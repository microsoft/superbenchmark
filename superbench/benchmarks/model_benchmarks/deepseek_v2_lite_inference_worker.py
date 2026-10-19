# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Official V2-Lite generation with vLLM, actual token counts and a separate reference."""

import argparse
import gc
import math
import os
import signal
import time
from pathlib import Path

from deepseek_v2_lite_utils import (
    FIXED_TEXTS, configure_cache, download_checkpoint, identity_report, interrupt_worker, packed_tokens,
    read_json, run_worker, write_json
)


def output_record(output, maximum, prompt_token_ids, include_logprobs=False):
    """Require finished requests, real tokens and bounded, finite probabilities."""
    if not output.finished or len(output.outputs) != 1 or not output.prompt_token_ids:
        raise RuntimeError('Missing or unfinished inference request.')
    if list(output.prompt_token_ids) != list(prompt_token_ids):
        raise RuntimeError('Engine output is associated with a different prompt.')
    completion = output.outputs[0]
    tokens = list(completion.token_ids)
    if not 0 < len(tokens) <= maximum or completion.finish_reason not in ('stop', 'length'):
        raise RuntimeError('Invalid completion count or finish reason.')
    record = {
        'prompt_token_ids': list(output.prompt_token_ids), 'token_ids': tokens,
        'finish_reason': completion.finish_reason,
    }
    if include_logprobs:
        if completion.logprobs is None or len(completion.logprobs) != len(tokens):
            raise RuntimeError('Missing per-token engine log probabilities.')
        values = []
        for token, probabilities in zip(tokens, completion.logprobs):
            if token not in probabilities:
                raise RuntimeError('Selected token has no log probability.')
            row = {str(key): float(value.logprob) for key, value in probabilities.items()}
            if any(not math.isfinite(value) for value in row.values()):
                raise RuntimeError('Nonfinite engine log probability.')
            values.append(row)
        record['logprobs'] = values
    return record


def request_times(output):
    """Use pinned engine timestamps, not fabricated prefill/decode durations."""
    metrics = output.metrics
    if metrics is None or metrics.is_corrupted:
        raise RuntimeError('Request statistics are missing or logits were corrupted.')
    count = len(output.outputs[0].token_ids)
    if metrics.num_generation_tokens != count:
        raise RuntimeError('Engine token statistics disagree with actual output.')
    prefill = metrics.first_token_ts - metrics.scheduled_ts
    ttft = metrics.first_token_latency
    decode = metrics.last_token_ts - metrics.first_token_ts
    if not math.isfinite(prefill) or prefill <= 0 or not math.isfinite(ttft) or ttft <= 0:
        raise RuntimeError('Invalid engine prefill/first-token timestamps.')
    if not math.isfinite(decode) or decode < 0 or (count > 1 and decode <= 0):
        raise RuntimeError('Invalid cached-decode timestamps.')
    return prefill * 1000, ttft * 1000, decode * 1000 / (count - 1) if count > 1 else None


def engine_checks(engine, documents, eos):
    """Exercise variable-length batches, single requests and EOS termination."""
    from vllm import SamplingParams

    sequences = [packed_tokens(documents, eos, length, 1, index * 7)[0] for index, length in enumerate((16, 23, 32))]
    prompts = [{'prompt_token_ids': sequence} for sequence in sequences]
    params = SamplingParams(temperature=0, max_tokens=4, logprobs=20, seed=20261015)
    batch = engine.generate(prompts, params, use_tqdm=False)
    if len(batch) != len(prompts):
        raise RuntimeError('Variable-length batch lost a request.')
    records = [output_record(output, 4, sequence, True) for output, sequence in zip(batch, sequences)]
    for prompt in prompts:
        single = engine.generate([prompt], params, use_tqdm=False)
        if len(single) != 1:
            raise RuntimeError('Single-request execution produced the wrong output count.')
        records.append(output_record(single[0], 4, prompt['prompt_token_ids'], True))
    stopped = engine.generate(
        [prompts[0]], SamplingParams(temperature=0, max_tokens=4, allowed_token_ids=[eos]), use_tqdm=False
    )
    if len(stopped) != 1 or list(stopped[0].outputs[0].token_ids) != [eos]:
        raise RuntimeError('EOS was not counted as exactly one generated token.')
    output_record(stopped[0], 4, sequences[0])
    if stopped[0].outputs[0].finish_reason != 'stop':
        raise RuntimeError('EOS did not terminate the request.')
    return records


def measured_generation(engine, documents, eos, args):
    """Measure complete generation batches, retaining actual request/token counts."""
    import torch
    from vllm import SamplingParams

    parameters = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, seed=20261015)
    step_times, input_counts, output_counts, prefills, ttfts, tpots = [], [], [], [], [], []
    for step in range(args.num_warmup + args.num_steps):
        sequences = packed_tokens(documents, eos, args.seq_len, args.batch_size, offset=step * args.seq_len)
        prompts = [{'prompt_token_ids': sequence} for sequence in sequences]
        torch.cuda.synchronize()
        start = time.monotonic()
        outputs = engine.generate(prompts, parameters, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = (time.monotonic() - start) * 1000
        if len(outputs) != len(prompts):
            raise RuntimeError('Generation returned fewer requests than submitted.')
        rows = [
            output_record(output, args.max_new_tokens, sequence) for output, sequence in zip(outputs, sequences)
        ]
        timings = [request_times(output) for output in outputs]
        if step >= args.num_warmup:
            step_times.append(elapsed)
            for row, (prefill, ttft, tpot) in zip(rows, timings):
                input_counts.append(len(row['prompt_token_ids']))
                output_counts.append(len(row['token_ids']))
                prefills.append(prefill)
                ttfts.append(ttft)
                if tpot is not None:
                    tpots.append(tpot)
        print('generation_batch={} actual_output_tokens={}'.format(
            step, sum(len(row['token_ids']) for row in rows)
        ), flush=True)
    if not tpots:
        raise RuntimeError('The profile never exercised multi-token cached decode.')
    return {
        'step_ms': step_times, 'input_token_counts': input_counts, 'output_token_counts': output_counts,
        'request_prefill_ms': prefills, 'ttft_ms': ttfts, 'decode_tpot_ms': tpots,
        'request_count': len(output_counts), 'num_gpus': 1,
        'metric_scope': 'engine scheduled-to-first-token and first-to-last-token; not isolated GPU kernel time',
    }


def main():
    """Include cold download, engine setup, generation, reference and shutdown."""
    from importlib.metadata import version

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache_dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference_python', required=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--num_warmup', type=int, default=2)
    parser.add_argument('--num_steps', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.8)
    parser.add_argument('--duration', type=int, default=3300)
    args = parser.parse_args()
    start = time.monotonic()
    signal.signal(signal.SIGTERM, interrupt_worker)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    configure_cache(args.cache_dir)
    if args.batch_size < 1 or args.seq_len < 32 or args.num_warmup < 0 or args.num_steps < 1:
        raise ValueError('Invalid generative inference dimensions.')
    if args.max_new_tokens < 4 or not 0 < args.gpu_memory_utilization <= 0.95 or not 30 < args.duration < 3600:
        raise ValueError('Invalid decode length, memory reservation or complete-job duration.')
    if version('vllm').split('+', 1)[0] != '0.31.0':
        raise RuntimeError('This adapter requires the pinned vLLM 0.31.0 statistics/shutdown contract.')
    import torch
    from vllm import LLM

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm GPU unavailable; no CPU or quantized fallback is allowed.')
    torch.cuda.set_device(0)
    report = identity_report()
    download_start = time.monotonic()
    checkpoint = download_checkpoint(args.cache_dir)
    report['download_verify_seconds'] = time.monotonic() - download_start
    engine_start = time.monotonic()
    engine = LLM(
        model=str(checkpoint), tokenizer=str(checkpoint), dtype='bfloat16',
        tensor_parallel_size=1, quantization=None, cpu_offload_gb=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.seq_len + args.max_new_tokens,
        max_num_seqs=args.batch_size, enable_prefix_caching=False, enable_chunked_prefill=False,
        disable_log_stats=False, seed=20261015, trust_remote_code=False,
    )
    try:
        if engine.llm_engine.model_config.dtype != torch.bfloat16 or engine.llm_engine.model_config.quantization:
            raise RuntimeError('Engine did not retain the requested unquantized BF16 model.')
        tokenizer = engine.get_tokenizer()
        documents = [tokenizer.encode(text, add_special_tokens=False) for text in FIXED_TEXTS]
        eos = tokenizer.eos_token_id
        report['engine_setup_seconds'] = time.monotonic() - engine_start
        records = engine_checks(engine, documents, eos)
        measurement_start = time.monotonic()
        report.update(measured_generation(engine, documents, eos, args))
        report['generation_seconds'] = time.monotonic() - measurement_start
    finally:
        # vLLM 0.31.0 exposes worker joining through its engine-core client.
        engine.llm_engine.engine_core.shutdown(timeout=10)
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    requests = args.output.parent / 'reference-requests.json'
    reference = args.output.parent / 'inference-reference.json'
    write_json(requests, records)
    reference_start = time.monotonic()
    run_worker(
        [args.reference_python, str(Path(__file__).with_name('deepseek_v2_lite_reference.py')),
         '--checkpoint', str(checkpoint), '--mode', 'inference',
         '--requests', str(requests), '--output', str(reference)],
        args.output.parent / 'reference.log', args.duration - (time.monotonic() - start) - 20,
    )
    result = read_json(reference)
    if result.get('correctness_passed') is not True or result.get('teardown_complete') is not True:
        raise RuntimeError('Official-code inference reference did not complete.')
    report['reference'] = result
    report['reference_seconds'] = time.monotonic() - reference_start
    report['correctness_passed'] = True
    report['teardown_complete'] = True
    report['vllm_version'] = version('vllm')
    report['torch_version'] = torch.__version__
    report['gpu_name'] = torch.cuda.get_device_name(0)
    report['total_seconds'] = time.monotonic() - start
    if report['total_seconds'] >= args.duration:
        raise TimeoutError('Generation exceeded the complete-job duration.')
    report['status'] = 'success'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
