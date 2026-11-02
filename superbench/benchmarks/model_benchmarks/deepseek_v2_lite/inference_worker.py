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

from . import numerics
from .utils import (
    FIXED_TEXTS, configure_cache, download_checkpoint, identity_report, interrupt_worker, packed_tokens, read_json,
    run_worker, write_json
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
        'prompt_token_ids': list(output.prompt_token_ids),
        'token_ids': tokens,
        'finish_reason': completion.finish_reason,
    }
    if include_logprobs:
        if completion.logprobs is None or len(completion.logprobs) != len(tokens):
            raise RuntimeError('Missing per-token engine log probabilities.')
        values = []
        for token, probabilities in zip(tokens, completion.logprobs):
            if token not in probabilities:
                raise RuntimeError('Selected token has no log probability.')
            if not 20 <= len(probabilities) <= 21:
                raise RuntimeError('Incomplete or oversized top-20 engine log probabilities.')
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
    measurement_start = None
    measurement_end = None
    for step in range(args.num_warmup + args.num_steps):
        sequences = packed_tokens(documents, eos, args.seq_len, args.batch_size, offset=step * args.seq_len)
        prompts = [{'prompt_token_ids': sequence} for sequence in sequences]
        torch.cuda.synchronize()
        if step == args.num_warmup:
            measurement_start = time.time()
        start = time.monotonic()
        outputs = engine.generate(prompts, parameters, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = (time.monotonic() - start) * 1000
        if len(outputs) != len(prompts):
            raise RuntimeError('Generation returned fewer requests than submitted.')
        rows = [output_record(output, args.max_new_tokens, sequence) for output, sequence in zip(outputs, sequences)]
        timings = [request_times(output) for output in outputs]
        if step >= args.num_warmup:
            measurement_end = time.time()
            step_times.append(elapsed)
            for row, (prefill, ttft, tpot) in zip(rows, timings):
                input_counts.append(len(row['prompt_token_ids']))
                output_counts.append(len(row['token_ids']))
                prefills.append(prefill)
                ttfts.append(ttft)
                if tpot is not None:
                    tpots.append(tpot)
        print(
            'generation_batch={} actual_output_tokens={}'.format(step, sum(len(row['token_ids']) for row in rows)),
            flush=True
        )
    if not tpots:
        raise RuntimeError('The profile never exercised multi-token cached decode.')
    return {
        'step_ms': step_times,
        'input_token_counts': input_counts,
        'output_token_counts': output_counts,
        'request_prefill_ms': prefills,
        'ttft_ms': ttfts,
        'decode_tpot_ms': tpots,
        'request_count': len(output_counts),
        'num_gpus': 1,
        'metric_scope': 'engine scheduled-to-first-token and first-to-last-token; not isolated GPU kernel time',
        'measurement_start_unix_seconds': measurement_start,
        'measurement_end_unix_seconds': measurement_end,
    }


def profile_checks(engine, documents, eos, args, offset=0, capture_all=False):
    """Numerically sample the real prompt length and batching, including later decode."""
    from vllm import SamplingParams

    sequences = packed_tokens(documents, eos, args.seq_len, args.batch_size, offset=offset)
    prompts = [{'prompt_token_ids': sequence} for sequence in sequences]
    parameters = SamplingParams(temperature=0, max_tokens=args.max_new_tokens, logprobs=20, seed=20261015)
    outputs = engine.generate(prompts, parameters, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise RuntimeError('Numerical profile check lost a request.')
    records = []
    selected = set(range(len(prompts))) if capture_all else {0, len(prompts) - 1}
    for index, (output, sequence) in enumerate(zip(outputs, sequences)):
        record = output_record(output, args.max_new_tokens, sequence, index in selected)
        if index in selected:
            count = len(record['token_ids'])
            record['full_prefix_steps'] = sorted({0, min(1, count - 1), count // 2, count - 1})
            record['profile_request_index'] = index
            record['profile_batch_size'] = args.batch_size
            records.append(record)
    return records


def qualified_checks(engine, documents, eos, args):
    """Reproduce every qualified short-case and complete-batch numerical window."""
    records = []
    for offset in numerics.OFFSETS:
        selected = engine_checks(engine, documents, eos)
        selected.extend(profile_checks(engine, documents, eos, args, offset=offset, capture_all=True))
        for record in selected:
            record['numerical_set_offset'] = offset
        records.extend(selected)
    return records


def prepare_numerical_policy(args):
    """Bind the explicit H100 policy before importing the engine."""
    numerics.validate_profile(args)
    if args.numerical_policy == numerics.POLICY:
        os.environ['VLLM_BATCH_INVARIANT'] = '1'
        os.environ['VLLM_MLA_DISABLE'] = '0'
        return numerics.load_contract()
    return None


def numerical_checks(engine, documents, eos, args):
    """Check actual engine precision before selecting the declared numerical coverage."""
    import torch

    config = engine.llm_engine.model_config
    if config.dtype != torch.bfloat16 or config.quantization:
        raise RuntimeError('Engine did not retain the requested unquantized BF16 model.')
    if args.numerical_policy == numerics.POLICY:
        if not config.use_mla:
            raise RuntimeError('The qualified H100 engine must retain its compressed-MLA implementation.')
        return qualified_checks(engine, documents, eos, args)
    records = engine_checks(engine, documents, eos)
    records.extend(profile_checks(engine, documents, eos, args))
    return records


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
    parser.add_argument('--numerical_policy', choices=numerics.POLICIES, default='strict_fp32')
    parser.add_argument('--duration', type=int, default=3300)
    args = parser.parse_args()
    start = time.monotonic()
    signal.signal(signal.SIGTERM, interrupt_worker)
    os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'] = '0'
    os.environ['NVIDIA_TF32_OVERRIDE'] = '0'
    os.environ['VLLM_FLOAT32_MATMUL_PRECISION'] = 'highest'
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'
    os.environ['VLLM_RAISE_ON_LOGIT_NANS'] = '1'
    configure_cache(args.cache_dir)
    if args.batch_size < 1 or args.seq_len < 32 or args.num_warmup < 0 or args.num_steps < 1:
        raise ValueError('Invalid generative inference dimensions.')
    if args.max_new_tokens < 4 or not 0 < args.gpu_memory_utilization <= 0.95 or not 30 < args.duration < 3600:
        raise ValueError('Invalid decode length, memory reservation or complete-job duration.')
    contract = prepare_numerical_policy(args)
    if version('vllm').split('+', 1)[0] != '0.31.0':
        raise RuntimeError('This adapter requires the pinned vLLM 0.31.0 statistics/shutdown contract.')
    import torch
    from vllm import LLM

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/ROCm GPU unavailable; no CPU or quantized fallback is allowed.')
    torch.cuda.set_device(0)
    if contract and torch.cuda.get_device_name(0) != contract['calibrated_contract']['runtime_identity']['gpu_name']:
        raise ValueError('The selected numerical policy requires its qualified H100 hardware.')
    report = identity_report()
    download_start = time.monotonic()
    checkpoint = download_checkpoint(args.cache_dir)
    report['download_verify_seconds'] = time.monotonic() - download_start
    engine_start = time.monotonic()
    engine = LLM(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        dtype='bfloat16',
        tensor_parallel_size=1,
        quantization=None,
        cpu_offload_gb=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.seq_len + args.max_new_tokens,
        max_num_seqs=args.batch_size,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        disable_log_stats=False,
        seed=20261015,
        trust_remote_code=False,
        hf_overrides={'moe_router_dtype': 'float32'},
    )
    try:
        tokenizer = engine.get_tokenizer()
        documents = [tokenizer.encode(text, add_special_tokens=False) for text in FIXED_TEXTS]
        eos = tokenizer.eos_token_id
        report['engine_setup_seconds'] = time.monotonic() - engine_start
        records = numerical_checks(engine, documents, eos, args)
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
    reference_command = (
        [
            args.reference_python, '-m', 'deepseek_v2_lite.numerics', '--checkpoint',
            str(checkpoint), '--requests',
            str(requests), '--output',
            str(reference)
        ] if contract else [
            args.reference_python, '-m', 'deepseek_v2_lite.reference', '--checkpoint',
            str(checkpoint), '--mode', 'inference', '--requests',
            str(requests), '--output',
            str(reference)
        ]
    )
    run_worker(
        reference_command,
        args.output.parent / 'reference.log',
        args.duration - (time.monotonic() - start) - 20,
    )
    result = read_json(reference)
    if result.get('correctness_passed') is not True or result.get('teardown_complete') is not True:
        raise RuntimeError('Official-code inference reference did not complete.')
    if contract:
        numerics.validate_reference_result(result)
    report['reference'] = result
    report['numerical_policy'] = args.numerical_policy
    report['batch_invariant'] = os.environ.get('VLLM_BATCH_INVARIANT') == '1'
    report['reference_seconds'] = time.monotonic() - reference_start
    report['correctness_passed'] = True
    report['teardown_complete'] = True
    report['vllm_version'] = version('vllm')
    report['router_dtype'] = 'float32'
    report['float32_matmul_precision'] = 'highest'
    report['triton_float32_precision'] = 'ieee'
    report['torch_version'] = torch.__version__
    report['gpu_name'] = torch.cuda.get_device_name(0)
    report['total_seconds'] = time.monotonic() - start
    if report['total_seconds'] >= args.duration:
        raise TimeoutError('Generation exceeded the complete-job duration.')
    report['status'] = 'success'
    write_json(args.output, report)


if __name__ == '__main__':
    main()
