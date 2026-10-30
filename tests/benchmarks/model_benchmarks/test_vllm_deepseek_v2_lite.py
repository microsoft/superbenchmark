# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU tests for the DeepSeek-V2-Lite inference benchmark: adapter, vLLM worker and H100 numerical policy."""

import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks import vllm_deepseek_v2_lite as adapter
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite import (
    cache as cache_cleanup, inference_worker, numerics as numerics_module
)
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite.utils import identity_report, write_json


# Adapter.
@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    """Use one local replica and an isolated output root."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    return adapter.DeepSeekV2LiteInference(
        'deepseek-v2-lite-inference',
        '--cache_root "{}" --num_steps 2 --batch_size 2 --seq_len 64 --duration 60'.format(tmp_path),
    )


def successful_worker(command, log, timeout, env):
    """Provide validated-shaped fixtures, not actual GPU measurements."""
    assert command[1:3] == ['-m', 'deepseek_v2_lite.inference_worker']
    assert env['PYTHONPATH'].split(os.pathsep)[0] == str(Path(adapter.__file__).resolve().parent)
    output = Path(command[command.index('--output') + 1])
    report = dict(
        identity_report(),
        status='success',
        correctness_passed=True,
        teardown_complete=True,
        step_ms=[10.0, 20.0],
        input_token_counts=[64, 64, 64, 64],
        output_token_counts=[1, 4, 3, 2],
        request_prefill_ms=[1.0] * 4,
        ttft_ms=[2.0] * 4,
        decode_tpot_ms=[3.0] * 3,
        request_count=4,
        num_gpus=1,
    )
    write_json(output, report)


def test_counts_and_model_result_shape(benchmark, monkeypatch):
    """Account for ten real output tokens, not four times the requested maximum."""
    monkeypatch.setattr(adapter, 'run_worker', successful_worker)
    assert benchmark.run()
    assert benchmark.type == BenchmarkType.MODEL
    assert benchmark.return_code == ReturnCode.SUCCESS
    assert benchmark.result['actual_output_tokens'] == [10.0]
    assert benchmark.result['bf16_inference_output_tokens_per_second'] == [10 * 1000 / 30]
    assert benchmark.raw_data['bf16_inference_output_token_count'] == [[1.0, 4.0, 3.0, 2.0]]


@pytest.mark.parametrize(
    'parameters', [
        '--seq_len 0', '--batch_size 0', '--max_new_tokens 1', '--num_steps 0', '--num_warmup -1',
        '--gpu_memory_utilization nan', '--duration 3600', '--run_count 2', '--precision float16'
    ]
)
def test_invalid_profiles(tmp_path, parameters):
    """Reject unsupported precisions, dimensions and multi-profile timing."""
    instance = adapter.DeepSeekV2LiteInference(
        'deepseek-v2-lite-inference', '--cache_root "{}" {}'.format(tmp_path, parameters)
    )
    assert not instance.run()
    assert instance.return_code == ReturnCode.INVALID_ARGUMENT


def test_cache_root_is_required():
    """Refuse to choose an implicit location for the per-run model cache."""
    instance = adapter.DeepSeekV2LiteInference('deepseek-v2-lite-inference', '--num_steps 2')
    assert not instance.run()
    assert instance.return_code == ReturnCode.INVALID_ARGUMENT


def test_parameters_render_in_help():
    """Keep every option compatible with the typed SuperBench help formatter."""
    instance = adapter.DeepSeekV2LiteInference('deepseek-v2-lite-inference')
    instance.add_parser_arguments()
    settings = instance.get_configurable_settings()
    assert '--numerical_policy' in settings
    assert '--cache_root' in settings


@pytest.mark.parametrize(
    'field,value', [
        ('output_token_counts', [1, 4, 3, 65]), ('output_token_counts', [1, 4, 3, 1.5]),
        ('input_token_counts', [63, 64, 64, 64]), ('ttft_ms', [0] * 4), ('decode_tpot_ms', [3.0] * 4),
        ('correctness_passed', False), ('teardown_complete', False)
    ]
)
def test_reject_invalid_generation_reports(benchmark, monkeypatch, field, value):
    """Validate output shapes and thresholds instead of accepting exit code zero."""
    def invalid_worker(command, log, timeout, env):
        successful_worker(command, log, timeout, env)
        output = Path(command[command.index('--output') + 1])
        report = adapter.read_json(output)
        report[field] = value
        write_json(output, report)

    monkeypatch.setattr(adapter, 'run_worker', invalid_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR


def test_deadline_failure(benchmark, monkeypatch):
    """A watchdog exit cannot be reported as a successful under-hour run."""
    def timeout(command, log, seconds, env):
        raise TimeoutError('whole job deadline')

    monkeypatch.setattr(adapter, 'run_worker', timeout)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT


@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout'])
def test_model_cache_is_removed_after_every_run(benchmark, monkeypatch, outcome):
    """Delete the per-run checkpoint cache on every exit path while keeping reports and logs."""
    jobs = []

    def worker(command, log, timeout, env):
        cache = Path(command[command.index('--cache_dir') + 1])
        (cache / 'checkpoint').mkdir(parents=True)
        (cache / 'checkpoint' / 'model-00001-of-000004.safetensors').write_bytes(b'weights')
        log.write_text('worker log\n')
        jobs.append(cache.parent)
        if outcome == 'failure':
            raise RuntimeError('worker failed')
        if outcome == 'timeout':
            raise TimeoutError('whole job deadline')
        successful_worker(command, log, timeout, env)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    assert benchmark.run() is (outcome == 'success')
    assert not (jobs[0] / 'cache').exists()
    assert (jobs[0] / 'worker.log').read_text() == 'worker log\n'
    assert (jobs[0] / 'report.json').is_file() is (outcome == 'success')


@pytest.mark.parametrize('cleanup_seconds', [7, 50, 70])
def test_cache_cleanup_counts_toward_the_job_deadline(benchmark, monkeypatch, cleanup_seconds):
    """Include cleanup in elapsed time and reject the exact deadline as well as an overrun."""
    clock = [100.0]
    run_cleanup = subprocess.run

    def worker(command, log, timeout, env):
        assert timeout == 40
        Path(command[command.index('--cache_dir') + 1]).mkdir(parents=True)
        successful_worker(command, log, timeout, env)
        clock[0] += 10

    def cleanup(command, **kwargs):
        assert kwargs['timeout'] == 50
        result = run_cleanup(command, **kwargs)
        clock[0] += cleanup_seconds
        return result

    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(adapter, 'time', fake_time)
    monkeypatch.setattr(cache_cleanup, 'time', fake_time)
    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', cleanup)
    assert benchmark.run() is (cleanup_seconds < 50)
    if cleanup_seconds < 50:
        assert benchmark.return_code == ReturnCode.SUCCESS
        assert benchmark.result['end_to_end_seconds'] == [10 + cleanup_seconds]
        assert benchmark.result['bf16_inference_output_tokens_per_second'] == [10 * 1000 / 30]
    else:
        assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT
        assert 'end_to_end_seconds' not in benchmark.result
        assert 'bf16_inference_output_tokens_per_second' not in benchmark.result


@pytest.mark.parametrize(
    'failure,code', [
        (RuntimeError('worker failed'), ReturnCode.RUNTIME_EXCEPTION_ERROR),
        (TimeoutError('whole job deadline'), ReturnCode.KILLED_BY_TIMEOUT),
    ]
)
def test_unfinished_failure_cleanup_keeps_the_original_error(benchmark, monkeypatch, failure, code):
    """Report the worker failure rather than a later cleanup timeout, and log both."""
    log_error = Mock()

    def worker(command, log, timeout, env):
        Path(command[command.index('--cache_dir') + 1]).mkdir(parents=True)
        raise failure

    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', Mock(side_effect=subprocess.TimeoutExpired(['cleanup'], 1)))
    monkeypatch.setattr(adapter.logger, 'error', log_error)
    assert not benchmark.run()
    assert benchmark.return_code == code
    messages = [call[0][0] % call[0][1:] for call in log_error.call_args_list]
    assert any(str(failure) in message for message in messages)
    assert any('after a failed run' in message for message in messages)


def test_h100_policy_cannot_use_an_unqualified_shape(benchmark, tmp_path):
    """Keep calibration/profile compatibility fail-closed in the frontend."""
    instance = adapter.DeepSeekV2LiteInference(
        'deepseek-v2-lite-inference', '--cache_root "{}" --numerical_policy h100_bf16_v6'.format(tmp_path)
    )
    assert not instance.run()
    assert instance.return_code == ReturnCode.INVALID_ARGUMENT


@pytest.mark.parametrize('fault', [None, 'policy', 'batch-invariance', 'reference'])
def test_h100_adapter_requires_matching_numerical_evidence(tmp_path, monkeypatch, fault):
    """Validate the selected policy before publishing performance or numerical counters."""
    arguments = '--cache_root "{}" --numerical_policy h100_bf16_v6 --num_steps 256'.format(tmp_path)
    instance = adapter.DeepSeekV2LiteInference('deepseek-v2-lite-inference', arguments)
    validator = Mock(side_effect=ValueError('bad reference') if fault == 'reference' else None)
    monkeypatch.setattr(adapter.numerics, 'validate_reference_result', validator)

    def worker(command, log, timeout, env):
        assert command[command.index('--numerical_policy') + 1] == adapter.numerics.POLICY
        report = dict(
            identity_report(),
            status='success',
            correctness_passed=True,
            teardown_complete=True,
            step_ms=[10.0] * 256,
            input_token_counts=[512] * 2048,
            output_token_counts=[64] * 2048,
            request_prefill_ms=[1.0] * 2048,
            ttft_ms=[2.0] * 2048,
            decode_tpot_ms=[3.0] * 2048,
            request_count=2048,
            num_gpus=1,
            numerical_policy=adapter.numerics.POLICY,
            batch_invariant=True,
            reference={
                'positive_positions': 3216,
                'negative_positions': 6438
            },
        )
        if fault == 'policy':
            report['numerical_policy'] = 'strict_fp32'
        if fault == 'batch-invariance':
            report['batch_invariant'] = False
        write_json(Path(command[command.index('--output') + 1]), report)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    assert instance.run() is (fault is None)
    if fault is None:
        assert instance.return_code == ReturnCode.SUCCESS
        assert instance.result['numerical_reference_positions'] == [3216]
        assert instance.result['numerical_fault_rejections'] == [6438]
        validator.assert_called_once_with({'positive_positions': 3216, 'negative_positions': 6438})
    else:
        assert instance.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR


# vLLM worker.
@pytest.fixture
def worker():
    """Return the worker, which imports vLLM and CUDA only inside its functions."""
    return inference_worker


def response(tokens=(7, 8)):
    """Construct engine-response fixtures without simulating GPU success."""
    return SimpleNamespace(
        finished=True,
        prompt_token_ids=[1, 2, 3],
        outputs=[
            SimpleNamespace(
                token_ids=list(tokens),
                finish_reason='length',
                logprobs=[
                    {key: SimpleNamespace(logprob=-0.5 if key == token else -2.0)
                     for key in set(range(20)) | {token}} for token in tokens
                ]
            )
        ],
        metrics=SimpleNamespace(
            is_corrupted=False,
            num_generation_tokens=len(tokens),
            scheduled_ts=10.0,
            first_token_ts=10.25,
            last_token_ts=10.5,
            first_token_latency=0.4
        ),
    )


def test_actual_output_tokens_not_maximum(worker):
    """Count emitted token IDs instead of treating max_tokens as completed work."""
    result = worker.output_record(response(), 64, [1, 2, 3], True)
    assert result['token_ids'] == [7, 8]
    assert len(result['token_ids']) == 2
    assert result['logprobs'][0]['7'] == result['logprobs'][1]['8'] == -0.5
    assert all(len(row) == 20 for row in result['logprobs'])


@pytest.mark.parametrize('tokens', [(), (1, 2, 3, 4, 5)])
def test_invalid_token_counts(worker, tokens):
    """Reject empty or over-limit outputs rather than inventing throughput."""
    with pytest.raises(RuntimeError, match='count'):
        worker.output_record(response(tokens), 4, [1, 2, 3])


@pytest.mark.parametrize('field,value', [('finished', False), ('prompt_token_ids', [])])
def test_incomplete_requests(worker, field, value):
    """Failed request batches do not count as complete model execution."""
    output = response()
    setattr(output, field, value)
    with pytest.raises(RuntimeError, match='unfinished'):
        worker.output_record(output, 64, [1, 2, 3])


def test_output_must_match_submitted_prompt(worker):
    """Do not validate a correct response against a different request's tokens."""
    with pytest.raises(RuntimeError, match='different prompt'):
        worker.output_record(response(), 64, [3, 2, 1])


def test_missing_logprobs(worker):
    """Missing numerical-reference evidence must fail explicitly."""
    output = response()
    output.outputs[0].logprobs = None
    with pytest.raises(RuntimeError, match='probabilities'):
        worker.output_record(output, 64, [1, 2, 3], True)


@pytest.mark.parametrize(
    'field,value', [
        ('is_corrupted', True), ('num_generation_tokens', 64), ('first_token_ts', 0.0),
        ('first_token_latency', float('nan')), ('last_token_ts', 10.0)
    ]
)
def test_invalid_statistics(worker, field, value):
    """Never substitute zero/default durations or hide corrupted logits."""
    output = response()
    setattr(output.metrics, field, value)
    with pytest.raises(RuntimeError):
        worker.request_times(output)


def test_measured_generation_excludes_warmup_and_counts_actual_tokens(worker, monkeypatch):
    """Exclude distinct warmup timings and retain mixed EOS/multi-token measurements."""
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    clock = iter([0.0, 0.1, 1.0, 1.2, 2.0, 2.3])
    monkeypatch.setattr(worker.time, 'monotonic', lambda: next(clock))
    batches = iter(
        [
            [(1, 0.05, 0.06, 0), (4, 0.075, 0.085, 0.015)],
            [(1, 0.01, 0.015, 0), (3, 0.02, 0.025, 0.12)],
            [(2, 0.03, 0.035, 0.08), (1, 0.04, 0.045, 0)],
        ]
    )
    submitted = []
    monkeypatch.setattr(worker.time, 'time', lambda: 1000.0 + 100 * len(submitted))

    class Engine:
        def generate(self, prompts, parameters, use_tqdm):
            submitted.append(prompts)
            outputs = []
            for prompt, (count, prefill, ttft, decode) in zip(prompts, next(batches)):
                output = response((99, ) if count == 1 else tuple(range(7, 7 + count)))
                output.prompt_token_ids = prompt['prompt_token_ids']
                output.metrics.first_token_ts = output.metrics.scheduled_ts + prefill
                output.metrics.first_token_latency = ttft
                output.metrics.last_token_ts = output.metrics.first_token_ts + decode
                if count == 1:
                    output.outputs[0].finish_reason = 'stop'
                outputs.append(output)
            return outputs

    args = SimpleNamespace(num_warmup=1, num_steps=2, seq_len=32, batch_size=2, max_new_tokens=4)
    report = worker.measured_generation(Engine(), [[1, 2, 3], [4, 5]], 99, args)
    assert len(submitted) == 3
    assert report['step_ms'] == pytest.approx([200, 300])
    assert report['request_count'] == 4
    assert report['input_token_counts'] == [32, 32, 32, 32]
    assert report['output_token_counts'] == [1, 3, 2, 1]
    assert report['request_prefill_ms'] == pytest.approx([10, 20, 30, 40])
    assert report['ttft_ms'] == pytest.approx([15, 25, 35, 45])
    assert report['decode_tpot_ms'] == pytest.approx([60, 80])
    assert report['measurement_start_unix_seconds'] == 1100
    assert report['measurement_end_unix_seconds'] == 1300


def test_measured_generation_rejects_missing_requests(worker, monkeypatch):
    """A partial generation batch cannot become a valid throughput sample."""
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(SamplingParams=SimpleNamespace))
    engine = SimpleNamespace(generate=lambda *args, **kwargs: [])
    args = SimpleNamespace(num_warmup=0, num_steps=1, seq_len=32, batch_size=2, max_new_tokens=4)
    with pytest.raises(RuntimeError, match='fewer requests'):
        worker.measured_generation(engine, [[1, 2, 3]], 99, args)


@pytest.fixture
def main_driver(worker, tmp_path, monkeypatch):
    """Exercise the full worker's control flow without real vLLM or GPU execution."""
    if sys.version_info < (3, 10):
        pytest.skip('The standalone inference runtime requires Python 3.10 or later.')
    import importlib.metadata
    torch = pytest.importorskip('torch')

    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.31.0' if name == 'vllm' else version(name))
    for variable in [
        'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE', 'VLLM_FLOAT32_MATMUL_PRECISION',
        'TRITON_F32_DEFAULT', 'VLLM_RAISE_ON_LOGIT_NANS', 'VLLM_BATCH_INVARIANT', 'VLLM_MLA_DISABLE'
    ]:
        monkeypatch.setenv(variable, 'test-original')
    monkeypatch.setattr(worker.signal, 'signal', Mock())
    monkeypatch.setattr(worker, 'configure_cache', Mock())
    download = Mock(return_value=tmp_path / 'checkpoint')
    monkeypatch.setattr(worker, 'download_checkpoint', download)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'synchronize', Mock())
    monkeypatch.setattr(torch.cuda, 'empty_cache', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'CPU orchestration fixture')
    shutdown = Mock()
    config = SimpleNamespace(dtype=torch.bfloat16, quantization=None)
    tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3], eos_token_id=99)

    def generate(prompts, parameters, use_tqdm):
        outputs = []
        stop = hasattr(parameters, 'allowed_token_ids')
        tokens = parameters.allowed_token_ids if stop else list(range(7, 7 + parameters.max_tokens))
        for prompt in prompts:
            output = response(tuple(tokens))
            output.prompt_token_ids = prompt['prompt_token_ids']
            if stop:
                output.outputs[0].finish_reason = 'stop'
                output.metrics.last_token_ts = output.metrics.first_token_ts
            outputs.append(output)
        return outputs

    engine = SimpleNamespace(
        llm_engine=SimpleNamespace(model_config=config, engine_core=SimpleNamespace(shutdown=shutdown)),
        get_tokenizer=lambda: tokenizer,
        generate=generate,
    )
    factory = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=factory, SamplingParams=SimpleNamespace))

    def reference(command, log, timeout):
        assert timeout > 0
        assert shutdown.called
        output = Path(command[command.index('--output') + 1])
        worker.write_json(output, {'correctness_passed': True, 'teardown_complete': True})

    oracle = Mock(side_effect=reference)
    monkeypatch.setattr(worker, 'run_worker', oracle)
    output = tmp_path / 'report.json'
    argv = [
        'worker',
        '--cache_dir',
        str(tmp_path / 'cache'),
        '--output',
        str(output),
        '--reference_python',
        'reference-python',
        '--batch_size',
        '2',
        '--seq_len',
        '64',
        '--num_warmup',
        '1',
        '--num_steps',
        '2',
        '--max_new_tokens',
        '4',
        '--duration',
        '60',
    ]
    monkeypatch.setattr(sys, 'argv', argv)
    return SimpleNamespace(
        output=output,
        argv=argv,
        factory=factory,
        engine=engine,
        config=config,
        shutdown=shutdown,
        oracle=oracle,
        download=download,
        torch=torch,
    )


def test_inference_main_keeps_precision_counts_and_reference_lifecycle(worker, main_driver):
    """Check the selected policies and report only after shutdown and reference success."""
    state = main_driver
    worker.main()
    options = state.factory.call_args[1]
    assert options['dtype'] == 'bfloat16' and options['quantization'] is None
    assert options['hf_overrides'] == {'moe_router_dtype': 'float32'}
    assert options['max_model_len'] == 68 and options['max_num_seqs'] == 2
    report = json.loads(state.output.read_text())
    assert report['status'] == 'success' and report['correctness_passed'] and report['teardown_complete']
    assert report['request_count'] == 4 and report['output_token_counts'] == [4] * 4
    assert report['router_dtype'] == 'float32' and report['float32_matmul_precision'] == 'highest'
    assert report['numerical_policy'] == 'strict_fp32' and report['batch_invariant'] is False
    state.shutdown.assert_called_once_with(timeout=10)
    state.oracle.assert_called_once()


def test_corrupted_measured_generation_cannot_publish_success(worker, main_driver, monkeypatch):
    """Corruption in unscored measured requests must prevent result publication."""
    generate = main_driver.engine.generate

    def corrupted_generation(*args, **kwargs):
        outputs = generate(*args, **kwargs)
        for output in outputs:
            output.metrics.is_corrupted = True
        return outputs

    monkeypatch.setattr(main_driver.engine, 'generate', corrupted_generation)
    with pytest.raises(RuntimeError, match='logits were corrupted'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    main_driver.oracle.assert_not_called()
    assert not main_driver.output.exists()


def test_engine_error_still_shuts_down(worker, main_driver, monkeypatch):
    """Generation failure cannot leave the engine alive or produce performance success."""
    monkeypatch.setattr(main_driver.engine, 'generate', Mock(side_effect=RuntimeError('generation failed')))
    with pytest.raises(RuntimeError, match='generation failed'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    main_driver.oracle.assert_not_called()
    assert not main_driver.output.exists()


def test_failed_reference_is_not_success(worker, main_driver):
    """A numerically rejected generation run must not publish a success report."""
    main_driver.oracle.side_effect = RuntimeError('numerical gate failed')
    with pytest.raises(RuntimeError, match='numerical gate'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    assert not main_driver.output.exists()


def test_wrong_engine_precision_is_rejected(worker, main_driver):
    """Do not silently replace the BF16 performance workload with another precision."""
    main_driver.config.dtype = main_driver.torch.float32
    with pytest.raises(RuntimeError, match='unquantized BF16'):
        worker.main()
    main_driver.shutdown.assert_called_once()
    assert not main_driver.output.exists()


def test_unsupported_engine_version_fails_before_download(worker, main_driver, monkeypatch):
    """Version-specific timing and shutdown contracts must not silently drift."""
    import importlib.metadata
    version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, 'version', lambda name: '0.32.0' if name == 'vllm' else version(name))
    with pytest.raises(RuntimeError, match='pinned vLLM'):
        worker.main()
    main_driver.download.assert_not_called()


def test_inference_does_not_fall_back_to_cpu(worker, main_driver, monkeypatch):
    """Unavailable GPU execution is an explicit failure, not a replacement benchmark."""
    monkeypatch.setattr(main_driver.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='GPU unavailable'):
        worker.main()
    main_driver.download.assert_not_called()


@pytest.mark.parametrize(
    'override', [
        ['--batch_size', '0'],
        ['--seq_len', '16'],
        ['--num_steps', '0'],
        ['--num_warmup', '-1'],
        ['--max_new_tokens', '3'],
        ['--gpu_memory_utilization', 'nan'],
        ['--duration', '3600'],
    ]
)
def test_inference_preflight_rejects_invalid_profile(worker, main_driver, monkeypatch, override):
    """Invalid generation parameters must fail before download or engine creation."""
    monkeypatch.setattr(sys, 'argv', main_driver.argv + override)
    with pytest.raises(ValueError):
        worker.main()
    main_driver.download.assert_not_called()
    main_driver.factory.assert_not_called()


def h100_arguments(worker, main_driver):
    """Select the frozen shape without changing the existing small fixture's defaults."""
    arguments = main_driver.argv + ['--numerical_policy', worker.numerics.POLICY]
    for key, value in worker.numerics.PROFILE.items():
        arguments.extend(['--' + key, str(value)])
    return arguments


def test_h100_policy_uses_pinned_engine_settings_and_separate_reference(worker, main_driver, monkeypatch):
    """Wire the explicit policy without replacing the measured BF16 model."""
    monkeypatch.setattr(sys, 'argv', h100_arguments(worker, main_driver))
    expected_gpu = worker.numerics.load_contract()['calibrated_contract']['runtime_identity']['gpu_name']
    monkeypatch.setattr(main_driver.torch.cuda, 'get_device_name', lambda index: expected_gpu)
    main_driver.config.use_mla = True
    validation = Mock()
    monkeypatch.setattr(worker.numerics, 'validate_reference_result', validation)
    worker.main()
    report = json.loads(main_driver.output.read_text())
    assert report['numerical_policy'] == worker.numerics.POLICY and report['batch_invariant'] is True
    assert main_driver.factory.call_args.kwargs['dtype'] == 'bfloat16'
    command = main_driver.oracle.call_args.args[0]
    assert command[1:3] == ['-m', 'deepseek_v2_lite.numerics'] and '--mode' not in command
    assert len(json.loads((main_driver.output.parent / 'reference-requests.json').read_text())) == 84
    validation.assert_called_once_with(report['reference'])
    assert worker.os.environ['VLLM_BATCH_INVARIANT'] == '1'
    assert worker.os.environ['VLLM_MLA_DISABLE'] == '0'
    main_driver.shutdown.assert_called_once()


def test_h100_policy_rejects_another_gpu_before_download(worker, main_driver, monkeypatch):
    """A qualified H100 policy cannot be silently reused on other hardware."""
    monkeypatch.setattr(sys, 'argv', h100_arguments(worker, main_driver))
    with pytest.raises(ValueError, match='qualified H100 hardware'):
        worker.main()
    main_driver.download.assert_not_called()


# H100 numerical policy.
@pytest.fixture
def numerics():
    """Return the numerical policy, which imports no engine at load time."""
    return numerics_module


def budgets():
    """Use deliberately small synthetic bounds for arithmetic controls."""
    return {
        'max_probability_error': 0.02,
        'relative_probability_mass_error': 0.03,
        'normalized_logprob_error': 0.5,
    }


@pytest.fixture
def policy_file(numerics, tmp_path, monkeypatch):
    """Create qualified-shaped test evidence without claiming GPU execution."""
    monkeypatch.setattr(numerics, '__file__', str(tmp_path / 'numerics.py'))
    path = tmp_path / 'h100_bf16_v6.json'
    candidate = {
        'method': numerics.METHOD,
        'reference_only': True,
        'input_matched_limit_rule': numerics.LIMIT_RULE,
        'qualification_offsets': list(numerics.OFFSETS),
        'profile': {key: numerics.PROFILE[key]
                    for key in ('batch_size', 'seq_len', 'max_new_tokens')},
        'qualification_stream': {key: numerics.PROFILE[key]
                                 for key in ('num_warmup', 'num_steps')},
    }
    candidate['frozen_limits_sha256'] = numerics.canonical_hash(candidate)
    contract = {
        'status': 'qualified',
        'calibrated_contract': candidate,
        'qualification': [{
            'profile_offset': offset
        } for offset in numerics.OFFSETS],
        'reference_validation': {
            'non_anchor_comparisons': 1
        },
    }

    def save():
        path.write_text(json.dumps(contract))
        monkeypatch.setattr(numerics, 'CONTRACT_SHA256', hashlib.sha256(path.read_bytes()).hexdigest())

    save()
    return SimpleNamespace(path=path, contract=contract, candidate=candidate, save=save)


def test_frozen_profile_is_explicit_and_legacy_policy_is_unchanged(numerics):
    """Do not apply one H100 qualification to other shapes or the default policy."""
    numerics.validate_profile(SimpleNamespace(numerical_policy='strict_fp32', batch_size=1))
    args = SimpleNamespace(numerical_policy=numerics.POLICY, **numerics.PROFILE)
    numerics.validate_profile(args)
    args.num_steps -= 1
    with pytest.raises(ValueError, match='complete frozen'):
        numerics.validate_profile(args)
    with pytest.raises(ValueError, match='Unknown'):
        numerics.validate_profile(SimpleNamespace(numerical_policy='other'))


def test_contract_is_bound_to_exact_file_bytes(numerics, policy_file):
    """Even a semantically equivalent substituted policy requires a new identity."""
    assert numerics.load_contract() == policy_file.contract
    policy_file.path.write_text(policy_file.path.read_text() + '\n')
    with pytest.raises(ValueError, match='changed'):
        numerics.load_contract()


@pytest.mark.parametrize('fault', ['status', 'method', 'shape', 'stream', 'offsets', 'evidence', 'anchor-only', 'hash'])
def test_contract_rejects_incomplete_qualification(numerics, policy_file, fault):
    """Validate semantics as well as a matching artifact digest."""
    candidate, contract = policy_file.candidate, policy_file.contract
    if fault == 'status':
        contract['status'] = 'candidate'
    elif fault == 'method':
        candidate['method'] = 'unqualified-method'
    elif fault == 'shape':
        candidate['profile']['batch_size'] = 1
    elif fault == 'stream':
        candidate['qualification_stream']['num_steps'] = 1
    elif fault == 'offsets':
        candidate['qualification_offsets'].pop()
    elif fault == 'evidence':
        contract['qualification'].pop()
    elif fault == 'anchor-only':
        contract['reference_validation']['non_anchor_comparisons'] = 0
    else:
        candidate['frozen_limits_sha256'] = 'changed'
    if fault != 'hash':
        candidate['frozen_limits_sha256'] = numerics.canonical_hash(
            {key: value
             for key, value in candidate.items() if key != 'frozen_limits_sha256'}
        )
    policy_file.save()
    with pytest.raises(ValueError, match='qualification'):
        numerics.load_contract()


@pytest.mark.parametrize(
    'expected,actual', [
        ([], []),
        ([-1.0], []),
        ([float('nan')], [-1.0]),
        ([-1.0], [float('inf')]),
        ([-1.0], [True]),
        ([-1.0], [0.1]),
        ([-1000.0], [-1000.0]),
        ([0.0, 0.0], [0.0, 0.0]),
    ]
)
def test_invalid_probability_data_is_rejected(numerics, expected, actual):
    """Reject malformed, nonfinite and unnormalized probabilities explicitly."""
    with pytest.raises(ValueError):
        numerics.probability_errors(expected, actual)


def test_input_matched_limits_are_reference_only(numerics):
    """Keep calibration unchanged and fail if uncertainty admits half normalization."""
    calibrated = budgets()
    original = dict(calibrated)
    reference = {metric: 0.01 for metric in numerics.METRICS}
    assert numerics.input_matched_limits(calibrated, reference) == {
        metric: original[metric] + reference[metric]
        for metric in numerics.METRICS
    }
    assert calibrated == original
    reference['relative_probability_mass_error'] = 0.5
    with pytest.raises(ValueError, match='half-normalization'):
        numerics.input_matched_limits(calibrated, reference)
    reference['relative_probability_mass_error'] = float('nan')
    with pytest.raises(ValueError, match='input-matched'):
        numerics.input_matched_limits(calibrated, reference)
    reference['relative_probability_mass_error'] = 0
    calibrated['normalized_logprob_error'] = 0
    with pytest.raises(ValueError, match='budget'):
        numerics.input_matched_limits(calibrated, reference)


def test_greedy_rules_and_invalid_bounds(numerics):
    """Near ties do not waive reported-greedy or probability-gap checks."""
    expected = {0: math.log(0.9), 1: math.log(0.1)}
    result = numerics.score_probabilities(expected, expected, 1, 0, budgets())
    assert not result['accepted'] and not result['greedy_match']
    assert 'reported token is not greedy' in result['failures']
    assert 'greedy choice exceeds calibrated uncertainty' in result['failures']
    assert numerics.score_probabilities(expected, expected, 0, 0, budgets())['accepted']
    with pytest.raises(ValueError, match='Incomplete'):
        numerics.score_probabilities(expected, {0: expected[0]}, 0, 0, budgets())
    limits = budgets()
    limits['normalized_logprob_error'] = float('inf')
    with pytest.raises(ValueError, match='bound'):
        numerics.score_probabilities(expected, expected, 0, 0, limits)


def test_top_k_shape_bounds_and_tail_fault(numerics):
    """A tiny tail still has to reject the fixed temperature fault."""
    torch = pytest.importorskip('torch')
    logits = torch.full((32, ), -40.0)
    logits[0] = 0
    expected = logits.log_softmax(-1)
    wrong = (expected / 2).log_softmax(-1)
    ids = wrong.topk(20).indices.tolist()
    observed = {str(key): wrong[key].item() for key in ids}
    result = numerics.evaluate_logits(expected, observed, 0, budgets())
    assert not result['accepted'] and 'normalized_logprob_error' in result['failures']
    with pytest.raises(ValueError, match='top-20'):
        numerics.evaluate_logits(expected, {'0': 0.0}, 0, budgets())
    observed['1000'] = observed.pop(str(ids[-1]))
    with pytest.raises(ValueError, match='out of range'):
        numerics.evaluate_logits(expected, observed, 0, budgets())
    missing = {str(key): -10.0 for key in range(1, 21)}
    assert not numerics.evaluate_logits(expected, missing, 1, budgets())['accepted']
    duplicate = {str(key): -10.0 for key in range(20)}
    duplicate['00'] = -10.0
    with pytest.raises(ValueError, match='duplicated'):
        numerics.evaluate_logits(expected, duplicate, 0, budgets())


@pytest.fixture
def cpu_reference(monkeypatch):
    """Exercise reference tensor operations on CPU with explicit fake device transport."""
    torch = pytest.importorskip('torch')
    tensor, ones = torch.tensor, torch.ones

    def cpu_tensor(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return tensor(*args, **kwargs)

    def cpu_ones(*args, **kwargs):
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return ones(*args, **kwargs)

    monkeypatch.setattr(torch, 'tensor', cpu_tensor)
    monkeypatch.setattr(torch, 'ones', cpu_ones)
    return torch


def test_complete_prefix_and_cached_reference_shapes(numerics, cpu_reference):
    """Check all attention-mask lengths and cache positions without a GPU."""
    torch = cpu_reference
    calls = []

    def model(input_ids, attention_mask, use_cache, past_key_values=None):
        calls.append((input_ids.shape, attention_mask.shape, use_cache, past_key_values))
        logits = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24)
        return SimpleNamespace(
            logits=logits.expand(input_ids.shape[0], input_ids.shape[1], 24), past_key_values=len(calls)
        )

    cases = [{'id': str(index), 'prompt_token_ids': [1, 2, 3], 'continuation_ids': [4, 5]} for index in range(3)]
    full = numerics.full_logits(model, [[1, 2, 3]])
    assert full.shape == (1, 24) and calls[-1][2:] == (False, None)
    values = numerics.collect_matched_bf16(model, cases, 2)
    assert set(values) == {'0', '1', '2'} and all(value.shape == (2, 24) for value in values.values())
    assert calls[1][0:2] == (torch.Size([2, 3]), torch.Size([2, 3]))
    assert calls[2][0:2] == (torch.Size([2, 1]), torch.Size([2, 4]))
    assert calls[2][3] is not None
    errors = numerics.tensor_errors(values['0'][0], values['1'][0])
    assert errors['greedy_match'] and all(errors[metric] == 0 for metric in numerics.METRICS)
    with pytest.raises(ValueError, match='uniquely'):
        numerics.configured_batches(cases + cases, 2)
    mismatched = [cases[0], dict(cases[1], continuation_ids=[4])]
    with pytest.raises(ValueError, match='matching sequence'):
        list(numerics.cached_logits(model, mismatched))


def test_reference_rejects_nonfinite_and_incomplete_output(numerics, cpu_reference, monkeypatch):
    """Reference failures cannot create success-shaped numerical evidence."""
    torch = cpu_reference

    def model(**kwargs):
        return SimpleNamespace(logits=torch.full((1, 2, 24), float('nan')), past_key_values=1)

    cases = [{'id': '0', 'prompt_token_ids': [1, 2], 'continuation_ids': [3]}]
    with pytest.raises(RuntimeError, match='Nonfinite'):
        numerics.full_logits(model, [[1, 2]])
    with pytest.raises(RuntimeError, match='Nonfinite'):
        list(numerics.cached_logits(model, cases))
    monkeypatch.setattr(numerics, 'cached_logits', lambda *args: iter([]))
    with pytest.raises(ValueError, match='Incomplete'):
        numerics.collect_matched_bf16(model, cases, 1)
    monkeypatch.setattr(numerics, 'cached_logits', lambda *args: iter([(1, torch.zeros((1, 24)))]))
    with pytest.raises(ValueError, match='out-of-order'):
        numerics.collect_matched_bf16(model, cases, 1)


def request_sets(numerics):
    """Create token-shaped fixtures, not reference or engine measurements."""
    documents, eos, records = [[1, 2, 3], [4, 5, 6]], 99, []
    short = [
        numerics.packed_tokens(documents, eos, length, 1, index * 7)[0] for index, length in enumerate((16, 23, 32))
    ]
    for offset in numerics.OFFSETS:
        prompts = short * 2 + numerics.packed_tokens(documents, eos, 512, 8, offset)
        for index, prompt in enumerate(prompts):
            count = 4 if index < 6 else 64
            row = {
                'prompt_token_ids': prompt,
                'token_ids': [1] * count,
                'logprobs': [{} for _ in range(count)],
                'numerical_set_offset': offset,
            }
            if index >= 6:
                row.update(profile_request_index=index - 6, profile_batch_size=8)
            records.append(row)
    return documents, eos, records


@pytest.mark.parametrize('fault', ['missing-window', 'wrong-prompt', 'short-decode', 'lost-row', 'bad-token', 'trace'])
def test_frozen_request_sets_cannot_drop_positions(numerics, fault):
    """Require all six windows and every configured-batch row."""
    documents, eos, requests = request_sets(numerics)
    groups = numerics.validate_requests(requests, documents, eos)
    assert list(groups) == list(numerics.OFFSETS)
    if fault == 'missing-window':
        requests.pop()
    elif fault == 'wrong-prompt':
        requests[0]['prompt_token_ids'] = [7]
    elif fault == 'short-decode':
        for row in requests[:14]:
            row['token_ids'] = row['token_ids'][:1]
            row['logprobs'] = row['logprobs'][:1]
    elif fault == 'lost-row':
        requests[6]['profile_request_index'] = 2
    elif fault == 'bad-token':
        requests[0]['token_ids'][0] = True
    else:
        requests[0]['logprobs'].pop()
    with pytest.raises(ValueError):
        numerics.validate_requests(requests, documents, eos)


@pytest.mark.parametrize('corrupt', [False, True])
def test_group_scoring_retains_all_positive_and_fault_positions(numerics, monkeypatch, corrupt):
    """A good synthetic pair must reject every fault; a corrupted pair must fail."""
    torch = pytest.importorskip('torch')
    vectors = [torch.arange(32, dtype=torch.float32), torch.arange(31, -1, -1, dtype=torch.float32)]
    expected = [value.log_softmax(-1) for value in vectors]
    requests, matched = [], {}
    for index in range(2):
        ids = expected[index].topk(20).indices.tolist()
        observed = {str(key): expected[index][key].item() for key in ids}
        if corrupt:
            observed = {key: value - math.log(2) for key, value in observed.items()}
        requests.append(
            {
                'prompt_token_ids': [index],
                'token_ids': [expected[index].argmax().item()] * 2,
                'logprobs': [observed, observed],
            }
        )
        matched['request-{}'.format(index)] = torch.stack([expected[index], expected[index]])
    monkeypatch.setattr(numerics, 'full_logits', lambda model, rows: vectors[rows[0][0]].unsqueeze(0))
    result = numerics.evaluate_group(None, requests, matched, budgets())
    assert result['positive_positions'] == 4 and result['negative_positions'] == 9
    assert len(result['positive_failures']) == (4 if corrupt else 0)
    assert not result['negative_failures']
    assert {row['fault']
            for row in result['all_negative_records']} == {
                'half-normalization',
                'temperature-two',
                'last-request-context-for-first',
            }


@pytest.mark.parametrize('failure', [False, True])
def test_reference_models_are_sequential_and_always_released(numerics, monkeypatch, failure):
    """Exercise lifecycle plumbing with explicit CPU-only model doubles."""
    if sys.version_info < (3, 10):
        pytest.skip('The standalone reference runtime requires Python 3.10 or later.')
    torch = pytest.importorskip('torch')
    from superbench.benchmarks.model_benchmarks.deepseek_v2_lite import reference

    loaded = []

    def load_model(checkpoint, dtype):
        loaded.append(dtype)
        model = SimpleNamespace()
        model.cuda = lambda: model
        model.eval = lambda: model
        return model

    monkeypatch.setattr(reference, 'load_model', load_model)
    monkeypatch.setattr(reference, 'fixed_documents', lambda checkpoint: ([[1, 2]], 99))
    monkeypatch.setattr(numerics, 'validate_requests', lambda *args: {0: [{'prompt_token_ids': [1], 'token_ids': [2]}]})
    monkeypatch.setattr(numerics, 'collect_matched_bf16', lambda *args: {})
    monkeypatch.setattr(torch, 'set_num_threads', Mock())
    monkeypatch.setattr(torch, 'set_float32_matmul_precision', Mock())
    monkeypatch.setattr(torch.cuda, 'set_device', Mock())
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda index: 'synthetic-h100')
    release = Mock()
    monkeypatch.setattr(torch.cuda, 'empty_cache', release)
    monkeypatch.setattr(torch.cuda, 'synchronize', Mock())
    from importlib.metadata import version
    identity = {
        'torch_version': torch.__version__,
        'transformers_version': version('transformers'),
        'gpu_name': 'synthetic-h100',
        'reference_loader_sha256': hashlib.sha256(inspect.getsource(load_model).encode()).hexdigest(),
        'tf32_allowed': False,
        'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    contract = {
        'calibrated_contract': {
            'runtime_identity': identity,
            'limits': budgets(),
            'frozen_limits_sha256': 'fixture'
        }
    }
    evaluation = Mock(
        side_effect=RuntimeError('reference failed') if failure else None,
        return_value={
            'positive_positions': 1,
            'negative_positions': 3,
            'positive_failures': [],
            'negative_failures': []
        },
    )
    monkeypatch.setattr(numerics, 'evaluate_group', evaluation)
    if failure:
        with pytest.raises(RuntimeError, match='reference failed'):
            numerics.run_reference(Path('fixture'), [1], contract)
    else:
        result = numerics.run_reference(Path('fixture'), [1], contract)
        assert result['correctness_passed'] is True and result['teardown_complete'] is True
    assert loaded == [torch.bfloat16, torch.float32]
    assert release.call_count == 2
    identity['gpu_name'] = 'different-hardware'
    with pytest.raises(ValueError, match='runtime or hardware'):
        numerics.run_reference(Path('fixture'), [1], contract)


@pytest.mark.parametrize('passed', [True, False])
def test_reference_cli_preserves_rejection_evidence(numerics, monkeypatch, tmp_path, passed):
    """A rejected report is saved but the process must still fail explicitly."""
    requests, output = tmp_path / 'requests.json', tmp_path / 'reference.json'
    requests.write_text('[]')
    monkeypatch.setattr(
        sys, 'argv', [
            'reference',
            '--checkpoint',
            str(tmp_path),
            '--requests',
            str(requests),
            '--output',
            str(output),
        ]
    )
    monkeypatch.setattr(numerics, 'load_contract', lambda: {'fixture': True})
    monkeypatch.setattr(numerics, 'run_reference', lambda *args: {'correctness_passed': passed})
    validate = Mock()
    monkeypatch.setattr(numerics, 'validate_reference_result', validate)
    if passed:
        numerics.main()
    else:
        with pytest.raises(RuntimeError, match='rejected'):
            numerics.main()
    assert json.loads(output.read_text())['correctness_passed'] is passed
    assert numerics.os.environ['NVIDIA_TF32_OVERRIDE'] == '0'
    assert validate.call_count == int(passed)


def serialized_reference(numerics):
    """Build complete control-flow evidence with explicit synthetic accepted/rejected flags."""
    windows = {}
    for offset in numerics.OFFSETS:
        positives, negatives = [], []
        for request in range(14):
            count = 4 if request < 6 else 64
            for step in range(count):
                positives.append({'request': request, 'step': step, 'accepted': True})
                negatives.extend(
                    [
                        {
                            'request': request,
                            'step': step,
                            'accepted': False,
                            'fault': fault
                        } for fault in ('half-normalization', 'temperature-two')
                    ]
                )
        negatives.append({'request': 0, 'step': 0, 'accepted': False, 'fault': 'last-request-context-for-first'})
        windows[str(offset)] = {
            'positive_positions': len(positives),
            'negative_positions': len(negatives),
            'positive_failures': [],
            'negative_failures': [],
            'all_positive_records': positives,
            'all_negative_records': negatives,
        }
    return {
        'correctness_passed': True,
        'teardown_complete': True,
        'numerical_policy': numerics.POLICY,
        'contract_sha256': numerics.CONTRACT_SHA256,
        'profile_offsets': list(numerics.OFFSETS),
        'request_count': 84,
        'positive_positions': 3216,
        'negative_positions': 6438,
        'windows': windows,
    }


@pytest.mark.parametrize(
    'fault', [
        'identity',
        'missing-window',
        'invalid-window',
        'missing-position',
        'accepted-fault',
        'failed-positive',
        'bad-index',
        'duplicate-position',
        'missing-step',
        'wrong-fault',
        'totals',
    ]
)
def test_serialized_reference_cannot_fabricate_complete_success(numerics, fault):
    """Validate all per-window records rather than trusting a summary flag."""
    report = serialized_reference(numerics)
    numerics.validate_reference_result(report)
    window = report['windows']['0']
    overrides = {'identity': ('contract_sha256', 'another-policy'), 'totals': ('positive_positions', 3215)}
    if fault in overrides:
        key, value = overrides[fault]
        report[key] = value
    elif fault == 'missing-window':
        report['windows'].pop('0')
    elif fault == 'invalid-window':
        report['windows']['0'] = None
    elif fault == 'missing-position':
        window['all_positive_records'].pop()
    elif fault == 'accepted-fault':
        window['all_negative_records'][0]['accepted'] = True
    elif fault == 'failed-positive':
        window['all_positive_records'][0]['accepted'] = False
    elif fault == 'bad-index':
        window['all_positive_records'][0]['step'] = True
    elif fault == 'duplicate-position':
        window['all_positive_records'][0] = dict(window['all_positive_records'][1])
    elif fault == 'missing-step':
        window['all_positive_records'][0]['step'] = 5
    else:
        window['all_negative_records'][-1]['fault'] = 'substituted-fault'
    with pytest.raises(ValueError):
        numerics.validate_reference_result(report)
