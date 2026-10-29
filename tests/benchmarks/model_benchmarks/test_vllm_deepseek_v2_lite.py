# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU adapter tests for the standalone complete-generation profile."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks import deepseek_v2_lite_cache as cache_cleanup
from superbench.benchmarks.model_benchmarks import vllm_deepseek_v2_lite as adapter
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite_utils import identity_report, write_json


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    """Use one local replica and an isolated output root."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    return adapter.DeepSeekV2LiteInference(
        'deepseek-v2-lite-inference',
        '--cache_root "{}" --num_steps 2 --batch_size 2 --seq_len 64 --duration 60'.format(tmp_path),
    )


def successful_worker(command, log, timeout):
    """Provide validated-shaped fixtures, not actual GPU measurements."""
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
    def invalid_worker(command, log, timeout):
        successful_worker(command, log, timeout)
        output = Path(command[command.index('--output') + 1])
        report = adapter.read_json(output)
        report[field] = value
        write_json(output, report)

    monkeypatch.setattr(adapter, 'run_worker', invalid_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR


def test_deadline_failure(benchmark, monkeypatch):
    """A watchdog exit cannot be reported as a successful under-hour run."""
    def timeout(command, log, seconds):
        raise TimeoutError('whole job deadline')

    monkeypatch.setattr(adapter, 'run_worker', timeout)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT


@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout'])
def test_model_cache_is_removed_after_every_run(benchmark, monkeypatch, outcome):
    """Delete the per-run checkpoint cache on every exit path while keeping reports and logs."""
    jobs = []

    def worker(command, log, timeout):
        cache = Path(command[command.index('--cache_dir') + 1])
        (cache / 'checkpoint').mkdir(parents=True)
        (cache / 'checkpoint' / 'model-00001-of-000004.safetensors').write_bytes(b'weights')
        log.write_text('worker log\n')
        jobs.append(cache.parent)
        if outcome == 'failure':
            raise RuntimeError('worker failed')
        if outcome == 'timeout':
            raise TimeoutError('whole job deadline')
        successful_worker(command, log, timeout)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    assert benchmark.run() is (outcome == 'success')
    assert not (jobs[0] / 'cache').exists()
    assert (jobs[0] / 'worker.log').read_text() == 'worker log\n'
    assert (jobs[0] / 'report.json').is_file() is (outcome == 'success')


@pytest.mark.parametrize(
    'failure',
    [PermissionError('read-only cache'),
     subprocess.CalledProcessError(1, ['cleanup'], output='read-only cache')]
)
def test_cache_removal_failure_keeps_a_valid_measurement(benchmark, monkeypatch, failure):
    """Log a cleanup error without discarding metrics that were already validated."""
    caches = []
    log_error = Mock()

    def worker(command, log, timeout):
        cache = Path(command[command.index('--cache_dir') + 1])
        cache.mkdir(parents=True)
        caches.append(cache)
        successful_worker(command, log, timeout)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', Mock(side_effect=failure))
    monkeypatch.setattr(adapter.logger, 'error', log_error)
    assert benchmark.run()
    assert benchmark.return_code == ReturnCode.SUCCESS
    assert caches[0].is_dir()
    log_error.assert_called_once()
    assert 'read-only cache' in str(log_error.call_args[0][-1])


def test_sigterm_during_cache_removal_still_reports_timeout(benchmark, monkeypatch):
    """A runner timeout during cleanup must not be reported as a completed run."""
    def worker(command, log, timeout):
        Path(command[command.index('--cache_dir') + 1]).mkdir(parents=True)
        successful_worker(command, log, timeout)

    cleanup = Mock(side_effect=TimeoutError('Killed by SIGTERM or timeout!'))
    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', cleanup)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT
    assert 'end_to_end_seconds' not in benchmark.result
    assert cleanup.call_count == 2


@pytest.mark.parametrize('cleanup_seconds', [7, 50, 70])
def test_cache_cleanup_counts_toward_the_job_deadline(benchmark, monkeypatch, cleanup_seconds):
    """Include cleanup in elapsed time and reject the exact deadline as well as an overrun."""
    clock = [100.0]
    run_cleanup = subprocess.run

    def worker(command, log, timeout):
        assert timeout == 40
        Path(command[command.index('--cache_dir') + 1]).mkdir(parents=True)
        successful_worker(command, log, timeout)
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


def test_cleanup_after_the_deadline_uses_the_failure_grace(benchmark, monkeypatch):
    """Fail an overrun job, then still remove its cache within the bounded failure grace."""
    clock = [100.0]
    timeouts = []
    caches = []
    run_cleanup = subprocess.run

    def worker(command, log, timeout):
        cache = Path(command[command.index('--cache_dir') + 1])
        cache.mkdir(parents=True)
        caches.append(cache)
        successful_worker(command, log, timeout)
        clock[0] += 60

    def cleanup(command, **kwargs):
        timeouts.append(kwargs['timeout'])
        return run_cleanup(command, **kwargs)

    fake_time = SimpleNamespace(monotonic=lambda: clock[0])
    monkeypatch.setattr(adapter, 'time', fake_time)
    monkeypatch.setattr(cache_cleanup, 'time', fake_time)
    monkeypatch.setattr(adapter, 'run_worker', worker)
    monkeypatch.setattr(cache_cleanup.subprocess, 'run', cleanup)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT
    assert 'end_to_end_seconds' not in benchmark.result
    assert timeouts == [cache_cleanup.FAILURE_CLEANUP_GRACE_SECONDS]
    assert not caches[0].exists()


@pytest.mark.parametrize(
    'failure,code', [
        (RuntimeError('worker failed'), ReturnCode.RUNTIME_EXCEPTION_ERROR),
        (TimeoutError('whole job deadline'), ReturnCode.KILLED_BY_TIMEOUT),
    ]
)
def test_unfinished_failure_cleanup_keeps_the_original_error(benchmark, monkeypatch, failure, code):
    """Report the worker failure rather than a later cleanup timeout, and log both."""
    log_error = Mock()

    def worker(command, log, timeout):
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

    def worker(command, log, timeout):
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
