# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU adapter tests; complete model correctness is checked by the GPU worker."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from superbench.benchmarks import BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks import deepseek_v2_lite_cache as cache_cleanup
from superbench.benchmarks.model_benchmarks import pytorch_deepseek_v2_lite as adapter
from superbench.benchmarks.model_benchmarks.deepseek_v2_lite_utils import identity_report, write_json


@pytest.fixture
def benchmark(tmp_path, monkeypatch):
    """Use an isolated cache root and a single local launcher."""
    monkeypatch.delenv('WORLD_SIZE', raising=False)
    monkeypatch.delenv('OMPI_COMM_WORLD_SIZE', raising=False)
    return adapter.DeepSeekV2LiteTraining(
        'deepseek-v2-lite-training',
        '--cache_root "{}" --num_steps 2 --seq_len 64 --duration 60'.format(tmp_path),
    )


def successful_worker(command, log, timeout):
    """Supply a worker-contract fixture, not a claim of model execution."""
    output = Path(command[command.index('--output') + 1])
    dispatch = command[command.index('--moe_dispatch') + 1]
    report = dict(
        identity_report(),
        status='success',
        correctness_passed=True,
        teardown_complete=True,
        step_ms=[10.0, 20.0],
        loss=[2.0, 1.9],
        gradient_norm=[1.0, 0.9],
        loss_token_count=2 * 8 * 63,
        updated_parameter_elements=100,
        world_size=4,
        moe_dispatch=dispatch,
        moe_layers_rewritten=26 if dispatch == 'stable_sorted' else 0,
    )
    write_json(output, report)


def test_successful_adapter_metrics(benchmark, monkeypatch):
    """Preserve the MODEL result shape and count predicted loss tokens correctly."""
    monkeypatch.setattr(adapter, 'run_worker', successful_worker)
    assert benchmark.run()
    assert benchmark.type == BenchmarkType.MODEL
    assert benchmark.return_code == ReturnCode.SUCCESS
    assert benchmark.result['bf16_train_step_time'] == [15.0]
    assert benchmark.raw_data['bf16_train_step_time'] == [[10.0, 20.0]]
    assert benchmark.result['bf16_train_tokens_per_second'] == [2 * 8 * 63 * 1000 / 30]


@pytest.mark.parametrize(
    'parameters', [
        '--seq_len 0', '--num_gpus 1', '--batch_size 0', '--num_steps 0', '--num_warmup -1', '--learning_rate nan',
        '--duration 3600', '--run_count 2', '--hidden_size 1024', '--moe_dispatch unknown'
    ]
)
def test_invalid_profiles(tmp_path, parameters):
    """Reject invalid shapes, architecture overrides and combined timing profiles."""
    benchmark = adapter.DeepSeekV2LiteTraining(
        'deepseek-v2-lite-training', '--cache_root "{}" {}'.format(tmp_path, parameters)
    )
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


def test_cache_root_is_required():
    """Refuse to choose an implicit location for the per-run model cache."""
    benchmark = adapter.DeepSeekV2LiteTraining('deepseek-v2-lite-training', '--num_steps 2')
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


def test_parameters_render_in_help():
    """Keep every option compatible with the typed SuperBench help formatter."""
    benchmark = adapter.DeepSeekV2LiteTraining('deepseek-v2-lite-training')
    benchmark.add_parser_arguments()
    settings = benchmark.get_configurable_settings()
    assert '--moe_dispatch' in settings
    assert '--cache_root' in settings


@pytest.mark.parametrize('world_size', ['8', 'not-an-integer'])
def test_duplicate_launcher_is_rejected(benchmark, monkeypatch, world_size):
    """An MPI worker group must not launch eight independent full-model jobs."""
    monkeypatch.setenv('WORLD_SIZE', world_size)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


def test_download_or_worker_failure_is_not_success(benchmark, monkeypatch):
    """Reject a failed subprocess rather than inventing throughput."""
    def failed_worker(command, log, timeout):
        raise RuntimeError('download failed')

    monkeypatch.setattr(adapter, 'run_worker', failed_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR
    assert 'bf16_train_step_time' not in benchmark.result


def test_timeout_is_failure(benchmark, monkeypatch):
    """Finishing through watchdog termination does not meet the time limit."""
    def timed_out_worker(command, log, timeout):
        raise TimeoutError('whole job deadline')

    monkeypatch.setattr(adapter, 'run_worker', timed_out_worker)
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
        assert benchmark.result['bf16_train_tokens_per_second'] == [2 * 8 * 63 * 1000 / 30]
    else:
        assert benchmark.return_code == ReturnCode.KILLED_BY_TIMEOUT
        assert 'end_to_end_seconds' not in benchmark.result
        assert 'bf16_train_tokens_per_second' not in benchmark.result


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


@pytest.mark.parametrize(
    'field,value', [
        ('step_ms', [-1, 20]), ('step_ms', [10]), ('correctness_passed', False), ('teardown_complete', False),
        ('updated_parameter_elements', True), ('loss_token_count', 1024), ('moe_dispatch', None),
        ('moe_dispatch', 'stable_sorted'), ('moe_layers_rewritten', False), ('moe_layers_rewritten', 26)
    ]
)
def test_invalid_worker_report(benchmark, monkeypatch, field, value):
    """Check required metric thresholds and shapes, not only process exit zero."""
    def invalid_worker(command, log, timeout):
        successful_worker(command, log, timeout)
        output = Path(command[command.index('--output') + 1])
        report = adapter.read_json(output)
        report[field] = value
        write_json(output, report)

    monkeypatch.setattr(adapter, 'run_worker', invalid_worker)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.RUNTIME_EXCEPTION_ERROR


def test_python_and_path_remain_separate_arguments(benchmark):
    """Never concatenate runtime paths or forward unchecked model flags."""
    assert benchmark._preprocess()
    benchmark._args.runtime_python = '/path with spaces/python'
    command = benchmark._worker_command(Path('/cache with spaces/job'))
    assert command[0] == '/path with spaces/python'
    assert command[command.index('--cache_dir') + 1] == '/cache with spaces/job/cache'
    assert '--nproc_per_node=4' in command
    assert command[command.index('--moe_dispatch') + 1] == 'official'


def test_opt_in_dispatch_reaches_the_worker_and_requires_matching_evidence(benchmark, tmp_path, monkeypatch):
    """Thread the explicit dispatch selection through the real adapter contract."""
    selected = adapter.DeepSeekV2LiteTraining(
        'deepseek-v2-lite-training',
        '--cache_root "{}" --num_steps 2 --seq_len 64 --duration 60 --moe_dispatch stable_sorted'.format(tmp_path),
    )
    commands = []

    def worker(command, log, timeout):
        commands.append(command)
        successful_worker(command, log, timeout)

    monkeypatch.setattr(adapter, 'run_worker', worker)
    assert selected.run()
    assert commands[0][commands[0].index('--moe_dispatch') + 1] == 'stable_sorted'
