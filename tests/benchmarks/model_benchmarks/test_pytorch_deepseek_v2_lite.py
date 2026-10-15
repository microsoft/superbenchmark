# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU adapter tests; complete model correctness is checked by the GPU worker."""

from pathlib import Path

import pytest

from superbench.benchmarks import BenchmarkType, ReturnCode
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
    report = dict(
        identity_report(), status='success', correctness_passed=True, teardown_complete=True,
        step_ms=[10.0, 20.0], loss=[2.0, 1.9], gradient_norm=[1.0, 0.9],
        loss_token_count=2 * 8 * 63, updated_parameter_elements=100, world_size=4,
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
    'parameters',
    ['--seq_len 0', '--num_gpus 1', '--batch_size 0', '--num_steps 0', '--num_warmup -1',
     '--learning_rate nan', '--duration 3600', '--run_count 2', '--hidden_size 1024']
)
def test_invalid_profiles(parameters):
    """Reject invalid shapes, architecture overrides and combined timing profiles."""
    benchmark = adapter.DeepSeekV2LiteTraining('deepseek-v2-lite-training', parameters)
    assert not benchmark.run()
    assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT


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


@pytest.mark.parametrize(
    'field,value',
    [('step_ms', [-1, 20]), ('step_ms', [10]), ('correctness_passed', False),
     ('teardown_complete', False), ('updated_parameter_elements', True), ('loss_token_count', 1024)]
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
