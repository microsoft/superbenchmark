# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CPU adapter tests for the standalone complete-generation profile."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from superbench.benchmarks import BenchmarkType, ReturnCode
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
def test_invalid_profiles(parameters):
    """Reject unsupported precisions, dimensions and multi-profile timing."""
    instance = adapter.DeepSeekV2LiteInference('deepseek-v2-lite-inference', parameters)
    assert not instance.run()
    assert instance.return_code == ReturnCode.INVALID_ARGUMENT


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


def test_h100_policy_cannot_use_an_unqualified_shape(benchmark):
    """Keep calibration/profile compatibility fail-closed in the frontend."""
    instance = adapter.DeepSeekV2LiteInference('deepseek-v2-lite-inference', '--numerical_policy h100_bf16_v6')
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
