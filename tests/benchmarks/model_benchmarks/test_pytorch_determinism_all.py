# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unified test for deterministic fingerprinting across all major PyTorch model benchmarks."""

from tests.helper import decorator
import os
import pytest
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, ReturnCode

# Set CUBLAS_WORKSPACE_CONFIG early to ensure deterministic cuBLAS behavior
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
# Set PYTORCH_CUDA_ALLOC_CONF to avoid memory fragmentation
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')


def run_deterministic_benchmark(model_name, params):
    """Helper to launch a deterministic benchmark and return the result."""
    parameters = params + ' --enable_determinism --deterministic_seed 42 --check_frequency 10'
    context = BenchmarkRegistry.create_benchmark_context(
        model_name,
        platform=Platform.CUDA,
        parameters=parameters,
        framework=Framework.PYTORCH,
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    return benchmark


MODELS = [
    (
        'resnet18',
        '--batch_size 2 --image_size 32 --num_classes 2 --num_warmup 1 --num_steps 20 '
        '--model_action train --precision float32',
    ),
    (
        'lstm',
        '--batch_size 1 --num_classes 2 --seq_len 4 --num_warmup 1 --num_steps 20 '
        '--model_action train '
        '--precision float32',
    ),
    (
        'gpt2-small',
        '--batch_size 1 --num_classes 2 --seq_len 4 --num_warmup 1 --num_steps 20 '
        '--model_action train --precision float32',
    ),
    pytest.param(
        'llama2-7b',
        '--batch_size 1 --seq_len 1 --num_warmup 1 --num_steps 20 --precision float32 --model_action train',
        marks=pytest.mark.skip(
            reason='Requires >26GB GPU memory for 7B model, and float16 is incompatible with deterministic mode'
        ),
    ),
    (
        'mixtral-8x7b',
        '--batch_size 1 --seq_len 4 --num_warmup 1 --num_steps 20 --precision float32 '
        '--hidden_size 128 --max_position_embeddings 32 '
        '--intermediate_size 256 --model_action train',
    ),
    (
        'bert-base',
        '--batch_size 1 --num_classes 2 --seq_len 4 --num_warmup 1 --num_steps 20 '
        '--model_action train --precision float32',
    ),
]


@decorator.cuda_test
@decorator.pytorch_test
@pytest.mark.parametrize('model_name, params', MODELS)
def test_pytorch_model_determinism(model_name, params):
    """Parameterised Test for PyTorch model determinism.

    Tests that deterministic metrics (loss, activation mean) are correctly recorded
    when --enable_determinism is enabled. Comparison against baseline should be done
    offline using `sb result diagnosis`.
    """
    benchmark = run_deterministic_benchmark(model_name, params)
    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    # Check args
    assert benchmark._args.enable_determinism is True
    assert benchmark._args.deterministic_seed == 42
    assert benchmark._args.check_frequency == 10

    # Check that detailed per-step fingerprints are captured in _model_run_periodic
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict), '_model_run_periodic should be a dict'

    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic, f"Key '{key}' missing in _model_run_periodic, got keys: {list(periodic.keys())}"
        assert isinstance(periodic[key], list) and len(periodic[key]) > 0, \
            f"Expected non-empty list for periodic['{key}']"

    # Verify loss values are reasonable (not None or inf)
    import math
    for loss_val in periodic['loss']:
        assert loss_val is not None, 'Loss value should not be None'
        assert isinstance(loss_val, (int, float)), f'Loss should be numeric, got {type(loss_val)}'
        if not math.isnan(loss_val):
            assert loss_val < 1e6, f'Loss seems unreasonably large: {loss_val}'

    # Verify deterministic metrics are in result (summarized form)
    result = benchmark._result.result
    metric_keys = [k for k in result.keys() if 'deterministic_' in k]
    assert len(metric_keys) > 0, f'Expected deterministic metrics in result, got keys: {list(result.keys())}'

    # Verify configuration parameters are in results for validation
    config_keys = [k for k in result.keys() if 'deterministic_config_' in k]
    assert len(config_keys) > 0, 'Expected deterministic_config metrics in result'

    # Verify specific config values match the arguments
    # Result values are stored as lists, so compare against list-wrapped values
    assert result.get('deterministic_config_deterministic_seed') == [42], \
        'deterministic_seed config should match args'
    assert result.get('deterministic_config_check_frequency') == [10], \
        'check_frequency config should match args'
    assert 'deterministic_config_batch_size' in result, \
        'batch_size should be in config metrics'


@decorator.cuda_test
@decorator.pytorch_test
@pytest.mark.parametrize('model_name, params', MODELS)
def test_pytorch_model_nondeterministic_default(model_name, params):
    """Parameterised Test for PyTorch model to verify non-determinism is default."""
    context = BenchmarkRegistry.create_benchmark_context(
        model_name,
        platform=Platform.CUDA,
        parameters=params,
        framework=Framework.PYTORCH,
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert (benchmark and benchmark.return_code == ReturnCode.SUCCESS), 'Benchmark did not run successfully.'
    args = benchmark._args
    assert getattr(args, 'enable_determinism', False) is False, 'Expected enable_determinism to be False by default.'
    assert (getattr(args, 'check_frequency', None) == 100), 'Expected check_frequency to be 100 by default.'

    # Periodic fingerprints exist but are empty when not deterministic
    assert hasattr(benchmark, '_model_run_periodic'), 'Benchmark missing _model_run_periodic attribute.'
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict), '_model_run_periodic should be a dict.'
    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic, f"Key '{key}' missing in _model_run_periodic."
        assert (len(periodic[key]) == 0), f"Expected empty list for periodic['{key}'], got {periodic[key]}."
