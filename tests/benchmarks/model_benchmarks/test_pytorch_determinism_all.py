# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unified test for deterministic fingerprinting across all major PyTorch model benchmarks."""

import sys
import os
import tempfile
import json
import pytest
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, ReturnCode
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

MODELS = [
    ('resnet18', '--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 --model_action train inference'),
    ('lstm', '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
    --model_action train inference --precision float32'),
    ('gpt2-large', '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 --model_action train inference'),
    ('llama2-7b', '--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 --model_action train inference'),
    ('mixtral-8x7b', '--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 --hidden_size 1024 --max_position_embeddings 2048 --intermediate_size 3584 --model_action train inference'),
    ('bert-large', '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 --model_action train inference'),
]

@pytest.mark.parametrize('model_name, params', MODELS)
def test_pytorch_model_determinism(model_name, params):
    print("**********", model_name)

    log_path = tempfile.mktemp(suffix='.json')
    parameters = params + f' --deterministic --random_seed 42 --generate-log --log-path {log_path} --check_frequency 10'
    context = BenchmarkRegistry.create_benchmark_context(
        model_name,
        platform=Platform.CUDA,
        parameters=parameters,
        framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    # Check args
    assert benchmark._args.deterministic is True
    assert getattr(benchmark._args, 'generate_log', False) is True
    assert benchmark._args.deterministic is True
    assert benchmark._args.random_seed == 42
    assert benchmark._args.check_frequency == 10

    # Log-file generation and contents
    assert os.path.exists(log_path)
    with open(log_path, 'r') as f:
        data = json.load(f)
    assert 'schema_version' in data
    assert 'metadata' in data
    assert 'per_step_fp32_loss' in data
    assert 'fingerprints' in data
    assert isinstance(data['per_step_fp32_loss'], list)
    assert isinstance(data['fingerprints'], dict)

    # Clean up
    os.remove(log_path)

@pytest.mark.parametrize('model_name, params', MODELS)
def test_pytorch_model_nondeterministoc_default(model_name, params):

    context = BenchmarkRegistry.create_benchmark_context(
        model_name, platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH
    )

    benchmark = BenchmarkRegistry.launch_benchmark(context)
    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS
    args = benchmark._args
    assert args.deterministic is False
    assert getattr(args, 'generate_log', False) is False
    assert getattr(args, 'log_path', None) is None
    assert getattr(args, 'compare_log', None) is None
    assert getattr(args, 'check_frequency', None) is 100

    # Periodic fingerprints exist but are empty when not deterministic
    assert hasattr(benchmark, '_model_run_periodic')
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict)
    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic
        assert len(periodic[key]) == 0
    pass
