# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mixtral model benchmarks."""

import sys
import os
import logging
import numpy as np
import tempfile
import json
import pytest

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode

# Check for Python version 3.8 or greater and conditionally import PytorchMixtral
if sys.version_info >= (3, 8):
    from superbench.benchmarks.model_benchmarks.pytorch_mixtral import PytorchMixtral


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_mixtral_8x7b():
    """Test pytorch-mixtral-8x7b benchmark for float16 train and inference."""
    context = BenchmarkRegistry.create_benchmark_context(
        'mixtral-8x7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 \
            --hidden_size 1024 --max_position_embeddings 2048 --intermediate_size 3584 \
            --model_action train inference',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchMixtral))
    assert (benchmark.name == 'pytorch-mixtral-8x7b')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of mixtral-8x7b model.
    assert (benchmark._args.hidden_size == 1024)
    assert (benchmark._args.num_hidden_layers == 32)
    assert (benchmark._args.num_attention_heads == 32)
    assert (benchmark._args.num_key_value_heads == 8)
    assert (benchmark._args.intermediate_size == 3584)
    assert (benchmark._args.max_position_embeddings == 2048)
    assert (benchmark._args.router_aux_loss_coef == 0.02)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 100)
    assert (benchmark._args.seq_len == 32)
    assert (benchmark._args.num_warmup == 1)
    assert (benchmark._args.num_steps == 2)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    for metric in [
        'fp16_train_step_time', 'fp16_train_throughput', 'fp16_inference_step_time', 'fp16_inference_throughput'
    ]:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_mixtral_periodic_and_logging_combined(caplog, monkeypatch):
    """Single run to verify periodic fingerprint logs, in-memory recording, and log-file generation."""
    if sys.version_info < (3, 8):
        return
    # Enable strict determinism if possible (must be before first CUDA init)
    try:
        import torch
        if torch.cuda.is_available() and not torch.cuda.is_initialized():
            monkeypatch.setenv('SB_STRICT_DETERMINISM', '1')
            monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    except Exception:
        pass

    caplog.set_level(logging.INFO, logger='superbench')

    log_path = tempfile.mktemp(suffix='.json')
    parameters = (
        '--hidden_size 1024 --num_hidden_layers 2 --num_attention_heads 8 --num_key_value_heads 4 '
        '--intermediate_size 2048 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 100 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train '
        f'--generate-log --log-path {log_path}'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'mixtral-8x7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    try:
        assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

        # Check determinism/logging args
        assert benchmark._args.deterministic is True
        assert benchmark._args.random_seed == 42
        assert getattr(benchmark._args, 'generate_log', False) is True

        # Expect one loss and one activation fingerprint log at step 100 (cadence = 100)
        messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
        assert any('Loss at step 100:' in m for m in messages)
        assert any('ActMean at step 100:' in m for m in messages)

        # In-memory records
        assert hasattr(benchmark, '_model_run_losses')
        assert isinstance(benchmark._model_run_losses, list)
        assert len(benchmark._model_run_losses) > 0

        assert hasattr(benchmark, '_model_run_periodic')
        periodic = benchmark._model_run_periodic
        assert isinstance(periodic, dict)
        assert 'loss' in periodic and 'act_mean' in periodic and 'step' in periodic
        assert len(periodic['loss']) > 0
        assert len(periodic['act_mean']) > 0
        assert len(periodic['step']) > 0

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
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_mixtral_nondeterministic_defaults():
    """Run in normal (non-deterministic) mode and assert new params are unset."""
    if sys.version_info < (3, 8):
        return
    parameters = (
        '--hidden_size 1024 --num_hidden_layers 2 --num_attention_heads 8 --num_key_value_heads 4 '
        '--intermediate_size 2048 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 5 '
        '--precision float32 --sample_count 2 --model_action train'
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'mixtral-8x7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS
    args = benchmark._args
    assert args.deterministic is False
    assert getattr(args, 'generate_log', False) is False
    assert getattr(args, 'log_path', None) is None
    assert getattr(args, 'compare_log', None) is None

    assert hasattr(benchmark, '_model_run_periodic')
    periodic = benchmark._model_run_periodic
    assert isinstance(periodic, dict)
    for key in ('loss', 'act_mean', 'step'):
        assert key in periodic
        assert len(periodic[key]) == 0


## Strict determinism test removed to align with Llama tests
