# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPT2 model benchmarks."""

import os
import logging
import json
import tempfile
import torch
import pytest

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_gpt2 import PytorchGPT2


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_gpt2_small():
    """Test pytorch-gpt2-small benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'gpt2-small',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchGPT2))
    assert (benchmark.name == 'pytorch-gpt2-small')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of gpt2-large model.
    assert (benchmark._args.hidden_size == 768)
    assert (benchmark._args.num_hidden_layers == 12)
    assert (benchmark._args.num_attention_heads == 12)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 8)
    assert (benchmark._args.num_warmup == 2)
    assert (benchmark._args.num_steps == 4)

    # Test Dataset.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in [
        'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
        'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time', 'fp16_inference_throughput'
    ]:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_gpt2_periodic_and_logging_combined(caplog, monkeypatch):
    """Verify periodic fingerprint logs, in-memory recording, and log-file generation in a single run.

    - Enables strict determinism envs if CUDA not initialized (optional).
    - Runs with --deterministic --random_seed 42 and num_steps=100 to hit cadence at step 100.
    - Enables --generate-log with a temp path; validates file contents and in-memory bookkeeping.
    - Confirms INFO logs contain Loss/ActMean at step 100.
    """

    # Ensure cuBLAS deterministic workspace is set before first CUDA init
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    caplog.set_level(logging.INFO, logger='superbench')

    log_path = tempfile.mktemp(suffix='.json')
    parameters = (
        '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 100 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train '
        f'--generate-log --log-path {log_path} --check_frequency 10'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'gpt2-small', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    try:
        assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

        # Check determinism/logging args
        assert benchmark._args.deterministic is True
        assert benchmark._args.random_seed == 42
        assert benchmark._args.generate_log is True
        assert benchmark._args.check_frequency == 10

        # Expect Loss/ActMean logs at step 100
        messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
        assert any(f'Loss at step {benchmark._args.check_frequency}:' in m for m in messages)
        assert any(f'ActMean at step {benchmark._args.check_frequency}:' in m for m in messages)

        # In-memory recording
        assert hasattr(benchmark, '_model_run_losses') and isinstance(benchmark._model_run_losses, list)
        assert len(benchmark._model_run_losses) > 0
        assert hasattr(benchmark, '_model_run_periodic') and isinstance(benchmark._model_run_periodic, dict)
        periodic = benchmark._model_run_periodic
        for key in ('loss', 'act_mean', 'step'):
            assert key in periodic
        assert len(periodic['loss']) > 0
        assert len(periodic['act_mean']) > 0
        assert len(periodic['step']) > 0

        # Log-file generation and contents
        assert os.path.exists(log_path)
        with open(log_path, 'r') as f:
            data = json.load(f)
        assert 'schema_version' in data
        assert 'metadata' in data
        assert 'per_step_fp32_loss' in data and isinstance(data['per_step_fp32_loss'], list)
        assert 'fingerprints' in data and isinstance(data['fingerprints'], dict)
        # Optional: verify step 100 present if any steps recorded
        fp = data['fingerprints']
        if 'step' in fp and isinstance(fp['step'], list) and len(fp['step']) > 0:
            assert 100 in fp['step']
            assert len(fp.get('loss', [])) == len(fp['step'])
            assert len(fp.get('act_mean', [])) == len(fp['step'])
    finally:
        if os.path.exists(log_path):
            os.remove(log_path)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_gpt2_nondeterministic_defaults():
    """Run without determinism/logging flags and assert defaults are unset and periodic is empty."""
    parameters = (
       '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
        --model_action train inference'
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'gpt2-small', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
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
