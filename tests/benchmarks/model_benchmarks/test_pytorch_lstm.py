# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for LSTM model benchmarks."""

import os
import logging
import json
import tempfile
import numpy as np
import pytest
import torch

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_lstm import PytorchLSTM


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_lstm_with_gpu():
    """Test pytorch-lstm benchmark with GPU."""
    run_pytorch_lstm(
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
            'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time',
            'fp16_inference_throughput'
        ]
    )


@decorator.pytorch_test
def test_pytorch_lstm_no_gpu():
    """Test pytorch-lstm benchmark with CPU."""
    run_pytorch_lstm(
        parameters='--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 4 \
            --model_action train inference --precision float32 --no_gpu',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp32_inference_step_time', 'fp32_inference_throughput'
        ]
    )


def run_pytorch_lstm(parameters='', check_metrics=[]):
    """Test pytorch-lstm benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'lstm', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchLSTM))
    assert (benchmark.name == 'pytorch-lstm')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of lstm model.
    assert (benchmark._args.input_size == 256)
    assert (benchmark._args.hidden_size == 1024)
    assert (benchmark._args.num_layers == 8)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.num_classes == 5)
    assert (benchmark._args.seq_len == 8)
    assert (benchmark._args.num_warmup == 2)
    assert (benchmark._args.num_steps == 4)

    # Check dataset scale.
    assert (len(benchmark._dataset) == benchmark._args.sample_count * benchmark._world_size)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for metric in check_metrics:
        assert (len(benchmark.raw_data[metric]) == benchmark.run_count)
        assert (len(benchmark.raw_data[metric][0]) == benchmark._args.num_steps)
        assert (len(benchmark.result[metric]) == benchmark.run_count)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_lstm_periodic_fingerprint_logging(caplog):
    """Verify periodic fingerprints, in-memory recording, and log-file generation together."""
    # Ensure deterministic cuBLAS workspace is set before first CUDA init (best-effort)
    if torch.cuda.is_available() and not torch.cuda.is_initialized():
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    caplog.set_level(logging.INFO, logger='superbench')

    log_path = tempfile.mktemp(suffix='.json')
    parameters = (
        '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 1 --num_steps 100 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train '
        f'--generate-log --log-path {log_path} --check_frequency 10'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'lstm', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
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
def test_pytorch_lstm_nondeterministic_defaults():
    """Run without determinism/logging flags and assert defaults are unset and periodic is empty."""
    parameters = (
        '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 2 --num_steps 5 '
        '--precision float32 --model_action train'
    )
    context = BenchmarkRegistry.create_benchmark_context(
        'lstm', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
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

