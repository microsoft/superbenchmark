# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for LSTM model benchmarks."""

import os
import logging
import numpy as np
import pytest

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
    """Emit Loss and ActMean logs at the periodic cadence under deterministic training."""
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

    caplog.set_level(logging.INFO, logger='superbench')

    parameters = (
        '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 1 --num_steps 100 '
        '--precision float32 --deterministic --random_seed 42 --model_action train'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'lstm', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
    assert any('Loss at step 100:' in m for m in messages)
    assert any('ActMean at step 100:' in m for m in messages)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_lstm_soft_determinism():
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    params = (
        '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 1 --num_steps 2 '
        '--precision float32 --deterministic --random_seed 42 --model_action train'
    )
    ctx1 = BenchmarkRegistry.create_benchmark_context('lstm', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b1 = BenchmarkRegistry.launch_benchmark(ctx1)
    ctx2 = BenchmarkRegistry.create_benchmark_context('lstm', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b2 = BenchmarkRegistry.launch_benchmark(ctx2)
    assert b1 and b2 and b1.return_code == ReturnCode.SUCCESS and b2.return_code == ReturnCode.SUCCESS
    m_loss = 'fp32_train_loss'
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.allclose(a1, a2, rtol=1e-6, atol=1e-7)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_lstm_strict_determinism():
    """Strict determinism: exact per-step loss equality under strict envs.

    This test verifies the strongest reproducibility guarantee: with strict determinism
    enabled and a fixed seed, two runs must produce identical fp32 per-step training
    losses (bitwise equality).

    Requirements and behavior:
    - Environment must be set before CUDA init: SB_STRICT_DETERMINISM=1 and
        CUBLAS_WORKSPACE_CONFIG (":4096:8" or ":16:8").
    - If these envs are not present, the test is skipped to avoid false failures.
    - The benchmark is invoked with --deterministic and --random_seed 42.
    - We compare the raw_data metric 'fp32_train_loss' via np.array_equal.

    Rationale:
    - Strict mode enforces deterministic kernels (warn_only=False) and will error if any
        nondeterministic op is used, ensuring reproducible numerics beyond soft determinism.
    """
    if os.environ.get('SB_STRICT_DETERMINISM') != '1' or 'CUBLAS_WORKSPACE_CONFIG' not in os.environ:
        pytest.skip('Strict determinism env not set; skipping test.')
    params = (
        '--batch_size 1 --num_classes 5 --seq_len 8 --num_warmup 1 --num_steps 2 '
        '--precision float32 --deterministic --random_seed 42 --model_action train'
    )
    ctx1 = BenchmarkRegistry.create_benchmark_context('lstm', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b1 = BenchmarkRegistry.launch_benchmark(ctx1)
    ctx2 = BenchmarkRegistry.create_benchmark_context('lstm', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b2 = BenchmarkRegistry.launch_benchmark(ctx2)
    assert b1 and b2 and b1.return_code == ReturnCode.SUCCESS and b2.return_code == ReturnCode.SUCCESS
    m_loss = 'fp32_train_loss'
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.array_equal(a1, a2)

