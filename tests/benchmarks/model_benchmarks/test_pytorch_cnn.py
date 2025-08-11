# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for CNN model benchmarks."""

import os
import logging
import numpy as np
import pytest

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.model_benchmarks.pytorch_cnn import PytorchCNN


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn_with_gpu():
    """Test pytorch cnn benchmarks with GPU."""
    run_pytorch_cnn(
        models=['resnet50', 'resnet101', 'resnet152', 'densenet169', 'densenet201', 'vgg11', 'vgg13', 'vgg16', 'vgg19'],
        parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
            --model_action train inference',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp16_train_step_time', 'fp16_train_throughput',
            'fp32_inference_step_time', 'fp32_inference_throughput', 'fp16_inference_step_time',
            'fp16_inference_throughput'
        ]
    )


@decorator.pytorch_test
def test_pytorch_cnn_no_gpu():
    """Test pytorch cnn benchmarks with CPU."""
    run_pytorch_cnn(
        models=['resnet50'],
        parameters='--batch_size 1 --image_size 224 --num_classes 5 --num_warmup 2 --num_steps 4 \
                --model_action train inference --precision float32 --no_gpu',
        check_metrics=[
            'fp32_train_step_time', 'fp32_train_throughput', 'fp32_inference_step_time', 'fp32_inference_throughput'
        ]
    )


def run_pytorch_cnn(models=[], parameters='', check_metrics=[]):
    """Run pytorch cnn benchmarks."""
    for model in models:
        context = BenchmarkRegistry.create_benchmark_context(
            model, platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
        )

        assert (BenchmarkRegistry.is_benchmark_context_valid(context))

        benchmark = BenchmarkRegistry.launch_benchmark(context)

        # Check basic information.
        assert (benchmark)
        assert (isinstance(benchmark, PytorchCNN))
        assert (benchmark.name == 'pytorch-' + model)
        assert (benchmark.type == BenchmarkType.MODEL)

        # Check predefined parameters of resnet101 model.
        assert (benchmark._args.model_type == model)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.batch_size == 1)
        assert (benchmark._args.image_size == 224)
        assert (benchmark._args.num_classes == 5)
        assert (benchmark._args.num_warmup == 2)
        assert (benchmark._args.num_steps == 4)

        # Check Dataset.
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
def test_pytorch_cnn_periodic_checksum_logging(caplog):
    """Emit checksum log at the periodic cadence when deterministic training is enabled."""
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

    caplog.set_level(logging.INFO, logger='superbench')

    # Use a relatively small model for speed
    parameters = (
        '--batch_size 1 --image_size 64 --num_classes 5 --num_warmup 1 --num_steps 100 '
        '--precision float32 --deterministic --random_seed 42 --model_action train'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'resnet18', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
    assert any('Checksum at step 100:' in m for m in messages)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn_soft_determinism():
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    params = (
        '--batch_size 1 --image_size 64 --num_classes 5 --num_warmup 1 --num_steps 2 '
        '--precision float32 --deterministic --random_seed 42 --model_action train'
    )
    ctx1 = BenchmarkRegistry.create_benchmark_context('resnet18', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b1 = BenchmarkRegistry.launch_benchmark(ctx1)
    ctx2 = BenchmarkRegistry.create_benchmark_context('resnet18', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b2 = BenchmarkRegistry.launch_benchmark(ctx2)
    assert b1 and b2 and b1.return_code == ReturnCode.SUCCESS and b2.return_code == ReturnCode.SUCCESS
    m_loss = 'fp32_train_loss'
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.allclose(a1, a2, rtol=1e-6, atol=1e-7)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_cnn_strict_determinism():
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
        '--batch_size 1 --image_size 64 --num_classes 5 --num_warmup 1 --num_steps 2 '
        '--precision float32 --deterministic --random_seed 42 --model_action train'
    )
    ctx1 = BenchmarkRegistry.create_benchmark_context('resnet18', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b1 = BenchmarkRegistry.launch_benchmark(ctx1)
    ctx2 = BenchmarkRegistry.create_benchmark_context('resnet18', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b2 = BenchmarkRegistry.launch_benchmark(ctx2)
    assert b1 and b2 and b1.return_code == ReturnCode.SUCCESS and b2.return_code == ReturnCode.SUCCESS
    m_loss = 'fp32_train_loss'
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.array_equal(a1, a2)
