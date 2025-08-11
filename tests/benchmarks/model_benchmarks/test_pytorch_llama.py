# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Llama model benchmarks."""

import os
import pytest
import torch
import numpy as np
import logging
from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode, Precision
from superbench.benchmarks.model_benchmarks.pytorch_llama import PytorchLlama

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_7b():
    """Test pytorch-llama2-7b benchmark for fp16 train and inference."""
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 \
            --model_action train inference',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (isinstance(benchmark, PytorchLlama))
    assert (benchmark.name == 'pytorch-llama2-7b')
    assert (benchmark.type == BenchmarkType.MODEL)

    # Check predefined parameters of llama2 7b model.
    assert (benchmark._args.hidden_size == 4096)
    assert (benchmark._args.num_hidden_layers == 32)
    assert (benchmark._args.num_attention_heads == 32)

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
def test_pytorch_llama_periodic_checksum_logging(caplog):
    """Emit checksum log at the periodic cadence when deterministic training is enabled.

    This test ensures that when deterministic training is enabled (but strict mode is off),
    the periodic checksum logging is triggered at the expected cadence.

    - Strict mode envs are explicitly unset to test only the periodic checksum behavior.
    - The benchmark is run with --deterministic and --random_seed 42.
    - We expect a checksum log at step 100 (cadence = 100).
    """
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

    caplog.set_level(logging.INFO, logger='superbench')

    parameters = (
        '--hidden_size 128 --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 4 '
        '--intermediate_size 512 --batch_size 1 --seq_len 16 --num_warmup 1 --num_steps 100 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    assert benchmark and benchmark.return_code == ReturnCode.SUCCESS

    # Expect one checksum log at step 100 (cadence = 100)
    messages = [rec.getMessage() for rec in caplog.records if rec.name == 'superbench']
    assert any('Checksum at step 100:' in m for m in messages)

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_soft_determinism():
    """Soft determinism: losses should be numerically close across runs without strict envs.

    This test checks that with deterministic training enabled (but strict mode envs unset),
    two runs produce numerically close (but not necessarily bitwise identical) fp32 per-step
    training losses.

    - Strict mode envs are explicitly unset to test only soft determinism.
    - The benchmark is run with --deterministic and --random_seed 42.
    - We compare the raw_data metric 'fp32_train_loss' via np.allclose.
    """
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

    params = (
        '--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 '
        '--num_key_value_heads 4 --intermediate_size 1024 --batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train'
    )

    ctx1 = BenchmarkRegistry.create_benchmark_context('llama2-7b', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b1 = BenchmarkRegistry.launch_benchmark(ctx1)
    ctx2 = BenchmarkRegistry.create_benchmark_context('llama2-7b', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b2 = BenchmarkRegistry.launch_benchmark(ctx2)

    assert b1 and b2 and b1.return_code == ReturnCode.SUCCESS and b2.return_code == ReturnCode.SUCCESS

    m_loss = 'fp32_train_loss'
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.allclose(a1, a2, rtol=1e-6, atol=1e-7)

@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_strict_deterministic_training():
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
        '--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 '
        '--num_key_value_heads 4 --intermediate_size 1024 --batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train'
    )

    ctx1 = BenchmarkRegistry.create_benchmark_context('llama2-7b', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b1 = BenchmarkRegistry.launch_benchmark(ctx1)
    ctx2 = BenchmarkRegistry.create_benchmark_context('llama2-7b', platform=Platform.CUDA, parameters=params, framework=Framework.PYTORCH)
    b2 = BenchmarkRegistry.launch_benchmark(ctx2)

    assert b1 and b2 and b1.return_code == ReturnCode.SUCCESS and b2.return_code == ReturnCode.SUCCESS

    m_loss = 'fp32_train_loss'
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.array_equal(a1, a2)
