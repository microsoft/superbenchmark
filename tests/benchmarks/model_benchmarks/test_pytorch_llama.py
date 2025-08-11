# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Llama model benchmarks."""

import os
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
def test_pytorch_llama_deterministic_training():
    """Test pytorch-llama2-7b benchmark with deterministic training enabled."""
    # Test that deterministic training parameters work and don't cause crashes

    parameters = '--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 4 --intermediate_size 1024 --batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train'
    # Run twice with the same seed and deterministic flag using the registry
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters=parameters,
        framework=Framework.PYTORCH
    )
    benchmark1 = BenchmarkRegistry.launch_benchmark(context)

    context2 = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters=parameters,
        framework=Framework.PYTORCH
    )
    benchmark2 = BenchmarkRegistry.launch_benchmark(context2)

    # Check that the run succeeded (basic checks)
    assert (benchmark1)
    assert (benchmark2)
    assert (isinstance(benchmark1, PytorchLlama))
    assert (isinstance(benchmark2, PytorchLlama))
    assert (benchmark1._args.deterministic == True)
    assert (benchmark2._args.deterministic == True)
    assert (benchmark1._args.random_seed == 42)
    assert (benchmark2._args.random_seed == 42)

    # Validate time metrics exist and shapes are correct (but don't require equality due to scheduler/async noise)
    m_time = 'fp32_train_step_time'
    assert m_time in benchmark1.raw_data and m_time in benchmark2.raw_data
    assert len(benchmark1.raw_data[m_time]) == benchmark1.run_count
    assert len(benchmark2.raw_data[m_time]) == benchmark2.run_count
    assert len(benchmark1.raw_data[m_time][0]) == benchmark1._args.num_steps
    assert len(benchmark2.raw_data[m_time][0]) == benchmark2._args.num_steps

    # Strict determinism check: compare per-step loss when strict mode + cuBLAS determinism are enabled
    m_loss = 'fp32_train_loss'
    assert m_loss in benchmark1.raw_data and m_loss in benchmark2.raw_data
    a1 = np.array(benchmark1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(benchmark2.raw_data[m_loss][0], dtype=float)
    # Require numeric (finite) values and exact equality
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.array_equal(a1, a2)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_non_deterministic_training():
    """Test pytorch-llama2-7b benchmark with non-deterministic training (default behavior)."""
    # Test that non-deterministic training works as expected

    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
    parameters='--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 4 --intermediate_size 1024 --batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 --model_action train',
        framework=Framework.PYTORCH
    )
    benchmark = BenchmarkRegistry.launch_benchmark(context)
    # Check that deterministic is disabled by default
    assert (benchmark._args.deterministic == False)
    assert (benchmark._args.random_seed == 42)  # Default value


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_soft_determinism():
    """Test soft determinism: deterministic=True without strict envs should yield repeatable numeric results."""
    # Ensure strict determinism is disabled within this test
    os.environ.pop('SB_STRICT_DETERMINISM', None)
    os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)

    parameters = (
        '--hidden_size 256 --num_hidden_layers 2 --num_attention_heads 4 --num_key_value_heads 4 '
        '--intermediate_size 1024 --batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 '
        '--precision float32 --sample_count 2 --deterministic --random_seed 42 --model_action train'
    )

    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    b1 = BenchmarkRegistry.launch_benchmark(context)

    context2 = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b', platform=Platform.CUDA, parameters=parameters, framework=Framework.PYTORCH
    )
    b2 = BenchmarkRegistry.launch_benchmark(context2)

    assert b1 and b2
    assert b1._args.deterministic and b2._args.deterministic

    # Check time metric shapes
    m_time = 'fp32_train_step_time'
    assert m_time in b1.raw_data and m_time in b2.raw_data
    assert len(b1.raw_data[m_time][0]) == b1._args.num_steps
    assert len(b2.raw_data[m_time][0]) == b2._args.num_steps

    # Compare per-step loss for closeness (soft determinism: allow tiny numeric diffs)
    m_loss = 'fp32_train_loss'
    assert m_loss in b1.raw_data and m_loss in b2.raw_data
    a1 = np.array(b1.raw_data[m_loss][0], dtype=float)
    a2 = np.array(b2.raw_data[m_loss][0], dtype=float)
    assert np.isfinite(a1).all() and np.isfinite(a2).all()
    assert np.allclose(a1, a2, rtol=1e-6, atol=1e-7)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_periodic_checksum_logging(caplog):
    """Emit checksum log at the periodic cadence when deterministic training is enabled."""
    # Ensure strict mode is off; only periodic checksum gated by deterministic should run
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
