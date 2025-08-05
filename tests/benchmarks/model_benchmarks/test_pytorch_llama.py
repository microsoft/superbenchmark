# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Llama model benchmarks."""

import torch
from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
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
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 \
            --model_action train --deterministic --random_seed 42',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    # Run benchmark with deterministic settings
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check that the run succeeded
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    # Check that deterministic parameters are set correctly
    assert (benchmark._args.deterministic == True)
    assert (benchmark._args.random_seed == 42)

    # Check that we have valid results (deterministic training should still produce results)
    assert 'fp16_train_step_time' in benchmark.result
    assert len(benchmark.result['fp16_train_step_time']) > 0
    assert all(isinstance(x, (int, float)) and x > 0 for x in benchmark.result['fp16_train_step_time'])

    # Check that throughput results are also valid
    if 'fp16_train_throughput' in benchmark.result:
        assert len(benchmark.result['fp16_train_throughput']) > 0
        assert all(isinstance(x, (int, float)) and x > 0 for x in benchmark.result['fp16_train_throughput'])

    # Verify that the benchmark completes without errors when deterministic mode is enabled
    # (This validates that our _enable_deterministic_training method works without crashes)


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_non_deterministic_training():
    """Test pytorch-llama2-7b benchmark with non-deterministic training (default behavior)."""
    # Test that non-deterministic training works as expected
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 \
            --model_action train',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check that benchmark runs successfully
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    # Check that deterministic is disabled by default
    assert (benchmark._args.deterministic == False)
    assert (benchmark._args.random_seed == 42)  # Default value


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_llama_deterministic_parameters():
    """Test pytorch-llama2-7b benchmark parameter parsing for deterministic training."""
    # Test parameter parsing for deterministic training
    context = BenchmarkRegistry.create_benchmark_context(
        'llama2-7b',
        platform=Platform.CUDA,
        parameters='--batch_size 1 --seq_len 32 --num_warmup 1 --num_steps 2 --precision float16 \
            --model_action train --deterministic --random_seed 123',
        framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic functionality
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    # Check that parameters are parsed correctly
    assert (benchmark._args.deterministic == True)
    assert (benchmark._args.random_seed == 123)

    # Check that all other parameters are still working
    assert (benchmark._args.batch_size == 1)
    assert (benchmark._args.seq_len == 32)
    assert (benchmark._args.num_warmup == 1)
    assert (benchmark._args.num_steps == 2)
