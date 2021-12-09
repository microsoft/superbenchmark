# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for matmul benchmark."""

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, Platform, Framework, BenchmarkType, ReturnCode
from superbench.benchmarks.micro_benchmarks.sharding_matmul import ShardingMode


@decorator.cuda_test
@decorator.pytorch_test
def test_pytorch_matmul():
    """Test pytorch-matmul benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'matmul', platform=Platform.CUDA, parameters='--run_count 2 --num_steps 20', framework=Framework.PYTORCH
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'pytorch-matmul')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check predefined parameters of sharding-matmul benchmark.
    assert (benchmark._args.mode == [ShardingMode.NOSHARDING])

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.run_count == 2)
    assert (benchmark._args.num_steps == 20)

    # Check results and metrics.
    assert (benchmark.run_count == 2)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (len(benchmark.raw_data['nosharding_time']) == benchmark.run_count)
    assert (len(benchmark.raw_data['nosharding_time'][0]) == benchmark._args.num_steps)
    assert (len(benchmark.result['nosharding_time']) == benchmark.run_count)
