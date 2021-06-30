# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nccl-bw benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.cuda_test
def test_kernel_launch_overhead():
    """Test nccl-bw benchmark."""
    context = BenchmarkRegistry.create_benchmark_context('nccl-bw', platform=Platform.CUDA, parameters='--gpu_count 1')
    assert (BenchmarkRegistry.is_benchmark_context_valid(context))
    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'nccl-bw')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (
        benchmark._args.bin_list == [
            'all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf'
        ]
    )
    assert (benchmark._args.gpu_count == 1)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    for bin_name in benchmark._args.bin_list:
        assert ('raw_output_' + bin_name in benchmark.raw_data)
        assert (len(benchmark.raw_data['raw_output_' + bin_name]) == 1)
        assert (isinstance(benchmark.raw_data['raw_output_' + bin_name][0], str))
        assert (bin_name in benchmark.result)
        assert (len(benchmark.result[bin_name]) == 1)
        assert (isinstance(benchmark.result[bin_name][0], numbers.Number))
