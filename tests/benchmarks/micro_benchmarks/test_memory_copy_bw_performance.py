# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mem-copy-bw benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.cuda_test
def test_memory_copy_bw_performance():
    """Test mem-copy-bw benchmark."""
    context = BenchmarkRegistry.create_benchmark_context('mem-copy-bw', Platform.CUDA)

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'mem-copy-bw')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)

    assert ('raw_output_0' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_0']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_0'][0], str))
    for metric in ['H2D_Mem_BW', 'D2H_Mem_BW', 'D2D_Mem_BW']:
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
