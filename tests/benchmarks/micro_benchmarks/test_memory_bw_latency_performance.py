# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for kernel-launch benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode


@decorator.cuda_test
def test_memory_bw_latency_performance():
    """Test memory bandwidth-latency benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'memory-bw-latency-benchmark', parameters='--tests bandwidth_matrix latency_matrix'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'memory-bw-latency-benchmark')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.tests == ['bandwidth_matrix','latency_matrix'])

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    # we skip the output from the first command that is enabling huge page
    assert ('raw_output_1' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_1']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_1'][0], str))
    for metric in ['Mem_bandwidth_matrix_numa_0_0_BW','Mem_latency_matrix_numa_0_0_Latency']:
        assert (metric in benchmark.result)
        assert (len(benchmark.result[metric]) == 1)
        assert (isinstance(benchmark.result[metric][0], numbers.Number))
