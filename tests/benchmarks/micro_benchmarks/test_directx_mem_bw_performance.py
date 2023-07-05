# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DirectXGPUMemBw benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.directx_test
def test_directx_gpu_mem_bw():
    """Test DirectXGPUMemBw benchmark."""
    # Test for default configuration
    context = BenchmarkRegistry.create_benchmark_context(
        'directx-gpu-mem-bw',
        platform=Platform.DIRECTX,
        parameters=r'--num_warm_up 0 --num_loop 100 --size 1073741824 --mode read write'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'directx-gpu-mem-bw')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_warm_up == 0)
    assert (benchmark._args.num_loop == 100)
    assert (benchmark._args.size == 1073741824)
    assert (sorted(benchmark._args.mode) == ['read', 'write'])

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output_read' in benchmark.raw_data)
    assert ('raw_output_write' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_read']) == 1)
    assert (len(benchmark.raw_data['raw_output_write']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_read'][0], str))
    assert (isinstance(benchmark.raw_data['raw_output_write'][0], str))

    assert ('read_1073741824_bw' in benchmark.result)
    assert ('write_1073741824_bw' in benchmark.result)
    assert (len(benchmark.result['read_1073741824_bw']) == 1)
    assert (len(benchmark.result['write_1073741824_bw']) == 1)
    assert (isinstance(benchmark.result['read_1073741824_bw'][0], numbers.Number))
    assert (isinstance(benchmark.result['write_1073741824_bw'][0], numbers.Number))
