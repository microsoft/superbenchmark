# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DirectXGPUCorefloops benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.directx_test
def test_directx_gpucoreflops():
    """Test DirectXGPUCoreFlops benchmark."""
    # Test for default configuration
    context = BenchmarkRegistry.create_benchmark_context(
        'directx-gpu-core-flops',
        platform=Platform.DIRECTX,
        parameters=r'--num_loops 10 --n 16384 --k 16384 --m 16384 --precision fp32'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'directx-gpu-core-flops')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.num_loops == 10)
    assert (benchmark._args.n == 16384)
    assert (benchmark._args.k == 16384)
    assert (benchmark._args.m == 16384)
    assert (sorted(benchmark._args.precision) == ['fp32'])

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output_fp32' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output_fp32']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output_fp32'][0], str))

    assert ('fp32_flops' in benchmark.result)
    assert (len(benchmark.result['fp32_flops']) == 1)
    assert (isinstance(benchmark.result['fp32_flops'][0], numbers.Number))
