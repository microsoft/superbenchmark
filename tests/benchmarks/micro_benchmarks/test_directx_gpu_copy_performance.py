# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DirectXGPUCopyBw benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.directx_test
def test_directx_gpu_copy_bw():
    """Test DirectXGPUCopyBw benchmark."""
    # Test for default configuration
    context = BenchmarkRegistry.create_benchmark_context(
        'directx-gpu-copy-bw',
        platform=Platform.DIRECTX,
        parameters=r'--warm_up 20 --num_loops 1000 --minbytes 64 --maxbytes 8388608 --mem_type htod dtoh'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'directx-gpu-copy-bw')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.warm_up == 20)
    assert (benchmark._args.num_loops == 1000)
    assert (benchmark._args.minbytes == 64)
    assert (benchmark._args.maxbytes == 8388608)
    assert (sorted(benchmark._args.mem_type) == ['dtoh', 'htod'])

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output' in benchmark.raw_data)
    assert (isinstance(benchmark.raw_data['raw_output'][0], str))
    size = 64
    while size <= 8388608:
        for mem_type in ['htod', 'dtoh']:
            assert (f'{mem_type}_{size}_bw' in benchmark.result)
            assert (len(benchmark.result[f'{mem_type}_{size}_bw']) == 1)
            assert (isinstance(benchmark.result[f'{mem_type}_{size}_bw'][0], numbers.Number))
        size *= 2
