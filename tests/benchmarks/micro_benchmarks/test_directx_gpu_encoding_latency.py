# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DirectXGPUEncodingLatency benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


@decorator.directx_test
def test_directx_gpuencodinglatency():
    """Test DirectXGPUEncodingLatency benchmark."""
    context = BenchmarkRegistry.create_benchmark_context(
        'directx-gpu-encoding-latency',
        platform=Platform.DIRECTX,
        parameters=r'--algo ASAP --codec H265 --format NV12 --frames 500' +
        r' --height 720 --width 1080 --output_height 720 --output_width 1080 --vcn 0'
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'directx-gpu-encoding-latency')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.algo == 'ASAP')
    assert (benchmark._args.codec == 'H265')
    assert (benchmark._args.format == 'NV12')
    assert (benchmark._args.frames == 500)
    assert (benchmark._args.height == 720)
    assert (benchmark._args.width == 1080)
    assert (benchmark._args.output_height == 720)
    assert (benchmark._args.output_width == 1080)
    assert (benchmark._args.vcn == 0)

    # Check results and metrics.
    assert (benchmark._args.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert ('raw_output' in benchmark.raw_data)
    assert (len(benchmark.raw_data['raw_output']) == 1)
    assert (isinstance(benchmark.raw_data['raw_output'][0], str))

    assert ('fps' in benchmark.result)
    assert ('min_lat' in benchmark.result)
    assert ('max_lat' in benchmark.result)
    assert ('avg_lat' in benchmark.result)
    assert (isinstance(benchmark.result['fps'][0], numbers.Number))
    assert (isinstance(benchmark.result['min_lat'][0], numbers.Number))
    assert (isinstance(benchmark.result['max_lat'][0], numbers.Number))
    assert (isinstance(benchmark.result['avg_lat'][0], numbers.Number))
