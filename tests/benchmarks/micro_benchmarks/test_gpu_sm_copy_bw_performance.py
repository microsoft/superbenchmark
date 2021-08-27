# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for disk-performance benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode


@decorator.cuda_test
@decorator.rocm_test
def test_gpu_sm_copy_bw_performance():
    """Test gpu-sm-copy-bw benchmark."""
    numa_node = 0
    gpu_id = 0
    size = 1048576
    num_loops = 10000
    copy_directions = ['dtoh', 'htod']
    context = BenchmarkRegistry.create_benchmark_context(
        'gpu-sm-copy-bw',
        parameters='--numa_nodes %d --gpu_ids %d --%s --%s --size %d --num_loops %d' %
        (numa_node, gpu_id, copy_directions[0], copy_directions[1], size, num_loops)
    )

    assert (BenchmarkRegistry.is_benchmark_context_valid(context))

    benchmark = BenchmarkRegistry.launch_benchmark(context)

    # Check basic information.
    assert (benchmark)
    assert (benchmark.name == 'gpu-sm-copy-bw')
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.numa_nodes == [numa_node])
    assert (benchmark._args.gpu_id == [gpu_id])
    assert (benchmark._args.dtoh)
    assert (benchmark._args.htod)
    assert (benchmark._args.size == size)
    assert (benchmark._args.num_loops == num_loops)

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for idx, copy_direction in enumerate(copy_directions):
        raw_output_key = 'raw_output_%d' % idx
        assert (raw_output_key in benchmark.raw_data)
        assert (len(benchmark.raw_data[raw_output_key]) == 1)
        assert (isinstance(benchmark.raw_data[raw_output_key][0], str))

        output_key = 'numa%d_gpu%d_%s' % (numa_node, gpu_id, copy_direction)
        assert (output_key in benchmark.result)
        assert (len(benchmark.result[output_key]) == 1)
        assert (isinstance(benchmark.result[output_key][0], numbers.Number))
