# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for disk-performance benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


def _test_gpu_sm_copy_bw_performance_impl(platform):
    """Test gpu-sm-copy-bw benchmark."""
    benchmark_name = 'gpu-sm-copy-bw'
    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
    assert (benchmark_class)

    numa_node = 0
    gpu_id = 0
    size = 1048576
    num_loops = 10000
    copy_directions = ['dtoh', 'htod']

    parameters = '--numa_nodes %d --gpu_ids %d --%s --%s --size %d --num_loops %d' % \
        (numa_node, gpu_id, copy_directions[0], copy_directions[1], size, num_loops)
    benchmark = benchmark_class(benchmark_name, parameters=parameters)

    # Check basic information
    assert (benchmark)
    ret = benchmark._preprocess()
    assert (ret is True)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (benchmark.name == benchmark_name)
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.numa_nodes == [numa_node])
    assert (benchmark._args.gpu_ids == [gpu_id])
    assert (benchmark._args.dtoh)
    assert (benchmark._args.htod)
    assert (benchmark._args.size == size)
    assert (benchmark._args.num_loops == num_loops)

    # Check and revise command list
    assert (len(copy_directions) * benchmark._GpuSmCopyBwBenchmark__num_gpus_in_system == len(benchmark._commands))
    for idx in range(benchmark._GpuSmCopyBwBenchmark__num_gpus_in_system):
        copy_direction = copy_directions[idx]
        assert (
            benchmark._commands[idx] == 'numactl -N %d -m %d %s %d %s %d %d' %
            (numa_node, numa_node, benchmark._GpuSmCopyBwBenchmark__bin_path, gpu_id, copy_direction, size, num_loops)
        )
        numactl_prefix = 'numactl -N %d -m %d ' % (numa_node, numa_node)
        # Remove numactl because test environment is not privileged
        benchmark._commands[idx] = benchmark._commands[idx][len(numactl_prefix):]

    # Run benchmark
    assert (benchmark._benchmark())

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for idx in range(benchmark._GpuSmCopyBwBenchmark__num_gpus_in_system):
        copy_direction = copy_directions[idx]

        raw_output_key = 'raw_output_%d' % idx
        assert (raw_output_key in benchmark.raw_data)
        assert (len(benchmark.raw_data[raw_output_key]) == 1)
        assert (isinstance(benchmark.raw_data[raw_output_key][0], str))

        output_key = 'numa%d_gpu%d_%s' % (numa_node, gpu_id, copy_direction)
        assert (output_key in benchmark.result)
        assert (len(benchmark.result[output_key]) == 1)
        assert (isinstance(benchmark.result[output_key][0], numbers.Number))


@decorator.cuda_test
def test_gpu_sm_copy_bw_performance_cuda():
    """Test gpu-sm-copy-bw benchmark, CUDA case."""
    _test_gpu_sm_copy_bw_performance_impl(Platform.CUDA)


@decorator.rocm_test
def test_gpu_sm_copy_bw_performance_rocm():
    """Test gpu-sm-copy-bw benchmark, ROCm case."""
    _test_gpu_sm_copy_bw_performance_impl(Platform.ROCM)
