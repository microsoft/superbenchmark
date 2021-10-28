# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for disk-performance benchmark."""

import numbers

from tests.helper import decorator
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


def _test_gpu_copy_bw_performance_impl(platform):
    """Test gpu-copy-bw benchmark."""
    benchmark_name = 'gpu-copy-bw'
    (benchmark_class,
     predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
    assert (benchmark_class)

    size = 1048576
    num_loops = 10000
    mem_types = ['htod', 'dtoh', 'dtod']
    copy_types = ['sm', 'dma']

    parameters = '--mem_type %s --copy_type %s --size %d --num_loops %d' % \
        (' '.join(mem_types), ' '.join(copy_types), size, num_loops)
    benchmark = benchmark_class(benchmark_name, parameters=parameters)

    # Check basic information
    assert (benchmark)
    ret = benchmark._preprocess()
    assert (ret is True)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (benchmark.name == benchmark_name)
    assert (benchmark.type == BenchmarkType.MICRO)

    # Check parameters specified in BenchmarkContext.
    assert (benchmark._args.mem_type == mem_types)
    assert (benchmark._args.copy_type == copy_types)
    assert (benchmark._args.size == size)
    assert (benchmark._args.num_loops == num_loops)

    # Check command
    assert (1 == len(benchmark._commands))
    assert (benchmark._commands[0].startswith(benchmark._GpuCopyBwBenchmark__bin_path))
    for mem_type in mem_types:
        assert ('--%s' % mem_type in benchmark._commands[0])
    for copy_type in copy_types:
        assert ('--%s_copy' % copy_type in benchmark._commands[0])
    assert ('--size %d' % size in benchmark._commands[0])
    assert ('--num_loops %d' % num_loops in benchmark._commands[0])

    # Run benchmark
    assert (benchmark._benchmark())

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    assert (1 == len(benchmark.raw_data))
    assert (len(benchmark.raw_data.splitlines()) == len(benchmark.result))
    for output_key in benchmark.result:
        assert (len(benchmark.result[output_key]) == 1)
        assert (isinstance(benchmark.result[output_key][0], numbers.Number))


@decorator.cuda_test
def test_gpu_sm_copy_bw_performance_cuda():
    """Test gpu-copy-bw benchmark, CUDA case."""
    _test_gpu_copy_bw_performance_impl(Platform.CUDA)


@decorator.rocm_test
def test_gpu_sm_copy_bw_performance_rocm():
    """Test gpu-copy-bw benchmark, ROCm case."""
    _test_gpu_copy_bw_performance_impl(Platform.ROCM)
