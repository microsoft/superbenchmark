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

    size = 1048576
    num_loops = 10000
    mem_types = ['dtoh', 'htod']

    parameters = '--mem_type %s --size %d --num_loops %d' % \
        (numa_node, gpu_id, ' '.join(mem_types), size, num_loops)
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
    assert (benchmark._args.size == size)
    assert (benchmark._args.num_loops == num_loops)

    # Check and revise command list
    assert (len(mem_types) == len(benchmark._commands))
    for idx, mem_type in enumerate(mem_types):
        assert (
            benchmark._commands[idx] == '%s 0 %s %d %d' %
            (benchmark._GpuSmCopyBwBenchmark__bin_path, mem_type, size, num_loops)
        )

    # Run benchmark
    assert (benchmark._benchmark())

    # Check results and metrics.
    assert (benchmark.run_count == 1)
    assert (benchmark.return_code == ReturnCode.SUCCESS)
    for idx, mem_type in enumerate(mem_types):
        raw_output_key = 'raw_output_%d' % idx
        assert (raw_output_key in benchmark.raw_data)
        assert (len(benchmark.raw_data[raw_output_key]) == 1)
        assert (isinstance(benchmark.raw_data[raw_output_key][0], str))

        output_key = mem_type
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
