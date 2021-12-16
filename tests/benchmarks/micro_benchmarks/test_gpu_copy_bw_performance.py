# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gpu-copy-bw benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class GpuCopyBwBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for gpu-copy-bw benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/gpu_copy'])

    def _test_gpu_copy_bw_performance_command_generation(self, platform):
        """Test gpu-copy benchmark command generation."""
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

    @decorator.cuda_test
    def test_gpu_copy_bw_performance_command_generation_cuda(self):
        """Test gpu-copy benchmark command generation, CUDA case."""
        self._test_gpu_copy_bw_performance_command_generation(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_copy_bw_performance_command_generation_rocm(self):
        """Test gpu-copy benchmark command generation, ROCm case."""
        self._test_gpu_copy_bw_performance_command_generation(Platform.ROCM)

    def _test_gpu_copy_bw_performance_result_parsing(self, platform):
        """Test gpu-copy benchmark result parsing."""
        benchmark_name = 'gpu-copy-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'gpu-copy-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Positive case - valid raw output.
        test_raw_output = """
cpu_to_gpu0_by_gpu0_using_sm_under_numa0 26.1755
cpu_to_gpu0_by_gpu0_using_dma_under_numa0 26.1894
gpu0_to_cpu_by_gpu0_using_sm_under_numa0 5.72584
gpu0_to_cpu_by_gpu0_using_dma_under_numa0 26.2623
gpu0_to_gpu0_by_gpu0_using_sm_under_numa0 659.275
gpu0_to_gpu0_by_gpu0_using_dma_under_numa0 636.401
cpu_to_gpu0_by_gpu0_using_sm_under_numa1 26.1589
cpu_to_gpu0_by_gpu0_using_dma_under_numa1 26.18
gpu0_to_cpu_by_gpu0_using_sm_under_numa1 5.07597
gpu0_to_cpu_by_gpu0_using_dma_under_numa1 25.2851
gpu0_to_gpu0_by_gpu0_using_sm_under_numa1 656.825
gpu0_to_gpu0_by_gpu0_using_dma_under_numa1 634.203
"""
        assert (benchmark._process_raw_result(0, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        assert (1 == len(benchmark.raw_data))
        print(test_raw_output.splitlines())
        test_raw_output_dict = {x.split()[0]: float(x.split()[1]) for x in test_raw_output.strip().splitlines()}
        assert (len(test_raw_output_dict) + benchmark.default_metric_count == len(benchmark.result))
        for output_key in benchmark.result:
            if output_key == 'return_code':
                assert (benchmark.result[output_key] == [0])
            else:
                assert (len(benchmark.result[output_key]) == 1)
                assert (isinstance(benchmark.result[output_key][0], numbers.Number))
                assert (output_key.strip('_bw') in test_raw_output_dict)
                assert (test_raw_output_dict[output_key.strip('_bw')] == benchmark.result[output_key][0])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(1, 'Invalid raw output') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)

    @decorator.cuda_test
    def test_gpu_copy_bw_performance_result_parsing_cuda(self):
        """Test gpu-copy benchmark result parsing, CUDA case."""
        self._test_gpu_copy_bw_performance_result_parsing(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_copy_bw_performance_result_parsing_rocm(self):
        """Test gpu-copy benchmark result parsing, ROCm case."""
        self._test_gpu_copy_bw_performance_result_parsing(Platform.ROCM)
