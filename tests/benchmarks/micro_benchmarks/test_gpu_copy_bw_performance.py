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

        parameters = '--mem_type %s --copy_type %s --size %d --num_loops %d --bidirectional' % \
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
        assert (benchmark._args.bidirectional)

        # Check command
        assert (1 == len(benchmark._commands))
        assert (benchmark._commands[0].startswith(benchmark._GpuCopyBwBenchmark__bin_path))
        for mem_type in mem_types:
            assert ('--%s' % mem_type in benchmark._commands[0])
        for copy_type in copy_types:
            assert ('--%s_copy' % copy_type in benchmark._commands[0])
        assert ('--size %d' % size in benchmark._commands[0])
        assert ('--num_loops %d' % num_loops in benchmark._commands[0])
        assert ('--bidirectional' in benchmark._commands[0])

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
cpu_to_gpu0_by_sm_under_numa0_uni 26.1736
cpu_to_gpu0_by_dma_under_numa0_uni 26.1878
gpu0_to_cpu_by_sm_under_numa0_uni 5.01589
gpu0_to_cpu_by_dma_under_numa0_uni 21.8659
gpu0_to_gpu0_by_sm_under_numa0_uni 655.759
gpu0_to_gpu0_by_dma_under_numa0_uni 633.325
gpu0_to_gpu1_write_by_sm_under_numa0_uni 250.122
gpu0_to_gpu1_write_by_dma_under_numa0_uni 274.951
gpu0_to_gpu1_read_by_sm_under_numa0_uni 253.563
gpu0_to_gpu1_read_by_dma_under_numa0_uni 264.009
cpu_to_gpu1_by_sm_under_numa0_uni 26.187
cpu_to_gpu1_by_dma_under_numa0_uni 26.207
gpu1_to_cpu_by_sm_under_numa0_uni 5.01132
gpu1_to_cpu_by_dma_under_numa0_uni 21.8635
gpu1_to_gpu0_write_by_sm_under_numa0_uni 249.824
gpu1_to_gpu0_write_by_dma_under_numa0_uni 275.123
gpu1_to_gpu0_read_by_sm_under_numa0_uni 253.469
gpu1_to_gpu0_read_by_dma_under_numa0_uni 264.908
gpu1_to_gpu1_by_sm_under_numa0_uni 658.338
gpu1_to_gpu1_by_dma_under_numa0_uni 631.148
cpu_to_gpu0_by_sm_under_numa1_uni 26.1542
cpu_to_gpu0_by_dma_under_numa1_uni 26.2007
gpu0_to_cpu_by_sm_under_numa1_uni 5.67356
gpu0_to_cpu_by_dma_under_numa1_uni 21.8599
gpu0_to_gpu0_by_sm_under_numa1_uni 656.935
gpu0_to_gpu0_by_dma_under_numa1_uni 631.974
gpu0_to_gpu1_write_by_sm_under_numa1_uni 250.118
gpu0_to_gpu1_write_by_dma_under_numa1_uni 274.778
gpu0_to_gpu1_read_by_sm_under_numa1_uni 253.625
gpu0_to_gpu1_read_by_dma_under_numa1_uni 264.347
cpu_to_gpu1_by_sm_under_numa1_uni 26.1905
cpu_to_gpu1_by_dma_under_numa1_uni 26.2007
gpu1_to_cpu_by_sm_under_numa1_uni 5.67716
gpu1_to_cpu_by_dma_under_numa1_uni 21.8579
gpu1_to_gpu0_write_by_sm_under_numa1_uni 250.064
gpu1_to_gpu0_write_by_dma_under_numa1_uni 274.924
gpu1_to_gpu0_read_by_sm_under_numa1_uni 253.746
gpu1_to_gpu0_read_by_dma_under_numa1_uni 264.256
gpu1_to_gpu1_by_sm_under_numa1_uni 655.623
gpu1_to_gpu1_by_dma_under_numa1_uni 634.062
cpu_to_gpu0_by_sm_under_numa0_bi 8.45975
cpu_to_gpu0_by_dma_under_numa0_bi 36.4282
gpu0_to_gpu0_by_sm_under_numa0_bi 689.063
gpu0_to_gpu0_by_dma_under_numa0_bi 661.7
gpu0_to_gpu1_write_by_sm_under_numa0_bi 427.446
gpu0_to_gpu1_write_by_dma_under_numa0_bi 521.577
gpu0_to_gpu1_read_by_sm_under_numa0_bi 446.835
gpu0_to_gpu1_read_by_dma_under_numa0_bi 503.158
cpu_to_gpu1_by_sm_under_numa0_bi 8.4487
cpu_to_gpu1_by_dma_under_numa0_bi 36.4272
cpu_to_gpu0_by_sm_under_numa1_bi 9.36164
cpu_to_gpu0_by_dma_under_numa1_bi 36.411
gpu0_to_gpu0_by_sm_under_numa1_bi 688.156
gpu0_to_gpu0_by_dma_under_numa1_bi 662.077
gpu0_to_gpu1_write_by_sm_under_numa1_bi 427.033
gpu0_to_gpu1_write_by_dma_under_numa1_bi 521.367
gpu0_to_gpu1_read_by_sm_under_numa1_bi 446.179
gpu0_to_gpu1_read_by_dma_under_numa1_bi 503.843
cpu_to_gpu1_by_sm_under_numa1_bi 9.37368
cpu_to_gpu1_by_dma_under_numa1_bi 36.4128
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
