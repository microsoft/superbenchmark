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
        mem_types = ['htod', 'dtoh', 'dtod', 'htod_with_dtoh', 'dtod_bidirectional']
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
cpu_to_gpu0_by_gpu0_using_sm_under_numa0_uni 26.1951
cpu_to_gpu0_by_gpu0_using_sm_under_numa0_bi 8.50481
cpu_to_gpu0_by_gpu0_using_dma_under_numa0_uni 26.1743
cpu_to_gpu0_by_gpu0_using_dma_under_numa0_bi 36.4315
gpu0_to_cpu_by_gpu0_using_sm_under_numa0_uni 5.01717
gpu0_to_cpu_by_gpu0_using_dma_under_numa0_uni 21.8663
gpu0_to_gpu0_by_gpu0_using_sm_under_numa0_uni 658.658
gpu0_to_gpu0_by_gpu0_using_sm_under_numa0_bi 687.988
gpu0_to_gpu0_by_gpu0_using_dma_under_numa0_uni 632.649
gpu0_to_gpu0_by_gpu0_using_dma_under_numa0_bi 661.429
gpu0_to_gpu1_by_gpu0_using_sm_under_numa0_uni 250.051
gpu0_to_gpu1_by_gpu0_using_sm_under_numa0_bi 426.696
gpu0_to_gpu1_by_gpu0_using_dma_under_numa0_uni 275.367
gpu0_to_gpu1_by_gpu0_using_dma_under_numa0_bi 522.122
gpu0_to_gpu1_by_gpu1_using_sm_under_numa0_uni 253.928
gpu0_to_gpu1_by_gpu1_using_sm_under_numa0_bi 447.098
gpu0_to_gpu1_by_gpu1_using_dma_under_numa0_uni 264.811
gpu0_to_gpu1_by_gpu1_using_dma_under_numa0_bi 503.603
cpu_to_gpu1_by_gpu1_using_sm_under_numa0_uni 26.1694
cpu_to_gpu1_by_gpu1_using_sm_under_numa0_bi 8.52593
cpu_to_gpu1_by_gpu1_using_dma_under_numa0_uni 26.2031
cpu_to_gpu1_by_gpu1_using_dma_under_numa0_bi 36.4327
gpu1_to_cpu_by_gpu1_using_sm_under_numa0_uni 5.01487
gpu1_to_cpu_by_gpu1_using_dma_under_numa0_uni 21.8666
gpu1_to_gpu0_by_gpu1_using_sm_under_numa0_uni 249.938
gpu1_to_gpu0_by_gpu1_using_sm_under_numa0_bi 426.169
gpu1_to_gpu0_by_gpu1_using_dma_under_numa0_uni 275.359
gpu1_to_gpu0_by_gpu1_using_dma_under_numa0_bi 522.436
gpu1_to_gpu0_by_gpu0_using_sm_under_numa0_uni 253.994
gpu1_to_gpu0_by_gpu0_using_sm_under_numa0_bi 447.08
gpu1_to_gpu0_by_gpu0_using_dma_under_numa0_uni 265.006
gpu1_to_gpu0_by_gpu0_using_dma_under_numa0_bi 503.195
gpu1_to_gpu1_by_gpu1_using_sm_under_numa0_uni 660.328
gpu1_to_gpu1_by_gpu1_using_sm_under_numa0_bi 690.055
gpu1_to_gpu1_by_gpu1_using_dma_under_numa0_uni 630.154
gpu1_to_gpu1_by_gpu1_using_dma_under_numa0_bi 663.159
cpu_to_gpu0_by_gpu0_using_sm_under_numa1_uni 26.1955
cpu_to_gpu0_by_gpu0_using_sm_under_numa1_bi 9.45794
cpu_to_gpu0_by_gpu0_using_dma_under_numa1_uni 26.1965
cpu_to_gpu0_by_gpu0_using_dma_under_numa1_bi 36.4124
gpu0_to_cpu_by_gpu0_using_sm_under_numa1_uni 5.67736
gpu0_to_cpu_by_gpu0_using_dma_under_numa1_uni 21.8603
gpu0_to_gpu0_by_gpu0_using_sm_under_numa1_uni 654.661
gpu0_to_gpu0_by_gpu0_using_sm_under_numa1_bi 689.692
gpu0_to_gpu0_by_gpu0_using_dma_under_numa1_uni 635.206
gpu0_to_gpu0_by_gpu0_using_dma_under_numa1_bi 661.315
gpu0_to_gpu1_by_gpu0_using_sm_under_numa1_uni 250.125
gpu0_to_gpu1_by_gpu0_using_sm_under_numa1_bi 427.628
gpu0_to_gpu1_by_gpu0_using_dma_under_numa1_uni 275.626
gpu0_to_gpu1_by_gpu0_using_dma_under_numa1_bi 522.218
gpu0_to_gpu1_by_gpu1_using_sm_under_numa1_uni 253.628
gpu0_to_gpu1_by_gpu1_using_sm_under_numa1_bi 447.183
gpu0_to_gpu1_by_gpu1_using_dma_under_numa1_uni 264.766
gpu0_to_gpu1_by_gpu1_using_dma_under_numa1_bi 503.57
cpu_to_gpu1_by_gpu1_using_sm_under_numa1_uni 26.1925
cpu_to_gpu1_by_gpu1_using_sm_under_numa1_bi 9.44989
cpu_to_gpu1_by_gpu1_using_dma_under_numa1_uni 26.2136
cpu_to_gpu1_by_gpu1_using_dma_under_numa1_bi 36.4129
gpu1_to_cpu_by_gpu1_using_sm_under_numa1_uni 5.67653
gpu1_to_cpu_by_gpu1_using_dma_under_numa1_uni 21.8599
gpu1_to_gpu0_by_gpu1_using_sm_under_numa1_uni 249.873
gpu1_to_gpu0_by_gpu1_using_sm_under_numa1_bi 427.317
gpu1_to_gpu0_by_gpu1_using_dma_under_numa1_uni 275.437
gpu1_to_gpu0_by_gpu1_using_dma_under_numa1_bi 521.908
gpu1_to_gpu0_by_gpu0_using_sm_under_numa1_uni 253.876
gpu1_to_gpu0_by_gpu0_using_sm_under_numa1_bi 447.028
gpu1_to_gpu0_by_gpu0_using_dma_under_numa1_uni 264.943
gpu1_to_gpu0_by_gpu0_using_dma_under_numa1_bi 502.893
gpu1_to_gpu1_by_gpu1_using_sm_under_numa1_uni 659.813
gpu1_to_gpu1_by_gpu1_using_sm_under_numa1_bi 689.295
gpu1_to_gpu1_by_gpu1_using_dma_under_numa1_uni 634.38
gpu1_to_gpu1_by_gpu1_using_dma_under_numa1_bi 663.085
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
