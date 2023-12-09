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
        num_warm_up = 20
        num_loops = 10000
        all_to_all_num_thread_blocks_per_rank = 8
        all_to_all_thread_block_size = 512
        mem_types = ['htod', 'dtoh', 'dtod', 'one_to_all', 'all_to_one', 'all_to_all']
        copy_types = ['sm', 'dma']

        parameters = '--mem_type %s --copy_type %s --size %d --num_warm_up %d --num_loops %d ' \
            '--all_to_all_num_thread_blocks_per_rank %d --all_to_all_thread_block_size %d ' \
            '--bidirectional --check_data' % \
            (' '.join(mem_types), ' '.join(copy_types), size, num_warm_up, num_loops,
             all_to_all_num_thread_blocks_per_rank, all_to_all_thread_block_size)
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
        assert (benchmark._args.num_warm_up == num_warm_up)
        assert (benchmark._args.num_loops == num_loops)
        assert (benchmark._args.all_to_all_num_thread_blocks_per_rank == all_to_all_num_thread_blocks_per_rank)
        assert (benchmark._args.all_to_all_thread_block_size == all_to_all_thread_block_size)
        assert (benchmark._args.bidirectional)
        assert (benchmark._args.check_data)

        # Check command
        assert (1 == len(benchmark._commands))
        assert (benchmark._commands[0].startswith(benchmark._GpuCopyBwBenchmark__bin_path))
        for mem_type in mem_types:
            assert ('--%s' % mem_type in benchmark._commands[0])
        for copy_type in copy_types:
            assert ('--%s_copy' % copy_type in benchmark._commands[0])
        assert ('--size %d' % size in benchmark._commands[0])
        assert ('--num_warm_up %d' % num_warm_up in benchmark._commands[0])
        assert ('--num_loops %d' % num_loops in benchmark._commands[0])
        assert (
            '--all_to_all_num_thread_blocks_per_rank %d' % all_to_all_num_thread_blocks_per_rank
            in benchmark._commands[0]
        )
        assert ('--all_to_all_thread_block_size %d' % all_to_all_thread_block_size in benchmark._commands[0])
        assert ('--bidirectional' in benchmark._commands[0])
        assert ('--check_data' in benchmark._commands[0])

    @decorator.cuda_test
    def test_gpu_copy_bw_performance_command_generation_cuda(self):
        """Test gpu-copy benchmark command generation, CUDA case."""
        self._test_gpu_copy_bw_performance_command_generation(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_copy_bw_performance_command_generation_rocm(self):
        """Test gpu-copy benchmark command generation, ROCm case."""
        self._test_gpu_copy_bw_performance_command_generation(Platform.ROCM)

    @decorator.load_data('tests/data/gpu_copy_bw_performance.log')
    def _test_gpu_copy_bw_performance_result_parsing(self, platform, test_raw_output):
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
