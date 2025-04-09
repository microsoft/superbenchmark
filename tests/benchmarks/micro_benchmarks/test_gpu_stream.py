# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gpu_stream benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class GpuStreamBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for gpu_stream benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/gpu_stream'])

    def _test_gpu_stream_command_generation(self, platform):
        """Test gpu-stream benchmark command generation."""
        benchmark_name = 'gpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)

        num_warm_up = 5
        num_loops = 10
        size = 25769803776

        parameters = '--num_warm_up %d --num_loops %d --size %d ' \
            '--check_data' % \
            (num_warm_up, num_loops, size)
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.size == size)
        assert (benchmark._args.num_warm_up == num_warm_up)
        assert (benchmark._args.num_loops == num_loops)
        assert (benchmark._args.check_data)

        # Check command
        assert (1 == len(benchmark._commands))
        assert (benchmark._commands[0].startswith(benchmark._GpuStreamBenchmark__bin_path))
        assert ('--size %d' % size in benchmark._commands[0])
        assert ('--num_warm_up %d' % num_warm_up in benchmark._commands[0])
        assert ('--num_loops %d' % num_loops in benchmark._commands[0])
        assert ('--check_data' in benchmark._commands[0])

    @decorator.cuda_test
    def test_gpu_stream_command_generation_cuda(self):
        """Test gpu-stream benchmark command generation, CUDA case."""
        self._test_gpu_stream_command_generation(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_stream_command_generation_rocm(self):
        """Test gpu-stream benchmark command generation, ROCm case."""
        self._test_gpu_stream_command_generation(Platform.ROCM)

    @decorator.load_data('tests/data/gpu_stream.log')
    def _test_gpu_stream_result_parsing(self, platform, test_raw_output):
        """Test gpu-stream benchmark result parsing."""
        benchmark_name = 'gpu-stream'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'gpu-stream')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Positive case - valid raw output.
        assert (benchmark._process_raw_result(0, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        assert (1 == len(benchmark.raw_data))
        # print(test_raw_output.splitlines())
        test_raw_output_dict = {x.split()[0]: [float(x.split()[1]), float(x.split()[2])] for x in test_raw_output.strip().splitlines() if x.startswith("STREAM_")}
        assert (len(test_raw_output_dict) * 2 + benchmark.default_metric_count == len(benchmark.result))
        for output_key in benchmark.result:
            if output_key == 'return_code':
                assert (benchmark.result[output_key] == [0])
            else:
                assert (len(benchmark.result[output_key]) == 1)
                assert (isinstance(benchmark.result[output_key][0], numbers.Number))
                if output_key.endswith('_bw'):
                    assert (output_key.strip('_bw') in test_raw_output_dict)
                    assert (test_raw_output_dict[output_key.strip('_bw')][0] == benchmark.result[output_key][0])
                else:
                    assert (output_key.strip('_pct') in test_raw_output_dict)
                    assert (test_raw_output_dict[output_key.strip('_pct')][1] == benchmark.result[output_key][0])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(1, 'Invalid raw output') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)

    @decorator.cuda_test
    def test_gpu_stream_result_parsing_cuda(self):
        """Test gpu-stream benchmark result parsing, CUDA case."""
        self._test_gpu_stream_result_parsing(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_stream_result_parsing_rocm(self):
        """Test gpu-stream benchmark result parsing, ROCm case."""
        self._test_gpu_stream_result_parsing(Platform.ROCM)
