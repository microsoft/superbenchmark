# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for dist-inference-cpp benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class DistInferenceCppTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for dist-inference-cpp benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/dist_inference_cpp'])

    def _test_dist_inference_cpp_command_generation(self, platform):
        """Test dist-inference-cpp benchmark command generation."""
        benchmark_name = 'dist-inference-cpp'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)

        mnk_list = ['1,2,3', '4,5,6']
        alpha = 7.0
        beta = 8.0
        num_layers = 9
        num_warmups = 10
        num_iters = 11
        wrapper_params_format_str = \
            '--mnk_list %s --alpha %g --beta %g --num_layers %d --num_warmups %d --num_iters %d --use_cuda_graph'
        parameters = wrapper_params_format_str % (' '.join(mnk_list), alpha, beta, num_layers, num_warmups, num_iters)
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.mnk_list == mnk_list)
        assert (benchmark._args.alpha == alpha)
        assert (benchmark._args.beta == beta)
        assert (benchmark._args.num_layers == num_layers)
        assert (benchmark._args.num_warmups == num_warmups)
        assert (benchmark._args.num_iters == num_iters)
        assert (benchmark._args.use_cuda_graph)

        # Check command
        assert (len(mnk_list) == len(benchmark._commands))
        for cmd, mnk in zip(benchmark._commands, mnk_list):
            m, n, k = [int(x) for x in mnk.split(',')]
            bench_params_format_str = \
                '%s -m %d -n %d -k %d --alpha %g --beta %g ' + \
                '--num_layers %d --num_warmups %d --num_iters %d --use_cuda_graph'
            assert (
                cmd == (
                    bench_params_format_str %
                    (benchmark._DistInferenceCpp__bin_path, m, n, k, alpha, beta, num_layers, num_warmups, num_iters)
                )
            )

    @decorator.cuda_test
    def test_dist_inference_cpp_command_generation_cuda(self):
        """Test dist-inference-cpp benchmark command generation, CUDA case."""
        self._test_dist_inference_cpp_command_generation(Platform.CUDA)

    @decorator.rocm_test
    def test_dist_inference_cpp_command_generation_rocm(self):
        """Test dist-inference-cpp benchmark command generation, ROCm case."""
        self._test_dist_inference_cpp_command_generation(Platform.ROCM)

    @decorator.load_data('tests/data/dist_inference_cpp.log')
    def _test_dist_inference_cpp_result_parsing(self, platform, test_raw_output):
        """Test dist-inference-cpp benchmark result parsing."""
        benchmark_name = 'dist-inference-cpp'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'dist-inference-cpp')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Positive case - valid raw output.
        assert (benchmark._process_raw_result(0, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        assert (2 == len(benchmark.raw_data)) # raw_data and post-processed data
        test_latency = float(test_raw_output.splitlines()[-1].split(' us per layer')[0].split()[-1])
        assert (benchmark.default_metric_count + 1 == len(benchmark.result))
        for output_key in benchmark.result:
            if output_key == 'return_code':
                assert (benchmark.result[output_key] == [0])
            else:
                assert (output_key == ('layer_latency_%s' % benchmark._args.mnk_list[0]))
                assert (len(benchmark.result[output_key]) == 1)
                assert (isinstance(benchmark.result[output_key][0], numbers.Number))
                assert (test_latency == benchmark.result[output_key][0])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(1, 'Invalid raw output') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)

    @decorator.cuda_test
    def test_dist_inference_cpp_result_parsing_cuda(self):
        """Test dist-inference-cpp benchmark result parsing, CUDA case."""
        self._test_dist_inference_cpp_result_parsing(Platform.CUDA)

    @decorator.rocm_test
    def test_dist_inference_cpp_result_parsing_rocm(self):
        """Test dist-inference-cpp benchmark result parsing, ROCm case."""
        self._test_dist_inference_cpp_result_parsing(Platform.ROCM)
