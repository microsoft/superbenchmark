# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gpu-stream benchmark."""

import unittest
from pathlib import Path

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class GpuStreamBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for gpu-stream benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/hip-stream'])

    @staticmethod
    def _load_fixture(filename):
        return (Path('tests/data') / filename).read_text()

    def _test_gpu_stream_command_generation(self, platform):
        """Test gpu-stream benchmark command generation."""
        benchmark_name = 'gpu-stream'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)

        parameters = '--array_size 268435456 --num_loops 20 --precision float'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        ret = benchmark._preprocess()

        assert (benchmark)
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._args.array_size == 268435456)
        assert (benchmark._args.num_loops == 20)
        assert (benchmark._args.precision == 'float')

        assert (1 == len(benchmark._commands))
        assert (benchmark._commands[0].startswith(benchmark._GpuStreamBenchmark__bin_path))
        assert ('--arraysize 268435456' in benchmark._commands[0])
        assert ('--numtimes 20' in benchmark._commands[0])
        assert ('--csv' in benchmark._commands[0])
        assert ('--float' in benchmark._commands[0])
        assert ('--device' not in benchmark._commands[0])

        benchmark = benchmark_class(benchmark_name, parameters='--array_size 1024 --num_loops 2')
        assert (benchmark._preprocess() is True)
        assert (benchmark._args.precision == 'double')
        assert ('--float' not in benchmark._commands[0])

    @decorator.cuda_test
    def test_gpu_stream_command_generation_cuda(self):
        """Test gpu-stream benchmark command generation, CUDA case."""
        self._test_gpu_stream_command_generation(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_stream_command_generation_rocm(self):
        """Test gpu-stream benchmark command generation, ROCm case."""
        self._test_gpu_stream_command_generation(Platform.ROCM)

    def _test_gpu_stream_result_parsing(self, platform):
        """Test gpu-stream benchmark result parsing."""
        benchmark_name = 'gpu-stream'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, platform)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--precision double')
        assert (benchmark)
        assert (benchmark._preprocess() is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == benchmark_name)
        assert (benchmark.type == BenchmarkType.MICRO)

        valid_output = self._load_fixture('gpu_stream.log')

        assert (benchmark._process_raw_result(0, valid_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert ('raw_output_0' in benchmark.raw_data)
        assert ('Device: BW150' in benchmark.raw_data['raw_output_0'][0])

        expected_metric_values = {
            'STREAM_INIT_double_array_268435456_bw': 6.77961,
            'STREAM_INIT_double_array_268435456_time': 0.950269,
            'STREAM_READ_double_array_268435456_bw': 1255.98,
            'STREAM_READ_double_array_268435456_time': 0.00512943,
            'STREAM_COPY_double_array_268435456_bw': 1345.22,
            'STREAM_COPY_double_array_268435456_time_min': 0.00319277,
            'STREAM_COPY_double_array_268435456_time_max': 0.00320985,
            'STREAM_COPY_double_array_268435456_time_avg': 0.00319879,
            'STREAM_MUL_double_array_268435456_bw': 1370.7,
            'STREAM_MUL_double_array_268435456_time_min': 0.00313342,
            'STREAM_MUL_double_array_268435456_time_max': 0.00314978,
            'STREAM_MUL_double_array_268435456_time_avg': 0.00313862,
            'STREAM_ADD_double_array_268435456_bw': 1292.74,
            'STREAM_ADD_double_array_268435456_time_min': 0.00498358,
            'STREAM_ADD_double_array_268435456_time_max': 0.00499938,
            'STREAM_ADD_double_array_268435456_time_avg': 0.00498747,
            'STREAM_TRIAD_double_array_268435456_bw': 1292.52,
            'STREAM_TRIAD_double_array_268435456_time_min': 0.00498439,
            'STREAM_TRIAD_double_array_268435456_time_max': 0.00499791,
            'STREAM_TRIAD_double_array_268435456_time_avg': 0.00498815,
            'STREAM_DOT_double_array_268435456_bw': 1271.19,
            'STREAM_DOT_double_array_268435456_time_min': 0.00337869,
            'STREAM_DOT_double_array_268435456_time_max': 0.00359398,
            'STREAM_DOT_double_array_268435456_time_avg': 0.0033883,
        }
        for metric_name, expected_value in expected_metric_values.items():
            assert (metric_name in benchmark.result)
            assert (abs(benchmark.result[metric_name][0] - expected_value) < 1e-6)

        assert (all(not metric.endswith('_ratio') for metric in benchmark.result))

        benchmark = benchmark_class(benchmark_name, parameters='--precision double')
        assert (benchmark._preprocess() is True)
        assert (benchmark._process_raw_result(0, 'Invalid raw output') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)

    @decorator.cuda_test
    def test_gpu_stream_result_parsing_cuda(self):
        """Test gpu-stream benchmark result parsing, CUDA case."""
        self._test_gpu_stream_result_parsing(Platform.CUDA)

    @decorator.rocm_test
    def test_gpu_stream_result_parsing_rocm(self):
        """Test gpu-stream benchmark result parsing, ROCm case."""
        self._test_gpu_stream_result_parsing(Platform.ROCM)
