# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvbench sleep kernel benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform


class TestNvbenchSleepKernelBenchmark(BenchmarkTestCase, unittest.TestCase):
    """Test class for NVBench Sleep Kernel benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/nvbench_sleep_kernel'])

    def test_nvbench_sleep_kernel_preprocess(self):
        """Test NVBench Sleep Kernel benchmark preprocess."""
        benchmark_name = 'nvbench-sleep-kernel'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test preprocess with default parameters
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Test preprocess with specified parameters
        parameters = (
            '--devices 0 '
            '--duration_us "[10,25,50,75]" '
            '--timeout 20 '
            '--min-samples 300 '
            '--stopping-criterion stdrel '
            '--min-time 2.0 '
            '--max-noise 0.5 '
            '--throttle-threshold 80.0 '
            '--throttle-recovery-delay 1.0'
        )
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('--devices 0' in benchmark._commands[0])
        assert ('--axis "Duration (us)=[10,25,50,75]"' in benchmark._commands[0])
        assert ('--timeout 20' in benchmark._commands[0])
        assert ('--min-samples 300' in benchmark._commands[0])
        assert ('--stopping-criterion stdrel' in benchmark._commands[0])
        assert ('--min-time 2.0' in benchmark._commands[0])
        assert ('--max-noise 0.5' in benchmark._commands[0])
        assert ('--throttle-threshold 80.0' in benchmark._commands[0])
        assert ('--throttle-recovery-delay 1.0' in benchmark._commands[0])

    @decorator.load_data('tests/data/nvbench_sleep_kernel.log')
    def test_nvbench_sleep_kernel_result_parsing_real_output(self, results):
        """Test NVBench Sleep Kernel benchmark result parsing."""
        benchmark_name = 'nvbench-sleep-kernel'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Parse the provided raw output
        assert benchmark._process_raw_result(0, results)
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Validate parsed results
        # assert benchmark.result['duration_us_25_samples'][0] == 10175
        self.assertAlmostEqual(benchmark.result['duration_us_25_cpu_time'][0], 42.123)
        # self.assertAlmostEqual(benchmark.result['duration_us_25_cpu_noise'][0], 69.78)
        self.assertAlmostEqual(benchmark.result['duration_us_25_gpu_time'][0], 25.321)
        # self.assertAlmostEqual(benchmark.result['duration_us_25_gpu_noise'][0], 0.93)
        # assert benchmark.result['duration_us_25_batch_samples'][0] == 17448
        self.assertAlmostEqual(benchmark.result['duration_us_25_batch_gpu_time'][0], 23.456)

        # assert benchmark.result['duration_us_50_samples'][0] == 8187
        # assert benchmark.result['duration_us_75_samples'][0] == 6279

    def test_nvbench_sleep_kernel_preprocess_duration_formats(self):
        """Test NVBench Sleep Kernel preprocess with different duration formats."""
        benchmark_name = 'nvbench-sleep-kernel'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test single value
        benchmark = benchmark_class(benchmark_name, parameters='--duration_us "50"')
        assert benchmark._preprocess()
        assert '--axis "Duration (us)=50"' in benchmark._commands[0]

        # Test list format
        benchmark = benchmark_class(benchmark_name, parameters='--duration_us "[25,50,75]"')
        assert benchmark._preprocess()
        assert '--axis "Duration (us)=[25,50,75]"' in benchmark._commands[0]

        # Test range format
        benchmark = benchmark_class(benchmark_name, parameters='--duration_us "[25:75]"')
        assert benchmark._preprocess()
        assert '--axis "Duration (us)=[25:75]"' in benchmark._commands[0]

        # Test range with step format
        benchmark = benchmark_class(benchmark_name, parameters='--duration_us "[0:50:10]"')
        assert benchmark._preprocess()
        assert '--axis "Duration (us)=[0:50:10]"' in benchmark._commands[0]

        # Test default format
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert '--axis "Duration (us)=[0,25,50,75,100]"' in benchmark._commands[0]

    def test_nvbench_sleep_kernel_process_raw_result_invalid_output(self):
        """Test NVBench Sleep Kernel benchmark result parsing with invalid output."""
        benchmark_name = 'nvbench-sleep-kernel'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Mock raw output with invalid format
        raw_output = 'Invalid output format'

        # Parse the provided raw output
        assert not benchmark._process_raw_result(0, raw_output)
        assert benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE


if __name__ == '__main__':
    unittest.main()
