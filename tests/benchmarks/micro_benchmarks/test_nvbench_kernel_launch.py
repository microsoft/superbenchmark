# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvbench kernel launch benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform


class TestNvbenchKernelLaunchBenchmark(BenchmarkTestCase, unittest.TestCase):
    """Test class for NVBench Kernel Launch benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/nvbench_kernel_launch'])

    def test_nvbench_kernel_launch_preprocess(self):
        """Test NVBench Kernel Launch benchmark preprocess."""
        benchmark_name = 'nvbench-kernel-launch'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test preprocess with default parameters
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Test preprocess with specified parameters
        parameters = (
            '--devices 0 '
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
        assert ('--timeout 20' in benchmark._commands[0])
        assert ('--min-samples 300' in benchmark._commands[0])
        assert ('--stopping-criterion stdrel' in benchmark._commands[0])
        assert ('--min-time 2.0' in benchmark._commands[0])
        assert ('--max-noise 0.5' in benchmark._commands[0])
        assert ('--throttle-threshold 80.0' in benchmark._commands[0])
        assert ('--throttle-recovery-delay 1.0' in benchmark._commands[0])

    @decorator.load_data('tests/data/nvbench_kernel_launch.log')
    def test_nvbench_kernel_launch_result_parsing_real_output(self, results):
        """Test NVBench Kernel Launch benchmark result parsing."""
        benchmark_name = 'nvbench-kernel-launch'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Parse the provided raw output
        assert benchmark._process_raw_result(0, results)
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Validate parsed results
        # assert benchmark.result['samples'][0] == 120000
        self.assertAlmostEqual(benchmark.result['cpu_time'][0], 24.222)
        # self.assertAlmostEqual(benchmark.result['cpu_noise'][0], 30.44)
        self.assertAlmostEqual(benchmark.result['gpu_time'][0], 7.808)
        # self.assertAlmostEqual(benchmark.result['gpu_noise'][0], 14.42)
        # assert benchmark.result['batch_samples'][0] == 300000
        self.assertAlmostEqual(benchmark.result['batch_gpu_time'][0], 6.024)

    def test_nvbench_kernel_launch_process_raw_result_invalid_output(self):
        """Test NVBench Kernel Launch benchmark result parsing with invalid output."""
        benchmark_name = 'nvbench-kernel-launch'
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
