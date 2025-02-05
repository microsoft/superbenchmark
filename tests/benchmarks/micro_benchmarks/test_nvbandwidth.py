# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvbandwidth benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform


class TestNvBandwidthBenchmark(BenchmarkTestCase, unittest.TestCase):
    """Test class for NV Bandwidth benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/nvbandwidth'])

    def test_nvbandwidth_preprocess(self):
        """Test NV Bandwidth benchmark preprocess."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test preprocess with default parameters
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Test preprocess with specified parameters
        parameters = (
            '--buffer_size 256 '
            '--test_cases host_to_device_memcpy_ce device_to_host_bidirectional_memcpy_ce '
            '--skip_verification '
            '--disable_affinity '
            '--use_mean '
            '--num_loops 100'
        )
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('--bufferSize 256' in benchmark._commands[0])
        assert ('--testcase host_to_device_memcpy_ce device_to_host_bidirectional_memcpy_ce' in benchmark._commands[0])
        assert ('--skipVerification' in benchmark._commands[0])
        assert ('--disableAffinity' in benchmark._commands[0])
        assert ('--useMean' in benchmark._commands[0])
        assert ('--testSamples 100' in benchmark._commands[0])

    @decorator.load_data('tests/data/nvbandwidth_results.log')
    def test_nvbandwidth_result_parsing_real_output(self, results):
        """Test NV Bandwidth benchmark result parsing."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Parse the provided raw output
        assert benchmark._process_raw_result(0, results)
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Validate parsed results
        assert benchmark.result['host_to_device_memcpy_ce_cpu0_gpu0_bw'][0] == 369.36
        assert benchmark.result['host_to_device_memcpy_ce_cpu0_gpu1_bw'][0] == 269.33
        assert benchmark.result['host_to_device_memcpy_ce_sum_bw'][0] == 1985.60
        assert benchmark.result['device_to_host_memcpy_ce_cpu0_gpu1_bw'][0] == 312.11
        assert benchmark.result['device_to_host_memcpy_ce_sum_bw'][0] == 607.26
        assert benchmark.result['host_device_latency_sm_cpu0_gpu0_lat'][0] == 772.58
        assert benchmark.result['host_device_latency_sm_sum_lat'][0] == 772.58

    def test_nvbandwidth_process_raw_result_unsupported_testcases(self):
        """Test NV Bandwidth benchmark result parsing with unsupported test cases."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Mock raw output with unsupported test cases
        raw_output = """
        ERROR: Testcase unsupported_testcase_1 not found!
        ERROR: Testcase unsupported_testcase_2 not found!
        """

        # Parse the provided raw output
        assert not benchmark._process_raw_result(0, raw_output)

        # Validate unsupported test cases
        assert 'unsupported_testcase_1' in benchmark._result.raw_data
        assert benchmark._result.raw_data['unsupported_testcase_1'][0] == 'Not supported'
        assert 'unsupported_testcase_2' in benchmark._result.raw_data
        assert benchmark._result.raw_data['unsupported_testcase_1'][0] == 'Not supported'

    def test_nvbandwidth_process_raw_result_waived_testcases(self):
        """Test NV Bandwidth benchmark result parsing with waived test cases."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Mock raw output with no executed test cases
        raw_output = """
        """

        # Set test cases to include some that will be waived
        benchmark._args.test_cases = ['waived_testcase_1', 'waived_testcase_2']

        # Parse the provided raw output
        assert not benchmark._process_raw_result(0, raw_output)

        # Validate waived test cases
        assert 'waived_testcase_1' in benchmark._result.raw_data
        assert benchmark._result.raw_data['waived_testcase_1'][0] == 'waived'
        assert 'waived_testcase_2' in benchmark._result.raw_data
        assert benchmark._result.raw_data['waived_testcase_2'][0] == 'waived'

    def test_get_all_test_cases(self):
        """Test _get_all_test_cases method."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Mock subprocess.run for successful execution with valid output
        with unittest.mock.patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = (
                '1, host_to_device_memcpy_ce:\n'
                '2, device_to_host_bidirectional_memcpy_ce:'
            )
            mock_run.return_value.stderr = ''
            test_cases = benchmark._get_all_test_cases()
            assert test_cases == ['host_to_device_memcpy_ce', 'device_to_host_bidirectional_memcpy_ce']

        # Mock subprocess.run for execution with non-zero return code
        with unittest.mock.patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ''
            mock_run.return_value.stderr = 'Error'
            test_cases = benchmark._get_all_test_cases()
            assert test_cases == []

        # Mock subprocess.run for execution with error message in stderr
        with unittest.mock.patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ''
            mock_run.return_value.stderr = 'Error'
            test_cases = benchmark._get_all_test_cases()
            assert test_cases == []
