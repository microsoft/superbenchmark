# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvbandwidth benchmark."""

import os

import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform
from superbench.benchmarks.micro_benchmarks import NvBandwidthBenchmark


class TestNvBandwidthBenchmark(BenchmarkTestCase, unittest.TestCase):
    """Test class for NV Bandwidth benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/nvbandwidth'])
        # Default discovery must return real case names, not an empty successful listing.
        with open(os.path.join(cls._tmp_dir, 'bin/nvbandwidth'), 'w') as mock_binary:
            mock_binary.write(
                '#!/bin/sh\n'
                'printf "0, host_to_device_memcpy_ce:\\n'
                '1, device_to_host_memcpy_ce:\\n'
                '2, host_device_latency_sm:\\n"\n'
            )

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

        # Test preprocess with numa
        os.environ['NUMA_NODES'] = '0'
        os.environ['PROC_RANK'] = '0'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS
        assert ('numactl --cpunodebind=0 --membind=0' in benchmark._commands[0])

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

    @decorator.load_data('tests/data/nvbandwidth_v0.11.log')
    def test_nvbandwidth_v011_single_node_output(self, results):
        """Parse native directional, latency, local and custom batch matrices without losing legacy aliases."""
        test_cases = (
            'device_to_device_bidirectional_memcpy_read_ce '
            'device_to_device_bidirectional_memcpy_write_sm '
            'device_to_device_latency_sm device_to_device_latency_tma device_to_device_latency_counted '
            'device_local_copy device_local_read_sm '
            'all_to_host_batch_memcpy_ce host_to_all_batch_memcpy_ce '
            'all_to_host_batch_memcpy_sm host_to_all_batch_memcpy_sm'
        )
        benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases ' + test_cases)
        assert benchmark._preprocess()
        assert benchmark._process_raw_result(0, results)
        assert benchmark.return_code == ReturnCode.SUCCESS

        for case, direction, values in (
            ('device_to_device_bidirectional_memcpy_read_ce', 'read', (41.0, 51.0, 92.0, 83.0, 103.0, 186.0)),
            ('device_to_device_bidirectional_memcpy_write_sm', 'write', (61.0, 71.0, 132.0, 123.0, 143.0, 266.0)),
        ):
            with self.subTest(case=case):
                assert benchmark.result[f'{case}_gpu0_gpu1_bw'] == [values[0]]
                assert benchmark.result[f'{case}_sum_bw'] == [values[3]]
                assert benchmark.result[f'{case}_{direction}1_gpu0_gpu1_bw'] == [values[0]]
                assert benchmark.result[f'{case}_{direction}2_gpu0_gpu1_bw'] == [values[1]]
                assert benchmark.result[f'{case}_total_gpu0_gpu1_bw'] == [values[2]]
                assert benchmark.result[f'{case}_{direction}1_sum_bw'] == [values[3]]
                assert benchmark.result[f'{case}_{direction}2_sum_bw'] == [values[4]]
                assert benchmark.result[f'{case}_total_sum_bw'] == [values[5]]
                assert f'{case}_gpu0_gpu0_bw' not in benchmark.result

        assert benchmark.result['device_to_device_latency_sm_gpu0_gpu1_lat'] == [154.25]
        assert benchmark.result['device_to_device_latency_sm_sum_lat'] == [310.0]
        assert benchmark.result['device_to_device_latency_tma_gpu0_gpu1_lat'] == [108.25]
        assert benchmark.result['device_to_device_latency_tma_sum_lat'] == [218.5]
        assert benchmark.result['device_to_device_latency_counted_gpu0_gpu1_lat'] == [160.5]
        assert benchmark.result['device_to_device_latency_counted_gpu1_gpu0_lat'] == [162.5]
        assert benchmark.result['device_to_device_latency_counted_sum_lat'] == [323.0]
        assert 'device_to_device_latency_counted_gpu0_gpu0_lat' not in benchmark.result
        assert benchmark.result['device_local_copy_gpu0_bw'] == [12000.0]
        assert benchmark.result['device_local_copy_gpu1_bw'] == [12500.0]
        assert benchmark.result['device_local_copy_sum_bw'] == [24500.0]
        assert benchmark.result['device_local_read_sm_gpu1_bw'] == [14000.0]
        for case in test_cases.split()[-4:]:
            assert benchmark.result[f'{case}_cpu0_gpu0_bw'] == [130.0]
            assert benchmark.result[f'{case}_cpu0_gpu1_bw'] == [140.0]
            assert benchmark.result[f'{case}_sum_bw'] == [270.0]

    def test_nvbandwidth_v011_matrix_lines(self):
        """A new matrix resets its columns without overwriting the first matrix's legacy alias."""
        benchmark = NvBandwidthBenchmark('nvbandwidth')
        case = 'device_to_device_bidirectional_memcpy_read_ce'
        status = {'test_name': case, 'matrix_index': 0, 'testcase_matrices': {case: set()}, 'results': {}}
        benchmark._process_matrix_line('memcpy CE GPU(row) <-> GPU(column) Read1 bandwidth (GB/s)', status)
        benchmark._process_matrix_line('0 1', status)
        benchmark._process_matrix_line('0 N/A 41.00', status)
        benchmark._process_matrix_line('memcpy CE GPU(row) <-> GPU(column) Read2 bandwidth (GB/s)', status)
        benchmark._process_matrix_line('0 1', status)
        benchmark._process_matrix_line('0 N/A 51.00', status)
        assert status['results'] == {
            f'{case}_gpu0_gpu1_bw': 41.0,
            f'{case}_read1_gpu0_gpu1_bw': 41.0,
            f'{case}_read2_gpu0_gpu1_bw': 51.0,
        }

    def test_nvbandwidth_v011_ublkcp_latency_title(self):
        """The non-LE counted latency title maps GPU(row->col) to GPU row and column axes."""
        benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases device_to_device_latency_counted')
        assert benchmark._preprocess()
        output = (
            'Running device_to_device_latency_counted.\n'
            'UBLKCP device-to-device counted latency GPU(row->col) (ns)\n'
            '           0         1\n'
            ' 0       N/A    900.50\n'
            ' 1    901.50       N/A\n'
            '\n'
            'SUM device_to_device_latency_counted 1802.00\n'
        )
        assert benchmark._process_raw_result(0, output)
        assert benchmark.return_code == ReturnCode.SUCCESS
        assert benchmark.result['device_to_device_latency_counted_gpu0_gpu1_lat'] == [900.5]
        assert benchmark.result['device_to_device_latency_counted_gpu1_gpu0_lat'] == [901.5]
        assert benchmark.result['device_to_device_latency_counted_sum_lat'] == [1802.0]

    @decorator.load_data('tests/data/nvbandwidth_v0.11.log')
    def test_nvbandwidth_v011_status_completeness(self, results):
        """Keep valid measurements but fail unsupported, waived, failed and missing requested cases."""
        statuses = (
            ('ERROR missing_case: Testcase missing_case not found!', 'Not supported', ReturnCode.INVALID_ARGUMENT),
            ('Waived missing_case: no accessible peers', 'waived', ReturnCode.INVALID_ARGUMENT),
            ('Running missing_case.\nWaived: buffer not supported', 'waived', ReturnCode.INVALID_ARGUMENT),
            ('Running missing_case.\nWaived:', 'waived', ReturnCode.INVALID_ARGUMENT),
            (
                'Running missing_case.\nError: verification failed',
                'Error: verification failed',
                ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE,
            ),
            ('Running missing_case.', 'waived', ReturnCode.INVALID_ARGUMENT),
            ('', 'waived', ReturnCode.INVALID_ARGUMENT),
        )
        for status, raw_status, return_code in statuses:
            with self.subTest(status=status):
                benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases missing_case')
                assert benchmark._preprocess()
                assert benchmark._process_raw_result(0, results + '\n' + status)
                assert benchmark.return_code == return_code
                assert benchmark._result.raw_data['missing_case'] == [raw_status]
                assert benchmark.result['device_local_copy_gpu0_bw'] == [12000.0]
                assert not any(metric.startswith('missing_case_') for metric in benchmark.result)

                empty_benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases missing_case')
                assert empty_benchmark._preprocess()
                assert not empty_benchmark._process_raw_result(0, status)
                assert empty_benchmark.return_code != ReturnCode.SUCCESS

        # A late status must not be hidden by an earlier valid SUM for the same testcase.
        for status, return_code in (
            ('Error: verification failed', ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE),
            ('Waived: buffer not supported', ReturnCode.INVALID_ARGUMENT),
            ('Waived:', ReturnCode.INVALID_ARGUMENT),
        ):
            with self.subTest(late_status=status):
                benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases host_to_all_batch_memcpy_sm')
                assert benchmark._preprocess()
                assert benchmark._process_raw_result(0, results + '\n' + status)
                assert benchmark.return_code == return_code
                assert benchmark.result['host_to_all_batch_memcpy_sm_sum_bw'] == [270.0]

    @decorator.load_data('tests/data/nvbandwidth_v0.11.log')
    def test_nvbandwidth_v011_incomplete_matrices(self, results):
        """A first bidirectional matrix is useful partial data, not a complete successful testcase."""
        case = 'device_to_device_bidirectional_memcpy_read_ce'
        benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases ' + case)
        assert benchmark._preprocess()
        first_matrix = results.split('memcpy CE GPU(row) <-> GPU(column) Read2 bandwidth (GB/s)')[0]
        assert benchmark._process_raw_result(0, first_matrix)
        assert benchmark.return_code == ReturnCode.INVALID_ARGUMENT
        assert benchmark.result[f'{case}_gpu0_gpu1_bw'] == [41.0]
        assert benchmark.result[f'{case}_read1_sum_bw'] == [83.0]
        assert f'{case}_read2_sum_bw' not in benchmark.result
        assert f'{case}_total_sum_bw' not in benchmark.result

        # A native SUM of an all-N/A matrix is not a measured zero.
        benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='--test_cases device_to_device_memcpy_read_ce')
        assert benchmark._preprocess()
        raw_output = (
            'Running device_to_device_memcpy_read_ce.\n'
            'memcpy CE GPU(row) -> GPU(column) bandwidth (GB/s)\n'
            '           0\n'
            ' 0       N/A\n\n'
            'SUM device_to_device_memcpy_read_ce 0.00\n'
        )
        assert not benchmark._process_raw_result(0, raw_output)
        assert benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE
        assert set(benchmark.result) == {'return_code'}

    def test_nvbandwidth_preprocess_discovery_failure(self):
        """Do not launch an untracked run when discovery fails or returns no testcases."""
        for return_code, stdout, stderr in ((0, '', ''), (0, 'Index, Name:\n', ''), (1, '', 'list failed')):
            with self.subTest(return_code=return_code, stdout=stdout):
                benchmark = NvBandwidthBenchmark('nvbandwidth', parameters='')
                with mock.patch('subprocess.run') as mock_run:
                    mock_run.return_value.returncode = return_code
                    mock_run.return_value.stdout = stdout
                    mock_run.return_value.stderr = stderr
                    with self.assertLogs('superbench', level='ERROR'):
                        assert not benchmark._preprocess()
                assert benchmark.return_code == ReturnCode.MICROBENCHMARK_EXECUTION_FAILURE
                assert benchmark._commands == []

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

        # Call preprocess to initialize _args
        assert benchmark._preprocess()

        # Mock subprocess.run for successful execution with valid output
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = (
                '1, host_to_device_memcpy_ce:\n'
                '2, device_to_host_bidirectional_memcpy_ce:'
            )
            mock_run.return_value.stderr = ''
            test_cases = benchmark._get_all_test_cases()
            assert test_cases == ['host_to_device_memcpy_ce', 'device_to_host_bidirectional_memcpy_ce']

        # Mock subprocess.run for execution with non-zero return code
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1
            mock_run.return_value.stdout = ''
            mock_run.return_value.stderr = 'Error'
            test_cases = benchmark._get_all_test_cases()
            assert test_cases == []

        # Mock subprocess.run for execution with error message in stderr
        with mock.patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = ''
            mock_run.return_value.stderr = 'Error'
            test_cases = benchmark._get_all_test_cases()
            assert test_cases == []
