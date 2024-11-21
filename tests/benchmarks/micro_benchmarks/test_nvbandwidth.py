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
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test preprocess with default parameters
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Test preprocess with specified parameters
        parameters = (
            '--buffer_size 256 '
            '--test_cases 0,1,2,19,20 '
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
        assert ('--testcase 0 1 2 19 20' in benchmark._commands[0])
        assert ('--skipVerification' in benchmark._commands[0])
        assert ('--disableAffinity' in benchmark._commands[0])
        assert ('--useMean' in benchmark._commands[0])
        assert ('--testSamples 100' in benchmark._commands[0])

    @decorator.load_data('tests/data/nvbandwidth_results.log')
    def test_nvbandwidth_result_parsing_real_output(self, results):
        """Test NV Bandwidth benchmark result parsing."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
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
