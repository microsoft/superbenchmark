# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPCNet benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType


class GPCNetBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for GPCNetBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/network_test', 'bin/network_load_test'])

    @decorator.load_data('tests/data/gpcnet_network_test.log')
    @decorator.load_data('tests/data/gpcnet_network_test_error.log')
    def test_gpcnet_network_test(self, raw_output, raw_output_no_execution):
        """Test gpcnet-network-test benchmark."""
        # Check registry.
        benchmark_name = 'gpcnet-network-test'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        benchmark = benchmark_class(benchmark_name)
        ret = benchmark._preprocess()
        assert (ret)

        expect_command = 'network_test'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output_no_execution))
        assert (len(benchmark.result) == benchmark.default_metric_count)

        # Check function process_raw_data.
        # Positive case - valid raw output.
        assert (benchmark._process_raw_result(0, raw_output))
        metric_list = [
            'rr_two-sided_lat',
            'rr_get_lat',
            'rr_two-sided_bw',
            'rr_put_bw',
            'rr_two-sided+sync_bw',
            'nat_two-sided_bw',
            'multiple_allreduce_time',
            'multiple_alltoall_bw',
        ]
        for metric_medium in metric_list:
            for suffix in ['avg', '99%']:
                metric = metric_medium + '_' + suffix
                assert (metric in benchmark.result)
                assert (len(benchmark.result[metric]) == 1)
                assert (isinstance(benchmark.result[metric][0], numbers.Number))

        # Negative case - Add invalid raw output.
        assert (benchmark._process_raw_result(0, 'ERROR') is False)

        # Check basic information.
        assert (benchmark.name == 'gpcnet-network-test')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'network_test')

    @decorator.load_data('tests/data/gpcnet_network_load.log')
    @decorator.load_data('tests/data/gpcnet_network_load_error.log')
    def test_gpcnet_network_load(self, raw_output, raw_output_no_execution):
        """Test gpcnet-network-load-test benchmark."""
        # Check registry.
        benchmark_name = 'gpcnet-network-load-test'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        benchmark = benchmark_class(benchmark_name)
        ret = benchmark._preprocess()
        assert (ret)

        expect_command = 'network_load_test'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        # Check function process_raw_data.
        assert (benchmark._process_raw_result(0, raw_output_no_execution))
        assert (len(benchmark.result) == benchmark.default_metric_count)
        # Positive case - valid raw output.
        assert (benchmark._process_raw_result(0, raw_output))
        metric_list = ['rr_two-sided_lat_x', 'rr_two-sided+sync_bw_x', 'multiple_allreduce_x']
        for metric_medium in metric_list:
            for suffix in ['avg', '99%']:
                metric = metric_medium + '_' + suffix
                assert (metric in benchmark.result)
                assert (len(benchmark.result[metric]) == 1)
                assert (isinstance(benchmark.result[metric][0], numbers.Number))

        # Negative case - Add invalid raw output.
        assert (benchmark._process_raw_result(0, 'ERROR') is False)

        # Check basic information.
        assert (benchmark.name == 'gpcnet-network-load-test')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'network_load_test')
