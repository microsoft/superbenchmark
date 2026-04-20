# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DTK mem-bw benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class DtkMemBwTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for DTK mem-bw benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/BandwidthTest'])

    @decorator.load_data('tests/data/dtk_memory_h2d_bw.log')
    @decorator.load_data('tests/data/dtk_memory_d2h_bw.log')
    @decorator.load_data('tests/data/dtk_memory_d2d_bw.log')
    def test_dtk_memory_bw_performance(self, raw_output_h2d, raw_output_d2h, raw_output_d2d):
        """Test DTK mem-bw benchmark."""
        benchmark_name = 'mem-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.DTK)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name)

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'mem-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list.
        expected_command = [
            'BandwidthTest --type 0 --index 0 --mode 0',
            'BandwidthTest --type 1 --index 0 --mode 0',
            'BandwidthTest --type 2 --index 0',
        ]
        for i in range(len(expected_command)):
            command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (command == expected_command[i])

        # Check results and metrics.
        raw_output = [raw_output_h2d, raw_output_d2h, raw_output_d2d]
        for i, metric in enumerate(['h2d_bw', 'd2h_bw', 'd2d_bw']):
            assert (benchmark._process_raw_result(i, raw_output[i]))
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark.result['h2d_bw'][0] == 30.274)
        assert (benchmark.result['d2h_bw'][0] == 32.058)
        assert (benchmark.result['d2d_bw'][0] == 1431.655)

    def test_dtk_memory_bw_performance_unpinned_command(self):
        """Test DTK mem-bw unpinned command generation."""
        benchmark_name = 'mem-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.DTK)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--memory unpinned')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        expected_command = [
            'BandwidthTest --type 0 --index 0 --mode 1',
            'BandwidthTest --type 1 --index 0 --mode 1',
            'BandwidthTest --type 2 --index 0',
        ]
        for i in range(len(expected_command)):
            command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (command == expected_command[i])
