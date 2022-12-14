# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mem-bw benchmark."""

import numbers
import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class CudaMemBwTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for cuda mem-bw benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/bandwidthTest'])

    @decorator.load_data('tests/data/cuda_memory_h2d_bw.log')
    @decorator.load_data('tests/data/cuda_memory_d2h_bw.log')
    @decorator.load_data('tests/data/cuda_memory_d2d_bw.log')
    def test_cuda_memory_bw_performance(self, raw_output_h2d, raw_output_d2h, raw_output_d2d):
        """Test cuda mem-bw benchmark."""
        benchmark_name = 'mem-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--shmoo_mode --memory=pinned')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'mem-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list
        expected_command = [
            'bandwidthTest --htod mode=shmoo memory=pinned --csv',
            'bandwidthTest --dtoh mode=shmoo memory=pinned --csv', 'bandwidthTest --dtod mode=shmoo memory=pinned --csv'
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

        benchmark = benchmark_class(benchmark_name, parameters='--memory=pinned --sleep 3')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check command list
        expected_command = [
            'bandwidthTest --htod memory=pinned --csv && sleep 3',
            'bandwidthTest --dtoh memory=pinned --csv && sleep 3', 'bandwidthTest --dtod memory=pinned --csv && sleep 3'
        ]
        for i in range(len(expected_command)):
            command = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (command == expected_command[i])
