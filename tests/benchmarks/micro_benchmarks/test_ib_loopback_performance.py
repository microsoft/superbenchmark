# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ib-loopback benchmark."""

import os
import numbers
import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType, ReturnCode
from superbench.common.utils import network
from superbench.benchmarks.micro_benchmarks import ib_loopback_performance


class IBLoopbackBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for IBLoopbackBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/run_perftest_loopback'])

    def test_ib_loopback_util(self):
        """Test util functions 'get_numa_cores' and 'get_free_port' used in ib-loopback benchmark."""
        port = network.get_free_port()
        assert (isinstance(port, numbers.Number))
        numa_cores = ib_loopback_performance.get_numa_cores(0)
        if numa_cores is None:
            # in case no NUMA support available on test system
            return
        assert (len(numa_cores) >= 2)
        for i in range(len(numa_cores)):
            assert (isinstance(numa_cores[i], numbers.Number))

    @decorator.load_data('tests/data/ib_loopback_all_sizes.log')
    @mock.patch('superbench.benchmarks.micro_benchmarks.ib_loopback_performance.get_numa_cores')
    @mock.patch('superbench.common.utils.network.get_ib_devices')
    def test_ib_loopback_all_sizes(self, raw_output, mock_ib_devices, mock_numa_cores):
        """Test ib-loopback benchmark for all sizes."""
        # Test without ib devices
        # Check registry.
        benchmark_name = 'ib-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        # Negative case
        parameters = '--ib_index 0 --numa 0 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
        parameters = '--ib_index 0 --numa 0 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_numa_cores.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
        # Positive case
        parameters = '--ib_index 0 --numa 0 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        mock_ib_devices.return_value = ['mlx5_0']
        mock_numa_cores.return_value = [0, 1, 2, 3]
        os.environ['PROC_RANK'] = '0'
        os.environ['IB_DEVICES'] = '0,2,4,6'
        os.environ['NUMA_NODES'] = '1,0,3,2'
        ret = benchmark._preprocess()
        assert (ret)

        port = benchmark._IBLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -a -F --iters=2000 -d mlx5_0 -p {port} -x 0 --report_gbits'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output))

        # Check function process_raw_data.
        # Positive case - valid raw output.
        metric_list = []
        for ib_command in benchmark._args.commands:
            for size in ['8388608', '4194304', '1024', '2']:
                metric = 'ib_{}_bw_{}:{}'.format(ib_command, size, str(benchmark._args.ib_index))
                metric_list.append(metric)
        for metric in metric_list:
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        # Negative case - Add invalid raw output.
        assert (benchmark._process_raw_result(0, 'Invalid raw output') is False)

        # Check basic information.
        assert (benchmark.name == 'ib-loopback')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'run_perftest_loopback')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.ib_index == 0)
        assert (benchmark._args.numa == 1)
        assert (benchmark._args.iters == 2000)
        assert (benchmark._args.commands == ['write'])

    @decorator.load_data('tests/data/ib_loopback_8M_size.log')
    @mock.patch('superbench.benchmarks.micro_benchmarks.ib_loopback_performance.get_numa_cores')
    @mock.patch('superbench.common.utils.network.get_ib_devices')
    def test_ib_loopback_8M_size(self, raw_output, mock_ib_devices, mock_numa_cores):
        """Test ib-loopback benchmark for 8M size."""
        # Test without ib devices
        # Check registry.
        benchmark_name = 'ib-loopback'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Check preprocess
        # Negative case
        parameters = '--ib_index 0 --numa 0 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_ib_devices.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
        parameters = '--ib_index 0 --numa 0 --iters 2000'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        mock_numa_cores.return_value = None
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_DEVICE_GETTING_FAILURE)
        # Positive case
        parameters = '--ib_index 0 --numa 0 --iters 2000 --msg_size 8388608'
        benchmark = benchmark_class(benchmark_name, parameters=parameters)

        mock_ib_devices.return_value = ['mlx5_0']
        mock_numa_cores.return_value = [0, 1, 2, 3]
        ret = benchmark._preprocess()
        assert (ret)

        port = benchmark._IBLoopbackBenchmark__sock_fds[-1].getsockname()[1]
        expect_command = 'run_perftest_loopback 3 1 ' + benchmark._args.bin_dir + \
            f'/ib_write_bw -s 8388608 -F --iters=2000 -d mlx5_0 -p {port} -x 0 --report_gbits'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output))

        # Check function process_raw_data.
        # Positive case - valid raw output.
        metric_list = []
        for ib_command in benchmark._args.commands:
            metric = 'ib_{}_bw_8388608:{}'.format(ib_command, str(benchmark._args.ib_index))
            metric_list.append(metric)
        for metric in metric_list:
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        # Negative case - Add invalid raw output.
        assert (benchmark._process_raw_result(0, 'Invalid raw output') is False)

        # Check basic information.
        assert (benchmark.name == 'ib-loopback')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'run_perftest_loopback')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.ib_index == 0)
        assert (benchmark._args.numa == 0)
        assert (benchmark._args.iters == 2000)
        assert (benchmark._args.msg_size == 8388608)
        assert (benchmark._args.commands == ['write'])
