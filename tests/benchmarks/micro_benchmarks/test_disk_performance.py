# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for disk-performance benchmark."""

import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class DiskBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Test class for disk-performance benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/fio'])

    def test_disk_performance_empty_param(self):
        """Test disk-performance benchmark command generation with empty parameter."""
        benchmark_name = 'disk-benchmark'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-benchmark')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Command list should be empty
        assert (0 == len(benchmark._commands))

    @mock.patch('pathlib.Path.is_block_device')
    def test_disk_performance_invalid_block_device(self, mock_is_block_device):
        """Test disk-performance benchmark command generation with invalid block device."""
        mock_is_block_device.return_value = False

        benchmark_name = 'disk-benchmark'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        block_devices = ['mock_block_device_0']
        block_device_option = '--block_devices ' + ' '.join(block_devices)

        benchmark = benchmark_class(benchmark_name, parameters=block_device_option)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is False)
        assert (benchmark.return_code == ReturnCode.INVALID_ARGUMENT)
        assert (benchmark.name == 'disk-benchmark')
        assert (benchmark.type == BenchmarkType.MICRO)

    @mock.patch('pathlib.Path.is_block_device')
    def test_disk_performance_benchmark_disabled(self, mock_is_block_device):
        """Test disk-performance benchmark command generation with all benchmarks disabled."""
        mock_is_block_device.return_value = True

        benchmark_name = 'disk-benchmark'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        block_devices = ['/dev/nvme0n1', '/dev/nvme1n1']
        block_device_option = '--block_devices ' + ' '.join(block_devices)

        param_str = block_device_option
        param_str += ' --rand_precond_time=0'
        param_str += ' --seq_read_runtime=0'
        param_str += ' --rand_read_runtime=0'
        benchmark = benchmark_class(benchmark_name, parameters=param_str)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-benchmark')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Command list should be empty
        assert (0 == len(benchmark._commands))

    @mock.patch('pathlib.Path.is_block_device')
    def test_disk_performance_benchmark_enabled(self, mock_is_block_device):
        """Test disk-performance benchmark command generation with all benchmarks enabled."""
        mock_is_block_device.return_value = True

        benchmark_name = 'disk-benchmark'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        block_devices = ['mock_block_device_0', 'mock_block_device_1']
        block_device_option = '--block_devices ' + ' '.join(block_devices)

        init_test_magic = 45
        curr_test_magic = init_test_magic
        param_str = block_device_option
        # Sequential precondition
        param_str += ' --enable_seq_precond'
        # Random precondition
        param_str += ' --rand_precond_time=%d' % curr_test_magic
        curr_test_magic += 1
        # Seq/rand read/write
        for io_pattern in ['seq', 'rand']:
            for io_type in ['read', 'write', 'readwrite']:
                io_str = '%s_%s' % (io_pattern, io_type)
                param_str += ' --%s_ramp_time=%d' % (io_str, curr_test_magic)
                curr_test_magic += 1
                param_str += ' --%s_runtime=%d' % (io_str, curr_test_magic)
                curr_test_magic += 1
                param_str += ' --%s_iodepth=%d' % (io_str, curr_test_magic)
                curr_test_magic += 1
                param_str += ' --%s_numjobs=%d' % (io_str, curr_test_magic)
                curr_test_magic += 1
        benchmark = benchmark_class(benchmark_name, parameters=param_str)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-benchmark')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list
        # 2 files * (2 preconditions + 3 io_patterns * 2 io_types) = 16 commands
        assert (16 == len(benchmark._commands))

        # Check parameter assignments
        command_idx = 0
        default_rwmixread = 80
        for block_device in block_devices:
            curr_test_magic = init_test_magic

            # Sequential precondition
            assert ('--filename=%s' % block_device in benchmark._commands[command_idx])
            command_idx += 1
            # Random precondition
            assert ('--filename=%s' % block_device in benchmark._commands[command_idx])
            assert ('--runtime=%d' % curr_test_magic in benchmark._commands[command_idx])
            curr_test_magic += 1
            command_idx += 1
            # Seq/rand read/write
            for io_pattern in ['seq', 'rand']:
                for io_type in ['read', 'write', 'rw']:
                    assert ('--filename=%s' % block_device in benchmark._commands[command_idx])
                    fio_rw = '%s%s' % (io_pattern if io_pattern == 'rand' else '', io_type)
                    assert ('--rw=%s' % fio_rw in benchmark._commands[command_idx])
                    assert ('--ramp_time=%d' % curr_test_magic in benchmark._commands[command_idx])
                    curr_test_magic += 1
                    assert ('--runtime=%d' % curr_test_magic in benchmark._commands[command_idx])
                    curr_test_magic += 1
                    assert ('--iodepth=%d' % curr_test_magic in benchmark._commands[command_idx])
                    curr_test_magic += 1
                    assert ('--numjobs=%d' % curr_test_magic in benchmark._commands[command_idx])
                    curr_test_magic += 1
                    if io_type == 'rw':
                        assert ('--rwmixread=%d' % default_rwmixread in benchmark._commands[command_idx])
                    command_idx += 1

    @decorator.load_data('tests/data/disk_performance.log')
    def test_disk_performance_result_parsing(self, test_raw_output):
        """Test disk-performance benchmark result parsing."""
        benchmark_name = 'disk-benchmark'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-benchmark')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Positive case - valid raw output.
        jobname_prefix = 'nvme0n1_rand_read_write'
        assert (benchmark._process_raw_result(0, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # bs + <read, write> x <iops, 95th, 99th, 99.9th>
        assert (9 + benchmark.default_metric_count == len(benchmark.result.keys()))

        assert (1 == len(benchmark.result[jobname_prefix + '_bs']))
        assert (4096 == benchmark.result[jobname_prefix + '_bs'][0])

        assert (1 == len(benchmark.result[jobname_prefix + '_read_iops']))
        assert (85138.890741 == benchmark.result[jobname_prefix + '_read_iops'][0])
        assert (1 == len(benchmark.result[jobname_prefix + '_write_iops']))
        assert (85066.128925 == benchmark.result[jobname_prefix + '_write_iops'][0])

        assert (1 == len(benchmark.result[jobname_prefix + '_read_lat_ns_95.0']))
        assert (1941504 == benchmark.result[jobname_prefix + '_read_lat_ns_95.0'][0])
        assert (1 == len(benchmark.result[jobname_prefix + '_read_lat_ns_99.0']))
        assert (2244608 == benchmark.result[jobname_prefix + '_read_lat_ns_99.0'][0])
        assert (1 == len(benchmark.result[jobname_prefix + '_read_lat_ns_99.9']))
        assert (3620864 == benchmark.result[jobname_prefix + '_read_lat_ns_99.9'][0])

        assert (1 == len(benchmark.result[jobname_prefix + '_write_lat_ns_95.0']))
        assert (1908736 == benchmark.result[jobname_prefix + '_write_lat_ns_95.0'][0])
        assert (1 == len(benchmark.result[jobname_prefix + '_write_lat_ns_99.0']))
        assert (2072576 == benchmark.result[jobname_prefix + '_write_lat_ns_99.0'][0])
        assert (1 == len(benchmark.result[jobname_prefix + '_write_lat_ns_99.9']))
        assert (2605056 == benchmark.result[jobname_prefix + '_write_lat_ns_99.9'][0])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(1, 'Invalid raw output') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
