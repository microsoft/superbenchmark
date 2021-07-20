# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for disk-performance benchmark."""

from pathlib import Path
import os
import unittest

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class DiskPerformanceTest(unittest.TestCase):
    """Test class for disk-performance benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin')
        Path(binary_path).mkdir(parents=True, exist_ok=True)
        self.__binary_file = Path(binary_path, 'fio')
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()

    def test_disk_performance_command_generation(self):
        """Test disk-performance benchmark command generation."""
        benchmark_name = 'disk-performance'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        # Test case 1: empty parameter
        benchmark = benchmark_class(benchmark_name, parameters='')

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-performance')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Command list should be empty
        assert (0 == len(benchmark._commands))

        filenames = ['/dev/nvme0n1', '/dev/nvme1n1']
        filename_option = '--filenames ' + ' '.join(filenames)

        # Test case 2: turn off all tests
        param_str = filename_option
        param_str += ' --enable_seq_precond=0'
        param_str += ' --rand_precond_time=0'
        param_str += ' --seq_read_runtime=0'
        param_str += ' --seq_write_runtime=0'
        param_str += ' --rand_read_runtime=0'
        param_str += ' --rand_write_runtime=0'
        benchmark = benchmark_class(benchmark_name, parameters=param_str)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-performance')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Command list should be empty
        assert (0 == len(benchmark._commands))

        # Test case 3: turn on all tests
        init_test_magic = 45
        curr_test_magic = init_test_magic
        param_str = filename_option
        # Sequential precondition
        param_str += ' --enable_seq_precond=1'
        # Random precondition
        param_str += ' --rand_precond_time=%d' % curr_test_magic
        curr_test_magic += 1
        # Seq/rand read/write
        for io_pattern in ['seq', 'rand']:
            for io_type in ['read', 'write']:
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
        assert (benchmark.name == 'disk-performance')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list
        # 2 files * (2 preconditions + 2 io_patterns * 2 io_types) = 12 commands
        assert (12 == len(benchmark._commands))

        # Check parameter assignments
        command_idx = 0

        for filename in filenames:
            curr_test_magic = init_test_magic

            # Sequential precondition
            assert ('--filename=%s' % filename in benchmark._commands[command_idx])
            command_idx += 1
            # Random precondition
            assert ('--filename=%s' % filename in benchmark._commands[command_idx])
            assert ('--runtime=%d' % curr_test_magic in benchmark._commands[command_idx])
            curr_test_magic += 1
            command_idx += 1
            # Seq/rand read/write
            for io_pattern in ['seq', 'rand']:
                for io_type in ['read', 'write']:
                    assert ('--filename=%s' % filename in benchmark._commands[command_idx])
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
                    command_idx += 1

    def test_disk_performance_result_parsing(self):
        """Test disk-performance benchmark result parsing."""
        benchmark_name = 'disk-performance'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'disk-performance')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Positive case - valid raw output.
        test_raw_output = """
{
  "fio version" : "fio-3.16",
  "timestamp" : 1626763278,
  "timestamp_ms" : 1626763278577,
  "time" : "Tue Jul 20 06:41:18 2021",
  "global options" : {
    "filename" : "/dev/nvme0n1",
    "ramp_time" : "10s",
    "runtime" : "30s",
    "iodepth" : "64",
    "numjobs" : "4",
    "randrepeat" : "1",
    "thread" : "1",
    "ioengine" : "libaio",
    "direct" : "1",
    "norandommap" : "1",
    "lat_percentiles" : "1",
    "group_reporting" : "1"
  },
  "jobs" : [
    {
      "jobname" : "rand_read_write",
      "groupid" : 0,
      "error" : 0,
      "eta" : 0,
      "elapsed" : 41,
      "job options" : {
        "name" : "rand_read",
        "rw" : "randrw",
        "bs" : "4096",
        "time_based" : "1"
      },
      "read" : {
        "io_bytes" : 10463010816,
        "io_kbytes" : 10217784,
        "bw_bytes" : 348743777,
        "bw" : 340570,
        "iops" : 85138.890741,
        "runtime" : 30002,
        "total_ios" : 2554337,
        "short_ios" : 0,
        "drop_ios" : 0,
        "slat_ns" : {
          "min" : 1332,
          "max" : 48691,
          "mean" : 2032.588341,
          "stddev" : 864.921965
        },
        "clat_ns" : {
          "min" : 278533,
          "max" : 10175655,
          "mean" : 1444476.063469,
          "stddev" : 300748.583131
        },
        "lat_ns" : {
          "min" : 280646,
          "max" : 10177629,
          "mean" : 1446562.147113,
          "stddev" : 300723.879349,
          "percentile" : {
            "1.000000" : 872448,
            "5.000000" : 1036288,
            "10.000000" : 1122304,
            "20.000000" : 1220608,
            "30.000000" : 1286144,
            "40.000000" : 1351680,
            "50.000000" : 1417216,
            "60.000000" : 1482752,
            "70.000000" : 1564672,
            "80.000000" : 1662976,
            "90.000000" : 1810432,
            "95.000000" : 1941504,
            "99.000000" : 2244608,
            "99.500000" : 2408448,
            "99.900000" : 3620864,
            "99.950000" : 4358144,
            "99.990000" : 6062080
          }
        },
        "bw_min" : 291288,
        "bw_max" : 380288,
        "bw_agg" : 99.999134,
        "bw_mean" : 340567.050000,
        "bw_dev" : 6222.338382,
        "bw_samples" : 240,
        "iops_min" : 72822,
        "iops_max" : 95072,
        "iops_mean" : 85141.733333,
        "iops_stddev" : 1555.582888,
        "iops_samples" : 240
      },
      "write" : {
        "io_bytes" : 10454208512,
        "io_kbytes" : 10209188,
        "bw_bytes" : 348450387,
        "bw" : 340283,
        "iops" : 85066.128925,
        "runtime" : 30002,
        "total_ios" : 2552154,
        "short_ios" : 0,
        "drop_ios" : 0,
        "slat_ns" : {
          "min" : 1383,
          "max" : 315361,
          "mean" : 2182.824623,
          "stddev" : 919.625590
        },
        "clat_ns" : {
          "min" : 433904,
          "max" : 6300941,
          "mean" : 1558511.433458,
          "stddev" : 207734.850159
        },
        "lat_ns" : {
          "min" : 441909,
          "max" : 6302845,
          "mean" : 1560749.444938,
          "stddev" : 207695.144244,
          "percentile" : {
            "1.000000" : 1155072,
            "5.000000" : 1269760,
            "10.000000" : 1318912,
            "20.000000" : 1384448,
            "30.000000" : 1449984,
            "40.000000" : 1499136,
            "50.000000" : 1531904,
            "60.000000" : 1597440,
            "70.000000" : 1646592,
            "80.000000" : 1728512,
            "90.000000" : 1826816,
            "95.000000" : 1908736,
            "99.000000" : 2072576,
            "99.500000" : 2179072,
            "99.900000" : 2605056,
            "99.950000" : 3031040,
            "99.990000" : 4358144
          }
        },
        "bw_min" : 288464,
        "bw_max" : 380080,
        "bw_agg" : 99.998134,
        "bw_mean" : 340276.650000,
        "bw_dev" : 6293.894521,
        "bw_samples" : 240,
        "iops_min" : 72116,
        "iops_max" : 95020,
        "iops_mean" : 85069.133333,
        "iops_stddev" : 1573.475038,
        "iops_samples" : 240
      },
      "trim" : {
        "io_bytes" : 0,
        "io_kbytes" : 0,
        "bw_bytes" : 0,
        "bw" : 0,
        "iops" : 0.000000,
        "runtime" : 0,
        "total_ios" : 0,
        "short_ios" : 0,
        "drop_ios" : 0,
        "slat_ns" : {
          "min" : 0,
          "max" : 0,
          "mean" : 0.000000,
          "stddev" : 0.000000
        },
        "clat_ns" : {
          "min" : 0,
          "max" : 0,
          "mean" : 0.000000,
          "stddev" : 0.000000
        },
        "lat_ns" : {
          "min" : 0,
          "max" : 0,
          "mean" : 0.000000,
          "stddev" : 0.000000,
          "percentile" : {
            "1.000000" : 0,
            "5.000000" : 0,
            "10.000000" : 0,
            "20.000000" : 0,
            "30.000000" : 0,
            "40.000000" : 0,
            "50.000000" : 0,
            "60.000000" : 0,
            "70.000000" : 0,
            "80.000000" : 0,
            "90.000000" : 0,
            "95.000000" : 0,
            "99.000000" : 0,
            "99.500000" : 0,
            "99.900000" : 0,
            "99.950000" : 0,
            "99.990000" : 0
          }
        },
        "bw_min" : 0,
        "bw_max" : 0,
        "bw_agg" : 0.000000,
        "bw_mean" : 0.000000,
        "bw_dev" : 0.000000,
        "bw_samples" : 0,
        "iops_min" : 0,
        "iops_max" : 0,
        "iops_mean" : 0.000000,
        "iops_stddev" : 0.000000,
        "iops_samples" : 0
      },
      "sync" : {
        "lat_ns" : {
          "min" : 0,
          "max" : 0,
          "mean" : 0.000000,
          "stddev" : 0.000000
        },
        "total_ios" : 0
      },
      "job_runtime" : 120004,
      "usr_cpu" : 4.833172,
      "sys_cpu" : 20.800973,
      "ctx" : 3542118,
      "majf" : 0,
      "minf" : 1263,
      "iodepth_level" : {
        "1" : 0.000000,
        "2" : 0.000000,
        "4" : 0.000000,
        "8" : 0.000000,
        "16" : 0.000000,
        "32" : 0.000000,
        ">=64" : 100.000000
      },
      "iodepth_submit" : {
        "0" : 0.000000,
        "4" : 100.000000,
        "8" : 0.000000,
        "16" : 0.000000,
        "32" : 0.000000,
        "64" : 0.000000,
        ">=64" : 0.000000
      },
      "iodepth_complete" : {
        "0" : 0.000000,
        "4" : 99.999922,
        "8" : 0.000000,
        "16" : 0.000000,
        "32" : 0.000000,
        "64" : 0.100000,
        ">=64" : 0.000000
      },
      "latency_ns" : {
        "2" : 0.000000,
        "4" : 0.000000,
        "10" : 0.000000,
        "20" : 0.000000,
        "50" : 0.000000,
        "100" : 0.000000,
        "250" : 0.000000,
        "500" : 0.000000,
        "750" : 0.000000,
        "1000" : 0.000000
      },
      "latency_us" : {
        "2" : 0.000000,
        "4" : 0.000000,
        "10" : 0.000000,
        "20" : 0.000000,
        "50" : 0.000000,
        "100" : 0.000000,
        "250" : 0.000000,
        "500" : 0.010000,
        "750" : 0.070126,
        "1000" : 1.756079
      },
      "latency_ms" : {
        "2" : 95.414131,
        "4" : 2.722457,
        "10" : 0.040830,
        "20" : 0.010000,
        "50" : 0.000000,
        "100" : 0.000000,
        "250" : 0.000000,
        "500" : 0.000000,
        "750" : 0.000000,
        "1000" : 0.000000,
        "2000" : 0.000000,
        ">=2000" : 0.000000
      },
      "latency_depth" : 64,
      "latency_target" : 0,
      "latency_percentile" : 100.000000,
      "latency_window" : 0
    }
  ],
  "disk_util" : [
    {
      "name" : "nvme0n1",
      "read_ios" : 3004914,
      "write_ios" : 3003760,
      "read_merges" : 0,
      "write_merges" : 0,
      "read_ticks" : 4269143,
      "write_ticks" : 4598453,
      "in_queue" : 11104,
      "util" : 99.840351
    }
  ]
}
"""
        jobname_prefix = 'disk_performance:/dev/nvme0n1:rand_read_write'
        assert (benchmark._process_raw_result(0, test_raw_output))

        # bs + <read, write> x <iops, 95th, 99th, 99.9th>
        assert (9 == len(benchmark.result.keys()))

        assert (1 == len(benchmark.result[jobname_prefix + ':bs']))
        assert (4096 == benchmark.result[jobname_prefix + ':bs'][0])

        assert (1 == len(benchmark.result[jobname_prefix + ':read:iops']))
        assert (85138.890741 == benchmark.result[jobname_prefix + ':read:iops'][0])
        assert (1 == len(benchmark.result[jobname_prefix + ':write:iops']))
        assert (85066.128925 == benchmark.result[jobname_prefix + ':write:iops'][0])

        assert (1 == len(benchmark.result[jobname_prefix + ':read:lat_ns:95.000000']))
        assert (1941504 == benchmark.result[jobname_prefix + ':read:lat_ns:95.000000'][0])
        assert (1 == len(benchmark.result[jobname_prefix + ':read:lat_ns:99.000000']))
        assert (2244608 == benchmark.result[jobname_prefix + ':read:lat_ns:99.000000'][0])
        assert (1 == len(benchmark.result[jobname_prefix + ':read:lat_ns:99.900000']))
        assert (3620864 == benchmark.result[jobname_prefix + ':read:lat_ns:99.900000'][0])

        assert (1 == len(benchmark.result[jobname_prefix + ':write:lat_ns:95.000000']))
        assert (1908736 == benchmark.result[jobname_prefix + ':write:lat_ns:95.000000'][0])
        assert (1 == len(benchmark.result[jobname_prefix + ':write:lat_ns:99.000000']))
        assert (2072576 == benchmark.result[jobname_prefix + ':write:lat_ns:99.000000'][0])
        assert (1 == len(benchmark.result[jobname_prefix + ':write:lat_ns:99.900000']))
        assert (2605056 == benchmark.result[jobname_prefix + ':write:lat_ns:99.900000'][0])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(1, 'Invalid raw output') is False)
