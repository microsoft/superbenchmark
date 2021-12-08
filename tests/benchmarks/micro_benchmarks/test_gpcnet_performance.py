# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for GPCNet benchmark."""

import os
import numbers
import unittest
from pathlib import Path

from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType


class GPCNetBenchmarkTest(unittest.TestCase):    # noqa: E501
    """Tests for GPCNetBenchmark benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench'
        binary_path = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin')
        Path(binary_path).mkdir(parents=True, exist_ok=True)
        self.__binary_files = []
        for bin_name in ['network_test', 'network_load_test']:
            self.__binary_files.append(Path(binary_path, bin_name))
            Path(binary_path, bin_name).touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        for bin_file in self.__binary_files:
            bin_file.unlink()

    def test_gpcnet_network_test(self):
        """Test gpcnet-network-test benchmark."""
        raw_output = """# noqa: E501
Network Tests v1.3
  Test with 2 MPI ranks (2 nodes)

  Legend
   RR = random ring communication pattern
   Nat = natural ring communication pattern
   Lat = latency
   BW = bandwidth
   BW+Sync = bandwidth with barrier
+------------------------------------------------------------------------------+
|                            Isolated Network Tests                            |
+---------------------------------+--------------+--------------+--------------+
|                            Name |          Avg |          99% |        Units |
+---------------------------------+--------------+--------------+--------------+
|          RR Two-sided Lat (8 B) |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+
|                RR Get Lat (8 B) |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+
|      RR Two-sided BW (131072 B) |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+
|            RR Put BW (131072 B) |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+
| RR Two-sided BW+Sync (131072 B) |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+
|     Nat Two-sided BW (131072 B) |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+
|        Multiple Allreduce (8 B) |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+
|      Multiple Alltoall (4096 B) |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+
"""
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

        raw_output_no_execution = """
ERROR: this application must be run on at least 2 nodes
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[63697,1],0]
  Exit code:    1
--------------------------------------------------------------------------
"""
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

    def test_gpcnet_network_load(self):    # noqa: C901
        """Test gpcnet-network-load-test benchmark."""
        raw_output = """# noqa: E501
NetworkLoad Tests v1.3
  Test with 10 MPI ranks (10 nodes)
  2 nodes running Network Tests
  8 nodes running Congestion Tests (min 100 nodes per congestor)

  Legend
   RR = random ring communication pattern
   Lat = latency
   BW = bandwidth
   BW+Sync = bandwidth with barrier
+------------------------------------------------------------------------------------------------------------------------------------------+
|                                                          Isolated Network Tests                                                          |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|                            Name |          Min |          Max |          Avg |   Avg(Worst) |          99% |        99.9% |        Units |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|          RR Two-sided Lat (8 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| RR Two-sided BW+Sync (131072 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|        Multiple Allreduce (8 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+

+------------------------------------------------------------------------------------------------------------------------------------------+
|                                                        Isolated Congestion Tests                                                         |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|                            Name |          Min |          Max |          Avg |   Avg(Worst) |          99% |        99.9% |        Units |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|               Alltoall (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|       Two-sided Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|             Put Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|              Get Bcast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+

+------------------------------------------------------------------------------------------------------------------------------------------+
|                             Network Tests running with Congestion Tests (    RR Two-sided Lat Network Test)                              |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|                            Name |          Min |          Max |          Avg |   Avg(Worst) |          99% |        99.9% |        Units |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|          RR Two-sided Lat (8 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|               Alltoall (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|       Two-sided Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|             Put Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|              Get Bcast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+

+------------------------------------------------------------------------------------------------------------------------------------------+
|                             Network Tests running with Congestion Tests (RR Two-sided BW+Sync Network Test)                              |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|                            Name |          Min |          Max |          Avg |   Avg(Worst) |          99% |        99.9% |        Units |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
| RR Two-sided BW+Sync (131072 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|               Alltoall (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|       Two-sided Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|             Put Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|              Get Bcast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+

+------------------------------------------------------------------------------------------------------------------------------------------+
|                             Network Tests running with Congestion Tests (  Multiple Allreduce Network Test)                              |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|                            Name |          Min |          Max |          Avg |   Avg(Worst) |          99% |        99.9% |        Units |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|        Multiple Allreduce (8 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |         usec |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|               Alltoall (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|       Two-sided Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|             Put Incast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+
|              Get Bcast (4096 B) |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |      10000.0 |   MiB/s/rank |
+---------------------------------+--------------+--------------+--------------+--------------+--------------+--------------+--------------+

+------------------------------------------------------------------------------+
|          Network Tests running with Congestion Tests - Key Results           |
+---------------------------------+--------------------------------------------+
|                            Name |                   Congestion Impact Factor |
+---------------------------------+----------------------+---------------------+
|                                 |                  Avg |                 99% |
+---------------------------------+----------------------+---------------------+
|          RR Two-sided Lat (8 B) |                 0.0X |                0.0X |
+---------------------------------+----------------------+---------------------+
| RR Two-sided BW+Sync (131072 B) |                 0.0X |                0.0X |
+---------------------------------+----------------------+---------------------+
|        Multiple Allreduce (8 B) |                 0.0X |                0.0X |
+---------------------------------+----------------------+---------------------+
"""
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
        raw_output_no_execution = """
ERROR: this application must be run on at least 10 nodes
--------------------------------------------------------------------------
Primary job  terminated normally, but 1 process returned
a non-zero exit code. Per user-direction, the job has been aborted.
--------------------------------------------------------------------------
--------------------------------------------------------------------------
mpirun detected that one or more processes exited with non-zero status, thus causing
the job to be terminated. The first process to do so was:

  Process name: [[63697,1],0]
  Exit code:    1
--------------------------------------------------------------------------
"""
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
