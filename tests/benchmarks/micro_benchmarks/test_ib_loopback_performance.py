# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ib-loopback benchmark."""

import os
import numbers
import unittest
from pathlib import Path
from unittest import mock

from superbench.benchmarks import BenchmarkRegistry, Platform, BenchmarkType, ReturnCode
from superbench.common.utils import network
from superbench.benchmarks.micro_benchmarks import ib_loopback_performance


class IBLoopbackBenchmarkTest(unittest.TestCase):
    """Tests for IBLoopbackBenchmark benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        if (len(network.get_ib_devices()) < 1):
            # Create fake binary file just for testing.
            os.environ['SB_MICRO_PATH'] = '/tmp/superbench'
            binary_path = Path(os.getenv('SB_MICRO_PATH'), 'bin')
            binary_path.mkdir(parents=True, exist_ok=True)
            self.__binary_file = Path(binary_path, 'run_perftest_loopback')
            self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        if (len(network.get_ib_devices()) < 1):
            self.__binary_file.unlink()

    def test_ib_loopback_util(self):
        """Test util functions 'get_numa_cores' and 'get_free_port' used in ib-loopback benchmark."""
        port = network.get_free_port()
        assert (isinstance(port, numbers.Number))
        numa_cores = ib_loopback_performance.get_numa_cores(0)
        assert (len(numa_cores) >= 2)
        for i in range(len(numa_cores)):
            assert (isinstance(numa_cores[i], numbers.Number))

    @mock.patch('superbench.common.utils.network.get_free_port')
    @mock.patch('superbench.benchmarks.micro_benchmarks.ib_loopback_performance.get_numa_cores')
    @mock.patch('superbench.common.utils.network.get_ib_devices')
    def test_ib_loopback_all_sizes(self, mock_ib_devices, mock_numa_cores, mock_port):
        """Test ib-loopback benchmark for all sizes."""
        raw_output = """
************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
Dual-port       : OFF          Device         : ibP257p0s0
Number of qps   : 1            Transport type : IB
Connection type : RC           Using SRQ      : OFF
PCIe relax order: ON
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
Dual-port       : OFF          Device         : ibP257p0s0
Number of qps   : 1            Transport type : IB
Connection type : RC           Using SRQ      : OFF
PCIe relax order: ON
ibv_wr* API     : ON
TX depth        : 128
CQ Moderation   : 100
Mtu             : 4096[B]
Link type       : IB
Max inline data : 0[B]
rdma_cm QPs     : OFF
Data ex. method : Ethernet
---------------------------------------------------------------------------------------
ibv_wr* API     : ON
CQ Moderation   : 100
Mtu             : 4096[B]
Link type       : IB
Max inline data : 0[B]
rdma_cm QPs     : OFF
Data ex. method : Ethernet
---------------------------------------------------------------------------------------
local address: LID 0xd06 QPN 0x092f PSN 0x3ff1bc RKey 0x080329 VAddr 0x007fc97ff50000
local address: LID 0xd06 QPN 0x092e PSN 0x3eb82d RKey 0x080228 VAddr 0x007f19adcbf000
remote address: LID 0xd06 QPN 0x092e PSN 0x3eb82d RKey 0x080228 VAddr 0x007f19adcbf000
remote address: LID 0xd06 QPN 0x092f PSN 0x3ff1bc RKey 0x080329 VAddr 0x007fc97ff50000
---------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------
#bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
#bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
2          2000             5.32               5.30               2.778732
4          2000             10.65              10.64              2.788833
8          2000             21.30              21.27              2.787609
16         2000             42.60              42.55              2.788268
32         2000             84.90              82.82              2.713896
64         2000             173.55             171.66             2.812504
128        2000             362.27             353.83             2.898535
256        2000             687.82             679.37             2.782698
512        2000             1337.12            1311.59            2.686135
1024       2000             2674.25            2649.39            2.712980
2048       2000             5248.56            5118.18            2.620509
4096       2000             10034.02            9948.41                   2.546793
8192       2000             18620.51            12782.56                  1.636168
16384      2000             23115.27            16782.50                  1.074080
32768      2000             22927.94            18586.03                  0.594753
65536      2000             23330.56            21167.79                  0.338685
131072     2000             22750.35            21443.14                  0.171545
262144     2000             22673.63            22411.35                  0.089645
524288     2000             22679.02            22678.86                  0.045358
1048576    2000             22817.06            22816.86                  0.022817
2097152    2000             22919.37            22919.27                  0.011460
4194304    2000             23277.93            23277.91                  0.005819
8388608    2000             23240.68            23240.68                  0.002905
---------------------------------------------------------------------------------------
8388608    2000             23240.68            23240.68                  0.002905
---------------------------------------------------------------------------------------
    """
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
        mock_port.return_value = 10000
        os.environ['PROC_RANK'] = '0'
        os.environ['IB_DEVICES'] = '0,2,4,6'
        os.environ['NUMA_NODES'] = '1,0,3,2'
        ret = benchmark._preprocess()
        assert (ret)

        expect_command = 'run_perftest_loopback 3 1 ib_write_bw -a -F --iters=2000 -d mlx5_0 -p 10000 -x 0'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output))

        # Check function process_raw_data.
        # Positive case - valid raw output.
        metric_list = []
        for ib_command in benchmark._args.commands:
            for size in ['8388608', '4194304', '1024', '2']:
                metric = 'IB_{}_{}_Avg_{}'.format(ib_command, size, str(benchmark._args.ib_index))
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

    @mock.patch('superbench.common.utils.network.get_free_port')
    @mock.patch('superbench.benchmarks.micro_benchmarks.ib_loopback_performance.get_numa_cores')
    @mock.patch('superbench.common.utils.network.get_ib_devices')
    def test_ib_loopback_8M_size(self, mock_ib_devices, mock_numa_cores, mock_port):
        """Test ib-loopback benchmark for 8M size."""
        raw_output = """
                        RDMA_Write BW Test
 Dual-port       : OFF		Device         : ibP257p0s0
 Number of qps   : 1		Transport type : IB
 Connection type : RC		Using SRQ      : OFF
 PCIe relax order: ON
 TX depth        : 128
 CQ Moderation   : 1
 Mtu             : 4096[B]
 Link type       : IB
 Max inline data : 0[B]
 rdma_cm QPs	 : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0xd06 QPN 0x095f PSN 0x3c9e82 RKey 0x080359 VAddr 0x007f9fc479c000
 remote address: LID 0xd06 QPN 0x095e PSN 0xbd024b RKey 0x080258 VAddr 0x007fe62504b000
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
 8388608    20000            24056.74            24056.72		   0.003007
************************************
* Waiting for client to connect... *
************************************
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF		Device         : ibP257p0s0
 Number of qps   : 1		Transport type : IB
 Connection type : RC		Using SRQ      : OFF
 PCIe relax order: ON
 CQ Moderation   : 1
 Mtu             : 4096[B]
 Link type       : IB
 Max inline data : 0[B]
 rdma_cm QPs	 : OFF
 Data ex. method : Ethernet
---------------------------------------------------------------------------------------
 local address: LID 0xd06 QPN 0x095e PSN 0xbd024b RKey 0x080258 VAddr 0x007fe62504b000
 remote address: LID 0xd06 QPN 0x095f PSN 0x3c9e82 RKey 0x080359 VAddr 0x007f9fc479c000
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[MB/sec]    BW average[MB/sec]   MsgRate[Mpps]
 8388608    20000            24056.74            24056.72		   0.003007
---------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------
"""
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
        mock_port.return_value = 10000
        ret = benchmark._preprocess()
        assert (ret)

        expect_command = 'run_perftest_loopback 3 1 ib_write_bw -s 8388608 -F --iters=2000 -d mlx5_0 -p 10000 -x 0'
        command = benchmark._bin_name + benchmark._commands[0].split(benchmark._bin_name)[1]
        assert (command == expect_command)

        assert (benchmark._process_raw_result(0, raw_output))

        # Check function process_raw_data.
        # Positive case - valid raw output.
        metric_list = []
        for ib_command in benchmark._args.commands:
            metric = 'IB_{}_8388608_Avg_{}'.format(ib_command, str(benchmark._args.ib_index))
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
