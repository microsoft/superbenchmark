# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nccl-bw benchmark."""

import os
import numbers
import unittest
from pathlib import Path

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class NcclBwTest(unittest.TestCase):
    """Tests for NcclBw benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin')
        Path(binary_path).mkdir(parents=True, exist_ok=True)
        self.__binary_files = []
        for bin_name in ['all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf']:
            self.__binary_files.append(Path(os.path.join(binary_path, bin_name)))
            Path(os.path.join(binary_path, bin_name)).touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        for binary_file in self.__binary_files:
            binary_file.unlink()

    def test_nccl_bw_performance(self):
        """Test nccl-bw benchmark."""
        benchmark_name = 'nccl-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name)

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'nccl-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.algo == ['allreduce', 'allgather', 'broadcast', 'reduce', 'reducescatter'])
        assert (benchmark._args.gpu_count == 8)

        # Check command list
        bin_names = ['all_reduce_perf', 'all_gather_perf', 'broadcast_perf', 'reduce_perf', 'reduce_scatter_perf']
        print(benchmark._commands)
        for i in range(len(benchmark._args.algo)):
            commnad = bin_names[i] + benchmark._commands[i].split(bin_names[i])[1]
            expected_command = '{} -b 1 -e {} -f 2 -g {} -c 0'.format(
                bin_names[i], benchmark._args.max_size, benchmark._args.gpu_count
            )
            assert (commnad == expected_command)

        # Check results and metrics.
        raw_output = {}
        raw_output['allgather'] = """
# nThread 1 nGpus 8 minBytes 1 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 validation: 0
#
# Using devices
#   Rank  0 Pid 112372 on sb-validation-scus-000001 device  0 [0x00] A100-SXM4-40GB
#   Rank  1 Pid 112372 on sb-validation-scus-000001 device  1 [0x00] A100-SXM4-40GB
#   Rank  2 Pid 112372 on sb-validation-scus-000001 device  2 [0x00] A100-SXM4-40GB
#   Rank  3 Pid 112372 on sb-validation-scus-000001 device  3 [0x00] A100-SXM4-40GB
#   Rank  4 Pid 112372 on sb-validation-scus-000001 device  4 [0x00] A100-SXM4-40GB
#   Rank  5 Pid 112372 on sb-validation-scus-000001 device  5 [0x00] A100-SXM4-40GB
#   Rank  6 Pid 112372 on sb-validation-scus-000001 device  6 [0x00] A100-SXM4-40GB
#   Rank  7 Pid 112372 on sb-validation-scus-000001 device  7 [0x00] A100-SXM4-40GB
#
#                                             out-of-place                       in-place
#       size         count    type     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)             (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0   float    34.27    0.00    0.00    N/A    33.57    0.00    0.00    N/A
           0             0   float    33.41    0.00    0.00    N/A    33.62    0.00    0.00    N/A
           0             0   float    33.94    0.00    0.00    N/A    33.48    0.00    0.00    N/A
           0             0   float    33.83    0.00    0.00    N/A    33.62    0.00    0.00    N/A
           0             0   float    33.82    0.00    0.00    N/A    33.57    0.00    0.00    N/A
          32             1   float    35.03    0.00    0.00    N/A    34.15    0.00    0.00    N/A
          64             2   float    34.36    0.00    0.00    N/A    33.83    0.00    0.00    N/A
         128             4   float    33.94    0.00    0.00    N/A    35.22    0.00    0.00    N/A
         256             8   float    34.44    0.01    0.01    N/A    34.82    0.01    0.01    N/A
         512            16   float    34.84    0.01    0.01    N/A    34.76    0.01    0.01    N/A
        1024            32   float    35.38    0.03    0.03    N/A    34.53    0.03    0.03    N/A
        2048            64   float    34.67    0.06    0.05    N/A    34.91    0.06    0.05    N/A
        4096           128   float    34.62    0.12    0.10    N/A    34.81    0.12    0.10    N/A
        8192           256   float    34.76    0.24    0.21    N/A    35.03    0.23    0.20    N/A
       16384           512   float    34.80    0.47    0.41    N/A    34.90    0.47    0.41    N/A
       32768          1024   float    34.54    0.95    0.83    N/A    35.23    0.93    0.81    N/A
       65536          2048   float    36.34    1.80    1.58    N/A    36.01    1.82    1.59    N/A
      131072          4096   float    40.18    3.26    2.85    N/A    39.43    3.32    2.91    N/A
      262144          8192   float    46.45    5.64    4.94    N/A    46.27    5.67    4.96    N/A
      524288         16384   float    58.48    8.96    7.84    N/A    60.40    8.68    7.60    N/A
     1048576         32768   float    72.95   14.37   12.58    N/A    73.07   14.35   12.56    N/A
     2097152         65536   float    77.28   27.14   23.75    N/A    75.84   27.65   24.20    N/A
     4194304        131072   float    100.7   41.64   36.43    N/A    99.56   42.13   36.86    N/A
     8388608        262144   float    123.5   67.94   59.44    N/A    120.7   69.51   60.82    N/A
    16777216        524288   float    167.7  100.03   87.52    N/A    164.6  101.94   89.20    N/A
    33554432       1048576   float    265.8  126.24  110.46    N/A    257.5  130.33  114.04    N/A
    67108864       2097152   float    379.7  176.74  154.65    N/A    367.6  182.57  159.75    N/A
   134217728       4194304   float    698.6  192.13  168.12    N/A    657.3  204.20  178.67    N/A
   268435456       8388608   float   1192.2  225.16  197.01    N/A   1136.0  236.29  206.76    N/A
   536870912      16777216   float   2304.1  233.01  203.88    N/A   2227.9  240.98  210.85    N/A
  1073741824      33554432   float   4413.4  243.29  212.88    N/A   4258.8  252.12  220.61    N/A
  2147483648      67108864   float   8658.8  248.01  217.01    N/A   8389.4  255.98  223.98    N/A
  4294967296     134217728   float    17016  252.40  220.85    N/A    16474  260.71  228.12    N/A
  8589934592     268435456   float    33646  255.31  223.39    N/A    32669  262.94  230.07    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 58.2651
#
"""
        raw_output['allreduce'] = """
# nThread 1 nGpus 8 minBytes 1 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 validation: 0
#
# Using devices
#   Rank  0 Pid 112424 on sb-validation-scus-000001 device  0 [0x00] A100-SXM4-40GB
#   Rank  1 Pid 112424 on sb-validation-scus-000001 device  1 [0x00] A100-SXM4-40GB
#   Rank  2 Pid 112424 on sb-validation-scus-000001 device  2 [0x00] A100-SXM4-40GB
#   Rank  3 Pid 112424 on sb-validation-scus-000001 device  3 [0x00] A100-SXM4-40GB
#   Rank  4 Pid 112424 on sb-validation-scus-000001 device  4 [0x00] A100-SXM4-40GB
#   Rank  5 Pid 112424 on sb-validation-scus-000001 device  5 [0x00] A100-SXM4-40GB
#   Rank  6 Pid 112424 on sb-validation-scus-000001 device  6 [0x00] A100-SXM4-40GB
#   Rank  7 Pid 112424 on sb-validation-scus-000001 device  7 [0x00] A100-SXM4-40GB
#
#                                                     out-of-place                       in-place
#       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0   float     sum    35.20    0.00    0.00    N/A    34.05    0.00    0.00    N/A
           0             0   float     sum    34.18    0.00    0.00    N/A    33.50    0.00    0.00    N/A
           4             1   float     sum    34.73    0.00    0.00    N/A    35.30    0.00    0.00    N/A
           8             2   float     sum    34.66    0.00    0.00    N/A    34.84    0.00    0.00    N/A
          16             4   float     sum    35.00    0.00    0.00    N/A    35.61    0.00    0.00    N/A
          32             8   float     sum    35.60    0.00    0.00    N/A    35.27    0.00    0.00    N/A
          64            16   float     sum    34.83    0.00    0.00    N/A    34.61    0.00    0.00    N/A
         128            32   float     sum    34.53    0.00    0.01    N/A    43.78    0.00    0.01    N/A
         256            64   float     sum    34.56    0.01    0.01    N/A    34.95    0.01    0.01    N/A
         512           128   float     sum    34.94    0.01    0.03    N/A    35.20    0.01    0.03    N/A
        1024           256   float     sum    36.07    0.03    0.05    N/A    35.77    0.03    0.05    N/A
        2048           512   float     sum    35.42    0.06    0.10    N/A    35.89    0.06    0.10    N/A
        4096          1024   float     sum    35.92    0.11    0.20    N/A    36.11    0.11    0.20    N/A
        8192          2048   float     sum    35.91    0.23    0.40    N/A    36.07    0.23    0.40    N/A
       16384          4096   float     sum    36.18    0.45    0.79    N/A    35.87    0.46    0.80    N/A
       32768          8192   float     sum    36.65    0.89    1.56    N/A    35.73    0.92    1.60    N/A
       65536         16384   float     sum    37.82    1.73    3.03    N/A    37.25    1.76    3.08    N/A
      131072         32768   float     sum    41.19    3.18    5.57    N/A    41.11    3.19    5.58    N/A
      262144         65536   float     sum    47.53    5.52    9.65    N/A    47.94    5.47    9.57    N/A
      524288        131072   float     sum    60.32    8.69   15.21    N/A    60.52    8.66   15.16    N/A
     1048576        262144   float     sum    74.78   14.02   24.54    N/A    76.17   13.77   24.09    N/A
     2097152        524288   float     sum    93.48   22.43   39.26    N/A    96.10   21.82   38.19    N/A
     4194304       1048576   float     sum    112.0   37.44   65.52    N/A    110.2   38.06   66.60    N/A
     8388608       2097152   float     sum    162.0   51.79   90.63    N/A    160.0   52.44   91.77    N/A
    16777216       4194304   float     sum    226.0   74.23  129.90    N/A    225.0   74.57  130.49    N/A
    33554432       8388608   float     sum    374.3   89.65  156.89    N/A    372.8   90.00  157.50    N/A
    67108864      16777216   float     sum    584.5  114.81  200.91    N/A    581.9  115.33  201.82    N/A
   134217728      33554432   float     sum   1162.2  115.49  202.11    N/A   1162.5  115.46  202.05    N/A
   268435456      67108864   float     sum   2112.2  127.09  222.40    N/A   2111.8  127.11  222.45    N/A
   536870912     134217728   float     sum   4200.3  127.82  223.68    N/A   4184.0  128.32  224.55    N/A
  1073741824     268435456   float     sum   8159.5  131.59  230.29    N/A   8176.5  131.32  229.81    N/A
  2147483648     536870912   float     sum    16215  132.44  231.76    N/A    16203  132.53  231.93    N/A
  4294967296    1073741824   float     sum    32070  133.92  234.37    N/A    32052  134.00  234.50    N/A
  8589934592    2147483648   float     sum    63896  134.44  235.26    N/A    63959  134.30  235.03    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 68.4048
#
"""
        raw_output['reduce'] = """
# nThread 1 nGpus 8 minBytes 1 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 validation: 0
#
# Using devices
#   Rank  0 Pid 112476 on sb-validation-scus-000001 device  0 [0x00] A100-SXM4-40GB
#   Rank  1 Pid 112476 on sb-validation-scus-000001 device  1 [0x00] A100-SXM4-40GB
#   Rank  2 Pid 112476 on sb-validation-scus-000001 device  2 [0x00] A100-SXM4-40GB
#   Rank  3 Pid 112476 on sb-validation-scus-000001 device  3 [0x00] A100-SXM4-40GB
#   Rank  4 Pid 112476 on sb-validation-scus-000001 device  4 [0x00] A100-SXM4-40GB
#   Rank  5 Pid 112476 on sb-validation-scus-000001 device  5 [0x00] A100-SXM4-40GB
#   Rank  6 Pid 112476 on sb-validation-scus-000001 device  6 [0x00] A100-SXM4-40GB
#   Rank  7 Pid 112476 on sb-validation-scus-000001 device  7 [0x00] A100-SXM4-40GB
#
#                                                     out-of-place                       in-place
#       size         count    type   redop    root     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                             (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0   float     sum       0    36.90    0.00    0.00    N/A    36.47    0.00    0.00    N/A
           0             0   float     sum       0    34.18    0.00    0.00    N/A    35.70    0.00    0.00    N/A
           4             1   float     sum       0    35.40    0.00    0.00    N/A    35.59    0.00    0.00    N/A
           8             2   float     sum       0    36.35    0.00    0.00    N/A    35.74    0.00    0.00    N/A
          16             4   float     sum       0    35.47    0.00    0.00    N/A    34.27    0.00    0.00    N/A
          32             8   float     sum       0    36.16    0.00    0.00    N/A    36.19    0.00    0.00    N/A
          64            16   float     sum       0    35.61    0.00    0.00    N/A    35.45    0.00    0.00    N/A
         128            32   float     sum       0    34.78    0.00    0.00    N/A    35.80    0.00    0.00    N/A
         256            64   float     sum       0    35.37    0.01    0.01    N/A    35.89    0.01    0.01    N/A
         512           128   float     sum       0    35.49    0.01    0.01    N/A    35.53    0.01    0.01    N/A
        1024           256   float     sum       0    35.38    0.03    0.03    N/A    35.52    0.03    0.03    N/A
        2048           512   float     sum       0    35.97    0.06    0.06    N/A    35.13    0.06    0.06    N/A
        4096          1024   float     sum       0    36.03    0.11    0.11    N/A    35.82    0.11    0.11    N/A
        8192          2048   float     sum       0    36.80    0.22    0.22    N/A    36.71    0.22    0.22    N/A
       16384          4096   float     sum       0    35.37    0.46    0.46    N/A    36.79    0.45    0.45    N/A
       32768          8192   float     sum       0    35.16    0.93    0.93    N/A    35.72    0.92    0.92    N/A
       65536         16384   float     sum       0    38.08    1.72    1.72    N/A    37.74    1.74    1.74    N/A
      131072         32768   float     sum       0    43.07    3.04    3.04    N/A    41.59    3.15    3.15    N/A
      262144         65536   float     sum       0    52.16    5.03    5.03    N/A    50.49    5.19    5.19    N/A
      524288        131072   float     sum       0    67.58    7.76    7.76    N/A    66.57    7.88    7.88    N/A
     1048576        262144   float     sum       0    76.74   13.66   13.66    N/A    80.47   13.03   13.03    N/A
     2097152        524288   float     sum       0    78.51   26.71   26.71    N/A    78.76   26.63   26.63    N/A
     4194304       1048576   float     sum       0    81.47   51.48   51.48    N/A    80.30   52.23   52.23    N/A
     8388608       2097152   float     sum       0    94.72   88.57   88.57    N/A    94.06   89.19   89.19    N/A
    16777216       4194304   float     sum       0    137.7  121.83  121.83    N/A    139.6  120.17  120.17    N/A
    33554432       8388608   float     sum       0    218.3  153.70  153.70    N/A    218.1  153.83  153.83    N/A
    67108864      16777216   float     sum       0    370.8  180.96  180.96    N/A    369.8  181.49  181.49    N/A
   134217728      33554432   float     sum       0    661.0  203.06  203.06    N/A    659.9  203.39  203.39    N/A
   268435456      67108864   float     sum       0   1251.4  214.52  214.52    N/A   1268.1  211.68  211.68    N/A
   536870912     134217728   float     sum       0   2421.6  221.70  221.70    N/A   2413.4  222.45  222.45    N/A
  1073741824     268435456   float     sum       0   4736.0  226.72  226.72    N/A   4757.9  225.68  225.68    N/A
  2147483648     536870912   float     sum       0   9323.5  230.33  230.33    N/A   9354.0  229.58  229.58    N/A
  4294967296    1073741824   float     sum       0    18594  230.99  230.99    N/A    18570  231.28  231.28    N/A
  8589934592    2147483648   float     sum       0    37613  228.38  228.38    N/A    37539  228.83  228.83    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 65.018
#
"""
        raw_output['broadcast'] = """
# nThread 1 nGpus 8 minBytes 1 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 validation: 0
#
# Using devices
#   Rank  0 Pid 112528 on sb-validation-scus-000001 device  0 [0x00] A100-SXM4-40GB
#   Rank  1 Pid 112528 on sb-validation-scus-000001 device  1 [0x00] A100-SXM4-40GB
#   Rank  2 Pid 112528 on sb-validation-scus-000001 device  2 [0x00] A100-SXM4-40GB
#   Rank  3 Pid 112528 on sb-validation-scus-000001 device  3 [0x00] A100-SXM4-40GB
#   Rank  4 Pid 112528 on sb-validation-scus-000001 device  4 [0x00] A100-SXM4-40GB
#   Rank  5 Pid 112528 on sb-validation-scus-000001 device  5 [0x00] A100-SXM4-40GB
#   Rank  6 Pid 112528 on sb-validation-scus-000001 device  6 [0x00] A100-SXM4-40GB
#   Rank  7 Pid 112528 on sb-validation-scus-000001 device  7 [0x00] A100-SXM4-40GB
#
#                                                     out-of-place                       in-place
#       size         count    type    root     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0   float       0    34.61    0.00    0.00    N/A    34.33    0.00    0.00    N/A
           0             0   float       0    34.43    0.00    0.00    N/A    35.06    0.00    0.00    N/A
           4             1   float       0    33.96    0.00    0.00    N/A    33.80    0.00    0.00    N/A
           8             2   float       0    34.16    0.00    0.00    N/A    34.32    0.00    0.00    N/A
          16             4   float       0    34.47    0.00    0.00    N/A    34.85    0.00    0.00    N/A
          32             8   float       0    35.24    0.00    0.00    N/A    34.75    0.00    0.00    N/A
          64            16   float       0    35.12    0.00    0.00    N/A    34.89    0.00    0.00    N/A
         128            32   float       0    34.67    0.00    0.00    N/A    34.36    0.00    0.00    N/A
         256            64   float       0    34.23    0.01    0.01    N/A    34.42    0.01    0.01    N/A
         512           128   float       0    34.26    0.01    0.01    N/A    35.20    0.01    0.01    N/A
        1024           256   float       0    34.87    0.03    0.03    N/A    34.80    0.03    0.03    N/A
        2048           512   float       0    34.90    0.06    0.06    N/A    35.27    0.06    0.06    N/A
        4096          1024   float       0    35.37    0.12    0.12    N/A    34.59    0.12    0.12    N/A
        8192          2048   float       0    34.95    0.23    0.23    N/A    34.79    0.24    0.24    N/A
       16384          4096   float       0    34.94    0.47    0.47    N/A    34.94    0.47    0.47    N/A
       32768          8192   float       0    35.03    0.94    0.94    N/A    34.71    0.94    0.94    N/A
       65536         16384   float       0    36.04    1.82    1.82    N/A    36.48    1.80    1.80    N/A
      131072         32768   float       0    40.09    3.27    3.27    N/A    39.92    3.28    3.28    N/A
      262144         65536   float       0    46.58    5.63    5.63    N/A    45.89    5.71    5.71    N/A
      524288        131072   float       0    58.37    8.98    8.98    N/A    59.67    8.79    8.79    N/A
     1048576        262144   float       0    76.02   13.79   13.79    N/A    78.43   13.37   13.37    N/A
     2097152        524288   float       0    78.12   26.85   26.85    N/A    78.84   26.60   26.60    N/A
     4194304       1048576   float       0    81.06   51.74   51.74    N/A    80.39   52.17   52.17    N/A
     8388608       2097152   float       0    97.20   86.30   86.30    N/A    96.09   87.30   87.30    N/A
    16777216       4194304   float       0    143.1  117.22  117.22    N/A    142.1  118.06  118.06    N/A
    33554432       8388608   float       0    223.4  150.21  150.21    N/A    221.3  151.61  151.61    N/A
    67108864      16777216   float       0    374.8  179.05  179.05    N/A    374.4  179.23  179.23    N/A
   134217728      33554432   float       0    672.2  199.67  199.67    N/A    670.0  200.34  200.34    N/A
   268435456      67108864   float       0   1271.5  211.11  211.11    N/A   1264.5  212.28  212.28    N/A
   536870912     134217728   float       0   2436.3  220.37  220.37    N/A   2434.5  220.53  220.53    N/A
  1073741824     268435456   float       0   4769.2  225.14  225.14    N/A   4697.5  228.58  228.58    N/A
  2147483648     536870912   float       0   9314.2  230.56  230.56    N/A   9248.3  232.20  232.20    N/A
  4294967296    1073741824   float       0    18487  232.33  232.33    N/A    18381  233.66  233.66    N/A
  8589934592    2147483648   float       0    36896  232.81  232.81    N/A    36599  234.70  234.70    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 64.8653
#
"""
        raw_output['reducescatter'] = """
# nThread 1 nGpus 8 minBytes 1 maxBytes 8589934592 step: 2(factor) warmup iters: 5 iters: 20 validation: 0
#
# Using devices
#   Rank  0 Pid 112580 on sb-validation-scus-000001 device  0 [0x00] A100-SXM4-40GB
#   Rank  1 Pid 112580 on sb-validation-scus-000001 device  1 [0x00] A100-SXM4-40GB
#   Rank  2 Pid 112580 on sb-validation-scus-000001 device  2 [0x00] A100-SXM4-40GB
#   Rank  3 Pid 112580 on sb-validation-scus-000001 device  3 [0x00] A100-SXM4-40GB
#   Rank  4 Pid 112580 on sb-validation-scus-000001 device  4 [0x00] A100-SXM4-40GB
#   Rank  5 Pid 112580 on sb-validation-scus-000001 device  5 [0x00] A100-SXM4-40GB
#   Rank  6 Pid 112580 on sb-validation-scus-000001 device  6 [0x00] A100-SXM4-40GB
#   Rank  7 Pid 112580 on sb-validation-scus-000001 device  7 [0x00] A100-SXM4-40GB
#
#                                                     out-of-place                       in-place
#       size         count    type   redop     time   algbw   busbw  error     time   algbw   busbw  error
#        (B)    (elements)                     (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)
           0             0   float     sum    34.88    0.00    0.00    N/A    33.65    0.00    0.00    N/A
           0             0   float     sum    33.54    0.00    0.00    N/A    33.72    0.00    0.00    N/A
           0             0   float     sum    33.45    0.00    0.00    N/A    33.44    0.00    0.00    N/A
           0             0   float     sum    34.07    0.00    0.00    N/A    33.44    0.00    0.00    N/A
           0             0   float     sum    33.55    0.00    0.00    N/A    33.43    0.00    0.00    N/A
          32             1   float     sum    35.06    0.00    0.00    N/A    35.14    0.00    0.00    N/A
          64             2   float     sum    34.82    0.00    0.00    N/A    34.76    0.00    0.00    N/A
         128             4   float     sum    34.38    0.00    0.00    N/A    34.52    0.00    0.00    N/A
         256             8   float     sum    34.75    0.01    0.01    N/A    34.32    0.01    0.01    N/A
         512            16   float     sum    34.71    0.01    0.01    N/A    35.43    0.01    0.01    N/A
        1024            32   float     sum    35.16    0.03    0.03    N/A    34.75    0.03    0.03    N/A
        2048            64   float     sum    35.43    0.06    0.05    N/A    35.29    0.06    0.05    N/A
        4096           128   float     sum    35.49    0.12    0.10    N/A    35.17    0.12    0.10    N/A
        8192           256   float     sum    35.18    0.23    0.20    N/A    35.77    0.23    0.20    N/A
       16384           512   float     sum    35.27    0.46    0.41    N/A    35.49    0.46    0.40    N/A
       32768          1024   float     sum    35.00    0.94    0.82    N/A    35.09    0.93    0.82    N/A
       65536          2048   float     sum    36.78    1.78    1.56    N/A    36.92    1.77    1.55    N/A
      131072          4096   float     sum    40.71    3.22    2.82    N/A    39.78    3.29    2.88    N/A
      262144          8192   float     sum    48.12    5.45    4.77    N/A    46.65    5.62    4.92    N/A
      524288         16384   float     sum    59.81    8.77    7.67    N/A    58.88    8.90    7.79    N/A
     1048576         32768   float     sum    72.37   14.49   12.68    N/A    74.95   13.99   12.24    N/A
     2097152         65536   float     sum    80.64   26.01   22.76    N/A    79.62   26.34   23.05    N/A
     4194304        131072   float     sum    108.9   38.53   33.72    N/A    109.3   38.37   33.57    N/A
     8388608        262144   float     sum    147.3   56.96   49.84    N/A    166.8   50.28   44.00    N/A
    16777216        524288   float     sum    152.4  110.11   96.34    N/A    152.8  109.82   96.09    N/A
    33554432       1048576   float     sum    240.5  139.50  122.06    N/A    240.8  139.33  121.91    N/A
    67108864       2097152   float     sum    356.1  188.45  164.89    N/A    352.1  190.57  166.75    N/A
   134217728       4194304   float     sum    618.1  217.15  190.01    N/A    615.2  218.18  190.90    N/A
   268435456       8388608   float     sum   1108.7  242.11  211.84    N/A   1112.6  241.27  211.11    N/A
   536870912      16777216   float     sum   2169.0  247.52  216.58    N/A   2181.8  246.07  215.31    N/A
  1073741824      33554432   float     sum   4203.0  255.47  223.54    N/A   4206.3  255.27  223.36    N/A
  2147483648      67108864   float     sum   8356.9  256.97  224.85    N/A   8323.5  258.00  225.75    N/A
  4294967296     134217728   float     sum    16400  261.89  229.15    N/A    16402  261.86  229.13    N/A
  8589934592     268435456   float     sum    32464  264.60  231.52    N/A    32502  264.29  231.25    N/A
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 60.168
#
"""

        for i, bin_name in enumerate(benchmark._args.algo):
            assert (benchmark._process_raw_result(i, raw_output[bin_name]))
            assert (bin_name in benchmark.result)
            assert (len(benchmark.result[bin_name]) == 1)
            assert (isinstance(benchmark.result[bin_name][0], numbers.Number))
