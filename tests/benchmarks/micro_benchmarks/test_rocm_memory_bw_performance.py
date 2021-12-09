# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mem-bw benchmark."""

import numbers
from pathlib import Path
import os
import unittest

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class RocmMemBwTest(unittest.TestCase):
    """Test class for rocm mem-bw benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin')
        Path(os.getenv('SB_MICRO_PATH'), 'bin').mkdir(parents=True, exist_ok=True)
        self.__binary_file = Path(binary_path, 'hipBusBandwidth')
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()

    def test_rocm_memory_bw_performance(self):
        """Test rocm mem-bw benchmark."""
        benchmark_name = 'mem-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.ROCM)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name)

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'mem-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list
        expected_command = ['hipBusBandwidth --h2d', 'hipBusBandwidth --d2h']
        for i in range(len(expected_command)):
            commnad = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (commnad == expected_command[i])

        # Check results and metrics.
        raw_output = {}
        raw_output[0] = """
Device:Device 738c Mem=32.0GB #CUs=120 Freq=1502Mhz  MallocMode=pinned
                   test        atts        units        median  mean    stddev  min     max
   H2D_Bandwidth_pinned      +064By       GB/sec        0.0000  0.0000  0.0000  0.0000  0.0000
   H2D_Bandwidth_pinned      +256By       GB/sec        0.0000  0.0000  0.0000  0.0000  0.0000
   H2D_Bandwidth_pinned      +512By       GB/sec        0.0000  0.0000  0.0000  0.0000  0.0000
   H2D_Bandwidth_pinned         1kB       GB/sec        0.0414  0.0411  0.0017  0.0189  0.0434
   H2D_Bandwidth_pinned         2kB       GB/sec        0.0828  0.0824  0.0018  0.0683  0.0862
   H2D_Bandwidth_pinned         4kB       GB/sec        0.1656  0.1652  0.0032  0.1374  0.1724
   H2D_Bandwidth_pinned         8kB       GB/sec        0.3268  0.3251  0.0117  0.1880  0.3425
   H2D_Bandwidth_pinned        16kB       GB/sec        0.6410  0.6365  0.0259  0.3597  0.6757
   H2D_Bandwidth_pinned        32kB       GB/sec        1.2422  1.2432  0.0278  0.9346  1.2987
   H2D_Bandwidth_pinned        64kB       GB/sec        2.3968  2.4161  0.1486  0.7242  2.6042
   H2D_Bandwidth_pinned       128kB       GB/sec        4.6786  4.6339  0.1310  4.1143  4.8162
   H2D_Bandwidth_pinned       256kB       GB/sec        7.8349  7.8369  0.1150  6.9093  8.0270
   H2D_Bandwidth_pinned       512kB       GB/sec        11.9963 11.9828 0.1287  11.2158 12.2201
   H2D_Bandwidth_pinned      1024kB       GB/sec        16.3342 16.3315 0.0956  16.0147 16.5823
   H2D_Bandwidth_pinned      2048kB       GB/sec        19.9790 19.9770 0.0853  19.7681 20.1635
   H2D_Bandwidth_pinned      4096kB       GB/sec        22.2706 22.2642 0.0552  22.0644 22.3847
   H2D_Bandwidth_pinned      8192kB       GB/sec        22.8232 22.7881 0.1669  21.3196 22.8930
   H2D_Bandwidth_pinned     16384kB       GB/sec        24.1521 24.1411 0.0429  24.0165 24.2162
   H2D_Bandwidth_pinned     32768kB       GB/sec        24.8695 24.7086 0.7491  20.6288 24.9035
   H2D_Bandwidth_pinned     65536kB       GB/sec        24.4840 24.0101 2.5769  6.1754  24.5292
   H2D_Bandwidth_pinned    131072kB       GB/sec        25.0487 24.9593 0.2601  24.1286 25.0711
   H2D_Bandwidth_pinned    262144kB       GB/sec        25.3280 25.2351 0.1788  24.8746 25.3498
   H2D_Bandwidth_pinned    524288kB       GB/sec        24.7523 24.6708 0.1586  24.3154 24.7880
         H2D_Timepinned      +064By           ms        0.0245  0.0253  0.0240  0.0232  0.7821
         H2D_Timepinned      +256By           ms        0.0243  0.0244  0.0013  0.0232  0.0546
         H2D_Timepinned      +512By           ms        0.0243  0.0244  0.0014  0.0230  0.0566
         H2D_Timepinned         1kB           ms        0.0242  0.0244  0.0016  0.0230  0.0530
         H2D_Timepinned         2kB           ms        0.0242  0.0243  0.0005  0.0232  0.0293
         H2D_Timepinned         4kB           ms        0.0242  0.0242  0.0005  0.0232  0.0291
         H2D_Timepinned         8kB           ms        0.0245  0.0247  0.0013  0.0234  0.0426
         H2D_Timepinned        16kB           ms        0.0250  0.0252  0.0015  0.0237  0.0445
         H2D_Timepinned        32kB           ms        0.0258  0.0258  0.0006  0.0246  0.0342
         H2D_Timepinned        64kB           ms        0.0271  0.0272  0.0045  0.0250  0.0898
         H2D_Timepinned       128kB           ms        0.0280  0.0283  0.0008  0.0272  0.0318
         H2D_Timepinned       256kB           ms        0.0334  0.0334  0.0005  0.0326  0.0379
         H2D_Timepinned       512kB           ms        0.0437  0.0437  0.0005  0.0429  0.0467
         H2D_Timepinned      1024kB           ms        0.0642  0.0642  0.0004  0.0632  0.0654
         H2D_Timepinned      2048kB           ms        0.1050  0.1050  0.0004  0.1040  0.1061
         H2D_Timepinned      4096kB           ms        0.1883  0.1884  0.0005  0.1874  0.1901
         H2D_Timepinned      8192kB           ms        0.3675  0.3681  0.0028  0.3664  0.3934
         H2D_Timepinned     16384kB           ms        0.6946  0.6950  0.0012  0.6928  0.6986
         H2D_Timepinned     32768kB           ms        1.3492  1.3595  0.0482  1.3474  1.6266
         H2D_Timepinned     65536kB           ms        2.7409  2.9163  1.1368  2.7358  10.8670
         H2D_Timepinned    131072kB           ms        5.3582  5.3780  0.0576  5.3534  5.5626
         H2D_Timepinned    262144kB           ms        10.5983 10.6379 0.0761  10.5892 10.7915
         H2D_Timepinned    524288kB           ms        21.6897 21.7622 0.1411  21.6585 22.0794

Note: results marked with (*) had missing values such as
might occur with a mixture of architectural capabilities.
    """
        raw_output[1] = """
Device:Device 738c Mem=32.0GB #CUs=120 Freq=1502Mhz  MallocMode=pinned
                   test        atts        units        median  mean    stddev  min     max
   D2H_Bandwidth_pinned      +064By       GB/sec        0.0000  0.0000  0.0000  0.0000  0.0000
   D2H_Bandwidth_pinned      +256By       GB/sec        0.0000  0.0000  0.0000  0.0000  0.0000
   D2H_Bandwidth_pinned      +512By       GB/sec        0.0000  0.0000  0.0000  0.0000  0.0000
   D2H_Bandwidth_pinned         1kB       GB/sec        0.0428  0.0426  0.0019  0.0114  0.0446
   D2H_Bandwidth_pinned         2kB       GB/sec        0.0850  0.0844  0.0034  0.0415  0.0893
   D2H_Bandwidth_pinned         4kB       GB/sec        0.1701  0.1687  0.0084  0.0504  0.1773
   D2H_Bandwidth_pinned         8kB       GB/sec        0.3378  0.3348  0.0168  0.1085  0.3546
   D2H_Bandwidth_pinned        16kB       GB/sec        0.6667  0.6606  0.0218  0.5618  0.6897
   D2H_Bandwidth_pinned        32kB       GB/sec        1.3072  1.2954  0.0663  0.5682  1.3605
   D2H_Bandwidth_pinned        64kB       GB/sec        2.5550  2.5339  0.0955  2.1382  2.6904
   D2H_Bandwidth_pinned       128kB       GB/sec        4.8162  4.7807  0.2331  2.0940  4.9621
   D2H_Bandwidth_pinned       256kB       GB/sec        8.2286  8.2192  0.1671  7.2456  8.5286
   D2H_Bandwidth_pinned       512kB       GB/sec        12.7930 12.7062 0.4407  7.1196  13.0478
   D2H_Bandwidth_pinned      1024kB       GB/sec        17.5603 17.4938 0.3921  12.7184 17.7989
   D2H_Bandwidth_pinned      2048kB       GB/sec        21.6275 21.5591 0.2233  20.6073 21.8076
   D2H_Bandwidth_pinned      4096kB       GB/sec        24.2708 24.2556 0.0942  23.5724 24.4292
   D2H_Bandwidth_pinned      8192kB       GB/sec        24.9287 24.9093 0.0733  24.7171 25.0359
   D2H_Bandwidth_pinned     16384kB       GB/sec        26.4588 26.1976 2.4387  1.9387  26.5191
   D2H_Bandwidth_pinned     32768kB       GB/sec        27.2939 27.1202 0.7941  23.2086 27.3277
   D2H_Bandwidth_pinned     65536kB       GB/sec        26.8278 26.7238 0.3894  24.7946 26.9000
   D2H_Bandwidth_pinned    131072kB       GB/sec        27.4751 27.3457 0.3968  25.4168 27.5098
   D2H_Bandwidth_pinned    262144kB       GB/sec        27.8236 27.7173 0.3072  26.7977 27.8525
   D2H_Bandwidth_pinned    524288kB       GB/sec        28.0193 27.9348 0.1912  27.4707 28.0314
        D2H_Time_pinned      +064By           ms        0.0229  0.0246  0.0457  0.0216  1.4690
        D2H_Time_pinned      +256By           ms        0.0232  0.0234  0.0013  0.0221  0.0378
        D2H_Time_pinned      +512By           ms        0.0234  0.0238  0.0063  0.0224  0.2091
        D2H_Time_pinned         1kB           ms        0.0234  0.0236  0.0028  0.0224  0.0875
        D2H_Time_pinned         2kB           ms        0.0235  0.0237  0.0014  0.0224  0.0482
        D2H_Time_pinned         4kB           ms        0.0235  0.0239  0.0031  0.0226  0.0794
        D2H_Time_pinned         8kB           ms        0.0237  0.0240  0.0027  0.0226  0.0738
        D2H_Time_pinned        16kB           ms        0.0240  0.0242  0.0009  0.0232  0.0285
        D2H_Time_pinned        32kB           ms        0.0245  0.0248  0.0021  0.0235  0.0563
        D2H_Time_pinned        64kB           ms        0.0254  0.0257  0.0011  0.0242  0.0304
        D2H_Time_pinned       128kB           ms        0.0272  0.0275  0.0026  0.0264  0.0626
        D2H_Time_pinned       256kB           ms        0.0318  0.0319  0.0007  0.0307  0.0362
        D2H_Time_pinned       512kB           ms        0.0410  0.0413  0.0024  0.0402  0.0736
        D2H_Time_pinned      1024kB           ms        0.0597  0.0599  0.0017  0.0589  0.0824
        D2H_Time_pinned      2048kB           ms        0.0970  0.0973  0.0010  0.0962  0.1018
        D2H_Time_pinned      4096kB           ms        0.1728  0.1729  0.0007  0.1717  0.1779
        D2H_Time_pinned      8192kB           ms        0.3365  0.3367  0.0010  0.3350  0.3394
        D2H_Time_pinned     16384kB           ms        0.6341  0.7147  0.7979  0.6326  8.6538
        D2H_Time_pinned     32768kB           ms        1.2294  1.2385  0.0420  1.2278  1.4458
        D2H_Time_pinned     65536kB           ms        2.5014  2.5117  0.0391  2.4947  2.7066
        D2H_Time_pinned    131072kB           ms        4.8850  4.9092  0.0748  4.8789  5.2806
        D2H_Time_pinned    262144kB           ms        9.6478  9.6860  0.1106  9.6377  10.0171
        D2H_Time_pinned    524288kB           ms        19.1607 19.2196 0.1333  19.1525 19.5434

Note: results marked with (*) had missing values such as
might occur with a mixture of architectural capabilities.
    """

        for i, metric in enumerate(['h2d_bw', 'd2h_bw']):
            assert (benchmark._process_raw_result(i, raw_output[i]))
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))

        assert (benchmark.result['h2d_bw'][0] == 25.2351)
        assert (benchmark.result['d2h_bw'][0] == 27.9348)
