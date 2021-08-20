# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mem-bw benchmark."""

import numbers
from pathlib import Path
import os
import unittest

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class CudaMemBwTest(unittest.TestCase):
    """Test class for cuda mem-bw benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin')
        Path(os.getenv('SB_MICRO_PATH'), 'bin').mkdir(parents=True, exist_ok=True)
        self.__binary_file = Path(binary_path, 'bandwidthTest')
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()

    def test_cuda_memory_bw_performance(self):
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
            'bandwidthTest --htod mode=shmoo memory=pinned --csv', 'bandwidthTest --dtoh mode=shmoo memory=pinned --csv'
        ]
        for i in range(len(expected_command)):
            commnad = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (commnad == expected_command[i])

        # Check results and metrics.
        raw_output = {}
        raw_output[0] = """
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla V100-PCIE-32GB
 Shmoo Mode

.................................................................................
bandwidthTest-H2D-Pinned, Bandwidth = 0.4 GB/s, Time = 0.00000 s, Size = 1000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 0.7 GB/s, Time = 0.00000 s, Size = 2000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 1.0 GB/s, Time = 0.00000 s, Size = 3000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 1.4 GB/s, Time = 0.00000 s, Size = 4000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 1.7 GB/s, Time = 0.00000 s, Size = 5000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 2.0 GB/s, Time = 0.00000 s, Size = 6000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 2.3 GB/s, Time = 0.00000 s, Size = 7000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 2.5 GB/s, Time = 0.00000 s, Size = 8000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 2.7 GB/s, Time = 0.00000 s, Size = 9000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 2.9 GB/s, Time = 0.00000 s, Size = 10000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 3.2 GB/s, Time = 0.00000 s, Size = 11000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 3.4 GB/s, Time = 0.00000 s, Size = 12000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 3.5 GB/s, Time = 0.00000 s, Size = 13000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 3.5 GB/s, Time = 0.00000 s, Size = 14000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 3.8 GB/s, Time = 0.00000 s, Size = 15000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 4.0 GB/s, Time = 0.00000 s, Size = 16000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 4.1 GB/s, Time = 0.00000 s, Size = 17000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 4.3 GB/s, Time = 0.00000 s, Size = 18000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 4.4 GB/s, Time = 0.00000 s, Size = 19000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 4.6 GB/s, Time = 0.00000 s, Size = 20000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 4.8 GB/s, Time = 0.00000 s, Size = 22000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 5.0 GB/s, Time = 0.00000 s, Size = 24000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 5.2 GB/s, Time = 0.00000 s, Size = 26000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 5.4 GB/s, Time = 0.00001 s, Size = 28000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 5.7 GB/s, Time = 0.00001 s, Size = 30000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 5.9 GB/s, Time = 0.00001 s, Size = 32000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 6.1 GB/s, Time = 0.00001 s, Size = 34000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 6.3 GB/s, Time = 0.00001 s, Size = 36000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 6.4 GB/s, Time = 0.00001 s, Size = 38000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 6.6 GB/s, Time = 0.00001 s, Size = 40000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 6.7 GB/s, Time = 0.00001 s, Size = 42000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 6.9 GB/s, Time = 0.00001 s, Size = 44000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 7.0 GB/s, Time = 0.00001 s, Size = 46000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 7.1 GB/s, Time = 0.00001 s, Size = 48000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 7.3 GB/s, Time = 0.00001 s, Size = 50000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 7.8 GB/s, Time = 0.00001 s, Size = 60000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 8.2 GB/s, Time = 0.00001 s, Size = 70000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 8.6 GB/s, Time = 0.00001 s, Size = 80000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 8.9 GB/s, Time = 0.00001 s, Size = 90000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 9.2 GB/s, Time = 0.00001 s, Size = 100000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 10.5 GB/s, Time = 0.00002 s, Size = 200000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.1 GB/s, Time = 0.00003 s, Size = 300000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.4 GB/s, Time = 0.00004 s, Size = 400000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.6 GB/s, Time = 0.00004 s, Size = 500000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.7 GB/s, Time = 0.00005 s, Size = 600000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.8 GB/s, Time = 0.00006 s, Size = 700000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.9 GB/s, Time = 0.00007 s, Size = 800000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.9 GB/s, Time = 0.00008 s, Size = 900000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.7 GB/s, Time = 0.00009 s, Size = 1000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.1 GB/s, Time = 0.00016 s, Size = 2000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.3 GB/s, Time = 0.00024 s, Size = 3000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.3 GB/s, Time = 0.00033 s, Size = 4000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.5 GB/s, Time = 0.00043 s, Size = 5000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.3 GB/s, Time = 0.00049 s, Size = 6000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.3 GB/s, Time = 0.00057 s, Size = 7000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.3 GB/s, Time = 0.00065 s, Size = 8000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.3 GB/s, Time = 0.00073 s, Size = 9000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00081 s, Size = 10000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00089 s, Size = 11000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00097 s, Size = 12000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00105 s, Size = 13000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00113 s, Size = 14000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00121 s, Size = 15000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00129 s, Size = 16000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00145 s, Size = 18000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00162 s, Size = 20000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00178 s, Size = 22000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00194 s, Size = 24000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00210 s, Size = 26000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00226 s, Size = 28000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00242 s, Size = 30000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 10.5 GB/s, Time = 0.00304 s, Size = 32000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.2 GB/s, Time = 0.00295 s, Size = 36000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 10.8 GB/s, Time = 0.00369 s, Size = 40000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00355 s, Size = 44000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00387 s, Size = 48000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.1 GB/s, Time = 0.00431 s, Size = 52000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 11.7 GB/s, Time = 0.00480 s, Size = 56000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00484 s, Size = 60000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.1 GB/s, Time = 0.00528 s, Size = 64000000 bytes, NumDevsUsed = 1
bandwidthTest-H2D-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00549 s, Size = 68000000 bytes, NumDevsUsed = 1
Result = PASS
    """
        raw_output[1] = """
[CUDA Bandwidth Test] - Starting...
Running on...

 Device 0: Tesla V100-PCIE-32GB
 Shmoo Mode

.................................................................................
bandwidthTest-D2H-Pinned, Bandwidth = 0.4 GB/s, Time = 0.00000 s, Size = 1000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 0.5 GB/s, Time = 0.00000 s, Size = 2000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 0.9 GB/s, Time = 0.00000 s, Size = 3000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 1.1 GB/s, Time = 0.00000 s, Size = 4000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 1.4 GB/s, Time = 0.00000 s, Size = 5000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 1.9 GB/s, Time = 0.00000 s, Size = 6000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 2.6 GB/s, Time = 0.00000 s, Size = 7000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 2.9 GB/s, Time = 0.00000 s, Size = 8000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 3.3 GB/s, Time = 0.00000 s, Size = 9000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 3.7 GB/s, Time = 0.00000 s, Size = 10000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 4.0 GB/s, Time = 0.00000 s, Size = 11000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 4.5 GB/s, Time = 0.00000 s, Size = 12000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 4.9 GB/s, Time = 0.00000 s, Size = 13000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 5.3 GB/s, Time = 0.00000 s, Size = 14000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 5.3 GB/s, Time = 0.00000 s, Size = 15000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 5.6 GB/s, Time = 0.00000 s, Size = 16000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 5.7 GB/s, Time = 0.00000 s, Size = 17000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 6.0 GB/s, Time = 0.00000 s, Size = 18000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 6.2 GB/s, Time = 0.00000 s, Size = 19000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 6.3 GB/s, Time = 0.00000 s, Size = 20000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 6.5 GB/s, Time = 0.00000 s, Size = 22000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 6.9 GB/s, Time = 0.00000 s, Size = 24000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 7.1 GB/s, Time = 0.00000 s, Size = 26000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 7.4 GB/s, Time = 0.00000 s, Size = 28000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 7.6 GB/s, Time = 0.00000 s, Size = 30000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 7.9 GB/s, Time = 0.00000 s, Size = 32000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 8.0 GB/s, Time = 0.00000 s, Size = 34000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 8.3 GB/s, Time = 0.00000 s, Size = 36000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 8.5 GB/s, Time = 0.00000 s, Size = 38000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 8.6 GB/s, Time = 0.00000 s, Size = 40000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 8.7 GB/s, Time = 0.00000 s, Size = 42000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 9.3 GB/s, Time = 0.00000 s, Size = 44000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 9.4 GB/s, Time = 0.00000 s, Size = 46000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 9.5 GB/s, Time = 0.00001 s, Size = 48000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 9.5 GB/s, Time = 0.00001 s, Size = 50000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 10.1 GB/s, Time = 0.00001 s, Size = 60000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 10.4 GB/s, Time = 0.00001 s, Size = 70000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 10.6 GB/s, Time = 0.00001 s, Size = 80000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 10.9 GB/s, Time = 0.00001 s, Size = 90000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 11.1 GB/s, Time = 0.00001 s, Size = 100000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.0 GB/s, Time = 0.00002 s, Size = 200000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.4 GB/s, Time = 0.00002 s, Size = 300000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.6 GB/s, Time = 0.00003 s, Size = 400000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.6 GB/s, Time = 0.00004 s, Size = 500000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.7 GB/s, Time = 0.00005 s, Size = 600000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.7 GB/s, Time = 0.00006 s, Size = 700000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.8 GB/s, Time = 0.00006 s, Size = 800000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.9 GB/s, Time = 0.00007 s, Size = 900000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.8 GB/s, Time = 0.00008 s, Size = 1000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.0 GB/s, Time = 0.00015 s, Size = 2000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.0 GB/s, Time = 0.00023 s, Size = 3000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00031 s, Size = 4000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00038 s, Size = 5000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00046 s, Size = 6000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00053 s, Size = 7000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00061 s, Size = 8000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.5 GB/s, Time = 0.00072 s, Size = 9000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00076 s, Size = 10000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00084 s, Size = 11000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00091 s, Size = 12000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00099 s, Size = 13000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00106 s, Size = 14000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00114 s, Size = 15000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00122 s, Size = 16000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00137 s, Size = 18000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00152 s, Size = 20000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00167 s, Size = 22000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00183 s, Size = 24000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 12.9 GB/s, Time = 0.00202 s, Size = 26000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00213 s, Size = 28000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00228 s, Size = 30000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00243 s, Size = 32000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00273 s, Size = 36000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00304 s, Size = 40000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00334 s, Size = 44000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00364 s, Size = 48000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00395 s, Size = 52000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00425 s, Size = 56000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.2 GB/s, Time = 0.00455 s, Size = 60000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00487 s, Size = 64000000 bytes, NumDevsUsed = 1
bandwidthTest-D2H-Pinned, Bandwidth = 13.1 GB/s, Time = 0.00520 s, Size = 68000000 bytes, NumDevsUsed = 1
Result = PASS
    """

        for i, metric in enumerate(['H2D_Mem_BW', 'D2H_Mem_BW']):
            assert (benchmark._process_raw_result(i, raw_output[i]))
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))
