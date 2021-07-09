# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for mem-bw benchmark."""

import numbers
from pathlib import Path
import os
import unittest

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class MemBwTest(unittest.TestCase):
    """Tests for MemBw benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = os.path.join(os.getenv('SB_MICRO_PATH'), 'bin')
        Path(binary_path).mkdir(parents=True, exist_ok=True)
        self.__binary_file = Path(os.path.join(binary_path, 'bandwidthTest'))
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()

    def test_memory_bw_performance(self):
        """Test mem-bw benchmark."""
        benchmark_name = 'mem-bw'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--mode=shmoo --memory=pinned')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark)
        assert (benchmark.name == 'mem-bw')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check command list
        expected_command = [
            'bandwidthTest --htod mode=shmoo memory=pinned', 'bandwidthTest --dtoh mode=shmoo memory=pinned',
            'bandwidthTest --dtod mode=shmoo memory=pinned'
        ]
        for i in range(len(expected_command)):
            commnad = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            assert (commnad == expected_command[i])

        # Check results and metrics.
        raw_output = {}
        raw_output[0] = """
    [CUDA Bandwidth Test] - Starting...
    Running on...

    Device 0: A100-SXM4-40GB
    Quick Mode

    Host to Device Bandwidth, 1 Device(s)
    PINNED Memory Transfers
    Transfer Size (Bytes)	Bandwidth(GB/s)
    32000000			26.1

    Result = PASS
    """
        raw_output[1] = """
    [CUDA Bandwidth Test] - Starting...
    Running on...

    Device 0: A100-SXM4-40GB
    Quick Mode

    Device to Host Bandwidth, 1 Device(s)
    PINNED Memory Transfers
    Transfer Size (Bytes)	Bandwidth(GB/s)
    32000000			23.5

    Result = PASS
    """
        raw_output[2] = """
    NOTE: The CUDA Samples are not meant for performance measurements. Results may vary when GPU Boost is enabled.
    [CUDA Bandwidth Test] - Starting...
    Running on...

    Device 0: A100-SXM4-40GB
    Quick Mode

    Device to Device Bandwidth, 1 Device(s)
    PINNED Memory Transfers
    Transfer Size (Bytes)	Bandwidth(GB/s)
    32000000			1113.3

    Result = PASS
    """
        for i, metric in enumerate(['H2D_Mem_BW', 'D2H_Mem_BW', 'D2D_Mem_BW']):
            assert (benchmark._process_raw_result(i, raw_output[i]))
            assert (metric in benchmark.result)
            assert (len(benchmark.result[metric]) == 1)
            assert (isinstance(benchmark.result[metric][0], numbers.Number))
