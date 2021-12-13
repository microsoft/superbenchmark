# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for cpu-memory-bw-latency benchmark."""

from pathlib import Path
import os
import unittest

from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform


class CpuMemBwLatencyBenchmarkTest(unittest.TestCase):
    """Test class for cpu-memory-bw-latency benchmark."""
    def setUp(self):
        """Method called to prepare the test fixture."""
        # Create fake binary file just for testing.
        self.__curr_micro_path = os.environ.get('SB_MICRO_PATH', '')
        os.environ['SB_MICRO_PATH'] = '/tmp/superbench/'
        binary_path = Path(os.getenv('SB_MICRO_PATH'), 'bin')
        binary_path.mkdir(parents=True, exist_ok=True)
        self.__binary_file = binary_path / 'mlc'
        self.__binary_file.touch(mode=0o755, exist_ok=True)

    def tearDown(self):
        """Method called after the test method has been called and the result recorded."""
        self.__binary_file.unlink()
        os.environ['SB_MICRO_PATH'] = self.__curr_micro_path

    def test_cpu_mem_bw_latency_benchmark_empty_param(self):
        """Test cpu-memory-bw-latency benchmark command generation with empty parameter."""
        benchmark_name = 'cpu-memory-bw-latency'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        default_mlc_test = 'bandwidth_matrix'
        benchmark = benchmark_class(benchmark_name, parameters='')

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'cpu-memory-bw-latency')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check commands
        assert (1 == len(benchmark._commands))
        assert ('mlc --%s;' % default_mlc_test in benchmark._commands[0])

    def test_cpu_mem_bw_latency_benchmark_result_parsing(self):
        """Test cpu-memory-bw-latency benchmark result parsing."""
        benchmark_name = 'cpu-memory-bw-latency'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CPU)
        assert (benchmark_class)

        all_mlc_tests = ['bandwidth_matrix', 'latency_matrix', 'max_bandwidth']
        param_str = '--tests %s' % ' '.join(all_mlc_tests)
        benchmark = benchmark_class(benchmark_name, parameters=param_str)

        # Check basic information
        assert (benchmark)
        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert (benchmark.name == 'cpu-memory-bw-latency')
        assert (benchmark.type == BenchmarkType.MICRO)

        # Check commands
        assert (len(all_mlc_tests) == len(benchmark._commands))
        for mlc_test, command in zip(all_mlc_tests, benchmark._commands):
            assert ('mlc --%s;' % mlc_test in command)

        # Positive case - valid bandwidth matrix output.
        test_raw_output = """
Intel(R) Memory Latency Checker - v3.9a
Command line parameters: --bandwidth_matrix

Using buffer size of 100.000MiB/thread for reads and an additional 100.000MiB/thread for writes
*** Unable to modify prefetchers (try executing 'modprobe msr')
*** So, enabling random access for latency measurements
Measuring Memory Bandwidths between nodes within system
Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
Using all the threads from each core if Hyper-threading is enabled
Using Read-only traffic type
                Numa node
Numa node            0       1
       0        82542.2 76679.9
       1        76536.0 82986.5
"""
        assert (benchmark._process_raw_result(0, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert ('raw_output_0' in benchmark.raw_data)
        assert ([test_raw_output] == benchmark.raw_data['raw_output_0'])
        assert ([82542.2] == benchmark.result['Mem_bandwidth_matrix_numa_0_0_BW'])
        assert ([76679.9] == benchmark.result['Mem_bandwidth_matrix_numa_0_1_BW'])
        assert ([76536.0] == benchmark.result['Mem_bandwidth_matrix_numa_1_0_BW'])
        assert ([82986.5] == benchmark.result['Mem_bandwidth_matrix_numa_1_1_BW'])

        # Positive case - valid latency matrix output.
        test_raw_output = """
Intel(R) Memory Latency Checker - v3.9a
Command line parameters: --latency_matrix

Using buffer size of 600.000MiB
*** Unable to modify prefetchers (try executing 'modprobe msr')
*** So, enabling random access for latency measurements
Measuring idle latencies (in ns)...
                Numa node
Numa node            0       1
       0          87.0   101.0
       1         101.9    86.9
"""
        assert (benchmark._process_raw_result(1, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert ('raw_output_1' in benchmark.raw_data)
        assert ([test_raw_output] == benchmark.raw_data['raw_output_1'])

        assert ([87.0] == benchmark.result['Mem_latency_matrix_numa_0_0_Latency'])
        assert ([101.0] == benchmark.result['Mem_latency_matrix_numa_0_1_Latency'])
        assert ([101.9] == benchmark.result['Mem_latency_matrix_numa_1_0_Latency'])
        assert ([86.9] == benchmark.result['Mem_latency_matrix_numa_1_1_Latency'])

        # Positive case - valid max bandwidth output.
        test_raw_output = """
Intel(R) Memory Latency Checker - v3.9a
Command line parameters: --max_bandwidth

Using buffer size of 100.000MiB/thread for reads and an additional 100.000MiB/thread for writes
*** Unable to modify prefetchers (try executing 'modprobe msr')
*** So, enabling random access for latency measurements

Measuring Maximum Memory Bandwidths for the system
Will take several minutes to complete as multiple injection rates will be tried to get the best bandwidth
Bandwidths are in MB/sec (1 MB/sec = 1,000,000 Bytes/sec)
Using all the threads from each core if Hyper-threading is enabled
Using traffic with the following read-write ratios
ALL Reads        :      165400.60
3:1 Reads-Writes :      154975.19
2:1 Reads-Writes :      158433.32
1:1 Reads-Writes :      157352.05
Stream-triad like:      157878.32

"""
        assert (benchmark._process_raw_result(2, test_raw_output))
        assert (benchmark.return_code == ReturnCode.SUCCESS)
        assert ('raw_output_2' in benchmark.raw_data)
        assert ([test_raw_output] == benchmark.raw_data['raw_output_2'])
        assert ([165400.60] == benchmark.result['Mem_max_bandwidth_ALL_Reads_BW'])
        assert ([154975.19] == benchmark.result['Mem_max_bandwidth_3_1_Reads-Writes_BW'])
        assert ([158433.32] == benchmark.result['Mem_max_bandwidth_2_1_Reads-Writes_BW'])
        assert ([157352.05] == benchmark.result['Mem_max_bandwidth_1_1_Reads-Writes_BW'])
        assert ([157878.32] == benchmark.result['Mem_max_bandwidth_Stream-triad_like_BW'])

        # Negative case - invalid raw output.
        assert (benchmark._process_raw_result(0, 'Invalid raw output') is False)
        assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE)
