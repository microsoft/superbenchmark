import unittest

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform


class TestNvBandwidthBenchmark(BenchmarkTestCase, unittest.TestCase):
    """Test class for NV Bandwidth benchmark."""

    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/nvbandwidth'])

    def test_nvbandwidth_result_parsing_real_output(self):
        """Test NV Bandwidth benchmark result parsing."""
        benchmark_name = 'nvbandwidth'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')

        # Preprocess and validate command
        assert benchmark._preprocess()

        # Provided raw output
        raw_output = """
        nvbandwidth Version: v0.6
        Built from Git version:

        CUDA Runtime Version: 12040
        CUDA Driver Version: 12040
        Driver Version: 550.54.15

        Device 0: NVIDIA GH200 480GB (00000009:01:00)

        Running host_to_device_memcpy_ce.
        memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)
                0         1
        0    337.55    2142.4
        1    2142.4    337.55

        SUM host_to_device_memcpy_ce 337.55

        Running device_to_host_memcpy_ce.
        memcpy CE CPU(row) <- GPU(column) bandwidth (GB/s)
                0         1        2
        0    295.23     241.2    254.0
        1    241.2      295.2    254.0

        SUM device_to_host_memcpy_ce 295.23

        Waived:
        Waived:
        Waived:
        Running host_to_device_bidirectional_memcpy_ce.
        memcpy CE CPU(row) <-> GPU(column) bandwidth (GB/s)
                0
        0    160.02
        Waived:
        """

        # Parse the provided raw output
        assert benchmark._process_raw_result(0, raw_output)
        assert benchmark.return_code == ReturnCode.SUCCESS
        print("output is: \n")
        print(benchmark.result)

        # Validate parsed results
        assert benchmark.result['host_to_device_memcpy_ce_bandwidth_cpu0_gpu0'][0] == 337.55
        assert benchmark.result['host_to_device_memcpy_ce_bandwidth_cpu0_gpu1'][0] == 2142.4
        assert benchmark.result['device_to_host_memcpy_ce_bandwidth_cpu0_gpu1'][0] == 241.2
        assert benchmark.result['device_to_host_memcpy_ce_sum_bandwidth'][0] == 295.23
        assert 'host_to_device_bidirectional_memcpy_ce_bandwidth_cpu0_gpu0' in benchmark.result
        assert benchmark.result['host_to_device_bidirectional_memcpy_ce_bandwidth_cpu0_gpu0'][0] == 160.02
