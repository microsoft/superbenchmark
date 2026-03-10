# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for nvbench auto throughput benchmark."""

import unittest

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform


class TestNvbenchAutoThroughputBenchmark(BenchmarkTestCase, unittest.TestCase):
    """Test class for NVBench Auto Throughput benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/nvbench_auto_throughput'])

    def test_nvbench_auto_throughput_preprocess(self):
        """Test NVBench Auto Throughput benchmark preprocess."""
        benchmark_name = 'nvbench-auto-throughput'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test preprocess with default parameters
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Test preprocess with specified parameters
        parameters = ('--devices 0 ' '--stride "[1,2,4,8]" ' '--timeout 20 ' '--min-samples 100')
        benchmark = benchmark_class(benchmark_name, parameters=parameters)
        assert benchmark._preprocess()
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Check command
        assert (1 == len(benchmark._commands))
        assert ('--devices 0' in benchmark._commands[0])
        assert ('--axis "Stride=[1,2,4,8]"' in benchmark._commands[0])
        assert ('--timeout 20' in benchmark._commands[0])
        assert ('--min-samples 100' in benchmark._commands[0])

    def test_nvbench_auto_throughput_stride_formats(self):
        """Test NVBench Auto Throughput preprocess with different stride formats."""
        benchmark_name = 'nvbench-auto-throughput'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test single value
        benchmark = benchmark_class(benchmark_name, parameters='--stride "2"')
        assert benchmark._preprocess()
        assert '--axis "Stride=2"' in benchmark._commands[0]

        # Test list format
        benchmark = benchmark_class(benchmark_name, parameters='--stride "[1,2,4]"')
        assert benchmark._preprocess()
        assert '--axis "Stride=[1,2,4]"' in benchmark._commands[0]

        # Test range format
        benchmark = benchmark_class(benchmark_name, parameters='--stride "[1:8]"')
        assert benchmark._preprocess()
        assert '--axis "Stride=[1:8]"' in benchmark._commands[0]

        # Test range with step format
        benchmark = benchmark_class(benchmark_name, parameters='--stride "[1:8:2]"')
        assert benchmark._preprocess()
        assert '--axis "Stride=[1:8:2]"' in benchmark._commands[0]

        # Test default format
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert '--axis "Stride=[1:4]"' in benchmark._commands[0]
        assert '--axis "BlockSize=[128,256,512,1024]"' in benchmark._commands[0]

    def test_nvbench_auto_throughput_block_size_formats(self):
        """Test NVBench Auto Throughput preprocess with different block_size formats."""
        benchmark_name = 'nvbench-auto-throughput'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Test single value
        benchmark = benchmark_class(benchmark_name, parameters='--block_size "256"')
        assert benchmark._preprocess()
        assert '--axis "BlockSize=256"' in benchmark._commands[0]

        # Test list format
        benchmark = benchmark_class(benchmark_name, parameters='--block_size "[128,256,512]"')
        assert benchmark._preprocess()
        assert '--axis "BlockSize=[128,256,512]"' in benchmark._commands[0]

        # Test default format
        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()
        assert '--axis "BlockSize=[128,256,512,1024]"' in benchmark._commands[0]

    @decorator.load_data('tests/data/nvbench_auto_throughput.log')
    def test_nvbench_auto_throughput_result_parsing(self, results):
        """Test NVBench Auto Throughput benchmark result parsing."""
        benchmark_name = 'nvbench-auto-throughput'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()

        # Parse the provided raw output
        assert benchmark._process_raw_result(0, results)
        assert benchmark.return_code == ReturnCode.SUCCESS

        # Validate timing metrics for ItemsPerThread=1, Stride=1, BlockSize=128
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_cpu_time'][0], 120.0)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_gpu_time'][0], 100.0)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_batch_gpu_time'][0], 95.0)

        # Validate CUPTI metrics for ItemsPerThread=1, Stride=1, BlockSize=128
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_hbw_peak'][0], 20.0)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_load_eff'][0], 100.0)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_store_eff'][0], 100.0)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_l1_hit_rate'][0], 0.0)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_l2_hit_rate'][0], 0.05)

        # Validate throughput for ItemsPerThread=1, Stride=1, BlockSize=128 (100G elem/s * 4 bytes = 400 GB/s)
        self.assertAlmostEqual(benchmark.result['ipt_1_stride_1_blk_128_throughput'][0], 400.0)

        # Validate timing metrics for ItemsPerThread=2, Stride=4, BlockSize=256
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_cpu_time'][0], 220.0)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_gpu_time'][0], 200.0)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_batch_gpu_time'][0], 195.0)

        # Validate CUPTI metrics for ItemsPerThread=2, Stride=4, BlockSize=256
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_hbw_peak'][0], 80.0)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_load_eff'][0], 12.5)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_store_eff'][0], 100.0)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_l1_hit_rate'][0], 30.0)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_l2_hit_rate'][0], 10.0)

        # Validate throughput for ItemsPerThread=2, Stride=4, BlockSize=256 (200G elem/s * 4 bytes = 800 GB/s)
        self.assertAlmostEqual(benchmark.result['ipt_2_stride_4_blk_256_throughput'][0], 800.0)

    def test_nvbench_auto_throughput_invalid_output(self):
        """Test NVBench Auto Throughput benchmark result parsing with invalid output."""
        benchmark_name = 'nvbench-auto-throughput'
        (benchmark_class, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='')
        assert benchmark._preprocess()

        # Mock raw output with invalid format
        raw_output = 'Invalid output format'

        # Parse the provided raw output
        assert not benchmark._process_raw_result(0, raw_output)
        assert benchmark.return_code == ReturnCode.MICROBENCHMARK_RESULT_PARSING_FAILURE


if __name__ == '__main__':
    unittest.main()
