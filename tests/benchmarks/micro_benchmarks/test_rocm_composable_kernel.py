# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ROCm composable kernel benchmark."""

import unittest
from types import SimpleNamespace

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.result import BenchmarkResult


class composable_kernelBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for composable kernel benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'composable-kernel-gemm'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/ckProfiler'])

    def get_benchmark(self):
        """Get Benchmark."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.ROCM)
        return benchmark_cls(self.benchmark_name, parameters='')

    def test_composable_kernel_gemm_cls(self):
        """Test composable-kernel-gemm benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.ROCM:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_composable_kernel_gemm_command_generation(self):
        """Test composable-kernel-gemm benchmark command generation."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.ROCM)
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=' --shapes 2,4,8 --in_types fp16 fp32',
        )

        self.assertTrue(benchmark._preprocess())
        self.assertEqual(len(benchmark._args.in_types), len(benchmark._commands))

        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=' --shapes 2,4,8 --in_types fp16 fp32 --splitk 2 4 --streamk -1',
        )

        self.assertTrue(benchmark._preprocess())
        self.assertEqual(4 * len(benchmark._args.in_types), len(benchmark._commands))
        for _t in ['fp16', 'fp32']:
            params = f'{benchmark._in_type_map[_t]} 0 0 1 0 1 2 4 8 -1 -1 -1'
            command = f'{benchmark._RocmComposableKernelBenchmark__bin_path} gemm {params} {benchmark._args.num_warmup} {benchmark._args.num_steps}'
            assert (command in benchmark._commands)

            for splitk in [2, 4]:
                command = f'{benchmark._RocmComposableKernelBenchmark__bin_path} gemm_splitk {params} {splitk} {benchmark._args.num_warmup} {benchmark._args.num_steps}'
                assert (command in benchmark._commands)

            command = f'{benchmark._RocmComposableKernelBenchmark__bin_path} gemm_streamk {params} -1 {benchmark._args.num_warmup} {benchmark._args.num_steps}'
            assert (command in benchmark._commands)

    def test_composable_kernel_gemm_result_parsing(self):
        """Test composable-kernel-gemm benchmark result parsing."""
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['8192,8192,8192'], in_types=['fp16'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        example_raw_output = """
Perf:    17.0853 ms, 64.3544 TFlops, 23.5673 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B256_Vec8x1x4_512x16x4x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    51.8717 ms, 21.1967 TFlops, 7.76248 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B64_Vec8x1x4_16x16x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    51.2179 ms, 21.4673 TFlops, 7.86157 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B64_Vec8x1x4_16x16x16x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    24.4389 ms, 44.9902 TFlops, 16.4759 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x1x4_16x32x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    12.0388 ms, 91.331 TFlops, 33.4464 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x2x4_16x64x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    12.8774 ms, 85.3828 TFlops, 31.2681 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x4x4_16x128x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    14.7506 ms, 74.54 TFlops, 27.2974 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x8x4_16x256x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    12.0325 ms, 91.3782 TFlops, 33.4637 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B256_Vec8x4x4_16x256x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:     26.055 ms, 42.1996 TFlops, 15.4539 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x1x4_32x16x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    13.9292 ms, 78.9358 TFlops, 28.9072 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x1x4_64x16x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:     8.0511 ms, 136.567 TFlops, 50.0122 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x1x4_128x16x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    18.9246 ms, 58.0995 TFlops, 21.2767 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B128_Vec8x1x4_256x16x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Perf:    15.0647 ms, 72.986 TFlops, 26.7283 GB/s, GemmXdlSplitKCShuffle_MNKPadding_RRR_B256_Vec8x1x4_256x16x8x8 LoopScheduler: Default, PipelineVersion: v2, KBatch 2
Best Perf for datatype = f16 ALayout =  RowMajor BLayout =  RowMajor M = 8192 N = 8192 K = 8192 StrideA = 8192 StrideB = 8192 StrideC = 8192 KBatch = 2 : 2.17246 ms, 506.113 TFlops, 185.344 GB/s, GemmXdlSplitKCShuffle_Default_RRR_B256_Vec8x2x8_256x128x4x8 LoopScheduler: Default, PipelineVersion: v1
"""
        # Positive case - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, example_raw_output))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertEqual(2, len(benchmark.result))
        self.assertEqual(506.113, benchmark.result['f16_8192_8192_8192_flops'][0])
