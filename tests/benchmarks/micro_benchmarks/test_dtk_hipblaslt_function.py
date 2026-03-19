# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DTK hipblaslt-bench benchmark."""

import unittest
from types import SimpleNamespace

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.result import BenchmarkResult


class DtkHipblasLtBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for DTK hipblaslt-bench benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'hipblaslt-gemm'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/hipblaslt-bench'])

    def get_benchmark(self):
        """Get benchmark."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.DTK)
        return benchmark_cls(self.benchmark_name, parameters='')

    def test_hipblaslt_gemm_cls(self):
        """Test DTK hipblaslt-bench benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.DTK:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_hipblaslt_gemm_command_generation(self):
        """Test DTK hipblaslt-bench benchmark command generation."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.DTK)
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters='--batch 4:2:-1 --shapes 2,4,8 --in_types fp16 fp32 fp64 int8',
        )
        self.assertFalse(benchmark._preprocess())
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=' --shapes 2,4,8 --in_types fp16 fp32 fp64 int8',
        )
        self.assertFalse(benchmark._preprocess())
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters=' --shapes 2:4,4:8 --in_types fp16 fp32',
        )
        self.assertFalse(benchmark._preprocess())
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters='--shapes 2:4,4:8,8:32 2:4,4:8,8:32:+4 --in_types fp16 fp32 bf16 fp8',
        )
        self.assertTrue(benchmark._preprocess())
        self.assertEqual((2 * 2 * 3 + 2 * 2 * 7) * len(benchmark._args.in_types), len(benchmark._commands))

        def cmd(t, b, m, n, k):
            if b == 0:
                return f'{benchmark._DtkHipBlasLtBenchmark__bin_path} ' + \
                    f'-m {m} -n {n} -k {k} -j 20 -i 50 {benchmark._in_type_map[t]}' + \
                    f' --transA {benchmark._args.transA} --transB {benchmark._args.transB}' + \
                    f' --initialization {benchmark._args.initialization}'
            else:
                return f'{benchmark._DtkHipBlasLtBenchmark__bin_path} ' + \
                    f'-m {m} -n {n} -k {k} -j 20 -i 50 {benchmark._in_type_map[t]} -b {b}' + \
                    f' --transA {benchmark._args.transA} --transB {benchmark._args.transB}' + \
                    f' --initialization {benchmark._args.initialization}'

        for _t in ['fp16', 'fp32', 'bf16', 'fp8']:
            for _m in [2, 4]:
                for _n in [4, 8]:
                    for _k in [8, 16, 32]:
                        self.assertIn(cmd(_t, 0, _m, _n, _k), benchmark._commands)
                    for _k in [8, 12, 16, 20, 24, 28, 32]:
                        self.assertIn(cmd(_t, 0, _m, _n, _k), benchmark._commands)

    def test_hipblaslt_gemm_result_parsing(self):
        """Test DTK hipblaslt-bench benchmark result parsing."""
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['4096,4096,4096'], in_types=['fp32'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        example_raw_output = """
hipBLASLt version: 1000
hipBLASLt git version: 4bd05bb5-dirty
Query device success: there are 1 devices
-------------------------------------------------------------------------------
Device ID 0 : BW150 gfx936:sramecc+:xnack-
with 68.7 GB memory, max. SCLK 1400 MHz, max. MCLK 1800 MHz, compute capability 9.3
maxGridDimX 2147483647, sharedMemPerBlock 65.5 KB, maxThreadsPerBlock 1024, warpSize 64
-------------------------------------------------------------------------------

Is supported 1 / Total solutions: 1
[0]:transA,transB,grouped_gemm,batch_count,m,n,k,alpha,lda,stride_a,beta,ldb,stride_b,ldc,stride_c,ldd,stride_d,a_type,b_type,c_type,d_type,compute_type,scaleA,scaleB,scaleC,scaleD,amaxD,activation_type,bias_vector,bias_type,hipblaslt-Gflops,us
    N,N,0,1,4096,4096,4096,1,4096,16777216,0,4096,16777216,4096,16777216,4096,16777216,f32_r,f32_r,f32_r,f32_r,f32_r,0,0,0,0,0,none,0,non-supported type,1595.18,86159.1
"""
        self.assertTrue(benchmark._process_raw_result(0, example_raw_output))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)
        self.assertEqual(2, len(benchmark.result))
        self.assertEqual(1.59518, benchmark.result['fp32_1_4096_4096_4096_flops'][0])

        self.assertFalse(benchmark._process_raw_result(1, 'HipBLAS API failed'))
