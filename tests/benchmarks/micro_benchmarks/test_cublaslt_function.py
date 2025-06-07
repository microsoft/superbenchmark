# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for cublaslt-gemm benchmark."""

import unittest
from types import GeneratorType, SimpleNamespace

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.result import BenchmarkResult
from superbench.benchmarks.micro_benchmarks.blaslt_function_base import mrange, validate_mrange


class CublasLtBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for cublaslt-gemm benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'cublaslt-gemm'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/cublaslt_gemm'])

    def get_benchmark(self):
        """Get Benchmark."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        return benchmark_cls(self.benchmark_name, parameters='')

    def test_cublaslt_gemm_cls(self):
        """Test cublaslt-gemm benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.CUDA:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_mrange(self):
        """Test mrange generation."""
        self.assertIsInstance(mrange(1), GeneratorType)
        self.assertListEqual([4, 8, 16, 32], list(mrange(4, 32, 2)))
        self.assertListEqual([2, 4, 8, 16], list(mrange(2, 31, 2)))
        self.assertListEqual([2, 4, 8], list(mrange(2, 8)))
        self.assertListEqual([2], list(mrange(2, 0, 2)))
        self.assertListEqual([2], list(mrange(2)))
        self.assertListEqual([2], list(mrange(2, 4, 1)))
        self.assertListEqual([2], list(mrange(2, 4, 0)))
        self.assertListEqual([0], list(mrange(0, 0)))
        self.assertListEqual([0], list(mrange(0)))
        self.assertListEqual([4, 8, 16, 32], list(mrange(4, 32, 2, 'x')))
        self.assertListEqual([4, 8, 12, 16, 20, 24, 28, 32], list(mrange(4, 32, 4, '+')))

    def test_validate_mrange(self):
        """Test mrange validation."""
        self.assertTrue(validate_mrange('2:32:2'))
        self.assertTrue(validate_mrange('4:32'))
        self.assertTrue(validate_mrange('8'))
        self.assertFalse(validate_mrange('2:32:2:4'))
        self.assertFalse(validate_mrange('2.5:32'))
        self.assertFalse(validate_mrange('2:32:2:x4'))
        self.assertFalse(validate_mrange('2:32:2:+4'))

    def test_cublaslt_gemm_command_generation(self):
        """Test cublaslt-gemm benchmark command generation."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        benchmark = benchmark_cls(
            self.benchmark_name,
            parameters='--batch 2:16:2 --shapes 2:4,4:8,8:32 32:128:4,128,128 --in_types fp16 fp32 fp64 int8',
        )
        self.assertTrue(benchmark._preprocess())
        self.assertEqual(4 * (2 * 2 * 3 + 2) * len(benchmark._args.in_types), len(benchmark._commands))

        def cmd(t, b, m, n, k):
            return f'{benchmark._CublasLtBenchmark__bin_path} -m {m} -n {n} -k {k} -b {b} -w 20 -i 50 -t {t}'

        for _t in ['fp16', 'fp32', 'fp64', 'int8']:
            for _b in [2, 4, 8, 16]:
                for _m in [2, 4]:
                    for _n in [4, 8]:
                        for _k in [8, 16, 32]:
                            self.assertIn(cmd(_t, _b, _m, _n, _k), benchmark._commands)
                for _m in [32, 128]:
                    self.assertIn(cmd(_t, _b, _m, 128, 128), benchmark._commands)

    def test_cublaslt_gemm_result_parsing(self):
        """Test cublaslt-gemm benchmark result parsing."""
        benchmark = self.get_benchmark()
        self.assertTrue(benchmark._preprocess())
        benchmark._args = SimpleNamespace(shapes=['16,16,16', '32,64,128'], in_types=['fp8e4m3'], log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # Positive case - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, '16   16    16    0       1.111      2.222'))
        self.assertTrue(benchmark._process_raw_result(1, '32   64    128    0       1.111      2.222'))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertEqual(3, len(benchmark.result))
        for shape in benchmark._args.shapes:
            self.assertEqual(2.222, benchmark.result[f'fp8e4m3_0_{shape.replace(",", "_")}_flops'][0])

        # Negative case - invalid raw output
        self.assertFalse(benchmark._process_raw_result(1, 'cuBLAS API failed'))
