# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for cublaslt-gemm benchmark."""

import unittest
from types import SimpleNamespace

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, ReturnCode, Platform
from superbench.benchmarks.result import BenchmarkResult


class CublasLtBenchmarkTestCase(BenchmarkTestCase, unittest.TestCase):
    """Class for cublaslt-gemm benchmark test cases."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'cublaslt-gemm'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/cublaslt_fp8_gemm'])

    def test_cublaslt_gemm_cls(self):
        """Test cublaslt-gemm benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.CUDA:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_cublaslt_gemm_result_parsing(self):
        """Test cublaslt-gemm benchmark result parsing."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.CUDA)
        benchmark = benchmark_cls(self.benchmark_name, parameters='')
        benchmark._args = SimpleNamespace(shapes=['16,16,16', '32,64,128'], in_type='fp8e4m3', log_raw_data=False)
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)

        # Positive case - valid raw output
        self.assertTrue(benchmark._process_raw_result(0, '16   16    16    0       1.111      2.222'))
        self.assertTrue(benchmark._process_raw_result(1, '32   64    128    0       1.111      2.222'))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertEqual(3, len(benchmark.result))
        for shape in benchmark._args.shapes:
            self.assertEqual(2.222, benchmark.result[f'fp8e4m3_{shape.replace(",", "_")}_flops'][0])

        # Negative case - invalid raw output
        self.assertFalse(benchmark._process_raw_result(1, 'cuBLAS API failed'))
