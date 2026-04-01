# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for DTK gpu-hpcg benchmark."""

import unittest
from types import SimpleNamespace

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, BenchmarkType, Platform, ReturnCode
from superbench.benchmarks.result import BenchmarkResult


class DtkHpcgBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for DTK gpu-hpcg benchmark."""

    example_raw_output = """
rocHPCG version: 0.8.8-62f1830-dirty (based on hpcg-3.1)

Setup Phase took 0.12 sec

Starting Reference CG Phase ...


Optimization Phase took 0.25 sec

Validation Testing Phase ...

Optimized CG Setup ...

HIP Initial Residual = 2.668768e+04

Total device memory usage: 19550 MByte (29152 MByte)

Starting Benchmarking Phase ...

Performing (at least) 2 CG sets in 1.0 seconds ...
CG set 1 / 2    6881.2186 GFlop/s     (215.0381 GFlop/s per process)    50%    0.0 sec left
CG set 2 / 2    6904.9453 GFlop/s     (215.7795 GFlop/s per process)    100%    0.0 sec left

Local domain: 560 x 280 x 280
Global domain: 2240 x 1120 x 560
Process domain: 4 x 4 x 2

Total Time: 7.55 sec
Setup Time: 0.12 sec
Optimization Time: 0.25 sec

*** WARNING *** INVALID RUN

DDOT   =  5849.4 GFlop/s (46794.9 GB/s)     182.8 GFlop/s per process ( 1462.3 GB/s per process)
WAXPBY =  3052.0 GFlop/s (36623.8 GB/s)      95.4 GFlop/s per process ( 1144.5 GB/s per process)
SpMV   =  5473.9 GFlop/s (34468.8 GB/s)     171.1 GFlop/s per process ( 1077.1 GB/s per process)
MG     =  7716.9 GFlop/s (59557.1 GB/s)     241.2 GFlop/s per process ( 1861.2 GB/s per process)
Total  =  6971.0 GFlop/s (52859.9 GB/s)     217.8 GFlop/s per process ( 1651.9 GB/s per process)
Final  =  6904.9 GFlop/s (52359.0 GB/s)     215.8 GFlop/s per process ( 1636.2 GB/s per process)

*** WARNING *** THIS IS NOT A VALID RUN ***
"""

    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.benchmark_name = 'gpu-hpcg'
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/run_rochpcg'])

    def get_benchmark(self):
        """Get benchmark."""
        (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, Platform.DTK)
        benchmark = benchmark_cls(self.benchmark_name, parameters='')
        benchmark._args = SimpleNamespace(log_raw_data=False)
        benchmark._curr_run_index = 0
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)
        return benchmark

    def test_dtk_hpcg_cls(self):
        """Test DTK gpu-hpcg benchmark class."""
        for platform in Platform:
            (benchmark_cls, _) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(self.benchmark_name, platform)
            if platform is Platform.DTK:
                self.assertIsNotNone(benchmark_cls)
            else:
                self.assertIsNone(benchmark_cls)

    def test_dtk_hpcg_result_parsing_with_wrapper_noise(self):
        """Test DTK gpu-hpcg result parsing with wrapper noise."""
        benchmark = self.get_benchmark()

        self.assertTrue(benchmark._process_raw_result(0, self.example_raw_output))
        self.assertEqual(ReturnCode.SUCCESS, benchmark.return_code)

        self.assertEqual(6904.9, benchmark.result['final_gflops'][0])
        self.assertEqual(215.8, benchmark.result['final_gflops_per_process'][0])
        self.assertEqual(5849.4, benchmark.result['ddot_gflops'][0])
        self.assertEqual(46794.9, benchmark.result['ddot_bandwidth'][0])
        self.assertEqual(182.8, benchmark.result['ddot_gflops_per_process'][0])
        self.assertEqual(1462.3, benchmark.result['ddot_bandwidth_per_process'][0])
        self.assertEqual(3052.0, benchmark.result['waxpby_gflops'][0])
        self.assertEqual(36623.8, benchmark.result['waxpby_bandwidth'][0])
        self.assertEqual(5473.9, benchmark.result['spmv_gflops'][0])
        self.assertEqual(34468.8, benchmark.result['spmv_bandwidth'][0])
        self.assertEqual(7716.9, benchmark.result['mg_gflops'][0])
        self.assertEqual(59557.1, benchmark.result['mg_bandwidth'][0])
        self.assertEqual(6971.0, benchmark.result['total_gflops'][0])
        self.assertEqual(52859.9, benchmark.result['total_bandwidth'][0])
        self.assertEqual(217.8, benchmark.result['total_gflops_per_process'][0])
        self.assertEqual(1651.9, benchmark.result['total_bandwidth_per_process'][0])
        self.assertEqual(0.12, benchmark.result['setup_time'][0])
        self.assertEqual(0.25, benchmark.result['optimization_time'][0])
        self.assertEqual(7.55, benchmark.result['total_time'][0])
        self.assertEqual(0, benchmark.result['is_valid'][0])
        self.assertEqual(560, benchmark.result['local_domain_x'][0])
        self.assertEqual(280, benchmark.result['local_domain_y'][0])
        self.assertEqual(280, benchmark.result['local_domain_z'][0])
        self.assertEqual(2240, benchmark.result['global_domain_x'][0])
        self.assertEqual(1120, benchmark.result['global_domain_y'][0])
        self.assertEqual(560, benchmark.result['global_domain_z'][0])
        self.assertEqual(4, benchmark.result['process_domain_x'][0])
        self.assertEqual(4, benchmark.result['process_domain_y'][0])
        self.assertEqual(2, benchmark.result['process_domain_z'][0])
        self.assertIn('raw_output_0', benchmark.raw_data)

    def test_dtk_hpcg_result_parsing_valid_by_absence_of_invalid_markers(self):
        """Test DTK gpu-hpcg valid detection by absence of invalid markers."""
        benchmark = self.get_benchmark()
        valid_output = self.example_raw_output.replace('*** WARNING *** INVALID RUN', '')
        valid_output = valid_output.replace('*** WARNING *** THIS IS NOT A VALID RUN ***', '')

        self.assertTrue(benchmark._process_raw_result(0, valid_output))
        self.assertEqual(1, benchmark.result['is_valid'][0])

    def test_dtk_hpcg_result_parsing_failure_when_required_summary_is_missing(self):
        """Test DTK gpu-hpcg parsing failure when required summary is missing."""
        benchmark = self.get_benchmark()
        invalid_output = self.example_raw_output.replace(
            'Process domain: 4 x 4 x 2\n\n',
            '',
        )

        self.assertFalse(benchmark._process_raw_result(0, invalid_output))
