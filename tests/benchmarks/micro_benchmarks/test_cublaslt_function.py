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
        benchmark._args = SimpleNamespace(
            shapes=['16,16,16', '32,64,128'], in_types=['fp8e4m3'], log_raw_data=False, enable_ncu_profiling=False
        )
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

        benchmark._args = SimpleNamespace(
            shapes=['16,16,16', '32,64,128'], in_types=['fp8e4m3'], log_raw_data=False, enable_ncu_profiling=True
        )
        benchmark._result = BenchmarkResult(self.benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS, run_count=1)
        raw_output = """
        ==PROF== Connected to process 371693 (/opt/superbench/bin/cublaslt_gemm)
2208    2048    5608    0       358.154755      141.598150
==PROF== Disconnected from process 371693
"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value","Rule Name","Rule Type","Rule Description","Estimated Speedup Type","Estimated Speedup"
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","DRAM Frequency","hz","3995313115.40",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","SM Frequency","hz","1239496049.13",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","Elapsed Cycles","cycle","508757",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","Memory Throughput","%","21.68",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","DRAM Throughput","%","0.74",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","Duration","ns","406240",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","L1/TEX Cache Throughput","%","23.23",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","L2 Cache Throughput","%","15.12",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","SM Active Cycles","cycle","462061.91",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU Speed Of Light Throughput","Compute (SM) Throughput","%","23.59",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","SpeedOfLight","","","","SOLBottleneck","OPT","This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.","",""
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Block Size","","384",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Cluster Scheduling Policy","","PolicySpread",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Cluster Size","","0",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Function Cache Configuration","","CachePreferNone",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Grid Size","","288",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Registers Per Thread","register/thread","168",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Shared Memory Configuration Size","byte","233472",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Driver Shared Memory Per Block","byte/block","1024",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Dynamic Shared Memory Per Block","byte/block","205696",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Static Shared Memory Per Block","byte/block","0",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","# SMs","SM","152",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Stack Size","","1760",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Threads","thread","110592",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","# TPCs","","76",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Enabled TPC IDs","","all",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Uses Green Context","","0",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Launch Statistics","Waves Per SM","","1.89",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","LaunchStats","","","","LaunchConfiguration","OPT","If you execute __syncthreads() to synchronize the threads of a block, it is recommended to have at least two blocks per multiprocessor (compared to the currently executed 1.9 blocks) This way, blocks that aren't waiting for __syncthreads() can keep the hardware busy.","",""
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","LaunchStats","","","","LaunchConfiguration","OPT","A wave of thread blocks is defined as the maximum number of blocks that can be executed in parallel on the target GPU. The number of blocks in a wave depends on the number of multiprocessors and the theoretical occupancy of the kernel. This kernel launch results in 1 full waves and a partial wave of 136 thread blocks. Under the assumption of a uniform execution duration of all thread blocks, this partial wave may account for up to 50.0% of the total runtime of this kernel. Try launching a grid with no partial wave. The overall impact of this tail effect also lessens with the number of full waves executed for a grid. See the Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model) description for more details on launch configurations.","global","50"
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Max Active Clusters","cluster","0",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Max Cluster Size","block","8",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Overall GPU Occupancy","%","0",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Cluster Occupancy","%","0",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Block Limit Barriers","block","9",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Block Limit SM","block","32",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Block Limit Registers","block","1",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Block Limit Shared Mem","block","1",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Block Limit Warps","block","5",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Theoretical Active Warps per SM","warp","12",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Theoretical Occupancy","%","18.75",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Achieved Occupancy","%","15.53",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","Achieved Active Warps Per SM","warp","9.94",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","Occupancy","","","","TheoreticalOccupancy","OPT","The 3.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 16. This kernel's theoretical occupancy (18.8%) is limited by the number of required registers, and the required amount of shared memory.","local","81.25"
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Average DRAM Active Cycles","cycle","12018",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Total DRAM Elapsed Cycles","cycle","103875584",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Average L1 Active Cycles","cycle","462061.91",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Total L1 Elapsed Cycles","cycle","75238502",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Average L2 Active Cycles","cycle","791529.25",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Total L2 Elapsed Cycles","cycle","127183034",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Average SM Active Cycles","cycle","462061.91",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Total SM Elapsed Cycles","cycle","75238502",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Average SMSP Active Cycles","cycle","461606.38",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","GPU and Memory Workload Distribution","Total SMSP Elapsed Cycles","cycle","300954008",
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","WorkloadDistribution","","","","WorkloadImbalance","OPT","One or more SMs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 6.23% above the average, while the minimum instance value is 40.85% below the average.","global","5.818"
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","WorkloadDistribution","","","","WorkloadImbalance","OPT","One or more SMSPs have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 6.69% above the average, while the minimum instance value is 41.54% below the average.","global","6.24"
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","WorkloadDistribution","","","","WorkloadImbalance","OPT","One or more L1 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 6.23% above the average, while the minimum instance value is 40.85% below the average.","global","5.818"
"0","371693","cublaslt_gemm","127.0.0.1","cutlass3x_sm100_tensorop_s64x256x32gemm_f8_f8_f32_f16_f16_64x256x128_1x1x1_0_tnn_align4_1sm_bias_f16_relu_aux_scalemax","1","7","(384, 1, 1)","(9, 32, 1)","0","10.0","WorkloadDistribution","","","","WorkloadImbalance","OPT","One or more L2 Slices have a much lower number of active cycles than the average number of active cycles. Maximum instance value is 12.44% above the average, while the minimum instance value is 19.49% below the average.","global","14.55"
"""
        self.assertTrue(benchmark._process_raw_result(1, raw_output))
