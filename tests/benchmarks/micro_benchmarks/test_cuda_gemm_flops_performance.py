# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gemm-flops benchmark."""

import fnmatch
import os
import unittest
from unittest import mock

from tests.helper import decorator
from tests.helper.testcase import BenchmarkTestCase
from superbench.common.utils import device_manager as dm
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform, BenchmarkType


class CudaGemmFlopsBenchmarkTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for CudaGemmFlopsBenchmark benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/cutlass_profiler', 'cutlass_sm107/bin/cutlass_profiler'])

    @decorator.cuda_test
    def test_flops_performance_cuda(self):
        """Test gemm-flops benchmark."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        # Negative case - MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE.
        benchmark = benchmark_class(
            benchmark_name,
            parameters='--num_warmup 200 --n 1024 --k 512 --m 2048 --precision fp32 tf32_tc fp16_tc int8_tc'
        )

        ret = benchmark._preprocess()
        if dm.device_manager.get_device_compute_capability() not in benchmark._CudaGemmFlopsBenchmark__kernel_map:
            assert (ret is False)
            assert (benchmark.return_code == ReturnCode.MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE)
        else:
            assert (ret is True)
            assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark.name == 'gemm-flops')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'cutlass_profiler')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.num_warmup == 200)
        assert (benchmark._args.n == 1024)
        assert (benchmark._args.k == 512)
        assert (benchmark._args.m == 2048)
        assert (benchmark._args.precision == ['fp32', 'tf32_tc', 'fp16_tc', 'int8_tc'])
        benchmark._CudaGemmFlopsBenchmark__precision_need_to_run = ['fp32', 'tf32_tc', 'fp16_tc', 'int8_tc']

        # Check results and metrics.
        raw_output_fp32 = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,alpha,beta,split_k_slices,batch_count,op_class,accum,cta_m,cta_n,cta_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_simt_sgemm_128x128_8x2_nn_align1,passed,success,universal,16384,16384,16384,f32:column,f32:column,f32:column,1,0,1,1,simt,f32,128,128,8,2,4,2,1,1,1,1,50,1024,3221225472,8796629893120,481.022,6.23672,18287.4
1,CUTLASS,gemm,cutlass_simt_sgemm_128x128_8x2_nt_align1,passed,success,universal,16384,16384,16384,f32:column,f32:row,f32:column,1,0,1,1,simt,f32,128,128,8,2,4,2,1,1,1,1,50,1024,3221225472,8796629893120,478.866,6.2648,18369.7
1,CUTLASS,gemm,cutlass_simt_sgemm_128x128_8x2_tn_align1,passed,success,universal,16384,16384,16384,f32:row,f32:column,f32:column,1,0,1,1,simt,f32,128,128,8,2,4,2,1,1,1,1,50,1024,3221225472,8796629893120,482.034,6.22363,18249
1,CUTLASS,gemm,cutlass_simt_sgemm_128x128_8x2_tt_align1,passed,success,universal,16384,16384,16384,f32:row,f32:row,f32:column,1,0,1,1,simt,f32,128,128,8,2,4,2,1,1,1,1,50,1024,3221225472,8796629893120,481.838,6.22616,18256.4
"""
        raw_output_tf32_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,alpha,beta,split_k_slices,batch_count,op_class,accum,cta_m,cta_n,cta_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_nn_align4,passed,success,universal,16384,16384,16384,tf32:column,tf32:column,tf32:column,1,0,1,1,tensorop,f32,256,128,16,3,4,2,1,16,8,8,80,1024,3221225472,8796629893120,88.5764,33.8691,99311.2
1,CUTLASS,gemm,cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_nt_align4,passed,success,universal,16384,16384,16384,tf32:column,tf32:row,tf32:column,1,0,1,1,tensorop,f32,256,128,16,3,4,2,1,16,8,8,80,1024,3221225472,8796629893120,70.3503,42.6438,125040
1,CUTLASS,gemm,cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_tn_align4,passed,success,universal,16384,16384,16384,tf32:row,tf32:column,tf32:column,1,0,1,1,tensorop,f32,256,128,16,3,4,2,1,16,8,8,80,1024,3221225472,8796629893120,86.5167,34.6754,101676
1,CUTLASS,gemm,cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_tt_align4,passed,success,universal,16384,16384,16384,tf32:row,tf32:row,tf32:column,1,0,1,1,tensorop,f32,256,128,16,3,4,2,1,16,8,8,80,1024,3221225472,8796629893120,68.3621,43.884,128677
"""
        raw_output_fp16_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,alpha,beta,split_k_slices,batch_count,op_class,accum,cta_m,cta_n,cta_k,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_h16816gemm_256x128_32x3_nn_align8,incorrect,success,universal,16384,16384,16384,f16:column,f16:column,f16:column,1,0,1,1,tensorop,f16,256,128,32,3,4,2,1,16,8,16,80,1024,1610612736,8796629893120,34.1575,43.9142,257531
1,CUTLASS,gemm,cutlass_tensorop_h16816gemm_256x128_32x3_nt_align8,incorrect,success,universal,16384,16384,16384,f16:column,f16:row,f16:column,1,0,1,1,tensorop,f16,256,128,32,3,4,2,1,16,8,16,80,1024,1610612736,8796629893120,34.6153,43.3334,254126
1,CUTLASS,gemm,cutlass_tensorop_h16816gemm_256x128_32x3_tn_align8,incorrect,success,universal,16384,16384,16384,f16:row,f16:column,f16:column,1,0,1,1,tensorop,f16,256,128,32,3,4,2,1,16,8,16,80,1024,1610612736,8796629893120,39.0413,38.4209,225316
1,CUTLASS,gemm,cutlass_tensorop_h16816gemm_256x128_32x3_tt_align8,incorrect,success,universal,16384,16384,16384,f16:row,f16:row,f16:column,1,0,1,1,tensorop,f16,256,128,32,3,4,2,1,16,8,16,80,1024,1610612736,8796629893120,31.2994,47.9243,281048
"""
        assert (benchmark._process_raw_result(0, raw_output_fp32))
        assert (benchmark._process_raw_result(1, raw_output_tf32_tc))
        assert (benchmark._process_raw_result(2, raw_output_fp16_tc))

        assert (benchmark.result['fp32_flops'][0] == 18369.7)
        assert (benchmark.result['tf32_tc_flops'][0] == 128677)
        assert (benchmark.result['fp16_tc_flops'][0] == 281048)

        # Negative case - Add invalid raw output.
        assert (benchmark._process_raw_result(3, 'Invalid raw output') is False)

    @decorator.cuda_test
    def test_flops_performance_cuda_sm100(self):
        """Test gemm-flops benchmark SM100 (Blackwell) UMMA kernel output parsing."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--precision tf32_tc bf16_tc fp16_tc int8_tc')
        # Parse args so _args and _result are initialized for _process_raw_result().
        benchmark.add_parser_arguments()
        ret, benchmark._args, _ = benchmark.parse_args()
        assert ret
        from superbench.benchmarks.result import BenchmarkResult
        from superbench.benchmarks import BenchmarkType, ReturnCode
        benchmark._result = BenchmarkResult(benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS)
        benchmark._precision_need_to_run = ['tf32_tc', 'bf16_tc', 'fp16_tc', 'int8_tc']

        # SM100 UMMA 3x kernel naming:
        #   cutlass3x_sm100_tensorop_gemm_{dtype_a}_{dtype_b}_{acc}_{c}_{d}_{tile}_{cluster}_{stages}_{layout}_align{al}_{schedule}
        raw_output_tf32_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_tf32gemm_f32_f32_f32_f32_f32_128x256x32_1x2x1_0_ntn_align4_1sm,passed,success,universal,16384,16384,16384,8.21,1072500
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_tf32gemm_f32_f32_f32_f32_f32_256x128x32_1x2x1_0_ntn_align4_1sm,passed,success,universal,16384,16384,16384,8.45,1043800
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_tf32gemm_f32_f32_f32_f32_f32_128x256x32_1x2x1_0_ntn_align4_stream_k_1sm,passed,success,universal,16384,16384,16384,8.10,1086300
"""
        raw_output_bf16_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_f32_f32_128x256x64_1x2x1_0_ntn_align8_1sm,passed,success,universal,16384,16384,16384,4.35,2024100
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_bf16_bf16_f32_void_f32_128x256x64_1x2x1_0_ntn_align8_1sm,passed,success,universal,16384,16384,16384,4.20,2096300
"""
        raw_output_fp16_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_f16_f16_f16_f16_f16_128x256x64_1x2x1_0_ntn_align8_1sm,passed,success,universal,16384,16384,16384,4.31,2042000
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_f16_f16_f32_f32_f32_128x256x64_1x2x1_0_ntn_align8_1sm,passed,success,universal,16384,16384,16384,4.28,2056200
"""
        raw_output_int8_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_s8_s8_s32_s32_s32_128x256x128_1x2x1_0_ntn_align16_1sm,passed,success,universal,16384,16384,16384,2.20,3998400
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_s8_s8_s32_void_s32_128x256x128_1x2x1_0_ntn_align16_1sm,passed,success,universal,16384,16384,16384,2.15,4091600
"""
        assert (benchmark._process_raw_result(0, raw_output_tf32_tc))
        assert (benchmark._process_raw_result(1, raw_output_bf16_tc))
        assert (benchmark._process_raw_result(2, raw_output_fp16_tc))
        assert (benchmark._process_raw_result(3, raw_output_int8_tc))

        assert (benchmark.result['tf32_tc_flops'][0] == 1086300)
        assert (benchmark.result['bf16_tc_flops'][0] == 2096300)
        assert (benchmark.result['fp16_tc_flops'][0] == 2056200)
        assert (benchmark.result['int8_tc_iops'][0] == 4091600)

    @decorator.cuda_test
    def test_flops_performance_cuda_fp8(self):
        """Test gemm-flops benchmark FP8 (e4m3) kernel output parsing for SM90 and SM100."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--precision fp8_tc')
        # Parse args so _args and _result are initialized for _process_raw_result().
        benchmark.add_parser_arguments()
        ret, benchmark._args, _ = benchmark.parse_args()
        assert ret
        from superbench.benchmarks.result import BenchmarkResult
        from superbench.benchmarks import BenchmarkType, ReturnCode
        benchmark._result = BenchmarkResult(benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS)
        benchmark._precision_need_to_run = ['fp8_tc', 'fp8_tc']

        # SM90 Hopper WGMMA 3x kernels from CUTLASS v3.9.2 (MMA shape before "gemm") and v4.3.5.
        raw_output_sm90_fp8 = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm90_tensorop_s64x256x32gemm_e4m3_e4m3_f32_f32_f32_128x256x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_nosmem,passed,success,universal,16384,16384,16384,3.79,2317200
1,CUTLASS,gemm,cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_f32_f32_128x256x128_2x1x1_0_tnn_align16_warpspecialized_cooperative_fp8_fastaccum_epi_nosmem,passed,success,universal,16384,16384,16384,3.85,2281500
"""
        pattern = benchmark._CudaGemmFlopsBenchmark__kernel_map[9.0]['fp8_tc']
        names = [line.split(',')[3] for line in raw_output_sm90_fp8.splitlines() if line.startswith('1,')]
        assert (names and all(fnmatch.fnmatchcase(name, '*' + pattern + '*') for name in names))
        # SM100 Blackwell UMMA 3x kernel: cutlass3x_sm100_tensorop_gemm_e4m3_{a}_{b}_{acc}_{c}_{d}_{tile}_...
        raw_output_sm100_fp8 = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_e4m3_e4m3_f32_bf16_bf16_128x256x64_1x2x1_0_ntn_align16_1sm,passed,success,universal,16384,16384,16384,2.10,5296000
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_e4m3_e4m3_f32_f32_f32_128x256x64_1x2x1_0_ntn_align16_1sm,passed,success,universal,16384,16384,16384,2.05,5425400
"""
        assert (benchmark._process_raw_result(0, raw_output_sm90_fp8))
        assert (benchmark._process_raw_result(1, raw_output_sm100_fp8))

        assert (benchmark.result['fp8_tc_flops'][0] == 2317200)
        assert (benchmark.result['fp8_tc_flops'][1] == 5425400)

    @decorator.cuda_test
    def test_flops_performance_cuda_nvfp4(self):
        """Test gemm-flops benchmark NVFP4 (block-scaled FP4) kernel selection and output parsing for SM103."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        benchmark = benchmark_class(benchmark_name, parameters='--precision nvfp4_tc')
        benchmark.add_parser_arguments()
        ret, benchmark._args, _ = benchmark.parse_args()
        assert ret
        from superbench.benchmarks.result import BenchmarkResult
        from superbench.benchmarks import BenchmarkType, ReturnCode
        benchmark._result = BenchmarkResult(benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS)
        benchmark._precision_need_to_run = ['nvfp4_tc']

        # SM103 Blackwell Ultra NVFP4 block-scaled kernel:
        # cutlass3x_sm103_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_{acc}_{c}_{d}_{tile}_..._ultra_2sm_*
        # OperationKind is "block_scaled_gemm" (not "gemm").
        raw_output_sm103_nvfp4 = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,block_scaled_gemm,cutlass3x_sm103_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_256x256x768_2x1x1_0_tnt_align32_o_vs16_ultra_2sm_nopf_epi_nosmem,passed,success,universal,16384,16384,16384,0.55,15998700
1,CUTLASS,block_scaled_gemm,cutlass3x_sm103_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_256x256x768_2x1x1_0_tnt_align32_o_vs16_ultra_2sm_tmapf_epi_nosmem,passed,success,universal,16384,16384,16384,0.53,16602100
"""
        # The SM10.3 --kernels pattern must select these real CUTLASS v4.3.5 names (matched like '*pattern*').
        pattern = benchmark._CudaGemmFlopsBenchmark__kernel_map[10.3]['nvfp4_tc']
        names = [line.split(',')[3] for line in raw_output_sm103_nvfp4.splitlines() if line.startswith('1,')]
        assert (names and all(fnmatch.fnmatchcase(name, '*' + pattern + '*') for name in names))

        assert (benchmark._process_raw_result(0, raw_output_sm103_nvfp4))

        assert (benchmark.result['nvfp4_tc_flops'][0] == 16602100)

    def test_flops_performance_cuda_sm107_kernels(self):
        """Test gemm-flops benchmark selects native SM107 (Rubin) kernels and parses their output."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        precisions = ['tf32_tc', 'bf16_tc', 'fp16_tc', 'fp8_tc', 'nvfp4_tc']
        benchmark = benchmark_class(benchmark_name, parameters='--precision ' + ' '.join(precisions))
        kernel_map = benchmark._CudaGemmFlopsBenchmark__kernel_map[10.7]
        assert (sorted(kernel_map) == sorted(precisions))
        # No legacy SM80 (Ampere) selectors on the Rubin performance path.
        assert (all(pattern.startswith('cutlass3x_sm107_') for pattern in kernel_map.values()))

        benchmark.add_parser_arguments()
        ret, benchmark._args, _ = benchmark.parse_args()
        assert ret
        from superbench.benchmarks.result import BenchmarkResult
        benchmark._result = BenchmarkResult(benchmark_name, BenchmarkType.MICRO, ReturnCode.SUCCESS)
        benchmark._precision_need_to_run = precisions

        # Kernel names from CUTLASS v4.8.0 generated for "107a"; FP8 and NVFP4 are native SM107 B-reuse kernels.
        raw_output_tf32_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_tf32gemm_f32_f32_f32_f32_f32_256x256x32_0x0x1_0_ntn_align4_stream_k_2sm,passed,success,universal,16384,16384,16384,8.0,1100000
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_tf32gemm_f32_f32_f32_f32_f32_256x256x32_4x4x1_0_tnn_align4_stream_k_2sm,passed,success,universal,16384,16384,16384,8.8,1000000
"""
        raw_output_bf16_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_gemm_bf16_bf16_f32_bf16_bf16_256x256x64_0x0x1_0_ntn_align8_stream_k_2sm,passed,success,universal,16384,16384,16384,4.4,2000000
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_gemm_bf16_bf16_f32_bf16_bf16_256x256x64_4x4x1_0_tnn_align8_stream_k_2sm,passed,success,universal,16384,16384,16384,4.2,2100000
"""
        raw_output_fp16_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_gemm_f16_f16_f32_f16_f16_256x256x64_0x0x1_0_ntn_align8_stream_k_2sm,passed,success,universal,16384,16384,16384,4.2,2100000
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_gemm_f16_f16_f32_f16_f16_256x256x64_4x4x1_0_tnn_align8_stream_k_2sm,passed,success,universal,16384,16384,16384,4.0,2200000
"""
        raw_output_fp8_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_gemm_f8_f8_f32_f32_f32_256x256x128_0x0x1_0_ntn_align16_no_breuse_2sm_epi_tma,passed,success,universal,16384,16384,16384,2.2,4000000
1,CUTLASS,gemm,cutlass3x_sm107_tensorop_gemm_f8_f8_f32_f32_f32_512x256x128_4x4x1_0_tnn_align16_breuse_2sm_epi_tma,passed,success,universal,16384,16384,16384,2.0,4400000
"""
        raw_output_nvfp4_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,block_scaled_gemm,cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_256x256x256_0x0x1_0_tnn_align32_o_vs16_no_breuse_2sm_epi_tma,passed,success,universal,16384,16384,16384,0.6,14000000
1,CUTLASS,block_scaled_gemm,cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_512x256x256_4x4x1_0_tnn_align32_o_vs16_breuse_2sm_epi_tma,passed,success,universal,16384,16384,16384,0.5,16000000
"""
        raw_outputs = [
            raw_output_tf32_tc, raw_output_bf16_tc, raw_output_fp16_tc, raw_output_fp8_tc, raw_output_nvfp4_tc
        ]
        for idx, raw_output in enumerate(raw_outputs):
            # The --kernels pattern must select the real names (matched like '*pattern*').
            pattern = kernel_map[precisions[idx]]
            names = [line.split(',')[3] for line in raw_output.splitlines() if line.startswith('1,')]
            assert (names and all(fnmatch.fnmatchcase(name, '*' + pattern + '*') for name in names))
            assert (benchmark._process_raw_result(idx, raw_output))

        assert (benchmark.result['tf32_tc_flops'][0] == 1100000)
        assert (benchmark.result['bf16_tc_flops'][0] == 2100000)
        assert (benchmark.result['fp16_tc_flops'][0] == 2200000)
        assert (benchmark.result['fp8_tc_flops'][0] == 4400000)
        assert (benchmark.result['nvfp4_tc_flops'][0] == 16000000)

        # Retargeted SM100-style FP8/NVFP4, 1SM, vs32 and MXFP4 (ue8m0) kernels are not selected.
        unselected = """
fp8_tc,cutlass3x_sm107_tensorop_gemm_f8_f8_f32_f32_f32_256x256x128_0x0x1_0_ntn_align16_stream_k_2sm
fp8_tc,cutlass3x_sm107_tensorop_gemm_f8_f8_f32_f32_f32_256x256x128_0x0x1_0_ntn_align16_breuse_1sm
nvfp4_tc,cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_256x256x256_0x0x1_0_tnn_align32_o_vs16_2sm
nvfp4_tc,cutlass3x_sm107_bstensorop_gemm_ue4m3xe2m1_ue4m3xe2m1_f32_void_f32_256x256x256_0x0x1_0_tnn_align32_o_vs32_no_breuse_2sm_epi_tma
nvfp4_tc,cutlass3x_sm107_bstensorop_gemm_ue8m0xe2m1_ue8m0xe2m1_f32_void_f32_256x256x256_0x0x1_0_tnn_align32_o_vs32_no_breuse_2sm_epi_tma
"""
        for line in unselected.split():
            precision, name = line.split(',')
            assert (not fnmatch.fnmatchcase(name, '*' + kernel_map[precision] + '*'))

    def test_flops_performance_cuda_sm107_commands(self):
        """Test gemm-flops benchmark runs the cutlass_sm107 install on Rubin and skips unsupported precisions."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.CUDA)
        assert (benchmark_class)

        prefix = os.path.join(self._tmp_dir, 'cutlass_sm107')
        with mock.patch.object(dm.device_manager, 'get_device_compute_capability', return_value=10.7):
            with mock.patch.dict(os.environ):
                benchmark = benchmark_class(
                    benchmark_name, parameters='--precision tf32_tc fp8_tc nvfp4_tc int8_tc int4_tc'
                )
                assert (benchmark._preprocess() is True)
                assert (os.environ['LD_LIBRARY_PATH'].split(os.pathsep)[0] == os.path.join(prefix, 'lib'))
            assert (benchmark.return_code == ReturnCode.SUCCESS)
            # SM107 has no INT8/INT4 kernels, so they are skipped.
            assert (benchmark._precision_need_to_run == ['tf32_tc', 'fp8_tc', 'nvfp4_tc'])
            assert (benchmark._args.bin_dir == os.path.join(prefix, 'bin'))
            kernel_map = benchmark._CudaGemmFlopsBenchmark__kernel_map[10.7]
            tf32_cmd, fp8_cmd, nvfp4_cmd = benchmark._commands
            assert (tf32_cmd.startswith(os.path.join(prefix, 'bin', 'cutlass_profiler') + ' '))
            assert ('--operation=gemm ' in tf32_cmd and tf32_cmd.endswith(' --kernels=' + kernel_map['tf32_tc']))
            assert ('--operation=gemm ' in fp8_cmd)
            assert (fp8_cmd.endswith(' --runtime_input_datatype_a=e4m3 --runtime_input_datatype_b=e4m3'))
            assert ('--operation=block_scaled_gemm ' in nvfp4_cmd and 'runtime_input_datatype' not in nvfp4_cmd)

            with mock.patch.dict(os.environ):
                benchmark = benchmark_class(benchmark_name, parameters='--precision int8_tc')
                assert (benchmark._preprocess() is False)
            assert (benchmark.return_code == ReturnCode.NO_SUPPORTED_PRECISION)

            # An explicit --bin_dir takes precedence over the per-architecture install.
            bin_dir = os.path.join(self._tmp_dir, 'bin')
            with mock.patch.dict(os.environ):
                benchmark = benchmark_class(benchmark_name, parameters='--precision fp16_tc --bin_dir ' + bin_dir)
                assert (benchmark._preprocess() is True)
            assert (benchmark._args.bin_dir == bin_dir)
