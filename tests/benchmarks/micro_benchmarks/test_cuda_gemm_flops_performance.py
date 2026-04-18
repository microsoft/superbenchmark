# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gemm-flops benchmark."""

import unittest

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
        cls.createMockFiles(cls, ['bin/cutlass_profiler'])

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

        # SM100 UMMA 3x kernel naming: cutlass3x_sm100_tensorop_gemm_{dtype_a}_{dtype_b}_{acc}_{c}_{d}_{tile}_{cluster}_{stages}_{layout}_align{al}_{schedule}
        raw_output_tf32_tc = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_tf32_tf32_f32_f32_f32_128x256x32_1x2x1_0_ntn_align4_1sm,passed,success,universal,16384,16384,16384,8.21,1072500
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_tf32_tf32_f32_f32_f32_256x128x32_1x2x1_0_ntn_align4_1sm,passed,success,universal,16384,16384,16384,8.45,1043800
1,CUTLASS,gemm,cutlass3x_sm100_tensorop_gemm_tf32_tf32_f32_void_f32_128x256x32_1x2x1_0_ntn_align4_1sm,passed,success,universal,16384,16384,16384,8.10,1086300
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

        # SM90 Hopper WGMMA 3x kernel: cutlass3x_sm90_tensorop_gemm_e4m3_{a}_{b}_{acc}_{c}_{d}_{tile}_...
        raw_output_sm90_fp8 = """
CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,Runtime,GFLOPs
1,CUTLASS,gemm,cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_bf16_bf16_128x128x64_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma,passed,success,universal,16384,16384,16384,3.85,2281500
1,CUTLASS,gemm,cutlass3x_sm90_tensorop_gemm_e4m3_e4m3_f32_f16_f16_128x128x64_2x1x1_0_tnn_align16_warpspecialized_cooperative_epi_tma,passed,success,universal,16384,16384,16384,3.79,2317200
"""
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
