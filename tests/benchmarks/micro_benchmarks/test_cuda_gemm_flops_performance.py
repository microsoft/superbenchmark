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
