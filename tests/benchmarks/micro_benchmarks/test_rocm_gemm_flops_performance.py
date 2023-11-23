# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for gemm-flops benchmark."""

import unittest

from tests.helper.testcase import BenchmarkTestCase
from superbench.benchmarks import BenchmarkRegistry, ReturnCode, Platform, BenchmarkType


class RocmGemmFlopsTest(BenchmarkTestCase, unittest.TestCase):
    """Tests for RocmGemmFlops benchmark."""
    @classmethod
    def setUpClass(cls):
        """Hook method for setting up class fixture before running tests in the class."""
        super().setUpClass()
        cls.createMockEnvs(cls)
        cls.createMockFiles(cls, ['bin/rocblas-bench'])

    def test_rocm_flops_performance(self):
        """Test gemm-flops benchmark."""
        benchmark_name = 'gemm-flops'
        (benchmark_class,
         predefine_params) = BenchmarkRegistry._BenchmarkRegistry__select_benchmark(benchmark_name, Platform.ROCM)
        assert (benchmark_class)

        # Negative case - MICROBENCHMARK_UNSUPPORTED_ARCHITECTURE.
        benchmark = benchmark_class(benchmark_name, parameters='--m 7680 --n 8192 --k 8192')

        ret = benchmark._preprocess()
        assert (ret is True)
        assert (benchmark.return_code == ReturnCode.SUCCESS)

        # Check basic information.
        assert (benchmark.name == 'gemm-flops')
        assert (benchmark.type == BenchmarkType.MICRO)
        assert (benchmark._bin_name == 'rocblas-bench')

        # Check parameters specified in BenchmarkContext.
        assert (benchmark._args.m == 7680)
        assert (benchmark._args.n == 8192)
        assert (benchmark._args.k == 8192)

        params = '--transposeA N --transposeB T -m 7680 -n 8192 -k 8192' + \
            ' --alpha 1 --beta 0 --lda 8384 --ldb 8384 --ldc 8384 --ldd 8384 --initialization hpl'
        # Check command list
        expected_command = [
            'rocblas-bench -r f64_r -f gemm ' + params,
            'rocblas-bench -r f32_r -f gemm_ex --compute_type f32_r ' + params,
            'rocblas-bench -r f16_r -f gemm_ex --compute_type f32_r ' + params,
            'rocblas-bench -r bf16_r -f gemm_ex --compute_type f32_r ' + params,
            'rocblas-bench --a_type i8_r --b_type i8_r --c_type i32_r --d_type i32_r -f gemm_ex --compute_type i32_r ' +
            params
        ]
        for i in range(len(expected_command)):
            commnad = benchmark._bin_name + benchmark._commands[i].split(benchmark._bin_name)[1]
            print(benchmark._commands)
            assert (commnad == expected_command[i])

        # Check results and metrics.
        raw_output_FP64 = """
transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,rocblas-Gflops,us
N,T,7680,8192,8192,1,8384,0,8384,8384, 10037.5, 102694
"""
        raw_output_FP32_X = """
transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,ldd,batch_count,rocblas-Gflops,us
N,T,8640,8640,8640,1,8640,0,8640,8640,8640,1, 39441.6, 32705.2
"""
        raw_output_FP16_X = """
transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,ldd,batch_count,rocblas-Gflops,us
N,T,7680,8192,8192,1,8384,0,8384,8384,8384,1, 153728, 6705.3
"""
        raw_output_BF16_X = """
transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,ldd,batch_count,rocblas-Gflops,us
N,T,7680,8192,8192,1,8384,0,8384,8384,8384,1, 81374.3, 12667.3
"""
        raw_output_INT8_X = """
transA,transB,M,N,K,alpha,lda,beta,ldb,ldc,ldd,batch_count,rocblas-Gflops,us
T,N,7680,8192,8192,1,8416,0,8416,8416,8416,1, 162675, 6336.5
"""
        assert (benchmark._process_raw_result(0, raw_output_FP64))
        assert (benchmark._process_raw_result(1, raw_output_FP32_X))
        assert (benchmark._process_raw_result(2, raw_output_FP16_X))
        assert (benchmark._process_raw_result(3, raw_output_BF16_X))
        assert (benchmark._process_raw_result(4, raw_output_INT8_X))

        assert (benchmark.result['fp64_flops'][0] == 10037.5)
        assert (benchmark.result['fp32_xdlops_flops'][0] == 39441.6)
        assert (benchmark.result['fp16_xdlops_flops'][0] == 153728)
        assert (benchmark.result['bf16_xdlops_flops'][0] == 81374.3)
        assert (benchmark.result['int8_xdlops_iops'][0] == 162675)

        # Negative case - Add invalid raw output.
        assert (benchmark._process_raw_result(4, 'Invalid raw output') is False)
