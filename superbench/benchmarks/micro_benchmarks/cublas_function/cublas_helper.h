// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cublas_helper.h
 * @brief  Header file for some functions related to cublas
 */

#pragma once

#include <sstream>
#include <string>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * @brief check cuda function running status and throw error str
 */
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#define CUDA_SAFE_CALL(x) check_cuda((x), #x, __FILE__, __LINE__)

/**
 * @brief check cublas function running status and throw error str
 */
void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line);
#define CUBLAS_SAFE_CALL(x) check_cublas((x), #x, __FILE__, __LINE__)

/**
 * @brief Cuda context init
 */
void cuda_init(cublasHandle_t *cublas_handle);

/**
 * @brief Cuda context free
 */
void cuda_free(cublasHandle_t *cublas_handle);

/**
 * @brief                   cublas function of gemm, wrapper of cublasSgemm
 * @param  handle           cublas handle
 * @param  transa           whether matrixA transpose
 * @param  transb           whether matrixB transpose
 * @param  m                m of matrix m*n,n*k
 * @param  n                n of matrix m*n,n*k
 * @param  k                k of matrix m*n,n*k
 * @param  a                input matrixA
 * @param  b                input matrixB
 * @param  c                output matrix
 */
void sgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a, const float *b,
           float *c);

/**
 * @brief                   cublas function of gemm, wrapper of cublasCgemm
 * @param  handle           cublas handle
 * @param  transa           whether matrixA transpose
 * @param  transb           whether matrixB transpose
 * @param  m                m of matrix m*n,n*k
 * @param  n                n of matrix m*n,n*k
 * @param  k                k of matrix m*n,n*k
 * @param  a                input matrixA
 * @param  b                input matrixB
 * @param  c                output matrix
 */
void cgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a, const cuComplex *b,
           cuComplex *c);

/**
 * @brief                   cublas function of GemmEx, wrapper of cublasGemmEx
 * @param  handle           cublas handle
 * @param  transa           whether matrixA transpose
 * @param  transb           whether matrixB transpose
 * @param  m                m of matrix m*n,n*k
 * @param  n                n of matrix m*n,n*k
 * @param  k                k of matrix m*n,n*k
 * @param  a                input matrixA
 * @param  b                input matrixB
 * @param  c                output matrix
 * @param  type             matrix type, 'float' or 'half'
 * @param  use_tensor_core  whether use tensor core
 */
void gemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *A, const void *B, void *C,
            std::string type, bool use_tensor_core);

/**
 * @brief                   cublas function of gemmStridedBatchedEx, wrapper of cublasGemmStridedBatchedEx
 * @param  handle           cublas handle
 * @param  transa           whether matrixA transpose
 * @param  transb           whether matrixB transpose
 * @param  m                m of matrix m*n,n*k
 * @param  n                n of matrix m*n,n*k
 * @param  k                k of matrix m*n,n*k
 * @param  a                input matrixA
 * @param  b                input matrixB
 * @param  c                output matrix
 * @param  type             matrix type, 'float' or 'half'
 * @param  use_tensor_core  whether use tensor core
 * @param  batchCount       My Param doc
 */
void gemmStridedBatchedEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a,
                          const void *b, void *c, std::string type, bool use_tensor_core, int batchCount);

/**
 * @brief                   cublas function of gemmStridedBatchedEx, wrapper of cublasGemmStridedBatchedEx
 * @param  handle           cublas handle
 * @param  transa           whether matrixA transpose
 * @param  transb           whether matrixB transpose
 * @param  m                m of matrix m*n,n*k
 * @param  n                n of matrix m*n,n*k
 * @param  k                k of matrix m*n,n*k
 * @param  a                input matrixA
 * @param  b                input matrixB
 * @param  c                output matrix
 * @param  batchCount       the count of batch used to compute
 */
void cgemm3mStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a,
                           const cuComplex *b, cuComplex *c, int batchCount);

/**
 * @brief
 * @brief                   cublas function of sgemmStridedBatched, wrapper of cublasSgemmStridedBatched
 * @param  handle           cublas handle
 * @param  transa           whether matrixA transpose
 * @param  transb           whether matrixB transpose
 * @param  m                m of matrix m*n,n*k
 * @param  n                n of matrix m*n,n*k
 * @param  k                k of matrix m*n,n*k
 * @param  a                input matrixA
 * @param  b                input matrixB
 * @param  c                output matrix
 * @param  batchCount       the count of batch used to compute
 */
void sgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a,
                         const float *b, float *c, int batchCount);
