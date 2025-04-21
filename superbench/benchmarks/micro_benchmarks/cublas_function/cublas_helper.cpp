// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cublas_helper.cpp
 * @brief  Cpp file for some functions related to cublas
 */

#include "cublas_benchmark.h"

/**
 * @brief check cuda function running status and throw error str
 */
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        const char *msg = cudaGetErrorString(result);
        std::stringstream safe_call_ss;
        safe_call_ss << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << msg;
        // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(safe_call_ss.str());
    }
}

/**
 * @brief check cublas function running status and throw error str
 */
void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {

        std::stringstream safe_call_ss;
        safe_call_ss << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << result;
        // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(safe_call_ss.str());
    }
}

/**
 * @brief Cuda context init
 */
void cuda_init(cublasHandle_t *cublas_handle) {
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDevice(0));
    // create streams/handles
    CUBLAS_SAFE_CALL(cublasCreate(cublas_handle));
}

/**
 * @brief Cuda context free
 */
void cuda_free(cublasHandle_t *cublas_handle) {
    CUBLAS_SAFE_CALL(cublasDestroy(*cublas_handle));
    CUDA_SAFE_CALL(cudaSetDevice(0));
}

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
           float *c) {
    float alpha = 1.0f;
    float beta = 1.0f;
    CUBLAS_SAFE_CALL(cublasSgemm(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                 n, k, &alpha, a, (transa ? k : m), b, (transb ? n : k), &beta, c, m));
}

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
           cuComplex *c) {
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);
    CUBLAS_SAFE_CALL(cublasCgemm(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                 n, k, &alpha, a, (transa ? k : m), b, (transb ? n : k), &beta, c, m));
}

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
void gemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a, const void *b, void *c,
            std::string type, bool use_tensor_core) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t matrix_type;
    cublasGemmAlgo_t algo;
    algo = (use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
    if (type.compare("float") == 0) {
        matrix_type = CUDA_R_32F;
    } else {
        if (type.compare("half") == 0) {
            matrix_type = CUDA_R_16F;
        } else {
            throw "invalid datatype";
        }
    }
    CUBLAS_SAFE_CALL(cublasGemmEx(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                  n, k, &alpha, a, matrix_type, (transa ? k : m), b, matrix_type, (transb ? n : k),
                                  &beta, c, matrix_type, m, compute_type, algo));
}

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
                          const void *b, void *c, std::string type, bool use_tensor_core, int batchCount) {
    float alpha = 1.0f;
    float beta = 1.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t matrix_type;
    cublasGemmAlgo_t algo;
    algo = (use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
    if (type.compare("float") == 0) {
        matrix_type = CUDA_R_32F;
    } else {
        if (type.compare("half") == 0) {
            matrix_type = CUDA_R_16F;
        } else {
            throw "invalid datatype";
        }
    }
    CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N),
                                                (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a, matrix_type,
                                                (transa ? k : m), static_cast<long long>(m) * k, b, matrix_type,
                                                (transb ? n : k), static_cast<long long>(n) * k, &beta, c, matrix_type,
                                                m, static_cast<long long>(m) * n, batchCount, compute_type, algo));
}

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
void sgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a,
                         const float *b, float *c, int batchCount) {
    float alpha = 1.0f;
    float beta = 1.0f;
    CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
        handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a,
        (transa ? k : m), static_cast<long long>(m) * k, b, (transb ? n : k), static_cast<long long>(n) * k, &beta, c,
        m, static_cast<long long>(m) * n, batchCount));
}

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
void cgemm3mStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a,
                           const cuComplex *b, cuComplex *c, int batchCount) {
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);
    CUBLAS_SAFE_CALL(cublasCgemm3mStridedBatched(
        handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a,
        (transa ? k : m), static_cast<long long>(m) * k, b, (transb ? n : k), static_cast<long long>(n) * k, &beta, c,
        m, static_cast<long long>(m) * n, batchCount));
}
