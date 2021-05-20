// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cublas_benchmark.h"

// Cuda context init
void cuda_init() {
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDevice(0));
    // create streams/handles
    CUBLAS_SAFE_CALL(cublasCreate(&cublas_handle));
}

// Cuda context free
void cuda_free() {
    CUDA_SAFE_CALL(cudaSetDevice(0));
    CUBLAS_SAFE_CALL(cublasDestroy(cublas_handle));
}

template <>
void gemm<float>(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a, const float *b,
                 float *c) {
    float alpha = 1.0f;
    float beta = 1.0f;
    CUBLAS_SAFE_CALL(cublasSgemm(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                 n, k, &alpha, a, (transa ? k : m), b, (transb ? n : k), &beta, c, m));
}

template <>
void gemm<double>(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const double *a, const double *b,
                  double *c) {
    double alpha = 1.0;
    double beta = 1.0;
    CUBLAS_SAFE_CALL(cublasDgemm(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                 n, k, &alpha, a, (transa ? k : m), b, (transb ? n : k), &beta, c, m));
}

template <>
void gemm<cuComplex>(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a,
                     const cuComplex *b, cuComplex *c) {
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);
    CUBLAS_SAFE_CALL(cublasCgemm(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                 n, k, &alpha, a, (transa ? k : m), b, (transb ? n : k), &beta, c, m));
}

template <>
void gemm<cuDoubleComplex>(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuDoubleComplex *a,
                           const cuDoubleComplex *b, cuDoubleComplex *c) {
    cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    cuDoubleComplex beta = make_cuDoubleComplex(1.0, 0.0);
    CUBLAS_SAFE_CALL(cublasZgemm(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m,
                                 n, k, &alpha, a, (transa ? k : m), b, (transb ? n : k), &beta, c, m));
}

void gemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a, const void *b, void *c,
            std::string type, bool use_tensor_core) {
    float alpha = 1.0f;
    float beta = 0.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t matrix_type;
    cublasGemmAlgo_t algo;
    algo = (use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
    if (type.compare("float")) {
        matrix_type = CUDA_R_32F;
        CUBLAS_SAFE_CALL(cublasGemmEx(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N),
                                      (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a, matrix_type,
                                      (transa ? k : m), b, matrix_type, (transb ? n : k), &beta, c, matrix_type, m,
                                      compute_type, algo));
    }
    if (type.compare("half")) {
        matrix_type = CUDA_R_16F;
        CUBLAS_SAFE_CALL(cublasGemmEx(handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N),
                                      (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a, matrix_type,
                                      (transa ? k : m), b, matrix_type, (transb ? n : k), &beta, c, matrix_type, m,
                                      compute_type, algo));
    }
}

void gemmStridedBatchedEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a,
                          const void *b, void *c, std::string type, bool use_tensor_core, int batchCount) {
    float alpha = 1.0f;
    float beta = 1.0f;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t matrix_type;
    cublasGemmAlgo_t algo;
    algo = (use_tensor_core ? CUBLAS_GEMM_DFALT_TENSOR_OP : CUBLAS_GEMM_DFALT);
    if (type.compare("float")) {
        matrix_type = CUDA_R_32F;
        CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
            handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a,
            matrix_type, (transa ? k : m), m * k, b, matrix_type, (transb ? n : k), n * k, &beta, c, matrix_type, m,
            m * n, batchCount, compute_type, algo));
    }
    if (type.compare("half")) {
        matrix_type = CUDA_R_16F;
        CUBLAS_SAFE_CALL(cublasGemmStridedBatchedEx(
            handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a,
            matrix_type, (transa ? k : m), m * k, b, matrix_type, (transb ? n : k), n * k, &beta, c, matrix_type, m,
            m * n, batchCount, compute_type, algo));
    }
}

void sgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a,
                         const float *b, float *c, int batchCount) {
    float alpha = 1.0f;
    float beta = 1.0f;
    CUBLAS_SAFE_CALL(cublasSgemmStridedBatched(
        handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a,
        (transa ? k : m), m * k, b, (transb ? n : k), n * k, &beta, c, m, m * n, batchCount));
}

void cgemm3mStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a,
                           const cuComplex *b, cuComplex *c, int batchCount) {
    cuComplex alpha = make_cuComplex(1.0f, 0.0f);
    cuComplex beta = make_cuComplex(0.0f, 0.0f);
    CUBLAS_SAFE_CALL(cublasCgemm3mStridedBatched(
        handle, (transa ? CUBLAS_OP_T : CUBLAS_OP_N), (transb ? CUBLAS_OP_T : CUBLAS_OP_N), m, n, k, &alpha, a,
        (transa ? k : m), m * k, b, (transb ? n : k), n * k, &beta, c, m, m * n, batchCount));
}
