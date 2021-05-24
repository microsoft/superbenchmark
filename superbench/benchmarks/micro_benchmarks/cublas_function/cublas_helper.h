/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cublas_helper.h
 * @brief  Header file for some functions related to cublas
 */

#include <sstream>
#include <string>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// global cublas handle;
cublasHandle_t cublas_handle;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
#define CUDA_SAFE_CALL(x) check_cuda((x), #x, __FILE__, __LINE__)

void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line);
#define CUBLAS_SAFE_CALL(x) check_cublas((x), #x, __FILE__, __LINE__)

// Cuda context init
void cuda_init();

// Cuda context free
void cuda_free();

// Wrappers of cublas functions
void sgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a, const float *b,
           float *c);

void cgemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a, const cuComplex *b,
           cuComplex *c);

void gemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *A, const void *B, void *C,
            std::string type, bool use_tensor_core);

void gemmStridedBatchedEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a,
                          const void *b, void *c, std::string type, bool use_tensor_core, int batchCount);

void cgemm3mStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a,
                           const cuComplex *b, cuComplex *c, int batchCount);
                           
void sgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a,
                         const float *b, float *c, int batchCount);
