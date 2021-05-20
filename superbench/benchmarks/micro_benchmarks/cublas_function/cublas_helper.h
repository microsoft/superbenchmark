// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <sstream>
#include <string>

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
#define CUDA_SAFE_CALL(x) check_cuda((x), #x, __FILE__, __LINE__)

void check_cublas(cublasStatus_t result, char const *const func, const char *const file, int const line) {
    if (result != CUBLAS_STATUS_SUCCESS) {

        std::stringstream safe_call_ss;
        safe_call_ss << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << result;
        // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(safe_call_ss.str());
    }
}
#define CUBLAS_SAFE_CALL(x) check_cublas((x), #x, __FILE__, __LINE__)

cublasHandle_t cublas_handle;

// Wrappers of cublas functions
template <class T>
void gemm(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const T *a, const T *b, T *c);

void gemmEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *A, const void *B, void *C,
            std::string type, bool use_tensor_core);

void gemmStridedBatchedEx(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const void *a,
                          const void *b, void *c, std::string type, bool use_tensor_core, int batchCount);

void cgemm3mStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const cuComplex *a,
                           const cuComplex *b, cuComplex *c, int batchCount);

void sgemmStridedBatched(cublasHandle_t handle, int transa, int transb, int m, int n, int k, const float *a,
                         const float *b, float *c, int batchCount);

// Cuda context init
void cuda_init();
// Cuda context free
void cuda_free();