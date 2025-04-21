// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @brief  Cpp file for some functions related to cudnn
 */

#include <cstdlib>
#include <numeric>

#include "cudnn_function.h"

namespace cudnn_test {
/**
 * @brief check cudnn function running status and throw error str
 */
void throw_cudnn_err(cudnnStatus_t result, const char *func, const char *file, int const line) {
    if (result != CUDNN_STATUS_SUCCESS) {
        const char *msg = cudnnGetErrorString(result);
        std::stringstream safe_call_ss;
        safe_call_ss << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << msg;
        throw std::runtime_error(safe_call_ss.str());
    }
}
/**
 * @brief check cudnn function running status and throw error str
 */
void check_cuda(cudaError_t result, const char *func, const char *file, int const line) {
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
 * @brief Cuda context init
 */
void cudnn_handle_init(cudnnHandle_t *cudnn_handle) {
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDevice(0));
    // create streams/handles
    CHECK_CUDNN_ERROR(cudnnCreate(cudnn_handle));
}
/**
 * @brief Cuda context free
 */
void cudnn_handle_free(cudnnHandle_t *cudnn_handle) { CHECK_CUDNN_ERROR(cudnnDestroy(*cudnn_handle)); }
/**
 * @brief Malloc cuda memory and fill in rand value
 * @tparam T
 * @param  input            the pointer of input
 * @param  dims_            the shape of input
 * @param  random_seed      the random seed to generate random data
 */
template <typename T> void rand(T **input, std::vector<int> dims_, int random_seed) {
    throw "unsupported rand data type";
}
template <> void rand(float **input, std::vector<int> dims_, int random_seed) {
    int size = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(float) * size));
    float *host_input;
    CUDA_SAFE_CALL(cudaMallocHost(&host_input, sizeof(float) * size));
    srand(random_seed);
    for (int i = 0; i < size; i++) {
        host_input[i] = (float)std::rand() / (float)(RAND_MAX);
    }
    // copy input data from host to device
    CUDA_SAFE_CALL(cudaMemcpy(*input, host_input, sizeof(float) * size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(host_input));
}
template <> void rand(half **input, std::vector<int> dims_, int random_seed) {
    int size = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(half) * size));
    half *host_input;
    CUDA_SAFE_CALL(cudaMallocHost(&host_input, sizeof(half) * size));
    for (int i = 0; i < size; i++) {
        host_input[i] = __float2half((float)std::rand() / (float)(RAND_MAX));
    }
    CUDA_SAFE_CALL(cudaMemcpy(host_input, *input, sizeof(half) * size, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(host_input));
}
} // namespace cudnn_test
