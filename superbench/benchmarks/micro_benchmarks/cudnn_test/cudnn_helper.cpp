// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

// Benchmark of cudnn functions
#include <bits/stdc++.h>

#include "cudnn_benchmark.h"

namespace cudnn_test {

/**
 * @brief Cuda context init
 */
void cuda_init(cudnnHandle_t *cudnn_handle) {
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDevice(0));
    // create streams/handles
    CHECK_CUDNN_ERROR(cudnnCreate(cudnn_handle));
}

/**
 * @brief Cuda context free
 */
void cuda_free(cudnnHandle_t *cudnn_handle) {
    CHECK_CUDNN_ERROR(cudnnDestroy(*cudnn_handle));
    CUDA_SAFE_CALL(cudaSetDevice(0));
}

template <typename T> void rand(T **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    zeros(input, dims_);
}

template <> void rand(float **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());

    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(float) * size_));

    curandGenerateUniform(curand_gen, *input, size_);
}

template <> void rand(double **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());

    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(double) * size_));

    curandGenerateUniformDouble(curand_gen, *input, size_);
}

template <> void rand(half **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(half) * size_));
    half *host_input;
    CUDA_SAFE_CALL(cudaMallocHost(&host_input, sizeof(half) * size_));
    for (int i = 0; i < size_; i++) {
        host_input[i] = __float2half((float)std::rand() / (float)(RAND_MAX));
    }
    CUDA_SAFE_CALL(cudaMemcpy(host_input, *input, sizeof(half) * size_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(host_input));
}
} // namespace cudnn_test