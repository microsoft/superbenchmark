// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_stream_kernels.hpp"

/**
 * @brief Fetches a value from source memory and writes it to a register.
 *
 * @details This inline device function fetches a value from the specified source memory
 * location and writes it to the provided register. The implementation references the following:
 * 1) NCCL:
 * https://github.com/NVIDIA/nccl/blob/7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/collectives/device/common_kernel.h#L483
 * 2) RCCL:
 * https://github.com/ROCmSoftwarePlatform/rccl/blob/5c8380ff5b5925cae4bce00b1879a5f930226e8d/src/collectives/device/common_kernel.h#L268
 *
 * @tparam T The type of the value to fetch.
 * @param[out] v The register to write the fetched value to.
 * @param[in] p The source memory location to fetch the value from.
 */
template <typename T>
inline __device__ void Fetch(T &v, const T *p) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    v = *p;
#else
    if constexpr (std::is_same<T, float>::value) {
        asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
    } else if constexpr (std::is_same<T, double>::value) {
        asm volatile("ld.volatile.global.f64 %0, [%1];" : "=d"(v) : "l"(p) : "memory");
    }
#endif
}

/**
 * @brief Stores a value from register and writes it to target memory.
 *
 * @details This inline device function stores a value from the provided register
 * and writes it to the specified target memory location. The implementation references the following:
 * 1) NCCL:
 * https://github.com/NVIDIA/nccl/blob/7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/collectives/device/common_kernel.h#L486
 * 2) RCCL:
 * https://github.com/ROCmSoftwarePlatform/rccl/blob/5c8380ff5b5925cae4bce00b1879a5f930226e8d/src/collectives/device/common_kernel.h#L276
 *
 * @tparam T The type of the value to store.
 * @param[out] p The target memory location to write the value to.
 * @param[in] v The register containing the value to be stored.
 */
template <typename T>
inline __device__ void Store(T *p, const T &v) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    *p = v;
#else
    if constexpr (std::is_same<T, float>::value) {
        asm volatile("st.volatile.global.f32 [%0], %1;" :: "l"(p), "f"(v) : "memory");
    } else if constexpr (std::is_same<T, double>::value) {
        asm volatile("st.volatile.global.f64 [%0], %1;" :: "l"(p), "d"(v) : "memory");
    }
#endif
}

/**
 * @brief Performs COPY, a simple copy operation from source to target. b = a
 *
 * @details This CUDA kernel performs a simple copy operation, copying data from the source array
 * to the target array. This is used to measure transfer rates without any arithmetic operations.
 *
 * @param[out] tgt The target array where data will be copied to.
 * @param[in] src The source array from which data will be copied.
 */
 __global__ void CopyKernel(double *tgt, const double *src) {
    uint64_t index = blockIdx.x * blockDim.x * kNumLoopUnrollAlias + threadIdx.x;
    double val[kNumLoopUnrollAlias];
#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++)
        Fetch(val[i], src + index + i * blockDim.x);
#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++)
        Store(tgt + index + i * blockDim.x, val[i]);
}

/**
 * @brief Performs SCALE, a scaling operation on the source data. b = x * a
 *
 * @details This CUDA kernel performs a simple arithmetic operation by scaling the source data
 * with a given scalar value and storing the result in the target array.
 *
 * @param[out] tgt The target array where the scaled data will be stored.
 * @param[in] src The source array containing the data to be scaled.
 * @param[in] scalar The scalar value used to scale the source data.
 */
 __global__ void ScaleKernel(double *tgt, const double *src, const long scalar) {
    uint64_t index = blockIdx.x * blockDim.x * kNumLoopUnrollAlias + threadIdx.x;
    double val[kNumLoopUnrollAlias];
#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++)
        Fetch(val[i], src + index + i * blockDim.x);
#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++) {
        val[i] *= scalar;
        Store(tgt + index + i * blockDim.x, val[i]);
    }
}

/**
 * @brief Performs ADD, an addition operation on two source arrays. c = a + b
 *
 * @details This CUDA kernel adds corresponding elements from two source arrays and stores the result
 * in the target array. This operation is used to measure transfer rates with a simple arithmetic addition.
 *
 * @param[out] tgt The target array where the result of the addition will be stored.
 * @param[in] src_a The first source array containing the first set of operands.
 * @param[in] src_b The second source array containing the second set of operands.
 */
 __global__ void AddKernel(double *tgt, const double *src_a, const double *src_b) {
    uint64_t index = blockIdx.x * blockDim.x * kNumLoopUnrollAlias + threadIdx.x;
    double val_a[kNumLoopUnrollAlias];
    double val_b[kNumLoopUnrollAlias];

#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++) {
        Fetch(val_a[i], src_a + index + i * blockDim.x);
        Fetch(val_b[i], src_b + index + i * blockDim.x);
    }
#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++) {
        val_a[i] += val_b[i];
        Store(tgt + index + i * blockDim.x, val_a[i]);
    }
}

/**
 * @brief Performs TRIAD, fused multiply/add operations on source arrays. a = b + x * c
 *
 * @details This CUDA kernel performs a fused multiply/add operation by multiplying elements from
 * the second source array with a scalar value, adding the result to corresponding elements from
 * the first source array, and storing the result in the target array.
 *
 * @param[out] tgt The target array where the result of the fused multiply/add operation will be stored.
 * @param[in] src_a The first source array containing the first set of operands.
 * @param[in] src_b The second source array containing the second set of operands to be multiplied by the scalar.
 * @param[in] scalar The scalar value used in the multiply/add operation.
 */
__global__ void TriadKernel(double *tgt, const double *src_a, const double *src_b, const long scalar) {
    uint64_t index = blockIdx.x * blockDim.x * kNumLoopUnrollAlias + threadIdx.x;
    double val_a[kNumLoopUnrollAlias];
    double val_b[kNumLoopUnrollAlias];

#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++) {
        Fetch(val_a[i], src_a + index + i * blockDim.x);
        Fetch(val_b[i], src_b + index + i * blockDim.x);
    }
#pragma unroll
    for (uint64_t i = 0; i < kNumLoopUnrollAlias; i++) {
        val_b[i] += (val_a[i] * scalar);
        Store(tgt + index + i * blockDim.x, val_b[i]);
    }
}