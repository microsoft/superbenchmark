// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "gpu_stream_utils.hpp"

/**
 * @brief Type trait mapping scalar types to their 128-bit aligned vector types.
 *
 * @details For optimal memory bandwidth, we use 128-bit (16 byte) vector loads/stores:
 * - double -> double2 (2 x 64-bit = 128-bit)
 * - float  -> float4  (4 x 32-bit = 128-bit)
 */
template <typename T> struct VectorType;
template <> struct VectorType<double> { using type = double2; };
template <> struct VectorType<float> { using type = float4; };

template <typename T> using VecT = typename VectorType<T>::type;

// Kernel declarations (visible to all compilers for function pointer usage)
template <typename T> __global__ void CopyKernel(VecT<T> *tgt, const VecT<T> *src);
template <typename T> __global__ void ScaleKernel(VecT<T> *tgt, const VecT<T> *src, const T scalar);
template <typename T> __global__ void AddKernel(VecT<T> *tgt, const VecT<T> *src_a, const VecT<T> *src_b);
template <typename T>
__global__ void TriadKernel(VecT<T> *tgt, const VecT<T> *src_a, const VecT<T> *src_b, const T scalar);

// Implementation section - only compiled by nvcc
#ifdef __CUDACC__

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
template <typename T> inline __device__ void Fetch(T &v, const T *p) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    v = *p;
#else
    if constexpr (std::is_same<T, double2>::value) {
        asm volatile("ld.volatile.global.v2.f64 {%0,%1}, [%2];" : "=d"(v.x), "=d"(v.y) : "l"(p) : "memory");
    } else if constexpr (std::is_same<T, float4>::value) {
        asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(p)
                     : "memory");
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
template <typename T> inline __device__ void Store(T *p, const T &v) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    *p = v;
#else
    if constexpr (std::is_same<T, double2>::value) {
        asm volatile("st.volatile.global.v2.f64 [%0], {%1,%2};" ::"l"(p), "d"(v.x), "d"(v.y) : "memory");
    } else if constexpr (std::is_same<T, float4>::value) {
        asm volatile("st.volatile.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(p), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w)
                     : "memory");
    }
#endif
}

/**
 * @brief Performs COPY, a simple copy operation from source to target. b = a
 *
 * @details This CUDA kernel performs a simple copy operation, copying data from the source array
 * to the target array. This is used to measure transfer rates without any arithmetic operations.
 *
 * @param[out] tgt The target array where data will be copied to (128-bit aligned).
 * @param[in] src The source array from which data will be copied (128-bit aligned).
 */
template <typename T> __global__ void CopyKernel(VecT<T> *tgt, const VecT<T> *src) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    VecT<T> val;
    Fetch(val, src + index);
    Store(tgt + index, val);
}

/**
 * @brief Performs SCALE, a scaling operation on the source data. b = x * a
 *
 * @details This CUDA kernel performs a simple arithmetic operation by scaling the source data
 * with a given scalar value and storing the result in the target array.
 *
 * @param[out] tgt The target array where the scaled data will be stored (128-bit aligned).
 * @param[in] src The source array containing the data to be scaled (128-bit aligned).
 * @param[in] scalar The scalar value used to scale the source data.
 */
template <typename T> __global__ void ScaleKernel(VecT<T> *tgt, const VecT<T> *src, const T scalar) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    VecT<T> val;
    Fetch(val, src + index);
    if constexpr (std::is_same<T, double>::value) {
        val.x *= scalar;
        val.y *= scalar;
    } else if constexpr (std::is_same<T, float>::value) {
        val.x *= scalar;
        val.y *= scalar;
        val.z *= scalar;
        val.w *= scalar;
    }
    Store(tgt + index, val);
}

/**
 * @brief Performs ADD, an addition operation on two source arrays. c = a + b
 *
 * @details This CUDA kernel adds corresponding elements from two source arrays and stores the result
 * in the target array. This operation is used to measure transfer rates with a simple arithmetic addition.
 *
 * @param[out] tgt The target array where the result of the addition will be stored (128-bit aligned).
 * @param[in] src_a The first source array containing the first set of operands (128-bit aligned).
 * @param[in] src_b The second source array containing the second set of operands (128-bit aligned).
 */
template <typename T> __global__ void AddKernel(VecT<T> *tgt, const VecT<T> *src_a, const VecT<T> *src_b) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    VecT<T> val_a;
    VecT<T> val_b;
    Fetch(val_a, src_a + index);
    Fetch(val_b, src_b + index);
    if constexpr (std::is_same<T, double>::value) {
        val_a.x += val_b.x;
        val_a.y += val_b.y;
    } else if constexpr (std::is_same<T, float>::value) {
        val_a.x += val_b.x;
        val_a.y += val_b.y;
        val_a.z += val_b.z;
        val_a.w += val_b.w;
    }
    Store(tgt + index, val_a);
}

/**
 * @brief Performs TRIAD, fused multiply/add operations on source arrays. c = b + x * a
 *
 * @details This CUDA kernel performs a fused multiply/add operation by multiplying elements from
 * the first source array with a scalar value, adding the result to corresponding elements from
 * the second source array, and storing the result in the target array.
 *
 * @param[out] tgt The target array where the result of the fused multiply/add operation will be stored (128-bit
 * aligned).
 * @param[in] src_a The first source array containing the first set of operands to be multiplied by the scalar
 * (128-bit aligned).
 * @param[in] src_b The second source array containing the second set of operands (128-bit aligned).
 * @param[in] scalar The scalar value used in the multiply/add operation.
 */
template <typename T>
__global__ void TriadKernel(VecT<T> *tgt, const VecT<T> *src_a, const VecT<T> *src_b, const T scalar) {
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    VecT<T> val_a;
    VecT<T> val_b;
    Fetch(val_a, src_a + index);
    Fetch(val_b, src_b + index);
    if constexpr (std::is_same<T, double>::value) {
        val_b.x += (val_a.x * scalar);
        val_b.y += (val_a.y * scalar);
    } else if constexpr (std::is_same<T, float>::value) {
        val_b.x += (val_a.x * scalar);
        val_b.y += (val_a.y * scalar);
        val_b.z += (val_a.z * scalar);
        val_b.w += (val_a.w * scalar);
    }
    Store(tgt + index, val_b);
}

#endif // __CUDACC__