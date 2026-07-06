// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "gpu_stream_utils.hpp"

#ifndef NON_HIP
#define NON_HIP (!defined(__HIP_PLATFORM_HCC__) && !defined(__HCC__) && !defined(__HIPCC__))
#endif

/**
 * @brief Type trait mapping scalar types to their 128-bit aligned vector types.
 *
 * @details For optimal memory bandwidth, we use 128-bit (16 byte) vector loads/stores:
 * - double -> double2 (2 x 64-bit = 128-bit)
 * - float  -> float4  (4 x 32-bit = 128-bit)
 */
template <typename T> struct VectorType;
template <> struct VectorType<double> {
    using type = double2;
};
template <> struct VectorType<float> {
    using type = float4;
};

template <typename T> using VecT = typename VectorType<T>::type;

// Number of vector elements each thread processes per unrolled loop iteration.
// This enables the compiler to pipeline multiple outstanding memory requests per thread,
// which is critical for saturating HBM bandwidth on modern GPUs.
constexpr int kUnroll = 4;

// Kernel declarations (visible to all compilers for function pointer usage)
template <typename T>
__global__ void CopyKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src, uint64_t n);
template <typename T>
__global__ void ScaleKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src, const T scalar, uint64_t n);
template <typename T>
__global__ void AddKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src_a,
                          const VecT<T> *__restrict__ src_b, uint64_t n);
template <typename T>
__global__ void TriadKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src_a,
                            const VecT<T> *__restrict__ src_b, const T scalar, uint64_t n);

// Implementation section - only compiled by nvcc
#ifdef __CUDACC__

// Compiler memory barrier: prevents nvcc from reordering memory operations across this point.
// Ensures all loads above are issued before any stores below, forcing the register allocator
// to keep all loaded values live simultaneously (= multiple outstanding memory requests).
#define STREAM_BARRIER() asm volatile("" ::: "memory")

// L2 cache-hinted load: uses createpolicy + ld.global.L2::cache_hint (PTX 7.8+, sm_80+)
// evict_last tells L2 to keep this data as long as possible, improving streaming BW by ~1.2%.
// Falls back to plain load on HIP/ROCm or if compiled for < sm_80.
inline __device__ void StreamLoadD2(double2 &v, const double2 *p) {
#if NON_HIP && (__CUDA_ARCH__ >= 800)
    uint64_t policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm("ld.global.L2::cache_hint.v2.f64 {%0,%1}, [%2], %3;" : "=d"(v.x), "=d"(v.y) : "l"(p), "l"(policy));
#else
    v = *p;
#endif
}

inline __device__ void StreamLoadF4(float4 &v, const float4 *p) {
#if NON_HIP && (__CUDA_ARCH__ >= 800)
    uint64_t policy;
    asm("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(policy));
    asm("ld.global.L2::cache_hint.v4.f32 {%0,%1,%2,%3}, [%4], %5;"
        : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
        : "l"(p), "l"(policy));
#else
    v = *p;
#endif
}

template <typename T> inline __device__ void StreamLoad(VecT<T> &v, const VecT<T> *p) {
    if constexpr (std::is_same<T, double>::value) {
        StreamLoadD2(v, p);
    } else if constexpr (std::is_same<T, float>::value) {
        StreamLoadF4(v, p);
    }
}

/**
 * @brief Performs COPY using a grid-stride loop with manual 4x unrolling. b = a
 *
 * @details Uses L2::evict_last cache hint on loads for ~1% better streaming BW.
 * Uses a compiler barrier between loads and stores to prevent nvcc from serializing.
 */
template <typename T>
__global__ __launch_bounds__(1024, 1) void CopyKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src,
                                                      uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t stride2 = stride * 2;
    uint64_t stride3 = stride * 3;
    uint64_t chunk = stride * 4;
    uint64_t bulk_end = (n / chunk) * chunk;

    for (uint64_t base = tid; base < bulk_end; base += chunk) {
        VecT<T> v0, v1, v2, v3;
        StreamLoad<T>(v0, src + base);
        StreamLoad<T>(v1, src + base + stride);
        StreamLoad<T>(v2, src + base + stride2);
        StreamLoad<T>(v3, src + base + stride3);
        STREAM_BARRIER();
        tgt[base] = v0;
        tgt[base + stride] = v1;
        tgt[base + stride2] = v2;
        tgt[base + stride3] = v3;
    }

    for (uint64_t idx = bulk_end + tid; idx < n; idx += stride) {
        tgt[idx] = src[idx];
    }
}

/**
 * @brief Performs SCALE with manual 4x unrolling. b = x * a
 */
template <typename T>
__global__ __launch_bounds__(1024, 1) void ScaleKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src,
                                                       const T scalar, uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t stride2 = stride * 2;
    uint64_t stride3 = stride * 3;
    uint64_t chunk = stride * 4;
    uint64_t bulk_end = (n / chunk) * chunk;

    for (uint64_t base = tid; base < bulk_end; base += chunk) {
        VecT<T> v0, v1, v2, v3;
        StreamLoad<T>(v0, src + base);
        StreamLoad<T>(v1, src + base + stride);
        StreamLoad<T>(v2, src + base + stride2);
        StreamLoad<T>(v3, src + base + stride3);
        STREAM_BARRIER();
        if constexpr (std::is_same<T, double>::value) {
            v0.x *= scalar;
            v0.y *= scalar;
            v1.x *= scalar;
            v1.y *= scalar;
            v2.x *= scalar;
            v2.y *= scalar;
            v3.x *= scalar;
            v3.y *= scalar;
        } else if constexpr (std::is_same<T, float>::value) {
            v0.x *= scalar;
            v0.y *= scalar;
            v0.z *= scalar;
            v0.w *= scalar;
            v1.x *= scalar;
            v1.y *= scalar;
            v1.z *= scalar;
            v1.w *= scalar;
            v2.x *= scalar;
            v2.y *= scalar;
            v2.z *= scalar;
            v2.w *= scalar;
            v3.x *= scalar;
            v3.y *= scalar;
            v3.z *= scalar;
            v3.w *= scalar;
        }
        tgt[base] = v0;
        tgt[base + stride] = v1;
        tgt[base + stride2] = v2;
        tgt[base + stride3] = v3;
    }

    for (uint64_t idx = bulk_end + tid; idx < n; idx += stride) {
        VecT<T> v = src[idx];
        if constexpr (std::is_same<T, double>::value) {
            v.x *= scalar;
            v.y *= scalar;
        } else if constexpr (std::is_same<T, float>::value) {
            v.x *= scalar;
            v.y *= scalar;
            v.z *= scalar;
            v.w *= scalar;
        }
        tgt[idx] = v;
    }
}

/**
 * @brief Performs ADD with manual 4x unrolling. c = a + b
 */
template <typename T>
__global__ __launch_bounds__(1024, 1) void AddKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src_a,
                                                     const VecT<T> *__restrict__ src_b, uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t stride2 = stride * 2;
    uint64_t stride3 = stride * 3;
    uint64_t chunk = stride * 4;
    uint64_t bulk_end = (n / chunk) * chunk;

    for (uint64_t base = tid; base < bulk_end; base += chunk) {
        VecT<T> a0, a1, a2, a3, b0, b1, b2, b3;
        StreamLoad<T>(a0, src_a + base);
        StreamLoad<T>(a1, src_a + base + stride);
        StreamLoad<T>(a2, src_a + base + stride2);
        StreamLoad<T>(a3, src_a + base + stride3);
        StreamLoad<T>(b0, src_b + base);
        StreamLoad<T>(b1, src_b + base + stride);
        StreamLoad<T>(b2, src_b + base + stride2);
        StreamLoad<T>(b3, src_b + base + stride3);
        STREAM_BARRIER();
        if constexpr (std::is_same<T, double>::value) {
            a0.x += b0.x;
            a0.y += b0.y;
            a1.x += b1.x;
            a1.y += b1.y;
            a2.x += b2.x;
            a2.y += b2.y;
            a3.x += b3.x;
            a3.y += b3.y;
        } else if constexpr (std::is_same<T, float>::value) {
            a0.x += b0.x;
            a0.y += b0.y;
            a0.z += b0.z;
            a0.w += b0.w;
            a1.x += b1.x;
            a1.y += b1.y;
            a1.z += b1.z;
            a1.w += b1.w;
            a2.x += b2.x;
            a2.y += b2.y;
            a2.z += b2.z;
            a2.w += b2.w;
            a3.x += b3.x;
            a3.y += b3.y;
            a3.z += b3.z;
            a3.w += b3.w;
        }
        tgt[base] = a0;
        tgt[base + stride] = a1;
        tgt[base + stride2] = a2;
        tgt[base + stride3] = a3;
    }

    for (uint64_t idx = bulk_end + tid; idx < n; idx += stride) {
        VecT<T> a = src_a[idx];
        VecT<T> b = src_b[idx];
        if constexpr (std::is_same<T, double>::value) {
            a.x += b.x;
            a.y += b.y;
        } else if constexpr (std::is_same<T, float>::value) {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            a.w += b.w;
        }
        tgt[idx] = a;
    }
}

/**
 * @brief Performs TRIAD with manual 4x unrolling. c = b + x * a
 */
template <typename T>
__global__ __launch_bounds__(1024, 1) void TriadKernel(VecT<T> *__restrict__ tgt, const VecT<T> *__restrict__ src_a,
                                                       const VecT<T> *__restrict__ src_b, const T scalar, uint64_t n) {
    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
    uint64_t stride2 = stride * 2;
    uint64_t stride3 = stride * 3;
    uint64_t chunk = stride * 4;
    uint64_t bulk_end = (n / chunk) * chunk;

    for (uint64_t base = tid; base < bulk_end; base += chunk) {
        VecT<T> a0, a1, a2, a3, b0, b1, b2, b3;
        StreamLoad<T>(a0, src_a + base);
        StreamLoad<T>(a1, src_a + base + stride);
        StreamLoad<T>(a2, src_a + base + stride2);
        StreamLoad<T>(a3, src_a + base + stride3);
        StreamLoad<T>(b0, src_b + base);
        StreamLoad<T>(b1, src_b + base + stride);
        StreamLoad<T>(b2, src_b + base + stride2);
        StreamLoad<T>(b3, src_b + base + stride3);
        STREAM_BARRIER();
        if constexpr (std::is_same<T, double>::value) {
            b0.x += a0.x * scalar;
            b0.y += a0.y * scalar;
            b1.x += a1.x * scalar;
            b1.y += a1.y * scalar;
            b2.x += a2.x * scalar;
            b2.y += a2.y * scalar;
            b3.x += a3.x * scalar;
            b3.y += a3.y * scalar;
        } else if constexpr (std::is_same<T, float>::value) {
            b0.x += a0.x * scalar;
            b0.y += a0.y * scalar;
            b0.z += a0.z * scalar;
            b0.w += a0.w * scalar;
            b1.x += a1.x * scalar;
            b1.y += a1.y * scalar;
            b1.z += a1.z * scalar;
            b1.w += a1.w * scalar;
            b2.x += a2.x * scalar;
            b2.y += a2.y * scalar;
            b2.z += a2.z * scalar;
            b2.w += a2.w * scalar;
            b3.x += a3.x * scalar;
            b3.y += a3.y * scalar;
            b3.z += a3.z * scalar;
            b3.w += a3.w * scalar;
        }
        tgt[base] = b0;
        tgt[base + stride] = b1;
        tgt[base + stride2] = b2;
        tgt[base + stride3] = b3;
    }

    for (uint64_t idx = bulk_end + tid; idx < n; idx += stride) {
        VecT<T> a = src_a[idx];
        VecT<T> b = src_b[idx];
        if constexpr (std::is_same<T, double>::value) {
            b.x += a.x * scalar;
            b.y += a.y * scalar;
        } else if constexpr (std::is_same<T, float>::value) {
            b.x += a.x * scalar;
            b.y += a.y * scalar;
            b.z += a.z * scalar;
            b.w += a.w * scalar;
        }
        tgt[idx] = b;
    }
}

#endif // __CUDACC__