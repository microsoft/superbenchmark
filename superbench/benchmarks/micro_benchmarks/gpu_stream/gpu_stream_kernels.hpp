// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "gpu_stream_utils.hpp"

template <typename T> struct VectorType;
template <> struct VectorType<double> {
    using type = double2;
};
template <> struct VectorType<float> {
    using type = float4;
};
template <typename T> using VecT = typename VectorType<T>::type;

#ifdef __CUDACC__
namespace stream_kernels {
using stream_config::Kernel;
using stream_config::LaunchConfig;

template <typename T, Kernel OP> __device__ __forceinline__ T Transform(T a, T b) {
    if constexpr (OP == Kernel::kCopy) {
        return a;
    } else if constexpr (OP == Kernel::kScale) {
        return static_cast<T>(stream_config::scalar) * a;
    } else if constexpr (OP == Kernel::kAdd) {
        return a + b;
    } else {
        static_assert(OP == Kernel::kTriad, "Unsupported STREAM operation");
        return b + static_cast<T>(stream_config::scalar) * a;
    }
}

template <typename T, Kernel OP> __device__ __forceinline__ VecT<T> TransformPacket(VecT<T> a, VecT<T> b) {
    VecT<T> result;
    result.x = Transform<T, OP>(a.x, b.x);
    result.y = Transform<T, OP>(a.y, b.y);
    if constexpr (std::is_same<T, float>::value) {
        result.z = Transform<T, OP>(a.z, b.z);
        result.w = Transform<T, OP>(a.w, b.w);
    }
    return result;
}

// Bounds checks and scalar remainders stay out of the complete-tile kernel.
template <typename T, Kernel OP, int BLOCK, int UNROLL, typename Index>
__global__ __launch_bounds__(BLOCK) void BulkKernel(VecT<T> *__restrict__ output, const VecT<T> *__restrict__ a,
                                                   const VecT<T> *__restrict__ b) {
    const Index base = static_cast<Index>(blockIdx.x) * (BLOCK * UNROLL) + threadIdx.x;
    VecT<T> first[UNROLL];
    VecT<T> second[UNROLL];
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        first[u] = a[base + u * BLOCK];
        if constexpr (OP == Kernel::kAdd || OP == Kernel::kTriad) {
            second[u] = b[base + u * BLOCK];
        }
    }
#pragma unroll
    for (int u = 0; u < UNROLL; ++u) {
        if constexpr (OP == Kernel::kCopy) {
            output[base + u * BLOCK] = first[u];
        } else if constexpr (OP == Kernel::kScale) {
            output[base + u * BLOCK] = TransformPacket<T, OP>(first[u], VecT<T>{});
        } else {
            output[base + u * BLOCK] = TransformPacket<T, OP>(first[u], second[u]);
        }
    }
}

template <typename T, Kernel OP>
__global__ void TailKernel(T *__restrict__ output, const T *__restrict__ a, const T *__restrict__ b,
                           uint64_t begin, uint64_t count) {
    const uint64_t i = begin + static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) {
        T right = 0;
        if constexpr (OP == Kernel::kAdd || OP == Kernel::kTriad) {
            right = b[i];
        }
        output[i] = Transform<T, OP>(a[i], right);
    }
}

template <typename T>
__global__ void InitializeData(T *a, T *b, uint64_t count) {
    for (uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
        a[i] = static_cast<T>(i % stream_config::kUInt8Mod);
        b[i] = a[i];
    }
}

template <typename T, Kernel OP, int BLOCK, int UNROLL, typename Index>
cudaError_t LaunchIndexed(T *output, const T *a, const T *b, uint64_t count, cudaStream_t stream, int max_grid_x) {
    static_assert(sizeof(VecT<T>) == 16 && alignof(VecT<T>) == 16, "STREAM packets must be 128-bit aligned");
    static_assert(std::is_unsigned<Index>::value, "STREAM indices must be unsigned");
    static_assert(UNROLL > 0 && BLOCK > 0, "STREAM tiles must be nonempty");
    constexpr uint64_t lanes = sizeof(VecT<T>) / sizeof(T);
    constexpr uint64_t tile_elements = BLOCK * UNROLL * lanes;
    const uint64_t blocks = count / tile_elements;
    const uint64_t tail_begin = blocks * tile_elements;
    if (count == 0 || count / lanes > std::numeric_limits<Index>::max()) {
        return cudaErrorInvalidValue;
    }
    const uint64_t tail_blocks = (count - tail_begin + 255) / 256;
    if (max_grid_x <= 0 || blocks > static_cast<uint64_t>(max_grid_x) ||
        tail_blocks > static_cast<uint64_t>(max_grid_x)) {
        return cudaErrorInvalidConfiguration;
    }
    if (blocks != 0) {
        BulkKernel<T, OP, BLOCK, UNROLL, Index><<<static_cast<unsigned int>(blocks), BLOCK, 0, stream>>>(
            reinterpret_cast<VecT<T> *>(output), reinterpret_cast<const VecT<T> *>(a),
            reinterpret_cast<const VecT<T> *>(b));
        const cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            return error;
        }
    }
    if (tail_begin < count) {
        TailKernel<T, OP><<<static_cast<unsigned int>(tail_blocks), 256, 0, stream>>>(output, a, b, tail_begin, count);
        return cudaGetLastError();
    }
    return cudaSuccess;
}

template <typename T, Kernel OP, int BLOCK, int UNROLL>
cudaError_t LaunchUnroll(T *output, const T *a, const T *b, uint64_t count, cudaStream_t stream, int max_grid_x) {
    if (count / (sizeof(VecT<T>) / sizeof(T)) <= std::numeric_limits<uint32_t>::max()) {
        return LaunchIndexed<T, OP, BLOCK, UNROLL, uint32_t>(output, a, b, count, stream, max_grid_x);
    }
    return LaunchIndexed<T, OP, BLOCK, UNROLL, uint64_t>(output, a, b, count, stream, max_grid_x);
}

template <typename T, Kernel OP, int BLOCK>
cudaError_t LaunchBlock(T *output, const T *a, const T *b, uint64_t count, cudaStream_t stream, int max_grid_x,
                       const LaunchConfig &config) {
    switch (config.unroll) {
    case 1:
        return LaunchUnroll<T, OP, BLOCK, 1>(output, a, b, count, stream, max_grid_x);
    case 2:
        return LaunchUnroll<T, OP, BLOCK, 2>(output, a, b, count, stream, max_grid_x);
    case 3:
        return LaunchUnroll<T, OP, BLOCK, 3>(output, a, b, count, stream, max_grid_x);
    case 4:
        return LaunchUnroll<T, OP, BLOCK, 4>(output, a, b, count, stream, max_grid_x);
    case 6:
        return LaunchUnroll<T, OP, BLOCK, 6>(output, a, b, count, stream, max_grid_x);
    case 8:
        return LaunchUnroll<T, OP, BLOCK, 8>(output, a, b, count, stream, max_grid_x);
    default:
        return cudaErrorInvalidConfiguration;
    }
}

template <typename T, Kernel OP>
cudaError_t Launch(T *output, const T *a, const T *b, uint64_t count, const LaunchConfig &config,
                   cudaStream_t stream, int max_grid_x) {
    switch (config.block_size) {
    case 128:
        return LaunchBlock<T, OP, 128>(output, a, b, count, stream, max_grid_x, config);
    case 256:
        return LaunchBlock<T, OP, 256>(output, a, b, count, stream, max_grid_x, config);
    case 512:
        return LaunchBlock<T, OP, 512>(output, a, b, count, stream, max_grid_x, config);
    case 1024:
        return LaunchBlock<T, OP, 1024>(output, a, b, count, stream, max_grid_x, config);
    default:
        return cudaErrorInvalidConfiguration;
    }
}
} // namespace stream_kernels
#endif
