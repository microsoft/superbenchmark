// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// GPU SM copy benchmark tests dtoh/htod data transfer bandwidth initiated by GPU SM.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

// Argurment index used in argument parsing.
enum class ArgIdx { kGpuId = 1, kCopyDirection, kSize, kNumLoops, kNumArgs };

// Stored arguments for this program.
struct Args {
    // ID of GPU used in this benchmark.
    int gpu_id = 0;

    // Data transfer direction, can be "dtoh" or "htod".
    std::string copy_direction;

    // Data buffer size used.
    uint64_t size = 0;

    // Number of loops in data transfer benchmark.
    uint64_t num_loops = 0;
};

struct Buffers {
    // Original data buffer.
    uint8_t *data_buf = nullptr;

    // Buffer to validate the correctness of data transfer.
    uint8_t *check_buf = nullptr;

    // Data buffer in host memory.
    uint8_t *host_buf = nullptr;

    // Device pointer of the data buffer in host memory.
    uint8_t *host_buf_dev_ptr = nullptr;

    // Data buffer in device memory
    uint8_t *dev_buf = nullptr;
};

// Pring usage of this program.
void PrintUsage() {
    printf("Usage: gpu_sm_copy "
           "<gpu-id> "
           "<copy-direction: dtoh|htod> "
           "<size> "
           "<num_loops>\n");
}

// Prepare data buffers to be used.
int PrepareBuf(const Args &args, Buffers *buffers) {
    cudaError_t cuda_err = cudaSuccess;

    // Generate data to copy
    buffers->data_buf = static_cast<uint8_t *>(malloc(args.size));
    for (int i = 0; i < args.size; i++) {
        buffers->data_buf[i] = static_cast<uint8_t>(i % 256);
    }

    // Reset check buffer
    buffers->check_buf = static_cast<uint8_t *>(malloc(args.size));
    memset(buffers->check_buf, 0, args.size);

    // Allocate host buffer
    buffers->host_buf = static_cast<uint8_t *>(malloc(args.size));
    cuda_err = cudaHostRegister(buffers->host_buf, args.size, cudaHostRegisterMapped);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "PrepareBuf::cudaHostRegister error: %d\n", cuda_err);
        return -1;
    }
    cuda_err = cudaHostGetDevicePointer((void **)(&(buffers->host_buf_dev_ptr)), buffers->host_buf, 0);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "PrepareBuf::cudaHostGetDevicePointer error: %d\n", cuda_err);
        return -1;
    }

    // Allocate device buffer
    cuda_err = cudaMalloc(&(buffers->dev_buf), args.size);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "PrepareBuf::cudaMalloc error: %d\n", cuda_err);
        return -1;
    }

    // Initialize source buffer
    if (args.copy_direction == "dtoh") {
        cuda_err = cudaMemcpy(buffers->dev_buf, buffers->data_buf, args.size, cudaMemcpyDefault);
    } else if (args.copy_direction == "htod") {
        cuda_err = cudaMemcpy(buffers->host_buf, buffers->data_buf, args.size, cudaMemcpyDefault);
    } else {
        fprintf(stderr, "Unrecognized copy direction: %s\n", args.copy_direction.c_str());
        return -1;
    }
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "PrepareBuf::cudaMemcpy error: %d\n", cuda_err);
        return -1;
    }

    return 0;
}

// Validate the result of data transfer.
int CheckBuf(const Args &args, Buffers *buffers) {
    cudaError_t cuda_err = cudaSuccess;

    // Copy result
    if (args.copy_direction == "dtoh") {
        cuda_err = cudaMemcpy(buffers->check_buf, buffers->host_buf, args.size, cudaMemcpyDefault);
    } else if (args.copy_direction == "htod") {
        cuda_err = cudaMemcpy(buffers->check_buf, buffers->dev_buf, args.size, cudaMemcpyDefault);
    }
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CheckBuf::cudaMemcpy error: %d\n", cuda_err);
        return -1;
    }

    // Validate result
    int memcmp_result = memcmp(buffers->data_buf, buffers->check_buf, args.size);
    if (memcmp_result) {
        fprintf(stderr, "Memory check failed\n");
        return -1;
    }

    return 0;
}

// Destroy data buffers
int DestroyBuf(Buffers *buffers) {
    int ret = 0;
    cudaError_t cuda_err = cudaSuccess;

    // Destroy original data buffer and check buffer
    if (buffers->data_buf != nullptr)
        free(buffers->data_buf);
    if (buffers->check_buf != nullptr)
        free(buffers->check_buf);

    // Destroy device buffer
    if (buffers->dev_buf != nullptr) {
        cuda_err = cudaFree(buffers->dev_buf);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "DestroyBuf::cudaFree error: %d\n", cuda_err);
            ret = -1;
        }
    }

    // Destroy host buffer
    if (buffers->host_buf != nullptr) {
        cuda_err = cudaHostUnregister(buffers->host_buf);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "DestroyBuf::cudaHostUnregister error: %d\n", cuda_err);
            ret = -1;
        }
        free(buffers->host_buf);
        buffers->host_buf_dev_ptr = nullptr;
    }

    return ret;
}

// Unroll depth in SM copy kernel
#define NUM_LOOP_UNROLL 2

// Thread block size
#define NUM_THREADS_IN_BLOCK 128

// Fetch a ulong2 from source memory and write to register
// This kernel references the implementation in
// 1) NCCL:
// https://github.com/NVIDIA/nccl/blob/7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/collectives/device/common_kernel.h#L483
// 2) RCCL:
// https://github.com/ROCmSoftwarePlatform/rccl/blob/5c8380ff5b5925cae4bce00b1879a5f930226e8d/src/collectives/device/common_kernel.h#L268
inline __device__ void FetchULong2(ulong2 &v, const ulong2 *p) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    v.x = p->x;
    v.y = p->y;
#else
    asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
#endif
}

// Store a ulong2 from register and write to target memory
// This kernel references the implementation in
// 1) NCCL:
// https://github.com/NVIDIA/nccl/blob/7e515921295adaab72adf56ea71a0fafb0ecb5f3/src/collectives/device/common_kernel.h#L486
// 2) RCCL:
// https://github.com/ROCmSoftwarePlatform/rccl/blob/5c8380ff5b5925cae4bce00b1879a5f930226e8d/src/collectives/device/common_kernel.h#L276
inline __device__ void StoreULong2(ulong2 *p, ulong2 &v) {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HCC__) || defined(__HIPCC__)
    p->x = v.x;
    p->y = v.y;
#else
    asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" ::"l"(p), "l"(v.x), "l"(v.y) : "memory");
#endif
}

// Fetch data from source memory into register first, and then write them to target memory
// Stride set to thread block size to best utilize cache
__global__ void SMCopyKernel(ulong2 *tgt, const ulong2 *src) {
    uint64_t index = blockIdx.x * blockDim.x * NUM_LOOP_UNROLL + threadIdx.x;
    ulong2 val[NUM_LOOP_UNROLL];
#pragma unroll
    for (uint64_t i = 0; i < NUM_LOOP_UNROLL; i++)
        FetchULong2(val[i], src + index + i * blockDim.x);
#pragma unroll
    for (uint64_t i = 0; i < NUM_LOOP_UNROLL; i++)
        StoreULong2(tgt + index + i * blockDim.x, val[i]);
}

// Run SM copy kernel benchmark
int BenchSMCopyKernel(const Args &args, Buffers *buffers) {
    cudaError_t cuda_err = cudaSuccess;
    cudaStream_t stream;
    uint8_t *src_buf = nullptr;
    uint8_t *tgt_buf = nullptr;

    // Determine source buffer and target buff
    if (args.copy_direction == "dtoh") {
        src_buf = buffers->dev_buf;
        tgt_buf = buffers->host_buf_dev_ptr;
    } else {
        src_buf = buffers->host_buf_dev_ptr;
        tgt_buf = buffers->dev_buf;
    }

    // Validate data size
    uint64_t num_elements_in_thread_block = NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK;
    uint64_t num_bytes_in_thread_block = num_elements_in_thread_block * sizeof(ulong2);
    if (args.size % num_bytes_in_thread_block) {
        fprintf(stderr, "Data size should be multiple of %lu\n", num_bytes_in_thread_block);
        return -1;
    }

    // Create stream to launch kernels
    cuda_err = cudaStreamCreate(&stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "BenchSMCopyKernel::cudaStreamCreate error: %d\n", cuda_err);
        return -1;
    }

    // Launch kernels and collect running time
    uint64_t num_thread_blocks = args.size / num_bytes_in_thread_block;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < args.num_loops; i++) {
        SMCopyKernel<<<num_thread_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(reinterpret_cast<ulong2 *>(tgt_buf),
                                                                             reinterpret_cast<ulong2 *>(src_buf));
    }
    cuda_err = cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "BenchSMCopyKernel::cudaStreamSynchronize error: %d\n", cuda_err);
        return -1;
    }

    // Destroy stream
    cuda_err = cudaStreamDestroy(stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "BenchSMCopyKernel::cudaStreamDestroy error: %d\n", cuda_err);
        return -1;
    }

    // Calculate and display bandwidth if no problem
    double time_in_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    printf("Bandwidth (GB/s): %g\n", args.size * args.num_loops / time_in_sec / 1e9);

    return 0;
}

int main(int argc, char **argv) {
    int ret = 0;
    int destroy_buf_ret = 0;
    cudaError_t cuda_err = cudaSuccess;
    Args args;
    Buffers buffers;

    if (argc != static_cast<int>(ArgIdx::kNumArgs)) {
        PrintUsage();
        return -1;
    }

    args.gpu_id = std::stoi(argv[static_cast<int>(ArgIdx::kGpuId)]);
    args.copy_direction = argv[static_cast<int>(ArgIdx::kCopyDirection)];
    args.size = std::stoul(argv[static_cast<int>(ArgIdx::kSize)]);
    args.num_loops = std::stoul(argv[static_cast<int>(ArgIdx::kNumLoops)]);

    // Set device context
    cuda_err = cudaSetDevice(args.gpu_id);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice error: %d\n", cuda_err);
        goto destroy_buf;
    }

    // Prepare data buffers
    ret = PrepareBuf(args, &buffers);
    if (ret != 0) {
        goto destroy_buf;
    }

    // Run benchmark
    ret = BenchSMCopyKernel(args, &buffers);
    if (ret != 0) {
        goto destroy_buf;
    }

    // Validate data
    ret = CheckBuf(args, &buffers);

destroy_buf:
    // Destroy buffers
    destroy_buf_ret = DestroyBuf(&buffers);
    if (ret == 0) {
        ret = destroy_buf_ret;
    }

    return ret;
}
