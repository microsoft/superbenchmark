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
enum class ArgIdx { kSrcDev = 1, kDstDev, kWorkingDev, kSize, kNumLoops, kNumArgs };

// Device type.
enum class DevType { kCpu, kGpu };

// Indices of devices involved.
enum class DevIdx { kSrcDevIdx, kDstDevIdx, kWorkingDevIdx, kNumDevices };

// Stored arguments for this program.
struct Args {
    // Type of source/destination/working devices.
    DevType dev_types[static_cast<int>(DevIdx::kNumDevices)] = {DevType::kGpu, DevType::kGpu, DevType::kGpu};

    // GPU IDs for source/destination/working devices.
    int gpu_ids[static_cast<int>(DevIdx::kNumDevices)] = {0, 0, 0};

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

    // Host pointers of the data buffers on source/destination devices.
    uint8_t *host_buf_ptrs[static_cast<int>(DevIdx::kNumDevices) - 1] = {nullptr, nullptr};

    // GPU pointers of the data buffers on source/destination devices.
    uint8_t *gpu_buf_ptrs[static_cast<int>(DevIdx::kNumDevices) - 1] = {nullptr, nullptr};
};

// Pring usage of this program.
void PrintUsage() {
    printf("Usage: gpu_sm_copy "
           "<src-dev: cpu|gpu[0-9]+> "
           "<dst-dev: cpu|gpu[0-9]+> "
           "<working-dev: gpu[0-9]+> "
           "<size> "
           "<num_loops>\n");
}

// Set GPU context according to device index and device types
int SetGpu(const Args &args, int dev_idx) {
    cudaError_t cuda_err = cudaSuccess;
    if (args.dev_types[dev_idx] == DevType::kCpu) {
        // Set to working device
        cuda_err = cudaSetDevice(args.gpu_ids[static_cast<int>(DevIdx::kWorkingDevIdx)]);
    } else {
        // Set to specified device
        cuda_err = cudaSetDevice(args.gpu_ids[dev_idx]);
    }
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "SetGpu::cudaSetDevice error: %d\n", cuda_err);
        return -1;
    }
    return 0;
}

// Prepare data buffers to be used.
int PrepareBuf(const Args &args, Buffers *buffers) {
    cudaError_t cuda_err = cudaSuccess;
    constexpr int uint8_mod = 256;

    // Generate data to copy
    buffers->data_buf = static_cast<uint8_t *>(malloc(args.size));
    for (int i = 0; i < args.size; i++) {
        buffers->data_buf[i] = static_cast<uint8_t>(i % uint8_mod);
    }

    // Reset check buffer
    buffers->check_buf = static_cast<uint8_t *>(malloc(args.size));
    memset(buffers->check_buf, 0, args.size);

    // Only allocate buffers for source/destination devices
    for (int i = 0; i < static_cast<int>(DevIdx::kNumDevices) - 1; i++) {
        // Allocate buffers
        if (args.dev_types[i] == DevType::kCpu) {
            // Set to working device for host memory buffer
            if (SetGpu(args, static_cast<int>(DevIdx::kWorkingDevIdx))) {
                return -1;
            }
            buffers->host_buf_ptrs[i] = static_cast<uint8_t *>(malloc(args.size));
            cuda_err = cudaHostRegister(buffers->host_buf_ptrs[i], args.size, cudaHostRegisterMapped);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "PrepareBuf::cudaHostRegister error: %d\n", cuda_err);
                return -1;
            }
            cuda_err = cudaHostGetDevicePointer((void **)(&(buffers->gpu_buf_ptrs[i])), buffers->host_buf_ptrs[i], 0);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "PrepareBuf::cudaHostGetDevicePointer error: %d\n", cuda_err);
                return -1;
            }
        } else {
            // Set to buffer device for GPU buffer
            if (SetGpu(args, i)) {
                return -1;
            }
            buffers->host_buf_ptrs[i] = nullptr;
            cuda_err = cudaMalloc(&(buffers->gpu_buf_ptrs[i]), args.size);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "PrepareBuf::cudaMalloc error: %d\n", cuda_err);
                return -1;
            }
        }
    }

    // Initialize source buffer
    if (SetGpu(args, static_cast<int>(DevIdx::kSrcDevIdx))) {
        return -1;
    }
    cuda_err = cudaMemcpy(buffers->gpu_buf_ptrs[0], buffers->data_buf, args.size, cudaMemcpyDefault);
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
    if (SetGpu(args, static_cast<int>(DevIdx::kDstDevIdx))) {
        return -1;
    }
    cuda_err = cudaMemcpy(buffers->check_buf, buffers->gpu_buf_ptrs[static_cast<int>(DevIdx::kDstDevIdx)], args.size,
                          cudaMemcpyDefault);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CheckBuf::cudaMemcpy error: %d\n", cuda_err);
        return -1;
    }

    // Validate result
    int memcmp_result = memcmp(buffers->data_buf, buffers->check_buf, args.size);
    if (memcmp_result) {
        fprintf(stderr, "CheckBuf: Memory check failed\n");
        return -1;
    }

    return 0;
}

// Destroy data buffers
int DestroyBuf(const Args &args, Buffers *buffers) {
    int ret = 0;
    cudaError_t cuda_err = cudaSuccess;

    // Destroy original data buffer and check buffer
    if (buffers->data_buf != nullptr)
        free(buffers->data_buf);
    if (buffers->check_buf != nullptr)
        free(buffers->check_buf);

    // Only destroy buffers for first 2 devices
    for (int i = 0; i < 2; i++) {
        // Destroy buffers
        if (args.dev_types[i] == DevType::kCpu) {
            if (buffers->host_buf_ptrs[i] == nullptr) {
                continue;
            }
            // Set to working device for host memory buffer
            if (SetGpu(args, static_cast<int>(DevIdx::kWorkingDevIdx))) {
                return -1;
            }
            cuda_err = cudaHostUnregister(buffers->host_buf_ptrs[i]);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "DestroyBuf::cudaHostUnregister error: %d\n", cuda_err);
                ret = -1;
            }
            free(buffers->host_buf_ptrs[i]);
            buffers->host_buf_ptrs[i] = nullptr;
            buffers->gpu_buf_ptrs[i] = nullptr;
        } else {
            if (buffers->gpu_buf_ptrs[i] == nullptr) {
                continue;
            }
            // Set to buffer device for GPU buffer
            if (SetGpu(args, i)) {
                return -1;
            }
            cuda_err = cudaFree(buffers->gpu_buf_ptrs[i]);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "DestroyBuf::cudaFree error: %d\n", cuda_err);
                ret = -1;
            }
            buffers->gpu_buf_ptrs[i] = nullptr;
        }
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

    // Set to working device
    if (SetGpu(args, static_cast<int>(DevIdx::kWorkingDevIdx))) {
        return -1;
    }

    // Validate data size
    uint64_t num_elements_in_thread_block = NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK;
    uint64_t num_bytes_in_thread_block = num_elements_in_thread_block * sizeof(ulong2);
    if (args.size % num_bytes_in_thread_block) {
        fprintf(stderr, "BenchSMCopyKernel: Data size should be multiple of %lu\n", num_bytes_in_thread_block);
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
        SMCopyKernel<<<num_thread_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
            reinterpret_cast<ulong2 *>(buffers->gpu_buf_ptrs[static_cast<int>(DevIdx::kDstDevIdx)]),
            reinterpret_cast<ulong2 *>(buffers->gpu_buf_ptrs[static_cast<int>(DevIdx::kSrcDevIdx)]));
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

// Enable P2P communication between all GPU pairs involved.
int EnableP2P(const Args &args) {
    cudaError_t cuda_err = cudaSuccess;
    int can_access = 0;
    for (int i = 0; i < static_cast<int>(DevIdx::kNumDevices); i++) {
        for (int j = 0; j < static_cast<int>(DevIdx::kNumDevices); j++) {
            if (args.gpu_ids[i] != args.gpu_ids[j] && args.dev_types[i] == DevType::kGpu &&
                args.dev_types[j] == DevType::kGpu) {
                cuda_err = cudaDeviceCanAccessPeer(&can_access, args.gpu_ids[i], args.gpu_ids[j]);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "EnableP2P::cudaDeviceCanAccessPeer error: %d\n", cuda_err);
                    return -1;
                }
                if (can_access) {
                    if (SetGpu(args, i)) {
                        return -1;
                    }
                    cuda_err = cudaDeviceEnablePeerAccess(args.gpu_ids[j], 0);
                    if (cuda_err != cudaErrorPeerAccessAlreadyEnabled && cuda_err != cudaSuccess) {
                        fprintf(stderr, "EnableP2P::cudaDeviceEnablePeerAccess error: %d\n", cuda_err);
                        return -1;
                    }
                } else {
                    fprintf(stderr, "EnableP2P: P2P communication between GPU %d and GPU %d cannot be enabled\n",
                            args.gpu_ids[i], args.gpu_ids[j]);
                    return -1;
                }
            }
        }
    }
    return 0;
}

// Parse arguments.
int ParseArgs(Args *args, char **argv) {
    const char *cpu_prefix = "cpu";
    const char *gpu_prefix = "gpu";
    constexpr int prefix_len = 3;
    ArgIdx dev_args[static_cast<int>(DevIdx::kNumDevices)] = {ArgIdx::kSrcDev, ArgIdx::kDstDev, ArgIdx::kWorkingDev};

    for (int i = 0; i < static_cast<int>(DevIdx::kNumDevices); i++) {
        std::string dev_str = argv[static_cast<int>(dev_args[i])];
        // Working device can only be GPU
        if (i < 2 && dev_str == cpu_prefix) {
            args->dev_types[i] = DevType::kCpu;
            args->gpu_ids[i] = 0;
        } else if (dev_str.rfind(gpu_prefix, 0) == 0) {
            args->dev_types[i] = DevType::kGpu;
            args->gpu_ids[i] = std::stoi(dev_str.substr(prefix_len));
        } else {
            fprintf(stderr, "ParseArgs: Invalid device %d: %s\n", i, dev_str.c_str());
            return -1;
        }
    }

    args->size = std::stoul(argv[static_cast<int>(ArgIdx::kSize)]);
    args->num_loops = std::stoul(argv[static_cast<int>(ArgIdx::kNumLoops)]);
    return 0;
}

int main(int argc, char **argv) {
    int ret = 0;
    int destroy_buf_ret = 0;
    Args args;
    Buffers buffers;

    if (argc != static_cast<int>(ArgIdx::kNumArgs)) {
        PrintUsage();
        return -1;
    }

    // Parse arguments
    ret = ParseArgs(&args, argv);
    if (ret != 0) {
        goto destroy_buf;
    }

    // Enable P2P access
    ret = EnableP2P(args);
    if (ret != 0) {
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
    destroy_buf_ret = DestroyBuf(args, &buffers);
    if (ret == 0) {
        ret = destroy_buf_ret;
    }

    return ret;
}
