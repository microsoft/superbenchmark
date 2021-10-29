// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// GPU copy benchmark tests dtoh/htod/dtod data transfer bandwidth by GPU SM/DMA.

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <getopt.h>
#include <numa.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Arguments for each benchmark run.
struct BenchArgs {
    // Whether source device is GPU.
    bool is_src_dev_gpu = false;

    // Whether destination device is GPU.
    bool is_dst_dev_gpu = false;

    // GPU IDs for source device (if applicable).
    int src_gpu_id = 0;

    // GPU IDs for destination device (if applicable).
    int dst_gpu_id = 0;

    // GPU IDs for worker device.
    int worker_gpu_id = 0;

    // Uses SM copy, otherwise DMA copy.
    bool is_sm_copy = false;

    // NUMA node under which the benchmark is done.
    uint64_t numa_id = 0;

    // Data buffer size used.
    uint64_t size = 0;

    // Number of loops to run.
    uint64_t num_loops = 0;
};

struct Buffers {
    // Original data buffer.
    uint8_t *data_buf = nullptr;

    // Buffer to validate the correctness of data transfer.
    uint8_t *check_buf = nullptr;

    // Host pointer of the data buffer on source device.
    uint8_t *src_dev_host_buf_ptr = nullptr;

    // GPU pointer of the data buffer on source devices.
    uint8_t *src_dev_gpu_buf_ptr = nullptr;

    // Host pointer of the data buffer on destination device.
    uint8_t *dst_dev_host_buf_ptr = nullptr;

    // GPU pointer of the data buffer on destination devices.
    uint8_t *dst_dev_gpu_buf_ptr = nullptr;
};

// Options accepted by this program.
struct Opts {
    // Data buffer size for copy benchmark.
    uint64_t size;

    // Data buffer size for copy benchmark.
    uint64_t num_loops;

    // Whether GPU SM copy needs to be evaluated.
    bool sm_copy_enabled = false;

    // Whether GPU DMA copy needs to be evaluated.
    bool dma_copy_enabled = false;

    // Whether host-to-device transfer needs to be evaluated.
    bool htod_enabled = false;

    // Whether device-to-host transfer needs to be evaluated.
    bool dtoh_enabled = false;

    // Whether device-to-device transfer needs to be evaluated.
    bool dtod_enabled = false;
};

// Pring usage of this program.
void PrintUsage() {
    printf("Usage: gpu_copy "
           "--size <size> "
           "--num_loops <num_loops> "
           "[--sm_copy] "
           "[--dma_copy] "
           "[--htod] "
           "[--dtoh] "
           "[--dtod]\n");
}

// Parse options of this program.
int ParseOpts(int argc, char **argv, Opts *opts) {
    enum class OptIdx { kSize, kNumIters, kEnableSmCopy, kEnableDmaCopy, kEnableHToD, kEnableDToH, kEnableDToD };
    const struct option options[] = {{"size", required_argument, nullptr, static_cast<int>(OptIdx::kSize)},
                                     {"num_loops", required_argument, nullptr, static_cast<int>(OptIdx::kNumIters)},
                                     {"sm_copy", no_argument, nullptr, static_cast<int>(OptIdx::kEnableSmCopy)},
                                     {"dma_copy", no_argument, nullptr, static_cast<int>(OptIdx::kEnableDmaCopy)},
                                     {"htod", no_argument, nullptr, static_cast<int>(OptIdx::kEnableHToD)},
                                     {"dtoh", no_argument, nullptr, static_cast<int>(OptIdx::kEnableDToH)},
                                     {"dtod", no_argument, nullptr, static_cast<int>(OptIdx::kEnableDToD)}};
    int getopt_ret = 0;
    int opt_idx = 0;
    bool size_specified = false;
    bool num_loops_specified = false;
    bool parse_err = false;
    while (true) {
        getopt_ret = getopt_long(argc, argv, "", options, &opt_idx);
        if (getopt_ret == -1) {
            if (!size_specified || !num_loops_specified) {
                parse_err = true;
            }
            break;
        } else if (getopt_ret == '?') {
            parse_err = true;
            break;
        }
        switch (opt_idx) {
        case static_cast<int>(OptIdx::kSize):
            if (1 != sscanf(optarg, "%lu", &(opts->size))) {
                fprintf(stderr, "Invalid size: %s\n", optarg);
                parse_err = true;
            } else {
                size_specified = true;
            }
            break;
        case static_cast<int>(OptIdx::kNumIters):
            if (1 != sscanf(optarg, "%lu", &(opts->num_loops))) {
                fprintf(stderr, "Invalid num_loops: %s\n", optarg);
                parse_err = true;
            } else {
                num_loops_specified = true;
            }
            break;
        case static_cast<int>(OptIdx::kEnableSmCopy):
            opts->sm_copy_enabled = true;
            break;
        case static_cast<int>(OptIdx::kEnableDmaCopy):
            opts->dma_copy_enabled = true;
            break;
        case static_cast<int>(OptIdx::kEnableHToD):
            opts->htod_enabled = true;
            break;
        case static_cast<int>(OptIdx::kEnableDToH):
            opts->dtoh_enabled = true;
            break;
        case static_cast<int>(OptIdx::kEnableDToD):
            opts->dtod_enabled = true;
            break;
        default:
            parse_err = true;
        }
        if (parse_err) {
            break;
        }
    }
    if (parse_err) {
        PrintUsage();
        return -1;
    }
    return 0;
}

// Get nubmer of GPUs available.
int GetGpuCount(int *gpu_count) {
    cudaError_t cuda_err = cudaGetDeviceCount(gpu_count);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "GetGpuCount::cudaGetDeviceCount error: %d\n", cuda_err);
        return -1;
    }
    return 0;
}

// Set GPU context according to device ID.
int SetGpu(int gpu_id) {
    cudaError_t cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "SetGpu::cudaSetDevice %d error: %d\n", gpu_id, cuda_err);
        return -1;
    }
    return 0;
}

// Prepare data buffers to be used.
int PrepareBuf(const BenchArgs &args, Buffers *buffers) {
    cudaError_t cuda_err = cudaSuccess;
    constexpr int uint8_mod = 256;

    // Generate data to copy
    buffers->data_buf = static_cast<uint8_t *>(numa_alloc_onnode(args.size, args.numa_id));
    for (int i = 0; i < args.size; i++) {
        buffers->data_buf[i] = static_cast<uint8_t>(i % uint8_mod);
    }

    // Reset check buffer
    buffers->check_buf = static_cast<uint8_t *>(numa_alloc_onnode(args.size, args.numa_id));
    memset(buffers->check_buf, 0, args.size);

    // Allocate buffers for src/dst devices
    constexpr int num_devices = 2;
    bool is_dev_gpu[num_devices] = {args.is_src_dev_gpu, args.is_dst_dev_gpu};
    int dev_ids[num_devices] = {args.src_gpu_id, args.dst_gpu_id};
    uint8_t **host_buf_ptrs[num_devices] = {&(buffers->src_dev_host_buf_ptr), &(buffers->dst_dev_host_buf_ptr)};
    uint8_t **gpu_buf_ptrs[num_devices] = {&(buffers->src_dev_gpu_buf_ptr), &(buffers->dst_dev_gpu_buf_ptr)};
    for (int i = 0; i < num_devices; i++) {
        // Allocate buffers
        if (is_dev_gpu[i]) {
            // Set to buffer device for GPU buffer
            if (SetGpu(dev_ids[i])) {
                return -1;
            }
            *(host_buf_ptrs[i]) = nullptr;
            cuda_err = cudaMalloc(gpu_buf_ptrs[i], args.size);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "PrepareBuf::cudaMalloc error: %d\n", cuda_err);
                return -1;
            }
        } else {
            // Set to worker device for host memory buffer
            if (SetGpu(args.worker_gpu_id)) {
                return -1;
            }
            *(host_buf_ptrs[i]) = static_cast<uint8_t *>(numa_alloc_onnode(args.size, args.numa_id));
            cuda_err = cudaHostRegister(*(host_buf_ptrs[i]), args.size, cudaHostRegisterMapped);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "PrepareBuf::cudaHostRegister error: %d\n", cuda_err);
                return -1;
            }
            cuda_err = cudaHostGetDevicePointer((void **)gpu_buf_ptrs[i], *(host_buf_ptrs[i]), 0);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "PrepareBuf::cudaHostGetDevicePointer error: %d\n", cuda_err);
                return -1;
            }
        }
    }

    // Initialize source buffer
    if (SetGpu(args.src_gpu_id)) {
        return -1;
    }
    cuda_err = cudaMemcpy(buffers->src_dev_gpu_buf_ptr, buffers->data_buf, args.size, cudaMemcpyDefault);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "PrepareBuf::cudaMemcpy error: %d\n", cuda_err);
        return -1;
    }

    return 0;
}

// Validate the result of data transfer.
int CheckBuf(const BenchArgs &args, const Buffers &buffers) {
    cudaError_t cuda_err = cudaSuccess;

    // Copy result
    if (SetGpu(args.dst_gpu_id)) {
        return -1;
    }
    cuda_err = cudaMemcpy(buffers.check_buf, buffers.src_dev_gpu_buf_ptr, args.size, cudaMemcpyDefault);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CheckBuf::cudaMemcpy error: %d\n", cuda_err);
        return -1;
    }

    // Validate result
    int memcmp_result = memcmp(buffers.data_buf, buffers.check_buf, args.size);
    if (memcmp_result) {
        fprintf(stderr, "CheckBuf: Memory check failed\n");
        return -1;
    }

    return 0;
}

// Destroy data buffers
int DestroyBuf(const BenchArgs &args, Buffers *buffers) {
    int ret = 0;
    cudaError_t cuda_err = cudaSuccess;

    // Destroy original data buffer and check buffer
    if (buffers->data_buf != nullptr)
        numa_free(buffers->data_buf, args.size);
    if (buffers->check_buf != nullptr)
        numa_free(buffers->check_buf, args.size);

    // Only destroy buffers for src/dst devices
    constexpr int num_devices = 2;
    bool is_dev_gpu[num_devices] = {args.is_src_dev_gpu, args.is_dst_dev_gpu};
    int dev_ids[num_devices] = {args.src_gpu_id, args.dst_gpu_id};
    uint8_t **host_buf_ptrs[num_devices] = {&(buffers->src_dev_host_buf_ptr), &(buffers->dst_dev_host_buf_ptr)};
    uint8_t **gpu_buf_ptrs[num_devices] = {&(buffers->src_dev_gpu_buf_ptr), &(buffers->dst_dev_gpu_buf_ptr)};
    for (int i = 0; i < num_devices; i++) {
        // Destroy buffers
        if (is_dev_gpu[i]) {
            if (*(gpu_buf_ptrs[i]) == nullptr) {
                continue;
            }
            // Set to buffer device for GPU buffer
            if (SetGpu(dev_ids[i])) {
                return -1;
            }
            cuda_err = cudaFree(*(gpu_buf_ptrs[i]));
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "DestroyBuf::cudaFree error: %d\n", cuda_err);
                ret = -1;
            }
            *(gpu_buf_ptrs[i]) = nullptr;
        } else {
            if (*(host_buf_ptrs[i]) == nullptr) {
                continue;
            }
            // Set to worker device for host memory buffer
            if (SetGpu(args.worker_gpu_id)) {
                return -1;
            }
            cuda_err = cudaHostUnregister(*(host_buf_ptrs[i]));
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "DestroyBuf::cudaHostUnregister error: %d\n", cuda_err);
                ret = -1;
            }
            numa_free(*(host_buf_ptrs[i]), args.size);
            *(host_buf_ptrs[i]) = nullptr;
            *(gpu_buf_ptrs[i]) = nullptr;
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

// Print result tag as <src_dev>_to_<dst_dev>_by_<worker_dev>_using_<sm|dma>_under_<numa_node>.
void PringResultTag(const BenchArgs &args) {
    if (args.is_src_dev_gpu) {
        printf("gpu%d", args.src_gpu_id);
    } else {
        printf("cpu");
    }
    printf("_to_");
    if (args.is_dst_dev_gpu) {
        printf("gpu%d", args.dst_gpu_id);
    } else {
        printf("cpu");
    }
    printf("_by_gpu%d_using_%s_under_numa%lu", args.worker_gpu_id, args.is_sm_copy ? "sm" : "dma", args.numa_id);
}

// Run copy benchmark.
int RunCopy(const BenchArgs &args, const Buffers &buffers) {
    cudaError_t cuda_err = cudaSuccess;
    cudaStream_t stream;
    uint64_t num_thread_blocks;

    // Set to worker device
    if (SetGpu(args.worker_gpu_id)) {
        return -1;
    }

    // Validate data size for SM copy
    if (args.is_sm_copy) {
        uint64_t num_elements_in_thread_block = NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK;
        uint64_t num_bytes_in_thread_block = num_elements_in_thread_block * sizeof(ulong2);
        if (args.size % num_bytes_in_thread_block) {
            fprintf(stderr, "RunCopy: Data size should be multiple of %lu\n", num_bytes_in_thread_block);
            return -1;
        }
        num_thread_blocks = args.size / num_bytes_in_thread_block;
    }

    // Create stream to launch kernels
    cuda_err = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "RunCopy::cudaStreamCreate error: %d\n", cuda_err);
        return -1;
    }

    // Launch jobs and collect running time
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < args.num_loops; i++) {
        if (args.is_sm_copy) {
            SMCopyKernel<<<num_thread_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
                reinterpret_cast<ulong2 *>(buffers.dst_dev_gpu_buf_ptr),
                reinterpret_cast<ulong2 *>(buffers.src_dev_gpu_buf_ptr));
        } else {
            cudaMemcpyAsync(buffers.dst_dev_gpu_buf_ptr, buffers.src_dev_gpu_buf_ptr, args.size, cudaMemcpyDefault,
                            stream);
        }
    }
    cuda_err = cudaStreamSynchronize(stream);
    auto end = std::chrono::steady_clock::now();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "RunCopy::cudaStreamSynchronize error: %d\n", cuda_err);
        return -1;
    }

    // Destroy stream
    cuda_err = cudaStreamDestroy(stream);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "RunCopy::cudaStreamDestroy error: %d\n", cuda_err);
        return -1;
    }

    // Calculate and display bandwidth if no problem
    double time_in_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    PringResultTag(args);
    printf(" %g\n", args.size * args.num_loops / time_in_sec / 1e9);

    return 0;
}

// Enable peer access between a GPU pair. Return whether succeeds.
int EnablePeerAccess(int src_gpu_id, int dst_gpu_id, int *can_access) {
    cudaError_t cuda_err = cudaSuccess;
    if (src_gpu_id == dst_gpu_id) {
        *can_access = 1;
        return 0;
    }
    cuda_err = cudaDeviceCanAccessPeer(can_access, src_gpu_id, dst_gpu_id);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "EnablePeerAccess::cudaDeviceCanAccessPeer error: %d\n", cuda_err);
        return -1;
    }
    if (can_access) {
        if (SetGpu(src_gpu_id)) {
            return -1;
        }
        cuda_err = cudaDeviceEnablePeerAccess(dst_gpu_id, 0);
        if (cuda_err != cudaErrorPeerAccessAlreadyEnabled && cuda_err != cudaSuccess) {
            fprintf(stderr, "EnablePeerAccess::cudaDeviceEnablePeerAccess error: %d\n", cuda_err);
            return -1;
        }
    }
    return 0;
}

int RunBench(const BenchArgs &args) {
    int ret = 0;
    int destroy_buf_ret = 0;
    Buffers buffers;
    ret = PrepareBuf(args, &buffers);
    if (ret == 0) {
        ret = RunCopy(args, buffers);
        if (ret == 0) {
            ret = CheckBuf(args, buffers);
        }
    }
    destroy_buf_ret = DestroyBuf(args, &buffers);
    if (ret == 0) {
        ret = destroy_buf_ret;
    }
    return ret;
}

int main(int argc, char **argv) {
    int ret = 0;
    int numa_count = 0;
    int gpu_count = 0;
    Opts opts;
    BenchArgs args;
    std::vector<BenchArgs> args_list;
    int can_access = 0;

    ret = ParseOpts(argc, argv, &opts);
    if (ret != 0) {
        return ret;
    }
    args.num_loops = opts.num_loops;
    args.size = opts.size;

    // Get number of NUMA nodes
    if (numa_available()) {
        fprintf(stderr, "main::numa_available error\n");
        return -1;
    }
    numa_count = numa_num_configured_nodes();

    // Get number of GPUs
    ret = GetGpuCount(&gpu_count);
    if (ret != 0) {
        return ret;
    }

    // Scan all NUMA nodes
    for (int i = 0; i < numa_count; i++) {
        args.numa_id = i;
        // Scan all GPUs
        for (int j = 0; j < gpu_count; j++) {
            // Host-to-device benchmark
            if (opts.htod_enabled) {
                args.is_src_dev_gpu = false;
                args.is_dst_dev_gpu = true;
                args.dst_gpu_id = j;
                args.worker_gpu_id = j;
                if (opts.sm_copy_enabled) {
                    args.is_sm_copy = true;
                    args_list.push_back(args);
                }
                if (opts.dma_copy_enabled) {
                    args.is_sm_copy = false;
                    args_list.push_back(args);
                }
            }
            // Device-to-host benchmark
            if (opts.dtoh_enabled) {
                args.is_src_dev_gpu = true;
                args.src_gpu_id = j;
                args.is_dst_dev_gpu = false;
                args.worker_gpu_id = j;
                if (opts.sm_copy_enabled) {
                    args.is_sm_copy = true;
                    args_list.push_back(args);
                }
                if (opts.dma_copy_enabled) {
                    args.is_sm_copy = false;
                    args_list.push_back(args);
                }
            }
            // Device-to-device benchmark
            if (opts.dtod_enabled) {
                args.is_src_dev_gpu = true;
                args.src_gpu_id = j;
                args.is_dst_dev_gpu = true;
                // Scan all peers
                for (int k = 0; k < gpu_count; k++) {
                    args.dst_gpu_id = k;
                    // P2P write
                    ret = EnablePeerAccess(j, k, &can_access);
                    if (ret != 0) {
                        return -1;
                    }
                    if (can_access) {
                        args.worker_gpu_id = j;
                        if (opts.sm_copy_enabled) {
                            args.is_sm_copy = true;
                            args_list.push_back(args);
                        }
                        if (opts.dma_copy_enabled) {
                            args.is_sm_copy = false;
                            args_list.push_back(args);
                        }
                    }
                    if (j == k) {
                        continue;
                    }
                    // P2P read
                    ret = EnablePeerAccess(k, j, &can_access);
                    if (ret != 0) {
                        return -1;
                    }
                    if (can_access) {
                        args.worker_gpu_id = k;
                        if (opts.sm_copy_enabled) {
                            args.is_sm_copy = true;
                            args_list.push_back(args);
                        }
                        if (opts.dma_copy_enabled) {
                            args.is_sm_copy = false;
                            args_list.push_back(args);
                        }
                    }
                }
            }
        }
    }

    for (const BenchArgs &curr_args : args_list) {
        ret = numa_run_on_node(curr_args.numa_id);
        if (ret != 0) {
            fprintf(stderr, "main::numa_run_on_node error: %d\n", errno);
            return -1;
        }
        ret = RunBench(curr_args);
        if (ret != 0) {
            return -1;
        }
    }

    return ret;
}
