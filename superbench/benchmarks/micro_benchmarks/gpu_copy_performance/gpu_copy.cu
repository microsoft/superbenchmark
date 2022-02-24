// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// GPU copy benchmark tests dtoh/htod/dtod data transfer bandwidth by GPU SM/DMA.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <getopt.h>
#include <numa.h>

#include <cuda.h>
#include <cuda_runtime.h>

// Arguments for each sub benchmark run.
struct SubBenchArgs {
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

    // CUDA stream to be used.
    cudaStream_t stream;

    // CUDA event to record start time.
    cudaEvent_t start_event;

    // CUDA event to record end time.
    cudaEvent_t end_event;
};

// Arguments for each benchmark run.
struct BenchArgs {
    // Max number of sub benchmarks.
    static const int kMaxNumSubs = 2;

    // Number of sub benchmarks in this benchmark run.
    // 1 for unidirectional, 2 for bidirectional.
    int num_subs = 0;

    // NUMA node under which the benchmark is done.
    uint64_t numa_id = 0;

    // Data buffer size used.
    uint64_t size = 0;

    // Number of warm up rounds to run.
    uint64_t num_warm_up = 0;

    // Number of loops to run.
    uint64_t num_loops = 0;

    // Uses SM copy, otherwise DMA copy.
    bool is_sm_copy = false;

    // Whether check data after copy.
    bool check_data = false;

    // Sub-benchmarks in parallel.
    SubBenchArgs subs[kMaxNumSubs];
};

// Options accepted by this program.
struct Opts {
    // Data buffer size for copy benchmark.
    uint64_t size = 0;

    // Number of warm up rounds to run.
    uint64_t num_warm_up = 0;

    // Number of loops to run.
    uint64_t num_loops = 0;

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

    // Whether bidirectional transfer is enabled.
    bool bidirectional_enabled = false;

    // Whether check data after copy.
    bool check_data = false;
};

// Print usage of this program.
void PrintUsage() {
    printf("Usage: gpu_copy "
           "--size <size> "
           "--num_warm_up <num_warm_up> "
           "--num_loops <num_loops> "
           "[--sm_copy] "
           "[--dma_copy] "
           "[--htod] "
           "[--dtoh] "
           "[--dtod] "
           "[--bidirectional] "
           "[--check_data]\n");
}

// Parse options of this program.
int ParseOpts(int argc, char **argv, Opts *opts) {
    enum class OptIdx {
        kSize,
        kNumWarmUp,
        kNumLoops,
        kEnableSmCopy,
        kEnableDmaCopy,
        kEnableHToD,
        kEnableDToH,
        kEnableDToD,
        kEnableBidirectional,
        kEnableCheckData
    };
    const struct option options[] = {
        {"size", required_argument, nullptr, static_cast<int>(OptIdx::kSize)},
        {"num_warm_up", required_argument, nullptr, static_cast<int>(OptIdx::kNumWarmUp)},
        {"num_loops", required_argument, nullptr, static_cast<int>(OptIdx::kNumLoops)},
        {"sm_copy", no_argument, nullptr, static_cast<int>(OptIdx::kEnableSmCopy)},
        {"dma_copy", no_argument, nullptr, static_cast<int>(OptIdx::kEnableDmaCopy)},
        {"htod", no_argument, nullptr, static_cast<int>(OptIdx::kEnableHToD)},
        {"dtoh", no_argument, nullptr, static_cast<int>(OptIdx::kEnableDToH)},
        {"dtod", no_argument, nullptr, static_cast<int>(OptIdx::kEnableDToD)},
        {"bidirectional", no_argument, nullptr, static_cast<int>(OptIdx::kEnableBidirectional)},
        {"check_data", no_argument, nullptr, static_cast<int>(OptIdx::kEnableCheckData)}};
    int getopt_ret = 0;
    int opt_idx = 0;
    bool size_specified = false;
    bool num_warm_up_specified = false;
    bool num_loops_specified = false;
    bool parse_err = false;
    while (true) {
        getopt_ret = getopt_long(argc, argv, "", options, &opt_idx);
        if (getopt_ret == -1) {
            if (!size_specified || !num_warm_up_specified || !num_loops_specified) {
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
        case static_cast<int>(OptIdx::kNumWarmUp):
            if (1 != sscanf(optarg, "%lu", &(opts->num_warm_up))) {
                fprintf(stderr, "Invalid num_warm_up: %s\n", optarg);
                parse_err = true;
            } else {
                num_warm_up_specified = true;
            }
            break;
        case static_cast<int>(OptIdx::kNumLoops):
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
        case static_cast<int>(OptIdx::kEnableBidirectional):
            opts->bidirectional_enabled = true;
            break;
        case static_cast<int>(OptIdx::kEnableCheckData):
            opts->check_data = true;
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

// Prepare data buffers and streams to be used.
int PrepareBufAndStream(BenchArgs *args) {
    cudaError_t cuda_err = cudaSuccess;
    constexpr int uint8_mod = 256;

    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];

        // Generate data to copy
        sub.data_buf = static_cast<uint8_t *>(numa_alloc_onnode(args->size, args->numa_id));

        if (args->check_data) {
            for (int j = 0; j < args->size; j++) {
                sub.data_buf[j] = static_cast<uint8_t>(j % uint8_mod);
            }
            // Allocate check buffer
            sub.check_buf = static_cast<uint8_t *>(numa_alloc_onnode(args->size, args->numa_id));
        }

        // Allocate buffers for src/dst devices
        constexpr int num_devices = 2;
        bool is_dev_gpu[num_devices] = {sub.is_src_dev_gpu, sub.is_dst_dev_gpu};
        int dev_ids[num_devices] = {sub.src_gpu_id, sub.dst_gpu_id};
        uint8_t **host_buf_ptrs[num_devices] = {&(sub.src_dev_host_buf_ptr), &(sub.dst_dev_host_buf_ptr)};
        uint8_t **gpu_buf_ptrs[num_devices] = {&(sub.src_dev_gpu_buf_ptr), &(sub.dst_dev_gpu_buf_ptr)};
        for (int j = 0; j < num_devices; j++) {
            // Allocate buffers
            if (is_dev_gpu[j]) {
                // Set to buffer device for GPU buffer
                if (SetGpu(dev_ids[j])) {
                    return -1;
                }
                *(host_buf_ptrs[j]) = nullptr;
                cuda_err = cudaMalloc(gpu_buf_ptrs[j], args->size);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "PrepareBufAndStream::cudaMalloc error: %d\n", cuda_err);
                    return -1;
                }
            } else {
                // Set to worker device for host memory buffer
                if (SetGpu(sub.worker_gpu_id)) {
                    return -1;
                }
                *(host_buf_ptrs[j]) = static_cast<uint8_t *>(numa_alloc_onnode(args->size, args->numa_id));
                cuda_err = cudaHostRegister(*(host_buf_ptrs[j]), args->size, cudaHostRegisterMapped);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "PrepareBufAndStream::cudaHostRegister error: %d\n", cuda_err);
                    return -1;
                }
                cuda_err = cudaHostGetDevicePointer((void **)gpu_buf_ptrs[j], *(host_buf_ptrs[j]), 0);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "PrepareBufAndStream::cudaHostGetDevicePointer error: %d\n", cuda_err);
                    return -1;
                }
            }
        }

        // Initialize source buffer
        if (sub.is_src_dev_gpu) {
            if (SetGpu(sub.src_gpu_id)) {
                return -1;
            }
        }
        cuda_err = cudaMemcpy(sub.src_dev_gpu_buf_ptr, sub.data_buf, args->size, cudaMemcpyDefault);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "PrepareBufAndStream::cudaMemcpy error: %d\n", cuda_err);
            return -1;
        }

        // Initialize stream on worker device
        if (SetGpu(sub.worker_gpu_id)) {
            return -1;
        }
        cuda_err = cudaStreamCreateWithFlags(&(sub.stream), cudaStreamNonBlocking);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "PrepareBufAndStream::cudaStreamCreate error: %d\n", cuda_err);
            return -1;
        }
    }

    return 0;
}

// Prepare events to be used.
int PrepareEvent(BenchArgs *args) {
    cudaError_t cuda_err = cudaSuccess;
    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];
        if (SetGpu(sub.worker_gpu_id)) {
            return -1;
        }
        cuda_err = cudaEventCreate(&(sub.start_event));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "PrepareEvent::cudaEventCreate error: %d\n", cuda_err);
            return -1;
        }
        cuda_err = cudaEventCreate(&(sub.end_event));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "PrepareEvent::cudaEventCreate error: %d\n", cuda_err);
            return -1;
        }
    }
    return 0;
}

// Validate the result of data transfer.
int CheckBuf(BenchArgs *args) {
    cudaError_t cuda_err = cudaSuccess;
    int memcmp_result = 0;

    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];

        // Copy result
        memset(sub.check_buf, 0, args->size);
        if (SetGpu(sub.dst_gpu_id)) {
            return -1;
        }
        cuda_err = cudaMemcpy(sub.check_buf, sub.dst_dev_gpu_buf_ptr, args->size, cudaMemcpyDefault);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CheckBuf::cudaMemcpy error: %d\n", cuda_err);
            return -1;
        }

        // Validate result
        memcmp_result = memcmp(sub.data_buf, sub.check_buf, args->size);
        if (memcmp_result) {
            fprintf(stderr, "CheckBuf: Memory check failed\n");
            return -1;
        }
    }

    return 0;
}

// Destroy data buffers and streams
int DestroyBufAndStream(BenchArgs *args) {
    int ret = 0;
    cudaError_t cuda_err = cudaSuccess;

    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];

        // Destroy original data buffer and check buffer
        if (sub.data_buf != nullptr) {
            numa_free(sub.data_buf, args->size);
        }
        if (sub.check_buf != nullptr) {
            numa_free(sub.check_buf, args->size);
        }

        // Only destroy buffers for src/dst devices
        constexpr int num_devices = 2;
        bool is_dev_gpu[num_devices] = {sub.is_src_dev_gpu, sub.is_dst_dev_gpu};
        int dev_ids[num_devices] = {sub.src_gpu_id, sub.dst_gpu_id};
        uint8_t **host_buf_ptrs[num_devices] = {&(sub.src_dev_host_buf_ptr), &(sub.dst_dev_host_buf_ptr)};
        uint8_t **gpu_buf_ptrs[num_devices] = {&(sub.src_dev_gpu_buf_ptr), &(sub.dst_dev_gpu_buf_ptr)};
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
                    fprintf(stderr, "DestroyBufAndStream::cudaFree error: %d\n", cuda_err);
                    ret = -1;
                }
                *(gpu_buf_ptrs[i]) = nullptr;
            } else {
                if (*(host_buf_ptrs[i]) == nullptr) {
                    continue;
                }
                // Set to worker device for host memory buffer
                if (SetGpu(sub.worker_gpu_id)) {
                    return -1;
                }
                cuda_err = cudaHostUnregister(*(host_buf_ptrs[i]));
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "DestroyBufAndStream::cudaHostUnregister error: %d\n", cuda_err);
                    ret = -1;
                }
                numa_free(*(host_buf_ptrs[i]), args->size);
                *(host_buf_ptrs[i]) = nullptr;
                *(gpu_buf_ptrs[i]) = nullptr;
            }
        }

        // Destroy stream on worker device
        if (SetGpu(sub.worker_gpu_id)) {
            return -1;
        }
        cuda_err = cudaStreamDestroy(sub.stream);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "DestroyBufAndStream::cudaStreamDestroy error: %d\n", cuda_err);
            return -1;
        }
    }

    return ret;
}

// Destroy events
int DestroyEvent(BenchArgs *args) {
    cudaError_t cuda_err = cudaSuccess;
    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];
        if (SetGpu(sub.worker_gpu_id)) {
            return -1;
        }
        cuda_err = cudaEventDestroy(sub.start_event);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "DestroyEvent::cudaEventDestroy error: %d\n", cuda_err);
            return -1;
        }
        cuda_err = cudaEventDestroy(sub.end_event);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "DestroyEvent::cudaEventDestroy error: %d\n", cuda_err);
            return -1;
        }
    }
    return 0;
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
void PrintResultTag(const BenchArgs &args) {
    if (args.subs[0].is_src_dev_gpu) {
        printf("gpu%d", args.subs[0].src_gpu_id);
    } else {
        printf("cpu");
    }
    printf("%s", args.num_subs == 1 ? "_to_" : "_and_");
    if (args.subs[0].is_dst_dev_gpu) {
        printf("gpu%d", args.subs[0].dst_gpu_id);
    } else {
        printf("cpu");
    }
    if (args.subs[0].is_src_dev_gpu && args.subs[0].is_dst_dev_gpu &&
        args.subs[0].src_gpu_id != args.subs[0].dst_gpu_id) {
        if (args.subs[0].src_gpu_id == args.subs[0].worker_gpu_id) {
            printf("_write");
        } else {
            printf("_read");
        }
    }
    printf("_by_%s", args.is_sm_copy ? "sm" : "dma");
    if (!args.subs[0].is_src_dev_gpu || !args.subs[0].is_dst_dev_gpu) {
        printf("_under_numa%lu", args.numa_id);
    }
}

// Run copy benchmark.
int RunCopy(BenchArgs *args) {
    cudaError_t cuda_err = cudaSuccess;
    uint64_t num_thread_blocks;

    // Validate data size for SM copy
    if (args->is_sm_copy) {
        uint64_t num_elements_in_thread_block = NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK;
        uint64_t num_bytes_in_thread_block = num_elements_in_thread_block * sizeof(ulong2);
        if (args->size % num_bytes_in_thread_block) {
            fprintf(stderr, "RunCopy: Data size should be multiple of %lu\n", num_bytes_in_thread_block);
            return -1;
        }
        num_thread_blocks = args->size / num_bytes_in_thread_block;
    }

    // Launch jobs and collect running time
    for (int i = 0; i < args->num_loops + args->num_warm_up; i++) {
        for (int j = 0; j < args->num_subs; j++) {
            SubBenchArgs &sub = args->subs[j];
            if (SetGpu(sub.worker_gpu_id)) {
                return -1;
            }
            if (i == args->num_warm_up) {
                cuda_err = cudaEventRecord(sub.start_event, sub.stream);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "RunCopy::cudaEventRecord error: %d\n", cuda_err);
                    return -1;
                }
            }
            if (args->is_sm_copy) {
                SMCopyKernel<<<num_thread_blocks, NUM_THREADS_IN_BLOCK, 0, sub.stream>>>(
                    reinterpret_cast<ulong2 *>(sub.dst_dev_gpu_buf_ptr),
                    reinterpret_cast<ulong2 *>(sub.src_dev_gpu_buf_ptr));
            } else {
                cuda_err = cudaMemcpyAsync(sub.dst_dev_gpu_buf_ptr, sub.src_dev_gpu_buf_ptr, args->size,
                                           cudaMemcpyDefault, sub.stream);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "RunCopy::cudaMemcpyAsync error: %d\n", cuda_err);
                    return -1;
                }
            }
            if (i + 1 == args->num_loops + args->num_warm_up) {
                cuda_err = cudaEventRecord(sub.end_event, sub.stream);
                if (cuda_err != cudaSuccess) {
                    fprintf(stderr, "RunCopy::cudaEventRecord error: %d\n", cuda_err);
                    return -1;
                }
            }
        }
    }
    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];
        cuda_err = cudaStreamSynchronize(sub.stream);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "RunCopy::cudaStreamSynchronize error: %d\n", cuda_err);
            return -1;
        }
    }

    // Calculate and display bandwidth if no problem
    float max_time_in_ms = 0;
    for (int i = 0; i < args->num_subs; i++) {
        SubBenchArgs &sub = args->subs[i];
        float time_in_ms = 0;
        cuda_err = cudaEventElapsedTime(&time_in_ms, sub.start_event, sub.end_event);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "RunCopy::cudaEventElapsedTime error: %d\n", cuda_err);
            return -1;
        }
        max_time_in_ms = time_in_ms > max_time_in_ms ? time_in_ms : max_time_in_ms;
    }

    PrintResultTag(*args);
    printf(" %g\n", args->size * args->num_loops * args->num_subs / max_time_in_ms / 1e6);

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
    if (*can_access) {
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

int RunBench(BenchArgs *args) {
    int ret = 0;
    int destroy_ret = 0;
    ret = PrepareBufAndStream(args);
    if (ret != 0) {
        goto destroy_buf;
    }
    ret = PrepareEvent(args);
    if (ret != 0) {
        goto destroy_event;
    }
    ret = RunCopy(args);
    if (ret == 0 && args->check_data) {
        ret = CheckBuf(args);
    }
destroy_event:
    destroy_ret = DestroyEvent(args);
    if (ret == 0) {
        ret = destroy_ret;
    }
destroy_buf:
    destroy_ret = DestroyBufAndStream(args);
    if (ret == 0) {
        ret = destroy_ret;
    }
    return ret;
}

void SetSubBenchArgsForHToD(int gpu_id, bool is_bidirectional, BenchArgs *args) {
    args->subs[0].is_src_dev_gpu = false;
    args->subs[0].is_dst_dev_gpu = true;
    args->subs[0].dst_gpu_id = gpu_id;
    args->subs[0].worker_gpu_id = gpu_id;
    if (is_bidirectional) {
        args->num_subs = 2;
        args->subs[1].is_src_dev_gpu = true;
        args->subs[1].is_dst_dev_gpu = false;
        args->subs[1].src_gpu_id = gpu_id;
        args->subs[1].worker_gpu_id = gpu_id;
    } else {
        args->num_subs = 1;
    }
}

void SetSubBenchArgsForDToH(int gpu_id, bool is_bidirectional, BenchArgs *args) {
    args->subs[0].is_src_dev_gpu = true;
    args->subs[0].is_dst_dev_gpu = false;
    args->subs[0].src_gpu_id = gpu_id;
    args->subs[0].worker_gpu_id = gpu_id;
    if (is_bidirectional) {
        args->num_subs = 2;
        args->subs[1].is_src_dev_gpu = false;
        args->subs[1].is_dst_dev_gpu = true;
        args->subs[1].dst_gpu_id = gpu_id;
        args->subs[1].worker_gpu_id = gpu_id;
    } else {
        args->num_subs = 1;
    }
}

void SetSubBenchArgsForDToD(int src_gpu_id, int dst_gpu_id, bool is_read, bool is_bidirectional, BenchArgs *args) {
    args->subs[0].is_src_dev_gpu = true;
    args->subs[0].is_dst_dev_gpu = true;
    args->subs[0].src_gpu_id = src_gpu_id;
    args->subs[0].dst_gpu_id = dst_gpu_id;
    args->subs[0].worker_gpu_id = is_read ? dst_gpu_id : src_gpu_id;
    if (is_bidirectional) {
        args->num_subs = 2;
        args->subs[1].is_src_dev_gpu = true;
        args->subs[1].is_dst_dev_gpu = true;
        args->subs[1].src_gpu_id = dst_gpu_id;
        args->subs[1].dst_gpu_id = src_gpu_id;
        args->subs[1].worker_gpu_id = is_read ? src_gpu_id : dst_gpu_id;
    } else {
        args->num_subs = 1;
    }
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
    args.num_warm_up = opts.num_warm_up;
    args.num_loops = opts.num_loops;
    args.size = opts.size;
    args.check_data = opts.check_data;

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
                if (opts.sm_copy_enabled) {
                    args.is_sm_copy = true;
                    SetSubBenchArgsForHToD(j, opts.bidirectional_enabled, &args);
                    args_list.push_back(args);
                }
                if (opts.dma_copy_enabled) {
                    args.is_sm_copy = false;
                    SetSubBenchArgsForHToD(j, opts.bidirectional_enabled, &args);
                    args_list.push_back(args);
                }
            }
            // Device-to-host benchmark
            if (opts.dtoh_enabled) {
                if (opts.sm_copy_enabled) {
                    args.is_sm_copy = true;
                    SetSubBenchArgsForDToH(j, opts.bidirectional_enabled, &args);
                    args_list.push_back(args);
                }
                if (opts.dma_copy_enabled) {
                    args.is_sm_copy = false;
                    SetSubBenchArgsForDToH(j, opts.bidirectional_enabled, &args);
                    args_list.push_back(args);
                }
            }
            if (args.numa_id != 0) {
                continue;
            }
            // Device-to-device benchmark
            if (opts.dtod_enabled) {
                // Scan all peers
                for (int k = 0; k < gpu_count; k++) {
                    // src_dev_id always <= dst_dev_id for bidirectional test
                    if (opts.bidirectional_enabled && j > k) {
                        continue;
                    }
                    // P2P write
                    ret = EnablePeerAccess(j, k, &can_access);
                    if (ret != 0) {
                        return -1;
                    }
                    if (can_access) {
                        if (opts.sm_copy_enabled) {
                            args.is_sm_copy = true;
                            SetSubBenchArgsForDToD(j, k, false, opts.bidirectional_enabled, &args);
                            args_list.push_back(args);
                        }
                        if (opts.dma_copy_enabled) {
                            args.is_sm_copy = false;
                            SetSubBenchArgsForDToD(j, k, false, opts.bidirectional_enabled, &args);
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
                        if (opts.sm_copy_enabled) {
                            args.is_sm_copy = true;
                            SetSubBenchArgsForDToD(j, k, true, opts.bidirectional_enabled, &args);
                            args_list.push_back(args);
                        }
                        if (opts.dma_copy_enabled) {
                            args.is_sm_copy = false;
                            SetSubBenchArgsForDToD(j, k, true, opts.bidirectional_enabled, &args);
                            args_list.push_back(args);
                        }
                    }
                }
            }
        }
    }

    for (BenchArgs &curr_args : args_list) {
        ret = numa_run_on_node(curr_args.numa_id);
        if (ret != 0) {
            fprintf(stderr, "main::numa_run_on_node error: %d\n", errno);
            return -1;
        }
        ret = RunBench(&curr_args);
        if (ret != 0) {
            return -1;
        }
    }

    return ret;
}
