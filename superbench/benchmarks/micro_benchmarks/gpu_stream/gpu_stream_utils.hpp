// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <chrono>
#include <getopt.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <nvml.h>

// Custom deleter for GPU buffers
struct GpuBufferDeleter {
    template <typename T> void operator()(T *ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

unsigned long long getCurrentTimestampInMicroseconds();

namespace stream_config {
constexpr std::array<int, 4> kThreadsPerBlock = {128, 256, 512, 1024}; // Threads per block
constexpr uint64_t kDefaultBufferSizeInBytes = 4294967296;             // Default buffer size 4GB
constexpr int kNumBuffers = 3;                                         // Number of buffers for triad, add kernel
constexpr int kNumValidationBuffers = 4;                          // Number of validation buffers, one for each kernel
constexpr int kUInt8Mod = 256;                                    // Modulo for unsigned long data type
constexpr std::array<int, 4> kBufferBwMultipliers = {2, 2, 3, 3}; // Buffer multiplier for triad, add kernel
constexpr double scalar = 11.0;                                   // Scalar for scale, triad kernel

// Enum for different kernels
enum class Kernel {
    kCopy,
    kScale,
    kAdd,
    kTriad,
    kCount // Add a count to keep track of the number of enums. Helpful for iterating over enums.
};

// Arguments for each sub benchmark run.
template <typename T> struct SubBenchArgs {
    // Unique pointer for GPU buffers
    using GpuBufferUniquePtr = std::unique_ptr<T, GpuBufferDeleter>;

    // Original data buffer.
    T *data_buf = nullptr;

    // Buffer to validate the correctness of data transfer.
    T *check_buf = nullptr;

    // GPU pointer of the data buffer on source devices.
    std::vector<GpuBufferUniquePtr> gpu_buf_ptrs;

    // Pointer of the validation buffers for each kernel. Order is same as Kernel enum.
    std::vector<std::vector<T>> validation_buf_ptrs;

    // CUDA stream to be used.
    cudaStream_t stream;

    // CUDA event to record start time.
    cudaEvent_t start_event;

    // CUDA event to record end time.
    cudaEvent_t end_event;

    // CUDA event to record end time.
    std::vector<std::vector<float>> times_in_ms;

    // Stream Kernel name.
    std::string kernel_name;
};

// Arguments for each benchmark run.
template <typename T> struct BenchArgs {

    // GPU ID for device (always 0 - actual GPU determined by CUDA_VISIBLE_DEVICES).
    int gpu_id = 0;

    // GPU device info
    cudaDeviceProp gpu_device_prop;

    // Data buffer size used.
    uint64_t size = kDefaultBufferSizeInBytes;

    // Number of warm up rounds to run.
    uint64_t num_warm_up = 0;

    // Number of loops to run.
    uint64_t num_loops = 1;

    // Whether check data after copy.
    bool check_data = false;

    // Sub-benchmarks in parallel.
    SubBenchArgs<T> sub;
};

// Options accepted by this program.
struct Opts {
    // Data buffer size for copy benchmark.
    uint64_t size = kDefaultBufferSizeInBytes;

    // Number of warm up rounds to run.
    uint64_t num_warm_up = 0;

    // Number of loops to run.
    uint64_t num_loops = 0;

    // Whether check data after copy.
    bool check_data = false;

    // Data type for the benchmark ("float" or "double").
    std::string data_type = "double";
};

std::string KernelToString(int); // Function to convert enum to string
int ParseOpts(int, char **, Opts *);
void PrintInputInfo(Opts &);
void PrintUsage();

} // namespace stream_config
