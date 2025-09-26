// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <getopt.h>
#include <iostream>
#include <memory>
#include <variant>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <numa.h>

#include "gpu_stream_kernels.hpp"
#include "gpu_stream_utils.hpp"

#define NON_HIP (!defined(__HIP_PLATFORM_HCC__) && !defined(__HCC__) && !defined(__HIPCC__))

using namespace stream_config;

class GpuStream {
  public:
    GpuStream() = delete;            // Delete default constructor
    GpuStream(Opts &) noexcept;      // Constructor
    ~GpuStream() noexcept = default; // Destructor

    GpuStream(const GpuStream &) = delete;
    GpuStream &operator=(const GpuStream &) = delete;
    GpuStream(GpuStream &&) noexcept = default;
    GpuStream &operator=(GpuStream &&) noexcept = default;

    int Run();

  private:
    using BenchArgsVariant = std::variant<std::unique_ptr<BenchArgs<double>>>;
    std::vector<BenchArgsVariant> bench_args_;
    Opts opts_;

    // Memory management functions
    template <typename T> cudaError_t GpuMallocDataBuf(T **, uint64_t);
    template <typename T> int PrepareValidationBuf(std::unique_ptr<BenchArgs<T>> &);
    template <typename T> int CheckBuf(std::unique_ptr<BenchArgs<T>> &, int);

    template <typename T> int PrepareEvent(std::unique_ptr<BenchArgs<T>> &);
    template <typename T> int PrepareBufAndStream(std::unique_ptr<BenchArgs<T>> &);

    template <typename T> int DestroyEvent(std::unique_ptr<BenchArgs<T>> &);
    template <typename T> int DestroyBufAndStream(std::unique_ptr<BenchArgs<T>> &);
    template <typename T> int Destroy(std::unique_ptr<BenchArgs<T>> &);

    // Benchmark functions
    template <typename T> int RunStreamKernel(std::unique_ptr<BenchArgs<T>> &, Kernel, int);
    float GetActualMemoryClockRate(int gpu_id);
    template <typename T> int RunStream(std::unique_ptr<BenchArgs<T>> &, const std::string &data_type, float peak_bw);

    // Helper functions
    int GetGpuCount(int *);
    int SetGpu(int gpu_id);
    float GetMemoryClockRate(int device_id, const cudaDeviceProp &prop);
    void PrintCudaDeviceInfo(int device_id, const cudaDeviceProp &prop, float memory_clock_mhz, float peak_bw);
};