// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Kernel launch benchmark with four metrics:
//   e2e_latency_us: single-shot end-to-end latency per kernel.
//   host_dispatch_us: host-side dispatch cost per kernel.
//   launch_throughput_mkps: steady-state launch throughput in million kernels/s.
//   device_launch_us: device-side average time per kernel measured by events.

#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>

#include "cuda_runtime.h"

__global__ void EmptyKernel() {}

namespace {

constexpr int kDeviceId = 0;
constexpr int kDispatchBatchSize = 32;

void CheckCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(result));
        exit(1);
    }
}

double test_e2e_latency_us(int num_warmups, int num_steps) {
    CheckCuda(cudaSetDevice(kDeviceId));

    for (int i = 0; i < num_warmups; i++) {
        EmptyKernel<<<1, 1>>>();
        CheckCuda(cudaDeviceSynchronize());
    }

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < num_steps; i++) {
        EmptyKernel<<<1, 1>>>();
        CheckCuda(cudaDeviceSynchronize());
    }
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::micro> elapsed = end - begin;
    return elapsed.count() / num_steps;
}

double test_host_dispatch_us(int num_warmups, int num_steps) {
    CheckCuda(cudaSetDevice(kDeviceId));

    for (int i = 0; i < num_warmups; i++) {
        EmptyKernel<<<1, 1>>>();
    }
    CheckCuda(cudaDeviceSynchronize());

    int remaining = num_steps;
    std::chrono::duration<double, std::micro> total_elapsed(0);

    while (remaining > 0) {
        const int current_batch = std::min(kDispatchBatchSize, remaining);

        CheckCuda(cudaDeviceSynchronize());

        auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < current_batch; i++) {
            EmptyKernel<<<1, 1>>>();
        }
        auto end = std::chrono::steady_clock::now();

        total_elapsed += end - begin;
        remaining -= current_batch;
    }

    CheckCuda(cudaDeviceSynchronize());

    return total_elapsed.count() / num_steps;
}

double test_launch_throughput_mkps(int num_warmups, int num_steps) {
    CheckCuda(cudaSetDevice(kDeviceId));

    for (int i = 0; i < num_warmups; i++) {
        EmptyKernel<<<1, 1>>>();
    }
    CheckCuda(cudaDeviceSynchronize());

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < num_steps; i++) {
        EmptyKernel<<<1, 1>>>();
    }
    CheckCuda(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed = end - begin;
    return static_cast<double>(num_steps) / elapsed.count() / 1e6;
}

double test_device_launch_us(int num_warmups, int num_steps) {
    CheckCuda(cudaSetDevice(kDeviceId));

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start));
    CheckCuda(cudaEventCreate(&stop));

    for (int i = 0; i < num_warmups; i++) {
        EmptyKernel<<<1, 1>>>();
    }
    CheckCuda(cudaDeviceSynchronize());

    CheckCuda(cudaEventRecord(start, 0));
    for (int i = 0; i < num_steps; i++) {
        EmptyKernel<<<1, 1>>>();
    }
    CheckCuda(cudaEventRecord(stop, 0));
    CheckCuda(cudaEventSynchronize(stop));

    float total_time_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&total_time_ms, start, stop));
    CheckCuda(cudaEventDestroy(start));
    CheckCuda(cudaEventDestroy(stop));

    return total_time_ms * 1000.0 / num_steps;
}

char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
    int num_warmups = 100;
    int num_steps = 2000000;
    int interval = 2000;

    if (char *value = getCmdOption(argv, argv + argc, "-w")) {
        num_warmups = std::stoi(value);
    }

    if (char *value = getCmdOption(argv, argv + argc, "-n")) {
        num_steps = std::stoi(value);
    }

    if (char *value = getCmdOption(argv, argv + argc, "-i")) {
        interval = std::stoi(value);
    }

    const double e2e_latency_us = test_e2e_latency_us(num_warmups, num_steps);
    printf("e2e_latency_us: %.6f\n", e2e_latency_us);

    std::this_thread::sleep_for(std::chrono::milliseconds(interval));

    const double host_dispatch_us = test_host_dispatch_us(num_warmups, num_steps);
    printf("host_dispatch_us: %.6f\n", host_dispatch_us);

    std::this_thread::sleep_for(std::chrono::milliseconds(interval));

    const double launch_throughput_mkps = test_launch_throughput_mkps(num_warmups, num_steps);
    printf("launch_throughput_mkps: %.6f\n", launch_throughput_mkps);

    std::this_thread::sleep_for(std::chrono::milliseconds(interval));

    const double device_launch_us = test_device_launch_us(num_warmups, num_steps);
    printf("device_launch_us: %.6f\n", device_launch_us);

    return 0;
}
