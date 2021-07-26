// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Kernel launch benchmark which will launch one empty kernel and record the cost in event mode and wall mode.
//   event mode: using hip event to record the elapsed time of kernel launch on device.
//   wall mode: using host timer to record the elapsed time kernel launch on both host and device.

#include <algorithm>
#include <chrono>
#include <stdio.h>
#include <string>
#include <sys/time.h>
#include <thread>

#include "hip/hip_runtime.h"

__global__ void EmptyKernel() {}

double test_rocm_kernel_launch_event_time(int num_warmups, int num_steps) {
    float time = 0.f;
    double total_time = 0.0;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    for (int i = 0; i < num_warmups; i++) {
        hipEventRecord(start, 0);
        EmptyKernel<<<1, 1>>>();
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
    }

    for (int i = 0; i < num_steps; i++) {
        hipEventRecord(start, 0);
        EmptyKernel<<<1, 1>>>();
        hipEventRecord(stop, 0);
        hipEventSynchronize(stop);
        hipEventElapsedTime(&time, start, stop);
        total_time += time;
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return total_time;
}

double test_rocm_kernel_launch_wall_time(int num_warmups, int num_steps) {
    double total_time = 0.0;

    for (int i = 0; i < num_warmups; i++) {
        EmptyKernel<<<1, 1>>>();
        hipDeviceSynchronize();
    }

    struct timeval begin_tv, end_tv;
    for (int i = 0; i < num_steps; i++) {
        gettimeofday(&begin_tv, NULL);
        EmptyKernel<<<1, 1>>>();
        hipDeviceSynchronize();
        gettimeofday(&end_tv, NULL);
        total_time += (((end_tv.tv_sec) * 1000 + (end_tv.tv_usec) / 1000) -
                       ((begin_tv.tv_sec) * 1000 + (begin_tv.tv_usec) / 1000));
    }

    return total_time;
}

char *getCmdOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

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

    // Test the kernel launch event time.
    double event_total_time = test_rocm_kernel_launch_event_time(num_warmups, num_steps);
    printf("Kernel launch overhead - event time: %3.5f ms \n", event_total_time / num_steps);

    // Sleep for interval milliseconds and run the next test.
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));

    // Test the kernel launch wall time.
    double wall_total_time = test_rocm_kernel_launch_wall_time(num_warmups, num_steps);
    printf("Kernel launch overhead - wall time: %3.5f ms \n", wall_total_time / num_steps);

    return 0;
}
