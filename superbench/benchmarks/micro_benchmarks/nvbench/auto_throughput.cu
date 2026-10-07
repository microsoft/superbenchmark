// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Based on nvbench example: auto_throughput.cu
// This benchmark measures memory throughput and cache hit rates.

#include <nvbench/nvbench.cuh>

// Thrust vectors simplify memory management:
#include <thrust/device_vector.h>

template <int ItemsPerThread>
__global__ void throughput_kernel(std::size_t stride, std::size_t elements, const nvbench::int32_t *__restrict__ in,
                                  nvbench::int32_t *__restrict__ out) {
    const std::size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const std::size_t step = gridDim.x * blockDim.x;

    for (std::size_t i = stride * tid; i < stride * elements; i += stride * step) {
        for (int j = 0; j < ItemsPerThread; j++) {
            const auto read_id = (ItemsPerThread * i + j) % elements;
            const auto write_id = tid + j * elements;
            out[write_id] = in[read_id];
        }
    }
}

// `throughput_bench` copies a 128 MiB buffer of int32_t, and reports throughput
// and cache hit rates.
//
// Calling state.collect_*() enables particular metric collection if nvbench
// was built with CUPTI support (CMake option: -DNVBench_ENABLE_CUPTI=ON).
template <int ItemsPerThread>
void throughput_bench(nvbench::state &state, nvbench::type_list<nvbench::enum_type<ItemsPerThread>>) {
    // Allocate input data:
    const std::size_t stride = static_cast<std::size_t>(state.get_int64("Stride"));
    const auto threads_in_block = static_cast<int>(state.get_int64("BlockSize"));
    const std::size_t elements = 128 * 1024 * 1024 / sizeof(nvbench::int32_t);
    thrust::device_vector<nvbench::int32_t> input(elements);
    thrust::device_vector<nvbench::int32_t> output(elements * ItemsPerThread);

    // Provide throughput information:
    state.add_element_count(elements, "Elements");

    // CUPTI metrics - these require nvbench built with -DNVBench_ENABLE_CUPTI=ON
    // If CUPTI is not available, these calls are no-ops
    state.collect_dram_throughput();
    state.collect_l1_hit_rates();
    state.collect_l2_hit_rates();
    state.collect_loads_efficiency();
    state.collect_stores_efficiency();

    const auto blocks_in_grid = static_cast<int>((elements + threads_in_block - 1) / threads_in_block);

    state.exec([&](nvbench::launch &launch) {
        throughput_kernel<ItemsPerThread><<<blocks_in_grid, threads_in_block, 0, launch.get_stream()>>>(
            stride, elements, thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()));
    });
}

using items_per_thread = nvbench::enum_type_list<1, 2>;

NVBENCH_BENCH_TYPES(throughput_bench, NVBENCH_TYPE_AXES(items_per_thread))
    .add_int64_axis("Stride", nvbench::range(1, 4, 3))
    .add_int64_axis("BlockSize", {128, 256, 512, 1024})
    .set_timeout(1);
