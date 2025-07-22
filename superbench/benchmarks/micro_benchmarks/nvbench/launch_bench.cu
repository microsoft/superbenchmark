#include <nvbench/nvbench.cuh>

__global__ void empty_kernel() {}

void launch_bench(nvbench::state &state) {
  state.exec([](nvbench::launch &launch) {
    empty_kernel<<<1, 1, 0, launch.get_stream()>>>();
  });
}

NVBENCH_BENCH(launch_bench);