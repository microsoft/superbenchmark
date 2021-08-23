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
enum class ArgIdx {
  kGpuId = 1,
  kCopyDirection,
  kSize,
  kNumLoops,
  kNumArgs
};

// ID of GPU used in this benchmark.
int g_gpu_id = 0;

// Data transfer direction, can be "dtoh" or "htod".
std::string g_copy_direction;

// Data buffer size used.
uint64_t g_size = 0;

// Number of loops in data transfer benchmark.
uint64_t g_num_loops = 0;

// Original data buffer.
uint8_t* g_data_buf = nullptr;

// Buffer to validate the correctness of data transfer.
uint8_t* g_check_buf = nullptr;

// Data buffer in host memory.
uint8_t* g_host_buf = nullptr;

// Device pointer of the data buffer in host memory.
uint8_t* g_host_buf_dev_ptr = nullptr;

// Data buffer in device memory
uint8_t* g_dev_buf = nullptr;

// Pring usage of this program.
void PrintUsage() {
  printf("Usage: gpu_sm_copy <gpu-id> <copy-direction: dtoh|htod> <size> <num_loops>\n");
}

// Prepare data buffers to be used.
int PrepareBuf() {
  cudaError_t cuda_err = cudaSuccess;

  // Generate data to copy
  g_data_buf = static_cast<uint8_t*>(malloc(g_size));
  for (int i = 0; i < g_size; i++) {
    g_data_buf[i] = static_cast<uint8_t>(i % 256);
  }

  // Reset check buffer
  g_check_buf = static_cast<uint8_t*>(malloc(g_size));
  for (int i = 0; i < g_size; i++) {
    g_check_buf[i] = static_cast<uint8_t>(0);
  }

  // Allocate host buffer
  g_host_buf = static_cast<uint8_t*>(malloc(g_size));
  cuda_err = cudaHostRegister(g_host_buf, g_size, cudaHostRegisterMapped);
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "PrepareBuf::cudaHostRegister error: %d\n", cuda_err);
    return -1;
  }
  cuda_err = cudaHostGetDevicePointer(&g_host_buf_dev_ptr, g_host_buf, 0);
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "PrepareBuf::cudaHostGetDevicePointer error: %d\n", cuda_err);
    return -1;
  }

  // ALlocate device buffer
  cuda_err = cudaMalloc(&g_dev_buf, g_size);
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "PrepareBuf::cudaMalloc error: %d\n", cuda_err);
    return -1;
  }

  // Initialize source buffer
  if (g_copy_direction == "dtoh") {
    cuda_err = cudaMemcpy(g_dev_buf, g_data_buf, g_size, cudaMemcpyDefault);
  } else if (g_copy_direction == "htod") {
    cuda_err = cudaMemcpy(g_host_buf, g_data_buf, g_size, cudaMemcpyDefault);
  } else {
    fprintf(stderr, "Unrecognized copy direction: %s\n", g_copy_direction.c_str());
    return -1;
  }
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "PrepareBuf::cudaMemcpy error: %d\n", cuda_err);
    return -1;
  }

  return 0;
}

// Validate the result of data transfer.
int CheckBuf() {
  cudaError_t cuda_err = cudaSuccess;

  // Copy result
  if (g_copy_direction == "dtoh") {
    cuda_err = cudaMemcpy(g_check_buf, g_host_buf, g_size, cudaMemcpyDefault);
  } else if (g_copy_direction == "htod") {
    cuda_err = cudaMemcpy(g_check_buf, g_dev_buf, g_size, cudaMemcpyDefault);
  }
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "CheckBuf::cudaMemcpy error: %d\n", cuda_err);
    return -1;
  }

  // Validate result
  int memcmp_result = memcmp(g_data_buf, g_check_buf, g_size);
  if (memcmp_result) {
    fprintf(stderr, "Memory check failed\n");
    return -1;
  }

  return 0;
}

// Destory data buffers
int DestroyBuf() {
  cudaError_t cuda_err = cudaSuccess;

  // Destroy original data buffer and check buffer
  if (g_data_buf != nullptr) free(g_data_buf);
  if (g_check_buf != nullptr) free(g_check_buf);

  // Destroy device buffer
  if (g_dev_buf != nullptr) {
    cuda_err = cudaFree(g_dev_buf);
    if (cuda_err != cudaSuccess) {
      fprintf(stderr, "DestroyBuf::cudaFree error: %d\n", cuda_err);
      return -1;
    }
  }

  // Destroy host buffer
  if (g_host_buf != nullptr) {
    cuda_err = cudaHostUnregister(g_host_buf);
    if (cuda_err != cudaSuccess) {
      fprintf(stderr, "DestroyBuf::cudaHostUnregister error: %d\n", cuda_err);
      return -1;
    }
    free(g_host_buf);
    g_host_buf_dev_ptr = nullptr;
  }

  return 0;
}

// Unroll depth in SM copy kernel
#define NUM_LOOP_UNROLL 64

// Thread block size
#define NUM_THREADS_IN_BLOCK 64

// Fetch a ulong2 from source memory and write to register
inline __device__ void FetchULong2(ulong2& v, const ulong2* p) {
  asm volatile("ld.volatile.global.v2.u64 {%0,%1}, [%2];" : "=l"(v.x), "=l"(v.y) : "l"(p) : "memory");
}

// Store a ulong2 from register and write to target memory
inline __device__ void StoreULong2(ulong2* p, ulong2& v) {
  asm volatile("st.volatile.global.v2.u64 [%0], {%1,%2};" :: "l"(p), "l"(v.x), "l"(v.y) : "memory");
}

// Fetch data from source memory into register first, and then write them to target memory
// Stride set to thread block size to best utilize cache
__global__ void SMCopyKernel(ulong2* tgt, const ulong2* src) {
  uint64_t index = blockIdx.x * blockDim.x * NUM_LOOP_UNROLL + threadIdx.x;
  ulong2 val[NUM_LOOP_UNROLL];
  #pragma unroll
  for (uint64_t i = 0; i < NUM_LOOP_UNROLL; i++) FetchULong2(val[i], src + index + i * blockDim.x);
  #pragma unroll
  for (uint64_t i = 0; i < NUM_LOOP_UNROLL; i++) StoreULong2(tgt + index + i * blockDim.x, val[i]);
}

// Run SM copy kernel benchmark
int BenchSMCopyKernel() {
  cudaError_t cuda_err = cudaSuccess;
  cudaStream_t stream;
  uint8_t* g_src_buf = nullptr;
  uint8_t* g_tgt_buf = nullptr;

  // Determine source buffer and target buff
  if (g_copy_direction == "dtoh") {
    g_src_buf = g_dev_buf;
    g_tgt_buf = g_host_buf_dev_ptr;
  } else {
    g_src_buf = g_host_buf_dev_ptr;
    g_tgt_buf = g_dev_buf;
  }

  // Validate data size
  uint64_t num_elements_in_thread_block = NUM_LOOP_UNROLL * NUM_THREADS_IN_BLOCK;
  uint64_t num_bytes_in_thread_block = num_elements_in_thread_block * sizeof(ulong2);
  if (g_size % num_bytes_in_thread_block) {
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
  uint64_t num_thread_blocks = g_size / num_bytes_in_thread_block;
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < g_num_loops; i++) {
    SMCopyKernel<<<num_thread_blocks, NUM_THREADS_IN_BLOCK, 0, stream>>>(
      reinterpret_cast<ulong2 *>(g_tgt_buf),
      reinterpret_cast<ulong2 *>(g_src_buf));
  }
  cuda_err = cudaStreamSynchronize(stream);
  auto end = std::chrono::steady_clock::now();
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "BenchSMCopyKernel::cudaStreamSynchronize error: %d\n", cuda_err);
    return -1;
  }

  // Destory stream
  cuda_err = cudaStreamDestroy(stream);
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "BenchSMCopyKernel::cudaStreamDestroy error: %d\n", cuda_err);
    return -1;
  }

  // Calculate and display bandwidth if no problem
  double time_in_sec = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
  printf("Bandwidth (GB/s): %g\n", g_size * g_num_loops / time_in_sec / 1e9);

  return 0;
}

int main(int argc, char** argv) {
  int ret = 0;
  cudaError_t cuda_err = cudaSuccess;

  if (argc != static_cast<int>(ArgIdx::kNumArgs)) {
    PrintUsage();
    return -1;
  }

  g_gpu_id = std::stoi(argv[static_cast<int>(ArgIdx::kGpuId)]);
  g_copy_direction = argv[static_cast<int>(ArgIdx::kCopyDirection)];
  g_size = std::stoul(argv[static_cast<uint64_t>(ArgIdx::kSize)]);
  g_num_loops = std::stoul(argv[static_cast<uint64_t>(ArgIdx::kNumLoops)]);

  // Set device context
  cuda_err = cudaSetDevice(g_gpu_id);
  if (cuda_err != cudaSuccess) {
    fprintf(stderr, "cudaSetDevice error: %d\n", cuda_err);
    goto destroy_buf;
  }

  // Prepare data buffers
  ret = PrepareBuf();
  if (ret != 0) {
    goto destroy_buf;
  }

  // Run benchmark
  ret = BenchSMCopyKernel();
  if (ret != 0) {
    goto destroy_buf;
  }

  // Validate data
  ret = CheckBuf();

destroy_buf:
  // Destory buffers
  ret = DestroyBuf();

  return ret;
}
