// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// GPU stream benchmark
// This benchmark is based on the STREAM benchmark, which is a simple  benchmark program that measures
// sustainable memory bandwidth (in MB/s) and the corresponding computation rate for simple (COPY, SCALE, ADD, TRIAD)
// kernels.

#include "gpu_stream.hpp"
#include <cassert>
#include <iostream>
#include <nvml.h>

/**
 * @brief Destroys the CUDA events used for benchmarking.
 *
 * @details This function cleans up and releases the resources associated with the CUDA events
 * used for benchmarking based on the provided arguments. It ensures that all allocated resources
 * are properly freed.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure
 *
 * @return int The status code indicating success or failure of the destruction process.
 */
template <typename T> int GpuStream::DestroyEvent(std::unique_ptr<BenchArgs<T>> &args) {

    cudaError_t cuda_err = cudaSuccess;
    if (SetGpu(args->gpu_id)) {
        return -1;
    }
    cuda_err = cudaEventDestroy(args->sub.start_event);
    if (cuda_err != cudaSuccess) {
        std::cerr << "DestroyEvent::cudaEventDestroy error: " << cuda_err << std::endl;
        return -1;
    }
    cuda_err = cudaEventDestroy(args->sub.end_event);
    if (cuda_err != cudaSuccess) {
        std::cerr << "DestroyEvent::cudaEventDestroy error: " << cuda_err << std::endl;
        return -1;
    }
    return 0;
}

/**
 * @brief Constructor for the GpuStream class.
 *
 * This constructor initializes the GpuStream opts with the given parameters.
 *
 * @param opts parsed command line options..
 */
GpuStream::GpuStream(Opts &opts) noexcept : opts_(opts) { PrintInputInfo(opts_); }

/**
 * @brief Sets the active GPU.
 *
 * @details This function sets the active GPU to the specified GPU ID.
 *
 * @param[in] gpu_id The ID of the GPU to set as active.
 *
 * @return int The status code indicating success or failure.
 */
int GpuStream::SetGpu(int gpu_id) {
    cudaError_t cuda_err = cudaSetDevice(gpu_id);
    if (cuda_err != cudaSuccess) {
        std::cerr << "SetGpu::cudaSetDevice " << gpu_id << "error: " << cuda_err << std::endl;
        return -1;
    }
    return 0;
}

/**
 * @brief Retrieves the number of GPUs available.
 *
 * @details This function retrieves the number of GPUs available on the system and stores the count
 * in the provided pointer.
 *
 * @param[out] gpu_count Pointer to an integer where the GPU count will be stored.
 *
 * @return int The status code indicating success or failure.
 */
int GpuStream::GetGpuCount(int *gpu_count) {
    cudaError_t cuda_err = cudaGetDeviceCount(gpu_count);
    if (cuda_err != cudaSuccess) {
        std::cerr << "GetGpuCount::cudaGetDeviceCount error: " << cuda_err << std::endl;
        return -1;
    }
    return 0;
}

/**
 * @brief destroys buff and stream resources.
 *
 * @details This method cleans up and releases the resources associated with the buffer and stream
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments.
 *
 * @return int The status code indicating success or failure.
 */
template <typename T> int GpuStream::Destroy(std::unique_ptr<BenchArgs<T>> &args) {
    int ret = DestroyBufAndStream(args);
    if (ret == 0) {
        ret = DestroyEvent(args);
    }
    return ret;
}

/**
 * @brief Prints CUDA device information.
 *
 * @details This function prints the properties of a CUDA device specified by the device ID.
 *
 * @param[in] device_id The ID of the CUDA device.
 * @param[out] prop The properties of the CUDA device.
 * @return void
 * */
float GpuStream::PrintCudaDeviceInfo(int device_id, const cudaDeviceProp &prop) {
    std::cout << "\nDevice " << device_id << ": \"" << prop.name << "\"";
    std::cout << "  " << prop.multiProcessorCount << " SMs(" << prop.major << "." << prop.minor << ")";

    float theoretical_bw = 0.0f;
    
    // Compute theoretical bw:
    // https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/#theoretical_bandwidth
#if CUDA_VERSION >= 12000
    // For CUDA 12.0+, try to get memory clock from NVML
    float memory_clock_mhz = GetActualMemoryClockRate(device_id);
    if (memory_clock_mhz <= 0.0f) {
        // NVML failed, warn and return failure
        std::cout << "  Memory: " << prop.memoryBusWidth << "-bit bus, " 
                  << prop.totalGlobalMem / (1024 * 1024 * 1024) << " GB total (NVML failed - peak BW unavailable)";
        std::cout << "  ECC is " << (prop.ECCEnabled ? "ON" : "OFF") << std::endl;
        return -1.0f;
    }
    
    // NVML succeeded, use the actual memory clock rate
    theoretical_bw = memory_clock_mhz * (prop.memoryBusWidth / 8) * 2 / 1000.0;
    std::cout << "  Memory: " << memory_clock_mhz << "MHz x " << prop.memoryBusWidth
              << "-bit = " << theoretical_bw << " GB/s PEAK ";
#else
    // For CUDA < 12.0, use memoryClockRate from cudaDeviceProp
    // Note: memoryClockRate is in kHz, need to convert to MHz first
    float memory_clock_mhz_converted = prop.memoryClockRate / 1000.0f; // Convert kHz to MHz
    theoretical_bw = memory_clock_mhz_converted * (prop.memoryBusWidth / 8) * 2 / 1000.0;
    std::cout << "  Memory: " << memory_clock_mhz_converted << "MHz x " << prop.memoryBusWidth
              << "-bit = " << theoretical_bw << " GB/s PEAK ";
#endif
    
    std::cout << "  ECC is " << (prop.ECCEnabled ? "ON" : "OFF") << std::endl;
    return theoretical_bw;
}

/**
 * @brief Allocates a buffer on the GPU.
 *
 * @details This function allocates a buffer of the specified size on the GPU and returns a pointer
 * to the allocated memory.
 *
 * @param[out] ptr Pointer to a pointer where the address of the allocated buffer will be stored.
 * @param[in] size The size of the buffer to allocate in bytes.
 *
 * @return cudaError_t The status code indicating success or failure of the memory allocation.
 */
template <typename T> cudaError_t GpuStream::GpuMallocDataBuf(T **ptr, uint64_t size) { return cudaMalloc(ptr, size); }

/**
 * @brief Prepares validation buffers for GPU stream benchmark.
 *
 * @details This function allocates and initializes validation buffers for different
 * kernels (copy, scale, add, and triad) used in the GPU stream benchmark. The buffer order
 * matches the kernel order in the Kernel enum.
 *
 * @param args A unique pointer to a BenchArgs structure containing the necessary arguments
 * for preparing the buffer and stream.
 *
 * @return int The status code indicating success or failure of the preparation.
 */
template <typename T> int GpuStream::PrepareValidationBuf(std::unique_ptr<BenchArgs<T>> &args) {
    args->sub.validation_buf_ptrs.resize(kNumValidationBuffers);

    // Compute and allocate validation buffers for add, scale and triad
    uint64_t size = args->size / sizeof(T);
    for (auto &buf_ptr : args->sub.validation_buf_ptrs) {
        buf_ptr.resize(size);
    }

    // Initialize validation buffer
    for (size_t j = 0; j < size; j++) {
        args->sub.validation_buf_ptrs[0][j] = static_cast<T>(j % kUInt8Mod);
        args->sub.validation_buf_ptrs[1][j] = static_cast<T>(j % kUInt8Mod) * scalar;
        args->sub.validation_buf_ptrs[2][j] = static_cast<T>(j % kUInt8Mod) + static_cast<T>(j % kUInt8Mod);
        args->sub.validation_buf_ptrs[3][j] = static_cast<T>(j % kUInt8Mod) + static_cast<T>(j % kUInt8Mod) * scalar;
    }
    return 0;
}

/**
 * @brief Prepares the buffer and stream for benchmarking.
 *
 * @details This function prepares the necessary buffer and stream for benchmarking based on the
 * provided arguments. It initializes and configures the buffer and stream as required.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments
 * for preparing the buffer and stream.
 *
 * @return int The status code indicating success or failure of the preparation.
 */
template <typename T> int GpuStream::PrepareBufAndStream(std::unique_ptr<BenchArgs<T>> &args) {

    cudaError_t cuda_err = cudaSuccess;

    if (args->check_data) {
        // Generate data to copy
        args->sub.data_buf = static_cast<T *>(numa_alloc_onnode(args->size * sizeof(T), args->numa_id));

        for (int j = 0; j < args->size / sizeof(T); j++) {
            args->sub.data_buf[j] = static_cast<T>(j % kUInt8Mod);
        }

        // Allocate check buffer
        args->sub.check_buf = static_cast<T *>(numa_alloc_onnode(args->size * sizeof(T), args->numa_id));
    }

    // Allocate buffers
    args->sub.gpu_buf_ptrs.resize(kNumBuffers);

    // Set to buffer device for GPU buffer
    if (SetGpu(args->gpu_id)) {
        return -1;
    }

    // Allocate buffers
    for (auto &buf_ptr : args->sub.gpu_buf_ptrs) {
        T *raw_ptr = nullptr;
        cuda_err = GpuMallocDataBuf(&raw_ptr, args->size * sizeof(T));
        if (cuda_err != cudaSuccess) {
            std::cerr << "PrepareBufAndStream::cudaMalloc error: " << cuda_err << std::endl;
            return -1;
        }
        buf_ptr.reset(raw_ptr); // Transfer ownership to the smart pointer
    }

    // Initialize source buffer
    if (args->check_data) {
        cuda_err = cudaMemcpy(args->sub.gpu_buf_ptrs[0].get(), args->sub.data_buf, args->size, cudaMemcpyDefault);
        if (cuda_err != cudaSuccess) {
            std::cerr << "PrepareBufAndStream::cudaMemcpy error: " << cuda_err << std::endl;
            return -1;
        }

        cuda_err = cudaMemcpy(args->sub.gpu_buf_ptrs[1].get(), args->sub.data_buf, args->size, cudaMemcpyDefault);
        if (cuda_err != cudaSuccess) {
            std::cerr << "PrepareBufAndStream::cudaMemcpy error: " << cuda_err << std::endl;
            return -1;
        }

        PrepareValidationBuf<T>(args);
    }

    cuda_err = cudaStreamCreateWithFlags(&(args->sub.stream), cudaStreamNonBlocking);
    if (cuda_err != cudaSuccess) {
        std::cerr << "PrepareBufAndStream::cudaStreamCreate error: " << cuda_err << std::endl;
        return -1;
    }
    return 0;
}

/**
 * @brief Prepares CUDA events for benchmarking.
 *
 * @details This function creates the necessary CUDA events for benchmarking based on the
 * provided arguments. It initializes and configures the events as required.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments
 * for preparing the CUDA events.
 *
 * @return int The status code indicating success or failure of the preparation.
 */
template <typename T> int GpuStream::PrepareEvent(std::unique_ptr<BenchArgs<T>> &args) {

    cudaError_t cuda_err = cudaSuccess;
    if (SetGpu(args->gpu_id)) {
        return -1;
    }

    cuda_err = cudaEventCreate(&(args->sub.start_event));
    if (cuda_err != cudaSuccess) {
        std::cerr << "PrepareEvent::cudaEventCreate error: " << cuda_err << std::endl;
        return -1;
    }

    cuda_err = cudaEventCreate(&(args->sub.end_event));
    if (cuda_err != cudaSuccess) {
        std::cerr << "PrepareEvent::cudaEventCreate error: " << cuda_err << std::endl;
        return -1;
    }

    return 0;
}

/**
 * @brief Validates the result of data transfer.
 *
 * @details This function checks the buffer to validate the result of a data transfer operation
 * based on the provided arguments. It ensures that the data transfer was successful and that
 * the buffer contains the expected data.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments
 * for validating the buffer.
 *
 * @return int The status code indicating success or failure of the validation.
 */
template <typename T> int GpuStream::CheckBuf(std::unique_ptr<BenchArgs<T>> &args, int kernel_idx) {
    cudaError_t cuda_err = cudaSuccess;
    int memcmp_result = 0;

    if (SetGpu(args->gpu_id)) {
        return -1;
    }

    // Copy buffer output from stream kernel to local buffer
    cuda_err = cudaMemcpy(args->sub.check_buf, args->sub.gpu_buf_ptrs[2].get(), args->size, cudaMemcpyDefault);
    if (cuda_err != cudaSuccess) {
        std::cerr << "CheckBuf::cudaMemcpy error: " << cuda_err << std::endl;
        return -1;
    }

    // Validate result by comparing the data buffer and check buffer
    memcmp_result = memcmp(args->sub.validation_buf_ptrs[kernel_idx].data(), args->sub.check_buf, args->size);
    if (memcmp_result) {
        std::cerr << "CheckBuf::Memory check failed for kernel index " << kernel_idx << std::endl;
        return -1;
    }

    return 0;
}

/**
 * @brief Destroys the buffer and stream used for benchmarking.
 *
 * @details This function cleans up and releases the resources associated with the buffer and stream
 * used for benchmarking based on the provided arguments. It ensures that all allocated buffers and streams
 * resources are properly freed.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments
 * for destroying the buffer and stream.
 *
 * @return int The status code indicating success or failure of the destruction process.
 */
template <typename T> int GpuStream::DestroyBufAndStream(std::unique_ptr<BenchArgs<T>> &args) {

    int ret = 0;
    cudaError_t cuda_err = cudaSuccess;

    // Destroy original data buffer and check buffer
    if (args->sub.data_buf != nullptr) {
        numa_free(args->sub.data_buf, args->size);
    }
    if (args->sub.check_buf != nullptr) {
        numa_free(args->sub.check_buf, args->size);
    }

    // Set to buffer device for GPU buffer
    if (SetGpu(args->gpu_id)) {
        return -1;
    }

    cuda_err = cudaStreamDestroy(args->sub.stream);
    if (cuda_err != cudaSuccess) {
        std::cerr << "DestroyBufAndStream::cudaStreamDestroy error: " << cuda_err << std::endl;
        return -1;
    }

    return ret;
}

/**
 * @brief Runs the STREAM benchmark.
 *
 * @details This function runs the STREAM benchmark using the specified kernel and number of threads per block.
 * It prepares the necessary arguments and configurations for the benchmark execution.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments for the
 benchmark.
 * @param[in] kernel The kernel function to be used for the benchmark.
 * @param[in] num_threads_per_block The number of threads per block to be used in the kernel execution.
 *
 * @return int The status code indicating success or failure of the benchmark execution.
 */
template <typename T>
int GpuStream::RunStreamKernel(std::unique_ptr<BenchArgs<T>> &args, Kernel kernel, int num_threads_per_block) {

    cudaError_t cuda_err = cudaSuccess;
    uint64_t num_thread_blocks;
    int size_factor = 2;

    // Validate data size
    uint64_t num_elements_in_thread_block = kNumLoopUnroll * num_threads_per_block;
    uint64_t num_bytes_in_thread_block = num_elements_in_thread_block * sizeof(T);
    if (args->size % num_bytes_in_thread_block) {
        std::cerr << "RunCopy: Data size should be multiple of " << num_bytes_in_thread_block << std::endl;
        return -1;
    }
    num_thread_blocks = args->size / num_bytes_in_thread_block;

    args->sub.times_in_ms.resize(static_cast<int>(Kernel::kCount));

    if (SetGpu(args->gpu_id)) {
        return -1;
    }

    // Launch jobs and collect running time
    for (int i = 0; i < args->num_loops + args->num_warm_up; i++) {

        // Record start event once warm up iterations are done
        if (i == args->num_warm_up) {
            cuda_err = cudaEventRecord(args->sub.start_event, args->sub.stream);
            if (cuda_err != cudaSuccess) {
                std::cerr << "RunStreamKernel::cudaEventRecord error: " << cuda_err << std::endl;
                return -1;
            }
        }

        switch (kernel) {
        case Kernel::kCopy:
            CopyKernel<<<num_thread_blocks, num_threads_per_block, 0, args->sub.stream>>>(
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[2].get()),
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[0].get()));
            args->sub.kernel_name = "COPY";
            break;
        case Kernel::kScale:
            ScaleKernel<<<num_thread_blocks, num_threads_per_block, 0, args->sub.stream>>>(
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[2].get()),
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[0].get()), scalar);
            args->sub.kernel_name = "SCALE";
            break;
        case Kernel::kAdd:
            AddKernel<<<num_thread_blocks, num_threads_per_block, 0, args->sub.stream>>>(
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[2].get()),
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[0].get()),
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[1].get()));
            size_factor = 3;
            args->sub.kernel_name = "ADD";
            break;
        case Kernel::kTriad:
            TriadKernel<<<num_thread_blocks, num_threads_per_block, 0, args->sub.stream>>>(
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[2].get()),
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[0].get()),
                reinterpret_cast<T *>(args->sub.gpu_buf_ptrs[1].get()), scalar);
            size_factor = 3;
            args->sub.kernel_name = "TRIAD";
            break;
        default:
            std::cerr << "RunStreamKernel::Invalid kernel: " << std::endl;
            break;
        }

        // Record end event at the end of iterations
        if (i + 1 == args->num_loops + args->num_warm_up) {
            cuda_err = cudaEventRecord(args->sub.end_event, args->sub.stream);
            if (cuda_err != cudaSuccess) {
                std::cerr << "RunStreamKernel::cudaEventRecord error: " << cuda_err << std::endl;
                return -1;
            }
        }
    }

    // Wait for the stream to finish
    cuda_err = cudaStreamSynchronize(args->sub.stream);
    if (cuda_err != cudaSuccess) {
        std::cerr << "RunStreamKernel::cudaStreamSynchronize error: " << cuda_err << std::endl;
        return -1;
    }

    // Calculate time
    float time_in_ms = 0;
    cuda_err = cudaEventElapsedTime(&time_in_ms, args->sub.start_event, args->sub.end_event);
    if (cuda_err != cudaSuccess) {
        std::cerr << "RunStreamKernel::cudaEventElapsedTime error: " << cuda_err << std::endl;
        return -1;
    }

    args->sub.times_in_ms[static_cast<int>(kernel)].push_back(time_in_ms / size_factor);

    return 0;
}

float GpuStream::GetActualMemoryClockRate(int gpu_id) {
    nvmlReturn_t result;
    nvmlDevice_t device;
    unsigned int clock_mhz;
    
    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 0.0f;
    }
    
    // Get device handle
    result = nvmlDeviceGetHandleByIndex(gpu_id, &device);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get device handle: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 0.0f;
    }
    
    // Get memory clock rate
    result = nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clock_mhz);
    if (result != NVML_SUCCESS) {
        std::cerr << "Failed to get memory clock: " << nvmlErrorString(result) << std::endl;
        nvmlShutdown();
        return 0.0f;
    }
    
    nvmlShutdown();
    return static_cast<float>(clock_mhz);
}

/**
 * @brief Runs the benchmark for various kernels and processes the results for a BenchArgs config.
 *
 * @details This function prepares the necessary buffers and streams, runs the benchmark for each kernel
 * with different thread per block configurations, checks the results, and processes the benchmark results.
 * It also handles cleanup of resources in case of errors.
 *
 * @param[in,out] args A unique pointer to a BenchArgs structure containing the necessary arguments for the
 benchmark.
 *
 * @return int The status code indicating success or failure of the benchmark execution.
 * */
template <typename T> int GpuStream::RunStream(std::unique_ptr<BenchArgs<T>> &args, const std::string &data_type, float peak_bw) {
    int ret = 0;
    ret = PrepareBufAndStream<T>(args);

    if (ret != 0) {
        return DestroyBufAndStream(args);
    }

    ret = PrepareEvent(args);
    if (ret != 0) {
        return DestroyEvent(args);
    }

    // benchmark over the kThreadsPerBlock array
    for (const int num_threads_in_block : kThreadsPerBlock) {
        // run the stream benchmark over the stream kernels
        for (int i = 0; i < static_cast<int>(Kernel::kCount); ++i) {
            Kernel kernel = static_cast<Kernel>(i);
            int ret = RunStreamKernel<T>(args, kernel, num_threads_in_block);
            if (ret == 0 && args->check_data) {
                // Compare buffer based on the kernel
                ret = CheckBuf(args, i);
            }
        }
    }

    // output formatted results to stdout
    // Tags are of format:
    // STREAM_<Kernelname>_datatype_gpu_<gpu_id>_buffer_<buffer_size>_block_<block_size>
    for (int i = 0; i < args->sub.times_in_ms.size(); i++) {
        std::string tag = "STREAM_" + KernelToString(i) + "_" + data_type + "_gpu_" + std::to_string(args->gpu_id) +
                          "_buffer_" + std::to_string(args->size);
        for (int j = 0; j < args->sub.times_in_ms[i].size(); j++) {
            // Calculate and display bandwidth
            double bw = args->size * args->num_loops / args->sub.times_in_ms[i][j] / 1e6;
            std::cout << tag << "_block_" << kThreadsPerBlock[j] << "\t" << bw << "\t";
            
            if (peak_bw < 0) { // cannot get peak_bw -> prints -1 for efficiency
                std::cout << "-1" << std::endl;
            } else {
                std::cout << std::fixed << std::setprecision(2) << bw / peak_bw * 100 << std::endl;
            }
        }
    }
    // cleanup buffer and streams for the curr arg
    Destroy(args);

    return ret;
}

/**
 * @brief Runs the Stream benchmark.
 *
 * @details This function processes the input args, validates and composes the BenchArgs structure for the
 availavble
 * GPUs, and runs the benchmark.
 *
 * @return int The status code indicating success or failure of the benchmark execution.
 * */
int GpuStream::Run() {

    int ret = 0;
    int gpu_count = 0;

    // Get number of NUMA nodes
    if (numa_available()) {
        std::cerr << "main::numa_available error" << std::endl;
        return -1;
    }

    // Get number of GPUs
    ret = GetGpuCount(&gpu_count);
    if (ret != 0) {
        return ret;
    }

    // find all GPUs and compose the Benchmarking data structure
    for (int j = 0; j < gpu_count; j++) {
        auto args = std::make_unique<BenchArgs<double>>();
        args->numa_id = 0;
        args->gpu_id = j;
        cudaGetDeviceProperties(&args->gpu_device_prop, j);

        args->num_warm_up = opts_.num_warm_up;
        args->num_loops = opts_.num_loops;
        args->size = opts_.size;
        args->check_data = opts_.check_data;
        args->numa_id = 0;
        args->gpu_id = j;

        // add data to vector
        bench_args_.emplace_back(std::move(args));
    }

    bool has_error = false;
    // Run the benchmark for all the configured data
    for (auto &variant_args : bench_args_) {
        std::visit(
            [&](auto &curr_args) {
                // Print device info and get the computed peak bandwidth
                float peak_bw = PrintCudaDeviceInfo(curr_args->gpu_id, curr_args->gpu_device_prop);
                
                // Set the NUMA node
                ret = numa_run_on_node(curr_args->numa_id);
                if (ret != 0) {
                    std::cerr << "Run::numa_run_on_node error: " << errno << std::endl;
                    has_error = true;
                    return;
                }

                // Run the stream benchmark for the configured data, passing the peak bandwidth
                if constexpr (std::is_same_v<std::decay_t<decltype(*curr_args)>, BenchArgs<float>>) {
                    ret = RunStream<float>(curr_args, "float", peak_bw);
                } else if constexpr (std::is_same_v<std::decay_t<decltype(*curr_args)>, BenchArgs<double>>) {
                    ret = RunStream<double>(curr_args, "double", peak_bw);
                } else {
                    std::cerr << "Run::Unknown type error" << std::endl;
                    has_error = true;
                    return;
                }

                if (ret != 0) {
                    std::cerr << "Run::RunStream error: " << errno << std::endl;
                    has_error = true;
                }
            },
            variant_args);
    }
    if (has_error) {
        return -1;
    }
    return ret;
}