// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <iomanip>
#include <tuple>

#include "cudnn_config.h"

namespace cudnn_test {
/**
 * @brief Generation of cudnn functions' params and run the benchmark of this function
 *
 * @tparam T1 the type of TensorDescriptor
 * @tparam T2 the type of ConvolutionDescriptor
 */
template <typename T1, typename T2> class CudnnFunction : public CudnnConfig {
  protected:
    cudnnHandle_t cudnn_handle;
    TensorDescriptorNd<T1> x_desc_;
    FilterDescriptorNd<T1> w_desc_;
    ConvolutionDescriptor<T2> conv_desc_;
    TensorDescriptorNd<T1> h_desc_;
    size_t fwd_workspace_size_;
    float *fwd_workspace_;
    T1 *x, *filter, *h;
    const float alpha_ = 1.f;
    const float beta_ = 0.f;

    /**
     * @brief Malloc cuda memory and fill in value for data params used in the cudnn function
     */
    void prepare_input();
    /**
     * @brief Generate some params used in the cudnn function
     */
    void prepare_for_function();
    /**
     * @brief Get and set convolution algorithm and workspace size used in cudnn convolution functions
     */
    virtual void get_workspace_size() {}
    /**
     * @brief launch the kernel/function
     */
    virtual void kernel_entry() {}
    /**
     * @brief Find the best algorithm for cudnn convolution functions
     */
    virtual void find_best_algo() {}

  public:
    /**
     * @brief Construct a new Cudnn Function object according to a CudnnConfig object, including initialization for
     * cudnn handle and curand
     *
     * @param config a CudnnConfig object including configuration of the params to the cudnn function
     */
    CudnnFunction(CudnnConfig &config) : CudnnConfig(config) {
        // Init cudnn handle and device
        cudnn_handle_init(&this->cudnn_handle);
    }
    /**
     * @brief Destroy the Cudnn Function object, including free cuda memory and handle of cudnn and curand
     */
    virtual ~CudnnFunction() {
        // free context and memory
        CUDA_SAFE_CALL(cudaFree(x));
        CUDA_SAFE_CALL(cudaFree(filter));
        CUDA_SAFE_CALL(cudaFree(h));
        cudnn_handle_free(&this->cudnn_handle);
    }
    /**
     * @brief The main procedure for cudnn function test, including warmup, function test and time measurement
     */
    void benchmark();
};

/**
 * @brief Generate some params used in the cudnn function
 */
template <typename T1, typename T2> void CudnnFunction<T1, T2>::prepare_for_function() {
    // Generate descriptor
    conv_desc_ =
        ConvolutionDescriptor<T2>(get_array_length(), get_padA(), get_filter_strideA(), get_dilationA(), get_mode());
    x_desc_ = TensorDescriptorNd<T1>(get_input_dims(), get_input_stride());
    w_desc_ = FilterDescriptorNd<T1>(get_filter_dims());
    h_desc_ = TensorDescriptorNd<T1>(get_output_dims(), get_output_stride());

    // Set Convolution MathType
    cudnnMathType_t algo = get_use_tensor_op() ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
    CHECK_CUDNN_ERROR(cudnnSetConvolutionMathType(conv_desc_.desc(), algo));
    if (this->auto_algo_) {
        find_best_algo();
    }
    // Set convolution algorithm and workspace size
    this->get_workspace_size();
    zeros<float>(&fwd_workspace_, std::vector<int>{static_cast<int>(this->fwd_workspace_size_ / sizeof(float)), 1});
}
/**
 * @brief Malloc cuda memory and fill in value for data params used in the cudnn function
 */
template <typename T1, typename T2> void CudnnFunction<T1, T2>::prepare_input() {
    // Allocate memory for filter data
    rand<T1>(&filter, get_filter_dims(), random_seed);
    // Allocate memory for input data
    rand<T1>(&x, get_input_dims(), random_seed);
    // Allocate memory for output data
    rand<T1>(&h, get_output_dims(), random_seed);
}
/**
 * @brief The main procedure for cudnn function test, including warmup, function test and time measurement
 */
template <typename T1, typename T2> void CudnnFunction<T1, T2>::benchmark() {
    // Prepare some Prerequisites for function running
    prepare_for_function();
    // Allocate memory and fill with data of input and output tensor
    prepare_input();

    // Warm up
    for (int i = 0; i < warm_up; ++i) {
        for (int j = 0; j < num_in_step; j++) {
            kernel_entry();
        }
    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    // Prepare some varibles for time measurement
    std::vector<float> iteration_time;
    // Benchmark in range of steps
    for (int i_ = 0; i_ < num_test; i_++) {
        // Collect time within each step, including #num_in_step times function invoking
        auto start = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < num_in_step; j++) {
            kernel_entry();
        }
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();

        // Convert step time to single function duration and update min and max duration
        float i = static_cast<float>(std::chrono::duration<double, std::milli>(end - start).count() / num_in_step);
        iteration_time.emplace_back(i);
    }

    // Output results
    std::cout << "[function config]: " << this->get_function_str() << std::endl;
    std::cout << "[raw_data]: ";
    for (int i = 0; i < iteration_time.size(); i++) {
        std::cout << iteration_time[i] << ",";
    }
    std::cout << std::endl;
}
} // namespace cudnn_test
