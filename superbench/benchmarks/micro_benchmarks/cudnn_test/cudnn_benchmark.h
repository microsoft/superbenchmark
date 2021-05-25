// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <chrono>
#include <iomanip>
#include <tuple>
#include <unordered_map>

#include "cudnn_helper.h"

namespace cudnn_test {
enum cudnn_function_name_enum {
    e_cudnnConvolutionForward = 0,
    e_cudnnConvolutionBackwardData,
    e_cudnnConvolutionBackwardFilter,
};

static std::unordered_map<std::string, cudnn_function_name_enum> const cudnn_function_name_string = {
    {"cudnnConvolutionForward", cudnn_function_name_enum::e_cudnnConvolutionForward},
    {"cudnnConvolutionBackwardData", cudnn_function_name_enum::e_cudnnConvolutionBackwardData},
    {"cudnnConvolutionBackwardFilter", cudnn_function_name_enum::e_cudnnConvolutionBackwardFilter},

};

// Store cudnn functions params configuration
class CudnnConfig {
  protected:
    int num_test;
    int warm_up;
    int num_in_step;
    std::string name;
    cudnn_function_name_enum e_name;
    std::vector<int> input_dims_;
    std::vector<int> input_stride_;
    std::vector<int> filter_dims_;
    std::vector<int> output_dims_;
    std::vector<int> output_stride_;
    int algo_;
    int arrayLength_;
    std::vector<int> padA_;
    std::vector<int> filterStrideA_;
    std::vector<int> dilationA_;
    cudnnConvolutionMode_t mode_;
    bool use_tensor_core_;
    cudnnDataType_t input_type_;
    cudnnDataType_t conv_type_;
    std::string to_str_;

  public:
    void set_num_test(int num_test) { this->num_test = num_test; }
    void set_warm_up(int warm_up) { this->warm_up = warm_up; }
    void set_num_in_step(int num_in_step) { this->num_in_step = num_in_step; }
    void set_name(std::string &n) { name = n; }
    void set_input_dims(std::vector<int> &input_dims) { input_dims_ = input_dims; }
    void set_input_stride(std::vector<int> &input_stride) { input_stride_ = input_stride; }
    void set_filter_dims(std::vector<int> &filter_dims) { filter_dims_ = filter_dims; }
    void set_output_dims(std::vector<int> &output_dims) { output_dims_ = output_dims; }
    void set_output_stride(std::vector<int> &output_stride) { output_stride_ = output_stride; }
    void set_algo(int algo) { algo_ = algo; }
    void set_arrayLength(int arrayLength) { arrayLength_ = arrayLength; }
    void set_padA(std::vector<int> &padA) { padA_ = padA; }
    void set_filterStrideA(std::vector<int> &filterStrideA) { filterStrideA_ = filterStrideA; }
    void set_dilationA(std::vector<int> &dilationA) { dilationA_ = dilationA; }
    void set_mode(cudnnConvolutionMode_t &mode) { mode_ = mode; }
    void set_use_tensor_core(bool use_tensor_core) { use_tensor_core_ = use_tensor_core; }
    void set_input_type(cudnnDataType_t &input_type) { input_type_ = input_type; }
    void set_conv_type(cudnnDataType_t &conv_type) { input_type_ = conv_type; }
    void set_str(std::string &str) { to_str_ = str; }

    std::string &get_name() { return name; }
    cudnn_function_name_enum &get_e_name() { return e_name; }
    std::vector<int> &get_input_dims() { return input_dims_; }
    std::vector<int> &get_input_stride() { return input_stride_; }
    std::vector<int> &get_filter_dims() { return filter_dims_; }
    std::vector<int> &get_output_dims() { return output_dims_; }
    std::vector<int> &get_output_stride() { return output_stride_; }
    int get_algo() { return algo_; }
    int get_arrayLength() { return arrayLength_; }
    std::vector<int> &get_padA() { return padA_; }
    std::vector<int> &get_filterStrideA() { return filterStrideA_; }
    std::vector<int> &get_dilationA() { return dilationA_; }
    cudnnConvolutionMode_t &get_mode() { return mode_; }
    bool get_use_tensor_core() { return use_tensor_core_; }
    cudnnDataType_t &get_input_type() { return input_type_; }
    cudnnDataType_t &get_conv_type() { return input_type_; }
    std::string &get_str() { return to_str_; }

    // convert function name to enum type
    cudnn_function_name_enum name2enum() {
        auto it = cudnn_function_name_string.find(this->name);
        if (it != cudnn_function_name_string.end()) {
            this->e_name = it->second;
            return e_name;
        } else {
            throw "ERROR: invalid input function name";
        }
    }
};

// Generation of cudnn functions' params and benchmark
template <typename T1, typename T2> class CudnnFunction : public CudnnConfig {
  protected:
    cudnnHandle_t cudnn_handle;
    curandGenerator_t curand_gen;
    TensorDescriptorNd<T1> x_desc_;
    FilterDescriptorNd<T1> w_desc_;
    ConvolutionDescriptor<T2> conv_desc_;
    TensorDescriptorNd<T1> h_desc_;

    size_t fwd_workspace_size_;
    float *fwd_workspace_;

    T1 *x, *filter, *h;

    const float alpha_ = 1.f;
    const float beta_ = 0.f;

    std::vector<int> get_output_dims() { return output_dims_; }
    // Generate some params used in the cudnn function
    void prepare_for_function();
    // Set convolution algorithm and workspace size used in cudnn convolution functions
    virtual void get_workspace_size() {}
    // Run a cudnn function
    virtual void kernel_entry() {}
    // Malloc cuda memory and fill in value for data params used in the cudnn function
    void prepare_input(curandGenerator_t curand_gen);

  public:
    CudnnFunction() {}
    CudnnFunction(CudnnConfig &config) : CudnnConfig(config) {
        // Init cudnn handle and device
        cuda_init(&this->cudnn_handle);
        // Create curandGenerator for cuda data random generation
        curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);
    }
    void benchmark();
    virtual ~CudnnFunction() {
        // free context and memory
        CUDA_SAFE_CALL(cudaFree(x));
        CUDA_SAFE_CALL(cudaFree(filter));
        CUDA_SAFE_CALL(cudaFree(h));
        curandDestroyGenerator(curand_gen);
        cuda_free(&this->cudnn_handle);
    }
};

template <typename T1, typename T2> void CudnnFunction<T1, T2>::prepare_for_function() {
    // Generate descriptor
    conv_desc_ =
        ConvolutionDescriptor<T2>(get_arrayLength(), get_padA(), get_filterStrideA(), get_dilationA(), get_mode());
    x_desc_ = TensorDescriptorNd<T1>(get_input_dims(), get_input_stride());
    w_desc_ = FilterDescriptorNd<T1>(get_filter_dims());
    h_desc_ = TensorDescriptorNd<T1>(get_output_dims(), get_output_stride());

    // Set Convolution MathType
    cudnnMathType_t algo = get_use_tensor_core() ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
    CHECK_CUDNN_ERROR(cudnnSetConvolutionMathType(conv_desc_.desc(), algo));
    // Set convolution algorithm and workspace size
    this->get_workspace_size();
    zeros<float>(&fwd_workspace_, std::vector<int>{static_cast<int>(fwd_workspace_size_ / sizeof(float)), 1});
}

template <typename T1, typename T2> void CudnnFunction<T1, T2>::prepare_input(curandGenerator_t curand_gen) {
    // Allocate memory for filter data
    rand<T1>(&filter, get_filter_dims(), curand_gen);

    // Allocate memory for input data
    rand<T1>(&x, get_input_dims(), curand_gen);

    // Allocate memory for output data
    zeros<T1>(&h, get_output_dims());
}

template <typename T1, typename T2> void CudnnFunction<T1, T2>::benchmark() {
    // Prepare some Prerequisites for function running
    prepare_for_function();
    // Allocate memory and fill with data of input and output tensor
    prepare_input(curand_gen);

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
        // Collect time within each step, including #repeat_in_one_step times function invoking
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
    std::cout << "[function config]: " << this->get_str() << std::endl;
    std::cout << "[raw_data]: ";
    for (int i = 0; i < iteration_time.size(); i++) {
        std::cout << iteration_time[i] << ",";
    }
    std::cout << std::endl;
}
}; // namespace cudnn_test