// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

// Benchmark of cudnn functions

#include "cudnn_benchmark.h"

namespace cudnn_test {

void cuda_init() {
    CUDA_SAFE_CALL(cudaDeviceReset());
    CUDA_SAFE_CALL(cudaSetDevice(0));
    CHECK_CUDNN_ERROR(cudnnCreate(&cudnn_handle_0));
}

void cuda_free() {
    CHECK_CUDNN_ERROR(cudnnDestroy(cudnn_handle_0));
    CUDA_SAFE_CALL(cudaDeviceReset());
}

template <typename T> void rand(T **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    zeros(input, dims_);
}

template <> void rand(float **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());

    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(float) * size_));

    curandGenerateUniform(curand_gen, *input, size_);
}

template <> void rand(double **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());

    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(double) * size_));

    curandGenerateUniformDouble(curand_gen, *input, size_);
}

template <> void rand(half **input, std::vector<int> dims_, curandGenerator_t curand_gen) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(half) * size_));
    half *host_input;
    CUDA_SAFE_CALL(cudaMallocHost(&host_input, sizeof(half) * size_));
    for (int i = 0; i < size_; i++) {
        host_input[i] = __float2half((float)std::rand() / (float)(RAND_MAX));
    }
    CUDA_SAFE_CALL(cudaMemcpy(host_input, *input, sizeof(half) * size_, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaFreeHost(host_input));
}

// helper function to convert json to cublasfunction
void from_json(const json &j, CudnnConfig &fn) {
    auto name = j.at("name").get<std::string>();
    fn.set_name(name);
    auto arrayLength = j.at("arrayLength").get<int>();
    fn.set_arrayLength(arrayLength);
    auto input_dims = j.at("input_dims").get<std::vector<int>>();
    fn.set_input_dims(input_dims);
    auto input_stride = j.at("input_stride").get<std::vector<int>>();
    fn.set_input_stride(input_stride);
    auto filter_dims = j.at("filter_dims").get<std::vector<int>>();
    fn.set_filter_dims(filter_dims);
    auto output_dims = j.at("output_dims").get<std::vector<int>>();
    fn.set_output_dims(output_dims);
    auto output_stride = j.at("output_stride").get<std::vector<int>>();
    fn.set_output_stride(output_stride);
    auto algo = j.at("algo").get<int>();
    fn.set_algo(algo);
    auto padA = j.at("padA").get<std::vector<int>>();
    fn.set_padA(padA);
    auto filterStrideA = j.at("filterStrideA").get<std::vector<int>>();
    fn.set_filterStrideA(filterStrideA);
    auto dilationA = j.at("dilationA").get<std::vector<int>>();
    fn.set_dilationA(dilationA);
    auto mode = j.at("mode").get<cudnnConvolutionMode_t>();
    fn.set_mode(mode);
    auto use_tensor_core = j.at("use_tensor_core").get<bool>();
    fn.set_use_tensor_core(use_tensor_core);
    auto input_type = j.at("input_type").get<cudnnDataType_t>();
    fn.set_input_type(input_type);
    auto conv_type = j.at("conv_type").get<cudnnDataType_t>();
    fn.set_conv_type(conv_type);
    auto str = j.dump();
    std::replace(str.begin(), str.end(), '\"', ' ');
    fn.set_str(str);
    fn.name2enum();
}
} // namespace cudnn_test