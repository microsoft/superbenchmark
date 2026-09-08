#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "cudnn_config.h"
#include "cudnn_function.h"

using cudnn_test::check_cuda;
using cudnn_test::throw_cudnn_err;

void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

void test_independent_types() {
    cudnn_test::CudnnConfig config{};
    config.set_input_type(CUDNN_DATA_HALF);
    config.set_conv_type(CUDNN_DATA_FLOAT);
    require(config.get_input_type() == CUDNN_DATA_HALF, "compute type overwrote half storage type");
    require(config.get_conv_type() == CUDNN_DATA_FLOAT, "compute type was not stored independently");
    config.set_conv_type(CUDNN_DATA_HALF);
    config.set_input_type(CUDNN_DATA_FLOAT);
    require(config.get_conv_type() == CUDNN_DATA_HALF, "storage type overwrote compute type");
    require(config.get_input_type() == CUDNN_DATA_FLOAT, "storage type did not change");
}

void test_half_initialization() {
    constexpr int count = 257;
    constexpr int seed = 33931;
    std::vector<float> expected(count);
    std::srand(seed);
    for (auto &value : expected)
        value = __half2float(__float2half(static_cast<float>(std::rand()) / RAND_MAX));
    std::vector<float> first;
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::srand(seed + repeat + 1);
        half *device = nullptr;
        try {
            cudnn_test::rand(&device, {count}, seed);
            std::vector<half> host(count);
            CUDA_SAFE_CALL(cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost));
            std::vector<float> actual(count);
            for (int index = 0; index < count; ++index)
                actual[index] = __half2float(host[index]);
            require(actual == expected, "half data does not match the supplied seed and host-to-device copy");
            if (repeat == 0)
                first = actual;
            else
                require(actual == first, "half initialization depends on ambient RNG state");
            CUDA_SAFE_CALL(cudaFree(device));
        } catch (...) {
            if (device != nullptr)
                cudaFree(device);
            throw;
        }
    }
}

void test_descriptor_types() {
    cudnn_test::TensorDescriptorNd<half> input({1, 2, 3, 3}, {18, 9, 3, 1});
    cudnn_test::FilterDescriptorNd<half> filter({2, 2, 1, 1});
    cudnn_test::ConvolutionDescriptor<float> convolution(2, {0, 0}, {1, 1}, {1, 1}, CUDNN_CROSS_CORRELATION);
    cudnnDataType_t storage, filter_type, compute;
    cudnnTensorFormat_t format;
    cudnnConvolutionMode_t mode;
    int rank, dims[4], strides[4], padding[2], dilation[2], conv_stride[2];
    CHECK_CUDNN_ERROR(cudnnGetTensorNdDescriptor(input.desc(), 4, &storage, &rank, dims, strides));
    CHECK_CUDNN_ERROR(cudnnGetFilterNdDescriptor(filter.desc(), 4, &filter_type, &format, &rank, dims));
    CHECK_CUDNN_ERROR(
        cudnnGetConvolutionNdDescriptor(convolution.desc(), 2, &rank, padding, conv_stride, dilation, &mode, &compute));
    require(storage == CUDNN_DATA_HALF && filter_type == CUDNN_DATA_HALF && compute == CUDNN_DATA_FLOAT,
            "actual cuDNN descriptors do not preserve half storage with float compute");
}

cudnn_test::CudnnConfig workspace_config() {
    cudnn_test::CudnnConfig config{};
    config.set_input_dims({1, 2, 3, 3});
    config.set_input_stride({18, 9, 3, 1});
    config.set_output_dims({1, 2, 3, 3});
    config.set_output_stride({18, 9, 3, 1});
    config.set_filter_dims({2, 2, 1, 1});
    config.set_array_length(2);
    config.set_padA({0, 0});
    config.set_filter_strideA({1, 1});
    config.set_dilationA({1, 1});
    config.set_mode(CUDNN_CROSS_CORRELATION);
    config.set_use_tensor_op(false);
    config.set_auto_algo(false);
    return config;
}

class WorkspaceProbe : public cudnn_test::CudnnFunction<float, float> {
    size_t requested_bytes_;

    void get_workspace_size() override { fwd_workspace_size_ = requested_bytes_; }

  public:
    WorkspaceProbe(cudnn_test::CudnnConfig &config, size_t requested_bytes)
        : CudnnFunction(config), requested_bytes_(requested_bytes) {
        x = nullptr;
        filter = nullptr;
        h = nullptr;
    }

    void prepare() { prepare_for_function(); }
    void prepare_data() { prepare_input(); }
    CUdeviceptr input() const { return reinterpret_cast<CUdeviceptr>(x); }
    CUdeviceptr workspace() const { return reinterpret_cast<CUdeviceptr>(fwd_workspace_); }
};

void check_workspace_size(size_t bytes) {
    auto config = workspace_config();
    WorkspaceProbe function(config, bytes);
    function.prepare();
    CUdeviceptr allocation;
    size_t allocated_bytes = 0;
    require(cuMemGetAddressRange(&allocation, &allocated_bytes, function.workspace()) == CUDA_SUCCESS,
            "workspace does not have a valid device allocation");
    require(allocated_bytes >= bytes, "workspace was rounded down below requested bytes");
}

void test_workspace_byte_size() { check_workspace_size(13); }
void test_single_byte_workspace() { check_workspace_size(1); }

void test_workspace_released() {
    auto config = workspace_config();
    CUdeviceptr pointer;
    {
        WorkspaceProbe function(config, 4096);
        function.prepare();
        pointer = function.workspace();
    }
    CUdeviceptr allocation;
    size_t bytes;
    require(cuMemGetAddressRange(&allocation, &bytes, pointer) != CUDA_SUCCESS,
            "workspace allocation remains live after destruction");
}

void test_unprepared_destruction() {
    auto config = workspace_config();
    cudnn_test::CudnnFunction<float, float> function(config);
}

void test_strided_allocation() {
    auto config = workspace_config();
    config.set_input_stride({40, 20, 5, 1});
    config.set_random_seed(33931);
    WorkspaceProbe function(config, 0);
    function.prepare_data();
    CUdeviceptr allocation;
    size_t bytes = 0;
    require(cuMemGetAddressRange(&allocation, &bytes, function.input()) == CUDA_SUCCESS,
            "strided input allocation is missing");
    require(bytes >= 33 * sizeof(float), "input allocation does not cover the descriptor stride extent");
}

void test_invalid_shape_rejected() {
    bool rejected = false;
    try {
        cudnn_test::TensorDescriptorNd<float> descriptor({1, 2, 3, 3}, {18});
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "mismatched stride rank was passed to cuDNN");
    rejected = false;
    try {
        cudnn_test::ConvolutionDescriptor<float> descriptor(2, {0}, {1, 1}, {1, 1}, CUDNN_CROSS_CORRELATION);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "mismatched convolution arrays were passed to cuDNN");
}

int main(int argc, char **argv) {
    const bool memcheck = argc == 2 && std::string(argv[1]) == "--memcheck";
    if (argc > 1 && !memcheck)
        return 2;
    std::cout.setf(std::ios::unitbuf);
    int failures = 0;
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        {"independent_types", test_independent_types},
        {"half_initialization", test_half_initialization},
        {"descriptor_types", test_descriptor_types},
        {"workspace_byte_size", test_workspace_byte_size},
        {"single_byte_workspace", test_single_byte_workspace},
        {"workspace_released", test_workspace_released},
        {"unprepared_destruction", test_unprepared_destruction},
        {"strided_allocation", test_strided_allocation},
        {"invalid_shape_rejected", test_invalid_shape_rejected},
    };
    for (const auto &test : tests) {
        if (memcheck && std::string(test.first) == "workspace_released") {
            std::cout << test.first << ": skipped negative driver query; use sanitizer leak check\n";
            continue;
        }
        try {
            test.second();
            std::cout << test.first << ": passed\n";
        } catch (const std::exception &error) {
            ++failures;
            std::cout << test.first << ": failed: " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}