#include <cstdlib>
#include <exception>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "cudnn_config.h"

using cudnn_test::check_cuda;

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
    for (int repeat = 0; repeat < 2; ++repeat) {
        std::srand(seed + repeat + 1);
        half *device = nullptr;
        try {
            cudnn_test::rand(&device, std::vector<int>{count}, seed);
            std::vector<half> host(count);
            CUDA_SAFE_CALL(cudaMemcpy(host.data(), device, count * sizeof(half), cudaMemcpyDeviceToHost));
            std::vector<float> actual(count);
            for (int index = 0; index < count; ++index)
                actual[index] = __half2float(host[index]);
            require(actual == expected, "half data does not match the supplied seed and host-to-device copy");
            CUDA_SAFE_CALL(cudaFree(device));
        } catch (...) {
            if (device != nullptr)
                cudaFree(device);
            throw;
        }
    }
}

int main() {
    int failures = 0;
    const std::vector<std::pair<const char *, void (*)()>> tests = {
        {"independent_types", test_independent_types},
        {"half_initialization", test_half_initialization},
    };
    for (const auto &test : tests) {
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