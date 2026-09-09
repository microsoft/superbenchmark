#include <cmath>
#include <cstdlib>

#include "cudnn_function_helper.h"

using namespace cudnn_test;

static json configuration() {
    return {{"name", "cudnnConvolutionBackwardFilter"},
            {"executionMode", "prepared"},
            {"inputDims", {2, 8, 5, 5}},
            {"inputStride", {200, 25, 5, 1}},
            {"filterDims", {8, 8, 3, 3}},
            {"outputDims", {2, 8, 5, 5}},
            {"outputStride", {200, 25, 5, 1}},
            {"inputType", 0},
            {"convType", 0},
            {"tensorOp", false},
            {"arrayLength", 2},
            {"padA", {1, 1}},
            {"filterStrideA", {1, 1}},
            {"dilationA", {1, 1}},
            {"mode", 1}};
}

static void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static void test_rejected_configs() {
    for (const auto &change : std::vector<json>{{{"name", "cudnnConvolutionForward"}},
                                                {{"inputStride", {201, 25, 5, 1}}},
                                                {{"convType", 2}},
                                                {{"algo", 1}},
                                                {{"workspaceLimitMiB", -1}},
                                                {{"executionMode", "unknown"}},
                                                {{"filterStrideA", {0, 1}}},
                                                {{"outputDims", {2, 8, 4, 5}}}}) {
        auto value = configuration();
        value.update(change);
        bool rejected = false;
        try {
            value.get<CudnnConfig>();
        } catch (const std::exception &) {
            rejected = true;
        }
        require(rejected, "invalid prepared configuration was accepted");
    }
}

template <typename Value> static float stored_float(Value value) { return static_cast<float>(value); }
template <> float stored_float<half>(half value) { return __half2float(value); }
template <typename Value> static Value stored_value(float value) { return static_cast<Value>(value); }
template <> half stored_value<half>(float value) { return __float2half(value); }

template <typename Value> static void test_execution(cudnnDataType_t type, unsigned seed) {
    auto value = configuration();
    value["inputType"] = static_cast<int>(type);
    value["tensorOp"] = type == CUDNN_DATA_HALF;
    CudnnConfig config = value.get<CudnnConfig>();
    struct Resources {
        cudnnHandle_t handle = nullptr;
        Value *input = nullptr;
        Value *gradient = nullptr;
        Value *filter = nullptr;
        ~Resources() {
            cudaFree(input);
            cudaFree(gradient);
            cudaFree(filter);
            if (handle) {
                cudnnDestroy(handle);
            }
        }
    } resources;
    std::vector<Value> input(400), gradient(400), result(576), first(576);
    std::srand(seed);
    for (size_t index = 0; index < input.size(); ++index) {
        input[index] = stored_value<Value>(2.f * std::rand() / RAND_MAX - 1.f);
        gradient[index] = stored_value<Value>(2.f * std::rand() / RAND_MAX - 1.f);
    }
    CHECK_CUDNN_ERROR(cudnnCreate(&resources.handle));
    CUDA_SAFE_CALL(cudaMalloc(&resources.input, input.size() * sizeof(Value)));
    CUDA_SAFE_CALL(cudaMalloc(&resources.gradient, gradient.size() * sizeof(Value)));
    CUDA_SAFE_CALL(cudaMalloc(&resources.filter, result.size() * sizeof(Value)));
    CUDA_SAFE_CALL(cudaMemcpy(resources.input, input.data(), input.size() * sizeof(Value), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(resources.gradient, gradient.data(), gradient.size() * sizeof(Value), cudaMemcpyHostToDevice));
    CudnnPreparedPlan plan(resources.handle, config, resources.input, resources.filter, resources.gradient);
    const auto identity = plan.json();
    for (int repeat = 0; repeat < 2; ++repeat) {
        CUDA_SAFE_CALL(cudaMemset(resources.filter, 0xff, result.size() * sizeof(Value)));
        for (int call = 0; call < 3; ++call) {
            plan.execute(resources.handle);
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(result.data(), resources.filter, result.size() * sizeof(Value), cudaMemcpyDeviceToHost));
        for (int channel_out = 0; channel_out < 8; ++channel_out) {
            for (int channel_in = 0; channel_in < 8; ++channel_in) {
                for (int kernel_y = 0; kernel_y < 3; ++kernel_y) {
                    for (int kernel_x = 0; kernel_x < 3; ++kernel_x) {
                        double reference = 0;
                        for (int batch = 0; batch < 2; ++batch) {
                            for (int output_y = 0; output_y < 5; ++output_y) {
                                for (int output_x = 0; output_x < 5; ++output_x) {
                                    int input_y = output_y + kernel_y - 1;
                                    int input_x = output_x + kernel_x - 1;
                                    if (input_y >= 0 && input_y < 5 && input_x >= 0 && input_x < 5) {
                                        reference +=
                                            static_cast<double>(stored_float(
                                                input[((batch * 8 + channel_in) * 5 + input_y) * 5 + input_x])) *
                                            stored_float(
                                                gradient[((batch * 8 + channel_out) * 5 + output_y) * 5 + output_x]);
                                    }
                                }
                            }
                        }
                        size_t index = ((channel_out * 8 + channel_in) * 3 + kernel_y) * 3 + kernel_x;
                        double actual = stored_float(result[index]);
                        double tolerance =
                            0.0005 + (0.0005 + (type == CUDNN_DATA_HALF ? 1.0 / 2048 : 0)) * std::abs(reference);
                        require(std::isfinite(actual) && std::abs(actual - reference) <= tolerance,
                                "prepared output differs from full stored-input CPU reference");
                        if (repeat != 0) {
                            require(actual == stored_float(first[index]), "prepared output is not repeatable");
                        }
                    }
                }
            }
        }
        first = result;
    }
    require(identity == plan.json(), "prepared plan identity changed during reuse");
}

int main() {
    try {
        test_rejected_configs();
        for (unsigned seed : {33931u, 58613u}) {
            test_execution<float>(CUDNN_DATA_FLOAT, seed);
            test_execution<half>(CUDNN_DATA_HALF, seed);
        }
        std::cout << "Prepared configuration, plan reuse and full CPU-reference checks passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}