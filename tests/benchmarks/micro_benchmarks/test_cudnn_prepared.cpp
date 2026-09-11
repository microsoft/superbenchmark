#include <cmath>
#include <cstdlib>
#include <cstring>

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

template <typename Value>
static void test_execution(cudnnDataType_t type, unsigned seed, json value = configuration(),
                           bool scalar_check = true) {
    value["inputType"] = static_cast<int>(type);
    value["tensorOp"] = type == CUDNN_DATA_HALF;
    CudnnConfig config = value.get<CudnnConfig>();
    if (config.get_input_type() != type) {
        std::cout << "Requested storage " << type << " resolves to " << config.get_input_type()
                  << "; skipping unavailable storage test" << std::endl;
        return;
    }
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
    const auto &input_dims = config.get_input_dims();
    const auto &output_dims = config.get_output_dims();
    const auto &filter_dims = config.get_filter_dims();
    auto count = [](const std::vector<int> &dimensions) {
        return std::accumulate(dimensions.begin(), dimensions.end(), size_t{1}, std::multiplies<size_t>());
    };
    std::vector<Value> input(count(input_dims)), gradient(count(output_dims)), result(count(filter_dims)),
        first(result.size());
    std::srand(seed);
    for (auto values : {&input, &gradient}) {
        for (auto &value : *values) {
            value = stored_value<Value>(2.f * std::rand() / RAND_MAX - 1.f);
        }
    }
    CHECK_CUDNN_ERROR(cudnnCreate(&resources.handle));
    CUDA_SAFE_CALL(cudaMalloc(&resources.input, input.size() * sizeof(Value)));
    CUDA_SAFE_CALL(cudaMalloc(&resources.gradient, gradient.size() * sizeof(Value)));
    CUDA_SAFE_CALL(cudaMalloc(&resources.filter, result.size() * sizeof(Value)));
    CUDA_SAFE_CALL(cudaMemcpy(resources.input, input.data(), input.size() * sizeof(Value), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(
        cudaMemcpy(resources.gradient, gradient.data(), gradient.size() * sizeof(Value), cudaMemcpyHostToDevice));
    CudnnPreparedPlan plan(resources.handle, config, resources.input, resources.filter, resources.gradient);
    require(plan.verification().at("passed").get<bool>(), "plan lacks a passing numerical screen");
    require(plan.verification().at("checked_elements").get<size_t>() == result.size() * 5,
            "plan was not checked over every output for all screening inputs");
    const auto identity = plan.json();
    for (int repeat = 0; repeat < 2; ++repeat) {
        CUDA_SAFE_CALL(cudaMemset(resources.filter, 0xff, result.size() * sizeof(Value)));
        for (int call = 0; call < 3; ++call) {
            plan.execute(resources.handle);
        }
        CUDA_SAFE_CALL(
            cudaMemcpy(result.data(), resources.filter, result.size() * sizeof(Value), cudaMemcpyDeviceToHost));
        for (int channel_out = 0; channel_out < filter_dims[0]; ++channel_out) {
            for (int channel_in = 0; channel_in < filter_dims[1]; ++channel_in) {
                for (int kernel_y = 0; kernel_y < filter_dims[2]; ++kernel_y) {
                    for (int kernel_x = 0; kernel_x < filter_dims[3]; ++kernel_x) {
                        double reference = 0;
                        for (int batch = 0; scalar_check && batch < input_dims[0]; ++batch) {
                            for (int output_y = 0; output_y < output_dims[2]; ++output_y) {
                                for (int output_x = 0; output_x < output_dims[3]; ++output_x) {
                                    int input_y = output_y * config.get_filter_strideA()[0] +
                                                  kernel_y * config.get_dilationA()[0] - config.get_padA()[0];
                                    int input_x = output_x * config.get_filter_strideA()[1] +
                                                  kernel_x * config.get_dilationA()[1] - config.get_padA()[1];
                                    if (input_y >= 0 && input_y < input_dims[2] && input_x >= 0 &&
                                        input_x < input_dims[3]) {
                                        reference +=
                                            static_cast<double>(stored_float(
                                                input[((batch * input_dims[1] + channel_in) * input_dims[2] + input_y) *
                                                          input_dims[3] +
                                                      input_x])) *
                                            stored_float(
                                                gradient[((batch * output_dims[1] + channel_out) * output_dims[2] +
                                                          output_y) *
                                                             output_dims[3] +
                                                         output_x]);
                                    }
                                }
                            }
                        }
                        size_t index =
                            ((channel_out * filter_dims[1] + channel_in) * filter_dims[2] + kernel_y) * filter_dims[3] +
                            kernel_x;
                        double actual = stored_float(result[index]);
                        double tolerance =
                            0.0005 + (0.0005 + (type == CUDNN_DATA_HALF ? 1.0 / 2048 : 0)) * std::abs(reference);
                        require(std::isfinite(actual) && (!scalar_check || std::abs(actual - reference) <= tolerance),
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
    std::vector<Value> restored_input(input.size()), restored_gradient(gradient.size());
    CUDA_SAFE_CALL(
        cudaMemcpy(restored_input.data(), resources.input, input.size() * sizeof(Value), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(restored_gradient.data(), resources.gradient, gradient.size() * sizeof(Value),
                              cudaMemcpyDeviceToHost));
    require(std::memcmp(restored_input.data(), input.data(), input.size() * sizeof(Value)) == 0 &&
                std::memcmp(restored_gradient.data(), gradient.data(), gradient.size() * sizeof(Value)) == 0,
            "numerical screening did not restore actual benchmark operands");
    if (scalar_check && type == CUDNN_DATA_FLOAT && seed == 33931u) {
        CudnnReference reference(config, resources.input, resources.filter, resources.gradient);
        require(!reference.accepts([&]() {}), "numerical screen accepted a no-op");
        require(reference.failures() != 0, "numerical rejection did not retain failure count");
        CUDA_SAFE_CALL(cudaMemset(resources.input, 0xff, input.size() * sizeof(Value)));
        bool rejected = false;
        try {
            CudnnReference invalid(config, resources.input, resources.filter, resources.gradient);
        } catch (const std::runtime_error &error) {
            rejected = std::string(error.what()).find("FP64 reference disagrees") != std::string::npos;
        }
        require(rejected, "nonfinite benchmark input was accepted by the numerical screen");
    }
}

static void test_default_shapes() {
    for (const auto &shape : std::vector<std::vector<int>>{{128, 32, 3}, {256, 1024, 1}, {512, 512, 3}}) {
        auto value = configuration();
        value["inputDims"] = {32, shape[0], 14, 14};
        value["inputStride"] = {shape[0] * 196, 196, 14, 1};
        value["outputDims"] = {32, shape[1], 14, 14};
        value["outputStride"] = {shape[1] * 196, 196, 14, 1};
        value["filterDims"] = {shape[1], shape[0], shape[2], shape[2]};
        value["padA"] = {shape[2] / 2, shape[2] / 2};
        test_execution<float>(CUDNN_DATA_FLOAT, 104729u, value, false);
        test_execution<half>(CUDNN_DATA_HALF, 104729u, value, false);
    }
}

int main() {
#if CUDNN_VERSION < 8900
    std::cout << "Prepared execution requires cuDNN 8.9 or newer" << std::endl;
    return 77;
#else
    try {
        test_rejected_configs();
        for (unsigned seed : {33931u, 58613u}) {
            test_execution<float>(CUDNN_DATA_FLOAT, seed);
            test_execution<half>(CUDNN_DATA_HALF, seed);
        }
        auto strided = configuration();
        strided["inputDims"] = {2, 8, 7, 6};
        strided["inputStride"] = {336, 42, 6, 1};
        strided["outputDims"] = {2, 8, 4, 6};
        strided["outputStride"] = {192, 24, 6, 1};
        strided["filterDims"] = {8, 8, 2, 3};
        strided["filterStrideA"] = {2, 1};
        strided["dilationA"] = {2, 1};
        test_execution<float>(CUDNN_DATA_FLOAT, 33931u, strided);
        test_execution<half>(CUDNN_DATA_HALF, 33931u, strided);
        test_default_shapes();
        std::cout
            << "Prepared available-storage screening, scalar checks, rejection and operand restoration passed"
            << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
#endif
}