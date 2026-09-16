#include <iostream>

#include "cudnn_convolution_workload.h"

using namespace cudnn_test;

static void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static CudnnConfig configuration() {
    CudnnConfig config{};
    config.set_name("cudnnConvolutionBackwardFilter");
    config.set_input_dims({2, 8, 5, 5});
    config.set_input_stride({200, 25, 5, 1});
    config.set_filter_dims({8, 8, 3, 3});
    config.set_output_dims({2, 8, 5, 5});
    config.set_output_stride({200, 25, 5, 1});
    config.set_padA({1, 1});
    config.set_filter_strideA({1, 1});
    config.set_dilationA({1, 1});
    config.set_input_type(CUDNN_DATA_FLOAT);
    config.set_conv_type(CUDNN_DATA_FLOAT);
    config.set_mode(CUDNN_CROSS_CORRELATION);
    config.set_array_length(2);
    return config;
}

int main() {
    try {
        auto config = configuration();
        const CudnnConvolutionWorkload workload(config);
        workload.validate_backward_filter();
        require(workload.input_type == config.get_input_type() && workload.compute_type == config.get_conv_type(),
                "workload did not retain actual parsed precision");
        config.set_input_dims({1, 1, 1, 1});
        config.set_use_tensor_op(true);
        config.set_workspace_limit_mib(-1);
        config.set_algo(99);
        workload.validate_backward_filter();
        require(workload.input_dims == std::vector<int>{2, 8, 5, 5}, "workload changed with mutable configuration");

        for (int change = 0; change < 8; ++change) {
            auto invalid = configuration();
            switch (change) {
            case 0:
                invalid.set_name("cudnnConvolutionForward");
                break;
            case 1:
                invalid.set_input_stride({201, 25, 5, 1});
                break;
            case 2:
                invalid.set_output_dims({2, 8, 4, 5});
                break;
            case 3:
                invalid.set_filter_strideA({0, 1});
                break;
            case 4:
                invalid.set_dilationA({1});
                break;
            case 5:
                invalid.set_padA({-1, 1});
                break;
            case 6:
                invalid.set_filter_dims({8, 8, 0, 3});
                break;
            case 7:
                invalid.set_array_length(3);
                break;
            }
            bool rejected = false;
            try {
                CudnnConvolutionWorkload(invalid).validate_backward_filter();
            } catch (const std::invalid_argument &) {
                rejected = true;
            }
            require(rejected, "invalid prepared workload was accepted");
        }
        std::cout << "Immutable prepared workload and shape validation passed without GPU calls" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
