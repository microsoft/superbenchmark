// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <stdexcept>

#include "cudnn_execution_config.h"

using namespace cudnn_test;

static void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main() {
    try {
        CudnnConfig config{};
        config.set_input_dims({2, 8, 5, 5});
        config.set_input_stride({200, 25, 5, 1});
        config.set_filter_dims({16, 8, 3, 3});
        config.set_output_dims({2, 16, 3, 3});
        config.set_output_stride({144, 9, 3, 1});
        config.set_padA({0, 0});
        config.set_filter_strideA({1, 1});
        config.set_dilationA({1, 1});
        config.set_mode(CUDNN_CROSS_CORRELATION);
        config.set_array_length(2);
        const CudnnConvolutionWorkload workload(config);
        config.set_input_dims({1, 1, 1, 1});
        config.set_padA({2, 2});
        require(workload.input_dims == std::vector<int>{2, 8, 5, 5} && workload.padding == std::vector<int>{0, 0},
                "workload changed with its source configuration");
        require(workload.input_stride == std::vector<int>{200, 25, 5, 1} &&
                    workload.filter_dims == std::vector<int>{16, 8, 3, 3} &&
                    workload.output_dims == std::vector<int>{2, 16, 3, 3} &&
                    workload.output_stride == std::vector<int>{144, 9, 3, 1} &&
                    workload.filter_stride == std::vector<int>{1, 1} && workload.dilation == std::vector<int>{1, 1} &&
                    workload.mode == CUDNN_CROSS_CORRELATION && workload.array_length == 2,
                "workload parameters were not preserved");
        for (bool tensor_op : {false, true}) {
            for (bool automatic : {false, true}) {
                config.set_use_tensor_op(tensor_op);
                config.set_auto_algo(automatic);
                config.set_algo(1);
                const CudnnExecutionPolicy policy(config);
                config.set_use_tensor_op(!tensor_op);
                config.set_auto_algo(!automatic);
                config.set_algo(7);
                require(policy.math_type == (tensor_op ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH) &&
                            policy.auto_algo == automatic && (automatic || policy.algorithm == 1),
                        "legacy math/algorithm policy was not preserved");
            }
        }
        std::cout << "Convolution workload and legacy execution policy snapshots passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
