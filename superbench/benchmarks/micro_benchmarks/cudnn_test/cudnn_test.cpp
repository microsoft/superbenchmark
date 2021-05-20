// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Cudnn function benchmark will read the functions' param fro json file 'para_info.json', and use these params to
// benchmark the wall time of cublas functions. params list:
//     num_test: test step nums
//     warm_up: warm up step nums
//     num_in_step: times each step will invoke the function
//     config path: the path of 'para_info.json'
// functions supported:
//     cudnnConvolutionForward
//     cudnnConvolutionBackwardData
//     cudnnConvolutionBackwardFilter

#include <limits>
#include <stdexcept>

#include "cudnn_benchmark.h"

int main(int argc, char *argv[]) {
    // parse arguments from cmd
    CommandLine cmdline(argc, argv);
    Options options;
    options.num_test = cmdline.get_cmd_line_argument<int>("--num_test");
    options.warm_up = cmdline.get_cmd_line_argument<int>("--warm_up");
    options.num_in_step = cmdline.get_cmd_line_argument<int>("--num_in_step");
    options.para_info_json = cmdline.get_cmd_line_argument<std::string>("--config_path");

    // read list of function params from 'para_info.json'
    auto config = options.read_params();

    // benchmark each function defined in 'para_info.json'
    for (auto &function_config_json : config) {
        try {
            // convert function params from json to CudnnConfig class
            auto function_config = function_config_json.get<cudnn_test::CudnnConfig>();

            if (function_config.get_input_type() == CUDNN_DATA_FLOAT &&
                function_config.get_conv_type() == CUDNN_DATA_FLOAT) {
                cudnn_test::CudnnFunction<float, float> one_function(&function_config);
                one_function.benchmark(&options);
            }
            if (function_config.get_input_type() == CUDNN_DATA_HALF &&
                function_config.get_conv_type() == CUDNN_DATA_FLOAT) {
                cudnn_test::CudnnFunction<half, float> one_function(&function_config);
                one_function.benchmark(&options);
            }
            if (function_config.get_input_type() == CUDNN_DATA_HALF &&
                function_config.get_conv_type() == CUDNN_DATA_HALF) {
                cudnn_test::CudnnFunction<half, half> one_function(&function_config);
                one_function.benchmark(&options);
            }
            throw "invalid input function type";
        } catch Exception(e) {
            std::cout << "Error: " + e.what() << std::endl;
        }
    }

    return 0;
}
