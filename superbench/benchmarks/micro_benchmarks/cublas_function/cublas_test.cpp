// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Cublas function benchmark will read the functions' param fro json file 'para_info.json', and use these params to
// benchmark the wall time of cublas functions. params list:
//     num_test: test step nums
//     warm_up: warm up step nums
//     num_in_step: times each step will invoke the function
//     config path: the path of 'para_info.json'
// functions supported:
//     cublasSgemm
//     cublasGemmEx
//     cublasSgemmStridedBatched
//     cublasGemmStridedBatchedEx
//     cublasCgemm
//     cublasCgemm3mStridedBatched

#include "cublas_benchmark.h"

int main(int argc, const char *argv[]) {
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
    for (auto &function_config : config) {
        auto one_function = function_config.get<CublasFunction>();
        switch (one_function.get_e_name()) {
        case e_cublasSgemm:
        case e_cublasGemmEx:
        case e_cublasSgemmStridedBatched:
        case e_cublasGemmStridedBatchedEx:
            one_function.benchmark<float>(&options);
            break;
        case e_cublasCgemm:
        case e_cublasCgemm3mStridedBatched:
            one_function.benchmark<cuComplex>(&options);
            break;
        default:
            std::cout << "Error: invalid enum name";
        }
    }
}
