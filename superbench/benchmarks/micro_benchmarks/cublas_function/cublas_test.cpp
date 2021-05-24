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

int main(int argc, char *argv[]) {
    // parse arguments from cmd
    CommandLine cmdline(argc, argv);
    Options options;
    try {
        options.num_test = cmdline.get_cmd_line_argument_int("--num_test");
        options.warm_up = cmdline.get_cmd_line_argument_int("--warm_up");
        options.num_in_step = cmdline.get_cmd_line_argument_int("--num_in_step");
        options.para_info_json = cmdline.get_cmd_line_argument_string("--config_path");

        // read list of function params from 'para_info.json'
        nlohmann::json config = options.read_params();

        // benchmark each function defined in 'para_info.json'
        for (auto &function_config : config) {
            try {
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
                    throw "invalid function name";
                }
            } catch (std::exception &e) {
                std::cout << "Error: " << e.what() << std::endl;
            }
        }
    } catch (std::exception) {
        std::cout << "Error: "
                  << "invalid argument" << std::endl;
        exit(-1);
    }
}
