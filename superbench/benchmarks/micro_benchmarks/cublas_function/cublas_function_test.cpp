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

#include <time.h>
#include <unistd.h>

#include "function.h"

int main(int argc, const char *argv[]) {
    clock_t start, end;
    start = clock();

    // parse arguments from cmd
    CommandLine cmdline(argc, argv);
    Options options;
    cmdline.get_cmd_line_argument("num_test", options.num_test, 1);
    cmdline.get_cmd_line_argument("warm_up", options.warm_up, 1);
    cmdline.get_cmd_line_argument("num_in_step", options.num_in_step, 1000);
    cmdline.get_cmd_line_argument("config_path", options.para_info_json,
                                  get_current_dir_name() + std::string("/../para_info.json"));

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
            std::cout << "invalid enum name";
        }
    }

    end = clock();
    std::cout << "program total time = " << double(end - start) / CLOCKS_PER_SEC << "s" << std::endl;
}
