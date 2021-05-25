// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Cudnn function benchmark will read the functions' param fro json file 'para_info.json', and use these params to
// benchmark the wall time of cudnn functions. params list:
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

#include "cudnn_function_helper.h"

int main(int argc, char *argv[]) {
    try {
        // parse arguments from cmd
        Options options(argc, argv);
        // benchmark each function defined in 'para_info.json'
        cudnn_test::run_benchmark(options);
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        exit(-1);
    }
    return 0;
}
