// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @brief Cudnn function benchmark will read the params from cmd, and use these params
 * to benchmark the wall time of the cudnn functions.
 */

#include <limits>
#include <stdexcept>

#include "cudnn_function_helper.h"

/**
 * @brief Main function and entry of cudnn benchmark

 * @details
 * params list:
 *  num_test: test step nums
 *  warm_up: warm up step nums
 *  num_in_step: times each step will invoke the function
 *  random_seed: the random seed to generate data
 *  config_json: the json string including the params of the function
 *  functions supported:
 *  cudnnConvolutionForward
 *  cudnnConvolutionBackwardData
 *  cudnnConvolutionBackwardFilter

 * @param  argc
 * @param  argv
 * @return int
 */
int main(int argc, char *argv[]) {
    try {
        // parse arguments from cmd
        cudnn_test::Options options(argc, argv);
        // benchmark the function
        cudnn_test::run_benchmark(options);
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        exit(-1);
    }
    return 0;
}
