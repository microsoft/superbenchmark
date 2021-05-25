/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cublas_test.cpp
 * @brief Cublas function benchmark will read the functions' param fro json file 'para_info.json', and use these params
 * to benchmark the wall time of cublas functions.
 */

#include "cublas_function_helper.h"

/**
 * @brief Main function and entry of cublas benchmark

 * @details
 * params list:
 *  num_test: test step nums
 *  warm_up: warm up step nums
 *  num_in_step: times each step will invoke the function
 *   config path: the path of 'para_info.json'
 * functions supported:
 *   cublasSgemm
 *   cublasGemmEx
 *   cublasSgemmStridedBatched
 *   cublasGemmStridedBatchedEx
 *   cublasCgemm
 *   cublasCgemm3mStridedBatched

 * @param  argc
 * @param  argv
 * @return int
 */
int main(int argc, char *argv[]) {
    try {
        // parse arguments from cmd
        Options options(argc, argv);
        // benchmark each function defined in 'para_info.json'
        run_benchmark(options);
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
        exit(-1);
    }
}
