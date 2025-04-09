// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_stream.hpp"

/**
 * @brief Main function and entry of gpu stream benchmark

 * @details
 * params list:
 *  num_warm_up: warm up count
 *  num_loops: num of runs for timing
 *  size: number of bytes to setup for the test
 * @param  argc argument count
 * @param  argv argument vector
 * @return int
 */
int main(int argc, char **argv) {
    int ret = 0;
    stream_config::Opts opts;

    // parse arguments from cmd
    ret = stream_config::ParseOpts(argc, argv, &opts);
    if (ret != 0) {
        return ret;
    }

    // run the stream benchmark
    GpuStream gpu_stream(opts);
    ret = gpu_stream.Run();
    if (ret != 0) {
        return ret;
    }

    return 0;
}
