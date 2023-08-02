// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "../directx_utils/Options.h"

class BenchmarkOptions : public Options {

  public:
    // Size of data for GPU copy.
    unsigned long long size;
    // Run size from min_size to max_size for GPU copy.
    unsigned long long min_size = 0;
    // Run size from min_size to max_size for GPU copy.
    unsigned long long max_size = 0;
    // Number of warm up copy times to run.
    int num_warm_up = 0;
    // Number of copy times to run.
    int num_loops = 0;
    // Host-to-device copy mode.
    bool htod_enabled = false;
    // device-to-host copy mode.
    bool dtoh_enabled = false;
    // Whether check data after copy.
    bool check_data = false;

    /**
     * @brief Construct a new BenchmarkOptions object.
     */
    BenchmarkOptions(int argc, char *argv[]) : Options(argc, argv) {}

    /**
     * @brief Parse the arguments.
     */
    virtual void parse_arguments() override {
        size = get_cmd_line_argument_int("--size", -1);
        num_warm_up = get_cmd_line_argument_int("--warm_up", 20);
        num_loops = get_cmd_line_argument_int("--num_loops", 100000);
        min_size = get_cmd_line_argument_int("--minbytes", 64);
        max_size = get_cmd_line_argument_ulonglong("--maxbytes", 8 * 1024 * 1024);
        htod_enabled = get_cmd_line_argument_bool("--htod");
        dtoh_enabled = get_cmd_line_argument_bool("--dtoh");
        check_data = get_cmd_line_argument_bool("--check");
        if (!htod_enabled && !dtoh_enabled) {
            std::cerr << "Error: Please specify copy mode!" << std::endl;
            exit(-1);
        }
    }

    /**
     * @brief Get the option usage.
     */
    void get_option_usage() override {
        std::cout << "Usage: " << std::endl;
        std::cout << "  --size <int>            Size of data for GPU copy." << std::endl;
        std::cout << "  --warm_up <int>         Number of warm up copy times to run." << std::endl;
        std::cout << "  --num_loops <int>       Number of copy times to run." << std::endl;
        std::cout << "  --minbytes <int>        Run size from min_size to max_size for GPU copy." << std::endl;
        std::cout << "  --maxbytes <int>        Run size from min_size to max_size for GPU copy." << std::endl;
        std::cout << "  --htod <bool>           Host-to-device copy mode." << std::endl;
        std::cout << "  --dtoh <bool>           Device-to-host copy mode." << std::endl;
        std::cout << "  --check <bool>          Whether check data after copy." << std::endl;
    }
};
