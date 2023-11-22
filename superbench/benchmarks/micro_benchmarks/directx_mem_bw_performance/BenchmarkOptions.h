// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "../directx_utils/Options.h"
#include "GPUMemRwBw.h"

enum Memtype {
    Read,
    Write,
    ReadWrite,
};
const std::string MemtypeString[] = {"Read", "Write", "ReadWrite"};

class BenchmarkOptions : public Options {
  public:
    // Number of warm up rounds.
    int num_warm_up = 0;
    // Number of loop rounds of dispatch to measure the performance.
    int num_loop = 0;
    // Size of data for GPU mem access.
    unsigned long long size;
    // Run size from min_size to max_size for GPU mem access.
    unsigned long long min_size = 0;
    // Run size from min_size to max_size for GPU mem access.
    unsigned long long max_size = 0;
    // Whether check data correctness.
    bool check_data = false;
    // Memory operation type.
    Memtype mem_type = Memtype::Write;
    // Number of threads to launch.
    UInt3 num_threads;

    /**
     * @brief Construct a new BenchmarkOptions object.
     */
    BenchmarkOptions(int argc, char *argv[]) : Options(argc, argv) {}

    /**
     * @brief Get the option usage.
     */
    void get_option_usage() override {
        std::cout << "Usage: " << std::endl;
        std::cout << "  --num_warm_up <num_warm_up> : Number of warm up rounds." << std::endl;
        std::cout << "  --num_loop <num_loop> : Number of loop times to measure the performance." << std::endl;
        std::cout << "  --minbytes <minbytes> : Lower data size bound to test." << std::endl;
        std::cout << "  --maxbytes <maxbytes> : Upper data size bound to test." << std::endl;
        std::cout << "  --check_data <check_data> : Whether check data correctness." << std::endl;
        std::cout << "  --read : Memory operation type is read." << std::endl;
        std::cout << "  --write : Memory operation type is write." << std::endl;
        std::cout << "  --readwrite : Memory operation type is readwrite." << std::endl;
        std::cout << "  --numthreads <x>,<y>,<z> : Number of threads in 3 dimenstions to launch." << std::endl;
        std::cout << "  --help : Print help message." << std::endl;
    }

    /**
     * @brief Parse the arguments.
     */
    virtual void parse_arguments() override {
        num_warm_up = get_cmd_line_argument_int("--num_warm_up", 0);
        num_loop = get_cmd_line_argument_int("--num_loop", 1);
        size = get_cmd_line_argument_ulonglong("--size", -1);
        min_size = get_cmd_line_argument_int("--minbytes", 4 * 1024);
        max_size =
            get_cmd_line_argument_ulonglong("--maxbytes", static_cast<unsigned long long>(1LL * 1024 * 1024 * 1024));
        check_data = get_cmd_line_argument_bool("--check_data");
        if (get_cmd_line_argument_bool("--read")) {
            mem_type = Memtype::Read;
        }
        if (get_cmd_line_argument_bool("--write")) {
            mem_type = Memtype::Write;
        }
        if (get_cmd_line_argument_bool("--readwrite")) {
            mem_type = Memtype::ReadWrite;
        }
        num_threads = get_cmd_line_argument_uint3("--numthreads", {256, 1, 1});
    }
};
