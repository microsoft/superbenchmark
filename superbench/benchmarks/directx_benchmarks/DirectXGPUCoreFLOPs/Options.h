// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

namespace Option {
enum Precision {
    F16,
    F32,
};
using PrecisionType = Option::Precision;
} // namespace Option

struct Options {
    char **begin;
    char **end;

    /**
     * @brief Get the char* value of the cmd line argument.
     * @param option the argument in cmd.
     * @return char*
     */
    char *get_cmd_option(const std::string &option) {
        char **itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) {
            return *itr;
        }
        return 0;
    }

    /**
     * @brief Get the int type value of cmd line argument.
     * @param option the cmd line argument.
     * @return int the int type value of cmd line argument 'option'.
     */
    int get_cmd_line_argument_int(const std::string &option, int defaults) {
        if (char *value = get_cmd_option(option)) {
            return std::stoi(value);
        }
        return defaults;
    }

    /**
     * @brief Get the string type value of cmd line argument.
     * @param  option the cmd line argument.
     * @return std::string the int type value of cmd line argument 'option'.
     */
    std::string get_cmd_line_argument_string(const std::string &option) {
        if (char *value = get_cmd_option(option)) {
            return std::string(value);
        }
        return "";
    }

    /**
     * @brief Get the boolean type value of cmd line argument.
     * @param  option the cmd line argument.
     * @return bool the boolean value.
     */
    bool get_cmd_line_argument_bool(const std::string &option) {
        if (char *value = get_cmd_option(option)) {
            bool b;
            std::istringstream(std::string(value)) >> b;
            return b;
        }
        return false;
    }

    /**
     * @brief Check if a argument exists.
     * @param  option the cmd line argument.
     * @return bool if a argument exists.
     */
    bool cmd_option_exists(const std::string &option) { return std::find(begin, end, option) != end; }

    void get_option_usage() {
        std::cout << "Usage: " << std::endl;
        std::cout << "  --help: Print help message." << std::endl;
        std::cout << "  --num_loops: The number of benchmark runs." << std::endl;
        std::cout << "  --num_loops_in_shader: The number of loops in compute shader." << std::endl;
        std::cout << "  --thread_num_per_block: The number of one block." << std::endl;
        std::cout << "  --num_data_elements: Total thread num." << std::endl;
        std::cout << "  --fp16: half precision to compute." << std::endl;
        std::cout << "  --fp32: float precision to compute." << std::endl;
    }

  public:
    // Data buffer size for copy benchmark.
    uint64_t size;
    // Number of warm up rounds to run.
    int num_warm_up = 0;
    // The number of benchmark runs.
    int num_loops = 0;
    // The number of loops in compute shader.
    int num_loops_in_shader = 0;
    // The number of one block.
    int thread_num_per_block = 0;
    // Total thread num.
    int num_data_elements = 0;
    // The precision of calculate.
    Option::PrecisionType mode_precision = Option::F32;

    /**
     * @brief Construct a new Command Line object.
     * @param argc the number of command line arguments.
     * @param argv the string array of comamnd line arguments.
     */
    Options(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
        if (cmd_option_exists("--help")) {
            get_option_usage();
        } else {
            num_loops = get_cmd_line_argument_int("--num_loops", 10);
            num_loops_in_shader = get_cmd_line_argument_int("--num_loops_in_shader", 100000);
            thread_num_per_block = get_cmd_line_argument_int("--thread_num_per_block", 256);
            num_data_elements = get_cmd_line_argument_int("--num_data_elements", thread_num_per_block * 512 * 8);

            if (get_cmd_line_argument_bool("--f16")) {
                mode_precision = Option::F16;
            }
            if (get_cmd_line_argument_bool("--f32")) {
                mode_precision = Option::F32;
            }
        }
    }
};
