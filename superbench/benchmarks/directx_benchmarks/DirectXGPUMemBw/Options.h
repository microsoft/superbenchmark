// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

#include "GPUMemRwBw.h"

namespace Option {
enum Opt {
    Read,
    Write,
    ReadWrite,
};
using OptType = Option::Opt;
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
    int get_cmd_line_argument_int(const std::string &option) {
        if (char *value = get_cmd_option(option)) {
            return std::stoi(value);
        }
        return 0;
    }


    /**
     * @brief Get the unsigned long long type value of cmd line argument.
     * @param option the cmd line argument.
     * @return unsigned long long the unsigned long long type value of cmd line argument 'option'.
     */
    unsigned long long get_cmd_line_argument_ulonglong(const std::string &option) {
        if (char *value = get_cmd_option(option)) {
            return std::stoull(value);
        }
        return 0;
    }

    /**
     * @brief Get the string type value of cmd line argument.
     * @param option the cmd line argument.
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
     * @param option the cmd line argument.
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
     * @param option the cmd line argument.
     * @return bool if a argument exists.
     */
    bool cmd_option_exists(const std::string &option) { return std::find(begin, end, option) != end; }

    void get_option_usage() {
        std::cout << "Usage: " << std::endl;
        std::cout << "  --num_warm_up <num_warm_up> : Number of warm up rounds." << std::endl;
        std::cout << "  --num_loop <num_loop> : Number of loop times to measure the performance." << std::endl;
        std::cout << "  --minbytes <minbytes> : Lower data size bound to test." << std::endl;
        std::cout << "  --maxbytes <maxbytes> : Upper data size bound to test." << std::endl;
        std::cout << "  --check_data <check_data> : Whether check data correctness." << std::endl;
        std::cout << "  --opt_type <opt_type> : Memory operation type." << std::endl;
        std::cout << "  --help : Print help message." << std::endl;
    }

  public:
    // Number of warm up rounds.
    int num_warm_up = 0;
    // Number of loop rounds of dispatch to measure the performance.
    int num_loop = 0;
    // Lower data size bound to test.
    int minbytes = 0;
    // Upper data size bound to test.
    unsigned long long maxbytes = 0;
    // Whether check data correctness.
    bool check_data = false;
    // Memory operation type.
    Option::OptType opt_type = Option::Write;

    /**
     * @brief Construct a new Command Line object.
     * @param argc the number of arguments.
     * @param argv the string array of arguments.
     */
    Options(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
        if (cmd_option_exists("--help")) {
            get_option_usage();
            exit(0);
        } else {
            num_warm_up = get_cmd_line_argument_int("--num_warm_up");
            num_warm_up = (num_warm_up == 0 ? 0 : num_warm_up);
            minbytes = get_cmd_line_argument_int("--minbytes");
            minbytes = (minbytes == 0 ? 4 * 1024 : minbytes);
            maxbytes = get_cmd_line_argument_ulonglong("--maxbytes");
            maxbytes = (maxbytes == 0 ? static_cast<unsigned long long>(1LL * 1024 * 1024 * 1024) : maxbytes);
            num_loop = get_cmd_line_argument_int("--num_loop");
            num_loop = (num_loop == 0 ? 1 : num_loop);
            check_data = get_cmd_line_argument_bool("--check");
            if (get_cmd_line_argument_bool("--r")) {
                opt_type = Option::Read;
            }
            if (get_cmd_line_argument_bool("--w")) {
                opt_type = Option::Write;
            }
            if (get_cmd_line_argument_bool("--rw")) {
                opt_type = Option::ReadWrite;
            }
        }
    }
};
