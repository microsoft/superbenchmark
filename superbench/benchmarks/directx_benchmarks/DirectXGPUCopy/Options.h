// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

struct Options {
    char **begin;
    char **end;

    /**
     * @brief Get the char* value of the cmd line argument
     * @param  option the argument in cmd.
     * @return char* of sub string.
     */
    char *get_cmd_option(const std::string &option) {
        char **itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) {
            return *itr;
        }
        return 0;
    }

    /**
     * @brief Get the int type value of cmd line argument
     * @param  option the cmd line argument.
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
     * @brief Get the string type value of cmd line argument
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

    void get_option_usage() {}

  public:
    // Data buffer size for copy benchmark.
    uint64_t size;
    // Number of warm up rounds to run.
    int num_warm_up = 0;
    // Number of loops to run.
    int num_loops = 0;
    // Lower limits of each buffer.
    int minbytes = 0;
    // Upper limits of each buffer.
    unsigned long long maxbytes = 0;
    // Whether host-to-device transfer needs to be evaluated.
    bool htod_enabled = false;
    // Whether device-to-host transfer needs to be evaluated.
    bool dtoh_enabled = false;
    // Whether check data after copy.
    bool check_data = false;

    /**
     * @brief Construct a new Command Line object
     * @param argc the number of arguments.
     * @param argv the string array of arguments.
     */
    Options(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
        if (cmd_option_exists("--help")) {
            get_option_usage();
        } else {
            size = get_cmd_line_argument_int("--size");
            size = (size == 0 ? 1LL * 64 * 1024 * 1024 : size);
            num_warm_up = get_cmd_line_argument_int("--warm_up");
            num_warm_up = (num_warm_up == 0 ? 20 : num_warm_up);
            num_loops = get_cmd_line_argument_int("--num_loops");
            num_loops = (num_loops == 0 ? 100000 : num_loops);
            minbytes = get_cmd_line_argument_int("--minbytes");
            minbytes = (minbytes == 0 ? 64 : minbytes);
            maxbytes = get_cmd_line_argument_ulonglong("--maxbytes");
            maxbytes = (maxbytes == 0 ? 8 * 1024 * 1024 : maxbytes);
            htod_enabled = get_cmd_line_argument_bool("--htod");
            dtoh_enabled = get_cmd_line_argument_bool("--dtoh");
            check_data = get_cmd_line_argument_bool("--check");
            if (!htod_enabled && !dtoh_enabled) {
                std::cerr << "Error: Please specify memory type!" << std::endl;
                exit(-1);
            }
        }
    }
};
