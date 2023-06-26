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
    int get_cmd_line_argument_int(const std::string &option, int defaults) {
        if (char *value = get_cmd_option(option)) {
            try {
                return std::stoi(value);
            } catch (const std::exception &e) {
                std::cout << "Error: Invalid argument - " << option << " should be INT" << e.what() << '\n';
                exit(1);
            }
        }
        return defaults;
    }

    /**
     * @brief Get the unsigned long long type value of cmd line argument.
     * @param option the cmd line argument.
     * @param defaults the default value.
     * @return unsigned long long the unsigned long long type value of cmd line argument 'option'.
     */
    unsigned long long get_cmd_line_argument_ulonglong(const std::string &option, unsigned long long defaults) {
        if (char *value = get_cmd_option(option)) {
            try {
                return std::stoull(value);
            } catch (const std::exception &e) {
                std::cout << "Error: Invalid argument - " << option << " should be unsigned long long" << e.what()
                          << '\n';
                exit(1);
            }
        }
        return defaults;
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
        if (cmd_option_exists(option)) {
            return true;
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
        std::cout << "  --size <int>            Size of data for GPU copy." << std::endl;
        std::cout << "  --warm_up <int>         Number of warm up copy times to run." << std::endl;
        std::cout << "  --num_loops <int>       Number of copy times to run." << std::endl;
        std::cout << "  --minbytes <int>        Run size from min_size to max_size for GPU copy." << std::endl;
        std::cout << "  --maxbytes <int>        Run size from min_size to max_size for GPU copy." << std::endl;
        std::cout << "  --htod <bool>           Host-to-device copy mode." << std::endl;
        std::cout << "  --dtoh <bool>           Device-to-host copy mode." << std::endl;
        std::cout << "  --check <bool>          Whether check data after copy." << std::endl;
        exit(0);
    }

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
    }
};
