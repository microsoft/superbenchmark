// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// Utility for storing and parsing command line arguments

#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

// Utility for storing command line arguments

struct Options {
    int num_test;
    int warm_up;
    int num_in_step;
    std::string para_info_json;

    // Read params of the functions in json from file 'para_file'
    json read_params() {
        std::ifstream para_info_fin(para_info_json);
        json config;
        if (!para_info_fin) {
            throw "Error: open function param file failed.";
        } else {
            para_info_fin >> config;
            para_info_fin.close();
            return config;
        }
    }
};

// Utility for parsing command line arguments
struct CommandLine {
    char **begin;
    char **end;

    // Construnctor
    CommandLine(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
    }
    // Return value str with "option value"
    char *get_cmd_option(const std::string &option) {
        char **itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) {
            return *itr;
        }
        return 0;
    }
    // Return value with "option value"
    template <typename T> T get_cmd_line_argument(const std::string &option);
};

template <> int CommandLine::get_cmd_line_argument<int>(const std::string &option) {
    if (char *value = get_cmd_option(option)) {
        return std::stoi(value);
    }
    return 0;
}

template <> std::string CommandLine::get_cmd_line_argument<std::string>(const std::string &option) {
    if (char *value = get_cmd_option(option)) {
        return std::string(value);
    }
    return "";
}