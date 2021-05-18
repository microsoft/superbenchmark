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
            std::cout << "Error: open function param file failed." << std::endl;
            exit(1);
        } else {
            para_info_fin >> config;
            para_info_fin.close();
            return config;
        }
    }
};

// Utility for parsing command line arguments
// Referenced by NVIDIA/cutlass/tools/util/include/cutlass/util/command_line.h
// https://github.com/NVIDIA/cutlass/blob/master/tools/util/include/cutlass/util/command_line.h
struct CommandLine {
    char **begin;
    char **end;

    CommandLine(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
    }
    char *getCmdOption(const std::string &option) {
        char **itr = std::find(begin, end, option);
        if (itr != end && ++itr != end) {
            return *itr;
        }
        return 0;
    }

    template <typename T> T get_cmd_line_argument(const std::string &option);
};

template <> int CommandLine::get_cmd_line_argument<int>(const std::string &option) {
    if (char *value = getCmdOption(option)) {
        return std::stoi(value);
    }
    return 0;
}

template <> std::string CommandLine::get_cmd_line_argument<std::string>(const std::string &option) {
    if (char *value = getCmdOption(option)) {
        return std::string(value);
    }
    return "";
}
