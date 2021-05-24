/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cmd_helper.h
 * @brief  Utility for storing and parsing command line arguments
 */

#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

using json = nlohmann::json;

/**
 * @brief Utility for storing command line arguments
 */
class Options {
  public:
    int num_test;
    int warm_up;
    int num_in_step;
    std::string para_info_json;

    /**
     * @brief Read params of the functions in json from file 'para_file'
     * @return json
     */
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

/**
 * @brief Utility for parsing command line arguments
 */
class CommandLine {
    char **begin;
    char **end;

  public:
    /**
     * @brief Construct a new Command Line object
     * @param  argc
     * @param  argv
     */
    CommandLine(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
    }

    // Get the char* value of the cmd line argument
    char *get_cmd_option(const std::string &option);

    // Get the value of cmd line argument
    int get_cmd_line_argument_int(const std::string &option);

    // Get the value of cmd line argument
    std::string get_cmd_line_argument_string(const std::string &option);
};

/**
 * @brief Get the char* value of the cmd line argument
 * @param  option           the argument in cmd
 * @return char*
 */
char *CommandLine::get_cmd_option(const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) {
        return *itr;
    }
    return 0;
}

/**
 * @brief Get the int type value of cmd line argument
 * @param  option           the cmd line argument
 * @return int              the int type value of cmd line argument 'option'
 */
int CommandLine::get_cmd_line_argument_int(const std::string &option) {
    if (char *value = get_cmd_option(option)) {
        return std::stoi(value);
    }
    return 0;
}

/**
 * @brief Get the string type value of cmd line argument
 * @param  option           the cmd line argument
 * @return std::string      the int type value of cmd line argument 'option'
 */
std::string CommandLine::get_cmd_line_argument_string(const std::string &option) {
    if (char *value = get_cmd_option(option)) {
        return std::string(value);
    }
    return "";
}