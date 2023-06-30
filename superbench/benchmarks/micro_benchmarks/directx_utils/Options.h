// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct UInt3 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

class Options {
  protected:
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
     * @param defaults the default value.
     * @return int the int type value of cmd line argument 'option'.
     */
    int get_cmd_line_argument_int(const std::string &option, int defaults) {
        if (char *value = get_cmd_option(option)) {
            try {
                return std::stoi(value);
            } catch (const std::exception &e) {
                std::cerr << "Error: Invalid argument - " << option << " should be INT " << e.what() << '\n';
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
            }
        }
        return defaults;
    }

    /**
     * @brief Split the string by ',' and convert to unsigned int.
     * @param str the string to be split.
     * @return std::vector<unsigned int> the vector of unsigned int.
     */
    std::vector<unsigned int> splitAndConvertToInt(const std::string &str) {
        std::vector<unsigned int> result;
        std::stringstream ss(str);
        std::string token;

        while (std::getline(ss, token, ',')) {
            try {
                result.push_back(std::stoul(token));
            } catch (std::invalid_argument &e) {
                throw std::invalid_argument("Invalid argument: " + token + e.what());
            }
        }
        return result;
    }

    /**
     * @brief Get the unsigned int 3 type value of cmd line argument.
     * @param option the cmd line argument.
     * @param defaults the default value.
     * @return unsigned int the unsigned int 3 type value of cmd line argument 'option'.
     */
    UInt3 get_cmd_line_argument_uint3(const std::string &option, const UInt3 &defaults) {
        if (char *value = get_cmd_option(option)) {
            try {
                std::vector<unsigned int> values = splitAndConvertToInt(value);
                if (values.size() != 3) {
                    std::cout << "Error: Invalid argument - " << option << " should be unsigned int3" << '\n';
                    exit(1);
                }
                return {values[0], values[1], values[2]};

            } catch (const std::exception &e) {
                std::cout << "Error: Invalid argument - " << option << " should be unsigned int3" << e.what() << '\n';
                exit(1);
            }
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
        if (cmd_option_exists(option)) {
            return true;
        }
        return false;
    }

    /**
     * @brief Check if a argument exists.
     * @param  option the cmd line argument.
     * @return bool if a argument exists.
     */
    bool cmd_option_exists(const std::string &option) { return std::find(begin, end, option) != end; }

    /**
     * @brief Get the option usage.
     */
    virtual void get_option_usage() = 0;

    /**
     * @brief Parse the arguments.
     */
    virtual void parse_arguments() = 0;

  public:
    /**
     * @brief Construct a new Command Line object.
     * @param argc the number of command line arguments.
     * @param argv the string array of comamnd line arguments.
     */
    Options(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
    }

    /**
     * @brief Init and parse the arguments.
     */
    virtual void init() {
        if (cmd_option_exists("--help")) {
            get_option_usage();
            exit(0);
        }
        try {
            parse_arguments();
        } catch (const std::exception &e) {
            std::cerr << "Error: Invalid argument - " << e.what() << '\n';
            exit(1);
        }
    };
};
