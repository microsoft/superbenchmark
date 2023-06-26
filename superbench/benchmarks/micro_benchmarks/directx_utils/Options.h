// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <sstream>
#include <string>

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
    virtual void get_option_usage(){};

    /**
     * @brief Parse the arguments.
     */
    virtual void parse_arguments(){};

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
