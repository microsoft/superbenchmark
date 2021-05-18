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

#include "thirdparty/json.hpp"

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
            std::cout << "Open function param file fail." << std::endl;
            exit(1);
        } else {
            para_info_fin >> config;
            para_info_fin.close();
            return config;
        }
    }
};

// Utility for parsing command line arguments
struct CommandLine {
    std::vector<std::string> keys;
    std::vector<std::string> values;
    std::vector<std::string> args;

    // Constructor
    CommandLine(int argc, const char **argv) {
        using namespace std;

        for (int i = 1; i < argc; i++) {
            string arg = argv[i];

            if ((arg[0] != '-') || (arg[1] != '-')) {
                args.push_back(arg);
                continue;
            }

            string::size_type pos;
            string key, val;
            if ((pos = arg.find('=')) == string::npos) {
                key = string(arg, 2, arg.length() - 2);
                val = "";
            } else {
                key = string(arg, 2, pos - 2);
                val = string(arg, pos + 1, arg.length() - 1);
            }

            keys.push_back(key);
            values.push_back(val);
        }
    }

    // Checks whether a flag "--<flag>" is present in the commandline
    bool check_cmd_line_flag(const char *arg_name) const {
        using namespace std;

        for (int i = 0; i < int(keys.size()); ++i) {
            if (keys[i] == string(arg_name))
                return true;
        }
        return false;
    }

    // Obtains the boolean value specified for a given commandline parameter --<flag>=<bool>
    void get_cmd_line_argument(const char *arg_name, bool &val, bool _default = true) const {
        val = _default;
        if (check_cmd_line_flag(arg_name)) {
            std::string value;
            get_cmd_line_argument(arg_name, value);

            val = !(value == "0" || value == "false");
        }
    }

    // Obtains the value specified for a given commandline parameter --<flag>=<value>
    template <typename value_t> void get_cmd_line_argument(const char *arg_name, value_t &val) const {

        get_cmd_line_argument(arg_name, val, val);
    }

    // Obtains the value specified for a given commandline parameter --<flag>=<value>
    template <typename value_t>
    void get_cmd_line_argument(const char *arg_name, value_t &val, value_t const &_default) const {
        using namespace std;
        val = _default;
        for (int i = 0; i < int(keys.size()); ++i) {
            if (keys[i] == string(arg_name)) {
                istringstream str_stream(values[i]);
                str_stream >> val;
            }
        }
    }
};
