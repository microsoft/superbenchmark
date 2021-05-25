/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cmd_helper.h
 * @brief  Helper for parsing command line arguments and pass params to cudnn function
 */

#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cudnn_benchmark.h"
#include "cudnn_function.h"

using json = nlohmann::json;

/**
 * @brief Utility for storing command line arguments
 */
class Options {
    char **begin;
    char **end;

    /**
     * @brief Get the char* value of the cmd line argument
     * @param  option           the argument in cmd
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
     * @brief Get the int type value of cmd line argument
     * @param  option           the cmd line argument
     * @return int              the int type value of cmd line argument 'option'
     */
    int get_cmd_line_argument_int(const std::string &option) {
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
    std::string get_cmd_line_argument_string(const std::string &option) {
        if (char *value = get_cmd_option(option)) {
            return std::string(value);
        }
        return "";
    }

  public:
    int num_test;
    int warm_up;
    int num_in_step;
    std::string para_info_json;

    /**
     * @brief Construct a new Command Line object
     * @param  argc
     * @param  argv
     */
    Options(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
        num_test = get_cmd_line_argument_int("--num_test");
        warm_up = get_cmd_line_argument_int("--warm_up");
        num_in_step = get_cmd_line_argument_int("--num_in_step");
        para_info_json = get_cmd_line_argument_string("--config_path");
    }
};

namespace cudnn_test {

/**
 * @brief  Helper function to convert from json to CudnnConfig
 *
 * The params required for each type of cudnn funcion is as below:
 *
 * @param  j    json including the params of a cudnn function read from 'config_path'
 * @param  fn   a CudnnConfig object
 */
void from_json(const json &j, cudnn_test::CudnnConfig &fn) {
    auto name = j.at("name").get<std::string>();
    fn.set_name(name);
    auto arrayLength = j.at("arrayLength").get<int>();
    fn.set_arrayLength(arrayLength);
    auto input_dims = j.at("input_dims").get<std::vector<int>>();
    fn.set_input_dims(input_dims);
    auto input_stride = j.at("input_stride").get<std::vector<int>>();
    fn.set_input_stride(input_stride);
    auto filter_dims = j.at("filter_dims").get<std::vector<int>>();
    fn.set_filter_dims(filter_dims);
    auto output_dims = j.at("output_dims").get<std::vector<int>>();
    fn.set_output_dims(output_dims);
    auto output_stride = j.at("output_stride").get<std::vector<int>>();
    fn.set_output_stride(output_stride);
    auto algo = j.at("algo").get<int>();
    fn.set_algo(algo);
    auto padA = j.at("padA").get<std::vector<int>>();
    fn.set_padA(padA);
    auto filterStrideA = j.at("filterStrideA").get<std::vector<int>>();
    fn.set_filterStrideA(filterStrideA);
    auto dilationA = j.at("dilationA").get<std::vector<int>>();
    fn.set_dilationA(dilationA);
    auto mode = j.at("mode").get<cudnnConvolutionMode_t>();
    fn.set_mode(mode);
    auto use_tensor_core = j.at("use_tensor_core").get<bool>();
    fn.set_use_tensor_core(use_tensor_core);
    auto input_type = j.at("input_type").get<cudnnDataType_t>();
    fn.set_input_type(input_type);
    auto conv_type = j.at("conv_type").get<cudnnDataType_t>();
    fn.set_conv_type(conv_type);
    auto str = j.dump();
    std::replace(str.begin(), str.end(), '\"', ' ');
    fn.set_str(str);
    fn.name2enum();
}

/**
 * @brief Get the cudnn function pointer of a specific child class object
 * @param  function         base class object of a cudnnFunction, used to initialize the base part of the child class
 * object
 * @return cudnnFunction*  return a base cudnn function pointer of a specific child class
 */
template <typename T1, typename T2> CudnnFunction<T1, T2> *get_cudnn_function_pointer(CudnnConfig &function) {
    switch (function.get_e_name()) {
    case e_cudnnConvolutionForward:
        return new ConvolutionForwardFunction<T1, T2>(function);
    case e_cudnnConvolutionBackwardData:
        return new ConvolutionBackwardDataFunction<T1, T2>(function);
    case e_cudnnConvolutionBackwardFilter:
        return new ConvolutionBackwardFilterFunction<T1, T2>(function);
    default:
        throw "invalid function name";
    }
}

/**
 * @brief run the entire process of benchmark according to cmd auguments
 *
 * first, read the para_info_json file in json array format representing multiple cudnn functions
 * then for each cudnn function, get the pointer of the class object the specific cudnn function
 * finally run the benchmark of each funcion
 *
 * @param  options  the cmd arguments of the application
 */
void run_benchmark(Options &options) {

    std::ifstream para_info_fin(options.para_info_json);
    json config;
    if (!para_info_fin) {
        throw "Error: open function param file failed.";
    } else {
        para_info_fin >> config;
        para_info_fin.close();
    }
    for (json &function_config : config) {
        try {
            // convert function params from json to CudnnConfig class
            cudnn_test::CudnnConfig function = function_config.get<cudnn_test::CudnnConfig>();
            function.set_num_test(options.num_test);
            function.set_warm_up(options.warm_up);
            function.set_num_in_step(options.num_in_step);
            std::cout << function.get_name() << std::endl;
            if (function.get_input_type() == CUDNN_DATA_FLOAT && function.get_conv_type() == CUDNN_DATA_FLOAT) {
                auto p_function = get_cudnn_function_pointer<float, float>(function);
                p_function->benchmark();
                delete p_function;
            }
            if (function.get_input_type() == CUDNN_DATA_HALF && function.get_conv_type() == CUDNN_DATA_FLOAT) {
                auto p_function = get_cudnn_function_pointer<half, float>(function);
                p_function->benchmark();
                delete p_function;
            }
            if (function.get_input_type() == CUDNN_DATA_HALF && function.get_conv_type() == CUDNN_DATA_HALF) {
                auto p_function = get_cudnn_function_pointer<half, half>(function);
                p_function->benchmark();
                delete p_function;
            }

        } catch (std::exception &e) {
            std::cout << "Error: " << e.what() << std::endl;
        }
    }
}
} // namespace cudnn_test
