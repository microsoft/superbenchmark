// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @brief  Helper for parsing command line arguments and pass params to CudnnConfig
 */

#pragma once

#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "convolution_backward_data.h"
#include "convolution_backward_filter.h"
#include "convolution_forward.h"

using json = nlohmann::json;

namespace cudnn_test {
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

    /** @brief Get the bool type value of cmd line argument
     * @param  option           the cmd line argument
     * @return bool             the bool type value of cmd line argument 'option'
     */
    bool get_cmd_line_argument_bool(const std::string &option) {
        if (std::find(begin, end, option) != end) {
            return true;
        }
        return false;
    }

  public:
    int num_test;
    int warm_up;
    int num_in_step;
    int random_seed;
    std::string para_info_json;
    bool auto_algo;

    /**
     * @brief Construct a new Command Line object
     * @param  argc
     * @param  argv
     */
    Options(int argc, char *argv[]) {
        begin = argv;
        end = argv + argc;
        num_test = get_cmd_line_argument_int("--num_test");
        num_test = (num_test == 0 ? 1 : num_test);
        warm_up = get_cmd_line_argument_int("--warm_up");
        warm_up = (warm_up == 0 ? 1 : warm_up);
        num_in_step = get_cmd_line_argument_int("--num_in_step");
        num_in_step = (num_in_step == 0 ? 100 : num_in_step);
        random_seed = get_cmd_line_argument_int("--random_seed");
        random_seed = (random_seed == 0 ? time(NULL) : random_seed);
        para_info_json = get_cmd_line_argument_string("--config_json");
        auto_algo = get_cmd_line_argument_bool("--enable_auto_algo");
        para_info_json =
            para_info_json == ""
                ? R"({"algo":0,"arrayLength":2,"convType":0,"dilationA":[1,1],"filterStrideA":[1,1],"filterDims":[32,128,3,3],"inputDims":[32,128,14,14],"inputStride":[25088,196,14,1],"inputType":0,"mode":1, "name":"cudnnConvolutionBackwardFilter","outputDims":[32,32,14,14],"outputStride":[6272,196,14,1],"padA":[1,1],"tensorOp":false})"
                : para_info_json;
    }
};

/**
 * @brief  Helper function to convert from json to CudnnConfig
 *
 * @param  j    json including the params of a cudnn function read from 'config_path'
 * @param  fn   a CudnnConfig object
 */
void from_json(const json &j, cudnn_test::CudnnConfig &fn) {
    auto str = j.dump();
    std::replace(str.begin(), str.end(), '\"', ' ');
    fn.set_function(str);
    auto name = j.at("name").get<std::string>();
    fn.set_name(name);
    auto input_dims = j.at("inputDims").get<std::vector<int>>();
    fn.set_input_dims(input_dims);
    auto output_dims = j.at("outputDims").get<std::vector<int>>();
    fn.set_output_dims(output_dims);
    auto filter_dims = j.at("filterDims").get<std::vector<int>>();
    fn.set_filter_dims(filter_dims);
    auto input_type = j.at("inputType").get<cudnnDataType_t>();
    fn.set_input_type(input_type);
    auto conv_type = j.at("convType").get<cudnnDataType_t>();
    fn.set_conv_type(conv_type);
    auto array_length = j.at("arrayLength").get<int>();
    fn.set_array_length(array_length);
    auto input_stride = j.at("inputStride").get<std::vector<int>>();
    fn.set_input_stride(input_stride);
    auto output_stride = j.at("outputStride").get<std::vector<int>>();
    fn.set_output_stride(output_stride);
    if (j.contains("algo")) {
        auto algo = j.at("algo").get<int>();
        fn.set_algo(algo);
    }
    auto padA = j.at("padA").get<std::vector<int>>();
    fn.set_padA(padA);
    auto filter_strideA = j.at("filterStrideA").get<std::vector<int>>();
    fn.set_filter_strideA(filter_strideA);
    auto dilationA = j.at("dilationA").get<std::vector<int>>();
    fn.set_dilationA(dilationA);
    auto mode = j.at("mode").get<cudnnConvolutionMode_t>();
    fn.set_mode(mode);
    auto use_tensor_op = j.at("tensorOp").get<bool>();
    fn.set_use_tensor_op(use_tensor_op);
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
    try {
        json function_config = json::parse(options.para_info_json);
        // convert function params from json to CudnnConfig class
        cudnn_test::CudnnConfig function = function_config.get<cudnn_test::CudnnConfig>();
        function.set_num_test(options.num_test);
        function.set_warm_up(options.warm_up);
        function.set_num_in_step(options.num_in_step);
        function.set_random_seed(options.random_seed);
        function.set_auto_algo(options.auto_algo);
        if (function.get_input_type() == CUDNN_DATA_FLOAT && function.get_conv_type() == CUDNN_DATA_FLOAT) {
            auto p_function = get_cudnn_function_pointer<float, float>(function);
            p_function->benchmark();
            delete p_function;
        } else {
            if (function.get_input_type() == CUDNN_DATA_HALF && function.get_conv_type() == CUDNN_DATA_FLOAT) {
                auto p_function = get_cudnn_function_pointer<half, float>(function);
                p_function->benchmark();
                delete p_function;
            } else {
                if (function.get_input_type() == CUDNN_DATA_HALF && function.get_conv_type() == CUDNN_DATA_HALF) {
                    auto p_function = get_cudnn_function_pointer<half, half>(function);
                    p_function->benchmark();
                    delete p_function;
                } else {
                    throw "invalid input and conv type";
                }
            }
        }
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}
} // namespace cudnn_test
