// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cublas_function_helper.h
 * @brief  Helper for parsing command line arguments and pass params to cublas function
 */

#pragma once

#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "cublas_function.h"
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
     * @brief Get the double type value of cmd line argument
     * @param  option           the cmd line argument
     * @return double           the double type value of cmd line argument 'option'
     */
    double get_cmd_line_argument_double(const std::string &option) {
        if (char *value = get_cmd_option(option)) {
            return std::atof(value);
        }
        return 0.0;
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

    /**
     * @brief Get the bool type value of cmd line argument
     * @param  option           the cmd line argument
     * @return std::string      the int type value of cmd line argument 'option'
     */
    bool get_cmd_line_argument_bool(const std::string &option) {
        char **itr = std::find(begin, end, option);
        if (itr != end) {
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
    bool correctness_check;
    double eps;
    bool random_data;

    /**
     * @brief Construct a options object according to cmd or set a default value used to test
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
        para_info_json = para_info_json == "" ? R"({"name":"cublasCgemm","m":512,"n":512,"k":32,"transa":1,"transb":0})"
                                              : para_info_json;
        correctness_check = get_cmd_line_argument_bool("--correctness");
        eps = get_cmd_line_argument_double("--eps");
        random_data = get_cmd_line_argument_bool("--random_data");
    }
};

/**
 * @brief  Helper function to convert from json to cublasfunction
 *
 * The params required for each type of cublas funcion is as below:
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| name                        | m        | n        | k        | transa   | transb   | batchCount | datatype |
 *use_tensor_core |
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| cublasSgemm                 | required | required | required | required | required | no         | no       | no |
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| cublasGemmEx                | required | required | required | required | required | no         | required |
 *required
 *|
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| cublasSgemmStridedBatched   | required | required | required | required | required | required   | no       | no |
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| cublasGemmStridedBatchedEx  | required | required | required | required | required | required   | required |
 *required
 *|
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| cublasCgemm                 | required | required | required | required | required | no         | no       | no |
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *| cublasCgemm3mStridedBatched | required | required | required | required | required | required   | no       | no |
 *+-----------------------------+----------+----------+----------+----------+----------+------------+----------+-----------------+
 *
 * @param  j    json including the params of a cublas function read from 'config_json'
 * @param  fn   a CublasFunction object
 */
void from_json(const json &j, CublasFunction &fn) {
    auto str = j.dump();
    std::replace(str.begin(), str.end(), '\"', ' ');
    fn.set_function(str);
    auto name = j.at("name").get<std::string>();
    fn.set_name(name);
    auto m = j.at("m").get<int>();
    fn.set_m(m);
    auto n = j.at("n").get<int>();
    fn.set_n(n);
    auto k = j.at("k").get<int>();
    fn.set_k(k);
    auto transa = j.at("transa").get<int>();
    fn.set_transa(transa);
    auto transb = j.at("transb").get<int>();
    fn.set_transb(transb);
    fn.name2enum();
    try {
        auto batch_count = j.at("batchCount").get<int>();
        fn.set_batch_count(batch_count);
    } catch (std::exception &e) {
        fn.set_batch_count(1);
    }
    try {
        if (str.find("datatype") == std::string::npos) {
            fn.set_datatype("unknown");
        } else {
            auto datatype = j.at("datatype").get<std::string>();
            fn.set_datatype(datatype);
        }

    } catch (std::exception &e) {
        throw std::runtime_error("datatype is required for cublasGemmEx and cublasGemmStridedBatchedEx");
    }
    try {
        if (str.find("use_tensor_core") == std::string::npos) {
            fn.set_use_tensor_core(false);
        } else {
            auto use_tensor_core = j.at("use_tensor_core").get<bool>();
            fn.set_use_tensor_core(use_tensor_core);
        }
    } catch (std::exception &e) {
        throw std::runtime_error("use_tensor_core is required for cublasGemmEx and cublasGemmStridedBatchedEx");
    }
}

/**
 * @brief Get the cublas function pointer of a specific child class
 * @param  function         base class object of a CublasFunction, used to initialize the base part of the child class
 * object
 * @return CublasFunction*  return a base cublas function pointer of a specific child class
 */
CublasFunction *get_cublas_function_pointer(CublasFunction &function) {
    switch (function.get_e_name()) {
    case e_cublasSgemm:
        return new SgemmFunction(function);
    case e_cublasGemmEx:
        return new GemmExFunction(function);
    case e_cublasSgemmStridedBatched:
        return new SgemmStridedBatchedFunction(function);
    case e_cublasGemmStridedBatchedEx:
        return new GemmStridedBatchedExFunction(function);
    case e_cublasCgemm:
        return new CgemmFunction(function);
    case e_cublasCgemm3mStridedBatched:
        return new Cgemm3mStridedBatchedFunction(function);
    default:
        throw "invalid function name";
    }
}

/**
 * @brief run the entire process of benchmark according to cmd auguments
 *
 * first, read the para_info_json string representing the params for a cublas function
 * then get the pointer of the class object the specific cublas function
 * finally run the benchmark of the funcion
 *
 * @param  options  the cmd arguments of the application
 */
void run_benchmark(Options &options) {
    try {
        json function_config = json::parse(options.para_info_json);
        CublasFunction function = function_config.get<CublasFunction>();
        function.set_num_test(options.num_test);
        function.set_warm_up(options.warm_up);
        function.set_num_in_step(options.num_in_step);
        function.set_random_seed(options.random_seed);
        function.set_correctness(options.correctness_check);
        function.set_eps(options.eps);
        function.set_random_data(options.random_data);
        CublasFunction *p_function = get_cublas_function_pointer(function);
        p_function->benchmark();
        delete p_function;
    } catch (std::exception &e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
}
