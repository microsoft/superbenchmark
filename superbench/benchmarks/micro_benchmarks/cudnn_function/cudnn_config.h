// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @file cudnn_config.h
 * @brief Unify the base class for cudnn function benchmark
 */

#pragma once

#include <unordered_map>

#include "cudnn_helper.h"

/**
 * @brief Enum of cudnn function name
 */
namespace cudnn_test {
enum cudnn_function_name_enum {
    e_cudnnConvolutionForward,
    e_cudnnConvolutionBackwardData,
    e_cudnnConvolutionBackwardFilter,
};

/**
 * @brief Map from cudnn function name to cudnn function name enum
 */
static std::unordered_map<std::string, cudnn_function_name_enum> const cudnn_function_name_string = {
    {"cudnnConvolutionForward", cudnn_function_name_enum::e_cudnnConvolutionForward},
    {"cudnnConvolutionBackwardData", cudnn_function_name_enum::e_cudnnConvolutionBackwardData},
    {"cudnnConvolutionBackwardFilter", cudnn_function_name_enum::e_cudnnConvolutionBackwardFilter},

};

/**
 * @brief Class to store the configuration of cudnn function params
 */
class CudnnConfig {
  protected:
    int num_test;
    int warm_up;
    int num_in_step;
    int random_seed;
    std::string name;
    cudnn_function_name_enum e_name;
    std::vector<int> input_dims_;
    std::vector<int> input_stride_;
    std::vector<int> filter_dims_;
    std::vector<int> output_dims_;
    std::vector<int> output_stride_;
    int algo_;
    int arrayLength_;
    std::vector<int> padA_;
    std::vector<int> filterStrideA_;
    std::vector<int> dilationA_;
    cudnnConvolutionMode_t mode_;
    bool use_tensor_core_;
    cudnnDataType_t input_type_;
    cudnnDataType_t conv_type_;
    std::string function_str_;

  public:
    void set_num_test(int num_test) { this->num_test = num_test; }
    void set_warm_up(int warm_up) { this->warm_up = warm_up; }
    void set_num_in_step(int num_in_step) { this->num_in_step = num_in_step; }
    void set_random_seed(int random_seed) { this->random_seed = random_seed; }
    void set_name(const std::string &n) { name = n; }
    void set_input_dims(const std::vector<int> &input_dims) { input_dims_ = input_dims; }
    void set_input_stride(const std::vector<int> &input_stride) { input_stride_ = input_stride; }
    void set_filter_dims(const std::vector<int> &filter_dims) { filter_dims_ = filter_dims; }
    void set_output_dims(const std::vector<int> &output_dims) { output_dims_ = output_dims; }
    void set_output_stride(const std::vector<int> &output_stride) { output_stride_ = output_stride; }
    void set_algo(int algo) { algo_ = algo; }
    void set_arrayLength(int arrayLength) { arrayLength_ = arrayLength; }
    void set_padA(const std::vector<int> &padA) { padA_ = padA; }
    void set_filterStrideA(const std::vector<int> &filterStrideA) { filterStrideA_ = filterStrideA; }
    void set_dilationA(const std::vector<int> &dilationA) { dilationA_ = dilationA; }
    void set_mode(const cudnnConvolutionMode_t &mode) { mode_ = mode; }
    void set_use_tensor_core(bool use_tensor_core) { use_tensor_core_ = use_tensor_core; }
    void set_input_type(const cudnnDataType_t &input_type) { input_type_ = input_type; }
    void set_conv_type(const cudnnDataType_t &conv_type) { input_type_ = conv_type; }
    void set_function(const std::string &str) { function_str_ = str; }

    std::vector<int> &get_input_dims() { return input_dims_; }
    std::vector<int> &get_input_stride() { return input_stride_; }
    std::vector<int> &get_filter_dims() { return filter_dims_; }
    std::vector<int> &get_output_dims() { return output_dims_; }
    std::vector<int> &get_output_stride() { return output_stride_; }
    int get_algo() { return algo_; }
    int get_arrayLength() { return arrayLength_; }
    std::vector<int> &get_padA() { return padA_; }
    std::vector<int> &get_filterStrideA() { return filterStrideA_; }
    std::vector<int> &get_dilationA() { return dilationA_; }
    cudnnConvolutionMode_t &get_mode() { return mode_; }
    bool get_use_tensor_core() { return use_tensor_core_; }
    cudnnDataType_t &get_input_type() { return input_type_; }
    cudnnDataType_t &get_conv_type() { return input_type_; }
    std::string &get_name() { return name; }
    cudnn_function_name_enum get_e_name() { return e_name; }
    std::string &get_str() { return function_str_; }
    /**
     * @brief Convert name string to enum name
     * @return cudnn_function_name_enum
     */
    cudnn_function_name_enum name2enum() {
        auto it = cudnn_function_name_string.find(this->name);
        if (it != cudnn_function_name_string.end()) {
            this->e_name = it->second;
            return e_name;
        } else {
            throw "ERROR: invalid input function name";
        }
    }
};
} // namespace cudnn_test