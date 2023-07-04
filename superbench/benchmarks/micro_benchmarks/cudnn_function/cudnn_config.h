// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>

#include "cudnn_helper.h"

namespace cudnn_test {
/**
 * @brief Enum of cudnn function name
 */
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
    int num_test;                    ///< the number of steps used to test and measure
    int warm_up;                     ///< the number of steps used to warm up
    int num_in_step;                 ///< the number of functions invoking in a step
    int random_seed;                 ///< the random seed used to generate random data
    std::string name;                ///< the name of the cudnn function
    cudnn_function_name_enum e_name; ///< enum cudnn functin name
    std::vector<int> input_dims_; ///< array of input dimension that contain the size of the tensor for every dimension
    std::vector<int>
        input_stride_; ///< array of input dimension that contain the stride of the tensor for every dimension
    std::vector<int>
        filter_dims_; ///< array of filter dimension that contain the size of the tensor for every dimension
    std::vector<int>
        output_dims_; ///< array of outpur dimension that contain the size of the tensor for every dimension
    std::vector<int>
        output_stride_; ///< array of output dimension that contain the stride of the tensor for every dimension
    int algo_;          ///< enumerant that specifies which convolution algorithm should be used to compute the results
    int array_length_;  ///< dimension of the convolution
    std::vector<int> padA_; ///< array of convolution dimension containing the zero-padding size for each dimension.
    std::vector<int>
        filter_strideA_;          ///< array of convolution dimension containing the filter stride for each dimension
    std::vector<int> dilationA_;  ///< array of dimension array_length containing the dilation factor for each dimension
    cudnnConvolutionMode_t mode_; ///< selects between CUDNN_CONVOLUTION and CUDNN_CROSS_CORRELATION
    bool use_tensor_op_;          ///< specify whether or not the use of tensor op is permitted in the library routines
                                  ///< associated with a given convolution descriptor
    cudnnDataType_t input_type_;  ///< selects the data type in which the computation will be done
    cudnnDataType_t conv_type_;   ///< selects the data type in which the convolution will be done
    std::string function_str_;    ///< the str representing the cudnn function with params
    bool auto_algo_;              ///< whether to use auto algo selection

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
    void set_array_length(int array_length) { array_length_ = array_length; }
    void set_padA(const std::vector<int> &padA) { padA_ = padA; }
    void set_filter_strideA(const std::vector<int> &filter_strideA) { filter_strideA_ = filter_strideA; }
    void set_dilationA(const std::vector<int> &dilationA) { dilationA_ = dilationA; }
    void set_mode(const cudnnConvolutionMode_t &mode) { mode_ = mode; }
    void set_use_tensor_op(bool use_tensor_op) { use_tensor_op_ = use_tensor_op; }
    void set_input_type(const cudnnDataType_t &input_type) { input_type_ = input_type; }
    void set_conv_type(const cudnnDataType_t &conv_type) { input_type_ = conv_type; }
    void set_function(const std::string &str) { function_str_ = str; }
    void set_auto_algo(bool auto_algo) { auto_algo_ = auto_algo; }

    std::vector<int> &get_input_dims() { return input_dims_; }
    std::vector<int> &get_input_stride() { return input_stride_; }
    std::vector<int> &get_filter_dims() { return filter_dims_; }
    std::vector<int> &get_output_dims() { return output_dims_; }
    std::vector<int> &get_output_stride() { return output_stride_; }
    int get_algo() { return algo_; }
    int get_array_length() { return array_length_; }
    std::vector<int> &get_padA() { return padA_; }
    std::vector<int> &get_filter_strideA() { return filter_strideA_; }
    std::vector<int> &get_dilationA() { return dilationA_; }
    cudnnConvolutionMode_t &get_mode() { return mode_; }
    bool get_use_tensor_op() { return use_tensor_op_; }
    cudnnDataType_t &get_input_type() { return input_type_; }
    cudnnDataType_t &get_conv_type() { return input_type_; }
    std::string &get_name() { return name; }
    cudnn_function_name_enum get_e_name() { return e_name; }
    std::string &get_function_str() { return function_str_; }
    bool get_auto_algo() { return auto_algo_; }
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
