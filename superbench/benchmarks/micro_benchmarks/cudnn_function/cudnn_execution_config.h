// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <numeric>

#include "cudnn_config.h"

namespace cudnn_test {
// Storage and compute types remain in the existing CudnnFunction template dispatch.
struct CudnnConvolutionWorkload {
    const std::vector<int> input_dims, input_stride, filter_dims, output_dims, output_stride;
    const std::vector<int> padding, filter_stride, dilation;
    const cudnnConvolutionMode_t mode;
    const int array_length;

    explicit CudnnConvolutionWorkload(CudnnConfig &config)
        : input_dims(config.get_input_dims()), input_stride(config.get_input_stride()),
          filter_dims(config.get_filter_dims()), output_dims(config.get_output_dims()),
          output_stride(config.get_output_stride()), padding(config.get_padA()),
          filter_stride(config.get_filter_strideA()), dilation(config.get_dilationA()), mode(config.get_mode()),
          array_length(config.get_array_length()) {}
};

struct CudnnExecutionPolicy {
    const cudnnMathType_t math_type;
    const bool auto_algo;
    const int algorithm;

    // Automatic selection does not require a configured algorithm index.
    explicit CudnnExecutionPolicy(CudnnConfig &config)
        : math_type(config.get_use_tensor_op() ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH),
          auto_algo(config.get_auto_algo()), algorithm(auto_algo ? 0 : config.get_algo()) {}
};
} // namespace cudnn_test
