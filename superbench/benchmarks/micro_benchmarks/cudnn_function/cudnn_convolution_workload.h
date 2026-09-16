#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "cudnn_config.h"

namespace cudnn_test {
class CudnnConvolutionWorkload {
  public:
    const std::string name;
    const std::vector<int> input_dims, input_stride, filter_dims, output_dims, output_stride;
    const std::vector<int> padding, filter_stride, dilation;
    const cudnnDataType_t input_type, compute_type;
    const cudnnConvolutionMode_t mode;
    const int array_length;

    explicit CudnnConvolutionWorkload(CudnnConfig &config)
        : name(config.get_name()), input_dims(config.get_input_dims()), input_stride(config.get_input_stride()),
          filter_dims(config.get_filter_dims()), output_dims(config.get_output_dims()),
          output_stride(config.get_output_stride()), padding(config.get_padA()),
          filter_stride(config.get_filter_strideA()), dilation(config.get_dilationA()),
          input_type(config.get_input_type()), compute_type(config.get_conv_type()), mode(config.get_mode()),
          array_length(config.get_array_length()) {}

    static std::vector<int64_t> packed_strides(const std::vector<int> &dimensions) {
        if (dimensions.size() != 4) {
            throw std::invalid_argument("prepared execution requires four-dimensional tensors");
        }
        std::vector<int64_t> strides(4, 1);
        int64_t elements = 1;
        for (size_t axis = dimensions.size(); axis-- > 0;) {
            if (dimensions[axis] <= 0 || elements > std::numeric_limits<int>::max() / dimensions[axis]) {
                throw std::invalid_argument("invalid or oversized prepared tensor");
            }
            strides[axis] = elements;
            elements *= dimensions[axis];
        }
        return strides;
    }

    void validate_backward_filter() const {
        if (name != "cudnnConvolutionBackwardFilter" || array_length != 2 || compute_type != CUDNN_DATA_FLOAT ||
            (input_type != CUDNN_DATA_FLOAT && input_type != CUDNN_DATA_HALF) || mode != CUDNN_CROSS_CORRELATION) {
            throw std::invalid_argument(
                "prepared execution requires 2D backward-filter, FP32 compute and FP32/FP16 storage");
        }
        for (auto tensors : {std::make_pair(input_dims, input_stride), std::make_pair(output_dims, output_stride)}) {
            auto strides = packed_strides(tensors.first);
            if (tensors.second.size() != strides.size() ||
                !std::equal(strides.begin(), strides.end(), tensors.second.begin())) {
                throw std::invalid_argument("prepared execution requires packed NCHW tensors");
            }
        }
        packed_strides(filter_dims);
        if (padding.size() != 2 || filter_stride.size() != 2 || dilation.size() != 2 ||
            input_dims[0] != output_dims[0] || input_dims[1] != filter_dims[1] || output_dims[1] != filter_dims[0]) {
            throw std::invalid_argument("inconsistent prepared convolution dimensions");
        }
        for (size_t axis = 0; axis < 2; ++axis) {
            const int64_t pad = padding[axis];
            const int64_t stride = filter_stride[axis];
            const int64_t spacing = dilation[axis];
            const int64_t extent = input_dims[axis + 2] + 2 * pad - spacing * (filter_dims[axis + 2] - 1) - 1;
            if (pad < 0 || stride <= 0 || spacing <= 0 || extent < 0 || output_dims[axis + 2] != extent / stride + 1) {
                throw std::invalid_argument("inconsistent prepared convolution output shape");
            }
        }
    }
};
} // namespace cudnn_test
