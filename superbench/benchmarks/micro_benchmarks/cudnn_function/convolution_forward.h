// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "cudnn_function.h"

namespace cudnn_test {
/**
 * @brief Class of ConvolutionForwardFunction
 * @tparam T1 input data type
 * @tparam T2 conv type
 */
template <typename T1, typename T2> class ConvolutionForwardFunction : public CudnnFunction<T1, T2> {
    cudnnConvolutionFwdAlgo_t selected_algorithm_;
    /**
     * @brief Execute the kernel/function
     */
    void kernel_entry() override {
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(this->cudnn_handle, &this->alpha_, this->x_desc_.desc(), this->x,
                                                  this->w_desc_.desc(), this->filter, this->conv_desc_.desc(),
                                                  selected_algorithm_, this->fwd_workspace_, this->fwd_workspace_size_,
                                                  &this->beta_, this->h_desc_.desc(), this->h));
    }
    /**
     * @brief Query workspace bytes for the selected forward algorithm
     */
    size_t get_workspace_size() const override {
        size_t bytes;
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(this->cudnn_handle, this->x_desc_.desc(),
                                                                  this->w_desc_.desc(), this->conv_desc_.desc(),
                                                                  this->h_desc_.desc(), selected_algorithm_, &bytes));
        return bytes;
    }
    /**
     * @brief Resolve the forward algorithm before querying workspace or running the convolution
     */
    void select_algorithm() override {
        selected_algorithm_ = select_convolution_algorithm<cudnnConvolutionFwdAlgoPerf_t>(
            *this, cudnnFindConvolutionForwardAlgorithm, this->cudnn_handle, this->x_desc_.desc(), this->w_desc_.desc(),
            this->conv_desc_.desc(), this->h_desc_.desc());
    }

  public:
    /**
     * @brief Construct a new Convolution Forward Function object
     */
    ConvolutionForwardFunction() {}
    /**
     * @brief Construct a new Convolution Forward Function object
     * @param  config         base class CudnnConfig object
     */
    ConvolutionForwardFunction(CudnnConfig &config) : CudnnFunction<T1, T2>(config) {}
};
} // namespace cudnn_test
