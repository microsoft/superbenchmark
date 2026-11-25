// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "cudnn_function.h"

namespace cudnn_test {
/**
 * @brief Class of ConvolutionBackwardDataFunction
 * @tparam T1 input data type
 * @tparam T2 conv type
 */
template <typename T1, typename T2> class ConvolutionBackwardDataFunction : public CudnnFunction<T1, T2> {
    cudnnConvolutionBwdDataAlgo_t selected_algorithm_;
    /**
     * @brief Execute the kernel/function
     */
    void kernel_entry() override {
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(
            this->cudnn_handle, &this->alpha_, this->w_desc_.desc(), this->filter, this->x_desc_.desc(), this->x,
            this->conv_desc_.desc(), selected_algorithm_, this->fwd_workspace_, this->fwd_workspace_size_, &this->beta_,
            this->h_desc_.desc(), this->h));
    }
    /**
     * @brief Query workspace bytes for the selected backward-data algorithm
     */
    size_t get_workspace_size() const override {
        size_t bytes;
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
            this->cudnn_handle, this->w_desc_.desc(), this->x_desc_.desc(), this->conv_desc_.desc(),
            this->h_desc_.desc(), selected_algorithm_, &bytes));
        return bytes;
    }

    /**
     * @brief Resolve the backward-data algorithm before querying workspace or running the convolution
     */
    void select_algorithm(const CudnnExecutionPolicy &policy) override {
        selected_algorithm_ = policy.resolve_algorithm([this]() -> cudnnConvolutionBwdDataAlgo_t {
            int algo_count;
            cudnnConvolutionBwdDataAlgoPerf_t perf_results;
            CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardDataAlgorithm(
                this->cudnn_handle, this->w_desc_.desc(), this->x_desc_.desc(), this->conv_desc_.desc(),
                this->h_desc_.desc(), 1, &algo_count, &perf_results));
            return perf_results.algo;
        });
    }

  public:
    /**
     * @brief Construct a new Convolution Backward Data Function object
     */
    ConvolutionBackwardDataFunction() {}
    /**
     * @brief Construct a new Convolution Backward Data Function object
     * @param  config         base class CudnnConfig object
     */
    ConvolutionBackwardDataFunction(CudnnConfig &config) : CudnnFunction<T1, T2>(config) {}
};
} // namespace cudnn_test
