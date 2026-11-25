// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "cudnn_function.h"

namespace cudnn_test {
/**
 * @brief Class of ConvolutionBackwardFilterFunction
 * @tparam T1 input data type
 * @tparam T2 conv type
 */
template <typename T1, typename T2> class ConvolutionBackwardFilterFunction : public CudnnFunction<T1, T2> {
    cudnnConvolutionBwdFilterAlgo_t selected_algorithm_;
    /**
     * @brief Execute the kernel/function
     */
    void kernel_entry() override {
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(
            this->cudnn_handle, &this->alpha_, this->x_desc_.desc(), this->x, this->h_desc_.desc(), this->h,
            this->conv_desc_.desc(), selected_algorithm_, this->fwd_workspace_, this->fwd_workspace_size_, &this->beta_,
            this->w_desc_.desc(), this->filter));
    }
    /**
     * @brief Query workspace bytes for the selected backward-filter algorithm
     */
    size_t get_workspace_size() const override {
        size_t bytes;
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            this->cudnn_handle, this->x_desc_.desc(), this->h_desc_.desc(), this->conv_desc_.desc(),
            this->w_desc_.desc(), selected_algorithm_, &bytes));
        return bytes;
    }
    /**
     * @brief Resolve the backward-filter algorithm before querying workspace or running the convolution
     */
    void select_algorithm(const CudnnExecutionPolicy &policy) override {
        selected_algorithm_ = policy.resolve_algorithm([this]() -> cudnnConvolutionBwdFilterAlgo_t {
            int algo_count;
            cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
            CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardFilterAlgorithm(
                this->cudnn_handle, this->x_desc_.desc(), this->h_desc_.desc(), this->conv_desc_.desc(),
                this->w_desc_.desc(), 1, &algo_count, &perf_results));
            return perf_results.algo;
        });
    }

  public:
    /**
     * @brief Construct a new Convolution Backward Filter Function object
     */
    ConvolutionBackwardFilterFunction() {}
    /**
     * @brief Construct a new Convolution Backward Filter Function object
     * @param  config         base class CudnnConfig object
     */
    ConvolutionBackwardFilterFunction(CudnnConfig &config) : CudnnFunction<T1, T2>(config) {}
};
} // namespace cudnn_test
