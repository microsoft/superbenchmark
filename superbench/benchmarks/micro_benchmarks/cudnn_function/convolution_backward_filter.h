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
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardFilter(
            this->cudnn_handle, &this->alpha_, this->x_desc_.desc(), this->x, this->h_desc_.desc(), this->h,
            this->conv_desc_.desc(), this->bwd_filter_algo_, this->fwd_workspace_, this->fwd_workspace_size_,
            &this->beta_, this->w_desc_.desc(), this->filter));
    }
    /**
     * @brief Get and set convolution algorithm and workspace size used in cudnn convolution functions
     */
    virtual void get_workspace_size() {
        bwd_filter_algo_ = cudnnConvolutionBwdFilterAlgo_t(this->algo_);
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            this->cudnn_handle, this->x_desc_.desc(), this->h_desc_.desc(), this->conv_desc_.desc(),
            this->w_desc_.desc(), this->bwd_filter_algo_, &this->fwd_workspace_size_));
    }
    /**
     * @brief Find the best algorithm for cudnn convolution functions
     */
    virtual void find_best_algo() {
        int algo_count;
        cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
        CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardFilterAlgorithm(
            this->cudnn_handle, this->x_desc_.desc(), this->h_desc_.desc(), this->conv_desc_.desc(),
            this->w_desc_.desc(), 1, &algo_count, &perf_results));
        this->algo_ = perf_results.algo;
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
