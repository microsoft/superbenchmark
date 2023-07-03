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
    cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        CHECK_CUDNN_ERROR(cudnnConvolutionBackwardData(
            this->cudnn_handle, &this->alpha_, this->w_desc_.desc(), this->filter, this->x_desc_.desc(), this->x,
            this->conv_desc_.desc(), this->bwd_data_algo_, this->fwd_workspace_, this->fwd_workspace_size_,
            &this->beta_, this->h_desc_.desc(), this->h));
    }
    /**
     * @brief Get and set convolution algorithm and workspace size used in cudnn convolution functions
     */
    virtual void get_workspace_size() {
        bwd_data_algo_ = cudnnConvolutionBwdDataAlgo_t(this->algo_);
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
            this->cudnn_handle, this->w_desc_.desc(), this->x_desc_.desc(), this->conv_desc_.desc(),
            this->h_desc_.desc(), this->bwd_data_algo_, &this->fwd_workspace_size_));
    }

    /**
     * @brief Find the best algorithm for cudnn convolution functions
     */
    virtual void find_best_algo() {
        int algo_count;
        cudnnConvolutionBwdDataAlgoPerf_t perf_results;
        CHECK_CUDNN_ERROR(cudnnFindConvolutionBackwardDataAlgorithm(
            this->cudnn_handle, this->w_desc_.desc(), this->x_desc_.desc(), this->conv_desc_.desc(),
            this->h_desc_.desc(), 1, &algo_count, &perf_results));
        this->algo_ = perf_results.algo;
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
