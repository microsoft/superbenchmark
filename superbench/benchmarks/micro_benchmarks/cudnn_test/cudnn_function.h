/**
 * @copyright Copyright (c) Microsoft Corporation
 * @file cudnn_function.h
 * @brief Implementation of specific cudnn function
 */

#pragma once

#include "cudnn_benchmark.h"

namespace cudnn_test {

template <typename T1, typename T2> class ConvolutionForwardFunction : public CudnnFunction<T1, T2> {
    cudnnConvolutionFwdAlgo_t fwd_algo_;
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        CHECK_CUDNN_ERROR(cudnnConvolutionForward(this->cudnn_handle, &this->alpha_, this->x_desc_.desc(), this->x,
                                                  this->w_desc_.desc(), this->filter, this->conv_desc_.desc(),
                                                  this->fwd_algo_, this->fwd_workspace_, this->fwd_workspace_size_,
                                                  &this->beta_, this->h_desc_.desc(), this->h));
    }
    virtual void get_workspace_size() {
        fwd_algo_ = cudnnConvolutionFwdAlgo_t(this->algo_);
        CHECK_CUDNN_ERROR(cudnnGetConvolutionForwardWorkspaceSize(
            this->cudnn_handle, this->x_desc_.desc(), this->w_desc_.desc(), this->conv_desc_.desc(),
            this->h_desc_.desc(), this->fwd_algo_, &this->fwd_workspace_size_));
    }

  public:
    /**
     * @brief Construct a new Sgemm Function object
     */
    ConvolutionForwardFunction() {}
    /**
     * @brief Construct a new Sgemm Function object
     * @param  function         base class cudnnFunction object
     */
    ConvolutionForwardFunction(CudnnConfig &config) : CudnnFunction<T1, T2>(config) {}
    /**
     * @brief Destroy the Sgemm Function object
     */
};

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

    virtual void get_workspace_size() {
        bwd_data_algo_ = cudnnConvolutionBwdDataAlgo_t(this->algo_);
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardDataWorkspaceSize(
            this->cudnn_handle, this->w_desc_.desc(), this->x_desc_.desc(), this->conv_desc_.desc(),
            this->h_desc_.desc(), this->bwd_data_algo_, &this->fwd_workspace_size_));
    }

  public:
    /**
     * @brief Construct a new Sgemm Function object
     */
    ConvolutionBackwardDataFunction() {}
    /**
     * @brief Construct a new Sgemm Function object
     * @param  function         base class cudnnFunction object
     */
    ConvolutionBackwardDataFunction(CudnnConfig &config) : CudnnFunction<T1, T2>(config) {}
    /**
     * @brief Destroy the Sgemm Function object
     */
};

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
    virtual void get_workspace_size() {
        bwd_filter_algo_ = cudnnConvolutionBwdFilterAlgo_t(this->algo_);
        CHECK_CUDNN_ERROR(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            this->cudnn_handle, this->x_desc_.desc(), this->h_desc_.desc(), this->conv_desc_.desc(),
            this->w_desc_.desc(), this->bwd_filter_algo_, &this->fwd_workspace_size_));
    }

  public:
    ConvolutionBackwardFilterFunction() {}
    ConvolutionBackwardFilterFunction(CudnnConfig &config) : CudnnFunction<T1, T2>(config) {}
};

} // namespace cudnn_test