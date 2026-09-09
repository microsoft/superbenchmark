// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "cudnn_function.h"
#include "cudnn_prepared_plan.h"
#include <nlohmann/json.hpp>

namespace cudnn_test {
/**
 * @brief Class of ConvolutionBackwardFilterFunction
 * @tparam T1 input data type
 * @tparam T2 conv type
 */
template <typename T1, typename T2> class ConvolutionBackwardFilterFunction : public CudnnFunction<T1, T2> {
    cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
    std::unique_ptr<CudnnPreparedPlan> prepared_plan_;
    double plan_build_ms_ = 0;

    void prepare_execution() override {
        if (this->get_prepared()) {
            auto start = std::chrono::steady_clock::now();
            prepared_plan_.reset(new CudnnPreparedPlan(this->cudnn_handle, *this, this->x, this->filter, this->h));
            plan_build_ms_ =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        }
    }

    void print_execution_info(double setup_ms, double first_call_ms, double benchmark_ms) override {
        nlohmann::json metadata = {{"execution_mode", "prepared"},
                                   {"policy", "deterministic-v1"},
                                   {"cudnn_version", cudnnGetVersion()},
                                   {"plan_build_ms", plan_build_ms_},
                                   {"setup_ms", setup_ms},
                                   {"first_call_ms", first_call_ms},
                                   {"benchmark_ms", benchmark_ms},
                                   {"plan", nlohmann::json::parse(prepared_plan_->json())}};
        std::cout << "[prepared_plan]: " << metadata.dump() << std::endl;
    }
    /**
     * @brief Execute the kernel/function
     */
    virtual void kernel_entry() {
        if (prepared_plan_) {
            prepared_plan_->execute(this->cudnn_handle);
            return;
        }
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
