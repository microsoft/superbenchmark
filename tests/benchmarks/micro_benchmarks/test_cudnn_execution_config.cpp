// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "cudnn_function.h"

namespace {
int expected, searches, queries, executions;

template <typename Find> auto trace_find(Find find) {
    return [find](auto handle, auto a, auto b, auto conv, auto out, int requested, int *count, auto *result) {
        ++searches;
        auto status = find(handle, a, b, conv, out, requested, count, result);
        if (status == CUDNN_STATUS_SUCCESS && *count == 1 && result->status == CUDNN_STATUS_SUCCESS)
            expected = result->algo;
        return status;
    };
}

template <size_t I, typename Fn, typename... Args> cudnnStatus_t trace(int &calls, Fn fn, Args... args) {
    if (std::get<I>(std::tie(args...)) != expected)
        throw std::runtime_error("cuDNN call did not receive the selected algorithm");
    ++calls;
    return fn(args...);
}
} // namespace

// Observe the actual cuDNN calls in this test translation unit; forward all arguments to the real library.
#define cudnnFindConvolutionForwardAlgorithm trace_find(cudnnFindConvolutionForwardAlgorithm)
#define cudnnFindConvolutionBackwardDataAlgorithm trace_find(cudnnFindConvolutionBackwardDataAlgorithm)
#define cudnnFindConvolutionBackwardFilterAlgorithm trace_find(cudnnFindConvolutionBackwardFilterAlgorithm)
#define cudnnGetConvolutionForwardWorkspaceSize(...)                                                                   \
    trace<5>(queries, cudnnGetConvolutionForwardWorkspaceSize, __VA_ARGS__)
#define cudnnGetConvolutionBackwardDataWorkspaceSize(...)                                                              \
    trace<5>(queries, cudnnGetConvolutionBackwardDataWorkspaceSize, __VA_ARGS__)
#define cudnnGetConvolutionBackwardFilterWorkspaceSize(...)                                                            \
    trace<5>(queries, cudnnGetConvolutionBackwardFilterWorkspaceSize, __VA_ARGS__)
#define cudnnConvolutionForward(...) trace<7>(executions, cudnnConvolutionForward, __VA_ARGS__)
#define cudnnConvolutionBackwardData(...) trace<7>(executions, cudnnConvolutionBackwardData, __VA_ARGS__)
#define cudnnConvolutionBackwardFilter(...) trace<7>(executions, cudnnConvolutionBackwardFilter, __VA_ARGS__)
#include "convolution_backward_data.h"
#include "convolution_backward_filter.h"
#include "convolution_forward.h"

using namespace cudnn_test;

static void check_search_errors(CudnnConfig &config) {
    config.set_auto_algo(true);
    for (int count : {0, 1}) {
        auto find = [count](int, int *n, cudnnConvolutionFwdAlgoPerf_t *result) {
            *n = count;
            result->status = count ? CUDNN_STATUS_ALLOC_FAILED : CUDNN_STATUS_SUCCESS;
            result->algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
            return CUDNN_STATUS_SUCCESS;
        };
        try {
            select_convolution_algorithm<cudnnConvolutionFwdAlgoPerf_t>(config, find);
        } catch (const std::runtime_error &e) {
            if (std::string(e.what()).find(count ? cudnnGetErrorString(CUDNN_STATUS_ALLOC_FAILED) : "got 0") !=
                std::string::npos)
                continue;
            throw;
        }
        throw std::runtime_error("accepted an empty or failed cuDNN search result");
    }
}

template <typename Op> void check_calls(CudnnConfig &config) {
    for (bool automatic : {false, true}) {
        config.set_auto_algo(automatic);
        expected = automatic ? -1 : config.get_algo();
        searches = queries = executions = 0;
        Op(config).benchmark();
        if (searches != static_cast<int>(automatic) || queries != 1 || executions != 2)
            throw std::runtime_error("unexpected number of cuDNN searches, workspace queries or convolutions");
    }
}

int main(int argc, char **argv) {
    try {
        CudnnConfig config{};
        if (argc == 1) {
            check_search_errors(config);
            return 0;
        }
        if (argc != 2 || std::string(argv[1]) != "--gpu")
            throw std::invalid_argument("usage: cudnn_config_test [--gpu]");
        config.set_input_dims({2, 8, 5, 5});
        config.set_output_dims(config.get_input_dims());
        config.set_input_stride({200, 25, 5, 1});
        config.set_output_stride(config.get_input_stride());
        config.set_filter_dims({8, 8, 3, 3});
        config.set_padA({1, 2});
        config.set_filter_strideA({1, 1});
        config.set_dilationA({1, 2});
        config.set_array_length(2);
        config.set_mode(CUDNN_CROSS_CORRELATION);
        config.set_algo(1);
        config.set_warm_up(1);
        config.set_num_test(1);
        config.set_num_in_step(1);
        check_calls<ConvolutionForwardFunction<float, float>>(config);
        check_calls<ConvolutionBackwardDataFunction<float, float>>(config);
        check_calls<ConvolutionBackwardFilterFunction<float, float>>(config);
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
