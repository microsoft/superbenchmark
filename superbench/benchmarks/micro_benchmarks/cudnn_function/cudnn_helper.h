// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

/**
 * @brief  Header file for some functions related to cudnn
 */

#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <curand_kernel.h>

namespace cudnn_test {
/**
 * @brief check cudnn function running status and throw error str
 */
void throw_cudnn_err(cudnnStatus_t result, const char *func, const char *file, int const line);
#define CHECK_CUDNN_ERROR(x) throw_cudnn_err((x), #x, __FILE__, __LINE__)
/**
 * @brief check cudnn function running status and throw error str
 */
void check_cuda(cudaError_t result, const char *func, const char *file, int const line);
#define CUDA_SAFE_CALL(x) check_cuda((x), #x, __FILE__, __LINE__)
/**
 * @brief Cuda context init
 */
void cudnn_handle_init(cudnnHandle_t *cudnn_handle);
/**
 * @brief Cuda context free
 */
void cudnn_handle_free(cudnnHandle_t *cudnn_handle);
/**
 * @brief Malloc cuda memory and fill in rand value
 * @tparam T
 * @param  input            the pointer of input
 * @param  dims_            the shape of input
 * @param  random_seed      the random seed to generate random data
 */
template <typename T> void rand(T **input, std::vector<int> dims_, int random_seed);
/**
 * @brief Malloc cuda memory and fill in zero
 * @tparam T
 * @param  input            the pointer of input
 * @param  dims_            the shape of input
 */
template <typename T> void zeros(T **input, std::vector<int> dims_) {
    int size = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(T) * size));
    CUDA_SAFE_CALL(cudaMemset((void *)*input, 0, sizeof(T) * size));
}
/**
 * @brief Get cudnn tensor format
 * @tparam T
 * @param  tensor_format  cudnnTensorFormat_t
 */
template <typename T> void get_tensor_format(cudnnTensorFormat_t &tensor_format) {
    // For int8 inference, the supported format is NHWC
    if (std::is_same<T, uint8_t>::value) {
        tensor_format = CUDNN_TENSOR_NHWC;
    } else {
        tensor_format = CUDNN_TENSOR_NCHW;
    }
}
/**
 * @brief Get cudnn tensor data type
 * @tparam T
 * @param  type             cudnnDataType_t
 */
template <typename T> void get_tensor_type(cudnnDataType_t &type) {
    if (std::is_same<T, float>::value) {
        type = CUDNN_DATA_FLOAT;
    } else if (std::is_same<T, half>::value) {
        type = CUDNN_DATA_HALF;
    }
#if CUDNN_MAJOR >= 6
    else if (std::is_same<T, uint8_t>::value)
        type = CUDNN_DATA_INT8;
#endif
    else
        throw("unknown type in tensorDescriptor");
}

/**
 * @brief RAII wrapper for TensorDescriptorNd
 * @tparam T
 */
template <typename T> class TensorDescriptorNd {
    std::shared_ptr<cudnnTensorDescriptor_t> desc_;

    struct TensorDescriptorNdDeleter {
        void operator()(cudnnTensorDescriptor_t *desc) {
            CHECK_CUDNN_ERROR(cudnnDestroyTensorDescriptor(*desc));
            delete desc;
        }
    };

  public:
    TensorDescriptorNd() {}
    TensorDescriptorNd(const std::vector<int> &dim, const std::vector<int> &stride)
        : desc_(new cudnnTensorDescriptor_t, TensorDescriptorNdDeleter()) {
        cudnnDataType_t type;
        get_tensor_type<T>(type);

        CHECK_CUDNN_ERROR(cudnnCreateTensorDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetTensorNdDescriptor(*desc_, type, dim.size(), dim.data(), stride.data()));
    }

    cudnnTensorDescriptor_t desc() const { return *desc_; }
};

/**
 * @brief RAII wrapper for FilterDescriptorNd
 * @tparam T
 */
template <typename T> class FilterDescriptorNd {
    std::shared_ptr<cudnnFilterDescriptor_t> desc_;

    struct FilterDescriptorNdDeleter {
        void operator()(cudnnFilterDescriptor_t *desc) {
            CHECK_CUDNN_ERROR(cudnnDestroyFilterDescriptor(*desc));
            delete desc;
        }
    };

  public:
    FilterDescriptorNd() {}

    FilterDescriptorNd(const std::vector<int> &dim) : desc_(new cudnnFilterDescriptor_t, FilterDescriptorNdDeleter()) {
        cudnnTensorFormat_t tensor_format;
        get_tensor_format<T>(tensor_format);
        cudnnDataType_t type;
        get_tensor_type<T>(type);

        CHECK_CUDNN_ERROR(cudnnCreateFilterDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetFilterNdDescriptor(*desc_, type, tensor_format, dim.size(), &dim[0]));
    }

    cudnnFilterDescriptor_t desc() { return *desc_; }
};

/**
 * @brief RAII wrapper for ConvolutionDescriptor
 * @tparam T
 */
template <typename T> class ConvolutionDescriptor {
    std::shared_ptr<cudnnConvolutionDescriptor_t> desc_;

    struct ConvolutionDescriptorDeleter {
        void operator()(cudnnConvolutionDescriptor_t *desc) {
            CHECK_CUDNN_ERROR(cudnnDestroyConvolutionDescriptor(*desc));
            delete desc;
        }
    };

  public:
    ConvolutionDescriptor() {}
    ConvolutionDescriptor(int array_length, const std::vector<int> &padA, const std::vector<int> &filter_strideA,
                          const std::vector<int> &dilationA, cudnnConvolutionMode_t mode)
        : desc_(new cudnnConvolutionDescriptor_t, ConvolutionDescriptorDeleter()) {
        cudnnDataType_t type;
        get_tensor_type<T>(type);

        CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetConvolutionNdDescriptor(*desc_, array_length, padA.data(), filter_strideA.data(),
                                                          dilationA.data(), mode, type));
    }

    cudnnConvolutionDescriptor_t desc() const { return *desc_; };
};
} // namespace cudnn_test
