// Copyright(c) Microsoft Corporation.
// Licensed under the MIT License.

// Helper functions of cuda and cudnn related

#pragma once

#include <iostream>
#include <memory>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <curand_kernel.h>

namespace cudnn_test {

cudnnHandle_t cudnn_handle_0 = NULL;

// Run cudnn function and check if succeed
void throw_cudnn_err(cudnnStatus_t result, const char *func, const char *file, int const line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        const char *msg = cudnnGetErrorString(result);
        std::stringstream safe_call_ss;
        safe_call_ss << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << msg;
        throw std::runtime_error(safe_call_ss.str());
    }
}
#define CHECK_CUDNN_ERROR(x) throw_cudnn_err((x), #x, __FILE__, __LINE__)

// Run cuda function and check if succeed
void check_cuda(cudaError_t result, const char *func, const char *file, int const line) {
    if (result != cudaSuccess) {
        const char *msg = cudaGetErrorString(result);
        std::stringstream safe_call_ss;
        safe_call_ss << func << " failed with error"
                     << "\nfile: " << file << "\nline: " << line << "\nmsg: " << msg;
        // Make sure we call CUDA Device Reset before exiting
        throw std::runtime_error(safe_call_ss.str());
    }
}
#define CUDA_SAFE_CALL(x) check_cuda((x), #x, __FILE__, __LINE__)

// Init cuda context
void cuda_init();
// Free cuda context
void cuda_free();
// Malloc cuda memory and fill in rand value
template <typename T> void rand(T **input, std::vector<int> dims_, curandGenerator_t curand_gen);
// Malloc cuda memory and fill in zero
template <typename T> void zeros(T **input, std::vector<int> dims_) {
    int size_ = std::accumulate(dims_.begin(), dims_.end(), 1, std::multiplies<int>());
    CUDA_SAFE_CALL(cudaMalloc((void **)input, sizeof(T) * size_));
    CUDA_SAFE_CALL(cudaMemset((void *)*input, 0, sizeof(T) * size_));
}

// Get cudnn tensor format
template <typename T> void get_tensor_format(cudnnTensorFormat_t &tensor_format) {
    // For int8 inference, the supported format is NHWC
    if (std::is_same<T, uint8_t>::value) {
        tensor_format = CUDNN_TENSOR_NHWC;
    } else {
        tensor_format = CUDNN_TENSOR_NCHW;
    }
}

// Get cudnn tensor data type
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
        throw("unknown type in tensorDecriptor");
}

// RAII wrapper for TensorDescriptorNd
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

// RAII wrapper for FilterDescriptorNd
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

// RAII wrapper for ConvolutionDescriptor
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
    ConvolutionDescriptor(int arrayLength, const std::vector<int> &padA, const std::vector<int> &filterStrideA,
                          const std::vector<int> &dilationA, cudnnConvolutionMode_t mode)
        : desc_(new cudnnConvolutionDescriptor_t, ConvolutionDescriptorDeleter()) {
        cudnnDataType_t type;
        get_tensor_type<T>(type);

        CHECK_CUDNN_ERROR(cudnnCreateConvolutionDescriptor(desc_.get()));
        CHECK_CUDNN_ERROR(cudnnSetConvolutionNdDescriptor(*desc_, arrayLength, padA.data(), filterStrideA.data(),
                                                          dilationA.data(), mode, type));
    }

    cudnnConvolutionDescriptor_t desc() const { return *desc_; };
};
} // namespace cudnn_test
