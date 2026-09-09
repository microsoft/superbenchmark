#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#include <cublas_v2.h>

#include "cudnn_config.h"

namespace cudnn_test {
class CudnnReference {
    struct DeviceBuffer {
        void *pointer = nullptr;
        explicit DeviceBuffer(size_t bytes) { CUDA_SAFE_CALL(cudaMalloc(&pointer, bytes)); }
        ~DeviceBuffer() { cudaFree(pointer); }
        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    };

    struct BlasHandle {
        cublasHandle_t handle = nullptr;
        BlasHandle() { check(cublasCreate(&handle)); }
        ~BlasHandle() { cublasDestroy(handle); }
        BlasHandle(const BlasHandle &) = delete;
        BlasHandle &operator=(const BlasHandle &) = delete;
    };

    struct Sample {
        std::vector<float> input;
        std::vector<float> gradient;
        std::vector<double> reference;
    };

    CudnnConfig &config_;
    void *input_;
    void *filter_;
    void *gradient_;
    std::vector<Sample> samples_;
    size_t filter_count_;
    size_t failures_ = 0;
    double maximum_ratio_ = 0;

    static void check(cublasStatus_t status) {
        if (status != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("FP64 reference cuBLAS failure: " + std::to_string(static_cast<int>(status)));
        }
    }

    static size_t count(const std::vector<int> &dimensions) {
        return std::accumulate(dimensions.begin(), dimensions.end(), size_t{1}, std::multiplies<size_t>());
    }

    bool is_half() const { return config_.get_input_type() == CUDNN_DATA_HALF; }

    std::vector<float> download(void *pointer, size_t elements) const {
        std::vector<float> values(elements);
        if (is_half()) {
            std::vector<half> stored(elements);
            CUDA_SAFE_CALL(cudaMemcpy(stored.data(), pointer, elements * sizeof(half), cudaMemcpyDeviceToHost));
            std::transform(stored.begin(), stored.end(), values.begin(),
                           [](half value) { return __half2float(value); });
        } else {
            CUDA_SAFE_CALL(cudaMemcpy(values.data(), pointer, elements * sizeof(float), cudaMemcpyDeviceToHost));
        }
        return values;
    }

    void upload(void *pointer, const std::vector<float> &values) const {
        if (is_half()) {
            std::vector<half> stored(values.size());
            std::transform(values.begin(), values.end(), stored.begin(),
                           [](float value) { return __float2half(value); });
            CUDA_SAFE_CALL(cudaMemcpy(pointer, stored.data(), stored.size() * sizeof(half), cudaMemcpyHostToDevice));
        } else {
            CUDA_SAFE_CALL(cudaMemcpy(pointer, values.data(), values.size() * sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    void fill(Sample &sample, unsigned seed, const std::string &pattern) const {
        std::mt19937 generator(seed);
        for (size_t tensor = 0; tensor < 3; ++tensor) {
            size_t elements = tensor == 0 ? sample.input.size() : tensor == 1 ? filter_count_ : sample.gradient.size();
            for (size_t index = 0; index < elements; ++index) {
                float unit = static_cast<float>((static_cast<double>(generator()) + 0.5) / 4294967296.0 - 0.5);
                float value = unit * 0.5f;
                if (pattern == "scaled") {
                    value = std::ldexp(value, static_cast<int>(generator() % 9) - 4);
                } else if (pattern == "cancellation") {
                    value = (tensor == 0 && index % 2 != 0 ? -1.f : 1.f) * (0.125f + unit * 0.01f);
                }
                if (is_half()) {
                    value = __half2float(__float2half(value));
                }
                if (tensor == 0) {
                    sample.input[index] = value;
                } else if (tensor == 2) {
                    sample.gradient[index] = value;
                }
            }
        }
    }

    double scalar(const Sample &sample, size_t element) const {
        const auto &input = config_.get_input_dims();
        const auto &output = config_.get_output_dims();
        const auto &filter = config_.get_filter_dims();
        const int kernel_x = element % filter[3];
        element /= filter[3];
        const int kernel_y = element % filter[2];
        element /= filter[2];
        const int channel = element % filter[1];
        const int channel_out = element / filter[1];
        double result = 0;
        for (int batch = 0; batch < input[0]; ++batch) {
            for (int output_y = 0; output_y < output[2]; ++output_y) {
                const int64_t input_y = int64_t{output_y} * config_.get_filter_strideA()[0] - config_.get_padA()[0] +
                                        int64_t{kernel_y} * config_.get_dilationA()[0];
                for (int output_x = 0; output_x < output[3]; ++output_x) {
                    const int64_t input_x = int64_t{output_x} * config_.get_filter_strideA()[1] -
                                            config_.get_padA()[1] + int64_t{kernel_x} * config_.get_dilationA()[1];
                    if (input_y >= 0 && input_y < input[2] && input_x >= 0 && input_x < input[3]) {
                        size_t source =
                            ((static_cast<size_t>(batch) * input[1] + channel) * input[2] + input_y) * input[3] +
                            input_x;
                        size_t gradient =
                            ((static_cast<size_t>(batch) * output[1] + channel_out) * output[2] + output_y) *
                                output[3] +
                            output_x;
                        result += static_cast<double>(sample.input[source]) * sample.gradient[gradient];
                    }
                }
            }
        }
        return result;
    }

    void reference(Sample &sample) const {
        const auto &input = config_.get_input_dims();
        const auto &output = config_.get_output_dims();
        const auto &filter = config_.get_filter_dims();
        const int features = filter[1] * filter[2] * filter[3];
        const int positions = output[2] * output[3];
        const int tile = std::max(1, std::min(positions, 1048576 / features));
        std::vector<double> patches(static_cast<size_t>(features) * tile);
        std::vector<double> gradients(static_cast<size_t>(output[1]) * tile);
        DeviceBuffer device_patches(patches.size() * sizeof(double));
        DeviceBuffer device_gradients(gradients.size() * sizeof(double));
        DeviceBuffer device_result(filter_count_ * sizeof(double));
        BlasHandle blas;
        CUDA_SAFE_CALL(cudaMemset(device_result.pointer, 0, filter_count_ * sizeof(double)));
        const double alpha = 1, beta = 1;
        for (int batch = 0; batch < input[0]; ++batch) {
            for (int start = 0; start < positions;) {
                int width = std::min(tile, positions - start);
                for (int position = 0; position < width; ++position) {
                    int output_y = (start + position) / output[3];
                    int output_x = (start + position) % output[3];
                    for (int feature = 0; feature < features; ++feature) {
                        int kernel_x = feature % filter[3];
                        int kernel_y = feature / filter[3] % filter[2];
                        int channel = feature / (filter[2] * filter[3]);
                        int64_t input_y = int64_t{output_y} * config_.get_filter_strideA()[0] - config_.get_padA()[0] +
                                          int64_t{kernel_y} * config_.get_dilationA()[0];
                        int64_t input_x = int64_t{output_x} * config_.get_filter_strideA()[1] - config_.get_padA()[1] +
                                          int64_t{kernel_x} * config_.get_dilationA()[1];
                        double value = 0;
                        if (input_y >= 0 && input_y < input[2] && input_x >= 0 && input_x < input[3]) {
                            size_t source =
                                ((static_cast<size_t>(batch) * input[1] + channel) * input[2] + input_y) * input[3] +
                                input_x;
                            value = sample.input[source];
                        }
                        patches[feature + static_cast<size_t>(features) * position] = value;
                    }
                    for (int channel = 0; channel < output[1]; ++channel) {
                        gradients[position + static_cast<size_t>(width) * channel] =
                            sample.gradient[(static_cast<size_t>(batch) * output[1] + channel) * positions + start +
                                            position];
                    }
                }
                CUDA_SAFE_CALL(cudaMemcpy(device_patches.pointer, patches.data(),
                                          static_cast<size_t>(features) * width * sizeof(double),
                                          cudaMemcpyHostToDevice));
                CUDA_SAFE_CALL(cudaMemcpy(device_gradients.pointer, gradients.data(),
                                          static_cast<size_t>(output[1]) * width * sizeof(double),
                                          cudaMemcpyHostToDevice));
                check(cublasDgemm(blas.handle, CUBLAS_OP_N, CUBLAS_OP_N, features, output[1], width, &alpha,
                                  static_cast<double *>(device_patches.pointer), features,
                                  static_cast<double *>(device_gradients.pointer), width, &beta,
                                  static_cast<double *>(device_result.pointer), features));
                start += width;
            }
        }
        sample.reference.resize(filter_count_);
        CUDA_SAFE_CALL(cudaMemcpy(sample.reference.data(), device_result.pointer, filter_count_ * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        const size_t checks = std::min(filter_count_, size_t{32});
        for (size_t index = 0; index < checks; ++index) {
            const size_t element = checks == 1 ? 0 : index * (filter_count_ - 1) / (checks - 1);
            const double expected = scalar(sample, element);
            const double actual = sample.reference[element];
            if (!std::isfinite(expected) || !std::isfinite(actual) ||
                std::abs(expected - actual) > 1e-10 * (1 + std::abs(expected))) {
                throw std::runtime_error("FP64 reference disagrees with independent CPU convolution");
            }
        }
    }

  public:
    CudnnReference(CudnnConfig &config, void *input, void *filter, void *gradient)
        : config_(config), input_(input), filter_(filter), gradient_(gradient),
          filter_count_(count(config.get_filter_dims())) {
        Sample actual{
            download(input_, count(config.get_input_dims())), download(gradient_, count(config.get_output_dims())), {}};
        for (const auto &probe : {std::make_pair(58613u, "scaled"), std::make_pair(91817u, "scaled"),
                                  std::make_pair(104729u, "uniform"), std::make_pair(130363u, "cancellation")}) {
            Sample sample{std::vector<float>(actual.input.size()), std::vector<float>(actual.gradient.size()), {}};
            fill(sample, probe.first, probe.second);
            reference(sample);
            samples_.push_back(std::move(sample));
        }
        reference(actual);
        samples_.push_back(std::move(actual));
    }

    template <typename Execute> bool accepts(Execute execute) {
        failures_ = 0;
        maximum_ratio_ = 0;
        for (const auto &sample : samples_) {
            upload(input_, sample.input);
            upload(gradient_, sample.gradient);
            CUDA_SAFE_CALL(cudaMemset(filter_, 0xff, filter_count_ * (is_half() ? sizeof(half) : sizeof(float))));
            execute();
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            auto result = download(filter_, filter_count_);
            for (size_t element = 0; element < result.size(); ++element) {
                double tolerance =
                    0.0005 + (0.0005 + (is_half() ? 1.0 / 2048 : 0)) * std::abs(sample.reference[element]);
                double ratio = std::abs(static_cast<double>(result[element]) - sample.reference[element]) / tolerance;
                if (!std::isfinite(result[element]) || !std::isfinite(sample.reference[element]) || ratio > 1) {
                    ++failures_;
                }
                maximum_ratio_ =
                    std::max(maximum_ratio_, std::isfinite(ratio) ? ratio : std::numeric_limits<double>::infinity());
            }
            if (failures_ != 0) {
                return false;
            }
        }
        return true;
    }

    size_t failures() const { return failures_; }
    double maximum_ratio() const { return maximum_ratio_; }
    size_t checked_elements() const { return filter_count_ * samples_.size(); }
    CudnnReference(const CudnnReference &) = delete;
    CudnnReference &operator=(const CudnnReference &) = delete;
};
} // namespace cudnn_test