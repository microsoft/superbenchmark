#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "cudnn_config.h"
#include "cudnn_reference.h"

namespace cudnn_test {
class CudnnPreparedPlan {
    nlohmann::json verification_;
    static std::vector<int64_t> packed_strides(const std::vector<int> &dimensions) {
        if (dimensions.size() != 4) {
            throw std::invalid_argument("prepared execution requires four-dimensional tensors");
        }
        std::vector<int64_t> strides(4, 1);
        int64_t elements = 1;
        for (size_t axis = dimensions.size(); axis-- > 0;) {
            if (dimensions[axis] <= 0 || elements > std::numeric_limits<int>::max() / dimensions[axis]) {
                throw std::invalid_argument("invalid or oversized prepared tensor");
            }
            strides[axis] = elements;
            elements *= dimensions[axis];
        }
        return strides;
    }

#if CUDNN_VERSION >= 8900
    struct Descriptors {
        std::vector<cudnnBackendDescriptor_t> values;
        ~Descriptors() {
            for (auto descriptor = values.rbegin(); descriptor != values.rend(); ++descriptor) {
                cudnnBackendDestroyDescriptor(*descriptor);
            }
        }
    } descriptors_;
    struct Workspace {
        void *pointer = nullptr;
        ~Workspace() { cudaFree(pointer); }
        void reset(size_t bytes) {
            void *previous = pointer;
            pointer = nullptr;
            CUDA_SAFE_CALL(cudaFree(previous));
            if (bytes != 0) {
                CUDA_SAFE_CALL(cudaMalloc(&pointer, bytes));
            }
        }
    } workspace_;
    cudnnBackendDescriptor_t plan_ = nullptr;
    cudnnBackendDescriptor_t pack_ = nullptr;
    int64_t workspace_bytes_ = 0;
    int64_t engine_index_ = -1;

    cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type) {
        cudnnBackendDescriptor_t descriptor = nullptr;
        CHECK_CUDNN_ERROR(cudnnBackendCreateDescriptor(type, &descriptor));
        try {
            descriptors_.values.push_back(descriptor);
        } catch (...) {
            cudnnBackendDestroyDescriptor(descriptor);
            throw;
        }
        return descriptor;
    }

    template <typename Value>
    void set(cudnnBackendDescriptor_t descriptor, cudnnBackendAttributeName_t name, cudnnBackendAttributeType_t type,
             const Value &value) {
        CHECK_CUDNN_ERROR(cudnnBackendSetAttribute(descriptor, name, type, 1, &value));
    }

    cudnnBackendDescriptor_t tensor(int64_t uid, const std::vector<int> &dimensions, cudnnDataType_t type) {
        auto descriptor = create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
        auto strides = packed_strides(dimensions);
        std::vector<int64_t> sizes(dimensions.begin(), dimensions.end());
        set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, uid);
        set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, type);
        set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, int64_t{16});
        CHECK_CUDNN_ERROR(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64,
                                                   sizes.size(), sizes.data()));
        CHECK_CUDNN_ERROR(cudnnBackendSetAttribute(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64,
                                                   strides.size(), strides.data()));
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(descriptor));
        return descriptor;
    }

    void bind(void *input, void *filter, void *output) {
        workspace_.reset(static_cast<size_t>(workspace_bytes_));
        pack_ = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        int64_t ids[] = {101, 102, 103};
        void *pointers[] = {input, filter, output};
        CHECK_CUDNN_ERROR(
            cudnnBackendSetAttribute(pack_, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, ids));
        CHECK_CUDNN_ERROR(
            cudnnBackendSetAttribute(pack_, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers));
        set(pack_, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, workspace_.pointer);
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(pack_));
    }

    void select(cudnnHandle_t handle, cudnnBackendDescriptor_t graph, CudnnConfig &config, void *input, void *filter,
                void *output) {
        CudnnReference reference(config, input, filter, output);
        verification_ = {{"policy", "full-output-v1"},
                         {"passed", false},
                         {"inputs", 5},
                         {"reference", "cuBLAS-FP64-with-CPU-crosschecks"},
                         {"atol", 0.0005},
                         {"rtol", 0.0005 + (config.get_input_type() == CUDNN_DATA_HALF ? 1.0 / 2048 : 0)},
                         {"checked_elements", reference.checked_elements()},
                         {"rejected", nlohmann::json::array()}};
        std::vector<std::string> attempted;
        for (auto mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
            auto heuristic = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            set(heuristic, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, graph);
            set(heuristic, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, mode);
            CHECK_CUDNN_ERROR(cudnnBackendFinalize(heuristic));
            int64_t count = 0;
            CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
            if (count < 0 || count > 256) {
                throw std::runtime_error("unexpected prepared-plan candidate count");
            }
            std::vector<cudnnBackendDescriptor_t> candidates;
            for (int64_t index = 0; index < count; ++index) {
                candidates.push_back(create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
            }
            if (count == 0) {
                continue;
            }
            CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(heuristic, CUDNN_ATTR_ENGINEHEUR_RESULTS,
                                                       CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &count,
                                                       candidates.data()));
            if (count < 0 || static_cast<size_t>(count) > candidates.size()) {
                throw std::runtime_error("unexpected prepared-plan result count");
            }
            candidates.resize(static_cast<size_t>(count));
            for (auto candidate : candidates) {
                auto engine = create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
                int64_t returned = 0;
                CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(candidate, CUDNN_ATTR_ENGINECFG_ENGINE,
                                                           CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned, &engine));
                std::vector<cudnnBackendNumericalNote_t> notes(CUDNN_NUMERICAL_NOTE_TYPE_COUNT);
                CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
                                                           CUDNN_TYPE_NUMERICAL_NOTE, notes.size(), &returned,
                                                           notes.data()));
                notes.resize(returned);
                const bool tensor_core =
                    std::find(notes.begin(), notes.end(), CUDNN_NUMERICAL_NOTE_TENSOR_CORE) != notes.end();
                if (tensor_core != config.get_use_tensor_op() ||
                    std::any_of(notes.begin(), notes.end(), [](cudnnBackendNumericalNote_t note) {
                        return note == CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC ||
                               note == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
                               note == CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION;
                    })) {
                    continue;
                }
                auto plan = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
                set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, handle);
                set(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, candidate);
                auto status = cudnnBackendFinalize(plan);
                if (status != CUDNN_STATUS_SUCCESS) {
#if CUDNN_MAJOR >= 9
                    if (CUDNN_STATUS_CATEGORY(status) != CUDNN_STATUS_NOT_SUPPORTED) {
#else
                    if (status != CUDNN_STATUS_NOT_SUPPORTED) {
#endif
                        CHECK_CUDNN_ERROR(status);
                    }
                    continue;
                }
                int64_t required = 0;
                CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
                                                           CUDNN_TYPE_INT64, 1, &returned, &required));
                if (required < 0) {
                    throw std::runtime_error("negative prepared-plan workspace size");
                }
                if (required > config.get_workspace_limit_mib() * 1024 * 1024) {
                    continue;
                }
                CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1,
                                                           &returned, &engine_index_));
                plan_ = plan;
                workspace_bytes_ = required;
                auto identity = json();
                if (std::find(attempted.begin(), attempted.end(), identity) != attempted.end()) {
                    continue;
                }
                attempted.push_back(identity);
                bind(input, filter, output);
                if (!reference.accepts([&]() { execute(handle); })) {
                    verification_["rejected"].push_back({{"engine", engine_index_},
                                                         {"failures", reference.failures()},
                                                         {"maximum_tolerance_ratio", reference.maximum_ratio()}});
                    continue;
                }
                verification_["passed"] = true;
                verification_["maximum_tolerance_ratio"] = reference.maximum_ratio();
                return;
            }
        }
        throw std::runtime_error("prepared execution unsupported under screened-v1 policy: " + verification_.dump());
    }
#endif

  public:
    static void validate(CudnnConfig &config) {
        if (config.get_name() != "cudnnConvolutionBackwardFilter" || config.get_array_length() != 2 ||
            config.get_conv_type() != CUDNN_DATA_FLOAT ||
            (config.get_input_type() != CUDNN_DATA_FLOAT && config.get_input_type() != CUDNN_DATA_HALF) ||
            config.get_mode() != CUDNN_CROSS_CORRELATION || config.get_workspace_limit_mib() < 0 ||
            config.get_workspace_limit_mib() > 1048576) {
            throw std::invalid_argument(
                "prepared execution requires 2D backward-filter, FP32 compute and FP32/FP16 storage");
        }
        for (auto tensors : {std::make_pair(config.get_input_dims(), config.get_input_stride()),
                             std::make_pair(config.get_output_dims(), config.get_output_stride())}) {
            auto strides = packed_strides(tensors.first);
            if (tensors.second.size() != strides.size() ||
                !std::equal(strides.begin(), strides.end(), tensors.second.begin())) {
                throw std::invalid_argument("prepared execution requires packed NCHW tensors");
            }
        }
        packed_strides(config.get_filter_dims());
        auto &input = config.get_input_dims();
        auto &output = config.get_output_dims();
        auto &filter = config.get_filter_dims();
        if (config.get_padA().size() != 2 || config.get_filter_strideA().size() != 2 ||
            config.get_dilationA().size() != 2 || input[0] != output[0] || input[1] != filter[1] ||
            output[1] != filter[0]) {
            throw std::invalid_argument("inconsistent prepared convolution dimensions");
        }
        for (size_t axis = 0; axis < 2; ++axis) {
            const int64_t padding = config.get_padA()[axis];
            const int64_t stride = config.get_filter_strideA()[axis];
            const int64_t dilation = config.get_dilationA()[axis];
            const int64_t extent = input[axis + 2] + 2 * padding - dilation * (filter[axis + 2] - 1) - 1;
            if (padding < 0 || stride <= 0 || dilation <= 0 || extent < 0 || output[axis + 2] != extent / stride + 1) {
                throw std::invalid_argument("inconsistent prepared convolution output shape");
            }
        }
#if CUDNN_VERSION < 8900
        throw std::runtime_error("prepared execution requires cuDNN 8.9 or newer");
#endif
    }

    CudnnPreparedPlan(cudnnHandle_t handle, CudnnConfig &config, void *input, void *filter, void *output) {
        validate(config);
#if CUDNN_VERSION >= 8900
        auto input_tensor = tensor(101, config.get_input_dims(), config.get_input_type());
        auto filter_tensor = tensor(102, config.get_filter_dims(), config.get_input_type());
        auto output_tensor = tensor(103, config.get_output_dims(), config.get_input_type());
        auto convolution = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        set(convolution, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(convolution, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, config.get_mode());
        set(convolution, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, int64_t{2});
        for (auto parameter : {std::make_pair(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, config.get_padA()),
                               std::make_pair(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, config.get_padA()),
                               std::make_pair(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, config.get_filter_strideA()),
                               std::make_pair(CUDNN_ATTR_CONVOLUTION_DILATIONS, config.get_dilationA())}) {
            std::vector<int64_t> values(parameter.second.begin(), parameter.second.end());
            CHECK_CUDNN_ERROR(
                cudnnBackendSetAttribute(convolution, parameter.first, CUDNN_TYPE_INT64, values.size(), values.data()));
        }
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(convolution));
        auto operation = create(CUDNN_BACKEND_OPERATION_CONVOLUTION_BACKWARD_FILTER_DESCRIPTOR);
        set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, input_tensor);
        set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DW, CUDNN_TYPE_BACKEND_DESCRIPTOR, filter_tensor);
        set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_DY, CUDNN_TYPE_BACKEND_DESCRIPTOR, output_tensor);
        set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR,
            convolution);
        set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_ALPHA, CUDNN_TYPE_FLOAT, 1.f);
        set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_BWD_FILTER_BETA, CUDNN_TYPE_FLOAT, 0.f);
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(operation));
        auto graph = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, handle);
        set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, operation);
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(graph));
        select(handle, graph, config, input, filter, output);
#endif
    }

    void execute(cudnnHandle_t handle) {
#if CUDNN_VERSION >= 8900
        CHECK_CUDNN_ERROR(cudnnBackendExecute(handle, plan_, pack_));
#endif
    }

    std::string json() const {
#if CUDNN_VERSION >= 8900
        int64_t count = 0;
        CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(plan_, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                                   CUDNN_TYPE_CHAR, 0, &count, nullptr));
        std::vector<char> text(count + 1, 0);
        CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(plan_, CUDNN_ATTR_EXECUTION_PLAN_JSON_REPRESENTATION,
                                                   CUDNN_TYPE_CHAR, count, &count, text.data()));
        return text.data();
#else
        return "{}";
#endif
    }

    const nlohmann::json &verification() const { return verification_; }
    CudnnPreparedPlan(const CudnnPreparedPlan &) = delete;
    CudnnPreparedPlan &operator=(const CudnnPreparedPlan &) = delete;
};
} // namespace cudnn_test