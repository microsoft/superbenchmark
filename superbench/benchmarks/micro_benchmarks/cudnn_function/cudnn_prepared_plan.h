#pragma once

#include <algorithm>
#include <stdexcept>

#include "cudnn_convolution_workload.h"
#include "cudnn_plan_policy.h"

namespace cudnn_test {
class CudnnPreparedPlan {
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
    cudnnHandle_t handle_ = nullptr;
    cudnnBackendDescriptor_t graph_ = nullptr;
    cudnnBackendDescriptor_t plan_ = nullptr;
    cudnnBackendDescriptor_t pack_ = nullptr;
    int64_t workspace_bytes_ = 0;
    int64_t engine_index_ = -1;
    void *input_ = nullptr;
    void *filter_ = nullptr;
    void *output_ = nullptr;

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
        auto strides = CudnnConvolutionWorkload::packed_strides(dimensions);
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

  public:
    struct Candidate {
        cudnnBackendDescriptor_t configuration;
        cudnnBackendDescriptor_t engine;
        CudnnPlanNumerics numerics;
    };

    void bind() {
        workspace_.reset(static_cast<size_t>(workspace_bytes_));
        pack_ = create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
        int64_t ids[] = {101, 102, 103};
        void *pointers[] = {input_, filter_, output_};
        CHECK_CUDNN_ERROR(
            cudnnBackendSetAttribute(pack_, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, ids));
        CHECK_CUDNN_ERROR(
            cudnnBackendSetAttribute(pack_, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers));
        set(pack_, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, workspace_.pointer);
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(pack_));
    }

    template <typename Visit> bool visit_candidates(Visit visit) {
        for (auto mode : {CUDNN_HEUR_MODE_A, CUDNN_HEUR_MODE_FALLBACK}) {
            auto heuristic = create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
            set(heuristic, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, graph_);
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
                auto contains = [&](cudnnBackendNumericalNote_t note) {
                    return std::find(notes.begin(), notes.end(), note) != notes.end();
                };
                const CudnnPlanNumerics numerics{contains(CUDNN_NUMERICAL_NOTE_TENSOR_CORE),
                                                 contains(CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC),
                                                 contains(CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS),
                                                 contains(CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION)};
                if (visit(Candidate{candidate, engine, numerics})) {
                    return true;
                }
            }
        }
        return false;
    }

    bool build(const Candidate &candidate) {
        auto plan = create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
        set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, handle_);
        set(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG, CUDNN_TYPE_BACKEND_DESCRIPTOR, candidate.configuration);
        auto status = cudnnBackendFinalize(plan);
        if (status != CUDNN_STATUS_SUCCESS) {
#if CUDNN_MAJOR >= 9
            if (CUDNN_STATUS_CATEGORY(status) != CUDNN_STATUS_NOT_SUPPORTED) {
#else
            if (status != CUDNN_STATUS_NOT_SUPPORTED) {
#endif
                CHECK_CUDNN_ERROR(status);
            }
            return false;
        }
        int64_t returned = 0;
        CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1,
                                                   &returned, &workspace_bytes_));
        CHECK_CUDNN_ERROR(cudnnBackendGetAttribute(candidate.engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64,
                                                   1, &returned, &engine_index_));
        plan_ = plan;
        return true;
    }

    int64_t workspace_bytes() const { return workspace_bytes_; }
    int64_t engine_index() const { return engine_index_; }
#endif

  public:
    static void validate(const CudnnConvolutionWorkload &workload) {
        workload.validate_backward_filter();
#if CUDNN_VERSION < 8900
        throw std::runtime_error("prepared execution requires cuDNN 8.9 or newer");
#endif
    }

    CudnnPreparedPlan(cudnnHandle_t handle, const CudnnConvolutionWorkload &workload, void *input, void *filter,
                      void *output) {
        validate(workload);
#if CUDNN_VERSION >= 8900
        handle_ = handle;
        input_ = input;
        filter_ = filter;
        output_ = output;
        auto input_tensor = tensor(101, workload.input_dims, workload.input_type);
        auto filter_tensor = tensor(102, workload.filter_dims, workload.input_type);
        auto output_tensor = tensor(103, workload.output_dims, workload.input_type);
        auto convolution = create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
        set(convolution, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, CUDNN_DATA_FLOAT);
        set(convolution, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, workload.mode);
        set(convolution, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, int64_t{2});
        for (auto parameter : {std::make_pair(CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, workload.padding),
                               std::make_pair(CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, workload.padding),
                               std::make_pair(CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, workload.filter_stride),
                               std::make_pair(CUDNN_ATTR_CONVOLUTION_DILATIONS, workload.dilation)}) {
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
        graph_ = create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
        set(graph_, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, handle);
        set(graph_, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, operation);
        CHECK_CUDNN_ERROR(cudnnBackendFinalize(graph_));
#endif
    }

    void execute(cudnnHandle_t handle) const {
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

    CudnnPreparedPlan(const CudnnPreparedPlan &) = delete;
    CudnnPreparedPlan &operator=(const CudnnPreparedPlan &) = delete;
};
} // namespace cudnn_test