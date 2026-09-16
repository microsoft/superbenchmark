#pragma once

#include <nlohmann/json.hpp>

#include "cudnn_plan_policy.h"
#include "cudnn_prepared_plan.h"
#include "cudnn_reference.h"

namespace cudnn_test {
class CudnnPreparedSelection {
    CudnnPreparedPlan plan_;
    const CudnnPlanPolicy policy_;
    nlohmann::json verification_;

  public:
    static void validate(CudnnConfig &config) {
        CudnnPreparedPlan::validate(CudnnConvolutionWorkload(config));
        CudnnPlanPolicy(config.get_use_tensor_op(), config.get_workspace_limit_mib());
    }

    CudnnPreparedSelection(cudnnHandle_t handle, const CudnnConvolutionWorkload &workload,
                           const CudnnPlanPolicy &policy, void *input, void *filter, void *output)
        : plan_(handle, workload, input, filter, output), policy_(policy) {
#if CUDNN_VERSION >= 8900
        const CudnnAccuracyPolicy accuracy(workload.input_type == CUDNN_DATA_HALF);
        CudnnReference reference(workload, accuracy, input, filter, output);
        verification_ = {{"policy", accuracy.name()},
                         {"passed", false},
                         {"inputs", 5},
                         {"reference", "cuBLAS-FP64-with-CPU-crosschecks"},
                         {"atol", accuracy.atol()},
                         {"rtol", accuracy.rtol()},
                         {"checked_elements", reference.checked_elements()},
                         {"rejected", nlohmann::json::array()}};
        bool selected = select_first_passing_plan(plan_, policy_, [&]() {
            if (!reference.accepts([&]() { plan_.execute(handle); })) {
                verification_["rejected"].push_back({{"engine", plan_.engine_index()},
                                                     {"failures", reference.failures()},
                                                     {"maximum_tolerance_ratio", reference.maximum_ratio()}});
                return false;
            }
            verification_["passed"] = true;
            verification_["maximum_tolerance_ratio"] = reference.maximum_ratio();
            return true;
        });
        if (!selected) {
            throw std::runtime_error(std::string("prepared execution unsupported under ") + policy_.name() +
                                     " policy: " + verification_.dump());
        }
#endif
    }

    const CudnnPreparedPlan &plan() const { return plan_; }
    const CudnnPlanPolicy &policy() const { return policy_; }
    const nlohmann::json &verification() const { return verification_; }
    CudnnPreparedSelection(const CudnnPreparedSelection &) = delete;
    CudnnPreparedSelection &operator=(const CudnnPreparedSelection &) = delete;
};
} // namespace cudnn_test
