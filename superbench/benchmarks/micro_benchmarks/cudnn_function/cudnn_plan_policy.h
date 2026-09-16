#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace cudnn_test {
struct CudnnPlanNumerics {
    bool tensor_cores;
    bool nondeterministic;
    bool input_down_conversion;
    bool reduced_precision_reduction;
};

class CudnnAccuracyPolicy {
    double relative_tolerance_;

  public:
    explicit CudnnAccuracyPolicy(bool half_storage) : relative_tolerance_(0.0005 + (half_storage ? 1.0 / 2048 : 0)) {}
    const char *name() const { return "full-output-v1"; }
    double atol() const { return 0.0005; }
    double rtol() const { return relative_tolerance_; }
    double tolerance(double reference) const { return atol() + rtol() * std::abs(reference); }
};

class CudnnPlanPolicy {
    bool require_tensor_cores_;
    int64_t workspace_limit_bytes_;

  public:
    CudnnPlanPolicy(bool require_tensor_cores, int64_t workspace_limit_mib)
        : require_tensor_cores_(require_tensor_cores) {
        if (workspace_limit_mib < 0 || workspace_limit_mib > 1048576) {
            throw std::invalid_argument("invalid prepared workspace limit");
        }
        workspace_limit_bytes_ = workspace_limit_mib * 1024 * 1024;
    }

    const char *name() const { return "screened-v1"; }

    bool accepts_math(const CudnnPlanNumerics &numerics) const {
        return numerics.tensor_cores == require_tensor_cores_ && !numerics.nondeterministic &&
               !numerics.input_down_conversion && !numerics.reduced_precision_reduction;
    }

    bool accepts_workspace(int64_t bytes) const {
        if (bytes < 0) {
            throw std::runtime_error("negative prepared-plan workspace size");
        }
        return bytes <= workspace_limit_bytes_;
    }
};

template <typename Plans, typename Qualify>
bool select_first_passing_plan(Plans &plans, const CudnnPlanPolicy &policy, Qualify qualify) {
    std::vector<std::string> attempted;
    return plans.visit_candidates([&](const typename Plans::Candidate &candidate) {
        if (!policy.accepts_math(candidate.numerics) || !plans.build(candidate) ||
            !policy.accepts_workspace(plans.workspace_bytes())) {
            return false;
        }
        const auto identity = plans.json();
        if (std::find(attempted.begin(), attempted.end(), identity) != attempted.end()) {
            return false;
        }
        attempted.push_back(identity);
        plans.bind();
        return qualify();
    });
}
} // namespace cudnn_test
