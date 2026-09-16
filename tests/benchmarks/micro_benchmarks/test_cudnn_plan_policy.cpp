#include <iostream>
#include <string>

#include "cudnn_plan_policy.h"

using namespace cudnn_test;

static void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

struct CandidatePlans {
    struct Candidate {
        CudnnPlanNumerics numerics;
        bool supported;
        int64_t workspace;
        std::string identity;
        bool accurate;
    };
    std::vector<Candidate> candidates;
    const Candidate *current = nullptr;
    template <typename Visit> bool visit_candidates(Visit visit) {
        for (const auto &candidate : candidates) {
            if (visit(candidate)) {
                return true;
            }
        }
        return false;
    }

    bool build(const Candidate &candidate) {
        current = &candidate;
        return candidate.supported;
    }

    int64_t workspace_bytes() const { return current->workspace; }
    std::string json() const { return current->identity; }
    void bind() {}
};

static void test_selection() {
    const CudnnPlanNumerics allowed{false, false, false, false};
    const CudnnPlanPolicy policy(false, 1);
    CandidatePlans plans;
    plans.candidates = {{{true, false, false, false}, true, 0, "wrong-math", true},
                        {allowed, false, 0, "unsupported", true},
                        {allowed, true, 1024 * 1024 + 1, "oversized", true},
                        {allowed, true, 0, "inaccurate", false},
                        {allowed, true, 0, "inaccurate", true},
                        {allowed, true, 0, "accepted", true},
                        {allowed, true, 0, "later", true}};
    std::vector<std::string> screened;
    require(select_first_passing_plan(plans, policy,
                                      [&]() {
                                          screened.push_back(plans.json());
                                          return plans.current->accurate;
                                      }),
            "no passing candidate was retained");
    require(screened == std::vector<std::string>{"inaccurate", "accepted"}, "screen order or deduplication changed");
    require(plans.json() == "accepted", "selection did not stop at the first passing candidate");
    require(!select_first_passing_plan(plans, policy, []() { return false; }),
            "failed numerical qualification was ignored");

    bool propagated = false;
    try {
        select_first_passing_plan(plans, policy, []() -> bool { throw std::runtime_error("execution failed"); });
    } catch (const std::runtime_error &error) {
        propagated = std::string(error.what()) == "execution failed";
    }
    require(propagated, "execution failure was swallowed");
}

int main() {
    try {
        for (bool half_storage : {false, true}) {
            const CudnnAccuracyPolicy accuracy(half_storage);
            require(accuracy.atol() == 0.0005, "absolute tolerance changed");
            require(accuracy.rtol() == 0.0005 + (half_storage ? 1.0 / 2048 : 0), "storage rounding term changed");
        }
        for (bool tensor_cores : {false, true}) {
            CudnnPlanPolicy policy(tensor_cores, 1024);
            for (int mask = 0; mask < 16; ++mask) {
                CudnnPlanNumerics numerics{bool(mask & 1), bool(mask & 2), bool(mask & 4), bool(mask & 8)};
                bool expected = numerics.tensor_cores == tensor_cores && !(mask & 14);
                require(policy.accepts_math(numerics) == expected, "screened-v1 candidate rules changed");
            }
        }
        const CudnnPlanPolicy policy(false, 1);
        require(policy.accepts_workspace(1024 * 1024), "workspace boundary was rejected");
        require(!policy.accepts_workspace(1024 * 1024 + 1), "workspace limit was exceeded");
        test_selection();
        std::cout << "Prepared policy, workspace, first-passing selection and rejection boundaries passed" << std::endl;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
    return 0;
}
