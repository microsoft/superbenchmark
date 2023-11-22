// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "../directx_utils/Options.h"

namespace Option {
enum Precision {
    F16,
    F32,
};
using PrecisionType = Option::Precision;
} // namespace Option

class BenchmarkOptions : public Options {
  public:
    // Number of warm up rounds to run.
    int num_warm_up = 0;
    // The number of benchmark runs.
    int num_loops = 0;
    // Dimension m of GEMM.
    int m = 0;
    // Dimension n of GEMM.
    int n = 0;
    // Dimension k of GEMM.
    int k = 0;
    // The precision of calculate.
    Option::PrecisionType mode_precision = Option::F32;

    /**
     * @brief Construct a new GPUCoreOptions object.
     */
    BenchmarkOptions(int argc, char *argv[]) : Options(argc, argv) {}

    /**
     * @brief Parse the arguments.
     */
    virtual void parse_arguments() {
        num_loops = get_cmd_line_argument_int("--num_loops", 10);
        num_warm_up = get_cmd_line_argument_int("--num_warm_up", 0);
        m = get_cmd_line_argument_int("--m", 16 * 256);
        n = get_cmd_line_argument_int("--n", 16 * 256);
        k = get_cmd_line_argument_int("--k", 16 * 256);
        if (get_cmd_line_argument_bool("--fp16")) {
            mode_precision = Option::F16;
        }
        if (get_cmd_line_argument_bool("--fp32")) {
            mode_precision = Option::F32;
        }
    }

    /**
     * @brief Get the option usage.
     */
    void get_option_usage() override {
        std::cout << "Usage: " << std::endl;
        std::cout << "  --help: Print help message." << std::endl;
        std::cout << "  --num_loops: The number of benchmark runs." << std::endl;
        std::cout << "  --num_warm_up: The number of warmup runs." << std::endl;
        std::cout << "  --m: m dimension of GEMM." << std::endl;
        std::cout << "  --n: n dimension of GEMM." << std::endl;
        std::cout << "  --k: l dimension of GEMM." << std::endl;
        std::cout << "  --fp16: half precision to compute." << std::endl;
        std::cout << "  --fp32: float precision to compute." << std::endl;
    }
};
