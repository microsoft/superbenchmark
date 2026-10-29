// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_stream_utils.hpp"

#include <algorithm>
#include <charconv>
#include <cstring>
#include <limits>

namespace {
bool ParseUnsigned(const char *text, uint64_t *value) {
    const char *end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, *value);
    return parsed.ec == std::errc{} && parsed.ptr == end;
}

template <size_t N> bool ParseChoice(const char *text, const std::array<int, N> &allowed, int *value) {
    uint64_t parsed = 0;
    if (!ParseUnsigned(text, &parsed) || parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        return false;
    }
    if (std::find(allowed.begin(), allowed.end(), static_cast<int>(parsed)) == allowed.end()) {
        return false;
    }
    *value = static_cast<int>(parsed);
    return true;
}
} // namespace

namespace stream_config {
std::string KernelToString(int kernel_idx) {
    const std::array<const char *, 4> names = {"COPY", "SCALE", "ADD", "TRIAD"};
    return kernel_idx >= 0 && kernel_idx < static_cast<int>(names.size()) ? names[kernel_idx] : "UNKNOWN";
}

void PrintUsage() {
    std::cout << "Usage: gpu_stream [--size <bytes>] [--num_warm_up <N>] [--num_loops <N>]\n"
              << "  [--data_type <float|double|both>] [--check_data]\n"
              << "  [--operation <copy|scale|add|triad|all>] [--block_size <N>] [--unroll <N>]\n";
}

void PrintInputInfo(Opts &opts) {
    std::cout << "STREAM Benchmark\nBuffer size(bytes): " << opts.size
              << "\nNumber of warm up runs: " << opts.num_warm_up << "\nNumber of loops: " << opts.num_loops
              << "\nData type: " << opts.data_type << "\nCheck data: " << (opts.check_data ? "Yes" : "No") << std::endl;
}

int ParseOpts(int argc, char **argv, Opts *opts) {
    enum OptIdx { kSize, kWarmup, kLoops, kCheck, kType, kBlock, kUnroll, kOperation };
    const option options[] = {
        {"size", required_argument, nullptr, kSize},
        {"num_warm_up", required_argument, nullptr, kWarmup},
        {"num_loops", required_argument, nullptr, kLoops},
        {"check_data", no_argument, nullptr, kCheck},
        {"data_type", required_argument, nullptr, kType},
        {"block_size", required_argument, nullptr, kBlock},
        {"unroll", required_argument, nullptr, kUnroll},
        {"operation", required_argument, nullptr, kOperation},
        {nullptr, 0, nullptr, 0},
    };
    int option_index = 0;
    int token = 0;
    while ((token = getopt_long(argc, argv, "", options, &option_index)) != -1) {
        bool valid = true;
        switch (token) {
        case kSize:
            valid = ParseUnsigned(optarg, &opts->size);
            break;
        case kWarmup:
            valid = ParseUnsigned(optarg, &opts->num_warm_up);
            break;
        case kLoops:
            valid = ParseUnsigned(optarg, &opts->num_loops);
            break;
        case kCheck:
            opts->check_data = true;
            break;
        case kType:
            opts->data_type = optarg;
            valid = opts->data_type == "float" || opts->data_type == "double" || opts->data_type == "both";
            break;
        case kBlock:
            valid = ParseChoice(optarg, kThreadsPerBlock, &opts->block_size);
            break;
        case kUnroll:
            valid = ParseChoice(optarg, kUnrollFactors, &opts->unroll);
            break;
        case kOperation: {
            const std::array<const char *, 4> names = {"copy", "scale", "add", "triad"};
            const auto found = std::find_if(names.begin(), names.end(),
                                            [&](const char *name) { return std::strcmp(name, optarg) == 0; });
            valid = found != names.end() || std::strcmp(optarg, "all") == 0;
            if (found != names.end()) {
                opts->operations = {static_cast<Kernel>(found - names.begin())};
            } else if (valid) {
                opts->operations = {Kernel::kCopy, Kernel::kScale, Kernel::kAdd, Kernel::kTriad};
            }
            break;
        }
        default:
            PrintUsage();
            return -1;
        }
        if (!valid) {
            std::cerr << "Invalid value for --" << options[option_index].name << ": " << optarg << std::endl;
            return -1;
        }
    }
    const uint64_t element_size = opts->data_type == "float" ? sizeof(float) : sizeof(double);
    if (optind != argc || opts->size == 0 || opts->size % element_size != 0) {
        std::cerr << "Size must be a positive multiple of " << element_size << " bytes; positional arguments are invalid"
                  << std::endl;
        return -1;
    }
    if (opts->num_loops == 0 || opts->num_warm_up > std::numeric_limits<uint64_t>::max() - opts->num_loops) {
        std::cerr << "num_loops must be positive and num_warm_up + num_loops must not overflow" << std::endl;
        return -1;
    }
    return 0;
}
} // namespace stream_config

unsigned long long getCurrentTimestampInMicroseconds() {
    const auto duration = std::chrono::system_clock::now().time_since_epoch();
    return static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::microseconds>(duration).count());
}
