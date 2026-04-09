
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "gpu_stream_utils.hpp"

namespace stream_config {
/**
 * @brief Converts a kernel index to its corresponding string representation.
 *
 * @details This function takes an integer representing a kernel index and returns the corresponding
 * string representation of the kernel. The mapping between kernel indices and their string representations
 * should be defined within the function.
 *
 * @param[in] kernel_idx The index of the kernel to be converted to a string.
 *
 * @return std::string The string representation of the kernel.
 */
std::string KernelToString(int kernel_idx) {
    switch (kernel_idx) {
    case static_cast<int>(Kernel::kCopy):
        return "COPY";
    case static_cast<int>(Kernel::kScale):
        return "SCALE";
    case static_cast<int>(Kernel::kAdd):
        return "ADD";
    case static_cast<int>(Kernel::kTriad):
        return "TRIAD";
    default:
        return "UNKNOWN";
    }
}

/**
 * @brief Print the usage of this program.
 *
 * @details Thus function prints the usage of this program.
 *
 * @return void.
 * */
void PrintUsage() {
    std::cout << "Usage: gpu_stream "
              << "--size <size in bytes> "
              << "--num_warm_up <num_warm_up> "
              << "--num_loops <num_loops> "
              << "[--data_type <float|double>] "
              << "[--check_data]" << std::endl;
}

/**
 * @brief Print the user provided inputs info.
 *
 * @details Thus function prints the parsed user provided inputs of this program..
 *
 * @param[in] opts The Opts struct that stores the parsed values.
 *
 * @return void
 * */
void PrintInputInfo(Opts &opts) {
    std::cout << "STREAM Benchmark" << std::endl;
    std::cout << "Buffer size(bytes): " << opts.size << std::endl;
    std::cout << "Number of warm up runs: " << opts.num_warm_up << std::endl;
    std::cout << "Number of loops: " << opts.num_loops << std::endl;
    std::cout << "Data type: " << opts.data_type << std::endl;
    std::cout << "Check data: " << (opts.check_data ? "Yes" : "No") << std::endl;
}

/**
 * @brief Parse the command line options.
 *
 * @details Thus function parses the command line options and stores the values in the Opts struct.
 *
 * @param[in] argc The number of command line options.
 * @param[in] argv The command line options.
 * @param[out] opts The Opts struct to store the parsed values.
 *
 * @return int The status code.
 * */
int ParseOpts(int argc, char **argv, Opts *opts) {
    enum class OptIdx { kSize, kNumWarmUp, kNumLoops, kEnableCheckData, kDataType };
    const struct option options[] = {{"size", required_argument, nullptr, static_cast<int>(OptIdx::kSize)},
                                     {"num_warm_up", required_argument, nullptr, static_cast<int>(OptIdx::kNumWarmUp)},
                                     {"num_loops", required_argument, nullptr, static_cast<int>(OptIdx::kNumLoops)},
                                     {"check_data", no_argument, nullptr, static_cast<int>(OptIdx::kEnableCheckData)},
                                     {"data_type", required_argument, nullptr, static_cast<int>(OptIdx::kDataType)}};
    int getopt_ret = 0;
    int opt_idx = 0;
    bool size_specified = true;
    bool num_warm_up_specified = false;
    bool num_loops_specified = false;

    bool parse_err = false;
    while (true) {
        getopt_ret = getopt_long(argc, argv, "", options, &opt_idx);
        if (getopt_ret == -1) {
            if (!size_specified || !num_warm_up_specified || !num_loops_specified) {
                parse_err = true;
            }
            break;
        } else if (getopt_ret == '?') {
            parse_err = true;
            break;
        }
        switch (opt_idx) {
        case static_cast<int>(OptIdx::kSize):
            if (1 != sscanf(optarg, "%lu", &(opts->size))) {
                std::cerr << "Invalid size: " << optarg << std::endl;
                parse_err = true;
            } else {
                size_specified = true;
            }
            break;
        case static_cast<int>(OptIdx::kNumWarmUp):
            if (1 != sscanf(optarg, "%lu", &(opts->num_warm_up))) {
                std::cerr << "Invalid num_warm_up: " << optarg << std::endl;
                parse_err = true;
            } else {
                num_warm_up_specified = true;
            }
            break;
        case static_cast<int>(OptIdx::kNumLoops):
            if (1 != sscanf(optarg, "%lu", &(opts->num_loops))) {
                std::cerr << "Invalid num_loops: " << optarg << std::endl;
                parse_err = true;
            } else {
                num_loops_specified = true;
            }
            break;
        case static_cast<int>(OptIdx::kEnableCheckData):
            opts->check_data = true;
            break;
        case static_cast<int>(OptIdx::kDataType):
            opts->data_type = optarg;
            if (opts->data_type != "float" && opts->data_type != "double") {
                std::cerr << "Invalid data_type: " << optarg << ". Must be 'float' or 'double'." << std::endl;
                parse_err = true;
            }
            break;
        default:
            parse_err = true;
        }
        if (parse_err) {
            break;
        }
    }
    if (parse_err) {
        PrintUsage();
        return -1;
    }
    return 0;
}

} // namespace stream_config

unsigned long long getCurrentTimestampInMicroseconds() {
    // Get the current time point
    auto now = std::chrono::system_clock::now();

    // Convert to time since epoch
    auto duration = now.time_since_epoch();

    // Convert to microseconds
    auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();

    return static_cast<unsigned long long>(microseconds);
}
