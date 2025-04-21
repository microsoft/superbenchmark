#include <chrono>
#include <cstring> // for memcpy
#include <getopt.h>
#include <iomanip> // for setting precision
#include <iostream>
#include <numa.h>
#include <numeric>
#include <vector>

// Options accepted by this program.
struct Opts {
    // Data buffer size for copy benchmark.
    uint64_t size = 0;

    // Number of warm up rounds to run.
    uint64_t num_warm_up = 0;

    // Number of loops to run.
    uint64_t num_loops = 0;

    // Whether check data after copy.
    bool check_data = false;
};

/**
 * @brief Print the usage instructions for this program.
 *
 * This function outputs the correct way to execute the program,
 * including any necessary command-line arguments and their descriptions.
 */
void PrintUsage() {
    std::cout << "Usage: cpu_copy "
              << "--size <size> "
              << "--num_warm_up <num_warm_up> "
              << "--num_loops <num_loops> "
              << "[--check_data]" << std::endl;
}

/**
 * @brief Checks if the system has memory available for a specific NUMA node.
 *
 * This function determines whether there is memory available on the specified
 * NUMA (Non-Uniform Memory Access) node.
 *
 * Empty NUMA nodes in Grace CPU are reserved for multi-instance GPUs (MIG).
 *
 * @param node The identifier of the NUMA node to check.
 * @return true if the specified NUMA node has sufficient memory available, false otherwise.
 */
bool HasMemForNumaNode(int node) {
    try {
        long free_memory = numa_node_size64(node, nullptr);
        return free_memory > 0;
    } catch (const std::exception &e) {
        std::cerr << "Failed to get memory size for NUMA node " << node << ". ERROR: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief Checks if the system has CPUs available for a specific NUMA node.
 *
 * This function determines whether there are CPUs available on the specified
 * NUMA (Non-Uniform Memory Access) node. It is useful for ensuring that CPU
 * affinity can be set to the desired NUMA node, which can help optimize memory
 * access patterns and performance in NUMA-aware applications.
 *
 * Memory-only or empty NUMA nodes in Grace CPU are for GPUs.
 *
 * @param node The identifier of the NUMA node to check.
 * @return true if the specified NUMA node has CPUs available, false otherwise.
 */
bool HasCPUsForNumaNode(int node) {
    struct bitmask *bm = numa_allocate_cpumask();

    int numa_err = numa_node_to_cpus(node, bm);
    if (numa_err != 0) {
        std::cerr << "Failed to get CPU mask for NUMA node " << node << ". ERROR: " << strerror(errno) << std::endl;

        numa_bitmask_free(bm);
        return false; // On error
    }

    // Check if any CPU is assigned to the NUMA node, has_cpus is false for mem only numa nodes
    bool has_cpus = (numa_bitmask_weight(bm) > 0);
    numa_bitmask_free(bm);
    return has_cpus;
}

/**
 * @brief Parses command-line options for the CPU copy performance benchmark.
 *
 * This function processes the command-line arguments provided to the benchmark
 * and sets the appropriate configuration options based on the input.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return An integer indicating the success or failure of the option parsing.
 *         Returns 0 on success, and a non-zero value on failure.
 */
/**/
int ParseOpts(int argc, char **argv, Opts *opts) {
    enum class OptIdx { kSize, kNumWarmUp, kNumLoops, kEnableCheckData };
    const struct option options[] = {{"size", required_argument, nullptr, static_cast<int>(OptIdx::kSize)},
                                     {"num_warm_up", required_argument, nullptr, static_cast<int>(OptIdx::kNumWarmUp)},
                                     {"num_loops", required_argument, nullptr, static_cast<int>(OptIdx::kNumLoops)},
                                     {"check_data", no_argument, nullptr, static_cast<int>(OptIdx::kEnableCheckData)}};
    int getopt_ret = 0;
    int opt_idx = 0;
    bool size_specified = false;
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

/**
 * @brief Benchmark the memory copy performance between two NUMA nodes.
 *
 * This function measures the performance of copying memory from a source NUMA node to a destination NUMA node.
 *
 * @param src_node The source NUMA node from which memory will be copied.
 * @param dst_node The destination NUMA node to which memory will be copied.
 * @param opts A reference to an Opts structure containing various options and configurations for the benchmark.
 * @return The performance metric of the memory copy operation, typically in terms of bandwidth or latency.
 */
double BenchmarkNUMACopy(int src_node, int dst_node, Opts &opts) {
    int ret = 0;

    // Set CPU affinity to the NUMA node with CPU cores assoiated
    int affinity_node = HasCPUsForNumaNode(src_node) ? src_node : dst_node;
    ret = numa_run_on_node(affinity_node);
    if (ret != 0) {
        std::cerr << "Failed to set CPU affinity to NUMA node " << src_node << std::endl;
        return 0;
    }

    // Allocate memory on the source and destination NUMA nodes
    char *src = (char *)numa_alloc_onnode(opts.size, src_node);
    if (!src) {
        std::cerr << "Memory allocation failed on node" << src_node << std::endl;
        return 0;
    }

    char *dst = (char *)numa_alloc_onnode(opts.size, dst_node);
    if (!dst) {
        std::cerr << "Memory allocation failed on node" << dst_node << std::endl;
        return 0;
    }

    // Initialize the source memory with some data
    memset(src, 1, opts.size);

    // Measure the time taken for memcpy between nodes
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the memory copy
    memcpy(dst, src, opts.size);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Calculate the latency (nanoseconds per byte)
    double total_time_ns = diff.count() * 1e9; // Convert seconds to nanoseconds

    // Free the allocated memory
    numa_free(src, opts.size);
    numa_free(dst, opts.size);

    if (opts.check_data) {
        // Check the data integrity after the copy
        if (memcmp(src, dst, opts.size) != 0) {
            std::cerr << "Data integrity check failed!" << dst_node << std::endl;

            return -1;
        }
    }

    return total_time_ns;
}

/**
 * @brief Runs the CPU copy benchmark between all pairs of NUMA nodes.
 *
 * This function runs the CPU copy benchmark between all pairs of NUMA nodes in the system.
 * It calculates the average bandwidth and latency for each pair of nodes and outputs the results.
 *
 * @param src_node The source NUMA node from which data will be copied.
 * @param dst_node The destination NUMA node to which data will be copied.
 * @param opts A reference to an Opts object containing various options and configurations for the benchmark.
 */
double RunCPUCopyBenchmark(int src_node, int dst_node, Opts &opts) {
    // Run warm up rounds
    for (int i = 0; i < opts.num_warm_up; i++) {
        BenchmarkNUMACopy(src_node, dst_node, opts);
    }

    double time_used_ns = 0;

    for (int i = 0; i < opts.num_loops; i++) {
        time_used_ns += BenchmarkNUMACopy(src_node, dst_node, opts);
    }

    return time_used_ns / opts.num_loops;
}

int main(int argc, char **argv) {
    Opts opts;
    int ret = -1;
    ret = ParseOpts(argc, argv, &opts);
    if (0 != ret) {
        return ret;
    }

    // Check if the system has multiple NUMA nodes
    if (-1 == numa_available()) {
        std::cerr << "NUMA is not available on this system!" << std::endl;
        return 1;
    }

    int num_of_numa_nodes = numa_num_configured_nodes();

    if (num_of_numa_nodes < 2) {
        std::cerr << "System has less than 2 NUMA nodes. Benchmark is not applicable." << std::endl;
        return 1;
    }

    // Run the benchmark
    for (int src_node = 0; src_node < num_of_numa_nodes; src_node++) {
        if (!HasMemForNumaNode(src_node)) {
            // Skip the NUMA node if there are no memory available
            continue;
        }

        for (int dst_node = 0; dst_node < num_of_numa_nodes; dst_node++) {
            if (src_node == dst_node) {
                // Skip the same NUMA node
                continue;
            }

            if (!HasMemForNumaNode(dst_node)) {
                // Skip the NUMA node if there are no memory available
                continue;
            }

            //
            if (!HasCPUsForNumaNode(src_node) && !HasCPUsForNumaNode(dst_node)) {
                // Skip the process if there are no CPUs available on both NUMA nodes
                continue;
            }

            double time_used_ns = RunCPUCopyBenchmark(src_node, dst_node, opts);
            double bw = opts.size / (time_used_ns / 1e9) / 1e6; // MB/s
            double latency = time_used_ns / opts.size;          // ns/byte

            // Output the result
            std::cout << "mem_bandwidth_matrix_numa_" << src_node << "_" << dst_node << "_bw: " << std::setprecision(9)
                      << bw << std::endl;
            std::cout << "mem_bandwidth_matrix_numa_" << src_node << "_" << dst_node << "_lat: " << std::setprecision(9)
                      << latency << std::endl;
        }
    }

    return 0;
}
