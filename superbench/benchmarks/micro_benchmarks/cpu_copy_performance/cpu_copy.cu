#include <chrono>
#include <cstring> // for memcpy
#include <iomanip> // for setting precision
#include <iostream>
#include <numa.h>

// using namespace std;

// Size of the memory to copy (100 MB in this example)
const size_t MEM_SIZE = 100 * 1024 * 1024 * 1024; // 1GB

/**
 * @brief Calculates the factorial of a number.
 *
 * This function computes the factorial of a non-negative integer n.
 *
 * @param n The non-negative integer for which to compute the factorial.
 * @return The factorial of n. Returns 1 if n is 0.
 * @throw std::invalid_argument if n is negative.
 */
int benchmark_numa_copy(int src_node, int dst_node) {
    int ret = 0;

    // Set CPU affinity to the source NUMA node
    ret = numa_run_on_node(src_node);
    if (ret != 0) {
        std::cerr << "Failed to set CPU affinity to NUMA node " << src_node << std::endl;
        return -1;
    }

    // Allocate memory on the source and destination NUMA nodes
    char *src = (char *)numa_alloc_onnode(MEM_SIZE, src_node);
    if (!src) {
        std::cerr << "Memory allocation failed on node" << src_node << std::endl;
        return -1;
    }

    char *dst = (char *)numa_alloc_onnode(MEM_SIZE, dst_node);
    if (!dst) {
        std::cerr << "Memory allocation failed on node" << dst_node << std::endl;
        return -1;
    }

    // Initialize the source memory with some data
    memset(src, 1, MEM_SIZE);

    // Measure the time taken for memcpy between nodes
    auto start = std::chrono::high_resolution_clock::now();

    // Perform the memory copy
    memcpy(dst, src, MEM_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    // Calculate the latency (nanoseconds per byte)
    double total_time_ns = diff.count() * 1e9; // Convert seconds to nanoseconds
    double latency_ns_per_byte = total_time_ns / MEM_SIZE;

    // Output the result
    std::cout << "cpu_copy_bw/" << src_node << "_to_" << dst_node << ": " << std::setprecision(9) << diff.count()
              << std::endl;
    std::cout << "cpu_copy_latency/" << src_node << "_to_" << dst_node << ": " << std::setprecision(9)
              << latency_ns_per_byte << std::endl;

    // Free the allocated memory
    numa_free(src, MEM_SIZE);
    numa_free(dst, MEM_SIZE);
}

int main() {
    int num_of_numa_nodes = numa_available() + 1;

    // Check if the system has multiple NUMA nodes
    if (0 == num_of_numa_nodes) {
        std::cerr << "NUMA is not available on this system!" << std::endl;
        return 1;
    }

    if (num_of_numa_nodes < 2) {
        std::cerr << "System has less than 2 NUMA nodes. Benchmark is not applicable." << std::endl;
        return 1;
    }

    // Run the benchmark
    for (int src_node = 0; src_node < num_of_numa_nodes; src_node++)
        for (int dst_node = 0; dst_node < num_of_numa_nodes; dst_node++) {
            if (src_node != dst_node && 0 != benchmark_numa_copy(src_node, dst_node)) {
                std::cerr << "Failed to run the benchmark. src node is " << src_node << ", dst ode is " << dst_node
                          << std::endl;
            }
        }

    return 0;
}
