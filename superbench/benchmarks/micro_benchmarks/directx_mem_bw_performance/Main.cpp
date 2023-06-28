// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "GPUMemRwBw.h"

int main(int argc, char *argv[]) {
    BenchmarkOptions option(argc, argv);
    option.init();
    if (option.size != -1) {
        // Run only one size
        GPUMemRwBw benchmark(&option);
        benchmark.Run();
    } else {
        // Run all sizes
        for (SIZE_T usize = option.min_size; usize <= option.max_size; usize += usize) {
            option.size = usize;
            GPUMemRwBw benchmark(&option);
            benchmark.Run();
        }
    }
}
