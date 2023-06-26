// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "GPUCopyBw.h"

int main(int argc, char *argv[]) {
    std::unique_ptr<BenchmarkOptions> opts = std::make_unique<BenchmarkOptions>(argc, argv);
    opts->init();
    if (opts->size != -1) {
        // Run only one size
        std::unique_ptr<GPUCopyBw> benchmark = std::make_unique<GPUCopyBw>(opts.get());
        benchmark->Run();
    } else {
        // Run all sizes
        for (SIZE_T usize = opts->min_size; usize <= opts->max_size; usize += usize) {
            opts->size = usize;
            std::unique_ptr<GPUCopyBw> benchmark = std::make_unique<GPUCopyBw>(opts.get());
            benchmark->Run();
        }
    }
}
