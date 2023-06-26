// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "GPUCopyBw.h"

int main(int argc, char *argv[]) {
    Options *opts = new Options(argc, argv);
    if (opts->size != -1) {
        // Run only one size
        auto gpucopy = new GPUCopyBw(opts);
        gpucopy->Run();
    } else {
        // Run all sizes
        for (SIZE_T usize = opts->min_size; usize <= opts->max_size; usize += usize) {
            opts->size = usize;
            auto gpucopy = new GPUCopyBw(opts);
            gpucopy->Run();
        }
    }
}
