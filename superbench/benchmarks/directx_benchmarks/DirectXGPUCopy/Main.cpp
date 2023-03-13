// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <iostream>
#include <sstream>

#include "GPUCopyBw.h"

int main(int argc, char *argv[]) {
    Options *opts = new Options(argc, argv);
    for (SIZE_T usize = opts->minbytes; usize <= opts->maxbytes; usize += usize) {
        std::cout << "size: " << usize << "B,";
        opts->size = usize;
        opts->htod_enabled = true;
        auto gpucopy = new GPUCopyBw(opts);
        gpucopy->Run();
    }
}
