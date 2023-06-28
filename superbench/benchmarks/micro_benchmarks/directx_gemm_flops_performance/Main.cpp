// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "GPUCore.h"

int main(int argc, char *argv[]) {
    std::unique_ptr<BenchmarkOptions> opts = std::make_unique<BenchmarkOptions>(argc, argv);
    opts->init();
    std::unique_ptr<GPUCore> gpucopy = std::make_unique<GPUCore>(opts.get());
    gpucopy->Run();
}
