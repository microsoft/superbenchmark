// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "GPUCore.h"

int main(int argc, char *argv[]) {
    std::unique_ptr<GPUCoreOptions> opts = std::make_unique<GPUCoreOptions>(argc, argv);
    opts->init();
    std::unique_ptr<GPUCore> gpucopy = std::make_unique<GPUCore>(opts.get());
    gpucopy->Run();
}
