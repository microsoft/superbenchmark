// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "GPUCore.h"
#include <iostream>
#include <sstream>

int main(int argc, char *argv[]) {
    Options *opts = new Options(argc, argv);
    auto gpucopy = new GPUCore(opts);
    gpucopy->Run();
}
