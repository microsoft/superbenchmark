#!/bin/bash

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH="${SB_MICRO_PATH:-/usr/local}"

# Build cutlass.
if [ -f cutlass/CMakeLists.txt ]; then
    SOURCE_DIR=./cutlass
    BUILD_ROOT=./cutlass/build
    mkdir -p $BUILD_ROOT
    cmake -DCMAKE_INSTALL_BINDIR=$SB_MICRO_PATH/bin -DCMAKE_INSTALL_LIBDIR=$SB_MICRO_PATH/lib -DCMAKE_BUILD_TYPE=Release \
          -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_ENABLE_TESTS=OFF -S $SOURCE_DIR -B $BUILD_ROOT
    cmake --build $BUILD_ROOT -j 16 --target install
fi
