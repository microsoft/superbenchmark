#!/bin/bash

# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License

set -e

MPI_HOME="${MPI_HOME:-/usr/local/mpi}"
SB_MICRO_PATH="${SB_MICRO_PATH:-/usr/local}"

for dir in micro_benchmarks/*/ ; do
    if [ -f $dir/CMakeLists.txt ]; then
        SOURCE_DIR=$dir
        BUILD_ROOT=$dir/build
        mkdir -p $BUILD_ROOT
        cmake -DCMAKE_PREFIX_PATH=$MPI_HOME -DCMAKE_INSTALL_PREFIX=$SB_MICRO_PATH -DCMAKE_BUILD_TYPE=Release -S $SOURCE_DIR -B $BUILD_ROOT
        cmake --build $BUILD_ROOT
        cmake --install $BUILD_ROOT
    fi
done
