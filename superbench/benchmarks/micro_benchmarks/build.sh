#!/bin/bash


# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_PATH="${SB_PATH:-/usr/local/bin}"
SB_LIB="${SB_LIB:-/usr/local/lib}"

for dir in */ ; do
    if [ -f $dir/CMakeLists.txt ]; then
        SOURCE_DIR=$dir
        BUILD_ROOT=$dir/build
        mkdir -p $BUILD_ROOT
        cmake -DCMAKE_INSTALL_PREFIX=$SB_PATH -DCMAKE_BUILD_TYPE=Release -S $SOURCE_DIR -B $BUILD_ROOT
        cmake --build $BUILD_ROOT --target install
    fi
done
