# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= "/usr/local"

.PHONY: all cutlass

# Build all targets.
all: cutlass

# Build cutlass.
cutlass:
ifneq (,$(wildcard cutlass/CMakeLists.txt))
	cmake -DCMAKE_INSTALL_BINDIR=$(SB_MICRO_PATH)/bin -DCMAKE_INSTALL_LIBDIR=$(SB_MICRO_PATH)/lib -DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS='70;80' -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_ENABLE_TESTS=OFF -S ./cutlass -B ./cutlass/build
	cmake --build ./cutlass/build --target install
endif
