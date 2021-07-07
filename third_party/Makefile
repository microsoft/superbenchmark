# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= "/usr/local"

.PHONY: all cutlass bandwidthTest

# Build all targets.
all: cutlass bandwidthTest

# Create $(SB_MICRO_PATH)/bin and  $(SB_MICRO_PATH)/lib if not exist.
sb_micro_path:
	if [ ! -d  $(SB_MICRO_PATH)/bin ]; then mkdir -p $(SB_MICRO_PATH)/bin; fi
	if [ ! -d $(SB_MICRO_PATH)/lib ]; then mkdir -p $(SB_MICRO_PATH)/lib; fi
# Build cutlass.
cutlass:
ifneq (,$(wildcard cutlass/CMakeLists.txt))
	cmake -DCMAKE_INSTALL_BINDIR=$(SB_MICRO_PATH)/bin -DCMAKE_INSTALL_LIBDIR=$(SB_MICRO_PATH)/lib -DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS='70;80' -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_ENABLE_TESTS=OFF -S ./cutlass -B ./cutlass/build
	cmake --build ./cutlass/build -j 8 --target install
endif
bandwidthTest: sb_micro_path
ifneq (,$(wildcard cuda-samples/Samples/bandwidthTest/Makefile))
	make -C ./cuda-samples/Samples/bandwidthTest TARGET_ARCH=x86_64 SMS="70 75 80 86" 
	cp ./cuda-samples/Samples/bandwidthTest/bandwidthTest $(SB_MICRO_PATH)/bin/
endif
