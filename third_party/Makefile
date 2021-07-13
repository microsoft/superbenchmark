# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= "/usr/local"

.PHONY: all cutlass bandwidthTest nccl_tests

# Build all targets.
all: cutlass bandwidthTest nccl_tests

# Create $(SB_MICRO_PATH)/bin and $(SB_MICRO_PATH)/lib, no error if existing, make parent directories as needed.
sb_micro_path:
	mkdir -p $(SB_MICRO_PATH)/bin
	mkdir -p $(SB_MICRO_PATH)/lib
# Build cutlass.
cutlass: 
ifneq (,$(wildcard cutlass/CMakeLists.txt))
	cmake -DCMAKE_INSTALL_BINDIR=$(SB_MICRO_PATH)/bin -DCMAKE_INSTALL_LIBDIR=$(SB_MICRO_PATH)/lib -DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS='70;80' -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_ENABLE_TESTS=OFF -S ./cutlass -B ./cutlass/build
	cmake --build ./cutlass/build -j 8 --target install
endif
# Build cuda-samples/Samples/bandwidthTest.
# cuda-samples is released together with CUDA, they have the exact same version. Like v10.0, v11.1 and so on.
# The version we use is the released tag of cuda-samples which is consistent with the cuda version in the environment or docker.
# The Makefile of bandwidthTest does not have 'install' target, so need to copy bin to $(SB_MICRO_PATH)/bin/ and create $(SB_MICRO_PATH)/bin/ if not existing.
bandwidthTest: sb_micro_path
ifneq (,$(wildcard cuda-samples/Samples/bandwidthTest/Makefile))
	cd cuda-samples && git checkout v$(shell nvcc --version | grep 'release' | awk '{print $$6}' | cut -c2- | cut -d '.' -f1-2)
	cd ./cuda-samples/Samples/bandwidthTest && make clean && make TARGET_ARCH=x86_64 SMS="70 75 80 86"
	cp -v ./cuda-samples/Samples/bandwidthTest/bandwidthTest $(SB_MICRO_PATH)/bin/
endif
# Build nccl-tests.
# The version we use is just the master branch, due to it didn't update for long time.
nccl_tests: sb_micro_path
ifneq (,$(wildcard nccl-tests/Makefile))
	cd nccl-tests && make MPI=1 MPI_HOME=/usr/local/mpi/ -j
	cp nccl-tests/build/* $(SB_MICRO_PATH)/bin/
