# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= "/usr/local"

.PHONY: all cutlass nccl_tests

# Build all targets.
all: cutlass nccl_tests

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
# Build nccl-tests.
nccl_tests: sb_micro_path
ifneq (,$(wildcard nccl-tests/Makefile))
	cd nccl-tests && make MPI=1 MPI_HOME=/usr/local/mpi/ -j
	cp nccl-tests/build/* $(SB_MICRO_PATH)/bin/
endif
