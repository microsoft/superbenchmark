
# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= "/usr/local"
MPI_HOME_PATH ?= "/usr/local/mpi"
HIP_HOME ?= "/opt/rocm/hip"
RCCL_HOME ?= "/opt/rocm/rccl"
USE_CUDA ?= "1"
USE_ROCM ?= "0"

.PHONY: all common cuda rocm cutlass bandwidthTest nccl_tests perftest fio rccl_tests

# Build all targets for cuda.
all: cuda rocm
cuda: common cutlass bandwidthTest nccl_tests
rocm: common rccl_tests
common: perftest fio

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
	if [ -d cuda-samples ]; then rm -rf cuda-samples; fi
	git clone -b v$(shell nvcc --version | grep 'release' | awk '{print $$6}' | cut -c2- | cut -d '.' -f1-2) https://github.com/NVIDIA/cuda-samples.git ./cuda-samples
	cd ./cuda-samples/Samples/bandwidthTest && make clean && make TARGET_ARCH=x86_64 SMS="70 75 80 86"
	cp -v ./cuda-samples/Samples/bandwidthTest/bandwidthTest $(SB_MICRO_PATH)/bin/

# Build nccl-tests.
# The version we use is commit 44df0bf from master branch, since it didn't update release tag for long time.
nccl_tests: sb_micro_path
ifneq (,$(wildcard nccl-tests/Makefile))
	cd ./nccl-tests && make MPI=1 MPI_HOME=$(MPI_HOME_PATH) -j
	cp -v ./nccl-tests/build/* $(SB_MICRO_PATH)/bin/
endif

# Build perftest.
# The version we use is the tag v4.5-0.2.
perftest:
ifneq (,$(wildcard perftest/autogen.sh))
ifeq (1,$(USE_CUDA))
	cd perftest && ./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h --prefix=$(SB_MICRO_PATH) && make -j && make install
else ifeq (1,$(USE_ROCM))
	cd perftest && ./autogen.sh && ./configure --enable-rocm --with-rocm=/opt/rocm --prefix=$(SB_MICRO_PATH) && make -j && make install
else
	cd perftest && ./autogen.sh && ./configure --prefix=$(SB_MICRO_PATH) && make -j && make install
endif
endif

# Build FIO from commit 0313e9 (fio-3.27 tag).
fio:
ifneq (,$(wildcard fio/Makefile))
	cd ./fio && ./configure --prefix=$(SB_MICRO_PATH) && make -j && make install
endif

# Build rccl-tests from commit cc34c5 of develop branch (default branch).
rccl_tests: sb_micro_path
ifneq (, $(wildcard rccl-tests/Makefile))
	cd ./rccl-tests && make MPI=1 MPI_HOME=$(MPI_HOME_PATH) HIP_HOME=$(HIP_HOME) RCCL_HOME=$(RCCL_HOME) -j
	cp -v ./rccl-tests/build/* $(SB_MICRO_PATH)/bin/
endif
