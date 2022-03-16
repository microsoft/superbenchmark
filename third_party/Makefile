
# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= /usr/local
MPI_HOME ?= /usr/local/mpi
HIP_HOME ?= /opt/rocm/hip
RCCL_HOME ?= /opt/rocm/rccl
ROCM_VERSION ?= rocm-$(shell dpkg -l | grep 'rocm-dev ' | awk '{print $$3}' | cut -d '.' -f1-3)
HPCX_HOME ?= /opt/hpcx

.PHONY: all cuda rocm common cuda_cutlass cuda_bandwidthTest cuda_nccl_tests cuda_perftest rocm_perftest fio rocm_rccl_tests rocm_rocblas rocm_bandwidthTest gpcnet cuda_gpuburn

# Build all targets.
all: cuda rocm
cuda: common cuda_cutlass cuda_bandwidthTest cuda_nccl_tests cuda_perftest gpcnet cuda_gpuburn
rocm: common rocm_perftest rocm_rccl_tests rocm_rocblas rocm_bandwidthTest
common: fio

# Create $(SB_MICRO_PATH)/bin and $(SB_MICRO_PATH)/lib, no error if existing, make parent directories as needed.
sb_micro_path:
	mkdir -p $(SB_MICRO_PATH)/bin
	mkdir -p $(SB_MICRO_PATH)/lib

# Build cutlass.
cuda_cutlass:
ifneq (,$(wildcard cutlass/CMakeLists.txt))
	cmake -DCMAKE_INSTALL_BINDIR=$(SB_MICRO_PATH)/bin -DCMAKE_INSTALL_LIBDIR=$(SB_MICRO_PATH)/lib -DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS='70;75;80;86' -DCUTLASS_ENABLE_EXAMPLES=OFF -DCUTLASS_ENABLE_TESTS=OFF -S ./cutlass -B ./cutlass/build
	cmake --build ./cutlass/build -j 8 --target install
endif

# Build cuda-samples/Samples/bandwidthTest.
# cuda-samples is released together with CUDA, they have the exact same version. Like v10.0, v11.1 and so on.
# The version we use is the released tag of cuda-samples which is consistent with the cuda version in the environment or docker.
# The Makefile of bandwidthTest does not have 'install' target, so need to copy bin to $(SB_MICRO_PATH)/bin/ and create $(SB_MICRO_PATH)/bin/ if not existing.
cuda_bandwidthTest: sb_micro_path
	if [ -d cuda-samples ]; then rm -rf cuda-samples; fi
	git clone -b v$(shell nvcc --version | grep 'release' | awk '{print $$6}' | cut -c2- | cut -d '.' -f1-2) https://github.com/NVIDIA/cuda-samples.git ./cuda-samples
	cd ./cuda-samples/Samples/bandwidthTest && make clean && make TARGET_ARCH=x86_64 SMS="70 75 80 86"
	cp -v ./cuda-samples/Samples/bandwidthTest/bandwidthTest $(SB_MICRO_PATH)/bin/

# Build nccl-tests.
# The version we use is commit 44df0bf from master branch, since it didn't update release tag for long time.
cuda_nccl_tests: sb_micro_path
ifneq (,$(wildcard nccl-tests/Makefile))
	cd ./nccl-tests && make MPI=1 MPI_HOME=$(MPI_HOME) -j
	cp -v ./nccl-tests/build/* $(SB_MICRO_PATH)/bin/
endif

# Build perftest.
# The version we use is the tag v4.5-0.2.
cuda_perftest:
ifneq (,$(wildcard perftest/autogen.sh))
	cd perftest && ./autogen.sh && ./configure CUDA_H_PATH=/usr/local/cuda/include/cuda.h --prefix=$(SB_MICRO_PATH) && make -j && make install
endif
rocm_perftest:
ifneq (,$(wildcard perftest/autogen.sh))
	cd perftest && ./autogen.sh && ./configure --enable-rocm --with-rocm=/opt/rocm --prefix=$(SB_MICRO_PATH) && make -j && make install
endif

# Build FIO from commit d83ac9 (fio-3.28 tag).
fio:
ifneq (,$(wildcard fio/Makefile))
	cd ./fio && ./configure --prefix=$(SB_MICRO_PATH) --disable-native && make -j && make install
endif

# Build rccl-tests from commit 602b745 of develop branch (default branch).
rocm_rccl_tests: sb_micro_path
ifneq (, $(wildcard rccl-tests/Makefile))
	cd ./rccl-tests && make MPI=1 MPI_HOME=$(MPI_HOME) HIP_HOME=$(HIP_HOME) RCCL_HOME=$(RCCL_HOME) -j
	cp -v ./rccl-tests/build/* $(SB_MICRO_PATH)/bin/
endif

# Build rocblas-bench.
# RocBLAS is released with rocm, like rocm-4.2.0 and so on.
# The version we use is the released tag which is consistent with the rocm version in the environment or docker.
# Since it takes several hours to build, avoid to build again if rocblas-bench exsists.
rocm_rocblas: sb_micro_path
ifeq (, $(wildcard $(SB_MICRO_PATH)/bin/rocblas-bench))
	if [ -d rocBLAS ]; then rm -rf rocBLAS; fi
	git clone -b ${ROCM_VERSION} https://github.com/ROCmSoftwarePlatform/rocBLAS.git ./rocBLAS
	cd ./rocBLAS && ./install.sh --dependencies --clients-only
	cp -v ./rocBLAS/build/release/clients/staging/rocblas-bench $(SB_MICRO_PATH)/bin/
endif

# Build hipBusBandwidth.
# HIP is released with rocm, like rocm-4.2.0 and so on.
# The version we use is the released tag which is consistent with the rocm version in the environment or docker.
rocm_bandwidthTest: sb_micro_path
	cp -r -v $(shell hipconfig -p)/samples/1_Utils/hipBusBandwidth ./
	cd ./hipBusBandwidth/ && mkdir -p build && cd build && cmake .. && make
	cp -v ./hipBusBandwidth/build/hipBusBandwidth $(SB_MICRO_PATH)/bin/

# Build GPCNET from commit c56fd9.
gpcnet: sb_micro_path
	bash -c "source ${HPCX_HOME}/hpcx-init.sh && hpcx_load && make CC=mpicc -C GPCNET all && hpcx_unload"
	cp -v ./GPCNET/network_test $(SB_MICRO_PATH)/bin/
	cp -v ./GPCNET/network_load_test $(SB_MICRO_PATH)/bin/

#Build GPU burn from main branch (only branch that exists)
cuda_gpuburn: sb_micro_path
ifneq (,$(wildcard gpu-burn/Makefile))
	cd ./gpu-burn && make
	cp -v ./gpu-burn/gpu_burn $(SB_MICRO_PATH)/bin/
	cp -v ./gpu-burn/compare.ptx $(SB_MICRO_PATH)/bin/
endif
