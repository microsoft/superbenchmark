# Copyright (c) Microsoft Corporation - All rights reserved
# Licensed under the MIT License


SB_MICRO_PATH ?= /usr/local
MPI_HOME ?= /usr/local/mpi
HIP_HOME ?= /opt/rocm/hip
RCCL_HOME ?= /opt/rocm/rccl
HPCX_HOME ?= /opt/hpcx
ROCM_PATH ?= /opt/rocm

CUDA_VER ?= $(shell nvcc --version | grep 'release' | awk '{print $$6}' | cut -c2- | cut -d '.' -f1-2)
ROCBLAS_BRANCH ?= rocm-$(shell dpkg -l | grep 'rocm-dev ' | awk '{print $$3}' | cut -d '.' -f1-3)
HIPBLASLT_BRANCH ?= rocm-$(shell dpkg -l | grep 'rocm-dev ' | awk '{print $$3}' | cut -d '.' -f1-3)
ROCM_VER ?= $(shell hipconfig -R | grep -oP '\d+\.\d+\.\d+' || echo "0.0.0")

NUM_MAKE_JOBS ?= $(shell nproc --ignore=2)

.PHONY: all cuda_with_msccl cuda rocm common cuda_cutlass cuda_bandwidthTest cuda_nccl_tests cuda_perftest cuda_msccl rocm_perftest fio rocm_rccl_tests rocm_rocblas rocm_bandwidthTest gpcnet cuda_gpuburn cpu_stream cpu_hpl directx_amf_encoding_latency directx_amd rocm_hipblaslt megatron_lm megatron_deepspeed apex_rocm nvbandwidth rocm_megatron_lm

# Build targets.
all: cuda rocm
# msccl: api change in cudaStreamUpdateCaptureDependencies
cuda_with_msccl: cuda cuda_msccl
# cuda13 target: 
# removed cuda_perftest due to API change in cuCtxCreate_v4 - CUresult CUDAAPI cuCtxCreate(CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev);
cuda13: common cuda_cutlass cuda_bandwidthTest cuda_nccl_tests gpcnet cuda_gpuburn megatron_lm megatron_deepspeed nvbandwidth
cuda: common cuda_cutlass cuda_bandwidthTest cuda_nccl_tests cuda_perftest gpcnet cuda_gpuburn megatron_lm megatron_deepspeed nvbandwidth
rocm: common rocm_perftest rocm_rccl_tests rocm_rocblas rocm_bandwidthTest rocm_hipblaslt megatron_deepspeed apex_rocm rocm_megatron_lm
cpu: common cpu_perftest
common: fio cpu_stream

# non aarch64 specific targets
ifneq ($(shell uname -m), aarch64)
common: cpu_hpl
directx_amd: directx_amf_encoding_latency
endif

# Create $(SB_MICRO_PATH)/bin and $(SB_MICRO_PATH)/lib, no error if existing, make parent directories as needed.
sb_micro_path:
	mkdir -p $(SB_MICRO_PATH)/bin
	mkdir -p $(SB_MICRO_PATH)/lib

# Build cutlass.
# for cuda 12.9 and later Build from commit v3.9 (3.9 release commit) for blackwell support
cuda_cutlass:
ifeq ($(shell echo $(CUDA_VER)">=12.9" | bc -l), 1)
	$(eval ARCHS := "100;103")
	if [ -d cutlass ]; then rm -rf cutlass; fi
	git clone --branch v4.1.0 --depth 1 https://github.com/NVIDIA/cutlass.git && cd cutlass
else ifeq ($(shell echo $(CUDA_VER)">=12.8" | bc -l), 1)
	$(eval ARCHS := "90;100")
	if [ -d cutlass ]; then rm -rf cutlass; fi
	git clone --branch v3.9.2 --depth 1 https://github.com/NVIDIA/cutlass.git && cd cutlass
else ifeq ($(shell echo $(CUDA_VER)">=11.8" | bc -l), 1)
	$(eval ARCHS := "70;75;80;86;89;90")
else
	$(eval ARCHS := "70;75;80;86")
endif

ifneq (,$(wildcard cutlass/CMakeLists.txt))
	cmake -DCMAKE_INSTALL_BINDIR=$(SB_MICRO_PATH)/bin \
		-DCMAKE_INSTALL_LIBDIR=$(SB_MICRO_PATH)/lib \
		-DCMAKE_BUILD_TYPE=Release \
		-DCUTLASS_NVCC_ARCHS=$(ARCHS) \
		-DCUTLASS_ENABLE_EXAMPLES=OFF \
		-DCUTLASS_ENABLE_TESTS=OFF \
		-S ./cutlass \
		-B ./cutlass/build \
		-DCUTLASS_LIBRARY_KERNELS="cutlass_simt_dgemm_128x128_8x2_*,\
cutlass_simt_sgemm_128x128_8x2_*,\
cutlass_simt_hgemm_256x128_8x2_*,\
cutlass_tensorop_h884gemm_256x128_32x2_*,\
cutlass_tensorop_d884gemm_128x128_16x3_*,\
cutlass_tensorop_tf32_s1688gemm_tf32_256x128_16x3_*,\
cutlass_tensorop_bf16_s16816gemm_bf16_256x128_32x3_*,\
cutlass_tensorop_h16816gemm_256x128_32x3_*,\
cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_*,\
cutlass_tensorop_s4_i16864gemm_s4_256x128_128x3_*"
	cmake --build ./cutlass/build -j $(shell nproc --ignore=2) --target install
	rm -rf ./cutlass/build
endif

# Build cuda-samples/Samples/bandwidthTest.
# cuda-samples is released together with CUDA, they have the exact same version. Like v10.0, v11.1 and so on.
# The version we use is the released tag of cuda-samples which is consistent with the cuda version in the environment or docker.
# The Makefile of bandwidthTest does not have 'install' target, so need to copy bin to $(SB_MICRO_PATH)/bin/ and create $(SB_MICRO_PATH)/bin/ if not existing.
# The bandwidthTest sample was out-of-date and has been removed as of the CUDA Samples 12.9 release (see the change log). For up-to-date bandwidth measurements, refer instead to the NVBandwith utility.
cuda_bandwidthTest: sb_micro_path
	if [ -d cuda-samples ]; then rm -rf cuda-samples; fi
	git clone --depth 1 -b v$(CUDA_VER) https://github.com/NVIDIA/cuda-samples.git
ifeq ($(shell echo $(CUDA_VER)">=12.9" | bc -l), 1)
	@echo "Skip cuda-samples build for CUDA>=12.9"
else ifeq ($(shell echo $(CUDA_VER)">=12.8" | bc -l), 1)
	$(eval TEST_PATH := "./cuda-samples/Samples/1_Utilities/bandwidthTest")
	cd ./$(TEST_PATH) && mkdir build && cd build && cmake .. && make
	cp -v ./$(TEST_PATH)/build/bandwidthTest $(SB_MICRO_PATH)/bin/
else ifeq ($(shell echo $(CUDA_VER)">=11.8" | bc -l), 1)
	$(eval TEST_PATH := "./cuda-samples/Samples/1_Utilities/bandwidthTest")
	$(eval ARCHS := "70 75 80 86 90")
	cd ./$(TEST_PATH) && make clean && make SMS=$(ARCHS)
	cp -v ./$(TEST_PATH)/bandwidthTest $(SB_MICRO_PATH)/bin/
else
	$(eval TEST_PATH := "./cuda-samples/Samples/bandwidthTest")
	$(eval ARCHS := "70 75 80 86")
	cd ./$(TEST_PATH) && make clean && make SMS=$(ARCHS)
	cp -v ./$(TEST_PATH)/bandwidthTest $(SB_MICRO_PATH)/bin/
endif

# Build nccl-tests.
# The version we use is the tag v2.13.13
cuda_nccl_tests: sb_micro_path
ifneq (,$(wildcard nccl-tests/Makefile))
	cd ./nccl-tests && make MPI=1 MPI_HOME=$(MPI_HOME) -j
	cp -v -r ./nccl-tests/build/* $(SB_MICRO_PATH)/bin/
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
cpu_perftest:
ifneq (,$(wildcard perftest/autogen.sh))
	cd perftest && ./autogen.sh && ./configure --prefix=$(SB_MICRO_PATH) && make -j && make install
endif

# Build FIO from commit d83ac9 (fio-3.28 tag).
fio:
ifneq (,$(wildcard fio/Makefile))
	cd ./fio && ./configure --prefix=$(SB_MICRO_PATH) --disable-native && make -j && make install
endif

# Build rccl-tests from commit 46375b1 of default branch.
rocm_rccl_tests: sb_micro_path
ifneq (, $(wildcard rccl-tests/Makefile))
	cd ./rccl-tests && make MPI=1 MPI_HOME=$(MPI_HOME) -j
	cp -v -r ./rccl-tests/build/* $(SB_MICRO_PATH)/bin/
endif

# Build rocblas-bench.
# RocBLAS is released with rocm, like rocm-4.2.0 and so on.
# The version we use is the released tag which is consistent with the rocm version in the environment or docker.
# Since it takes several hours to build, avoid to build again if rocblas-bench exsists.
rocm_rocblas: sb_micro_path
	@if [ ! -e $(SB_MICRO_PATH)/bin/rocblas-bench ] && [ -z `which rocblas-bench` ]; then \
		if [ -d rocBLAS ]; then rm -rf rocBLAS; fi; \
		git clone -b ${ROCBLAS_BRANCH} https://github.com/ROCmSoftwarePlatform/rocBLAS.git ./rocBLAS; \
		sed -i 's|#include "gemm.hpp"|#include "Tensile/gemm.hpp"|' rocBLAS/clients/benchmarks/../../library/src/blas3/rocblas_trtri.hpp; \
		cd ./rocBLAS && ./install.sh --dependencies --clients-only; \
		cp -v $(SB_MICRO_PATH)/third_party/rocBLAS/build/release/clients/staging/rocblas-bench $(SB_MICRO_PATH)/bin/; \
	fi

# Build hipblaslt-bench.
# hipBLASLt is released with rocm, like rocm-4.2.0 and so on.
# The version we use is the released tag which is consistent with the rocm version in the environment or docker.
# Since it takes several hours to build, avoid to build again if hipblaslt-bench exsists.
rocm_hipblaslt: sb_micro_path
	@if [ ! -e $(SB_MICRO_PATH)/bin/hipblaslt-bench ] && [ -z `which hipblaslt-bench` ]; then \
		if [ -d hipBLASLt ]; then rm -rf hipBLASLt; fi; \
		git clone -b ${HIPBLASLT_BRANCH} https://github.com/ROCmSoftwarePlatform/hipBLASLt.git ./hipBLASLt; \
		cd ./hipBLASLt && ./install.sh -dc; \
		cp -v $(SB_MICRO_PATH)/third_party/hipBLASLt/build/release/clients/staging/hipblaslt-bench $(SB_MICRO_PATH)/bin/;  \
	fi

# Build hipBusBandwidth.
# HIP is released with rocm, like rocm-4.2.0 and so on.
# The version we use is the released tag which is consistent with the rocm version in the environment or docker.
rocm_bandwidthTest: sb_micro_path
	git clone -b ${ROCM_VER} https://github.com/ROCm-Developer-Tools/HIP.git
	cd ./HIP/samples/1_Utils/hipBusBandwidth/ && mkdir -p build && cd build && cmake .. && make
	cp -v ./HIP/samples/1_Utils/hipBusBandwidth/build/hipBusBandwidth $(SB_MICRO_PATH)/bin/

# Build GPCNET from commit c56fd9.
gpcnet: sb_micro_path
	bash -c "source ${HPCX_HOME}/hpcx-init.sh && hpcx_load && make CC=mpicc -C GPCNET all && hpcx_unload"
	cp -v ./GPCNET/network_test $(SB_MICRO_PATH)/bin/
	cp -v ./GPCNET/network_load_test $(SB_MICRO_PATH)/bin/

# Build GPU burn from main branch (only branch that exists)
cuda_gpuburn: sb_micro_path
ifneq (,$(wildcard gpu-burn/Makefile))
	cd ./gpu-burn && make
	cp -v ./gpu-burn/gpu_burn $(SB_MICRO_PATH)/bin/
	cp -v ./gpu-burn/compare.ptx $(SB_MICRO_PATH)/bin/
endif

# Build HPL from main branch
cpu_hpl: sb_micro_path
ifneq (,$(wildcard hpl-tests/Makefile))
	cd ./hpl-tests && \
    wget https://netlib.org/benchmark/hpl/hpl-2.3.tar.gz && \
	tar xzf hpl-2.3.tar.gz && \
	cp Make.Linux_zen3 hpl-2.3 && \
	cp Make.Linux_zen4 hpl-2.3 && \
	make all
	cp -v ./hpl-tests/hpl-2.3/bin/Linux_zen3/xhpl $(SB_MICRO_PATH)/bin/xhpl_z3
	cp -v ./hpl-tests/hpl-2.3/bin/Linux_zen4/xhpl $(SB_MICRO_PATH)/bin/xhpl_z4
	cp -v ./hpl-tests/hpl_run.sh $(SB_MICRO_PATH)/bin/
	cp -v ./hpl-tests/bindmem.sh $(SB_MICRO_PATH)/bin/
	cp -v ./hpl-tests/template_hpl.dat $(SB_MICRO_PATH)/bin/
endif

# Build STREAM
cpu_stream: sb_micro_path
ifneq (,$(wildcard stream-tests/Makefile))
	cd ./stream-tests && \
    wget https://www.cs.virginia.edu/stream/FTP/Code/stream.c && \
	make all
	cp -v ./stream-tests/stream* $(SB_MICRO_PATH)/bin/
endif

# Build AMD Encoder Latency Test
directx_amf_encoding_latency:
	@if not exist "AMF" (git clone -b v1.4.29 https://github.com/GPUOpen-LibrariesAndSDKs/AMF.git)
	@if exist "AMF\amf\public\samples\CPPSamples_vs2019.sln" ( \
		curl -L -o vs_buildtools.exe https://aka.ms/vs/16/release/vs_buildtools.exe && echo "Downloaded vs_buildtools.exe" && \
		start /wait vs_buildtools.exe --quiet --wait --norestart --nocache --installPath C:/temp/BuildTools --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.ATLMFC --includeRecommended  && echo "Installed VS Build Tools" && \
		del vs_buildtools.exe && echo "Deleted vs_buildtools.exe" && \
		"C:\temp\BuildTools\MSBuild\Current\Bin\MSBuild.exe" "AMF\amf\public\samples\CPPSamples_vs2019.sln" /t:EncoderLatency /p:Platform=x64 /p:Configuration=Release /p:OutDir="%SB_MICRO_PATH%\bin" \
	)

# Install requirements for Megatron-LM
megatron_lm:
	cd Megatron && \
	apt install -y python3-mpi4py && \
	python -m pip install --no-cache-dir -r requirements.txt

# Install requirements for Megatron-DeepSpeed
megatron_deepspeed:
	cd Megatron && \
	apt install -y python3-mpi4py && \
	python -m pip install --no-cache-dir -r requirements.txt && \
	python -m pip install DeepSpeed

rocm_megatron_lm:
	cd Megatron && mkdir -p rocm && cd rocm && \
	if [ ! -d "Megatron-LM" ]; then \
		git clone -b rocm_dev https://github.com/ROCm/Megatron-LM.git ; \
	fi
	cp Megatron/rocm/Megatron-LM/examples/deepseek_v2/pretrain_deepseek.py Megatron/rocm/Megatron-LM/
	git clone https://github.com/caaatch22/grouped_gemm.git &&\
    	cd grouped_gemm &&\
    	git checkout 8a9b438 &&\
    	git submodule update --init --recursive &&\
    	pip install .

# Instal apex of ROCm due to dependency of Megatron
apex_rocm:
	$(eval TORCH_VERSION ?= $(shell python -c "import torch; print(torch.__version__)"))
	$(eval TORCH_MAJOR_VERSION ?= $(word 1,$(subst ., ,$(TORCH_VERSION))))
	$(eval TORCH_MINOR_VERSION ?= $(word 2,$(subst ., ,$(TORCH_VERSION))))
	if [ ! -d "apex" ]; then \
		git clone https://github.com/ROCmSoftwarePlatform/apex.git ; \
	fi
	cd apex && \
	if [ "$$(expr $(TORCH_MAJOR_VERSION) \> 2)" -eq 1 ] && [ "$$(expr $(TORCH_MINOR_VERSION) \> 1)" -eq 1 ]; then \
		git checkout master ; \
	elif [ "$$(expr $(TORCH_MAJOR_VERSION) == 2)" -eq 1 ] && [ "$$(expr $(TORCH_MINOR_VERSION) == 1)" -eq 1 ]; then \
		git checkout release/1.1.0 ; \
	elif [ "$$(expr $(TORCH_MAJOR_VERSION) == 2)" -eq 1 ] && [ "$$(expr $(TORCH_MINOR_VERSION) == 0)" -eq 1 ]; then \
		git checkout release/1.0.0 ; \
	elif [ "$$(expr $(TORCH_MAJOR_VERSION) == 1)" -eq 1 ]; then \
	    git checkout release/1.0.0 ; \
	fi
	pip install -v --disable-pip-version-check --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex

# Build MSCCL for CUDA
cuda_msccl: sb_micro_path
ifeq ($(shell echo $(CUDA_VER)">=12.9" | bc -l), 1)
	$(eval ARCHS := 100 103)
	if [ -d msccl ]; then rm -rf msccl; fi; \
	git clone --single-branch --branch main https://github.com/Azure/msccl.git \
	&& git -C msccl checkout 87048bd && git -C msccl submodule update --recursive --init
else ifeq ($(shell echo $(CUDA_VER)">=12.8" | bc -l), 1)
    # Get commit 87048bd from msscl to support updated nccl and sm_100
	$(eval ARCHS := 75 80 86 89 90 100)
	if [ -d msccl ]; then rm -rf msccl; fi; \
	git clone --single-branch --branch main https://github.com/Azure/msccl.git \
    && git -C msccl checkout 87048bd && git -C msccl submodule update --recursive --init
else ifeq ($(shell echo $(CUDA_VER)">=11.8" | bc -l), 1)
	$(eval ARCHS := 70 75 80 86 89 90)
else
	$(eval ARCHS := 70 75 80 86")
endif
	$(eval NVCC_GENCODE := "$(foreach arch, $(ARCHS), $(NVCC_GENCODE) -gencode=arch=compute_$(arch),code=sm_$(arch))")
ifneq (,$(wildcard msccl/executor/msccl-executor-nccl/Makefile))
	cd ./msccl/executor/msccl-executor-nccl && \
	make -j ${NUM_MAKE_JOBS} src.build NVCC_GENCODE=$(NVCC_GENCODE) && \
	cd ../../..
	mkdir -p $(SB_MICRO_PATH)/lib/msccl-executor-nccl && \
	cp -r -v ./msccl/executor/msccl-executor-nccl/build/* $(SB_MICRO_PATH)/lib/msccl-executor-nccl/
endif
ifneq (,$(wildcard msccl/scheduler/msccl-scheduler/Makefile))
	cd ./msccl/scheduler/msccl-scheduler && \
	CXX=nvcc BIN_HOME=$(SB_MICRO_PATH)/lib/msccl-executor-nccl SRC_HOME=../../../msccl/executor/msccl-executor-nccl make -j ${NUM_MAKE_JOBS} && \
	cd ../../..
	mkdir -p $(SB_MICRO_PATH)/lib/msccl-scheduler && \
	cp -r -v ./msccl/scheduler/msccl-scheduler/build/* $(SB_MICRO_PATH)/lib/msccl-scheduler/
endif
ifneq (,$(wildcard msccl/tests/msccl-tests-nccl/Makefile))
	cd ./msccl/tests/msccl-tests-nccl && \
	make MPI=1 MPI_HOME=$(MPI_HOME) NCCL_HOME=$(SB_MICRO_PATH)/lib/msccl-executor-nccl NVCC_GENCODE=$(NVCC_GENCODE) -j ${NUM_MAKE_JOBS} && cd ../../..
	mkdir -p $(SB_MICRO_PATH)/bin/msccl-tests-nccl && \
	cp -r -v ./msccl/tests/msccl-tests-nccl/build/* $(SB_MICRO_PATH)/bin/msccl-tests-nccl/
endif

# Build nvbandwidth.
nvbandwidth: sb_micro_path
	cd ./nvbandwidth && mkdir -p build && cd build && \
	cmake .. && make -j && cd ../..
	cp -v ./nvbandwidth/build/nvbandwidth $(SB_MICRO_PATH)/bin
