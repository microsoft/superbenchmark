# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.24)

project(gpu_copy LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND) # CUDA environment
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(gpu_copy gpu_copy.cu)
    set_property(TARGET gpu_copy PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_link_libraries(gpu_copy numa)
else() # ROCm environment
    include(../rocm_common.cmake)
    hipify_sources(HIP_FILES gpu_copy.cu)
    add_executable(gpu_copy ${HIP_FILES})
    target_compile_options(gpu_copy PRIVATE -O2)
    target_link_libraries(gpu_copy PRIVATE numa)

    include(CheckSymbolExists)
    check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)
    if(${HIP_UNCACHED_MEMORY})
        target_compile_definitions(gpu_copy PRIVATE HIP_UNCACHED_MEMORY)
    endif()
endif()

install(TARGETS gpu_copy RUNTIME DESTINATION bin)
