# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.24)

project(kernel_launch_overhead LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND) # CUDA environment
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(kernel_launch_overhead kernel_launch.cu)
    set_property(TARGET kernel_launch_overhead PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
else() # ROCm environment
    include(../rocm_common.cmake)
    hipify_sources(HIP_FILES kernel_launch.cu)
    add_executable(kernel_launch_overhead ${HIP_FILES})
endif()

install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
