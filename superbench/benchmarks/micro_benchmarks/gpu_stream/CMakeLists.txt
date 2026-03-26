# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

# Source files
set(SOURCES
    gpu_stream_test.cpp
    gpu_stream_utils.cpp
    gpu_stream.cu
    gpu_stream_kernels.cu
)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(gpu_stream ${SOURCES})
    set_property(TARGET gpu_stream PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_include_directories(gpu_stream PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gpu_stream numa nvidia-ml)
else()
    message(WARNING "gpu_stream: CUDA not found, skipping build (requires NVIDIA GPU with NVML)")
    return()
endif()

install(TARGETS gpu_stream RUNTIME DESTINATION bin)
