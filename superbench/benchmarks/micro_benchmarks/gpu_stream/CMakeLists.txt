# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

if(NOT CUDAToolkit_FOUND)
    message(WARNING "gpu_stream: CUDA not found, skipping build (requires NVIDIA GPU with NVML)")
    return()
endif()

# Check for NVML (nvidia-ml) library, required for querying memory clock rates
find_library(NVML_LIBRARY nvidia-ml PATHS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES stubs)
if(NOT NVML_LIBRARY)
    message(WARNING "gpu_stream: NVML (nvidia-ml) not found, skipping build")
    return()
endif()

message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

# Source files
set(SOURCES
    gpu_stream_test.cpp
    gpu_stream_utils.cpp
    gpu_stream.cu
    gpu_stream_kernels.cu
)

include(../cuda_common.cmake)
add_executable(gpu_stream ${SOURCES})
set_property(TARGET gpu_stream PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
target_include_directories(gpu_stream PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(gpu_stream numa ${NVML_LIBRARY})

install(TARGETS gpu_stream RUNTIME DESTINATION bin)
