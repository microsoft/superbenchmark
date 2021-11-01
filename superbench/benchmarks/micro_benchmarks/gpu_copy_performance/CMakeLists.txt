# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_copy LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(gpu_copy gpu_copy.cu)
    set_property(TARGET gpu_copy PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
else()
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found ROCm: " ${HIP_VERSION})

        # Convert cuda code to hip code inplace
        execute_process(COMMAND hipify-perl -inplace -print-stats gpu_copy.cu
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

        # Add HIP targets
        set_source_files_properties(gpu_copy.cu PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
        # Link with HIP
        hip_add_executable(gpu_copy gpu_copy.cu)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()

install(TARGETS gpu_copy RUNTIME DESTINATION bin)
target_link_libraries(gpu_copy numa)
