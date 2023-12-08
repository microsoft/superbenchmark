# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(kernel_launch_overhead LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(kernel_launch_overhead kernel_launch.cu)
    set_property(TARGET kernel_launch_overhead PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
else()
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(HIP QUIET)
    if(HIP_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})

        # Convert cuda code to hip code inplace
        execute_process(COMMAND hipify-perl -inplace -print-stats kernel_launch.cu
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

        # Add HIP targets
        set_source_files_properties(kernel_launch.cu PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
        # Link with HIP
        hip_add_executable(kernel_launch_overhead kernel_launch.cu)
        # Install tergets
        install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()
