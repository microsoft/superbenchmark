# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(kernel_launch_overhead LANGUAGES CXX)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    # Cuda environment
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
    enable_language(CUDA)

    include(../cuda_common.cmake)

    add_executable(kernel_launch_overhead cuda_kernel_launch.cu)
    set_property(TARGET kernel_launch_overhead PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
else()
    # ROCm environment
    # Set ROCM_PATH
    if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH /opt/rocm)
    else()
    set(ROCM_PATH $ENV{ROCM_PATH})
    endif()
    # Set HIP_PATH
    if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH ${ROCM_PATH}/hip)
    else()
    set(HIP_PATH $ENV{HIP_PATH})
    endif()
    # Check if ROCm environment
    if(EXISTS ${HIP_PATH})
        # Search for hip in common locations
        list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH})
        set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
        set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})

        # Find hip
        find_package(HIP QUIET)
        if(HIP_FOUND)
            message(STATUS "Found HIP: " ${HIP_VERSION})

            # Add HIP targets
            set_source_files_properties(rocm_kernel_launch.cc PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
            # Link with HIP
            hip_add_executable(kernel_launch_overhead rocm_kernel_launch.cc)
            install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
        else()
            message(FATAL_ERROR "No HIP found. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
        endif()
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()


