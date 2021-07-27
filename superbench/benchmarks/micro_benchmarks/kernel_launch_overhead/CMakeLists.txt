# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(kernel_launch_overhead LANGUAGES CXX)

include(../cuda_common.cmake)
find_package(CUDAToolkit QUIET)
include(../rocm_common.cmake)
find_package(HIP QUIET)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
    enable_language(CUDA)

    add_executable(kernel_launch_overhead cuda_kernel_launch.cu)
    set_property(TARGET kernel_launch_overhead PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)

# ROCm environment
elseif(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})

    # Add HIP targets
    set_source_files_properties(rocm_kernel_launch.cc PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    # Link with HIP
    hip_add_executable(kernel_launch_overhead rocm_kernel_launch.cc)
    # Install tergets
    install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)

else()
    message(FATAL_ERROR "No CUDA or ROCm environment found.")
endif()

