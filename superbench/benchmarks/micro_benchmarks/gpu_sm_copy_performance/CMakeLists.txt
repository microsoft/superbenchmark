# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_sm_copy LANGUAGES CXX)

include(../cuda_common.cmake)
find_package(CUDAToolkit QUIET)
include(../rocm_common.cmake)
find_package(HIP QUIET)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})
    enable_language(CUDA)

    add_executable(gpu_sm_copy gpu_sm_copy.cu)
    set_property(TARGET gpu_sm_copy PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    install(TARGETS gpu_sm_copy RUNTIME DESTINATION bin)

# ROCm environment
elseif(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
    
    # Convert cuda code to hip code inplace
    execute_process(COMMAND hipify-perl -inplace -print-stats gpu_sm_copy.cu
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

    # Add HIP targets
    set_source_files_properties(gpu_sm_copy.cu PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
    # Link with HIP
    hip_add_executable(gpu_sm_copy gpu_sm_copy.cu)
    # Install tergets
    install(TARGETS gpu_sm_copy RUNTIME DESTINATION bin)

else()
    message(FATAL_ERROR "No CUDA or ROCm environment found.")
endif()

