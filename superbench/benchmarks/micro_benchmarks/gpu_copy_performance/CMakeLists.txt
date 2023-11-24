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
    target_link_libraries(gpu_copy numa)
else()
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found ROCm: " ${HIP_VERSION})
        # Convert cuda code to hip code inplace
        execute_process(COMMAND hipify-perl -print-stats -o gpu_copy.cpp gpu_copy.cu
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
 
        # Add HIP targets
        add_executable(gpu_copy gpu_copy.cpp)
        add_compile_options(-O2)
        target_link_libraries(gpu_copy numa hip::device)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()
 
install(TARGETS gpu_copy RUNTIME DESTINATION bin)