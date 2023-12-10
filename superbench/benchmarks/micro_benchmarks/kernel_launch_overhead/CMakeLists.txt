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
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found HIP: " ${HIP_VERSION})

        # Convert cuda code to hip code in cpp
        execute_process(COMMAND hipify-perl -print-stats -o kernel_launch.cpp kernel_launch.cu WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

        # link hip device lib
        add_executable(kernel_launch_overhead kernel_launch.cpp)
        target_link_libraries(kernel_launch_overhead hip::device)
        # Install tergets
        install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()
