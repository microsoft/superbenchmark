# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(c2c_copy LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(c2c_copy c2c_copy.cu)
    set_property(TARGET c2c_copy PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_link_libraries(c2c_copy numa)
else()
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found ROCm: " ${HIP_VERSION})

        # Convert cuda code to hip code in cpp
        execute_process(COMMAND hipify-perl -print-stats -o c2c_copy.cpp c2c_copy.cu WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

        # link hip device lib
        add_executable(c2c_copy c2c_copy.cpp)

        include(CheckSymbolExists)
        check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)
        if(${HIP_UNCACHED_MEMORY})
            target_compile_definitions(c2c_copy PRIVATE HIP_UNCACHED_MEMORY)
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
        target_link_libraries(c2c_copy numa hip::device)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()

install(TARGETS c2c_copy RUNTIME DESTINATION bin)
