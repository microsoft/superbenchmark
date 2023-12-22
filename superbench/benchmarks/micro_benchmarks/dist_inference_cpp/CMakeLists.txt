# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(dist_inference LANGUAGES CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(CUDAToolkit QUIET)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(dist_inference dist_inference.cu)
    set_property(TARGET dist_inference PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_link_libraries(dist_inference MPI::MPI_CXX nccl cublasLt)
else()
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found ROCm: " ${HIP_VERSION})

        # Convert cuda code to hip code in cpp
        execute_process(COMMAND hipify-perl -print-stats -o dist_inference.cpp dist_inference.cu WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)

        # link hip device lib
        add_executable(dist_inference dist_inference.cpp)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DROCM_USE_FLOAT16=1")
        if(DEFINED ENV{USE_HIPBLASLT_DATATYPE})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_HIPBLASLT_DATATYPE=1")
        elseif(DEFINED ENV{USE_HIP_DATATYPE})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_HIP_DATATYPE=1")
        endif()
        if(DEFINED ENV{USE_HIPBLAS_COMPUTETYPE})
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_HIPBLAS_COMPUTETYPE=1")
        endif()
        target_link_libraries(dist_inference MPI::MPI_CXX rccl hipblaslt hip::device)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()

install(TARGETS dist_inference RUNTIME DESTINATION bin)
