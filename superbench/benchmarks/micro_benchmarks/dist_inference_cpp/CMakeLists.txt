# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.24)

project(dist_inference LANGUAGES CXX)

find_package(MPI REQUIRED)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND) # CUDA environment
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(dist_inference dist_inference.cu)
    set_property(TARGET dist_inference PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_link_libraries(dist_inference PRIVATE MPI::MPI_CXX nccl cublasLt)
else() # ROCm environment
    include(../rocm_common.cmake)

    hipify_sources(HIP_FILES dist_inference.cu)
    add_executable(dist_inference ${HIP_FILES})
    target_compile_options(dist_inference PRIVATE -O2)
    target_compile_definitions(dist_inference PRIVATE ROCM_USE_FLOAT16=1)

    if(DEFINED ENV{USE_HIPBLASLT_DATATYPE})
        target_compile_definitions(dist_inference PRIVATE USE_HIPBLASLT_DATATYPE=1)
    elseif(DEFINED ENV{USE_HIP_DATATYPE})
        target_compile_definitions(dist_inference PRIVATE USE_HIP_DATATYPE=1)
    endif()
    if(DEFINED ENV{USE_HIPBLAS_COMPUTETYPE})
        target_compile_definitions(dist_inference PRIVATE USE_HIPBLAS_COMPUTETYPE=1)
    endif()

    target_link_libraries(dist_inference PRIVATE MPI::MPI_CXX rccl hipblaslt)
endif()

install(TARGETS dist_inference RUNTIME DESTINATION bin)
