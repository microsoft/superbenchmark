# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(cublaslt_fp8_gemm LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND AND NOT CUDAToolkit_VERSION VERSION_LESS 11.8)
    set(CMAKE_CUDA_STANDARD 17)
    include(../cuda_common.cmake)

    add_library(cublaslt_utils SHARED cublaslt_utils.cc)
    target_link_libraries(cublaslt_utils CUDA::cublas CUDA::cublasLt)
    set_target_properties(cublaslt_utils PROPERTIES LINK_FLAGS_RELEASE -s)
    install(TARGETS cublaslt_utils LIBRARY DESTINATION lib)

    add_executable(cublaslt_fp8_gemm cublaslt_fp8_gemm.cu)
    target_link_libraries(cublaslt_fp8_gemm cublaslt_utils)
    set_target_properties(cublaslt_fp8_gemm PROPERTIES CUDA_ARCHITECTURES "80;86;90")
    install(TARGETS cublaslt_fp8_gemm RUNTIME DESTINATION bin)
endif()
