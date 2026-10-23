# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
project(cublaslt_gemm LANGUAGES CXX)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND AND NOT CUDAToolkit_VERSION VERSION_LESS 11.8)
    include(CheckCXXSourceCompiles)
    set(CMAKE_CUDA_STANDARD 17)
    include(../cuda_common.cmake)
    set(CUDA_ARCH_LIST "80;86;90")
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8.0")
        list(APPEND CUDA_ARCH_LIST "100")
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.9.0")
        list(APPEND CUDA_ARCH_LIST "103")
    endif()
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.4.0")
        list(APPEND CUDA_ARCH_LIST "107")
    endif()

    set(CMAKE_REQUIRED_INCLUDES ${CUDAToolkit_INCLUDE_DIRS})
    check_cxx_source_compiles("
        #include <cublasLt.h>
        int main() {
            auto vec32 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_MN_K4_UE8M0;
            auto vec128 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC128_MN_K4_UE8M0;
            return static_cast<int>(vec32) + static_cast<int>(vec128);
        }
    " SUPERBENCH_HAS_MXFP8_MN_K4_SCALE)

    add_library(cublaslt_utils SHARED cublaslt_utils.cc)
    if(SUPERBENCH_HAS_MXFP8_MN_K4_SCALE)
        target_compile_definitions(cublaslt_utils PRIVATE SUPERBENCH_HAS_MXFP8_MN_K4_SCALE=1)
    endif()
    target_link_libraries(cublaslt_utils CUDA::cublas CUDA::cublasLt CUDA::cudart)
    set_target_properties(cublaslt_utils PROPERTIES LINK_FLAGS_RELEASE -s)
    install(TARGETS cublaslt_utils LIBRARY DESTINATION lib)

    add_executable(cublaslt_gemm cublaslt_gemm.cu)
    if(SUPERBENCH_HAS_MXFP8_MN_K4_SCALE)
        target_compile_definitions(cublaslt_gemm PRIVATE SUPERBENCH_HAS_MXFP8_MN_K4_SCALE=1)
    endif()
    target_link_libraries(cublaslt_gemm cublaslt_utils)
    set_target_properties(cublaslt_gemm PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCH_LIST}")

    install(TARGETS cublaslt_gemm RUNTIME DESTINATION bin)
endif()
