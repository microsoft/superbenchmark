# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(CublasBenchmark)

include(../cuda_common.cmake)

SET(SRC "function.cu" CACHE STRING "source file")
SET(TARGET_NAME "cublas_function" CACHE STRING "target name")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_ARCHS_SUPPORTED}")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -ftemplate-depth=4096 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -cudart shared")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
cuda_add_library(${TARGET_NAME} SHARED ${SRC})

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES cuda/include include)

include_directories(${CUDNN_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})

find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

find_library(CUDA_cuda_LIBRARY cuda ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)
find_library(CUDA_cudart_LIBRARY libcudart.so ${CUDA_TOOLKIT_ROOT_DIR}/lib64)

target_link_libraries(${TARGET_NAME}
    ${CUDA_cuda_LIBRARY}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDNN_LIBRARY})

cuda_add_executable(CublasBenchmark cublas_function_test.cpp)   
target_link_libraries(CublasBenchmark ${TARGET_NAME}) 

install(TARGETS ${TARGET_NAME}  LIBRARY DESTINATION /usr/lib)
install(TARGETS CublasBenchmark EXPORT CublasBenchmarkTargets)