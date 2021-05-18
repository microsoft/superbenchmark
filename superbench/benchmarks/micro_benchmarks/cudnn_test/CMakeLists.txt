# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(cudnnBenchmark)

include(../cuda_common.cmake)

SET(SRC "cudnn_benchmark.cu" CACHE STRING "source file")
SET(TARGET_NAME "cudnn_function" CACHE STRING "target name")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra -std=c++11 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(CUDA)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_ARCHS_SUPPORTED}")
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

include(FetchContent)

FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.7.3)

FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()


target_link_libraries(${TARGET_NAME}
    ${CUDA_cuda_LIBRARY}
    ${CUDA_cudart_LIBRARY}
    ${CUDA_curand_LIBRARY}
    ${CUDA_LIBRARIES}
    ${CUDA_cudnn_LIBRARIES}
    ${CUDNN_LIBRARY} nlohmann_json::nlohmann_json)

cuda_add_executable(cudnnBenchmark cudnn_test.cpp)   
target_link_libraries(cudnnBenchmark ${TARGET_NAME} ) 

install(TARGETS cudnnBenchmark ${TARGET_NAME} EXPORT cudnnBenchmarkTargets)






