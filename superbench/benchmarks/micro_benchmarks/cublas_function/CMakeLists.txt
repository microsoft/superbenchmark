# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(CublasBenchmark)

include(../cuda_common.cmake)

SET(SRC "cublas_benchmark.cu" CACHE STRING "source file")
SET(TARGET_NAME "cublas_function" CACHE STRING "target name")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
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
include_directories( ${CUDA_INCLUDE_DIRS})
find_library(CUDA_cuda_LIBRARY cuda ${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs)

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
    ${CUDA_LIBRARIES}
    nlohmann_json::nlohmann_json)

cuda_add_executable(CublasBenchmark cublas_test.cpp)   
target_link_libraries(CublasBenchmark ${TARGET_NAME} -lcublas) 

install(TARGETS CublasBenchmark ${TARGET_NAME} EXPORT CublasBenchmarkTargets)