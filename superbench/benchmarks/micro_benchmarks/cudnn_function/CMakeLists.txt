# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(CudnnBenchmark LANGUAGES CUDA CXX)

include(../cuda_common.cmake)

SET(SRC "cudnn_helper.cpp" CACHE STRING "source file")
SET(TARGET_NAME "cudnn_function" CACHE STRING "target name")

find_package(CUDAToolkit REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_ARCHS_SUPPORTED}")
add_library(${TARGET_NAME} SHARED ${SRC})
link_directories( ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_TARGET_DIR})
include_directories( ${CUDAToolkit_INCLUDE_DIRS})
find_library(CUDNN_LIBRARY cudnn
    HINTS ${CUDAToolkit_ROOT_DIR}
    PATH_SUFFIXES lib lib64 cuda/lib cuda/lib64 lib/x64)

include(FetchContent)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.7.3)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(CudnnBenchmark cudnn_test.cpp)   
target_link_libraries(CudnnBenchmark ${TARGET_NAME} nlohmann_json::nlohmann_json CUDA::cudart ${CUDNN_LIBRARY}) 
install(TARGETS CudnnBenchmark ${TARGET_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
