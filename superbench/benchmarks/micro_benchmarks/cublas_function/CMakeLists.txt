# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(CublasBenchmark LANGUAGES CUDA CXX)

include(../cuda_common.cmake)

SET(SRC "cublas_helper.cpp" CACHE STRING "source file")
SET(TARGET_NAME "cublas_function" CACHE STRING "target name")

find_package(CUDAToolkit REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${NVCC_ARCHS_SUPPORTED}")
add_library(${TARGET_NAME} SHARED ${SRC})
link_directories( ${CUDAToolkit_LIBRARY_DIR} ${CUDAToolkit_TARGET_DIR})
include_directories( ${CUDAToolkit_INCLUDE_DIRS})

include(FetchContent)
FetchContent_Declare(json
  GIT_REPOSITORY https://github.com/ArthurSonzogni/nlohmann_json_cmake_fetchcontent
  GIT_TAG v3.7.3)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

add_executable(CublasBenchmark cublas_test.cpp)   
target_link_libraries(CublasBenchmark ${TARGET_NAME} nlohmann_json::nlohmann_json CUDA::cudart CUDA::cublas) 

install(TARGETS CublasBenchmark ${TARGET_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib)