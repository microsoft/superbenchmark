# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(cudnn_benchmark LANGUAGES CXX)

include(CTest)

set(CUDNN_POLICY_TEST_SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../tests/benchmarks/micro_benchmarks/test_cudnn_plan_policy.cpp)
if(BUILD_TESTING AND EXISTS "${CUDNN_POLICY_TEST_SOURCE}")
  add_executable(cudnn_policy_test ${CUDNN_POLICY_TEST_SOURCE})
  target_include_directories(cudnn_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_features(cudnn_policy_test PRIVATE cxx_std_11)
  add_test(NAME cudnn_policy COMMAND cudnn_policy_test)
  set_tests_properties(cudnn_policy PROPERTIES LABELS "cpu;cudnn")
endif()

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  include(../cuda_common.cmake)
  set(SRC "cudnn_helper.cpp" CACHE STRING "source file")
  set(TARGET_NAME "cudnn_function" CACHE STRING "target name")

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

  add_executable(cudnn_benchmark cudnn_test.cpp)
  target_link_libraries(cudnn_benchmark ${TARGET_NAME} nlohmann_json::nlohmann_json CUDA::cudart CUDA::cublas ${CUDNN_LIBRARY})

  install(TARGETS cudnn_benchmark ${TARGET_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
endif()
