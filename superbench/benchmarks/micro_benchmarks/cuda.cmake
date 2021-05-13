# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

set(NVCC_ARCHS_SUPPORTED "")
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 7.5)
  list(APPEND NVCC_ARCHS_SUPPORTED 53)
endif()
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 8.0)
  list(APPEND NVCC_ARCHS_SUPPORTED 60 61)
endif()
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.0)
  list(APPEND NVCC_ARCHS_SUPPORTED 70)
endif()
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 9.2)
  list(APPEND NVCC_ARCHS_SUPPORTED 72)
endif()
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 10.0)
  list(APPEND NVCC_ARCHS_SUPPORTED 75)
endif()
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
  list(APPEND NVCC_ARCHS_SUPPORTED 80)
endif()
if (NOT CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.1)
  list(APPEND NVCC_ARCHS_SUPPORTED 86)
endif()
