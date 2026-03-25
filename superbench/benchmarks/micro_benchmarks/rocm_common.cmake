# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Set ROCM_PATH
if(NOT DEFINED ENV{ROCM_PATH})
    # Run hipconfig -p to get ROCm path
  execute_process(
    COMMAND hipconfig -R
    RESULT_VARIABLE HIPCONFIG_RESULT
    OUTPUT_VARIABLE ROCM_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Check if hipconfig was successful
  if(NOT HIPCONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to run hipconfig -p. Make sure ROCm is installed and hipconfig is available.")
  endif()

else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

# Set HIP_PATH
if(NOT DEFINED ENV{HIP_PATH})
  execute_process(
    COMMAND hipconfig -p
    RESULT_VARIABLE HIPCONFIG_RESULT
    OUTPUT_VARIABLE HIP_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Check if hipconfig was successful
  if(NOT HIPCONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to run hipconfig -p. Make sure ROCm is installed and hipconfig is available.")
  endif()
else()
    set(HIP_PATH $ENV{HIP_PATH})
endif()

# Set HIP architectures from AMDGPU_TARGETS environment variable if available.
# AMDGPU_TARGETS should be a space-separated list of GPU architectures,
# e.g. "gfx908 gfx90a gfx942".
# Both the CMake variable AMDGPU_TARGETS and CMAKE_HIP_ARCHITECTURES must be set:
# - AMDGPU_TARGETS is read by ROCm's hip-config-amd.cmake to set --offload-arch flags
# - CMAKE_HIP_ARCHITECTURES is the native CMake variable for HIP (requires >= 3.21)
if(DEFINED ENV{AMDGPU_TARGETS})
    string(REPLACE " " ";" HIP_ARCH_LIST "$ENV{AMDGPU_TARGETS}")
    set(AMDGPU_TARGETS ${HIP_ARCH_LIST} CACHE STRING "AMD GPU targets to compile for")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
        set(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH_LIST})
    endif()
    message(STATUS "Using AMDGPU_TARGETS from environment: ${HIP_ARCH_LIST}")
else()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
        set(CMAKE_HIP_ARCHITECTURES OFF)
    endif()
    message(STATUS "AMDGPU_TARGETS not set, relying on hipcc auto-detection")
endif()

if(EXISTS ${HIP_PATH})
    # Search for hip in common locations
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH} ${ROCM_PATH}/hsa ${ROCM_PATH}/hip ${ROCM_PATH}/share/rocm/cmake/)
    set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()
