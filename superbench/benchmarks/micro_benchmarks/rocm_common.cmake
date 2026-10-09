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

# Set HIP architectures from the AMDGPU_TARGETS environment variable if available.
# Accepts the common separators a user might pass: whitespace, ',', or ';'
# (e.g. "gfx908 gfx90a gfx942", "gfx908,gfx90a", or "gfx908;gfx90a").
# In this repository's micro-benchmarks, AMDGPU_TARGETS is what actually drives
# --offload-arch selection, via ROCm's hip-config-amd.cmake and hipcc (the C++
# compiler). CMAKE_HIP_ARCHITECTURES is set (when supported by CMake >= 3.21)
# for compatibility with projects that enable the CMake HIP language, but is
# not required for these CXX-only projects.
set(_amdgpu_targets_raw "$ENV{AMDGPU_TARGETS}")
string(STRIP "${_amdgpu_targets_raw}" _amdgpu_targets_stripped)
# Collapse runs of any common separator (spaces, tabs, CR/LF, ',', ';') into a
# single ';' so the result is a well-formed CMake list with no empty elements.
string(REGEX REPLACE "[ \t\r\n,;]+" ";" HIP_ARCH_LIST "${_amdgpu_targets_stripped}")
if(NOT HIP_ARCH_LIST STREQUAL "")
    # Use a normal (non-cache) directory-scoped variable so we do not pollute
    # the global CMake cache or override AMDGPU_TARGETS in nested projects.
    # The env var is re-read on every reconfigure, so this still wins over
    # stale state when the user changes AMDGPU_TARGETS.
    set(AMDGPU_TARGETS ${HIP_ARCH_LIST})
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
        set(CMAKE_HIP_ARCHITECTURES ${HIP_ARCH_LIST})
    endif()
    message(STATUS "Using AMDGPU_TARGETS from environment: ${HIP_ARCH_LIST}")
else()
    if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
        set(CMAKE_HIP_ARCHITECTURES OFF)
    endif()
    message(STATUS "AMDGPU_TARGETS not set (or empty), relying on hipcc auto-detection")
endif()

if(EXISTS ${HIP_PATH})
    # Search for hip in common locations
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH} ${ROCM_PATH}/hsa ${ROCM_PATH}/hip ${ROCM_PATH}/share/rocm/cmake/)
    set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()
