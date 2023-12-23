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

# Turn off CMAKE_HIP_ARCHITECTURES Feature if cmake version is 3.21+
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.21.0)
    set(CMAKE_HIP_ARCHITECTURES OFF)
endif()
message(STATUS "CMAKE HIP ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")

if(EXISTS ${HIP_PATH})
    # Search for hip in common locations
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH} ${ROCM_PATH}/hsa ${ROCM_PATH}/hip ${ROCM_PATH}/share/rocm/cmake/)
    set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()
