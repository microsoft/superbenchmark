# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Set ROCM_PATH
if(NOT DEFINED ENV{ROCM_PATH})
    set(ROCM_PATH /opt/rocm)
else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

# Set HIP_PATH
if(NOT DEFINED ENV{HIP_PATH})
    set(HIP_PATH ${ROCM_PATH}/hip)
        if(NOT EXISTS ${HIP_PATH})
            execute_process(COMMAND hipconfig --path OUTPUT_VARIABLE HIP_PATH)
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
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH})
    set(CMAKE_PREFIX_PATH /opt/rocm ROCM_PATH)
    set(CMAKE_CXX_COMPILER "${HIP_PATH}/bin/hipcc")
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()
