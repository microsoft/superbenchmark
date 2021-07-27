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
else()
set(HIP_PATH $ENV{HIP_PATH})
endif()

if(EXISTS ${HIP_PATH})
    # Search for hip in common locations
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH})
    set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})
    set(CMAKE_MODULE_PATH "${HIP_PATH}/lib/cmake/hip" ${CMAKE_MODULE_PATH})
endif()