# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Source files (CUDA-style names; hipify-perl translates them on ROCm path)
set(GPU_STREAM_SOURCES
    gpu_stream_test.cpp
    gpu_stream_utils.cpp
    gpu_stream.cu
    gpu_stream_kernels.cu
)
set(GPU_STREAM_HEADERS
    gpu_stream.hpp
    gpu_stream_utils.hpp
    gpu_stream_kernels.hpp
)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    # NVIDIA / CUDA path
    find_library(NVML_LIBRARY nvidia-ml PATHS ${CUDAToolkit_LIBRARY_DIR} PATH_SUFFIXES stubs)
    if(NOT NVML_LIBRARY)
        message(WARNING "gpu_stream: NVML (nvidia-ml) not found, skipping build")
        return()
    endif()

    message(STATUS "Found CUDA: ${CUDAToolkit_VERSION}")
    include(../cuda_common.cmake)
    add_executable(gpu_stream ${GPU_STREAM_SOURCES})
    set_property(TARGET gpu_stream PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_include_directories(gpu_stream PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gpu_stream numa ${NVML_LIBRARY})
    install(TARGETS gpu_stream RUNTIME DESTINATION bin)
    return()
endif()

# ROCm / HIP path
include(../rocm_common.cmake)
find_package(hip QUIET)

if(NOT hip_FOUND)
    message(WARNING "gpu_stream: neither CUDA nor HIP found, skipping build")
    return()
endif()

message(STATUS "Found ROCm: ${HIP_VERSION}")
enable_language(HIP)

# rocm_smi is required for actual memory clock queries (NVML equivalent).
find_path(ROCM_SMI_INCLUDE_DIR rocm_smi/rocm_smi.h
    HINTS ${ROCM_PATH} /opt/rocm
    PATH_SUFFIXES include
)
find_library(ROCM_SMI_LIBRARY
    NAMES rocm_smi64
    HINTS ${ROCM_PATH} /opt/rocm
    PATH_SUFFIXES lib lib64
)
if(NOT ROCM_SMI_INCLUDE_DIR OR NOT ROCM_SMI_LIBRARY)
    message(WARNING "gpu_stream: rocm_smi not found, skipping build")
    return()
endif()
message(STATUS "Found rocm_smi: ${ROCM_SMI_LIBRARY}")

# Translate the CUDA-named sources/headers to HIP via hipify-perl into the
# build directory. hipify-perl translates cuda* identifiers and CUDA includes
# but leaves NVML/rocm_smi symbols untouched (they are gated by #ifdef in the
# source).
find_program(HIPIFY_PERL hipify-perl REQUIRED)

set(HIPIFY_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipified)
file(MAKE_DIRECTORY ${HIPIFY_OUTPUT_DIR})

set(HIP_SOURCES "")
set(HIPIFY_OUTPUTS "")
foreach(SRC IN LISTS GPU_STREAM_SOURCES GPU_STREAM_HEADERS)
    set(SRC_IN ${CMAKE_CURRENT_SOURCE_DIR}/${SRC})
    set(SRC_OUT ${HIPIFY_OUTPUT_DIR}/${SRC})
    add_custom_command(
        OUTPUT ${SRC_OUT}
        COMMAND ${HIPIFY_PERL} -o ${SRC_OUT} ${SRC_IN}
        DEPENDS ${SRC_IN}
        COMMENT "hipify-perl ${SRC}"
        VERBATIM
    )
    list(APPEND HIPIFY_OUTPUTS ${SRC_OUT})
    if(SRC MATCHES "\\.(cpp|cu)$")
        list(APPEND HIP_SOURCES ${SRC_OUT})
    endif()
endforeach()

# Aggregate target so all hipified outputs (including headers) are produced
# before any object is compiled.
add_custom_target(gpu_stream_hipify DEPENDS ${HIPIFY_OUTPUTS})

set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)

include(CheckSymbolExists)
check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)
if(HIP_UNCACHED_MEMORY)
    add_compile_definitions(HIP_UNCACHED_MEMORY)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(gpu_stream ${HIP_SOURCES})
add_dependencies(gpu_stream gpu_stream_hipify)
target_include_directories(gpu_stream PRIVATE
    ${HIPIFY_OUTPUT_DIR}
    ${ROCM_SMI_INCLUDE_DIR}
)
target_link_libraries(gpu_stream numa hip::device ${ROCM_SMI_LIBRARY})

install(TARGETS gpu_stream RUNTIME DESTINATION bin)
