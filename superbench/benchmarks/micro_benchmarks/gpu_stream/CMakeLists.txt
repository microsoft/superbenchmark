# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)

project(gpu_stream LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

# Source files
set(SOURCES
    gpu_stream_test.cpp
    gpu_stream_utils.cpp
    gpu_stream.cu
    gpu_stream_kernels.cu
)

# Cuda environment
if(CUDAToolkit_FOUND)
    message(STATUS "Found CUDA: " ${CUDAToolkit_VERSION})

    include(../cuda_common.cmake)
    add_executable(gpu_stream ${SOURCES})
    set_property(TARGET gpu_stream PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
    target_include_directories(gpu_stream PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(gpu_stream numa nvidia-ml)
else()
    # ROCm environment
    include(../rocm_common.cmake)
    find_package(hip QUIET)
    if(hip_FOUND)
        message(STATUS "Found ROCm: " ${HIP_VERSION})
        enable_language(HIP)

        # Hipify each source and header file individually into the build directory
        set(HIPIFY_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/hipified)
        file(MAKE_DIRECTORY ${HIPIFY_OUTPUT_DIR})
        file(GLOB ALL_GPU_STREAM_SRCS
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )
        foreach(SRC_FILE ${ALL_GPU_STREAM_SRCS})
            get_filename_component(SRC_NAME ${SRC_FILE} NAME)
            # hipify-perl converts CUDA API to HIP API
            execute_process(
                COMMAND hipify-perl -o ${HIPIFY_OUTPUT_DIR}/${SRC_NAME} ${SRC_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
        endforeach()
        # Remove nvml.h include (NVIDIA-only) from hipified files
        # and replace with rocm_smi for memory clock queries
        execute_process(
            COMMAND sed -i
                "s|#include <nvml.h>|#include <rocm_smi/rocm_smi.h>|g"
                ${HIPIFY_OUTPUT_DIR}/gpu_stream_utils.hpp
        )
        execute_process(
            COMMAND sed -i
                "s|#include <nvml.h>|#include <rocm_smi/rocm_smi.h>|g"
                ${HIPIFY_OUTPUT_DIR}/gpu_stream.cu
        )
        # Replace the NVML-based GetActualMemoryClockRate with rocm_smi equivalent
        file(WRITE ${HIPIFY_OUTPUT_DIR}/rocm_mem_clock.inc [=[
    // ROCm SMI implementation
    rsmi_status_t ret;
    ret = rsmi_init(0);
    if (ret != RSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to initialize ROCm SMI" << std::endl;
        return -1.0f;
    }
    rsmi_frequencies_t freq;
    ret = rsmi_dev_gpu_clk_freq_get(gpu_id, RSMI_CLK_TYPE_MEM, &freq);
    if (ret != RSMI_STATUS_SUCCESS) {
        std::cerr << "Failed to get memory clock" << std::endl;
        rsmi_shut_down();
        return -1.0f;
    }
    float clock_mhz = static_cast<float>(freq.frequency[freq.current] / 1000000);
    rsmi_shut_down();
    return clock_mhz;
]=])
        execute_process(
            COMMAND sed -i "/nvmlReturn_t/,/return static_cast<float>(clock_mhz);/{
                /nvmlReturn_t/r ${HIPIFY_OUTPUT_DIR}/rocm_mem_clock.inc
                d
            }" ${HIPIFY_OUTPUT_DIR}/gpu_stream.cu
        )

        set(HIP_SOURCES
            ${HIPIFY_OUTPUT_DIR}/gpu_stream_test.cpp
            ${HIPIFY_OUTPUT_DIR}/gpu_stream_utils.cpp
            ${HIPIFY_OUTPUT_DIR}/gpu_stream.cu
            ${HIPIFY_OUTPUT_DIR}/gpu_stream_kernels.cu
        )
        set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
        add_executable(gpu_stream ${HIP_SOURCES})
        target_include_directories(gpu_stream PRIVATE ${HIPIFY_OUTPUT_DIR})

        include(CheckSymbolExists)
        check_symbol_exists("hipDeviceMallocUncached" "hip/hip_runtime_api.h" HIP_UNCACHED_MEMORY)
        if(${HIP_UNCACHED_MEMORY})
            target_compile_definitions(gpu_stream PRIVATE HIP_UNCACHED_MEMORY)
        endif()

        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
        target_link_libraries(gpu_stream numa hip::device rocm_smi64)
    else()
        message(FATAL_ERROR "No CUDA or ROCm environment found.")
    endif()
endif()

install(TARGETS gpu_stream RUNTIME DESTINATION bin)
