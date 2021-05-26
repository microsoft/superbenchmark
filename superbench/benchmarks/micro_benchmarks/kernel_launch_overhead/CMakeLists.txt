# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(kernel_launch_overhead LANGUAGES CUDA CXX)

include(../cuda_common.cmake)

add_executable(kernel_launch_overhead cuda_kernel_launch.cu)
set_property(TARGET kernel_launch_overhead PROPERTY CUDA_ARCHITECTURES ${NVCC_ARCHS_SUPPORTED})
install(TARGETS kernel_launch_overhead RUNTIME DESTINATION bin)
