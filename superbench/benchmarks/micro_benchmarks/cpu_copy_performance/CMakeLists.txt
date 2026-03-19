# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.24)

project(cpu_copy LANGUAGES CXX)

add_executable(cpu_copy cpu_copy.cpp)
target_compile_options(cpu_copy PRIVATE -O2)
target_link_libraries(cpu_copy PRIVATE numa)

install(TARGETS cpu_copy RUNTIME DESTINATION bin)
