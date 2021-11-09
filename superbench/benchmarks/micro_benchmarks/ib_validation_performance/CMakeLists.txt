# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(ib_validation)

set(CMAKE_CXX_STANDARD 14)

# -fopenmp flag (enables OpenMP)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -pedantic -lm -O3 -funroll-loops -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(ib_validation ib_validation_performance.cc)

# MPI
find_package(MPI REQUIRED)

# Boost
execute_process(COMMAND wget -O boost_1_71_0.tar.gz https://boostorg.jfrog.io/artifactory/main/release/1.71.0/source/boost_1_71_0.tar.gz WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
execute_process(COMMAND tar xzvf boost_1_71_0.tar.gz WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
execute_process(COMMAND ./bootstrap.sh WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/boost_1_71_0/)
execute_process(COMMAND ./b2 --with-program_options install WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/boost_1_71_0/)
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/boost_1_71_0/build/)

find_package(Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(ib_validation PUBLIC MPI::MPI_CXX ${Boost_LIBRARIES})
install(TARGETS ib_validation RUNTIME DESTINATION bin)
