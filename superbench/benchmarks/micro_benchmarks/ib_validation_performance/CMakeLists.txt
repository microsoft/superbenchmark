# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(ib_mpi)

set(CMAKE_CXX_STANDARD 14)

# -fopenmp flag (enables OpenMP)
set(GCC_COVERAGE_COMPILE_FLAGS "-Wall -pedantic -lm -O3 -funroll-loops -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}")

add_executable(ib_mpi ib_validation_performance.cc)

# MPI
find_package(MPI REQUIRED)

# Boost
execute_process(COMMAND apt-get install libboost-all-dev -y WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/)
find_package(Boost REQUIRED COMPONENTS system thread program_options)
include_directories(${Boost_INCLUDE_DIRS})
target_link_libraries(ib_mpi PUBLIC MPI::MPI_CXX ${Boost_LIBRARIES})
