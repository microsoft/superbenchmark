cmake_minimum_required(VERSION 3.0)
project(ROCBLAS_GEMM_TEST)

# Set the C++ standard (e.g., C++11)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_PREFIX_PATH /opt/rocm)

add_compile_definitions(__gfx940__)

# Find ROCBLAS package
find_package(rocblas REQUIRED)

# Add the test executable
add_executable(rocblas_gemm_fp8 rocblas_gemm.cc)

# Link the ROCBLAS library to the test executable
target_link_libraries(rocblas_gemm_fp8 roc::rocblas)
