# Standalone CMake for building hipblaslt-bench against system-installed
# hipBLASLt, bypassing the upstream build system.
#
# Used by dockerfile/rocm7.2.x.dockerfile because the upstream 7.2 source
# tree pulls in AMD-internal "origami" headers and a new tensilelite-host
# C++ library that conflict with the goal of building only the bench tool.
#
# Place this file at the root of an upstream hipBLASLt source tree as the
# top-level CMakeLists.txt and configure it as a normal CMake project, e.g.:
#
#   cp /path/to/this/file /path/to/hipBLASLt/CMakeLists.txt
#   cmake -S /path/to/hipBLASLt -B /path/to/hipBLASLt/build
#   cmake --build /path/to/hipBLASLt/build --target hipblaslt-bench

cmake_minimum_required(VERSION 3.21)
project(hipblaslt-bench-standalone LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The directory containing this build script is the hipBLASLt source root.
set(HIPBLASLT_SRC "${CMAKE_CURRENT_SOURCE_DIR}")

# --- Dependencies (all from system / preinstalled) ---
find_package(hip REQUIRED)
find_package(hipblaslt CONFIG REQUIRED)
find_package(LAPACK REQUIRED)  # also brings BLAS via implicit find_package(BLAS)
find_package(OpenMP REQUIRED)
find_package(rocm_smi)         # optional

# Locate cblas explicitly (not part of LAPACK's standard targets).
# cblas_interface.cpp uses cblas_sgemm/dgemm so we need the C BLAS library.
find_library(CBLAS_LIBRARY NAMES cblas PATHS /usr/local/lib /usr/lib REQUIRED)
message(STATUS "Found CBLAS: ${CBLAS_LIBRARY}")

# --- The bench static helper library ---
add_library(hipblaslt-clients-common STATIC
    "${HIPBLASLT_SRC}/clients/common/src/singletons.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/utility.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/efficiency_monitor.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/cblas_interface.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/argument_model.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_parse_data.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_arguments.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_random.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_init_device.cpp"
)

# These .cpp files are HIP code (use __device__/__host__, hip_runtime APIs,
# half/bfloat16 types). Compiling them as plain CXX with gcc fails. Force HIP.
set_source_files_properties(
    "${HIPBLASLT_SRC}/clients/common/src/utility.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/cblas_interface.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_init_device.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_arguments.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_random.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/argument_model.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/hipblaslt_parse_data.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/efficiency_monitor.cpp"
    "${HIPBLASLT_SRC}/clients/common/src/singletons.cpp"
    PROPERTIES LANGUAGE HIP
)

target_include_directories(hipblaslt-clients-common
    PUBLIC
        "${HIPBLASLT_SRC}/clients/common/include"
        "${HIPBLASLT_SRC}/clients/bench/include"
        # Source's library/include comes BEFORE system include so we get
        # hipblaslt_xfloat32.h (not shipped in the system install).
        "${HIPBLASLT_SRC}/library/include"
        # Internal headers used by clients (rocblaslt/rocblaslt-types.h etc.)
        "${HIPBLASLT_SRC}/library/src/amd_detail/include"
        "${HIPBLASLT_SRC}/library/src/amd_detail/rocblaslt/include"
        "${HIPBLASLT_SRC}/library/src/amd_detail/rocblaslt/src/include"
        # tensilelite headers used by clients (e.g. client/include/Utility.hpp).
        "${HIPBLASLT_SRC}/tensilelite"
)

target_compile_definitions(hipblaslt-clients-common
    PUBLIC
        # Critical: in 7.2 the upstream build sets ROCM_USE_FLOAT16 only
        # via the in-tree hipblaslt target's INTERFACE_COMPILE_DEFINITIONS.
        # The system find_package(hipblaslt) does not propagate it. Without
        # this, hipblasLtHalf is the struct version with no operator float,
        # which breaks hipblaslt_ostream.hpp.
        ROCM_USE_FLOAT16
        __HIP_PLATFORM_AMD__
        HIPBLASLT_BENCH
        HIPBLASLT_INTERNAL_API
)

target_link_libraries(hipblaslt-clients-common
    PUBLIC
        hip::host
        hip::device
        # Order matters: cblas -> lapack -> blas -> gfortran (lapack needs blas
        # which needs Fortran runtime).
        ${CBLAS_LIBRARY}
        ${LAPACK_LIBRARIES}
        ${BLAS_LIBRARIES}
        gfortran
        OpenMP::OpenMP_CXX
)

if(rocm_smi_FOUND)
    target_link_libraries(hipblaslt-clients-common PRIVATE rocm_smi64)
endif()

# Link against the system hipblaslt .so directly via library name to avoid
# inheriting INTERFACE_COMPILE_DEFINITIONS (HIPBLASLT_USE_ROCROLLER) from
# the imported roc::hipblaslt target. We only need linkage, not propagated
# defines.
target_link_directories(hipblaslt-clients-common PUBLIC /opt/rocm/lib)
target_link_libraries(hipblaslt-clients-common PUBLIC hipblaslt)

# --- The bench executable ---
add_executable(hipblaslt-bench
    "${HIPBLASLT_SRC}/clients/bench/src/client.cpp"
)
set_source_files_properties(
    "${HIPBLASLT_SRC}/clients/bench/src/client.cpp"
    PROPERTIES LANGUAGE HIP
)
target_link_libraries(hipblaslt-bench PRIVATE hipblaslt-clients-common)
