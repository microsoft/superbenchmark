cmake_minimum_required(VERSION 3.20)
project(nvbench_benchmarks LANGUAGES CUDA)

find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
  include(../cuda_common.cmake)

  find_package(nvbench CONFIG REQUIRED)
  add_executable(nvbench_kernel_launch launch_bench.cu)
  target_compile_features(nvbench_kernel_launch PUBLIC cuda_std_17)
  target_link_libraries(nvbench_kernel_launch
    PRIVATE nvbench::nvbench nvbench::main
  )
  install(TARGETS nvbench_kernel_launch RUNTIME DESTINATION bin)
endif()