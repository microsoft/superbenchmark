cmake_minimum_required(VERSION 3.20)
project(nvbench_benchmarks LANGUAGES CUDA)

find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
  include(../cuda_common.cmake)
  find_package(nvbench CONFIG REQUIRED)

  # list all your CUDA benchmark source files here
  set(NVBENCH_SOURCES
    kernel_launch.cu
    sleep_kernel.cu
    # add more *.cu as needed
  )

  foreach(src ${NVBENCH_SOURCES})
    # strip ".cu" → NAME_WE
    get_filename_component(basename ${src} NAME_WE)
    set(target nvbench_${basename})

    add_executable(${target} ${src})
    target_compile_features(${target} PUBLIC cuda_std_17)
    target_link_libraries(${target}
      PRIVATE nvbench::nvbench nvbench::main
    )
    install(TARGETS ${target} RUNTIME DESTINATION bin)
  endforeach()
else()
  message(STATUS "CUDA not found, skipping nvbench benchmarks.")
endif()