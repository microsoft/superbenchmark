cmake_minimum_required(VERSION 3.18)
project(nvbench_benchmarks LANGUAGES CUDA)

# Check if we have a recent enough CMake for nvbench (which requires 3.30.4)
if(CMAKE_VERSION VERSION_LESS "3.30.4")
  message(STATUS "CMake version ${CMAKE_VERSION} is less than 3.30.4 (required by nvbench), skipping nvbench benchmarks")
  return()
endif()

find_package(CUDAToolkit QUIET)
if (CUDAToolkit_FOUND)
  include(../cuda_common.cmake)
  
  # Try to find nvbench, but don't require it
  find_package(nvbench CONFIG QUIET)
  
  if (nvbench_FOUND)
    message(STATUS "Found nvbench, building nvbench benchmarks")
    
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
    message(STATUS "nvbench not found, skipping nvbench benchmarks.")
    message(STATUS "To build nvbench benchmarks, first build the submodule in third_party/nvbench")
  endif()
else()
  message(STATUS "CUDA not found, skipping nvbench benchmarks.")
endif()