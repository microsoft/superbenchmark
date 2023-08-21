# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

cmake_minimum_required(VERSION 3.18)
project(cuda_decode_performance)

find_package(CUDA QUIET)
if(CUDA_FOUND)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)

  set(THIRD_PARTY_SAMPLE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/Video_Codec_SDK/Samples)
  set(NVCODEC_PUBLIC_INTERFACE_DIR ${THIRD_PARTY_SAMPLE_DIR}/../Interface)
  set(NVCODEC_UTILS_DIR ${THIRD_PARTY_SAMPLE_DIR}/Utils)
  set(NV_CODEC_DIR ${THIRD_PARTY_SAMPLE_DIR}/NvCodec)
  set(NV_DEC_DIR ${THIRD_PARTY_SAMPLE_DIR}/NvCodec/NvDecoder)

  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PC_AVCODEC REQUIRED IMPORTED_TARGET libavcodec)
    pkg_check_modules(PC_AVFORMAT REQUIRED IMPORTED_TARGET libavformat)
    pkg_check_modules(PC_AVUTIL REQUIRED IMPORTED_TARGET libavutil)
    pkg_check_modules(PC_SWRESAMPLE REQUIRED IMPORTED_TARGET libswresample)

    set(NV_FFMPEG_HDRS ${PC_AVCODEC_INCLUDE_DIRS})
    find_library(AVCODEC_LIBRARY NAMES avcodec
    HINTS
    ${PC_AVCODEC_LIBDIR}
    ${PC_AVCODEC_LIBRARY_DIRS}
    )
    find_library(AVFORMAT_LIBRARY NAMES avformat
    HINTS
    ${PC_AVFORMAT_LIBDIR}
    ${PC_AVFORMAT_LIBRARY_DIRS}
    )
    find_library(AVUTIL_LIBRARY NAMES avutil
    HINTS
    ${PC_AVUTIL_LIBDIR}
    ${PC_AVUTIL_LIBRARY_DIRS}
    )
    find_library(SWRESAMPLE_LIBRARY NAMES swresample
    HINTS
    ${PC_SWRESAMPLE_LIBDIR}
    ${PC_SWRESAMPLE_LIBRARY_DIRS}
    )
    set(AVCODEC_LIB ${AVCODEC_LIBRARY})
    set(AVFORMAT_LIB ${AVFORMAT_LIBRARY})
    set(AVUTIL_LIB ${AVUTIL_LIBRARY})
    set(SWRESAMPLE_LIB ${SWRESAMPLE_LIBRARY})
  endif()

  set(APP_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/AppDecPerf.cpp
  )

  set(NV_DEC_SOURCES
  ${NV_DEC_DIR}/NvDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/OptimizedNvDecoder.cpp
  )

  set(NV_DEC_HDRS
  ${NV_DEC_DIR}/NvDecoder.h
  ${NVCODEC_PUBLIC_INTERFACE_DIR}/cuviddec.h
  ${NVCODEC_PUBLIC_INTERFACE_DIR}/nvcuvid.h
  ${NVCODEC_UTILS_DIR}/NvCodecUtils.h
  ${NVCODEC_UTILS_DIR}/FFmpegDemuxer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ThreadPoolUtils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/OptimizedNvDecoder.h
  )

  source_group( "headers" FILES ${NV_DEC_HDRS} )
  source_group( "sources" FILES ${APP_SOURCES} ${NV_DEC_SOURCES})
  set(CMAKE_LIBRARY_PATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64/stubs;${CUDA_TOOLKIT_ROOT_DIR}/lib/stubs;${CUDA_TOOLKIT_ROOT_DIR}/lib64;${CUDA_TOOLKIT_ROOT_DIR}/lib;${CMAKE_LIBRARY_PATH}")
  find_package(CUDA)
  set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=\"sm_50,compute_50\")
  if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
      list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
  endif()

  # Check if the file exists
  if (NOT EXISTS "/usr/local/lib/libnvcuvid.so" )
      execute_process(
        COMMAND sudo ln -s /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/local/lib/libnvcuvid.so
        RESULT_VARIABLE result
      )  
      if(result)
        message(FATAL_ERROR "Failed to create symbolic link for nvcuvid lib: ${result}")
      endif()
  endif ()

  find_library(CUVID_LIB nvcuvid
  HINTS
  "/usr/local/lib/"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../../third_party/Video_Codec_SDK/Lib/linux/stubs/x86_64/"
  )

  cuda_add_executable(${PROJECT_NAME} ${APP_SOURCES} ${NV_DEC_SOURCES} ${NV_DEC_HDRS})

  set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS}
  ${NVCODEC_PUBLIC_INTERFACE_DIR}
  ${NVCODEC_UTILS_DIR}
  ${NV_CODEC_DIR}
  ${NV_APPDEC_COMMON_DIR}
  ${NV_FFMPEG_HDRS}
  ${THIRD_PARTY_SAMPLE_DIR}
  )

  target_link_libraries(${PROJECT_NAME} ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${CUVID_LIB} ${AVCODEC_LIB}
  ${AVFORMAT_LIB} ${AVUTIL_LIB} ${SWRESAMPLE_LIB})

  install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin LIBRARY DESTINATION lib)
endif()
