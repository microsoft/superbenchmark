# Copyright 2020-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.po

# 3.7 is required for FindVulkan module support in CMake.
cmake_minimum_required(VERSION 3.7)

project(NvCodec)

# Set C++11 for all projects and disable non-standard extensions
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_INSTALL_PREFIX .)

# Check for WSL
if (EXISTS /usr/lib/wsl/lib)
    set(WSL_FLAG TRUE)
endif()

set(NVCODEC_PUBLIC_INTERFACE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../Interface)
set(NVCODEC_UTILS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/Utils)
set(NV_CODEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec)
set(NV_DEC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/NvCodec/NvDecoder)
set(NV_APPDEC_COMMON_DIR  ${CMAKE_CURRENT_SOURCE_DIR}/AppDecode/Common)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(NVCODEC_SAMPLES_INSTALL_DIR ${CMAKE_BINARY_DIR})
else()
    set(NVCODEC_SAMPLES_INSTALL_DIR ${CMAKE_BINARY_DIR})
endif()

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        Set(CUVID_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Lib/x64/nvcuvid.lib)
    else()
        Set(CUVID_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../Lib/Win32/nvcuvid.lib)
    endif()
else()
    find_library(CUVID_LIB nvcuvid)
endif()

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

if(WIN32)
    # check if FFMpeg, GLUT and GLEW library cmake variables are set and
    # add projects having above dependencies accordingly for Windows
    if(DEFINED FFMPEG_DIR)
        add_subdirectory(AppDecode/AppDecPerf)
    elseif((NOT DEFINED FFMPEG_DIR) AND (NOT DEFINED SKIP_FFMPEG_DEPENDENCY))
        message("FFMPEG Libraries are needed to build the following apps :")
        message("AppDec, AppDec3D, AppDecGL, AppDecImageProvider, AppDecLowLatency, AppDecMem,")
        message("AppDecMultiFiles, AppDecMultiInput, AppDecPerf, AppEncDec, AppTranscode")
        message("It is required to fetch the prebuilt LGPL FFMPEG libraries and to set the variable FFMPEG_DIR to point to this libraries")
        message("To skip building of apps having FFMPEG dependency, add -DSKIP_FFMPEG_DEPENDENCY=TRUE to commandline")
                message("See Windows Configuration Requirements section in Read_Me.pdf for detailed instructions")
        message(FATAL_ERROR "FFMPEG_DIR not set")
    endif()

else()
    add_subdirectory(AppDecode/AppDecPerf)
endif()

