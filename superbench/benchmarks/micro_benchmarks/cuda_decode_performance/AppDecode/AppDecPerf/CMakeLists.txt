# Copyright 2020-2022 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.

# 3.7 is required for FindVulkan module support in CMake.
cmake_minimum_required(VERSION 3.7)

project(AppDecPerf)

set(APP_SOURCES
 ${CMAKE_CURRENT_SOURCE_DIR}/AppDecPerf.cpp
)

if(WIN32)
    set(NV_FFMPEG_HDRS ${FFMPEG_DIR}/include)
endif()

set(NV_DEC_SOURCES
 ${NV_DEC_DIR}/NvDecoder.cpp
)



set(NV_DEC_HDRS
 ${NV_DEC_DIR}/NvDecoder.h
 ${NVCODEC_PUBLIC_INTERFACE_DIR}/cuviddec.h
 ${NVCODEC_PUBLIC_INTERFACE_DIR}/nvcuvid.h
 ${NVCODEC_UTILS_DIR}/NvCodecUtils.h
 ${NVCODEC_UTILS_DIR}/FFmpegDemuxer.h
 ${CMAKE_CURRENT_SOURCE_DIR}/ThreadPoolUtils.h
)

source_group( "headers" FILES ${NV_DEC_HDRS} )
source_group( "sources" FILES ${APP_SOURCES} ${NV_DEC_SOURCES})

find_package(CUDA)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=\"sm_50,compute_50\")
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

cuda_add_executable(${PROJECT_NAME}  ${APP_SOURCES} ${NV_DEC_SOURCES} ${NV_DEC_HDRS})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS}
 ${NVCODEC_PUBLIC_INTERFACE_DIR}
 ${NVCODEC_UTILS_DIR}
 ${NV_CODEC_DIR}
 ${NV_APPDEC_COMMON_DIR}
 ${NV_FFMPEG_HDRS}
)

if(WIN32)
    Set(AVCODEC_LIB ${FFMPEG_DIR}/lib/avcodec.lib)
    Set(AVFORMAT_LIB ${FFMPEG_DIR}/lib/avformat.lib)
    Set(AVUTIL_LIB ${FFMPEG_DIR}/lib/avutil.lib)
    Set(SWRESAMPLE_LIB ${FFMPEG_DIR}/lib/swresample.lib)
    Set(FFMPEG_DLL_DIR ${FFMPEG_DIR}/bin/)
    Set(FFMPEG_LIB_DIR ${FFMPEG_DIR}/lib/)
    #copy dll's
    install(DIRECTORY ${FFMPEG_DLL_DIR} DESTINATION ${NVCODEC_SAMPLES_INSTALL_DIR} FILES_MATCHING PATTERN "*.dll")
    install(DIRECTORY ${FFMPEG_LIB_DIR} DESTINATION ${NVCODEC_SAMPLES_INSTALL_DIR} FILES_MATCHING PATTERN "*.lib")
endif()


target_link_libraries(${PROJECT_NAME} ${CUDA_CUDA_LIBRARY} ${CMAKE_DL_LIBS} ${CUVID_LIB} ${AVCODEC_LIB}
 ${AVFORMAT_LIB} ${AVUTIL_LIB} ${SWRESAMPLE_LIB})

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION ${NVCODEC_SAMPLES_INSTALL_DIR})
if (MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
    set_target_properties( ${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${NVCODEC_SAMPLES_INSTALL_DIR}/$<CONFIG>/ )
    # copy ffmpeg .dll and .lib files
    file(GLOB externalLibList 
        ${FFMPEG_LIB_DIR}/*.lib 
        ${FFMPEG_DLL_DIR}/*.dll
    )    
    foreach(externalLib ${externalLibList})
        add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD  COMMAND ${CMAKE_COMMAND} -E copy ${externalLib} ${NVCODEC_SAMPLES_INSTALL_DIR}/$<CONFIG>/)
    endforeach()
    #copy pdb's
    install(FILES $<TARGET_PDB_FILE:${PROJECT_NAME}> DESTINATION ${NVCODEC_SAMPLES_INSTALL_DIR}  CONFIGURATIONS Debug)
endif()

