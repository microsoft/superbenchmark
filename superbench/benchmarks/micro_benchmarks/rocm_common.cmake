# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

enable_language(HIP)

# Set ROCM_PATH
if(NOT DEFINED ENV{ROCM_PATH})
    # Run hipconfig -p to get ROCm path
  execute_process(
    COMMAND hipconfig -R
    RESULT_VARIABLE HIPCONFIG_RESULT
    OUTPUT_VARIABLE ROCM_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Check if hipconfig was successful
  if(NOT HIPCONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to run hipconfig -p. Make sure ROCm is installed and hipconfig is available.")
  endif()

else()
    set(ROCM_PATH $ENV{ROCM_PATH})
endif()

# Set HIP_PATH
if(NOT DEFINED ENV{HIP_PATH})
  execute_process(
    COMMAND hipconfig -p
    RESULT_VARIABLE HIPCONFIG_RESULT
    OUTPUT_VARIABLE HIP_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  # Check if hipconfig was successful
  if(NOT HIPCONFIG_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to run hipconfig -p. Make sure ROCm is installed and hipconfig is available.")
  endif()
else()
    set(HIP_PATH $ENV{HIP_PATH})
endif()

if(EXISTS ${HIP_PATH})
    # Search for hip in common locations
    list(APPEND CMAKE_PREFIX_PATH ${HIP_PATH} ${ROCM_PATH} ${ROCM_PATH}/hsa ${ROCM_PATH}/hip ${ROCM_PATH}/share/rocm/cmake/)
    list(APPEND CMAKE_MODULE_PATH
        "${HIP_PATH}/cmake"
        "${HIP_PATH}/lib/cmake/hip"
    )
endif()

function(hipify_sources OUTPUT_VAR_NAME)
    if(NOT HIPIFY_TOOL)
        find_program(HIPIFY_TOOL hipify-perl PATHS $ENV{ROCM_PATH}/bin)
        if(NOT HIPIFY_TOOL)
            message(FATAL_ERROR "hipify-perl not found! Cannot translate CUDA to HIP.")
        endif()
    endif()

    set(HIP_SOURCE_EXTS ".hip" ".cpp" ".cc" ".cxx")
    set(GENERATED_HIP_FILES "")

    foreach(SRC_FILE ${ARGN})
        get_filename_component(FILE_ABS ${SRC_FILE} ABSOLUTE)
        get_filename_component(FILE_NAME_WE ${SRC_FILE} NAME_WE)
        get_filename_component(FILE_EXT ${SRC_FILE} EXT)

        if(FILE_EXT STREQUAL ".cu")
            set(OUT_EXT ".hip")
        else()
            set(OUT_EXT ${FILE_EXT})
        endif()

        set(OUT_FILE "${CMAKE_CURRENT_BINARY_DIR}/${FILE_NAME_WE}${OUT_EXT}")

        add_custom_command(
            OUTPUT ${OUT_FILE}
            COMMAND ${HIPIFY_TOOL} -print-stats -o ${OUT_FILE} ${FILE_ABS}
            DEPENDS ${FILE_ABS}
            COMMENT "Auto-hipifying ${SRC_FILE}..."
        )
        if(OUT_EXT IN_LIST HIP_SOURCE_EXTS)
            set_source_files_properties(${OUT_FILE} PROPERTIES
                COMPILE_OPTIONS "-Wno-unused-result;-Wno-return-type"
                LANGUAGE HIP
            )
        endif()
        list(APPEND GENERATED_HIP_FILES ${OUT_FILE})
    endforeach()

    set(${OUTPUT_VAR_NAME} ${GENERATED_HIP_FILES} PARENT_SCOPE)
endfunction()
