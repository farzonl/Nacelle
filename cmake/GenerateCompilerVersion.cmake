cmake_minimum_required(VERSION 3.24)
if(NOT MANIFEST_FILE OR NOT SOURCE_ROOT OR NOT OUTPUT_FILE)
    message(FATAL_ERROR "GenerateCompilerVersion.cmake requires MANIFEST_FILE, SOURCE_ROOT, OUTPUT_FILE")
endif()

if(NOT EXISTS "${MANIFEST_FILE}")
    message(FATAL_ERROR "Compiler hash manifest not found: ${MANIFEST_FILE}")
endif()

file(STRINGS "${MANIFEST_FILE}" MANIFEST_LINES)

set(HASH_ACCUMULATOR "")
foreach(LINE IN LISTS MANIFEST_LINES)
    string(STRIP "${LINE}" LINE_STRIPPED)
    if(LINE_STRIPPED STREQUAL "")
        continue()
    endif()

    if(LINE_STRIPPED MATCHES "^LLVM_PACKAGE_VERSION=")
        string(APPEND HASH_ACCUMULATOR "${LINE_STRIPPED}\n")
    else()
        if(NOT EXISTS "${LINE_STRIPPED}")
            message(FATAL_ERROR "Compiler hash input file does not exist: ${LINE_STRIPPED}")
        endif()
        file(SHA256 "${LINE_STRIPPED}" FILE_SHA256)
        cmake_path(IS_PREFIX SOURCE_ROOT "${LINE_STRIPPED}" NORMALIZE IS_IN_SOURCE)
        if(IS_IN_SOURCE)
            file(RELATIVE_PATH ENTRY_LABEL "${SOURCE_ROOT}" "${LINE_STRIPPED}")
        else()
            get_filename_component(ENTRY_LABEL "${LINE_STRIPPED}" NAME)
        endif()
        string(APPEND HASH_ACCUMULATOR "${ENTRY_LABEL}:${FILE_SHA256}\n")
    endif()
endforeach()

string(SHA256 COMPILER_BUILD_HASH "${HASH_ACCUMULATOR}")

get_filename_component(OUT_DIR "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OUT_DIR}")

file(WRITE "${OUTPUT_FILE}"
"#pragma once\n\n"
"inline constexpr char kCompilerBuildHash[] = \"${COMPILER_BUILD_HASH}\";\n"
)
