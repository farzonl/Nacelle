cmake_minimum_required(VERSION 3.24)
if(NOT INPUT_FILE OR NOT OUTPUT_FILE OR NOT SYMBOL_NAME)
    message(FATAL_ERROR "EmbedBinary.cmake requires INPUT_FILE, OUTPUT_FILE, SYMBOL_NAME")
endif()

file(READ "${INPUT_FILE}" HEX_DATA HEX)
string(LENGTH "${HEX_DATA}" HEX_LEN)
math(EXPR BYTE_LEN "${HEX_LEN} / 2")

string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " C_BYTES "${HEX_DATA}")

get_filename_component(OUT_DIR "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${OUT_DIR}")

file(WRITE "${OUTPUT_FILE}"
"#pragma once\n"
"#include <cstddef>\n"
"#include <cstdint>\n\n"
"inline constexpr uint8_t ${SYMBOL_NAME}[] = {\n"
"    ${C_BYTES}\n"
"};\n"
"inline constexpr size_t ${SYMBOL_NAME}_size = ${BYTE_LEN};\n"
)
