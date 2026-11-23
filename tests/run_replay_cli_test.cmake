cmake_minimum_required(VERSION 3.24)
foreach(input PROGRAM CLI REPLAY)
    if(NOT DEFINED ${input})
        message(FATAL_ERROR "Replay CLI test requires ${input}")
    endif()
endforeach()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    NACELLE_REPLAY_KEEP_OUTPUT=1 NACELLE_SIMD_WIDTH=4 NACELLE_DISABLE_TIER2=0
    NACELLE_DISABLE_GEMM_FASTPATH=1 "NACELLE_CACHE_DIR=${REPLAY}-cache"
    "${PROGRAM}" "${REPLAY}" table RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Replay fixture failed: ${output}\n${error}")
endif()
execute_process(COMMAND "${CLI}" "${REPLAY}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "no captured code executed")
    message(FATAL_ERROR "CLI validation must not execute: ${output}\n${error}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E env
    NACELLE_SIMD_WIDTH=4 NACELLE_DISABLE_TIER2=0 NACELLE_DISABLE_GEMM_FASTPATH=1
    "NACELLE_CACHE_DIR=${REPLAY}-execution-cache"
    "${CLI}" "${REPLAY}" --execute-trusted-local --repetitions 3
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error)
if(NOT result EQUAL 0 OR NOT output MATCHES "matched across 3 fresh-state runs")
    message(FATAL_ERROR "CLI repeated output validation failed: ${output}\n${error}")
endif()
file(REMOVE "${REPLAY}")
