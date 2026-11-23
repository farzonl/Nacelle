foreach(input PROGRAM LLDB MODE WIDTH LINKAGE)
    if(NOT DEFINED ${input})
        message(FATAL_ERROR "Optimized source debugger test requires ${input}")
    endif()
endforeach()
set(ENV{NACELLE_SIMD_WIDTH} "${WIDTH}")
set(ENV{NACELLE_DISABLE_TIER2} "0")
set(ENV{NACELLE_TEST_SOURCE_OPTIMIZED_MODE} "1")
set(variant "${INPUT_O0}")
if(MIXED)
    set(variant mixed)
elseif(ASYNC)
    set(variant async)
endif()
set(ENV{NACELLE_CACHE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/optimized-source-${MODE}-${LINKAGE}-${WIDTH}-${variant}-cache")
if(INPUT_O0)
    set(ENV{NACELLE_TEST_SOURCE_INPUT_O0} "1")
endif()
set(asynchronous)
set(earlier_work)
if(ASYNC)
    set(ENV{NACELLE_TEST_SOURCE_ASYNC} "1")
    set(compiler_break "breakpoint set --func-regex emitObjectBytesO3")
    if(MIXED)
        set(ENV{NACELLE_TEST_SOURCE_MIXED} "1")
        set(compiler_break "breakpoint set --name nacelle::source_debug::bindCompilation --condition 'entryVersion == 2'")
    endif()
    list(APPEND asynchronous
        -o "script lldb.nacelle_compilers = []"
        -o "${compiler_break}"
        -o "breakpoint command add -s python -o \"return not (frame.GetThread().Suspend() and lldb.nacelle_compilers.append(frame.GetThread()) is None)\" 2")
    list(APPEND earlier_work
        -o "script assert all(thread.Resume() for thread in lldb.nacelle_compilers)"
        -o "breakpoint disable 2"
        -o hlsl-arm-step -o continue -o hlsl-location -o hlsl-values
        -o hlsl-arm-cancel -o continue -o hlsl-location -o hlsl-values)
endif()
execute_process(COMMAND "${LLDB}" --no-lldbinit --batch
    -o "settings set target.disable-aslr false"
    -o "command source \"${CMAKE_CURRENT_LIST_DIR}/../tools/nacelle-dbg/optimized-source-debug.lldb\""
    ${asynchronous}
    -o run -o hlsl-location -o hlsl-lanes -o hlsl-values -o hlsl-code
    ${earlier_work}
    -o hlsl-arm-step -o continue -o hlsl-location -o hlsl-values
    -o hlsl-arm-cancel -o "breakpoint disable" -o continue
    -- "${PROGRAM}" unused "${MODE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 150)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/optimized-source-${MODE}-${LINKAGE}-${WIDTH}-${variant}.log" "${output}\n${error}")
if(NOT result EQUAL 0 OR NOT output MATCHES "stop reason = breakpoint" OR
   NOT output MATCHES "tier = 2" OR NOT output MATCHES "entry_version = 2" OR
   NOT output MATCHES "OPTIMIZED SOURCE TIER2 LOCAL STEP PASS" OR
   NOT output MATCHES "SOURCE DEBUG OUTPUT PASS" OR NOT output MATCHES "exited with status = 0" OR
   output MATCHES "Traceback")
    message(FATAL_ERROR "Real optimized source step/locals/output acceptance failed\n${output}\n${error}")
endif()
if(MIXED AND NOT output MATCHES "OPTIMIZED SOURCE MIXED VS2 PS1 READY")
    message(FATAL_ERROR "Independent graphics stage versions were not observed\n${output}\n${error}")
endif()
if(ASYNC AND (NOT output MATCHES "OPTIMIZED SOURCE TIER1 LOCAL STEP PASS" OR
   NOT output MATCHES "OPTIMIZED SOURCE ASYNCHRONOUS ENTRY ISOLATION PASS"))
    message(FATAL_ERROR "Earlier entry attribution and asynchronous promotion failed\n${output}\n${error}")
endif()
file(GLOB_RECURSE cached LIST_DIRECTORIES false
    "${CMAKE_CURRENT_BINARY_DIR}/optimized-source-${MODE}-${LINKAGE}-${WIDTH}-${variant}-cache/*")
if(cached)
    message(FATAL_ERROR "Private optimized debug objects were written to the disk cache: ${cached}")
endif()
message(STATUS "Optimized LLDB source stepping, actual Tier-2 locals and unchanged full outputs passed")
