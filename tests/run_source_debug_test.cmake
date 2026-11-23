foreach(input PROGRAM LLDB MODE LINKAGE)
    if(NOT DEFINED ${input})
        message(FATAL_ERROR "Source debugger test requires ${input}")
    endif()
endforeach()
set(ENV{NACELLE_SIMD_WIDTH} "${WIDTH}")
if(LOOP)
    set(ENV{NACELLE_TEST_SOURCE_LOOP} "1")
endif()
if(HELPER)
    set(ENV{NACELLE_TEST_SOURCE_HELPER} "1")
    set(helper_step -o hlsl-arm-step -o continue -o "frame variable snapshot->site snapshot->stop snapshot->locals")
endif()
if(STALE)
    set(stale_command -o "expression -- command->session = 0")
endif()
execute_process(COMMAND "${LLDB}" --no-lldbinit --batch
    -o "settings set target.disable-aslr false"
    -o "command source \"${CMAKE_CURRENT_LIST_DIR}/../tools/nacelle-dbg/source-debug.lldb\""
    -o run -o "frame variable snapshot->site snapshot->code snapshot->active_mask snapshot->real_mask snapshot->selected_lane snapshot->stop"
    -o hlsl-arm-step
    -o continue -o "frame variable snapshot->site snapshot->stop"
    -o hlsl-arm-step
    -o continue -o "frame variable snapshot->site snapshot->stop snapshot->locals"
    ${helper_step}
    ${stale_command}
    -o hlsl-arm-cancel
    -o "breakpoint disable" -o continue -- "${PROGRAM}" unused "${MODE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 150)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${MODE}-${LINKAGE}-${WIDTH}-${LOOP}-${HELPER}-${STALE}-lldb.log" "${output}\n${error}")
if(STALE AND NOT "${output}\n${error}" MATCHES "Debugger mailbox has stale identities or an invalid action")
    message(FATAL_ERROR "Stale debugger mailbox was not explicitly rejected\n${output}\n${error}")
endif()
if(NOT result EQUAL 0 OR NOT output MATCHES "stop reason = breakpoint" OR
   NOT output MATCHES "SOURCE DEBUG LOCAL PASS" OR
   NOT output MATCHES "SOURCE DEBUG STEP PASS stops=[34]" OR NOT output MATCHES "SOURCE DEBUG OUTPUT PASS" OR
   NOT output MATCHES "exited with status = 0")
    message(FATAL_ERROR "LLDB source breakpoint/step/locals/output acceptance failed\n${output}\n${error}")
endif()
string(REGEX MATCH "SOURCE DEBUG READY session=1 line=([0-9]+)" ready "${output}")
set(expected "${CMAKE_MATCH_1}")
string(REGEX MATCHALL "line = [0-9]+" lines "${output}")
list(LENGTH lines line_count)
if(HELPER)
    if(NOT line_count EQUAL 4 OR output MATCHES "inline_depth = [1-7]")
        message(FATAL_ERROR "Source step entered an inlined helper or missed its caller sites\n${output}")
    endif()
    list(GET lines 0 first)
    list(GET lines 1 second)
    list(GET lines 2 third)
    list(GET lines 3 fourth)
    if(NOT first STREQUAL "line = ${expected}" OR first STREQUAL second OR
       NOT second STREQUAL third OR third STREQUAL fourth)
        message(FATAL_ERROR "Source step-over did not preserve the caller's actual mapped sites\n${output}")
    endif()
    message(STATUS "LLDB source step-over preserved inline frames without entering the helper")
    return()
endif()
if(NOT line_count EQUAL 3)
    message(FATAL_ERROR "Expected three actual HLSL source stops\n${output}")
endif()
list(GET lines 0 first)
list(GET lines 1 second)
list(GET lines 2 third)
if(NOT first STREQUAL "line = ${expected}" OR first STREQUAL second OR second STREQUAL third)
    message(FATAL_ERROR "LLDB source breakpoint/step did not resolve and advance actual HLSL lines\n${output}")
endif()
if(LOOP AND NOT first STREQUAL third)
    message(FATAL_ERROR "Source stepping did not revisit the loop body\n${output}")
endif()
message(STATUS "LLDB source stop, step, cancellation/draining and output equality passed")
