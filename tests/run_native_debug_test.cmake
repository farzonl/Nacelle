foreach(input PROGRAM LLDB STAGE LINKAGE)
    if(NOT DEFINED ${input})
        message(FATAL_ERROR "Native debugger test requires ${input}")
    endif()
endforeach()
if(STAGE STREQUAL "lifetime")
    execute_process(COMMAND "${LLDB}" --no-lldbinit --batch
        -o "settings set symbols.enable-external-lookup false"
        -o "settings set symbols.enable-lldb-index-cache false"
        -o "settings set target.disable-aslr false"
        -o "settings set plugin.jit-loader.gdb.enable on"
        -o "breakpoint set --name nacelle_cs_lifetime"
        -o "breakpoint set --name nacelle_native_debug_unloaded"
        -o run -o "breakpoint list 1" -o continue -o "breakpoint list 1"
        -o "breakpoint disable" -o continue -- "${PROGRAM}"
        RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 120)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/native-debug-lifetime.log" "${output}\n${error}")
    if(NOT result EQUAL 0 OR NOT output MATCHES "stop reason = breakpoint 1" OR
       NOT output MATCHES "stop reason = breakpoint 2" OR
       NOT output MATCHES "resolved = 1, hit count = 1" OR
       NOT output MATCHES "address = <Unknown>[^\n]*unresolved, hit count = 1" OR
       NOT output MATCHES "NATIVE DEBUG LIFETIME PASS" OR NOT output MATCHES "exited with status = 0")
        message(FATAL_ERROR "LLDB unregister/stale-breakpoint acceptance failed\n${output}\n${error}")
    endif()
    message(STATUS "LLDB removes the unloaded JIT symbol and leaves the old breakpoint unresolved")
    return()
endif()
set(ENV{NACELLE_DISABLE_TIER2} "0")
set(ENV{NACELLE_SIMD_WIDTH} "4")
set(ENV{NACELLE_CACHE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/native-debug-${LINKAGE}-${STAGE}-cache")
if(STAGE STREQUAL "cs")
    set(mode native-compute)
    set(warmup cached)
else()
    set(mode native-graphics)
    set(warmup graphics-cached)
endif()
string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef suffix)
set(profile "${CMAKE_CURRENT_BINARY_DIR}/native-debug-${LINKAGE}-${STAGE}-${suffix}.nacelle-profile")
execute_process(COMMAND "${PROGRAM}" "${profile}" "${warmup}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 120)
file(REMOVE "${profile}" "${profile}.warmup")
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Optimized cache warmup failed: ${result}\n${output}\n${error}")
endif()
file(GLOB_RECURSE before_files "$ENV{NACELLE_CACHE_DIR}/*")
if(NOT before_files)
    message(FATAL_ERROR "Optimized cache warmup produced no objects")
endif()
foreach(path IN LISTS before_files)
    file(SHA256 "${path}" digest)
    list(APPEND before "${path}:${digest}")
endforeach()
execute_process(COMMAND "${LLDB}" --no-lldbinit --batch
    -o "settings set symbols.enable-external-lookup false"
    -o "settings set symbols.enable-lldb-index-cache false"
    -o "settings set target.disable-aslr false"
    -o "settings set plugin.jit-loader.gdb.enable on"
    -o "breakpoint set --func-regex ^nacelle_${STAGE}_"
    -o run -o "frame info" -o "register read --format hex pc"
    -o "thread step-inst" -o "thread step-inst"
    -o "register read --format hex pc" -o "thread backtrace -c 4"
    -o "breakpoint disable" -o continue
    -- "${PROGRAM}" unused "${mode}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE error TIMEOUT 120)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/native-debug-${LINKAGE}-${STAGE}.log" "${output}\n${error}")
if(NOT result EQUAL 0 OR NOT output MATCHES "stop reason = breakpoint" OR
   NOT output MATCHES "stop reason = instruction step" OR
   NOT output MATCHES "frame #0: [^\n]*JIT[^\n]*nacelle_${STAGE}_" OR
   NOT output MATCHES "frame #1: [^\n]*nacelle::" OR
   NOT output MATCHES "NATIVE DEBUG OUTPUT PASS" OR
   NOT output MATCHES "exited with status = 0")
    message(FATAL_ERROR "LLDB shader breakpoint/step/frame/output acceptance failed: ${result}\n${output}\n${error}")
endif()
string(REGEX MATCH "NATIVE DEBUG READY address=([0-9]+) end=([0-9]+) symbol=nacelle_${STAGE}_" range "${output}")
if(NOT range)
    message(FATAL_ERROR "Missing exact shader code range\n${output}")
endif()
set(begin "${CMAKE_MATCH_1}")
set(end "${CMAKE_MATCH_2}")
string(REGEX MATCHALL "rip = 0x[0-9a-f]+" registers "${output}")
list(LENGTH registers count)
if(NOT count EQUAL 2)
    message(FATAL_ERROR "Expected two actual PC register reads\n${output}")
endif()
list(GET registers 0 pc_before)
list(GET registers 1 pc_after)
string(REPLACE "rip = " "" pc_before "${pc_before}")
string(REPLACE "rip = " "" pc_after "${pc_after}")
math(EXPR pc_before "${pc_before}")
math(EXPR pc_after "${pc_after}")
if(NOT pc_before EQUAL begin OR pc_after LESS_EQUAL pc_before OR pc_after GREATER_EQUAL end)
    message(FATAL_ERROR "LLDB did not step inside the exact executed native code version")
endif()
file(GLOB_RECURSE after_files "$ENV{NACELLE_CACHE_DIR}/*")
foreach(path IN LISTS after_files)
    file(SHA256 "${path}" digest)
    list(APPEND after "${path}:${digest}")
endforeach()
if(NOT before STREQUAL after)
    message(FATAL_ERROR "Native debug mode modified the optimized cache")
endif()
message(STATUS "LLDB ${LINKAGE}/${STAGE}: exact JIT entry, instruction advancement, frames, outputs, cache isolation passed")
