cmake_minimum_required(VERSION 3.24)

foreach(input BUILD_DIR LIBRARY_FILE NM_TOOL)
    if(NOT DEFINED ${input} OR NOT EXISTS "${${input}}")
        message(FATAL_ERROR "Profile isolation check requires an existing ${input}")
    endif()
endforeach()

file(STRINGS "${BUILD_DIR}/CMakeCache.txt" profile_options
    REGEX "^NACELLE_(ENABLE_INSTRUMENTATION|BUILD_PROFILE_(TOOLS|TESTS)):BOOL=")
foreach(option NACELLE_ENABLE_INSTRUMENTATION NACELLE_BUILD_PROFILE_TOOLS NACELLE_BUILD_PROFILE_TESTS)
    if(NOT "${option}:BOOL=OFF" IN_LIST profile_options)
        message(FATAL_ERROR "Profile isolation check requires ${option}=OFF")
    endif()
endforeach()

if(NOT EXISTS "${BUILD_DIR}/compile_commands.json")
    message(FATAL_ERROR "Profile isolation check requires CMAKE_EXPORT_COMPILE_COMMANDS=ON")
endif()
file(READ "${BUILD_DIR}/compile_commands.json" compile_commands)
if(compile_commands MATCHES "NACELLE_ENABLE_INSTRUMENTATION|nacelle_profile_|native_debug|source_debug|replay_(capture|execute|shader|format)|profile-zstd|profile-glaze")
    message(FATAL_ERROR "Instrumentation definitions or support sources leaked into the OFF build")
endif()

execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target help
    RESULT_VARIABLE result OUTPUT_VARIABLE targets ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect OFF build targets: ${error}")
endif()
if(targets MATCHES "nacelle_profile_|nacelle-replay|replay_runtime|replay_format|profile_memory_test|profile_writer_diagnostics|profile-zstd|profile-glaze|CLI11")
    message(FATAL_ERROR "Profiling support targets leaked into the OFF build")
endif()

execute_process(COMMAND "${NM_TOOL}" --demangle "${LIBRARY_FILE}"
    RESULT_VARIABLE result OUTPUT_VARIABLE symbols ERROR_VARIABLE error)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Cannot inspect OFF library symbols: ${error}")
endif()
if(symbols MATCHES "Nacelle(Begin|End|Destroy)ProfileCapture|Nacelle(ConfigureNativeDebug|GetNativeDebugCode)|Nacelle[A-Za-z]*SourceDebug|nacelle_(optimized_)?source_debug|Nacelle(ConfigureReplay|WaitForReplayCompilation|BeginReplayCapture|EndReplayCapture|DestroyReplayCapture|ReplayBundle)|nacelle::(profile|replay|native_debug|source_debug)::")
    message(FATAL_ERROR "Capture symbols or references leaked into the OFF library")
endif()

message(STATUS "Instrumentation-OFF compile definitions, support targets, and library symbols are isolated")
