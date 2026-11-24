get_filename_component(NACELLE_PROFILE_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(nacelle_add_profile_support)
    set(root "${NACELLE_PROFILE_ROOT}")
    foreach(dependency glaze miniz)
        if(NOT EXISTS "${root}/third_party/${dependency}/CMakeLists.txt")
            message(FATAL_ERROR "Initialize profiling submodules: git submodule update --init third_party/glaze third_party/miniz third_party/zstd third_party/CLI11 third_party/DirectX-Headers")
        endif()
    endforeach()
    if(NOT EXISTS "${root}/third_party/zstd/build/cmake/CMakeLists.txt")
        message(FATAL_ERROR "Initialize third_party/zstd before enabling profiling")
    endif()
    set(glaze_BUILD_EXAMPLES OFF)
    add_subdirectory("${root}/third_party/glaze" "${CMAKE_CURRENT_BINARY_DIR}/profile-glaze" EXCLUDE_FROM_ALL)
    set(ZSTD_BUILD_PROGRAMS OFF)
    set(ZSTD_BUILD_TESTS OFF)
    set(ZSTD_BUILD_SHARED OFF)
    set(ZSTD_BUILD_STATIC ON)
    set(ZSTD_MULTITHREAD_SUPPORT OFF)
    add_subdirectory("${root}/third_party/zstd/build/cmake" "${CMAKE_CURRENT_BINARY_DIR}/profile-zstd" EXCLUDE_FROM_ALL)

    add_library(nacelle_profile_miniz STATIC
        "${root}/third_party/miniz/miniz.c"
        "${root}/third_party/miniz/miniz_tdef.c"
        "${root}/third_party/miniz/miniz_tinfl.c"
        "${root}/third_party/miniz/miniz_zip.c")
    include(GenerateExportHeader)
    generate_export_header(nacelle_profile_miniz BASE_NAME MINIZ
        EXPORT_FILE_NAME "${CMAKE_CURRENT_BINARY_DIR}/profile-generated/miniz_export.h")
    target_include_directories(nacelle_profile_miniz PUBLIC
        "${root}/third_party/miniz" "${CMAKE_CURRENT_BINARY_DIR}/profile-generated")
    target_compile_definitions(nacelle_profile_miniz PUBLIC MINIZ_STATIC_DEFINE)

    add_library(nacelle_profile_io STATIC "${root}/src/instrumentation/profile_writer.cpp"
        "${root}/src/instrumentation/cpu_counters.cpp" "${root}/src/instrumentation/profile_counters.cpp"
        "${root}/src/instrumentation/artifacts.cpp" "${root}/src/instrumentation/spatial.cpp"
        "${root}/src/instrumentation/replay_format.cpp")
    target_include_directories(nacelle_profile_io PRIVATE "${root}/third_party/xxHash")
    target_link_libraries(nacelle_profile_io PRIVATE glaze::glaze nacelle_profile_miniz libzstd_static)
    if(WIN32)
        target_link_libraries(nacelle_profile_io PRIVATE advapi32 bcrypt)
    endif()
    target_compile_definitions(nacelle_profile_io PRIVATE NACELLE_ENABLE_INSTRUMENTATION=1)
    if(NACELLE_ENABLE_CPU_COUNTERS)
        target_compile_definitions(nacelle_profile_io PUBLIC NACELLE_ENABLE_CPU_COUNTERS=1)
    endif()
    target_include_directories(nacelle_profile_io PRIVATE "${root}/include" "${root}/third_party/DirectX-Headers/include")
    if(NOT WIN32)
        target_include_directories(nacelle_profile_io PRIVATE "${root}/third_party/DirectX-Headers/include/wsl/stubs")
    endif()
    execute_process(COMMAND git rev-parse HEAD WORKING_DIRECTORY "${root}"
        OUTPUT_VARIABLE revision OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE revision_result)
    if(NOT revision_result EQUAL 0)
        message(FATAL_ERROR "Cannot determine profile producer revision")
    endif()
    target_compile_definitions(nacelle_profile_io PRIVATE
        NACELLE_PROFILE_COMPILER="${CMAKE_CXX_COMPILER_ID}-${CMAKE_CXX_COMPILER_VERSION}"
        NACELLE_PROFILE_OS="${CMAKE_SYSTEM_NAME}"
        NACELLE_PROFILE_ARCH="${CMAKE_SYSTEM_PROCESSOR}"
        NACELLE_PROFILE_REVISION="${revision}")
    add_library(nacelle_profile_capture STATIC "${root}/src/instrumentation/capture.cpp"
        "${root}/src/instrumentation/artifact_store.cpp")
    find_package(Threads REQUIRED)
    target_link_libraries(nacelle_profile_capture PUBLIC Threads::Threads PRIVATE nacelle_profile_io)
    target_compile_features(nacelle_profile_capture PUBLIC cxx_std_20)
    target_compile_definitions(nacelle_profile_capture PUBLIC NACELLE_ENABLE_INSTRUMENTATION=1)
    if(NACELLE_ENABLE_CPU_COUNTERS)
        target_compile_definitions(nacelle_profile_capture PUBLIC NACELLE_ENABLE_CPU_COUNTERS=1)
    endif()
    target_include_directories(nacelle_profile_capture PUBLIC
        "${root}/include" "${root}/src" "${root}/third_party/DirectX-Headers/include")
    if(NOT WIN32)
        target_include_directories(nacelle_profile_capture PUBLIC
            "${root}/third_party/DirectX-Headers/include/wsl/stubs")
    endif()
    foreach(target nacelle_profile_capture nacelle_profile_io nacelle_profile_miniz libzstd_static)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endforeach()
endfunction()

function(nacelle_add_profile_tests)
    add_executable(cpu_counter_provider_test "${NACELLE_PROFILE_ROOT}/tests/cpu_counter_provider_test.cpp")
    target_link_libraries(cpu_counter_provider_test PRIVATE nacelle_profile_io)
    target_include_directories(cpu_counter_provider_test PRIVATE "${NACELLE_PROFILE_ROOT}/src")
    add_test(NAME cpu_counter_provider_test COMMAND cpu_counter_provider_test)
    add_test(NAME cpu_counter_real_provider_test COMMAND cpu_counter_provider_test --require-real)
    set_tests_properties(cpu_counter_real_provider_test PROPERTIES SKIP_RETURN_CODE 77)
    if(NACELLE_ENABLE_CPU_COUNTERS)
        add_executable(cpu_counter_capture_test "${NACELLE_PROFILE_ROOT}/tests/cpu_counter_capture_test.cpp")
        target_link_libraries(cpu_counter_capture_test PRIVATE nacelle_profile_capture nacelle_profile_io)
        add_test(NAME cpu_counter_capture_test COMMAND cpu_counter_capture_test)
    endif()
    add_executable(replay_format_test "${NACELLE_PROFILE_ROOT}/tests/replay_format_test.cpp")
    target_link_libraries(replay_format_test PRIVATE nacelle_profile_io)
    add_test(NAME replay_format_test COMMAND replay_format_test)
    add_executable(instrumentation_test "${NACELLE_PROFILE_ROOT}/tests/instrumentation_test.cpp")
    target_link_libraries(instrumentation_test PRIVATE
        nacelle_profile_capture nacelle_profile_io nacelle_profile_miniz libzstd_static)
    add_test(NAME instrumentation_test COMMAND instrumentation_test)
    add_library(profile_writer_diagnostics OBJECT
        "${NACELLE_PROFILE_ROOT}/src/instrumentation/profile_writer.cpp")
    get_target_property(io_includes nacelle_profile_io INCLUDE_DIRECTORIES)
    get_target_property(io_definitions nacelle_profile_io COMPILE_DEFINITIONS)
    target_include_directories(profile_writer_diagnostics PRIVATE ${io_includes})
    target_compile_definitions(profile_writer_diagnostics PRIVATE
        ${io_definitions} NACELLE_PROFILE_MEMORY_DIAGNOSTICS=1)
    target_link_libraries(profile_writer_diagnostics PRIVATE glaze::glaze nacelle_profile_miniz libzstd_static)
    add_executable(profile_memory_test "${NACELLE_PROFILE_ROOT}/tests/profile_memory_test.cpp"
        $<TARGET_OBJECTS:profile_writer_diagnostics>)
    target_link_libraries(profile_memory_test PRIVATE nacelle_profile_capture nacelle_profile_io)
    target_include_directories(profile_memory_test PRIVATE "${NACELLE_PROFILE_ROOT}/tools/nacelle-bench")
    if(WIN32)
        target_link_libraries(profile_memory_test PRIVATE psapi)
    endif()
    add_test(NAME profile_memory_streaming_test COMMAND profile_memory_test 1 near-full)
    add_test(NAME profile_memory_overflow_test COMMAND profile_memory_test 1 overflow)
    add_test(NAME profile_memory_threads_test COMMAND profile_memory_test 1 threads)
    add_test(NAME profile_memory_metadata_test COMMAND profile_memory_test 1 metadata)
    add_test(NAME profile_memory_versions_test COMMAND profile_memory_test 1 versions)
    foreach(stage metadata-begin ordering-begin trace-count-begin compression-begin archive-begin write-begin write-progress)
        add_test(NAME profile_memory_${stage}_failure_test
            COMMAND profile_memory_test 1 near-full "${stage}")
    endforeach()
endfunction()
