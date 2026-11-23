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
        "${root}/src/instrumentation/artifacts.cpp")
    target_include_directories(nacelle_profile_io PRIVATE "${root}/third_party/xxHash")
    target_link_libraries(nacelle_profile_io PRIVATE glaze::glaze nacelle_profile_miniz libzstd_static)
    target_compile_definitions(nacelle_profile_io PRIVATE NACELLE_ENABLE_INSTRUMENTATION=1)
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
    add_library(nacelle_profile_capture STATIC "${root}/src/instrumentation/capture.cpp")
    find_package(Threads REQUIRED)
    target_link_libraries(nacelle_profile_capture PUBLIC Threads::Threads PRIVATE nacelle_profile_io)
    target_compile_features(nacelle_profile_capture PUBLIC cxx_std_20)
    target_compile_definitions(nacelle_profile_capture PUBLIC NACELLE_ENABLE_INSTRUMENTATION=1)
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
    add_executable(instrumentation_test "${NACELLE_PROFILE_ROOT}/tests/instrumentation_test.cpp")
    target_link_libraries(instrumentation_test PRIVATE
        nacelle_profile_capture nacelle_profile_io nacelle_profile_miniz libzstd_static)
    add_test(NAME instrumentation_test COMMAND instrumentation_test)
endfunction()
