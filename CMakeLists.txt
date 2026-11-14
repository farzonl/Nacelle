cmake_minimum_required(VERSION 3.24)
project(Nacelle LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Auto-detect LLVM/MLIR if not explicitly provided
if(APPLE AND NOT LLVM_DIR)
    if(EXISTS "/opt/homebrew/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake")
        set(LLVM_DIR "/opt/homebrew/opt/llvm/lib/cmake/llvm")
        set(MLIR_DIR "/opt/homebrew/opt/llvm/lib/cmake/mlir")
    elseif(EXISTS "/usr/local/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake")
        set(LLVM_DIR "/usr/local/opt/llvm/lib/cmake/llvm")
        set(MLIR_DIR "/usr/local/opt/llvm/lib/cmake/mlir")
    endif()
elseif(UNIX AND NOT APPLE AND NOT LLVM_DIR)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.toolchains/llvm/lib/cmake/llvm/LLVMConfig.cmake" AND
       EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.toolchains/llvm/lib/cmake/mlir/MLIRConfig.cmake")
        set(LLVM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.toolchains/llvm/lib/cmake/llvm")
        set(MLIR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/.toolchains/llvm/lib/cmake/mlir")
    else()
        foreach(_nacelle_llvm_ver IN ITEMS 24 23 22 21)
            if(EXISTS "/usr/lib/llvm-${_nacelle_llvm_ver}/lib/cmake/llvm/LLVMConfig.cmake" AND
               EXISTS "/usr/lib/llvm-${_nacelle_llvm_ver}/lib/cmake/mlir/MLIRConfig.cmake")
                set(LLVM_DIR "/usr/lib/llvm-${_nacelle_llvm_ver}/lib/cmake/llvm")
                set(MLIR_DIR "/usr/lib/llvm-${_nacelle_llvm_ver}/lib/cmake/mlir")
                break()
            endif()
        endforeach()
    endif()
endif()
if(LLVM_DIR AND NOT MLIR_DIR AND EXISTS "${LLVM_DIR}/../mlir/MLIRConfig.cmake")
    get_filename_component(MLIR_DIR "${LLVM_DIR}/../mlir" REALPATH)
endif()

# 1. DirectX-Headers (cross-platform: Win32 SDK on Windows, wsl/stubs on macOS/Linux)
set(DXHEADERS_BUILD_TEST OFF CACHE BOOL "" FORCE)
set(DXHEADERS_BUILD_GOOGLE_TEST OFF CACHE BOOL "" FORCE)
set(DXHEADERS_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/DirectX-Headers)

# 2. LLVM & MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} (${LLVM_DIR}), dylib=${LLVM_LINK_LLVM_DYLIB}")
message(STATUS "Found MLIR (${MLIR_DIR})")

if(LLVM_LINK_LLVM_DYLIB)
    set(NACELLE_LLVM_LIBS LLVM)
else()
    llvm_map_components_to_libnames(NACELLE_LLVM_LIBS
        Core Object BitReader BitWriter IRReader Linker Passes Analysis
        TransformUtils ScalarOpts Vectorize InstCombine AggressiveInstCombine
        IPO OrcJIT OrcDebugging OrcTargetProcess OrcShared ExecutionEngine JITLink MC
        Support Target TargetParser native
    )
endif()

set(NACELLE_MLIR_LIBS
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRDLTIDialect
    MLIRLLVMDialect
    MLIRSCFDialect
    MLIRControlFlowDialect
    MLIRFuncDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRVectorDialect
    MLIRUBDialect
    MLIRTargetLLVMIRImport
    MLIRTargetLLVMIRExport
    MLIRBuiltinToLLVMIRTranslation
    MLIRLLVMToLLVMIRTranslation
    MLIRLLVMIRToLLVMTranslation
    MLIRControlFlowToSCF
    MLIRSCFToControlFlow
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRArithToLLVM
    MLIRMathToLLVM
    MLIRFuncToLLVM
    MLIRMathToLibm
    MLIRControlFlowToLLVM
    MLIRUBToLLVM
    MLIRReconcileUnrealizedCasts
)

# 3. Compile texture_sampler.cpp to portable LLVM bitcode (.bc) and embed as C array
set(TEX_SAMPLER_BC "${CMAKE_CURRENT_BINARY_DIR}/generated/texture_sampler.bc")
set(TEX_SAMPLER_BC_HDR "${CMAKE_CURRENT_BINARY_DIR}/generated/texture_sampler_bc.h")

find_program(NACELLE_CLANGXX NAMES clang++ HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)

add_custom_command(
    OUTPUT "${TEX_SAMPLER_BC}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND "${NACELLE_CLANGXX}" -std=c++20 -O3 -emit-llvm -c
            -fno-exceptions -fno-rtti -ffreestanding -fno-math-errno
            "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_bc/texture_sampler.cpp"
            -o "${TEX_SAMPLER_BC}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/runtime_bc/texture_sampler.cpp"
    COMMENT "Compiling freestanding texture_sampler.cpp to LLVM bitcode"
)

add_custom_command(
    OUTPUT "${TEX_SAMPLER_BC_HDR}"
    COMMAND ${CMAKE_COMMAND}
            "-DINPUT_FILE=${TEX_SAMPLER_BC}"
            "-DOUTPUT_FILE=${TEX_SAMPLER_BC_HDR}"
            "-DSYMBOL_NAME=kTextureSamplerBitcode"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    DEPENDS "${TEX_SAMPLER_BC}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedBinary.cmake"
    COMMENT "Embedding texture_sampler.bc into generated header"
)

# Compute deterministic build-time SHA-256 stamp across all compiler, ABI, and runtime bitcode inputs
set(COMPILER_VERSION_HDR "${CMAKE_CURRENT_BINARY_DIR}/generated/compiler_version.h")
set(COMPILER_HASH_MANIFEST "${CMAKE_CURRENT_BINARY_DIR}/generated/compiler_hash_inputs.txt")
set(NACELLE_COMPILER_HASH_INPUTS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/d3d12_types.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/api/pipeline.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common/format_utils.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/dxil_loader.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/dxil_loader.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit_engine.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/jit_engine.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/mlir_pipeline.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/mlir_pipeline.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/pso_synthesizer.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/pso_synthesizer.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/shader_compiler.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/simt_to_simd_pass.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/simt_to_simd_pass.h"
    "${TEX_SAMPLER_BC}"
)
string(JOIN "\n" COMPILER_HASH_MANIFEST_CONTENT
    "LLVM_PACKAGE_VERSION=${LLVM_PACKAGE_VERSION}"
    ${NACELLE_COMPILER_HASH_INPUTS}
)
file(GENERATE
    OUTPUT "${COMPILER_HASH_MANIFEST}"
    CONTENT "${COMPILER_HASH_MANIFEST_CONTENT}\n"
)

add_custom_command(
    OUTPUT "${COMPILER_VERSION_HDR}"
    COMMAND ${CMAKE_COMMAND}
            "-DMANIFEST_FILE=${COMPILER_HASH_MANIFEST}"
            "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DOUTPUT_FILE=${COMPILER_VERSION_HDR}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateCompilerVersion.cmake"
    DEPENDS ${NACELLE_COMPILER_HASH_INPUTS}
            "${COMPILER_HASH_MANIFEST}"
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GenerateCompilerVersion.cmake"
    COMMENT "Generating compiler build hash header"
)

add_custom_target(nacelle_embedded_bc DEPENDS "${TEX_SAMPLER_BC_HDR}" "${COMPILER_VERSION_HDR}")

# 4. Core Nacelle Object Library (shared between static libnacelle and shared libnacelle_d3d12)
set(NACELLE_SOURCES
    src/common/format_utils.cpp
    src/api/resource.cpp
    src/api/descriptors.cpp
    src/api/pipeline.cpp
    src/api/command_list.cpp
    src/api/device.cpp
    src/api/dxcore.cpp
    src/compiler/dxil_loader.cpp
    src/compiler/mlir_pipeline.cpp
    src/compiler/simt_to_simd_pass.cpp
    src/compiler/jit_engine.cpp
    src/compiler/pso_synthesizer.cpp
    src/rasterizer/geometry_pipe.cpp
    src/rasterizer/raytracing_bvh.cpp
    src/rasterizer/raytracing_pipeline.cpp
    src/rasterizer/tile_binner.cpp
    src/rasterizer/quad_rasterizer.cpp
)

separate_arguments(LLVM_DEFS_LIST NATIVE_COMMAND "${LLVM_DEFINITIONS}")

add_library(nacelle_objs OBJECT ${NACELLE_SOURCES})
add_dependencies(nacelle_objs nacelle_embedded_bc)
set_target_properties(nacelle_objs PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(nacelle_objs
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectXMath/Inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bcdec"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/thread-pool/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxHash"
)

if(NOT WIN32)
    target_include_directories(nacelle_objs PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/common/posix_sal")
endif()

target_include_directories(nacelle_objs SYSTEM PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)

target_compile_definitions(nacelle_objs PRIVATE ${LLVM_DEFS_LIST})
target_link_libraries(nacelle_objs PUBLIC Microsoft::DirectX-Headers Microsoft::DirectX-Guids)

# Static library: libnacelle
add_library(nacelle STATIC $<TARGET_OBJECTS:nacelle_objs>)
target_include_directories(nacelle PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
)
target_link_libraries(nacelle
    PUBLIC
        Microsoft::DirectX-Headers
        Microsoft::DirectX-Guids
    PRIVATE
        ${NACELLE_MLIR_LIBS}
        ${NACELLE_LLVM_LIBS}
)
if(UNIX AND NOT APPLE)
    target_link_libraries(nacelle PRIVATE pthread dl)
endif()

# Shared D3D12/DXCore drop-in library: libnacelle_d3d12
add_library(nacelle_d3d12 SHARED ${NACELLE_SOURCES})
if(WIN32)
    target_sources(nacelle_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/api/nacelle_d3d12.def")
endif()
add_dependencies(nacelle_d3d12 nacelle_embedded_bc)
target_include_directories(nacelle_d3d12
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_BINARY_DIR}/generated"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectXMath/Inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/bcdec"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/thread-pool/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/xxHash"
)
if(NOT WIN32)
    target_include_directories(nacelle_d3d12 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/common/posix_sal")
endif()
target_include_directories(nacelle_d3d12 SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(nacelle_d3d12 PRIVATE ${LLVM_DEFS_LIST} NACELLE_EXPORT_D3D12_SYMBOLS)
target_link_libraries(nacelle_d3d12
    PUBLIC
        Microsoft::DirectX-Headers
        Microsoft::DirectX-Guids
    PRIVATE
        ${NACELLE_MLIR_LIBS}
        ${NACELLE_LLVM_LIBS}
)
if(UNIX AND NOT APPLE)
    target_link_libraries(nacelle_d3d12 PRIVATE pthread dl)
endif()

# 5. CLI Tools
add_executable(nacelle-compile tools/nacelle_compile.cpp)
target_include_directories(nacelle-compile PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/thread-pool/include"
)
if(NOT WIN32)
    target_include_directories(nacelle-compile PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/common/posix_sal")
endif()
target_include_directories(nacelle-compile SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(nacelle-compile PRIVATE ${LLVM_DEFS_LIST})
target_link_libraries(nacelle-compile PRIVATE nacelle ${NACELLE_MLIR_LIBS} ${NACELLE_LLVM_LIBS})

if(APPLE)
    add_executable(nacelle-bench tools/nacelle_bench.cpp tools/metal_bench.mm)
    set_source_files_properties(tools/metal_bench.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    find_path(METAL_IRCONVERTER_INC metal_irconverter/metal_irconverter.h HINTS /usr/local/include /opt/homebrew/include)
    find_library(METAL_IRCONVERTER_LIB metalirconverter HINTS /usr/local/lib /opt/homebrew/lib)
    target_link_libraries(nacelle-bench PRIVATE "-framework Metal" "-framework Foundation")
    if(METAL_IRCONVERTER_INC AND METAL_IRCONVERTER_LIB)
        target_compile_definitions(nacelle-bench PRIVATE NACELLE_HAS_METAL_IRCONVERTER=1)
        target_include_directories(nacelle-bench PRIVATE "${METAL_IRCONVERTER_INC}")
        target_link_libraries(nacelle-bench PRIVATE "${METAL_IRCONVERTER_LIB}")
        get_filename_component(METAL_IRCONVERTER_LIBDIR "${METAL_IRCONVERTER_LIB}" DIRECTORY)
        set_target_properties(nacelle-bench PROPERTIES BUILD_RPATH "${METAL_IRCONVERTER_LIBDIR}")
    endif()
else()
    add_executable(nacelle-bench tools/nacelle_bench.cpp)
endif()
add_dependencies(nacelle-bench nacelle_test_shaders)
target_include_directories(nacelle-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/thread-pool/include"
)
target_compile_definitions(nacelle-bench PRIVATE
    NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
)
target_link_libraries(nacelle-bench PRIVATE nacelle)
if(UNIX AND NOT APPLE)
    target_link_libraries(nacelle-bench PRIVATE dl)
endif()

# 6. SDL3 Interactive + Headless Viewer
option(NACELLE_FETCH_SDL3 "Automatically fetch SDL3 when SDL3 is not installed on the system" ON)
find_package(SDL3 QUIET HINTS /opt/homebrew /usr/local)
if(WIN32 AND NOT SDL3_FOUND AND NACELLE_FETCH_SDL3)
    include(FetchContent)
    FetchContent_Declare(
        nacelle_sdl3_vc
        URL "https://github.com/libsdl-org/SDL/releases/download/release-3.4.20/SDL3-devel-3.4.20-VC.zip"
        URL_HASH SHA256=1ec3009655ee37b1f73e52b3b120acc5afd8abaf9532f3303b1bf1dfe91b5969
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nacelle_sdl3_vc)
    find_package(SDL3 QUIET CONFIG PATHS "${nacelle_sdl3_vc_SOURCE_DIR}/cmake" NO_DEFAULT_PATH)
elseif(UNIX AND NOT APPLE AND NOT SDL3_FOUND AND NACELLE_FETCH_SDL3)
    include(FetchContent)
    find_package(X11 QUIET)
    set(SDL_UNIX_CONSOLE_BUILD ON CACHE BOOL "" FORCE)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
    set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
    set(SDL_TESTS OFF CACHE BOOL "" FORCE)
    set(SDL_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SDL_INSTALL OFF CACHE BOOL "" FORCE)
    set(SDL_UNINSTALL OFF CACHE BOOL "" FORCE)
    set(SDL_AUDIO OFF CACHE BOOL "" FORCE)
    set(SDL_CAMERA OFF CACHE BOOL "" FORCE)
    set(SDL_DIALOG OFF CACHE BOOL "" FORCE)
    set(SDL_GPU OFF CACHE BOOL "" FORCE)
    set(SDL_HAPTIC OFF CACHE BOOL "" FORCE)
    set(SDL_HIDAPI OFF CACHE BOOL "" FORCE)
    set(SDL_JOYSTICK OFF CACHE BOOL "" FORCE)
    set(SDL_POWER OFF CACHE BOOL "" FORCE)
    set(SDL_SENSOR OFF CACHE BOOL "" FORCE)
    set(SDL_TRAY OFF CACHE BOOL "" FORCE)
    if(X11_FOUND)
        if(X11_Xcursor_FOUND)
            set(SDL_X11_XCURSOR ON CACHE BOOL "" FORCE)
        else()
            set(SDL_X11_XCURSOR OFF CACHE BOOL "" FORCE)
        endif()
        if(X11_Xi_FOUND)
            set(SDL_X11_XINPUT ON CACHE BOOL "" FORCE)
        else()
            set(SDL_X11_XINPUT OFF CACHE BOOL "" FORCE)
        endif()
        if(X11_Xfixes_FOUND AND X11_Xi_FOUND)
            set(SDL_X11_XFIXES ON CACHE BOOL "" FORCE)
        else()
            set(SDL_X11_XFIXES OFF CACHE BOOL "" FORCE)
        endif()
        if(X11_Xrandr_FOUND)
            set(SDL_X11_XRANDR ON CACHE BOOL "" FORCE)
        else()
            set(SDL_X11_XRANDR OFF CACHE BOOL "" FORCE)
        endif()
        if(X11_Xss_FOUND)
            set(SDL_X11_XSCRNSAVER ON CACHE BOOL "" FORCE)
        else()
            set(SDL_X11_XSCRNSAVER OFF CACHE BOOL "" FORCE)
        endif()
        if(X11_Xtst_FOUND)
            set(SDL_X11_XTEST ON CACHE BOOL "" FORCE)
        else()
            set(SDL_X11_XTEST OFF CACHE BOOL "" FORCE)
        endif()
    endif()
    FetchContent_Declare(
        nacelle_sdl3
        URL "https://github.com/libsdl-org/SDL/releases/download/release-3.4.20/SDL3-3.4.20.tar.gz"
        URL_HASH SHA256=f3bb06f55bed9557227a700af2ea62e49497ebdf58037732a731a5fb3ef8581f
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(nacelle_sdl3)
    get_directory_property(_nacelle_sdl3_have_x11 DIRECTORY "${nacelle_sdl3_SOURCE_DIR}" DEFINITION HAVE_X11)
    get_directory_property(_nacelle_sdl3_have_wayland DIRECTORY "${nacelle_sdl3_SOURCE_DIR}" DEFINITION HAVE_WAYLAND)
    if(NOT _nacelle_sdl3_have_wayland)
        unset(__pkg_config_checked_PC_WAYLAND CACHE)
        unset(__pkg_config_checked_PC_LIBDECOR CACHE)
    endif()
    if(NOT _nacelle_sdl3_have_x11 AND NOT _nacelle_sdl3_have_wayland)
        message(WARNING
            "SDL3 was built without X11 or Wayland desktop video backends (missing development headers). "
            "Install X11/Wayland dev packages (e.g. libx11-dev libxext-dev pkg-config libwayland-dev "
            "libegl-dev libxkbcommon-dev libdecor-0-dev) and re-run CMake to enable desktop windows."
        )
    else()
        message(STATUS "Built SDL3 from source (X11=${_nacelle_sdl3_have_x11}, Wayland=${_nacelle_sdl3_have_wayland})")
    endif()
endif()
add_executable(nacelle_viewer apps/nacelle_viewer.cpp)
add_dependencies(nacelle_viewer nacelle_test_shaders)
target_include_directories(nacelle_viewer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
)
if(NOT WIN32)
    target_include_directories(nacelle_viewer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/common/posix_sal")
endif()
target_include_directories(nacelle_viewer SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(nacelle_viewer PRIVATE
    ${LLVM_DEFS_LIST}
    NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
)
target_link_libraries(nacelle_viewer PRIVATE nacelle)
if(UNIX AND NOT APPLE)
    target_link_libraries(nacelle_viewer PRIVATE pthread dl)
endif()
if(SDL3_FOUND OR TARGET SDL3::SDL3)
    set(NACELLE_VIEWER_HAS_SDL3 ON)
    message(STATUS "Found SDL3; enabling windowed mode in nacelle_viewer")
    target_compile_definitions(nacelle_viewer PRIVATE NACELLE_HAS_SDL3=1)
    target_link_libraries(nacelle_viewer PRIVATE SDL3::SDL3)
    if(WIN32 AND TARGET SDL3::SDL3-shared)
        add_custom_command(TARGET nacelle_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:SDL3::SDL3-shared>"
                "$<TARGET_FILE_DIR:nacelle_viewer>"
        )
    endif()
else()
    set(NACELLE_VIEWER_HAS_SDL3 OFF)
    message(STATUS "SDL3 not found; nacelle_viewer will build in headless-only mode")
endif()

# 7. Tests
enable_testing()
add_subdirectory(tests)
