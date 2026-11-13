cmake_minimum_required(VERSION 3.24)
project(Nacelle LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Auto-detect Homebrew LLVM/MLIR on macOS if not explicitly provided
if(APPLE AND NOT LLVM_DIR)
    if(EXISTS "/opt/homebrew/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake")
        set(LLVM_DIR "/opt/homebrew/opt/llvm/lib/cmake/llvm")
        set(MLIR_DIR "/opt/homebrew/opt/llvm/lib/cmake/mlir")
    elseif(EXISTS "/usr/local/opt/llvm/lib/cmake/llvm/LLVMConfig.cmake")
        set(LLVM_DIR "/usr/local/opt/llvm/lib/cmake/llvm")
        set(MLIR_DIR "/usr/local/opt/llvm/lib/cmake/mlir")
    endif()
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
            -fno-exceptions -fno-rtti -ffreestanding
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

add_custom_target(nacelle_embedded_bc DEPENDS "${TEX_SAMPLER_BC_HDR}")

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
find_package(SDL3 QUIET HINTS /opt/homebrew /usr/local)
add_executable(nacelle_viewer apps/nacelle_viewer.cpp)
add_dependencies(nacelle_viewer nacelle_test_shaders)
target_include_directories(nacelle_viewer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/DirectX-Headers/include"
)
target_compile_definitions(nacelle_viewer PRIVATE
    NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
)
target_link_libraries(nacelle_viewer PRIVATE nacelle)
if(SDL3_FOUND)
    message(STATUS "Found SDL3; enabling windowed mode in nacelle_viewer")
    target_compile_definitions(nacelle_viewer PRIVATE NACELLE_HAS_SDL3=1)
    target_link_libraries(nacelle_viewer PRIVATE SDL3::SDL3)
else()
    message(STATUS "SDL3 not found; nacelle_viewer will build in headless-only mode")
endif()

# 7. Tests
enable_testing()
add_subdirectory(tests)
