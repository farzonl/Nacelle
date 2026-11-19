set(NACELLE_D3D12_SDK_VERSION "0" CACHE STRING "DirectX 12 Agility SDK version number")

# Worker: Nacelle (links libnacelle / LLVM)
add_executable(bench-worker-nacelle backends/nacelle/worker.cpp)
nacelle_apply_lto_pgo(bench-worker-nacelle)
set_target_properties(bench-worker-nacelle PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_include_directories(bench-worker-nacelle PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    "${PROJECT_SOURCE_DIR}/third_party/thread-pool/include"
)
target_compile_definitions(bench-worker-nacelle PRIVATE
    NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
)
target_link_libraries(bench-worker-nacelle PRIVATE nacelle)

# Worker: Vulkan (isolated from Nacelle/LLVM to avoid ICD libLLVM symbol collisions)
add_executable(bench-worker-vulkan backends/vulkan/worker.cpp)
set_target_properties(bench-worker-vulkan PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_include_directories(bench-worker-vulkan PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(bench-worker-vulkan PRIVATE
    NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
)
if(UNIX AND NOT APPLE)
    target_link_libraries(bench-worker-vulkan PRIVATE dl)
endif()

set(NACELLE_BENCH_WORKERS bench-worker-nacelle bench-worker-vulkan)

# Worker: Metal / IRConverter (macOS)
if(APPLE)
    add_executable(bench-worker-metal
        backends/metal/worker.cpp
        backends/metal/metal_bench.mm
    )
    set_source_files_properties(backends/metal/metal_bench.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    set_target_properties(bench-worker-metal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    find_path(METAL_IRCONVERTER_INC metal_irconverter/metal_irconverter.h HINTS /usr/local/include /opt/homebrew/include)
    find_library(METAL_IRCONVERTER_LIB metalirconverter HINTS /usr/local/lib /opt/homebrew/lib)
    target_include_directories(bench-worker-metal PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_compile_definitions(bench-worker-metal PRIVATE
        NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
    )
    target_link_libraries(bench-worker-metal PRIVATE "-framework Metal" "-framework Foundation")
    if(METAL_IRCONVERTER_INC AND METAL_IRCONVERTER_LIB)
        target_compile_definitions(bench-worker-metal PRIVATE NACELLE_HAS_METAL_IRCONVERTER=1)
        target_include_directories(bench-worker-metal PRIVATE "${METAL_IRCONVERTER_INC}")
        target_link_libraries(bench-worker-metal PRIVATE "${METAL_IRCONVERTER_LIB}")
        get_filename_component(METAL_IRCONVERTER_LIBDIR "${METAL_IRCONVERTER_LIB}" DIRECTORY)
        set_target_properties(bench-worker-metal PROPERTIES BUILD_RPATH "${METAL_IRCONVERTER_LIBDIR}")
    endif()
    list(APPEND NACELLE_BENCH_WORKERS bench-worker-metal)
endif()

# Worker: DirectX 12 / WARP (Windows)
if(WIN32)
    add_executable(bench-worker-d3d12 backends/d3d12/worker.cpp)
    set_target_properties(bench-worker-d3d12 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
    target_include_directories(bench-worker-d3d12 PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${PROJECT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    )
    target_compile_definitions(bench-worker-d3d12 PRIVATE
        NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
    )
    if(NACELLE_D3D12_SDK_VERSION GREATER 0)
        target_compile_definitions(bench-worker-d3d12 PRIVATE
            NACELLE_D3D12_SDK_VERSION=${NACELLE_D3D12_SDK_VERSION}
        )
    endif()
    list(APPEND NACELLE_BENCH_WORKERS bench-worker-d3d12)
endif()

# Harness: nacelle-bench (pure orchestrator; spawns per-backend worker binaries)
add_executable(nacelle-bench main.cpp)
add_dependencies(nacelle-bench nacelle_test_shaders ${NACELLE_BENCH_WORKERS})
set_target_properties(nacelle-bench PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_include_directories(nacelle-bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
)
target_compile_definitions(nacelle-bench PRIVATE
    NACELLE_SHADER_BIN_DIR="${CMAKE_BINARY_DIR}/tests/shaders"
)
