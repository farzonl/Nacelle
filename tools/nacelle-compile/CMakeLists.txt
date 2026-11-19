add_executable(nacelle-compile main.cpp)
nacelle_apply_lto_pgo(nacelle-compile)
set_target_properties(nacelle-compile PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
target_include_directories(nacelle-compile PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/third_party/DirectX-Headers/include"
    "${PROJECT_SOURCE_DIR}/third_party/thread-pool/include"
)
if(NOT WIN32)
    target_include_directories(nacelle-compile PRIVATE "${PROJECT_SOURCE_DIR}/src/common/posix_sal")
endif()
target_include_directories(nacelle-compile SYSTEM PRIVATE ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
target_compile_definitions(nacelle-compile PRIVATE ${LLVM_DEFS_LIST})
target_link_libraries(nacelle-compile PRIVATE nacelle)
