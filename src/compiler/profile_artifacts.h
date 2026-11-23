#pragma once
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include "shader_compiler.h"
namespace llvm { class Module; }
namespace mlir { class ModuleOp; }
namespace nacelle {
void retainProfileShader(const D3D12_SHADER_BYTECODE& bytecode, const std::string& identity);
void retainProfileObject(const void* bytes, size_t size, const std::string& identity);
void retainProfileLLVM(llvm::Module& module, const std::string& identity);
void retainProfileMLIR(mlir::ModuleOp module, const std::string& identity);
}
#endif
