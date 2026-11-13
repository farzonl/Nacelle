#pragma once

#include <memory>
#include <string>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

namespace llvm {
class Module;
}

namespace nacelle {

/// Registers all dialects and LLVM-IR import/export interfaces needed by Nacelle on `ctx`.
void configureNacelleMLIRContext(mlir::MLIRContext& ctx);

/// Translates an upgraded DXIL LLVM IR module to MLIR and lifts entry CFG to structured `scf`.
mlir::OwningOpRef<mlir::ModuleOp> importAndLiftDXILToSCF(
    std::unique_ptr<llvm::Module> llvmModule,
    const std::string& entryName,
    mlir::MLIRContext& mlirCtx,
    std::string& outError,
    const std::string& extraEntryName = "");

} // namespace nacelle
