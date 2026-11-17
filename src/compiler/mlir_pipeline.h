#pragma once

#include <memory>
#include <string>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
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

/// Attaches `llvm.intr.expect` to a scalar `i1` branch condition so LLVM emits `MD_prof` branch_weights.
inline mlir::Value emitExpectBool(
    mlir::OpBuilder& b,
    mlir::Location loc,
    mlir::Value cond,
    bool expected) {
    if (!cond || cond.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        return cond;
    }
    mlir::Type i1Ty = b.getI1Type();
    if (cond.getType() != i1Ty) {
        return cond;
    }
    mlir::Value expVal = mlir::LLVM::ConstantOp::create(b, loc, i1Ty, b.getIntegerAttr(i1Ty, expected ? 1 : 0));
    return mlir::LLVM::ExpectOp::create(b, loc, cond, expVal);
}

} // namespace nacelle
