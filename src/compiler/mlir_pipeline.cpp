#include "mlir_pipeline.h"

#include <llvm/IR/Module.h>
#include <mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/DLTI/DLTI.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Import.h>
#include <mlir/Transforms/Passes.h>

namespace nacelle {

void configureNacelleMLIRContext(mlir::MLIRContext& ctx) {
    mlir::DialectRegistry registry;
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectImport(registry);
    mlir::registerLLVMDialectTranslation(registry);
    registry.insert<
        mlir::DLTIDialect,
        mlir::func::FuncDialect,
        mlir::cf::ControlFlowDialect,
        mlir::scf::SCFDialect,
        mlir::arith::ArithDialect,
        mlir::math::MathDialect,
        mlir::vector::VectorDialect,
        mlir::ub::UBDialect,
        mlir::LLVM::LLVMDialect>();
    ctx.appendDialectRegistry(registry);
    ctx.loadAllAvailableDialects();
}

static bool convertLLVMFuncShellToCF(mlir::ModuleOp mod, const std::string& entryName, std::string& outError) {
    auto llvmFunc = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>(entryName);
    if (!llvmFunc || llvmFunc.isDeclaration()) {
        outError = "Entry LLVMFuncOp not found after MLIR import: " + entryName;
        return false;
    }

    mlir::OpBuilder b(llvmFunc);
    auto llvmFnTy = llvmFunc.getFunctionType();
    llvm::SmallVector<mlir::Type, 4> resTys;
    if (!llvm::isa<mlir::LLVM::LLVMVoidType>(llvmFnTy.getReturnType())) {
        resTys.push_back(llvmFnTy.getReturnType());
    }
    auto fnTy = b.getFunctionType(llvmFnTy.getParams(), resTys);

    auto funcOp = mlir::func::FuncOp::create(b, llvmFunc.getLoc(), llvmFunc.getName(), fnTy);
    funcOp.getBody().takeBody(llvmFunc.getBody());
    llvmFunc.erase();

    // Rewrite LLVM branch/return terminators in `funcOp` to `cf.*` / `func.return` so
    // `createLiftControlFlowToSCFPass()` can lift the entire shader CFG.
    llvm::SmallVector<mlir::Block*, 16> blocks;
    for (mlir::Block& bb : funcOp.getBody()) blocks.push_back(&bb);

    for (mlir::Block* bb : blocks) {
        if (bb->empty()) continue;
        mlir::Operation* term = bb->getTerminator();
        if (!term) continue;
        mlir::OpBuilder tb(term);
        mlir::Location loc = term->getLoc();

        if (auto br = llvm::dyn_cast<mlir::LLVM::BrOp>(term)) {
            mlir::cf::BranchOp::create(tb, loc, br.getDest(), br.getDestOperands());
            br.erase();
        } else if (auto cbr = llvm::dyn_cast<mlir::LLVM::CondBrOp>(term)) {
            mlir::cf::CondBranchOp::create(
                tb, loc, cbr.getCondition(),
                cbr.getTrueDest(), cbr.getTrueDestOperands(),
                cbr.getFalseDest(), cbr.getFalseDestOperands());
            cbr.erase();
        } else if (auto sw = llvm::dyn_cast<mlir::LLVM::SwitchOp>(term)) {
            auto caseVals = sw.getCaseValues();
            auto caseDests = sw.getCaseDestinations();
            mlir::Value condVal = sw.getValue();
            if (!caseVals || caseDests.empty()) {
                mlir::cf::BranchOp::create(tb, loc, sw.getDefaultDestination(), sw.getDefaultOperands());
            } else {
                mlir::Block* curBB = bb;
                for (unsigned i = 0; i < caseDests.size(); ++i) {
                    mlir::OpBuilder cb = (i == 0) ? tb : mlir::OpBuilder::atBlockEnd(curBB);
                    auto cmpCst = mlir::arith::ConstantIntOp::create(
                        cb, loc, condVal.getType(),
                        caseVals->getValues<llvm::APInt>()[i].getZExtValue());
                    auto eq = mlir::arith::CmpIOp::create(
                        cb, loc, mlir::arith::CmpIPredicate::eq, condVal, cmpCst);
                    if (i + 1 == caseDests.size()) {
                        mlir::cf::CondBranchOp::create(
                            cb, loc, eq,
                            caseDests[i], sw.getCaseOperands(i),
                            sw.getDefaultDestination(), sw.getDefaultOperands());
                    } else {
                        mlir::Block* nextBB = new mlir::Block();
                        funcOp.getBody().push_back(nextBB);
                        mlir::cf::CondBranchOp::create(
                            cb, loc, eq,
                            caseDests[i], sw.getCaseOperands(i),
                            nextBB, mlir::ValueRange{});
                        curBB = nextBB;
                    }
                }
            }
            sw.erase();
        } else if (auto ret = llvm::dyn_cast<mlir::LLVM::ReturnOp>(term)) {
            mlir::func::ReturnOp::create(tb, loc, ret.getOperands());
            ret.erase();
        } else if (auto unreach = llvm::dyn_cast<mlir::LLVM::UnreachableOp>(term)) {
            llvm::SmallVector<mlir::Value, 2> dummyRets;
            for (mlir::Type rt : resTys) {
                dummyRets.push_back(mlir::ub::PoisonOp::create(tb, loc, rt));
            }
            mlir::func::ReturnOp::create(tb, loc, dummyRets);
            unreach.erase();
        }
    }

    // `createLiftControlFlowToSCFPass()` requires at most one exit (`func.return`) block.
    llvm::SmallVector<mlir::Block*, 4> returnBlocks;
    for (mlir::Block& bb : funcOp.getBody()) {
        if (!bb.empty() && llvm::isa<mlir::func::ReturnOp>(bb.getTerminator())) {
            returnBlocks.push_back(&bb);
        }
    }
    if (returnBlocks.size() > 1) {
        mlir::Block* unifiedExit = new mlir::Block();
        funcOp.getBody().push_back(unifiedExit);
        mlir::Location exitLoc = funcOp.getLoc();
        llvm::SmallVector<mlir::Value, 4> exitArgs;
        for (mlir::Type rt : resTys) {
            exitArgs.push_back(unifiedExit->addArgument(rt, exitLoc));
        }
        for (mlir::Block* rb : returnBlocks) {
            auto retOp = llvm::cast<mlir::func::ReturnOp>(rb->getTerminator());
            mlir::OpBuilder eb(retOp);
            mlir::cf::BranchOp::create(eb, retOp.getLoc(), unifiedExit, retOp.getOperands());
            retOp.erase();
        }
        mlir::OpBuilder exitB = mlir::OpBuilder::atBlockEnd(unifiedExit);
        mlir::func::ReturnOp::create(exitB, exitLoc, exitArgs);
    }
    return true;
}

mlir::OwningOpRef<mlir::ModuleOp> importAndLiftDXILToSCF(
    std::unique_ptr<llvm::Module> llvmModule,
    const std::string& entryName,
    mlir::MLIRContext& mlirCtx,
    std::string& outError) {
    configureNacelleMLIRContext(mlirCtx);

    auto mlirMod = mlir::translateLLVMIRToModule(std::move(llvmModule), &mlirCtx);
    if (!mlirMod) {
        outError = "translateLLVMIRToModule failed";
        return nullptr;
    }

    if (!convertLLVMFuncShellToCF(*mlirMod, entryName, outError)) {
        return nullptr;
    }

    mlir::PassManager pm(&mlirCtx);
    pm.addPass(mlir::createLiftControlFlowToSCFPass());
    pm.addPass(mlir::createCanonicalizerPass());
    if (mlir::failed(pm.run(*mlirMod))) {
        outError = "MLIR ControlFlowToSCF pass failed";
        return nullptr;
    }

    if (mlir::failed(mlir::verify(*mlirMod))) {
        outError = "MLIR module verification failed after ControlFlowToSCF";
        return nullptr;
    }
    return mlirMod;
}

} // namespace nacelle
