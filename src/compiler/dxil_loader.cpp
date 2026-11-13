#include "dxil_loader.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <optional>
#include <unordered_map>

#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/ReplaceConstant.h>
#include <llvm/Object/DXContainer.h>
#include <llvm/Config/llvm-config.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Utils/DXILUpgrade.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Utils/Mem2Reg.h>

#include "../api/pipeline.h"

namespace nacelle {

namespace {

std::once_flag gHostInfoOnce;
std::string gHostDataLayoutStr;
std::string gHostTripleStr;

void initHostInfo() {
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    auto jtmbOrErr = llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!jtmbOrErr) {
        llvm::consumeError(jtmbOrErr.takeError());
        gHostDataLayoutStr = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32";
        return;
    }
    gHostTripleStr = jtmbOrErr->getTargetTriple().str();
    auto tmOrErr = jtmbOrErr->createTargetMachine();
    if (!tmOrErr) {
        llvm::consumeError(tmOrErr.takeError());
        return;
    }
    gHostDataLayoutStr = (*tmOrErr)->createDataLayout().getStringRepresentation();
}

uint32_t mdAsUInt(const llvm::Metadata* md, uint32_t fallback = 0) {
    if (!md) return fallback;
    if (auto* cmd = llvm::dyn_cast<llvm::ConstantAsMetadata>(md)) {
        if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(cmd->getValue())) {
            return static_cast<uint32_t>(ci->getZExtValue());
        }
    }
    return fallback;
}

std::string mdAsString(const llvm::Metadata* md) {
    if (auto* s = llvm::dyn_cast_or_null<llvm::MDString>(md)) {
        return s->getString().str();
    }
    return {};
}

void parseSignatureList(const llvm::MDNode* listNode, std::vector<DxilSigRecord>& out) {
    if (!listNode) return;
    for (unsigned i = 0; i < listNode->getNumOperands(); ++i) {
        auto* elem = llvm::dyn_cast_or_null<llvm::MDNode>(listNode->getOperand(i).get());
        if (!elem || elem->getNumOperands() < 10) continue;
        DxilSigRecord rec;
        rec.sigId = mdAsUInt(elem->getOperand(0).get());
        rec.semanticName = mdAsString(elem->getOperand(1).get());
        rec.componentType = mdAsUInt(elem->getOperand(2).get(), 9);
        rec.systemValueType = mdAsUInt(elem->getOperand(3).get(), 0);
        if (auto* idxVec = llvm::dyn_cast_or_null<llvm::MDNode>(elem->getOperand(4).get())) {
            if (idxVec->getNumOperands() > 0) {
                rec.semanticIndex = mdAsUInt(idxVec->getOperand(0).get(), 0);
            }
        }
        rec.interpMode = mdAsUInt(elem->getOperand(5).get(), 2);
        rec.rows = mdAsUInt(elem->getOperand(6).get(), 1);
        rec.cols = mdAsUInt(elem->getOperand(7).get(), 4);
        rec.startRow = mdAsUInt(elem->getOperand(8).get(), 0);
        rec.startCol = mdAsUInt(elem->getOperand(9).get(), 0);
        out.push_back(std::move(rec));
    }
}

void parseResourcesNode(const llvm::MDNode* resRoot, std::vector<DxilResourceRecord>& out) {
    if (!resRoot) return;
    for (uint32_t resClass = 0; resClass < 4 && resClass < resRoot->getNumOperands(); ++resClass) {
        auto* list = llvm::dyn_cast_or_null<llvm::MDNode>(resRoot->getOperand(resClass).get());
        if (!list) continue;
        for (unsigned i = 0; i < list->getNumOperands(); ++i) {
            auto* recMd = llvm::dyn_cast_or_null<llvm::MDNode>(list->getOperand(i).get());
            if (!recMd || recMd->getNumOperands() < 7) continue;
            DxilResourceRecord r;
            r.resClass = resClass;
            r.rangeId = mdAsUInt(recMd->getOperand(0).get());
            r.space = mdAsUInt(recMd->getOperand(3).get());
            r.lowerBound = mdAsUInt(recMd->getOperand(4).get());
            r.rangeSize = mdAsUInt(recMd->getOperand(5).get(), 1);
            r.resourceKind = mdAsUInt(recMd->getOperand(6).get());
            if (resClass == 2) {
                // CBV: operand 6 is byte size
                r.structuredStride = r.resourceKind;
                r.resourceKind = 13;
            } else if (resClass == 3) {
                r.resourceKind = 14;
            } else if (r.resourceKind == 12) {
                // StructuredBuffer: properties tuple contains !{i32 1, i32 stride}
                unsigned tagIdx = (resClass == 0) ? 8 : 10;
                if (recMd->getNumOperands() > tagIdx) {
                    if (auto* props = llvm::dyn_cast_or_null<llvm::MDNode>(recMd->getOperand(tagIdx).get())) {
                        if (props->getNumOperands() >= 2 && mdAsUInt(props->getOperand(0).get()) == 1) {
                            r.structuredStride = mdAsUInt(props->getOperand(1).get(), 0);
                        }
                    }
                }
            }
            out.push_back(r);
        }
    }
}

void extractDxilMetadata(llvm::Module& mod, DxilLoadedShader& out) {
    if (auto* sm = mod.getNamedMetadata("dx.shaderModel")) {
        if (sm->getNumOperands() > 0) {
            auto* n = sm->getOperand(0);
            if (n->getNumOperands() >= 3) {
                out.stage = mdAsString(n->getOperand(0).get());
                out.smMajor = mdAsUInt(n->getOperand(1).get(), 6);
                out.smMinor = mdAsUInt(n->getOperand(2).get(), 0);
            }
        }
    }

    if (auto* res = mod.getNamedMetadata("dx.resources")) {
        if (res->getNumOperands() > 0) {
            parseResourcesNode(res->getOperand(0), out.resources);
        }
    }

    if (auto* eps = mod.getNamedMetadata("dx.entryPoints")) {
        for (unsigned e = 0; e < eps->getNumOperands(); ++e) {
            auto* ep = eps->getOperand(e);
            if (!ep || ep->getNumOperands() < 5) continue;
            if (auto* fnVal = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(ep->getOperand(0).get())) {
                if (auto* fn = llvm::dyn_cast<llvm::Function>(fnVal->getValue())) {
                    out.entryName = fn->getName().str();
                }
            } else if (out.entryName.empty()) {
                out.entryName = mdAsString(ep->getOperand(1).get());
            }
            if (auto* sigTuple = llvm::dyn_cast_or_null<llvm::MDNode>(ep->getOperand(2).get())) {
                if (sigTuple->getNumOperands() >= 1) {
                    parseSignatureList(llvm::dyn_cast_or_null<llvm::MDNode>(sigTuple->getOperand(0).get()), out.inputSigs);
                }
                if (sigTuple->getNumOperands() >= 2) {
                    parseSignatureList(llvm::dyn_cast_or_null<llvm::MDNode>(sigTuple->getOperand(1).get()), out.outputSigs);
                }
                if (sigTuple->getNumOperands() >= 3) {
                    parseSignatureList(llvm::dyn_cast_or_null<llvm::MDNode>(sigTuple->getOperand(2).get()), out.patchConstSigs);
                }
            }
            if (out.resources.empty()) {
                if (auto* epRes = llvm::dyn_cast_or_null<llvm::MDNode>(ep->getOperand(3).get())) {
                    parseResourcesNode(epRes, out.resources);
                }
            }
            if (auto* props = llvm::dyn_cast_or_null<llvm::MDNode>(ep->getOperand(4).get())) {
                for (unsigned p = 0; p + 1 < props->getNumOperands(); p += 2) {
                    uint32_t tag = mdAsUInt(props->getOperand(p).get());
                    if (tag == 1) { // kDxilGSStateTag
                        if (auto* gs = llvm::dyn_cast_or_null<llvm::MDNode>(props->getOperand(p + 1).get())) {
                            if (gs->getNumOperands() >= 5) {
                                out.gsInputPrimitive = mdAsUInt(gs->getOperand(0).get(), 0);
                                out.gsMaxVertexCount = mdAsUInt(gs->getOperand(1).get(), 0);
                                out.gsActiveStreamMask = mdAsUInt(gs->getOperand(2).get(), 1);
                                out.gsOutputTopology = mdAsUInt(gs->getOperand(3).get(), 5);
                                out.gsInstanceCount = std::max(1u, mdAsUInt(gs->getOperand(4).get(), 1));
                            }
                        }
                    } else if (tag == 2) { // kDxilDSStateTag
                        if (auto* ds = llvm::dyn_cast_or_null<llvm::MDNode>(props->getOperand(p + 1).get())) {
                            if (ds->getNumOperands() >= 2) {
                                out.tessDomain = mdAsUInt(ds->getOperand(0).get(), 0);
                                out.dsInputControlPointCount = mdAsUInt(ds->getOperand(1).get(), 0);
                            }
                        }
                    } else if (tag == 3) { // kDxilHSStateTag
                        if (auto* hs = llvm::dyn_cast_or_null<llvm::MDNode>(props->getOperand(p + 1).get())) {
                            if (hs->getNumOperands() >= 7) {
                                if (auto* pcFnVal = llvm::dyn_cast_or_null<llvm::ValueAsMetadata>(hs->getOperand(0).get())) {
                                    if (auto* pcFn = llvm::dyn_cast<llvm::Function>(pcFnVal->getValue())) {
                                        out.hsPatchConstFnName = pcFn->getName().str();
                                    }
                                }
                                out.hsInputControlPointCount = mdAsUInt(hs->getOperand(1).get(), 0);
                                out.hsOutputControlPointCount = mdAsUInt(hs->getOperand(2).get(), 0);
                                out.tessDomain = mdAsUInt(hs->getOperand(3).get(), 0);
                                out.tessPartitioning = mdAsUInt(hs->getOperand(4).get(), 0);
                                out.tessOutputPrimitive = mdAsUInt(hs->getOperand(5).get(), 0);
                                if (auto* c = llvm::dyn_cast_or_null<llvm::ConstantAsMetadata>(hs->getOperand(6).get())) {
                                    if (auto* cf = llvm::dyn_cast<llvm::ConstantFP>(c->getValue())) {
                                        out.hsMaxTessFactor = cf->getValueAPF().convertToFloat();
                                    }
                                }
                            }
                        }
                    } else if (tag == 4) { // kDxilNumThreadsTag
                        if (auto* tgs = llvm::dyn_cast_or_null<llvm::MDNode>(props->getOperand(p + 1).get())) {
                            for (unsigned k = 0; k < 3 && k < tgs->getNumOperands(); ++k) {
                                out.threadGroupSize[k] = std::max(1u, mdAsUInt(tgs->getOperand(k).get(), 1));
                            }
                        }
                    }
                }
            }
            if (!out.entryName.empty()) break;
        }
    }

    if (out.entryName.empty()) {
        for (const auto& f : mod) {
            if (!f.isDeclaration()) {
                out.entryName = f.getName().str();
                break;
            }
        }
    }
}

/// Lowers addrspace(3) groupshared globals to flat offsets accessed via `@nacelle.groupshared.ptr(i32 offset)`.
uint32_t lowerGroupSharedGlobals(llvm::Module& mod, llvm::Function* entryFn) {
    llvm::SmallVector<llvm::GlobalVariable*, 8> gsGlobals;
    for (auto& gv : mod.globals()) {
        if (gv.getAddressSpace() == 3) {
            gsGlobals.push_back(&gv);
        }
    }
    if (gsGlobals.empty() || !entryFn || entryFn->isDeclaration()) return 0;

    llvm::SmallVector<llvm::Constant*, 8> gsConsts;
    for (auto* gv : gsGlobals) gsConsts.push_back(gv);
    llvm::convertUsersOfConstantsToInstructions(gsConsts, nullptr, true, false);

    const llvm::DataLayout& dl = mod.getDataLayout();
    llvm::LLVMContext& ctx = mod.getContext();
    llvm::PointerType* ptrTy = llvm::PointerType::get(ctx, 0);
    llvm::FunctionCallee helperFn = mod.getOrInsertFunction(
        "nacelle.groupshared.ptr",
        llvm::FunctionType::get(ptrTy, {llvm::Type::getInt32Ty(ctx)}, false));

    uint32_t totalBytes = 0;
    llvm::IRBuilder<> entryBuilder(&entryFn->getEntryBlock(), entryFn->getEntryBlock().getFirstInsertionPt());
    std::unordered_map<llvm::Value*, llvm::Value*> rewrittenPtrs;

    for (auto* gv : gsGlobals) {
        uint64_t typeSize = dl.getTypeAllocSize(gv->getValueType());
        uint32_t align = std::max<uint32_t>(16, gv->getAlign().valueOrOne().value());
        totalBytes = (totalBytes + align - 1) & ~(align - 1);
        llvm::Value* basePtr = entryBuilder.CreateCall(
            helperFn, {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), totalBytes)},
            gv->getName() + ".gs_ptr");
        rewrittenPtrs[gv] = basePtr;
        totalBytes += static_cast<uint32_t>(typeSize);
    }

    // Rewrite pointer-producing instructions that operate on addrspace(3) to addrspace(0)
    // in topological (basic-block/instruction) order. Repeated until fixed point for cyclic loop PHIs.
    auto mapVal = [&](llvm::Value* v) -> llvm::Value* {
        auto it = rewrittenPtrs.find(v);
        return (it != rewrittenPtrs.end()) ? it->second : v;
    };

    llvm::SmallVector<llvm::Instruction*, 32> toErase;
    for (auto& bb : *entryFn) {
        for (auto& inst : bb) {
            if (auto* phi = llvm::dyn_cast<llvm::PHINode>(&inst)) {
                if (phi->getType()->isPointerTy() && phi->getType()->getPointerAddressSpace() == 3) {
                    auto* newPhi = llvm::PHINode::Create(ptrTy, phi->getNumIncomingValues(), phi->getName() + ".gs", phi->getIterator());
                    rewrittenPtrs[phi] = newPhi;
                    toErase.push_back(phi);
                }
            }
        }
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (auto& bb : *entryFn) {
            for (auto& inst : bb) {
                if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&inst)) {
                    if (gep->getType()->isPointerTy() && gep->getType()->getPointerAddressSpace() == 3 && !rewrittenPtrs.count(gep)) {
                        llvm::Value* base = mapVal(gep->getPointerOperand());
                        if (base->getType() == ptrTy) {
                            llvm::SmallVector<llvm::Value*, 4> idx(gep->indices());
                            llvm::IRBuilder<> b(gep);
                            auto* newGep = b.CreateGEP(gep->getSourceElementType(), base, idx, gep->getName() + ".gs");
                            if (auto* gInst = llvm::dyn_cast<llvm::GetElementPtrInst>(newGep)) {
                                gInst->setIsInBounds(gep->isInBounds());
                            }
                            rewrittenPtrs[gep] = newGep;
                            toErase.push_back(gep);
                            changed = true;
                        }
                    }
                } else if (auto* asc = llvm::dyn_cast<llvm::AddrSpaceCastInst>(&inst)) {
                    if (asc->getSrcAddressSpace() == 3 && !rewrittenPtrs.count(asc)) {
                        llvm::Value* base = mapVal(asc->getPointerOperand());
                        if (base->getType() == asc->getType()) {
                            rewrittenPtrs[asc] = base;
                            toErase.push_back(asc);
                            changed = true;
                        }
                    }
                } else if (auto* sel = llvm::dyn_cast<llvm::SelectInst>(&inst)) {
                    if (sel->getType()->isPointerTy() && sel->getType()->getPointerAddressSpace() == 3 && !rewrittenPtrs.count(sel)) {
                        llvm::Value* tv = mapVal(sel->getTrueValue());
                        llvm::Value* fv = mapVal(sel->getFalseValue());
                        if (tv->getType() == ptrTy && fv->getType() == ptrTy) {
                            llvm::IRBuilder<> b(sel);
                            rewrittenPtrs[sel] = b.CreateSelect(sel->getCondition(), tv, fv, sel->getName() + ".gs");
                            toErase.push_back(sel);
                            changed = true;
                        }
                    }
                } else if (auto* bc = llvm::dyn_cast<llvm::BitCastInst>(&inst)) {
                    if (bc->getType()->isPointerTy() && bc->getType()->getPointerAddressSpace() == 3 && !rewrittenPtrs.count(bc)) {
                        llvm::Value* base = mapVal(bc->getOperand(0));
                        if (base->getType() == ptrTy) {
                            rewrittenPtrs[bc] = base;
                            toErase.push_back(bc);
                            changed = true;
                        }
                    }
                }
            }
        }
    }

    for (auto& bb : *entryFn) {
        for (auto& inst : bb) {
            if (auto* phi = llvm::dyn_cast<llvm::PHINode>(&inst)) {
                auto it = rewrittenPtrs.find(phi);
                if (it != rewrittenPtrs.end()) {
                    auto* newPhi = llvm::cast<llvm::PHINode>(it->second);
                    for (unsigned k = 0; k < phi->getNumIncomingValues(); ++k) {
                        newPhi->addIncoming(mapVal(phi->getIncomingValue(k)), phi->getIncomingBlock(k));
                    }
                }
            } else {
                for (unsigned op = 0; op < inst.getNumOperands(); ++op) {
                    llvm::Value* cur = inst.getOperand(op);
                    llvm::Value* mapped = mapVal(cur);
                    if (mapped != cur && mapped->getType() == cur->getType()) {
                        inst.setOperand(op, mapped);
                    } else if (mapped != cur) {
                        // Load/Store/Atomic directly consuming addrspace(3) pointer
                        if (auto* li = llvm::dyn_cast<llvm::LoadInst>(&inst); li && op == 0) {
                            li->setOperand(0, mapped);
                        } else if (auto* si = llvm::dyn_cast<llvm::StoreInst>(&inst); si && op == 1) {
                            si->setOperand(1, mapped);
                        } else if (auto* rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&inst); rmw && op == 0) {
                            rmw->setOperand(0, mapped);
                        } else if (auto* cx = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&inst); cx && op == 0) {
                            cx->setOperand(0, mapped);
                        }
                    }
                }
            }
        }
    }

    // Also replace any remaining direct uses in addrspace(0) casts
    for (auto* inst : toErase) {
        auto it = rewrittenPtrs.find(inst);
        if (it != rewrittenPtrs.end() && it->second->getType() == inst->getType()) {
            inst->replaceAllUsesWith(it->second);
        }
    }
    for (auto it = toErase.rbegin(); it != toErase.rend(); ++it) {
        if ((*it)->use_empty()) (*it)->eraseFromParent();
    }
    for (auto* gv : gsGlobals) {
        if (gv->use_empty()) gv->eraseFromParent();
    }

    return (totalBytes + 63) & ~63u;
}

/// Unifies all `ret` terminators in `fn` into a single exit block so ControlFlowToSCF lifts cleanly.
void unifyExitBlocks(llvm::Function& fn) {
    llvm::SmallVector<llvm::ReturnInst*, 4> rets;
    for (auto& bb : fn) {
        if (auto* ri = llvm::dyn_cast<llvm::ReturnInst>(bb.getTerminator())) {
            rets.push_back(ri);
        }
    }
    if (rets.size() <= 1) return;

    auto* exitBB = llvm::BasicBlock::Create(fn.getContext(), "nacelle.unified.exit", &fn);
    llvm::Type* retTy = fn.getReturnType();
    llvm::PHINode* retPhi = nullptr;
    if (!retTy->isVoidTy()) {
        retPhi = llvm::PHINode::Create(retTy, static_cast<unsigned>(rets.size()), "ret.val", exitBB);
        llvm::ReturnInst::Create(fn.getContext(), retPhi, exitBB);
    } else {
        llvm::ReturnInst::Create(fn.getContext(), nullptr, exitBB);
    }

    for (auto* ri : rets) {
        llvm::BasicBlock* srcBB = ri->getParent();
        if (retPhi) retPhi->addIncoming(ri->getReturnValue(), srcBB);
        llvm::UncondBrInst::Create(exitBB, srcBB);
        ri->eraseFromParent();
    }
}

void scanDxilFlags(llvm::Function& fn, DxilLoadedShader& out) {
    for (auto& bb : fn) {
        for (auto& inst : bb) {
            auto* ci = llvm::dyn_cast<llvm::CallInst>(&inst);
            if (!ci) continue;
            llvm::Function* callee = ci->getCalledFunction();
            if (!callee || !callee->getName().starts_with("dx.op.")) continue;
            if (ci->arg_size() < 1) continue;
            auto* opConst = llvm::dyn_cast<llvm::ConstantInt>(ci->getArgOperand(0));
            if (!opConst) continue;
            uint32_t op = static_cast<uint32_t>(opConst->getZExtValue());
            switch (op) {
            case 80: // Barrier
                out.hasBarrier = true;
                break;
            case 60: // Sample
            case 61: // SampleBias
            case 64: // SampleCmp
            case 81: // CalculateLevelOfDetail
            case 83: // DerivCoarseX
            case 84: // DerivCoarseY
            case 85: // DerivFineX
            case 86: // DerivFineY
                out.hasDerivativesOrImplicitSample = true;
                break;
            case 82: // Discard
                out.usesDiscard = true;
                break;
            case 67:  // TextureStore
            case 69:  // BufferStore
            case 70:  // BufferUpdateCounter
            case 78:  // AtomicBinOp
            case 79:  // AtomicCompareExchange
            case 140: // RawBufferStore
                out.hasSideEffects = true;
                break;
            case 5: { // StoreOutput
                if (ci->arg_size() >= 2) {
                    uint32_t sigId = mdAsUInt(nullptr);
                    if (auto* idc = llvm::dyn_cast<llvm::ConstantInt>(ci->getArgOperand(1))) {
                        sigId = static_cast<uint32_t>(idc->getZExtValue());
                        if (const auto* sig = out.findOutputSig(sigId)) {
                            if (sig->systemValueType >= 17 && sig->systemValueType <= 19) {
                                out.writesDepth = true;
                            }
                        }
                    }
                }
                break;
            }
            default:
                break;
            }
        }
    }
}

} // namespace

const DxilResourceRecord* DxilLoadedShader::findResource(uint32_t resClass, uint32_t rangeId) const {
    for (const auto& r : resources) {
        if (r.resClass == resClass && r.rangeId == rangeId) return &r;
    }
    return nullptr;
}

const DxilSigRecord* DxilLoadedShader::findInputSig(uint32_t sigId) const {
    for (const auto& s : inputSigs) {
        if (s.sigId == sigId) return &s;
    }
    return nullptr;
}

const DxilSigRecord* DxilLoadedShader::findOutputSig(uint32_t sigId) const {
    for (const auto& s : outputSigs) {
        if (s.sigId == sigId) return &s;
    }
    return nullptr;
}

const DxilSigRecord* DxilLoadedShader::findPatchConstSig(uint32_t sigId) const {
    for (const auto& s : patchConstSigs) {
        if (s.sigId == sigId) return &s;
    }
    return nullptr;
}

const std::string& getHostDataLayoutString() {
    std::call_once(gHostInfoOnce, initHostInfo);
    return gHostDataLayoutStr;
}

const std::string& getHostTargetTripleString() {
    std::call_once(gHostInfoOnce, initHostInfo);
    return gHostTripleStr;
}

bool loadAndUpgradeDXIL(
    const D3D12_SHADER_BYTECODE& bytecode,
    llvm::LLVMContext& llvmCtx,
    DxilLoadedShader& outShader,
    std::string& outError) {
    if (!bytecode.pShaderBytecode || bytecode.BytecodeLength < 4) {
        outError = "Empty shader bytecode";
        return false;
    }

    const std::string& hostDL = getHostDataLayoutString();
    const char* rawBytes = static_cast<const char*>(bytecode.pShaderBytecode);
    size_t rawLen = bytecode.BytecodeLength;

    llvm::StringRef bitcodeSlice;
    if (rawLen >= 4 && std::memcmp(rawBytes, "DXBC", 4) == 0) {
        llvm::MemoryBufferRef bufRef(llvm::StringRef(rawBytes, rawLen), "dxbc");
        auto containerOrErr = llvm::object::DXContainer::create(bufRef);
        if (!containerOrErr) {
            outError = llvm::toString(containerOrErr.takeError());
            return false;
        }
        auto dxilOpt = containerOrErr->getDXIL(false);
        if (!dxilOpt.has_value()) {
            outError = "DXContainer missing DXIL program part";
            return false;
        }
        bitcodeSlice = llvm::StringRef(dxilOpt->second, dxilOpt->first.Bitcode.Size);
        if (auto hashOpt = containerOrErr->getShaderHash()) {
            std::memcpy(outShader.shaderHash, hashOpt->Digest, 16);
        }
        if (containerOrErr->getRootSignature().has_value()) {
            outShader.hasEmbeddedRootSig =
                SUCCEEDED(NacelleRootSignature::parseBlob(rawBytes, rawLen, outShader.embeddedRootSig));
        }
    } else {
        // Raw LLVM/DXIL bitcode blob
        bitcodeSlice = llvm::StringRef(rawBytes, rawLen);
    }

    llvm::ParserCallbacks callbacks;
    callbacks.DataLayout = [&hostDL](llvm::StringRef, llvm::StringRef) -> std::optional<std::string> {
        return hostDL;
    };

    auto modOrErr = llvm::parseBitcodeFile(llvm::MemoryBufferRef(bitcodeSlice, "dxil.bc"), llvmCtx, callbacks);
    if (!modOrErr) {
        outError = llvm::toString(modOrErr.takeError());
        return false;
    }

    std::unique_ptr<llvm::Module> mod = std::move(*modOrErr);
    mod->setTargetTriple(llvm::Triple(getHostTargetTripleString()));

    extractDxilMetadata(*mod, outShader);

    // Strip target-specific DXIL metadata/attributes that interfere with LLVM/MLIR import, while
    // running DXILUpgradePass first to convert legacy DXIL constructs and typed pointers.
#if LLVM_VERSION_MAJOR >= 24
    llvm::PassBuilder pb(llvmCtx);
#else
    llvm::PassBuilder pb;
#endif
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);

    // Force internal helper functions inline into entry (keep hull patch-constant entry intact)
    for (auto& fn : *mod) {
        if (!fn.isDeclaration() && fn.getName() != outShader.entryName &&
            (outShader.hsPatchConstFnName.empty() || fn.getName() != outShader.hsPatchConstFnName)) {
            fn.setLinkage(llvm::GlobalValue::InternalLinkage);
            fn.addFnAttr(llvm::Attribute::AlwaysInline);
        }
    }

    {
        llvm::ModulePassManager mpm;
        mpm.addPass(llvm::DXILUpgradePass());
        mpm.addPass(llvm::AlwaysInlinerPass());
        mpm.run(*mod, mam);
    }

    llvm::Function* entryFn = mod->getFunction(outShader.entryName);
    if (!entryFn) {
        outError = "Entry function not found in DXIL module: " + outShader.entryName;
        return false;
    }
    llvm::Function* pcFn = outShader.hsPatchConstFnName.empty()
                               ? nullptr
                               : mod->getFunction(outShader.hsPatchConstFnName);

    // Erase non-entry helper definitions after inlining
    llvm::SmallVector<llvm::Function*, 8> deadFns;
    for (auto& fn : *mod) {
        if (!fn.isDeclaration() && &fn != entryFn && &fn != pcFn && fn.use_empty()) {
            deadFns.push_back(&fn);
        }
    }
    for (auto* df : deadFns) df->eraseFromParent();

    outShader.groupSharedBytes = lowerGroupSharedGlobals(*mod, entryFn);
    unifyExitBlocks(*entryFn);
    scanDxilFlags(*entryFn, outShader);
    if (pcFn && !pcFn->isDeclaration()) {
        unifyExitBlocks(*pcFn);
        scanDxilFlags(*pcFn, outShader);
    }

    // Promote allocas and clean CFG (removing switch ops) before MLIR import
    fam.clear();
    {
        llvm::FunctionPassManager fpm;
        fpm.addPass(llvm::PromotePass());
        fpm.addPass(llvm::SROAPass(llvm::SROAOptions::ModifyCFG));
        fpm.addPass(llvm::LowerSwitchPass());
        fpm.addPass(llvm::SimplifyCFGPass());
        fpm.run(*entryFn, fam);
        if (pcFn && !pcFn->isDeclaration()) {
            fpm.run(*pcFn, fam);
        }
    }
    unifyExitBlocks(*entryFn);
    if (pcFn && !pcFn->isDeclaration()) {
        unifyExitBlocks(*pcFn);
    }

    // Remove DXIL-specific named metadata (already extracted into DxilLoadedShader)
    llvm::SmallVector<llvm::NamedMDNode*, 8> mdToDrop;
    for (auto& nmd : mod->named_metadata()) {
        if (nmd.getName().starts_with("dx.")) mdToDrop.push_back(&nmd);
    }
    for (auto* nmd : mdToDrop) mod->eraseNamedMetadata(nmd);

    outShader.module = std::move(mod);
    return true;
}

} // namespace nacelle
