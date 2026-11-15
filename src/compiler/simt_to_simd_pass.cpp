#include "simt_to_simd_pass.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <unordered_map>

#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>

namespace nacelle {

namespace {

static llvm::AllocaInst* createAlignedAlloca(
    llvm::IRBuilder<>& b,
    llvm::Type* elemTy,
    llvm::Value* arraySize,
    uint64_t alignBytes) {
    auto* ai = b.CreateAlloca(elemTy, arraySize);
    if (alignBytes > 0) ai->setAlignment(llvm::Align(alignBytes));
    return ai;
}

template <typename ThenFn>
static void emitIf(llvm::IRBuilder<>& b, llvm::Value* cond, ThenFn&& thenFn) {
    llvm::Function* fn = b.GetInsertBlock()->getParent();
    llvm::LLVMContext& ctx = b.getContext();
    llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(ctx, "if.then", fn);
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(ctx, "if.end", fn);
    b.CreateCondBr(cond, thenBB, mergeBB);
    b.SetInsertPoint(thenBB);
    thenFn(b);
    if (!b.GetInsertBlock()->getTerminatorOrNull()) b.CreateBr(mergeBB);
    b.SetInsertPoint(mergeBB);
}

template <typename ThenFn, typename ElseFn>
static void emitIfElse(llvm::IRBuilder<>& b, llvm::Value* cond, ThenFn&& thenFn, ElseFn&& elseFn) {
    llvm::Function* fn = b.GetInsertBlock()->getParent();
    llvm::LLVMContext& ctx = b.getContext();
    llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(ctx, "if.then", fn);
    llvm::BasicBlock* elseBB = llvm::BasicBlock::Create(ctx, "if.else", fn);
    llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(ctx, "if.end", fn);
    b.CreateCondBr(cond, thenBB, elseBB);
    b.SetInsertPoint(thenBB);
    thenFn(b);
    if (!b.GetInsertBlock()->getTerminatorOrNull()) b.CreateBr(mergeBB);
    b.SetInsertPoint(elseBB);
    elseFn(b);
    if (!b.GetInsertBlock()->getTerminatorOrNull()) b.CreateBr(mergeBB);
    b.SetInsertPoint(mergeBB);
}

template <typename BodyFn>
static void emitForLoop(
    llvm::IRBuilder<>& b,
    llvm::Value* lb,
    llvm::Value* ub,
    llvm::Value* step,
    BodyFn&& bodyFn) {
    llvm::Function* fn = b.GetInsertBlock()->getParent();
    llvm::LLVMContext& ctx = b.getContext();
    llvm::BasicBlock* preBB = b.GetInsertBlock();
    llvm::BasicBlock* hdrBB = llvm::BasicBlock::Create(ctx, "for.hdr", fn);
    llvm::BasicBlock* bodyBB = llvm::BasicBlock::Create(ctx, "for.body", fn);
    llvm::BasicBlock* exitBB = llvm::BasicBlock::Create(ctx, "for.end", fn);
    b.CreateBr(hdrBB);
    b.SetInsertPoint(hdrBB);
    llvm::PHINode* iv = b.CreatePHI(lb->getType(), 2, "for.iv");
    iv->addIncoming(lb, preBB);
    llvm::Value* cond = b.CreateICmpSLT(iv, ub);
    b.CreateCondBr(cond, bodyBB, exitBB);
    b.SetInsertPoint(bodyBB);
    bodyFn(b, iv);
    llvm::BasicBlock* latchBB = b.GetInsertBlock();
    if (!latchBB->getTerminatorOrNull()) {
        llvm::Value* nextIv = b.CreateAdd(iv, step);
        iv->addIncoming(nextIv, latchBB);
        b.CreateBr(hdrBB);
    }
    b.SetInsertPoint(exitBB);
}

static std::optional<uint64_t> getConstIntValue(llvm::Value* v) {
    if (!v) return std::nullopt;
    if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(v)) {
        return ci->getZExtValue();
    }
    if (auto* cdv = llvm::dyn_cast<llvm::ConstantDataVector>(v)) {
        if (cdv->isSplat() && cdv->getElementType()->isIntegerTy()) {
            return cdv->getElementAsInteger(0);
        }
    }
    if (auto* cv = llvm::dyn_cast<llvm::ConstantVector>(v)) {
        if (auto* ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(cv->getSplatValue())) {
            return ci->getZExtValue();
        }
    }
    return std::nullopt;
}

static bool isUndefOrPoison(llvm::Value* v) {
    return v && llvm::isa<llvm::UndefValue>(v);
}

static bool extractConstStructInts(llvm::Value* v, llvm::SmallVectorImpl<uint64_t>& out) {
    if (!v) return false;
    auto* st = llvm::dyn_cast<llvm::StructType>(v->getType());
    if (!st) return false;
    out.assign(st->getNumElements(), 0);

    if (auto* cs = llvm::dyn_cast<llvm::ConstantStruct>(v)) {
        for (unsigned i = 0; i < cs->getNumOperands() && i < out.size(); ++i) {
            if (auto cv = getConstIntValue(cs->getOperand(i))) {
                out[i] = *cv;
            }
        }
        return true;
    }
    if (llvm::isa<llvm::ConstantAggregateZero>(v) || isUndefOrPoison(v)) {
        return true;
    }

    llvm::SmallVector<std::pair<size_t, llvm::Value*>, 8> inserts;
    llvm::Value* cur = v;
    while (auto* iv = llvm::dyn_cast<llvm::InsertValueInst>(cur)) {
        auto pos = iv->getIndices();
        if (pos.size() == 1 && static_cast<size_t>(pos[0]) < out.size()) {
            inserts.push_back({static_cast<size_t>(pos[0]), iv->getInsertedValueOperand()});
        }
        cur = iv->getAggregateOperand();
    }

    bool recognized = !inserts.empty();
    if (llvm::isa<llvm::ConstantAggregateZero>(cur) || isUndefOrPoison(cur)) {
        recognized = true;
    } else if (auto* cs = llvm::dyn_cast<llvm::ConstantStruct>(cur)) {
        recognized = true;
        for (unsigned i = 0; i < cs->getNumOperands() && i < out.size(); ++i) {
            if (auto cv = getConstIntValue(cs->getOperand(i))) {
                out[i] = *cv;
            }
        }
    }

    for (auto it = inserts.rbegin(); it != inserts.rend(); ++it) {
        if (auto cv = getConstIntValue(it->second)) {
            out[it->first] = *cv;
        }
    }
    return recognized;
}

static bool isBarrierOp(llvm::Instruction* op) {
    auto* call = llvm::dyn_cast<llvm::CallInst>(op);
    if (!call) return false;
    llvm::Function* callee = call->getCalledFunction();
    if (!callee || !callee->getName().starts_with("dx.op.barrier")) return false;
    if (call->arg_size() == 0) return true;
    uint64_t opNum = getConstIntValue(call->getArgOperand(0)).value_or(80);
    if (opNum == 80 && call->arg_size() > 1) {
        uint64_t mode = getConstIntValue(call->getArgOperand(1)).value_or(1);
        return (mode & 1u) != 0; // DXIL::BarrierMode::SYNC_THREAD_GROUP
    }
    if ((opNum == 244 || opNum == 245) && call->arg_size() > 2) {
        uint64_t sem = getConstIntValue(call->getArgOperand(2)).value_or(1);
        return (sem & 1u) != 0; // DXIL::BarrierSemanticFlags::GroupSync
    }
    return true;
}

static bool nodeContainsBarrier(const RegionNode& node) {
    switch (node.kind) {
    case RegionNode::Kind::Block: {
        const auto& bn = llvm::cast<BlockNode>(node);
        for (llvm::Instruction& inst : *bn.bb) {
            if (isBarrierOp(&inst)) return true;
        }
        return false;
    }
    case RegionNode::Kind::Seq: {
        const auto& sn = llvm::cast<SeqNode>(node);
        for (const auto& child : sn.children) {
            if (nodeContainsBarrier(*child)) return true;
        }
        return false;
    }
    case RegionNode::Kind::If: {
        const auto& in = llvm::cast<IfNode>(node);
        if (in.thenRegion && nodeContainsBarrier(*in.thenRegion)) return true;
        if (in.elseRegion && nodeContainsBarrier(*in.elseRegion)) return true;
        return false;
    }
    case RegionNode::Kind::While: {
        const auto& wn = llvm::cast<WhileNode>(node);
        return wn.body && nodeContainsBarrier(*wn.body);
    }
    }
    return false;
}

static llvm::Value* llvmExtract(llvm::IRBuilder<>& builder, llvm::Value* vec, uint32_t idx) {
    return builder.CreateExtractElement(vec, builder.getInt32(idx));
}

static llvm::Value* llvmInsert(llvm::IRBuilder<>& builder, llvm::Value* vec, llvm::Value* elem, uint32_t idx) {
    return builder.CreateInsertElement(vec, elem, builder.getInt32(idx));
}

static llvm::Value* llvmSplat(llvm::IRBuilder<>& builder, llvm::Type* elemTy, llvm::Value* scalar, uint32_t W) {
    auto* vt = llvm::FixedVectorType::get(elemTy, W);
    llvm::Value* p = llvm::PoisonValue::get(vt);
    llvm::Value* ins = llvmInsert(builder, p, scalar, 0);
    llvm::SmallVector<int32_t, 16> zeros(W, 0);
    return builder.CreateShuffleVector(ins, p, zeros);
}

class SimdPacketizer {
public:
    struct BufferDescInfo {
        llvm::Value* descP = nullptr;
        llvm::Value* basePtr = nullptr;
        llvm::Value* byteSize = nullptr;
        llvm::Value* isReserved = nullptr;
        llvm::Value* okPtr = nullptr;
        llvm::Value* elemStride = nullptr;
        llvm::Value* stride64 = nullptr;
    };

    struct DescInfoCacheKey {
        llvm::Value* descPtr;
        bool isRootConstants = false;
        uint32_t stride = 0;
        bool needElemStride = false;
        bool needStride64 = false;
        bool operator==(const DescInfoCacheKey& o) const {
            return descPtr == o.descPtr && isRootConstants == o.isRootConstants && stride == o.stride &&
                   needElemStride == o.needElemStride && needStride64 == o.needStride64;
        }
    };
    using DescInfoCache = llvm::SmallVector<std::pair<DescInfoCacheKey, BufferDescInfo>, 8>;

    SimdPacketizer(
        llvm::IRBuilder<>& builder,
        llvm::Module& mod,
        const DxilLoadedShader& shader,
        const RootSignatureLayout& rootLayout,
        uint32_t simdWidth,
        llvm::Value* rootCtxPtr,
        StageHooks& hooks,
        DescInfoCache* sharedDescCache = nullptr)
        : b_(builder),
          mod_(mod),
          shader_(shader),
          rootLayout_(rootLayout),
          W_(simdWidth),
          rootCtxPtr_(rootCtxPtr),
          hooks_(hooks),
          descCache_(sharedDescCache ? sharedDescCache : &ownedDescCache_) {
        const char* strictFpEnv = std::getenv("NACELLE_STRICT_FP");
        strictFp_ = strictFpEnv && strictFpEnv[0] == '1';
        if (!strictFp_) {
            llvm::FastMathFlags fmf;
            fmf.setAllowReassoc();
            fmf.setNoSignedZeros();
            fmf.setAllowReciprocal();
            fmf.setAllowContract();
            fmf.setApproxFunc();
            b_.setFastMathFlags(fmf);
        }
    }

    llvm::IRBuilder<>& builder() { return b_; }
    uint32_t width() const { return W_; }

    llvm::Type* ptrTy() const { return llvm::PointerType::getUnqual(b_.getContext()); }
    llvm::Type* i1Ty() const { return b_.getInt1Ty(); }
    llvm::Type* i8Ty() const { return b_.getInt8Ty(); }
    llvm::Type* i32Ty() const { return b_.getInt32Ty(); }
    llvm::Type* i64Ty() const { return b_.getInt64Ty(); }
    llvm::Type* f32Ty() const { return b_.getFloatTy(); }
    llvm::FixedVectorType* vecTy(llvm::Type* elem) const { return llvm::FixedVectorType::get(elem, W_); }

    llvm::Value* cInt(llvm::Type* ty, uint64_t val) {
        return llvm::ConstantInt::get(ty, val);
    }
    llvm::Value* cFlt(llvm::Type* ty, double val) {
        return llvm::ConstantFP::get(ty, val);
    }
    llvm::Value* splatInt(llvm::Type* elemTy, uint64_t val) {
        return llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W_), llvm::ConstantInt::get(elemTy, val));
    }
    llvm::Value* splatFlt(llvm::Type* elemTy, double val) {
        return llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W_), llvm::ConstantFP::get(elemTy, val));
    }
    llvm::Value* laneIota() {
        llvm::SmallVector<llvm::Constant*, 16> vals;
        for (uint32_t i = 0; i < W_; ++i) vals.push_back(b_.getInt32(i));
        return llvm::ConstantVector::get(vals);
    }
    llvm::Value* llvmExtract(llvm::IRBuilder<>& builder, llvm::Value* vec, uint32_t idx) {
        return ::nacelle::llvmExtract(builder, vec, idx);
    }
    llvm::Value* llvmExtract(llvm::Value* vec, uint32_t idx) {
        return ::nacelle::llvmExtract(b_, vec, idx);
    }
    llvm::Value* llvmInsert(llvm::IRBuilder<>& builder, llvm::Value* vec, llvm::Value* elem, uint32_t idx) {
        return ::nacelle::llvmInsert(builder, vec, elem, idx);
    }
    llvm::Value* llvmInsert(llvm::Value* vec, llvm::Value* elem, uint32_t idx) {
        return ::nacelle::llvmInsert(b_, vec, elem, idx);
    }
    llvm::Value* llvmSplat(llvm::IRBuilder<>& builder, llvm::Type* elemTy, llvm::Value* scalar) {
        return ::nacelle::llvmSplat(builder, elemTy, scalar, W_);
    }
    llvm::Value* llvmSplat(llvm::Type* elemTy, llvm::Value* scalar) {
        return ::nacelle::llvmSplat(b_, elemTy, scalar, W_);
    }

    llvm::Value* createFAdd(llvm::IRBuilder<>& builder, llvm::Value* l, llvm::Value* r) { return builder.CreateFAdd(l, r); }
    llvm::Value* createFAdd(llvm::Value* l, llvm::Value* r) { return b_.CreateFAdd(l, r); }
    llvm::Value* createFSub(llvm::IRBuilder<>& builder, llvm::Value* l, llvm::Value* r) { return builder.CreateFSub(l, r); }
    llvm::Value* createFSub(llvm::Value* l, llvm::Value* r) { return b_.CreateFSub(l, r); }
    llvm::Value* createFMul(llvm::IRBuilder<>& builder, llvm::Value* l, llvm::Value* r) { return builder.CreateFMul(l, r); }
    llvm::Value* createFMul(llvm::Value* l, llvm::Value* r) { return b_.CreateFMul(l, r); }
    llvm::Value* createFDiv(llvm::IRBuilder<>& builder, llvm::Value* l, llvm::Value* r) { return builder.CreateFDiv(l, r); }
    llvm::Value* createFDiv(llvm::Value* l, llvm::Value* r) { return b_.CreateFDiv(l, r); }
    llvm::Value* createFRem(llvm::IRBuilder<>& builder, llvm::Value* l, llvm::Value* r) { return builder.CreateFRem(l, r); }
    llvm::Value* createFRem(llvm::Value* l, llvm::Value* r) { return b_.CreateFRem(l, r); }
    llvm::Value* createFNeg(llvm::IRBuilder<>& builder, llvm::Value* v) { return builder.CreateFNeg(v); }
    llvm::Value* createFNeg(llvm::Value* v) { return b_.CreateFNeg(v); }
    llvm::Value* createFMA(llvm::Value* a, llvm::Value* b, llvm::Value* c) {
        return b_.CreateIntrinsic(llvm::Intrinsic::fma, {a->getType()}, {a, b, c});
    }
    llvm::Value* createSqrt(llvm::Value* v) {
        return b_.CreateIntrinsic(llvm::Intrinsic::sqrt, {v->getType()}, {v});
    }
    llvm::Value* createFAbs(llvm::Value* v) {
        return b_.CreateIntrinsic(llvm::Intrinsic::fabs, {v->getType()}, {v});
    }
    llvm::Value* createMinNum(llvm::Value* a, llvm::Value* b) {
        return b_.CreateIntrinsic(llvm::Intrinsic::minnum, {a->getType()}, {a, b});
    }
    llvm::Value* createMaxNum(llvm::Value* a, llvm::Value* b) {
        return b_.CreateIntrinsic(llvm::Intrinsic::maxnum, {a->getType()}, {a, b});
    }

    llvm::Value* emitMathUnary(llvm::StringRef baseName, llvm::Value* vecArg) {
        auto* vt = llvm::cast<llvm::FixedVectorType>(vecArg->getType());
        llvm::Type* elemTy = vt->getElementType();
        if (!strictFp_ && elemTy->isFloatTy() && (W_ == 4 || W_ == 8 || W_ == 16)) {
            std::string fnName = ("nacelle_math_v" + std::to_string(W_) + "_" + baseName + "f").str();
            llvm::FunctionType* fnTy = llvm::FunctionType::get(b_.getVoidTy(), {ptrTy(), ptrTy()}, false);
            llvm::FunctionCallee callee = mod_.getOrInsertFunction(fnName, fnTy);
            if (auto* f = llvm::dyn_cast<llvm::Function>(callee.getCallee())) {
                f->setLinkage(llvm::GlobalValue::ExternalLinkage);
            }
            llvm::Function* curFn = b_.GetInsertBlock()->getParent();
            llvm::IRBuilder<> eb(&curFn->getEntryBlock().front());
            llvm::AllocaInst* inSlot = eb.CreateAlloca(vt, nullptr, "math.in");
            llvm::AllocaInst* outSlot = eb.CreateAlloca(vt, nullptr, "math.out");
            b_.CreateStore(vecArg, inSlot);
            b_.CreateCall(callee, {outSlot, inSlot});
            return b_.CreateLoad(vt, outSlot, "math.res");
        }
        auto intr = [&](llvm::Intrinsic::ID id) {
            return b_.CreateIntrinsic(id, {vt}, {vecArg});
        };
        if (baseName == "sin") return intr(llvm::Intrinsic::sin);
        if (baseName == "cos") return intr(llvm::Intrinsic::cos);
        if (baseName == "tan") return intr(llvm::Intrinsic::tan);
        if (baseName == "acos") return intr(llvm::Intrinsic::acos);
        if (baseName == "asin") return intr(llvm::Intrinsic::asin);
        if (baseName == "atan") return intr(llvm::Intrinsic::atan);
        if (baseName == "cosh") return intr(llvm::Intrinsic::cosh);
        if (baseName == "sinh") return intr(llvm::Intrinsic::sinh);
        if (baseName == "tanh") return intr(llvm::Intrinsic::tanh);
        if (baseName == "exp2") return intr(llvm::Intrinsic::exp2);
        if (baseName == "log2") return intr(llvm::Intrinsic::log2);
        // Scalar libm fallback for strict-FP or non-f32 transcendental ops.
        std::string libmName = (elemTy->isFloatTy() ? (baseName + "f") : baseName).str();
        llvm::FunctionCallee libmFn = getOrDeclareLibmFn(libmName, elemTy);
        llvm::Value* outVec = llvm::PoisonValue::get(vt);
        for (uint32_t l = 0; l < W_; ++l) {
            llvm::Value* x = llvmExtract(vecArg, l);
            llvm::Value* r = b_.CreateCall(libmFn, {x});
            outVec = llvmInsert(outVec, r, l);
        }
        return outVec;
    }

    llvm::FunctionCallee getOrDeclareLibmFn(llvm::StringRef name, llvm::Type* ty) {
        auto* fnTy = llvm::FunctionType::get(ty, {ty}, false);
        return mod_.getOrInsertFunction(name, fnTy);
    }

    llvm::Value* orReduce(llvm::Value* vecI1) {
        return b_.CreateOrReduce(vecI1);
    }

    llvm::Value* allTrueMask() { return splatInt(i1Ty(), 1); }
    llvm::Value* allFalseMask() { return splatInt(i1Ty(), 0); }

    bool isMaskConstAllTrue(llvm::Value* maskVec) const {
        if (!maskVec) return false;
        if (auto* cdv = llvm::dyn_cast<llvm::ConstantDataVector>(maskVec)) {
            return cdv->getElementType()->isIntegerTy(1) && cdv->isSplat() && cdv->getElementAsInteger(0) != 0;
        }
        if (auto* cv = llvm::dyn_cast<llvm::ConstantVector>(maskVec)) {
            if (auto* ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(cv->getSplatValue())) {
                return !ci->isZero();
            }
        }
        return false;
    }

    bool isLaneMaskConstFalse(llvm::Value* maskVec, uint32_t lane, uint32_t depth = 0) const {
        if (!maskVec || depth > 8) return false;
        if (llvm::isa<llvm::ConstantAggregateZero>(maskVec)) return true;
        if (auto* cdv = llvm::dyn_cast<llvm::ConstantDataVector>(maskVec)) {
            if (!cdv->getElementType()->isIntegerTy(1)) return false;
            if (cdv->isSplat()) return cdv->getElementAsInteger(0) == 0;
            if (lane < cdv->getNumElements()) return cdv->getElementAsInteger(lane) == 0;
            return false;
        }
        if (auto* cv = llvm::dyn_cast<llvm::ConstantVector>(maskVec)) {
            if (lane < cv->getNumOperands()) {
                if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(cv->getOperand(lane))) {
                    return ci->isZero();
                }
            }
            return false;
        }
        if (auto* bin = llvm::dyn_cast<llvm::BinaryOperator>(maskVec)) {
            if (bin->getOpcode() == llvm::Instruction::And) {
                return isLaneMaskConstFalse(bin->getOperand(0), lane, depth + 1) ||
                       isLaneMaskConstFalse(bin->getOperand(1), lane, depth + 1);
            }
            if (bin->getOpcode() == llvm::Instruction::Or) {
                return isLaneMaskConstFalse(bin->getOperand(0), lane, depth + 1) &&
                       isLaneMaskConstFalse(bin->getOperand(1), lane, depth + 1);
            }
        }
        return false;
    }

    llvm::Value* expandMaskQuads(llvm::Value* mask) {
        // For quad q (4 lanes: 4q..4q+3), any active lane makes all 4 lanes execute in execMask.
        llvm::SmallVector<int32_t, 16> s1, s2;
        for (uint32_t l = 0; l < W_; ++l) {
            s1.push_back(static_cast<int32_t>(l ^ 1u));
            s2.push_back(static_cast<int32_t>(l ^ 2u));
        }
        llvm::Value* pair = b_.CreateOr(mask, b_.CreateShuffleVector(mask, mask, s1));
        return b_.CreateOr(pair, b_.CreateShuffleVector(pair, pair, s2));
    }

    SimdVal makeZeroSimd(llvm::Type* origTy) {
        if (origTy->isIntegerTy()) {
            return SimdVal::makeVector(splatInt(origTy, 0), origTy, SimdVal::LanePattern::Uniform);
        }
        if (origTy->isFloatingPointTy()) {
            return SimdVal::makeVector(splatFlt(origTy, 0.0), origTy, SimdVal::LanePattern::Uniform);
        }
        if (origTy->isPointerTy()) {
            auto zeroPtr = llvm::Constant::getNullValue(ptrTy());
            return SimdVal::makeLanes({zeroPtr}, origTy);
        }
        if (auto* st = llvm::dyn_cast<llvm::StructType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (llvm::Type* fTy : st->elements()) {
                sv.fields.push_back(makeZeroSimd(fTy));
            }
            return sv;
        }
        if (auto* at = llvm::dyn_cast<llvm::ArrayType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < at->getNumElements(); ++i) {
                sv.fields.push_back(makeZeroSimd(at->getElementType()));
            }
            return sv;
        }
        if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                sv.fields.push_back(makeZeroSimd(vt->getElementType()));
            }
            return sv;
        }
        return SimdVal::makeVector(splatInt(i32Ty(), 0), origTy, SimdVal::LanePattern::Uniform);
    }

    SimdVal getMapped(llvm::Value* orig) {
        if (!orig) return {};
        auto it = valueMap_.find(orig);
        if (it != valueMap_.end()) return it->second;

        if (isUndefOrPoison(orig) || llvm::isa<llvm::ConstantPointerNull>(orig) ||
            llvm::isa<llvm::ConstantAggregateZero>(orig)) {
            return makeZeroSimd(orig->getType());
        }
        if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(orig)) {
            return SimdVal::makeVector(splatInt(ci->getType(), ci->getZExtValue()), ci->getType(), SimdVal::LanePattern::Uniform);
        }
        if (auto* cf = llvm::dyn_cast<llvm::ConstantFP>(orig)) {
            return SimdVal::makeVector(splatFlt(cf->getType(), cf->getValueAPF().convertToDouble()), cf->getType(), SimdVal::LanePattern::Uniform);
        }
        if (auto* gv = llvm::dyn_cast<llvm::GlobalVariable>(orig)) {
            return SimdVal::makeLanes({gv}, gv->getType());
        }
        if (auto* cdv = llvm::dyn_cast<llvm::ConstantDataVector>(orig)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = cdv->getType();
            llvm::Type* et = cdv->getElementType();
            for (unsigned i = 0; i < cdv->getNumElements(); ++i) {
                if (et->isIntegerTy()) {
                    sv.fields.push_back(SimdVal::makeVector(splatInt(et, cdv->getElementAsInteger(i)), et, SimdVal::LanePattern::Uniform));
                } else {
                    sv.fields.push_back(SimdVal::makeVector(splatFlt(et, cdv->getElementAsAPFloat(i).convertToDouble()), et, SimdVal::LanePattern::Uniform));
                }
            }
            return sv;
        }
        if (auto* cv = llvm::dyn_cast<llvm::ConstantVector>(orig)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = cv->getType();
            for (unsigned i = 0; i < cv->getNumOperands(); ++i) {
                sv.fields.push_back(getMapped(cv->getOperand(i)));
            }
            return sv;
        }
        if (auto* cs = llvm::dyn_cast<llvm::ConstantStruct>(orig)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = cs->getType();
            for (unsigned i = 0; i < cs->getNumOperands(); ++i) {
                sv.fields.push_back(getMapped(cs->getOperand(i)));
            }
            return sv;
        }
        if (auto* ca = llvm::dyn_cast<llvm::ConstantArray>(orig)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = ca->getType();
            for (unsigned i = 0; i < ca->getNumOperands(); ++i) {
                sv.fields.push_back(getMapped(ca->getOperand(i)));
            }
            return sv;
        }
        return makeZeroSimd(orig->getType());
    }

    void setMapped(llvm::Value* orig, const SimdVal& sv) { valueMap_[orig] = sv; }

    llvm::Value* asVec(const SimdVal& sv) {
        if (sv.kind == SimdVal::Kind::Vector) return sv.vec;
        if (sv.kind == SimdVal::Kind::Lanes && !sv.lanes.empty()) {
            if (sv.lanes.size() == 1) {
                return llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(sv.lanes[0]->getType()))->getElementType(), sv.lanes[0]);
            }
            llvm::Type* elemTy = sv.lanes[0]->getType();
            llvm::Value* v = llvm::PoisonValue::get(vecTy(elemTy));
            for (uint32_t l = 0; l < W_; ++l) {
                v = llvmInsert(b_, v, sv.lanes[l], static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            return v;
        }
        return nullptr;
    }

    llvm::Value* laneValue(const SimdVal& sv, uint32_t l) {
        if (sv.kind == SimdVal::Kind::Vector) {
            return llvmExtract(b_, sv.vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
        }
        if (sv.kind == SimdVal::Kind::Lanes && !sv.lanes.empty()) {
            return sv.lanes.size() == 1 ? sv.lanes[0] : sv.lanes[l];
        }
        if (sv.kind == SimdVal::Kind::Handle && !sv.lanes.empty()) {
            return sv.lanes.size() == 1 ? sv.lanes[0] : sv.lanes[l];
        }
        if (sv.kind == SimdVal::Kind::Struct && sv.fields.size() == 1) {
            return laneValue(sv.fields[0], l);
        }
        return nullptr;
    }

    SimdVal selectVal(llvm::Value* condVec, const SimdVal& t, const SimdVal& f) {
        if (t.kind == SimdVal::Kind::Vector && f.kind == SimdVal::Kind::Vector) {
            llvm::Value* s = b_.CreateSelect(condVec, t.vec, f.vec);
            return SimdVal::makeVector(s, t.origType);
        }
        if (t.kind == SimdVal::Kind::Lanes || f.kind == SimdVal::Kind::Lanes ||
            t.kind == SimdVal::Kind::Handle || f.kind == SimdVal::Kind::Handle) {
            SimdVal out = t;
            out.lanes.clear();
            if (t.isUniform() && f.isUniform() && t.lanes[0] == f.lanes[0]) {
                out.lanes.push_back(t.lanes[0]);
                return out;
            }
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* cL = llvmExtract(b_, condVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* tL = laneValue(t, l);
                llvm::Value* fL = laneValue(f, l);
                out.lanes.push_back(b_.CreateSelect(cL, tL, fL));
            }
            return out;
        }
        if (t.kind == SimdVal::Kind::Struct && f.kind == SimdVal::Kind::Struct &&
            t.fields.size() == f.fields.size()) {
            SimdVal out = t;
            out.fields.clear();
            for (size_t i = 0; i < t.fields.size(); ++i) {
                out.fields.push_back(selectVal(condVec, t.fields[i], f.fields[i]));
            }
            return out;
        }
        return t;
    }

    /// Ensures a SimdVal only contains Vector/Lanes/Handle values with unambiguous flat storage.
    void flattenSimdTypes(llvm::Type* origTy, llvm::SmallVectorImpl<llvm::Type*>& outVecTypes) {
        if (origTy->isIntegerTy() || origTy->isFloatingPointTy()) {
            outVecTypes.push_back(vecTy(origTy));
        } else if (origTy->isPointerTy()) {
            outVecTypes.push_back(vecTy(ptrTy()));
        } else if (auto* st = llvm::dyn_cast<llvm::StructType>(origTy)) {
            for (llvm::Type* f : st->elements()) flattenSimdTypes(f, outVecTypes);
        } else if (auto* at = llvm::dyn_cast<llvm::ArrayType>(origTy)) {
            for (unsigned i = 0; i < at->getNumElements(); ++i) flattenSimdTypes(at->getElementType(), outVecTypes);
        } else if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(origTy)) {
            for (unsigned i = 0; i < vt->getNumElements(); ++i) flattenSimdTypes(vt->getElementType(), outVecTypes);
        } else {
            outVecTypes.push_back(vecTy(i32Ty()));
        }
    }

    void packSimdToValues(const SimdVal& sv, llvm::SmallVectorImpl<llvm::Value*>& out) {
        if (sv.kind == SimdVal::Kind::Vector) {
            out.push_back(sv.vec);
        } else if (sv.kind == SimdVal::Kind::Lanes || sv.kind == SimdVal::Kind::Handle) {
            if (sv.isUniform()) {
                out.push_back(llvmSplat(ptrTy(), laneValue(sv, 0)));
            } else {
                llvm::Value* vec = llvm::PoisonValue::get(vecTy(ptrTy()));
                for (uint32_t l = 0; l < W_; ++l) {
                    vec = llvmInsert(vec, laneValue(sv, l), l);
                }
                out.push_back(vec);
            }
        } else if (sv.kind == SimdVal::Kind::Struct) {
            for (const auto& f : sv.fields) packSimdToValues(f, out);
        } else {
            out.push_back(splatInt(i32Ty(), 0));
        }
    }

    SimdVal unpackSimdFromValues(llvm::Type* origTy, llvm::ArrayRef<llvm::Value*> vals, size_t& cursor, const SimdVal* proto = nullptr) {
        if (proto && proto->kind == SimdVal::Kind::Handle) {
            SimdVal h = *proto;
            h.lanes.clear();
            llvm::Value* ptrVec = vals[cursor++];
            if (h.isUniform()) {
                llvm::Value* p0 = llvmExtract(ptrVec, 0);
                h.lanes.assign(W_, p0);
            } else {
                for (uint32_t l = 0; l < W_; ++l) {
                    h.lanes.push_back(llvmExtract(ptrVec, l));
                }
            }
            return h;
        }
        if (origTy->isIntegerTy() || origTy->isFloatingPointTy()) {
            return SimdVal::makeVector(vals[cursor++], origTy);
        }
        if (origTy->isPointerTy()) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = origTy;
            llvm::Value* ptrVec = vals[cursor++];
            for (uint32_t l = 0; l < W_; ++l) {
                sv.lanes.push_back(llvmExtract(ptrVec, l));
            }
            return sv;
        }
        if (auto* st = llvm::dyn_cast<llvm::StructType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < st->getNumElements(); ++i) {
                const SimdVal* fp = (proto && i < proto->fields.size()) ? &proto->fields[i] : nullptr;
                sv.fields.push_back(unpackSimdFromValues(st->getElementType(i), vals, cursor, fp));
            }
            return sv;
        }
        if (auto* at = llvm::dyn_cast<llvm::ArrayType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < at->getNumElements(); ++i) {
                const SimdVal* fp = (proto && i < proto->fields.size()) ? &proto->fields[i] : nullptr;
                sv.fields.push_back(unpackSimdFromValues(at->getElementType(), vals, cursor, fp));
            }
            return sv;
        }
        if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                const SimdVal* fp = (proto && i < proto->fields.size()) ? &proto->fields[i] : nullptr;
                sv.fields.push_back(unpackSimdFromValues(vt->getElementType(), vals, cursor, fp));
            }
            return sv;
        }
        return SimdVal::makeVector(vals[cursor++], origTy);
    }

    struct SimdSlot {
        llvm::Type* origTy = nullptr;
        llvm::SmallVector<llvm::Value*, 4> allocas;
        llvm::SmallVector<llvm::Type*, 4> flatTys;
        SimdVal proto;
    };

    SimdSlot allocSimdSlot(llvm::Type* origTy, const SimdVal* proto = nullptr) {
        SimdSlot s;
        s.origTy = origTy;
        if (proto) s.proto = *proto;
        flattenSimdTypes(origTy, s.flatTys);
        for (llvm::Type* ft : s.flatTys) {
            s.allocas.push_back(createEntryAlloca(ft, 1));
        }
        return s;
    }

    void storeSimdSlot(const SimdSlot& s, const SimdVal& sv) {
        llvm::SmallVector<llvm::Value*, 4> packed;
        packSimdToValues(sv, packed);
        for (size_t i = 0; i < s.allocas.size() && i < packed.size(); ++i) {
            b_.CreateStore(packed[i], s.allocas[i]);
        }
    }

    SimdVal loadSimdSlot(const SimdSlot& s) {
        llvm::SmallVector<llvm::Value*, 4> vals;
        for (size_t i = 0; i < s.allocas.size(); ++i) {
            vals.push_back(b_.CreateLoad(s.flatTys[i], s.allocas[i]));
        }
        size_t cur = 0;
        return unpackSimdFromValues(s.origTy, vals, cur, s.proto.valid() ? &s.proto : nullptr);
    }

    static void collectBlocks(const RegionNode& node, llvm::DenseSet<const llvm::BasicBlock*>& out) {
        switch (node.kind) {
        case RegionNode::Kind::Block:
            out.insert(llvm::cast<BlockNode>(node).bb);
            break;
        case RegionNode::Kind::Seq:
            for (const auto& c : llvm::cast<SeqNode>(node).children) collectBlocks(*c, out);
            break;
        case RegionNode::Kind::If: {
            const auto& in = llvm::cast<IfNode>(node);
            if (in.thenRegion) collectBlocks(*in.thenRegion, out);
            if (in.elseRegion) collectBlocks(*in.elseRegion, out);
            break;
        }
        case RegionNode::Kind::While: {
            const auto& wn = llvm::cast<WhileNode>(node);
            if (wn.body) collectBlocks(*wn.body, out);
            break;
        }
        }
    }

    struct WhileExitContext {
        const WhileNode* wn = nullptr;
        llvm::Value* continueMask = nullptr;
        llvm::DenseMap<llvm::PHINode*, SimdSlot> exitPhiSlots;
    };

    bool packetizeBlockRange(
        llvm::BasicBlock::iterator beginIt,
        llvm::BasicBlock::iterator endIt,
        llvm::Value* execMask,
        llvm::Value*& realMask,
        std::string& outError) {
        for (auto it = beginIt; it != endIt; ++it) {
            llvm::Instruction& inst = *it;
            if (llvm::isa<llvm::PHINode>(&inst)) continue;
            if (inst.isTerminator()) break;
            if (!packetizeOp(inst, execMask, realMask, outError)) return false;
        }
        return true;
    }

    bool packetizeBlock(llvm::BasicBlock& srcBlock, llvm::Value* execMask, llvm::Value*& realMask, std::string& outError) {
        return packetizeBlockRange(srcBlock.begin(), srcBlock.end(), execMask, realMask, outError);
    }

    bool packetizeSeq(
        const SeqNode& seq,
        llvm::Value* execMask,
        llvm::Value*& realMask,
        std::string& outError,
        WhileExitContext* whileCtx = nullptr) {
        llvm::Value* curExec = execMask;
        for (const auto& child : seq.children) {
            switch (child->kind) {
            case RegionNode::Kind::Block: {
                const auto& bn = llvm::cast<BlockNode>(*child);
                if (!packetizeBlock(*bn.bb, curExec, realMask, outError)) return false;
                break;
            }
            case RegionNode::Kind::Seq: {
                if (!packetizeSeq(llvm::cast<SeqNode>(*child), curExec, realMask, outError, whileCtx)) return false;
                break;
            }
            case RegionNode::Kind::If: {
                if (!packetizeIfNode(llvm::cast<IfNode>(*child), curExec, realMask, outError, whileCtx)) return false;
                if (whileCtx) curExec = whileCtx->continueMask;
                break;
            }
            case RegionNode::Kind::While: {
                if (!packetizeWhileNode(llvm::cast<WhileNode>(*child), curExec, realMask, outError)) return false;
                break;
            }
            }
        }
        return true;
    }

    bool packetizeIfNode(
        const IfNode& ifn,
        llvm::Value* execMask,
        llvm::Value*& realMask,
        std::string& outError,
        WhileExitContext* whileCtx = nullptr) {
        llvm::Value* condVec = asVec(getMapped(ifn.cond));
        llvm::Value* realThen = b_.CreateAnd(realMask, condVec);
        llvm::Value* notCond = b_.CreateNot(condVec);
        llvm::Value* realElse = b_.CreateAnd(realMask, notCond);

        llvm::Value* execThen = b_.CreateAnd(execMask, condVec);
        llvm::Value* execElse = b_.CreateAnd(execMask, notCond);
        if (hooks_.isPixelShader && shader_.hasDerivativesOrImplicitSample) {
            execThen = expandMaskQuads(realThen);
            execElse = expandMaskQuads(realElse);
        }

        if (whileCtx) {
            if (ifn.thenExitsLoop) {
                llvm::Value* exitLanes = b_.CreateAnd(execMask, condVec);
                whileCtx->continueMask = b_.CreateAnd(whileCtx->continueMask, notCond);
                if (whileCtx->wn && whileCtx->wn->exitBB) {
                    for (llvm::PHINode& phi : whileCtx->wn->exitBB->phis()) {
                        int idx = phi.getBasicBlockIndex(ifn.headerBB);
                        if (idx >= 0) {
                            SimdVal incSv = getMapped(phi.getIncomingValue(idx));
                            auto& slot = whileCtx->exitPhiSlots[&phi];
                            SimdVal prevSv = loadSimdSlot(slot);
                            storeSimdSlot(slot, selectVal(exitLanes, incSv, prevSv));
                        }
                    }
                }
            }
            if (ifn.elseExitsLoop) {
                llvm::Value* exitLanes = b_.CreateAnd(execMask, notCond);
                whileCtx->continueMask = b_.CreateAnd(whileCtx->continueMask, condVec);
                if (whileCtx->wn && whileCtx->wn->exitBB) {
                    for (llvm::PHINode& phi : whileCtx->wn->exitBB->phis()) {
                        int idx = phi.getBasicBlockIndex(ifn.headerBB);
                        if (idx >= 0) {
                            SimdVal incSv = getMapped(phi.getIncomingValue(idx));
                            auto& slot = whileCtx->exitPhiSlots[&phi];
                            SimdVal prevSv = loadSimdSlot(slot);
                            storeSimdSlot(slot, selectVal(exitLanes, incSv, prevSv));
                        }
                    }
                }
            }
        }

        bool hasRealMaskOut = hooks_.isPixelShader && shader_.usesDiscard;
        llvm::Value* realMaskSlot = hasRealMaskOut ? createEntryAlloca(vecTy(i1Ty()), 1) : nullptr;

        auto savedMap = valueMap_;
        llvm::DenseMap<llvm::Value*, SimdVal> thenMap = savedMap;
        llvm::DenseMap<llvm::Value*, SimdVal> elseMap = savedMap;

        llvm::Value* realAfterThen = realThen;
        if (!ifn.thenExitsLoop && ifn.thenRegion && !ifn.thenRegion->children.empty()) {
            if (realMaskSlot) b_.CreateStore(realThen, realMaskSlot);
            emitIf(b_, orReduce(execThen), [&](llvm::IRBuilder<>&) {
                llvm::Value* curReal = realThen;
                if (!packetizeSeq(*ifn.thenRegion, execThen, curReal, outError, whileCtx)) return;
                if (realMaskSlot) b_.CreateStore(curReal, realMaskSlot);
                thenMap = std::move(valueMap_);
            });
            valueMap_ = savedMap;
            if (!outError.empty()) return false;
            if (realMaskSlot) realAfterThen = b_.CreateLoad(vecTy(i1Ty()), realMaskSlot);
        }

        llvm::Value* realAfterElse = realElse;
        if (!ifn.elseExitsLoop && ifn.elseRegion && !ifn.elseRegion->children.empty()) {
            if (realMaskSlot) b_.CreateStore(realElse, realMaskSlot);
            emitIf(b_, orReduce(execElse), [&](llvm::IRBuilder<>&) {
                llvm::Value* curReal = realElse;
                if (!packetizeSeq(*ifn.elseRegion, execElse, curReal, outError, whileCtx)) return;
                if (realMaskSlot) b_.CreateStore(curReal, realMaskSlot);
                elseMap = std::move(valueMap_);
            });
            valueMap_ = savedMap;
            if (!outError.empty()) return false;
            if (realMaskSlot) realAfterElse = b_.CreateLoad(vecTy(i1Ty()), realMaskSlot);
        }

        if (ifn.mergeBB) {
            llvm::DenseSet<const llvm::BasicBlock*> thenBlocks, elseBlocks;
            if (ifn.thenRegion) collectBlocks(*ifn.thenRegion, thenBlocks);
            if (ifn.elseRegion) collectBlocks(*ifn.elseRegion, elseBlocks);

            auto lookupFrom = [&](llvm::DenseMap<llvm::Value*, SimdVal>& m, llvm::Value* v) -> SimdVal {
                auto it = m.find(v);
                if (it != m.end()) return it->second;
                return getMapped(v);
            };

            for (llvm::PHINode& phi : ifn.mergeBB->phis()) {
                SimdVal tv, ev;
                bool hasT = false, hasE = false;
                for (unsigned i = 0; i < phi.getNumIncomingValues(); ++i) {
                    llvm::BasicBlock* incBB = phi.getIncomingBlock(i);
                    llvm::Value* incV = phi.getIncomingValue(i);
                    if (thenBlocks.count(incBB) || (thenBlocks.empty() && incBB == ifn.headerBB)) {
                        tv = lookupFrom(thenMap, incV);
                        hasT = true;
                    } else if (elseBlocks.count(incBB) || (elseBlocks.empty() && incBB == ifn.headerBB)) {
                        ev = lookupFrom(elseMap, incV);
                        hasE = true;
                    }
                }
                if (hasT && hasE) {
                    setMapped(&phi, selectVal(condVec, tv, ev));
                } else if (hasT) {
                    setMapped(&phi, tv);
                } else if (hasE) {
                    setMapped(&phi, ev);
                } else {
                    setMapped(&phi, makeZeroSimd(phi.getType()));
                }
            }
        }

        if (ifn.elseExitsLoop) {
            for (auto& kv : thenMap) valueMap_[kv.first] = kv.second;
        } else if (ifn.thenExitsLoop) {
            for (auto& kv : elseMap) valueMap_[kv.first] = kv.second;
        }

        if (hasRealMaskOut) {
            realMask = b_.CreateOr(realAfterThen, realAfterElse);
        }
        return true;
    }

    bool packetizeWhileNode(
        const WhileNode& wn,
        llvm::Value* execMask,
        llvm::Value*& realMask,
        std::string& outError) {
        llvm::Value* loopExecSlot = createEntryAlloca(vecTy(i1Ty()), 1);
        llvm::Value* aliveMaskSlot = createEntryAlloca(vecTy(i1Ty()), 1);
        b_.CreateStore(execMask, loopExecSlot);
        b_.CreateStore(realMask, aliveMaskSlot);

        llvm::DenseMap<llvm::PHINode*, SimdSlot> hdrPhiSlots;
        for (llvm::PHINode& phi : wn.headerBB->phis()) {
            SimdVal initSv;
            if (wn.preheaderBB) {
                initSv = getMapped(phi.getIncomingValueForBlock(wn.preheaderBB));
            } else {
                for (unsigned i = 0; i < phi.getNumIncomingValues(); ++i) {
                    if (!wn.loop->contains(phi.getIncomingBlock(i))) {
                        initSv = getMapped(phi.getIncomingValue(i));
                        break;
                    }
                }
            }
            if (!initSv.valid()) initSv = makeZeroSimd(phi.getType());
            SimdSlot slot = allocSimdSlot(phi.getType(), &initSv);
            storeSimdSlot(slot, initSv);
            hdrPhiSlots[&phi] = std::move(slot);
        }

        WhileExitContext exitCtx;
        exitCtx.wn = &wn;
        if (wn.exitBB) {
            for (llvm::PHINode& phi : wn.exitBB->phis()) {
                SimdVal z = makeZeroSimd(phi.getType());
                SimdSlot slot = allocSimdSlot(phi.getType(), &z);
                storeSimdSlot(slot, z);
                exitCtx.exitPhiSlots[&phi] = std::move(slot);
            }
        }

        llvm::Function* curFn = b_.GetInsertBlock()->getParent();
        llvm::BasicBlock* loopHdrBB = llvm::BasicBlock::Create(b_.getContext(), "while.hdr", curFn);
        llvm::BasicBlock* loopExitBB = llvm::BasicBlock::Create(b_.getContext(), "while.exit", curFn);
        b_.CreateBr(loopHdrBB);

        b_.SetInsertPoint(loopHdrBB);
        llvm::Value* loopExec = b_.CreateLoad(vecTy(i1Ty()), loopExecSlot);
        llvm::Value* aliveMask = b_.CreateLoad(vecTy(i1Ty()), aliveMaskSlot);
        llvm::Value* curReal = b_.CreateAnd(loopExec, aliveMask);

        auto savedMap = valueMap_;
        llvm::DenseMap<llvm::PHINode*, SimdVal> curHdrVals;
        for (auto& kv : hdrPhiSlots) {
            SimdVal sv = loadSimdSlot(kv.second);
            curHdrVals[kv.first] = sv;
            setMapped(kv.first, sv);
        }

        exitCtx.continueMask = loopExec;
        if (wn.body && !packetizeSeq(*wn.body, loopExec, curReal, outError, &exitCtx)) {
            valueMap_ = std::move(savedMap);
            return false;
        }

        llvm::Value* nextExec = exitCtx.continueMask;
        if (wn.latchBB) {
            if (auto* br = llvm::dyn_cast<llvm::BranchInst>(wn.latchBB->getTerminator())) {
                if (br->isConditional()) {
                    llvm::Value* condVec = asVec(getMapped(br->getCondition()));
                    llvm::Value* contCond = (br->getSuccessor(0) == wn.headerBB) ? condVec : b_.CreateNot(condVec);
                    llvm::Value* exitLanes = b_.CreateAnd(nextExec, b_.CreateNot(contCond));
                    nextExec = b_.CreateAnd(nextExec, contCond);
                    if (wn.exitBB) {
                        for (llvm::PHINode& phi : wn.exitBB->phis()) {
                            int idx = phi.getBasicBlockIndex(wn.latchBB);
                            if (idx >= 0) {
                                SimdVal incSv = getMapped(phi.getIncomingValue(idx));
                                auto& slot = exitCtx.exitPhiSlots[&phi];
                                SimdVal prevSv = loadSimdSlot(slot);
                                storeSimdSlot(slot, selectVal(exitLanes, incSv, prevSv));
                            }
                        }
                    }
                }
            }
            for (auto& kv : hdrPhiSlots) {
                int idx = kv.first->getBasicBlockIndex(wn.latchBB);
                if (idx >= 0) {
                    SimdVal nextSv = getMapped(kv.first->getIncomingValue(idx));
                    SimdVal blended = selectVal(loopExec, nextSv, curHdrVals[kv.first]);
                    storeSimdSlot(kv.second, blended);
                }
            }
        }

        llvm::Value* survived = b_.CreateOr(b_.CreateNot(loopExec), curReal);
        llvm::Value* aliveAfter = b_.CreateAnd(aliveMask, survived);
        b_.CreateStore(aliveAfter, aliveMaskSlot);

        nextExec = b_.CreateAnd(nextExec, aliveAfter);
        if (hooks_.isPixelShader && shader_.hasDerivativesOrImplicitSample) {
            nextExec = expandMaskQuads(nextExec);
        }
        b_.CreateStore(nextExec, loopExecSlot);
        b_.CreateCondBr(orReduce(nextExec), loopHdrBB, loopExitBB);

        b_.SetInsertPoint(loopExitBB);
        valueMap_ = std::move(savedMap);
        realMask = b_.CreateLoad(vecTy(i1Ty()), aliveMaskSlot);
        if (wn.exitBB) {
            for (auto& kv : exitCtx.exitPhiSlots) {
                setMapped(kv.first, loadSimdSlot(kv.second));
            }
        }
        return true;
    }

private:
    llvm::Value* gepByteOffset(llvm::Value* basePtr, int64_t byteOff) {
        if (byteOff == 0) return basePtr;
        return b_.CreateGEP(i8Ty(), basePtr, b_.getInt32(static_cast<int32_t>(byteOff)));
    }

    llvm::Value* gepDynByteOffset(llvm::Value* basePtr, llvm::Value* byteOff) {
        return b_.CreateGEP(i8Ty(), basePtr, llvm::ArrayRef<llvm::Value*>{byteOff});
    }

    llvm::Value* loadScalar(llvm::Type* ty, llvm::Value* ptr) {
        return b_.CreateLoad(ty, ptr);
    }
    llvm::Value* loadInvariantScalar(llvm::Type* ty, llvm::Value* ptr) {
        auto* li = b_.CreateLoad(ty, ptr);
        li->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(b_.getContext(), {}));
        return li;
    }

    void storeScalar(llvm::Value* val, llvm::Value* ptr) {
        b_.CreateStore(val, ptr);
    }

    llvm::Value* createEntryAlloca(llvm::Type* elemTy, uint64_t count, unsigned align = 64) {
        llvm::BasicBlock* entryBlk = hooks_.entryBlock;
        if (!entryBlk) {
            llvm::BasicBlock* curBlk = b_.GetInsertBlock();
            if (curBlk && curBlk->getParent()) {
                entryBlk = &curBlk->getParent()->getEntryBlock();
            }
        }
        if (entryBlk) {
            llvm::IRBuilder<> eb(entryBlk, entryBlk->begin());
            llvm::Value* sz = llvm::ConstantInt::get(eb.getInt32Ty(), count);
            return createAlignedAlloca(eb, elemTy, sz, align);
        }
        return createAlignedAlloca(b_, elemTy, cInt(i32Ty(), count), align);
    }

    llvm::Value* getOrCreateScratchAlloca() {
        if (!scratchAlloca_) {
            // 16-byte aligned scratch slot in entry block for masked-off alloca lanes.
            scratchAlloca_ = createEntryAlloca(i64Ty(), 2, 16);
        }
        return scratchAlloca_;
    }

    llvm::FunctionCallee getOrDeclareRuntimeFn(
        llvm::StringRef name, llvm::Type* retTy, llvm::ArrayRef<llvm::Type*> argTys) {
        auto* fnTy = llvm::FunctionType::get(retTy ? retTy : b_.getVoidTy(), argTys, false);
        return mod_.getOrInsertFunction(name, fnTy);
    }

    bool packetizeOp(llvm::Instruction& op, llvm::Value* execMask, llvm::Value*& realMask, std::string& outError) {
        // Constants / Undef / Poison / Freeze
        if (auto* frz = llvm::dyn_cast<llvm::FreezeInst>(&op)) {
            setMapped(frz, getMapped(frz->getOperand(0)));
            return true;
        }

        // Calls (including intrinsics and lifetime markers)
        if (auto* call = llvm::dyn_cast<llvm::CallInst>(&op)) {
            if (call->isLifetimeStartOrEnd()) return true;
            if (packetizeVectorizableOp(op, execMask)) return true;
            return packetizeCall(call, execMask, realMask, outError);
        }

        // Vector element insert/extract/shuffle (DXIL long vectors)
        if (auto* ee = llvm::dyn_cast<llvm::ExtractElementInst>(&op)) {
            SimdVal vec = getMapped(ee->getVectorOperand());
            if (vec.kind != SimdVal::Kind::Struct || vec.fields.empty()) {
                setMapped(ee, makeZeroSimd(ee->getType()));
                return true;
            }
            if (auto idxC = getConstIntValue(ee->getIndexOperand())) {
                size_t idx = static_cast<size_t>(*idxC);
                setMapped(ee, idx < vec.fields.size() ? vec.fields[idx] : makeZeroSimd(ee->getType()));
                return true;
            }
            llvm::Value* idxVec = asVec(getMapped(ee->getIndexOperand()));
            llvm::Type* idxTy = llvm::cast<llvm::FixedVectorType>(idxVec->getType())->getElementType();
            llvm::Type* elemTy = ee->getType();
            llvm::Value* res = asVec(vec.fields[0]);
            for (size_t i = 1; i < vec.fields.size(); ++i) {
                llvm::Value* eq = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, idxVec, splatInt(idxTy, i));
                res = b_.CreateSelect(eq, asVec(vec.fields[i]), res);
            }
            setMapped(ee, SimdVal::makeVector(res, elemTy));
            return true;
        }
        if (auto* ie = llvm::dyn_cast<llvm::InsertElementInst>(&op)) {
            SimdVal vec = getMapped(ie->getOperand(0));
            if (vec.kind != SimdVal::Kind::Struct || vec.fields.empty()) {
                vec = makeZeroSimd(ie->getType());
            }
            SimdVal val = getMapped(ie->getOperand(1));
            if (auto idxC = getConstIntValue(ie->getOperand(2))) {
                size_t idx = static_cast<size_t>(*idxC);
                if (idx < vec.fields.size()) vec.fields[idx] = val;
                setMapped(ie, vec);
                return true;
            }
            llvm::Value* idxVec = asVec(getMapped(ie->getOperand(2)));
            llvm::Type* idxTy = llvm::cast<llvm::FixedVectorType>(idxVec->getType())->getElementType();
            llvm::Value* valVec = asVec(val);
            for (size_t i = 0; i < vec.fields.size(); ++i) {
                llvm::Value* eq = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, idxVec, splatInt(idxTy, i));
                llvm::Value* cur = asVec(vec.fields[i]);
                vec.fields[i].vec = b_.CreateSelect(eq, valVec, cur);
            }
            setMapped(ie, vec);
            return true;
        }
        if (auto* sv = llvm::dyn_cast<llvm::ShuffleVectorInst>(&op)) {
            SimdVal v1 = getMapped(sv->getOperand(0));
            SimdVal v2 = getMapped(sv->getOperand(1));
            auto* resVt = llvm::cast<llvm::FixedVectorType>(sv->getType());
            llvm::Type* elemTy = resVt->getElementType();
            size_t n1 = (v1.kind == SimdVal::Kind::Struct) ? v1.fields.size() : 0;
            size_t n2 = (v2.kind == SimdVal::Kind::Struct) ? v2.fields.size() : 0;
            SimdVal out;
            out.kind = SimdVal::Kind::Struct;
            out.origType = resVt;
            for (int m : sv->getShuffleMask()) {
                if (m >= 0 && static_cast<size_t>(m) < n1) {
                    out.fields.push_back(v1.fields[static_cast<size_t>(m)]);
                } else if (m >= 0 && static_cast<size_t>(m) < n1 + n2) {
                    out.fields.push_back(v2.fields[static_cast<size_t>(m) - n1]);
                } else {
                    out.fields.push_back(makeZeroSimd(elemTy));
                }
            }
            setMapped(sv, out);
            return true;
        }

        // Aggregate insert/extract
        if (auto* ev = llvm::dyn_cast<llvm::ExtractValueInst>(&op)) {
            SimdVal agg = getMapped(ev->getAggregateOperand());
            SimdVal cur = agg;
            for (unsigned idx : ev->getIndices()) {
                if (cur.kind == SimdVal::Kind::Struct && static_cast<size_t>(idx) < cur.fields.size()) {
                    SimdVal next = cur.fields[static_cast<size_t>(idx)];
                    cur = std::move(next);
                } else {
                    cur = makeZeroSimd(ev->getType());
                    break;
                }
            }
            setMapped(ev, cur);
            return true;
        }
        if (auto* iv = llvm::dyn_cast<llvm::InsertValueInst>(&op)) {
            SimdVal agg = getMapped(iv->getAggregateOperand());
            SimdVal elem = getMapped(iv->getInsertedValueOperand());
            auto pos = iv->getIndices();
            std::function<void(SimdVal&, size_t)> setRec = [&](SimdVal& cur, size_t d) {
                if (cur.kind != SimdVal::Kind::Struct) return;
                size_t idx = static_cast<size_t>(pos[d]);
                if (idx >= cur.fields.size()) return;
                if (d + 1 == pos.size()) cur.fields[idx] = elem;
                else setRec(cur.fields[idx], d + 1);
            };
            setRec(agg, 0);
            setMapped(iv, agg);
            return true;
        }

        // Stack Alloca / GEP / Load / Store / Atomic
        if (auto* allocaOp = llvm::dyn_cast<llvm::AllocaInst>(&op)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            sv.isAllocaPtr = true;
            llvm::Type* elemTy = allocaOp->getAllocatedType();
            uint64_t count = getConstIntValue(allocaOp->getArraySize()).value_or(1);
            llvm::Type* slotTy = (count == 1) ? elemTy : llvm::ArrayType::get(elemTy, count);
            uint32_t numWaves = std::max<uint32_t>(1u, hooks_.numWaves);
            llvm::Value* baseAlloca = createEntryAlloca(slotTy, static_cast<uint64_t>(numWaves) * W_, 64);
            if (hooks_.waveIdx && numWaves > 1u) {
                llvm::Value* w32 = hooks_.waveIdx;
                llvm::Value* waveBase = b_.CreateMul(w32, cInt(i32Ty(), W_));
                for (uint32_t l = 0; l < W_; ++l) {
                    llvm::Value* slotIdx = b_.CreateAdd(waveBase, cInt(i32Ty(), l));
                    sv.lanes.push_back(b_.CreateGEP(slotTy, baseAlloca, llvm::ArrayRef<llvm::Value*>{slotIdx}));
                }
            } else {
                for (uint32_t l = 0; l < W_; ++l) {
                    sv.lanes.push_back(b_.CreateGEP(slotTy, baseAlloca, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), l)}));
                }
            }
            setMapped(allocaOp, sv);
            return true;
        }
        if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(&op)) {
            SimdVal base = getMapped(gep->getPointerOperand());
            llvm::SmallVector<SimdVal, 4> idxVals;
            llvm::SmallVector<bool, 4> isDyn;
            for (llvm::Value* iv : gep->indices()) {
                if (llvm::isa<llvm::ConstantInt>(iv)) {
                    isDyn.push_back(false);
                } else {
                    isDyn.push_back(true);
                    idxVals.push_back(getMapped(iv));
                }
            }

            bool uniform = base.isUniform();
            if (uniform) {
                for (const auto& iv : idxVals) {
                    if (!iv.isUniform()) {
                        uniform = false;
                        break;
                    }
                }
            }

            SimdVal out;
            out.kind = SimdVal::Kind::Lanes;
            out.origType = gep->getType();
            out.isAllocaPtr = base.isAllocaPtr;
            if (uniform) {
                out.pattern = SimdVal::LanePattern::Uniform;
            } else if (base.isUniform() && idxVals.size() == 1 && idxVals[0].isUnitStride() &&
                       gep->getNumIndices() == 1 && isDyn[0]) {
                out.pattern = SimdVal::LanePattern::UnitStride;
            }
            uint32_t lanesOut = uniform ? 1u : W_;
            for (uint32_t l = 0; l < lanesOut; ++l) {
                llvm::SmallVector<llvm::Value*, 4> gepArgs;
                size_t dynIdx = 0;
                for (llvm::Value* iv : gep->indices()) {
                    if (llvm::isa<llvm::ConstantInt>(iv)) {
                        gepArgs.push_back(iv);
                    } else {
                        gepArgs.push_back(laneValue(idxVals[dynIdx++], l));
                    }
                }
                llvm::Value* lp = b_.CreateGEP(gep->getSourceElementType(), laneValue(base, l), gepArgs, "", gep->getNoWrapFlags());
                out.lanes.push_back(lp);
            }
            setMapped(gep, out);
            return true;
        }
        if (auto* loadOp = llvm::dyn_cast<llvm::LoadInst>(&op)) {
            SimdVal addr = getMapped(loadOp->getPointerOperand());
            llvm::Type* elemTy = loadOp->getType();
            uint64_t align = std::max<uint64_t>(1, loadOp->getAlign().value());
            bool isVecScalar = elemTy->isIntegerTy() || elemTy->isFloatingPointTy() || elemTy->isPointerTy();
            if (addr.isUniform() && isVecScalar) {
                llvm::Value* s = loadScalar(elemTy, laneValue(addr, 0));
                llvm::Value* bvec = llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(elemTy))->getElementType(), s);
                SimdVal sv = SimdVal::makeVector(bvec, elemTy);
                sv.pattern = SimdVal::LanePattern::Uniform;
                setMapped(loadOp, std::move(sv));
                return true;
            }
            if (isVecScalar) {
                if (addr.isUnitStride() && !addr.isAllocaPtr && isMaskConstAllTrue(execMask)) {
                    llvm::Value* v = b_.CreateAlignedLoad(vecTy(elemTy), laneValue(addr, 0), llvm::Align(align));
                    setMapped(loadOp, SimdVal::makeVector(v, elemTy));
                    return true;
                }
                if (addr.isAllocaPtr) {
                    llvm::Value* v = llvm::UndefValue::get(vecTy(elemTy));
                    llvm::Value* scratch = getOrCreateScratchAlloca();
                    for (uint32_t l = 0; l < W_; ++l) {
                        if (isLaneMaskConstFalse(execMask, l)) continue;
                        llvm::Value* mL = llvmExtract(b_, execMask, l);
                        llvm::Value* pL = laneValue(addr, l);
                        llvm::Value* safePtr = b_.CreateSelect(mL, pL, scratch);
                        llvm::Value* s = loadScalar(elemTy, safePtr);
                        v = llvmInsert(b_, v, s, l);
                    }
                    setMapped(loadOp, SimdVal::makeVector(v, elemTy));
                    return true;
                }
                llvm::Value* ptrs = asVec(addr);
                llvm::Value* undefVec = llvm::UndefValue::get(vecTy(elemTy));
                llvm::Value* v = b_.CreateMaskedGather(vecTy(elemTy), ptrs, llvm::Align(align), execMask, undefVec);
                setMapped(loadOp, SimdVal::makeVector(v, elemTy));
                return true;
            }
            if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(elemTy)) {
                llvm::Type* scalarTy = vt->getElementType();
                unsigned n = vt->getNumElements();
                bool scalarIsVec = scalarTy->isIntegerTy() || scalarTy->isFloatingPointTy() || scalarTy->isPointerTy();
                SimdVal res;
                res.kind = SimdVal::Kind::Struct;
                res.origType = vt;
                res.fields.resize(n);
                if (addr.isUniform()) {
                    for (unsigned i = 0; i < n; ++i) {
                        llvm::Value* ep = b_.CreateGEP(scalarTy, addr.lanes[0], llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                        llvm::Value* s = loadScalar(scalarTy, ep);
                        llvm::Value* bvec = llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(scalarTy))->getElementType(), s);
                        res.fields[i] = SimdVal::makeVector(bvec, scalarTy);
                    }
                } else if (addr.isAllocaPtr && scalarIsVec) {
                    llvm::Value* scratch = getOrCreateScratchAlloca();
                    for (unsigned i = 0; i < n; ++i) {
                        llvm::Value* v = llvm::UndefValue::get(vecTy(scalarTy));
                        for (uint32_t l = 0; l < W_; ++l) {
                            if (isLaneMaskConstFalse(execMask, l)) continue;
                            llvm::Value* mL = llvmExtract(b_, execMask, l);
                            llvm::Value* pL = laneValue(addr, l);
                            llvm::Value* ep = (i == 0)
                                                 ? pL
                                                 : b_.CreateGEP(scalarTy, pL, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                            llvm::Value* safePtr = b_.CreateSelect(mL, ep, scratch);
                            llvm::Value* s = loadScalar(scalarTy, safePtr);
                            v = llvmInsert(b_, v, s, l);
                        }
                        res.fields[i] = SimdVal::makeVector(v, scalarTy);
                    }
                } else if (scalarIsVec) {
                    llvm::Value* ptrs = asVec(addr);
                    llvm::Value* undefVec = llvm::UndefValue::get(vecTy(scalarTy));
                    for (unsigned i = 0; i < n; ++i) {
                        llvm::Value* epVec = (i == 0)
                                                ? ptrs
                                                : b_.CreateGEP(scalarTy, ptrs, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                        llvm::Value* v = b_.CreateMaskedGather(vecTy(scalarTy), epVec, llvm::Align(align), execMask, undefVec);
                        res.fields[i] = SimdVal::makeVector(v, scalarTy);
                    }
                } else {
                    llvm::Value* tmpSlot = createEntryAlloca(scalarTy, 1, 16);
                    for (unsigned i = 0; i < n; ++i) {
                        llvm::Value* v = llvm::UndefValue::get(vecTy(scalarTy));
                        for (uint32_t l = 0; l < W_; ++l) {
                            if (isLaneMaskConstFalse(execMask, l)) continue;
                            llvm::Value* mL = llvmExtract(b_, execMask, l);
                            llvm::Value* pL = laneValue(addr, l);
                            b_.CreateStore(llvm::UndefValue::get(scalarTy), tmpSlot);
                            emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                                llvm::Value* ep = nb.CreateGEP(scalarTy, pL, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                                nb.CreateStore(nb.CreateLoad(scalarTy, ep), tmpSlot);
                            });
                            llvm::Value* loaded = b_.CreateLoad(scalarTy, tmpSlot);
                            v = llvmInsert(b_, v, loaded, l);
                        }
                        res.fields[i] = SimdVal::makeVector(v, scalarTy);
                    }
                }
                setMapped(loadOp, res);
                return true;
            }
            SimdVal res;
            res.kind = SimdVal::Kind::Lanes;
            res.origType = elemTy;
            for (uint32_t l = 0; l < (addr.isUniform() ? 1u : W_); ++l) {
                res.lanes.push_back(loadScalar(elemTy, laneValue(addr, l)));
            }
            setMapped(loadOp, res);
            return true;
        }
        if (auto* storeOp = llvm::dyn_cast<llvm::StoreInst>(&op)) {
            SimdVal val = getMapped(storeOp->getValueOperand());
            SimdVal addr = getMapped(storeOp->getPointerOperand());
            llvm::Type* valTy = storeOp->getValueOperand()->getType();
            uint64_t align = std::max<uint64_t>(1, storeOp->getAlign().value());
            if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(valTy)) {
                if (val.kind == SimdVal::Kind::Struct && val.fields.size() == vt->getNumElements()) {
                    llvm::Type* scalarTy = vt->getElementType();
                    bool scalarIsVec = scalarTy->isIntegerTy() || scalarTy->isFloatingPointTy() || scalarTy->isPointerTy();
                    if (scalarIsVec) {
                        if (addr.isAllocaPtr) {
                            llvm::Value* scratch = getOrCreateScratchAlloca();
                            for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                                for (uint32_t l = 0; l < W_; ++l) {
                                    if (isLaneMaskConstFalse(execMask, l)) continue;
                                    llvm::Value* mL = llvmExtract(b_, execMask, l);
                                    llvm::Value* pL = laneValue(addr, l);
                                    llvm::Value* ep = (i == 0)
                                                         ? pL
                                                         : b_.CreateGEP(scalarTy, pL, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                                    llvm::Value* safePtr = b_.CreateSelect(mL, ep, scratch);
                                    llvm::Value* vL = laneValue(val.fields[i], l);
                                    storeScalar(vL, safePtr);
                                }
                            }
                            return true;
                        }
                        llvm::Value* ptrs = asVec(addr);
                        for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                            llvm::Value* valVec = asVec(val.fields[i]);
                            llvm::Value* epVec = (i == 0)
                                                    ? ptrs
                                                    : b_.CreateGEP(scalarTy, ptrs, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                            b_.CreateMaskedScatter(valVec, epVec, llvm::Align(align), execMask);
                        }
                        return true;
                    }
                    for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                        for (uint32_t l = 0; l < W_; ++l) {
                            if (isLaneMaskConstFalse(execMask, l)) continue;
                            llvm::Value* mL = llvmExtract(b_, execMask, l);
                            llvm::Value* pL = laneValue(addr, l);
                            llvm::Value* vL = laneValue(val.fields[i], l);
                            emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                                llvm::Value* ep = nb.CreateGEP(scalarTy, pL, llvm::ArrayRef<llvm::Value*>{cInt(i32Ty(), i)});
                                nb.CreateStore(vL, ep);
                            });
                        }
                    }
                    return true;
                }
            }
            if (valTy->isIntegerTy() || valTy->isFloatingPointTy() || valTy->isPointerTy()) {
                if (addr.isUnitStride() && !addr.isAllocaPtr && isMaskConstAllTrue(execMask)) {
                    if (llvm::Value* valVec = asVec(val)) {
                        b_.CreateAlignedStore(valVec, laneValue(addr, 0), llvm::Align(align));
                        return true;
                    }
                }
                if (addr.isAllocaPtr) {
                    llvm::Value* scratch = getOrCreateScratchAlloca();
                    for (uint32_t l = 0; l < W_; ++l) {
                        if (isLaneMaskConstFalse(execMask, l)) continue;
                        llvm::Value* mL = llvmExtract(b_, execMask, l);
                        llvm::Value* pL = laneValue(addr, l);
                        llvm::Value* safePtr = b_.CreateSelect(mL, pL, scratch);
                        llvm::Value* vL = laneValue(val, l);
                        storeScalar(vL, safePtr);
                    }
                    return true;
                }
                llvm::Value* valVec = asVec(val);
                llvm::Value* ptrs = asVec(addr);
                if (valVec && ptrs) {
                    b_.CreateMaskedScatter(valVec, ptrs, llvm::Align(align), execMask);
                    return true;
                }
            }
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, execMask, l);
                llvm::Value* pL = laneValue(addr, l);
                llvm::Value* vL = laneValue(val, l);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    nb.CreateStore(vL, pL);
                });
            }
            return true;
        }
        if (auto* rmw = llvm::dyn_cast<llvm::AtomicRMWInst>(&op)) {
            SimdVal addr = getMapped(rmw->getPointerOperand());
            SimdVal val = getMapped(rmw->getValOperand());
            llvm::Type* ty = rmw->getType();
            llvm::Value* outVec = ty->isFloatingPointTy() ? splatFlt(ty, 0.0) : splatInt(ty, 0);
            llvm::Value* tmpSlot = createEntryAlloca(ty, 1, 16);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, realMask, l);
                llvm::Value* pL = laneValue(addr, l);
                llvm::Value* vL = laneValue(val, l);
                b_.CreateStore(llvm::UndefValue::get(ty), tmpSlot);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    llvm::Value* r;
                    if (ty->isFloatingPointTy() && rmw->getOperation() == llvm::AtomicRMWInst::Xchg) {
                        llvm::Type* intTy = nb.getIntNTy(ty->getPrimitiveSizeInBits());
                        llvm::Value* vInt = nb.CreateBitCast(vL, intTy);
                        llvm::Value* rInt = nb.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg, pL, vInt, llvm::MaybeAlign(), rmw->getOrdering());
                        r = nb.CreateBitCast(rInt, ty);
                    } else {
                        r = nb.CreateAtomicRMW(rmw->getOperation(), pL, vL, llvm::MaybeAlign(), rmw->getOrdering());
                    }
                    nb.CreateStore(r, tmpSlot);
                });
                llvm::Value* rVal = b_.CreateLoad(ty, tmpSlot);
                outVec = llvmInsert(b_, outVec, rVal, l);
            }
            setMapped(rmw, SimdVal::makeVector(outVec, ty));
            return true;
        }
        if (auto* cx = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&op)) {
            SimdVal addr = getMapped(cx->getPointerOperand());
            SimdVal cmp = getMapped(cx->getCompareOperand());
            SimdVal val = getMapped(cx->getNewValOperand());
            llvm::Type* elemTy = cmp.origType;
            llvm::Value* outValVec = elemTy->isFloatingPointTy() ? splatFlt(elemTy, 0.0) : splatInt(elemTy, 0);
            llvm::Value* outOkVec = splatInt(i1Ty(), 0);
            llvm::Value* valSlot = createEntryAlloca(elemTy, 1, 16);
            llvm::Value* okSlot = createEntryAlloca(i1Ty(), 1, 4);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, realMask, l);
                llvm::Value* pL = laneValue(addr, l);
                llvm::Value* cL = laneValue(cmp, l);
                llvm::Value* vL = laneValue(val, l);
                b_.CreateStore(llvm::UndefValue::get(elemTy), valSlot);
                b_.CreateStore(cInt(i1Ty(), 0), okSlot);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    llvm::Value* cOp = cL;
                    llvm::Value* vOp = vL;
                    if (elemTy->isFloatingPointTy()) {
                        llvm::Type* intTy = nb.getIntNTy(elemTy->getPrimitiveSizeInBits());
                        cOp = nb.CreateBitCast(cL, intTy);
                        vOp = nb.CreateBitCast(vL, intTy);
                    }
                    auto* r = nb.CreateAtomicCmpXchg(pL, cOp, vOp, llvm::MaybeAlign(), cx->getSuccessOrdering(), cx->getFailureOrdering());
                    llvm::Value* oldV = nb.CreateExtractValue(r, 0u);
                    if (elemTy->isFloatingPointTy()) {
                        oldV = nb.CreateBitCast(oldV, elemTy);
                    }
                    llvm::Value* ok = nb.CreateExtractValue(r, 1u);
                    nb.CreateStore(oldV, valSlot);
                    nb.CreateStore(ok, okSlot);
                });
                outValVec = llvmInsert(b_, outValVec, b_.CreateLoad(elemTy, valSlot), l);
                outOkVec = llvmInsert(b_, outOkVec, b_.CreateLoad(i1Ty(), okSlot), l);
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = cx->getType();
            st.fields.push_back(SimdVal::makeVector(outValVec, elemTy));
            st.fields.push_back(SimdVal::makeVector(outOkVec, i1Ty()));
            setMapped(cx, st);
            return true;
        }
        if (auto* f = llvm::dyn_cast<llvm::FenceInst>(&op)) {
            b_.CreateFence(f->getOrdering());
            return true;
        }

        // Vectorizable LLVM ops
        if (packetizeVectorizableOp(op, execMask)) return true;

        // PtrToInt / IntToPtr scalar fallback
        if (auto* p2i = llvm::dyn_cast<llvm::PtrToIntInst>(&op)) {
            SimdVal src = getMapped(p2i->getPointerOperand());
            llvm::Type* dstTy = p2i->getType();
            llvm::Value* outVec = llvm::PoisonValue::get(vecTy(dstTy));
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* iv = b_.CreatePtrToInt(laneValue(src, l), dstTy);
                outVec = llvmInsert(b_, outVec, iv, l);
            }
            setMapped(p2i, SimdVal::makeVector(outVec, dstTy));
            return true;
        }
        if (auto* i2p = llvm::dyn_cast<llvm::IntToPtrInst>(&op)) {
            SimdVal src = getMapped(i2p->getOperand(0));
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            for (uint32_t l = 0; l < W_; ++l) {
                sv.lanes.push_back(b_.CreateIntToPtr(laneValue(src, l), ptrTy()));
            }
            setMapped(i2p, sv);
            return true;
        }

        std::string opStr;
        llvm::raw_string_ostream os(opStr);
        op.print(os);
        outError = "Unsupported LLVM instruction in SIMT packetizer: " + opStr;
        return false;
    }

    bool packetizeVectorizableOp(llvm::Instruction& op, llvm::Value* /*execMask*/) {
        // Unary and binary LLVM instructions and intrinsics that natively accept vector<W x T>.
        auto vecUnary = [&](auto tagFn) -> bool {
            llvm::Type* resTy = op.getType();
            SimdVal aSv = getMapped(op.getOperand(0));
            auto outPat = aSv.isUniform() ? SimdVal::LanePattern::Uniform : SimdVal::LanePattern::Divergent;
            if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(resTy)) {
                llvm::Type* resElem = vt->getElementType();
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = resTy;
                out.pattern = outPat;
                for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                    llvm::Value* ai = (aSv.kind == SimdVal::Kind::Struct && i < aSv.fields.size())
                                         ? asVec(aSv.fields[i])
                                         : asVec(aSv);
                    llvm::Value* r = tagFn(vecTy(resElem), ai);
                    SimdVal fsv = SimdVal::makeVector(r, resElem);
                    fsv.pattern = outPat;
                    out.fields.push_back(std::move(fsv));
                }
                setMapped(&op, out);
                return true;
            }
            llvm::Value* a = asVec(aSv);
            llvm::Value* r = tagFn(vecTy(resTy), a);
            SimdVal sv = SimdVal::makeVector(r, resTy);
            sv.pattern = outPat;
            setMapped(&op, std::move(sv));
            return true;
        };
        auto vecBinary = [&](auto tagFn) -> bool {
            llvm::Type* resTy = op.getType();
            SimdVal aSv = getMapped(op.getOperand(0));
            SimdVal cSv = getMapped(op.getOperand(1));
            SimdVal::LanePattern outPat = SimdVal::LanePattern::Divergent;
            if (aSv.isUniform() && cSv.isUniform()) {
                outPat = SimdVal::LanePattern::Uniform;
            } else if (op.getOpcode() == llvm::Instruction::Add &&
                       ((aSv.isUnitStride() && cSv.isUniform()) || (aSv.isUniform() && cSv.isUnitStride()))) {
                outPat = SimdVal::LanePattern::UnitStride;
            } else if (op.getOpcode() == llvm::Instruction::Sub &&
                       aSv.isUnitStride() && cSv.isUniform()) {
                outPat = SimdVal::LanePattern::UnitStride;
            }
            if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(resTy)) {
                llvm::Type* resElem = vt->getElementType();
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = resTy;
                out.pattern = outPat;
                for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                    llvm::Value* ai = (aSv.kind == SimdVal::Kind::Struct && i < aSv.fields.size())
                                         ? asVec(aSv.fields[i])
                                         : asVec(aSv);
                    llvm::Value* ci = (cSv.kind == SimdVal::Kind::Struct && i < cSv.fields.size())
                                         ? asVec(cSv.fields[i])
                                         : asVec(cSv);
                    llvm::Value* r = tagFn(vecTy(resElem), ai, ci);
                    SimdVal fsv = SimdVal::makeVector(r, resElem);
                    fsv.pattern = outPat;
                    out.fields.push_back(std::move(fsv));
                }
                setMapped(&op, out);
                return true;
            }
            llvm::Value* a = asVec(aSv);
            llvm::Value* c = asVec(cSv);
            llvm::Value* r = tagFn(vecTy(resTy), a, c);
            SimdVal sv = SimdVal::makeVector(r, resTy);
            sv.pattern = outPat;
            setMapped(&op, std::move(sv));
            return true;
        };

        switch (op.getOpcode()) {
        case llvm::Instruction::Add:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateAdd(a, c); });
        case llvm::Instruction::Sub:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateSub(a, c); });
        case llvm::Instruction::Mul:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateMul(a, c); });
        case llvm::Instruction::SDiv:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateSDiv(a, c); });
        case llvm::Instruction::UDiv:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateUDiv(a, c); });
        case llvm::Instruction::SRem:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateSRem(a, c); });
        case llvm::Instruction::URem:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateURem(a, c); });
        case llvm::Instruction::Shl:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateShl(a, c); });
        case llvm::Instruction::LShr:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateLShr(a, c); });
        case llvm::Instruction::AShr:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateAShr(a, c); });
        case llvm::Instruction::And:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateAnd(a, c); });
        case llvm::Instruction::Or:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateOr(a, c); });
        case llvm::Instruction::Xor:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return b_.CreateXor(a, c); });

        case llvm::Instruction::FAdd:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createFAdd(a, c); });
        case llvm::Instruction::FSub:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createFSub(a, c); });
        case llvm::Instruction::FMul:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createFMul(a, c); });
        case llvm::Instruction::FDiv:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createFDiv(a, c); });
        case llvm::Instruction::FRem:
            return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createFRem(a, c); });
        case llvm::Instruction::FNeg:
            return vecUnary([&](llvm::Type*, llvm::Value* a) { return createFNeg(a); });

        case llvm::Instruction::ZExt:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateZExt(a, rt); });
        case llvm::Instruction::SExt:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateSExt(a, rt); });
        case llvm::Instruction::Trunc:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateTrunc(a, rt); });
        case llvm::Instruction::FPTrunc:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateFPTrunc(a, rt); });
        case llvm::Instruction::FPExt:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateFPExt(a, rt); });
        case llvm::Instruction::FPToSI:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateFPToSI(a, rt); });
        case llvm::Instruction::FPToUI:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateFPToUI(a, rt); });
        case llvm::Instruction::SIToFP:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateSIToFP(a, rt); });
        case llvm::Instruction::UIToFP:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateUIToFP(a, rt); });
        case llvm::Instruction::BitCast:
            return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateBitCast(a, rt); });
        default:
            break;
        }

        if (auto* icmp = llvm::dyn_cast<llvm::ICmpInst>(&op)) {
            if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(icmp->getType())) {
                SimdVal aSv = getMapped(icmp->getOperand(0));
                SimdVal cSv = getMapped(icmp->getOperand(1));
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = vt;
                for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                    llvm::Value* ai = asVec(aSv.fields[i]);
                    llvm::Value* ci = asVec(cSv.fields[i]);
                    llvm::Value* r = b_.CreateICmp(icmp->getPredicate(), ai, ci);
                    out.fields.push_back(SimdVal::makeVector(r, i1Ty()));
                }
                setMapped(icmp, out);
                return true;
            }
            if (icmp->getOperand(0)->getType()->isPointerTy()) {
                SimdVal a = getMapped(icmp->getOperand(0));
                SimdVal c = getMapped(icmp->getOperand(1));
                llvm::Value* outVec = splatInt(i1Ty(), 0);
                for (uint32_t l = 0; l < W_; ++l) {
                    llvm::Value* r = b_.CreateICmp(icmp->getPredicate(), laneValue(a, l), laneValue(c, l));
                    outVec = llvmInsert(b_, outVec, r, l);
                }
                setMapped(icmp, SimdVal::makeVector(outVec, i1Ty()));
                return true;
            }
            llvm::Value* a = asVec(getMapped(icmp->getOperand(0)));
            llvm::Value* c = asVec(getMapped(icmp->getOperand(1)));
            llvm::Value* r = b_.CreateICmp(icmp->getPredicate(), a, c);
            setMapped(icmp, SimdVal::makeVector(r, i1Ty()));
            return true;
        }
        if (auto* fcmp = llvm::dyn_cast<llvm::FCmpInst>(&op)) {
            if (auto* vt = llvm::dyn_cast<llvm::FixedVectorType>(fcmp->getType())) {
                SimdVal aSv = getMapped(fcmp->getOperand(0));
                SimdVal cSv = getMapped(fcmp->getOperand(1));
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = vt;
                for (unsigned i = 0; i < vt->getNumElements(); ++i) {
                    llvm::Value* ai = asVec(aSv.fields[i]);
                    llvm::Value* ci = asVec(cSv.fields[i]);
                    llvm::Value* r = b_.CreateFCmp(fcmp->getPredicate(), ai, ci);
                    out.fields.push_back(SimdVal::makeVector(r, i1Ty()));
                }
                setMapped(fcmp, out);
                return true;
            }
            llvm::Value* a = asVec(getMapped(fcmp->getOperand(0)));
            llvm::Value* c = asVec(getMapped(fcmp->getOperand(1)));
            llvm::Value* r = b_.CreateFCmp(fcmp->getPredicate(), a, c);
            setMapped(fcmp, SimdVal::makeVector(r, i1Ty()));
            return true;
        }
        if (auto* sel = llvm::dyn_cast<llvm::SelectInst>(&op)) {
            SimdVal condSv = getMapped(sel->getCondition());
            SimdVal t = getMapped(sel->getTrueValue());
            SimdVal f = getMapped(sel->getFalseValue());
            if (condSv.kind == SimdVal::Kind::Struct && t.kind == SimdVal::Kind::Struct && f.kind == SimdVal::Kind::Struct) {
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = sel->getType();
                for (size_t i = 0; i < condSv.fields.size() && i < t.fields.size() && i < f.fields.size(); ++i) {
                    out.fields.push_back(selectVal(asVec(condSv.fields[i]), t.fields[i], f.fields[i]));
                }
                setMapped(sel, out);
                return true;
            }
            llvm::Value* c = asVec(condSv);
            setMapped(sel, selectVal(c, t, f));
            return true;
        }

        // Standard LLVM intrinsics (which DXIL/Clang might emit alongside dx.op.*)
        if (auto* ii = llvm::dyn_cast<llvm::IntrinsicInst>(&op)) {
            auto castIntVec = [&](llvm::Value* v, llvm::Type* dstElem) -> llvm::Value* {
                llvm::Type* srcElem = llvm::cast<llvm::FixedVectorType>(v->getType())->getElementType();
                if (srcElem == dstElem) return v;
                return (srcElem->getPrimitiveSizeInBits() > dstElem->getPrimitiveSizeInBits())
                           ? b_.CreateTrunc(v, vecTy(dstElem))
                           : b_.CreateZExt(v, vecTy(dstElem));
            };
            switch (ii->getIntrinsicID()) {
            case llvm::Intrinsic::fabs:
                return vecUnary([&](llvm::Type*, llvm::Value* a) { return createFAbs(a); });
            case llvm::Intrinsic::sqrt:
                return vecUnary([&](llvm::Type*, llvm::Value* a) { return createSqrt(a); });
            case llvm::Intrinsic::sin:
                return vecUnary([&](llvm::Type*, llvm::Value* a) { return emitMathUnary("sin", a); });
            case llvm::Intrinsic::cos:
                return vecUnary([&](llvm::Type*, llvm::Value* a) { return emitMathUnary("cos", a); });
            case llvm::Intrinsic::exp2:
                return vecUnary([&](llvm::Type*, llvm::Value* a) { return emitMathUnary("exp2", a); });
            case llvm::Intrinsic::log2:
                return vecUnary([&](llvm::Type*, llvm::Value* a) { return emitMathUnary("log2", a); });
            case llvm::Intrinsic::floor:
                return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateIntrinsic(llvm::Intrinsic::floor, {rt}, {a}); });
            case llvm::Intrinsic::ceil:
                return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateIntrinsic(llvm::Intrinsic::ceil, {rt}, {a}); });
            case llvm::Intrinsic::trunc:
                return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateIntrinsic(llvm::Intrinsic::trunc, {rt}, {a}); });
            case llvm::Intrinsic::roundeven:
                return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateIntrinsic(llvm::Intrinsic::roundeven, {rt}, {a}); });
            case llvm::Intrinsic::minnum:
                return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createMinNum(a, c); });
            case llvm::Intrinsic::maxnum:
                return vecBinary([&](llvm::Type*, llvm::Value* a, llvm::Value* c) { return createMaxNum(a, c); });
            case llvm::Intrinsic::umin:
                return vecBinary([&](llvm::Type* rt, llvm::Value* a, llvm::Value* c) { return b_.CreateIntrinsic(llvm::Intrinsic::umin, {rt}, {a, c}); });
            case llvm::Intrinsic::umax:
                return vecBinary([&](llvm::Type* rt, llvm::Value* a, llvm::Value* c) { return b_.CreateIntrinsic(llvm::Intrinsic::umax, {rt}, {a, c}); });
            case llvm::Intrinsic::smin:
                return vecBinary([&](llvm::Type* rt, llvm::Value* a, llvm::Value* c) { return b_.CreateIntrinsic(llvm::Intrinsic::smin, {rt}, {a, c}); });
            case llvm::Intrinsic::smax:
                return vecBinary([&](llvm::Type* rt, llvm::Value* a, llvm::Value* c) { return b_.CreateIntrinsic(llvm::Intrinsic::smax, {rt}, {a, c}); });
            case llvm::Intrinsic::ctpop: {
                llvm::Type* resElem = ii->getType();
                llvm::Value* a = asVec(getMapped(ii->getArgOperand(0)));
                llvm::Value* r = castIntVec(b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {a->getType()}, {a}), resElem);
                setMapped(ii, SimdVal::makeVector(r, resElem));
                return true;
            }
            case llvm::Intrinsic::bitreverse:
                return vecUnary([&](llvm::Type* rt, llvm::Value* a) { return b_.CreateIntrinsic(llvm::Intrinsic::bitreverse, {rt}, {a}); });
            case llvm::Intrinsic::fma:
            case llvm::Intrinsic::fmuladd: {
                llvm::Type* resElem = ii->getType();
                llvm::Value* a = asVec(getMapped(ii->getArgOperand(0)));
                llvm::Value* x = asVec(getMapped(ii->getArgOperand(1)));
                llvm::Value* c = asVec(getMapped(ii->getArgOperand(2)));
                llvm::Value* r = createFMA(a, x, c);
                setMapped(ii, SimdVal::makeVector(r, resElem));
                return true;
            }
            case llvm::Intrinsic::ctlz: {
                llvm::Type* resElem = ii->getType();
                llvm::Value* a = asVec(getMapped(ii->getArgOperand(0)));
                llvm::Value* r = castIntVec(b_.CreateIntrinsic(llvm::Intrinsic::ctlz, {a->getType()}, {a, b_.getFalse()}), resElem);
                setMapped(ii, SimdVal::makeVector(r, resElem));
                return true;
            }
            case llvm::Intrinsic::cttz: {
                llvm::Type* resElem = ii->getType();
                llvm::Value* a = asVec(getMapped(ii->getArgOperand(0)));
                llvm::Value* r = castIntVec(b_.CreateIntrinsic(llvm::Intrinsic::cttz, {a->getType()}, {a, b_.getFalse()}), resElem);
                setMapped(ii, SimdVal::makeVector(r, resElem));
                return true;
            }
            case llvm::Intrinsic::is_fpclass: {
                llvm::Value* a = asVec(getMapped(ii->getArgOperand(0)));
                auto* bitC = llvm::cast<llvm::ConstantInt>(ii->getArgOperand(1));
                llvm::Value* r = b_.CreateIntrinsic(llvm::Intrinsic::is_fpclass, {a->getType()}, {a, b_.getInt32(bitC->getZExtValue())});
                setMapped(ii, SimdVal::makeVector(r, i1Ty()));
                return true;
            }
            default:
                break;
            }
        }
        return false;
    }

    bool packetizeCall(llvm::CallInst* call, llvm::Value* execMask, llvm::Value*& realMask, std::string& outError) {
        auto callee = (call->getCalledFunction() ? std::optional<llvm::StringRef>(call->getCalledFunction()->getName()) : std::nullopt);
        if (!callee) {
            outError = "Indirect call inside DXIL shader is not supported";
            return false;
        }
        if (*callee == "nacelle.groupshared.ptr") {
            uint64_t byteOff = getConstIntValue(call->getArgOperand(0)).value_or(0);
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            sv.lanes.push_back(gepByteOffset(hooks_.groupSharedBase, static_cast<int64_t>(byteOff)));
            setMapped(call, sv);
            return true;
        }
        if (callee->starts_with("dx.op.")) {
            uint32_t opNum = static_cast<uint32_t>(getConstIntValue(call->getArgOperand(0)).value_or(0));
            return packetizeDxOp(call, opNum, execMask, realMask, outError);
        }
        outError = "Unsupported call inside DXIL shader: " + callee->str();
        return false;
    }

    // --- Resource Handle Resolution ---
    SimdVal resolveStaticBinding(uint32_t resClass, uint32_t space, uint32_t lowerBound,
                                 uint32_t resourceKind, uint32_t stride, llvm::Value* indexVal, llvm::Type* origTy,
                                 llvm::Value* execMask) {
        RootBindingTarget tgt = rootLayout_.resolveBinding(resClass, space, lowerBound);
        SimdVal h;
        h.kind = SimdVal::Kind::Handle;
        h.origType = origTy;
        h.resClass = resClass;
        h.resourceKind = resourceKind;
        h.stride = stride;

        if (tgt.kind == RootBindingTarget::RootConstants) {
            h.isRootConstants = true;
            int64_t off = static_cast<int64_t>(offsetof(DriverRootContext, inline_constants)) +
                          static_cast<int64_t>(tgt.tableOffsetOrConstStart) * 4;
            h.lanes.push_back(gepByteOffset(rootCtxPtr_, off));
            return h;
        }
        if (tgt.kind == RootBindingTarget::RootDescriptor) {
            int64_t off = static_cast<int64_t>(offsetof(DriverRootContext, inline_descriptors)) +
                          static_cast<int64_t>(tgt.rootParamIndex) * 64;
            h.lanes.push_back(gepByteOffset(rootCtxPtr_, off));
            return h;
        }
        if (tgt.kind == RootBindingTarget::StaticSampler) {
            int64_t off = static_cast<int64_t>(offsetof(DriverRootContext, static_samplers)) +
                          static_cast<int64_t>(tgt.staticSamplerIndex) * 64;
            h.lanes.push_back(gepByteOffset(rootCtxPtr_, off));
            return h;
        }
        if (tgt.kind == RootBindingTarget::DescriptorTableOffset) {
            int64_t slotOff = static_cast<int64_t>(offsetof(DriverRootContext, root_slots)) +
                              static_cast<int64_t>(tgt.rootParamIndex) * 8;
            llvm::Value* tblBase = loadInvariantScalar(ptrTy(), gepByteOffset(rootCtxPtr_, slotOff));
            auto idxConst = getConstIntValue(indexVal);
            if (idxConst.has_value()) {
                int64_t regIdx = static_cast<int64_t>(*idxConst);
                int64_t tableIdx = static_cast<int64_t>(tgt.tableOffsetOrConstStart) +
                                   (regIdx - static_cast<int64_t>(lowerBound));
                h.lanes.push_back(gepByteOffset(tblBase, tableIdx * 64));
                return h;
            }
            bool isSamp = (resClass == 3);
            int64_t heapBaseOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_base))
                                         : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_base));
            int64_t heapCntOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_count))
                                        : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_count));
            llvm::Value* heapBase = loadInvariantScalar(ptrTy(), gepByteOffset(rootCtxPtr_, heapBaseOff));
            llvm::Value* heapCnt = loadInvariantScalar(i32Ty(), gepByteOffset(rootCtxPtr_, heapCntOff));
            llvm::Value* heapI64 = b_.CreatePtrToInt(heapBase, i64Ty());
            llvm::Value* tblI64 = b_.CreatePtrToInt(tblBase, i64Ty());
            llvm::Value* heapBytes = b_.CreateMul(b_.CreateZExt(heapCnt, i64Ty()), cInt(i64Ty(), 64));
            llvm::Value* heapEnd = b_.CreateAdd(heapI64, heapBytes);
            llvm::Value* heapNonNull = b_.CreateICmp(llvm::CmpInst::ICMP_NE, heapI64, cInt(i64Ty(), 0));
            llvm::Value* tblInHeap = b_.CreateAnd(heapNonNull, b_.CreateAnd(
                    b_.CreateICmp(llvm::CmpInst::ICMP_UGE, tblI64, heapI64),
                    b_.CreateICmp(llvm::CmpInst::ICMP_ULT, tblI64, heapEnd)));

            llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
            SimdVal idxSimd = getMapped(indexVal);
            llvm::Value* idxVec = asVec(idxSimd);
            int64_t bias = static_cast<int64_t>(tgt.tableOffsetOrConstStart) - static_cast<int64_t>(lowerBound);
            for (uint32_t l = 0; l < W_; ++l) {
                if (execMask && isLaneMaskConstFalse(execMask, l)) {
                    h.lanes.push_back(nullPtr);
                    continue;
                }
                llvm::Value* idxL = llvmExtract(b_, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* idx64 = b_.CreateSExt(idxL, i64Ty());
                llvm::Value* slot = b_.CreateAdd(idx64, cInt(i64Ty(), bias));
                llvm::Value* byteOff = b_.CreateMul(slot, cInt(i64Ty(), 64));
                llvm::Value* candPtr = gepDynByteOffset(tblBase, byteOff);

                llvm::Value* slotNonNeg = b_.CreateICmp(llvm::CmpInst::ICMP_SGE, slot, cInt(i64Ty(), 0));
                llvm::Value* candI64 = b_.CreatePtrToInt(candPtr, i64Ty());
                llvm::Value* inHeap = b_.CreateAnd(b_.CreateICmp(llvm::CmpInst::ICMP_UGE, candI64, heapI64), b_.CreateICmp(llvm::CmpInst::ICMP_ULT, candI64, heapEnd));
                // If tblBase is inside the descriptor heap, enforce [heapBase, heapEnd);
                // otherwise allow non-negative slot offsets (e.g. standalone test tables).
                llvm::Value* boundsOk = b_.CreateSelect(tblInHeap, inHeap, slotNonNeg);
                if (execMask) {
                    llvm::Value* mL = llvmExtract(b_, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    boundsOk = b_.CreateAnd(mL, boundsOk);
                }
                h.lanes.push_back(b_.CreateSelect(boundsOk, candPtr, nullPtr));
            }
            return h;
        }

        // Unbound fallback -> null descriptor pointer (bounds checks return 0 safely)
        h.lanes.push_back(llvm::Constant::getNullValue(ptrTy()));
        return h;
    }

    llvm::Value* maskDescPtrForLane(const SimdVal& h, llvm::Value* maskVec, uint32_t lane) {
        llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
        if (maskVec && isLaneMaskConstFalse(maskVec, lane)) {
            return nullPtr;
        }
        llvm::Value* descP = laneValue(h, lane);
        if (maskVec && h.lanes.size() > 1) {
            llvm::Value* mL = llvmExtract(b_, maskVec, static_cast<uint32_t>(static_cast<int64_t>(lane)));
            descP = b_.CreateSelect(mL, descP, nullPtr);
        }
        return descP;
    }

    // Loads NacelleDescriptor fields safely (if descPtr is null, returns 0)
    llvm::Value* loadDescBasePtr(llvm::Value* descPtr) {
        llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
        llvm::Value* isNull = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, descPtr, nullPtr);
        llvm::Value* slot = createEntryAlloca(ptrTy(), 1, 8);
        emitIfElse(b_, isNull,
                   [&](llvm::IRBuilder<>& nb) { nb.CreateStore(nullPtr, slot); },
                   [&](llvm::IRBuilder<>& nb) {
                       auto* p = nb.CreateLoad(ptrTy(), descPtr);
                       p->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(nb.getContext(), {}));
                       nb.CreateStore(p, slot);
                   });
        return b_.CreateLoad(ptrTy(), slot);
    }

    llvm::Value* loadDescU32Field(llvm::Value* descPtr, int64_t byteOff, uint32_t fallback = 0) {
        llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
        llvm::Value* isNull = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, descPtr, nullPtr);
        llvm::Value* slot = createEntryAlloca(i32Ty(), 1, 4);
        emitIfElse(b_, isNull,
                   [&](llvm::IRBuilder<>& nb) { nb.CreateStore(cInt(i32Ty(), fallback), slot); },
                   [&](llvm::IRBuilder<>& nb) {
                       llvm::Value* fp = nb.CreateGEP(i8Ty(), descPtr, nb.getInt32(static_cast<int32_t>(byteOff)));
                       auto* val = nb.CreateLoad(i32Ty(), fp);
                       val->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(nb.getContext(), {}));
                       nb.CreateStore(val, slot);
                   });
        return b_.CreateLoad(i32Ty(), slot);
    }

    llvm::Value* loadDescU64Field(llvm::Value* descPtr, int64_t byteOff) {
        llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
        llvm::Value* isNull = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, descPtr, nullPtr);
        llvm::Value* slot = createEntryAlloca(i64Ty(), 1, 8);
        emitIfElse(b_, isNull,
                   [&](llvm::IRBuilder<>& nb) { nb.CreateStore(cInt(i64Ty(), 0), slot); },
                   [&](llvm::IRBuilder<>& nb) {
                       llvm::Value* fp = nb.CreateGEP(i8Ty(), descPtr, nb.getInt32(static_cast<int32_t>(byteOff)));
                       auto* val = nb.CreateLoad(i64Ty(), fp);
                       val->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(nb.getContext(), {}));
                       nb.CreateStore(val, slot);
                   });
        return b_.CreateLoad(i64Ty(), slot);
    }

    bool packetizeDxOp(llvm::CallInst* call, uint32_t op, llvm::Value* execMask, llvm::Value*& realMask, std::string& outError) {
        auto arg = [&](unsigned i) -> llvm::Value* { return i < call->arg_size() ? call->getArgOperand(i) : nullptr; };
        auto resTy = [&]() -> llvm::Type* { return (!call->getType()->isVoidTy()) ? call->getType() : nullptr; };

        // 1. Thread / Group system values (93..96)
        if (op == 93 || op == 94 || op == 95) {
            uint32_t comp = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            llvm::Value* v = (op == 93) ? (comp < hooks_.threadId.size() ? hooks_.threadId[comp] : splatInt(i32Ty(), 0))
                            : (op == 94) ? (comp < hooks_.groupId.size() ? hooks_.groupId[comp] : splatInt(i32Ty(), 0))
                                         : (comp < hooks_.threadIdInGroup.size() ? hooks_.threadIdInGroup[comp] : splatInt(i32Ty(), 0));
            SimdVal sv = SimdVal::makeVector(v, i32Ty());
            if (op == 94) {
                sv.pattern = SimdVal::LanePattern::Uniform;
            } else if ((op == 93 || op == 95) && comp == 0 && !hooks_.isPixelShader) {
                sv.pattern = SimdVal::LanePattern::UnitStride;
            }
            setMapped(call, std::move(sv));
            return true;
        }
        if (op == 96) {
            llvm::Value* v = hooks_.flatGroupId ? hooks_.flatGroupId : splatInt(i32Ty(), 0);
            SimdVal sv = SimdVal::makeVector(v, i32Ty());
            sv.pattern = SimdVal::LanePattern::Uniform;
            setMapped(call, std::move(sv));
            return true;
        }

        // 2. Handles (57, 217, 218, 216)
        if (op == 57) {
            uint32_t resClass = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            uint32_t rangeId = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            const DxilResourceRecord* rec = shader_.findResource(resClass, rangeId);
            uint32_t space = rec ? rec->space : 0;
            uint32_t lower = rec ? rec->lowerBound : rangeId;
            uint32_t kind = rec ? rec->resourceKind : 0;
            uint32_t stride = rec ? rec->structuredStride : 0;
            setMapped(call, resolveStaticBinding(resClass, space, lower, kind, stride, arg(3), resTy(), execMask));
            return true;
        }
        if (op == 217) {
            // createHandleFromBinding(op, ResBind{lower, upper, space, class}, index, nonUniform)
            uint32_t lower = 0, space = 0, resClass = 0;
            llvm::SmallVector<uint64_t, 4> rbFields;
            if (extractConstStructInts(arg(1), rbFields) && rbFields.size() >= 4) {
                lower = static_cast<uint32_t>(rbFields[0]);
                space = static_cast<uint32_t>(rbFields[2]);
                resClass = static_cast<uint32_t>(rbFields[3]);
            } else {
                // Fallback: resolve from Struct fields if constructed dynamically
                SimdVal rb = getMapped(arg(1));
                if (rb.fields.size() >= 4) {
                    lower = static_cast<uint32_t>(getConstIntValue(rb.fields[0].vec).value_or(0));
                    space = static_cast<uint32_t>(getConstIntValue(rb.fields[2].vec).value_or(0));
                    resClass = static_cast<uint32_t>(getConstIntValue(rb.fields[3].vec).value_or(0));
                }
            }
            // Match stride/kind from shader_.resources if present
            uint32_t kind = 0, stride = 0;
            for (const auto& r : shader_.resources) {
                if (r.resClass == resClass && r.space == space && lower >= r.lowerBound &&
                    (r.rangeSize == UINT32_MAX || lower < r.lowerBound + r.rangeSize)) {
                    kind = r.resourceKind;
                    stride = r.structuredStride;
                    break;
                }
            }
            setMapped(call, resolveStaticBinding(resClass, space, lower, kind, stride, arg(2), resTy(), execMask));
            return true;
        }
        if (op == 218) {
            // createHandleFromHeap(op, index, samplerHeap, nonUniform)
            bool isSamp = getConstIntValue(arg(2)).value_or(0) != 0;
            int64_t baseOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_base))
                                     : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_base));
            int64_t cntOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_count))
                                    : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_count));
            llvm::Value* heapBase = loadInvariantScalar(ptrTy(), gepByteOffset(rootCtxPtr_, baseOff));
            llvm::Value* heapCnt = loadInvariantScalar(i32Ty(), gepByteOffset(rootCtxPtr_, cntOff));

            SimdVal h;
            h.kind = SimdVal::Kind::Handle;
            h.origType = resTy();
            h.resClass = isSamp ? 3u : 0u;
            if (auto idxC = getConstIntValue(arg(1))) {
                llvm::Value* ok = b_.CreateICmp(llvm::CmpInst::ICMP_ULT, cInt(i32Ty(), *idxC), heapCnt);
                llvm::Value* slot = createEntryAlloca(ptrTy(), 1, 8);
                emitIfElse(b_, ok,
                           [&](llvm::IRBuilder<>& nb) {
                               nb.CreateStore(nb.CreateGEP(i8Ty(), heapBase, nb.getInt32(static_cast<int32_t>((*idxC) * 64))), slot);
                           },
                           [&](llvm::IRBuilder<>& nb) {
                               nb.CreateStore(llvm::Constant::getNullValue(ptrTy()), slot);
                           });
                h.lanes.push_back(b_.CreateLoad(ptrTy(), slot));
                setMapped(call, h);
                return true;
            }
            llvm::Value* idxVec = asVec(getMapped(arg(1)));
            llvm::Value* slot = createEntryAlloca(ptrTy(), 1, 8);
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* idxL = llvmExtract(b_, idxVec, l);
                llvm::Value* ok = b_.CreateICmp(llvm::CmpInst::ICMP_ULT, idxL, heapCnt);
                emitIfElse(b_, ok,
                           [&](llvm::IRBuilder<>& nb) {
                               llvm::Value* idx64 = nb.CreateZExt(idxL, i64Ty());
                               llvm::Value* bOff = nb.CreateMul(idx64, nb.getInt64(64));
                               nb.CreateStore(nb.CreateGEP(i8Ty(), heapBase, bOff), slot);
                           },
                           [&](llvm::IRBuilder<>& nb) {
                               nb.CreateStore(llvm::Constant::getNullValue(ptrTy()), slot);
                           });
                h.lanes.push_back(b_.CreateLoad(ptrTy(), slot));
            }
            setMapped(call, h);
            return true;
        }
        if (op == 160) {
            // createHandleForLib(op, Resource) -> trace Resource load/GEP back to global symbol
            llvm::Value* ptrVal = arg(1);
            if (auto* loadOp = llvm::dyn_cast<llvm::LoadInst>(ptrVal)) {
                ptrVal = loadOp->getPointerOperand();
            }
            llvm::Value* indexVal = nullptr;
            if (auto* gepOp = llvm::dyn_cast<llvm::GEPOperator>(ptrVal)) {
                if (gepOp->getNumIndices() > 0) {
                    indexVal = *(gepOp->idx_end() - 1);
                }
                ptrVal = gepOp->getPointerOperand();
            }
            std::string symName;
            if (auto* gv = llvm::dyn_cast<llvm::GlobalVariable>(ptrVal)) {
                symName = gv->getName().str();
            }
            const DxilResourceRecord* rec = nullptr;
            for (const auto& r : shader_.resources) {
                if (!symName.empty() && r.symbolName == symName) {
                    rec = &r;
                    break;
                }
            }
            if (!rec) {
                outError = "createHandleForLib: unresolved resource symbol '" + symName + "'";
                return false;
            }
            llvm::Value* regIdx = indexVal;
            if (!regIdx) {
                regIdx = cInt(i32Ty(), rec->lowerBound);
            } else {
                // GEP offset is relative to lowerBound
                if (auto cIdx = getConstIntValue(regIdx)) {
                    regIdx = cInt(i32Ty(), rec->lowerBound + static_cast<uint32_t>(*cIdx));
                }
            }
            setMapped(call,
                      resolveStaticBinding(rec->resClass, rec->space, rec->lowerBound,
                                           rec->resourceKind, rec->structuredStride, regIdx, resTy(), execMask));
            return true;
        }
        if (op == 216) {
            // annotateHandle(op, handle, ResourceProperties{i32 kindFlags, i32 strideOrFmt})
            SimdVal h = getMapped(arg(1));
            uint32_t kindFlags = 0, strideOrFmt = 0;
            llvm::SmallVector<uint64_t, 4> propFields;
            if (extractConstStructInts(arg(2), propFields) && propFields.size() >= 2) {
                kindFlags = static_cast<uint32_t>(propFields[0]);
                strideOrFmt = static_cast<uint32_t>(propFields[1]);
            } else {
                SimdVal rp = getMapped(arg(2));
                if (rp.fields.size() >= 2) {
                    kindFlags = static_cast<uint32_t>(getConstIntValue(rp.fields[0].vec).value_or(0));
                    strideOrFmt = static_cast<uint32_t>(getConstIntValue(rp.fields[1].vec).value_or(0));
                }
            }
            if ((kindFlags & 0xFFu) != 0) h.resourceKind = kindFlags & 0xFFu;
            if ((kindFlags & 0xFFu) == 12u && strideOrFmt > 0) h.stride = strideOrFmt;
            if ((kindFlags & 0x1000u) != 0) h.resClass = 1u;
            setMapped(call, h);
            return true;
        }

        // 3. CBuffer Load (59 legacy, 58)
        if (op == 59 || op == 58) {
            return packetizeCBufferLoad(call, op == 59, outError);
        }

        // 4. Buffer Load/Store (68, 69, 139, 140, 70)
        if (op == 68 || op == 139) return packetizeBufferLoad(call, op == 139, outError);
        if (op == 69 || op == 140) return packetizeBufferStore(call, op == 140, realMask, outError);
        if (op == 303) return packetizeBufferVectorLoad(call, outError);
        if (op == 304) return packetizeBufferVectorStore(call, realMask, outError);
        if (op == 311) { // FDot(a, b) -> scalar float
            SimdVal aSv = getMapped(arg(1));
            SimdVal bSv = getMapped(arg(2));
            llvm::Type* resElem = call->getType();
            llvm::Type* vT = vecTy(resElem);
            if (aSv.kind != SimdVal::Kind::Struct || bSv.kind != SimdVal::Kind::Struct || aSv.fields.empty()) {
                setMapped(call, makeZeroSimd(resElem));
                return true;
            }
            size_t n = std::min(aSv.fields.size(), bSv.fields.size());
            llvm::Value* acc = b_.CreateFMul(asVec(aSv.fields[0]), asVec(bSv.fields[0]));
            for (size_t i = 1; i < n; ++i) {
                llvm::Value* prod = b_.CreateFMul(asVec(aSv.fields[i]), asVec(bSv.fields[i]));
                acc = b_.CreateFAdd(acc, prod);
            }
            setMapped(call, SimdVal::makeVector(acc, resElem));
            return true;
        }
        if (op == 70) {
            // bufferUpdateCounter: extra_ptr holds atomic counter ptr if bound.
            // DXIL passes inc as i8 (+1 for IncrementCounter, -1 for DecrementCounter);
            // sign-extend from i8 and return pre-op value for increment, post-op value for decrement.
            SimdVal h = getMapped(arg(1));
            int32_t inc = static_cast<int32_t>(static_cast<int8_t>(getConstIntValue(arg(2)).value_or(1)));
            llvm::Value* outVec = splatInt(i32Ty(), 0);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* descP = laneValue(h, l);
                llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
                llvm::Value* isValid = b_.CreateAnd(mL, b_.CreateICmp(llvm::CmpInst::ICMP_NE, descP, nullP));
                llvm::Value* slot = createEntryAlloca(i32Ty(), 1, 4);
                b_.CreateStore(cInt(i32Ty(), 0), slot);
                emitIf(b_, isValid, [&](llvm::IRBuilder<>& nb) {
                    llvm::Value* ep = nb.CreateGEP(i8Ty(), descP, nb.getInt32(static_cast<int32_t>(offsetof(NacelleDescriptor, extra_ptr))));
                    llvm::Value* cptr = nb.CreateLoad(ptrTy(), ep);
                    llvm::Value* cValid = nb.CreateICmp(llvm::CmpInst::ICMP_NE, cptr, llvm::Constant::getNullValue(ptrTy()));
                    emitIf(nb, cValid, [&](llvm::IRBuilder<>& ib) {
                        llvm::Value* incVal = cInt(i32Ty(), inc);
                        llvm::Value* old = ib.CreateAtomicRMW(llvm::AtomicRMWInst::Add, cptr, incVal, llvm::MaybeAlign(), llvm::AtomicOrdering::SequentiallyConsistent);
                        llvm::Value* retVal = (inc < 0) ? ib.CreateAdd(old, incVal) : old;
                        ib.CreateStore(retVal, slot);
                    });
                });
                outVec = llvmInsert(b_, outVec, b_.CreateLoad(i32Ty(), slot), l);
            }
            setMapped(call, SimdVal::makeVector(outVec, i32Ty()));
            return true;
        }
        if (op == 71) { // CheckAccessFullyMapped(status) -> i1
            llvm::Value* stVec = asVec(getMapped(arg(1)));
            llvm::Value* mapped = b_.CreateICmp(llvm::CmpInst::ICMP_NE, stVec, splatInt(i32Ty(), 0));
            setMapped(call, SimdVal::makeVector(mapped, b_.getInt1Ty()));
            return true;
        }

        // 5. Atomics (78, 79) & Memory Fences without GroupSync (80, 244, 245)
        if (op == 78 || op == 79) return packetizeAtomicOp(call, op == 79, realMask, outError);
        if (op == 80 || op == 244 || op == 245) {
            b_.CreateFence(llvm::AtomicOrdering::AcquireRelease);
            return true;
        }

        // 6. Textures (60..67, 72, 73, 74, 81)
        if ((op >= 60 && op <= 67) || op == 72 || op == 73 || op == 74 || op == 81) {
            return packetizeTextureOp(call, op, execMask, realMask, outError);
        }

        // 7. VS / GS / HS / DS / PS Input/Output, Stream, & Discard
        if (op == 4) { // loadInput(sigId, row, col, gsVertexAxis)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal res;
            bool ok = false;
            if (hooks_.loadInputWithVertex) {
                SimdVal vAxis = (call->arg_size() > 4) ? getMapped(arg(4)) : makeZeroSimd(i32Ty());
                ok = hooks_.loadInputWithVertex(b_, sigId, row, col, vAxis, resTy(), res);
            } else if (hooks_.loadInput) {
                ok = hooks_.loadInput(b_, sigId, row, col, resTy(), res);
            }
            if (!ok) res = makeZeroSimd(resTy());
            setMapped(call, res);
            return true;
        }
        if (op == 5) { // storeOutput
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(arg(4));
            if (hooks_.storeOutput) {
                if (!hooks_.storeOutput(b_, sigId, row, col, val, realMask)) return false;
            }
            return true;
        }
        if (op == 97 || op == 98 || op == 99) { // emitStream / cutStream / emitThenCutStream
            uint32_t streamId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            if ((op == 97 || op == 99) && hooks_.emitStream) {
                if (!hooks_.emitStream(b_, streamId, realMask)) return false;
            }
            if ((op == 98 || op == 99) && hooks_.cutStream) {
                if (!hooks_.cutStream(b_, streamId, realMask)) return false;
            }
            return true;
        }
        if (op == 100) { // gsInstanceID
            SimdVal res;
            if (!hooks_.loadGSInstanceId || !hooks_.loadGSInstanceId(b_, resTy(), res)) {
                res = makeZeroSimd(resTy());
            }
            setMapped(call, res);
            return true;
        }
        if (op == 103) { // loadOutputControlPoint(sigId, row, col, cpId)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal cpId = getMapped(arg(4));
            SimdVal res;
            if (!hooks_.loadOutputControlPoint ||
                !hooks_.loadOutputControlPoint(b_, sigId, row, col, cpId, resTy(), res)) {
                res = makeZeroSimd(resTy());
            }
            setMapped(call, res);
            return true;
        }
        if (op == 104) { // loadPatchConstant(sigId, row, col)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal res;
            if (!hooks_.loadPatchConstant ||
                !hooks_.loadPatchConstant(b_, sigId, row, col, resTy(), res)) {
                res = makeZeroSimd(resTy());
            }
            setMapped(call, res);
            return true;
        }
        if (op == 105) { // domainLocation(col)
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal res;
            if (!hooks_.loadDomainLocation || !hooks_.loadDomainLocation(b_, col, resTy(), res)) {
                res = makeZeroSimd(resTy());
            }
            setMapped(call, res);
            return true;
        }
        if (op == 106) { // storePatchConstant(sigId, row, col, val)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(arg(4));
            if (hooks_.storePatchConstant) {
                if (!hooks_.storePatchConstant(b_, sigId, row, col, val, realMask)) return false;
            }
            return true;
        }
        if (op == 107) { // outputControlPointID
            SimdVal res;
            if (!hooks_.loadOutputControlPointId || !hooks_.loadOutputControlPointId(b_, resTy(), res)) {
                res = makeZeroSimd(resTy());
            }
            setMapped(call, res);
            return true;
        }
        if (op == 108) { // primitiveID
            SimdVal res;
            if (!hooks_.loadPrimitiveId || !hooks_.loadPrimitiveId(b_, resTy(), res)) {
                res = makeZeroSimd(resTy());
            }
            setMapped(call, res);
            return true;
        }
        if (op == 82) { // discard(cond)
            llvm::Value* condVec = asVec(getMapped(arg(1)));
            if (hooks_.onDiscard) {
                hooks_.onDiscard(b_, condVec, execMask, realMask);
            } else {
                llvm::Value* killed = b_.CreateAnd(execMask, condVec);
                llvm::Value* keep = b_.CreateXor(killed, allTrueMask());
                realMask = b_.CreateAnd(realMask, keep);
            }
            return true;
        }

        // 8. Wave & Quad Intrinsics (110..121, 123, 135, 136, 166)
        if ((op >= 110 && op <= 121) || op == 123 || op == 135 || op == 136 || op == 166) {
            return packetizeWaveOp(call, op, realMask, outError);
        }

        // 8a. DXR Raytracing Pipeline (141..159, 161, 213)
        if ((op >= 141 && op <= 159) || op == 161 || op == 213) {
            return packetizeDxrOp(call, op, execMask, outError);
        }

        // 8b. Mesh & Amplification Shader Intrinsics (168..173)
        if (op == 168) { // SetMeshOutputCounts(op, numVertices, numPrimitives)
            SimdVal nv = getMapped(arg(1));
            SimdVal np = getMapped(arg(2));
            if (hooks_.setMeshOutputCounts) {
                if (!hooks_.setMeshOutputCounts(b_, nv, np, realMask)) return false;
            }
            return true;
        }
        if (op == 169) { // EmitIndices(op, primIdx, v0, v1, v2)
            SimdVal pIdx = getMapped(arg(1));
            SimdVal v0 = getMapped(arg(2));
            SimdVal v1 = getMapped(arg(3));
            SimdVal v2 = (call->arg_size() > 4) ? getMapped(arg(4)) : makeZeroSimd(i32Ty());
            if (hooks_.emitIndices) {
                if (!hooks_.emitIndices(b_, pIdx, v0, v1, v2, realMask)) return false;
            }
            return true;
        }
        if (op == 170) { // GetMeshPayload(op) -> ptr
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            sv.lanes.push_back(hooks_.meshPayloadBase ? hooks_.meshPayloadBase
                                                      : llvm::Constant::getNullValue(ptrTy()));
            setMapped(call, sv);
            return true;
        }
        if (op == 171) { // StoreVertexOutput(op, sigId, row, col, val, vertIdx)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(arg(4));
            SimdVal vIdx = getMapped(arg(5));
            if (hooks_.storeVertexOutput) {
                if (!hooks_.storeVertexOutput(b_, sigId, row, col, val, vIdx, realMask)) return false;
            }
            return true;
        }
        if (op == 172) { // StorePrimitiveOutput(op, sigId, row, col, val, primIdx)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(arg(4));
            SimdVal pIdx = getMapped(arg(5));
            if (hooks_.storePrimitiveOutput) {
                if (!hooks_.storePrimitiveOutput(b_, sigId, row, col, val, pIdx, realMask)) return false;
            }
            return true;
        }
        if (op == 173) { // DispatchMesh(op, countX, countY, countZ, payloadPtr)
            SimdVal cx = getMapped(arg(1));
            SimdVal cy = getMapped(arg(2));
            SimdVal cz = getMapped(arg(3));
            SimdVal pp = (call->arg_size() > 4) ? getMapped(arg(4)) : makeZeroSimd(ptrTy());
            if (hooks_.dispatchMesh) {
                if (!hooks_.dispatchMesh(b_, cx, cy, cz, pp, realMask)) return false;
            }
            return true;
        }

        // 8c. Inline Raytracing (RayQuery) (178..215, 258)
        if (op == 258 || (op >= 178 && op <= 215)) {
            return packetizeRayQueryOp(call, op, execMask);
        }

        // 9. Math, Bitwise & Quad Derivatives (6..56, 83..86)
        return packetizeMathOp(call, op, outError);
    }

    BufferDescInfo loadBufferDescInfo(
        llvm::IRBuilder<>& nb,
        const SimdVal& h,
        llvm::Value* descPtr,
        bool needElemStride,
        bool needStride64) {
        llvm::BasicBlock* entryBlock = getFuncEntryBlock();
        auto* descInst = llvm::dyn_cast<llvm::Instruction>(descPtr);
        const bool canCache = (entryBlock != nullptr) && (llvm::isa<llvm::Argument>(descPtr) || (descInst && descInst->getParent() == entryBlock));
        DescInfoCacheKey key{descPtr, h.isRootConstants, h.stride, needElemStride, needStride64};
        if (canCache) {
            for (const auto& entry : *descCache_) {
                if (entry.first == key) return entry.second;
            }
        }

        llvm::BasicBlock* savedBlock = nb.GetInsertBlock();
        const bool hoistToEntry = canCache && (savedBlock != entryBlock);
        std::optional<llvm::IRBuilderBase::InsertPointGuard> guard;
        llvm::BasicBlock* contBB = nullptr;
        if (hoistToEntry) {
            guard.emplace(nb);
            llvm::BasicBlock::iterator splitPt = entryBlock->getFirstInsertionPt();
            if (auto* defInst = llvm::dyn_cast<llvm::Instruction>(descPtr)) {
                splitPt = std::next(defInst->getIterator());
            }
            if (!h.isRootConstants) {
                contBB = entryBlock->splitBasicBlock(splitPt, "desc.cont");
                entryBlock->getTerminator()->eraseFromParent();
                nb.SetInsertPoint(entryBlock);
            } else {
                nb.SetInsertPoint(entryBlock, splitPt);
            }
        }

        BufferDescInfo info;
        info.descP = descPtr;
        if (h.isRootConstants) {
            info.basePtr = descPtr;
            info.byteSize = cInt(i64Ty(), UINT64_MAX);
            info.isReserved = cInt(i1Ty(), 0);
            llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
            info.okPtr = nb.CreateICmp(llvm::CmpInst::ICMP_NE, descPtr, nullPtr);
            if (needElemStride) info.elemStride = cInt(i32Ty(), 0);
            if (needStride64) info.stride64 = cInt(i64Ty(), h.stride);
            return info;
        }

        bool loadEs = needElemStride || (needStride64 && h.stride == 0);
        llvm::Value* nullPtr = llvm::Constant::getNullValue(ptrTy());
        llvm::Value* isNull = nb.CreateICmp(llvm::CmpInst::ICMP_EQ, descPtr, nullPtr);
        llvm::Value* bpSlot = createEntryAlloca(ptrTy(), 1, 8);
        llvm::Value* bsSlot = createEntryAlloca(i64Ty(), 1, 8);
        llvm::Value* vfSlot = createEntryAlloca(i32Ty(), 1, 4);
        llvm::Value* esSlot = loadEs ? createEntryAlloca(i32Ty(), 1, 4) : nullptr;
        emitIfElse(nb, isNull,
                   [&](llvm::IRBuilder<>& ib) {
                       ib.CreateStore(nullPtr, bpSlot);
                       ib.CreateStore(cInt(i64Ty(), 0), bsSlot);
                       ib.CreateStore(cInt(i32Ty(), 0), vfSlot);
                       if (loadEs) ib.CreateStore(cInt(i32Ty(), 0), esSlot);
                   },
                   [&](llvm::IRBuilder<>& ib) {
                       auto* p = ib.CreateLoad(ptrTy(), descPtr);
                       p->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(ib.getContext(), {}));
                       ib.CreateStore(p, bpSlot);
                       llvm::Value* bsp = ib.CreateGEP(i8Ty(), descPtr, ib.getInt32(static_cast<int32_t>(offsetof(NacelleDescriptor, byte_size))));
                       auto* bs = ib.CreateLoad(i64Ty(), bsp);
                       bs->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(ib.getContext(), {}));
                       ib.CreateStore(bs, bsSlot);
                       llvm::Value* vfp = ib.CreateGEP(i8Ty(), descPtr, ib.getInt32(static_cast<int32_t>(offsetof(NacelleDescriptor, view_flags))));
                       auto* vf = ib.CreateLoad(i32Ty(), vfp);
                       vf->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(ib.getContext(), {}));
                       ib.CreateStore(vf, vfSlot);
                       if (loadEs) {
                           llvm::Value* esp = ib.CreateGEP(i8Ty(), descPtr, ib.getInt32(static_cast<int32_t>(offsetof(NacelleDescriptor, elem_stride))));
                           auto* es = ib.CreateLoad(i32Ty(), esp);
                           es->setMetadata(llvm::LLVMContext::MD_invariant_load, llvm::MDNode::get(ib.getContext(), {}));
                           ib.CreateStore(es, esSlot);
                       }
                   });
        info.basePtr = nb.CreateLoad(ptrTy(), bpSlot);
        info.byteSize = nb.CreateLoad(i64Ty(), bsSlot);
        llvm::Value* vf = nb.CreateLoad(i32Ty(), vfSlot);
        llvm::Value* flag = nb.CreateAnd(vf, cInt(i32Ty(), kDescViewFlagReservedBuffer));
        info.isReserved = nb.CreateICmp(llvm::CmpInst::ICMP_NE, flag, cInt(i32Ty(), 0));
        info.okPtr = nb.CreateICmp(llvm::CmpInst::ICMP_NE, info.basePtr, nullPtr);
        llvm::Value* loadedEs = loadEs ? nb.CreateLoad(i32Ty(), esSlot) : nullptr;
        if (needElemStride) info.elemStride = loadedEs;
        if (needStride64) {
            info.stride64 = (h.stride > 0)
                                ? cInt(i64Ty(), h.stride)
                                : nb.CreateZExt(loadedEs, i64Ty());
        }
        if (contBB) {
            nb.CreateBr(contBB);
        }
        if (canCache) {
            descCache_->push_back({key, info});
        }
        return info;
    }

    llvm::Value* resolveReservedBufferElemPtr(
        llvm::IRBuilder<>& nb,
        llvm::Value* descP,
        llvm::Value* tileTablePtr,
        llvm::Value* cOff,
        uint32_t elemBytes) {
        auto ci64 = [&](uint64_t v) { return cInt(i64Ty(), v); };
        auto loadU32 = [&](int64_t fOff) -> llvm::Value* {
            llvm::Value* fp = nb.CreateGEP(i8Ty(), descP, nb.getInt32(static_cast<int32_t>(fOff)));
            return nb.CreateLoad(i32Ty(), fp);
        };
        llvm::Value* lo = nb.CreateZExt(loadU32(offsetof(NacelleDescriptor, row_pitch)), i64Ty());
        llvm::Value* hi = nb.CreateZExt(loadU32(offsetof(NacelleDescriptor, depth_or_array)), i64Ty());
        llvm::Value* viewOff = nb.CreateOr(nb.CreateShl(hi, ci64(32)), lo);
        llvm::Value* resOff = nb.CreateAdd(viewOff, cOff);
        llvm::Value* tileIdx = nb.CreateLShr(resOff, ci64(16));
        llvm::Value* endTileIdx =
            (elemBytes > 1)
                ? nb.CreateLShr(nb.CreateAdd(resOff, ci64(elemBytes - 1)), ci64(16))
                : tileIdx;
        llvm::Value* totalTiles64 =
            nb.CreateZExt(loadU32(offsetof(NacelleDescriptor, height)), i64Ty());
        llvm::Value* validTile = nb.CreateAnd(nb.CreateICmp(llvm::CmpInst::ICMP_ULT, tileIdx, totalTiles64), nb.CreateICmp(llvm::CmpInst::ICMP_EQ, tileIdx, endTileIdx));

        llvm::Value* epSlot = createEntryAlloca(ptrTy(), 1, 8);
        nb.CreateStore(llvm::Constant::getNullValue(ptrTy()), epSlot);
        emitIf(nb, validTile, [&](llvm::IRBuilder<>& tb) {
            llvm::Value* slotByteOff = tb.CreateShl(tileIdx, ci64(3));
            llvm::Value* slotPtr = tb.CreateGEP(i8Ty(), tileTablePtr, slotByteOff);
            llvm::Value* tileBase = tb.CreateLoad(ptrTy(), slotPtr);
            llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
            llvm::Value* hasPage = tb.CreateICmp(llvm::CmpInst::ICMP_NE, tileBase, nullP);
            emitIf(tb, hasPage, [&](llvm::IRBuilder<>& pb) {
                llvm::Value* inTile = pb.CreateAnd(resOff, ci64(0xFFFF));
                pb.CreateStore(pb.CreateGEP(i8Ty(), tileBase, inTile), epSlot);
            });
        });
        return nb.CreateLoad(ptrTy(), epSlot);
    }

    std::pair<llvm::Value*, llvm::Value*> emitScalarBufferComponentLoad(
        llvm::Type* elemTy,
        uint32_t elemBytes,
        llvm::Value* descP,
        llvm::Value* basePtr,
        llvm::Value* isReserved,
        llvm::Value* inBounds,
        llvm::Value* cOff) {
        llvm::Value* vSlot = createEntryAlloca(elemTy, 1, 16);
        llvm::Value* mSlot = createEntryAlloca(i1Ty(), 1, 1);
        llvm::Value* zeroVal = elemTy->isFloatingPointTy() ? cFlt(elemTy, 0.0) : cInt(elemTy, 0);
        b_.CreateStore(zeroVal, vSlot);
        b_.CreateStore(cInt(i1Ty(), 1), mSlot);
        emitIf(b_, inBounds, [&](llvm::IRBuilder<>& nb) {
            emitIfElse(nb, isReserved,
                       [&](llvm::IRBuilder<>& rb) {
                           llvm::Value* ep = resolveReservedBufferElemPtr(rb, descP, basePtr, cOff, elemBytes);
                           llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
                           llvm::Value* mapped = rb.CreateICmp(llvm::CmpInst::ICMP_NE, ep, nullP);
                           rb.CreateStore(mapped, mSlot);
                           emitIf(rb, mapped, [&](llvm::IRBuilder<>& lb) {
                               lb.CreateStore(lb.CreateLoad(elemTy, ep), vSlot);
                           });
                       },
                       [&](llvm::IRBuilder<>& rb) {
                           llvm::Value* ep = rb.CreateGEP(i8Ty(), basePtr, cOff);
                           rb.CreateStore(rb.CreateLoad(elemTy, ep), vSlot);
                           rb.CreateStore(cInt(i1Ty(), 1), mSlot);
                       });
        });
        return {b_.CreateLoad(elemTy, vSlot), b_.CreateLoad(i1Ty(), mSlot)};
    }

    void emitScalarBufferComponentStore(
        uint32_t elemBytes,
        llvm::Value* descP,
        llvm::Value* basePtr,
        llvm::Value* isReserved,
        llvm::Value* doStore,
        llvm::Value* cOff,
        llvm::Value* valScalar) {
        emitIf(b_, doStore, [&](llvm::IRBuilder<>& nb) {
            emitIfElse(nb, isReserved,
                       [&](llvm::IRBuilder<>& rb) {
                           llvm::Value* ep = resolveReservedBufferElemPtr(rb, descP, basePtr, cOff, elemBytes);
                           llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
                           llvm::Value* mapped = rb.CreateICmp(llvm::CmpInst::ICMP_NE, ep, nullP);
                           emitIf(rb, mapped, [&](llvm::IRBuilder<>& sb) {
                               sb.CreateStore(valScalar, ep);
                           });
                       },
                       [&](llvm::IRBuilder<>& rb) {
                           llvm::Value* ep = rb.CreateGEP(i8Ty(), basePtr, cOff);
                           rb.CreateStore(valScalar, ep);
                       });
        });
    }

    std::pair<llvm::Value*, llvm::Value*> emitReservedElemLoad(
        llvm::IRBuilder<>& rb,
        llvm::Type* elemTy,
        uint32_t elemBytes,
        llvm::Value* descP,
        llvm::Value* tileTablePtr,
        llvm::Value* inBounds,
        llvm::Value* cOff) {
        llvm::Value* zeroVal = elemTy->isFloatingPointTy() ? cFlt(elemTy, 0.0) : cInt(elemTy, 0);
        llvm::Value* vSlot = createEntryAlloca(elemTy, 1, 16);
        llvm::Value* mSlot = createEntryAlloca(i1Ty(), 1, 1);
        rb.CreateStore(zeroVal, vSlot);
        rb.CreateStore(cInt(i1Ty(), 1), mSlot);
        emitIf(rb, inBounds, [&](llvm::IRBuilder<>& tb) {
            llvm::Value* ep = resolveReservedBufferElemPtr(tb, descP, tileTablePtr, cOff, elemBytes);
            llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
            llvm::Value* mapped = tb.CreateICmp(llvm::CmpInst::ICMP_NE, ep, nullP);
            tb.CreateStore(mapped, mSlot);
            emitIf(tb, mapped, [&](llvm::IRBuilder<>& lb) {
                lb.CreateStore(lb.CreateLoad(elemTy, ep), vSlot);
            });
        });
        return {rb.CreateLoad(elemTy, vSlot), rb.CreateLoad(i1Ty(), mSlot)};
    }

    void emitReservedElemStore(
        llvm::IRBuilder<>& rb,
        uint32_t elemBytes,
        llvm::Value* descP,
        llvm::Value* tileTablePtr,
        llvm::Value* doStore,
        llvm::Value* cOff,
        llvm::Value* valScalar) {
        emitIf(rb, doStore, [&](llvm::IRBuilder<>& tb) {
            llvm::Value* ep = resolveReservedBufferElemPtr(tb, descP, tileTablePtr, cOff, elemBytes);
            llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
            llvm::Value* mapped = tb.CreateICmp(llvm::CmpInst::ICMP_NE, ep, nullP);
            emitIf(tb, mapped, [&](llvm::IRBuilder<>& sb) {
                sb.CreateStore(valScalar, ep);
            });
        });
    }

    llvm::Value* computeBufferByteOffsetScalar(
        llvm::IRBuilder<>& nb, const SimdVal& h, llvm::Value* stride64,
        llvm::Value* idxL, llvm::Value* elemOffL, bool isRawOp, bool hasElemOff) {
        llvm::Value* idx64 = nb.CreateZExt(idxL, i64Ty());
        if (isRawOp) {
            if (!hasElemOff) return idx64;
            llvm::Value* e64 = nb.CreateZExt(elemOffL, i64Ty());
            return nb.CreateAdd(nb.CreateMul(idx64, stride64), e64);
        }
        if (hasElemOff) {
            llvm::Value* e64 = nb.CreateZExt(elemOffL, i64Ty());
            return nb.CreateAdd(nb.CreateMul(idx64, stride64), e64);
        }
        if (h.resourceKind == 11) return idx64;
        return nb.CreateMul(idx64, stride64);
    }

    bool tryGetUniformScalar(llvm::Value* origVal, llvm::Value*& outScalar) {
        if (auto c = getConstIntValue(origVal)) {
            outScalar = cInt(i32Ty(), static_cast<int64_t>(*c));
            return true;
        }
        SimdVal sv = getMapped(origVal);
        if (sv.isUniform()) {
            outScalar = !sv.lanes.empty()
                            ? sv.lanes[0]
                            : llvmExtract(b_, sv.vec, 0u);
            return true;
        }
        if (sv.kind == SimdVal::Kind::Vector) {
            if (auto* shuf = llvm::dyn_cast<llvm::ShuffleVectorInst>(sv.vec)) {
                auto mask = shuf->getShuffleMask();
                if (!mask.empty() && llvm::all_of(mask, [](int idx) { return idx == 0; })) {
                    if (auto* ins = llvm::dyn_cast<llvm::InsertElementInst>(shuf->getOperand(0))) {
                        if (getConstIntValue(ins->getOperand(2)) == 0) {
                            outScalar = ins->getOperand(1);
                            return true;
                        }
                    }
                }
            }
            if (auto* cst = llvm::dyn_cast<llvm::Constant>(sv.vec)) {
                if (llvm::Constant* sp = cst->getSplatValue()) {
                    outScalar = sp;
                    return true;
                }
            }
        }
        return false;
    }

    std::pair<llvm::SmallVector<llvm::Value*, 16>, llvm::Value*> emitCoalescedUniformBufferLoad(
        llvm::Type* elemTy,
        uint32_t elemBytes,
        const BufferDescInfo& info,
        bool isRootConstants,
        bool nonStraddling,
        llvm::Value* byteOff,
        unsigned totalComps,
        llvm::ArrayRef<unsigned> activeComps) {
        llvm::Value* zeroVec = elemTy->isFloatingPointTy() ? splatFlt(elemTy, 0.0) : splatInt(elemTy, 0);
        if (activeComps.empty()) {
            return {llvm::SmallVector<llvm::Value*, 16>(totalComps, zeroVec), splatInt(i32Ty(), 1)};
        }

        llvm::Value* scalarZero = elemTy->isFloatingPointTy() ? cFlt(elemTy, 0.0) : cInt(elemTy, 0);

        auto compInBounds = [&](llvm::IRBuilder<>& zb, unsigned c) -> std::pair<llvm::Value*, llvm::Value*> {
            llvm::Value* cOff = (c == 0)
                                   ? byteOff
                                   : zb.CreateAdd(byteOff, cInt(i64Ty(), uint64_t(c) * elemBytes));
            if (isRootConstants) return {cOff, info.okPtr};
            llvm::Value* cEnd = zb.CreateAdd(cOff, cInt(i64Ty(), elemBytes));
            llvm::Value* ok = zb.CreateAnd(info.okPtr, zb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
            if (info.elemStride) {
                llvm::Value* elemEnd = cInt(i32Ty(), (c + 1) * elemBytes);
                llvm::Value* strideZero = zb.CreateICmp(llvm::CmpInst::ICMP_EQ, info.elemStride, cInt(i32Ty(), 0));
                llvm::Value* inElem = zb.CreateICmp(llvm::CmpInst::ICMP_ULE, elemEnd, info.elemStride);
                ok = zb.CreateAnd(ok, zb.CreateOr(strideZero, inElem));
            }
            return {cOff, ok};
        };

        llvm::SmallVector<llvm::Value*, 16> valSlots;
        valSlots.reserve(activeComps.size());
        for (size_t i = 0; i < activeComps.size(); ++i) {
            llvm::Value* sl = createEntryAlloca(elemTy, 1, 16);
            b_.CreateStore(scalarZero, sl);
            valSlots.push_back(sl);
        }

        auto emitNormalScalarLoads = [&](llvm::IRBuilder<>& nb) {
            auto [lastOff, fullOk] = compInBounds(nb, activeComps.back());
            (void)lastOff;
            emitIfElse(nb, fullOk,
                       [&](llvm::IRBuilder<>& tb) {
                           llvm::Value* baseElemPtr = tb.CreateGEP(i8Ty(), info.basePtr, byteOff);
                           for (size_t i = 0; i < activeComps.size(); ++i) {
                               unsigned c = activeComps[i];
                               llvm::Value* ep = (c == 0)
                                                    ? baseElemPtr
                                                    : tb.CreateGEP(i8Ty(), baseElemPtr, tb.getInt32(static_cast<int32_t>(c * elemBytes)));
                               tb.CreateStore(tb.CreateLoad(elemTy, ep), valSlots[i]);
                           }
                       },
                       [&](llvm::IRBuilder<>& eb) {
                           if (activeComps.size() == 1 || nonStraddling || isRootConstants) {
                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                   eb.CreateStore(scalarZero, valSlots[i]);
                               }
                               return;
                           }
                           auto [firstOff, firstOk] = compInBounds(eb, activeComps.front());
                           emitIf(eb, firstOk, [&](llvm::IRBuilder<>& pb) {
                               llvm::Value* ep0 = pb.CreateGEP(i8Ty(), info.basePtr, firstOff);
                               pb.CreateStore(pb.CreateLoad(elemTy, ep0), valSlots[0]);
                               for (size_t idx = 1; idx + 1 < activeComps.size(); ++idx) {
                                   auto [mOff, mOk] = compInBounds(pb, activeComps[idx]);
                                   emitIf(pb, mOk, [&](llvm::IRBuilder<>& mb) {
                                       llvm::Value* mep = mb.CreateGEP(i8Ty(), info.basePtr, mOff);
                                       mb.CreateStore(mb.CreateLoad(elemTy, mep), valSlots[idx]);
                                   });
                               }
                           });
                       });
        };

        llvm::SmallVector<llvm::Value*, 16> outVecs(totalComps, zeroVec);
        if (isRootConstants) {
            emitNormalScalarLoads(b_);
            for (size_t i = 0; i < activeComps.size(); ++i) {
                llvm::Value* sc = b_.CreateLoad(elemTy, valSlots[i]);
                outVecs[activeComps[i]] = llvmSplat(b_, elemTy, sc);
            }
            return {outVecs, splatInt(i32Ty(), 1)};
        }

        llvm::Value* statusSlot = createEntryAlloca(i32Ty(), 1, 4);
        b_.CreateStore(cInt(i32Ty(), 1), statusSlot);

        auto emitPerCompReserved = [&](llvm::IRBuilder<>& rb) {
            llvm::Value* allMapped = cInt(i1Ty(), 1);
            for (size_t i = 0; i < activeComps.size(); ++i) {
                unsigned c = activeComps[i];
                auto [cOff, ok] = compInBounds(rb, c);
                auto [v, mapped] = emitReservedElemLoad(
                    rb, elemTy, elemBytes, info.descP, info.basePtr, ok, cOff);
                rb.CreateStore(v, valSlots[i]);
                allMapped = rb.CreateAnd(allMapped, mapped);
            }
            rb.CreateStore(rb.CreateZExt(allMapped, i32Ty()), statusSlot);
        };

        emitIfElse(b_, info.isReserved,
                   [&](llvm::IRBuilder<>& rb) {
                       if (activeComps.size() <= 1) {
                           emitPerCompReserved(rb);
                           return;
                       }
                       auto [lastOff, fullOk] = compInBounds(rb, activeComps.back());
                       (void)lastOff;
                       uint32_t spanBytes = (activeComps.back() + 1u) * elemBytes;
                       emitIfElse(rb, fullOk,
                                  [&](llvm::IRBuilder<>& tb) {
                                      llvm::Value* spanPtr = resolveReservedBufferElemPtr(
                                          tb, info.descP, info.basePtr, byteOff, spanBytes);
                                      llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
                                      llvm::Value* samePageMapped = tb.CreateICmp(llvm::CmpInst::ICMP_NE, spanPtr, nullP);
                                      emitIfElse(tb, samePageMapped,
                                                 [&](llvm::IRBuilder<>& lb) {
                                                     for (size_t i = 0; i < activeComps.size(); ++i) {
                                                         unsigned c = activeComps[i];
                                                         llvm::Value* ep = (c == 0)
                                                                              ? spanPtr
                                                                              : lb.CreateGEP(i8Ty(), spanPtr, lb.getInt32(static_cast<int32_t>(c * elemBytes)));
                                                         lb.CreateStore(lb.CreateLoad(elemTy, ep), valSlots[i]);
                                                     }
                                                     lb.CreateStore(cInt(i32Ty(), 1), statusSlot);
                                                 },
                                                 [&](llvm::IRBuilder<>& lb) {
                                                     emitPerCompReserved(lb);
                                                 });
                                  },
                                  [&](llvm::IRBuilder<>& tb) {
                                      emitPerCompReserved(tb);
                                  });
                   },
                   [&](llvm::IRBuilder<>& rb) {
                       emitNormalScalarLoads(rb);
                       rb.CreateStore(cInt(i32Ty(), 1), statusSlot);
                   });

        for (size_t i = 0; i < activeComps.size(); ++i) {
            llvm::Value* sc = b_.CreateLoad(elemTy, valSlots[i]);
            outVecs[activeComps[i]] = llvmSplat(b_, elemTy, sc);
        }
        llvm::Value* statusVec = llvmSplat(b_, i32Ty(), b_.CreateLoad(i32Ty(), statusSlot));
        return {outVecs, statusVec};
    }

    llvm::Value* computeBufferByteOffsetVec(
        llvm::IRBuilder<>& nb, const SimdVal& h, llvm::Value* stride64Vec,
        llvm::Value* idxVec, llvm::Value* elemOffVec, bool isRawOp, bool hasElemOff) {
        llvm::Value* idx64Vec = nb.CreateZExt(idxVec, vecTy(i64Ty()));
        if (isRawOp) {
            if (!hasElemOff) return idx64Vec;
            llvm::Value* e64Vec = nb.CreateZExt(elemOffVec, vecTy(i64Ty()));
            return nb.CreateAdd(nb.CreateMul(idx64Vec, stride64Vec), e64Vec);
        }
        if (hasElemOff) {
            llvm::Value* e64Vec = nb.CreateZExt(elemOffVec, vecTy(i64Ty()));
            return nb.CreateAdd(nb.CreateMul(idx64Vec, stride64Vec), e64Vec);
        }
        if (h.resourceKind == 11) return idx64Vec;
        return nb.CreateMul(idx64Vec, stride64Vec);
    }

    llvm::Value* computeBufferBoundsMaskVec(
        llvm::IRBuilder<>& nb, llvm::Value* byteOffVec, llvm::Value* byteSizeVec,
        llvm::Value* okPtrVec, llvm::Value* elemStrideOkVec, uint32_t c, uint32_t elemBytes) {
        llvm::Value* endBytesSplat = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W_), llvm::ConstantInt::get(i64Ty(), uint64_t(c + 1) * elemBytes));
        llvm::Value* cEndVec = nb.CreateAdd(byteOffVec, endBytesSplat);
        llvm::Value* inBoundsVec = nb.CreateAnd(okPtrVec, nb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEndVec, byteSizeVec));
        if (elemStrideOkVec) {
            inBoundsVec = nb.CreateAnd(inBoundsVec, elemStrideOkVec);
        }
        return inBoundsVec;
    }

    std::pair<llvm::Value*, llvm::Value*> checkContiguousByteOffsets(
        llvm::IRBuilder<>& nb, llvm::Value* byteOffVec, uint32_t elemBytes) {
        llvm::Value* lane0Off = llvmExtract(nb, byteOffVec, static_cast<uint32_t>(0));
        if (W_ <= 1) {
            return {llvm::ConstantInt::get(i1Ty(), 1), lane0Off};
        }
        llvm::SmallVector<llvm::Constant*, 16> stepConsts;
        for (uint32_t l = 0; l < W_; ++l) {
            stepConsts.push_back(llvm::ConstantInt::get(i64Ty(), static_cast<uint64_t>(l) * elemBytes));
        }
        llvm::Value* stepConst = llvm::ConstantVector::get(stepConsts);
        llvm::Value* expectedVec = nb.CreateAdd(llvmSplat(nb, llvm::cast<llvm::FixedVectorType>(vecTy(i64Ty()))->getElementType(), lane0Off), stepConst);
        llvm::Value* eqVec = nb.CreateICmp(llvm::CmpInst::ICMP_EQ, byteOffVec, expectedVec);
        llvm::Value* isContig = nb.CreateAndReduce(eqVec);
        return {isContig, lane0Off};
    }

    llvm::Value* computeScalarElemStrideOkVec(
        llvm::IRBuilder<>& nb, llvm::Value* elemStride, uint32_t c, uint32_t elemBytes) {
        if (!elemStride) return {};
        llvm::Value* elemEnd = llvm::ConstantInt::get(i32Ty(), (c + 1) * elemBytes);
        llvm::Value* strideZero = nb.CreateICmp(llvm::CmpInst::ICMP_EQ, elemStride, llvm::ConstantInt::get(i32Ty(), 0));
        llvm::Value* inElem = nb.CreateICmp(llvm::CmpInst::ICMP_ULE, elemEnd, elemStride);
        llvm::Value* ok = nb.CreateOr(strideZero, inElem);
        return llvmSplat(nb, llvm::cast<llvm::FixedVectorType>(vecTy(i1Ty()))->getElementType(), ok);
    }

    llvm::Value* computeVecElemStrideOkVec(
        llvm::IRBuilder<>& nb, llvm::Value* elemStrideVec, uint32_t c, uint32_t elemBytes) {
        if (!elemStrideVec) return {};
        auto splatI32 = [&](uint32_t v) {
            return llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W_), llvm::ConstantInt::get(i32Ty(), v));
        };
        llvm::Value* strideZero = nb.CreateICmp(llvm::CmpInst::ICMP_EQ, elemStrideVec, splatI32(0));
        llvm::Value* inElem = nb.CreateICmp(llvm::CmpInst::ICMP_ULE, splatI32((c + 1) * elemBytes), elemStrideVec);
        return nb.CreateOr(strideZero, inElem);
    }

    bool packetizeCBufferLoad(llvm::CallInst* call, bool isLegacy, std::string& outError) {
        SimdVal h = getMapped(call->getArgOperand(1));
        llvm::Value* idxArg = call->getArgOperand(2);
        auto st = llvm::dyn_cast<llvm::StructType>(call->getType());
        if (!st) {
            outError = "cbufferLoad expected struct return type";
            return false;
        }
        const unsigned numFields = st->getNumElements();
        llvm::Type* elemTy = st->getElementType(0);
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy->getScalarSizeInBits() / 8);

        uint32_t usedMask = 0;
        bool allExtract = true;
        for (llvm::User* user : call->users()) {
            if (auto* ev = llvm::dyn_cast<llvm::ExtractValueInst>(user)) {
                auto pos = ev->getIndices();
                if (!pos.empty() && pos[0] < 32) {
                    usedMask |= (1u << pos[0]);
                    continue;
                }
            }
            allExtract = false;
            break;
        }
        if (!allExtract) {
            usedMask = (numFields >= 32) ? UINT32_MAX : ((1u << numFields) - 1u);
        }
        llvm::SmallVector<unsigned, 8> activeComps;
        for (unsigned f = 0; f < numFields; ++f) {
            if (f >= 32 || (usedMask & (1u << f)) != 0) activeComps.push_back(f);
        }

        llvm::Value* uniformIdx;
        if (h.isUniform() && tryGetUniformScalar(idxArg, uniformIdx)) {
            llvm::Value* byteOff;
            if (auto idxConst = getConstIntValue(idxArg)) {
                byteOff = cInt(i64Ty(), isLegacy ? (*idxConst * 16) : *idxConst);
            } else {
                llvm::Value* idx64 = b_.CreateZExt(uniformIdx, i64Ty());
                byteOff = isLegacy ? static_cast<llvm::Value*>(b_.CreateMul(idx64, cInt(i64Ty(), 16))) : idx64;
            }
            BufferDescInfo info = loadBufferDescInfo(b_, h, h.lanes[0], /*needElemStride=*/false, /*needStride64=*/false);
            auto [compVecs, _] = emitCoalescedUniformBufferLoad(
                elemTy, elemBytes, info, h.isRootConstants, /*nonStraddling=*/false, byteOff, numFields, activeComps);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = st;
            for (unsigned f = 0; f < numFields; ++f) {
                outSt.fields.push_back(SimdVal::makeVector(compVecs[f], elemTy));
            }
            setMapped(call, outSt);
            return true;
        }

        llvm::Value* zeroElemVec = elemTy->isFloatingPointTy() ? splatFlt(elemTy, 0.0) : splatInt(elemTy, 0);
        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        if (activeComps.empty()) {
            for (unsigned f = 0; f < numFields; ++f) {
                outSt.fields.push_back(SimdVal::makeVector(zeroElemVec, elemTy));
            }
            setMapped(call, outSt);
            return true;
        }

        llvm::Value* idxVec = asVec(getMapped(idxArg));
        if (h.isUniform()) {
            BufferDescInfo info = loadBufferDescInfo(b_, h, h.lanes[0], /*needElemStride=*/false, /*needStride64=*/false);
            llvm::Value* idx64Vec = b_.CreateZExt(idxVec, vecTy(i64Ty()));
            llvm::Value* off64Vec = isLegacy ? b_.CreateMul(idx64Vec, splatInt(i64Ty(), 16)) : idx64Vec;

            auto emitGatherCBuffer = [&](llvm::IRBuilder<>& gb) -> llvm::SmallVector<llvm::Value*, 8> {
                llvm::Value* okPtrVec = llvmSplat(gb, i1Ty(), info.okPtr);
                llvm::Value* byteSizeVec = h.isRootConstants
                                              ? nullptr
                                              : llvmSplat(gb, i64Ty(), info.byteSize);
                llvm::SmallVector<llvm::Value*, 8> gathered;
                gathered.reserve(activeComps.size());
                for (unsigned f : activeComps) {
                    llvm::Value* inBoundsVec = h.isRootConstants
                                                  ? okPtrVec
                                                  : computeBufferBoundsMaskVec(
                                                        gb, off64Vec, byteSizeVec, okPtrVec, nullptr, f, elemBytes);
                    llvm::Value* compBase = (f == 0)
                                               ? info.basePtr
                                               : gb.CreateGEP(i8Ty(), info.basePtr, gb.getInt32(static_cast<int32_t>(f * elemBytes)));
                    llvm::Value* ptrs = gb.CreateGEP(i8Ty(), compBase, off64Vec);
                    gathered.push_back(gb.CreateMaskedGather(vecTy(elemTy), ptrs, llvm::Align(1), inBoundsVec, zeroElemVec));
                }
                return gathered;
            };

            llvm::SmallVector<llvm::Value*, 8> activeVecs;
            if (h.isRootConstants) {
                activeVecs = emitGatherCBuffer(b_);
            } else {
                llvm::SmallVector<llvm::Value*, 8> vecSlots;
                vecSlots.reserve(activeComps.size());
                for (size_t i = 0; i < activeComps.size(); ++i) {
                    llvm::Value* sl = createEntryAlloca(vecTy(elemTy), 1, 64);
                    b_.CreateStore(zeroElemVec, sl);
                    vecSlots.push_back(sl);
                }
                emitIfElse(b_, info.isReserved,
                           [&](llvm::IRBuilder<>& nb) {
                               emitForLoop(nb, cInt(i32Ty(), 0), cInt(i32Ty(), W_), cInt(i32Ty(), 1),
                                           [&](llvm::IRBuilder<>& lb, llvm::Value* lIdx) {
                                               llvm::Value* offL = lb.CreateExtractElement(off64Vec, lIdx);
                                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                                   unsigned f = activeComps[i];
                                                   llvm::Value* cOff = (f == 0)
                                                                          ? offL
                                                                          : lb.CreateAdd(offL, cInt(i64Ty(), uint64_t(f) * elemBytes));
                                                   llvm::Value* cEnd = lb.CreateAdd(cOff, cInt(i64Ty(), elemBytes));
                                                   llvm::Value* inBounds = lb.CreateAnd(info.okPtr, lb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
                                                   auto [val, _] = emitReservedElemLoad(
                                                       lb, elemTy, elemBytes, info.descP, info.basePtr, inBounds, cOff);
                                                   llvm::Value* curVec = lb.CreateLoad(vecTy(elemTy), vecSlots[i]);
                                                   lb.CreateStore(lb.CreateInsertElement(curVec, val, lIdx), vecSlots[i]);
                                               }
                                           });
                           },
                           [&](llvm::IRBuilder<>& nb) {
                               auto g = emitGatherCBuffer(nb);
                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                   nb.CreateStore(g[i], vecSlots[i]);
                               }
                           });
                for (size_t i = 0; i < activeComps.size(); ++i) {
                    activeVecs.push_back(b_.CreateLoad(vecTy(elemTy), vecSlots[i]));
                }
            }

            llvm::SmallVector<llvm::Value*, 8> fieldVecs(numFields, zeroElemVec);
            for (size_t i = 0; i < activeComps.size(); ++i) fieldVecs[activeComps[i]] = activeVecs[i];
            for (unsigned f = 0; f < numFields; ++f) {
                outSt.fields.push_back(SimdVal::makeVector(fieldVecs[f], elemTy));
            }
            setMapped(call, outSt);
            return true;
        }

        llvm::SmallVector<llvm::Value*, 8> fieldVecs(numFields, zeroElemVec);
        for (uint32_t l = 0; l < W_; ++l) {
            llvm::Value* ptr = laneValue(h, l);
            BufferDescInfo info = loadBufferDescInfo(b_, h, ptr, /*needElemStride=*/false, /*needStride64=*/false);
            llvm::Value* idxL = llvmExtract(b_, idxVec, l);
            llvm::Value* idx64 = b_.CreateZExt(idxL, i64Ty());
            llvm::Value* off64 = isLegacy ? b_.CreateMul(idx64, cInt(i64Ty(), 16)) : idx64;
            for (unsigned f : activeComps) {
                llvm::Value* cOff = (f == 0) ? off64 : b_.CreateAdd(off64, cInt(i64Ty(), uint64_t(f) * elemBytes));
                llvm::Value* cEnd = b_.CreateAdd(cOff, cInt(i64Ty(), elemBytes));
                llvm::Value* inBounds = h.isRootConstants
                                           ? info.okPtr
                                           : b_.CreateAnd(info.okPtr, b_.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
                auto [val, _] = emitScalarBufferComponentLoad(
                    elemTy, elemBytes, ptr, info.basePtr, info.isReserved, inBounds, cOff);
                fieldVecs[f] = llvmInsert(b_, fieldVecs[f], val, l);
            }
        }
        for (unsigned f = 0; f < numFields; ++f) {
            outSt.fields.push_back(SimdVal::makeVector(fieldVecs[f], elemTy));
        }
        setMapped(call, outSt);
        return true;
    }

    std::pair<llvm::SmallVector<llvm::Value*, 16>, llvm::Value*> emitPacketizedBufferLoads(
        const SimdVal& h,
        llvm::Value* idxArg,
        llvm::Value* elemOffArg,
        bool hasElemOff,
        bool isRawOp,
        llvm::Type* elemTy,
        unsigned totalComps,
        llvm::ArrayRef<unsigned> activeComps) {
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy->getPrimitiveSizeInBits() / 8);
        llvm::Value* zeroElemVec = elemTy->isFloatingPointTy() ? splatFlt(elemTy, 0.0) : splatInt(elemTy, 0);
        if (activeComps.empty()) {
            return {llvm::SmallVector<llvm::Value*, 16>(totalComps, zeroElemVec), splatInt(i32Ty(), 1)};
        }

        bool hasElemStride = (!isRawOp && !hasElemOff && h.resourceKind != 11);
        bool needsStride = hasElemOff || (!isRawOp && h.resourceKind != 11);
        auto elemOffConst = hasElemOff ? getConstIntValue(elemOffArg) : std::optional<uint64_t>{};
        unsigned lastActiveComp = activeComps.back();
        bool nonStraddling = hasElemOff && h.resourceKind == 12 && h.stride > 0 && elemOffConst.has_value() &&
                             (*elemOffConst + uint64_t(lastActiveComp + 1) * elemBytes <= h.stride);

        llvm::Value* uniformIdx = nullptr;
        llvm::Value* uniformElemOff = nullptr;
        bool isUniformIdx = tryGetUniformScalar(idxArg, uniformIdx);
        bool isUniformElemOff = !hasElemOff || tryGetUniformScalar(elemOffArg, uniformElemOff);
        if (h.isUniform() && isUniformIdx && isUniformElemOff) {
            if (!hasElemOff) uniformElemOff = cInt(i32Ty(), 0);
            BufferDescInfo info = loadBufferDescInfo(b_, h, h.lanes[0], hasElemStride, needsStride);
            llvm::Value* byteOff = computeBufferByteOffsetScalar(
                b_, h, info.stride64, uniformIdx, uniformElemOff, isRawOp, hasElemOff);
            return emitCoalescedUniformBufferLoad(
                elemTy, elemBytes, info, h.isRootConstants, nonStraddling, byteOff, totalComps, activeComps);
        }

        llvm::Value* idxVec = asVec(getMapped(idxArg));
        llvm::Value* elemOffVec = hasElemOff ? asVec(getMapped(elemOffArg)) : splatInt(i32Ty(), 0);

        if (h.isUniform()) {
            BufferDescInfo info = loadBufferDescInfo(b_, h, h.lanes[0], hasElemStride, needsStride);
            llvm::Value* stride64Vec = needsStride ? llvmSplat(b_, i64Ty(), info.stride64) : nullptr;

            llvm::SmallVector<llvm::Value*, 16> valSlots;
            valSlots.reserve(activeComps.size());
            for (size_t i = 0; i < activeComps.size(); ++i) {
                llvm::Value* sl = createEntryAlloca(vecTy(elemTy), 1, 64);
                b_.CreateStore(zeroElemVec, sl);
                valSlots.push_back(sl);
            }
            llvm::Value* statusSlot = createEntryAlloca(vecTy(i32Ty()), 1, 64);
            b_.CreateStore(splatInt(i32Ty(), 1), statusSlot);

            emitIfElse(b_, info.isReserved,
                       [&](llvm::IRBuilder<>& nb) {
                           emitForLoop(nb, cInt(i32Ty(), 0), cInt(i32Ty(), W_), cInt(i32Ty(), 1),
                                       [&](llvm::IRBuilder<>& lb, llvm::Value* lIdx) {
                                           llvm::Value* idxL = lb.CreateExtractElement(idxVec, lIdx);
                                           llvm::Value* eL = hasElemOff
                                                                ? lb.CreateExtractElement(elemOffVec, lIdx)
                                                                : cInt(i32Ty(), 0);
                                           llvm::Value* byteOff = computeBufferByteOffsetScalar(
                                               lb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                                           llvm::Value* laneMapped = cInt(i1Ty(), 1);
                                           for (size_t i = 0; i < activeComps.size(); ++i) {
                                               unsigned c = activeComps[i];
                                               llvm::Value* cOff = (c == 0)
                                                                      ? byteOff
                                                                      : lb.CreateAdd(byteOff, cInt(i64Ty(), uint64_t(c) * elemBytes));
                                               llvm::Value* cEnd = lb.CreateAdd(cOff, cInt(i64Ty(), elemBytes));
                                               llvm::Value* inBounds = lb.CreateAnd(info.okPtr, lb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
                                               if (info.elemStride) {
                                                   llvm::Value* elemEnd = cInt(i32Ty(), (c + 1) * elemBytes);
                                                   llvm::Value* strideZero = lb.CreateICmp(llvm::CmpInst::ICMP_EQ, info.elemStride, cInt(i32Ty(), 0));
                                                   llvm::Value* inElem = lb.CreateICmp(llvm::CmpInst::ICMP_ULE, elemEnd, info.elemStride);
                                                   inBounds = lb.CreateAnd(inBounds, lb.CreateOr(strideZero, inElem));
                                               }
                                               auto [val, compMapped] = emitReservedElemLoad(
                                                   lb, elemTy, elemBytes, info.descP, info.basePtr, inBounds, cOff);
                                               laneMapped = lb.CreateAnd(laneMapped, compMapped);
                                               llvm::Value* curVec = lb.CreateLoad(vecTy(elemTy), valSlots[i]);
                                               lb.CreateStore(lb.CreateInsertElement(curVec, val, lIdx), valSlots[i]);
                                           }
                                           llvm::Value* laneStatus = lb.CreateZExt(laneMapped, i32Ty());
                                           llvm::Value* curSt = lb.CreateLoad(vecTy(i32Ty()), statusSlot);
                                           lb.CreateStore(lb.CreateInsertElement(curSt, laneStatus, lIdx), statusSlot);
                                       });
                       },
                       [&](llvm::IRBuilder<>& nb) {
                           llvm::Value* byteOffVec = computeBufferByteOffsetVec(
                               nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                           llvm::Value* byteSizeVec = llvmSplat(nb, i64Ty(), info.byteSize);
                           llvm::Value* okPtrVec = llvmSplat(nb, i1Ty(), info.okPtr);

                           llvm::SmallVector<llvm::Value*, 16> inBounds(activeComps.size());
                           if (nonStraddling) {
                               llvm::Value* esOk = computeScalarElemStrideOkVec(nb, info.elemStride, lastActiveComp, elemBytes);
                               llvm::Value* sharedMask = computeBufferBoundsMaskVec(
                                   nb, byteOffVec, byteSizeVec, okPtrVec, esOk, lastActiveComp, elemBytes);
                               for (size_t i = 0; i < activeComps.size(); ++i) inBounds[i] = sharedMask;
                           } else {
                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                   unsigned c = activeComps[i];
                                   llvm::Value* esOk = computeScalarElemStrideOkVec(nb, info.elemStride, c, elemBytes);
                                   inBounds[i] = computeBufferBoundsMaskVec(
                                       nb, byteOffVec, byteSizeVec, okPtrVec, esOk, c, elemBytes);
                               }
                           }

                           auto emitGatherLoads = [&](llvm::IRBuilder<>& gb) {
                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                   unsigned c = activeComps[i];
                                   llvm::Value* compBase = (c == 0)
                                                              ? info.basePtr
                                                              : gb.CreateGEP(i8Ty(), info.basePtr, gb.getInt32(static_cast<int32_t>(c * elemBytes)));
                                   llvm::Value* ptrs = gb.CreateGEP(i8Ty(), compBase, byteOffVec);
                                   gb.CreateStore(gb.CreateMaskedGather(vecTy(elemTy), ptrs, llvm::Align(1), inBounds[i], zeroElemVec), valSlots[i]);
                               }
                           };

                           bool canBeContig = !(W_ > 1 && ((hasElemOff && h.stride > elemBytes && isUniformElemOff) || activeComps.size() > 4));
                           if (canBeContig) {
                               auto [isContig, lane0Off] = checkContiguousByteOffsets(nb, byteOffVec, elemBytes);
                               emitIfElse(nb, isContig,
                                          [&](llvm::IRBuilder<>& cb) {
                                              llvm::Value* baseLane0Ptr = cb.CreateGEP(i8Ty(), info.basePtr, lane0Off);
                                              for (size_t i = 0; i < activeComps.size(); ++i) {
                                                  unsigned c = activeComps[i];
                                                  llvm::Value* ptr0 = (c == 0)
                                                                         ? baseLane0Ptr
                                                                         : cb.CreateGEP(i8Ty(), baseLane0Ptr, cb.getInt32(static_cast<int32_t>(c * elemBytes)));
                                                  cb.CreateStore(cb.CreateMaskedLoad(vecTy(elemTy), ptr0, llvm::Align(1), inBounds[i], zeroElemVec), valSlots[i]);
                                              }
                                          },
                                          [&](llvm::IRBuilder<>& cb) {
                                              emitGatherLoads(cb);
                                          });
                           } else {
                               emitGatherLoads(nb);
                           }
                           nb.CreateStore(splatInt(i32Ty(), 1), statusSlot);
                       });

            llvm::SmallVector<llvm::Value*, 16> compVecs(totalComps, zeroElemVec);
            for (size_t i = 0; i < activeComps.size(); ++i) {
                compVecs[activeComps[i]] = b_.CreateLoad(vecTy(elemTy), valSlots[i]);
            }
            return {compVecs, b_.CreateLoad(vecTy(i32Ty()), statusSlot)};
        }

        llvm::SmallVector<BufferDescInfo, 16> descInfos(W_);
        llvm::Value* basePtrVec = llvm::PoisonValue::get(vecTy(ptrTy()));
        llvm::Value* byteSizeVec = llvm::PoisonValue::get(vecTy(i64Ty()));
        llvm::Value* okPtrVec = llvm::PoisonValue::get(vecTy(i1Ty()));
        llvm::Value* isReservedVec = llvm::PoisonValue::get(vecTy(i1Ty()));
        llvm::Value* stride64Vec = needsStride ? llvm::PoisonValue::get(vecTy(i64Ty())) : nullptr;
        llvm::Value* elemStrideVec = hasElemStride ? llvm::PoisonValue::get(vecTy(i32Ty())) : nullptr;

        for (uint32_t l = 0; l < W_; ++l) {
            llvm::Value* descP = laneValue(h, l);
            BufferDescInfo info = loadBufferDescInfo(b_, h, descP, hasElemStride, needsStride);
            descInfos[l] = info;
            basePtrVec = llvmInsert(b_, basePtrVec, info.basePtr, l);
            byteSizeVec = llvmInsert(b_, byteSizeVec, info.byteSize, l);
            okPtrVec = llvmInsert(b_, okPtrVec, info.okPtr, l);
            isReservedVec = llvmInsert(b_, isReservedVec, info.isReserved, l);
            if (hasElemStride) {
                elemStrideVec = llvmInsert(b_, elemStrideVec, info.elemStride, l);
            }
            if (needsStride) {
                stride64Vec = llvmInsert(b_, stride64Vec, info.stride64, l);
            }
        }
        llvm::Value* anyReserved = orReduce(isReservedVec);
        llvm::SmallVector<llvm::Value*, 16> valSlots;
        valSlots.reserve(activeComps.size());
        for (size_t i = 0; i < activeComps.size(); ++i) {
            llvm::Value* sl = createEntryAlloca(vecTy(elemTy), 1, 64);
            b_.CreateStore(zeroElemVec, sl);
            valSlots.push_back(sl);
        }
        llvm::Value* statusSlot = createEntryAlloca(vecTy(i32Ty()), 1, 64);
        b_.CreateStore(splatInt(i32Ty(), 1), statusSlot);

        emitIfElse(b_, anyReserved,
                   [&](llvm::IRBuilder<>& nb) {
                       SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                       llvm::SmallVector<llvm::Value*, 16> activeVecs(
                           activeComps.size(), llvm::PoisonValue::get(vecTy(elemTy)));
                       llvm::Value* statusVec = tmp.splatInt(i32Ty(), 1);
                       for (uint32_t l = 0; l < W_; ++l) {
                           const BufferDescInfo& info = descInfos[l];
                           llvm::Value* idxL = llvmExtract(nb, idxVec, l);
                           llvm::Value* eL = llvmExtract(nb, elemOffVec, l);
                           llvm::Value* byteOff = tmp.computeBufferByteOffsetScalar(
                               nb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                           llvm::Value* laneMapped = tmp.cInt(i1Ty(), 1);
                           for (size_t i = 0; i < activeComps.size(); ++i) {
                               unsigned c = activeComps[i];
                               llvm::Value* cOff = (c == 0)
                                                      ? byteOff
                                                      : nb.CreateAdd(byteOff, tmp.cInt(i64Ty(), uint64_t(c) * elemBytes));
                               llvm::Value* cEnd = nb.CreateAdd(cOff, tmp.cInt(i64Ty(), elemBytes));
                               llvm::Value* inBounds = nb.CreateAnd(info.okPtr, nb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
                               if (info.elemStride) {
                                   llvm::Value* elemEnd = tmp.cInt(i32Ty(), (c + 1) * elemBytes);
                                   llvm::Value* strideZero = nb.CreateICmp(llvm::CmpInst::ICMP_EQ, info.elemStride, tmp.cInt(i32Ty(), 0));
                                   llvm::Value* inElem = nb.CreateICmp(llvm::CmpInst::ICMP_ULE, elemEnd, info.elemStride);
                                   inBounds = nb.CreateAnd(inBounds, nb.CreateOr(strideZero, inElem));
                               }
                               auto [val, compMapped] = tmp.emitScalarBufferComponentLoad(
                                   elemTy, elemBytes, info.descP, info.basePtr, info.isReserved, inBounds, cOff);
                               laneMapped = nb.CreateAnd(laneMapped, compMapped);
                               activeVecs[i] = llvmInsert(nb, activeVecs[i], val, l);
                           }
                           llvm::Value* laneStatus = nb.CreateZExt(laneMapped, i32Ty());
                           statusVec = llvmInsert(nb, statusVec, laneStatus, l);
                       }
                       for (size_t i = 0; i < activeComps.size(); ++i) {
                           nb.CreateStore(activeVecs[i], valSlots[i]);
                       }
                       nb.CreateStore(statusVec, statusSlot);
                   },
                   [&](llvm::IRBuilder<>& nb) {
                       llvm::Value* byteOffVec = computeBufferByteOffsetVec(
                           nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                       llvm::Value* sharedMask = nullptr;
                       if (nonStraddling) {
                           llvm::Value* esOk = computeVecElemStrideOkVec(nb, elemStrideVec, lastActiveComp, elemBytes);
                           sharedMask = computeBufferBoundsMaskVec(
                               nb, byteOffVec, byteSizeVec, okPtrVec, esOk, lastActiveComp, elemBytes);
                       }
                       for (size_t i = 0; i < activeComps.size(); ++i) {
                           unsigned c = activeComps[i];
                           llvm::Value* inBoundsVec = sharedMask;
                           if (!inBoundsVec) {
                               llvm::Value* esOk = computeVecElemStrideOkVec(nb, elemStrideVec, c, elemBytes);
                               inBoundsVec = computeBufferBoundsMaskVec(
                                   nb, byteOffVec, byteSizeVec, okPtrVec, esOk, c, elemBytes);
                           }
                           llvm::Value* cOffVec =
                               (c == 0)
                                   ? byteOffVec
                                   : nb.CreateAdd(byteOffVec, splatInt(i64Ty(), uint64_t(c) * elemBytes));
                           llvm::Value* ptrs = nb.CreateGEP(i8Ty(), basePtrVec, cOffVec);
                           nb.CreateStore(nb.CreateMaskedGather(vecTy(elemTy), ptrs, llvm::Align(1), inBoundsVec, zeroElemVec), valSlots[i]);
                       }
                       nb.CreateStore(splatInt(i32Ty(), 1), statusSlot);
                   });

        llvm::SmallVector<llvm::Value*, 16> compVecs(totalComps, zeroElemVec);
        for (size_t i = 0; i < activeComps.size(); ++i) {
            compVecs[activeComps[i]] = b_.CreateLoad(vecTy(elemTy), valSlots[i]);
        }
        return {compVecs, b_.CreateLoad(vecTy(i32Ty()), statusSlot)};
    }

    void emitPacketizedBufferStores(
        const SimdVal& h,
        llvm::Value* idxArg,
        llvm::Value* elemOffArg,
        bool hasElemOff,
        bool isRawOp,
        llvm::Value* realMask,
        llvm::Type* elemTy,
        llvm::ArrayRef<unsigned> activeComps,
        llvm::ArrayRef<llvm::Value*> activeValVecs) {
        if (activeComps.empty()) return;
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy->getPrimitiveSizeInBits() / 8);
        bool hasElemStride = (!isRawOp && !hasElemOff && h.resourceKind != 11);
        bool needsStride = hasElemOff || (!isRawOp && h.resourceKind != 11);
        auto elemOffConst = hasElemOff ? getConstIntValue(elemOffArg) : std::optional<uint64_t>{};
        unsigned lastActiveComp = activeComps.back();
        bool nonStraddling = hasElemOff && h.resourceKind == 12 && h.stride > 0 && elemOffConst.has_value() &&
                             (*elemOffConst + uint64_t(lastActiveComp + 1) * elemBytes <= h.stride);

        llvm::Value* dummyUniformElemOff = nullptr;
        bool isUniformElemOff = !hasElemOff || tryGetUniformScalar(elemOffArg, dummyUniformElemOff);
        llvm::Value* idxVec = asVec(getMapped(idxArg));
        llvm::Value* elemOffVec = hasElemOff ? asVec(getMapped(elemOffArg)) : splatInt(i32Ty(), 0);

        if (h.isUniform()) {
            BufferDescInfo info = loadBufferDescInfo(b_, h, h.lanes[0], hasElemStride, needsStride);
            llvm::Value* stride64Vec = needsStride ? llvmSplat(b_, i64Ty(), info.stride64) : nullptr;

            emitIfElse(b_, info.isReserved,
                       [&](llvm::IRBuilder<>& nb) {
                           emitForLoop(nb, cInt(i32Ty(), 0), cInt(i32Ty(), W_), cInt(i32Ty(), 1),
                                       [&](llvm::IRBuilder<>& lb, llvm::Value* lIdx) {
                                           llvm::Value* mL = lb.CreateExtractElement(realMask, lIdx);
                                           llvm::Value* okPtr = lb.CreateAnd(mL, info.okPtr);
                                           llvm::Value* idxL = lb.CreateExtractElement(idxVec, lIdx);
                                           llvm::Value* eL = hasElemOff
                                                                ? lb.CreateExtractElement(elemOffVec, lIdx)
                                                                : cInt(i32Ty(), 0);
                                           llvm::Value* byteOff = computeBufferByteOffsetScalar(
                                               lb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                                           for (size_t i = 0; i < activeComps.size(); ++i) {
                                               unsigned c = activeComps[i];
                                               llvm::Value* vL = lb.CreateExtractElement(activeValVecs[i], lIdx);
                                               llvm::Value* cOff = (c == 0)
                                                                      ? byteOff
                                                                      : lb.CreateAdd(byteOff, cInt(i64Ty(), uint64_t(c) * elemBytes));
                                               llvm::Value* cEnd = lb.CreateAdd(cOff, cInt(i64Ty(), elemBytes));
                                               llvm::Value* doStore = lb.CreateAnd(okPtr, lb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
                                               if (info.elemStride) {
                                                   llvm::Value* elemEnd = cInt(i32Ty(), (c + 1) * elemBytes);
                                                   llvm::Value* strideZero = lb.CreateICmp(llvm::CmpInst::ICMP_EQ, info.elemStride, cInt(i32Ty(), 0));
                                                   llvm::Value* inElem = lb.CreateICmp(llvm::CmpInst::ICMP_ULE, elemEnd, info.elemStride);
                                                   doStore = lb.CreateAnd(doStore, lb.CreateOr(strideZero, inElem));
                                               }
                                               emitReservedElemStore(lb, elemBytes, info.descP, info.basePtr, doStore, cOff, vL);
                                           }
                                       });
                       },
                       [&](llvm::IRBuilder<>& nb) {
                           llvm::Value* byteOffVec = computeBufferByteOffsetVec(
                               nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                           llvm::Value* byteSizeVec = llvmSplat(nb, i64Ty(), info.byteSize);
                           llvm::Value* activeOkVec = nb.CreateAnd(realMask, llvmSplat(nb, i1Ty(), info.okPtr));

                           llvm::SmallVector<llvm::Value*, 16> doStores(activeComps.size());
                           if (nonStraddling) {
                               llvm::Value* esOk = computeScalarElemStrideOkVec(nb, info.elemStride, lastActiveComp, elemBytes);
                               llvm::Value* sharedMask = computeBufferBoundsMaskVec(
                                   nb, byteOffVec, byteSizeVec, activeOkVec, esOk, lastActiveComp, elemBytes);
                               for (size_t i = 0; i < activeComps.size(); ++i) doStores[i] = sharedMask;
                           } else {
                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                   unsigned c = activeComps[i];
                                   llvm::Value* esOk = computeScalarElemStrideOkVec(nb, info.elemStride, c, elemBytes);
                                   doStores[i] = computeBufferBoundsMaskVec(
                                       nb, byteOffVec, byteSizeVec, activeOkVec, esOk, c, elemBytes);
                               }
                           }

                           auto emitScatterStores = [&](llvm::IRBuilder<>& sb) {
                               for (size_t i = 0; i < activeComps.size(); ++i) {
                                   unsigned c = activeComps[i];
                                   llvm::Value* compBase = (c == 0)
                                                              ? info.basePtr
                                                              : sb.CreateGEP(i8Ty(), info.basePtr, sb.getInt32(static_cast<int32_t>(c * elemBytes)));
                                   llvm::Value* ptrs = sb.CreateGEP(i8Ty(), compBase, byteOffVec);
                                   sb.CreateMaskedScatter(activeValVecs[i], ptrs, llvm::Align(1), doStores[i]);
                               }
                           };

                           bool canBeContig = !(W_ > 1 && ((hasElemOff && h.stride > elemBytes && isUniformElemOff) || activeComps.size() > 4));
                           if (canBeContig) {
                               auto [isContig, lane0Off] = checkContiguousByteOffsets(nb, byteOffVec, elemBytes);
                               emitIfElse(nb, isContig,
                                          [&](llvm::IRBuilder<>& cb) {
                                              llvm::Value* baseLane0Ptr = cb.CreateGEP(i8Ty(), info.basePtr, lane0Off);
                                              for (size_t i = 0; i < activeComps.size(); ++i) {
                                                  unsigned c = activeComps[i];
                                                  llvm::Value* ptr0 = (c == 0)
                                                                         ? baseLane0Ptr
                                                                         : cb.CreateGEP(i8Ty(), baseLane0Ptr, cb.getInt32(static_cast<int32_t>(c * elemBytes)));
                                                  cb.CreateMaskedStore(activeValVecs[i], ptr0, llvm::Align(1), doStores[i]);
                                              }
                                          },
                                          [&](llvm::IRBuilder<>& cb) {
                                              emitScatterStores(cb);
                                          });
                           } else {
                               emitScatterStores(nb);
                           }
                       });
            return;
        }

        llvm::SmallVector<BufferDescInfo, 16> descInfos(W_);
        llvm::Value* basePtrVec = llvm::PoisonValue::get(vecTy(ptrTy()));
        llvm::Value* byteSizeVec = llvm::PoisonValue::get(vecTy(i64Ty()));
        llvm::Value* okPtrVec = llvm::PoisonValue::get(vecTy(i1Ty()));
        llvm::Value* isReservedVec = llvm::PoisonValue::get(vecTy(i1Ty()));
        llvm::Value* stride64Vec = needsStride ? llvm::PoisonValue::get(vecTy(i64Ty())) : nullptr;
        llvm::Value* elemStrideVec = hasElemStride ? llvm::PoisonValue::get(vecTy(i32Ty())) : nullptr;

        for (uint32_t l = 0; l < W_; ++l) {
            llvm::Value* descP = laneValue(h, l);
            BufferDescInfo info = loadBufferDescInfo(b_, h, descP, hasElemStride, needsStride);
            descInfos[l] = info;
            basePtrVec = llvmInsert(b_, basePtrVec, info.basePtr, l);
            byteSizeVec = llvmInsert(b_, byteSizeVec, info.byteSize, l);
            okPtrVec = llvmInsert(b_, okPtrVec, info.okPtr, l);
            isReservedVec = llvmInsert(b_, isReservedVec, info.isReserved, l);
            if (hasElemStride) {
                elemStrideVec = llvmInsert(b_, elemStrideVec, info.elemStride, l);
            }
            if (needsStride) {
                stride64Vec = llvmInsert(b_, stride64Vec, info.stride64, l);
            }
        }
        llvm::Value* anyReserved = orReduce(isReservedVec);
        llvm::Value* activeOkVec = b_.CreateAnd(realMask, okPtrVec);

        emitIfElse(b_, anyReserved,
                   [&](llvm::IRBuilder<>& nb) {
                       SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                       for (uint32_t l = 0; l < W_; ++l) {
                           const BufferDescInfo& info = descInfos[l];
                           llvm::Value* mL = llvmExtract(nb, realMask, l);
                           llvm::Value* okPtr = nb.CreateAnd(mL, info.okPtr);
                           llvm::Value* idxL = llvmExtract(nb, idxVec, l);
                           llvm::Value* eL = llvmExtract(nb, elemOffVec, l);
                           llvm::Value* byteOff = tmp.computeBufferByteOffsetScalar(
                               nb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                           for (size_t i = 0; i < activeComps.size(); ++i) {
                               unsigned c = activeComps[i];
                               llvm::Value* vL = llvmExtract(nb, activeValVecs[i], l);
                               llvm::Value* cOff = (c == 0)
                                                      ? byteOff
                                                      : nb.CreateAdd(byteOff, tmp.cInt(i64Ty(), uint64_t(c) * elemBytes));
                               llvm::Value* cEnd = nb.CreateAdd(cOff, tmp.cInt(i64Ty(), elemBytes));
                               llvm::Value* doStore = nb.CreateAnd(okPtr, nb.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, info.byteSize));
                               if (info.elemStride) {
                                   llvm::Value* elemEnd = tmp.cInt(i32Ty(), (c + 1) * elemBytes);
                                   llvm::Value* strideZero = nb.CreateICmp(llvm::CmpInst::ICMP_EQ, info.elemStride, tmp.cInt(i32Ty(), 0));
                                   llvm::Value* inElem = nb.CreateICmp(llvm::CmpInst::ICMP_ULE, elemEnd, info.elemStride);
                                   doStore = nb.CreateAnd(doStore, nb.CreateOr(strideZero, inElem));
                               }
                               tmp.emitScalarBufferComponentStore(
                                   elemBytes, info.descP, info.basePtr, info.isReserved, doStore, cOff, vL);
                           }
                       }
                   },
                   [&](llvm::IRBuilder<>& nb) {
                       llvm::Value* byteOffVec = computeBufferByteOffsetVec(
                           nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                       llvm::Value* sharedMask = nullptr;
                       if (nonStraddling) {
                           llvm::Value* esOk = computeVecElemStrideOkVec(nb, elemStrideVec, lastActiveComp, elemBytes);
                           sharedMask = computeBufferBoundsMaskVec(
                               nb, byteOffVec, byteSizeVec, activeOkVec, esOk, lastActiveComp, elemBytes);
                       }
                       for (size_t i = 0; i < activeComps.size(); ++i) {
                           unsigned c = activeComps[i];
                           llvm::Value* doStoreVec = sharedMask;
                           if (!doStoreVec) {
                               llvm::Value* esOk = computeVecElemStrideOkVec(nb, elemStrideVec, c, elemBytes);
                               doStoreVec = computeBufferBoundsMaskVec(
                                   nb, byteOffVec, byteSizeVec, activeOkVec, esOk, c, elemBytes);
                           }
                           llvm::Value* cOffVec =
                               (c == 0)
                                   ? byteOffVec
                                   : nb.CreateAdd(byteOffVec, splatInt(i64Ty(), uint64_t(c) * elemBytes));
                           llvm::Value* ptrs = nb.CreateGEP(i8Ty(), basePtrVec, cOffVec);
                           nb.CreateMaskedScatter(activeValVecs[i], ptrs, llvm::Align(1), doStoreVec);
                       }
                   });
    }

    bool packetizeBufferLoad(llvm::CallInst* call, bool isRawOp, std::string& outError) {
        SimdVal h = getMapped(call->getArgOperand(1));
        llvm::Value* idxArg = call->getArgOperand(2);
        llvm::Value* elemOffArg = call->getArgOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);
        uint32_t mask = isRawOp ? static_cast<uint32_t>(getConstIntValue(call->getArgOperand(4)).value_or(15)) : 15u;

        auto st = llvm::dyn_cast<llvm::StructType>(call->getType());
        if (!st || st->elements().size() < 5) {
            outError = "bufferLoad expected ResRet struct";
            return false;
        }
        llvm::Type* elemTy = st->elements()[0];

        uint32_t usedCompMask = 0;
        bool statusUsed = false;
        bool allExtract = true;
        for (llvm::User* user : call->users()) {
            if (auto* ev = llvm::dyn_cast<llvm::ExtractValueInst>(user)) {
                auto pos = ev->getIndices();
                if (!pos.empty()) {
                    if (pos[0] < 4) {
                        usedCompMask |= (1u << pos[0]);
                        continue;
                    }
                    if (pos[0] == 4) {
                        statusUsed = true;
                        continue;
                    }
                }
            }
            allExtract = false;
            break;
        }
        if (allExtract && !statusUsed) {
            mask &= usedCompMask;
        }

        llvm::SmallVector<unsigned, 4> activeComps;
        for (unsigned c = 0; c < 4; ++c) {
            if (mask & (1u << c)) activeComps.push_back(c);
        }

        auto [compVecs, statusVec] = emitPacketizedBufferLoads(
            h, idxArg, elemOffArg, hasElemOff, isRawOp, elemTy, /*totalComps=*/4, activeComps);

        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        for (unsigned c = 0; c < 4; ++c) {
            outSt.fields.push_back(SimdVal::makeVector(compVecs[c], elemTy));
        }
        outSt.fields.push_back(SimdVal::makeVector(statusVec, i32Ty()));
        setMapped(call, outSt);
        return true;
    }

    bool packetizeBufferStore(llvm::CallInst* call, bool isRawOp, llvm::Value* realMask, std::string&) {
        SimdVal h = getMapped(call->getArgOperand(1));
        llvm::Value* idxArg = call->getArgOperand(2);
        llvm::Value* elemOffArg = call->getArgOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);
        uint32_t mask = static_cast<uint32_t>(getConstIntValue(call->getArgOperand(8)).value_or(15));

        llvm::SmallVector<unsigned, 4> activeComps;
        llvm::SmallVector<llvm::Value*, 4> activeValVecs;
        llvm::Type* elemTy = i32Ty();
        for (unsigned c = 0; c < 4; ++c) {
            if (mask & (1u << c)) {
                llvm::Value* v = asVec(getMapped(call->getArgOperand(4 + c)));
                elemTy = llvm::cast<llvm::FixedVectorType>(v->getType())->getElementType();
                activeComps.push_back(c);
                activeValVecs.push_back(v);
            }
        }
        emitPacketizedBufferStores(
            h, idxArg, elemOffArg, hasElemOff, isRawOp, realMask, elemTy, activeComps, activeValVecs);
        return true;
    }

    bool packetizeBufferVectorLoad(llvm::CallInst* call, std::string& outError) {
        SimdVal h = getMapped(call->getArgOperand(1));
        llvm::Value* idxArg = call->getArgOperand(2);
        llvm::Value* elemOffArg = call->getArgOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);

        auto* st = llvm::dyn_cast<llvm::StructType>(call->getType());
        auto* vt = st ? llvm::dyn_cast<llvm::FixedVectorType>(st->elements()[0]) : nullptr;
        if (!st || !vt || st->elements().size() < 2) {
            outError = "rawBufferVectorLoad expected ResRet struct with vector element";
            return false;
        }
        llvm::Type* elemTy = vt->getElementType();
        unsigned numElems = vt->getNumElements();

        llvm::SmallVector<bool, 16> elemUsed(numElems, false);
        bool statusUsed = false;
        bool canPrune = true;
        for (llvm::User* user : call->users()) {
            auto* ev = llvm::dyn_cast<llvm::ExtractValueInst>(user);
            if (!ev || ev->getIndices().empty()) {
                canPrune = false;
                break;
            }
            unsigned fieldIdx = ev->getIndices()[0];
            if (fieldIdx == 1) {
                statusUsed = true;
            } else if (fieldIdx == 0) {
                for (llvm::User* vUser : ev->users()) {
                    auto* ee = llvm::dyn_cast<llvm::ExtractElementInst>(vUser);
                    auto idxC = ee ? getConstIntValue(ee->getIndexOperand()) : std::optional<uint64_t>{};
                    if (!idxC.has_value() || *idxC >= numElems) {
                        canPrune = false;
                        break;
                    }
                    elemUsed[*idxC] = true;
                }
                if (!canPrune) break;
            } else {
                canPrune = false;
                break;
            }
        }

        llvm::SmallVector<unsigned, 16> activeComps;
        activeComps.reserve(numElems);
        for (unsigned c = 0; c < numElems; ++c) {
            if (!canPrune || statusUsed || elemUsed[c]) activeComps.push_back(c);
        }

        auto [compVecs, statusVec] = emitPacketizedBufferLoads(
            h, idxArg, elemOffArg, hasElemOff, /*isRawOp=*/true, elemTy, numElems, activeComps);

        SimdVal vecVal;
        vecVal.kind = SimdVal::Kind::Struct;
        vecVal.origType = vt;
        for (unsigned c = 0; c < numElems; ++c) {
            vecVal.fields.push_back(SimdVal::makeVector(compVecs[c], elemTy));
        }
        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        outSt.fields.push_back(std::move(vecVal));
        outSt.fields.push_back(SimdVal::makeVector(statusVec, i32Ty()));
        setMapped(call, outSt);
        return true;
    }

    bool packetizeBufferVectorStore(llvm::CallInst* call, llvm::Value* realMask, std::string& outError) {
        SimdVal h = getMapped(call->getArgOperand(1));
        llvm::Value* idxArg = call->getArgOperand(2);
        llvm::Value* elemOffArg = call->getArgOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);
        SimdVal valSv = getMapped(call->getArgOperand(4));
        if (valSv.kind != SimdVal::Kind::Struct || valSv.fields.empty()) {
            outError = "rawBufferVectorStore expected vector value";
            return false;
        }
        unsigned numElems = static_cast<unsigned>(valSv.fields.size());
        llvm::SmallVector<unsigned, 16> activeComps;
        llvm::SmallVector<llvm::Value*, 16> activeValVecs;
        activeComps.reserve(numElems);
        activeValVecs.reserve(numElems);
        for (unsigned c = 0; c < numElems; ++c) {
            activeComps.push_back(c);
            activeValVecs.push_back(asVec(valSv.fields[c]));
        }
        llvm::Type* elemTy = llvm::cast<llvm::FixedVectorType>(activeValVecs[0]->getType())->getElementType();
        emitPacketizedBufferStores(
            h, idxArg, elemOffArg, hasElemOff, /*isRawOp=*/true, realMask, elemTy, activeComps, activeValVecs);
        return true;
    }

    bool packetizeAtomicOp(llvm::CallInst* call, bool isCmpXchg, llvm::Value* realMask, std::string&) {
        // atomicBinOp(op, handle, atomicOp, off0, off1, off2, newVal)
        // atomicCompareExchange(op, handle, off0, off1, off2, cmpVal, newVal)
        SimdVal h = getMapped(call->getArgOperand(1));
        uint32_t atomicOp = isCmpXchg ? 0 : static_cast<uint32_t>(getConstIntValue(call->getArgOperand(2)).value_or(0));
        unsigned offStart = isCmpXchg ? 2u : 3u;
        llvm::Value* off0 = asVec(getMapped(call->getArgOperand(offStart)));
        bool hasOff1 = !isUndefOrPoison(call->getArgOperand(offStart + 1));
        llvm::Value* off1 = hasOff1 ? asVec(getMapped(call->getArgOperand(offStart + 1))) : splatInt(i32Ty(), 0);
        llvm::Value* cmpValVec = isCmpXchg ? asVec(getMapped(call->getArgOperand(5))) : nullptr;
        llvm::Value* newValVec = asVec(getMapped(call->getArgOperand(6)));

        auto toBinOp = [](uint32_t dxAtomic) -> llvm::AtomicRMWInst::BinOp {
            switch (dxAtomic) {
            case 0: return llvm::AtomicRMWInst::Add;
            case 1: return llvm::AtomicRMWInst::And;
            case 2: return llvm::AtomicRMWInst::Or;
            case 3: return llvm::AtomicRMWInst::Xor;
            case 4: return llvm::AtomicRMWInst::Min;
            case 5: return llvm::AtomicRMWInst::Max;
            case 6: return llvm::AtomicRMWInst::UMin;
            case 7: return llvm::AtomicRMWInst::UMax;
            case 8: return llvm::AtomicRMWInst::Xchg;
            default: return llvm::AtomicRMWInst::Add;
            }
        };

        llvm::Type* resTy = call->getType();
        uint32_t bitWidth = resTy->getScalarSizeInBits();
        uint32_t elemBytes = std::max<uint32_t>(1u, bitWidth / 8u);
        bool isFloat = resTy->isFloatingPointTy();
        llvm::Type* intTy = b_.getIntNTy(bitWidth);
        const bool isTexAtomic = (h.resourceKind >= 1 && h.resourceKind <= 9);
        const bool isRawBufAtomic = (h.resourceKind == 11 || h.resourceKind == 12);
        const bool needAtomicStride = !isTexAtomic && (hasOff1 || !isRawBufAtomic);

        std::optional<BufferDescInfo> uniformBufInfo;
        if (!isTexAtomic && h.isUniform()) {
            uniformBufInfo = loadBufferDescInfo(b_, h, h.lanes[0], /*needElemStride=*/false, needAtomicStride);
        }

        llvm::Value* outVec = isFloat ? splatFlt(resTy, 0.0) : splatInt(resTy, 0);
        for (uint32_t l = 0; l < W_; ++l) {
            if (isLaneMaskConstFalse(realMask, l)) continue;
            llvm::Value* mL = llvmExtract(b_, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
            llvm::Value* descP;
            llvm::Value* basePtr;
            llvm::Value* isReserved;
            llvm::Value* byteSize;
            llvm::Value* okPtr;
            llvm::Value* o0L = llvmExtract(b_, off0, static_cast<uint32_t>(static_cast<int64_t>(l)));
            llvm::Value* o1L = llvmExtract(b_, off1, static_cast<uint32_t>(static_cast<int64_t>(l)));
            llvm::Value* byteOff;
            if (isTexAtomic) {
                descP = maskDescPtrForLane(h, realMask, l);
                basePtr = loadDescBasePtr(descP);
                isReserved = cInt(i1Ty(), 0);
                byteSize = loadDescU64Field(descP, offsetof(NacelleDescriptor, byte_size));
                llvm::Value* nullP = llvm::Constant::getNullValue(ptrTy());
                okPtr = b_.CreateAnd(mL, b_.CreateICmp(llvm::CmpInst::ICMP_NE, basePtr, nullP));
                // RWTexture2D atomic: off0=x, off1=y
                llvm::Value* rp = loadDescU32Field(descP, offsetof(NacelleDescriptor, row_pitch), 0);
                llvm::Value* y64 = b_.CreateMul(b_.CreateZExt(o1L, i64Ty()), b_.CreateZExt(rp, i64Ty()));
                llvm::Value* x64 = b_.CreateMul(b_.CreateZExt(o0L, i64Ty()), cInt(i64Ty(), elemBytes));
                byteOff = b_.CreateAdd(y64, x64);
            } else {
                BufferDescInfo info = uniformBufInfo.has_value()
                                          ? *uniformBufInfo
                                          : loadBufferDescInfo(b_, h, maskDescPtrForLane(h, realMask, l), /*needElemStride=*/false, needAtomicStride);
                descP = info.descP;
                basePtr = info.basePtr;
                isReserved = info.isReserved;
                byteSize = info.byteSize;
                okPtr = b_.CreateAnd(mL, info.okPtr);
                byteOff = computeBufferByteOffsetScalar(b_, h, info.stride64, o0L, o1L, isRawBufAtomic, hasOff1);
            }
            llvm::Value* cEnd = b_.CreateAdd(byteOff, cInt(i64Ty(), elemBytes));
            llvm::Value* doOp = b_.CreateAnd(okPtr, b_.CreateICmp(llvm::CmpInst::ICMP_ULE, cEnd, byteSize));
            llvm::Value* newL = llvmExtract(b_, newValVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
            llvm::Value* cmpL = isCmpXchg ? llvmExtract(b_, cmpValVec, static_cast<uint32_t>(static_cast<int64_t>(l))) : nullptr;

            llvm::Value* zeroRes = isFloat ? cFlt(resTy, 0.0) : cInt(resTy, 0);
            llvm::Value* resSlot = createEntryAlloca(resTy, 1, 8);
            b_.CreateStore(zeroRes, resSlot);
            emitIf(b_, doOp, [&](llvm::IRBuilder<>& nb) {
                llvm::Value* epSlot = createEntryAlloca(ptrTy(), 1, 8);
                emitIfElse(nb, isReserved,
                           [&](llvm::IRBuilder<>& rb) {
                               rb.CreateStore(resolveReservedBufferElemPtr(rb, descP, basePtr, byteOff, elemBytes), epSlot);
                           },
                           [&](llvm::IRBuilder<>& rb) {
                               rb.CreateStore(rb.CreateGEP(i8Ty(), basePtr, byteOff), epSlot);
                           });
                llvm::Value* ep = nb.CreateLoad(ptrTy(), epSlot);
                llvm::Value* epValid = nb.CreateICmp(llvm::CmpInst::ICMP_NE, ep, llvm::Constant::getNullValue(ptrTy()));
                emitIf(nb, epValid, [&](llvm::IRBuilder<>& ab) {
                    llvm::Value* ar = nullptr;
                    if (isCmpXchg) {
                        llvm::Value* cOp = isFloat ? ab.CreateBitCast(cmpL, intTy) : cmpL;
                        llvm::Value* nOp = isFloat ? ab.CreateBitCast(newL, intTy) : newL;
                        auto* cx = ab.CreateAtomicCmpXchg(ep, cOp, nOp, llvm::MaybeAlign(), llvm::AtomicOrdering::SequentiallyConsistent, llvm::AtomicOrdering::SequentiallyConsistent);
                        ar = ab.CreateExtractValue(cx, 0u);
                        if (isFloat) ar = ab.CreateBitCast(ar, resTy);
                    } else if (isFloat && toBinOp(atomicOp) == llvm::AtomicRMWInst::Xchg) {
                        llvm::Value* nOp = ab.CreateBitCast(newL, intTy);
                        ar = ab.CreateAtomicRMW(llvm::AtomicRMWInst::Xchg, ep, nOp, llvm::MaybeAlign(), llvm::AtomicOrdering::SequentiallyConsistent);
                        ar = ab.CreateBitCast(ar, resTy);
                    } else {
                        ar = ab.CreateAtomicRMW(toBinOp(atomicOp), ep, newL, llvm::MaybeAlign(), llvm::AtomicOrdering::SequentiallyConsistent);
                    }
                    ab.CreateStore(ar, resSlot);
                });
            });
            outVec = llvmInsert(b_, outVec, b_.CreateLoad(resTy, resSlot), l);
        }
        setMapped(call, SimdVal::makeVector(outVec, resTy));
        return true;
    }

    std::pair<llvm::Value*, llvm::Value*> computeQuadFiniteDiff(llvm::Value* vec, bool fine) {
        llvm::SmallVector<int32_t, 16> loX, hiX, loY, hiY;
        for (int32_t q = 0; q < static_cast<int32_t>(W_); q += 4) {
            if (fine) {
                loX.insert(loX.end(), {q, q, q + 2, q + 2});
                hiX.insert(hiX.end(), {q + 1, q + 1, q + 3, q + 3});
                loY.insert(loY.end(), {q, q + 1, q, q + 1});
                hiY.insert(hiY.end(), {q + 2, q + 3, q + 2, q + 3});
            } else {
                loX.insert(loX.end(), {q, q, q, q});
                hiX.insert(hiX.end(), {q + 1, q + 1, q + 1, q + 1});
                loY.insert(loY.end(), {q, q, q, q});
                hiY.insert(hiY.end(), {q + 2, q + 2, q + 2, q + 2});
            }
        }
        llvm::Value* ax = b_.CreateShuffleVector(vec, vec, loX);
        llvm::Value* bx = b_.CreateShuffleVector(vec, vec, hiX);
        llvm::Value* ay = b_.CreateShuffleVector(vec, vec, loY);
        llvm::Value* by = b_.CreateShuffleVector(vec, vec, hiY);
        return {b_.CreateFSub(bx, ax), b_.CreateFSub(by, ay)};
    }

    bool packetizeTextureOp(llvm::CallInst* call, uint32_t op, llvm::Value* execMask, llvm::Value* realMask, std::string&) {
        auto vecOrZero = [&](unsigned argIdx, llvm::Type* ty, double z = 0.0) -> llvm::Value* {
            llvm::Value* v = call->getArgOperand(argIdx);
            if (isUndefOrPoison(v)) return ty->isFloatingPointTy() ? splatFlt(ty, z) : splatInt(ty, static_cast<uint64_t>(z));
            return asVec(getMapped(v));
        };

        if (op == 72) { // getDimensions(op, handle, mip)
            SimdVal h = getMapped(call->getArgOperand(1));
            llvm::Value* mipVec = vecOrZero(2, i32Ty(), 0.0);
            auto fn = getOrDeclareRuntimeFn("nacelle_tex_get_dims", nullptr, {ptrTy(), i32Ty(), ptrTy()});
            llvm::Value* tmp4 = createEntryAlloca(llvm::ArrayType::get(i32Ty(), 4), 1, 16);
            llvm::SmallVector<llvm::Value*, 4> dims(4, splatInt(i32Ty(), 0));
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* m = llvmExtract(b_, mipVec, l);
                b_.CreateCall(fn, {laneValue(h, l), m, tmp4});
                for (unsigned c = 0; c < 4; ++c) {
                    llvm::Value* ep = gepByteOffset(tmp4, c * 4);
                    dims[c] = llvmInsert(b_, dims[c], loadScalar(i32Ty(), ep), l);
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call->getType();
            for (unsigned c = 0; c < 4; ++c) st.fields.push_back(SimdVal::makeVector(dims[c], i32Ty()));
            setMapped(call, st);
            return true;
        }

        if (op == 67) { // textureStore(op, uav, c0, c1, c2, v0, v1, v2, v3, mask)
            SimdVal h = getMapped(call->getArgOperand(1));
            llvm::Value* x = vecOrZero(2, i32Ty(), 0.0);
            llvm::Value* y = vecOrZero(3, i32Ty(), 0.0);
            llvm::Value* z = vecOrZero(4, i32Ty(), 0.0);
            llvm::Value* r = vecOrZero(5, f32Ty(), 0.0);
            llvm::Value* g = vecOrZero(6, f32Ty(), 0.0);
            llvm::Value* b = vecOrZero(7, f32Ty(), 0.0);
            llvm::Value* a = vecOrZero(8, f32Ty(), 0.0);
            auto fn = getOrDeclareRuntimeFn("nacelle_tex_store", nullptr,
                                            {ptrTy(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty()});
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, realMask, l);
                llvm::Value* dp = laneValue(h, l);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    auto ext = [&](llvm::Value* v) {
                        llvm::Value* ev = llvmExtract(nb, v, l);
                        if (auto* it = llvm::dyn_cast<llvm::IntegerType>(ev->getType())) {
                            if (it->getBitWidth() < 32) {
                                ev = nb.CreateZExt(ev, nb.getInt32Ty());
                            } else if (it->getBitWidth() > 32) {
                                ev = nb.CreateTrunc(ev, nb.getInt32Ty());
                            }
                            ev = nb.CreateBitCast(ev, nb.getFloatTy());
                        }
                        return ev;
                    };
                    nb.CreateCall(fn, {
                            dp,
                            llvmExtract(nb, x, l),
                            llvmExtract(nb, y, l),
                            llvmExtract(nb, z, l),
                            ext(r), ext(g), ext(b), ext(a)});
                });
            }
            return true;
        }

        if (op == 66) { // textureLoad(op, srv, mip, x, y, z, off0, off1, off2)
            SimdVal h = getMapped(call->getArgOperand(1));
            llvm::Value* mip = vecOrZero(2, i32Ty(), 0.0);
            llvm::Value* x = b_.CreateAdd(vecOrZero(3, i32Ty(), 0.0), vecOrZero(6, i32Ty(), 0.0));
            llvm::Value* y = b_.CreateAdd(vecOrZero(4, i32Ty(), 0.0), vecOrZero(7, i32Ty(), 0.0));
            llvm::Value* z = b_.CreateAdd(vecOrZero(5, i32Ty(), 0.0), vecOrZero(8, i32Ty(), 0.0));
            auto fn = getOrDeclareRuntimeFn("nacelle_tex_load", i32Ty(), {ptrTy(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            llvm::Value* out4 = createEntryAlloca(llvm::ArrayType::get(f32Ty(), 4), 1, 16);
            llvm::Value* stSlot = createEntryAlloca(i32Ty(), 1, 4);
            llvm::SmallVector<llvm::Value*, 4> rgba(4, splatFlt(f32Ty(), 0.0));
            llvm::Value* statusVec = splatInt(i32Ty(), 1);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, execMask, l);
                llvm::Value* dp = laneValue(h, l);
                b_.CreateStore(cInt(i32Ty(), 1), stSlot);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    auto* c = nb.CreateCall(fn, {
                            dp,
                            llvmExtract(nb, x, l),
                            llvmExtract(nb, y, l),
                            llvmExtract(nb, z, l),
                            llvmExtract(nb, mip, l),
                            out4});
                    nb.CreateStore(c, stSlot);
                });
                statusVec = llvmInsert(b_, statusVec, b_.CreateLoad(i32Ty(), stSlot), l);
                for (unsigned c = 0; c < 4; ++c) {
                    llvm::Value* ep = gepByteOffset(out4, c * 4);
                    rgba[c] = llvmInsert(b_, rgba[c], loadScalar(f32Ty(), ep), l);
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call->getType();
            llvm::Type* resElem = llvm::cast<llvm::StructType>(st.origType)->elements()[0];
            for (unsigned c = 0; c < 4; ++c) {
                llvm::Value* v = rgba[c];
                if (auto* it = llvm::dyn_cast<llvm::IntegerType>(resElem)) {
                    v = b_.CreateBitCast(v, vecTy(i32Ty()));
                    if (it->getBitWidth() < 32) {
                        v = b_.CreateTrunc(v, vecTy(resElem));
                    } else if (it->getBitWidth() > 32) {
                        v = b_.CreateZExt(v, vecTy(resElem));
                    }
                }
                st.fields.push_back(SimdVal::makeVector(v, resElem));
            }
            st.fields.push_back(SimdVal::makeVector(statusVec, i32Ty()));
            setMapped(call, st);
            return true;
        }

        if (op == 73 || op == 74) { // textureGather / textureGatherCmp
            bool isCmp = (op == 74);
            SimdVal texH = getMapped(call->getArgOperand(1));
            SimdVal sampH = getMapped(call->getArgOperand(2));
            llvm::Value* u = vecOrZero(3, f32Ty(), 0.0);
            llvm::Value* v = vecOrZero(4, f32Ty(), 0.0);
            llvm::Value* w = vecOrZero(5, f32Ty(), 0.0);
            llvm::Value* offX = vecOrZero(7, i32Ty(), 0.0);
            llvm::Value* offY = vecOrZero(8, i32Ty(), 0.0);
            llvm::Value* comp = vecOrZero(9, i32Ty(), 0.0);
            llvm::Value* cmpV = isCmp ? vecOrZero(10, f32Ty(), 0.0) : nullptr;
            auto fn = isCmp
                          ? getOrDeclareRuntimeFn(
                                "nacelle_tex_gather_cmp", i32Ty(),
                                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), ptrTy()})
                          : getOrDeclareRuntimeFn(
                                "nacelle_tex_gather", i32Ty(),
                                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            llvm::Value* out4 = createEntryAlloca(llvm::ArrayType::get(f32Ty(), 4), 1, 16);
            llvm::Value* stSlot = createEntryAlloca(i32Ty(), 1, 4);
            llvm::SmallVector<llvm::Value*, 4> rgba(4, splatFlt(f32Ty(), 0.0));
            llvm::Value* statusVec = splatInt(i32Ty(), 1);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, execMask, l);
                llvm::Value* tp = laneValue(texH, l);
                llvm::Value* sp = laneValue(sampH, l);
                b_.CreateStore(cInt(i32Ty(), 1), stSlot);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    llvm::SmallVector<llvm::Value*, 10> args{
                        tp, sp,
                        llvmExtract(nb, u, l),
                        llvmExtract(nb, v, l),
                        llvmExtract(nb, w, l),
                        llvmExtract(nb, offX, l),
                        llvmExtract(nb, offY, l),
                        llvmExtract(nb, comp, l)};
                    if (isCmp) args.push_back(llvmExtract(nb, cmpV, l));
                    args.push_back(out4);
                    auto* c = nb.CreateCall(fn, args);
                    nb.CreateStore(c, stSlot);
                });
                statusVec = llvmInsert(b_, statusVec, b_.CreateLoad(i32Ty(), stSlot), l);
                for (unsigned c = 0; c < 4; ++c) {
                    llvm::Value* ep = gepByteOffset(out4, c * 4);
                    rgba[c] = llvmInsert(b_, rgba[c], loadScalar(f32Ty(), ep), l);
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call->getType();
            for (unsigned c = 0; c < 4; ++c) st.fields.push_back(SimdVal::makeVector(rgba[c], f32Ty()));
            st.fields.push_back(SimdVal::makeVector(statusVec, i32Ty()));
            setMapped(call, st);
            return true;
        }

        if (op == 81) { // calculateLOD(op, srv, samp, coord0, coord1, coord2, clamped) -> float
            SimdVal texH = getMapped(call->getArgOperand(1));
            SimdVal sampH = getMapped(call->getArgOperand(2));
            llvm::Value* u = vecOrZero(3, f32Ty(), 0.0);
            llvm::Value* v = vecOrZero(4, f32Ty(), 0.0);
            llvm::Value* w = vecOrZero(5, f32Ty(), 0.0);
            llvm::Value* clampedVec = asVec(getMapped(call->getArgOperand(6)));

            auto projFn = getOrDeclareRuntimeFn(
                "nacelle_tex_project_lod_coords", nullptr,
                {ptrTy(), f32Ty(), f32Ty(), f32Ty(), ptrTy()});
            llvm::Value* out3 = createEntryAlloca(llvm::ArrayType::get(f32Ty(), 3), 1, 16);
            llvm::Value* su = splatFlt(f32Ty(), 0.0);
            llvm::Value* sv = su;
            llvm::Value* sw = su;
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, execMask, l);
                llvm::Value* tp = laneValue(texH, l);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    nb.CreateCall(projFn, {
                            tp,
                            llvmExtract(nb, u, l),
                            llvmExtract(nb, v, l),
                            llvmExtract(nb, w, l),
                            out3});
                });
                su = llvmInsert(b_, su, loadScalar(f32Ty(), out3), l);
                sv = llvmInsert(b_, sv, loadScalar(f32Ty(), gepByteOffset(out3, 4)), l);
                sw = llvmInsert(b_, sw, loadScalar(f32Ty(), gepByteOffset(out3, 8)), l);
            }

            llvm::Value* ddxU = splatFlt(f32Ty(), 0.0), *ddxV = ddxU, *ddxW = ddxU;
            llvm::Value* ddyU = ddxU, *ddyV = ddxU, *ddyW = ddxU;
            if (hooks_.isPixelShader) {
                auto [ux, uy] = computeQuadFiniteDiff(su, false);
                auto [vx, vy] = computeQuadFiniteDiff(sv, false);
                auto [wx, wy] = computeQuadFiniteDiff(sw, false);
                ddxU = ux; ddyU = uy; ddxV = vx; ddyV = vy; ddxW = wx; ddyW = wy;
            }

            auto lodFn = getOrDeclareRuntimeFn(
                "nacelle_tex_calc_lod", f32Ty(),
                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), i32Ty()});
            llvm::Value* resSlot = createEntryAlloca(f32Ty(), 1, 4);
            llvm::Value* resVec = splatFlt(f32Ty(), 0.0);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, execMask, l);
                llvm::Value* tp = laneValue(texH, l);
                llvm::Value* sp = laneValue(sampH, l);
                b_.CreateStore(cFlt(f32Ty(), 0.0), resSlot);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    llvm::Value* c1 = llvmExtract(nb, clampedVec, l);
                    llvm::Value* c32 = nb.CreateZExt(c1, nb.getInt32Ty());
                    auto* c = nb.CreateCall(lodFn, {
                            tp, sp,
                            llvmExtract(nb, ddxU, l),
                            llvmExtract(nb, ddxV, l),
                            llvmExtract(nb, ddxW, l),
                            llvmExtract(nb, ddyU, l),
                            llvmExtract(nb, ddyV, l),
                            llvmExtract(nb, ddyW, l),
                            c32});
                    nb.CreateStore(c, resSlot);
                });
                resVec = llvmInsert(b_, resVec, b_.CreateLoad(f32Ty(), resSlot), l);
            }
            setMapped(call, SimdVal::makeVector(resVec, f32Ty()));
            return true;
        }

        if (op == 64 || op == 65) { // sampleCmp / sampleCmpLevelZero
            SimdVal texH = getMapped(call->getArgOperand(1));
            SimdVal sampH = getMapped(call->getArgOperand(2));
            llvm::Value* u = vecOrZero(3, f32Ty(), 0.0);
            llvm::Value* v = vecOrZero(4, f32Ty(), 0.0);
            llvm::Value* w = vecOrZero(5, f32Ty(), 0.0);
            llvm::Value* offX = vecOrZero(7, i32Ty(), 0.0);
            llvm::Value* offY = vecOrZero(8, i32Ty(), 0.0);
            llvm::Value* cmpV = vecOrZero(10, f32Ty(), 0.0);
            uint32_t isL0 = (op == 65) ? 1u : 0u;
            llvm::Value* expLod = splatFlt(f32Ty(), 0.0);
            if (op == 64 && hooks_.isPixelShader) {
                auto [dux, duy] = computeQuadFiniteDiff(u, true);
                auto [dvx, dvy] = computeQuadFiniteDiff(v, true);
                llvm::Value* dx2 = b_.CreateFAdd(b_.CreateFMul(dux, dux), b_.CreateFMul(dvx, dvx));
                llvm::Value* dy2 = b_.CreateFAdd(b_.CreateFMul(duy, duy), b_.CreateFMul(dvy, dvy));
                llvm::Value* d2 = b_.CreateIntrinsic(llvm::Intrinsic::maxnum, {dx2->getType()}, {dx2, dy2});
                expLod = b_.CreateFMul(splatFlt(f32Ty(), 0.5), emitMathUnary("log2", createMaxNum(d2, splatFlt(f32Ty(), 1e-12))));
                expLod = b_.CreateIntrinsic(llvm::Intrinsic::maxnum, {expLod->getType()}, {expLod, splatFlt(f32Ty(), 0.0)});
            }
            auto fn = getOrDeclareRuntimeFn(
                "nacelle_tex_sample_cmp", f32Ty(),
                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            llvm::Value* statusPtr = createEntryAlloca(i32Ty(), 1, 4);
            llvm::Value* resSlot = createEntryAlloca(f32Ty(), 1, 4);
            llvm::Value* stSlot = createEntryAlloca(i32Ty(), 1, 4);
            llvm::Value* resVec = splatFlt(f32Ty(), 0.0);
            llvm::Value* statusVec = splatInt(i32Ty(), 1);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                llvm::Value* mL = llvmExtract(b_, execMask, l);
                llvm::Value* tp = laneValue(texH, l);
                llvm::Value* sp = laneValue(sampH, l);
                b_.CreateStore(cFlt(f32Ty(), 0.0), resSlot);
                b_.CreateStore(cInt(i32Ty(), 1), stSlot);
                emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                    auto* c = nb.CreateCall(fn, {
                            tp, sp,
                            llvmExtract(nb, u, l),
                            llvmExtract(nb, v, l),
                            llvmExtract(nb, w, l),
                            llvmExtract(nb, cmpV, l),
                            llvmExtract(nb, expLod, l),
                            cInt(i32Ty(), isL0),
                            llvmExtract(nb, offX, l),
                            llvmExtract(nb, offY, l),
                            statusPtr});
                    nb.CreateStore(c, resSlot);
                    nb.CreateStore(nb.CreateLoad(i32Ty(), statusPtr), stSlot);
                });
                resVec = llvmInsert(b_, resVec, b_.CreateLoad(f32Ty(), resSlot), l);
                statusVec = llvmInsert(b_, statusVec, b_.CreateLoad(i32Ty(), stSlot), l);
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call->getType();
            st.fields.push_back(SimdVal::makeVector(resVec, f32Ty()));
            for (unsigned c = 1; c < 4; ++c) st.fields.push_back(SimdVal::makeVector(splatFlt(f32Ty(), 0.0), f32Ty()));
            st.fields.push_back(SimdVal::makeVector(statusVec, i32Ty()));
            setMapped(call, st);
            return true;
        }

        // Sample (60), SampleBias (61), SampleLevel (62), SampleGrad (63)
        SimdVal texH = getMapped(call->getArgOperand(1));
        SimdVal sampH = getMapped(call->getArgOperand(2));
        llvm::Value* u = vecOrZero(3, f32Ty(), 0.0);
        llvm::Value* v = vecOrZero(4, f32Ty(), 0.0);
        llvm::Value* w = vecOrZero(5, f32Ty(), 0.0);
        llvm::Value* c3 = vecOrZero(6, f32Ty(), 0.0);
        llvm::Value* offX = vecOrZero(7, i32Ty(), 0.0);
        llvm::Value* offY = vecOrZero(8, i32Ty(), 0.0);
        llvm::Value* offZ = vecOrZero(9, i32Ty(), 0.0);

        uint32_t lodMode = 0;
        llvm::Value* ddxU = splatFlt(f32Ty(), 0.0), *ddxV = ddxU, *ddxW = ddxU, *ddyU = ddxU, *ddyV = ddxU, *ddyW = ddxU;
        llvm::Value* bias = ddxU, *expLod = ddxU, *clampLod = splatFlt(f32Ty(), -1.0);

        if (op == 60 || op == 61) {
            lodMode = (op == 61) ? 1u : 0u;
            if (hooks_.isPixelShader) {
                auto [ux, uy] = computeQuadFiniteDiff(u, false);
                auto [vx, vy] = computeQuadFiniteDiff(v, false);
                auto [wx, wy] = computeQuadFiniteDiff(w, false);
                ddxU = ux; ddyU = uy; ddxV = vx; ddyV = vy; ddxW = wx; ddyW = wy;
            }
            if (op == 61) {
                bias = vecOrZero(10, f32Ty(), 0.0);
                clampLod = vecOrZero(11, f32Ty(), -1.0);
            } else {
                clampLod = vecOrZero(10, f32Ty(), -1.0);
            }
        } else if (op == 62) {
            lodMode = 2u;
            expLod = vecOrZero(10, f32Ty(), 0.0);
        } else if (op == 63) {
            lodMode = 0u;
            ddxU = vecOrZero(10, f32Ty(), 0.0);
            ddxV = vecOrZero(11, f32Ty(), 0.0);
            ddxW = vecOrZero(12, f32Ty(), 0.0);
            ddyU = vecOrZero(13, f32Ty(), 0.0);
            ddyV = vecOrZero(14, f32Ty(), 0.0);
            ddyW = vecOrZero(15, f32Ty(), 0.0);
            clampLod = vecOrZero(16, f32Ty(), -1.0);
        }

        auto fn = getOrDeclareRuntimeFn(
            "nacelle_tex_sample", i32Ty(),
            {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(),
             f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(),
             f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), ptrTy()});
        llvm::Value* out4 = createEntryAlloca(llvm::ArrayType::get(f32Ty(), 4), 1, 16);
        llvm::Value* stSlot = createEntryAlloca(i32Ty(), 1, 4);
        llvm::SmallVector<llvm::Value*, 4> rgba(4, splatFlt(f32Ty(), 0.0));
        llvm::Value* statusVec = splatInt(i32Ty(), 1);
        for (uint32_t l = 0; l < W_; ++l) {
            if (isLaneMaskConstFalse(execMask, l)) continue;
            llvm::Value* mL = llvmExtract(b_, execMask, l);
            llvm::Value* tp = laneValue(texH, l);
            llvm::Value* sp = laneValue(sampH, l);
            b_.CreateStore(cInt(i32Ty(), 1), stSlot);
            emitIf(b_, mL, [&](llvm::IRBuilder<>& nb) {
                auto ex = [&](llvm::Value* vv) { return llvmExtract(nb, vv, l); };
                auto* c = nb.CreateCall(fn, {
                        tp, sp, ex(u), ex(v), ex(w), ex(c3),
                        ex(ddxU), ex(ddxV), ex(ddxW), ex(ddyU), ex(ddyV), ex(ddyW),
                        ex(bias), ex(expLod),
                        cInt(i32Ty(), lodMode),
                        ex(offX), ex(offY), ex(offZ), ex(clampLod), out4});
                nb.CreateStore(c, stSlot);
            });
            statusVec = llvmInsert(b_, statusVec, b_.CreateLoad(i32Ty(), stSlot), l);
            for (unsigned c = 0; c < 4; ++c) {
                llvm::Value* ep = gepByteOffset(out4, c * 4);
                rgba[c] = llvmInsert(b_, rgba[c], loadScalar(f32Ty(), ep), l);
            }
        }
        SimdVal st;
        st.kind = SimdVal::Kind::Struct;
        st.origType = call->getType();
        for (unsigned c = 0; c < 4; ++c) st.fields.push_back(SimdVal::makeVector(rgba[c], f32Ty()));
        st.fields.push_back(SimdVal::makeVector(statusVec, i32Ty()));
        setMapped(call, st);
        return true;
    }

    bool packetizeWaveOp(llvm::CallInst* call, uint32_t op, llvm::Value* activeMask, std::string&) {
        auto arg = [&](unsigned i) { return call->getArgOperand(i); };

        if (op == 111) { // waveGetLaneIndex
            setMapped(call, SimdVal::makeVector(laneIota(), i32Ty()));
            return true;
        }
        if (op == 112) { // waveGetLaneCount
            setMapped(call, {SimdVal::Kind::Vector, i32Ty(), splatInt(i32Ty(), W_), {}, {}});
            return true;
        }

        // Convert activeMask (vector<Wxi1>) into 4 x i32 ballot words covering up to 128 lanes.
        auto buildBallotWords = [&](llvm::Value* m) -> std::array<llvm::Value*, 4> {
            llvm::Value* zero = cInt(i32Ty(), 0);
            if (W_ <= 32) {
                llvm::Value* bits = b_.CreateZExt(m, vecTy(i32Ty()));
                llvm::Value* shl = b_.CreateShl(bits, laneIota());
                llvm::Value* w0 = b_.CreateOrReduce(shl);
                return {w0, zero, zero, zero};
            }
            llvm::FixedVectorType* v32i32 = llvm::FixedVectorType::get(i32Ty(), 32);
            llvm::SmallVector<llvm::Constant*, 32> iotaConsts(32);
            for (int32_t i = 0; i < 32; ++i) iotaConsts[i] = llvm::ConstantInt::get(i32Ty(), i);
            llvm::Value* iota32 = llvm::ConstantVector::get(iotaConsts);
            std::array<llvm::Value*, 4> words{zero, zero, zero, zero};
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (uint32_t w = 0; w < numWords; ++w) {
                llvm::SmallVector<int32_t, 32> sliceIdx(32);
                for (int32_t i = 0; i < 32; ++i) sliceIdx[i] = static_cast<int32_t>(w * 32u) + i;
                llvm::Value* slice = b_.CreateShuffleVector(m, m, sliceIdx);
                llvm::Value* bits = b_.CreateZExt(slice, v32i32);
                llvm::Value* shl = b_.CreateShl(bits, iota32);
                words[w] = b_.CreateOrReduce(shl);
            }
            return words;
        };

        auto buildFirstActiveLane = [&](llvm::Value* m) -> llvm::Value* {
            auto words = buildBallotWords(m);
            if (W_ <= 32) {
                llvm::Value* tz = b_.CreateIntrinsic(llvm::Intrinsic::cttz, {words[0]->getType()}, {words[0], b_.getFalse()});
                return b_.CreateIntrinsic(llvm::Intrinsic::umin, {tz->getType()}, {tz, cInt(i32Ty(), W_ - 1)});
            }
            llvm::Value* firstIdx = cInt(i32Ty(), W_ - 1);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (int w = static_cast<int>(numWords) - 1; w >= 0; --w) {
                llvm::Value* nz = b_.CreateICmp(llvm::CmpInst::ICMP_NE, words[w], cInt(i32Ty(), 0));
                llvm::Value* tz = b_.CreateIntrinsic(llvm::Intrinsic::cttz, {words[w]->getType()}, {words[w], b_.getFalse()});
                llvm::Value* idx = b_.CreateAdd(cInt(i32Ty(), static_cast<uint64_t>(w) * 32u), tz);
                firstIdx = b_.CreateSelect(nz, idx, firstIdx);
            }
            return firstIdx;
        };

        if (op == 110) { // waveIsFirstLane
            llvm::Value* firstIdx = buildFirstActiveLane(activeMask);
            llvm::Value* firstVec = llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(i32Ty()))->getElementType(), firstIdx);
            llvm::Value* isFirst = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, laneIota(), firstVec);
            isFirst = b_.CreateAnd(isFirst, activeMask);
            setMapped(call, SimdVal::makeVector(isFirst, i1Ty()));
            return true;
        }
        if (op == 113 || op == 114) { // waveAnyTrue / waveAllTrue
            llvm::Value* condVec = asVec(getMapped(arg(1)));
            llvm::Value* scalarRes;
            if (op == 113) {
                llvm::Value* actCond = b_.CreateAnd(condVec, activeMask);
                scalarRes = b_.CreateOrReduce(actCond);
            } else {
                // !active || cond across all lanes
                llvm::Value* inact = b_.CreateXor(activeMask, allTrueMask());
                llvm::Value* impl = b_.CreateOr(condVec, inact);
                scalarRes = b_.CreateAndReduce(impl);
            }
            llvm::Value* bvec = llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(i1Ty()))->getElementType(), scalarRes);
            setMapped(call, SimdVal::makeVector(bvec, i1Ty()));
            return true;
        }
        if (op == 116) { // waveActiveBallot(cond) -> %dx.types.fouri32
            llvm::Value* condVec = b_.CreateAnd(asVec(getMapped(arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call->getType();
            for (int i = 0; i < 4; ++i) {
                st.fields.push_back({SimdVal::Kind::Vector, i32Ty(),
                                     llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(i32Ty()))->getElementType(), words[i]), {}, {}});
            }
            setMapped(call, st);
            return true;
        }
        if (op == 118) { // waveReadLaneFirst(val)
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            llvm::Value* firstIdx = buildFirstActiveLane(activeMask);
            llvm::Value* scalar = b_.CreateExtractElement(valVec, firstIdx);
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            setMapped(call, {SimdVal::Kind::Vector, et, llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(et))->getElementType(), scalar), {}, {}});
            return true;
        }
        if (op == 117) { // waveReadLaneAt(val, lane)
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            llvm::Value* laneVec = asVec(getMapped(arg(2)));
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            llvm::Value* outVec = llvm::PoisonValue::get(vecTy(et));
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* li = llvmExtract(b_, laneVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* clamped = b_.CreateIntrinsic(llvm::Intrinsic::umin, {li->getType()}, {li, cInt(i32Ty(), W_ - 1)});
                llvm::Value* s = b_.CreateExtractElement(valVec, clamped);
                outVec = llvmInsert(b_, outVec, s, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(call, SimdVal::makeVector(outVec, et));
            return true;
        }
        if (op == 115) { // waveActiveAllEqual(val)
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            llvm::Value* firstIdx = buildFirstActiveLane(activeMask);
            llvm::Value* firstScalar = b_.CreateExtractElement(valVec, firstIdx);
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            llvm::Value* firstVec = llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(et))->getElementType(), firstScalar);
            llvm::Value* eq = et->isFloatingPointTy()
                                 ? b_.CreateFCmp(llvm::CmpInst::FCMP_OEQ, valVec, firstVec)
                                 : b_.CreateICmp(llvm::CmpInst::ICMP_EQ, valVec, firstVec);
            llvm::Value* inact = b_.CreateXor(activeMask, allTrueMask());
            llvm::Value* allEq = b_.CreateAndReduce(b_.CreateOr(eq, inact));
            setMapped(call, {SimdVal::Kind::Vector, i1Ty(), llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(i1Ty()))->getElementType(), allEq), {}, {}});
            return true;
        }
        if (op == 119 || op == 120) { // waveActiveOp / waveActiveBit
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            uint32_t sop = (op == 119) ? static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0)) : 0u;
            bool isFloat = et->isFloatingPointTy();

            llvm::Value* identity;
            enum class WaveRedKind { And, Or, Xor, Add, Mul, MinF, MaxF, MinUI, MaxUI, MinSI, MaxSI };
            WaveRedKind rk = WaveRedKind::Add;
            if (op == 120) {
                if (kind == 0) { // AND
                    rk = WaveRedKind::And;
                    identity = splatInt(et, ~uint64_t(0));
                } else if (kind == 1) { // OR
                    rk = WaveRedKind::Or;
                    identity = splatInt(et, 0);
                } else { // XOR
                    rk = WaveRedKind::Xor;
                    identity = splatInt(et, 0);
                }
            } else if (kind == 0) { // SUM
                rk = WaveRedKind::Add;
                identity = isFloat ? splatFlt(et, 0.0) : splatInt(et, 0);
            } else if (kind == 1) { // PROD
                rk = WaveRedKind::Mul;
                identity = isFloat ? splatFlt(et, 1.0) : splatInt(et, 1);
            } else if (kind == 2) { // MIN
                if (isFloat) {
                    rk = WaveRedKind::MinF;
                    identity = splatFlt(et, std::numeric_limits<double>::infinity());
                } else if (sop != 0) {
                    rk = WaveRedKind::MinUI;
                    identity = splatInt(et, ~uint64_t(0));
                } else {
                    rk = WaveRedKind::MinSI;
                    unsigned bw = et->getScalarSizeInBits();
                    uint64_t sMax = (bw >= 64) ? uint64_t(INT64_MAX) : ((uint64_t(1) << (bw - 1)) - 1);
                    identity = splatInt(et, sMax);
                }
            } else { // MAX
                if (isFloat) {
                    rk = WaveRedKind::MaxF;
                    identity = splatFlt(et, -std::numeric_limits<double>::infinity());
                } else if (sop != 0) {
                    rk = WaveRedKind::MaxUI;
                    identity = splatInt(et, 0);
                } else {
                    rk = WaveRedKind::MaxSI;
                    unsigned bw = et->getPrimitiveSizeInBits();
                    uint64_t sMin = (bw >= 64) ? uint64_t(INT64_MIN) : (uint64_t(1) << (bw - 1));
                    identity = splatInt(et, sMin);
                }
            }
            llvm::Value* masked = b_.CreateSelect(activeMask, valVec, identity);
            llvm::Value* reduced = nullptr;
            switch (rk) {
            case WaveRedKind::And: reduced = b_.CreateAndReduce(masked); break;
            case WaveRedKind::Or: reduced = b_.CreateOrReduce(masked); break;
            case WaveRedKind::Xor: reduced = b_.CreateXorReduce(masked); break;
            case WaveRedKind::Add:
                reduced = isFloat
                              ? b_.CreateFAddReduce(cFlt(et, 0.0), masked)
                              : b_.CreateAddReduce(masked);
                break;
            case WaveRedKind::Mul:
                reduced = isFloat
                              ? b_.CreateFMulReduce(cFlt(et, 1.0), masked)
                              : b_.CreateMulReduce(masked);
                break;
            case WaveRedKind::MinF: reduced = b_.CreateFPMinReduce(masked); break;
            case WaveRedKind::MaxF: reduced = b_.CreateFPMaxReduce(masked); break;
            case WaveRedKind::MinUI: reduced = b_.CreateIntMinReduce(masked, /*IsSigned=*/false); break;
            case WaveRedKind::MaxUI: reduced = b_.CreateIntMaxReduce(masked, /*IsSigned=*/false); break;
            case WaveRedKind::MinSI: reduced = b_.CreateIntMinReduce(masked, /*IsSigned=*/true); break;
            case WaveRedKind::MaxSI: reduced = b_.CreateIntMaxReduce(masked, /*IsSigned=*/true); break;
            }
            setMapped(call, SimdVal::makeVector(llvmSplat(b_, et, reduced), et));
            return true;
        }
        if (op == 121) { // wavePrefixOp (exclusive prefix sum or prod)
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            bool isMul = (kind == 1);
            bool isFloat = et->isFloatingPointTy();
            llvm::Value* idScalar = isFloat ? cFlt(et, isMul ? 1.0 : 0.0) : cInt(et, isMul ? 1 : 0);
            llvm::Value* idVec = isFloat ? splatFlt(et, isMul ? 1.0 : 0.0) : splatInt(et, isMul ? 1 : 0);
            llvm::Value* contrib = b_.CreateSelect(activeMask, valVec, idVec);
            llvm::Value* acc = idScalar;
            llvm::Value* outVec = llvm::PoisonValue::get(vecTy(et));
            for (uint32_t l = 0; l < W_; ++l) {
                outVec = llvmInsert(b_, outVec, acc, l);
                llvm::Value* vL = llvmExtract(b_, contrib, l);
                if (isFloat) {
                    acc = isMul ? b_.CreateFMul(acc, vL)
                                : b_.CreateFAdd(acc, vL);
                } else {
                    acc = isMul ? b_.CreateMul(acc, vL)
                                : b_.CreateAdd(acc, vL);
                }
            }
            setMapped(call, SimdVal::makeVector(outVec, et));
            return true;
        }
        if (op == 123) { // quadOp(val, kind): 0=ReadAcrossX, 1=ReadAcrossY, 2=ReadAcrossDiagonal
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            uint32_t xorMask = (kind == 0) ? 1u : (kind == 1 ? 2u : 3u);
            llvm::SmallVector<int32_t, 16> indices;
            indices.reserve(W_);
            for (uint32_t l = 0; l < W_; ++l) {
                uint32_t peer = l ^ xorMask;
                indices.push_back(peer < W_ ? static_cast<int32_t>(peer) : static_cast<int32_t>(l));
            }
            llvm::Value* res = b_.CreateShuffleVector(valVec, valVec, indices);
            setMapped(call, SimdVal::makeVector(res, et));
            return true;
        }
        if (op == 135) { // waveAllBitCount(cond) -> i32
            llvm::Value* condVec = b_.CreateAnd(asVec(getMapped(arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            llvm::Value* cnt = b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {words[0]->getType()}, {words[0]});
            for (uint32_t w = 1; w < numWords; ++w) {
                llvm::Value* wc = b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {words[w]->getType()}, {words[w]});
                cnt = b_.CreateAdd(cnt, wc);
            }
            setMapped(call, SimdVal::makeVector(llvmSplat(b_, i32Ty(), cnt), i32Ty()));
            return true;
        }
        if (op == 136) { // wavePrefixBitCount(cond) -> i32
            llvm::Value* condVec = b_.CreateAnd(asVec(getMapped(arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            if (W_ <= 32) {
                llvm::Value* ballotVec = llvmSplat(b_, i32Ty(), words[0]);
                llvm::SmallVector<llvm::Constant*, 32> laneMasks;
                laneMasks.reserve(W_);
                for (uint32_t l = 0; l < W_; ++l) {
                    laneMasks.push_back(llvm::ConstantInt::get(i32Ty(), (uint32_t(1) << l) - 1u));
                }
                llvm::Value* maskVec = llvm::ConstantVector::get(laneMasks);
                llvm::Value* masked = b_.CreateAnd(ballotVec, maskVec);
                llvm::Value* cntVec = b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {masked->getType()}, {masked});
                setMapped(call, SimdVal::makeVector(cntVec, i32Ty()));
                return true;
            }
            llvm::FixedVectorType* v32i32 = llvm::FixedVectorType::get(i32Ty(), 32);
            llvm::SmallVector<llvm::Constant*, 32> laneMasks32(32);
            for (int32_t l = 0; l < 32; ++l) {
                laneMasks32[l] = llvm::ConstantInt::get(i32Ty(), (uint32_t(1) << l) - 1u);
            }
            llvm::Value* maskVec32 = llvm::ConstantVector::get(laneMasks32);
            llvm::Value* p32 = llvm::PoisonValue::get(v32i32);
            llvm::SmallVector<int32_t, 32> zeros32(32, 0);
            auto splat32 = [&](llvm::Value* s) {
                llvm::Value* ins = llvmInsert(b_, p32, s, 0);
                return b_.CreateShuffleVector(ins, p32, zeros32);
            };
            llvm::Value* outVec = llvm::PoisonValue::get(vecTy(i32Ty()));
            llvm::Value* prefixSum = cInt(i32Ty(), 0);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (uint32_t w = 0; w < numWords; ++w) {
                llvm::Value* bVec = splat32(words[w]);
                llvm::Value* anded = b_.CreateAnd(bVec, maskVec32);
                llvm::Value* inWord = b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {anded->getType()}, {anded});
                llvm::Value* chunk = b_.CreateAdd(inWord, splat32(prefixSum));
                for (uint32_t i = 0; i < 32u && (w * 32u + i) < W_; ++i) {
                    outVec = llvmInsert(b_, outVec, llvmExtract(b_, chunk, i), w * 32u + i);
                }
                if (w + 1 < numWords) {
                    prefixSum = b_.CreateAdd(prefixSum, b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {words[w]->getType()}, {words[w]}));
                }
            }
            setMapped(call, SimdVal::makeVector(outVec, i32Ty()));
            return true;
        }
        if (op == 166) { // waveMultiPrefixOp(value, mask0, mask1, mask2, mask3, op, sop)
            llvm::Value* valVec = asVec(getMapped(arg(1)));
            std::array<llvm::Value*, 4> maskVecs{
                asVec(getMapped(arg(2))),
                asVec(getMapped(arg(3))),
                asVec(getMapped(arg(4))),
                asVec(getMapped(arg(5))),
            };
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(valVec->getType())->getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(6)).value_or(0)); // 0=Sum, 1=BitAnd, 2=BitOr, 3=BitXor, 4=Prod
            bool isFloat = et->isFloatingPointTy();
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);

            std::string fnName = "__nacelle_wave_multi_prefix_" + std::to_string(kind) + "_" +
                                 (isFloat ? "f" : "i") + std::to_string(et->getPrimitiveSizeInBits()) +
                                 "_w" + std::to_string(W_);
            llvm::Function* helperFn = mod_.getFunction(fnName);
            if (!helperFn) {
                llvm::SmallVector<llvm::Type*, 6> argTypes;
                argTypes.push_back(vecTy(et));
                for (uint32_t w = 0; w < numWords; ++w) {
                    argTypes.push_back(vecTy(i32Ty()));
                }
                argTypes.push_back(vecTy(i1Ty()));
                auto* fnTy = llvm::FunctionType::get(vecTy(et), argTypes, false);

                llvm::IRBuilderBase::InsertPointGuard guard(b_);
                helperFn = llvm::Function::Create(fnTy, llvm::GlobalValue::InternalLinkage, fnName, &mod_);
                llvm::BasicBlock* entry = llvm::BasicBlock::Create(b_.getContext(), "entry", helperFn);
                b_.SetInsertPoint(entry);

                llvm::Value* fnValVec = helperFn->getArg(0);
                llvm::SmallVector<llvm::Value*, 4> fnMaskVecs;
                for (uint32_t w = 0; w < numWords; ++w) {
                    fnMaskVecs.push_back(helperFn->getArg(1 + w));
                }
                llvm::Value* fnActiveMask = helperFn->getArg(1 + numWords);

                llvm::Value* idScalar = nullptr;
                if (kind == 1) {
                    idScalar = cInt(et, ~uint64_t(0));
                } else if (kind == 4) {
                    idScalar = isFloat ? cFlt(et, 1.0) : cInt(et, 1);
                } else {
                    idScalar = isFloat ? cFlt(et, 0.0) : cInt(et, 0);
                }

                llvm::Value* c0 = cInt(i32Ty(), 0);
                llvm::Value* c1 = cInt(i32Ty(), 1);
                llvm::Value* cW = cInt(i32Ty(), W_);
                llvm::Value* outSlot = b_.CreateAlloca(vecTy(et));
                llvm::Value* accSlot = b_.CreateAlloca(et);
                b_.CreateStore(llvm::PoisonValue::get(vecTy(et)), outSlot);

                emitForLoop(b_, c0, cW, c1,
                            [&](llvm::IRBuilder<>& ob, llvm::Value* lIdx) {
                                llvm::SmallVector<llvm::Value*, 4> maskL(numWords);
                                for (uint32_t w = 0; w < numWords; ++w) {
                                    maskL[w] = ob.CreateExtractElement(fnMaskVecs[w], lIdx);
                                }
                                ob.CreateStore(idScalar, accSlot);
                                emitForLoop(ob, c0, lIdx, c1,
                                            [&](llvm::IRBuilder<>& ib, llvm::Value* jIdx) {
                                                llvm::Value* acc = ib.CreateLoad(et, accSlot);
                                                llvm::Value* j32 = jIdx;
                                                llvm::Value* maskWordL = maskL[0];
                                                if (numWords > 1) {
                                                    llvm::Value* wIdx = ib.CreateLShr(j32, cInt(i32Ty(), 5));
                                                    for (uint32_t w = 1; w < numWords; ++w) {
                                                        llvm::Value* isW = ib.CreateICmp(llvm::CmpInst::ICMP_EQ, wIdx, cInt(i32Ty(), w));
                                                        maskWordL = ib.CreateSelect(isW, maskL[w], maskWordL);
                                                    }
                                                }
                                                llvm::Value* bitIdx = ib.CreateAnd(j32, cInt(i32Ty(), 31));
                                                llvm::Value* shifted = ib.CreateLShr(maskWordL, bitIdx);
                                                llvm::Value* bitJ = ib.CreateAnd(shifted, cInt(i32Ty(), 1));
                                                llvm::Value* inGroup = ib.CreateICmp(llvm::CmpInst::ICMP_NE, bitJ, cInt(i32Ty(), 0));
                                                llvm::Value* actJ = ib.CreateExtractElement(fnActiveMask, jIdx);
                                                llvm::Value* take = ib.CreateAnd(inGroup, actJ);
                                                llvm::Value* valJ = ib.CreateExtractElement(fnValVec, jIdx);

                                                llvm::Value* cand = nullptr;
                                                if (kind == 0) {
                                                    cand = isFloat ? ib.CreateFAdd(acc, valJ)
                                                                   : ib.CreateAdd(acc, valJ);
                                                } else if (kind == 1) {
                                                    cand = ib.CreateAnd(acc, valJ);
                                                } else if (kind == 2) {
                                                    cand = ib.CreateOr(acc, valJ);
                                                } else if (kind == 3) {
                                                    cand = ib.CreateXor(acc, valJ);
                                                } else {
                                                    cand = isFloat ? ib.CreateFMul(acc, valJ)
                                                                   : ib.CreateMul(acc, valJ);
                                                }
                                                ib.CreateStore(ib.CreateSelect(take, cand, acc), accSlot);
                                            });
                                llvm::Value* curOut = ob.CreateLoad(vecTy(et), outSlot);
                                llvm::Value* accRes = ob.CreateLoad(et, accSlot);
                                ob.CreateStore(ob.CreateInsertElement(curOut, accRes, lIdx), outSlot);
                            });
                b_.CreateRet(b_.CreateLoad(vecTy(et), outSlot));
            }

            llvm::SmallVector<llvm::Value*, 6> callArgs;
            callArgs.push_back(valVec);
            for (uint32_t w = 0; w < numWords; ++w) {
                callArgs.push_back(maskVecs[w]);
            }
            callArgs.push_back(activeMask);
            llvm::Value* callRes = b_.CreateCall(helperFn, callArgs);
            setMapped(call, SimdVal::makeVector(callRes, et));
            return true;
        }
        return false;
    }

    bool packetizeDxrOp(llvm::CallInst* call, uint32_t op, llvm::Value* execMask, std::string& outError) {
        auto arg = [&](unsigned i) -> llvm::Value* {
            return i < call->arg_size() ? call->getArgOperand(i) : nullptr;
        };
        auto laneVec = [&](llvm::Value* v) -> llvm::Value* {
            return asVec(getMapped(v));
        };
        auto activeAt = [&](uint32_t l) -> llvm::Value* {
            return llvmExtract(b_, execMask, l);
        };

        auto runPerActiveLaneVoid = [&](auto&& emitCall) {
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* act = activeAt(l);
                emitIf(b_, act, [&](llvm::IRBuilder<>& nb) {
                    emitCall(nb, l);
                });
            }
        };

        auto runPerActiveLaneScalar = [&](llvm::Type* elemTy, llvm::Value* fallbackVal, auto&& emitCall) {
            llvm::Value* outVec = llvmSplat(b_, elemTy, fallbackVal);
            llvm::Value* slot = createEntryAlloca(elemTy, 1, 16);
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* act = activeAt(l);
                b_.CreateStore(fallbackVal, slot);
                emitIf(b_, act, [&](llvm::IRBuilder<>& nb) {
                    nb.CreateStore(emitCall(nb, l), slot);
                });
                outVec = llvmInsert(b_, outVec, b_.CreateLoad(elemTy, slot), l);
            }
            setMapped(call, SimdVal::makeVector(outVec, elemTy));
        };

        // DispatchRaysIndex (145) & DispatchRaysDimensions (146)
        if (op == 145 || op == 146) {
            const auto& arr = (op == 145) ? hooks_.threadId : hooks_.dispatchRaysDim;
            auto bcastI32 = [&](llvm::Value* scalar) -> llvm::Value* {
                return llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(i32Ty()))->getElementType(), scalar);
            };
            if (auto comp = getConstIntValue(arg(1))) {
                const uint32_t c = static_cast<uint32_t>(*comp);
                llvm::Value* v = (c < 3 && arr[c]) ? arr[c] : bcastI32(cInt(i32Ty(), 0));
                setMapped(call, SimdVal::makeVector(v, i32Ty()));
                return true;
            }
            llvm::Value* compVec = b_.CreateSExt(laneVec(arg(1)), vecTy(i32Ty()));
            llvm::Value* v0 = arr[0] ? arr[0] : bcastI32(cInt(i32Ty(), 0));
            llvm::Value* v1 = arr[1] ? arr[1] : bcastI32(cInt(i32Ty(), 0));
            llvm::Value* v2 = arr[2] ? arr[2] : bcastI32(cInt(i32Ty(), 0));
            llvm::Value* is1 = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, compVec, bcastI32(cInt(i32Ty(), 1)));
            llvm::Value* is2 = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, compVec, bcastI32(cInt(i32Ty(), 2)));
            llvm::Value* sel01 = b_.CreateSelect(is1, v1, v0);
            llvm::Value* res = b_.CreateSelect(is2, v2, sel01);
            setMapped(call, SimdVal::makeVector(res, i32Ty()));
            return true;
        }

        // TraceRay (157)
        if (op == 157) {
            SimdVal tlasHandle = getMapped(arg(1));
            llvm::Value* uniformTlasPtr = nullptr;
            if (tlasHandle.lanes.size() == 1) {
                uniformTlasPtr = loadDescBasePtr(tlasHandle.lanes[0]);
            }
            llvm::Value* flagsVec = laneVec(arg(2));
            llvm::Value* maskVec = laneVec(arg(3));
            llvm::Value* rayContribVec = laneVec(arg(4));
            llvm::Value* geomMultVec = laneVec(arg(5));
            llvm::Value* missIdxVec = laneVec(arg(6));
            llvm::Value* oxVec = laneVec(arg(7));
            llvm::Value* oyVec = laneVec(arg(8));
            llvm::Value* ozVec = laneVec(arg(9));
            llvm::Value* tMinVec = laneVec(arg(10));
            llvm::Value* dxVec = laneVec(arg(11));
            llvm::Value* dyVec = laneVec(arg(12));
            llvm::Value* dzVec = laneVec(arg(13));
            llvm::Value* tMaxVec = laneVec(arg(14));
            SimdVal payloadVal = getMapped(arg(15));

            auto fn = getOrDeclareRuntimeFn(
                "nacelle_dxr_trace_ray", nullptr,
                {ptrTy(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), i32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty(), ptrTy()});

            runPerActiveLaneVoid([&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* tlasPtr = uniformTlasPtr;
                if (!tlasPtr) {
                    llvm::Value* descPtr = l < tlasHandle.lanes.size()
                                              ? tlasHandle.lanes[l]
                                              : llvm::Constant::getNullValue(ptrTy());
                    tlasPtr = nb.CreateLoad(ptrTy(), descPtr);
                }
                llvm::Value* pPtr = l < payloadVal.lanes.size()
                                       ? payloadVal.lanes[l]
                                       : (!payloadVal.lanes.empty()
                                              ? payloadVal.lanes[0]
                                              : llvm::Constant::getNullValue(ptrTy()));
                auto ext = [&](llvm::Value* v) -> llvm::Value* {
                    return llvmExtract(nb, v, static_cast<uint32_t>(static_cast<int64_t>(l)));
                };
                nb.CreateCall(fn, {tlasPtr, ext(flagsVec), ext(maskVec), ext(rayContribVec),
                                     ext(geomMultVec), ext(missIdxVec), ext(oxVec), ext(oyVec),
                                     ext(ozVec), ext(tMinVec), ext(dxVec), ext(dyVec),
                                     ext(dzVec), ext(tMaxVec), pPtr});
            });
            return true;
        }

        // ReportHit (158)
        if (op == 158) {
            llvm::Value* tHitVec = laneVec(arg(1));
            llvm::Value* hitKindVec = laneVec(arg(2));
            SimdVal attrVal = getMapped(arg(3));
            int32_t attrBytes = 8;
            if (auto* allocaOp = llvm::dyn_cast<llvm::AllocaInst>(arg(3))) {
                attrBytes = static_cast<int32_t>(mod_.getDataLayout().getTypeAllocSize(allocaOp->getAllocatedType()));
            }
            llvm::Value* attrBytesConst = cInt(i32Ty(), static_cast<int64_t>(attrBytes));
            auto fn = getOrDeclareRuntimeFn(
                "nacelle_dxr_report_hit", i32Ty(), {f32Ty(), i32Ty(), ptrTy(), i32Ty()});
            runPerActiveLaneScalar(i1Ty(), cInt(i1Ty(), 0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* t = llvmExtract(nb, tHitVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* k = llvmExtract(nb, hitKindVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* aPtr = l < attrVal.lanes.size()
                                       ? attrVal.lanes[l]
                                       : (!attrVal.lanes.empty()
                                              ? attrVal.lanes[0]
                                              : llvm::Constant::getNullValue(ptrTy()));
                llvm::Value* r = nb.CreateCall(fn, {t, k, aPtr, attrBytesConst});
                return nb.CreateTrunc(r, i1Ty());
            });
            return true;
        }

        // CallShader (159)
        if (op == 159) {
            llvm::Value* idxVec = laneVec(arg(1));
            SimdVal paramVal = getMapped(arg(2));
            auto fn = getOrDeclareRuntimeFn("nacelle_dxr_call_shader", nullptr, {i32Ty(), ptrTy()});
            runPerActiveLaneVoid([&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* idx = llvmExtract(nb, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* pPtr = l < paramVal.lanes.size()
                                       ? paramVal.lanes[l]
                                       : (!paramVal.lanes.empty()
                                              ? paramVal.lanes[0]
                                              : llvm::Constant::getNullValue(ptrTy()));
                nb.CreateCall(fn, {idx, pPtr});
            });
            return true;
        }

        // IgnoreHit (155) & AcceptHitAndEndSearch (156)
        if (op == 155 || op == 156) {
            const char* fnName = (op == 155) ? "nacelle_dxr_ignore_hit" : "nacelle_dxr_accept_hit_and_end_search";
            auto fn = getOrDeclareRuntimeFn(fnName, nullptr, {});
            runPerActiveLaneVoid([&](llvm::IRBuilder<>& nb, uint32_t) {
                nb.CreateCall(fn, {});
            });
            return true;
        }

        // Scalar float state queries: RayTMin (153), RayTCurrent (154)
        if (op == 153 || op == 154) {
            auto fn = getOrDeclareRuntimeFn("nacelle_dxr_state_f32", f32Ty(), {i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(f32Ty(), 0.0), [&](llvm::IRBuilder<>& nb, uint32_t) {
                return nb.CreateCall(fn, {opConst});
            });
            return true;
        }

        // Vector float state queries: WorldRayOrigin (147), WorldRayDirection (148),
        // ObjectRayOrigin (149), ObjectRayDirection (150)
        if (op >= 147 && op <= 150) {
            llvm::Value* compVec = b_.CreateSExt(laneVec(arg(1)), vecTy(i32Ty()));
            auto fn = getOrDeclareRuntimeFn("nacelle_dxr_state_vec", f32Ty(), {i32Ty(), i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(f32Ty(), 0.0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* c = llvmExtract(nb, compVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {opConst, c});
            });
            return true;
        }

        // Matrix float state queries: ObjectToWorld (151), WorldToObject (152)
        if (op == 151 || op == 152) {
            llvm::Value* rowVec = laneVec(arg(1));
            llvm::Value* colVec = b_.CreateSExt(laneVec(arg(2)), vecTy(i32Ty()));
            auto fn = getOrDeclareRuntimeFn("nacelle_dxr_state_mat", f32Ty(), {i32Ty(), i32Ty(), i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(f32Ty(), 0.0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* r = llvmExtract(nb, rowVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* c = llvmExtract(nb, colVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {opConst, r, c});
            });
            return true;
        }

        // Scalar i32 state queries: InstanceID (141), InstanceIndex (142), HitKind (143),
        // RayFlags (144), PrimitiveIndex (161), GeometryIndex (213)
        if (op == 141 || op == 142 || op == 143 || op == 144 || op == 161 || op == 213) {
            auto fn = getOrDeclareRuntimeFn("nacelle_dxr_state_i32", i32Ty(), {i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(i32Ty(), cInt(i32Ty(), 0), [&](llvm::IRBuilder<>& nb, uint32_t) {
                return nb.CreateCall(fn, {opConst});
            });
            return true;
        }

        outError = "Unsupported DXR opcode: " + std::to_string(op);
        return false;
    }

    bool packetizeRayQueryOp(llvm::CallInst* call, uint32_t op, llvm::Value* execMask) {
        auto arg = [&](unsigned i) -> llvm::Value* {
            return i < call->arg_size() ? call->getArgOperand(i) : nullptr;
        };
        auto laneVec = [&](llvm::Value* v) -> llvm::Value* {
            return asVec(getMapped(v));
        };
        auto laneAt = [&](llvm::Value* vec, uint32_t l) -> llvm::Value* {
            return llvmExtract(b_, vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
        };
        auto activeAt = [&](uint32_t l) -> llvm::Value* {
            return llvmExtract(b_, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
        };

        auto runPerActiveLaneVoid = [&](auto&& emitCall) {
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* act = activeAt(l);
                emitIf(b_, act, [&](llvm::IRBuilder<>& nb) {
                    emitCall(nb, l);
                });
            }
        };

        auto runPerActiveLaneScalar = [&](llvm::Type* elemTy, llvm::Value* fallbackVal, auto&& emitCall) {
            llvm::Value* outVec = llvmSplat(b_, llvm::cast<llvm::FixedVectorType>(vecTy(elemTy))->getElementType(), fallbackVal);
            llvm::Value* laneSlot = createEntryAlloca(elemTy, 1);
            for (uint32_t l = 0; l < W_; ++l) {
                llvm::Value* act = activeAt(l);
                emitIfElse(b_, act,
                    [&](llvm::IRBuilder<>& nb) {
                        llvm::Value* val = emitCall(nb, l);
                        nb.CreateStore(val, laneSlot);
                    },
                    [&](llvm::IRBuilder<>& nb) {
                        nb.CreateStore(fallbackVal, laneSlot);
                    });
                llvm::Value* laneRes = b_.CreateLoad(elemTy, laneSlot);
                outVec = llvmInsert(b_, outVec, laneRes, static_cast<uint32_t>(l));
            }
            setMapped(call, SimdVal::makeVector(outVec, elemTy));
        };

        // AllocateRayQuery (178, 258)
        if (op == 178 || op == 258) {
            llvm::Value* flagsVec = laneVec(arg(1));
            auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_alloc", i32Ty(), {i32Ty()});
            runPerActiveLaneScalar(i32Ty(), cInt(i32Ty(), 0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* fl = llvmExtract(nb, flagsVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {fl});
            });
            return true;
        }

        // RayQuery_TraceRayInline (179)
        if (op == 179) {
            llvm::Value* qVec = laneVec(arg(1));
            SimdVal tlasHandle = getMapped(arg(2));
            llvm::Value* uniformTlasPtr = nullptr;
            if (tlasHandle.lanes.size() == 1) {
                uniformTlasPtr = loadDescBasePtr(tlasHandle.lanes[0]);
            }
            llvm::Value* flagsVec = laneVec(arg(3));
            llvm::Value* maskVec = laneVec(arg(4));
            llvm::Value* oxVec = laneVec(arg(5));
            llvm::Value* oyVec = laneVec(arg(6));
            llvm::Value* ozVec = laneVec(arg(7));
            llvm::Value* tMinVec = laneVec(arg(8));
            llvm::Value* dxVec = laneVec(arg(9));
            llvm::Value* dyVec = laneVec(arg(10));
            llvm::Value* dzVec = laneVec(arg(11));
            llvm::Value* tMaxVec = laneVec(arg(12));

            auto fn = getOrDeclareRuntimeFn(
                "nacelle_rayquery_trace_inline", nullptr,
                {i32Ty(), ptrTy(), i32Ty(), i32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty()});

            runPerActiveLaneVoid([&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* tlasPtr = uniformTlasPtr;
                if (!tlasPtr) {
                    llvm::Value* descPtr = l < tlasHandle.lanes.size()
                                              ? tlasHandle.lanes[l]
                                              : llvm::Constant::getNullValue(ptrTy());
                    tlasPtr = nb.CreateLoad(ptrTy(), descPtr);
                }
                auto ext = [&](llvm::Value* v) -> llvm::Value* {
                    return llvmExtract(nb, v, static_cast<uint32_t>(static_cast<int64_t>(l)));
                };
                nb.CreateCall(fn, {ext(qVec), tlasPtr, ext(flagsVec), ext(maskVec),
                                     ext(oxVec), ext(oyVec), ext(ozVec), ext(tMinVec),
                                     ext(dxVec), ext(dyVec), ext(dzVec), ext(tMaxVec)});
            });
            return true;
        }

        // RayQuery_Proceed (180)
        if (op == 180) {
            llvm::Value* qVec = laneVec(arg(1));
            auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_proceed", i32Ty(), {i32Ty()});
            runPerActiveLaneScalar(i1Ty(), cInt(i1Ty(), 0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* r = nb.CreateCall(fn, {q});
                return nb.CreateTrunc(r, i1Ty());
            });
            return true;
        }

        // RayQuery_Abort (181) & RayQuery_CommitNonOpaqueTriangleHit (182)
        if (op == 181 || op == 182) {
            llvm::Value* qVec = laneVec(arg(1));
            const char* fnName = (op == 181) ? "nacelle_rayquery_abort" : "nacelle_rayquery_commit_triangle";
            auto fn = getOrDeclareRuntimeFn(fnName, nullptr, {i32Ty()});
            runPerActiveLaneVoid([&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                nb.CreateCall(fn, {q});
            });
            return true;
        }

        // RayQuery_CommitProceduralPrimitiveHit (183)
        if (op == 183) {
            llvm::Value* qVec = laneVec(arg(1));
            llvm::Value* tVec = laneVec(arg(2));
            auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_commit_procedural", nullptr, {i32Ty(), f32Ty()});
            runPerActiveLaneVoid([&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* t = llvmExtract(nb, tVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                nb.CreateCall(fn, {q, t});
            });
            return true;
        }

        // Scalar float state queries: RayTMin (198), CandidateTriangleRayT (199), CommittedRayT (200)
        if (op == 198 || op == 199 || op == 200) {
            llvm::Value* qVec = laneVec(arg(1));
            auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_state_f32", f32Ty(), {i32Ty(), i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(f32Ty(), 0.0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {opConst, q});
            });
            return true;
        }

        // Vector float state queries: Barycentrics (193, 194), WorldRay* (196, 197), ObjectRay* (205, 206, 211, 212)
        if (op == 193 || op == 194 || op == 196 || op == 197 ||
            op == 205 || op == 206 || op == 211 || op == 212) {
            llvm::Value* qVec = laneVec(arg(1));
            llvm::Value* compVec = b_.CreateSExt(laneVec(arg(2)), vecTy(i32Ty()));
            auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_state_vec", f32Ty(), {i32Ty(), i32Ty(), i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(f32Ty(), 0.0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* c = llvmExtract(nb, compVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {opConst, q, c});
            });
            return true;
        }

        // Matrix float state queries: ObjectToWorld3x4 / WorldToObject3x4 (186..189)
        if (op >= 186 && op <= 189) {
            llvm::Value* qVec = laneVec(arg(1));
            llvm::Value* rowVec = laneVec(arg(2));
            llvm::Value* colVec = b_.CreateSExt(laneVec(arg(3)), vecTy(i32Ty()));
            auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_state_mat", f32Ty(), {i32Ty(), i32Ty(), i32Ty(), i32Ty()});
            llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(f32Ty(), 0.0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* r = llvmExtract(nb, rowVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* c = llvmExtract(nb, colVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {opConst, q, r, c});
            });
            return true;
        }

        // Scalar i32 / i1 state queries (184, 185, 190..192, 195, 201..204, 207..210, 214, 215)
        llvm::Value* qVec = laneVec(arg(1));
        auto fn = getOrDeclareRuntimeFn("nacelle_rayquery_state_i32", i32Ty(), {i32Ty(), i32Ty()});
        llvm::Value* opConst = cInt(i32Ty(), static_cast<int64_t>(op));
        const bool isBool = call->getType()->isIntegerTy(1);
        if (isBool) {
            runPerActiveLaneScalar(i1Ty(), cInt(i1Ty(), 0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                llvm::Value* r = nb.CreateCall(fn, {opConst, q});
                return nb.CreateTrunc(r, i1Ty());
            });
        } else {
            runPerActiveLaneScalar(i32Ty(), cInt(i32Ty(), 0), [&](llvm::IRBuilder<>& nb, uint32_t l) {
                llvm::Value* q = llvmExtract(nb, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return nb.CreateCall(fn, {opConst, q});
            });
        }
        return true;
    }

    bool packetizeMathOp(llvm::CallInst* call, uint32_t op, std::string& outError) {
        if ((!call->getType()->isVoidTy()) == 0) {
            outError = "Unsupported dx.op opcode: " + std::to_string(op);
            return false;
        }
        auto fullResTy = call->getType();
        auto vt = llvm::dyn_cast<llvm::FixedVectorType>(fullResTy);
        auto resTy = vt ? vt->getElementType() : fullResTy;
        auto vT = vecTy(resTy);

        if (vt) {
            SimdVal outVecStruct;
            outVecStruct.kind = SimdVal::Kind::Struct;
            outVecStruct.origType = fullResTy;
            unsigned n = static_cast<unsigned>(vt->getNumElements());
            for (unsigned elemIdx = 0; elemIdx < n; ++elemIdx) {
                llvm::Value* elemOut;
                auto argV = [&](unsigned i) {
                    SimdVal sv = getMapped(call->getArgOperand(i));
                    if (sv.kind == SimdVal::Kind::Struct && elemIdx < sv.fields.size()) {
                        return asVec(sv.fields[elemIdx]);
                    }
                    return asVec(sv);
                };
                auto store = [&](llvm::Value* r) {
                    elemOut = r;
                    return true;
                };
                auto flushDenormF = [&](llvm::Value* v) -> llvm::Value* {
                    if (!resTy->isFloatTy()) return v;
                    llvm::Value* bits = b_.CreateBitCast(v, vecTy(i32Ty()));
                    llvm::Value* absBits = b_.CreateAnd(bits, splatInt(i32Ty(), 0x7FFFFFFFu));
                    llvm::Value* isDenorm = b_.CreateICmp(llvm::CmpInst::ICMP_ULT, absBits, splatInt(i32Ty(), 0x00800000u));
                    llvm::Value* signZero = b_.CreateAnd(bits, splatInt(i32Ty(), 0x80000000u));
                    llvm::Value* flushed = b_.CreateSelect(isDenorm, signZero, bits);
                    return b_.CreateBitCast(flushed, vT);
                };
                if (!packetizeScalarMathSwitch(call, op, resTy, vT, argV, store, flushDenormF, outError)) {
                    return false;
                }
                outVecStruct.fields.push_back(SimdVal::makeVector(elemOut, resTy));
            }
            setMapped(call, outVecStruct);
            return true;
        }

        auto argV = [&](unsigned i) { return asVec(getMapped(call->getArgOperand(i))); };
        auto store = [&](llvm::Value* r) {
            setMapped(call, SimdVal::makeVector(r, resTy));
            return true;
        };

        auto flushDenormF = [&](llvm::Value* v) -> llvm::Value* {
            if (!resTy->isFloatTy()) return v;
            llvm::Value* bits = b_.CreateBitCast(v, vecTy(i32Ty()));
            llvm::Value* absBits = b_.CreateAnd(bits, splatInt(i32Ty(), 0x7FFFFFFFu));
            llvm::Value* isDenorm = b_.CreateICmp(llvm::CmpInst::ICMP_ULT, absBits, splatInt(i32Ty(), 0x00800000u));
            llvm::Value* signZero = b_.CreateAnd(bits, splatInt(i32Ty(), 0x80000000u));
            llvm::Value* flushed = b_.CreateSelect(isDenorm, signZero, bits);
            return b_.CreateBitCast(flushed, vT);
        };

        return packetizeScalarMathSwitch(call, op, resTy, vT, argV, store, flushDenormF, outError);
    }

    template <typename ArgVFn, typename StoreFn, typename FlushFn>
    bool packetizeScalarMathSwitch(
        llvm::CallInst* call, uint32_t op, llvm::Type* resTy, llvm::Type* vT,
        ArgVFn&& argV, StoreFn&& store, FlushFn&& flushDenormF, std::string& outError) {
        switch (op) {
        case 6: return store(createFAbs(argV(1)));
        case 7: { // Saturate
            llvm::Value* lo = createMaxNum(argV(1), splatFlt(resTy, 0.0));
            return store(createMinNum(lo, splatFlt(resTy, 1.0)));
        }
        case 8: return store(b_.CreateIntrinsic(llvm::Intrinsic::is_fpclass, {argV(1)->getType()}, {argV(1), b_.getInt32(static_cast<uint32_t>(llvm::fcNan))}));
        case 9: return store(b_.CreateIntrinsic(llvm::Intrinsic::is_fpclass, {argV(1)->getType()}, {argV(1), b_.getInt32(static_cast<uint32_t>(llvm::fcInf))}));
        case 10: return store(b_.CreateIntrinsic(llvm::Intrinsic::is_fpclass, {argV(1)->getType()}, {argV(1), b_.getInt32(static_cast<uint32_t>(llvm::fcNormal | llvm::fcSubnormal | llvm::fcZero))}));
        case 11: return store(b_.CreateIntrinsic(llvm::Intrinsic::is_fpclass, {argV(1)->getType()}, {argV(1), b_.getInt32(static_cast<uint32_t>(llvm::fcNormal))}));
        case 12: return store(emitMathUnary("cos", flushDenormF(argV(1))));
        case 13: return store(emitMathUnary("sin", flushDenormF(argV(1))));
        case 14: return store(emitMathUnary("tan", flushDenormF(argV(1))));
        case 15: return store(emitMathUnary("acos", flushDenormF(argV(1))));
        case 16: return store(emitMathUnary("asin", flushDenormF(argV(1))));
        case 17: return store(emitMathUnary("atan", flushDenormF(argV(1))));
        case 18: return store(emitMathUnary("cosh", flushDenormF(argV(1))));
        case 19: return store(emitMathUnary("sinh", flushDenormF(argV(1))));
        case 20: return store(emitMathUnary("tanh", flushDenormF(argV(1))));
        case 21: return store(emitMathUnary("exp2", flushDenormF(argV(1)))); // DXIL Exp is base-2!
        case 22: { // Frc = x - floor(x)
            llvm::Value* x = argV(1);
            return store(createFSub(x, b_.CreateIntrinsic(llvm::Intrinsic::floor, {vT}, {x})));
        }
        case 23: return store(emitMathUnary("log2", flushDenormF(argV(1)))); // DXIL Log is base-2!
        case 24: return store(createSqrt(argV(1)));
        case 25: { // Rsqrt = 1.0 / sqrt(x)
            llvm::Value* sq = createSqrt(argV(1));
            return store(createFDiv(splatFlt(resTy, 1.0), sq));
        }
        case 26: return store(b_.CreateIntrinsic(llvm::Intrinsic::roundeven, {vT}, {argV(1)}));
        case 27: return store(b_.CreateIntrinsic(llvm::Intrinsic::floor, {vT}, {argV(1)}));
        case 28: return store(b_.CreateIntrinsic(llvm::Intrinsic::ceil, {vT}, {argV(1)}));
        case 29: return store(b_.CreateIntrinsic(llvm::Intrinsic::trunc, {vT}, {argV(1)}));
        case 30: return store(b_.CreateIntrinsic(llvm::Intrinsic::bitreverse, {vT}, {argV(1)}));
        case 31: { // Countbits: population count, result is always i32 in DXIL
            llvm::Value* x = argV(1);
            llvm::Type* argTy = llvm::cast<llvm::FixedVectorType>(x->getType())->getElementType();
            llvm::Value* pop = b_.CreateIntrinsic(llvm::Intrinsic::ctpop, {vecTy(argTy)}, {x});
            if (argTy != resTy) {
                pop = (argTy->getScalarSizeInBits() > resTy->getScalarSizeInBits())
                          ? static_cast<llvm::Value*>(b_.CreateTrunc(pop, vT))
                          : static_cast<llvm::Value*>(b_.CreateZExt(pop, vT));
            }
            return store(pop);
        }
        case 32: { // FirstbitLo: index of lowest set bit, or -1 if 0
            llvm::Value* x = argV(1);
            llvm::Type* argTy = llvm::cast<llvm::FixedVectorType>(x->getType())->getElementType();
            llvm::Value* tz = b_.CreateIntrinsic(llvm::Intrinsic::cttz, {vecTy(argTy)}, {x, b_.getFalse()});
            if (argTy != resTy) {
                tz = (argTy->getScalarSizeInBits() > resTy->getScalarSizeInBits())
                         ? static_cast<llvm::Value*>(b_.CreateTrunc(tz, vT))
                         : static_cast<llvm::Value*>(b_.CreateZExt(tz, vT));
            }
            llvm::Value* isZero = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, x, splatInt(argTy, 0));
            return store(b_.CreateSelect(isZero, splatInt(resTy, ~uint64_t(0)), tz));
        }
        case 33: // FirstbitHi: clz(x), or -1 if 0 (DXIL returns leading-zero count from MSB; DXC emits bitWidth-1 - res)
        case 34: { // FirstbitSHi: clz(x < 0 ? ~x : x), or -1 if 0 or -1
            llvm::Value* x = argV(1);
            llvm::Type* argTy = llvm::cast<llvm::FixedVectorType>(x->getType())->getElementType();
            if (op == 34) {
                llvm::Value* isNeg = b_.CreateICmp(llvm::CmpInst::ICMP_SLT, x, splatInt(argTy, 0));
                llvm::Value* inv = b_.CreateXor(x, splatInt(argTy, ~uint64_t(0)));
                x = b_.CreateSelect(isNeg, inv, x);
            }
            llvm::Value* lz = b_.CreateIntrinsic(llvm::Intrinsic::ctlz, {vecTy(argTy)}, {x, b_.getFalse()});
            if (argTy != resTy) {
                lz = (argTy->getScalarSizeInBits() > resTy->getScalarSizeInBits())
                         ? static_cast<llvm::Value*>(b_.CreateTrunc(lz, vT))
                         : static_cast<llvm::Value*>(b_.CreateZExt(lz, vT));
            }
            llvm::Value* isZero = b_.CreateICmp(llvm::CmpInst::ICMP_EQ, x, splatInt(argTy, 0));
            return store(b_.CreateSelect(isZero, splatInt(resTy, ~uint64_t(0)), lz));
        }
        case 35: return store(createMaxNum(argV(1), argV(2)));
        case 36: return store(createMinNum(argV(1), argV(2)));
        case 37: return store(b_.CreateIntrinsic(llvm::Intrinsic::smax, {vT}, {argV(1), argV(2)}));
        case 38: return store(b_.CreateIntrinsic(llvm::Intrinsic::smin, {vT}, {argV(1), argV(2)}));
        case 39: return store(b_.CreateIntrinsic(llvm::Intrinsic::umax, {vT}, {argV(1), argV(2)}));
        case 40: return store(b_.CreateIntrinsic(llvm::Intrinsic::umin, {vT}, {argV(1), argV(2)}));
        case 44: { // UAddc(a, b) -> { sum, carry }
            auto st = llvm::dyn_cast<llvm::StructType>(resTy);
            if (!st || st->getNumElements() < 2) {
                outError = "UAddc expected { T, i1 } struct return type";
                return false;
            }
            llvm::Type* elemTy = st->elements()[0];
            llvm::Value* a = argV(1);
            llvm::Value* b = argV(2);
            llvm::Value* sum = b_.CreateAdd(a, b);
            llvm::Value* carry = b_.CreateICmp(llvm::CmpInst::ICMP_ULT, sum, a);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            outSt.fields.push_back(SimdVal::makeVector(sum, elemTy));
            outSt.fields.push_back(SimdVal::makeVector(carry, i1Ty()));
            setMapped(call, outSt);
            return true;
        }
        case 46:
        case 47: return store(createFMA(argV(1), argV(2), argV(3)));
        case 48:
        case 49: {
            llvm::Value* prod = b_.CreateMul(argV(1), argV(2));
            return store(b_.CreateAdd(prod, argV(3)));
        }
        case 54: // Dot2
        case 55: // Dot3
        case 56: { // Dot4
            unsigned n = (op == 54) ? 2u : (op == 55 ? 3u : 4u);
            llvm::Value* sum = b_.CreateFMul(argV(1), argV(1 + n));
            for (unsigned c = 1; c < n; ++c) {
                llvm::Value* p = b_.CreateFMul(argV(1 + c), argV(1 + n + c));
                sum = b_.CreateFAdd(sum, p);
            }
            return store(sum);
        }
        case 83: // DerivCoarseX
        case 84: // DerivCoarseY
        case 85: // DerivFineX
        case 86: { // DerivFineY
            if (!hooks_.isPixelShader) return store(splatFlt(resTy, 0.0));
            bool fine = (op == 85 || op == 86);
            auto [dx, dy] = computeQuadFiniteDiff(argV(1), fine);
            return store((op == 83 || op == 85) ? dx : dy);
        }
        case 101: { // MakeDouble(lo, hi) -> f64
            llvm::Value* lo64 = b_.CreateZExt(argV(1), vecTy(i64Ty()));
            llvm::Value* hi64 = b_.CreateZExt(argV(2), vecTy(i64Ty()));
            llvm::Value* hiShift = b_.CreateShl(hi64, splatInt(i64Ty(), 32));
            llvm::Value* bits = b_.CreateOr(lo64, hiShift);
            return store(b_.CreateBitCast(bits, vecTy(b_.getDoubleTy())));
        }
        case 102: { // SplitDouble(val) -> { i32 lo, i32 hi }
            auto st = llvm::dyn_cast<llvm::StructType>(resTy);
            if (!st || st->getNumElements() < 2) {
                outError = "SplitDouble expected { i32, i32 } struct return type";
                return false;
            }
            llvm::Value* bits = b_.CreateBitCast(argV(1), vecTy(i64Ty()));
            llvm::Value* lo = b_.CreateTrunc(bits, vecTy(i32Ty()));
            llvm::Value* hi = b_.CreateTrunc(b_.CreateLShr(bits, splatInt(i64Ty(), 32)), vecTy(i32Ty()));
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            outSt.fields.push_back(SimdVal::makeVector(lo, i32Ty()));
            outSt.fields.push_back(SimdVal::makeVector(hi, i32Ty()));
            setMapped(call, outSt);
            return true;
        }
        case 130: { // LegacyF32ToF16(val) -> i32
            llvm::Value* f16v = b_.CreateFPTrunc(argV(1), vecTy(b_.getHalfTy()));
            llvm::Value* i16v = b_.CreateBitCast(f16v, vecTy(b_.getInt16Ty()));
            return store(b_.CreateZExt(i16v, vecTy(i32Ty())));
        }
        case 131: { // LegacyF16ToF32(val) -> f32
            llvm::Value* i16v = b_.CreateTrunc(argV(1), vecTy(b_.getInt16Ty()));
            llvm::Value* f16v = b_.CreateBitCast(i16v, vecTy(b_.getHalfTy()));
            return store(b_.CreateFPExt(f16v, vecTy(f32Ty())));
        }
        case 162: { // Dot2AddHalf(acc, ax, ay, bx, by) -> f32
            llvm::Value* acc = argV(1);
            llvm::Value* ax = b_.CreateFPExt(argV(2), vecTy(f32Ty()));
            llvm::Value* ay = b_.CreateFPExt(argV(3), vecTy(f32Ty()));
            llvm::Value* bx = b_.CreateFPExt(argV(4), vecTy(f32Ty()));
            llvm::Value* by = b_.CreateFPExt(argV(5), vecTy(f32Ty()));
            llvm::Value* px = b_.CreateFMul(ax, bx);
            llvm::Value* py = b_.CreateFMul(ay, by);
            llvm::Value* sum = b_.CreateFAdd(acc, b_.CreateFAdd(px, py));
            return store(sum);
        }
        case 163: // Dot4AddI8Packed(acc, a, b)
        case 164: { // Dot4AddU8Packed(acc, a, b)
            bool isSigned = (op == 163);
            llvm::Value* acc = argV(1);
            llvm::Value* a = argV(2);
            llvm::Value* b = argV(3);
            llvm::Value* maskFF = splatInt(i32Ty(), 0xFF);
            llvm::Value* c24 = splatInt(i32Ty(), 24);
            for (unsigned i = 0; i < 4; ++i) {
                llvm::Value* ai = nullptr;
                llvm::Value* bi = nullptr;
                if (isSigned) {
                    llvm::Value* shlAmt = splatInt(i32Ty(), 24 - i * 8);
                    ai = b_.CreateAShr(b_.CreateShl(a, shlAmt), c24);
                    bi = b_.CreateAShr(b_.CreateShl(b, shlAmt), c24);
                } else {
                    llvm::Value* shrAmt = splatInt(i32Ty(), i * 8);
                    ai = b_.CreateAnd(b_.CreateLShr(a, shrAmt), maskFF);
                    bi = b_.CreateAnd(b_.CreateLShr(b, shrAmt), maskFF);
                }
                acc = b_.CreateAdd(acc, b_.CreateMul(ai, bi));
            }
            return store(acc);
        }
        case 219: { // Unpack4x8(unpackMode, pk) -> %dx.types.fouri32 / fouri16
            auto st = llvm::dyn_cast<llvm::StructType>(resTy);
            if (!st || st->getNumElements() < 4) {
                outError = "Unpack4x8 expected 4-element struct return type";
                return false;
            }
            uint32_t mode = static_cast<uint32_t>(getConstIntValue(call->getArgOperand(1)).value_or(0));
            llvm::Value* pk = argV(2);
            llvm::Type* elemTy = st->getElementType(0);
            bool needTrunc = elemTy->getScalarSizeInBits() < 32;
            llvm::Value* maskFF = splatInt(i32Ty(), 0xFF);
            llvm::Value* c24 = splatInt(i32Ty(), 24);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            for (unsigned i = 0; i < 4; ++i) {
                llvm::Value* comp;
                if (mode == 1) { // Signed
                    llvm::Value* shlAmt = splatInt(i32Ty(), 24 - i * 8);
                    comp = b_.CreateAShr(b_.CreateShl(pk, shlAmt), c24);
                } else { // Unsigned
                    llvm::Value* shrAmt = splatInt(i32Ty(), i * 8);
                    comp = b_.CreateAnd(b_.CreateLShr(pk, shrAmt), maskFF);
                }
                if (needTrunc) {
                    comp = b_.CreateTrunc(comp, vecTy(elemTy));
                }
                outSt.fields.push_back(SimdVal::makeVector(comp, elemTy));
            }
            setMapped(call, outSt);
            return true;
        }
        case 220: { // Pack4x8(packMode, x, y, z, w) -> i32
            uint32_t mode = static_cast<uint32_t>(getConstIntValue(call->getArgOperand(1)).value_or(0));
            llvm::Value* maskFF = splatInt(i32Ty(), 0xFF);
            llvm::Value* acc = splatInt(i32Ty(), 0);
            for (unsigned i = 0; i < 4; ++i) {
                llvm::Value* c = argV(2 + i);
                llvm::Type* cet = llvm::cast<llvm::FixedVectorType>(c->getType())->getElementType();
                if (cet->getScalarSizeInBits() < 32) {
                    c = b_.CreateSExt(c, vecTy(i32Ty()));
                }
                if (mode == 1) { // U8Clamp [0, 255]
                    c = b_.CreateIntrinsic(llvm::Intrinsic::smax, {splatInt(i32Ty(), 0)->getType()}, {splatInt(i32Ty(), 0), b_.CreateIntrinsic(llvm::Intrinsic::smin, {splatInt(i32Ty(), 255)->getType()}, {splatInt(i32Ty(), 255), c})});
                } else if (mode == 2) { // S8Clamp [-128, 127]
                    c = b_.CreateIntrinsic(llvm::Intrinsic::smax, {splatInt(i32Ty(), -128)->getType()}, {splatInt(i32Ty(), -128), b_.CreateIntrinsic(llvm::Intrinsic::smin, {splatInt(i32Ty(), 127)->getType()}, {splatInt(i32Ty(), 127), c})});
                }
                c = b_.CreateAnd(c, maskFF);
                if (i > 0) {
                    c = b_.CreateShl(c, splatInt(i32Ty(), i * 8));
                }
                acc = b_.CreateOr(acc, c);
            }
            return store(acc);
        }
        default:
            outError = "Unsupported dx.op opcode: " + std::to_string(op);
            return false;
        }
    }

    llvm::IRBuilder<>& b_;
    llvm::Module& mod_;
    const DxilLoadedShader& shader_;
    const RootSignatureLayout& rootLayout_;
    uint32_t W_;
    llvm::Value* rootCtxPtr_;
    StageHooks& hooks_;
    llvm::Value* scratchAlloca_;
    bool strictFp_ = false;
    DescInfoCache ownedDescCache_;
    DescInfoCache* descCache_ = nullptr;

    llvm::BasicBlock* getFuncEntryBlock() const {
        if (auto* bb = b_.GetInsertBlock()) {
            if (auto* fn = bb->getParent()) {
                return &fn->getEntryBlock();
            }
        }
        return nullptr;
    }

public:
    llvm::DenseMap<llvm::Value*, SimdVal> valueMap_;
};

// --- Barrier Wave-Loop Fission Helpers ---

struct SpillSlot {
    llvm::Value* arrPtr; // `!llvm.ptr` pointing to [numWaves x packedStructOrArray]
    llvm::Type* origTy;
    llvm::Type* packedValTy;
    bool isHandle = false;
    SimdVal handleProto;
};

static llvm::Value* buildFlatTidAndMask(
    llvm::IRBuilder<>& b, uint32_t W,
    llvm::Value* waveIdx, llvm::Value* totalThreads, llvm::Value*& outActiveMask) {
    llvm::Type* i32Ty = b.getInt32Ty();
    llvm::FixedVectorType* vI32 = llvm::FixedVectorType::get(i32Ty, W);
    if (auto wConst = getConstIntValue(waveIdx)) {
        if (auto totConst = getConstIntValue(totalThreads)) {
            uint64_t baseVal = (*wConst) * W;
            llvm::SmallVector<uint32_t, 16> tids;
            llvm::SmallVector<llvm::Constant*, 16> maskConsts;
            tids.reserve(W);
            maskConsts.reserve(W);
            for (uint32_t i = 0; i < W; ++i) {
                uint64_t tid = baseVal + i;
                tids.push_back(static_cast<uint32_t>(tid));
                maskConsts.push_back(b.getInt1(tid < *totConst));
            }
            outActiveMask = llvm::ConstantVector::get(maskConsts);
            return llvm::ConstantDataVector::get(b.getContext(), llvm::ArrayRef<uint32_t>(tids));
        }
    }
    llvm::SmallVector<uint32_t, 16> iota;
    for (uint32_t i = 0; i < W; ++i) iota.push_back(i);
    llvm::Value* iotaVec = llvm::ConstantDataVector::get(b.getContext(), llvm::ArrayRef<uint32_t>(iota));
    llvm::Value* w32 = waveIdx;
    llvm::Value* base = b.CreateMul(w32, llvm::ConstantInt::get(i32Ty, W));
    llvm::Value* baseVec = llvmSplat(b, i32Ty, base, W);
    llvm::Value* flatTid = b.CreateAdd(baseVec, iotaVec);
    llvm::Value* totVec = llvmSplat(b, i32Ty, totalThreads, W);
    outActiveMask = b.CreateICmp(llvm::CmpInst::ICMP_ULT, flatTid, totVec);
    return flatTid;
}

static void setupComputeThreadIds(
    llvm::IRBuilder<>& b, uint32_t W, const uint32_t tgs[3],
    llvm::Value* flatTid, llvm::Value* gx, llvm::Value* gy, llvm::Value* gz,
    StageHooks& hooks) {
    llvm::Type* i32Ty = b.getInt32Ty();
    auto splat = [&](uint32_t val) {
        return llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W), llvm::ConstantInt::get(i32Ty, val));
    };
    llvm::Value* zeroVec = splat(0);
    llvm::Value* tgsX = splat(tgs[0]);
    llvm::Value* tgsY = splat(tgs[1]);
    llvm::Value* tgsZ = splat(tgs[2]);
    llvm::Value* xy = splat(tgs[0] * tgs[1]);

    llvm::Value* lx = (tgs[0] == 1) ? zeroVec : static_cast<llvm::Value*>(b.CreateURem(flatTid, tgsX));
    llvm::Value* qx = (tgs[0] == 1) ? flatTid : static_cast<llvm::Value*>(b.CreateUDiv(flatTid, tgsX));
    llvm::Value* ly = (tgs[1] == 1) ? zeroVec : static_cast<llvm::Value*>(b.CreateURem(qx, tgsY));
    llvm::Value* lz = (tgs[2] == 1) ? zeroVec : static_cast<llvm::Value*>(b.CreateUDiv(flatTid, xy));

    llvm::Value* gxVec = llvmSplat(b, i32Ty, gx, W);
    llvm::Value* gyVec = llvmSplat(b, i32Ty, gy, W);
    llvm::Value* gzVec = llvmSplat(b, i32Ty, gz, W);

    hooks.groupId = {gxVec, gyVec, gzVec};
    hooks.threadIdInGroup = {lx, ly, lz};
    hooks.flatGroupId = (tgs[0] == 1 && tgs[1] == 1 && tgs[2] == 1) ? zeroVec : flatTid;
    hooks.threadId = {
        (tgs[0] == 1) ? gxVec
                      : static_cast<llvm::Value*>(b.CreateAdd(b.CreateMul(gxVec, tgsX), lx)),
        (tgs[1] == 1) ? gyVec
                      : static_cast<llvm::Value*>(b.CreateAdd(b.CreateMul(gyVec, tgsY), ly)),
        (tgs[2] == 1) ? gzVec
                      : static_cast<llvm::Value*>(b.CreateAdd(b.CreateMul(gzVec, tgsZ), lz))};
}

// Collects all SSA values used inside `[segBegin, segEnd)` that are already tracked in `spills`,
// and all values defined inside `[segBegin, segEnd)` that are used outside `[segBegin, segEnd)`.
static void analyzeSegmentLiveness(
    llvm::BasicBlock::iterator segBegin, llvm::BasicBlock::iterator segEnd,
    const llvm::DenseMap<llvm::Value*, SpillSlot>& spills,
    llvm::SmallVectorImpl<llvm::Value*>& needReload,
    llvm::SmallVectorImpl<llvm::Value*>& needSpill) {
    llvm::DenseSet<llvm::Instruction*> inSegOps;
    for (auto it = segBegin; it != segEnd; ++it) inSegOps.insert(&*it);

    llvm::DenseSet<llvm::Value*> seenReload;
    for (auto it = segBegin; it != segEnd; ++it) {
        for (llvm::Value* opnd : it->operands()) {
            if (spills.count(opnd) && seenReload.insert(opnd).second) {
                needReload.push_back(opnd);
            }
        }
    }

    auto isUsedOutsideSeg = [&](llvm::Instruction* inst) -> bool {
        for (llvm::User* u : inst->users()) {
            if (auto* ui = llvm::dyn_cast<llvm::Instruction>(u)) {
                if (!inSegOps.count(ui)) return true;
            }
        }
        return false;
    };

    for (auto it = segBegin; it != segEnd; ++it) {
        llvm::Instruction* inst = &*it;
        if (!inst->getType()->isVoidTy() && !spills.count(inst) && isUsedOutsideSeg(inst)) {
            needSpill.push_back(inst);
        }
    }
}

static bool seqContainsBarrier(const SeqNode& seq) {
    for (const auto& child : seq.children) {
        switch (child->kind) {
        case RegionNode::Kind::Block: {
            const auto& bn = static_cast<const BlockNode&>(*child);
            if (bn.bb) {
                for (llvm::Instruction& inst : *bn.bb) {
                    if (isBarrierOp(&inst)) return true;
                }
            }
            break;
        }
        case RegionNode::Kind::Seq:
            if (seqContainsBarrier(static_cast<const SeqNode&>(*child))) return true;
            break;
        case RegionNode::Kind::If: {
            const auto& in = static_cast<const IfNode&>(*child);
            if ((in.thenRegion && seqContainsBarrier(*in.thenRegion)) ||
                (in.elseRegion && seqContainsBarrier(*in.elseRegion))) {
                return true;
            }
            break;
        }
        case RegionNode::Kind::While: {
            const auto& wn = static_cast<const WhileNode&>(*child);
            if (wn.body && seqContainsBarrier(*wn.body)) return true;
            break;
        }
        }
    }
    return false;
}

static bool packetizeComputeSeqWithBarriers(
    llvm::IRBuilder<>& b, llvm::Module& mod,
    const SeqNode& srcSeq, const DxilLoadedShader& shader, const RootSignatureLayout& rootLayout,
    uint32_t W, uint32_t numWavesVal, llvm::Value* rootCtxPtr, llvm::Value* gsBase,
    llvm::Value* gx, llvm::Value* gy, llvm::Value* gz,
    llvm::Value* totalThreads, llvm::Value* numWaves,
    llvm::Value* waveMaskArr, // [numWaves x <W x i1>] current active mask per wave
    llvm::DenseMap<llvm::Value*, SpillSlot>& spills,
    llvm::Instruction* entryAllocaAnchor,
    const StageHooks& baseHooks,
    std::string& outError) {
    llvm::Type* i32Ty = b.getInt32Ty();
    llvm::FixedVectorType* maskVecTy = llvm::FixedVectorType::get(b.getInt1Ty(), W);

    auto allocSpillSlot = [&](llvm::Type* origTy, const SimdVal* proto = nullptr) -> SpillSlot {
        SpillSlot slot;
        slot.origTy = origTy;
        if (proto && proto->kind == SimdVal::Kind::Handle) {
            slot.isHandle = true;
            slot.handleProto = *proto;
        }
        llvm::IRBuilder<> eb(entryAllocaAnchor);
        StageHooks unusedHooks;
        SimdPacketizer helper(eb, mod, shader, rootLayout, W, rootCtxPtr, unusedHooks);
        llvm::SmallVector<llvm::Type*, 4> flatTys;
        helper.flattenSimdTypes(origTy, flatTys);
        slot.packedValTy = (flatTys.size() == 1) ? flatTys[0] : llvm::StructType::get(b.getContext(), flatTys);
        llvm::Value* n32 = llvm::ConstantInt::get(i32Ty, numWavesVal);
        uint32_t vecAlign = std::max<uint32_t>(64, W * 8);
        slot.arrPtr = createAlignedAlloca(eb, slot.packedValTy, n32, vecAlign);
        return slot;
    };

    auto storeSpillAtWave = [&](llvm::IRBuilder<>& wb, SimdPacketizer& pkt, const SpillSlot& slot,
                                llvm::Value* waveIdx, const SimdVal& sv) {
        llvm::SmallVector<llvm::Value*, 4> packed;
        pkt.packSimdToValues(sv, packed);
        llvm::Value* valToStore;
        if (packed.size() == 1) {
            valToStore = packed[0];
        } else {
            valToStore = llvm::UndefValue::get(slot.packedValTy);
            for (size_t i = 0; i < packed.size(); ++i) {
                valToStore = wb.CreateInsertValue(valToStore, packed[i], {static_cast<unsigned>(i)});
            }
        }
        llvm::Value* wp = wb.CreateGEP(slot.packedValTy, slot.arrPtr, {waveIdx});
        wb.CreateAlignedStore(valToStore, wp, llvm::Align(1u));
    };

    auto loadSpillAtWave = [&](llvm::IRBuilder<>& wb, SimdPacketizer& pkt, const SpillSlot& slot,
                               llvm::Value* waveIdx) -> SimdVal {
        llvm::Value* wp = wb.CreateGEP(slot.packedValTy, slot.arrPtr, {waveIdx});
        llvm::Value* loaded = wb.CreateAlignedLoad(slot.packedValTy, wp, llvm::Align(1u));
        llvm::SmallVector<llvm::Value*, 4> flatVals;
        if (auto* st = llvm::dyn_cast<llvm::StructType>(slot.packedValTy);
            st && !slot.isHandle && !llvm::isa<llvm::StructType>(slot.origTy) && !llvm::isa<llvm::ArrayType>(slot.origTy)) {
            flatVals.push_back(loaded);
        } else if (auto* st2 = llvm::dyn_cast<llvm::StructType>(slot.packedValTy)) {
            for (unsigned i = 0; i < st2->getNumElements(); ++i) {
                flatVals.push_back(wb.CreateExtractValue(loaded, {i}));
            }
        } else {
            flatVals.push_back(loaded);
        }
        size_t cur = 0;
        return pkt.unpackSimdFromValues(slot.origTy, flatVals, cur, slot.isHandle ? &slot.handleProto : nullptr);
    };

    llvm::Value* c0 = llvm::ConstantInt::get(i32Ty, 0);
    llvm::Value* c1 = llvm::ConstantInt::get(i32Ty, 1);

    auto emitWaveSegment = [&](llvm::BasicBlock::iterator segBegin, llvm::BasicBlock::iterator segEnd) -> bool {
        if (segBegin == segEnd) return true;
        llvm::SmallVector<llvm::Value*, 8> needReload, needSpill;
        analyzeSegmentLiveness(segBegin, segEnd, spills, needReload, needSpill);

        for (llvm::Value* v : needSpill) {
            spills[v] = allocSpillSlot(v->getType());
        }

        emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
            llvm::Value* baseMask = nullptr;
            llvm::Value* flatTid = buildFlatTidAndMask(wb, W, waveIdx, totalThreads, baseMask);
            llvm::Value* mp = wb.CreateGEP(maskVecTy, waveMaskArr, {waveIdx});
            llvm::Value* activeMask = wb.CreateAlignedLoad(maskVecTy, mp, llvm::Align(1u));

            StageHooks hooks = baseHooks;
            hooks.entryBlock = entryAllocaAnchor ? entryAllocaAnchor->getParent() : baseHooks.entryBlock;
            hooks.groupSharedBase = gsBase;
            hooks.waveIdx = waveIdx;
            hooks.numWaves = numWavesVal;
            setupComputeThreadIds(wb, W, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
            SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, hooks);
            for (llvm::Value* rv : needReload) {
                pkt.valueMap_[rv] = loadSpillAtWave(wb, pkt, spills[rv], waveIdx);
            }
            llvm::Value* realMask = activeMask;
            if (!pkt.packetizeBlockRange(segBegin, segEnd, activeMask, realMask, outError)) {
                return;
            }
            for (llvm::Value* sv : needSpill) {
                const auto& mapped = pkt.valueMap_[sv];
                if (mapped.kind == SimdVal::Kind::Handle) {
                    spills[sv].isHandle = true;
                    spills[sv].handleProto = mapped;
                }
                storeSpillAtWave(wb, pkt, spills[sv], waveIdx, mapped);
            }
        });
        return outError.empty();
    };

    for (const auto& child : srcSeq.children) {
        switch (child->kind) {
        case RegionNode::Kind::Block: {
            const auto& bn = static_cast<const BlockNode&>(*child);
            if (!bn.bb) break;
            auto it = bn.bb->begin();
            auto segBegin = it;
            while (it != bn.bb->end() && !it->isTerminator()) {
                llvm::Instruction* op = &*it;
                if (isBarrierOp(op)) {
                    if (!emitWaveSegment(segBegin, it)) return false;
                    b.CreateFence(llvm::AtomicOrdering::AcquireRelease);
                    ++it;
                    segBegin = it;
                    continue;
                }
                ++it;
            }
            if (!emitWaveSegment(segBegin, it)) return false;
            break;
        }
        case RegionNode::Kind::Seq: {
            const auto& sn = static_cast<const SeqNode&>(*child);
            if (!packetizeComputeSeqWithBarriers(
                    b, mod, sn, shader, rootLayout, W, numWavesVal, rootCtxPtr, gsBase,
                    gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
                    entryAllocaAnchor, baseHooks, outError)) {
                return false;
            }
            break;
        }
        case RegionNode::Kind::If: {
            const auto& in = static_cast<const IfNode&>(*child);
            // Evaluate uniform condition at wave 0 lane 0
            StageHooks dummyHooks;
            SimdPacketizer condPkt(b, mod, shader, rootLayout, W, rootCtxPtr, dummyHooks);
            SimdVal cVal = spills.count(in.cond)
                               ? loadSpillAtWave(b, condPkt, spills[in.cond], c0)
                               : condPkt.getMapped(in.cond);
            llvm::Value* condScalar = llvmExtract(b, condPkt.asVec(cVal), 0);

            emitIfElse(b, condScalar,
                [&](llvm::IRBuilder<>& ib) {
                    if (in.thenRegion) {
                        (void)packetizeComputeSeqWithBarriers(
                            ib, mod, *in.thenRegion, shader, rootLayout, W, numWavesVal,
                            rootCtxPtr, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
                            entryAllocaAnchor, baseHooks, outError);
                    }
                },
                [&](llvm::IRBuilder<>& ib) {
                    if (in.elseRegion && !in.elseRegion->children.empty()) {
                        (void)packetizeComputeSeqWithBarriers(
                            ib, mod, *in.elseRegion, shader, rootLayout, W, numWavesVal,
                            rootCtxPtr, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
                            entryAllocaAnchor, baseHooks, outError);
                    }
                });
            if (!outError.empty()) return false;
            break;
        }
        case RegionNode::Kind::While: {
            const auto& wn = static_cast<const WhileNode&>(*child);
            // Allocate spill slots for header PHIs and initialize from preheader/outside incoming values
            llvm::SmallVector<std::pair<llvm::PHINode*, SpillSlot>, 4> hdrPhis;
            if (wn.headerBB) {
                for (llvm::PHINode& phi : wn.headerBB->phis()) {
                    llvm::Value* initV = nullptr;
                    if (wn.preheaderBB) {
                        initV = phi.getIncomingValueForBlock(wn.preheaderBB);
                    } else {
                        for (unsigned i = 0; i < phi.getNumIncomingValues(); ++i) {
                            if (!wn.loop || !wn.loop->contains(phi.getIncomingBlock(i))) {
                                initV = phi.getIncomingValue(i);
                                break;
                            }
                        }
                    }
                    SpillSlot slot = allocSpillSlot(phi.getType());
                    spills[&phi] = slot;
                    hdrPhis.push_back({&phi, slot});
                    if (initV) {
                        emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
                            StageHooks dh;
                            SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, dh);
                            SimdVal sv = spills.count(initV) ? loadSpillAtWave(wb, pkt, spills[initV], waveIdx)
                                                             : pkt.getMapped(initV);
                            if (!sv.valid()) sv = pkt.makeZeroSimd(phi.getType());
                            storeSpillAtWave(wb, pkt, slot, waveIdx, sv);
                        });
                    }
                }
            }

            // Save outer wave masks on stack
            llvm::Value* savedWaveMaskArr = nullptr;
            {
                llvm::IRBuilder<> eb(entryAllocaAnchor);
                savedWaveMaskArr = createAlignedAlloca(eb, maskVecTy, llvm::ConstantInt::get(i32Ty, numWavesVal), std::max<uint32_t>(64, W * 8));
            }
            emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
                llvm::Value* srcP = wb.CreateGEP(maskVecTy, waveMaskArr, {waveIdx});
                llvm::Value* dstP = wb.CreateGEP(maskVecTy, savedWaveMaskArr, {waveIdx});
                wb.CreateAlignedStore(wb.CreateAlignedLoad(maskVecTy, srcP, llvm::Align(1u)), dstP, llvm::Align(1u));
            });

            llvm::Function* fn = b.GetInsertBlock()->getParent();
            llvm::BasicBlock* bodyBB = llvm::BasicBlock::Create(b.getContext(), "barrier.while.body", fn);
            llvm::BasicBlock* exitBB = llvm::BasicBlock::Create(b.getContext(), "barrier.while.exit", fn);

            b.CreateBr(bodyBB);
            b.SetInsertPoint(bodyBB);

            if (wn.body && !packetizeComputeSeqWithBarriers(
                    b, mod, *wn.body, shader, rootLayout, W, numWavesVal, rootCtxPtr, gsBase,
                    gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills, entryAllocaAnchor, baseHooks, outError)) {
                return false;
            }

            llvm::AllocaInst* anySlot = nullptr;
            {
                llvm::IRBuilder<> eb(entryAllocaAnchor);
                anySlot = eb.CreateAlloca(b.getInt1Ty(), nullptr, "barrier.while.any");
            }
            b.CreateStore(b.getFalse(), anySlot);

            llvm::CondBrInst* latchCondBr = wn.latchBB ? llvm::dyn_cast<llvm::CondBrInst>(wn.latchBB->getTerminator()) : nullptr;
            emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
                llvm::Value* baseMask = nullptr;
                llvm::Value* flatTid = buildFlatTidAndMask(wb, W, waveIdx, totalThreads, baseMask);
                llvm::Value* mp = wb.CreateGEP(maskVecTy, waveMaskArr, {waveIdx});
                llvm::Value* curMask = wb.CreateAlignedLoad(maskVecTy, mp, llvm::Align(1u));

                StageHooks hooks = baseHooks;
                hooks.entryBlock = entryAllocaAnchor ? entryAllocaAnchor->getParent() : nullptr;
                hooks.groupSharedBase = gsBase;
                hooks.waveIdx = waveIdx;
                hooks.numWaves = numWavesVal;
                setupComputeThreadIds(wb, W, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
                SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, hooks);

                llvm::Value* nextMask = curMask;
                if (latchCondBr) {
                    llvm::Value* condOpnd = latchCondBr->getCondition();
                    SimdVal condSimd = spills.count(condOpnd)
                                           ? loadSpillAtWave(wb, pkt, spills[condOpnd], waveIdx)
                                           : pkt.getMapped(condOpnd);
                    llvm::Value* condVec = pkt.asVec(condSimd);
                    if (latchCondBr->getSuccessor(0) != wn.headerBB) {
                        condVec = wb.CreateNot(condVec);
                    }
                    nextMask = wb.CreateAnd(curMask, condVec);
                    wb.CreateAlignedStore(nextMask, mp, llvm::Align(1u));
                }

                for (auto& kv : hdrPhis) {
                    if (wn.latchBB) {
                        int idx = kv.first->getBasicBlockIndex(wn.latchBB);
                        if (idx >= 0) {
                            llvm::Value* latchV = kv.first->getIncomingValue(static_cast<unsigned>(idx));
                            SimdVal nextV = spills.count(latchV) ? loadSpillAtWave(wb, pkt, spills[latchV], waveIdx)
                                                                 : pkt.getMapped(latchV);
                            SimdVal prevV = loadSpillAtWave(wb, pkt, kv.second, waveIdx);
                            storeSpillAtWave(wb, pkt, kv.second, waveIdx, pkt.selectVal(nextMask, nextV, prevV));
                        }
                    }
                }

                llvm::Value* waveAny = pkt.orReduce(nextMask);
                llvm::Value* prevAny = wb.CreateLoad(b.getInt1Ty(), anySlot);
                wb.CreateStore(wb.CreateOr(prevAny, waveAny), anySlot);
            });

            llvm::Value* anyActive = b.CreateLoad(b.getInt1Ty(), anySlot);
            b.CreateCondBr(anyActive, bodyBB, exitBB);
            b.SetInsertPoint(exitBB);

            // Restore outer wave masks
            emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
                llvm::Value* srcP = wb.CreateGEP(maskVecTy, savedWaveMaskArr, {waveIdx});
                llvm::Value* dstP = wb.CreateGEP(maskVecTy, waveMaskArr, {waveIdx});
                wb.CreateAlignedStore(wb.CreateAlignedLoad(maskVecTy, srcP, llvm::Align(1u)), dstP, llvm::Align(1u));
            });
            break;
        }
        }
    }
    return true;
}

} // namespace

bool packetizeThreadGroupRegion(
    llvm::IRBuilder<>& b,
    llvm::Module& mod,
    const SeqNode& srcRegion,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    llvm::Value* rootCtx,
    llvm::Value* gsBase,
    llvm::Value* gx,
    llvm::Value* gy,
    llvm::Value* gz,
    llvm::BasicBlock* entry,
    const StageHooks& baseHooks,
    std::string& outError) {
    if (srcRegion.children.empty()) return true;
    llvm::Type* i32Ty = b.getInt32Ty();

    const uint32_t totalThreadsVal =
        std::max(1u, shader.threadGroupSize[0] * shader.threadGroupSize[1] * shader.threadGroupSize[2]);
    const uint32_t numWavesVal = (totalThreadsVal + simdWidth - 1) / simdWidth;
    llvm::Value* totalThreads = llvm::ConstantInt::get(i32Ty, totalThreadsVal);
    llvm::Value* c0 = llvm::ConstantInt::get(i32Ty, 0);
    llvm::Value* c1 = llvm::ConstantInt::get(i32Ty, 1);
    llvm::Value* numWaves = llvm::ConstantInt::get(i32Ty, numWavesVal);

    if (!seqContainsBarrier(srcRegion)) {
        auto emitWaveBody = [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
            llvm::Value* activeMask = nullptr;
            llvm::Value* flatTid = buildFlatTidAndMask(wb, simdWidth, waveIdx, totalThreads, activeMask);
            StageHooks hooks = baseHooks;
            hooks.entryBlock = entry;
            hooks.groupSharedBase = gsBase;
            hooks.waveIdx = nullptr;
            hooks.numWaves = 1;
            setupComputeThreadIds(wb, simdWidth, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
            SimdPacketizer pkt(wb, mod, shader, rootLayout, simdWidth, rootCtx, hooks);
            llvm::Value* realMask = activeMask;
            (void)pkt.packetizeSeq(srcRegion, activeMask, realMask, outError);
        };
        if (numWavesVal == 1) {
            emitWaveBody(b, c0);
        } else {
            emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
                emitWaveBody(wb, waveIdx);
            });
        }
    } else {
        llvm::FixedVectorType* maskVecTy = llvm::FixedVectorType::get(b.getInt1Ty(), simdWidth);
        llvm::AllocaInst* waveMaskArr = createAlignedAlloca(b, maskVecTy, llvm::ConstantInt::get(i32Ty, numWavesVal), std::max<uint32_t>(64, simdWidth * 8));
        // Initialize per-wave active masks
        emitForLoop(b, c0, numWaves, c1, [&](llvm::IRBuilder<>& wb, llvm::Value* waveIdx) {
            llvm::Value* activeMask = nullptr;
            (void)buildFlatTidAndMask(wb, simdWidth, waveIdx, totalThreads, activeMask);
            llvm::Value* mp = wb.CreateGEP(maskVecTy, waveMaskArr, {waveIdx});
            wb.CreateAlignedStore(activeMask, mp, llvm::Align(1u));
        });

        llvm::DenseMap<llvm::Value*, SpillSlot> spills;
        (void)packetizeComputeSeqWithBarriers(
            b, mod, srcRegion, shader, rootLayout, simdWidth, numWavesVal,
            rootCtx, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
            waveMaskArr, baseHooks, outError);
    }
    return outError.empty();
}

bool packetizeComputeEntry(
    llvm::Module& mod,
    const StructuredFunction& srcTree,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError) {
    if (!srcTree.fn || srcTree.fn->isDeclaration() || !srcTree.root) {
        outError = "Source compute function not found: " + shader.entryName;
        return false;
    }

    llvm::LLVMContext& ctx = mod.getContext();
    llvm::Type* ptrTy = llvm::PointerType::getUnqual(ctx);
    llvm::Type* i32Ty = llvm::Type::getInt32Ty(ctx);

    // void fn(ptr rootCtx, ptr gsBase, i32 gx, i32 gy, i32 gz, i32 ngx, i32 ngy, i32 ngz)
    auto* fnTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(ctx),
        {ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty}, false);
    llvm::Function* dstFunc = llvm::Function::Create(
        fnTy, llvm::GlobalValue::ExternalLinkage, symbolName, &mod);
    dstFunc->addParamAttr(0, llvm::Attribute::NoAlias);
    dstFunc->addParamAttr(0, llvm::Attribute::ReadOnly);

    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", dstFunc);
    llvm::IRBuilder<> b(entry);

    llvm::Value* rootCtx = dstFunc->getArg(0);
    llvm::Value* gsBase = dstFunc->getArg(1);
    llvm::Value* gx = dstFunc->getArg(2);
    llvm::Value* gy = dstFunc->getArg(3);
    llvm::Value* gz = dstFunc->getArg(4);

    StageHooks baseHooks;
    if (!packetizeThreadGroupRegion(b, mod, *srcTree.root, shader, rootLayout,
                                    simdWidth, rootCtx, gsBase, gx, gy, gz, entry, baseHooks, outError)) {
        return false;
    }
    b.CreateRetVoid();

    srcTree.fn->eraseFromParent();
    if (llvm::Function* helper = mod.getFunction("nacelle.groupshared.ptr")) {
        if (helper->use_empty()) helper->eraseFromParent();
    }
    llvm::SmallVector<llvm::Function*, 16> deadDxOps;
    for (llvm::Function& fnOp : mod) {
        if (fnOp.isDeclaration() && fnOp.getName().starts_with("dx.op.") && fnOp.use_empty()) {
            deadDxOps.push_back(&fnOp);
        }
    }
    for (llvm::Function* fnOp : deadDxOps) fnOp->eraseFromParent();

    std::string verifyErr;
    llvm::raw_string_ostream verifyOs(verifyErr);
    if (llvm::verifyModule(mod, &verifyOs)) {
        outError = "LLVM verification failed after compute SIMT-to-SIMD packetization: " + verifyErr;
        return false;
    }
    return true;
}

bool packetizeRegionWithHooks(
    llvm::IRBuilder<>& builder,
    llvm::Module& mod,
    const SeqNode& srcRegion,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    llvm::Value* rootCtxPtr,
    llvm::Value* initExecMask,
    llvm::Value*& inoutRealMask,
    StageHooks& hooks,
    std::string& outError) {
    if (srcRegion.children.empty()) return true;
    SimdPacketizer pkt(builder, mod, shader, rootLayout, simdWidth, rootCtxPtr, hooks);
    return pkt.packetizeSeq(srcRegion, initExecMask, inoutRealMask, outError);
}

bool packetizeDxrEntry(
    llvm::Module& mod,
    const StructuredFunction& srcTree,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError) {
    if (!srcTree.fn || srcTree.fn->isDeclaration() || !srcTree.root) {
        outError = "Source DXR function not found: " + shader.entryName;
        return false;
    }

    llvm::LLVMContext& ctx = mod.getContext();
    llvm::Type* ptrTy = llvm::PointerType::getUnqual(ctx);
    llvm::Type* i32Ty = llvm::Type::getInt32Ty(ctx);
    llvm::Type* i1Ty = llvm::Type::getInt1Ty(ctx);
    llvm::FixedVectorType* vecI32Ty = llvm::FixedVectorType::get(i32Ty, simdWidth);
    llvm::FixedVectorType* vecI1Ty = llvm::FixedVectorType::get(i1Ty, simdWidth);

    // void fn(ptr rootCtx, i32 activeMask, ptr rxPtr, ptr ryPtr, ptr rzPtr,
    //         i32 dimX, i32 dimY, i32 dimZ, ptr payloadPtr, ptr attrPtr)
    auto* fnTy = llvm::FunctionType::get(
        llvm::Type::getVoidTy(ctx),
        {ptrTy, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, ptrTy, ptrTy}, false);
    llvm::Function* dstFunc = llvm::Function::Create(
        fnTy, llvm::GlobalValue::ExternalLinkage, symbolName, &mod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx, "entry", dstFunc);
    llvm::IRBuilder<> b(entry);

    llvm::Value* rootCtx = dstFunc->getArg(0);
    llvm::Value* activeMaskArg = dstFunc->getArg(1);
    llvm::Value* rxPtr = dstFunc->getArg(2);
    llvm::Value* ryPtr = dstFunc->getArg(3);
    llvm::Value* rzPtr = dstFunc->getArg(4);
    llvm::Value* dimX = dstFunc->getArg(5);
    llvm::Value* dimY = dstFunc->getArg(6);
    llvm::Value* dimZ = dstFunc->getArg(7);
    llvm::Value* payloadPtr = dstFunc->getArg(8);
    llvm::Value* attrPtr = dstFunc->getArg(9);

    llvm::Value* c0 = llvm::ConstantInt::get(i32Ty, 0);
    llvm::Value* c1 = llvm::ConstantInt::get(i32Ty, 1);
    llvm::Value* initExecMask = llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(simdWidth), b.getFalse());
    for (uint32_t l = 0; l < simdWidth; ++l) {
        llvm::Value* shift = llvm::ConstantInt::get(i32Ty, l);
        llvm::Value* shifted = b.CreateLShr(activeMaskArg, shift);
        llvm::Value* bit = b.CreateAnd(shifted, c1);
        llvm::Value* cond = b.CreateICmp(llvm::CmpInst::ICMP_NE, bit, c0);
        initExecMask = llvmInsert(b, initExecMask, cond, l);
    }

    StageHooks hooks;
    hooks.entryBlock = entry;
    hooks.threadId[0] = b.CreateAlignedLoad(vecI32Ty, rxPtr, llvm::Align(4u));
    hooks.threadId[1] = b.CreateAlignedLoad(vecI32Ty, ryPtr, llvm::Align(4u));
    hooks.threadId[2] = b.CreateAlignedLoad(vecI32Ty, rzPtr, llvm::Align(4u));
    hooks.dispatchRaysDim[0] = llvmSplat(b, i32Ty, dimX, simdWidth);
    hooks.dispatchRaysDim[1] = llvmSplat(b, i32Ty, dimY, simdWidth);
    hooks.dispatchRaysDim[2] = llvmSplat(b, i32Ty, dimZ, simdWidth);

    SimdPacketizer pkt(b, mod, shader, rootLayout, simdWidth, rootCtx, hooks);
    for (unsigned a = 0; a < srcTree.fn->arg_size() && a < 2; ++a) {
        llvm::Value* ptrArg = (a == 0) ? payloadPtr : attrPtr;
        llvm::SmallVector<llvm::Value*, 8> lanes(simdWidth, ptrArg);
        pkt.setMapped(srcTree.fn->getArg(a), {SimdVal::Kind::Lanes, ptrTy, nullptr, std::move(lanes), {}});
    }

    llvm::Value* realMask = initExecMask;
    if (!pkt.packetizeSeq(*srcTree.root, initExecMask, realMask, outError)) {
        return false;
    }

    b.CreateRetVoid();
    srcTree.fn->eraseFromParent();

    llvm::SmallVector<llvm::Function*, 16> deadDxOps;
    for (llvm::Function& fnOp : mod) {
        if (fnOp.isDeclaration() && fnOp.getName().starts_with("dx.op.") && fnOp.use_empty()) {
            deadDxOps.push_back(&fnOp);
        }
    }
    for (llvm::Function* fnOp : deadDxOps) fnOp->eraseFromParent();

    std::string verifyErr;
    llvm::raw_string_ostream verifyOs(verifyErr);
    if (llvm::verifyModule(mod, &verifyOs)) {
        outError = "LLVM verification failed after DXR SIMT-to-SIMD packetization: " + verifyErr;
        return false;
    }
    return true;
}

} // namespace nacelle
