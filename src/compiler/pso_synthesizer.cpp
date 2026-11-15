#include "pso_synthesizer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <unordered_map>

#include <llvm/IR/Constants.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Linker/Linker.h>

#include "dxil_loader.h"
#include "jit_engine.h"
#include "simt_to_simd_pass.h"
#include "structured_cfg.h"
#include "../common/format_utils.h"

namespace nacelle {

namespace {

static llvm::Value* createAlignedAlloca(
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

std::string toLowerAscii(const std::string& s) {
    std::string out = s;
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::optional<uint32_t> tryGetUniformConstUInt(const SimdVal& sv) {
    if (sv.kind != SimdVal::Kind::Vector || !sv.vec) return std::nullopt;
    if (auto* cdv = llvm::dyn_cast<llvm::ConstantDataVector>(sv.vec)) {
        if (cdv->isSplat()) {
            return static_cast<uint32_t>(cdv->getElementAsInteger(0));
        }
    }
    if (auto* cv = llvm::dyn_cast<llvm::ConstantVector>(sv.vec)) {
        if (auto* splat = llvm::dyn_cast_or_null<llvm::ConstantInt>(cv->getSplatValue())) {
            return static_cast<uint32_t>(splat->getZExtValue());
        }
    }
    if (llvm::isa<llvm::ConstantAggregateZero>(sv.vec)) {
        return 0u;
    }
    if (auto* shuf = llvm::dyn_cast<llvm::ShuffleVectorInst>(sv.vec)) {
        if (auto* ins = llvm::dyn_cast<llvm::InsertElementInst>(shuf->getOperand(0))) {
            if (auto* ci = llvm::dyn_cast<llvm::ConstantInt>(ins->getOperand(1))) {
                return static_cast<uint32_t>(ci->getZExtValue());
            }
        }
    }
    return std::nullopt;
}

std::vector<ShaderSigElement> convertSigElements(const std::vector<DxilSigRecord>& recs) {
    std::vector<ShaderSigElement> out;
    out.reserve(recs.size());
    for (const auto& r : recs) {
        ShaderSigElement e;
        e.semanticName = r.semanticName;
        e.semanticIndex = r.semanticIndex;
        e.systemValueType = r.systemValueType;
        e.componentType = (r.componentType == 9) ? 3 : ((r.componentType == 4) ? 2 : 1);
        e.registerIndex = r.startRow;
        uint8_t m = 0;
        for (uint32_t c = 0; c < r.cols && (r.startCol + c) < 4; ++c) {
            m |= static_cast<uint8_t>(1u << (r.startCol + c));
        }
        e.mask = m;
        e.rwMask = m;
        out.push_back(std::move(e));
    }
    return out;
}

uint32_t formatByteSizeForVertex(DXGI_FORMAT fmt) {
    switch (fmt) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
        return 16;
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
        return 12;
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
        return 8;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R32_FLOAT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT:
        return 4;
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16_SINT:
        return 2;
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8_SINT:
        return 1;
    default:
        return 16;
    }
}

bool isUnormRenderTargetFormat(DXGI_FORMAT fmt) {
    switch (fmt) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM_SRGB:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM_SRGB:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_A8_UNORM:
        return true;
    default:
        return false;
    }
}

static llvm::Value* llvmExtract(llvm::IRBuilder<>& builder, llvm::Value* vec, uint32_t idx) {
    llvm::Type* i32Ty = builder.getInt32Ty();
    llvm::Value* idxVal = llvm::ConstantInt::get(i32Ty, idx);
    return builder.CreateExtractElement(vec, idxVal);
}

static llvm::Value* llvmInsert(llvm::IRBuilder<>& builder, llvm::Value* vec, llvm::Value* elem, uint32_t idx) {
    llvm::Type* i32Ty = builder.getInt32Ty();
    llvm::Value* idxVal = llvm::ConstantInt::get(i32Ty, idx);
    return builder.CreateInsertElement(vec, elem, idxVal);
}

static llvm::Value* llvmSplat(llvm::IRBuilder<>& builder, llvm::Type* elemTy, llvm::Value* scalar, uint32_t W) {
    auto vt = llvm::FixedVectorType::get(elemTy, W);
    llvm::Value* p = llvm::PoisonValue::get(vt);
    llvm::Value* ins = llvmInsert(builder, p, scalar, 0);
    llvm::SmallVector<int32_t, 16> zeros(W, 0);
    return builder.CreateShuffleVector(ins, p, zeros);
}

struct IRHelpers {
    llvm::IRBuilder<>& b;
    ;
    uint32_t W;

    llvm::Type* i1Ty() const { return b.getInt1Ty(); }
    llvm::Type* i8Ty() const { return b.getInt8Ty(); }
    llvm::Type* i16Ty() const { return b.getInt16Ty(); }
    llvm::Type* i32Ty() const { return b.getInt32Ty(); }
    llvm::Type* i64Ty() const { return b.getInt64Ty(); }
    llvm::Type* f16Ty() const { return b.getHalfTy(); }
    llvm::Type* f32Ty() const { return b.getFloatTy(); }
    llvm::Type* ptrTy() const { return llvm::PointerType::getUnqual(b.getContext()); }
    llvm::FixedVectorType* vecTy(llvm::Type* elem) const { return llvm::FixedVectorType::get(elem, W); }

    llvm::Value* cInt(llvm::Type* ty, int64_t v) const {
        return llvm::ConstantInt::get(ty, v);
    }
    llvm::Value* cFlt(double v) const {
        return llvm::ConstantFP::get(f32Ty(), static_cast<float>(v));
    }
    llvm::Value* splatInt(llvm::Type* ty, int64_t v) const {
        auto vt = vecTy(ty);
        return llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W), llvm::ConstantInt::get(ty, v));
    }
    llvm::Value* splatFlt(double v) const {
        auto vt = vecTy(f32Ty());
        return llvm::ConstantVector::getSplat(llvm::ElementCount::getFixed(W), llvm::ConstantFP::get(f32Ty(), static_cast<float>(v)));
    }
    SimdVal splatScalar(llvm::Value* scalar) const {
        SimdVal sv;
        sv.kind = SimdVal::Kind::Vector;
        sv.origType = scalar->getType();
        sv.vec = llvmSplat(b, scalar->getType(), scalar, W);
        return sv;
    }
    llvm::Value* allTrueMask() const { return splatInt(i1Ty(), 1); }

    llvm::Value* laneIota() const {
        llvm::Value* v = llvm::PoisonValue::get(vecTy(i32Ty()));
        for (uint32_t l = 0; l < W; ++l) {
            v = llvmInsert(b, v, cInt(i32Ty(), l), l);
        }
        return v;
    }

    llvm::Value* gepBytes(llvm::Value* ptr, int64_t byteOff) const {
        if (byteOff == 0) return ptr;
        return b.CreateGEP(i8Ty(), ptr, b.getInt32(static_cast<int32_t>(byteOff)));
    }
    llvm::Value* gepBytesDyn(llvm::Value* ptr, llvm::Value* byteOffI64) const {
        return b.CreateGEP(i8Ty(), ptr, llvm::ArrayRef<llvm::Value*>{byteOffI64});
    }
    llvm::Value* gepArray(llvm::Value* basePtr, llvm::Type* elemTy, llvm::Value* idx) const {
        return b.CreateGEP(elemTy, basePtr, llvm::ArrayRef<llvm::Value*>{idx});
    }
    llvm::Value* loadScalar(llvm::Type* ty, llvm::Value* ptr) const {
        unsigned align = llvm::isa<llvm::FixedVectorType>(ty) ? 4u : 1u;
        return b.CreateAlignedLoad(ty, ptr, llvm::Align(align));
    }
    void storeScalar(llvm::Value* val, llvm::Value* ptr) const {
        unsigned align = llvm::isa<llvm::FixedVectorType>(val->getType()) ? 4u : 1u;
        b.CreateAlignedStore(val, ptr, llvm::Align(align));
    }

    llvm::Value* orReduce(llvm::Value* maskVec) const {
        return b.CreateOrReduce(maskVec);
    }

    llvm::Value* expandMaskQuads(llvm::Value* mask) const {
        llvm::SmallVector<int32_t, 16> s0, s1, s2, s3;
        for (uint32_t q = 0; q < W; q += 4) {
            for (uint32_t k = 0; k < 4; ++k) {
                s0.push_back(q);
                s1.push_back(q + 1);
                s2.push_back(q + 2);
                s3.push_back(q + 3);
            }
        }
        llvm::Value* m0 = b.CreateShuffleVector(mask, mask, s0);
        llvm::Value* m1 = b.CreateShuffleVector(mask, mask, s1);
        llvm::Value* m2 = b.CreateShuffleVector(mask, mask, s2);
        llvm::Value* m3 = b.CreateShuffleVector(mask, mask, s3);
        llvm::Value* o01 = b.CreateOr(m0, m1);
        llvm::Value* o23 = b.CreateOr(m2, m3);
        return b.CreateOr(o01, o23);
    }

    llvm::Value* clamp01(llvm::Value* vVec) const {
        llvm::Value* zero = splatFlt(0.0);
        llvm::Value* one = splatFlt(1.0);
        return b.CreateIntrinsic(llvm::Intrinsic::minnum, {b.CreateIntrinsic(llvm::Intrinsic::maxnum, {vVec->getType()}, {vVec, zero})->getType()}, {b.CreateIntrinsic(llvm::Intrinsic::maxnum, {vVec->getType()}, {vVec, zero}), one});
    }

    llvm::Value* evalDepthCmp(D3D12_COMPARISON_FUNC func, llvm::Value* newZ, llvm::Value* oldZ) const {
        switch (func) {
        case D3D12_COMPARISON_FUNC_NEVER:
            return splatInt(i1Ty(), 0);
        case D3D12_COMPARISON_FUNC_LESS:
            return b.CreateFCmp(llvm::CmpInst::FCMP_OLT, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_EQUAL:
            return b.CreateFCmp(llvm::CmpInst::FCMP_OEQ, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_LESS_EQUAL:
            return b.CreateFCmp(llvm::CmpInst::FCMP_OLE, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_GREATER:
            return b.CreateFCmp(llvm::CmpInst::FCMP_OGT, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_NOT_EQUAL:
            return b.CreateFCmp(llvm::CmpInst::FCMP_ONE, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
            return b.CreateFCmp(llvm::CmpInst::FCMP_OGE, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_ALWAYS:
        default:
            return allTrueMask();
        }
    }

    llvm::Value* evalStencilCmp(D3D12_COMPARISON_FUNC func, llvm::Value* refVec, llvm::Value* bufVec) const {
        switch (func) {
        case D3D12_COMPARISON_FUNC_NEVER:
            return splatInt(i1Ty(), 0);
        case D3D12_COMPARISON_FUNC_LESS:
            return b.CreateICmp(llvm::CmpInst::ICMP_ULT, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_EQUAL:
            return b.CreateICmp(llvm::CmpInst::ICMP_EQ, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_LESS_EQUAL:
            return b.CreateICmp(llvm::CmpInst::ICMP_ULE, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_GREATER:
            return b.CreateICmp(llvm::CmpInst::ICMP_UGT, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_NOT_EQUAL:
            return b.CreateICmp(llvm::CmpInst::ICMP_NE, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
            return b.CreateICmp(llvm::CmpInst::ICMP_UGE, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_ALWAYS:
        default:
            return allTrueMask();
        }
    }

    llvm::Value* applyStencilOp(D3D12_STENCIL_OP sop, llvm::Value* oldVal, llvm::Value* refVal) const {
        switch (sop) {
        case D3D12_STENCIL_OP_KEEP:
            return oldVal;
        case D3D12_STENCIL_OP_ZERO:
            return splatInt(i8Ty(), 0);
        case D3D12_STENCIL_OP_REPLACE:
            return refVal;
        case D3D12_STENCIL_OP_INCR_SAT: {
            llvm::Value* sat = b.CreateICmp(llvm::CmpInst::ICMP_EQ, oldVal, splatInt(i8Ty(), 255));
            llvm::Value* inc = b.CreateAdd(oldVal, splatInt(i8Ty(), 1));
            return b.CreateSelect(sat, oldVal, inc);
        }
        case D3D12_STENCIL_OP_DECR_SAT: {
            llvm::Value* z = b.CreateICmp(llvm::CmpInst::ICMP_EQ, oldVal, splatInt(i8Ty(), 0));
            llvm::Value* dec = b.CreateSub(oldVal, splatInt(i8Ty(), 1));
            return b.CreateSelect(z, oldVal, dec);
        }
        case D3D12_STENCIL_OP_INVERT:
            return b.CreateXor(oldVal, splatInt(i8Ty(), -1));
        case D3D12_STENCIL_OP_INCR:
            return b.CreateAdd(oldVal, splatInt(i8Ty(), 1));
        case D3D12_STENCIL_OP_DECR:
            return b.CreateSub(oldVal, splatInt(i8Ty(), 1));
        default:
            return oldVal;
        }
    }

    llvm::Value* evalBlendFactorRGB(
        D3D12_BLEND factor,
        llvm::Value* const src[4],
        llvm::Value* const dst[4],
        llvm::Value* const blendFactor[4],
        int comp) const {
        llvm::Value* one = splatFlt(1.0);
        switch (factor) {
        case D3D12_BLEND_ZERO: return splatFlt(0.0);
        case D3D12_BLEND_ONE: return one;
        case D3D12_BLEND_SRC_COLOR: return src[comp];
        case D3D12_BLEND_INV_SRC_COLOR: return b.CreateFSub(one, src[comp]);
        case D3D12_BLEND_SRC_ALPHA: return src[3];
        case D3D12_BLEND_INV_SRC_ALPHA: return b.CreateFSub(one, src[3]);
        case D3D12_BLEND_DEST_ALPHA: return dst[3];
        case D3D12_BLEND_INV_DEST_ALPHA: return b.CreateFSub(one, dst[3]);
        case D3D12_BLEND_DEST_COLOR: return dst[comp];
        case D3D12_BLEND_INV_DEST_COLOR: return b.CreateFSub(one, dst[comp]);
        case D3D12_BLEND_SRC_ALPHA_SAT: {
            if (comp == 3) return one;
            llvm::Value* invDstA = b.CreateFSub(one, dst[3]);
            return b.CreateIntrinsic(llvm::Intrinsic::minnum, {src[3]->getType()}, {src[3], invDstA});
        }
        case D3D12_BLEND_BLEND_FACTOR: return blendFactor[comp];
        case D3D12_BLEND_INV_BLEND_FACTOR: return b.CreateFSub(one, blendFactor[comp]);
        default: return one;
        }
    }

    llvm::Value* evalBlendOp(D3D12_BLEND_OP op, llvm::Value* sTerm, llvm::Value* dTerm) const {
        switch (op) {
        case D3D12_BLEND_OP_ADD: return b.CreateFAdd(sTerm, dTerm);
        case D3D12_BLEND_OP_SUBTRACT: return b.CreateFSub(sTerm, dTerm);
        case D3D12_BLEND_OP_REV_SUBTRACT: return b.CreateFSub(dTerm, sTerm);
        case D3D12_BLEND_OP_MIN: return b.CreateIntrinsic(llvm::Intrinsic::minnum, {sTerm->getType()}, {sTerm, dTerm});
        case D3D12_BLEND_OP_MAX: return b.CreateIntrinsic(llvm::Intrinsic::maxnum, {sTerm->getType()}, {sTerm, dTerm});
        default: return b.CreateFAdd(sTerm, dTerm);
        }
    }
};

bool structurizeAndLinkStage(
    std::unique_ptr<llvm::Module> stageMod,
    const std::string& entryName,
    const std::string& tempTag,
    llvm::Module& dstMod,
    StructuredFunction& outTree,
    llvm::Function*& outLinkedFn,
    std::string& outError,
    const std::string& extraEntryName = "",
    StructuredFunction* outExtraTree = nullptr,
    llvm::Function** outExtraLinkedFn = nullptr) {
    if (!stageMod) return false;
    llvm::Function* fn = stageMod->getFunction(entryName);
    if (!fn) {
        outError = "Stage entry function not found: " + entryName;
        return false;
    }
    std::string renamedEntry = "__nacelle_src_" + tempTag;
    fn->setName(renamedEntry);
    if (!structurizeAndBuildRegionTree(*fn, outTree, outError)) {
        return false;
    }

    std::string renamedExtra;
    if (!extraEntryName.empty() && outExtraTree) {
        llvm::Function* extraFn = stageMod->getFunction(extraEntryName);
        if (!extraFn) {
            outError = "Stage extra entry function not found: " + extraEntryName;
            return false;
        }
        renamedExtra = "__nacelle_src_" + tempTag + "_extra";
        extraFn->setName(renamedExtra);
        if (!structurizeAndBuildRegionTree(*extraFn, *outExtraTree, outError)) {
            return false;
        }
    }

    if (llvm::Linker::linkModules(dstMod, std::move(stageMod))) {
        outError = "llvm::Linker failed linking stage " + tempTag;
        return false;
    }
    outLinkedFn = dstMod.getFunction(renamedEntry);
    if (outExtraLinkedFn && !renamedExtra.empty()) {
        *outExtraLinkedFn = dstMod.getFunction(renamedExtra);
    }
    return true;
}

void eraseUnusedDxOps(llvm::Module& mod) {
    llvm::SmallVector<llvm::Function*, 16> deadDxOps;
    for (llvm::Function& f : mod) {
        if (f.isDeclaration() && f.getName().starts_with("dx.op.") && f.use_empty()) {
            deadDxOps.push_back(&f);
        }
    }
    for (llvm::Function* f : deadDxOps) f->eraseFromParent();
}

struct LinkedVSSigSlot {
    int32_t psTargetRow = -1;
    uint32_t psStartCol = 0;
};

struct PredecodedVertexElement {
    bool valid = false;
    llvm::Value* compsF32[4]{};
    llvm::Value* compsI32[4]{};
};

PredecodedVertexElement predecodeVertexElementSIMD(
    IRHelpers& h,
    const D3D12_INPUT_ELEMENT_DESC& el,
    llvm::Value* activeMask,
    llvm::Value* vbBasePtr,
    llvm::Value* vbStrideVec,
    llvm::Value* vbSizeVec,
    llvm::Value* idxVec,
    uint32_t byteOffsetInElem,
    uint8_t neededColsMask) {
    PredecodedVertexElement res;
    res.valid = true;
    llvm::IRBuilder<>& b = h.b;
    auto i8Ty = h.i8Ty();
    auto i16Ty = h.i16Ty();
    auto i32Ty = h.i32Ty();
    auto i64Ty = h.i64Ty();
    auto f16Ty = h.f16Ty();
    auto f32Ty = h.f32Ty();
    auto ptrTy = h.ptrTy();

    llvm::Value* zeroI32 = h.splatInt(i32Ty, 0);
    llvm::Value* zeroF32 = h.splatFlt(0.0);
    llvm::Value* oneF32 = h.splatFlt(1.0);
    for (uint32_t c = 0; c < 4; ++c) {
        res.compsF32[c] = (c == 3) ? oneF32 : zeroF32;
        res.compsI32[c] = zeroI32;
    }
    if (neededColsMask == 0) return res;

    uint32_t fmtBytes = formatByteSizeForVertex(el.Format);
    llvm::Value* idx64Vec = b.CreateZExt(idxVec, h.vecTy(i64Ty));
    llvm::Value* str64Vec = b.CreateZExt(vbStrideVec, h.vecTy(i64Ty));
    llvm::Value* off64Vec = b.CreateAdd(b.CreateMul(idx64Vec, str64Vec), h.splatInt(i64Ty, byteOffsetInElem));
    llvm::Value* end64Vec = b.CreateAdd(off64Vec, h.splatInt(i64Ty, fmtBytes));
    llvm::Value* sz64Vec = b.CreateZExt(vbSizeVec, h.vecTy(i64Ty));
    llvm::Value* inBoundsVec = b.CreateICmp(llvm::CmpInst::ICMP_ULE, end64Vec, sz64Vec);
    llvm::Value* ptrNotNull = b.CreateICmp(llvm::CmpInst::ICMP_NE, vbBasePtr, llvm::Constant::getNullValue(ptrTy));
    llvm::Value* ptrNotNullVec = llvmSplat(b, h.i1Ty(), ptrNotNull, h.W);
    llvm::Value* laneOkMask = b.CreateAnd(b.CreateAnd(activeMask, ptrNotNullVec), inBoundsVec);

    auto gatherElem = [&](llvm::Type* elemTy, int64_t deltaBytes, llvm::Value* passThruVec) -> llvm::Value* {
        llvm::Value* compBase = h.gepBytes(vbBasePtr, deltaBytes);
        llvm::Value* ptrVec = b.CreateGEP(i8Ty, compBase, llvm::ArrayRef<llvm::Value*>{off64Vec});
        return b.CreateMaskedGather(h.vecTy(elemTy), ptrVec, llvm::Align(1), laneOkMask, passThruVec);
    };

    auto setFromFloat = [&](uint32_t c, llvm::Value* fv, bool applyOobAlphaDefault) {
        if (c == 3 && applyOobAlphaDefault) {
            fv = b.CreateSelect(laneOkMask, fv, oneF32);
        }
        res.compsF32[c] = fv;
        llvm::Value* asInt = b.CreateBitCast(fv, h.vecTy(i32Ty));
        res.compsI32[c] = b.CreateSelect(laneOkMask, asInt, zeroI32);
    };

    auto setFromInt = [&](uint32_t c, llvm::Value* iv) {
        res.compsI32[c] = iv;
        llvm::Value* asFlt = b.CreateBitCast(iv, h.vecTy(f32Ty));
        res.compsF32[c] = (c == 3) ? b.CreateSelect(laneOkMask, asFlt, oneF32) : asFlt;
    };

    switch (el.Format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT: {
        uint32_t maxComps = fmtBytes / 4;
        for (uint32_t c = 0; c < maxComps; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            llvm::Value* defF = (c == 3) ? oneF32 : zeroF32;
            llvm::Value* fv = gatherElem(f32Ty, static_cast<int64_t>(c * 4), defF);
            setFromFloat(c, fv, /*applyOobAlphaDefault=*/false);
        }
        break;
    }
    case DXGI_FORMAT_R32G32B32A32_UINT:
    case DXGI_FORMAT_R32G32B32A32_SINT:
    case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT:
    case DXGI_FORMAT_R32G32_UINT:
    case DXGI_FORMAT_R32G32_SINT:
    case DXGI_FORMAT_R32_UINT:
    case DXGI_FORMAT_R32_SINT: {
        uint32_t maxComps = fmtBytes / 4;
        for (uint32_t c = 0; c < maxComps; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            llvm::Value* iv = gatherElem(i32Ty, static_cast<int64_t>(c * 4), zeroI32);
            setFromInt(c, iv);
        }
        break;
    }
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16_FLOAT: {
        if (el.Format == DXGI_FORMAT_R16_FLOAT) {
            if (neededColsMask & 1u) {
                llvm::Value* zeroF16 = b.CreateBitCast(h.splatInt(i16Ty, 0), h.vecTy(f16Ty));
                llvm::Value* hv = gatherElem(f16Ty, 0, zeroF16);
                setFromFloat(0, b.CreateFPExt(hv, h.vecTy(f32Ty)), false);
            }
        } else {
            auto unpackPair = [&](llvm::Value* raw32, uint32_t c0, uint32_t c1) {
                if (neededColsMask & (1u << c0)) {
                    llvm::Value* lo16 = b.CreateTrunc(raw32, h.vecTy(i16Ty));
                    llvm::Value* hv0 = b.CreateBitCast(lo16, h.vecTy(f16Ty));
                    setFromFloat(c0, b.CreateFPExt(hv0, h.vecTy(f32Ty)), c0 == 3);
                }
                if (neededColsMask & (1u << c1)) {
                    llvm::Value* hi32 = b.CreateLShr(raw32, h.splatInt(i32Ty, 16));
                    llvm::Value* hi16 = b.CreateTrunc(hi32, h.vecTy(i16Ty));
                    llvm::Value* hv1 = b.CreateBitCast(hi16, h.vecTy(f16Ty));
                    setFromFloat(c1, b.CreateFPExt(hv1, h.vecTy(f32Ty)), c1 == 3);
                }
            };
            if (neededColsMask & 0x3u) {
                unpackPair(gatherElem(i32Ty, 0, zeroI32), 0, 1);
            }
            if (fmtBytes == 8 && (neededColsMask & 0xCu)) {
                unpackPair(gatherElem(i32Ty, 4, zeroI32), 2, 3);
            }
        }
        break;
    }
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16_UNORM:
    case DXGI_FORMAT_R16G16B16A16_SNORM:
    case DXGI_FORMAT_R16G16_SNORM:
    case DXGI_FORMAT_R16_SNORM:
    case DXGI_FORMAT_R16G16B16A16_UINT:
    case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16_UINT:
    case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R16G16_SINT:
    case DXGI_FORMAT_R16_SINT: {
        bool isSigned = (el.Format == DXGI_FORMAT_R16G16B16A16_SNORM || el.Format == DXGI_FORMAT_R16G16_SNORM ||
                         el.Format == DXGI_FORMAT_R16_SNORM || el.Format == DXGI_FORMAT_R16G16B16A16_SINT ||
                         el.Format == DXGI_FORMAT_R16G16_SINT || el.Format == DXGI_FORMAT_R16_SINT);
        bool isNorm = (el.Format == DXGI_FORMAT_R16G16B16A16_UNORM || el.Format == DXGI_FORMAT_R16G16_UNORM ||
                       el.Format == DXGI_FORMAT_R16_UNORM || el.Format == DXGI_FORMAT_R16G16B16A16_SNORM ||
                       el.Format == DXGI_FORMAT_R16G16_SNORM || el.Format == DXGI_FORMAT_R16_SNORM);
        auto decode16 = [&](uint32_t c, llvm::Value* raw32, bool highHalf) {
            llvm::Value* v32;
            if (isSigned) {
                llvm::Value* shl = highHalf ? raw32 : b.CreateShl(raw32, h.splatInt(i32Ty, 16));
                v32 = b.CreateAShr(shl, h.splatInt(i32Ty, 16));
            } else {
                v32 = highHalf ? b.CreateLShr(raw32, h.splatInt(i32Ty, 16))
                               : b.CreateAnd(raw32, h.splatInt(i32Ty, 0xFFFF));
            }
            if (isNorm) {
                llvm::Value* fv = isSigned
                                     ? b.CreateSIToFP(v32, h.vecTy(f32Ty))
                                     : b.CreateUIToFP(v32, h.vecTy(f32Ty));
                fv = b.CreateFMul(fv, h.splatFlt(isSigned ? (1.0 / 32767.0) : (1.0 / 65535.0)));
                if (isSigned) fv = b.CreateIntrinsic(llvm::Intrinsic::maxnum, {fv->getType()}, {fv, h.splatFlt(-1.0)});
                setFromFloat(c, fv, c == 3);
            } else {
                setFromInt(c, v32);
            }
        };
        if (fmtBytes == 2) {
            if (neededColsMask & 1u) {
                llvm::Value* u16 = gatherElem(i16Ty, 0, h.splatInt(i16Ty, 0));
                llvm::Value* raw32 = b.CreateZExt(u16, h.vecTy(i32Ty));
                decode16(0, raw32, false);
            }
        } else {
            if (neededColsMask & 0x3u) {
                llvm::Value* rawLo = gatherElem(i32Ty, 0, zeroI32);
                if (neededColsMask & 1u) decode16(0, rawLo, false);
                if (neededColsMask & 2u) decode16(1, rawLo, true);
            }
            if (fmtBytes == 8 && (neededColsMask & 0xCu)) {
                llvm::Value* rawHi = gatherElem(i32Ty, 4, zeroI32);
                if (neededColsMask & 4u) decode16(2, rawHi, false);
                if (neededColsMask & 8u) decode16(3, rawHi, true);
            }
        }
        break;
    }
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_B8G8R8X8_UNORM:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8G8B8A8_SNORM:
    case DXGI_FORMAT_R8G8_SNORM:
    case DXGI_FORMAT_R8_SNORM:
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8G8B8A8_SINT:
    case DXGI_FORMAT_R8G8_SINT:
    case DXGI_FORMAT_R8_SINT: {
        bool isBGRA = (el.Format == DXGI_FORMAT_B8G8R8A8_UNORM || el.Format == DXGI_FORMAT_B8G8R8X8_UNORM);
        bool isBGRX = (el.Format == DXGI_FORMAT_B8G8R8X8_UNORM);
        bool isSigned = (el.Format == DXGI_FORMAT_R8G8B8A8_SNORM || el.Format == DXGI_FORMAT_R8G8_SNORM ||
                         el.Format == DXGI_FORMAT_R8_SNORM || el.Format == DXGI_FORMAT_R8G8B8A8_SINT ||
                         el.Format == DXGI_FORMAT_R8G8_SINT || el.Format == DXGI_FORMAT_R8_SINT);
        bool isNorm = (el.Format == DXGI_FORMAT_R8G8B8A8_UNORM || el.Format == DXGI_FORMAT_B8G8R8A8_UNORM ||
                       el.Format == DXGI_FORMAT_B8G8R8X8_UNORM || el.Format == DXGI_FORMAT_R8G8_UNORM ||
                       el.Format == DXGI_FORMAT_R8_UNORM || el.Format == DXGI_FORMAT_R8G8B8A8_SNORM ||
                       el.Format == DXGI_FORMAT_R8G8_SNORM || el.Format == DXGI_FORMAT_R8_SNORM);
        uint32_t maxComps = fmtBytes;
        llvm::Value* raw32;
        if (fmtBytes == 4) {
            raw32 = gatherElem(i32Ty, 0, zeroI32);
        } else if (fmtBytes == 2) {
            raw32 = b.CreateZExt(gatherElem(i16Ty, 0, h.splatInt(i16Ty, 0)), h.vecTy(i32Ty));
        } else {
            raw32 = b.CreateZExt(gatherElem(i8Ty, 0, h.splatInt(i8Ty, 0)), h.vecTy(i32Ty));
        }
        for (uint32_t c = 0; c < maxComps; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            if (isBGRX && c == 3) {
                setFromFloat(3, oneF32, false);
                continue;
            }
            uint32_t byteIdx = (isBGRA && c == 0) ? 2u : ((isBGRA && c == 2) ? 0u : c);
            llvm::Value* v32;
            if (isSigned) {
                uint32_t shlAmt = 24u - byteIdx * 8u;
                llvm::Value* shl = (shlAmt > 0)
                                      ? b.CreateShl(raw32, h.splatInt(i32Ty, shlAmt))
                                      : raw32;
                v32 = b.CreateAShr(shl, h.splatInt(i32Ty, 24));
            } else {
                llvm::Value* shr = (byteIdx > 0)
                                      ? b.CreateLShr(raw32, h.splatInt(i32Ty, byteIdx * 8u))
                                      : raw32;
                v32 = (byteIdx == 3) ? shr : b.CreateAnd(shr, h.splatInt(i32Ty, 0xFF));
            }
            if (isNorm) {
                llvm::Value* fv = isSigned
                                     ? b.CreateSIToFP(v32, h.vecTy(f32Ty))
                                     : b.CreateUIToFP(v32, h.vecTy(f32Ty));
                fv = b.CreateFMul(fv, h.splatFlt(isSigned ? (1.0 / 127.0) : (1.0 / 255.0)));
                if (isSigned) fv = b.CreateIntrinsic(llvm::Intrinsic::maxnum, {fv->getType()}, {fv, h.splatFlt(-1.0)});
                setFromFloat(c, fv, c == 3);
            } else {
                setFromInt(c, v32);
            }
        }
        break;
    }
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UINT: {
        bool isNorm = (el.Format == DXGI_FORMAT_R10G10B10A2_UNORM);
        llvm::Value* raw32 = gatherElem(i32Ty, 0, zeroI32);
        llvm::Value* mask10 = h.splatInt(i32Ty, 0x3FF);
        for (uint32_t c = 0; c < 4; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            llvm::Value* u32;
            if (c == 0) {
                u32 = b.CreateAnd(raw32, mask10);
            } else if (c == 1) {
                u32 = b.CreateAnd(b.CreateLShr(raw32, h.splatInt(i32Ty, 10)), mask10);
            } else if (c == 2) {
                u32 = b.CreateAnd(b.CreateLShr(raw32, h.splatInt(i32Ty, 20)), mask10);
            } else {
                u32 = b.CreateLShr(raw32, h.splatInt(i32Ty, 30));
            }
            if (isNorm) {
                llvm::Value* fv = b.CreateUIToFP(u32, h.vecTy(f32Ty));
                fv = b.CreateFMul(fv, h.splatFlt(c == 3 ? (1.0 / 3.0) : (1.0 / 1023.0)));
                setFromFloat(c, fv, c == 3);
            } else {
                setFromInt(c, u32);
            }
        }
        break;
    }
    default:
        break;
    }
    return res;
}

struct CanonicalVSLayoutInfo {
    int32_t posOutputReg = 0;
    uint32_t numVSOutputRegs = 0;
    uint32_t numCanonicalAttrRegs = 0;
    uint32_t clipDistMask = 0;
    uint32_t cullDistMask = 0;
    uint32_t clipAttrOffset = 0;
    uint32_t cullAttrOffset = 0;
    bool hasViewportArrayIndex = false;
    uint32_t vpIndexAttrOffset = 0;
    bool hasPerPrimShadingRate = false;
    uint32_t shadingRateAttrOffset = 0;
    uint32_t totalVSScalarAttrs = 0;
    std::unordered_map<uint32_t, uint32_t> clipSigBaseIdx;
    std::unordered_map<uint32_t, uint32_t> cullSigBaseIdx;
};

CanonicalVSLayoutInfo computeCanonicalVSLayout(const std::vector<DxilSigRecord>& vsOutputSigs) {
    CanonicalVSLayoutInfo info;
    for (const auto& vo : vsOutputSigs) {
        if (vo.startRow < 64) {
            info.numVSOutputRegs = std::max(info.numVSOutputRegs, vo.startRow + vo.rows);
        }
        if (vo.systemValueType == 3 && vo.startRow < 64) {
            info.posOutputReg = static_cast<int32_t>(vo.startRow);
        }
        if (vo.systemValueType == 0 && vo.startRow < 32) {
            info.numCanonicalAttrRegs = std::max(info.numCanonicalAttrRegs, vo.startRow + vo.rows);
        }
    }

    uint32_t nextExtraAttr = info.numCanonicalAttrRegs * 4;
    info.clipAttrOffset = nextExtraAttr;
    uint32_t clipScalarCount = 0;
    for (const auto& vo : vsOutputSigs) {
        if (vo.systemValueType == 6) {
            info.clipSigBaseIdx[vo.sigId] = clipScalarCount;
            uint32_t count = std::min<uint32_t>(8u - clipScalarCount, vo.rows * vo.cols);
            for (uint32_t k = 0; k < count; ++k) {
                info.clipDistMask |= (1u << (clipScalarCount + k));
            }
            clipScalarCount += count;
        }
    }
    nextExtraAttr += clipScalarCount;
    info.cullAttrOffset = nextExtraAttr;
    uint32_t cullScalarCount = 0;
    for (const auto& vo : vsOutputSigs) {
        if (vo.systemValueType == 7) {
            info.cullSigBaseIdx[vo.sigId] = cullScalarCount;
            uint32_t count = std::min<uint32_t>(8u - cullScalarCount, vo.rows * vo.cols);
            for (uint32_t k = 0; k < count; ++k) {
                info.cullDistMask |= (1u << (cullScalarCount + k));
            }
            cullScalarCount += count;
        }
    }
    nextExtraAttr += cullScalarCount;
    for (const auto& vo : vsOutputSigs) {
        if (vo.systemValueType == 5 && !info.hasViewportArrayIndex) {
            info.hasViewportArrayIndex = true;
            info.vpIndexAttrOffset = nextExtraAttr++;
        } else if (vo.systemValueType == 29 && !info.hasPerPrimShadingRate) {
            info.hasPerPrimShadingRate = true;
            info.shadingRateAttrOffset = nextExtraAttr++;
        }
    }
    info.totalVSScalarAttrs = nextExtraAttr;
    return info;
}

bool emitVertexStageFunction(
    llvm::Module& targetMod,
    const StructuredFunction& vsTree,
    const DxilLoadedShader& vsLoaded,
    const RootSignatureLayout& vsEffectiveLayout,
    const D3D12_INPUT_LAYOUT_DESC& inputLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    bool hasTess,
    bool hasGS,
    const CanonicalVSLayoutInfo* canonicalLayout,
    const std::unordered_map<uint32_t, LinkedVSSigSlot>* vsToNextMap,
    std::string& outError) {
    if (!vsTree.root) {
        outError = "Vertex shader structured region tree missing";
        return false;
    }

    auto ptrTy = llvm::PointerType::getUnqual(targetMod.getContext());
    auto i32Ty = llvm::Type::getInt32Ty(targetMod.getContext());
    auto fnTy = llvm::FunctionType::get(llvm::Type::getVoidTy(targetMod.getContext()), {ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy}, false);
    auto* vsFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym, targetMod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(targetMod.getContext(), "entry", vsFn);
    llvm::IRBuilder<> b(entry);
    IRHelpers h{b, simdWidth};

    llvm::Value* rootCtx = entry->getParent()->getArg(0);
    llvm::Value* vertexIdsPtr = entry->getParent()->getArg(1);
    llvm::Value* instanceIdsPtr = entry->getParent()->getArg(2);
    llvm::Value* activeCount = entry->getParent()->getArg(3);
    llvm::Value* outPosXYZW = entry->getParent()->getArg(4);
    llvm::Value* outAttrs = entry->getParent()->getArg(5);

    llvm::Value* acVec = llvmSplat(b, i32Ty, activeCount, h.W);
    llvm::Value* activeMask = b.CreateICmp(llvm::CmpInst::ICMP_ULT, h.laneIota(), acVec);

    // Callers pad tail lanes of vertexIdsPtr & instanceIdsPtr to W; load directly without SelectOp
    // so LLVM does not scalarize into W conditional scalar loads.
    llvm::Value* zeroI32Vec = h.splatInt(i32Ty, 0);
    llvm::Value* vidVec = h.loadScalar(h.vecTy(i32Ty), vertexIdsPtr);
    llvm::Value* iidVec = h.loadScalar(h.vecTy(i32Ty), instanceIdsPtr);

    // Resolve InputLayout offsets per slot and match against VS input signatures.
    std::vector<uint32_t> elementByteOffsets(inputLayout.NumElements, 0);
    uint32_t slotRunningOffset[16]{};
    for (UINT i = 0; i < inputLayout.NumElements && inputLayout.pInputElementDescs; ++i) {
        const auto& el = inputLayout.pInputElementDescs[i];
        UINT slot = std::min<UINT>(el.InputSlot, 15);
        uint32_t off = (el.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT)
                           ? slotRunningOffset[slot]
                           : el.AlignedByteOffset;
        elementByteOffsets[i] = off;
        slotRunningOffset[slot] = off + formatByteSizeForVertex(el.Format);
    }

    std::unordered_map<uint32_t, UINT> sigToElemIdx;
    std::vector<uint8_t> elemNeededCols(inputLayout.NumElements, 0);
    for (const auto& sig : vsLoaded.inputSigs) {
        if (sig.systemValueType != 0) continue;
        std::string sigNameLower = toLowerAscii(sig.semanticName);
        for (UINT i = 0; i < inputLayout.NumElements && inputLayout.pInputElementDescs; ++i) {
            const auto& el = inputLayout.pInputElementDescs[i];
            if (el.SemanticName && toLowerAscii(el.SemanticName) == sigNameLower &&
                el.SemanticIndex == sig.semanticIndex) {
                sigToElemIdx[sig.sigId] = i;
                uint8_t colMask = 0;
                for (uint32_t c = 0; c < sig.cols && c < 4; ++c) {
                    colMask |= static_cast<uint8_t>(1u << c);
                }
                elemNeededCols[i] |= (colMask != 0) ? colMask : 0xFu;
                break;
            }
        }
    }

    // Preload referenced vertex buffer descriptors and pre-decode all referenced InputLayout elements
    // once in the entry block so SSA values dominate all control flow and gathers are never duplicated.
    llvm::Value* vbBasePtrs[16]{};
    llvm::Value* vbStrideVecs[16]{};
    llvm::Value* vbSizeVecs[16]{};
    std::vector<PredecodedVertexElement> predecodedElems(inputLayout.NumElements);
    for (UINT i = 0; i < inputLayout.NumElements && inputLayout.pInputElementDescs; ++i) {
        if (elemNeededCols[i] == 0) continue;
        const auto& el = inputLayout.pInputElementDescs[i];
        UINT slot = std::min<UINT>(el.InputSlot, 15);
        if (!vbBasePtrs[slot]) {
            llvm::Value* addrInt = h.loadScalar(
                h.i64Ty(), h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_buffers) + slot * 8)));
            vbBasePtrs[slot] = b.CreateIntToPtr(addrInt, ptrTy);
            llvm::Value* str32 = h.loadScalar(
                i32Ty, h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_strides) + slot * 4)));
            llvm::Value* sz32 = h.loadScalar(
                i32Ty, h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_sizes) + slot * 4)));
            vbStrideVecs[slot] = llvmSplat(b, i32Ty, str32, h.W);
            vbSizeVecs[slot] = llvmSplat(b, i32Ty, sz32, h.W);
        }
        llvm::Value* idxVec = vidVec;
        if (el.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA) {
            if (el.InstanceDataStepRate == 0) {
                idxVec = zeroI32Vec;
            } else if (el.InstanceDataStepRate == 1) {
                idxVec = iidVec;
            } else {
                idxVec = b.CreateUDiv(iidVec, h.splatInt(i32Ty, el.InstanceDataStepRate));
            }
        }
        predecodedElems[i] = predecodeVertexElementSIMD(
            h, el, activeMask, vbBasePtrs[slot], vbStrideVecs[slot], vbSizeVecs[slot],
            idxVec, elementByteOffsets[i], elemNeededCols[i]);
    }

    StageHooks vsHooks;
    vsHooks.isPixelShader = false;
    vsHooks.loadInput = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                            const SimdVal&, uint32_t col, llvm::Type* scalarTy, SimdVal& out) -> bool {
        const DxilSigRecord* sig = vsLoaded.findInputSig(sigId);
        if (!sig) return false;
        IRHelpers lh{lb, simdWidth};
        bool wantInt = scalarTy->isIntegerTy();

        if (sig->systemValueType == 1) { // SV_VertexID
            llvm::Value* v = wantInt ? vidVec : lb.CreateBitCast(vidVec, lh.vecTy(scalarTy));
            if (wantInt && scalarTy != i32Ty) v = lb.CreateTrunc(v, lh.vecTy(scalarTy));
            out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
            return true;
        }
        if (sig->systemValueType == 2) { // SV_InstanceID
            llvm::Value* v = wantInt ? iidVec : lb.CreateBitCast(iidVec, lh.vecTy(scalarTy));
            if (wantInt && scalarTy != i32Ty) v = lb.CreateTrunc(v, lh.vecTy(scalarTy));
            out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
            return true;
        }

        auto eit = sigToElemIdx.find(sigId);
        if (eit != sigToElemIdx.end() && col < 4 && predecodedElems[eit->second].valid) {
            const auto& pe = predecodedElems[eit->second];
            llvm::Value* decoded = wantInt ? pe.compsI32[col] : pe.compsF32[col];
            if (wantInt && scalarTy != i32Ty) {
                decoded = lb.CreateTrunc(decoded, lh.vecTy(scalarTy));
            } else if (!wantInt && scalarTy != lh.f32Ty()) {
                decoded = lb.CreateFPTrunc(decoded, lh.vecTy(scalarTy));
            }
            out = {SimdVal::Kind::Vector, scalarTy, decoded, {}, {}};
            return true;
        }
        return false;
    };

    vsHooks.storeOutput = [&](llvm::IRBuilder<>& sb, uint32_t sigId,
                              const SimdVal& row, uint32_t col, const SimdVal& val, llvm::Value* writeMask) -> bool {
        const DxilSigRecord* sig = vsLoaded.findOutputSig(sigId);
        if (!sig || val.kind != SimdVal::Kind::Vector) return true;
        IRHelpers sh{sb, simdWidth};
        llvm::Value* vVec = val.vec;
        llvm::Type* elemTy = llvm::cast<llvm::FixedVectorType>(vVec->getType())->getElementType();
        if (elemTy != sh.f32Ty()) {
            if (elemTy->isIntegerTy()) {
                if (elemTy->getScalarSizeInBits() < 32) {
                    vVec = sb.CreateZExt(vVec, sh.vecTy(i32Ty));
                }
                vVec = sb.CreateBitCast(vVec, sh.vecTy(sh.f32Ty()));
            } else {
                vVec = sb.CreateFPExt(vVec, sh.vecTy(sh.f32Ty()));
            }
        }

        auto storeVecSOA = [&](llvm::Value* dstPtr) {
            if (!writeMask || writeMask == activeMask) {
                sb.CreateStore(vVec, dstPtr);
            } else {
                llvm::Value* prev = sh.loadScalar(sh.vecTy(sh.f32Ty()), dstPtr);
                llvm::Value* merged = sb.CreateSelect(writeMask, vVec, prev);
                sb.CreateStore(merged, dstPtr);
            }
        };

        auto storeAttrAtSlot = [&](uint32_t baseSlot) {
            auto rowConst = tryGetUniformConstUInt(row);
            if (rowConst.has_value()) {
                uint32_t scalarSlot = baseSlot + (*rowConst) * 4 + col;
                llvm::Value* dst = sh.gepBytes(outAttrs, static_cast<int64_t>(scalarSlot * simdWidth * 4));
                storeVecSOA(dst);
            } else {
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    llvm::Value* mL = llvmExtract(sb, writeMask ? writeMask : activeMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    emitIf(sb, mL, [&](llvm::IRBuilder<>& nb) {
                        IRHelpers nh{nb, simdWidth};
                        llvm::Value* rL = llvmExtract(nb, row.vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* r32 = (rL->getType() == i32Ty)
                                              ? rL
                                              : nb.CreateZExt(rL, i32Ty);
                        llvm::Value* slot32 = nb.CreateAdd(nh.cInt(i32Ty, static_cast<int64_t>(baseSlot + col)), nb.CreateMul(r32, nh.cInt(i32Ty, 4)));
                        llvm::Value* elemIdx = nb.CreateAdd(nb.CreateMul(slot32, nh.cInt(i32Ty, simdWidth)), nh.cInt(i32Ty, static_cast<int64_t>(l)));
                        llvm::Value* byteOff = nb.CreateMul(nb.CreateZExt(elemIdx, nh.i64Ty()), nh.cInt(nh.i64Ty(), 4));
                        llvm::Value* vL = llvmExtract(nb, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        nb.CreateStore(vL, nh.gepBytesDyn(outAttrs, byteOff));
                    });
                }
            }
        };

        if (sig->systemValueType == 3) { // SV_Position -> outPosXYZW[col * W + lane]
            if (col < 4) {
                llvm::Value* dst = sh.gepBytes(outPosXYZW, static_cast<int64_t>(col * simdWidth * 4));
                storeVecSOA(dst);
            }
            if (!hasTess) return true;
        }

        if (!hasGS && !hasTess && canonicalLayout) {
            if (sig->systemValueType == 6) { // SV_ClipDistance
                auto cit = canonicalLayout->clipSigBaseIdx.find(sigId);
                if (cit != canonicalLayout->clipSigBaseIdx.end()) {
                    uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (canonicalLayout->clipDistMask & (1u << idx))) {
                        storeVecSOA(sh.gepBytes(outAttrs, static_cast<int64_t>((canonicalLayout->clipAttrOffset + idx) * simdWidth * 4)));
                    }
                }
                return true;
            }
            if (sig->systemValueType == 7) { // SV_CullDistance
                auto cit = canonicalLayout->cullSigBaseIdx.find(sigId);
                if (cit != canonicalLayout->cullSigBaseIdx.end()) {
                    uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (canonicalLayout->cullDistMask & (1u << idx))) {
                        storeVecSOA(sh.gepBytes(outAttrs, static_cast<int64_t>((canonicalLayout->cullAttrOffset + idx) * simdWidth * 4)));
                    }
                }
                return true;
            }
            if (sig->systemValueType == 5 && canonicalLayout->hasViewportArrayIndex && col == 0) {
                storeVecSOA(sh.gepBytes(outAttrs, static_cast<int64_t>(canonicalLayout->vpIndexAttrOffset * simdWidth * 4)));
                return true;
            }
            if (sig->systemValueType == 29 && canonicalLayout->hasPerPrimShadingRate && col == 0) {
                storeVecSOA(sh.gepBytes(outAttrs, static_cast<int64_t>(canonicalLayout->shadingRateAttrOffset * simdWidth * 4)));
                return true;
            }
            if (sig->systemValueType == 0 && sig->startRow < 32) {
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol;
                storeAttrAtSlot(baseSlot);
            }
            return true;
        }

        if (vsToNextMap) {
            auto it = vsToNextMap->find(sigId);
            if (it != vsToNextMap->end() && it->second.psTargetRow >= 0) {
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol;
                storeAttrAtSlot(baseSlot);
            }
        }
        return true;
    };

    llvm::Value* vsRealMask = activeMask;
    if (!packetizeRegionWithHooks(b, targetMod, *vsTree.root, vsLoaded, vsEffectiveLayout,
                                  simdWidth, rootCtx, activeMask, vsRealMask, vsHooks, outError)) {
        return false;
    }
    b.CreateRetVoid();
    return true;
}

} // namespace

bool synthesizeVertexStageLLVM(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    llvm::LLVMContext& llvmCtx,
    std::unique_ptr<llvm::Module>& outVSMod,
    CachedVSMeta& outVSMeta,
    std::vector<ShaderSigElement>& outVSInputs,
    std::vector<ShaderSigElement>& outVSOutputs,
    std::vector<DxilSigRecord>& outVSOutputRecs,
    std::string& outError) {
    if (!desc.VS.pShaderBytecode || desc.VS.BytecodeLength == 0) {
        outError = "Graphics PSO requires a valid Vertex Shader bytecode";
        return false;
    }

    DxilLoadedShader vsLoaded;
    if (!loadAndUpgradeDXIL(desc.VS, llvmCtx, vsLoaded, outError)) {
        return false;
    }

    const RootSignatureLayout& vsEffectiveLayout =
        (rootLayout.params.empty() && rootLayout.staticSamplers.empty() && vsLoaded.hasEmbeddedRootSig)
            ? vsLoaded.embeddedRootSig
            : rootLayout;

    outVSMod = std::make_unique<llvm::Module>("nacelle_vs", llvmCtx);
    llvm::Module& vsMod = *outVSMod;
    StructuredFunction vsTree;
    llvm::Function* vsSrcFn = nullptr;
    if (!structurizeAndLinkStage(std::move(vsLoaded.module), vsLoaded.entryName, "vs", vsMod, vsTree, vsSrcFn, outError)) {
        return false;
    }

    outVSInputs = convertSigElements(vsLoaded.inputSigs);
    outVSOutputs = convertSigElements(vsLoaded.outputSigs);
    outVSOutputRecs = vsLoaded.outputSigs;

    CanonicalVSLayoutInfo canonical = computeCanonicalVSLayout(vsLoaded.outputSigs);
    if (!emitVertexStageFunction(
            vsMod, vsTree, vsLoaded, vsEffectiveLayout, desc.InputLayout,
            simdWidth, vsSym, /*hasTess=*/false, /*hasGS=*/false, &canonical, nullptr, outError)) {
        return false;
    }
    if (vsSrcFn) vsSrcFn->eraseFromParent();
    eraseUnusedDxOps(vsMod);

    if (llvm::verifyModule(vsMod, &llvm::errs())) {
        outError = "LLVM verification failed after Vertex Stage synthesis";
        return false;
    }

    outVSMeta = CachedVSMeta{};
    outVSMeta.simdWidth = simdWidth;
    outVSMeta.numVSOutputRegs = canonical.numVSOutputRegs;
    outVSMeta.posOutputReg = canonical.posOutputReg;
    outVSMeta.clipDistMask = canonical.clipDistMask;
    outVSMeta.cullDistMask = canonical.cullDistMask;
    outVSMeta.clipAttrOffset = canonical.clipAttrOffset;
    outVSMeta.cullAttrOffset = canonical.cullAttrOffset;
    outVSMeta.hasViewportArrayIndex = canonical.hasViewportArrayIndex ? 1 : 0;
    outVSMeta.vpIndexAttrOffset = canonical.vpIndexAttrOffset;
    outVSMeta.hasPerPrimShadingRate = canonical.hasPerPrimShadingRate ? 1 : 0;
    outVSMeta.shadingRateAttrOffset = canonical.shadingRateAttrOffset;
    outVSMeta.totalVSScalarAttrs = canonical.totalVSScalarAttrs;
    return true;
}

bool synthesizeGraphicsPSOLLVM(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    const std::string& psSym,
    llvm::LLVMContext& llvmCtx,
    std::unique_ptr<llvm::Module>& outCombinedMod,
    CachedPSOMeta& outMeta,
    std::vector<ShaderSigElement>& outVSInputs,
    std::vector<ShaderSigElement>& outVSOutputs,
    std::vector<ShaderSigElement>& outPSInputs,
    std::vector<ShaderSigElement>& outPSOutputs,
    std::string& outError,
    const MeshShaderStages& meshStages,
    bool skipVS,
    const CachedVSMeta* precompiledVSMeta,
    const std::vector<DxilSigRecord>* precompiledVSOutputRecs) {
    const bool hasMS = (meshStages.MS.pShaderBytecode && meshStages.MS.BytecodeLength > 0);
    const bool hasAS = hasMS && (meshStages.AS.pShaderBytecode && meshStages.AS.BytecodeLength > 0);
    if (!hasMS && (!desc.VS.pShaderBytecode || desc.VS.BytecodeLength == 0)) {
        outError = "Graphics PSO requires a valid Vertex or Mesh Shader bytecode";
        return false;
    }

    bool hasTess = !hasMS && (desc.HS.pShaderBytecode && desc.HS.BytecodeLength > 0 &&
                              desc.DS.pShaderBytecode && desc.DS.BytecodeLength > 0);
    bool hasGS = !hasMS && (desc.GS.pShaderBytecode && desc.GS.BytecodeLength > 0);
    if (hasMS || hasTess || hasGS) {
        skipVS = false;
    }

    outCombinedMod = std::make_unique<llvm::Module>("nacelle_pso", llvmCtx);
    llvm::Module& combinedMod = *outCombinedMod;

    DxilLoadedShader msLoaded;
    StructuredFunction msTree;
    llvm::Function* msSrcFn = nullptr;
    DxilLoadedShader asLoaded;
    StructuredFunction asTree;
    llvm::Function* asSrcFn = nullptr;
    if (hasMS) {
        if (!loadAndUpgradeDXIL(meshStages.MS, llvmCtx, msLoaded, outError)) {
            return false;
        }
        if (!structurizeAndLinkStage(std::move(msLoaded.module), msLoaded.entryName, "ms", combinedMod, msTree, msSrcFn, outError)) {
            return false;
        }
        if (hasAS) {
            if (!loadAndUpgradeDXIL(meshStages.AS, llvmCtx, asLoaded, outError)) {
                return false;
            }
            if (!structurizeAndLinkStage(std::move(asLoaded.module), asLoaded.entryName, "as", combinedMod, asTree, asSrcFn, outError)) {
                return false;
            }
        }
    }

    const bool emptyRoot = rootLayout.params.empty() && rootLayout.staticSamplers.empty();
    const RootSignatureLayout& msEffectiveLayout =
        (hasMS && emptyRoot && msLoaded.hasEmbeddedRootSig)
            ? msLoaded.embeddedRootSig
            : ((hasAS && emptyRoot && asLoaded.hasEmbeddedRootSig) ? asLoaded.embeddedRootSig : rootLayout);
    const RootSignatureLayout& asEffectiveLayout =
        (hasAS && emptyRoot && asLoaded.hasEmbeddedRootSig) ? asLoaded.embeddedRootSig : msEffectiveLayout;

    DxilLoadedShader vsLoaded;
    StructuredFunction vsTree;
    llvm::Function* vsSrcFn = nullptr;
    bool loadedVSFromDXIL = false;
    if (!hasMS) {
        if (!skipVS || !precompiledVSOutputRecs || precompiledVSOutputRecs->empty() || emptyRoot) {
            if (!loadAndUpgradeDXIL(desc.VS, llvmCtx, vsLoaded, outError)) {
                return false;
            }
            loadedVSFromDXIL = true;
        }
        if (!skipVS) {
            if (!structurizeAndLinkStage(std::move(vsLoaded.module), vsLoaded.entryName, "vs", combinedMod, vsTree, vsSrcFn, outError)) {
                return false;
            }
        }
    }

    const RootSignatureLayout& vsEffectiveLayout =
        (loadedVSFromDXIL && emptyRoot && vsLoaded.hasEmbeddedRootSig)
            ? vsLoaded.embeddedRootSig
            : (hasMS ? msEffectiveLayout : rootLayout);

    const std::vector<DxilSigRecord>& vsOutputSigsRef =
        (skipVS && precompiledVSOutputRecs && !precompiledVSOutputRecs->empty())
            ? *precompiledVSOutputRecs
            : vsLoaded.outputSigs;

    DxilLoadedShader hsLoaded;
    StructuredFunction hsTree;
    StructuredFunction hspcTree;
    llvm::Function* hsSrcFn = nullptr;
    llvm::Function* hspcSrcFn = nullptr;
    DxilLoadedShader dsLoaded;
    StructuredFunction dsTree;
    llvm::Function* dsSrcFn = nullptr;
    if (hasTess) {
        if (!loadAndUpgradeDXIL(desc.HS, llvmCtx, hsLoaded, outError)) {
            return false;
        }
        if (!structurizeAndLinkStage(
                std::move(hsLoaded.module), hsLoaded.entryName, "hs", combinedMod, hsTree, hsSrcFn, outError,
                hsLoaded.hsPatchConstFnName, &hspcTree, &hspcSrcFn)) {
            return false;
        }

        if (!loadAndUpgradeDXIL(desc.DS, llvmCtx, dsLoaded, outError)) {
            return false;
        }
        if (!structurizeAndLinkStage(std::move(dsLoaded.module), dsLoaded.entryName, "ds", combinedMod, dsTree, dsSrcFn, outError)) {
            return false;
        }
    }

    const RootSignatureLayout& hsEffectiveLayout =
        (hasTess && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && hsLoaded.hasEmbeddedRootSig)
            ? hsLoaded.embeddedRootSig
            : vsEffectiveLayout;
    const RootSignatureLayout& dsEffectiveLayout =
        (hasTess && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && dsLoaded.hasEmbeddedRootSig)
            ? dsLoaded.embeddedRootSig
            : vsEffectiveLayout;

    DxilLoadedShader gsLoaded;
    StructuredFunction gsTree;
    llvm::Function* gsSrcFn = nullptr;
    if (hasGS) {
        if (!loadAndUpgradeDXIL(desc.GS, llvmCtx, gsLoaded, outError)) {
            return false;
        }
        if (!structurizeAndLinkStage(std::move(gsLoaded.module), gsLoaded.entryName, "gs", combinedMod, gsTree, gsSrcFn, outError)) {
            return false;
        }
    }

    const RootSignatureLayout& gsEffectiveLayout =
        (hasGS && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && gsLoaded.hasEmbeddedRootSig)
            ? gsLoaded.embeddedRootSig
            : vsEffectiveLayout;

    bool hasPS = (desc.PS.pShaderBytecode && desc.PS.BytecodeLength > 0);
    DxilLoadedShader psLoaded;
    StructuredFunction psTree;
    llvm::Function* psSrcFn = nullptr;
    if (hasPS) {
        if (!loadAndUpgradeDXIL(desc.PS, llvmCtx, psLoaded, outError)) {
            return false;
        }
        if (!structurizeAndLinkStage(std::move(psLoaded.module), psLoaded.entryName, "ps", combinedMod, psTree, psSrcFn, outError)) {
            return false;
        }
    }

    const RootSignatureLayout& psEffectiveLayout =
        (hasPS && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && psLoaded.hasEmbeddedRootSig)
            ? psLoaded.embeddedRootSig
            : vsEffectiveLayout;

    llvm::IRBuilder<> b(llvmCtx);

    if (hasMS) {
        outVSInputs.clear();
        outVSOutputs = convertSigElements(msLoaded.outputSigs);
    } else if (loadedVSFromDXIL) {
        outVSInputs = convertSigElements(vsLoaded.inputSigs);
        outVSOutputs = convertSigElements(vsLoaded.outputSigs);
    }
    bool psWritesDepth = false;
    bool psUsesDiscard = hasPS && psLoaded.usesDiscard;
    bool psHasSideEffects = hasPS && psLoaded.hasSideEffects;
    bool psUsesPerspBarycentrics = false;
    if (hasPS) {
        outPSInputs = convertSigElements(psLoaded.inputSigs);
        outPSOutputs = convertSigElements(psLoaded.outputSigs);
        for (const auto& po : psLoaded.outputSigs) {
            if (po.systemValueType >= 17 && po.systemValueType <= 19) psWritesDepth = true;
        }
        for (const auto& pi : psLoaded.inputSigs) {
            if (pi.systemValueType == 28 && pi.interpMode != 4) {
                psUsesPerspBarycentrics = true;
            }
        }
    } else {
        outPSInputs.clear();
        outPSOutputs.clear();
    }
    uint32_t numPSInputRegs = 0;
    uint32_t flatInterpMask = 0;
    if (hasPS) {
        for (const auto& pi : psLoaded.inputSigs) {
            if (pi.systemValueType == 0) {
                numPSInputRegs = std::max(numPSInputRegs, pi.startRow + pi.rows);
                for (uint32_t r = 0; r < pi.rows; ++r) {
                    uint32_t reg = pi.startRow + r;
                    if (pi.interpMode == 1) {
                        flatInterpMask |= (1u << std::min<uint32_t>(reg, 15u));
                    } else if (pi.interpMode == 4 && reg < 16) {
                        flatInterpMask |= (1u << (16u + reg));
                    }
                }
            }
        }
    }

    uint32_t numHSInputRegs = 0;
    uint32_t numHSOutputRegs = 0;
    uint32_t numDSInputRegs = 0;
    uint32_t numPatchConstRegs = 0;
    if (hasTess) {
        for (const auto& hi : hsLoaded.inputSigs) {
            numHSInputRegs = std::max(numHSInputRegs, hi.startRow + hi.rows);
        }
        for (const auto& ho : hsLoaded.outputSigs) {
            numHSOutputRegs = std::max(numHSOutputRegs, ho.startRow + ho.rows);
        }
        for (const auto& di : dsLoaded.inputSigs) {
            numDSInputRegs = std::max(numDSInputRegs, di.startRow + di.rows);
        }
        for (const auto& hpo : hsLoaded.patchConstSigs) {
            if (hpo.systemValueType != 25 && hpo.systemValueType != 26) {
                numPatchConstRegs = std::max(numPatchConstRegs, hpo.startRow + hpo.rows);
            }
        }
        for (const auto& dpi : dsLoaded.patchConstSigs) {
            if (dpi.systemValueType != 25 && dpi.systemValueType != 26) {
                numPatchConstRegs = std::max(numPatchConstRegs, dpi.startRow + dpi.rows);
            }
        }
    }
    uint32_t numHSInputScalars = numHSInputRegs * 4;
    uint32_t numHSOutputScalars = std::max(numHSOutputRegs, numDSInputRegs) * 4;
    uint32_t numPatchConstScalars = numPatchConstRegs * 4;

    uint32_t numGSInputRegs = 0;
    if (hasGS) {
        for (const auto& gi : gsLoaded.inputSigs) {
            if (gi.systemValueType == 0) {
                numGSInputRegs = std::max(numGSInputRegs, gi.startRow + gi.rows);
            }
        }
    }

    std::unordered_map<uint32_t, LinkedVSSigSlot> vsToNextMap;
    std::unordered_map<uint32_t, LinkedVSSigSlot> vsToPsMap;
    if (hasTess) {
        for (const auto& vo : vsOutputSigsRef) {
            if (vo.startRow == UINT32_MAX) continue;
            std::string vn = toLowerAscii(vo.semanticName);
            for (const auto& hi : hsLoaded.inputSigs) {
                if (toLowerAscii(hi.semanticName) == vn && hi.semanticIndex == vo.semanticIndex) {
                    vsToNextMap[vo.sigId] = {static_cast<int32_t>(hi.startRow), hi.startCol};
                    break;
                }
            }
        }
    } else if (hasGS) {
        for (const auto& vo : vsOutputSigsRef) {
            if (vo.startRow == UINT32_MAX) continue;
            if (vo.systemValueType == 3) continue;
            std::string vn = toLowerAscii(vo.semanticName);
            for (const auto& gi : gsLoaded.inputSigs) {
                if (gi.systemValueType != 0) continue;
                if (toLowerAscii(gi.semanticName) == vn && gi.semanticIndex == vo.semanticIndex) {
                    vsToNextMap[vo.sigId] = {static_cast<int32_t>(gi.startRow), gi.startCol};
                    break;
                }
            }
        }
    } else if (hasMS && hasPS) {
        for (const auto& vo : msLoaded.outputSigs) {
            if (vo.startRow == UINT32_MAX) continue;
            if (vo.systemValueType == 3) continue;
            std::string vn = toLowerAscii(vo.semanticName);
            for (const auto& pi : psLoaded.inputSigs) {
                if (pi.systemValueType != 0) continue;
                if (toLowerAscii(pi.semanticName) == vn && pi.semanticIndex == vo.semanticIndex) {
                    vsToPsMap[vo.sigId] = {static_cast<int32_t>(pi.startRow), pi.startCol};
                    break;
                }
            }
        }
    }

    std::unordered_map<uint32_t, LinkedVSSigSlot> hsToDsMap;
    std::unordered_map<uint32_t, LinkedVSSigSlot> hsPcToDsMap;
    std::unordered_map<uint32_t, LinkedVSSigSlot> dsToNextMap;
    if (hasTess) {
        for (const auto& ho : hsLoaded.outputSigs) {
            std::string hn = toLowerAscii(ho.semanticName);
            for (const auto& di : dsLoaded.inputSigs) {
                if (toLowerAscii(di.semanticName) == hn && di.semanticIndex == ho.semanticIndex) {
                    hsToDsMap[ho.sigId] = {static_cast<int32_t>(di.startRow), di.startCol};
                    break;
                }
            }
        }
        for (const auto& hpo : hsLoaded.patchConstSigs) {
            if (hpo.systemValueType == 25 || hpo.systemValueType == 26) continue;
            std::string hpn = toLowerAscii(hpo.semanticName);
            for (const auto& dpi : dsLoaded.patchConstSigs) {
                if (dpi.systemValueType == 25 || dpi.systemValueType == 26) continue;
                if (toLowerAscii(dpi.semanticName) == hpn && dpi.semanticIndex == hpo.semanticIndex) {
                    hsPcToDsMap[hpo.sigId] = {static_cast<int32_t>(dpi.startRow), dpi.startCol};
                    break;
                }
            }
        }
        if (hasGS) {
            for (const auto& dso : dsLoaded.outputSigs) {
                if (dso.systemValueType == 3) continue;
                std::string dn = toLowerAscii(dso.semanticName);
                for (const auto& gi : gsLoaded.inputSigs) {
                    if (gi.systemValueType != 0) continue;
                    if (toLowerAscii(gi.semanticName) == dn && gi.semanticIndex == dso.semanticIndex) {
                        dsToNextMap[dso.sigId] = {static_cast<int32_t>(gi.startRow), gi.startCol};
                        break;
                    }
                }
            }
        } else if (hasPS) {
            for (const auto& dso : dsLoaded.outputSigs) {
                if (dso.systemValueType == 3) continue;
                std::string dn = toLowerAscii(dso.semanticName);
                for (const auto& pi : psLoaded.inputSigs) {
                    if (pi.systemValueType != 0) continue;
                    if (toLowerAscii(pi.semanticName) == dn && pi.semanticIndex == dso.semanticIndex) {
                        dsToNextMap[dso.sigId] = {static_cast<int32_t>(pi.startRow), pi.startCol};
                        break;
                    }
                }
            }
        }
    }

    std::unordered_map<uint32_t, LinkedVSSigSlot> gsToPsMap;
    if (hasGS && hasPS) {
        for (const auto& go : gsLoaded.outputSigs) {
            if (go.systemValueType == 3) continue;
            std::string gn = toLowerAscii(go.semanticName);
            for (const auto& pi : psLoaded.inputSigs) {
                if (pi.systemValueType != 0) continue;
                if (toLowerAscii(pi.semanticName) == gn && pi.semanticIndex == go.semanticIndex) {
                    gsToPsMap[go.sigId] = {static_cast<int32_t>(pi.startRow), pi.startCol};
                    break;
                }
            }
        }
    }

    std::unordered_map<uint32_t, LinkedVSSigSlot> msPrimToPsMap;
    uint64_t msPrimAttrMask = 0;
    if (hasMS && hasPS) {
        for (const auto& po : msLoaded.patchConstSigs) {
            if (po.systemValueType != 0) continue;
            std::string pn = toLowerAscii(po.semanticName);
            for (const auto& pi : psLoaded.inputSigs) {
                if (pi.systemValueType != 0) continue;
                if (toLowerAscii(pi.semanticName) == pn && pi.semanticIndex == po.semanticIndex) {
                    msPrimToPsMap[po.sigId] = {static_cast<int32_t>(pi.startRow), pi.startCol};
                    for (uint32_t r = 0; r < po.rows; ++r) {
                        for (uint32_t c = 0; c < po.cols && (pi.startCol + c) < 4; ++c) {
                            uint32_t slot = (pi.startRow + r) * 4 + (pi.startCol + c);
                            if (slot < 64) msPrimAttrMask |= (1ull << slot);
                        }
                    }
                    break;
                }
            }
        }
    }

    CanonicalVSLayoutInfo canonicalVS{};
    int32_t posOutputReg = 0;
    uint32_t numVSOutputRegs = 0;
    uint32_t clipDistMask = 0;
    uint32_t cullDistMask = 0;
    uint32_t clipAttrOffset = 0;
    uint32_t cullAttrOffset = 0;
    bool hasViewportArrayIndex = false;
    uint32_t vpIndexAttrOffset = 0;
    bool hasPerPrimShadingRate = false;
    uint32_t shadingRateAttrOffset = 0;
    uint32_t totalVSScalarAttrs = 0;
    std::unordered_map<uint32_t, uint32_t> clipSigBaseIdx;
    std::unordered_map<uint32_t, uint32_t> cullSigBaseIdx;

    if (hasMS) {
        for (const auto& vo : msLoaded.outputSigs) {
            if (vo.startRow == UINT32_MAX) continue;
            numVSOutputRegs = std::max(numVSOutputRegs, vo.startRow + vo.rows);
            if (vo.systemValueType == 3) posOutputReg = static_cast<int32_t>(vo.startRow);
        }
    } else {
        canonicalVS = computeCanonicalVSLayout(vsOutputSigsRef);
        posOutputReg = canonicalVS.posOutputReg;
        numVSOutputRegs = canonicalVS.numVSOutputRegs;
        clipDistMask = canonicalVS.clipDistMask;
        cullDistMask = canonicalVS.cullDistMask;
        clipAttrOffset = canonicalVS.clipAttrOffset;
        cullAttrOffset = canonicalVS.cullAttrOffset;
        hasViewportArrayIndex = canonicalVS.hasViewportArrayIndex;
        vpIndexAttrOffset = canonicalVS.vpIndexAttrOffset;
        hasPerPrimShadingRate = canonicalVS.hasPerPrimShadingRate;
        shadingRateAttrOffset = canonicalVS.shadingRateAttrOffset;
        totalVSScalarAttrs = canonicalVS.totalVSScalarAttrs;
        clipSigBaseIdx = canonicalVS.clipSigBaseIdx;
        cullSigBaseIdx = canonicalVS.cullSigBaseIdx;

        if (!hasTess && !hasGS && skipVS && precompiledVSMeta) {
            posOutputReg = precompiledVSMeta->posOutputReg;
            numVSOutputRegs = precompiledVSMeta->numVSOutputRegs;
            clipDistMask = precompiledVSMeta->clipDistMask;
            cullDistMask = precompiledVSMeta->cullDistMask;
            clipAttrOffset = precompiledVSMeta->clipAttrOffset;
            cullAttrOffset = precompiledVSMeta->cullAttrOffset;
            hasViewportArrayIndex = precompiledVSMeta->hasViewportArrayIndex != 0;
            vpIndexAttrOffset = precompiledVSMeta->vpIndexAttrOffset;
            hasPerPrimShadingRate = precompiledVSMeta->hasPerPrimShadingRate != 0;
            shadingRateAttrOffset = precompiledVSMeta->shadingRateAttrOffset;
            totalVSScalarAttrs = precompiledVSMeta->totalVSScalarAttrs;
        }
    }

    uint32_t totalPostDSVertexAttrs = 0;
    uint32_t totalPostGSVertexAttrs = 0;
    if (hasMS || hasTess || hasGS) {
        const auto& preRastOutputs = hasMS ? msLoaded.outputSigs
                                           : (hasGS ? gsLoaded.outputSigs : dsLoaded.outputSigs);
        uint32_t nextExtraAttr = numPSInputRegs * 4;
        clipDistMask = 0;
        cullDistMask = 0;
        clipAttrOffset = nextExtraAttr;
        uint32_t clipScalarCount = 0;
        cullAttrOffset = 0;
        uint32_t cullScalarCount = 0;
        clipSigBaseIdx.clear();
        cullSigBaseIdx.clear();
        for (const auto& vo : preRastOutputs) {
            if (vo.systemValueType == 6) {
                clipSigBaseIdx[vo.sigId] = clipScalarCount;
                uint32_t count = std::min<uint32_t>(8u - clipScalarCount, vo.rows * vo.cols);
                for (uint32_t k = 0; k < count; ++k) {
                    clipDistMask |= (1u << (clipScalarCount + k));
                }
                clipScalarCount += count;
            }
        }
        nextExtraAttr += clipScalarCount;
        cullAttrOffset = nextExtraAttr;
        for (const auto& vo : preRastOutputs) {
            if (vo.systemValueType == 7) {
                cullSigBaseIdx[vo.sigId] = cullScalarCount;
                uint32_t count = std::min<uint32_t>(8u - cullScalarCount, vo.rows * vo.cols);
                for (uint32_t k = 0; k < count; ++k) {
                    cullDistMask |= (1u << (cullScalarCount + k));
                }
                cullScalarCount += count;
            }
        }
        nextExtraAttr += cullScalarCount;
        hasViewportArrayIndex = false;
        vpIndexAttrOffset = 0;
        hasPerPrimShadingRate = false;
        shadingRateAttrOffset = 0;
        for (const auto& vo : preRastOutputs) {
            if (vo.systemValueType == 5 && !hasViewportArrayIndex) {
                hasViewportArrayIndex = true;
                vpIndexAttrOffset = nextExtraAttr++;
            } else if (vo.systemValueType == 29 && !hasPerPrimShadingRate) {
                hasPerPrimShadingRate = true;
                shadingRateAttrOffset = nextExtraAttr++;
            }
        }
        if (hasMS) {
            for (const auto& po : msLoaded.patchConstSigs) {
                if (po.systemValueType == 5) {
                    if (!hasViewportArrayIndex) {
                        hasViewportArrayIndex = true;
                        vpIndexAttrOffset = nextExtraAttr++;
                    }
                    if (vpIndexAttrOffset < 64) msPrimAttrMask |= (1ull << vpIndexAttrOffset);
                } else if (po.systemValueType == 29) {
                    if (!hasPerPrimShadingRate) {
                        hasPerPrimShadingRate = true;
                        shadingRateAttrOffset = nextExtraAttr++;
                    }
                    if (shadingRateAttrOffset < 64) msPrimAttrMask |= (1ull << shadingRateAttrOffset);
                }
            }
        }
        totalVSScalarAttrs = hasMS ? nextExtraAttr : (hasTess ? numHSInputScalars : (numGSInputRegs * 4));
        totalPostDSVertexAttrs = hasGS ? (numGSInputRegs * 4) : nextExtraAttr;
        totalPostGSVertexAttrs = nextExtraAttr;
    }

    std::array<int16_t, 128> psInputFromVSSlot;
    psInputFromVSSlot.fill(-1);
    if (!hasMS && !hasTess && !hasGS) {
        if (hasPS) {
            for (const auto& pi : psLoaded.inputSigs) {
                if (pi.systemValueType != 0) continue;
                std::string pn = toLowerAscii(pi.semanticName);
                for (const auto& vo : vsOutputSigsRef) {
                    if (vo.systemValueType != 0 || vo.startRow >= 32) continue;
                    if (toLowerAscii(vo.semanticName) == pn && vo.semanticIndex == pi.semanticIndex) {
                        uint32_t commonRows = std::min(pi.rows, vo.rows);
                        uint32_t commonCols = std::min(pi.cols, vo.cols);
                        for (uint32_t r = 0; r < commonRows; ++r) {
                            for (uint32_t c = 0; c < commonCols; ++c) {
                                if ((pi.startCol + c) >= 4 || (vo.startCol + c) >= 4) continue;
                                uint32_t psSlot = (pi.startRow + r) * 4 + pi.startCol + c;
                                uint32_t vsSlot = (vo.startRow + r) * 4 + vo.startCol + c;
                                if (psSlot < 128) {
                                    psInputFromVSSlot[psSlot] = static_cast<int16_t>(vsSlot);
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    } else {
        uint32_t limit = std::min<uint32_t>(128u, numPSInputRegs * 4);
        for (uint32_t s = 0; s < limit; ++s) {
            psInputFromVSSlot[s] = static_cast<int16_t>(s);
        }
    }

    // --- 1. Emit @<vsSym> unless skipped by VS stage cache hit or Mesh Shader PSO ---
    if (!hasMS && !skipVS) {
        if (!emitVertexStageFunction(
                combinedMod, vsTree, vsLoaded, vsEffectiveLayout, desc.InputLayout,
                simdWidth, vsSym, hasTess, hasGS,
                (!hasTess && !hasGS) ? &canonicalVS : nullptr,
                (hasTess || hasGS) ? &vsToNextMap : nullptr,
                outError)) {
            return false;
        }
    }

    // --- 1-mesh. Emit @<vsSym>_as (optional) and @<vsSym>_ms ---
    if (hasMS) {
        auto extractLaneI32 = [](llvm::IRBuilder<>& lb, const SimdVal& sv, uint32_t lane) -> llvm::Value* {
            llvm::Type* i32 = lb.getInt32Ty();
            llvm::Value* v = (sv.kind == SimdVal::Kind::Vector)
                                ? llvmExtract(lb, sv.vec, static_cast<uint32_t>(static_cast<int64_t>(lane)))
                                : (sv.isUniform() ? sv.lanes[0] : sv.lanes[lane]);
            if (v->getType() != i32) {
                if (v->getType()->isFloatingPointTy()) {
                    if (v->getType()->getScalarSizeInBits() != 32) {
                        v = lb.CreateFPExt(v, lb.getFloatTy());
                    }
                    v = lb.CreateBitCast(v, i32);
                } else if (v->getType()->getScalarSizeInBits() < 32) {
                    v = lb.CreateZExt(v, i32);
                } else {
                    v = lb.CreateTrunc(v, i32);
                }
            }
            return v;
        };

        auto convertSimdValToF32Vec = [&](llvm::IRBuilder<>& lb, const SimdVal& val) -> llvm::Value* {
            IRHelpers lh{lb, simdWidth};
            llvm::Value* vVec = val.vec;
            llvm::Type* elemTy = llvm::cast<llvm::FixedVectorType>(vVec->getType())->getElementType();
            if (elemTy != lh.f32Ty()) {
                if (elemTy->isIntegerTy()) {
                    if (elemTy->getScalarSizeInBits() < 32) {
                        vVec = lb.CreateZExt(vVec, lh.vecTy(lh.i32Ty()));
                    }
                    vVec = lb.CreateBitCast(vVec, lh.vecTy(lh.f32Ty()));
                } else {
                    vVec = lb.CreateFPExt(vVec, lh.vecTy(lh.f32Ty()));
                }
            }
            return vVec;
        };

        if (hasAS) {
            if (!asTree.root) {
                outError = "Amplification shader entry function missing after structured CFG lift";
                return false;
            }

            llvm::IRBuilder<> b(combinedMod.getContext());
            IRHelpers h{b, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy}, false);
            auto* asFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym + "_as", combinedMod);
            llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", asFn);
            b.SetInsertPoint(entry);

            llvm::Value* argsPtr = entry->getParent()->getArg(0);
            llvm::Value* rootCtx = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, rootCtx)));
            llvm::Value* gsBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, groupSharedScratch)));
            llvm::Value* gx = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, gx)));
            llvm::Value* gy = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, gy)));
            llvm::Value* gz = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, gz)));
            llvm::Value* outDispatchDims = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, outDispatchDims)));
            llvm::Value* outPayload = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, outPayload)));

            StageHooks asHooks;
            asHooks.dispatchMesh = [&](llvm::IRBuilder<>& db,
                                       const SimdVal& countX, const SimdVal& countY, const SimdVal& countZ,
                                       const SimdVal& payloadArg, llvm::Value* realMask) -> bool {
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    llvm::Value* actL = llvmExtract(db, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    emitIf(db, actL, [&](llvm::IRBuilder<>& lb) {
                        IRHelpers lh{lb, simdWidth};
                        llvm::Value* already = lh.loadScalar(i32Ty, lh.gepBytes(outDispatchDims, 12));
                        llvm::Value* notYet = lb.CreateICmp(llvm::CmpInst::ICMP_EQ, already, lh.cInt(i32Ty, 0));
                        emitIf(lb, notYet, [&](llvm::IRBuilder<>& cb) {
                            IRHelpers ch{cb, simdWidth};
                            ch.storeScalar(ch.cInt(i32Ty, 1), ch.gepBytes(outDispatchDims, 12));
                            ch.storeScalar(extractLaneI32(cb, countX, l), ch.gepBytes(outDispatchDims, 0));
                            ch.storeScalar(extractLaneI32(cb, countY, l), ch.gepBytes(outDispatchDims, 4));
                            ch.storeScalar(extractLaneI32(cb, countZ, l), ch.gepBytes(outDispatchDims, 8));
                            if (asLoaded.asPayloadBytes > 0 && payloadArg.kind == SimdVal::Kind::Lanes && !payloadArg.lanes.empty()) {
                                llvm::Value* srcPtr = payloadArg.isUniform() ? payloadArg.lanes[0] : payloadArg.lanes[l];
                                llvm::Value* nullP = llvm::Constant::getNullValue(cb.getPtrTy());
                                llvm::Value* srcOk = cb.CreateICmp(llvm::CmpInst::ICMP_NE, srcPtr, nullP);
                                llvm::Value* dstOk = cb.CreateICmp(llvm::CmpInst::ICMP_NE, outPayload, nullP);
                                emitIf(cb, cb.CreateAnd(srcOk, dstOk),
                                                        [&](llvm::IRBuilder<>& mb) {
                                                            llvm::Type* i64T = mb.getInt64Ty();
                                                            llvm::Value* c0 = llvm::ConstantInt::get(i64T, 0);
                                                            llvm::Value* c1 = llvm::ConstantInt::get(i64T, 1);
                                                            llvm::Value* cN = llvm::ConstantInt::get(i64T, asLoaded.asPayloadBytes);
                                                            emitForLoop(mb, c0, cN, c1, [&](llvm::IRBuilder<>& fb, llvm::Value* bi) {
                                                                    llvm::Value* bi64 = bi;
                                                                    llvm::Value* sp = fb.CreateGEP(fb.getInt8Ty(), srcPtr, llvm::ArrayRef<llvm::Value*>{bi64});
                                                                    llvm::Value* dp = fb.CreateGEP(fb.getInt8Ty(), outPayload, llvm::ArrayRef<llvm::Value*>{bi64});
                                                                    llvm::Value* bv = fb.CreateLoad(fb.getInt8Ty(), sp);
                                                                    fb.CreateStore(bv, dp);
                                                                });
                                                        });
                            }
                        });
                    });
                }
                return true;
            };

            if (!packetizeThreadGroupRegion(b, combinedMod, *asTree.root, asLoaded, asEffectiveLayout,
                                            simdWidth, rootCtx, gsBase, gx, gy, gz, entry, asHooks, outError)) {
                return false;
            }
            b.CreateRetVoid();
        }

        // Emit @<vsSym>_ms(const MeshThreadGroupArgs* args)
        {
            if (!msTree.root) {
                outError = "Mesh shader entry function missing after MLIR SCF lift";
                return false;
            }

            llvm::IRBuilder<> b(combinedMod.getContext());
            IRHelpers h{b, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto f32Ty = h.f32Ty();
            auto i8Ty = h.i8Ty();
            auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy}, false);
            auto* msFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym + "_ms", combinedMod);
            llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", msFn);
            b.SetInsertPoint(entry);

            llvm::Value* argsPtr = entry->getParent()->getArg(0);
            llvm::Value* rootCtx = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, rootCtx)));
            llvm::Value* gsBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, groupSharedScratch)));
            llvm::Value* payloadPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, payloadPtr)));
            llvm::Value* gx = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, gx)));
            llvm::Value* gy = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, gy)));
            llvm::Value* gz = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, gz)));
            llvm::Value* outCounts = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outCounts)));
            llvm::Value* outVertPosXYZW = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outVertPosXYZW)));
            llvm::Value* outVertAttrs = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outVertAttrs)));
            llvm::Value* outPrimIndices = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimIndices)));
            llvm::Value* outPrimAttrs = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimAttrs)));
            llvm::Value* outPrimCull = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimCull)));
            llvm::Value* outPrimId = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimId)));

            llvm::Value* countsSetPtr = createAlignedAlloca(b, i32Ty, h.cInt(i32Ty, 1), 4);
            h.storeScalar(h.cInt(i32Ty, 0), countsSetPtr);

            const uint32_t maxVerts = msLoaded.msMaxVertexCount;
            const uint32_t maxPrims = msLoaded.msMaxPrimitiveCount;

            StageHooks msHooks;
            msHooks.meshPayloadBase = payloadPtr;

            msHooks.setMeshOutputCounts = [&](llvm::IRBuilder<>& sb,
                                              const SimdVal& numVerts, const SimdVal& numPrims,
                                              llvm::Value* realMask) -> bool {
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    emitIf(sb, actL, [&](llvm::IRBuilder<>& lb) {
                        IRHelpers lh{lb, simdWidth};
                        llvm::Value* already = lh.loadScalar(i32Ty, countsSetPtr);
                        llvm::Value* notYet = lb.CreateICmp(llvm::CmpInst::ICMP_EQ, already, lh.cInt(i32Ty, 0));
                        emitIf(lb, notYet, [&](llvm::IRBuilder<>& cb) {
                            IRHelpers ch{cb, simdWidth};
                            ch.storeScalar(ch.cInt(i32Ty, 1), countsSetPtr);
                            ch.storeScalar(extractLaneI32(cb, numVerts, l), ch.gepBytes(outCounts, 0));
                            ch.storeScalar(extractLaneI32(cb, numPrims, l), ch.gepBytes(outCounts, 4));
                        });
                    });
                }
                return true;
            };

            msHooks.emitIndices = [&](llvm::IRBuilder<>& sb,
                                      const SimdVal& primIdx, const SimdVal& v0,
                                      const SimdVal& v1, const SimdVal& v2,
                                      llvm::Value* realMask) -> bool {
                if (maxPrims == 0) return true;
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    llvm::Value* pIdx = extractLaneI32(sb, primIdx, l);
                    llvm::Value* inB = sb.CreateICmp(llvm::CmpInst::ICMP_ULT, pIdx, llvm::ConstantInt::get(i32Ty, maxPrims));
                    emitIf(sb, sb.CreateAnd(actL, inB),
                                            [&](llvm::IRBuilder<>& lb) {
                                                IRHelpers lh{lb, simdWidth};
                                                llvm::Value* base3 = lb.CreateMul(pIdx, lh.cInt(i32Ty, 3));
                                                lh.storeScalar(extractLaneI32(lb, v0, l), lh.gepArray(outPrimIndices, i32Ty, base3));
                                                lh.storeScalar(extractLaneI32(lb, v1, l),
                                                               lh.gepArray(outPrimIndices, i32Ty, lb.CreateAdd(base3, lh.cInt(i32Ty, 1))));
                                                lh.storeScalar(extractLaneI32(lb, v2, l),
                                                               lh.gepArray(outPrimIndices, i32Ty, lb.CreateAdd(base3, lh.cInt(i32Ty, 2))));
                                            });
                }
                return true;
            };

            msHooks.storeVertexOutput = [&](llvm::IRBuilder<>& sb, uint32_t sigId,
                                            const SimdVal& row, uint32_t col, const SimdVal& val,
                                            const SimdVal& vertIdx, llvm::Value* realMask) -> bool {
                const DxilSigRecord* sig = msLoaded.findOutputSig(sigId);
                if (!sig || val.kind != SimdVal::Kind::Vector || maxVerts == 0) return true;
                llvm::Value* vVec = convertSimdValToF32Vec(sb, val);

                if (sig->systemValueType == 3) { // SV_Position
                    if (col < 4) {
                        for (uint32_t l = 0; l < simdWidth; ++l) {
                            llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                            llvm::Value* vIdx = extractLaneI32(sb, vertIdx, l);
                            llvm::Value* inB = sb.CreateICmp(llvm::CmpInst::ICMP_ULT, vIdx, llvm::ConstantInt::get(i32Ty, maxVerts));
                            emitIf(sb, sb.CreateAnd(actL, inB),
                                                    [&](llvm::IRBuilder<>& lb) {
                                                        IRHelpers lh{lb, simdWidth};
                                                        llvm::Value* elemIdx = lb.CreateAdd(lb.CreateMul(vIdx, lh.cInt(i32Ty, 4)), lh.cInt(i32Ty, static_cast<int64_t>(col)));
                                                        llvm::Value* vL = llvmExtract(lb, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                        lh.storeScalar(vL, lh.gepArray(outVertPosXYZW, f32Ty, elemIdx));
                                                    });
                        }
                    }
                    return true;
                }

                bool hasTarget = false;
                uint32_t baseSlot = 0;
                bool useDynRow = false;
                if (sig->systemValueType == 6) {
                    auto cit = clipSigBaseIdx.find(sigId);
                    if (cit != clipSigBaseIdx.end()) {
                        uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                        uint32_t idx = cit->second + r * sig->cols + col;
                        if (idx < 8 && (clipDistMask & (1u << idx))) {
                            hasTarget = true;
                            baseSlot = clipAttrOffset + idx;
                        }
                    }
                } else if (sig->systemValueType == 7) {
                    auto cit = cullSigBaseIdx.find(sigId);
                    if (cit != cullSigBaseIdx.end()) {
                        uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                        uint32_t idx = cit->second + r * sig->cols + col;
                        if (idx < 8 && (cullDistMask & (1u << idx))) {
                            hasTarget = true;
                            baseSlot = cullAttrOffset + idx;
                        }
                    }
                } else if (sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                    hasTarget = true;
                    baseSlot = vpIndexAttrOffset;
                } else if (sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                    hasTarget = true;
                    baseSlot = shadingRateAttrOffset;
                } else {
                    auto it = vsToPsMap.find(sigId);
                    if (it != vsToPsMap.end() && it->second.psTargetRow >= 0) {
                        hasTarget = true;
                        baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                        useDynRow = true;
                    }
                }

                if (hasTarget && totalVSScalarAttrs > 0) {
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* vIdx = extractLaneI32(sb, vertIdx, l);
                        llvm::Value* inB = sb.CreateICmp(llvm::CmpInst::ICMP_ULT, vIdx, llvm::ConstantInt::get(i32Ty, maxVerts));
                        emitIf(sb, sb.CreateAnd(actL, inB),
                                                [&](llvm::IRBuilder<>& lb) {
                                                    IRHelpers lh{lb, simdWidth};
                                                    llvm::Value* slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot));
                                                    if (useDynRow) {
                                                        if (auto rc = tryGetUniformConstUInt(row)) {
                                                            if (*rc != 0) {
                                                                slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot + (*rc) * 4));
                                                            }
                                                        } else {
                                                            llvm::Value* r32 = extractLaneI32(lb, row, l);
                                                            slot32 = lb.CreateAdd(slot32, lb.CreateMul(r32, lh.cInt(i32Ty, 4)));
                                                        }
                                                    }
                                                    llvm::Value* elemIdx = lb.CreateAdd(lb.CreateMul(vIdx, lh.cInt(i32Ty, totalVSScalarAttrs)), slot32);
                                                    llvm::Value* vL = llvmExtract(lb, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                    lh.storeScalar(vL, lh.gepArray(outVertAttrs, f32Ty, elemIdx));
                                                });
                    }
                }
                return true;
            };

            msHooks.storePrimitiveOutput = [&](llvm::IRBuilder<>& sb, uint32_t sigId,
                                               const SimdVal& row, uint32_t col, const SimdVal& val,
                                               const SimdVal& primIdx, llvm::Value* realMask) -> bool {
                const DxilSigRecord* sig = msLoaded.findPatchConstSig(sigId);
                if (!sig || val.kind != SimdVal::Kind::Vector || maxPrims == 0) return true;

                if (sig->systemValueType == 30) { // SV_CullPrimitive
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* pIdx = extractLaneI32(sb, primIdx, l);
                        llvm::Value* inB = sb.CreateICmp(llvm::CmpInst::ICMP_ULT, pIdx, llvm::ConstantInt::get(i32Ty, maxPrims));
                        emitIf(sb, sb.CreateAnd(actL, inB),
                                                [&](llvm::IRBuilder<>& lb) {
                                                    IRHelpers lh{lb, simdWidth};
                                                    llvm::Value* vL = llvmExtract(lb, val.vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                    llvm::Value* nz;
                                                    if (vL->getType()->isIntegerTy()) {
                                                        nz = lb.CreateICmp(llvm::CmpInst::ICMP_NE, vL, lh.cInt(vL->getType(), 0));
                                                    } else {
                                                        nz = lb.CreateFCmp(llvm::CmpInst::FCMP_ONE, vL, lh.cFlt(0.0));
                                                    }
                                                    llvm::Value* cByte = lb.CreateSelect(nz, lh.cInt(i8Ty, 1), lh.cInt(i8Ty, 0));
                                                    lh.storeScalar(cByte, lh.gepArray(outPrimCull, i8Ty, pIdx));
                                                });
                    }
                    return true;
                }

                if (sig->systemValueType == 10) { // SV_PrimitiveID
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* pIdx = extractLaneI32(sb, primIdx, l);
                        llvm::Value* inB = sb.CreateICmp(llvm::CmpInst::ICMP_ULT, pIdx, llvm::ConstantInt::get(i32Ty, maxPrims));
                        emitIf(sb, sb.CreateAnd(actL, inB),
                                                [&](llvm::IRBuilder<>& lb) {
                                                    IRHelpers lh{lb, simdWidth};
                                                    lh.storeScalar(extractLaneI32(lb, val, l), lh.gepArray(outPrimId, i32Ty, pIdx));
                                                });
                    }
                    return true;
                }

                llvm::Value* vVec = convertSimdValToF32Vec(sb, val);
                bool hasTarget = false;
                uint32_t baseSlot = 0;
                bool useDynRow = false;
                if (sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                    hasTarget = true;
                    baseSlot = vpIndexAttrOffset;
                } else if (sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                    hasTarget = true;
                    baseSlot = shadingRateAttrOffset;
                } else {
                    auto it = msPrimToPsMap.find(sigId);
                    if (it != msPrimToPsMap.end() && it->second.psTargetRow >= 0) {
                        hasTarget = true;
                        baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                        useDynRow = true;
                    }
                }

                if (hasTarget && totalVSScalarAttrs > 0) {
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        llvm::Value* actL = llvmExtract(sb, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* pIdx = extractLaneI32(sb, primIdx, l);
                        llvm::Value* inB = sb.CreateICmp(llvm::CmpInst::ICMP_ULT, pIdx, llvm::ConstantInt::get(i32Ty, maxPrims));
                        emitIf(sb, sb.CreateAnd(actL, inB),
                                                [&](llvm::IRBuilder<>& lb) {
                                                    IRHelpers lh{lb, simdWidth};
                                                    llvm::Value* slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot));
                                                    if (useDynRow) {
                                                        if (auto rc = tryGetUniformConstUInt(row)) {
                                                            if (*rc != 0) {
                                                                slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot + (*rc) * 4));
                                                            }
                                                        } else {
                                                            llvm::Value* r32 = extractLaneI32(lb, row, l);
                                                            slot32 = lb.CreateAdd(slot32, lb.CreateMul(r32, lh.cInt(i32Ty, 4)));
                                                        }
                                                    }
                                                    llvm::Value* elemIdx = lb.CreateAdd(lb.CreateMul(pIdx, lh.cInt(i32Ty, totalVSScalarAttrs)), slot32);
                                                    llvm::Value* vL = llvmExtract(lb, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                    lh.storeScalar(vL, lh.gepArray(outPrimAttrs, f32Ty, elemIdx));
                                                });
                    }
                }
                return true;
            };

            if (!packetizeThreadGroupRegion(b, combinedMod, *msTree.root, msLoaded, msEffectiveLayout,
                                            simdWidth, rootCtx, gsBase, gx, gy, gz, entry, msHooks, outError)) {
                return false;
            }
            b.CreateRetVoid();
        }
    }

    // --- 1a-tess. Emit @<vsSym>_hs, @<vsSym>_hspc, and @<vsSym>_ds ---
    if (hasTess) {
        auto computeScalarPtr = [](llvm::IRBuilder<>& lb, uint32_t simdW,
                                   llvm::Value* basePtr, uint32_t cpStride,
                                   const SimdVal* vAxis, const SimdVal& row, uint32_t baseSlot) -> llvm::Value* {
            IRHelpers lh{lb, simdW};
            auto i32Ty = lh.i32Ty();
            auto i64Ty = lh.i64Ty();

            llvm::Value* idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot));
            if (vAxis && cpStride > 0) {
                if (auto vc = tryGetUniformConstUInt(*vAxis)) {
                    idx32 = lh.cInt(i32Ty, static_cast<int64_t>((*vc) * cpStride + baseSlot));
                } else {
                    llvm::Value* v0 = llvmExtract(lb, vAxis->vec, static_cast<uint32_t>(0));
                    if (v0->getType() != i32Ty) v0 = lb.CreateZExt(v0, i32Ty);
                    idx32 = lb.CreateAdd(idx32, lb.CreateMul(v0, lh.cInt(i32Ty, cpStride)));
                }
            }
            if (auto rc = tryGetUniformConstUInt(row)) {
                if (*rc != 0) {
                    idx32 = lb.CreateAdd(idx32, lh.cInt(i32Ty, static_cast<int64_t>((*rc) * 4)));
                }
            } else {
                llvm::Value* r0 = llvmExtract(lb, row.vec, static_cast<uint32_t>(0));
                if (r0->getType() != i32Ty) r0 = lb.CreateZExt(r0, i32Ty);
                idx32 = lb.CreateAdd(idx32, lb.CreateMul(r0, lh.cInt(i32Ty, 4)));
            }
            llvm::Value* byteOff = lb.CreateMul(lb.CreateZExt(idx32, i64Ty), lh.cInt(i64Ty, 4));
            return lh.gepBytesDyn(basePtr, byteOff);
        };

        auto convertLoadedF32ToSimd = [](llvm::IRBuilder<>& lb, uint32_t simdW,
                                         llvm::Value* f32Scalar, llvm::Type* scalarTy) -> SimdVal {
            IRHelpers lh{lb, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            llvm::Value* outVal = f32Scalar;
            if (scalarTy != f32Ty) {
                if (scalarTy->isIntegerTy()) {
                    outVal = lb.CreateBitCast(outVal, i32Ty);
                    if (scalarTy != i32Ty) outVal = lb.CreateTrunc(outVal, scalarTy);
                } else if (scalarTy->isFloatingPointTy()) {
                    outVal = lb.CreateFPTrunc(outVal, scalarTy);
                }
            }
            return lh.splatScalar(outVal);
        };

        auto convertF32VecToSimd = [](llvm::IRBuilder<>& lb, uint32_t simdW,
                                      llvm::Value* fVec, llvm::Type* scalarTy) -> SimdVal {
            IRHelpers lh{lb, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            llvm::Value* outVec = fVec;
            if (scalarTy != f32Ty) {
                if (scalarTy->isIntegerTy()) {
                    outVec = lb.CreateBitCast(outVec, lh.vecTy(i32Ty));
                    if (scalarTy != i32Ty) {
                        outVec = lb.CreateTrunc(outVec, lh.vecTy(scalarTy));
                    }
                } else if (scalarTy->isFloatingPointTy()) {
                    outVec = lb.CreateFPTrunc(outVec, lh.vecTy(scalarTy));
                }
            }
            return {SimdVal::Kind::Vector, scalarTy, outVec, {}, {}};
        };

        auto toI32Vec = [](llvm::IRBuilder<>& lb, uint32_t simdW, const SimdVal& sv) -> llvm::Value* {
            IRHelpers lh{lb, simdW};
            auto i32Ty = lh.i32Ty();
            llvm::Type* et = llvm::cast<llvm::FixedVectorType>(sv.vec->getType())->getElementType();
            if (et == i32Ty) return sv.vec;
            if (et->getScalarSizeInBits() < 32) {
                return lb.CreateZExt(sv.vec, lh.vecTy(i32Ty));
            }
            return lb.CreateTrunc(sv.vec, lh.vecTy(i32Ty));
        };

        auto loadPatchInputSIMD = [&](llvm::IRBuilder<>& lb, uint32_t simdW,
                                      llvm::Value* basePtr, uint32_t cpStride,
                                      const SimdVal* vAxis, const SimdVal& row, uint32_t rowStride,
                                      uint32_t baseSlot, llvm::Type* scalarTy, llvm::Value* gatherMask) -> SimdVal {
            IRHelpers lh{lb, simdW};
            auto i32Ty = lh.i32Ty();
            auto i64Ty = lh.i64Ty();
            auto f32Ty = lh.f32Ty();

            std::optional<uint32_t> vc = vAxis ? tryGetUniformConstUInt(*vAxis) : std::optional<uint32_t>(0u);
            std::optional<uint32_t> rc = tryGetUniformConstUInt(row);
            if (vc.has_value() && rc.has_value()) {
                int64_t offBytes = static_cast<int64_t>((*vc) * cpStride + baseSlot + (*rc) * rowStride) * 4;
                llvm::Value* fVal = lh.loadScalar(f32Ty, lh.gepBytes(basePtr, offBytes));
                return convertLoadedF32ToSimd(lb, simdW, fVal, scalarTy);
            }

            llvm::Value* idxVec = lh.splatInt(i32Ty, static_cast<int64_t>(baseSlot));
            if (vAxis && cpStride > 0) {
                idxVec = lb.CreateAdd(idxVec, lb.CreateMul(toI32Vec(lb, simdW, *vAxis), lh.splatInt(i32Ty, cpStride)));
            }
            if (rowStride > 0) {
                idxVec = lb.CreateAdd(idxVec, lb.CreateMul(toI32Vec(lb, simdW, row), lh.splatInt(i32Ty, rowStride)));
            }
            llvm::Value* byteOffVec = lb.CreateMul(lb.CreateZExt(idxVec, lh.vecTy(i64Ty)), lh.splatInt(i64Ty, 4));
            llvm::Value* basePtrVec = llvmSplat(lb, lh.ptrTy(), basePtr, lh.W);
            llvm::Value* ptrVec = lb.CreateGEP(lh.i8Ty(), basePtrVec, byteOffVec);
            llvm::Value* fVec = lb.CreateMaskedGather(lh.vecTy(f32Ty), ptrVec, llvm::Align(1), gatherMask, lh.splatFlt(0.0));
            return convertF32VecToSimd(lb, simdW, fVec, scalarTy);
        };

        auto storeVecToSOA = [](llvm::IRBuilder<>& sb, uint32_t simdW,
                                const SimdVal& val, llvm::Value* writeMask, llvm::Value* activeMask,
                                llvm::Value* outBasePtr, uint32_t baseSlotWithCol, const SimdVal* dynRow) {
            if (val.kind != SimdVal::Kind::Vector) return;
            IRHelpers sh{sb, simdW};
            auto i32Ty = sh.i32Ty();
            auto f32Ty = sh.f32Ty();
            llvm::Value* vVec = val.vec;
            llvm::Type* elemTy = llvm::cast<llvm::FixedVectorType>(vVec->getType())->getElementType();
            if (elemTy != f32Ty) {
                if (elemTy->isIntegerTy()) {
                    if (elemTy->getScalarSizeInBits() < 32) {
                        vVec = sb.CreateZExt(vVec, sh.vecTy(i32Ty));
                    }
                    vVec = sb.CreateBitCast(vVec, sh.vecTy(f32Ty));
                } else {
                    vVec = sb.CreateFPExt(vVec, sh.vecTy(f32Ty));
                }
            }

            std::optional<uint32_t> rc = dynRow ? tryGetUniformConstUInt(*dynRow) : std::optional<uint32_t>(0u);
            if (rc.has_value()) {
                uint32_t scalarSlot = baseSlotWithCol + (*rc) * 4;
                llvm::Value* dstPtr = sh.gepBytes(outBasePtr, static_cast<int64_t>(scalarSlot * simdW * 4));
                if (!writeMask || writeMask == activeMask) {
                    sb.CreateStore(vVec, dstPtr);
                } else {
                    llvm::Value* prev = sh.loadScalar(sh.vecTy(f32Ty), dstPtr);
                    llvm::Value* merged = sb.CreateSelect(writeMask, vVec, prev);
                    sb.CreateStore(merged, dstPtr);
                }
            } else {
                llvm::Value* effMask = writeMask ? writeMask : activeMask;
                for (uint32_t l = 0; l < simdW; ++l) {
                    llvm::Value* mL = llvmExtract(sb, effMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    emitIf(sb, mL, [&](llvm::IRBuilder<>& nb) {
                        IRHelpers nh{nb, simdW};
                        llvm::Value* rL = llvmExtract(nb, dynRow->vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* r32 = (rL->getType() == i32Ty)
                                              ? rL
                                              : nb.CreateZExt(rL, i32Ty);
                        llvm::Value* slot32 = nb.CreateAdd(nh.cInt(i32Ty, static_cast<int64_t>(baseSlotWithCol)), nb.CreateMul(r32, nh.cInt(i32Ty, 4)));
                        llvm::Value* elemIdx = nb.CreateAdd(nb.CreateMul(slot32, nh.cInt(i32Ty, simdW)), nh.cInt(i32Ty, static_cast<int64_t>(l)));
                        llvm::Value* byteOff = nb.CreateMul(nb.CreateZExt(elemIdx, nh.i64Ty()), nh.cInt(nh.i64Ty(), 4));
                        llvm::Value* vL = llvmExtract(nb, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        nb.CreateStore(vL, nh.gepBytesDyn(outBasePtr, byteOff));
                    });
                }
            }
        };

        auto storeLane0ToF32Ptr = [](llvm::IRBuilder<>& lb, uint32_t simdW,
                                     const SimdVal& val, llvm::Value* realMask, llvm::Value* dstPtr) {
            IRHelpers lh{lb, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            llvm::Value* v0 = llvmExtract(lb, val.vec, static_cast<uint32_t>(0));
            if (v0->getType() != f32Ty) {
                if (v0->getType()->isIntegerTy()) {
                    if (v0->getType() != i32Ty) v0 = lb.CreateZExt(v0, i32Ty);
                    v0 = lb.CreateBitCast(v0, f32Ty);
                } else {
                    v0 = lb.CreateFPExt(v0, f32Ty);
                }
            }
            llvm::Value* l0Active = llvmExtract(lb, realMask, static_cast<uint32_t>(0));
            emitIf(lb, l0Active, [&](llvm::IRBuilder<>& sb) {
                sb.CreateStore(v0, dstPtr);
            });
        };

        // 1) @<vsSym>_hs(rootCtx, inCPAttrs, patchId, baseOutCpId, activeCount, outCPAttrsSOA)
        {
            if (!hsTree.root) {
                outError = "Hull shader control-point entry function missing after MLIR SCF lift";
                return false;
            }
            llvm::IRBuilder<> b(combinedMod.getContext());
            IRHelpers h{b, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, ptrTy}, false);
            auto* hsFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym + "_hs", combinedMod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", hsFn);
            b.SetInsertPoint(entry);

            llvm::Value* rootCtx = entry->getParent()->getArg(0);
            llvm::Value* inCPAttrs = entry->getParent()->getArg(1);
            llvm::Value* patchIdArg = entry->getParent()->getArg(2);
            llvm::Value* baseOutCpIdArg = entry->getParent()->getArg(3);
            llvm::Value* activeCountArg = entry->getParent()->getArg(4);
            llvm::Value* outCPAttrsSOA = entry->getParent()->getArg(5);

            llvm::Value* acVec = llvmSplat(b, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getElementType(), activeCountArg, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getNumElements());
            llvm::Value* activeMask = b.CreateICmp(llvm::CmpInst::ICMP_ULT, h.laneIota(), acVec);
            llvm::Value* outCpIdVec = b.CreateSelect(activeMask, b.CreateAdd(llvmSplat(b, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getElementType(), baseOutCpIdArg, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getNumElements()), h.laneIota()), h.splatInt(i32Ty, 0));

            StageHooks hsHooks;
            llvm::Value* hsRealMask = activeMask;
            hsHooks.loadInputWithVertex = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                              const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                              llvm::Type* scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = hsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                out = loadPatchInputSIMD(
                    lb, simdWidth, inCPAttrs, numHSInputScalars, &vAxis, row, 4u, baseSlot, scalarTy, activeMask);
                return true;
            };
            hsHooks.storeOutput = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                      const SimdVal& row, uint32_t col, const SimdVal& val,
                                      llvm::Value* realMask) -> bool {
                auto it = hsToDsMap.find(sigId);
                if (it == hsToDsMap.end() || it->second.psTargetRow < 0) return true;
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outCPAttrsSOA, baseSlot, &row);
                return true;
            };
            hsHooks.loadOutputControlPointId = [&](llvm::IRBuilder<>& lb, llvm::Type* scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, simdWidth};
                llvm::Value* v = (scalarTy == i32Ty)
                                    ? outCpIdVec
                                    : lb.CreateTrunc(outCpIdVec, lh.vecTy(scalarTy));
                out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                return true;
            };
            hsHooks.loadPrimitiveId = [&](llvm::IRBuilder<>& lb, llvm::Type* scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, simdWidth};
                llvm::Value* v = patchIdArg;
                if (scalarTy != i32Ty) v = lb.CreateTrunc(v, scalarTy);
                out = lh.splatScalar(v);
                return true;
            };

            if (!packetizeRegionWithHooks(b, combinedMod, *hsTree.root, hsLoaded, hsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, hsRealMask, hsHooks, outError)) {
                return false;
            }
            b.CreateRetVoid();
        }

        // 2) @<vsSym>_hspc(rootCtx, inCPAttrs, outCPAttrs, patchId, outPatchConsts, outTessFactors)
        {
            if (!hspcTree.root) {
                outError = "Hull shader patch-constant function missing after MLIR SCF lift";
                return false;
            }
            llvm::IRBuilder<> b(combinedMod.getContext());
            IRHelpers h{b, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto f32Ty = h.f32Ty();
            auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy}, false);
            auto* hspcFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym + "_hspc", combinedMod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", hspcFn);
            b.SetInsertPoint(entry);

            llvm::Value* rootCtx = entry->getParent()->getArg(0);
            llvm::Value* inCPAttrs = entry->getParent()->getArg(1);
            llvm::Value* outCPAttrs = entry->getParent()->getArg(2);
            llvm::Value* patchIdArg = entry->getParent()->getArg(3);
            llvm::Value* outPatchConsts = entry->getParent()->getArg(4);
            llvm::Value* outTessFactors = entry->getParent()->getArg(5);

            llvm::Value* activeMask = b.CreateICmp(llvm::CmpInst::ICMP_EQ, h.laneIota(), h.splatInt(i32Ty, 0));

            StageHooks hspcHooks;
            hspcHooks.loadInputWithVertex = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                                const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                                llvm::Type* scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = hsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                llvm::Value* ptr = computeScalarPtr(lb, simdWidth, inCPAttrs, numHSInputScalars, &vAxis, row, baseSlot);
                llvm::Value* fVal = IRHelpers{lb, simdWidth}.loadScalar(f32Ty, ptr);
                out = convertLoadedF32ToSimd(lb, simdWidth, fVal, scalarTy);
                return true;
            };
            hspcHooks.loadOutputControlPoint = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                                   const SimdVal& row, uint32_t col, const SimdVal& cpIdx,
                                                   llvm::Type* scalarTy, SimdVal& out) -> bool {
                auto it = hsToDsMap.find(sigId);
                uint32_t baseSlot = 0;
                if (it != hsToDsMap.end() && it->second.psTargetRow >= 0) {
                    baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                } else {
                    const DxilSigRecord* sig = hsLoaded.findOutputSig(sigId);
                    if (!sig) return false;
                    baseSlot = sig->startRow * 4 + sig->startCol + col;
                }
                llvm::Value* ptr = computeScalarPtr(lb, simdWidth, outCPAttrs, numHSOutputScalars, &cpIdx, row, baseSlot);
                llvm::Value* fVal = IRHelpers{lb, simdWidth}.loadScalar(f32Ty, ptr);
                out = convertLoadedF32ToSimd(lb, simdWidth, fVal, scalarTy);
                return true;
            };
            hspcHooks.storePatchConstant = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                               const SimdVal& row, uint32_t col, const SimdVal& val,
                                               llvm::Value* realMask) -> bool {
                const DxilSigRecord* sig = hsLoaded.findPatchConstSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 25 || sig->systemValueType == 26) {
                    // SV_TessFactor -> outTessFactors[0..3], SV_InsideTessFactor -> outTessFactors[4..5]
                    IRHelpers lh{lb, simdWidth};
                    uint32_t baseIdx = (sig->systemValueType == 25) ? 0u : 4u;
                    llvm::Value* idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseIdx));
                    if (auto rc = tryGetUniformConstUInt(row)) {
                        idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseIdx + *rc));
                    } else {
                        llvm::Value* r0 = llvmExtract(lb, row.vec, static_cast<uint32_t>(0));
                        if (r0->getType() != i32Ty) r0 = lb.CreateZExt(r0, i32Ty);
                        idx32 = lb.CreateAdd(idx32, r0);
                    }
                    llvm::Value* byteOff = lb.CreateMul(lb.CreateZExt(idx32, lh.i64Ty()), lh.cInt(lh.i64Ty(), 4));
                    storeLane0ToF32Ptr(lb, simdWidth, val, realMask, lh.gepBytesDyn(outTessFactors, byteOff));
                    return true;
                }
                auto it = hsPcToDsMap.find(sigId);
                if (it == hsPcToDsMap.end() || it->second.psTargetRow < 0) return true;
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                llvm::Value* ptr = computeScalarPtr(lb, simdWidth, outPatchConsts, 0, nullptr, row, baseSlot);
                storeLane0ToF32Ptr(lb, simdWidth, val, realMask, ptr);
                return true;
            };
            hspcHooks.loadPrimitiveId = [&](llvm::IRBuilder<>& lb, llvm::Type* scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, simdWidth};
                llvm::Value* v = patchIdArg;
                if (scalarTy != i32Ty) v = lb.CreateTrunc(v, scalarTy);
                out = lh.splatScalar(v);
                return true;
            };

            llvm::Value* hspcRealMask = activeMask;
            if (!packetizeRegionWithHooks(b, combinedMod, *hspcTree.root, hsLoaded, hsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, hspcRealMask, hspcHooks, outError)) {
                return false;
            }
            b.CreateRetVoid();
        }

        // 3) @<vsSym>_ds(rootCtx, inCPAttrs, inPatchConsts, inTessFactors, patchId, domUPtr, domVPtr, domWPtr, activeCount, outPosXYZW, outAttrs)
        {
            if (!dsTree.root) {
                outError = "Domain shader entry function missing after MLIR SCF lift";
                return false;
            }
            llvm::IRBuilder<> b(combinedMod.getContext());
            IRHelpers h{b, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto f32Ty = h.f32Ty();
            auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy}, false);
            auto* dsFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym + "_ds", combinedMod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", dsFn);
            b.SetInsertPoint(entry);

            llvm::Value* rootCtx = entry->getParent()->getArg(0);
            llvm::Value* inCPAttrs = entry->getParent()->getArg(1);
            llvm::Value* inPatchConsts = entry->getParent()->getArg(2);
            llvm::Value* inTessFactors = entry->getParent()->getArg(3);
            llvm::Value* patchIdArg = entry->getParent()->getArg(4);
            llvm::Value* domUPtr = entry->getParent()->getArg(5);
            llvm::Value* domVPtr = entry->getParent()->getArg(6);
            llvm::Value* domWPtr = entry->getParent()->getArg(7);
            llvm::Value* activeCountArg = entry->getParent()->getArg(8);
            llvm::Value* outPosXYZW = entry->getParent()->getArg(9);
            llvm::Value* outAttrs = entry->getParent()->getArg(10);

            llvm::Value* acVec = llvmSplat(b, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getElementType(), activeCountArg, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getNumElements());
            llvm::Value* activeMask = b.CreateICmp(llvm::CmpInst::ICMP_ULT, h.laneIota(), acVec);
            llvm::Value* zeroFltVec = h.splatFlt(0.0);
            llvm::Value* domUVec = b.CreateSelect(activeMask, h.loadScalar(h.vecTy(f32Ty), domUPtr), zeroFltVec);
            llvm::Value* domVVec = b.CreateSelect(activeMask, h.loadScalar(h.vecTy(f32Ty), domVPtr), zeroFltVec);
            llvm::Value* domWVec = b.CreateSelect(activeMask, h.loadScalar(h.vecTy(f32Ty), domWPtr), zeroFltVec);

            StageHooks dsHooks;
            dsHooks.loadInputWithVertex = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                              const SimdVal& row, uint32_t col, const SimdVal& cpIdx,
                                              llvm::Type* scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = dsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                out = loadPatchInputSIMD(
                    lb, simdWidth, inCPAttrs, numHSOutputScalars, &cpIdx, row, 4u, baseSlot, scalarTy, activeMask);
                return true;
            };
            dsHooks.loadPatchConstant = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                            const SimdVal& row, uint32_t col,
                                            llvm::Type* scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = dsLoaded.findPatchConstSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 25 || sig->systemValueType == 26) {
                    uint32_t baseIdx = (sig->systemValueType == 25) ? 0u : 4u;
                    out = loadPatchInputSIMD(
                        lb, simdWidth, inTessFactors, 0u, nullptr, row, 1u, baseIdx, scalarTy, activeMask);
                    return true;
                }
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                out = loadPatchInputSIMD(
                    lb, simdWidth, inPatchConsts, 0u, nullptr, row, 4u, baseSlot, scalarTy, activeMask);
                return true;
            };
            dsHooks.loadDomainLocation = [&](llvm::IRBuilder<>& lb, uint32_t comp,
                                             llvm::Type* scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, simdWidth};
                llvm::Value* v = (comp == 0) ? domUVec : ((comp == 1) ? domVVec : ((comp == 2) ? domWVec : lh.splatFlt(0.0)));
                if (scalarTy != f32Ty) {
                    v = lb.CreateFPTrunc(v, lh.vecTy(scalarTy));
                }
                out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                return true;
            };
            dsHooks.loadPrimitiveId = [&](llvm::IRBuilder<>& lb, llvm::Type* scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, simdWidth};
                llvm::Value* v = patchIdArg;
                if (scalarTy != i32Ty) v = lb.CreateTrunc(v, scalarTy);
                out = lh.splatScalar(v);
                return true;
            };
            dsHooks.storeOutput = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                      const SimdVal& row, uint32_t col, const SimdVal& val,
                                      llvm::Value* realMask) -> bool {
                const DxilSigRecord* sig = dsLoaded.findOutputSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 3) {
                    if (col < 4) {
                        storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outPosXYZW, col, nullptr);
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 6) {
                    auto cit = clipSigBaseIdx.find(sigId);
                    if (cit != clipSigBaseIdx.end()) {
                        uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                        uint32_t idx = cit->second + r * sig->cols + col;
                        if (idx < 8 && (clipDistMask & (1u << idx))) {
                            storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outAttrs, clipAttrOffset + idx, nullptr);
                        }
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 7) {
                    auto cit = cullSigBaseIdx.find(sigId);
                    if (cit != cullSigBaseIdx.end()) {
                        uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                        uint32_t idx = cit->second + r * sig->cols + col;
                        if (idx < 8 && (cullDistMask & (1u << idx))) {
                            storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outAttrs, cullAttrOffset + idx, nullptr);
                        }
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                    storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outAttrs, vpIndexAttrOffset, nullptr);
                    return true;
                }
                if (!hasGS && sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                    storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outAttrs, shadingRateAttrOffset, nullptr);
                    return true;
                }
                auto it = dsToNextMap.find(sigId);
                if (it != dsToNextMap.end() && it->second.psTargetRow >= 0) {
                    uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                    storeVecToSOA(lb, simdWidth, val, realMask, activeMask, outAttrs, baseSlot, &row);
                }
                return true;
            };

            llvm::Value* dsRealMask = activeMask;
            if (!packetizeRegionWithHooks(b, combinedMod, *dsTree.root, dsLoaded, dsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, dsRealMask, dsHooks, outError)) {
                return false;
            }
            b.CreateRetVoid();
        }
    }

    // --- 1b. Emit @<vsSym>_gs(rootCtx, inPosXYZW, inAttrs, primIdsPtr, gsInstId, activeCount, outPosXYZW, outAttrs, outStripCounts, outTotalVerts, outNumStrips) ---
    uint32_t numGSInputScalars = numGSInputRegs * 4;
    if (hasGS) {
        if (!gsTree.root) {
            outError = "Geometry shader entry function missing after MLIR SCF lift";
            return false;
        }

        llvm::IRBuilder<> b(combinedMod.getContext());
        IRHelpers h{b, simdWidth};
        auto ptrTy = h.ptrTy();
        auto i32Ty = h.i32Ty();
        auto f32Ty = h.f32Ty();
        auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy, ptrTy, ptrTy, ptrTy, i32Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy}, false);
        auto* gsFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, vsSym + "_gs", combinedMod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", gsFn);
        b.SetInsertPoint(entry);

        llvm::Value* rootCtx = entry->getParent()->getArg(0);
        llvm::Value* inPosXYZW = entry->getParent()->getArg(1);
        llvm::Value* inAttrs = entry->getParent()->getArg(2);
        llvm::Value* primIdsPtr = entry->getParent()->getArg(3);
        llvm::Value* gsInstIdArg = entry->getParent()->getArg(4);
        llvm::Value* activeCountArg = entry->getParent()->getArg(5);
        llvm::Value* outPosXYZW = entry->getParent()->getArg(6);
        llvm::Value* outAttrs = entry->getParent()->getArg(7);
        llvm::Value* outStripCounts = entry->getParent()->getArg(8);
        llvm::Value* outTotalVerts = entry->getParent()->getArg(9);
        llvm::Value* outNumStrips = entry->getParent()->getArg(10);

        llvm::Value* acVec = llvmSplat(b, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getElementType(), activeCountArg, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getNumElements());
        llvm::Value* activeMask = b.CreateICmp(llvm::CmpInst::ICMP_ULT, h.laneIota(), acVec);
        llvm::Value* primIdVec = b.CreateSelect(activeMask, h.loadScalar(h.vecTy(i32Ty), primIdsPtr), h.splatInt(i32Ty, 0));

        uint32_t maxVerts = std::max<uint32_t>(1, gsLoaded.gsMaxVertexCount);
        uint32_t outAttrScalars = totalPostGSVertexAttrs;
        uint32_t vecAlign = std::max<uint32_t>(64, simdWidth * 4);

        llvm::Value* stagingPos = createAlignedAlloca(b, h.vecTy(f32Ty), h.cInt(i32Ty, 4), vecAlign);
        llvm::Value* stagingAttrs = createAlignedAlloca(b, h.vecTy(f32Ty), h.cInt(i32Ty, std::max<uint32_t>(1, outAttrScalars)), vecAlign);
        for (uint32_t c = 0; c < 4; ++c) {
            h.storeScalar(h.splatFlt(0.0), h.gepBytes(stagingPos, static_cast<int64_t>(c * simdWidth * 4)));
        }
        for (uint32_t s = 0; s < outAttrScalars; ++s) {
            h.storeScalar(h.splatFlt(0.0), h.gepBytes(stagingAttrs, static_cast<int64_t>(s * simdWidth * 4)));
        }

        llvm::Value* emitCountPtr = createAlignedAlloca(b, h.vecTy(i32Ty), h.cInt(i32Ty, 1), vecAlign);
        llvm::Value* curStripPtr = createAlignedAlloca(b, h.vecTy(i32Ty), h.cInt(i32Ty, 1), vecAlign);
        llvm::Value* numStripsPtr = createAlignedAlloca(b, h.vecTy(i32Ty), h.cInt(i32Ty, 1), vecAlign);
        h.storeScalar(h.splatInt(i32Ty, 0), emitCountPtr);
        h.storeScalar(h.splatInt(i32Ty, 0), curStripPtr);
        h.storeScalar(h.splatInt(i32Ty, 0), numStripsPtr);

        StageHooks gsHooks;
        gsHooks.loadInputWithVertex = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                          const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                          llvm::Type* scalarTy, SimdVal& out) -> bool {
            const DxilSigRecord* sig = gsLoaded.findInputSig(sigId);
            if (!sig) return false;
            IRHelpers lh{lb, simdWidth};
            auto toI32V = [&](const SimdVal& sv) -> llvm::Value* {
                llvm::Type* et = llvm::cast<llvm::FixedVectorType>(sv.vec->getType())->getElementType();
                if (et == i32Ty) return sv.vec;
                if (et->getScalarSizeInBits() < 32) {
                    return lb.CreateZExt(sv.vec, lh.vecTy(i32Ty));
                }
                return lb.CreateTrunc(sv.vec, lh.vecTy(i32Ty));
            };

            auto vcOpt = tryGetUniformConstUInt(vAxis);
            auto rcOpt = tryGetUniformConstUInt(row);
            llvm::Value* vecVal;
            if (vcOpt.has_value() && rcOpt.has_value()) {
                uint32_t vIdx = *vcOpt;
                uint32_t r = *rcOpt;
                if (sig->systemValueType == 3) {
                    vecVal = lh.loadScalar(
                        lh.vecTy(f32Ty),
                        lh.gepBytes(inPosXYZW, static_cast<int64_t>((vIdx * 4 + col) * simdWidth * 4)));
                } else {
                    uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                    if (slot < numGSInputScalars) {
                        vecVal = lh.loadScalar(
                            lh.vecTy(f32Ty),
                            lh.gepBytes(inAttrs, static_cast<int64_t>((vIdx * numGSInputScalars + slot) * simdWidth * 4)));
                    } else {
                        vecVal = lh.splatFlt(0.0);
                    }
                }
            } else {
                llvm::Value* vVec32 = toI32V(vAxis);
                if (sig->systemValueType == 3) {
                    llvm::Value* planeIdx = lb.CreateAdd(lb.CreateMul(vVec32, lh.splatInt(i32Ty, 4)), lh.splatInt(i32Ty, col));
                    llvm::Value* elemIdx = lb.CreateAdd(lb.CreateMul(planeIdx, lh.splatInt(i32Ty, simdWidth)), lh.laneIota());
                    llvm::Value* byteOff = lb.CreateMul(lb.CreateZExt(elemIdx, lh.vecTy(lh.i64Ty())), lh.splatInt(lh.i64Ty(), 4));
                    llvm::Value* ptrVec = lb.CreateGEP(lh.i8Ty(), llvmSplat(lb, ptrTy, inPosXYZW, lh.W), byteOff);
                    vecVal = lb.CreateMaskedGather(lh.vecTy(f32Ty), ptrVec, llvm::Align(1), activeMask, lh.splatFlt(0.0));
                } else {
                    llvm::Value* rVec32 = toI32V(row);
                    llvm::Value* slotVec = lb.CreateAdd(lb.CreateMul(lb.CreateAdd(lh.splatInt(i32Ty, sig->startRow), rVec32), lh.splatInt(i32Ty, 4)), lh.splatInt(i32Ty, sig->startCol + col));
                    llvm::Value* inBounds = lb.CreateICmp(llvm::CmpInst::ICMP_ULT, slotVec, lh.splatInt(i32Ty, numGSInputScalars));
                    llvm::Value* gatherMask = lb.CreateAnd(activeMask, inBounds);
                    llvm::Value* planeIdx = lb.CreateAdd(lb.CreateMul(vVec32, lh.splatInt(i32Ty, numGSInputScalars)), slotVec);
                    llvm::Value* elemIdx = lb.CreateAdd(lb.CreateMul(planeIdx, lh.splatInt(i32Ty, simdWidth)), lh.laneIota());
                    llvm::Value* byteOff = lb.CreateMul(lb.CreateZExt(elemIdx, lh.vecTy(lh.i64Ty())), lh.splatInt(lh.i64Ty(), 4));
                    llvm::Value* ptrVec = lb.CreateGEP(lh.i8Ty(), llvmSplat(lb, ptrTy, inAttrs, lh.W), byteOff);
                    vecVal = lb.CreateMaskedGather(lh.vecTy(f32Ty), ptrVec, llvm::Align(1), gatherMask, lh.splatFlt(0.0));
                }
            }
            if (scalarTy != f32Ty) {
                if (scalarTy->isIntegerTy()) {
                    vecVal = lb.CreateBitCast(vecVal, lh.vecTy(i32Ty));
                    if (scalarTy != i32Ty) {
                        vecVal = lb.CreateTrunc(vecVal, lh.vecTy(scalarTy));
                    }
                } else {
                    vecVal = lb.CreateBitCast(vecVal, lh.vecTy(scalarTy));
                }
            }
            out = {SimdVal::Kind::Vector, scalarTy, vecVal, {}, {}};
            return true;
        };

        gsHooks.loadPrimitiveId = [&](llvm::IRBuilder<>& lb, llvm::Type* scalarTy, SimdVal& out) -> bool {
            IRHelpers lh{lb, simdWidth};
            llvm::Value* v = (scalarTy == i32Ty)
                                ? primIdVec
                                : lb.CreateTrunc(primIdVec, lh.vecTy(scalarTy));
            out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
            return true;
        };

        gsHooks.loadGSInstanceId = [&](llvm::IRBuilder<>& lb, llvm::Type* scalarTy, SimdVal& out) -> bool {
            IRHelpers lh{lb, simdWidth};
            llvm::Value* v = gsInstIdArg;
            if (scalarTy != i32Ty) v = lb.CreateTrunc(v, scalarTy);
            out = {SimdVal::Kind::Vector, scalarTy, llvmSplat(lb, llvm::cast<llvm::FixedVectorType>(lh.vecTy(scalarTy))->getElementType(), v, llvm::cast<llvm::FixedVectorType>(lh.vecTy(scalarTy))->getNumElements()), {}, {}};
            return true;
        };

        gsHooks.storeOutput = [&](llvm::IRBuilder<>& sb, uint32_t sigId,
                                  const SimdVal& row, uint32_t col, const SimdVal& val, llvm::Value* writeMask) -> bool {
            const DxilSigRecord* sig = gsLoaded.findOutputSig(sigId);
            if (!sig || val.kind != SimdVal::Kind::Vector) return true;
            IRHelpers sh{sb, simdWidth};
            llvm::Value* vF32 = val.vec;
            llvm::Type* elemTy = llvm::cast<llvm::FixedVectorType>(vF32->getType())->getElementType();
            if (elemTy != f32Ty) {
                if (elemTy->isIntegerTy()) {
                    if (elemTy->getScalarSizeInBits() < 32) {
                        vF32 = sb.CreateZExt(vF32, sh.vecTy(i32Ty));
                    }
                    vF32 = sb.CreateBitCast(vF32, sh.vecTy(f32Ty));
                } else {
                    vF32 = sb.CreateFPExt(vF32, sh.vecTy(f32Ty));
                }
            }
            auto storeStaged = [&](llvm::Value* basePtr, uint32_t slotIdx) {
                llvm::Value* slotPtr = sh.gepBytes(basePtr, static_cast<int64_t>(slotIdx * simdWidth * 4));
                llvm::Value* prev = sh.loadScalar(sh.vecTy(f32Ty), slotPtr);
                llvm::Value* effMask = writeMask ? writeMask : activeMask;
                sh.storeScalar(sb.CreateSelect(effMask, vF32, prev), slotPtr);
            };
            if (sig->systemValueType == 3) {
                if (col < 4) storeStaged(stagingPos, col);
                return true;
            }
            uint32_t r = tryGetUniformConstUInt(row).value_or(0);
            if (sig->systemValueType == 6) {
                auto cit = clipSigBaseIdx.find(sigId);
                if (cit != clipSigBaseIdx.end()) {
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (clipDistMask & (1u << idx))) {
                        storeStaged(stagingAttrs, clipAttrOffset + idx);
                    }
                }
                return true;
            }
            if (sig->systemValueType == 7) {
                auto cit = cullSigBaseIdx.find(sigId);
                if (cit != cullSigBaseIdx.end()) {
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (cullDistMask & (1u << idx))) {
                        storeStaged(stagingAttrs, cullAttrOffset + idx);
                    }
                }
                return true;
            }
            if (sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                storeStaged(stagingAttrs, vpIndexAttrOffset);
                return true;
            }
            if (sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                storeStaged(stagingAttrs, shadingRateAttrOffset);
                return true;
            }
            auto it = gsToPsMap.find(sigId);
            if (it != gsToPsMap.end() && it->second.psTargetRow >= 0) {
                uint32_t slot = (static_cast<uint32_t>(it->second.psTargetRow) + r) * 4 + it->second.psStartCol + col;
                if (slot < outAttrScalars) {
                    storeStaged(stagingAttrs, slot);
                }
            }
            return true;
        };

        gsHooks.emitStream = [&](llvm::IRBuilder<>& eb, uint32_t streamId, llvm::Value* realMask) -> bool {
            if (streamId != 0) return true;
            IRHelpers eh{eb, simdWidth};
            llvm::Value* curCountVec = eh.loadScalar(eh.vecTy(i32Ty), emitCountPtr);
            llvm::Value* hasRoomVec = eb.CreateICmp(llvm::CmpInst::ICMP_ULT, curCountVec, eh.splatInt(i32Ty, maxVerts));
            llvm::Value* doEmitMask = eb.CreateAnd(realMask, hasRoomVec);
            llvm::Value* anyEmit = eh.orReduce(doEmitMask);
            emitIf(eb, anyEmit, [&](llvm::IRBuilder<>& nb) {
                IRHelpers nh{nb, simdWidth};
                llvm::Value* posVecs[4];
                for (uint32_t c = 0; c < 4; ++c) {
                    posVecs[c] = nh.loadScalar(nh.vecTy(f32Ty), nh.gepBytes(stagingPos, static_cast<int64_t>(c * simdWidth * 4)));
                }
                llvm::SmallVector<llvm::Value*, 16> attrVecs(outAttrScalars);
                for (uint32_t s = 0; s < outAttrScalars; ++s) {
                    attrVecs[s] = nh.loadScalar(nh.vecTy(f32Ty), nh.gepBytes(stagingAttrs, static_cast<int64_t>(s * simdWidth * 4)));
                }

                for (uint32_t l = 0; l < simdWidth; ++l) {
                    llvm::Value* laneDo = llvmExtract(nb, doEmitMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    emitIf(nb, laneDo, [&](llvm::IRBuilder<>& lb2) {
                        IRHelpers lh2{lb2, simdWidth};
                        llvm::Value* cL = llvmExtract(lb2, curCountVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* vIdx = lb2.CreateAdd(lh2.cInt(i32Ty, static_cast<int64_t>(l * maxVerts)), cL);
                        llvm::Value* vIdx64 = lb2.CreateZExt(vIdx, lh2.i64Ty());
                        llvm::Value* posBase = lb2.CreateMul(vIdx64, lh2.cInt(lh2.i64Ty(), 16));
                        for (uint32_t c = 0; c < 4; ++c) {
                            llvm::Value* pv = llvmExtract(lb2, posVecs[c], static_cast<uint32_t>(static_cast<int64_t>(l)));
                            llvm::Value* off = lb2.CreateAdd(posBase, lh2.cInt(lh2.i64Ty(), static_cast<int64_t>(c * 4)));
                            lh2.storeScalar(pv, lh2.gepBytesDyn(outPosXYZW, off));
                        }
                        if (outAttrScalars > 0) {
                            llvm::Value* attrBase = lb2.CreateMul(vIdx64, lh2.cInt(lh2.i64Ty(), static_cast<int64_t>(outAttrScalars * 4)));
                            for (uint32_t s = 0; s < outAttrScalars; ++s) {
                                llvm::Value* av = llvmExtract(lb2, attrVecs[s], static_cast<uint32_t>(static_cast<int64_t>(l)));
                                llvm::Value* off = lb2.CreateAdd(attrBase, lh2.cInt(lh2.i64Ty(), static_cast<int64_t>(s * 4)));
                                lh2.storeScalar(av, lh2.gepBytesDyn(outAttrs, off));
                            }
                        }
                    });
                }

                llvm::Value* nextCount = nb.CreateSelect(doEmitMask, nb.CreateAdd(curCountVec, nh.splatInt(i32Ty, 1)), curCountVec);
                nh.storeScalar(nextCount, emitCountPtr);
                llvm::Value* csVec = nh.loadScalar(nh.vecTy(i32Ty), curStripPtr);
                llvm::Value* nextCS = nb.CreateSelect(doEmitMask, nb.CreateAdd(csVec, nh.splatInt(i32Ty, 1)), csVec);
                nh.storeScalar(nextCS, curStripPtr);
            });
            return true;
        };

        auto emitCutForMask = [&](llvm::IRBuilder<>& cb, llvm::Value* maskVec) {
            IRHelpers ch{cb, simdWidth};
            llvm::Value* csVec = ch.loadScalar(ch.vecTy(i32Ty), curStripPtr);
            llvm::Value* hasVertsVec = cb.CreateICmp(llvm::CmpInst::ICMP_UGT, csVec, ch.splatInt(i32Ty, 0));
            llvm::Value* doCutMask = cb.CreateAnd(maskVec, hasVertsVec);
            llvm::Value* anyCut = ch.orReduce(doCutMask);
            emitIf(cb, anyCut, [&](llvm::IRBuilder<>& nb) {
                IRHelpers nh{nb, simdWidth};
                llvm::Value* nsVec = nh.loadScalar(nh.vecTy(i32Ty), numStripsPtr);
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    llvm::Value* laneCut = llvmExtract(nb, doCutMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    emitIf(nb, laneCut, [&](llvm::IRBuilder<>& lb2) {
                        IRHelpers lh2{lb2, simdWidth};
                        llvm::Value* nsL = llvmExtract(lb2, nsVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* csL = llvmExtract(lb2, csVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        llvm::Value* stripIdx = lb2.CreateAdd(lh2.cInt(i32Ty, static_cast<int64_t>(l * maxVerts)), nsL);
                        llvm::Value* off = lb2.CreateMul(lb2.CreateZExt(stripIdx, lh2.i64Ty()), lh2.cInt(lh2.i64Ty(), 4));
                        lh2.storeScalar(csL, lh2.gepBytesDyn(outStripCounts, off));
                    });
                }
                nh.storeScalar(
                    nb.CreateSelect(doCutMask, nb.CreateAdd(nsVec, nh.splatInt(i32Ty, 1)), nsVec),
                    numStripsPtr);
                nh.storeScalar(
                    nb.CreateSelect(doCutMask, nh.splatInt(i32Ty, 0), csVec),
                    curStripPtr);
            });
        };

        gsHooks.cutStream = [&](llvm::IRBuilder<>& cb, uint32_t streamId, llvm::Value* realMask) -> bool {
            if (streamId != 0) return true;
            emitCutForMask(cb, realMask);
            return true;
        };

        llvm::Value* gsRealMask = activeMask;
        if (!packetizeRegionWithHooks(b, combinedMod, *gsTree.root, gsLoaded, gsEffectiveLayout,
                                      simdWidth, rootCtx, activeMask, gsRealMask, gsHooks, outError)) {
            return false;
        }

        // Flush any remaining vertices in current strip across all active lanes
        emitCutForMask(b, activeMask);

        b.CreateAlignedStore(h.loadScalar(h.vecTy(i32Ty), emitCountPtr), outTotalVerts, llvm::Align(/*alignment=*/4));
        b.CreateAlignedStore(h.loadScalar(h.vecTy(i32Ty), numStripsPtr), outNumStrips, llvm::Align(/*alignment=*/4));
        b.CreateRetVoid();
    }

    // --- 2. Emit @<psSym>(const PixelBatchArgs* args) ---
    {
        llvm::IRBuilder<> b(combinedMod.getContext());
        IRHelpers h{b, simdWidth};
        auto ptrTy = h.ptrTy();
        auto i32Ty = h.i32Ty();
        auto f32Ty = h.f32Ty();
        auto fnTy = llvm::FunctionType::get(b.getVoidTy(), {ptrTy}, false);
        auto* psFn = llvm::Function::Create(fnTy, llvm::Function::ExternalLinkage, psSym, combinedMod);
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(b.getContext(), "entry", psFn);
        b.SetInsertPoint(entry);

        llvm::Value* argsPtr = entry->getParent()->getArg(0);
        llvm::Value* rootCtx = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, rootCtx)));
        llvm::Value* covMask32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, coverageMask)));
        llvm::Value* isFront32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, isFrontFace)));
        llvm::Value* primId32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, primitiveId)));

        llvm::Value* covVec = llvmSplat(b, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getElementType(), covMask32, llvm::cast<llvm::FixedVectorType>(h.vecTy(i32Ty))->getNumElements());
        llvm::Value* shifted = b.CreateLShr(covVec, h.laneIota());
        llvm::Value* bitOne = b.CreateAnd(shifted, h.splatInt(i32Ty, 1));
        llvm::Value* initRealMask = b.CreateICmp(llvm::CmpInst::ICMP_NE, bitOne, h.splatInt(i32Ty, 0));

        // Load PixelBatchArgs vectors
        auto loadVecFromArgField = [&](size_t fieldOffset) -> llvm::Value* {
            llvm::Value* p = h.loadScalar(ptrTy, h.gepBytes(argsPtr, static_cast<int64_t>(fieldOffset)));
            return b.CreateAlignedLoad(h.vecTy(f32Ty), p, llvm::Align(/*alignment=*/4));
        };
        llvm::Value* pixelX = loadVecFromArgField(offsetof(PixelBatchArgs, pixelX));
        llvm::Value* pixelY = loadVecFromArgField(offsetof(PixelBatchArgs, pixelY));
        llvm::Value* baryI = loadVecFromArgField(offsetof(PixelBatchArgs, baryI));
        llvm::Value* baryJ = loadVecFromArgField(offsetof(PixelBatchArgs, baryJ));

        // Load quad offsets (W/4 elements)
        uint32_t numQuads = simdWidth / 4;
        llvm::Value* quadOffPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, quadOffsets)));
        llvm::SmallVector<llvm::Value*, 4> quadOffsets(numQuads);
        for (uint32_t q = 0; q < numQuads; ++q) {
            quadOffsets[q] = h.loadScalar(i32Ty, h.gepBytes(quadOffPtr, static_cast<int64_t>(q * 4)));
        }

        // Interpolate Z from triDepth[0..2]
        llvm::Value* triDepthPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, triDepth)));
        llvm::Value* z0 = llvmSplat(b, f32Ty, h.loadScalar(f32Ty, triDepthPtr), simdWidth);
        llvm::Value* dz1 = llvmSplat(b, f32Ty, h.loadScalar(f32Ty, h.gepBytes(triDepthPtr, 4)), simdWidth);
        llvm::Value* dz2 = llvmSplat(b, f32Ty, h.loadScalar(f32Ty, h.gepBytes(triDepthPtr, 8)), simdWidth);
        llvm::Value* interpZ = b.CreateFAdd(z0, b.CreateFAdd(
                b.CreateFMul(baryI, dz1),
                b.CreateFMul(baryJ, dz2)));
        if (!desc.RasterizerState.DepthClipEnable) {
            interpZ = h.clamp01(interpZ);
        }

        llvm::Value* tileDepthBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, tileDepthSOA)));
        llvm::Value* tileStencilBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, tileStencilSOA)));
        llvm::Value* tileColorBases[8]{};
        for (uint32_t rt = 0; rt < 8; ++rt) {
            tileColorBases[rt] = h.loadScalar(
                ptrTy, h.gepBytes(argsPtr, static_cast<int64_t>(offsetof(PixelBatchArgs, tileColorSOA) + rt * 8)));
        }

        // Helper lambdas for quad-swizzled depth & stencil SOA access (zero branches on loads, masked on writes)
        auto loadDepthVec = [&](llvm::IRBuilder<>& db) -> llvm::Value* {
            IRHelpers dh{db, simdWidth};
            llvm::Value* out = dh.splatFlt(1.0);
            auto v4f = llvm::FixedVectorType::get(f32Ty, 4);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::Value* qByte = db.CreateMul(db.CreateZExt(quadOffsets[q], dh.i64Ty()), dh.cInt(dh.i64Ty(), 4));
                llvm::Value* qp = dh.gepBytesDyn(tileDepthBase, qByte);
                llvm::Value* q4 = dh.loadScalar(v4f, qp);
                for (uint32_t s = 0; s < 4; ++s) {
                    llvm::Value* sc = llvmExtract(db, q4, static_cast<uint32_t>(static_cast<int64_t>(s)));
                    out = llvmInsert(db, out, sc, static_cast<uint32_t>(static_cast<int64_t>(q * 4 + s)));
                }
            }
            return out;
        };

        auto storeDepthVec = [&](llvm::IRBuilder<>& db, llvm::Value* maskVec, llvm::Value* newZ, llvm::Value* oldZ) {
            IRHelpers dh{db, simdWidth};
            llvm::Value* merged = db.CreateSelect(maskVec, newZ, oldZ);
            auto v4f = llvm::FixedVectorType::get(f32Ty, 4);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::SmallVector<int32_t, 4> qIdx{static_cast<int32_t>(q * 4), static_cast<int32_t>(q * 4 + 1), static_cast<int32_t>(q * 4 + 2), static_cast<int32_t>(q * 4 + 3)};
                llvm::Value* qMask = db.CreateShuffleVector(maskVec, maskVec, qIdx);
                llvm::Value* anyQ = db.CreateOrReduce(qMask);
                emitIf(db, anyQ, [&](llvm::IRBuilder<>& qb) {
                    IRHelpers qh{qb, simdWidth};
                    llvm::Value* q4 = qb.CreateShuffleVector(merged, merged, qIdx);
                    llvm::Value* qByte = qb.CreateMul(qb.CreateZExt(quadOffsets[q], qh.i64Ty()), qh.cInt(qh.i64Ty(), 4));
                    llvm::Value* qp = qh.gepBytesDyn(tileDepthBase, qByte);
                    qb.CreateStore(q4, qp);
                });
            }
        };

        auto loadStencilVec = [&](llvm::IRBuilder<>& db) -> llvm::Value* {
            IRHelpers dh{db, simdWidth};
            llvm::Value* out = dh.splatInt(dh.i8Ty(), 0);
            auto v4i8 = llvm::FixedVectorType::get(dh.i8Ty(), 4);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::Value* q64 = db.CreateZExt(quadOffsets[q], dh.i64Ty());
                llvm::Value* qp = dh.gepBytesDyn(tileStencilBase, q64);
                llvm::Value* q4 = dh.loadScalar(v4i8, qp);
                for (uint32_t s = 0; s < 4; ++s) {
                    llvm::Value* sc = llvmExtract(db, q4, static_cast<uint32_t>(static_cast<int64_t>(s)));
                    out = llvmInsert(db, out, sc, static_cast<uint32_t>(static_cast<int64_t>(q * 4 + s)));
                }
            }
            return out;
        };

        auto storeStencilVec = [&](llvm::IRBuilder<>& db, llvm::Value* maskVec, llvm::Value* newS, llvm::Value* oldS) {
            IRHelpers dh{db, simdWidth};
            llvm::Value* merged = db.CreateSelect(maskVec, newS, oldS);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::SmallVector<int32_t, 4> qIdx{static_cast<int32_t>(q * 4), static_cast<int32_t>(q * 4 + 1), static_cast<int32_t>(q * 4 + 2), static_cast<int32_t>(q * 4 + 3)};
                llvm::Value* qMask = db.CreateShuffleVector(maskVec, maskVec, qIdx);
                llvm::Value* anyQ = db.CreateOrReduce(qMask);
                emitIf(db, anyQ, [&](llvm::IRBuilder<>& qb) {
                    IRHelpers qh{qb, simdWidth};
                    llvm::Value* q4 = qb.CreateShuffleVector(merged, merged, qIdx);
                    llvm::Value* q64 = qb.CreateZExt(quadOffsets[q], qh.i64Ty());
                    llvm::Value* qp = qh.gepBytesDyn(tileStencilBase, q64);
                    qb.CreateStore(q4, qp);
                });
            }
        };

        auto runDepthStencil = [&](llvm::IRBuilder<>& db, llvm::Value* curRealMask, llvm::Value* candZ,
                                   llvm::Value* candStencilRef8 = nullptr) -> llvm::Value* {
            IRHelpers dh{db, simdWidth};
            const auto& ds = desc.DepthStencilState;
            bool depthOn = ds.DepthEnable && (desc.DSVFormat != DXGI_FORMAT_UNKNOWN);
            bool stencilOn = ds.StencilEnable && (desc.DSVFormat != DXGI_FORMAT_UNKNOWN);
            if (!depthOn && !stencilOn) return curRealMask;

            llvm::Value* nextMask = curRealMask;
            llvm::Value* stencilPassMask = dh.allTrueMask();
            llvm::Value* oldStencil;
            llvm::Value* ref8;
            llvm::Value* writeMask8;
            D3D12_DEPTH_STENCILOP_DESC activeSOp = ds.FrontFace;

            if (stencilOn) {
                oldStencil = loadStencilVec(db);
                if (candStencilRef8) {
                    ref8 = candStencilRef8;
                } else {
                    llvm::Value* ref32 = dh.loadScalar(i32Ty, dh.gepBytes(rootCtx, offsetof(DriverRootContext, stencil_ref)));
                    llvm::Value* refScalar8 = db.CreateTrunc(ref32, dh.i8Ty());
                    ref8 = llvmSplat(db, llvm::cast<llvm::FixedVectorType>(dh.vecTy(dh.i8Ty()))->getElementType(), refScalar8, llvm::cast<llvm::FixedVectorType>(dh.vecTy(dh.i8Ty()))->getNumElements());
                }
                llvm::Value* rMask8 = dh.splatInt(dh.i8Ty(), ds.StencilReadMask);
                writeMask8 = dh.splatInt(dh.i8Ty(), ds.StencilWriteMask);
                llvm::Value* refMasked = db.CreateAnd(ref8, rMask8);
                llvm::Value* bufMasked = db.CreateAnd(oldStencil, rMask8);
                llvm::Value* isFrontBool = db.CreateICmp(llvm::CmpInst::ICMP_NE, isFront32, dh.cInt(i32Ty, 0));
                llvm::Value* frontPass = dh.evalStencilCmp(ds.FrontFace.StencilFunc, refMasked, bufMasked);
                llvm::Value* backPass = dh.evalStencilCmp(ds.BackFace.StencilFunc, refMasked, bufMasked);
                stencilPassMask = db.CreateSelect(llvmSplat(db, llvm::cast<llvm::FixedVectorType>(dh.vecTy(dh.i1Ty()))->getElementType(), isFrontBool, llvm::cast<llvm::FixedVectorType>(dh.vecTy(dh.i1Ty()))->getNumElements()), frontPass, backPass);
                nextMask = db.CreateAnd(nextMask, stencilPassMask);
            }

            llvm::Value* zPassMask = dh.allTrueMask();
            if (depthOn) {
                llvm::Value* oldZ = loadDepthVec(db);
                zPassMask = dh.evalDepthCmp(ds.DepthFunc, candZ, oldZ);
                nextMask = db.CreateAnd(nextMask, zPassMask);
                if (ds.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL) {
                    storeDepthVec(db, nextMask, candZ, oldZ);
                }
            }

            if (stencilOn && ds.StencilWriteMask != 0) {
                llvm::Value* sFailMask = db.CreateAnd(curRealMask, db.CreateXor(stencilPassMask, dh.allTrueMask()));
                llvm::Value* zFailMask = db.CreateAnd(db.CreateAnd(curRealMask, stencilPassMask), db.CreateXor(zPassMask, dh.allTrueMask()));
                llvm::Value* sUpdated = oldStencil;
                if (activeSOp.StencilFailOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = db.CreateSelect(sFailMask, dh.applyStencilOp(activeSOp.StencilFailOp, oldStencil, ref8), sUpdated);
                }
                if (activeSOp.StencilDepthFailOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = db.CreateSelect(zFailMask, dh.applyStencilOp(activeSOp.StencilDepthFailOp, oldStencil, ref8), sUpdated);
                }
                if (activeSOp.StencilPassOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = db.CreateSelect(nextMask, dh.applyStencilOp(activeSOp.StencilPassOp, oldStencil, ref8), sUpdated);
                }
                llvm::Value* keepBits = dh.splatInt(dh.i8Ty(), static_cast<uint8_t>(~ds.StencilWriteMask));
                llvm::Value* maskedNewS = db.CreateOr(db.CreateAnd(oldStencil, keepBits), db.CreateAnd(sUpdated, writeMask8));
                storeStencilVec(db, curRealMask, maskedNewS, oldStencil);
            }
            return nextMask;
        };

        bool psWritesDepth = hasPS && psLoaded.writesDepth;
        bool psWritesStencilRef = false;
        if (hasPS) {
            for (const auto& po : psLoaded.outputSigs) {
                if (po.systemValueType == 20) { // SV_StencilRef
                    psWritesStencilRef = true;
                    break;
                }
            }
        }
        bool psUsesDiscard = hasPS && psLoaded.usesDiscard;
        bool psHasSideEffects = hasPS && psLoaded.hasSideEffects;
        bool canEarlyZ = !psWritesDepth && !psWritesStencilRef && !psUsesDiscard && !psHasSideEffects;

        llvm::Value* realMaskAfterEarly = initRealMask;
        if (canEarlyZ) {
            realMaskAfterEarly = runDepthStencil(b, initRealMask, interpZ);
        }

        if (hasPS) {
            if (!psTree.root) {
                outError = "Pixel shader entry function missing after MLIR SCF lift";
                return false;
            }

            auto emitPSBodyAndOM = [&](llvm::IRBuilder<>& pb, llvm::Value* enterRealMask) {
                IRHelpers ph{pb, simdWidth};
                llvm::Value* curRealMask = enterRealMask;
                llvm::Value* execMask = psLoaded.hasDerivativesOrImplicitSample
                                           ? ph.expandMaskQuads(curRealMask)
                                           : curRealMask;

                // Interpolate PS inputs
                llvm::Value* wVec = ph.splatFlt(1.0);
                llvm::Value* invWVec = ph.splatFlt(1.0);
                bool needInvW = false;
                for (const auto& pi : psLoaded.inputSigs) {
                    if (pi.systemValueType == 3 ||
                        (pi.systemValueType == 0 && pi.interpMode == 2) ||
                        (pi.systemValueType == 28 && pi.interpMode != 4)) {
                        needInvW = true;
                        break;
                    }
                }
                if (needInvW) {
                    llvm::Value* triInvWPtr = ph.loadScalar(ptrTy, ph.gepBytes(argsPtr, offsetof(PixelBatchArgs, triInvW)));
                    llvm::Value* iw0 = llvmSplat(pb, f32Ty, ph.loadScalar(f32Ty, triInvWPtr), simdWidth);
                    llvm::Value* diw1 = llvmSplat(pb, f32Ty, ph.loadScalar(f32Ty, ph.gepBytes(triInvWPtr, 4)), simdWidth);
                    llvm::Value* diw2 = llvmSplat(pb, f32Ty, ph.loadScalar(f32Ty, ph.gepBytes(triInvWPtr, 8)), simdWidth);
                    invWVec = pb.CreateFAdd(iw0, pb.CreateFAdd(
                            pb.CreateFMul(baryI, diw1),
                            pb.CreateFMul(baryJ, diw2)));
                    llvm::Value* safeInvW = pb.CreateSelect(pb.CreateFCmp(llvm::CmpInst::FCMP_OGT, pb.CreateIntrinsic(llvm::Intrinsic::fabs, {invWVec->getType()}, {invWVec}), ph.splatFlt(1e-20)), invWVec, ph.splatFlt(1.0));
                    wVec = pb.CreateFDiv(ph.splatFlt(1.0), safeInvW);
                }

                llvm::Value* triAttrBasePtr = ph.loadScalar(ptrTy, ph.gepBytes(argsPtr, offsetof(PixelBatchArgs, triAttrBase)));
                llvm::Value* triAttrFlatPtr = ph.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, triAttrFlat)));

                // Cache interpolated scalar registers (max 32 * 4 = 128 scalars)
                std::array<llvm::Value*, 128> interpScalars{};
                for (const auto& pi : psLoaded.inputSigs) {
                    if (pi.systemValueType != 0) continue;
                    for (uint32_t r = 0; r < pi.rows; ++r) {
                        for (uint32_t c = 0; c < pi.cols && (pi.startCol + c) < 4; ++c) {
                            uint32_t slot = (pi.startRow + r) * 4 + (pi.startCol + c);
                            if (slot >= 128) continue;
                            bool isMsPrimAttr = (slot < 64) && ((msPrimAttrMask & (1ull << slot)) != 0);
                            if (pi.interpMode == 1 || isMsPrimAttr) { // nointerpolation or MS per-primitive
                                llvm::Value* fv = ph.loadScalar(f32Ty, ph.gepBytes(triAttrFlatPtr, static_cast<int64_t>(slot * 4)));
                                interpScalars[slot] = llvmSplat(pb, llvm::cast<llvm::FixedVectorType>(ph.vecTy(f32Ty))->getElementType(), fv, llvm::cast<llvm::FixedVectorType>(ph.vecTy(f32Ty))->getNumElements());
                            } else {
                                int64_t baseOff = static_cast<int64_t>(slot * 3 * 4);
                                llvm::Value* a0 = llvmSplat(pb, f32Ty, ph.loadScalar(f32Ty, ph.gepBytes(triAttrBasePtr, baseOff)), simdWidth);
                                llvm::Value* da1 = llvmSplat(pb, f32Ty, ph.loadScalar(f32Ty, ph.gepBytes(triAttrBasePtr, baseOff + 4)), simdWidth);
                                llvm::Value* da2 = llvmSplat(pb, f32Ty, ph.loadScalar(f32Ty, ph.gepBytes(triAttrBasePtr, baseOff + 8)), simdWidth);
                                llvm::Value* lin = pb.CreateFAdd(a0, pb.CreateFAdd(
                                        pb.CreateFMul(baryI, da1),
                                        pb.CreateFMul(baryJ, da2)));
                                interpScalars[slot] = (pi.interpMode == 4)
                                                          ? lin
                                                          : pb.CreateFMul(lin, wVec);
                            }
                        }
                    }
                }

                // Allocate stack slots for PS outputs (SV_Target0..7 and optional SV_Depth) so writes inside scf.if work naturally
                std::array<std::array<llvm::Value*, 4>, 8> rtOutPtrs{};
                for (uint32_t rt = 0; rt < 8; ++rt) {
                    if (rt < desc.NumRenderTargets && desc.RTVFormats[rt] != DXGI_FORMAT_UNKNOWN) {
                        for (uint32_t c = 0; c < 4; ++c) {
                            rtOutPtrs[rt][c] = createAlignedAlloca(pb, ph.vecTy(f32Ty), ph.cInt(i32Ty, 1), 64);
                            ph.storeScalar(ph.splatFlt(c == 3 ? 1.0 : 0.0), rtOutPtrs[rt][c]);
                        }
                    }
                }
                llvm::Value* outDepthPtr = createAlignedAlloca(pb, ph.vecTy(f32Ty), ph.cInt(i32Ty, 1), 64);
                ph.storeScalar(interpZ, outDepthPtr);

                llvm::Value* outStencilRefPtr = nullptr;
                if (psWritesStencilRef) {
                    outStencilRefPtr = createAlignedAlloca(pb, ph.vecTy(ph.i8Ty()), ph.cInt(i32Ty, 1), 64);
                    llvm::Value* initRef32 = ph.loadScalar(i32Ty, ph.gepBytes(rootCtx, offsetof(DriverRootContext, stencil_ref)));
                    llvm::Value* initRef8 = pb.CreateTrunc(initRef32, ph.i8Ty());
                    ph.storeScalar(llvmSplat(pb, llvm::cast<llvm::FixedVectorType>(ph.vecTy(ph.i8Ty()))->getElementType(), initRef8, llvm::cast<llvm::FixedVectorType>(ph.vecTy(ph.i8Ty()))->getNumElements()), outStencilRefPtr);
                }

                StageHooks psHooks;
                psHooks.isPixelShader = true;
                psHooks.loadInput = [&](llvm::IRBuilder<>& lb, uint32_t sigId,
                                        const SimdVal& row, uint32_t col, llvm::Type* scalarTy, SimdVal& out) -> bool {
                    const DxilSigRecord* sig = psLoaded.findInputSig(sigId);
                    if (!sig) return false;
                    IRHelpers lh{lb, simdWidth};
                    if (sig->systemValueType == 3) { // SV_Position
                        llvm::Value* v = (col == 0) ? pixelX : ((col == 1) ? pixelY : ((col == 2) ? interpZ : invWVec));
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 5) { // SV_ViewportArrayIndex
                        llvm::Value* vp32 = lh.loadScalar(
                            i32Ty, lh.gepBytes(argsPtr, offsetof(PixelBatchArgs, viewportArrayIndex)));
                        llvm::Value* v = llvmSplat(lb, i32Ty, vp32, simdWidth);
                        if (scalarTy != i32Ty) {
                            v = lb.CreateTrunc(v, lh.vecTy(scalarTy));
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 10) { // SV_PrimitiveID
                        llvm::Value* v = llvmSplat(lb, i32Ty, primId32, simdWidth);
                        if (scalarTy != i32Ty) {
                            v = lb.CreateTrunc(v, lh.vecTy(scalarTy));
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 28) { // SV_Barycentrics
                        llvm::Value* v = lh.splatFlt(0.0);
                        if (sig->interpMode == 4) { // noperspective
                            llvm::Value* b0 = lb.CreateFSub(lh.splatFlt(1.0), lb.CreateFAdd(baryI, baryJ));
                            if (col == 0) v = b0;
                            else if (col == 1) v = baryI;
                            else if (col == 2) v = baryJ;
                        } else if (col < 3) { // perspective (linear) using original-primitive barycentrics
                            llvm::Value* triOrigBaryPtr = lh.loadScalar(
                                ptrTy, lh.gepBytes(argsPtr, offsetof(PixelBatchArgs, triOrigBary)));
                            int64_t baseOff = static_cast<int64_t>(col * 3 * 4);
                            llvm::Value* a0 = llvmSplat(lb, f32Ty, lh.loadScalar(f32Ty, lh.gepBytes(triOrigBaryPtr, baseOff)), simdWidth);
                            llvm::Value* da1 = llvmSplat(lb, f32Ty, lh.loadScalar(f32Ty, lh.gepBytes(triOrigBaryPtr, baseOff + 4)), simdWidth);
                            llvm::Value* da2 = llvmSplat(lb, f32Ty, lh.loadScalar(f32Ty, lh.gepBytes(triOrigBaryPtr, baseOff + 8)), simdWidth);
                            llvm::Value* lin = lb.CreateFAdd(a0, lb.CreateFAdd(
                                    lb.CreateFMul(baryI, da1),
                                    lb.CreateFMul(baryJ, da2)));
                            v = lb.CreateFMul(lin, wVec);
                        }
                        if (scalarTy != f32Ty) {
                            v = lb.CreateFPTrunc(v, lh.vecTy(scalarTy));
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 29) { // SV_ShadingRate
                        llvm::Value* sr32 = lh.loadScalar(
                            i32Ty, lh.gepBytes(argsPtr, offsetof(PixelBatchArgs, shadingRate)));
                        llvm::Value* v = llvmSplat(lb, i32Ty, sr32, simdWidth);
                        if (scalarTy != i32Ty) {
                            v = lb.CreateTrunc(v, lh.vecTy(scalarTy));
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 13) { // SV_IsFrontFace
                        llvm::Value* ffCond = lb.CreateICmp(llvm::CmpInst::ICMP_NE, isFront32, lh.cInt(i32Ty, 0));
                        llvm::Value* v = (scalarTy == lh.i1Ty())
                                            ? llvmSplat(lb, lh.i1Ty(), ffCond, simdWidth)
                                            : llvmSplat(lb, i32Ty, lb.CreateZExt(ffCond, i32Ty), simdWidth);
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    auto rowConst = tryGetUniformConstUInt(row);
                    llvm::Value* v;
                    if (rowConst.has_value() || sig->rows <= 1) {
                        uint32_t r = rowConst.value_or(0);
                        uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                        v = (slot < 128 && interpScalars[slot]) ? interpScalars[slot] : lh.splatFlt(0.0);
                    } else {
                        uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                        v = (baseSlot < 128 && interpScalars[baseSlot]) ? interpScalars[baseSlot] : lh.splatFlt(0.0);
                        llvm::Type* rElemTy = llvm::cast<llvm::FixedVectorType>(row.vec->getType())->getElementType();
                        for (uint32_t r = 1; r < sig->rows; ++r) {
                            uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                            if (slot >= 128 || !interpScalars[slot]) continue;
                            llvm::Value* cmp = lb.CreateICmp(llvm::CmpInst::ICMP_EQ, row.vec, lh.splatInt(rElemTy, r));
                            v = lb.CreateSelect(cmp, interpScalars[slot], v);
                        }
                    }
                    if (scalarTy != f32Ty) {
                        if (scalarTy->isIntegerTy()) {
                            v = lb.CreateBitCast(v, lh.vecTy(i32Ty));
                            if (scalarTy != i32Ty) {
                                v = lb.CreateTrunc(v, lh.vecTy(scalarTy));
                            }
                        } else {
                            v = lb.CreateBitCast(v, lh.vecTy(scalarTy));
                        }
                    }
                    out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                    return true;
                };

                psHooks.storeOutput = [&](llvm::IRBuilder<>& sb, uint32_t sigId,
                                          const SimdVal&, uint32_t col, const SimdVal& val, llvm::Value* writeMask) -> bool {
                    const DxilSigRecord* sig = psLoaded.findOutputSig(sigId);
                    if (!sig || val.kind != SimdVal::Kind::Vector) return true;
                    IRHelpers sh{sb, simdWidth};
                    llvm::Value* vVec = val.vec;
                    llvm::Type* et = llvm::cast<llvm::FixedVectorType>(vVec->getType())->getElementType();
                    if (sig->systemValueType == 20 && outStencilRefPtr) { // SV_StencilRef
                        llvm::Value* refVec8;
                        if (llvm::isa<llvm::IntegerType>(et)) {
                            refVec8 = (et == sh.i8Ty())
                                          ? vVec
                                          : sb.CreateTrunc(vVec, sh.vecTy(sh.i8Ty()));
                        } else {
                            llvm::Value* asI32 = sb.CreateBitCast(vVec, sh.vecTy(i32Ty));
                            refVec8 = sb.CreateTrunc(asI32, sh.vecTy(sh.i8Ty()));
                        }
                        llvm::Value* prev = sh.loadScalar(sh.vecTy(sh.i8Ty()), outStencilRefPtr);
                        sh.storeScalar(sb.CreateSelect(writeMask, refVec8, prev), outStencilRefPtr);
                        return true;
                    }
                    if (et != f32Ty) {
                        vVec = llvm::isa<llvm::IntegerType>(et)
                                   ? sb.CreateBitCast(vVec, sh.vecTy(f32Ty))
                                   : sb.CreateFPExt(vVec, sh.vecTy(f32Ty));
                    }
                    if (sig->systemValueType == 17 || sig->systemValueType == 18 || sig->systemValueType == 19) { // SV_Depth*
                        llvm::Value* prev = sh.loadScalar(sh.vecTy(f32Ty), outDepthPtr);
                        sh.storeScalar(sb.CreateSelect(writeMask, vVec, prev), outDepthPtr);
                        return true;
                    }
                    if (sig->systemValueType == 16 && sig->semanticIndex < 8 && col < 4) { // SV_TargetN
                        llvm::Value* slotPtr = rtOutPtrs[sig->semanticIndex][col];
                        if (slotPtr) {
                            llvm::Value* prev = sh.loadScalar(sh.vecTy(f32Ty), slotPtr);
                            sh.storeScalar(sb.CreateSelect(writeMask, vVec, prev), slotPtr);
                        }
                    }
                    return true;
                };

                if (!packetizeRegionWithHooks(pb, combinedMod, *psTree.root, psLoaded, psEffectiveLayout,
                                              simdWidth, rootCtx, execMask, curRealMask, psHooks, outError)) {
                    return;
                }

                // Late-Z if Early-Z was skipped
                if (!canEarlyZ) {
                    llvm::Value* finalZ = ph.loadScalar(ph.vecTy(f32Ty), outDepthPtr);
                    llvm::Value* finalStencilRef = psWritesStencilRef
                                                      ? ph.loadScalar(ph.vecTy(ph.i8Ty()), outStencilRefPtr)
                                                      : nullptr;
                    curRealMask = runDepthStencil(pb, curRealMask, finalZ, finalStencilRef);
                }

                // Fixed-function Output Merger Blending across active MRTs
                llvm::Value* anyWrite = ph.orReduce(curRealMask);
                emitIf(pb, anyWrite, [&](llvm::IRBuilder<>& ob) {
                    IRHelpers oh{ob, simdWidth};
                    llvm::Value* blendFactors[4]{};
                    bool loadedBlendFactor = false;
                    auto ensureBlendFactor = [&]() {
                        if (loadedBlendFactor) return;
                        for (int c = 0; c < 4; ++c) {
                            llvm::Value* bf = oh.loadScalar(
                                f32Ty, oh.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, blend_factor) + c * 4)));
                            blendFactors[c] = llvmSplat(ob, llvm::cast<llvm::FixedVectorType>(oh.vecTy(f32Ty))->getElementType(), bf, llvm::cast<llvm::FixedVectorType>(oh.vecTy(f32Ty))->getNumElements());
                        }
                        loadedBlendFactor = true;
                    };

                    for (uint32_t rt = 0; rt < desc.NumRenderTargets && rt < 8; ++rt) {
                        if (desc.RTVFormats[rt] == DXGI_FORMAT_UNKNOWN) continue;
                        const auto& rtBlend = desc.BlendState.RenderTarget[desc.BlendState.IndependentBlendEnable ? rt : 0];
                        uint8_t writeMask = rtBlend.RenderTargetWriteMask;
                        if (writeMask == 0) continue;

                        llvm::Value* src[4];
                        for (int c = 0; c < 4; ++c) {
                            src[c] = rtOutPtrs[rt][c] ? oh.loadScalar(oh.vecTy(f32Ty), rtOutPtrs[rt][c]) : oh.splatFlt(c == 3 ? 1.0 : 0.0);
                            if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) {
                                src[c] = oh.clamp01(src[c]);
                            }
                        }

                        // Load current tile color for all W lanes (needed for partial quad masking and blending)
                        llvm::Value* dst[4];
                        auto v4f = llvm::FixedVectorType::get(f32Ty, 4);
                        for (int c = 0; c < 4; ++c) {
                            dst[c] = oh.splatFlt(0.0);
                            for (uint32_t q = 0; q < numQuads; ++q) {
                                llvm::Value* pixIdx = ob.CreateAdd(quadOffsets[q], oh.cInt(i32Ty, c * 4096));
                                llvm::Value* byteOff = ob.CreateMul(ob.CreateZExt(pixIdx, oh.i64Ty()), oh.cInt(oh.i64Ty(), 4));
                                llvm::Value* q4 = oh.loadScalar(v4f, oh.gepBytesDyn(tileColorBases[rt], byteOff));
                                for (uint32_t s = 0; s < 4; ++s) {
                                    llvm::Value* sc = llvmExtract(ob, q4, static_cast<uint32_t>(static_cast<int64_t>(s)));
                                    dst[c] = llvmInsert(ob, dst[c], sc, static_cast<uint32_t>(static_cast<int64_t>(q * 4 + s)));
                                }
                            }
                        }

                        llvm::Value* outColor[4]{src[0], src[1], src[2], src[3]};
                        if (rtBlend.BlendEnable) {
                            if (rtBlend.SrcBlend == D3D12_BLEND_BLEND_FACTOR || rtBlend.SrcBlend == D3D12_BLEND_INV_BLEND_FACTOR ||
                                rtBlend.DestBlend == D3D12_BLEND_BLEND_FACTOR || rtBlend.DestBlend == D3D12_BLEND_INV_BLEND_FACTOR ||
                                rtBlend.SrcBlendAlpha == D3D12_BLEND_BLEND_FACTOR || rtBlend.SrcBlendAlpha == D3D12_BLEND_INV_BLEND_FACTOR ||
                                rtBlend.DestBlendAlpha == D3D12_BLEND_BLEND_FACTOR || rtBlend.DestBlendAlpha == D3D12_BLEND_INV_BLEND_FACTOR) {
                                ensureBlendFactor();
                            }
                            for (int c = 0; c < 3; ++c) {
                                llvm::Value* sf = oh.evalBlendFactorRGB(rtBlend.SrcBlend, src, dst, blendFactors, c);
                                llvm::Value* df = oh.evalBlendFactorRGB(rtBlend.DestBlend, src, dst, blendFactors, c);
                                llvm::Value* sTerm = ob.CreateFMul(src[c], sf);
                                llvm::Value* dTerm = ob.CreateFMul(dst[c], df);
                                outColor[c] = oh.evalBlendOp(rtBlend.BlendOp, sTerm, dTerm);
                                if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) outColor[c] = oh.clamp01(outColor[c]);
                            }
                            llvm::Value* saf = oh.evalBlendFactorRGB(rtBlend.SrcBlendAlpha, src, dst, blendFactors, 3);
                            llvm::Value* daf = oh.evalBlendFactorRGB(rtBlend.DestBlendAlpha, src, dst, blendFactors, 3);
                            outColor[3] = oh.evalBlendOp(
                                rtBlend.BlendOpAlpha,
                                ob.CreateFMul(src[3], saf),
                                ob.CreateFMul(dst[3], daf));
                            if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) outColor[3] = oh.clamp01(outColor[3]);
                        }

                        // Masked SOA quad write-back respecting RenderTargetWriteMask & curRealMask
                        for (uint32_t q = 0; q < numQuads; ++q) {
                            llvm::SmallVector<int32_t, 4> qIdx{static_cast<int32_t>(q * 4), static_cast<int32_t>(q * 4 + 1), static_cast<int32_t>(q * 4 + 2), static_cast<int32_t>(q * 4 + 3)};
                            llvm::Value* qMask = ob.CreateShuffleVector(curRealMask, curRealMask, qIdx);
                            llvm::Value* anyQ = ob.CreateOrReduce(qMask);
                            emitIf(ob, anyQ, [&](llvm::IRBuilder<>& qb) {
                                IRHelpers qh{qb, simdWidth};
                                for (int c = 0; c < 4; ++c) {
                                    if (!(writeMask & (1u << c))) continue;
                                    llvm::Value* merged = qb.CreateSelect(curRealMask, outColor[c], dst[c]);
                                    llvm::Value* q4 = qb.CreateShuffleVector(merged, merged, qIdx);
                                    llvm::Value* pixIdx = qb.CreateAdd(quadOffsets[q], qh.cInt(i32Ty, c * 4096));
                                    llvm::Value* byteOff = qb.CreateMul(qb.CreateZExt(pixIdx, qh.i64Ty()), qh.cInt(qh.i64Ty(), 4));
                                    qb.CreateStore(q4, qh.gepBytesDyn(tileColorBases[rt], byteOff));
                                }
                            });
                        }
                    }
                });
            };

            if (canEarlyZ) {
                llvm::Value* anySurvive = h.orReduce(realMaskAfterEarly);
                emitIf(b, anySurvive, [&](llvm::IRBuilder<>& nb) {
                    emitPSBodyAndOM(nb, realMaskAfterEarly);
                });
                if (!outError.empty()) return false;
            } else {
                emitPSBodyAndOM(b, realMaskAfterEarly);
                if (!outError.empty()) return false;
            }
        }

        b.CreateRetVoid();
    }


    for (llvm::Function* srcFn : {vsSrcFn, msSrcFn, asSrcFn, hsSrcFn, hspcSrcFn, dsSrcFn, gsSrcFn, psSrcFn}) {
        if (srcFn) srcFn->eraseFromParent();
    }
    eraseUnusedDxOps(combinedMod);

    if (llvm::verifyModule(combinedMod, &llvm::errs())) {
        outError = "LLVM verification failed after Graphics Uber-PSO synthesis";
        return false;
    }

    outMeta = CachedPSOMeta{};
    outMeta.simdWidth = simdWidth;
    outMeta.numVSOutputRegs = numVSOutputRegs;
    outMeta.posOutputReg = posOutputReg;
    outMeta.numPSInputRegs = numPSInputRegs;
    outMeta.flatInterpMask = flatInterpMask;
    outMeta.clipDistMask = clipDistMask;
    outMeta.cullDistMask = cullDistMask;
    outMeta.clipAttrOffset = clipAttrOffset;
    outMeta.cullAttrOffset = cullAttrOffset;
    outMeta.hasViewportArrayIndex = hasViewportArrayIndex ? 1 : 0;
    outMeta.vpIndexAttrOffset = vpIndexAttrOffset;
    outMeta.hasPerPrimShadingRate = hasPerPrimShadingRate ? 1 : 0;
    outMeta.shadingRateAttrOffset = shadingRateAttrOffset;
    outMeta.totalVSScalarAttrs = totalVSScalarAttrs;
    outMeta.hasGS = hasGS ? 1 : 0;
    outMeta.gsMaxVertexCount = hasGS ? gsLoaded.gsMaxVertexCount : 0;
    outMeta.gsOutputTopology = hasGS ? gsLoaded.gsOutputTopology : 5;
    outMeta.gsInstanceCount = hasGS ? gsLoaded.gsInstanceCount : 1;
    outMeta.numGSInputScalars = hasGS ? numGSInputScalars : 0;
    outMeta.hasTess = hasTess ? 1 : 0;
    outMeta.hsInputCPCount = hasTess ? hsLoaded.hsInputControlPointCount : 0;
    outMeta.hsOutputCPCount = hasTess ? hsLoaded.hsOutputControlPointCount : 0;
    outMeta.tessDomain = hasTess ? dsLoaded.tessDomain : 0;
    outMeta.tessPartitioning = hasTess ? hsLoaded.tessPartitioning : 0;
    outMeta.tessOutputPrimitive = hasTess ? hsLoaded.tessOutputPrimitive : 0;
    outMeta.hsMaxTessFactor = hasTess ? hsLoaded.hsMaxTessFactor : 64.0f;
    outMeta.numHSInputScalars = hasTess ? numHSInputScalars : 0;
    outMeta.numHSOutputScalars = hasTess ? numHSOutputScalars : 0;
    outMeta.numPatchConstScalars = hasTess ? numPatchConstScalars : 0;
    outMeta.hasMS = hasMS ? 1 : 0;
    outMeta.hasAS = hasAS ? 1 : 0;
    if (hasMS) {
        outMeta.msThreadGroupSize[0] = msLoaded.threadGroupSize[0];
        outMeta.msThreadGroupSize[1] = msLoaded.threadGroupSize[1];
        outMeta.msThreadGroupSize[2] = msLoaded.threadGroupSize[2];
        outMeta.msGroupSharedBytes = msLoaded.groupSharedBytes;
        outMeta.msMaxVertexCount = msLoaded.msMaxVertexCount;
        outMeta.msMaxPrimitiveCount = msLoaded.msMaxPrimitiveCount;
        outMeta.msOutputTopology = msLoaded.msOutputTopology;
        outMeta.msPayloadBytes = msLoaded.msPayloadBytes;
        outMeta.msPrimAttrMask = msPrimAttrMask;
    }
    if (hasAS) {
        outMeta.asThreadGroupSize[0] = asLoaded.threadGroupSize[0];
        outMeta.asThreadGroupSize[1] = asLoaded.threadGroupSize[1];
        outMeta.asThreadGroupSize[2] = asLoaded.threadGroupSize[2];
        outMeta.asGroupSharedBytes = asLoaded.groupSharedBytes;
        outMeta.asPayloadBytes = asLoaded.asPayloadBytes;
    }
    outMeta.psInputFromVSSlot = psInputFromVSSlot;
    outMeta.psUsesPerspBarycentrics = psUsesPerspBarycentrics ? 1 : 0;
    outMeta.psWritesDepth = psWritesDepth ? 1 : 0;
    outMeta.psUsesDiscard = psUsesDiscard ? 1 : 0;
    outMeta.psHasSideEffects = psHasSideEffects ? 1 : 0;
    return true;
}

bool compileGraphicsPSO(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    CompiledGraphicsPSO& outPSO,
    std::string& outError,
    const MeshShaderStages& meshStages) {
    uint32_t simdWidth = getTargetSimdWidth();
    uint64_t lo = 0, hi = 0;
    computeGraphicsPSOHash(desc, rootLayout, simdWidth, lo, hi, meshStages);
    outPSO.cacheHashLo = lo;
    outPSO.cacheHashHi = hi;

    const bool hasMS = (meshStages.MS.pShaderBytecode && meshStages.MS.BytecodeLength > 0);
    bool hasTess = !hasMS && (desc.HS.pShaderBytecode && desc.HS.BytecodeLength > 0 &&
                              desc.DS.pShaderBytecode && desc.DS.BytecodeLength > 0);
    bool hasGS = !hasMS && (desc.GS.pShaderBytecode && desc.GS.BytecodeLength > 0);

    std::string hex = hashToHex128(lo, hi);
    std::string psSym = "nacelle_ps_" + hex;

    auto populatePSOFromMeta = [&](const CachedPSOMeta& meta) {
        outPSO.simdWidth = meta.simdWidth;
        outPSO.numVSOutputRegs = meta.numVSOutputRegs;
        outPSO.posOutputReg = meta.posOutputReg;
        outPSO.numPSInputRegs = meta.numPSInputRegs;
        outPSO.flatInterpMask = meta.flatInterpMask;
        outPSO.clipDistMask = meta.clipDistMask;
        outPSO.cullDistMask = meta.cullDistMask;
        outPSO.clipAttrOffset = meta.clipAttrOffset;
        outPSO.cullAttrOffset = meta.cullAttrOffset;
        outPSO.hasViewportArrayIndex = meta.hasViewportArrayIndex != 0;
        outPSO.vpIndexAttrOffset = meta.vpIndexAttrOffset;
        outPSO.hasPerPrimShadingRate = meta.hasPerPrimShadingRate != 0;
        outPSO.shadingRateAttrOffset = meta.shadingRateAttrOffset;
        outPSO.totalVSScalarAttrs = meta.totalVSScalarAttrs;
        outPSO.hasGS = meta.hasGS != 0;
        outPSO.gsMaxVertexCount = meta.gsMaxVertexCount;
        outPSO.gsOutputTopology = meta.gsOutputTopology;
        outPSO.gsInstanceCount = meta.gsInstanceCount;
        outPSO.numGSInputScalars = meta.numGSInputScalars;
        outPSO.hasTess = meta.hasTess != 0;
        outPSO.hsInputCPCount = meta.hsInputCPCount;
        outPSO.hsOutputCPCount = meta.hsOutputCPCount;
        outPSO.tessDomain = meta.tessDomain;
        outPSO.tessPartitioning = meta.tessPartitioning;
        outPSO.tessOutputPrimitive = meta.tessOutputPrimitive;
        outPSO.hsMaxTessFactor = meta.hsMaxTessFactor;
        outPSO.numHSInputScalars = meta.numHSInputScalars;
        outPSO.numHSOutputScalars = meta.numHSOutputScalars;
        outPSO.numPatchConstScalars = meta.numPatchConstScalars;
        outPSO.hasMS = meta.hasMS != 0;
        outPSO.hasAS = meta.hasAS != 0;
        outPSO.msThreadGroupSize[0] = meta.msThreadGroupSize[0];
        outPSO.msThreadGroupSize[1] = meta.msThreadGroupSize[1];
        outPSO.msThreadGroupSize[2] = meta.msThreadGroupSize[2];
        outPSO.msGroupSharedBytes = meta.msGroupSharedBytes;
        outPSO.msMaxVertexCount = meta.msMaxVertexCount;
        outPSO.msMaxPrimitiveCount = meta.msMaxPrimitiveCount;
        outPSO.msOutputTopology = meta.msOutputTopology;
        outPSO.msPayloadBytes = meta.msPayloadBytes;
        outPSO.msPrimAttrMask = meta.msPrimAttrMask;
        outPSO.asThreadGroupSize[0] = meta.asThreadGroupSize[0];
        outPSO.asThreadGroupSize[1] = meta.asThreadGroupSize[1];
        outPSO.asThreadGroupSize[2] = meta.asThreadGroupSize[2];
        outPSO.asGroupSharedBytes = meta.asGroupSharedBytes;
        outPSO.asPayloadBytes = meta.asPayloadBytes;
        outPSO.psInputFromVSSlot = meta.psInputFromVSSlot;
        outPSO.psUsesPerspBarycentrics = meta.psUsesPerspBarycentrics != 0;
        outPSO.psWritesDepth = meta.psWritesDepth != 0;
        outPSO.psUsesDiscard = meta.psUsesDiscard != 0;
        outPSO.psHasSideEffects = meta.psHasSideEffects != 0;
    };

    if (!hasMS && !hasTess && !hasGS) {
        uint64_t vsHashLo = 0, vsHashHi = 0;
        computeVertexStageHash(desc, rootLayout, simdWidth, vsHashLo, vsHashHi);
        outPSO.vsCacheHashLo = vsHashLo;
        outPSO.vsCacheHashHi = vsHashHi;
        std::string vsSym = "nacelle_vs_" + hashToHex128(vsHashLo, vsHashHi);

        if (tryLoadGraphicsPSOFromCache(lo, hi, vsSym, psSym, outPSO)) {
            return true;
        }

        CachedVSMeta vsMeta{};
        CompiledVertexFn cachedVSFn = nullptr;
        std::vector<DxilSigRecord> vsOutputRecs;
        if (tryLoadVertexStageFromCache(
                vsHashLo, vsHashHi, vsSym, vsMeta, cachedVSFn,
                &outPSO.vsInputs, &outPSO.vsOutputs, &vsOutputRecs)) {
            outPSO.vsSymbolName = vsSym;
            outPSO.vsFn.store(cachedVSFn, std::memory_order_release);
        } else {
            auto vsLlvmCtx = std::make_unique<llvm::LLVMContext>();
            std::unique_ptr<llvm::Module> vsMod;
            if (!synthesizeVertexStageLLVM(
                    desc, rootLayout, simdWidth, vsSym, *vsLlvmCtx, vsMod, vsMeta,
                    outPSO.vsInputs, outPSO.vsOutputs, vsOutputRecs, outError)) {
                return false;
            }
            if (!linkAndFinalizeLLVMModule(*vsMod, outError)) return false;
            if (!jitCompileVertexStageTier1AndScheduleTier2(
                    std::move(vsMod), std::move(vsLlvmCtx), vsHashLo, vsHashHi, vsSym,
                    vsMeta, outPSO.vsInputs, outPSO.vsOutputs, vsOutputRecs, outPSO, outError)) {
                return false;
            }
        }

        auto psLlvmCtx = std::make_unique<llvm::LLVMContext>();
        std::unique_ptr<llvm::Module> psMod;
        CachedPSOMeta meta{};
        if (!synthesizeGraphicsPSOLLVM(
                desc, rootLayout, simdWidth, vsSym, psSym, *psLlvmCtx, psMod, meta,
                outPSO.vsInputs, outPSO.vsOutputs, outPSO.psInputs, outPSO.psOutputs, outError,
                meshStages, /*skipVS=*/true, &vsMeta, &vsOutputRecs)) {
            return false;
        }
        meta.separateVSObject = 1;
        meta.vsHashLo = vsHashLo;
        meta.vsHashHi = vsHashHi;

        if (!linkAndFinalizeLLVMModule(*psMod, outError)) return false;

        populatePSOFromMeta(meta);
        return jitCompilePixelStageTier1AndScheduleTier2(
            std::move(psMod), std::move(psLlvmCtx), vsHashLo, vsHashHi, psSym, meta, outPSO, outError);
    }

    std::string vsSym = "nacelle_vs_" + hex;
    if (tryLoadGraphicsPSOFromCache(lo, hi, vsSym, psSym, outPSO)) {
        return true;
    }

    auto outLlvmCtx = std::make_unique<llvm::LLVMContext>();
    std::unique_ptr<llvm::Module> combinedMod;
    CachedPSOMeta meta;
    if (!synthesizeGraphicsPSOLLVM(
            desc, rootLayout, simdWidth, vsSym, psSym, *outLlvmCtx, combinedMod, meta,
            outPSO.vsInputs, outPSO.vsOutputs, outPSO.psInputs, outPSO.psOutputs, outError, meshStages)) {
        return false;
    }

    if (!linkAndFinalizeLLVMModule(*combinedMod, outError)) return false;

    populatePSOFromMeta(meta);
    return jitCompileGraphicsTier1AndScheduleTier2(
        std::move(combinedMod), std::move(outLlvmCtx), vsSym, psSym, meta, outPSO, outError);
}

} // namespace nacelle
