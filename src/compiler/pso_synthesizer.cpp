#include "pso_synthesizer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <unordered_map>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Verifier.h>

#include "dxil_loader.h"
#include "jit_engine.h"
#include "mlir_pipeline.h"
#include "simt_to_simd_pass.h"
#include "../common/format_utils.h"

namespace nacelle {

namespace {

std::string toLowerAscii(const std::string& s) {
    std::string out = s;
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::optional<uint32_t> tryGetUniformConstUInt(const SimdVal& sv) {
    if (sv.kind != SimdVal::Kind::Vector || !sv.vec) return std::nullopt;
    auto extractAttr = [](mlir::Attribute attr) -> std::optional<uint32_t> {
        if (auto splat = llvm::dyn_cast<mlir::SplatElementsAttr>(attr)) {
            if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(splat.getSplatValue<mlir::Attribute>())) {
                return static_cast<uint32_t>(ia.getValue().getZExtValue());
            }
        }
        if (auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(attr); dense && dense.isSplat()) {
            return static_cast<uint32_t>(dense.getSplatValue<llvm::APInt>().getZExtValue());
        }
        return std::nullopt;
    };
    if (auto c = sv.vec.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        if (auto v = extractAttr(c.getValue())) return v;
    }
    if (auto shuf = sv.vec.getDefiningOp<mlir::LLVM::ShuffleVectorOp>()) {
        if (auto ins = shuf.getV1().getDefiningOp<mlir::LLVM::InsertElementOp>()) {
            if (auto c = ins.getValue().getDefiningOp<mlir::LLVM::ConstantOp>()) {
                if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
                    return static_cast<uint32_t>(ia.getValue().getZExtValue());
                }
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

static mlir::Value llvmExtract(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value vec, uint32_t idx) {
    mlir::Type i32Ty = builder.getI32Type();
    mlir::Value idxVal = mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, builder.getI32IntegerAttr(idx));
    return mlir::LLVM::ExtractElementOp::create(builder, loc, vec, idxVal);
}

static mlir::Value llvmInsert(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value vec, mlir::Value elem, uint32_t idx) {
    mlir::Type i32Ty = builder.getI32Type();
    mlir::Value idxVal = mlir::LLVM::ConstantOp::create(builder, loc, i32Ty, builder.getI32IntegerAttr(idx));
    return mlir::LLVM::InsertElementOp::create(builder, loc, vec, elem, idxVal);
}

static mlir::Value llvmSplat(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemTy, mlir::Value scalar, uint32_t W) {
    auto vt = mlir::VectorType::get({W}, elemTy);
    mlir::Value p = mlir::LLVM::PoisonOp::create(builder, loc, vt);
    mlir::Value ins = llvmInsert(builder, loc, p, scalar, 0);
    llvm::SmallVector<int32_t, 16> zeros(W, 0);
    return mlir::LLVM::ShuffleVectorOp::create(builder, loc, ins, p, zeros);
}

struct IRHelpers {
    mlir::OpBuilder& b;
    mlir::Location loc;
    uint32_t W;

    mlir::Type i1Ty() const { return b.getI1Type(); }
    mlir::Type i8Ty() const { return b.getIntegerType(8); }
    mlir::Type i16Ty() const { return b.getIntegerType(16); }
    mlir::Type i32Ty() const { return b.getI32Type(); }
    mlir::Type i64Ty() const { return b.getI64Type(); }
    mlir::Type f16Ty() const { return b.getF16Type(); }
    mlir::FloatType f32Ty() const { return b.getF32Type(); }
    mlir::Type ptrTy() const { return mlir::LLVM::LLVMPointerType::get(b.getContext()); }
    mlir::VectorType vecTy(mlir::Type elem) const { return mlir::VectorType::get({W}, elem); }

    mlir::Value cInt(mlir::Type ty, int64_t v) const {
        return mlir::LLVM::ConstantOp::create(b, loc, ty, b.getIntegerAttr(ty, v));
    }
    mlir::Value cFlt(double v) const {
        return mlir::LLVM::ConstantOp::create(b, loc, f32Ty(), b.getF32FloatAttr(static_cast<float>(v)));
    }
    mlir::Value splatInt(mlir::Type ty, int64_t v) const {
        auto vt = vecTy(ty);
        return mlir::LLVM::ConstantOp::create(b, loc, vt, mlir::DenseElementsAttr::get(vt, b.getIntegerAttr(ty, v)));
    }
    mlir::Value splatFlt(double v) const {
        auto vt = vecTy(f32Ty());
        return mlir::LLVM::ConstantOp::create(b, loc, vt, mlir::DenseElementsAttr::get(vt, b.getF32FloatAttr(static_cast<float>(v))));
    }
    SimdVal splatScalar(mlir::Value scalar) const {
        SimdVal sv;
        sv.kind = SimdVal::Kind::Vector;
        sv.origType = scalar.getType();
        sv.vec = llvmSplat(b, loc, scalar.getType(), scalar, W);
        return sv;
    }
    mlir::Value allTrueMask() const { return splatInt(i1Ty(), 1); }

    mlir::Value laneIota() const {
        mlir::Value v = mlir::LLVM::PoisonOp::create(b, loc, vecTy(i32Ty()));
        for (uint32_t l = 0; l < W; ++l) {
            v = llvmInsert(b, loc, v, cInt(i32Ty(), l), l);
        }
        return v;
    }

    mlir::Value gepBytes(mlir::Value ptr, int64_t byteOff) const {
        if (byteOff == 0) return ptr;
        return mlir::LLVM::GEPOp::create(
            b, loc, ptrTy(), i8Ty(), ptr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(byteOff)});
    }
    mlir::Value gepBytesDyn(mlir::Value ptr, mlir::Value byteOffI64) const {
        return mlir::LLVM::GEPOp::create(
            b, loc, ptrTy(), i8Ty(), ptr, mlir::ValueRange{byteOffI64});
    }
    mlir::Value gepArray(mlir::Value basePtr, mlir::Type elemTy, mlir::Value idx) const {
        return mlir::LLVM::GEPOp::create(
            b, loc, ptrTy(), elemTy, basePtr, mlir::ValueRange{idx});
    }
    mlir::Value loadScalar(mlir::Type ty, mlir::Value ptr) const {
        unsigned align = llvm::isa<mlir::VectorType>(ty) ? 4u : 0u;
        return mlir::LLVM::LoadOp::create(b, loc, ty, ptr, align);
    }
    void storeScalar(mlir::Value val, mlir::Value ptr) const {
        unsigned align = llvm::isa<mlir::VectorType>(val.getType()) ? 4u : 0u;
        mlir::LLVM::StoreOp::create(b, loc, val, ptr, align);
    }

    mlir::Value orReduce(mlir::Value maskVec) const {
        return mlir::LLVM::vector_reduce_or::create(b, loc, b.getI1Type(), maskVec);
    }

    mlir::Value expandMaskQuads(mlir::Value mask) const {
        llvm::SmallVector<int32_t, 16> s0, s1, s2, s3;
        for (uint32_t q = 0; q < W; q += 4) {
            for (uint32_t k = 0; k < 4; ++k) {
                s0.push_back(q);
                s1.push_back(q + 1);
                s2.push_back(q + 2);
                s3.push_back(q + 3);
            }
        }
        mlir::Value m0 = mlir::LLVM::ShuffleVectorOp::create(b, loc, mask, mask, s0);
        mlir::Value m1 = mlir::LLVM::ShuffleVectorOp::create(b, loc, mask, mask, s1);
        mlir::Value m2 = mlir::LLVM::ShuffleVectorOp::create(b, loc, mask, mask, s2);
        mlir::Value m3 = mlir::LLVM::ShuffleVectorOp::create(b, loc, mask, mask, s3);
        mlir::Value o01 = mlir::LLVM::OrOp::create(b, loc, m0, m1);
        mlir::Value o23 = mlir::LLVM::OrOp::create(b, loc, m2, m3);
        return mlir::LLVM::OrOp::create(b, loc, o01, o23);
    }

    mlir::Value clamp01(mlir::Value vVec) const {
        mlir::Value zero = splatFlt(0.0);
        mlir::Value one = splatFlt(1.0);
        return mlir::LLVM::MinNumOp::create(
            b, loc, mlir::LLVM::MaxNumOp::create(b, loc, vVec, zero), one);
    }

    mlir::Value evalDepthCmp(D3D12_COMPARISON_FUNC func, mlir::Value newZ, mlir::Value oldZ) const {
        switch (func) {
        case D3D12_COMPARISON_FUNC_NEVER:
            return splatInt(i1Ty(), 0);
        case D3D12_COMPARISON_FUNC_LESS:
            return mlir::LLVM::FCmpOp::create(b, loc, mlir::LLVM::FCmpPredicate::olt, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_EQUAL:
            return mlir::LLVM::FCmpOp::create(b, loc, mlir::LLVM::FCmpPredicate::oeq, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_LESS_EQUAL:
            return mlir::LLVM::FCmpOp::create(b, loc, mlir::LLVM::FCmpPredicate::ole, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_GREATER:
            return mlir::LLVM::FCmpOp::create(b, loc, mlir::LLVM::FCmpPredicate::ogt, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_NOT_EQUAL:
            return mlir::LLVM::FCmpOp::create(b, loc, mlir::LLVM::FCmpPredicate::one, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
            return mlir::LLVM::FCmpOp::create(b, loc, mlir::LLVM::FCmpPredicate::oge, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_ALWAYS:
        default:
            return allTrueMask();
        }
    }

    mlir::Value evalStencilCmp(D3D12_COMPARISON_FUNC func, mlir::Value refVec, mlir::Value bufVec) const {
        switch (func) {
        case D3D12_COMPARISON_FUNC_NEVER:
            return splatInt(i1Ty(), 0);
        case D3D12_COMPARISON_FUNC_LESS:
            return mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::ult, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_EQUAL:
            return mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::eq, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_LESS_EQUAL:
            return mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::ule, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_GREATER:
            return mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::ugt, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_NOT_EQUAL:
            return mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::ne, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
            return mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::uge, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_ALWAYS:
        default:
            return allTrueMask();
        }
    }

    mlir::Value applyStencilOp(D3D12_STENCIL_OP sop, mlir::Value oldVal, mlir::Value refVal) const {
        switch (sop) {
        case D3D12_STENCIL_OP_KEEP:
            return oldVal;
        case D3D12_STENCIL_OP_ZERO:
            return splatInt(i8Ty(), 0);
        case D3D12_STENCIL_OP_REPLACE:
            return refVal;
        case D3D12_STENCIL_OP_INCR_SAT: {
            mlir::Value sat = mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::eq, oldVal, splatInt(i8Ty(), 255));
            mlir::Value inc = mlir::LLVM::AddOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
            return mlir::LLVM::SelectOp::create(b, loc, sat, oldVal, inc);
        }
        case D3D12_STENCIL_OP_DECR_SAT: {
            mlir::Value z = mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::eq, oldVal, splatInt(i8Ty(), 0));
            mlir::Value dec = mlir::LLVM::SubOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
            return mlir::LLVM::SelectOp::create(b, loc, z, oldVal, dec);
        }
        case D3D12_STENCIL_OP_INVERT:
            return mlir::LLVM::XOrOp::create(b, loc, oldVal, splatInt(i8Ty(), -1));
        case D3D12_STENCIL_OP_INCR:
            return mlir::LLVM::AddOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
        case D3D12_STENCIL_OP_DECR:
            return mlir::LLVM::SubOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
        default:
            return oldVal;
        }
    }

    mlir::Value evalBlendFactorRGB(
        D3D12_BLEND factor,
        const mlir::Value src[4],
        const mlir::Value dst[4],
        const mlir::Value blendFactor[4],
        int comp) const {
        mlir::Value one = splatFlt(1.0);
        switch (factor) {
        case D3D12_BLEND_ZERO: return splatFlt(0.0);
        case D3D12_BLEND_ONE: return one;
        case D3D12_BLEND_SRC_COLOR: return src[comp];
        case D3D12_BLEND_INV_SRC_COLOR: return mlir::LLVM::FSubOp::create(b, loc, one, src[comp]);
        case D3D12_BLEND_SRC_ALPHA: return src[3];
        case D3D12_BLEND_INV_SRC_ALPHA: return mlir::LLVM::FSubOp::create(b, loc, one, src[3]);
        case D3D12_BLEND_DEST_ALPHA: return dst[3];
        case D3D12_BLEND_INV_DEST_ALPHA: return mlir::LLVM::FSubOp::create(b, loc, one, dst[3]);
        case D3D12_BLEND_DEST_COLOR: return dst[comp];
        case D3D12_BLEND_INV_DEST_COLOR: return mlir::LLVM::FSubOp::create(b, loc, one, dst[comp]);
        case D3D12_BLEND_SRC_ALPHA_SAT: {
            if (comp == 3) return one;
            mlir::Value invDstA = mlir::LLVM::FSubOp::create(b, loc, one, dst[3]);
            return mlir::LLVM::MinNumOp::create(b, loc, src[3], invDstA);
        }
        case D3D12_BLEND_BLEND_FACTOR: return blendFactor[comp];
        case D3D12_BLEND_INV_BLEND_FACTOR: return mlir::LLVM::FSubOp::create(b, loc, one, blendFactor[comp]);
        default: return one;
        }
    }

    mlir::Value evalBlendOp(D3D12_BLEND_OP op, mlir::Value sTerm, mlir::Value dTerm) const {
        switch (op) {
        case D3D12_BLEND_OP_ADD: return mlir::LLVM::FAddOp::create(b, loc, sTerm, dTerm);
        case D3D12_BLEND_OP_SUBTRACT: return mlir::LLVM::FSubOp::create(b, loc, sTerm, dTerm);
        case D3D12_BLEND_OP_REV_SUBTRACT: return mlir::LLVM::FSubOp::create(b, loc, dTerm, sTerm);
        case D3D12_BLEND_OP_MIN: return mlir::LLVM::MinNumOp::create(b, loc, sTerm, dTerm);
        case D3D12_BLEND_OP_MAX: return mlir::LLVM::MaxNumOp::create(b, loc, sTerm, dTerm);
        default: return mlir::LLVM::FAddOp::create(b, loc, sTerm, dTerm);
        }
    }
};

void cloneGlobalsIntoModule(mlir::ModuleOp srcMod, mlir::ModuleOp dstMod) {
    mlir::OpBuilder dstBuilder(dstMod.getBody(), dstMod.getBody()->begin());
    for (auto gv : srcMod.getOps<mlir::LLVM::GlobalOp>()) {
        if (!dstMod.lookupSymbol(gv.getSymName())) {
            dstBuilder.clone(*gv.getOperation());
        }
    }
}

struct LinkedVSSigSlot {
    int32_t psTargetRow = -1;
    uint32_t psStartCol = 0;
};

struct PredecodedVertexElement {
    bool valid = false;
    mlir::Value compsF32[4]{};
    mlir::Value compsI32[4]{};
};

PredecodedVertexElement predecodeVertexElementSIMD(
    IRHelpers& h,
    const D3D12_INPUT_ELEMENT_DESC& el,
    mlir::Value activeMask,
    mlir::Value vbBasePtr,
    mlir::Value vbStrideVec,
    mlir::Value vbSizeVec,
    mlir::Value idxVec,
    uint32_t byteOffsetInElem,
    uint8_t neededColsMask,
    llvm::ArrayRef<mlir::Value> preloadedWords = {},
    uint32_t preloadedBaseOff = 0,
    mlir::Value preloadedLaneOkMask = mlir::Value()) {
    PredecodedVertexElement res;
    res.valid = true;
    mlir::OpBuilder& b = h.b;
    mlir::Location loc = h.loc;
    auto i8Ty = h.i8Ty();
    auto i16Ty = h.i16Ty();
    auto i32Ty = h.i32Ty();
    auto i64Ty = h.i64Ty();
    auto f16Ty = h.f16Ty();
    auto f32Ty = h.f32Ty();
    auto ptrTy = h.ptrTy();

    mlir::Value zeroI32 = h.splatInt(i32Ty, 0);
    mlir::Value zeroF32 = h.splatFlt(0.0);
    mlir::Value oneF32 = h.splatFlt(1.0);
    for (uint32_t c = 0; c < 4; ++c) {
        res.compsF32[c] = (c == 3) ? oneF32 : zeroF32;
        res.compsI32[c] = zeroI32;
    }
    if (neededColsMask == 0) return res;

    uint32_t fmtBytes = formatByteSizeForVertex(el.Format);
    const bool usePreloaded = !preloadedWords.empty();
    mlir::Value off64Vec;
    mlir::Value laneOkMask = preloadedLaneOkMask;
    if (!usePreloaded) {
        mlir::Value idx64Vec = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(i64Ty), idxVec);
        mlir::Value str64Vec = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(i64Ty), vbStrideVec);
        off64Vec = mlir::LLVM::AddOp::create(
            b, loc,
            mlir::LLVM::MulOp::create(b, loc, idx64Vec, str64Vec),
            h.splatInt(i64Ty, byteOffsetInElem));
        mlir::Value end64Vec = mlir::LLVM::AddOp::create(b, loc, off64Vec, h.splatInt(i64Ty, fmtBytes));
        mlir::Value sz64Vec = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(i64Ty), vbSizeVec);
        mlir::Value inBoundsVec = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::ule, end64Vec, sz64Vec);
        mlir::Value ptrNotNull = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::ne, vbBasePtr, mlir::LLVM::ZeroOp::create(b, loc, ptrTy));
        mlir::Value ptrNotNullVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(h.i1Ty())).getElementType(), ptrNotNull, llvm::cast<mlir::VectorType>(h.vecTy(h.i1Ty())).getNumElements());
        laneOkMask = mlir::LLVM::AndOp::create(
            b, loc, mlir::LLVM::AndOp::create(b, loc, activeMask, ptrNotNullVec), inBoundsVec);
    }

    auto gatherElem = [&](mlir::Type elemTy, int64_t deltaBytes, mlir::Value passThruVec) -> mlir::Value {
        if (usePreloaded && (elemTy == i32Ty || elemTy == f32Ty) &&
            byteOffsetInElem + deltaBytes >= preloadedBaseOff &&
            ((byteOffsetInElem + deltaBytes - preloadedBaseOff) % 4) == 0) {
            size_t wordIdx = static_cast<size_t>((byteOffsetInElem + deltaBytes - preloadedBaseOff) / 4);
            if (wordIdx < preloadedWords.size()) {
                mlir::Value w = preloadedWords[wordIdx];
                if (elemTy == f32Ty) {
                    w = mlir::LLVM::BitcastOp::create(b, loc, h.vecTy(f32Ty), w);
                    if (passThruVec == oneF32) {
                        w = mlir::LLVM::SelectOp::create(b, loc, laneOkMask, w, oneF32);
                    }
                }
                return w;
            }
        }
        mlir::Value compBase = h.gepBytes(vbBasePtr, deltaBytes);
        mlir::Value ptrVec = mlir::LLVM::GEPOp::create(
            b, loc, h.vecTy(ptrTy), i8Ty, compBase, mlir::ValueRange{off64Vec});
        return mlir::LLVM::masked_gather::create(
            b, loc, h.vecTy(elemTy), ptrVec, laneOkMask, mlir::ValueRange{passThruVec}, /*alignment=*/1);
    };

    auto setFromFloat = [&](uint32_t c, mlir::Value fv, bool applyOobAlphaDefault) {
        if (c == 3 && applyOobAlphaDefault) {
            fv = mlir::LLVM::SelectOp::create(b, loc, laneOkMask, fv, oneF32);
        }
        res.compsF32[c] = fv;
        mlir::Value asInt = mlir::LLVM::BitcastOp::create(b, loc, h.vecTy(i32Ty), fv);
        res.compsI32[c] = mlir::LLVM::SelectOp::create(b, loc, laneOkMask, asInt, zeroI32);
    };

    auto setFromInt = [&](uint32_t c, mlir::Value iv) {
        res.compsI32[c] = iv;
        mlir::Value asFlt = mlir::LLVM::BitcastOp::create(b, loc, h.vecTy(f32Ty), iv);
        res.compsF32[c] = (c == 3) ? mlir::LLVM::SelectOp::create(b, loc, laneOkMask, asFlt, oneF32).getResult() : asFlt;
    };

    switch (el.Format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT: {
        uint32_t maxComps = fmtBytes / 4;
        for (uint32_t c = 0; c < maxComps; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            mlir::Value defF = (c == 3) ? oneF32 : zeroF32;
            mlir::Value fv = gatherElem(f32Ty, static_cast<int64_t>(c * 4), defF);
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
            mlir::Value iv = gatherElem(i32Ty, static_cast<int64_t>(c * 4), zeroI32);
            setFromInt(c, iv);
        }
        break;
    }
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16_FLOAT: {
        if (el.Format == DXGI_FORMAT_R16_FLOAT) {
            if (neededColsMask & 1u) {
                mlir::Value zeroF16 = mlir::LLVM::BitcastOp::create(b, loc, h.vecTy(f16Ty), h.splatInt(i16Ty, 0));
                mlir::Value hv = gatherElem(f16Ty, 0, zeroF16);
                setFromFloat(0, mlir::LLVM::FPExtOp::create(b, loc, h.vecTy(f32Ty), hv), false);
            }
        } else {
            auto unpackPair = [&](mlir::Value raw32, uint32_t c0, uint32_t c1) {
                if (neededColsMask & (1u << c0)) {
                    mlir::Value lo16 = mlir::LLVM::TruncOp::create(b, loc, h.vecTy(i16Ty), raw32);
                    mlir::Value hv0 = mlir::LLVM::BitcastOp::create(b, loc, h.vecTy(f16Ty), lo16);
                    setFromFloat(c0, mlir::LLVM::FPExtOp::create(b, loc, h.vecTy(f32Ty), hv0), c0 == 3);
                }
                if (neededColsMask & (1u << c1)) {
                    mlir::Value hi32 = mlir::LLVM::LShrOp::create(b, loc, raw32, h.splatInt(i32Ty, 16));
                    mlir::Value hi16 = mlir::LLVM::TruncOp::create(b, loc, h.vecTy(i16Ty), hi32);
                    mlir::Value hv1 = mlir::LLVM::BitcastOp::create(b, loc, h.vecTy(f16Ty), hi16);
                    setFromFloat(c1, mlir::LLVM::FPExtOp::create(b, loc, h.vecTy(f32Ty), hv1), c1 == 3);
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
        auto decode16 = [&](uint32_t c, mlir::Value raw32, bool highHalf) {
            mlir::Value v32;
            if (isSigned) {
                mlir::Value shl = highHalf ? raw32 : mlir::LLVM::ShlOp::create(b, loc, raw32, h.splatInt(i32Ty, 16)).getResult();
                v32 = mlir::LLVM::AShrOp::create(b, loc, shl, h.splatInt(i32Ty, 16));
            } else {
                v32 = highHalf ? mlir::LLVM::LShrOp::create(b, loc, raw32, h.splatInt(i32Ty, 16)).getResult()
                               : mlir::LLVM::AndOp::create(b, loc, raw32, h.splatInt(i32Ty, 0xFFFF)).getResult();
            }
            if (isNorm) {
                mlir::Value fv = isSigned
                                     ? mlir::LLVM::SIToFPOp::create(b, loc, h.vecTy(f32Ty), v32).getResult()
                                     : mlir::LLVM::UIToFPOp::create(b, loc, h.vecTy(f32Ty), v32).getResult();
                fv = mlir::LLVM::FMulOp::create(b, loc, fv, h.splatFlt(isSigned ? (1.0 / 32767.0) : (1.0 / 65535.0)));
                if (isSigned) fv = mlir::LLVM::MaxNumOp::create(b, loc, fv, h.splatFlt(-1.0));
                setFromFloat(c, fv, c == 3);
            } else {
                setFromInt(c, v32);
            }
        };
        if (fmtBytes == 2) {
            if (neededColsMask & 1u) {
                mlir::Value u16 = gatherElem(i16Ty, 0, h.splatInt(i16Ty, 0));
                mlir::Value raw32 = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(i32Ty), u16);
                decode16(0, raw32, false);
            }
        } else {
            if (neededColsMask & 0x3u) {
                mlir::Value rawLo = gatherElem(i32Ty, 0, zeroI32);
                if (neededColsMask & 1u) decode16(0, rawLo, false);
                if (neededColsMask & 2u) decode16(1, rawLo, true);
            }
            if (fmtBytes == 8 && (neededColsMask & 0xCu)) {
                mlir::Value rawHi = gatherElem(i32Ty, 4, zeroI32);
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
        mlir::Value raw32;
        if (fmtBytes == 4) {
            raw32 = gatherElem(i32Ty, 0, zeroI32);
        } else if (fmtBytes == 2) {
            raw32 = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(i32Ty), gatherElem(i16Ty, 0, h.splatInt(i16Ty, 0)));
        } else {
            raw32 = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(i32Ty), gatherElem(i8Ty, 0, h.splatInt(i8Ty, 0)));
        }
        for (uint32_t c = 0; c < maxComps; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            if (isBGRX && c == 3) {
                setFromFloat(3, oneF32, false);
                continue;
            }
            uint32_t byteIdx = (isBGRA && c == 0) ? 2u : ((isBGRA && c == 2) ? 0u : c);
            mlir::Value v32;
            if (isSigned) {
                uint32_t shlAmt = 24u - byteIdx * 8u;
                mlir::Value shl = (shlAmt > 0)
                                      ? mlir::LLVM::ShlOp::create(b, loc, raw32, h.splatInt(i32Ty, shlAmt)).getResult()
                                      : raw32;
                v32 = mlir::LLVM::AShrOp::create(b, loc, shl, h.splatInt(i32Ty, 24));
            } else {
                mlir::Value shr = (byteIdx > 0)
                                      ? mlir::LLVM::LShrOp::create(b, loc, raw32, h.splatInt(i32Ty, byteIdx * 8u)).getResult()
                                      : raw32;
                v32 = (byteIdx == 3) ? shr : mlir::LLVM::AndOp::create(b, loc, shr, h.splatInt(i32Ty, 0xFF)).getResult();
            }
            if (isNorm) {
                mlir::Value fv = isSigned
                                     ? mlir::LLVM::SIToFPOp::create(b, loc, h.vecTy(f32Ty), v32).getResult()
                                     : mlir::LLVM::UIToFPOp::create(b, loc, h.vecTy(f32Ty), v32).getResult();
                fv = mlir::LLVM::FMulOp::create(b, loc, fv, h.splatFlt(isSigned ? (1.0 / 127.0) : (1.0 / 255.0)));
                if (isSigned) fv = mlir::LLVM::MaxNumOp::create(b, loc, fv, h.splatFlt(-1.0));
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
        mlir::Value raw32 = gatherElem(i32Ty, 0, zeroI32);
        mlir::Value mask10 = h.splatInt(i32Ty, 0x3FF);
        for (uint32_t c = 0; c < 4; ++c) {
            if (!(neededColsMask & (1u << c))) continue;
            mlir::Value u32;
            if (c == 0) {
                u32 = mlir::LLVM::AndOp::create(b, loc, raw32, mask10);
            } else if (c == 1) {
                u32 = mlir::LLVM::AndOp::create(b, loc, mlir::LLVM::LShrOp::create(b, loc, raw32, h.splatInt(i32Ty, 10)), mask10);
            } else if (c == 2) {
                u32 = mlir::LLVM::AndOp::create(b, loc, mlir::LLVM::LShrOp::create(b, loc, raw32, h.splatInt(i32Ty, 20)), mask10);
            } else {
                u32 = mlir::LLVM::LShrOp::create(b, loc, raw32, h.splatInt(i32Ty, 30));
            }
            if (isNorm) {
                mlir::Value fv = mlir::LLVM::UIToFPOp::create(b, loc, h.vecTy(f32Ty), u32);
                fv = mlir::LLVM::FMulOp::create(b, loc, fv, h.splatFlt(c == 3 ? (1.0 / 3.0) : (1.0 / 1023.0)));
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
    mlir::ModuleOp targetMod,
    mlir::func::FuncOp vsSrcFunc,
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
    if (!vsSrcFunc || vsSrcFunc.getBody().empty()) {
        outError = "Vertex shader entry function missing after MLIR SCF lift";
        return false;
    }

    mlir::Location loc = targetMod.getLoc();
    mlir::OpBuilder b(targetMod.getBody(), targetMod.getBody()->end());
    IRHelpers h{b, loc, simdWidth};
    auto ptrTy = h.ptrTy();
    auto i32Ty = h.i32Ty();
    auto fnTy = b.getFunctionType({ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy}, {});
    auto vsFn = mlir::func::FuncOp::create(b, loc, vsSym, fnTy);
    mlir::Block* entry = vsFn.addEntryBlock();
    b.setInsertionPointToEnd(entry);

    mlir::Value rootCtx = entry->getArgument(0);
    mlir::Value vertexIdsPtr = entry->getArgument(1);
    mlir::Value instanceIdsPtr = entry->getArgument(2);
    mlir::Value activeCount = entry->getArgument(3);
    mlir::Value outPosXYZW = entry->getArgument(4);
    mlir::Value outAttrs = entry->getArgument(5);

    mlir::Value acVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), activeCount, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());
    mlir::Value activeMask = mlir::LLVM::ICmpOp::create(
        b, loc, mlir::LLVM::ICmpPredicate::ult, h.laneIota(), acVec);

    // Callers pad tail lanes of vertexIdsPtr & instanceIdsPtr to W; load directly without SelectOp
    // so LLVM does not scalarize into W conditional scalar loads.
    mlir::Value zeroI32Vec = h.splatInt(i32Ty, 0);
    mlir::Value vidVec = h.loadScalar(h.vecTy(i32Ty), vertexIdsPtr);
    mlir::Value iidVec = h.loadScalar(h.vecTy(i32Ty), instanceIdsPtr);

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
    mlir::Value vbBasePtrs[16]{};
    mlir::Value vbStrideVecs[16]{};
    mlir::Value vbSizeVecs[16]{};

    struct CoalescedSlotData {
        bool valid = false;
        uint32_t baseByteOff = 0;
        mlir::Value laneOkMask;
        llvm::SmallVector<mlir::Value, 16> words;
    };
    CoalescedSlotData coalescedSlots[16];

    for (UINT slot = 0; slot < 16; ++slot) {
        bool anyNeeded = false;
        bool allPerVertex4Aligned = true;
        uint32_t minOff = UINT32_MAX;
        uint32_t maxEnd = 0;
        uint32_t elemCount = 0;
        for (UINT i = 0; i < inputLayout.NumElements && inputLayout.pInputElementDescs; ++i) {
            if (elemNeededCols[i] == 0) continue;
            const auto& el = inputLayout.pInputElementDescs[i];
            if (std::min<UINT>(el.InputSlot, 15) != slot) continue;
            anyNeeded = true;
            ++elemCount;
            uint32_t fmtBytes = formatByteSizeForVertex(el.Format);
            if (el.InputSlotClass != D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA ||
                (elementByteOffsets[i] % 4) != 0 || (fmtBytes % 4) != 0 || fmtBytes == 0) {
                allPerVertex4Aligned = false;
                break;
            }
            minOff = std::min(minOff, elementByteOffsets[i]);
            maxEnd = std::max(maxEnd, elementByteOffsets[i] + fmtBytes);
        }
        if (!anyNeeded) continue;

        mlir::Value addrInt = h.loadScalar(
            h.i64Ty(), h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_buffers) + slot * 8)));
        vbBasePtrs[slot] = mlir::LLVM::IntToPtrOp::create(b, loc, ptrTy, addrInt);
        mlir::Value str32 = h.loadScalar(
            i32Ty, h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_strides) + slot * 4)));
        mlir::Value sz32 = h.loadScalar(
            i32Ty, h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_sizes) + slot * 4)));
        vbStrideVecs[slot] = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), str32, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());
        vbSizeVecs[slot] = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), sz32, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());

        uint32_t spanBytes = (maxEnd > minOff) ? (maxEnd - minOff) : 0;
        if (allPerVertex4Aligned && spanBytes >= 8 && spanBytes <= 64 && elemCount >= 1) {
            uint32_t numWords = spanBytes / 4;
            mlir::Value idx64Vec = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(h.i64Ty()), vidVec);
            mlir::Value str64Vec = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(h.i64Ty()), vbStrideVecs[slot]);
            mlir::Value off64Vec = mlir::LLVM::MulOp::create(b, loc, idx64Vec, str64Vec);
            mlir::Value end64Vec = mlir::LLVM::AddOp::create(b, loc, off64Vec, h.splatInt(h.i64Ty(), maxEnd));
            mlir::Value sz64Vec = mlir::LLVM::ZExtOp::create(b, loc, h.vecTy(h.i64Ty()), vbSizeVecs[slot]);
            mlir::Value inBoundsVec = mlir::LLVM::ICmpOp::create(b, loc, mlir::LLVM::ICmpPredicate::ule, end64Vec, sz64Vec);
            mlir::Value ptrNotNull = mlir::LLVM::ICmpOp::create(
                b, loc, mlir::LLVM::ICmpPredicate::ne, vbBasePtrs[slot], mlir::LLVM::ZeroOp::create(b, loc, ptrTy));
            mlir::Value ptrNotNullVec = llvmSplat(
                b, loc, llvm::cast<mlir::VectorType>(h.vecTy(h.i1Ty())).getElementType(), ptrNotNull, simdWidth);
            mlir::Value slotOkMask = mlir::LLVM::AndOp::create(
                b, loc, mlir::LLVM::AndOp::create(b, loc, activeMask, ptrNotNullVec), inBoundsVec);

            mlir::Value allOk = mlir::LLVM::vector_reduce_and::create(b, loc, h.i1Ty(), slotOkMask);
            mlir::Value zeroI32Splat = h.splatInt(i32Ty, 0);
            llvm::SmallVector<mlir::Type, 16> resTypes(numWords, h.vecTy(i32Ty));

            auto emitLaneLoads = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Value baseOff64,
                                     llvm::SmallVectorImpl<mlir::Value>& outWordVecs, uint32_t lane) {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value vtxPtr = lh.gepBytesDyn(
                    vbBasePtrs[slot],
                    mlir::LLVM::AddOp::create(lb, lloc, baseOff64, lh.cInt(h.i64Ty(), minOff)));
                uint32_t w = 0;
                while (w < numWords) {
                    uint32_t rem = numWords - w;
                    uint32_t chunk = (rem >= 4) ? 4 : ((rem >= 2) ? 2 : 1);
                    mlir::Value ptr = (w == 0) ? vtxPtr : lh.gepBytes(vtxPtr, static_cast<int64_t>(w * 4));
                    if (chunk == 1) {
                        mlir::Value val = mlir::LLVM::LoadOp::create(lb, lloc, i32Ty, ptr, /*alignment=*/4);
                        outWordVecs[w] = llvmInsert(lb, lloc, outWordVecs[w], val, lane);
                    } else {
                        auto cTy = mlir::VectorType::get({static_cast<int64_t>(chunk)}, i32Ty);
                        mlir::Value cVal = mlir::LLVM::LoadOp::create(lb, lloc, cTy, ptr, /*alignment=*/4);
                        for (uint32_t k = 0; k < chunk; ++k) {
                            mlir::Value elem = llvmExtract(lb, lloc, cVal, k);
                            outWordVecs[w + k] = llvmInsert(lb, lloc, outWordVecs[w + k], elem, lane);
                        }
                    }
                    w += chunk;
                }
            };

            auto ifOp = mlir::scf::IfOp::create(b, loc, resTypes, allOk, /*withElseRegion=*/true);
            {
                mlir::OpBuilder::InsertionGuard g(b);
                b.setInsertionPointToStart(&ifOp.getThenRegion().front());
                llvm::SmallVector<mlir::Value, 16> wordVecs(numWords, mlir::LLVM::PoisonOp::create(b, loc, h.vecTy(i32Ty)));
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value offL = llvmExtract(b, loc, off64Vec, l);
                    emitLaneLoads(b, loc, offL, wordVecs, l);
                }
                mlir::scf::YieldOp::create(b, loc, wordVecs);

                b.setInsertionPointToStart(&ifOp.getElseRegion().front());
                llvm::SmallVector<mlir::Value, 16> elseVecs(numWords, zeroI32Splat);
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value mL = llvmExtract(b, loc, slotOkMask, l);
                    auto laneIf = mlir::scf::IfOp::create(b, loc, resTypes, mL, /*withElseRegion=*/true);
                    {
                        mlir::OpBuilder::InsertionGuard lg(b);
                        b.setInsertionPointToStart(&laneIf.getThenRegion().front());
                        llvm::SmallVector<mlir::Value, 16> cur = elseVecs;
                        mlir::Value offL = llvmExtract(b, loc, off64Vec, l);
                        emitLaneLoads(b, loc, offL, cur, l);
                        mlir::scf::YieldOp::create(b, loc, cur);

                        b.setInsertionPointToStart(&laneIf.getElseRegion().front());
                        mlir::scf::YieldOp::create(b, loc, elseVecs);
                    }
                    elseVecs.assign(laneIf.getResults().begin(), laneIf.getResults().end());
                }
                mlir::scf::YieldOp::create(b, loc, elseVecs);
            }

            coalescedSlots[slot].valid = true;
            coalescedSlots[slot].baseByteOff = minOff;
            coalescedSlots[slot].laneOkMask = slotOkMask;
            coalescedSlots[slot].words.assign(ifOp.getResults().begin(), ifOp.getResults().end());
        }
    }

    std::vector<PredecodedVertexElement> predecodedElems(inputLayout.NumElements);
    for (UINT i = 0; i < inputLayout.NumElements && inputLayout.pInputElementDescs; ++i) {
        if (elemNeededCols[i] == 0) continue;
        const auto& el = inputLayout.pInputElementDescs[i];
        UINT slot = std::min<UINT>(el.InputSlot, 15);
        mlir::Value idxVec = vidVec;
        if (el.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA) {
            if (el.InstanceDataStepRate == 0) {
                idxVec = zeroI32Vec;
            } else if (el.InstanceDataStepRate == 1) {
                idxVec = iidVec;
            } else {
                idxVec = mlir::LLVM::UDivOp::create(
                    b, loc, iidVec, h.splatInt(i32Ty, el.InstanceDataStepRate));
            }
        }
        if (coalescedSlots[slot].valid) {
            predecodedElems[i] = predecodeVertexElementSIMD(
                h, el, activeMask, vbBasePtrs[slot], vbStrideVecs[slot], vbSizeVecs[slot],
                idxVec, elementByteOffsets[i], elemNeededCols[i],
                coalescedSlots[slot].words, coalescedSlots[slot].baseByteOff, coalescedSlots[slot].laneOkMask);
        } else {
            predecodedElems[i] = predecodeVertexElementSIMD(
                h, el, activeMask, vbBasePtrs[slot], vbStrideVecs[slot], vbSizeVecs[slot],
                idxVec, elementByteOffsets[i], elemNeededCols[i]);
        }
    }

    StageHooks vsHooks;
    vsHooks.isPixelShader = false;
    vsHooks.entryBlock = entry;
    vsHooks.loadInput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                            const SimdVal&, uint32_t col, mlir::Type scalarTy, SimdVal& out) -> bool {
        const DxilSigRecord* sig = vsLoaded.findInputSig(sigId);
        if (!sig) return false;
        IRHelpers lh{lb, lloc, simdWidth};
        bool wantInt = mlir::isa<mlir::IntegerType>(scalarTy);

        if (sig->systemValueType == 1) { // SV_VertexID
            mlir::Value v = wantInt ? vidVec : mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), vidVec).getResult();
            if (wantInt && scalarTy != i32Ty) v = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
            out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
            return true;
        }
        if (sig->systemValueType == 2) { // SV_InstanceID
            mlir::Value v = wantInt ? iidVec : mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), iidVec).getResult();
            if (wantInt && scalarTy != i32Ty) v = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
            out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
            return true;
        }

        auto eit = sigToElemIdx.find(sigId);
        if (eit != sigToElemIdx.end() && col < 4 && predecodedElems[eit->second].valid) {
            const auto& pe = predecodedElems[eit->second];
            mlir::Value decoded = wantInt ? pe.compsI32[col] : pe.compsF32[col];
            if (wantInt && scalarTy != i32Ty) {
                decoded = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), decoded);
            } else if (!wantInt && scalarTy != lh.f32Ty()) {
                decoded = mlir::LLVM::FPTruncOp::create(lb, lloc, lh.vecTy(scalarTy), decoded);
            }
            out = {SimdVal::Kind::Vector, scalarTy, decoded, {}, {}};
            return true;
        }
        return false;
    };

    vsHooks.storeOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                              const SimdVal& row, uint32_t col, const SimdVal& val, mlir::Value writeMask) -> bool {
        const DxilSigRecord* sig = vsLoaded.findOutputSig(sigId);
        if (!sig || val.kind != SimdVal::Kind::Vector) return true;
        IRHelpers sh{sb, sloc, simdWidth};
        mlir::Value vVec = val.vec;
        mlir::Type elemTy = llvm::cast<mlir::VectorType>(vVec.getType()).getElementType();
        if (elemTy != sh.f32Ty()) {
            if (mlir::isa<mlir::IntegerType>(elemTy)) {
                if (elemTy.getIntOrFloatBitWidth() < 32) {
                    vVec = mlir::LLVM::ZExtOp::create(sb, sloc, sh.vecTy(i32Ty), vVec);
                }
                vVec = mlir::LLVM::BitcastOp::create(sb, sloc, sh.vecTy(sh.f32Ty()), vVec);
            } else {
                vVec = mlir::LLVM::FPExtOp::create(sb, sloc, sh.vecTy(sh.f32Ty()), vVec);
            }
        }

        auto storeVecSOA = [&](mlir::Value dstPtr) {
            if (!writeMask || writeMask == activeMask) {
                mlir::LLVM::StoreOp::create(sb, sloc, vVec, dstPtr, /*alignment=*/4);
            } else {
                mlir::Value prev = sh.loadScalar(sh.vecTy(sh.f32Ty()), dstPtr);
                mlir::Value merged = mlir::LLVM::SelectOp::create(sb, sloc, writeMask, vVec, prev);
                mlir::LLVM::StoreOp::create(sb, sloc, merged, dstPtr, /*alignment=*/4);
            }
        };

        auto storeAttrAtSlot = [&](uint32_t baseSlot) {
            auto rowConst = tryGetUniformConstUInt(row);
            if (rowConst.has_value()) {
                uint32_t scalarSlot = baseSlot + (*rowConst) * 4 + col;
                mlir::Value dst = sh.gepBytes(outAttrs, static_cast<int64_t>(scalarSlot * simdWidth * 4));
                storeVecSOA(dst);
            } else {
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value mL = llvmExtract(sb, sloc, writeMask ? writeMask : activeMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::scf::IfOp::create(sb, sloc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        IRHelpers nh{nb, nloc, simdWidth};
                        mlir::Value rL = llvmExtract(nb, nloc, row.vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value r32 = (rL.getType() == i32Ty)
                                              ? rL
                                              : mlir::LLVM::ZExtOp::create(nb, nloc, i32Ty, rL).getResult();
                        mlir::Value slot32 = mlir::LLVM::AddOp::create(
                            nb, nloc, nh.cInt(i32Ty, static_cast<int64_t>(baseSlot + col)),
                            mlir::LLVM::MulOp::create(nb, nloc, r32, nh.cInt(i32Ty, 4)));
                        mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                            nb, nloc, mlir::LLVM::MulOp::create(nb, nloc, slot32, nh.cInt(i32Ty, simdWidth)),
                            nh.cInt(i32Ty, static_cast<int64_t>(l)));
                        mlir::Value byteOff = mlir::LLVM::MulOp::create(
                            nb, nloc, mlir::LLVM::ZExtOp::create(nb, nloc, nh.i64Ty(), elemIdx), nh.cInt(nh.i64Ty(), 4));
                        mlir::Value vL = llvmExtract(nb, nloc, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::LLVM::StoreOp::create(nb, nloc, vL, nh.gepBytesDyn(outAttrs, byteOff));
                        mlir::scf::YieldOp::create(nb, nloc);
                    });
                }
            }
        };

        if (sig->systemValueType == 3) { // SV_Position -> outPosXYZW[col * W + lane]
            if (col < 4) {
                mlir::Value dst = sh.gepBytes(outPosXYZW, static_cast<int64_t>(col * simdWidth * 4));
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

    mlir::Value vsRealMask = activeMask;
    if (!packetizeRegionWithHooks(b, loc, targetMod, vsSrcFunc.getBody(), vsLoaded, vsEffectiveLayout,
                                  simdWidth, rootCtx, activeMask, vsRealMask, vsHooks, outError)) {
        return false;
    }
    mlir::func::ReturnOp::create(b, loc);
    return true;
}

} // namespace

bool synthesizeVertexStageMLIR(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    mlir::MLIRContext& mlirCtx,
    mlir::OwningOpRef<mlir::ModuleOp>& outVSMod,
    CachedVSMeta& outVSMeta,
    std::vector<ShaderSigElement>& outVSInputs,
    std::vector<ShaderSigElement>& outVSOutputs,
    std::vector<DxilSigRecord>& outVSOutputRecs,
    std::string& outError) {
    if (!desc.VS.pShaderBytecode || desc.VS.BytecodeLength == 0) {
        outError = "Graphics PSO requires a valid Vertex Shader bytecode";
        return false;
    }

    auto vsLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader vsLoaded;
    if (!loadAndUpgradeDXIL(desc.VS, *vsLlvmCtx, vsLoaded, outError)) {
        return false;
    }

    const RootSignatureLayout& vsEffectiveLayout =
        (rootLayout.params.empty() && rootLayout.staticSamplers.empty() && vsLoaded.hasEmbeddedRootSig)
            ? vsLoaded.embeddedRootSig
            : rootLayout;

    auto vsMlirMod = importAndLiftDXILToSCF(std::move(vsLoaded.module), vsLoaded.entryName, mlirCtx, outError);
    if (!vsMlirMod) return false;

    configureNacelleMLIRContext(mlirCtx);
    mlir::Location loc = mlir::UnknownLoc::get(&mlirCtx);
    outVSMod = mlir::ModuleOp::create(loc);
    mlir::ModuleOp vsMod = *outVSMod;
    cloneGlobalsIntoModule(*vsMlirMod, vsMod);

    outVSInputs = convertSigElements(vsLoaded.inputSigs);
    outVSOutputs = convertSigElements(vsLoaded.outputSigs);
    outVSOutputRecs = vsLoaded.outputSigs;

    CanonicalVSLayoutInfo canonical = computeCanonicalVSLayout(vsLoaded.outputSigs);
    auto vsSrcFunc = vsMlirMod->lookupSymbol<mlir::func::FuncOp>(vsLoaded.entryName);
    if (!emitVertexStageFunction(
            vsMod, vsSrcFunc, vsLoaded, vsEffectiveLayout, desc.InputLayout,
            simdWidth, vsSym, /*hasTess=*/false, /*hasGS=*/false, &canonical, nullptr, outError)) {
        return false;
    }

    llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 16> deadDxOps;
    for (auto fnOp : vsMod.getOps<mlir::LLVM::LLVMFuncOp>()) {
        if (fnOp.isDeclaration() && fnOp.getName().starts_with("dx.op.") && fnOp.use_empty()) {
            deadDxOps.push_back(fnOp);
        }
    }
    for (auto fnOp : deadDxOps) fnOp.erase();

    if (mlir::failed(mlir::verify(vsMod))) {
        outError = "MLIR verification failed after Vertex Stage synthesis";
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

bool synthesizeGraphicsPSOMLIR(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    const std::string& psSym,
    mlir::MLIRContext& mlirCtx,
    mlir::OwningOpRef<mlir::ModuleOp>& outCombinedMod,
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

    auto msLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader msLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> msMlirMod;
    auto asLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader asLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> asMlirMod;
    if (hasMS) {
        if (!loadAndUpgradeDXIL(meshStages.MS, *msLlvmCtx, msLoaded, outError)) {
            return false;
        }
        msMlirMod = importAndLiftDXILToSCF(std::move(msLoaded.module), msLoaded.entryName, mlirCtx, outError);
        if (!msMlirMod) return false;
        if (hasAS) {
            if (!loadAndUpgradeDXIL(meshStages.AS, *asLlvmCtx, asLoaded, outError)) {
                return false;
            }
            asMlirMod = importAndLiftDXILToSCF(std::move(asLoaded.module), asLoaded.entryName, mlirCtx, outError);
            if (!asMlirMod) return false;
        }
    }

    const bool emptyRoot = rootLayout.params.empty() && rootLayout.staticSamplers.empty();
    const RootSignatureLayout& msEffectiveLayout =
        (hasMS && emptyRoot && msLoaded.hasEmbeddedRootSig)
            ? msLoaded.embeddedRootSig
            : ((hasAS && emptyRoot && asLoaded.hasEmbeddedRootSig) ? asLoaded.embeddedRootSig : rootLayout);
    const RootSignatureLayout& asEffectiveLayout =
        (hasAS && emptyRoot && asLoaded.hasEmbeddedRootSig) ? asLoaded.embeddedRootSig : msEffectiveLayout;

    auto vsLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader vsLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> vsMlirMod;
    bool loadedVSFromDXIL = false;
    if (!hasMS) {
        if (!skipVS || !precompiledVSOutputRecs || precompiledVSOutputRecs->empty() || emptyRoot) {
            if (!loadAndUpgradeDXIL(desc.VS, *vsLlvmCtx, vsLoaded, outError)) {
                return false;
            }
            loadedVSFromDXIL = true;
        }
        if (!skipVS) {
            vsMlirMod = importAndLiftDXILToSCF(std::move(vsLoaded.module), vsLoaded.entryName, mlirCtx, outError);
            if (!vsMlirMod) return false;
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

    auto hsLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader hsLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> hsMlirMod;
    auto dsLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader dsLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> dsMlirMod;
    if (hasTess) {
        if (!loadAndUpgradeDXIL(desc.HS, *hsLlvmCtx, hsLoaded, outError)) {
            return false;
        }
        hsMlirMod = importAndLiftDXILToSCF(
            std::move(hsLoaded.module), hsLoaded.entryName, mlirCtx, outError, hsLoaded.hsPatchConstFnName);
        if (!hsMlirMod) return false;

        if (!loadAndUpgradeDXIL(desc.DS, *dsLlvmCtx, dsLoaded, outError)) {
            return false;
        }
        dsMlirMod = importAndLiftDXILToSCF(std::move(dsLoaded.module), dsLoaded.entryName, mlirCtx, outError);
        if (!dsMlirMod) return false;
    }

    const RootSignatureLayout& hsEffectiveLayout =
        (hasTess && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && hsLoaded.hasEmbeddedRootSig)
            ? hsLoaded.embeddedRootSig
            : vsEffectiveLayout;
    const RootSignatureLayout& dsEffectiveLayout =
        (hasTess && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && dsLoaded.hasEmbeddedRootSig)
            ? dsLoaded.embeddedRootSig
            : vsEffectiveLayout;

    auto gsLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader gsLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> gsMlirMod;
    if (hasGS) {
        if (!loadAndUpgradeDXIL(desc.GS, *gsLlvmCtx, gsLoaded, outError)) {
            return false;
        }
        gsMlirMod = importAndLiftDXILToSCF(std::move(gsLoaded.module), gsLoaded.entryName, mlirCtx, outError);
        if (!gsMlirMod) return false;
    }

    const RootSignatureLayout& gsEffectiveLayout =
        (hasGS && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && gsLoaded.hasEmbeddedRootSig)
            ? gsLoaded.embeddedRootSig
            : vsEffectiveLayout;

    bool hasPS = (desc.PS.pShaderBytecode && desc.PS.BytecodeLength > 0);
    auto psLlvmCtx = std::make_unique<llvm::LLVMContext>();
    DxilLoadedShader psLoaded;
    mlir::OwningOpRef<mlir::ModuleOp> psMlirMod;
    if (hasPS) {
        if (!loadAndUpgradeDXIL(desc.PS, *psLlvmCtx, psLoaded, outError)) {
            return false;
        }
        psMlirMod = importAndLiftDXILToSCF(std::move(psLoaded.module), psLoaded.entryName, mlirCtx, outError);
        if (!psMlirMod) return false;
    }

    const RootSignatureLayout& psEffectiveLayout =
        (hasPS && rootLayout.params.empty() && rootLayout.staticSamplers.empty() && psLoaded.hasEmbeddedRootSig)
            ? psLoaded.embeddedRootSig
            : vsEffectiveLayout;

    configureNacelleMLIRContext(mlirCtx);
    mlir::Location loc = mlir::UnknownLoc::get(&mlirCtx);
    outCombinedMod = mlir::ModuleOp::create(loc);
    mlir::ModuleOp combinedMod = *outCombinedMod;

    if (hasMS) {
        cloneGlobalsIntoModule(*msMlirMod, combinedMod);
        if (hasAS) cloneGlobalsIntoModule(*asMlirMod, combinedMod);
    } else if (!skipVS && vsMlirMod) {
        cloneGlobalsIntoModule(*vsMlirMod, combinedMod);
    }
    if (hasTess) {
        cloneGlobalsIntoModule(*hsMlirMod, combinedMod);
        cloneGlobalsIntoModule(*dsMlirMod, combinedMod);
    }
    if (hasGS) {
        cloneGlobalsIntoModule(*gsMlirMod, combinedMod);
    }
    if (hasPS) {
        cloneGlobalsIntoModule(*psMlirMod, combinedMod);
    }

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
    bool psUsesSampleFrequency = false;
    if (hasPS) {
        outPSInputs = convertSigElements(psLoaded.inputSigs);
        outPSOutputs = convertSigElements(psLoaded.outputSigs);
        for (const auto& po : psLoaded.outputSigs) {
            if (po.systemValueType >= 17 && po.systemValueType <= 19) psWritesDepth = true;
            if (po.systemValueType == 14) psUsesSampleFrequency = true; // SV_Coverage output
        }
        for (const auto& pi : psLoaded.inputSigs) {
            if (pi.systemValueType == 28 && pi.interpMode != 4) {
                psUsesPerspBarycentrics = true;
            }
            if (pi.systemValueType == 14 || pi.systemValueType == 15 || pi.systemValueType == 18 ||
                pi.interpMode == 5 || pi.interpMode == 7) {
                psUsesSampleFrequency = true;
            }
        }
    } else {
        outPSInputs.clear();
        outPSOutputs.clear();
    }
    const uint32_t msaaSampleCount = std::max<uint32_t>(1u, desc.SampleDesc.Count);
    const bool psPixelFreqMSAA =
        hasPS && msaaSampleCount > 1u && msaaSampleCount <= 8u &&
        !desc.BlendState.AlphaToCoverageEnable && !psUsesSampleFrequency;
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
        auto vsSrcFunc = vsMlirMod->lookupSymbol<mlir::func::FuncOp>(vsLoaded.entryName);
        if (!emitVertexStageFunction(
                combinedMod, vsSrcFunc, vsLoaded, vsEffectiveLayout, desc.InputLayout,
                simdWidth, vsSym, hasTess, hasGS,
                (!hasTess && !hasGS) ? &canonicalVS : nullptr,
                (hasTess || hasGS) ? &vsToNextMap : nullptr,
                outError)) {
            return false;
        }
    }

    // --- 1-mesh. Emit @<vsSym>_as (optional) and @<vsSym>_ms ---
    if (hasMS) {
        auto extractLaneI32 = [](mlir::OpBuilder& lb, mlir::Location lloc, const SimdVal& sv, uint32_t lane) -> mlir::Value {
            mlir::Type i32 = lb.getI32Type();
            mlir::Value v = (sv.kind == SimdVal::Kind::Vector)
                                ? llvmExtract(lb, lloc, sv.vec, static_cast<uint32_t>(static_cast<int64_t>(lane)))
                                : (sv.lanes.size() == 1 ? sv.lanes[0] : sv.lanes[lane]);
            if (v.getType() != i32) {
                if (mlir::isa<mlir::FloatType>(v.getType())) {
                    if (v.getType().getIntOrFloatBitWidth() != 32) {
                        v = mlir::LLVM::FPExtOp::create(lb, lloc, lb.getF32Type(), v);
                    }
                    v = mlir::LLVM::BitcastOp::create(lb, lloc, i32, v);
                } else if (v.getType().getIntOrFloatBitWidth() < 32) {
                    v = mlir::LLVM::ZExtOp::create(lb, lloc, i32, v);
                } else {
                    v = mlir::LLVM::TruncOp::create(lb, lloc, i32, v);
                }
            }
            return v;
        };

        auto convertSimdValToF32Vec = [&](mlir::OpBuilder& lb, mlir::Location lloc, const SimdVal& val) -> mlir::Value {
            IRHelpers lh{lb, lloc, simdWidth};
            mlir::Value vVec = val.vec;
            mlir::Type elemTy = llvm::cast<mlir::VectorType>(vVec.getType()).getElementType();
            if (elemTy != lh.f32Ty()) {
                if (mlir::isa<mlir::IntegerType>(elemTy)) {
                    if (elemTy.getIntOrFloatBitWidth() < 32) {
                        vVec = mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(lh.i32Ty()), vVec);
                    }
                    vVec = mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(lh.f32Ty()), vVec);
                } else {
                    vVec = mlir::LLVM::FPExtOp::create(lb, lloc, lh.vecTy(lh.f32Ty()), vVec);
                }
            }
            return vVec;
        };

        if (hasAS) {
            auto asSrcFunc = asMlirMod->lookupSymbol<mlir::func::FuncOp>(asLoaded.entryName);
            if (!asSrcFunc || asSrcFunc.getBody().empty()) {
                outError = "Amplification shader entry function missing after MLIR SCF lift";
                return false;
            }

            mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
            IRHelpers h{b, loc, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto fnTy = b.getFunctionType({ptrTy}, {});
            auto asFn = mlir::func::FuncOp::create(b, loc, vsSym + "_as", fnTy);
            mlir::Block* entry = asFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value argsPtr = entry->getArgument(0);
            mlir::Value rootCtx = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, rootCtx)));
            mlir::Value gsBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, groupSharedScratch)));
            mlir::Value gx = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, gx)));
            mlir::Value gy = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, gy)));
            mlir::Value gz = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, gz)));
            mlir::Value outDispatchDims = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, outDispatchDims)));
            mlir::Value outPayload = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(AmplificationThreadGroupArgs, outPayload)));

            StageHooks asHooks;
            asHooks.entryBlock = entry;
            asHooks.dispatchMesh = [&](mlir::OpBuilder& db, mlir::Location dloc,
                                       const SimdVal& countX, const SimdVal& countY, const SimdVal& countZ,
                                       const SimdVal& payloadArg, mlir::Value realMask) -> bool {
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value actL = llvmExtract(db, dloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::scf::IfOp::create(db, dloc, actL, [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                        IRHelpers lh{lb, lloc, simdWidth};
                        mlir::Value already = lh.loadScalar(i32Ty, lh.gepBytes(outDispatchDims, 12));
                        mlir::Value notYet = mlir::LLVM::ICmpOp::create(
                            lb, lloc, mlir::LLVM::ICmpPredicate::eq, already, lh.cInt(i32Ty, 0));
                        mlir::scf::IfOp::create(lb, lloc, notYet, [&](mlir::OpBuilder& cb, mlir::Location cloc) {
                            IRHelpers ch{cb, cloc, simdWidth};
                            ch.storeScalar(ch.cInt(i32Ty, 1), ch.gepBytes(outDispatchDims, 12));
                            ch.storeScalar(extractLaneI32(cb, cloc, countX, l), ch.gepBytes(outDispatchDims, 0));
                            ch.storeScalar(extractLaneI32(cb, cloc, countY, l), ch.gepBytes(outDispatchDims, 4));
                            ch.storeScalar(extractLaneI32(cb, cloc, countZ, l), ch.gepBytes(outDispatchDims, 8));
                            if (asLoaded.asPayloadBytes > 0 && payloadArg.kind == SimdVal::Kind::Lanes && !payloadArg.lanes.empty()) {
                                mlir::Value srcPtr = payloadArg.lanes.size() == 1 ? payloadArg.lanes[0] : payloadArg.lanes[l];
                                mlir::Value nullP = mlir::LLVM::ZeroOp::create(cb, cloc, ptrTy);
                                mlir::Value srcOk = mlir::LLVM::ICmpOp::create(cb, cloc, mlir::LLVM::ICmpPredicate::ne, srcPtr, nullP);
                                mlir::Value dstOk = mlir::LLVM::ICmpOp::create(cb, cloc, mlir::LLVM::ICmpPredicate::ne, outPayload, nullP);
                                mlir::scf::IfOp::create(cb, cloc, mlir::LLVM::AndOp::create(cb, cloc, srcOk, dstOk),
                                                        [&](mlir::OpBuilder& mb, mlir::Location mloc) {
                                                            mlir::Type i64T = mb.getI64Type();
                                                            mlir::Value c0 = mlir::LLVM::ConstantOp::create(mb, mloc, i64T, mb.getI64IntegerAttr(0));
                                                            mlir::Value c1 = mlir::LLVM::ConstantOp::create(mb, mloc, i64T, mb.getI64IntegerAttr(1));
                                                            mlir::Value cN = mlir::LLVM::ConstantOp::create(mb, mloc, i64T, mb.getI64IntegerAttr(asLoaded.asPayloadBytes));
                                                            mlir::scf::ForOp::create(
                                                                mb, mloc, c0, cN, c1, mlir::ValueRange{},
                                                                [&](mlir::OpBuilder& fb, mlir::Location floc, mlir::Value bi, mlir::ValueRange) {
                                                                    mlir::Value bi64 = bi;
                                                                    mlir::Value sp = mlir::LLVM::GEPOp::create(fb, floc, ptrTy, fb.getI8Type(), srcPtr, mlir::ValueRange{bi64});
                                                                    mlir::Value dp = mlir::LLVM::GEPOp::create(fb, floc, ptrTy, fb.getI8Type(), outPayload, mlir::ValueRange{bi64});
                                                                    mlir::Value bv = mlir::LLVM::LoadOp::create(fb, floc, fb.getI8Type(), sp);
                                                                    mlir::LLVM::StoreOp::create(fb, floc, bv, dp);
                                                                    mlir::scf::YieldOp::create(fb, floc);
                                                                });
                                                            mlir::scf::YieldOp::create(mb, mloc);
                                                        });
                            }
                            mlir::scf::YieldOp::create(cb, cloc);
                        });
                        mlir::scf::YieldOp::create(lb, lloc);
                    });
                }
                return true;
            };

            if (!packetizeThreadGroupRegion(b, loc, combinedMod, asSrcFunc.getBody(), asLoaded, asEffectiveLayout,
                                            simdWidth, rootCtx, gsBase, gx, gy, gz, entry, asHooks, outError)) {
                return false;
            }
            mlir::func::ReturnOp::create(b, loc);
        }

        // Emit @<vsSym>_ms(const MeshThreadGroupArgs* args)
        {
            auto msSrcFunc = msMlirMod->lookupSymbol<mlir::func::FuncOp>(msLoaded.entryName);
            if (!msSrcFunc || msSrcFunc.getBody().empty()) {
                outError = "Mesh shader entry function missing after MLIR SCF lift";
                return false;
            }

            mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
            IRHelpers h{b, loc, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto f32Ty = h.f32Ty();
            auto i8Ty = h.i8Ty();
            auto fnTy = b.getFunctionType({ptrTy}, {});
            auto msFn = mlir::func::FuncOp::create(b, loc, vsSym + "_ms", fnTy);
            mlir::Block* entry = msFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value argsPtr = entry->getArgument(0);
            mlir::Value rootCtx = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, rootCtx)));
            mlir::Value gsBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, groupSharedScratch)));
            mlir::Value payloadPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, payloadPtr)));
            mlir::Value gx = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, gx)));
            mlir::Value gy = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, gy)));
            mlir::Value gz = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, gz)));
            mlir::Value outCounts = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outCounts)));
            mlir::Value outVertPosXYZW = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outVertPosXYZW)));
            mlir::Value outVertAttrs = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outVertAttrs)));
            mlir::Value outPrimIndices = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimIndices)));
            mlir::Value outPrimAttrs = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimAttrs)));
            mlir::Value outPrimCull = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimCull)));
            mlir::Value outPrimId = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(MeshThreadGroupArgs, outPrimId)));

            mlir::Value countsSetPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, i32Ty, h.cInt(i32Ty, 1), 4);
            h.storeScalar(h.cInt(i32Ty, 0), countsSetPtr);

            const uint32_t maxVerts = msLoaded.msMaxVertexCount;
            const uint32_t maxPrims = msLoaded.msMaxPrimitiveCount;

            StageHooks msHooks;
            msHooks.entryBlock = entry;
            msHooks.meshPayloadBase = payloadPtr;

            msHooks.setMeshOutputCounts = [&](mlir::OpBuilder& sb, mlir::Location sloc,
                                              const SimdVal& numVerts, const SimdVal& numPrims,
                                              mlir::Value realMask) -> bool {
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::scf::IfOp::create(sb, sloc, actL, [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                        IRHelpers lh{lb, lloc, simdWidth};
                        mlir::Value already = lh.loadScalar(i32Ty, countsSetPtr);
                        mlir::Value notYet = mlir::LLVM::ICmpOp::create(
                            lb, lloc, mlir::LLVM::ICmpPredicate::eq, already, lh.cInt(i32Ty, 0));
                        mlir::scf::IfOp::create(lb, lloc, notYet, [&](mlir::OpBuilder& cb, mlir::Location cloc) {
                            IRHelpers ch{cb, cloc, simdWidth};
                            ch.storeScalar(ch.cInt(i32Ty, 1), countsSetPtr);
                            ch.storeScalar(extractLaneI32(cb, cloc, numVerts, l), ch.gepBytes(outCounts, 0));
                            ch.storeScalar(extractLaneI32(cb, cloc, numPrims, l), ch.gepBytes(outCounts, 4));
                            mlir::scf::YieldOp::create(cb, cloc);
                        });
                        mlir::scf::YieldOp::create(lb, lloc);
                    });
                }
                return true;
            };

            msHooks.emitIndices = [&](mlir::OpBuilder& sb, mlir::Location sloc,
                                      const SimdVal& primIdx, const SimdVal& v0,
                                      const SimdVal& v1, const SimdVal& v2,
                                      mlir::Value realMask) -> bool {
                if (maxPrims == 0) return true;
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::Value pIdx = extractLaneI32(sb, sloc, primIdx, l);
                    mlir::Value inB = mlir::LLVM::ICmpOp::create(
                        sb, sloc, mlir::LLVM::ICmpPredicate::ult, pIdx,
                        mlir::LLVM::ConstantOp::create(sb, sloc, i32Ty, sb.getIntegerAttr(i32Ty, maxPrims)));
                    mlir::scf::IfOp::create(sb, sloc, mlir::LLVM::AndOp::create(sb, sloc, actL, inB),
                                            [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                                IRHelpers lh{lb, lloc, simdWidth};
                                                mlir::Value base3 = mlir::LLVM::MulOp::create(lb, lloc, pIdx, lh.cInt(i32Ty, 3));
                                                lh.storeScalar(extractLaneI32(lb, lloc, v0, l), lh.gepArray(outPrimIndices, i32Ty, base3));
                                                lh.storeScalar(extractLaneI32(lb, lloc, v1, l),
                                                               lh.gepArray(outPrimIndices, i32Ty, mlir::LLVM::AddOp::create(lb, lloc, base3, lh.cInt(i32Ty, 1))));
                                                lh.storeScalar(extractLaneI32(lb, lloc, v2, l),
                                                               lh.gepArray(outPrimIndices, i32Ty, mlir::LLVM::AddOp::create(lb, lloc, base3, lh.cInt(i32Ty, 2))));
                                                mlir::scf::YieldOp::create(lb, lloc);
                                            });
                }
                return true;
            };

            msHooks.storeVertexOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                                            const SimdVal& row, uint32_t col, const SimdVal& val,
                                            const SimdVal& vertIdx, mlir::Value realMask) -> bool {
                const DxilSigRecord* sig = msLoaded.findOutputSig(sigId);
                if (!sig || val.kind != SimdVal::Kind::Vector || maxVerts == 0) return true;
                mlir::Value vVec = convertSimdValToF32Vec(sb, sloc, val);

                if (sig->systemValueType == 3) { // SV_Position
                    if (col < 4) {
                        for (uint32_t l = 0; l < simdWidth; ++l) {
                            mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                            mlir::Value vIdx = extractLaneI32(sb, sloc, vertIdx, l);
                            mlir::Value inB = mlir::LLVM::ICmpOp::create(
                                sb, sloc, mlir::LLVM::ICmpPredicate::ult, vIdx,
                                mlir::LLVM::ConstantOp::create(sb, sloc, i32Ty, sb.getIntegerAttr(i32Ty, maxVerts)));
                            mlir::scf::IfOp::create(sb, sloc, mlir::LLVM::AndOp::create(sb, sloc, actL, inB),
                                                    [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                                        IRHelpers lh{lb, lloc, simdWidth};
                                                        mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                                                            lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, vIdx, lh.cInt(i32Ty, 4)),
                                                            lh.cInt(i32Ty, static_cast<int64_t>(col)));
                                                        mlir::Value vL = llvmExtract(lb, lloc, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                        lh.storeScalar(vL, lh.gepArray(outVertPosXYZW, f32Ty, elemIdx));
                                                        mlir::scf::YieldOp::create(lb, lloc);
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
                        mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value vIdx = extractLaneI32(sb, sloc, vertIdx, l);
                        mlir::Value inB = mlir::LLVM::ICmpOp::create(
                            sb, sloc, mlir::LLVM::ICmpPredicate::ult, vIdx,
                            mlir::LLVM::ConstantOp::create(sb, sloc, i32Ty, sb.getIntegerAttr(i32Ty, maxVerts)));
                        mlir::scf::IfOp::create(sb, sloc, mlir::LLVM::AndOp::create(sb, sloc, actL, inB),
                                                [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                                    IRHelpers lh{lb, lloc, simdWidth};
                                                    mlir::Value slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot));
                                                    if (useDynRow) {
                                                        if (auto rc = tryGetUniformConstUInt(row)) {
                                                            if (*rc != 0) {
                                                                slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot + (*rc) * 4));
                                                            }
                                                        } else {
                                                            mlir::Value r32 = extractLaneI32(lb, lloc, row, l);
                                                            slot32 = mlir::LLVM::AddOp::create(
                                                                lb, lloc, slot32, mlir::LLVM::MulOp::create(lb, lloc, r32, lh.cInt(i32Ty, 4)));
                                                        }
                                                    }
                                                    mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                                                        lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, vIdx, lh.cInt(i32Ty, totalVSScalarAttrs)),
                                                        slot32);
                                                    mlir::Value vL = llvmExtract(lb, lloc, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                    lh.storeScalar(vL, lh.gepArray(outVertAttrs, f32Ty, elemIdx));
                                                    mlir::scf::YieldOp::create(lb, lloc);
                                                });
                    }
                }
                return true;
            };

            msHooks.storePrimitiveOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                                               const SimdVal& row, uint32_t col, const SimdVal& val,
                                               const SimdVal& primIdx, mlir::Value realMask) -> bool {
                const DxilSigRecord* sig = msLoaded.findPatchConstSig(sigId);
                if (!sig || val.kind != SimdVal::Kind::Vector || maxPrims == 0) return true;

                if (sig->systemValueType == 30) { // SV_CullPrimitive
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value pIdx = extractLaneI32(sb, sloc, primIdx, l);
                        mlir::Value inB = mlir::LLVM::ICmpOp::create(
                            sb, sloc, mlir::LLVM::ICmpPredicate::ult, pIdx,
                            mlir::LLVM::ConstantOp::create(sb, sloc, i32Ty, sb.getIntegerAttr(i32Ty, maxPrims)));
                        mlir::scf::IfOp::create(sb, sloc, mlir::LLVM::AndOp::create(sb, sloc, actL, inB),
                                                [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                                    IRHelpers lh{lb, lloc, simdWidth};
                                                    mlir::Value vL = llvmExtract(lb, lloc, val.vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                    mlir::Value nz;
                                                    if (mlir::isa<mlir::IntegerType>(vL.getType())) {
                                                        nz = mlir::LLVM::ICmpOp::create(
                                                            lb, lloc, mlir::LLVM::ICmpPredicate::ne, vL, lh.cInt(vL.getType(), 0));
                                                    } else {
                                                        nz = mlir::LLVM::FCmpOp::create(
                                                            lb, lloc, mlir::LLVM::FCmpPredicate::one, vL, lh.cFlt(0.0));
                                                    }
                                                    mlir::Value cByte = mlir::LLVM::SelectOp::create(
                                                        lb, lloc, nz, lh.cInt(i8Ty, 1), lh.cInt(i8Ty, 0));
                                                    lh.storeScalar(cByte, lh.gepArray(outPrimCull, i8Ty, pIdx));
                                                    mlir::scf::YieldOp::create(lb, lloc);
                                                });
                    }
                    return true;
                }

                if (sig->systemValueType == 10) { // SV_PrimitiveID
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value pIdx = extractLaneI32(sb, sloc, primIdx, l);
                        mlir::Value inB = mlir::LLVM::ICmpOp::create(
                            sb, sloc, mlir::LLVM::ICmpPredicate::ult, pIdx,
                            mlir::LLVM::ConstantOp::create(sb, sloc, i32Ty, sb.getIntegerAttr(i32Ty, maxPrims)));
                        mlir::scf::IfOp::create(sb, sloc, mlir::LLVM::AndOp::create(sb, sloc, actL, inB),
                                                [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                                    IRHelpers lh{lb, lloc, simdWidth};
                                                    lh.storeScalar(extractLaneI32(lb, lloc, val, l), lh.gepArray(outPrimId, i32Ty, pIdx));
                                                    mlir::scf::YieldOp::create(lb, lloc);
                                                });
                    }
                    return true;
                }

                mlir::Value vVec = convertSimdValToF32Vec(sb, sloc, val);
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
                        mlir::Value actL = llvmExtract(sb, sloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value pIdx = extractLaneI32(sb, sloc, primIdx, l);
                        mlir::Value inB = mlir::LLVM::ICmpOp::create(
                            sb, sloc, mlir::LLVM::ICmpPredicate::ult, pIdx,
                            mlir::LLVM::ConstantOp::create(sb, sloc, i32Ty, sb.getIntegerAttr(i32Ty, maxPrims)));
                        mlir::scf::IfOp::create(sb, sloc, mlir::LLVM::AndOp::create(sb, sloc, actL, inB),
                                                [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                                    IRHelpers lh{lb, lloc, simdWidth};
                                                    mlir::Value slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot));
                                                    if (useDynRow) {
                                                        if (auto rc = tryGetUniformConstUInt(row)) {
                                                            if (*rc != 0) {
                                                                slot32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot + (*rc) * 4));
                                                            }
                                                        } else {
                                                            mlir::Value r32 = extractLaneI32(lb, lloc, row, l);
                                                            slot32 = mlir::LLVM::AddOp::create(
                                                                lb, lloc, slot32, mlir::LLVM::MulOp::create(lb, lloc, r32, lh.cInt(i32Ty, 4)));
                                                        }
                                                    }
                                                    mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                                                        lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, pIdx, lh.cInt(i32Ty, totalVSScalarAttrs)),
                                                        slot32);
                                                    mlir::Value vL = llvmExtract(lb, lloc, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                                    lh.storeScalar(vL, lh.gepArray(outPrimAttrs, f32Ty, elemIdx));
                                                    mlir::scf::YieldOp::create(lb, lloc);
                                                });
                    }
                }
                return true;
            };

            if (!packetizeThreadGroupRegion(b, loc, combinedMod, msSrcFunc.getBody(), msLoaded, msEffectiveLayout,
                                            simdWidth, rootCtx, gsBase, gx, gy, gz, entry, msHooks, outError)) {
                return false;
            }
            mlir::func::ReturnOp::create(b, loc);
        }
    }

    // --- 1a-tess. Emit @<vsSym>_hs, @<vsSym>_hspc, and @<vsSym>_ds ---
    if (hasTess) {
        auto computeScalarPtr = [](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW,
                                   mlir::Value basePtr, uint32_t cpStride,
                                   const SimdVal* vAxis, const SimdVal& row, uint32_t baseSlot) -> mlir::Value {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            auto i64Ty = lh.i64Ty();

            mlir::Value idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseSlot));
            if (vAxis && cpStride > 0) {
                if (auto vc = tryGetUniformConstUInt(*vAxis)) {
                    idx32 = lh.cInt(i32Ty, static_cast<int64_t>((*vc) * cpStride + baseSlot));
                } else {
                    mlir::Value v0 = llvmExtract(lb, lloc, vAxis->vec, static_cast<uint32_t>(0));
                    if (v0.getType() != i32Ty) v0 = mlir::LLVM::ZExtOp::create(lb, lloc, i32Ty, v0);
                    idx32 = mlir::LLVM::AddOp::create(
                        lb, lloc, idx32, mlir::LLVM::MulOp::create(lb, lloc, v0, lh.cInt(i32Ty, cpStride)));
                }
            }
            if (auto rc = tryGetUniformConstUInt(row)) {
                if (*rc != 0) {
                    idx32 = mlir::LLVM::AddOp::create(lb, lloc, idx32, lh.cInt(i32Ty, static_cast<int64_t>((*rc) * 4)));
                }
            } else {
                mlir::Value r0 = llvmExtract(lb, lloc, row.vec, static_cast<uint32_t>(0));
                if (r0.getType() != i32Ty) r0 = mlir::LLVM::ZExtOp::create(lb, lloc, i32Ty, r0);
                idx32 = mlir::LLVM::AddOp::create(
                    lb, lloc, idx32, mlir::LLVM::MulOp::create(lb, lloc, r0, lh.cInt(i32Ty, 4)));
            }
            mlir::Value byteOff = mlir::LLVM::MulOp::create(
                lb, lloc, mlir::LLVM::ZExtOp::create(lb, lloc, i64Ty, idx32), lh.cInt(i64Ty, 4));
            return lh.gepBytesDyn(basePtr, byteOff);
        };

        auto convertLoadedF32ToSimd = [](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW,
                                         mlir::Value f32Scalar, mlir::Type scalarTy) -> SimdVal {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            mlir::Value outVal = f32Scalar;
            if (scalarTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(scalarTy)) {
                    outVal = mlir::LLVM::BitcastOp::create(lb, lloc, i32Ty, outVal);
                    if (scalarTy != i32Ty) outVal = mlir::LLVM::TruncOp::create(lb, lloc, scalarTy, outVal);
                } else if (mlir::isa<mlir::FloatType>(scalarTy)) {
                    outVal = mlir::LLVM::FPTruncOp::create(lb, lloc, scalarTy, outVal);
                }
            }
            return lh.splatScalar(outVal);
        };

        auto convertF32VecToSimd = [](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW,
                                      mlir::Value fVec, mlir::Type scalarTy) -> SimdVal {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            mlir::Value outVec = fVec;
            if (scalarTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(scalarTy)) {
                    outVec = mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(i32Ty), outVec);
                    if (scalarTy != i32Ty) {
                        outVec = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), outVec);
                    }
                } else if (mlir::isa<mlir::FloatType>(scalarTy)) {
                    outVec = mlir::LLVM::FPTruncOp::create(lb, lloc, lh.vecTy(scalarTy), outVec);
                }
            }
            return {SimdVal::Kind::Vector, scalarTy, outVec, {}, {}};
        };

        auto toI32Vec = [](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW, const SimdVal& sv) -> mlir::Value {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            mlir::Type et = llvm::cast<mlir::VectorType>(sv.vec.getType()).getElementType();
            if (et == i32Ty) return sv.vec;
            if (et.getIntOrFloatBitWidth() < 32) {
                return mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(i32Ty), sv.vec);
            }
            return mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(i32Ty), sv.vec);
        };

        auto loadPatchInputSIMD = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW,
                                      mlir::Value basePtr, uint32_t cpStride,
                                      const SimdVal* vAxis, const SimdVal& row, uint32_t rowStride,
                                      uint32_t baseSlot, mlir::Type scalarTy, mlir::Value gatherMask) -> SimdVal {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            auto i64Ty = lh.i64Ty();
            auto f32Ty = lh.f32Ty();

            std::optional<uint32_t> vc = vAxis ? tryGetUniformConstUInt(*vAxis) : std::optional<uint32_t>(0u);
            std::optional<uint32_t> rc = tryGetUniformConstUInt(row);
            if (vc.has_value() && rc.has_value()) {
                int64_t offBytes = static_cast<int64_t>((*vc) * cpStride + baseSlot + (*rc) * rowStride) * 4;
                mlir::Value fVal = lh.loadScalar(f32Ty, lh.gepBytes(basePtr, offBytes));
                return convertLoadedF32ToSimd(lb, lloc, simdW, fVal, scalarTy);
            }

            mlir::Value idxVec = lh.splatInt(i32Ty, static_cast<int64_t>(baseSlot));
            if (vAxis && cpStride > 0) {
                idxVec = mlir::LLVM::AddOp::create(
                    lb, lloc, idxVec,
                    mlir::LLVM::MulOp::create(lb, lloc, toI32Vec(lb, lloc, simdW, *vAxis), lh.splatInt(i32Ty, cpStride)));
            }
            if (rowStride > 0) {
                idxVec = mlir::LLVM::AddOp::create(
                    lb, lloc, idxVec,
                    mlir::LLVM::MulOp::create(lb, lloc, toI32Vec(lb, lloc, simdW, row), lh.splatInt(i32Ty, rowStride)));
            }
            mlir::Value byteOffVec = mlir::LLVM::MulOp::create(
                lb, lloc, mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(i64Ty), idxVec), lh.splatInt(i64Ty, 4));
            mlir::Value basePtrVec = llvmSplat(lb, lloc, llvm::cast<mlir::VectorType>(lh.vecTy(lh.ptrTy())).getElementType(), basePtr, llvm::cast<mlir::VectorType>(lh.vecTy(lh.ptrTy())).getNumElements());
            mlir::Value ptrVec = mlir::LLVM::GEPOp::create(
                lb, lloc, lh.vecTy(lh.ptrTy()), lh.i8Ty(), basePtrVec, mlir::ValueRange{byteOffVec});
            mlir::Value fVec = mlir::LLVM::masked_gather::create(
                lb, lloc, lh.vecTy(f32Ty), ptrVec, gatherMask, mlir::ValueRange{lh.splatFlt(0.0)}, /*alignment=*/4);
            return convertF32VecToSimd(lb, lloc, simdW, fVec, scalarTy);
        };

        auto storeVecToSOA = [](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t simdW,
                                const SimdVal& val, mlir::Value writeMask, mlir::Value activeMask,
                                mlir::Value outBasePtr, uint32_t baseSlotWithCol, const SimdVal* dynRow) {
            if (val.kind != SimdVal::Kind::Vector) return;
            IRHelpers sh{sb, sloc, simdW};
            auto i32Ty = sh.i32Ty();
            auto f32Ty = sh.f32Ty();
            mlir::Value vVec = val.vec;
            mlir::Type elemTy = llvm::cast<mlir::VectorType>(vVec.getType()).getElementType();
            if (elemTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(elemTy)) {
                    if (elemTy.getIntOrFloatBitWidth() < 32) {
                        vVec = mlir::LLVM::ZExtOp::create(sb, sloc, sh.vecTy(i32Ty), vVec);
                    }
                    vVec = mlir::LLVM::BitcastOp::create(sb, sloc, sh.vecTy(f32Ty), vVec);
                } else {
                    vVec = mlir::LLVM::FPExtOp::create(sb, sloc, sh.vecTy(f32Ty), vVec);
                }
            }

            std::optional<uint32_t> rc = dynRow ? tryGetUniformConstUInt(*dynRow) : std::optional<uint32_t>(0u);
            if (rc.has_value()) {
                uint32_t scalarSlot = baseSlotWithCol + (*rc) * 4;
                mlir::Value dstPtr = sh.gepBytes(outBasePtr, static_cast<int64_t>(scalarSlot * simdW * 4));
                if (!writeMask || writeMask == activeMask) {
                    mlir::LLVM::StoreOp::create(sb, sloc, vVec, dstPtr, /*alignment=*/4);
                } else {
                    mlir::Value prev = sh.loadScalar(sh.vecTy(f32Ty), dstPtr);
                    mlir::Value merged = mlir::LLVM::SelectOp::create(sb, sloc, writeMask, vVec, prev);
                    mlir::LLVM::StoreOp::create(sb, sloc, merged, dstPtr, /*alignment=*/4);
                }
            } else {
                mlir::Value effMask = writeMask ? writeMask : activeMask;
                for (uint32_t l = 0; l < simdW; ++l) {
                    mlir::Value mL = llvmExtract(sb, sloc, effMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::scf::IfOp::create(sb, sloc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        IRHelpers nh{nb, nloc, simdW};
                        mlir::Value rL = llvmExtract(nb, nloc, dynRow->vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value r32 = (rL.getType() == i32Ty)
                                              ? rL
                                              : mlir::LLVM::ZExtOp::create(nb, nloc, i32Ty, rL).getResult();
                        mlir::Value slot32 = mlir::LLVM::AddOp::create(
                            nb, nloc, nh.cInt(i32Ty, static_cast<int64_t>(baseSlotWithCol)),
                            mlir::LLVM::MulOp::create(nb, nloc, r32, nh.cInt(i32Ty, 4)));
                        mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                            nb, nloc, mlir::LLVM::MulOp::create(nb, nloc, slot32, nh.cInt(i32Ty, simdW)),
                            nh.cInt(i32Ty, static_cast<int64_t>(l)));
                        mlir::Value byteOff = mlir::LLVM::MulOp::create(
                            nb, nloc, mlir::LLVM::ZExtOp::create(nb, nloc, nh.i64Ty(), elemIdx), nh.cInt(nh.i64Ty(), 4));
                        mlir::Value vL = llvmExtract(nb, nloc, vVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::LLVM::StoreOp::create(nb, nloc, vL, nh.gepBytesDyn(outBasePtr, byteOff));
                        mlir::scf::YieldOp::create(nb, nloc);
                    });
                }
            }
        };

        auto storeLane0ToF32Ptr = [](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW,
                                     const SimdVal& val, mlir::Value realMask, mlir::Value dstPtr) {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            mlir::Value v0 = llvmExtract(lb, lloc, val.vec, static_cast<uint32_t>(0));
            if (v0.getType() != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(v0.getType())) {
                    if (v0.getType() != i32Ty) v0 = mlir::LLVM::ZExtOp::create(lb, lloc, i32Ty, v0);
                    v0 = mlir::LLVM::BitcastOp::create(lb, lloc, f32Ty, v0);
                } else {
                    v0 = mlir::LLVM::FPExtOp::create(lb, lloc, f32Ty, v0);
                }
            }
            mlir::Value l0Active = llvmExtract(lb, lloc, realMask, static_cast<uint32_t>(0));
            mlir::scf::IfOp::create(lb, lloc, l0Active, [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                mlir::LLVM::StoreOp::create(sb, sloc, v0, dstPtr, /*alignment=*/4);
                mlir::scf::YieldOp::create(sb, sloc);
            });
        };

        // 1) @<vsSym>_hs(rootCtx, inCPAttrs, patchId, baseOutCpId, activeCount, outCPAttrsSOA)
        {
            auto hsSrcFunc = hsMlirMod->lookupSymbol<mlir::func::FuncOp>(hsLoaded.entryName);
            if (!hsSrcFunc || hsSrcFunc.getBody().empty()) {
                outError = "Hull shader control-point entry function missing after MLIR SCF lift";
                return false;
            }
            mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
            IRHelpers h{b, loc, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto fnTy = b.getFunctionType({ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, ptrTy}, {});
            auto hsFn = mlir::func::FuncOp::create(b, loc, vsSym + "_hs", fnTy);
            mlir::Block* entry = hsFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value rootCtx = entry->getArgument(0);
            mlir::Value inCPAttrs = entry->getArgument(1);
            mlir::Value patchIdArg = entry->getArgument(2);
            mlir::Value baseOutCpIdArg = entry->getArgument(3);
            mlir::Value activeCountArg = entry->getArgument(4);
            mlir::Value outCPAttrsSOA = entry->getArgument(5);

            mlir::Value acVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), activeCountArg, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());
            mlir::Value activeMask = mlir::LLVM::ICmpOp::create(
                b, loc, mlir::LLVM::ICmpPredicate::ult, h.laneIota(), acVec);
            mlir::Value outCpIdVec = mlir::LLVM::SelectOp::create(
                b, loc, activeMask,
                mlir::LLVM::AddOp::create(
                    b, loc, llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), baseOutCpIdArg, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements()), h.laneIota()),
                h.splatInt(i32Ty, 0));

            StageHooks hsHooks;
            hsHooks.entryBlock = entry;
            mlir::Value hsRealMask = activeMask;
            hsHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                              const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                              mlir::Type scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = hsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                out = loadPatchInputSIMD(
                    lb, lloc, simdWidth, inCPAttrs, numHSInputScalars, &vAxis, row, 4u, baseSlot, scalarTy, activeMask);
                return true;
            };
            hsHooks.storeOutput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                      const SimdVal& row, uint32_t col, const SimdVal& val,
                                      mlir::Value realMask) -> bool {
                auto it = hsToDsMap.find(sigId);
                if (it == hsToDsMap.end() || it->second.psTargetRow < 0) return true;
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outCPAttrsSOA, baseSlot, &row);
                return true;
            };
            hsHooks.loadOutputControlPointId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value v = (scalarTy == i32Ty)
                                    ? outCpIdVec
                                    : mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), outCpIdVec).getResult();
                out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                return true;
            };
            hsHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value v = patchIdArg;
                if (scalarTy != i32Ty) v = mlir::LLVM::TruncOp::create(lb, lloc, scalarTy, v);
                out = lh.splatScalar(v);
                return true;
            };

            if (!packetizeRegionWithHooks(b, loc, combinedMod, hsSrcFunc.getBody(), hsLoaded, hsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, hsRealMask, hsHooks, outError)) {
                return false;
            }
            mlir::func::ReturnOp::create(b, loc);
        }

        // 2) @<vsSym>_hspc(rootCtx, inCPAttrs, outCPAttrs, patchId, outPatchConsts, outTessFactors)
        {
            auto hspcSrcFunc = hsMlirMod->lookupSymbol<mlir::func::FuncOp>(hsLoaded.hsPatchConstFnName);
            if (!hspcSrcFunc || hspcSrcFunc.getBody().empty()) {
                outError = "Hull shader patch-constant function missing after MLIR SCF lift";
                return false;
            }
            mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
            IRHelpers h{b, loc, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto f32Ty = h.f32Ty();
            auto fnTy = b.getFunctionType({ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy}, {});
            auto hspcFn = mlir::func::FuncOp::create(b, loc, vsSym + "_hspc", fnTy);
            mlir::Block* entry = hspcFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value rootCtx = entry->getArgument(0);
            mlir::Value inCPAttrs = entry->getArgument(1);
            mlir::Value outCPAttrs = entry->getArgument(2);
            mlir::Value patchIdArg = entry->getArgument(3);
            mlir::Value outPatchConsts = entry->getArgument(4);
            mlir::Value outTessFactors = entry->getArgument(5);

            mlir::Value activeMask = mlir::LLVM::ICmpOp::create(
                b, loc, mlir::LLVM::ICmpPredicate::eq, h.laneIota(), h.splatInt(i32Ty, 0));

            StageHooks hspcHooks;
            hspcHooks.entryBlock = entry;
            hspcHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                                const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                                mlir::Type scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = hsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, inCPAttrs, numHSInputScalars, &vAxis, row, baseSlot);
                mlir::Value fVal = IRHelpers{lb, lloc, simdWidth}.loadScalar(f32Ty, ptr);
                out = convertLoadedF32ToSimd(lb, lloc, simdWidth, fVal, scalarTy);
                return true;
            };
            hspcHooks.loadOutputControlPoint = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                                   const SimdVal& row, uint32_t col, const SimdVal& cpIdx,
                                                   mlir::Type scalarTy, SimdVal& out) -> bool {
                auto it = hsToDsMap.find(sigId);
                uint32_t baseSlot = 0;
                if (it != hsToDsMap.end() && it->second.psTargetRow >= 0) {
                    baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                } else {
                    const DxilSigRecord* sig = hsLoaded.findOutputSig(sigId);
                    if (!sig) return false;
                    baseSlot = sig->startRow * 4 + sig->startCol + col;
                }
                mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, outCPAttrs, numHSOutputScalars, &cpIdx, row, baseSlot);
                mlir::Value fVal = IRHelpers{lb, lloc, simdWidth}.loadScalar(f32Ty, ptr);
                out = convertLoadedF32ToSimd(lb, lloc, simdWidth, fVal, scalarTy);
                return true;
            };
            hspcHooks.storePatchConstant = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                               const SimdVal& row, uint32_t col, const SimdVal& val,
                                               mlir::Value realMask) -> bool {
                const DxilSigRecord* sig = hsLoaded.findPatchConstSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 25 || sig->systemValueType == 26) {
                    // SV_TessFactor -> outTessFactors[0..3], SV_InsideTessFactor -> outTessFactors[4..5]
                    IRHelpers lh{lb, lloc, simdWidth};
                    uint32_t baseIdx = (sig->systemValueType == 25) ? 0u : 4u;
                    mlir::Value idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseIdx));
                    if (auto rc = tryGetUniformConstUInt(row)) {
                        idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseIdx + *rc));
                    } else {
                        mlir::Value r0 = llvmExtract(lb, lloc, row.vec, static_cast<uint32_t>(0));
                        if (r0.getType() != i32Ty) r0 = mlir::LLVM::ZExtOp::create(lb, lloc, i32Ty, r0);
                        idx32 = mlir::LLVM::AddOp::create(lb, lloc, idx32, r0);
                    }
                    mlir::Value byteOff = mlir::LLVM::MulOp::create(
                        lb, lloc, mlir::LLVM::ZExtOp::create(lb, lloc, lh.i64Ty(), idx32), lh.cInt(lh.i64Ty(), 4));
                    storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, lh.gepBytesDyn(outTessFactors, byteOff));
                    return true;
                }
                auto it = hsPcToDsMap.find(sigId);
                if (it == hsPcToDsMap.end() || it->second.psTargetRow < 0) return true;
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, outPatchConsts, 0, nullptr, row, baseSlot);
                storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, ptr);
                return true;
            };
            hspcHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value v = patchIdArg;
                if (scalarTy != i32Ty) v = mlir::LLVM::TruncOp::create(lb, lloc, scalarTy, v);
                out = lh.splatScalar(v);
                return true;
            };

            mlir::Value hspcRealMask = activeMask;
            if (!packetizeRegionWithHooks(b, loc, combinedMod, hspcSrcFunc.getBody(), hsLoaded, hsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, hspcRealMask, hspcHooks, outError)) {
                return false;
            }
            mlir::func::ReturnOp::create(b, loc);
        }

        // 3) @<vsSym>_ds(rootCtx, inCPAttrs, inPatchConsts, inTessFactors, patchId, domUPtr, domVPtr, domWPtr, activeCount, outPosXYZW, outAttrs)
        {
            auto dsSrcFunc = dsMlirMod->lookupSymbol<mlir::func::FuncOp>(dsLoaded.entryName);
            if (!dsSrcFunc || dsSrcFunc.getBody().empty()) {
                outError = "Domain shader entry function missing after MLIR SCF lift";
                return false;
            }
            mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
            IRHelpers h{b, loc, simdWidth};
            auto ptrTy = h.ptrTy();
            auto i32Ty = h.i32Ty();
            auto f32Ty = h.f32Ty();
            auto fnTy = b.getFunctionType({ptrTy, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, ptrTy, ptrTy}, {});
            auto dsFn = mlir::func::FuncOp::create(b, loc, vsSym + "_ds", fnTy);
            mlir::Block* entry = dsFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value rootCtx = entry->getArgument(0);
            mlir::Value inCPAttrs = entry->getArgument(1);
            mlir::Value inPatchConsts = entry->getArgument(2);
            mlir::Value inTessFactors = entry->getArgument(3);
            mlir::Value patchIdArg = entry->getArgument(4);
            mlir::Value domUPtr = entry->getArgument(5);
            mlir::Value domVPtr = entry->getArgument(6);
            mlir::Value domWPtr = entry->getArgument(7);
            mlir::Value activeCountArg = entry->getArgument(8);
            mlir::Value outPosXYZW = entry->getArgument(9);
            mlir::Value outAttrs = entry->getArgument(10);

            mlir::Value acVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), activeCountArg, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());
            mlir::Value activeMask = mlir::LLVM::ICmpOp::create(
                b, loc, mlir::LLVM::ICmpPredicate::ult, h.laneIota(), acVec);
            mlir::Value zeroFltVec = h.splatFlt(0.0);
            mlir::Value domUVec = mlir::LLVM::SelectOp::create(b, loc, activeMask, h.loadScalar(h.vecTy(f32Ty), domUPtr), zeroFltVec);
            mlir::Value domVVec = mlir::LLVM::SelectOp::create(b, loc, activeMask, h.loadScalar(h.vecTy(f32Ty), domVPtr), zeroFltVec);
            mlir::Value domWVec = mlir::LLVM::SelectOp::create(b, loc, activeMask, h.loadScalar(h.vecTy(f32Ty), domWPtr), zeroFltVec);

            StageHooks dsHooks;
            dsHooks.entryBlock = entry;
            dsHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                              const SimdVal& row, uint32_t col, const SimdVal& cpIdx,
                                              mlir::Type scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = dsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                out = loadPatchInputSIMD(
                    lb, lloc, simdWidth, inCPAttrs, numHSOutputScalars, &cpIdx, row, 4u, baseSlot, scalarTy, activeMask);
                return true;
            };
            dsHooks.loadPatchConstant = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                            const SimdVal& row, uint32_t col,
                                            mlir::Type scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = dsLoaded.findPatchConstSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 25 || sig->systemValueType == 26) {
                    uint32_t baseIdx = (sig->systemValueType == 25) ? 0u : 4u;
                    out = loadPatchInputSIMD(
                        lb, lloc, simdWidth, inTessFactors, 0u, nullptr, row, 1u, baseIdx, scalarTy, activeMask);
                    return true;
                }
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                out = loadPatchInputSIMD(
                    lb, lloc, simdWidth, inPatchConsts, 0u, nullptr, row, 4u, baseSlot, scalarTy, activeMask);
                return true;
            };
            dsHooks.loadDomainLocation = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t comp,
                                             mlir::Type scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value v = (comp == 0) ? domUVec : ((comp == 1) ? domVVec : ((comp == 2) ? domWVec : lh.splatFlt(0.0)));
                if (scalarTy != f32Ty) {
                    v = mlir::LLVM::FPTruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                }
                out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                return true;
            };
            dsHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value v = patchIdArg;
                if (scalarTy != i32Ty) v = mlir::LLVM::TruncOp::create(lb, lloc, scalarTy, v);
                out = lh.splatScalar(v);
                return true;
            };
            dsHooks.storeOutput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                      const SimdVal& row, uint32_t col, const SimdVal& val,
                                      mlir::Value realMask) -> bool {
                const DxilSigRecord* sig = dsLoaded.findOutputSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 3) {
                    if (col < 4) {
                        storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outPosXYZW, col, nullptr);
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 6) {
                    auto cit = clipSigBaseIdx.find(sigId);
                    if (cit != clipSigBaseIdx.end()) {
                        uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                        uint32_t idx = cit->second + r * sig->cols + col;
                        if (idx < 8 && (clipDistMask & (1u << idx))) {
                            storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outAttrs, clipAttrOffset + idx, nullptr);
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
                            storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outAttrs, cullAttrOffset + idx, nullptr);
                        }
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                    storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outAttrs, vpIndexAttrOffset, nullptr);
                    return true;
                }
                if (!hasGS && sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                    storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outAttrs, shadingRateAttrOffset, nullptr);
                    return true;
                }
                auto it = dsToNextMap.find(sigId);
                if (it != dsToNextMap.end() && it->second.psTargetRow >= 0) {
                    uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                    storeVecToSOA(lb, lloc, simdWidth, val, realMask, activeMask, outAttrs, baseSlot, &row);
                }
                return true;
            };

            mlir::Value dsRealMask = activeMask;
            if (!packetizeRegionWithHooks(b, loc, combinedMod, dsSrcFunc.getBody(), dsLoaded, dsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, dsRealMask, dsHooks, outError)) {
                return false;
            }
            mlir::func::ReturnOp::create(b, loc);
        }
    }

    // --- 1b. Emit @<vsSym>_gs(rootCtx, inPosXYZW, inAttrs, primIdsPtr, gsInstId, activeCount, outPosXYZW, outAttrs, outStripCounts, outTotalVerts, outNumStrips) ---
    uint32_t numGSInputScalars = numGSInputRegs * 4;
    if (hasGS) {
        auto gsSrcFunc = gsMlirMod->lookupSymbol<mlir::func::FuncOp>(gsLoaded.entryName);
        if (!gsSrcFunc || gsSrcFunc.getBody().empty()) {
            outError = "Geometry shader entry function missing after MLIR SCF lift";
            return false;
        }

        mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
        IRHelpers h{b, loc, simdWidth};
        auto ptrTy = h.ptrTy();
        auto i32Ty = h.i32Ty();
        auto f32Ty = h.f32Ty();
        auto fnTy = b.getFunctionType({ptrTy, ptrTy, ptrTy, ptrTy, i32Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy}, {});
        auto gsFn = mlir::func::FuncOp::create(b, loc, vsSym + "_gs", fnTy);
        mlir::Block* entry = gsFn.addEntryBlock();
        b.setInsertionPointToEnd(entry);

        mlir::Value rootCtx = entry->getArgument(0);
        mlir::Value inPosXYZW = entry->getArgument(1);
        mlir::Value inAttrs = entry->getArgument(2);
        mlir::Value primIdsPtr = entry->getArgument(3);
        mlir::Value gsInstIdArg = entry->getArgument(4);
        mlir::Value activeCountArg = entry->getArgument(5);
        mlir::Value outPosXYZW = entry->getArgument(6);
        mlir::Value outAttrs = entry->getArgument(7);
        mlir::Value outStripCounts = entry->getArgument(8);
        mlir::Value outTotalVerts = entry->getArgument(9);
        mlir::Value outNumStrips = entry->getArgument(10);

        mlir::Value acVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), activeCountArg, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());
        mlir::Value activeMask = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::ult, h.laneIota(), acVec);
        mlir::Value primIdVec = mlir::LLVM::SelectOp::create(
            b, loc, activeMask, h.loadScalar(h.vecTy(i32Ty), primIdsPtr), h.splatInt(i32Ty, 0));

        uint32_t maxVerts = std::max<uint32_t>(1, gsLoaded.gsMaxVertexCount);
        uint32_t outAttrScalars = totalPostGSVertexAttrs;
        uint32_t vecAlign = std::max<uint32_t>(64, simdWidth * 4);

        mlir::Value stagingPos = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, h.vecTy(f32Ty), h.cInt(i32Ty, 4), vecAlign);
        mlir::Value stagingAttrs = mlir::LLVM::AllocaOp::create(
            b, loc, ptrTy, h.vecTy(f32Ty), h.cInt(i32Ty, std::max<uint32_t>(1, outAttrScalars)), vecAlign);
        for (uint32_t c = 0; c < 4; ++c) {
            h.storeScalar(h.splatFlt(0.0), h.gepBytes(stagingPos, static_cast<int64_t>(c * simdWidth * 4)));
        }
        for (uint32_t s = 0; s < outAttrScalars; ++s) {
            h.storeScalar(h.splatFlt(0.0), h.gepBytes(stagingAttrs, static_cast<int64_t>(s * simdWidth * 4)));
        }

        mlir::Value emitCountPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, h.vecTy(i32Ty), h.cInt(i32Ty, 1), vecAlign);
        mlir::Value curStripPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, h.vecTy(i32Ty), h.cInt(i32Ty, 1), vecAlign);
        mlir::Value numStripsPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, h.vecTy(i32Ty), h.cInt(i32Ty, 1), vecAlign);
        h.storeScalar(h.splatInt(i32Ty, 0), emitCountPtr);
        h.storeScalar(h.splatInt(i32Ty, 0), curStripPtr);
        h.storeScalar(h.splatInt(i32Ty, 0), numStripsPtr);

        StageHooks gsHooks;
        gsHooks.entryBlock = entry;
        gsHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                          const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                          mlir::Type scalarTy, SimdVal& out) -> bool {
            const DxilSigRecord* sig = gsLoaded.findInputSig(sigId);
            if (!sig) return false;
            IRHelpers lh{lb, lloc, simdWidth};
            auto toI32V = [&](const SimdVal& sv) -> mlir::Value {
                mlir::Type et = llvm::cast<mlir::VectorType>(sv.vec.getType()).getElementType();
                if (et == i32Ty) return sv.vec;
                if (et.getIntOrFloatBitWidth() < 32) {
                    return mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(i32Ty), sv.vec);
                }
                return mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(i32Ty), sv.vec);
            };

            auto vcOpt = tryGetUniformConstUInt(vAxis);
            auto rcOpt = tryGetUniformConstUInt(row);
            mlir::Value vecVal;
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
                mlir::Value vVec32 = toI32V(vAxis);
                if (sig->systemValueType == 3) {
                    mlir::Value planeIdx = mlir::LLVM::AddOp::create(
                        lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, vVec32, lh.splatInt(i32Ty, 4)),
                        lh.splatInt(i32Ty, col));
                    mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                        lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, planeIdx, lh.splatInt(i32Ty, simdWidth)),
                        lh.laneIota());
                    mlir::Value byteOff = mlir::LLVM::MulOp::create(
                        lb, lloc, mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(lh.i64Ty()), elemIdx),
                        lh.splatInt(lh.i64Ty(), 4));
                    mlir::Value ptrVec = mlir::LLVM::GEPOp::create(
                        lb, lloc, lh.vecTy(ptrTy), lh.i8Ty(),
                        llvmSplat(lb, lloc, llvm::cast<mlir::VectorType>(lh.vecTy(ptrTy)).getElementType(), inPosXYZW, llvm::cast<mlir::VectorType>(lh.vecTy(ptrTy)).getNumElements()),
                        mlir::ValueRange{byteOff});
                    vecVal = mlir::LLVM::masked_gather::create(
                        lb, lloc, lh.vecTy(f32Ty), ptrVec, activeMask, mlir::ValueRange{lh.splatFlt(0.0)}, /*alignment=*/4);
                } else {
                    mlir::Value rVec32 = toI32V(row);
                    mlir::Value slotVec = mlir::LLVM::AddOp::create(
                        lb, lloc,
                        mlir::LLVM::MulOp::create(
                            lb, lloc, mlir::LLVM::AddOp::create(lb, lloc, lh.splatInt(i32Ty, sig->startRow), rVec32),
                            lh.splatInt(i32Ty, 4)),
                        lh.splatInt(i32Ty, sig->startCol + col));
                    mlir::Value inBounds = mlir::LLVM::ICmpOp::create(
                        lb, lloc, mlir::LLVM::ICmpPredicate::ult, slotVec, lh.splatInt(i32Ty, numGSInputScalars));
                    mlir::Value gatherMask = mlir::LLVM::AndOp::create(lb, lloc, activeMask, inBounds);
                    mlir::Value planeIdx = mlir::LLVM::AddOp::create(
                        lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, vVec32, lh.splatInt(i32Ty, numGSInputScalars)),
                        slotVec);
                    mlir::Value elemIdx = mlir::LLVM::AddOp::create(
                        lb, lloc, mlir::LLVM::MulOp::create(lb, lloc, planeIdx, lh.splatInt(i32Ty, simdWidth)),
                        lh.laneIota());
                    mlir::Value byteOff = mlir::LLVM::MulOp::create(
                        lb, lloc, mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(lh.i64Ty()), elemIdx),
                        lh.splatInt(lh.i64Ty(), 4));
                    mlir::Value ptrVec = mlir::LLVM::GEPOp::create(
                        lb, lloc, lh.vecTy(ptrTy), lh.i8Ty(),
                        llvmSplat(lb, lloc, llvm::cast<mlir::VectorType>(lh.vecTy(ptrTy)).getElementType(), inAttrs, llvm::cast<mlir::VectorType>(lh.vecTy(ptrTy)).getNumElements()),
                        mlir::ValueRange{byteOff});
                    vecVal = mlir::LLVM::masked_gather::create(
                        lb, lloc, lh.vecTy(f32Ty), ptrVec, gatherMask, mlir::ValueRange{lh.splatFlt(0.0)}, /*alignment=*/4);
                }
            }
            if (scalarTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(scalarTy)) {
                    vecVal = mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(i32Ty), vecVal);
                    if (scalarTy != i32Ty) {
                        vecVal = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), vecVal);
                    }
                } else {
                    vecVal = mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), vecVal);
                }
            }
            out = {SimdVal::Kind::Vector, scalarTy, vecVal, {}, {}};
            return true;
        };

        gsHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
            IRHelpers lh{lb, lloc, simdWidth};
            mlir::Value v = (scalarTy == i32Ty)
                                ? primIdVec
                                : mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), primIdVec).getResult();
            out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
            return true;
        };

        gsHooks.loadGSInstanceId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
            IRHelpers lh{lb, lloc, simdWidth};
            mlir::Value v = gsInstIdArg;
            if (scalarTy != i32Ty) v = mlir::LLVM::TruncOp::create(lb, lloc, scalarTy, v);
            out = {SimdVal::Kind::Vector, scalarTy, llvmSplat(lb, lloc, llvm::cast<mlir::VectorType>(lh.vecTy(scalarTy)).getElementType(), v, llvm::cast<mlir::VectorType>(lh.vecTy(scalarTy)).getNumElements()), {}, {}};
            return true;
        };

        gsHooks.storeOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                                  const SimdVal& row, uint32_t col, const SimdVal& val, mlir::Value writeMask) -> bool {
            const DxilSigRecord* sig = gsLoaded.findOutputSig(sigId);
            if (!sig || val.kind != SimdVal::Kind::Vector) return true;
            IRHelpers sh{sb, sloc, simdWidth};
            mlir::Value vF32 = val.vec;
            mlir::Type elemTy = llvm::cast<mlir::VectorType>(vF32.getType()).getElementType();
            if (elemTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(elemTy)) {
                    if (elemTy.getIntOrFloatBitWidth() < 32) {
                        vF32 = mlir::LLVM::ZExtOp::create(sb, sloc, sh.vecTy(i32Ty), vF32);
                    }
                    vF32 = mlir::LLVM::BitcastOp::create(sb, sloc, sh.vecTy(f32Ty), vF32);
                } else {
                    vF32 = mlir::LLVM::FPExtOp::create(sb, sloc, sh.vecTy(f32Ty), vF32);
                }
            }
            auto storeStaged = [&](mlir::Value basePtr, uint32_t slotIdx) {
                mlir::Value slotPtr = sh.gepBytes(basePtr, static_cast<int64_t>(slotIdx * simdWidth * 4));
                mlir::Value prev = sh.loadScalar(sh.vecTy(f32Ty), slotPtr);
                mlir::Value effMask = writeMask ? writeMask : activeMask;
                sh.storeScalar(mlir::LLVM::SelectOp::create(sb, sloc, effMask, vF32, prev), slotPtr);
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

        gsHooks.emitStream = [&](mlir::OpBuilder& eb, mlir::Location eloc, uint32_t streamId, mlir::Value realMask) -> bool {
            if (streamId != 0) return true;
            IRHelpers eh{eb, eloc, simdWidth};
            mlir::Value curCountVec = eh.loadScalar(eh.vecTy(i32Ty), emitCountPtr);
            mlir::Value hasRoomVec = mlir::LLVM::ICmpOp::create(
                eb, eloc, mlir::LLVM::ICmpPredicate::ult, curCountVec, eh.splatInt(i32Ty, maxVerts));
            mlir::Value doEmitMask = mlir::LLVM::AndOp::create(eb, eloc, realMask, hasRoomVec);
            mlir::Value anyEmit = eh.orReduce(doEmitMask);
            mlir::scf::IfOp::create(eb, eloc, anyEmit, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                IRHelpers nh{nb, nloc, simdWidth};
                mlir::Value posVecs[4];
                for (uint32_t c = 0; c < 4; ++c) {
                    posVecs[c] = nh.loadScalar(nh.vecTy(f32Ty), nh.gepBytes(stagingPos, static_cast<int64_t>(c * simdWidth * 4)));
                }
                llvm::SmallVector<mlir::Value, 16> attrVecs(outAttrScalars);
                for (uint32_t s = 0; s < outAttrScalars; ++s) {
                    attrVecs[s] = nh.loadScalar(nh.vecTy(f32Ty), nh.gepBytes(stagingAttrs, static_cast<int64_t>(s * simdWidth * 4)));
                }

                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value laneDo = llvmExtract(nb, nloc, doEmitMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::scf::IfOp::create(nb, nloc, laneDo, [&](mlir::OpBuilder& lb2, mlir::Location lloc2) {
                        IRHelpers lh2{lb2, lloc2, simdWidth};
                        mlir::Value cL = llvmExtract(lb2, lloc2, curCountVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value vIdx = mlir::LLVM::AddOp::create(
                            lb2, lloc2, lh2.cInt(i32Ty, static_cast<int64_t>(l * maxVerts)), cL);
                        mlir::Value vIdx64 = mlir::LLVM::ZExtOp::create(lb2, lloc2, lh2.i64Ty(), vIdx);
                        mlir::Value posBase = mlir::LLVM::MulOp::create(lb2, lloc2, vIdx64, lh2.cInt(lh2.i64Ty(), 16));
                        for (uint32_t c = 0; c < 4; ++c) {
                            mlir::Value pv = llvmExtract(lb2, lloc2, posVecs[c], static_cast<uint32_t>(static_cast<int64_t>(l)));
                            mlir::Value off = mlir::LLVM::AddOp::create(
                                lb2, lloc2, posBase, lh2.cInt(lh2.i64Ty(), static_cast<int64_t>(c * 4)));
                            lh2.storeScalar(pv, lh2.gepBytesDyn(outPosXYZW, off));
                        }
                        if (outAttrScalars > 0) {
                            mlir::Value attrBase = mlir::LLVM::MulOp::create(
                                lb2, lloc2, vIdx64, lh2.cInt(lh2.i64Ty(), static_cast<int64_t>(outAttrScalars * 4)));
                            for (uint32_t s = 0; s < outAttrScalars; ++s) {
                                mlir::Value av = llvmExtract(lb2, lloc2, attrVecs[s], static_cast<uint32_t>(static_cast<int64_t>(l)));
                                mlir::Value off = mlir::LLVM::AddOp::create(
                                    lb2, lloc2, attrBase, lh2.cInt(lh2.i64Ty(), static_cast<int64_t>(s * 4)));
                                lh2.storeScalar(av, lh2.gepBytesDyn(outAttrs, off));
                            }
                        }
                        mlir::scf::YieldOp::create(lb2, lloc2);
                    });
                }

                mlir::Value nextCount = mlir::LLVM::SelectOp::create(
                    nb, nloc, doEmitMask, mlir::LLVM::AddOp::create(nb, nloc, curCountVec, nh.splatInt(i32Ty, 1)), curCountVec);
                nh.storeScalar(nextCount, emitCountPtr);
                mlir::Value csVec = nh.loadScalar(nh.vecTy(i32Ty), curStripPtr);
                mlir::Value nextCS = mlir::LLVM::SelectOp::create(
                    nb, nloc, doEmitMask, mlir::LLVM::AddOp::create(nb, nloc, csVec, nh.splatInt(i32Ty, 1)), csVec);
                nh.storeScalar(nextCS, curStripPtr);
                mlir::scf::YieldOp::create(nb, nloc);
            });
            return true;
        };

        auto emitCutForMask = [&](mlir::OpBuilder& cb, mlir::Location cloc, mlir::Value maskVec) {
            IRHelpers ch{cb, cloc, simdWidth};
            mlir::Value csVec = ch.loadScalar(ch.vecTy(i32Ty), curStripPtr);
            mlir::Value hasVertsVec = mlir::LLVM::ICmpOp::create(
                cb, cloc, mlir::LLVM::ICmpPredicate::ugt, csVec, ch.splatInt(i32Ty, 0));
            mlir::Value doCutMask = mlir::LLVM::AndOp::create(cb, cloc, maskVec, hasVertsVec);
            mlir::Value anyCut = ch.orReduce(doCutMask);
            mlir::scf::IfOp::create(cb, cloc, anyCut, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                IRHelpers nh{nb, nloc, simdWidth};
                mlir::Value nsVec = nh.loadScalar(nh.vecTy(i32Ty), numStripsPtr);
                for (uint32_t l = 0; l < simdWidth; ++l) {
                    mlir::Value laneCut = llvmExtract(nb, nloc, doCutMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::scf::IfOp::create(nb, nloc, laneCut, [&](mlir::OpBuilder& lb2, mlir::Location lloc2) {
                        IRHelpers lh2{lb2, lloc2, simdWidth};
                        mlir::Value nsL = llvmExtract(lb2, lloc2, nsVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value csL = llvmExtract(lb2, lloc2, csVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value stripIdx = mlir::LLVM::AddOp::create(
                            lb2, lloc2, lh2.cInt(i32Ty, static_cast<int64_t>(l * maxVerts)), nsL);
                        mlir::Value off = mlir::LLVM::MulOp::create(
                            lb2, lloc2, mlir::LLVM::ZExtOp::create(lb2, lloc2, lh2.i64Ty(), stripIdx),
                            lh2.cInt(lh2.i64Ty(), 4));
                        lh2.storeScalar(csL, lh2.gepBytesDyn(outStripCounts, off));
                        mlir::scf::YieldOp::create(lb2, lloc2);
                    });
                }
                nh.storeScalar(
                    mlir::LLVM::SelectOp::create(
                        nb, nloc, doCutMask, mlir::LLVM::AddOp::create(nb, nloc, nsVec, nh.splatInt(i32Ty, 1)), nsVec),
                    numStripsPtr);
                nh.storeScalar(
                    mlir::LLVM::SelectOp::create(nb, nloc, doCutMask, nh.splatInt(i32Ty, 0), csVec),
                    curStripPtr);
                mlir::scf::YieldOp::create(nb, nloc);
            });
        };

        gsHooks.cutStream = [&](mlir::OpBuilder& cb, mlir::Location cloc, uint32_t streamId, mlir::Value realMask) -> bool {
            if (streamId != 0) return true;
            emitCutForMask(cb, cloc, realMask);
            return true;
        };

        mlir::Value gsRealMask = activeMask;
        if (!packetizeRegionWithHooks(b, loc, combinedMod, gsSrcFunc.getBody(), gsLoaded, gsEffectiveLayout,
                                      simdWidth, rootCtx, activeMask, gsRealMask, gsHooks, outError)) {
            return false;
        }

        // Flush any remaining vertices in current strip across all active lanes
        emitCutForMask(b, loc, activeMask);

        mlir::LLVM::StoreOp::create(b, loc, h.loadScalar(h.vecTy(i32Ty), emitCountPtr), outTotalVerts, /*alignment=*/4);
        mlir::LLVM::StoreOp::create(b, loc, h.loadScalar(h.vecTy(i32Ty), numStripsPtr), outNumStrips, /*alignment=*/4);
        mlir::func::ReturnOp::create(b, loc);
    }

    // --- 2. Emit @<psSym>(const PixelBatchArgs* args) ---
    {
        mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
        IRHelpers h{b, loc, simdWidth};
        auto ptrTy = h.ptrTy();
        auto i32Ty = h.i32Ty();
        auto f32Ty = h.f32Ty();
        auto fnTy = b.getFunctionType({ptrTy}, {});
        auto psFn = mlir::func::FuncOp::create(b, loc, psSym, fnTy);
        mlir::Block* entry = psFn.addEntryBlock();
        b.setInsertionPointToEnd(entry);

        mlir::Value argsPtr = entry->getArgument(0);
        mlir::Value rootCtx = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, rootCtx)));
        mlir::Value isFront32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, isFrontFace)));
        mlir::Value primId32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, primitiveId)));

        mlir::Value batchCountRaw = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, batchCount)));
        mlir::Value batchCountNonZero = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::ne, batchCountRaw, h.cInt(i32Ty, 0));
        mlir::Value batchCount32 = mlir::LLVM::SelectOp::create(
            b, loc, batchCountNonZero, batchCountRaw, h.cInt(i32Ty, 1));
        mlir::Value covMasksPtrRaw = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, coverageMasks)));
        mlir::Value nullPtrVal = mlir::LLVM::ZeroOp::create(b, loc, ptrTy);
        mlir::Value covMasksIsNull = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::eq, covMasksPtrRaw, nullPtrVal);
        mlir::Value covMasksPtr = mlir::LLVM::SelectOp::create(
            b, loc, covMasksIsNull, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, coverageMask)), covMasksPtrRaw);
        mlir::Value sampleCovMasksPtr = psPixelFreqMSAA
            ? h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, sampleCoverageMasks)))
            : mlir::Value();

        mlir::Value pixelXPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, pixelX)));
        mlir::Value pixelYPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, pixelY)));
        mlir::Value baryIPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, baryI)));
        mlir::Value baryJPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, baryJ)));
        uint32_t numQuads = simdWidth / 4;
        mlir::Value quadOffPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, quadOffsets)));
        mlir::Value triDataPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, triData)));
        mlir::Value quadTriIdxPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, quadTriIndices)));
        mlir::Value isUniformTri = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::eq, quadTriIdxPtr, nullPtrVal);

        auto broadcastQuadScalars = [&](mlir::OpBuilder& qb, mlir::Location qloc, mlir::Type elemTy,
                                        llvm::ArrayRef<mlir::Value> quadVals) -> mlir::Value {
            if (numQuads == 1) {
                return llvmSplat(qb, qloc, elemTy, quadVals[0], simdWidth);
            }
            if (numQuads == 2) {
                mlir::Value q0 = llvmSplat(qb, qloc, elemTy, quadVals[0], 4);
                mlir::Value q1 = llvmSplat(qb, qloc, elemTy, quadVals[1], 4);
                return mlir::LLVM::ShuffleVectorOp::create(
                    qb, qloc, q0, q1, llvm::ArrayRef<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
            }
            if (numQuads == 4) {
                mlir::Value q0 = llvmSplat(qb, qloc, elemTy, quadVals[0], 4);
                mlir::Value q1 = llvmSplat(qb, qloc, elemTy, quadVals[1], 4);
                mlir::Value q2 = llvmSplat(qb, qloc, elemTy, quadVals[2], 4);
                mlir::Value q3 = llvmSplat(qb, qloc, elemTy, quadVals[3], 4);
                mlir::Value lo = mlir::LLVM::ShuffleVectorOp::create(
                    qb, qloc, q0, q1, llvm::ArrayRef<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
                mlir::Value hi = mlir::LLVM::ShuffleVectorOp::create(
                    qb, qloc, q2, q3, llvm::ArrayRef<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
                return mlir::LLVM::ShuffleVectorOp::create(
                    qb, qloc, lo, hi,
                    llvm::ArrayRef<int32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
            }
            mlir::Value vec = llvmSplat(qb, qloc, elemTy, quadVals[0], simdWidth);
            for (uint32_t q = 1; q < numQuads; ++q) {
                for (uint32_t s = 0; s < 4; ++s) {
                    vec = llvmInsert(qb, qloc, vec, quadVals[q], q * 4 + s);
                }
            }
            return vec;
        };

        mlir::Value tileDepthBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, tileDepthSOA)));
        mlir::Value tileStencilBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, tileStencilSOA)));
        mlir::Value tileColorBases[8]{};
        for (uint32_t rt = 0; rt < 8; ++rt) {
            tileColorBases[rt] = h.loadScalar(
                ptrTy, h.gepBytes(argsPtr, static_cast<int64_t>(offsetof(PixelBatchArgs, tileColorSOA) + rt * 8)));
        }

        bool batchEmitOk = true;
        mlir::Value zeroI64 = h.cInt(h.i64Ty(), 0);
        mlir::Value oneI64 = h.cInt(h.i64Ty(), 1);
        mlir::Value batchCountI64 = mlir::LLVM::ZExtOp::create(b, loc, h.i64Ty(), batchCount32);

        auto createScfIfResults = [&](mlir::OpBuilder& ib, mlir::Location iloc, mlir::TypeRange types, mlir::Value cond,
                                      llvm::function_ref<void(mlir::OpBuilder&, mlir::Location)> thenFn,
                                      llvm::function_ref<void(mlir::OpBuilder&, mlir::Location)> elseFn) {
            auto ifOp = mlir::scf::IfOp::create(ib, iloc, types, cond, /*withElseRegion=*/true);
            {
                mlir::OpBuilder::InsertionGuard g(ib);
                ib.setInsertionPointToStart(&ifOp.getThenRegion().front());
                thenFn(ib, iloc);
                ib.setInsertionPointToStart(&ifOp.getElseRegion().front());
                elseFn(ib, iloc);
            }
            return ifOp;
        };

        mlir::scf::ForOp::create(
            b, loc, zeroI64, batchCountI64, oneI64, mlir::ValueRange{},
            [&](mlir::OpBuilder& b, mlir::Location loc, mlir::Value batchIdxI64, mlir::ValueRange) {
                IRHelpers h{b, loc, simdWidth};
                mlir::Value covByteOff = mlir::LLVM::MulOp::create(b, loc, batchIdxI64, h.cInt(h.i64Ty(), 4));
                mlir::Value covMask32 = h.loadScalar(i32Ty, h.gepBytesDyn(covMasksPtr, covByteOff));

                mlir::Value covVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getElementType(), covMask32, llvm::cast<mlir::VectorType>(h.vecTy(i32Ty)).getNumElements());
                mlir::Value shifted = mlir::LLVM::LShrOp::create(b, loc, covVec, h.laneIota());
                mlir::Value bitOne = mlir::LLVM::AndOp::create(b, loc, shifted, h.splatInt(i32Ty, 1));
                mlir::Value initRealMask = mlir::LLVM::ICmpOp::create(
                    b, loc, mlir::LLVM::ICmpPredicate::ne, bitOne, h.splatInt(i32Ty, 0));

                mlir::Value laneVecByteOff = mlir::LLVM::MulOp::create(
                    b, loc, batchIdxI64, h.cInt(h.i64Ty(), static_cast<int64_t>(simdWidth * 4)));
                auto loadBatchVec = [&](mlir::Value basePtr) -> mlir::Value {
                    return mlir::LLVM::LoadOp::create(
                        b, loc, h.vecTy(f32Ty), h.gepBytesDyn(basePtr, laneVecByteOff), /*alignment=*/4);
                };
                mlir::Value pixelX = loadBatchVec(pixelXPtr);
                mlir::Value pixelY = loadBatchVec(pixelYPtr);
                mlir::Value baryI = loadBatchVec(baryIPtr);
                mlir::Value baryJ = loadBatchVec(baryJPtr);

                mlir::Value quadOffBatchByteOff = mlir::LLVM::MulOp::create(
                    b, loc, batchIdxI64, h.cInt(h.i64Ty(), static_cast<int64_t>(numQuads * 4)));
                mlir::Value quadOffBatchPtr = h.gepBytesDyn(quadOffPtr, quadOffBatchByteOff);
                llvm::SmallVector<mlir::Value, 4> quadOffsets(numQuads);
                for (uint32_t q = 0; q < numQuads; ++q) {
                    quadOffsets[q] = h.loadScalar(i32Ty, h.gepBytes(quadOffBatchPtr, static_cast<int64_t>(q * 4)));
                }

                auto loadQTriPtrs = [&](mlir::OpBuilder& qb, mlir::Location qloc) -> llvm::SmallVector<mlir::Value, 4> {
                    IRHelpers qh{qb, qloc, simdWidth};
                    mlir::Value qtBatchByteOff = mlir::LLVM::MulOp::create(
                        qb, qloc, batchIdxI64, qh.cInt(qh.i64Ty(), static_cast<int64_t>(numQuads * 4)));
                    mlir::Value qtBatchPtr = qh.gepBytesDyn(quadTriIdxPtr, qtBatchByteOff);
                    llvm::SmallVector<mlir::Value, 4> ptrs(numQuads);
                    for (uint32_t q = 0; q < numQuads; ++q) {
                        mlir::Value tIdx = qh.loadScalar(i32Ty, qh.gepBytes(qtBatchPtr, static_cast<int64_t>(q * 4)));
                        mlir::Value tByteOff = mlir::LLVM::MulOp::create(
                            qb, qloc, mlir::LLVM::ZExtOp::create(qb, qloc, qh.i64Ty(), tIdx),
                            qh.cInt(qh.i64Ty(), static_cast<int64_t>(sizeof(PixelTriData))));
                        ptrs[q] = qh.gepBytesDyn(triDataPtr, tByteOff);
                    }
                    return ptrs;
                };

                auto zIf = createScfIfResults(
                    b, loc, mlir::TypeRange{h.vecTy(f32Ty), h.vecTy(f32Ty), h.vecTy(f32Ty)}, isUniformTri,
                    [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                        IRHelpers th{tb, tloc, simdWidth};
                        mlir::Value triDepthPtr = th.loadScalar(ptrTy, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, triDepth)));
                        mlir::Value z0U = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, triDepthPtr), simdWidth);
                        mlir::Value dz1U = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triDepthPtr, 4)), simdWidth);
                        mlir::Value dz2U = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triDepthPtr, 8)), simdWidth);
                        mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{z0U, dz1U, dz2U});
                    },
                    [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                        IRHelpers eh{eb, eloc, simdWidth};
                        auto ptrs = loadQTriPtrs(eb, eloc);
                        llvm::SmallVector<mlir::Value, 4> z0s(numQuads), dz1s(numQuads), dz2s(numQuads);
                        for (uint32_t q = 0; q < numQuads; ++q) {
                            mlir::Value tdPtr = eh.loadScalar(ptrTy, eh.gepBytes(ptrs[q], offsetof(PixelTriData, triDepth)));
                            z0s[q] = eh.loadScalar(f32Ty, tdPtr);
                            dz1s[q] = eh.loadScalar(f32Ty, eh.gepBytes(tdPtr, 4));
                            dz2s[q] = eh.loadScalar(f32Ty, eh.gepBytes(tdPtr, 8));
                        }
                        mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{
                            broadcastQuadScalars(eb, eloc, f32Ty, z0s),
                            broadcastQuadScalars(eb, eloc, f32Ty, dz1s),
                            broadcastQuadScalars(eb, eloc, f32Ty, dz2s)});
                    });
                mlir::Value z0 = zIf.getResult(0);
                mlir::Value dz1 = zIf.getResult(1);
                mlir::Value dz2 = zIf.getResult(2);

                auto getIsFrontBoolVec = [&](mlir::OpBuilder& gb, mlir::Location gloc) -> mlir::Value {
                    IRHelpers gh{gb, gloc, simdWidth};
                    auto ffIf = createScfIfResults(
                        gb, gloc, mlir::TypeRange{gh.vecTy(gh.i1Ty())}, isUniformTri,
                        [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                            IRHelpers th{tb, tloc, simdWidth};
                            mlir::Value isFrontBool = mlir::LLVM::ICmpOp::create(
                                tb, tloc, mlir::LLVM::ICmpPredicate::ne, isFront32, th.cInt(i32Ty, 0));
                            mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{llvmSplat(tb, tloc, th.i1Ty(), isFrontBool, simdWidth)});
                        },
                        [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                            IRHelpers eh{eb, eloc, simdWidth};
                            auto ptrs = loadQTriPtrs(eb, eloc);
                            llvm::SmallVector<mlir::Value, 4> ffs(numQuads);
                            for (uint32_t q = 0; q < numQuads; ++q) {
                                mlir::Value ff32 = eh.loadScalar(i32Ty, eh.gepBytes(ptrs[q], offsetof(PixelTriData, isFrontFace)));
                                ffs[q] = mlir::LLVM::ICmpOp::create(
                                    eb, eloc, mlir::LLVM::ICmpPredicate::ne, ff32, eh.cInt(i32Ty, 0));
                            }
                            mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{broadcastQuadScalars(eb, eloc, eh.i1Ty(), ffs)});
                        });
                    return ffIf.getResult(0);
                };

                    mlir::Value interpZ = mlir::LLVM::FAddOp::create(
                    b, loc, z0,
                    mlir::LLVM::FAddOp::create(
                        b, loc, mlir::LLVM::FMulOp::create(b, loc, baryI, dz1),
                        mlir::LLVM::FMulOp::create(b, loc, baryJ, dz2)));
                if (!desc.RasterizerState.DepthClipEnable) {
                    interpZ = h.clamp01(interpZ);
                }

        // Helper lambdas for quad-swizzled depth & stencil SOA access (zero branches on loads, masked on writes)
        auto addTileOffset = [&](mlir::OpBuilder& ab, mlir::Location aloc, mlir::Value qOff, uint32_t tileElemOffset) -> mlir::Value {
            if (tileElemOffset == 0) return qOff;
            IRHelpers ah{ab, aloc, simdWidth};
            return mlir::LLVM::AddOp::create(ab, aloc, qOff, ah.cInt(i32Ty, static_cast<int64_t>(tileElemOffset)));
        };

        auto loadDepthVec = [&](mlir::OpBuilder& db, mlir::Location dloc, uint32_t tileElemOffset = 0) -> mlir::Value {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value out = dh.splatFlt(1.0);
            auto v4f = mlir::VectorType::get({4}, f32Ty);
            for (uint32_t q = 0; q < numQuads; ++q) {
                mlir::Value off = addTileOffset(db, dloc, quadOffsets[q], tileElemOffset);
                mlir::Value qByte = mlir::LLVM::MulOp::create(
                    db, dloc, mlir::LLVM::ZExtOp::create(db, dloc, dh.i64Ty(), off), dh.cInt(dh.i64Ty(), 4));
                mlir::Value qp = dh.gepBytesDyn(tileDepthBase, qByte);
                mlir::Value q4 = dh.loadScalar(v4f, qp);
                for (uint32_t s = 0; s < 4; ++s) {
                    mlir::Value sc = llvmExtract(db, dloc, q4, static_cast<uint32_t>(static_cast<int64_t>(s)));
                    out = llvmInsert(db, dloc, out, sc, static_cast<uint32_t>(static_cast<int64_t>(q * 4 + s)));
                }
            }
            return out;
        };

        auto storeDepthVec = [&](mlir::OpBuilder& db, mlir::Location dloc, mlir::Value maskVec, mlir::Value newZ, mlir::Value oldZ, uint32_t tileElemOffset = 0) {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value merged = mlir::LLVM::SelectOp::create(db, dloc, maskVec, newZ, oldZ);
            auto v4f = mlir::VectorType::get({4}, f32Ty);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::SmallVector<int32_t, 4> qIdx{static_cast<int32_t>(q * 4), static_cast<int32_t>(q * 4 + 1), static_cast<int32_t>(q * 4 + 2), static_cast<int32_t>(q * 4 + 3)};
                mlir::Value qMask = mlir::LLVM::ShuffleVectorOp::create(db, dloc, maskVec, maskVec, qIdx);
                mlir::Value anyQ = mlir::LLVM::vector_reduce_or::create(db, dloc, db.getI1Type(), qMask);
                mlir::scf::IfOp::create(db, dloc, anyQ, [&](mlir::OpBuilder& qb, mlir::Location qloc) {
                    IRHelpers qh{qb, qloc, simdWidth};
                    mlir::Value q4 = mlir::LLVM::ShuffleVectorOp::create(qb, qloc, merged, merged, qIdx);
                    mlir::Value off = addTileOffset(qb, qloc, quadOffsets[q], tileElemOffset);
                    mlir::Value qByte = mlir::LLVM::MulOp::create(
                        qb, qloc, mlir::LLVM::ZExtOp::create(qb, qloc, qh.i64Ty(), off), qh.cInt(qh.i64Ty(), 4));
                    mlir::Value qp = qh.gepBytesDyn(tileDepthBase, qByte);
                    mlir::LLVM::StoreOp::create(qb, qloc, q4, qp);
                    mlir::scf::YieldOp::create(qb, qloc);
                });
            }
        };

        auto loadStencilVec = [&](mlir::OpBuilder& db, mlir::Location dloc, uint32_t tileElemOffset = 0) -> mlir::Value {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value out = dh.splatInt(dh.i8Ty(), 0);
            auto v4i8 = mlir::VectorType::get({4}, dh.i8Ty());
            for (uint32_t q = 0; q < numQuads; ++q) {
                mlir::Value off = addTileOffset(db, dloc, quadOffsets[q], tileElemOffset);
                mlir::Value q64 = mlir::LLVM::ZExtOp::create(db, dloc, dh.i64Ty(), off);
                mlir::Value qp = dh.gepBytesDyn(tileStencilBase, q64);
                mlir::Value q4 = dh.loadScalar(v4i8, qp);
                for (uint32_t s = 0; s < 4; ++s) {
                    mlir::Value sc = llvmExtract(db, dloc, q4, static_cast<uint32_t>(static_cast<int64_t>(s)));
                    out = llvmInsert(db, dloc, out, sc, static_cast<uint32_t>(static_cast<int64_t>(q * 4 + s)));
                }
            }
            return out;
        };

        auto storeStencilVec = [&](mlir::OpBuilder& db, mlir::Location dloc, mlir::Value maskVec, mlir::Value newS, mlir::Value oldS, uint32_t tileElemOffset = 0) {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value merged = mlir::LLVM::SelectOp::create(db, dloc, maskVec, newS, oldS);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::SmallVector<int32_t, 4> qIdx{static_cast<int32_t>(q * 4), static_cast<int32_t>(q * 4 + 1), static_cast<int32_t>(q * 4 + 2), static_cast<int32_t>(q * 4 + 3)};
                mlir::Value qMask = mlir::LLVM::ShuffleVectorOp::create(db, dloc, maskVec, maskVec, qIdx);
                mlir::Value anyQ = mlir::LLVM::vector_reduce_or::create(db, dloc, db.getI1Type(), qMask);
                mlir::scf::IfOp::create(db, dloc, anyQ, [&](mlir::OpBuilder& qb, mlir::Location qloc) {
                    IRHelpers qh{qb, qloc, simdWidth};
                    mlir::Value q4 = mlir::LLVM::ShuffleVectorOp::create(qb, qloc, merged, merged, qIdx);
                    mlir::Value off = addTileOffset(qb, qloc, quadOffsets[q], tileElemOffset);
                    mlir::Value q64 = mlir::LLVM::ZExtOp::create(qb, qloc, qh.i64Ty(), off);
                    mlir::Value qp = qh.gepBytesDyn(tileStencilBase, q64);
                    mlir::LLVM::StoreOp::create(qb, qloc, q4, qp);
                    mlir::scf::YieldOp::create(qb, qloc);
                });
            }
        };

        auto runDepthStencil = [&](mlir::OpBuilder& db, mlir::Location dloc, mlir::Value curRealMask, mlir::Value candZ,
                                   mlir::Value candStencilRef8 = mlir::Value(), uint32_t tileElemOffset = 0) -> mlir::Value {
            IRHelpers dh{db, dloc, simdWidth};
            const auto& ds = desc.DepthStencilState;
            bool depthOn = ds.DepthEnable && (desc.DSVFormat != DXGI_FORMAT_UNKNOWN);
            bool stencilOn = ds.StencilEnable && (desc.DSVFormat != DXGI_FORMAT_UNKNOWN);
            if (!depthOn && !stencilOn) return curRealMask;

            mlir::Value nextMask = curRealMask;
            mlir::Value stencilPassMask = dh.allTrueMask();
            mlir::Value oldStencil;
            mlir::Value ref8;
            mlir::Value writeMask8;
            D3D12_DEPTH_STENCILOP_DESC activeSOp = ds.FrontFace;

            if (stencilOn) {
                oldStencil = loadStencilVec(db, dloc, tileElemOffset);
                if (candStencilRef8) {
                    ref8 = candStencilRef8;
                } else {
                    mlir::Value ref32 = dh.loadScalar(i32Ty, dh.gepBytes(rootCtx, offsetof(DriverRootContext, stencil_ref)));
                    mlir::Value refScalar8 = mlir::LLVM::TruncOp::create(db, dloc, dh.i8Ty(), ref32);
                    ref8 = llvmSplat(db, dloc, llvm::cast<mlir::VectorType>(dh.vecTy(dh.i8Ty())).getElementType(), refScalar8, llvm::cast<mlir::VectorType>(dh.vecTy(dh.i8Ty())).getNumElements());
                }
                mlir::Value rMask8 = dh.splatInt(dh.i8Ty(), ds.StencilReadMask);
                writeMask8 = dh.splatInt(dh.i8Ty(), ds.StencilWriteMask);
                mlir::Value refMasked = mlir::LLVM::AndOp::create(db, dloc, ref8, rMask8);
                mlir::Value bufMasked = mlir::LLVM::AndOp::create(db, dloc, oldStencil, rMask8);
                mlir::Value isFrontBoolVec = getIsFrontBoolVec(db, dloc);
                mlir::Value frontPass = dh.evalStencilCmp(ds.FrontFace.StencilFunc, refMasked, bufMasked);
                mlir::Value backPass = dh.evalStencilCmp(ds.BackFace.StencilFunc, refMasked, bufMasked);
                stencilPassMask = mlir::LLVM::SelectOp::create(
                    db, dloc, isFrontBoolVec, frontPass, backPass);
                nextMask = mlir::LLVM::AndOp::create(db, dloc, nextMask, stencilPassMask);
            }

            mlir::Value zPassMask = dh.allTrueMask();
            if (depthOn) {
                mlir::Value oldZ = loadDepthVec(db, dloc, tileElemOffset);
                zPassMask = dh.evalDepthCmp(ds.DepthFunc, candZ, oldZ);
                nextMask = mlir::LLVM::AndOp::create(db, dloc, nextMask, zPassMask);
                if (ds.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL) {
                    storeDepthVec(db, dloc, nextMask, candZ, oldZ, tileElemOffset);
                }
            }

            if (stencilOn && ds.StencilWriteMask != 0) {
                mlir::Value sFailMask = mlir::LLVM::AndOp::create(
                    db, dloc, curRealMask, mlir::LLVM::XOrOp::create(db, dloc, stencilPassMask, dh.allTrueMask()));
                mlir::Value zFailMask = mlir::LLVM::AndOp::create(
                    db, dloc, mlir::LLVM::AndOp::create(db, dloc, curRealMask, stencilPassMask),
                    mlir::LLVM::XOrOp::create(db, dloc, zPassMask, dh.allTrueMask()));
                mlir::Value sUpdated = oldStencil;
                if (activeSOp.StencilFailOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = mlir::LLVM::SelectOp::create(
                        db, dloc, sFailMask, dh.applyStencilOp(activeSOp.StencilFailOp, oldStencil, ref8), sUpdated);
                }
                if (activeSOp.StencilDepthFailOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = mlir::LLVM::SelectOp::create(
                        db, dloc, zFailMask, dh.applyStencilOp(activeSOp.StencilDepthFailOp, oldStencil, ref8), sUpdated);
                }
                if (activeSOp.StencilPassOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = mlir::LLVM::SelectOp::create(
                        db, dloc, nextMask, dh.applyStencilOp(activeSOp.StencilPassOp, oldStencil, ref8), sUpdated);
                }
                mlir::Value keepBits = dh.splatInt(dh.i8Ty(), static_cast<uint8_t>(~ds.StencilWriteMask));
                mlir::Value maskedNewS = mlir::LLVM::OrOp::create(
                    db, dloc, mlir::LLVM::AndOp::create(db, dloc, oldStencil, keepBits),
                    mlir::LLVM::AndOp::create(db, dloc, sUpdated, writeMask8));
                storeStencilVec(db, dloc, curRealMask, maskedNewS, oldStencil, tileElemOffset);
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

        llvm::SmallVector<mlir::Value, 8> msaaSampleMasks;
        if (psPixelFreqMSAA) {
            mlir::Value sampleCovNotNull = mlir::LLVM::ICmpOp::create(
                b, loc, mlir::LLVM::ICmpPredicate::ne, sampleCovMasksPtr, nullPtrVal);
            msaaSampleMasks.resize(msaaSampleCount);
            for (uint32_t s = 0; s < msaaSampleCount; ++s) {
                mlir::Value sMask = mlir::scf::IfOp::create(
                    b, loc, sampleCovNotNull,
                    [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                        IRHelpers sh{sb, sloc, simdWidth};
                        mlir::Value sIdx = mlir::LLVM::AddOp::create(
                            sb, sloc,
                            mlir::LLVM::MulOp::create(sb, sloc, batchIdxI64, sh.cInt(h.i64Ty(), static_cast<int64_t>(msaaSampleCount))),
                            sh.cInt(h.i64Ty(), static_cast<int64_t>(s)));
                        mlir::Value sByteOff = mlir::LLVM::MulOp::create(sb, sloc, sIdx, sh.cInt(h.i64Ty(), 4));
                        mlir::Value sMask32 = sh.loadScalar(i32Ty, sh.gepBytesDyn(sampleCovMasksPtr, sByteOff));
                        mlir::Value sVec = llvmSplat(sb, sloc, llvm::cast<mlir::VectorType>(sh.vecTy(i32Ty)).getElementType(), sMask32, simdWidth);
                        mlir::Value sShifted = mlir::LLVM::LShrOp::create(sb, sloc, sVec, sh.laneIota());
                        mlir::Value sBitOne = mlir::LLVM::AndOp::create(sb, sloc, sShifted, sh.splatInt(i32Ty, 1));
                        mlir::Value m = mlir::LLVM::ICmpOp::create(
                            sb, sloc, mlir::LLVM::ICmpPredicate::ne, sBitOne, sh.splatInt(i32Ty, 0));
                        mlir::scf::YieldOp::create(sb, sloc, mlir::ValueRange{m});
                    },
                    [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                        mlir::scf::YieldOp::create(sb, sloc, mlir::ValueRange{initRealMask});
                    }).getResult(0);
                msaaSampleMasks[s] = sMask;
            }
        }

        mlir::Value realMaskAfterEarly = initRealMask;
        llvm::SmallVector<mlir::Value, 8> msaaEarlyDSMasks;
        if (canEarlyZ) {
            if (psPixelFreqMSAA) {
                msaaEarlyDSMasks.resize(msaaSampleCount);
                for (uint32_t s = 0; s < msaaSampleCount; ++s) {
                    msaaEarlyDSMasks[s] = runDepthStencil(
                        b, loc, msaaSampleMasks[s], interpZ, mlir::Value(), s * kRasterTilePixels);
                    realMaskAfterEarly = (s == 0)
                        ? msaaEarlyDSMasks[0]
                        : mlir::LLVM::OrOp::create(b, loc, realMaskAfterEarly, msaaEarlyDSMasks[s]);
                }
            } else {
                realMaskAfterEarly = runDepthStencil(b, loc, initRealMask, interpZ);
            }
        }

        if (hasPS) {
            mlir::func::FuncOp psSrcFunc = psMlirMod->lookupSymbol<mlir::func::FuncOp>(psLoaded.entryName);
            if (!psSrcFunc || psSrcFunc.getBody().empty()) {
                outError = "Pixel shader entry function missing after MLIR SCF lift";
                batchEmitOk = false;
                mlir::scf::YieldOp::create(b, loc);
                return;
            }

            auto emitPSBodyAndOM = [&](mlir::OpBuilder& pb, mlir::Location ploc, mlir::Value enterRealMask) {
                IRHelpers ph{pb, ploc, simdWidth};
                mlir::Value curRealMask = enterRealMask;
                mlir::Value execMask = psLoaded.hasDerivativesOrImplicitSample
                                           ? ph.expandMaskQuads(curRealMask)
                                           : curRealMask;

                // Interpolate PS inputs
                mlir::Value wVec = ph.splatFlt(1.0);
                mlir::Value invWVec = ph.splatFlt(1.0);
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
                    auto iwIf = createScfIfResults(
                        pb, ploc, mlir::TypeRange{ph.vecTy(f32Ty), ph.vecTy(f32Ty), ph.vecTy(f32Ty)}, isUniformTri,
                        [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                            IRHelpers th{tb, tloc, simdWidth};
                            mlir::Value triInvWPtr = th.loadScalar(ptrTy, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, triInvW)));
                            mlir::Value iw0U = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, triInvWPtr), simdWidth);
                            mlir::Value diw1U = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triInvWPtr, 4)), simdWidth);
                            mlir::Value diw2U = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triInvWPtr, 8)), simdWidth);
                            mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{iw0U, diw1U, diw2U});
                        },
                        [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                            IRHelpers eh{eb, eloc, simdWidth};
                            auto ptrs = loadQTriPtrs(eb, eloc);
                            llvm::SmallVector<mlir::Value, 4> iw0s(numQuads), diw1s(numQuads), diw2s(numQuads);
                            for (uint32_t q = 0; q < numQuads; ++q) {
                                mlir::Value p = eh.loadScalar(ptrTy, eh.gepBytes(ptrs[q], offsetof(PixelTriData, triInvW)));
                                iw0s[q] = eh.loadScalar(f32Ty, p);
                                diw1s[q] = eh.loadScalar(f32Ty, eh.gepBytes(p, 4));
                                diw2s[q] = eh.loadScalar(f32Ty, eh.gepBytes(p, 8));
                            }
                            mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{
                                broadcastQuadScalars(eb, eloc, f32Ty, iw0s),
                                broadcastQuadScalars(eb, eloc, f32Ty, diw1s),
                                broadcastQuadScalars(eb, eloc, f32Ty, diw2s)});
                        });
                    mlir::Value iw0 = iwIf.getResult(0);
                    mlir::Value diw1 = iwIf.getResult(1);
                    mlir::Value diw2 = iwIf.getResult(2);
                    invWVec = mlir::LLVM::FAddOp::create(
                        pb, ploc, iw0,
                        mlir::LLVM::FAddOp::create(
                            pb, ploc, mlir::LLVM::FMulOp::create(pb, ploc, baryI, diw1),
                            mlir::LLVM::FMulOp::create(pb, ploc, baryJ, diw2)));
                    mlir::Value safeInvW = mlir::LLVM::SelectOp::create(
                        pb, ploc,
                        mlir::LLVM::FCmpOp::create(pb, ploc, mlir::LLVM::FCmpPredicate::ogt,
                                                    mlir::LLVM::FAbsOp::create(pb, ploc, invWVec), ph.splatFlt(1e-20)),
                        invWVec, ph.splatFlt(1.0));
                    wVec = mlir::LLVM::FDivOp::create(pb, ploc, ph.splatFlt(1.0), safeInvW);
                }

                // Cache interpolated scalar registers (max 32 * 4 = 128 scalars)
                struct ActiveInterpSlot {
                    uint32_t slot = 0;
                    uint32_t interpMode = 0;
                    bool isFlat = false;
                };
                llvm::SmallVector<ActiveInterpSlot, 32> activeSlots;
                for (const auto& pi : psLoaded.inputSigs) {
                    if (pi.systemValueType != 0) continue;
                    for (uint32_t r = 0; r < pi.rows; ++r) {
                        for (uint32_t c = 0; c < pi.cols && (pi.startCol + c) < 4; ++c) {
                            uint32_t slot = (pi.startRow + r) * 4 + (pi.startCol + c);
                            if (slot >= 128) continue;
                            bool isMsPrimAttr = (slot < 64) && ((msPrimAttrMask & (1ull << slot)) != 0);
                            activeSlots.push_back({slot, pi.interpMode, (pi.interpMode == 1 || isMsPrimAttr)});
                        }
                    }
                }

                std::array<mlir::Value, 128> interpScalars{};
                if (!activeSlots.empty()) {
                    llvm::SmallVector<mlir::Type, 32> slotTypes(activeSlots.size(), ph.vecTy(f32Ty));
                    auto slotsIf = createScfIfResults(
                        pb, ploc, slotTypes, isUniformTri,
                        [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                            IRHelpers th{tb, tloc, simdWidth};
                            mlir::Value triAttrBasePtr = th.loadScalar(ptrTy, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, triAttrBase)));
                            mlir::Value triAttrFlatPtr = th.loadScalar(ptrTy, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, triAttrFlat)));
                            llvm::SmallVector<mlir::Value, 32> vals;
                            vals.reserve(activeSlots.size());
                            for (const auto& as : activeSlots) {
                                if (as.isFlat) {
                                    mlir::Value fv = th.loadScalar(f32Ty, th.gepBytes(triAttrFlatPtr, static_cast<int64_t>(as.slot * 4)));
                                    vals.push_back(llvmSplat(tb, tloc, f32Ty, fv, simdWidth));
                                } else {
                                    int64_t baseOff = static_cast<int64_t>(as.slot * 3 * 4);
                                    mlir::Value a0 = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triAttrBasePtr, baseOff)), simdWidth);
                                    mlir::Value da1 = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triAttrBasePtr, baseOff + 4)), simdWidth);
                                    mlir::Value da2 = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triAttrBasePtr, baseOff + 8)), simdWidth);
                                    mlir::Value lin = mlir::LLVM::FAddOp::create(
                                        tb, tloc, a0,
                                        mlir::LLVM::FAddOp::create(
                                            tb, tloc, mlir::LLVM::FMulOp::create(tb, tloc, baryI, da1),
                                            mlir::LLVM::FMulOp::create(tb, tloc, baryJ, da2)));
                                    vals.push_back((as.interpMode == 4)
                                                       ? lin
                                                       : mlir::LLVM::FMulOp::create(tb, tloc, lin, wVec).getResult());
                                }
                            }
                            mlir::scf::YieldOp::create(tb, tloc, vals);
                        },
                        [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                            IRHelpers eh{eb, eloc, simdWidth};
                            auto ptrs = loadQTriPtrs(eb, eloc);
                            llvm::SmallVector<mlir::Value, 4> qBasePtrs(numQuads), qFlatPtrs(numQuads);
                            bool anyFlat = false, anyInterp = false;
                            for (const auto& as : activeSlots) {
                                if (as.isFlat) anyFlat = true;
                                else anyInterp = true;
                            }
                            for (uint32_t q = 0; q < numQuads; ++q) {
                                if (anyInterp) {
                                    qBasePtrs[q] = eh.loadScalar(ptrTy, eh.gepBytes(ptrs[q], offsetof(PixelTriData, triAttrBase)));
                                }
                                if (anyFlat) {
                                    qFlatPtrs[q] = eh.loadScalar(ptrTy, eh.gepBytes(ptrs[q], offsetof(PixelTriData, triAttrFlat)));
                                }
                            }
                            llvm::SmallVector<mlir::Value, 32> vals;
                            vals.reserve(activeSlots.size());
                            for (const auto& as : activeSlots) {
                                if (as.isFlat) {
                                    llvm::SmallVector<mlir::Value, 4> fvs(numQuads);
                                    for (uint32_t q = 0; q < numQuads; ++q) {
                                        fvs[q] = eh.loadScalar(f32Ty, eh.gepBytes(qFlatPtrs[q], static_cast<int64_t>(as.slot * 4)));
                                    }
                                    vals.push_back(broadcastQuadScalars(eb, eloc, f32Ty, fvs));
                                } else {
                                    int64_t baseOff = static_cast<int64_t>(as.slot * 3 * 4);
                                    llvm::SmallVector<mlir::Value, 4> a0s(numQuads), da1s(numQuads), da2s(numQuads);
                                    for (uint32_t q = 0; q < numQuads; ++q) {
                                        a0s[q] = eh.loadScalar(f32Ty, eh.gepBytes(qBasePtrs[q], baseOff));
                                        da1s[q] = eh.loadScalar(f32Ty, eh.gepBytes(qBasePtrs[q], baseOff + 4));
                                        da2s[q] = eh.loadScalar(f32Ty, eh.gepBytes(qBasePtrs[q], baseOff + 8));
                                    }
                                    mlir::Value a0 = broadcastQuadScalars(eb, eloc, f32Ty, a0s);
                                    mlir::Value da1 = broadcastQuadScalars(eb, eloc, f32Ty, da1s);
                                    mlir::Value da2 = broadcastQuadScalars(eb, eloc, f32Ty, da2s);
                                    mlir::Value lin = mlir::LLVM::FAddOp::create(
                                        eb, eloc, a0,
                                        mlir::LLVM::FAddOp::create(
                                            eb, eloc, mlir::LLVM::FMulOp::create(eb, eloc, baryI, da1),
                                            mlir::LLVM::FMulOp::create(eb, eloc, baryJ, da2)));
                                    vals.push_back((as.interpMode == 4)
                                                       ? lin
                                                       : mlir::LLVM::FMulOp::create(eb, eloc, lin, wVec).getResult());
                                }
                            }
                            mlir::scf::YieldOp::create(eb, eloc, vals);
                        });
                    for (size_t i = 0; i < activeSlots.size(); ++i) {
                        interpScalars[activeSlots[i].slot] = slotsIf.getResult(static_cast<unsigned>(i));
                    }
                }

                // Allocate stack slots for PS outputs (SV_Target0..7 and optional SV_Depth) in the function entry block
                // so batch loop iterations do not grow the stack dynamically.
                mlir::OpBuilder entryAllocBuilder(entry, entry->begin());
                IRHelpers eh{entryAllocBuilder, ploc, simdWidth};
                std::array<std::array<mlir::Value, 4>, 8> rtOutPtrs{};
                for (uint32_t rt = 0; rt < 8; ++rt) {
                    if (rt < desc.NumRenderTargets && desc.RTVFormats[rt] != DXGI_FORMAT_UNKNOWN) {
                        for (uint32_t c = 0; c < 4; ++c) {
                            rtOutPtrs[rt][c] = mlir::LLVM::AllocaOp::create(
                                entryAllocBuilder, ploc, ptrTy, eh.vecTy(f32Ty), eh.cInt(i32Ty, 1), std::max<uint32_t>(64, ph.W * 8));
                            ph.storeScalar(ph.splatFlt(c == 3 ? 1.0 : 0.0), rtOutPtrs[rt][c]);
                        }
                    }
                }
                mlir::Value outDepthPtr = mlir::LLVM::AllocaOp::create(
                    entryAllocBuilder, ploc, ptrTy, eh.vecTy(f32Ty), eh.cInt(i32Ty, 1), std::max<uint32_t>(64, ph.W * 8));
                ph.storeScalar(interpZ, outDepthPtr);

                mlir::Value outStencilRefPtr;
                if (psWritesStencilRef) {
                    outStencilRefPtr = mlir::LLVM::AllocaOp::create(
                        entryAllocBuilder, ploc, ptrTy, eh.vecTy(eh.i8Ty()), eh.cInt(i32Ty, 1), std::max<uint32_t>(64, ph.W * 8));
                    mlir::Value initRef32 = ph.loadScalar(i32Ty, ph.gepBytes(rootCtx, offsetof(DriverRootContext, stencil_ref)));
                    mlir::Value initRef8 = mlir::LLVM::TruncOp::create(pb, ploc, ph.i8Ty(), initRef32);
                    ph.storeScalar(llvmSplat(pb, ploc, llvm::cast<mlir::VectorType>(ph.vecTy(ph.i8Ty())).getElementType(), initRef8, llvm::cast<mlir::VectorType>(ph.vecTy(ph.i8Ty())).getNumElements()), outStencilRefPtr);
                }

                StageHooks psHooks;
                psHooks.isPixelShader = true;
                psHooks.entryBlock = entry;
                psHooks.loadInput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                        const SimdVal& row, uint32_t col, mlir::Type scalarTy, SimdVal& out) -> bool {
                    const DxilSigRecord* sig = psLoaded.findInputSig(sigId);
                    if (!sig) return false;
                    IRHelpers lh{lb, lloc, simdWidth};
                    if (sig->systemValueType == 3) { // SV_Position
                        mlir::Value v = (col == 0) ? pixelX : ((col == 1) ? pixelY : ((col == 2) ? interpZ : invWVec));
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 5) { // SV_ViewportArrayIndex
                        auto vpIf = createScfIfResults(
                            lb, lloc, mlir::TypeRange{lh.vecTy(i32Ty)}, isUniformTri,
                            [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                                IRHelpers th{tb, tloc, simdWidth};
                                mlir::Value vp32 = th.loadScalar(
                                    i32Ty, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, viewportArrayIndex)));
                                mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{llvmSplat(tb, tloc, i32Ty, vp32, simdWidth)});
                            },
                            [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                                IRHelpers eh{eb, eloc, simdWidth};
                                auto ptrs = loadQTriPtrs(eb, eloc);
                                llvm::SmallVector<mlir::Value, 4> vps(numQuads);
                                for (uint32_t q = 0; q < numQuads; ++q) {
                                    vps[q] = eh.loadScalar(i32Ty, eh.gepBytes(ptrs[q], offsetof(PixelTriData, viewportArrayIndex)));
                                }
                                mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{broadcastQuadScalars(eb, eloc, i32Ty, vps)});
                            });
                        mlir::Value v = vpIf.getResult(0);
                        if (scalarTy != i32Ty) {
                            v = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 10) { // SV_PrimitiveID
                        auto pidIf = createScfIfResults(
                            lb, lloc, mlir::TypeRange{lh.vecTy(i32Ty)}, isUniformTri,
                            [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                                mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{llvmSplat(tb, tloc, i32Ty, primId32, simdWidth)});
                            },
                            [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                                IRHelpers eh{eb, eloc, simdWidth};
                                auto ptrs = loadQTriPtrs(eb, eloc);
                                llvm::SmallVector<mlir::Value, 4> pids(numQuads);
                                for (uint32_t q = 0; q < numQuads; ++q) {
                                    pids[q] = eh.loadScalar(i32Ty, eh.gepBytes(ptrs[q], offsetof(PixelTriData, primitiveId)));
                                }
                                mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{broadcastQuadScalars(eb, eloc, i32Ty, pids)});
                            });
                        mlir::Value v = pidIf.getResult(0);
                        if (scalarTy != i32Ty) {
                            v = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 28) { // SV_Barycentrics
                        mlir::Value v = lh.splatFlt(0.0);
                        if (sig->interpMode == 4) { // noperspective
                            mlir::Value b0 = mlir::LLVM::FSubOp::create(
                                lb, lloc, lh.splatFlt(1.0),
                                mlir::LLVM::FAddOp::create(lb, lloc, baryI, baryJ));
                            if (col == 0) v = b0;
                            else if (col == 1) v = baryI;
                            else if (col == 2) v = baryJ;
                        } else if (col < 3) { // perspective (linear) using original-primitive barycentrics
                            int64_t baseOff = static_cast<int64_t>(col * 3 * 4);
                            auto obIf = createScfIfResults(
                                lb, lloc, mlir::TypeRange{lh.vecTy(f32Ty), lh.vecTy(f32Ty), lh.vecTy(f32Ty)}, isUniformTri,
                                [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                                    IRHelpers th{tb, tloc, simdWidth};
                                    mlir::Value triOrigBaryPtr = th.loadScalar(
                                        ptrTy, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, triOrigBary)));
                                    mlir::Value a0 = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triOrigBaryPtr, baseOff)), simdWidth);
                                    mlir::Value da1 = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triOrigBaryPtr, baseOff + 4)), simdWidth);
                                    mlir::Value da2 = llvmSplat(tb, tloc, f32Ty, th.loadScalar(f32Ty, th.gepBytes(triOrigBaryPtr, baseOff + 8)), simdWidth);
                                    mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{a0, da1, da2});
                                },
                                [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                                    IRHelpers eh{eb, eloc, simdWidth};
                                    auto ptrs = loadQTriPtrs(eb, eloc);
                                    llvm::SmallVector<mlir::Value, 4> a0s(numQuads), da1s(numQuads), da2s(numQuads);
                                    for (uint32_t q = 0; q < numQuads; ++q) {
                                        mlir::Value p = eh.loadScalar(ptrTy, eh.gepBytes(ptrs[q], offsetof(PixelTriData, triOrigBary)));
                                        a0s[q] = eh.loadScalar(f32Ty, eh.gepBytes(p, baseOff));
                                        da1s[q] = eh.loadScalar(f32Ty, eh.gepBytes(p, baseOff + 4));
                                        da2s[q] = eh.loadScalar(f32Ty, eh.gepBytes(p, baseOff + 8));
                                    }
                                    mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{
                                        broadcastQuadScalars(eb, eloc, f32Ty, a0s),
                                        broadcastQuadScalars(eb, eloc, f32Ty, da1s),
                                        broadcastQuadScalars(eb, eloc, f32Ty, da2s)});
                                });
                            mlir::Value lin = mlir::LLVM::FAddOp::create(
                                lb, lloc, obIf.getResult(0),
                                mlir::LLVM::FAddOp::create(
                                    lb, lloc, mlir::LLVM::FMulOp::create(lb, lloc, baryI, obIf.getResult(1)),
                                    mlir::LLVM::FMulOp::create(lb, lloc, baryJ, obIf.getResult(2))));
                            v = mlir::LLVM::FMulOp::create(lb, lloc, lin, wVec);
                        }
                        if (scalarTy != f32Ty) {
                            v = mlir::LLVM::FPTruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 29) { // SV_ShadingRate
                        auto srIf = createScfIfResults(
                            lb, lloc, mlir::TypeRange{lh.vecTy(i32Ty)}, isUniformTri,
                            [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                                IRHelpers th{tb, tloc, simdWidth};
                                mlir::Value sr32 = th.loadScalar(
                                    i32Ty, th.gepBytes(argsPtr, offsetof(PixelBatchArgs, shadingRate)));
                                mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{llvmSplat(tb, tloc, i32Ty, sr32, simdWidth)});
                            },
                            [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                                IRHelpers eh{eb, eloc, simdWidth};
                                auto ptrs = loadQTriPtrs(eb, eloc);
                                llvm::SmallVector<mlir::Value, 4> srs(numQuads);
                                for (uint32_t q = 0; q < numQuads; ++q) {
                                    srs[q] = eh.loadScalar(i32Ty, eh.gepBytes(ptrs[q], offsetof(PixelTriData, shadingRate)));
                                }
                                mlir::scf::YieldOp::create(eb, eloc, mlir::ValueRange{broadcastQuadScalars(eb, eloc, i32Ty, srs)});
                            });
                        mlir::Value v = srIf.getResult(0);
                        if (scalarTy != i32Ty) {
                            v = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 13) { // SV_IsFrontFace
                        mlir::Value ffVec = getIsFrontBoolVec(lb, lloc);
                        mlir::Value v = (scalarTy == lh.i1Ty())
                                            ? ffVec
                                            : mlir::LLVM::ZExtOp::create(lb, lloc, lh.vecTy(i32Ty), ffVec).getResult();
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    auto rowConst = tryGetUniformConstUInt(row);
                    mlir::Value v;
                    if (rowConst.has_value() || sig->rows <= 1) {
                        uint32_t r = rowConst.value_or(0);
                        uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                        v = (slot < 128 && interpScalars[slot]) ? interpScalars[slot] : lh.splatFlt(0.0);
                    } else {
                        uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                        v = (baseSlot < 128 && interpScalars[baseSlot]) ? interpScalars[baseSlot] : lh.splatFlt(0.0);
                        mlir::Type rElemTy = llvm::cast<mlir::VectorType>(row.vec.getType()).getElementType();
                        for (uint32_t r = 1; r < sig->rows; ++r) {
                            uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                            if (slot >= 128 || !interpScalars[slot]) continue;
                            mlir::Value cmp = mlir::LLVM::ICmpOp::create(
                                lb, lloc, mlir::LLVM::ICmpPredicate::eq, row.vec, lh.splatInt(rElemTy, r));
                            v = mlir::LLVM::SelectOp::create(lb, lloc, cmp, interpScalars[slot], v);
                        }
                    }
                    if (scalarTy != f32Ty) {
                        if (mlir::isa<mlir::IntegerType>(scalarTy)) {
                            v = mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(i32Ty), v);
                            if (scalarTy != i32Ty) {
                                v = mlir::LLVM::TruncOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                            }
                        } else {
                            v = mlir::LLVM::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                    }
                    out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                    return true;
                };

                psHooks.storeOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                                          const SimdVal&, uint32_t col, const SimdVal& val, mlir::Value writeMask) -> bool {
                    const DxilSigRecord* sig = psLoaded.findOutputSig(sigId);
                    if (!sig || val.kind != SimdVal::Kind::Vector) return true;
                    IRHelpers sh{sb, sloc, simdWidth};
                    mlir::Value vVec = val.vec;
                    mlir::Type et = llvm::cast<mlir::VectorType>(vVec.getType()).getElementType();
                    if (sig->systemValueType == 20 && outStencilRefPtr) { // SV_StencilRef
                        mlir::Value refVec8;
                        if (mlir::isa<mlir::IntegerType>(et)) {
                            refVec8 = (et == sh.i8Ty())
                                          ? vVec
                                          : mlir::LLVM::TruncOp::create(sb, sloc, sh.vecTy(sh.i8Ty()), vVec).getResult();
                        } else {
                            mlir::Value asI32 = mlir::LLVM::BitcastOp::create(sb, sloc, sh.vecTy(i32Ty), vVec);
                            refVec8 = mlir::LLVM::TruncOp::create(sb, sloc, sh.vecTy(sh.i8Ty()), asI32);
                        }
                        mlir::Value prev = sh.loadScalar(sh.vecTy(sh.i8Ty()), outStencilRefPtr);
                        sh.storeScalar(mlir::LLVM::SelectOp::create(sb, sloc, writeMask, refVec8, prev), outStencilRefPtr);
                        return true;
                    }
                    if (et != f32Ty) {
                        vVec = mlir::isa<mlir::IntegerType>(et)
                                   ? mlir::LLVM::BitcastOp::create(sb, sloc, sh.vecTy(f32Ty), vVec).getResult()
                                   : mlir::LLVM::FPExtOp::create(sb, sloc, sh.vecTy(f32Ty), vVec).getResult();
                    }
                    if (sig->systemValueType == 17 || sig->systemValueType == 18 || sig->systemValueType == 19) { // SV_Depth*
                        mlir::Value prev = sh.loadScalar(sh.vecTy(f32Ty), outDepthPtr);
                        sh.storeScalar(mlir::LLVM::SelectOp::create(sb, sloc, writeMask, vVec, prev), outDepthPtr);
                        return true;
                    }
                    if (sig->systemValueType == 16 && sig->semanticIndex < 8 && col < 4) { // SV_TargetN
                        mlir::Value slotPtr = rtOutPtrs[sig->semanticIndex][col];
                        if (slotPtr) {
                            mlir::Value prev = sh.loadScalar(sh.vecTy(f32Ty), slotPtr);
                            sh.storeScalar(mlir::LLVM::SelectOp::create(sb, sloc, writeMask, vVec, prev), slotPtr);
                        }
                    }
                    return true;
                };

                if (!packetizeRegionWithHooks(pb, ploc, combinedMod, psSrcFunc.getBody(), psLoaded, psEffectiveLayout,
                                              simdWidth, rootCtx, execMask, curRealMask, psHooks, outError)) {
                    return;
                }

                // Preload and clamp PS color outputs once; reuse across all MSAA samples when psPixelFreqMSAA is active.
                std::array<std::array<mlir::Value, 4>, 8> rtSrcColors{};
                for (uint32_t rt = 0; rt < desc.NumRenderTargets && rt < 8; ++rt) {
                    if (desc.RTVFormats[rt] == DXGI_FORMAT_UNKNOWN) continue;
                    const auto& rtBlend = desc.BlendState.RenderTarget[desc.BlendState.IndependentBlendEnable ? rt : 0];
                    if (rtBlend.RenderTargetWriteMask == 0) continue;
                    for (int c = 0; c < 4; ++c) {
                        rtSrcColors[rt][c] = rtOutPtrs[rt][c]
                                                 ? ph.loadScalar(ph.vecTy(f32Ty), rtOutPtrs[rt][c])
                                                 : ph.splatFlt(c == 3 ? 1.0 : 0.0);
                        if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) {
                            rtSrcColors[rt][c] = ph.clamp01(rtSrcColors[rt][c]);
                        }
                    }
                }

                auto emitOutputMerger = [&](mlir::OpBuilder& mb, mlir::Location mloc, mlir::Value omMask, uint32_t tileColorElemOffset) {
                    IRHelpers mh{mb, mloc, simdWidth};
                    mlir::Value anyWrite = mh.orReduce(omMask);
                    mlir::scf::IfOp::create(mb, mloc, anyWrite, [&](mlir::OpBuilder& ob, mlir::Location oloc) {
                        IRHelpers oh{ob, oloc, simdWidth};
                        mlir::Value blendFactors[4]{};
                        bool loadedBlendFactor = false;
                        auto ensureBlendFactor = [&]() {
                            if (loadedBlendFactor) return;
                            for (int c = 0; c < 4; ++c) {
                                mlir::Value bf = oh.loadScalar(
                                    f32Ty, oh.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, blend_factor) + c * 4)));
                                blendFactors[c] = llvmSplat(ob, oloc, llvm::cast<mlir::VectorType>(oh.vecTy(f32Ty)).getElementType(), bf, llvm::cast<mlir::VectorType>(oh.vecTy(f32Ty)).getNumElements());
                            }
                            loadedBlendFactor = true;
                        };

                        for (uint32_t rt = 0; rt < desc.NumRenderTargets && rt < 8; ++rt) {
                            if (desc.RTVFormats[rt] == DXGI_FORMAT_UNKNOWN) continue;
                            const auto& rtBlend = desc.BlendState.RenderTarget[desc.BlendState.IndependentBlendEnable ? rt : 0];
                            uint8_t writeMask = rtBlend.RenderTargetWriteMask;
                            if (writeMask == 0) continue;

                            mlir::Value src[4]{rtSrcColors[rt][0], rtSrcColors[rt][1], rtSrcColors[rt][2], rtSrcColors[rt][3]};

                            // Load current tile color only when blending is enabled; for opaque draws,
                            // partial quads load destination pixels lazily inside the masked store branch.
                            mlir::Value dst[4];
                            auto v4f = mlir::VectorType::get({4}, f32Ty);
                            if (rtBlend.BlendEnable) {
                                for (int c = 0; c < 4; ++c) {
                                    dst[c] = oh.splatFlt(0.0);
                                    for (uint32_t q = 0; q < numQuads; ++q) {
                                        mlir::Value pixIdx = mlir::LLVM::AddOp::create(
                                            ob, oloc, quadOffsets[q], oh.cInt(i32Ty, static_cast<int64_t>(tileColorElemOffset + c * kRasterTilePixels)));
                                        mlir::Value byteOff = mlir::LLVM::MulOp::create(
                                            ob, oloc, mlir::LLVM::ZExtOp::create(ob, oloc, oh.i64Ty(), pixIdx), oh.cInt(oh.i64Ty(), 4));
                                        mlir::Value q4 = oh.loadScalar(v4f, oh.gepBytesDyn(tileColorBases[rt], byteOff));
                                        for (uint32_t s = 0; s < 4; ++s) {
                                            mlir::Value sc = llvmExtract(ob, oloc, q4, static_cast<uint32_t>(static_cast<int64_t>(s)));
                                            dst[c] = llvmInsert(ob, oloc, dst[c], sc, static_cast<uint32_t>(static_cast<int64_t>(q * 4 + s)));
                                        }
                                    }
                                }
                            }

                            mlir::Value outColor[4]{src[0], src[1], src[2], src[3]};
                            if (rtBlend.BlendEnable) {
                                if (rtBlend.SrcBlend == D3D12_BLEND_BLEND_FACTOR || rtBlend.SrcBlend == D3D12_BLEND_INV_BLEND_FACTOR ||
                                    rtBlend.DestBlend == D3D12_BLEND_BLEND_FACTOR || rtBlend.DestBlend == D3D12_BLEND_INV_BLEND_FACTOR ||
                                    rtBlend.SrcBlendAlpha == D3D12_BLEND_BLEND_FACTOR || rtBlend.SrcBlendAlpha == D3D12_BLEND_INV_BLEND_FACTOR ||
                                    rtBlend.DestBlendAlpha == D3D12_BLEND_BLEND_FACTOR || rtBlend.DestBlendAlpha == D3D12_BLEND_INV_BLEND_FACTOR) {
                                    ensureBlendFactor();
                                }
                                for (int c = 0; c < 3; ++c) {
                                    mlir::Value sf = oh.evalBlendFactorRGB(rtBlend.SrcBlend, src, dst, blendFactors, c);
                                    mlir::Value df = oh.evalBlendFactorRGB(rtBlend.DestBlend, src, dst, blendFactors, c);
                                    mlir::Value sTerm = mlir::LLVM::FMulOp::create(ob, oloc, src[c], sf);
                                    mlir::Value dTerm = mlir::LLVM::FMulOp::create(ob, oloc, dst[c], df);
                                    outColor[c] = oh.evalBlendOp(rtBlend.BlendOp, sTerm, dTerm);
                                    if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) outColor[c] = oh.clamp01(outColor[c]);
                                }
                                mlir::Value saf = oh.evalBlendFactorRGB(rtBlend.SrcBlendAlpha, src, dst, blendFactors, 3);
                                mlir::Value daf = oh.evalBlendFactorRGB(rtBlend.DestBlendAlpha, src, dst, blendFactors, 3);
                                outColor[3] = oh.evalBlendOp(
                                    rtBlend.BlendOpAlpha,
                                    mlir::LLVM::FMulOp::create(ob, oloc, src[3], saf),
                                    mlir::LLVM::FMulOp::create(ob, oloc, dst[3], daf));
                                if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) outColor[3] = oh.clamp01(outColor[3]);
                            }

                            // Masked SOA quad write-back respecting RenderTargetWriteMask & omMask
                            for (uint32_t q = 0; q < numQuads; ++q) {
                                llvm::SmallVector<int32_t, 4> qIdx{static_cast<int32_t>(q * 4), static_cast<int32_t>(q * 4 + 1), static_cast<int32_t>(q * 4 + 2), static_cast<int32_t>(q * 4 + 3)};
                                mlir::Value qMask = mlir::LLVM::ShuffleVectorOp::create(ob, oloc, omMask, omMask, qIdx);
                                mlir::Value anyQ = mlir::LLVM::vector_reduce_or::create(ob, oloc, ob.getI1Type(), qMask);
                                mlir::scf::IfOp::create(ob, oloc, anyQ, [&](mlir::OpBuilder& qb, mlir::Location qloc) {
                                    IRHelpers qh{qb, qloc, simdWidth};
                                    mlir::Value allQ = rtBlend.BlendEnable
                                        ? mlir::Value{}
                                        : mlir::LLVM::vector_reduce_and::create(qb, qloc, qb.getI1Type(), qMask);
                                    auto writeQuadChannels = [&](mlir::OpBuilder& wb, mlir::Location wloc, bool loadDstForPartial) {
                                        IRHelpers wh{wb, wloc, simdWidth};
                                        for (int c = 0; c < 4; ++c) {
                                            if (!(writeMask & (1u << c))) continue;
                                            mlir::Value outQ4 = mlir::LLVM::ShuffleVectorOp::create(wb, wloc, outColor[c], outColor[c], qIdx);
                                            mlir::Value pixIdx = mlir::LLVM::AddOp::create(
                                                wb, wloc, quadOffsets[q], wh.cInt(i32Ty, static_cast<int64_t>(tileColorElemOffset + c * kRasterTilePixels)));
                                            mlir::Value byteOff = mlir::LLVM::MulOp::create(
                                                wb, wloc, mlir::LLVM::ZExtOp::create(wb, wloc, wh.i64Ty(), pixIdx), wh.cInt(wh.i64Ty(), 4));
                                            mlir::Value dstPtr = wh.gepBytesDyn(tileColorBases[rt], byteOff);
                                            mlir::Value storeQ4 = outQ4;
                                            if (rtBlend.BlendEnable) {
                                                mlir::Value dstQ4 = mlir::LLVM::ShuffleVectorOp::create(wb, wloc, dst[c], dst[c], qIdx);
                                                storeQ4 = mlir::LLVM::SelectOp::create(wb, wloc, qMask, outQ4, dstQ4);
                                            } else if (loadDstForPartial) {
                                                mlir::Value dstQ4 = wh.loadScalar(v4f, dstPtr);
                                                storeQ4 = mlir::LLVM::SelectOp::create(wb, wloc, qMask, outQ4, dstQ4);
                                            }
                                            mlir::LLVM::StoreOp::create(wb, wloc, storeQ4, dstPtr);
                                        }
                                    };
                                    if (rtBlend.BlendEnable) {
                                        writeQuadChannels(qb, qloc, false);
                                    } else {
                                        mlir::scf::IfOp::create(
                                            qb, qloc, allQ,
                                            [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                                                writeQuadChannels(tb, tloc, false);
                                                mlir::scf::YieldOp::create(tb, tloc);
                                            },
                                            [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                                                writeQuadChannels(eb, eloc, true);
                                                mlir::scf::YieldOp::create(eb, eloc);
                                            });
                                    }
                                    mlir::scf::YieldOp::create(qb, qloc);
                                });
                            }
                        }
                        mlir::scf::YieldOp::create(ob, oloc);
                    });
                };

                if (psPixelFreqMSAA) {
                    mlir::Value finalZ = !canEarlyZ ? ph.loadScalar(ph.vecTy(f32Ty), outDepthPtr) : mlir::Value();
                    mlir::Value finalStencilRef = (!canEarlyZ && psWritesStencilRef)
                                                      ? ph.loadScalar(ph.vecTy(ph.i8Ty()), outStencilRefPtr)
                                                      : mlir::Value();
                    for (uint32_t s = 0; s < msaaSampleCount; ++s) {
                        mlir::Value sMask = canEarlyZ
                            ? msaaEarlyDSMasks[s]
                            : runDepthStencil(pb, ploc,
                                              mlir::LLVM::AndOp::create(pb, ploc, msaaSampleMasks[s], curRealMask),
                                              finalZ, finalStencilRef, s * kRasterTilePixels);
                        emitOutputMerger(pb, ploc, sMask, s * 4u * kRasterTilePixels);
                    }
                } else {
                    // Late-Z if Early-Z was skipped
                    if (!canEarlyZ) {
                        mlir::Value finalZ = ph.loadScalar(ph.vecTy(f32Ty), outDepthPtr);
                        mlir::Value finalStencilRef = psWritesStencilRef
                                                          ? ph.loadScalar(ph.vecTy(ph.i8Ty()), outStencilRefPtr)
                                                          : mlir::Value();
                        curRealMask = runDepthStencil(pb, ploc, curRealMask, finalZ, finalStencilRef);
                    }
                    emitOutputMerger(pb, ploc, curRealMask, 0);
                }
            };

            if (canEarlyZ) {
                mlir::Value anySurvive = h.orReduce(realMaskAfterEarly);
                mlir::scf::IfOp::create(b, loc, anySurvive, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    emitPSBodyAndOM(nb, nloc, realMaskAfterEarly);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
                if (!outError.empty()) { batchEmitOk = false; }
            } else {
                emitPSBodyAndOM(b, loc, realMaskAfterEarly);
                if (!outError.empty()) { batchEmitOk = false; }
            }
        }
        mlir::scf::YieldOp::create(b, loc);
    });
        if (!batchEmitOk) return false;

        mlir::func::ReturnOp::create(b, loc);
    }


    // Remove dead dx.op.* declarations left over in combinedMod
    llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 16> deadDxOps;
    for (auto fnOp : combinedMod.getOps<mlir::LLVM::LLVMFuncOp>()) {
        if (fnOp.isDeclaration() && fnOp.getName().starts_with("dx.op.") && fnOp.use_empty()) {
            deadDxOps.push_back(fnOp);
        }
    }
    for (auto fnOp : deadDxOps) fnOp.erase();

    if (mlir::failed(mlir::verify(combinedMod))) {
        outError = "MLIR verification failed after Graphics Uber-PSO synthesis";
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
    outMeta.psPixelFreqMSAA = psPixelFreqMSAA ? 1 : 0;
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
        outPSO.psPixelFreqMSAA = meta.psPixelFreqMSAA != 0;
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
            std::unique_ptr<llvm::LLVMContext> vsLlvmCtx;
            std::unique_ptr<llvm::Module> vsLinkedMod;
            {
                mlir::MLIRContext vsMlirCtx(mlir::MLIRContext::Threading::DISABLED);
                mlir::OwningOpRef<mlir::ModuleOp> vsMod;
                if (!synthesizeVertexStageMLIR(
                        desc, rootLayout, simdWidth, vsSym, vsMlirCtx, vsMod, vsMeta,
                        outPSO.vsInputs, outPSO.vsOutputs, vsOutputRecs, outError)) {
                    return false;
                }
                vsLlvmCtx = std::make_unique<llvm::LLVMContext>();
                vsLinkedMod = translateMLIRToLinkedLLVMIR(*vsMod, *vsLlvmCtx, outError, /*linkRuntime=*/false);
                if (!vsLinkedMod) return false;
            }
            if (!jitCompileVertexStageTier1AndScheduleTier2(
                    std::move(vsLinkedMod), std::move(vsLlvmCtx), vsHashLo, vsHashHi, vsSym,
                    vsMeta, outPSO.vsInputs, outPSO.vsOutputs, vsOutputRecs, outPSO, outError)) {
                return false;
            }
        }

        CachedPSOMeta meta{};
        std::unique_ptr<llvm::LLVMContext> psLlvmCtx;
        std::unique_ptr<llvm::Module> psLinkedMod;
        {
            mlir::MLIRContext psMlirCtx(mlir::MLIRContext::Threading::DISABLED);
            mlir::OwningOpRef<mlir::ModuleOp> psMod;
            if (!synthesizeGraphicsPSOMLIR(
                    desc, rootLayout, simdWidth, vsSym, psSym, psMlirCtx, psMod, meta,
                    outPSO.vsInputs, outPSO.vsOutputs, outPSO.psInputs, outPSO.psOutputs, outError,
                    meshStages, /*skipVS=*/true, &vsMeta, &vsOutputRecs)) {
                return false;
            }
            meta.separateVSObject = 1;
            meta.vsHashLo = vsHashLo;
            meta.vsHashHi = vsHashHi;

            psLlvmCtx = std::make_unique<llvm::LLVMContext>();
            psLinkedMod = translateMLIRToLinkedLLVMIR(*psMod, *psLlvmCtx, outError, /*linkRuntime=*/false);
            if (!psLinkedMod) return false;
        }

        populatePSOFromMeta(meta);
        return jitCompilePixelStageTier1AndScheduleTier2(
            std::move(psLinkedMod), std::move(psLlvmCtx), vsHashLo, vsHashHi, psSym, meta, outPSO, outError);
    }

    std::string vsSym = "nacelle_vs_" + hex;
    if (tryLoadGraphicsPSOFromCache(lo, hi, vsSym, psSym, outPSO)) {
        return true;
    }

    CachedPSOMeta meta;
    std::unique_ptr<llvm::LLVMContext> outLlvmCtx;
    std::unique_ptr<llvm::Module> linkedMod;
    {
        mlir::MLIRContext mlirCtx(mlir::MLIRContext::Threading::DISABLED);
        mlir::OwningOpRef<mlir::ModuleOp> combinedMod;
        if (!synthesizeGraphicsPSOMLIR(
                desc, rootLayout, simdWidth, vsSym, psSym, mlirCtx, combinedMod, meta,
                outPSO.vsInputs, outPSO.vsOutputs, outPSO.psInputs, outPSO.psOutputs, outError, meshStages)) {
            return false;
        }

        outLlvmCtx = std::make_unique<llvm::LLVMContext>();
        linkedMod = translateMLIRToLinkedLLVMIR(*combinedMod, *outLlvmCtx, outError, /*linkRuntime=*/false);
        if (!linkedMod) return false;
    }

    populatePSOFromMeta(meta);
    return jitCompileGraphicsTier1AndScheduleTier2(
        std::move(linkedMod), std::move(outLlvmCtx), vsSym, psSym, meta, outPSO, outError);
}

} // namespace nacelle
