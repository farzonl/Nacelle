#include "pso_synthesizer.h"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstring>
#include <unordered_map>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
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
    if (auto c = sv.vec.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto splat = llvm::dyn_cast<mlir::SplatElementsAttr>(c.getValue())) {
            if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(splat.getSplatValue<mlir::Attribute>())) {
                return static_cast<uint32_t>(ia.getValue().getZExtValue());
            }
        }
        if (auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(c.getValue()); dense && dense.isSplat()) {
            return static_cast<uint32_t>(dense.getSplatValue<llvm::APInt>().getZExtValue());
        }
    }
    if (auto bc = sv.vec.getDefiningOp<mlir::vector::BroadcastOp>()) {
        if (auto c = bc.getSource().getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
                return static_cast<uint32_t>(ia.getValue().getZExtValue());
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
    case DXGI_FORMAT_R16_FLOAT:
    case DXGI_FORMAT_R16_UNORM:
        return 2;
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R8_UINT:
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
        return mlir::arith::ConstantIntOp::create(b, loc, ty, v);
    }
    mlir::Value cFlt(double v) const {
        return mlir::arith::ConstantFloatOp::create(b, loc, f32Ty(), llvm::APFloat(static_cast<float>(v)));
    }
    mlir::Value splatInt(mlir::Type ty, int64_t v) const {
        return mlir::vector::BroadcastOp::create(b, loc, vecTy(ty), cInt(ty, v));
    }
    mlir::Value splatFlt(double v) const {
        return mlir::vector::BroadcastOp::create(b, loc, vecTy(f32Ty()), cFlt(v));
    }
    SimdVal splatScalar(mlir::Value scalar) const {
        SimdVal sv;
        sv.kind = SimdVal::Kind::Vector;
        sv.origType = scalar.getType();
        sv.vec = mlir::vector::BroadcastOp::create(b, loc, vecTy(scalar.getType()), scalar);
        return sv;
    }
    mlir::Value allTrueMask() const { return splatInt(i1Ty(), 1); }

    mlir::Value laneIota() const {
        mlir::Value v = mlir::LLVM::PoisonOp::create(b, loc, vecTy(i32Ty()));
        for (uint32_t l = 0; l < W; ++l) {
            v = mlir::vector::InsertOp::create(b, loc, cInt(i32Ty(), l), v, static_cast<int64_t>(l));
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
        return mlir::vector::ReductionOp::create(b, loc, mlir::vector::CombiningKind::OR, maskVec);
    }

    mlir::Value expandMaskQuads(mlir::Value mask) const {
        llvm::SmallVector<int64_t, 16> s0, s1, s2, s3;
        for (uint32_t q = 0; q < W; q += 4) {
            for (uint32_t k = 0; k < 4; ++k) {
                s0.push_back(q);
                s1.push_back(q + 1);
                s2.push_back(q + 2);
                s3.push_back(q + 3);
            }
        }
        mlir::Value m0 = mlir::vector::ShuffleOp::create(b, loc, mask, mask, s0);
        mlir::Value m1 = mlir::vector::ShuffleOp::create(b, loc, mask, mask, s1);
        mlir::Value m2 = mlir::vector::ShuffleOp::create(b, loc, mask, mask, s2);
        mlir::Value m3 = mlir::vector::ShuffleOp::create(b, loc, mask, mask, s3);
        mlir::Value o01 = mlir::arith::OrIOp::create(b, loc, m0, m1);
        mlir::Value o23 = mlir::arith::OrIOp::create(b, loc, m2, m3);
        return mlir::arith::OrIOp::create(b, loc, o01, o23);
    }

    mlir::Value clamp01(mlir::Value vVec) const {
        mlir::Value zero = splatFlt(0.0);
        mlir::Value one = splatFlt(1.0);
        return mlir::arith::MinimumFOp::create(
            b, loc, mlir::arith::MaximumFOp::create(b, loc, vVec, zero), one);
    }

    mlir::Value evalDepthCmp(D3D12_COMPARISON_FUNC func, mlir::Value newZ, mlir::Value oldZ) const {
        switch (func) {
        case D3D12_COMPARISON_FUNC_NEVER:
            return splatInt(i1Ty(), 0);
        case D3D12_COMPARISON_FUNC_LESS:
            return mlir::arith::CmpFOp::create(b, loc, mlir::arith::CmpFPredicate::OLT, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_EQUAL:
            return mlir::arith::CmpFOp::create(b, loc, mlir::arith::CmpFPredicate::OEQ, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_LESS_EQUAL:
            return mlir::arith::CmpFOp::create(b, loc, mlir::arith::CmpFPredicate::OLE, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_GREATER:
            return mlir::arith::CmpFOp::create(b, loc, mlir::arith::CmpFPredicate::OGT, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_NOT_EQUAL:
            return mlir::arith::CmpFOp::create(b, loc, mlir::arith::CmpFPredicate::ONE, newZ, oldZ);
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
            return mlir::arith::CmpFOp::create(b, loc, mlir::arith::CmpFPredicate::OGE, newZ, oldZ);
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
            return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ult, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_EQUAL:
            return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::eq, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_LESS_EQUAL:
            return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ule, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_GREATER:
            return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ugt, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_NOT_EQUAL:
            return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ne, refVec, bufVec);
        case D3D12_COMPARISON_FUNC_GREATER_EQUAL:
            return mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::uge, refVec, bufVec);
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
            mlir::Value sat = mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::eq, oldVal, splatInt(i8Ty(), 255));
            mlir::Value inc = mlir::arith::AddIOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
            return mlir::arith::SelectOp::create(b, loc, sat, oldVal, inc);
        }
        case D3D12_STENCIL_OP_DECR_SAT: {
            mlir::Value z = mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::eq, oldVal, splatInt(i8Ty(), 0));
            mlir::Value dec = mlir::arith::SubIOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
            return mlir::arith::SelectOp::create(b, loc, z, oldVal, dec);
        }
        case D3D12_STENCIL_OP_INVERT:
            return mlir::arith::XOrIOp::create(b, loc, oldVal, splatInt(i8Ty(), -1));
        case D3D12_STENCIL_OP_INCR:
            return mlir::arith::AddIOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
        case D3D12_STENCIL_OP_DECR:
            return mlir::arith::SubIOp::create(b, loc, oldVal, splatInt(i8Ty(), 1));
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
        case D3D12_BLEND_INV_SRC_COLOR: return mlir::arith::SubFOp::create(b, loc, one, src[comp]);
        case D3D12_BLEND_SRC_ALPHA: return src[3];
        case D3D12_BLEND_INV_SRC_ALPHA: return mlir::arith::SubFOp::create(b, loc, one, src[3]);
        case D3D12_BLEND_DEST_ALPHA: return dst[3];
        case D3D12_BLEND_INV_DEST_ALPHA: return mlir::arith::SubFOp::create(b, loc, one, dst[3]);
        case D3D12_BLEND_DEST_COLOR: return dst[comp];
        case D3D12_BLEND_INV_DEST_COLOR: return mlir::arith::SubFOp::create(b, loc, one, dst[comp]);
        case D3D12_BLEND_SRC_ALPHA_SAT: {
            if (comp == 3) return one;
            mlir::Value invDstA = mlir::arith::SubFOp::create(b, loc, one, dst[3]);
            return mlir::arith::MinimumFOp::create(b, loc, src[3], invDstA);
        }
        case D3D12_BLEND_BLEND_FACTOR: return blendFactor[comp];
        case D3D12_BLEND_INV_BLEND_FACTOR: return mlir::arith::SubFOp::create(b, loc, one, blendFactor[comp]);
        default: return one;
        }
    }

    mlir::Value evalBlendOp(D3D12_BLEND_OP op, mlir::Value sTerm, mlir::Value dTerm) const {
        switch (op) {
        case D3D12_BLEND_OP_ADD: return mlir::arith::AddFOp::create(b, loc, sTerm, dTerm);
        case D3D12_BLEND_OP_SUBTRACT: return mlir::arith::SubFOp::create(b, loc, sTerm, dTerm);
        case D3D12_BLEND_OP_REV_SUBTRACT: return mlir::arith::SubFOp::create(b, loc, dTerm, sTerm);
        case D3D12_BLEND_OP_MIN: return mlir::arith::MinimumFOp::create(b, loc, sTerm, dTerm);
        case D3D12_BLEND_OP_MAX: return mlir::arith::MaximumFOp::create(b, loc, sTerm, dTerm);
        default: return mlir::arith::AddFOp::create(b, loc, sTerm, dTerm);
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

mlir::Value decodeVertexComponent(
    IRHelpers& h,
    const D3D12_INPUT_ELEMENT_DESC& el,
    mlir::Value activeMask,
    mlir::Value vbBasePtr,
    mlir::Value vbStrideVec,
    mlir::Value vbSizeVec,
    mlir::Value idxVec,
    uint32_t byteOffsetInElem,
    uint32_t col,
    bool wantInt) {
    mlir::OpBuilder& b = h.b;
    mlir::Location loc = h.loc;
    uint32_t W = h.W;
    uint32_t fmtBytes = formatByteSizeForVertex(el.Format);
    float defaultF = (col == 3) ? 1.0f : 0.0f;
    mlir::Value outVec = wantInt ? h.splatInt(h.i32Ty(), 0) : h.splatFlt(defaultF);

    auto activePerLaneLoad = [&](auto loadScalarFn) -> mlir::Value {
        mlir::Value res = outVec;
        for (uint32_t l = 0; l < W; ++l) {
            mlir::Value actL = mlir::vector::ExtractOp::create(b, loc, activeMask, static_cast<int64_t>(l));
            mlir::Value idxL = mlir::vector::ExtractOp::create(b, loc, idxVec, static_cast<int64_t>(l));
            mlir::Value strL = mlir::vector::ExtractOp::create(b, loc, vbStrideVec, static_cast<int64_t>(l));
            mlir::Value szL = mlir::vector::ExtractOp::create(b, loc, vbSizeVec, static_cast<int64_t>(l));

            mlir::Value idx64 = mlir::arith::ExtUIOp::create(b, loc, h.i64Ty(), idxL);
            mlir::Value str64 = mlir::arith::ExtUIOp::create(b, loc, h.i64Ty(), strL);
            mlir::Value off64 = mlir::arith::AddIOp::create(
                b, loc,
                mlir::arith::MulIOp::create(b, loc, idx64, str64),
                h.cInt(h.i64Ty(), byteOffsetInElem));
            mlir::Value end64 = mlir::arith::AddIOp::create(b, loc, off64, h.cInt(h.i64Ty(), fmtBytes));
            mlir::Value sz64 = mlir::arith::ExtUIOp::create(b, loc, h.i64Ty(), szL);
            mlir::Value inBounds = mlir::arith::CmpIOp::create(b, loc, mlir::arith::CmpIPredicate::ule, end64, sz64);
            mlir::Value ptrNotNull = mlir::LLVM::ICmpOp::create(
                b, loc, mlir::LLVM::ICmpPredicate::ne, vbBasePtr, mlir::LLVM::ZeroOp::create(b, loc, h.ptrTy()));
            mlir::Value ok = mlir::arith::AndIOp::create(b, loc, mlir::arith::AndIOp::create(b, loc, actL, ptrNotNull), inBounds);

            auto ifOp = mlir::scf::IfOp::create(
                b, loc, ok,
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value addr = mlir::LLVM::GEPOp::create(nb, nloc, h.ptrTy(), h.i8Ty(), vbBasePtr, mlir::ValueRange{off64});
                    mlir::Value val = loadScalarFn(nb, nloc, addr);
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{val});
                },
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value defV = wantInt
                                           ? mlir::arith::ConstantIntOp::create(nb, nloc, h.i32Ty(), 0).getResult()
                                           : mlir::arith::ConstantFloatOp::create(nb, nloc, h.f32Ty(), llvm::APFloat(defaultF)).getResult();
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{defV});
                });
            res = mlir::vector::InsertOp::create(b, loc, ifOp.getResult(0), res, static_cast<int64_t>(l));
        }
        return res;
    };

    switch (el.Format) {
    case DXGI_FORMAT_R32G32B32A32_FLOAT:
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32_FLOAT:
    case DXGI_FORMAT_R32_FLOAT: {
        uint32_t maxComps = (el.Format == DXGI_FORMAT_R32_FLOAT) ? 1 :
                            (el.Format == DXGI_FORMAT_R32G32_FLOAT) ? 2 :
                            (el.Format == DXGI_FORMAT_R32G32B32_FLOAT) ? 3 : 4;
        if (col >= maxComps) return outVec;
        return activePerLaneLoad([&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value addr) {
            mlir::Value p = mlir::LLVM::GEPOp::create(
                nb, nloc, h.ptrTy(), h.i8Ty(), addr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(col * 4)});
            mlir::Value v = mlir::LLVM::LoadOp::create(nb, nloc, wantInt ? h.i32Ty() : h.f32Ty(), p);
            return v;
        });
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
        if (col >= maxComps) return outVec;
        return activePerLaneLoad([&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value addr) {
            mlir::Value p = mlir::LLVM::GEPOp::create(
                nb, nloc, h.ptrTy(), h.i8Ty(), addr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(col * 4)});
            mlir::Value v = mlir::LLVM::LoadOp::create(nb, nloc, h.i32Ty(), p);
            if (!wantInt) v = mlir::arith::BitcastOp::create(nb, nloc, h.f32Ty(), v);
            return v;
        });
    }
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R16G16_FLOAT:
    case DXGI_FORMAT_R16_FLOAT: {
        uint32_t maxComps = fmtBytes / 2;
        if (col >= maxComps) return outVec;
        return activePerLaneLoad([&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value addr) {
            mlir::Value p = mlir::LLVM::GEPOp::create(
                nb, nloc, h.ptrTy(), h.i8Ty(), addr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(col * 2)});
            mlir::Value hv = mlir::LLVM::LoadOp::create(nb, nloc, h.f16Ty(), p);
            return mlir::arith::ExtFOp::create(nb, nloc, h.f32Ty(), hv).getResult();
        });
    }
    case DXGI_FORMAT_R16G16B16A16_UNORM:
    case DXGI_FORMAT_R16G16_UNORM:
    case DXGI_FORMAT_R16_UNORM: {
        uint32_t maxComps = fmtBytes / 2;
        if (col >= maxComps) return outVec;
        return activePerLaneLoad([&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value addr) {
            mlir::Value p = mlir::LLVM::GEPOp::create(
                nb, nloc, h.ptrTy(), h.i8Ty(), addr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(col * 2)});
            mlir::Value u16 = mlir::LLVM::LoadOp::create(nb, nloc, h.i16Ty(), p);
            mlir::Value fv = mlir::arith::UIToFPOp::create(nb, nloc, h.f32Ty(), u16);
            mlir::Value scale = mlir::arith::ConstantFloatOp::create(nb, nloc, h.f32Ty(), llvm::APFloat(1.0f / 65535.0f));
            return mlir::arith::MulFOp::create(nb, nloc, fv, scale).getResult();
        });
    }
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R8_UNORM: {
        uint32_t maxComps = fmtBytes;
        if (col >= maxComps) return outVec;
        return activePerLaneLoad([&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value addr) {
            mlir::Value p = mlir::LLVM::GEPOp::create(
                nb, nloc, h.ptrTy(), h.i8Ty(), addr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(col)});
            mlir::Value u8 = mlir::LLVM::LoadOp::create(nb, nloc, h.i8Ty(), p);
            mlir::Value fv = mlir::arith::UIToFPOp::create(nb, nloc, h.f32Ty(), u8);
            mlir::Value scale = mlir::arith::ConstantFloatOp::create(nb, nloc, h.f32Ty(), llvm::APFloat(1.0f / 255.0f));
            return mlir::arith::MulFOp::create(nb, nloc, fv, scale).getResult();
        });
    }
    case DXGI_FORMAT_R8G8B8A8_UINT:
    case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8_UINT: {
        uint32_t maxComps = fmtBytes;
        if (col >= maxComps) return outVec;
        return activePerLaneLoad([&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value addr) {
            mlir::Value p = mlir::LLVM::GEPOp::create(
                nb, nloc, h.ptrTy(), h.i8Ty(), addr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(col)});
            mlir::Value u8 = mlir::LLVM::LoadOp::create(nb, nloc, h.i8Ty(), p);
            mlir::Value u32 = mlir::arith::ExtUIOp::create(nb, nloc, h.i32Ty(), u8);
            return wantInt ? u32 : mlir::arith::BitcastOp::create(nb, nloc, h.f32Ty(), u32).getResult();
        });
    }
    default:
        return outVec;
    }
}

struct LinkedVSSigSlot {
    int32_t psTargetRow = -1;
    uint32_t psStartCol = 0;
};

} // namespace

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

    bool hasTess = (desc.HS.pShaderBytecode && desc.HS.BytecodeLength > 0 &&
                    desc.DS.pShaderBytecode && desc.DS.BytecodeLength > 0);
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

    bool hasGS = (desc.GS.pShaderBytecode && desc.GS.BytecodeLength > 0);
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

    cloneGlobalsIntoModule(*vsMlirMod, combinedMod);
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

    outVSInputs = convertSigElements(vsLoaded.inputSigs);
    outVSOutputs = convertSigElements(vsLoaded.outputSigs);
    bool psWritesDepth = false;
    bool psUsesDiscard = false;
    bool psHasSideEffects = false;
    if (hasPS) {
        outPSInputs = convertSigElements(psLoaded.inputSigs);
        outPSOutputs = convertSigElements(psLoaded.outputSigs);
        for (const auto& po : psLoaded.outputSigs) {
            if (po.systemValueType >= 17 && po.systemValueType <= 19) psWritesDepth = true;
        }
    } else {
        outPSInputs.clear();
        outPSOutputs.clear();
    }

    // Link VS output signatures -> PS input signatures by (semanticName, semanticIndex)
    int32_t posOutputReg = 0;
    uint32_t numVSOutputRegs = 0;
    for (const auto& vo : vsLoaded.outputSigs) {
        numVSOutputRegs = std::max(numVSOutputRegs, vo.startRow + vo.rows);
        if (vo.systemValueType == 3) posOutputReg = static_cast<int32_t>(vo.startRow);
    }

    uint32_t numPSInputRegs = 0;
    uint32_t flatInterpMask = 0;
    if (hasPS) {
        for (const auto& pi : psLoaded.inputSigs) {
            if (pi.systemValueType == 0) {
                numPSInputRegs = std::max(numPSInputRegs, pi.startRow + pi.rows);
                for (uint32_t r = 0; r < pi.rows; ++r) {
                    uint32_t reg = pi.startRow + r;
                    if (reg < 16) {
                        if (pi.interpMode == 1) flatInterpMask |= (1u << reg);
                        else if (pi.interpMode == 4) flatInterpMask |= (1u << (16u + reg));
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

    std::unordered_map<uint32_t, LinkedVSSigSlot> vsToPsMap;
    if (hasTess) {
        for (const auto& vo : vsLoaded.outputSigs) {
            std::string vn = toLowerAscii(vo.semanticName);
            for (const auto& hi : hsLoaded.inputSigs) {
                if (toLowerAscii(hi.semanticName) == vn && hi.semanticIndex == vo.semanticIndex) {
                    vsToPsMap[vo.sigId] = {static_cast<int32_t>(hi.startRow), hi.startCol};
                    break;
                }
            }
        }
    } else if (hasGS) {
        for (const auto& vo : vsLoaded.outputSigs) {
            if (vo.systemValueType == 3) continue; // SV_Position goes to outPosXYZW
            std::string vn = toLowerAscii(vo.semanticName);
            for (const auto& gi : gsLoaded.inputSigs) {
                if (gi.systemValueType != 0) continue;
                if (toLowerAscii(gi.semanticName) == vn && gi.semanticIndex == vo.semanticIndex) {
                    vsToPsMap[vo.sigId] = {static_cast<int32_t>(gi.startRow), gi.startCol};
                    break;
                }
            }
        }
    } else if (hasPS) {
        for (const auto& vo : vsLoaded.outputSigs) {
            if (vo.systemValueType == 3) continue; // SV_Position
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

    // Allocate scalar attribute slots for SV_ClipDistance (systemValueType == 6) and
    // SV_CullDistance (systemValueType == 7) after the PS input attribute registers.
    const auto& preRastOutputs = hasGS ? gsLoaded.outputSigs : (hasTess ? dsLoaded.outputSigs : vsLoaded.outputSigs);
    uint32_t nextExtraAttr = numPSInputRegs * 4;
    uint32_t clipDistMask = 0;
    uint32_t cullDistMask = 0;
    uint32_t clipAttrOffset = nextExtraAttr;
    uint32_t clipScalarCount = 0;
    uint32_t cullAttrOffset = 0;
    uint32_t cullScalarCount = 0;
    std::unordered_map<uint32_t, uint32_t> clipSigBaseIdx;
    std::unordered_map<uint32_t, uint32_t> cullSigBaseIdx;
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
    bool hasViewportArrayIndex = false;
    uint32_t vpIndexAttrOffset = 0;
    bool hasPerPrimShadingRate = false;
    uint32_t shadingRateAttrOffset = 0;
    for (const auto& vo : preRastOutputs) {
        if (vo.systemValueType == 5 && !hasViewportArrayIndex) {
            hasViewportArrayIndex = true;
            vpIndexAttrOffset = nextExtraAttr++;
        } else if (vo.systemValueType == 29 && !hasPerPrimShadingRate) {
            hasPerPrimShadingRate = true;
            shadingRateAttrOffset = nextExtraAttr++;
        }
    }
    uint32_t totalVSScalarAttrs = hasTess ? numHSInputScalars : (hasGS ? (numGSInputRegs * 4) : nextExtraAttr);
    uint32_t totalPostDSVertexAttrs = hasGS ? (numGSInputRegs * 4) : nextExtraAttr;
    uint32_t totalPostGSVertexAttrs = nextExtraAttr;

    // Resolve InputLayout offsets per slot
    std::vector<uint32_t> elementByteOffsets(desc.InputLayout.NumElements, 0);
    uint32_t slotRunningOffset[16]{};
    for (UINT i = 0; i < desc.InputLayout.NumElements && desc.InputLayout.pInputElementDescs; ++i) {
        const auto& el = desc.InputLayout.pInputElementDescs[i];
        UINT slot = std::min<UINT>(el.InputSlot, 15);
        uint32_t off = (el.AlignedByteOffset == D3D12_APPEND_ALIGNED_ELEMENT)
                           ? slotRunningOffset[slot]
                           : el.AlignedByteOffset;
        elementByteOffsets[i] = off;
        slotRunningOffset[slot] = off + formatByteSizeForVertex(el.Format);
    }

    // --- 1. Emit @<vsSym>(ptr %rootCtx, ptr %vertexIds, ptr %instanceIds, i32 %activeCount, ptr %outPosXYZW, ptr %outAttrs) ---
    {
        auto vsSrcFunc = vsMlirMod->lookupSymbol<mlir::func::FuncOp>(vsLoaded.entryName);
        if (!vsSrcFunc || vsSrcFunc.getBody().empty()) {
            outError = "Vertex shader entry function missing after MLIR SCF lift";
            return false;
        }

        mlir::OpBuilder b(combinedMod.getBody(), combinedMod.getBody()->end());
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

        mlir::Value acVec = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(i32Ty), activeCount);
        mlir::Value activeMask = mlir::arith::CmpIOp::create(
            b, loc, mlir::arith::CmpIPredicate::ult, h.laneIota(), acVec);

        mlir::Value vidVec = h.splatInt(i32Ty, 0);
        mlir::Value iidVec = h.splatInt(i32Ty, 0);
        // Clean helper without i8TyOrByte typo:
        vidVec = h.splatInt(i32Ty, 0);
        iidVec = h.splatInt(i32Ty, 0);
        for (uint32_t l = 0; l < simdWidth; ++l) {
            mlir::Value mL = mlir::vector::ExtractOp::create(b, loc, activeMask, static_cast<int64_t>(l));
            auto rVid = mlir::scf::IfOp::create(
                b, loc, mL,
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value ep = mlir::LLVM::GEPOp::create(
                        nb, nloc, ptrTy, i32Ty, vertexIdsPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(l)});
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, i32Ty, ep)});
                },
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(nb, nloc, i32Ty, 0)});
                });
            auto rIid = mlir::scf::IfOp::create(
                b, loc, mL,
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value ep = mlir::LLVM::GEPOp::create(
                        nb, nloc, ptrTy, i32Ty, instanceIdsPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(l)});
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, i32Ty, ep)});
                },
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(nb, nloc, i32Ty, 0)});
                });
            vidVec = mlir::vector::InsertOp::create(b, loc, rVid.getResult(0), vidVec, static_cast<int64_t>(l));
            iidVec = mlir::vector::InsertOp::create(b, loc, rIid.getResult(0), iidVec, static_cast<int64_t>(l));
        }

        // Preload vertex buffer slots referenced by InputLayout
        mlir::Value vbBasePtrs[16]{};
        mlir::Value vbStrideVecs[16]{};
        mlir::Value vbSizeVecs[16]{};
        for (UINT i = 0; i < desc.InputLayout.NumElements && desc.InputLayout.pInputElementDescs; ++i) {
            UINT slot = std::min<UINT>(desc.InputLayout.pInputElementDescs[i].InputSlot, 15);
            if (!vbBasePtrs[slot]) {
                mlir::Value addrInt = h.loadScalar(
                    h.i64Ty(), h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_buffers) + slot * 8)));
                vbBasePtrs[slot] = mlir::LLVM::IntToPtrOp::create(b, loc, ptrTy, addrInt);
                mlir::Value str32 = h.loadScalar(
                    i32Ty, h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_strides) + slot * 4)));
                mlir::Value sz32 = h.loadScalar(
                    i32Ty, h.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, vertex_sizes) + slot * 4)));
                vbStrideVecs[slot] = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(i32Ty), str32);
                vbSizeVecs[slot] = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(i32Ty), sz32);
            }
        }

        StageHooks vsHooks;
        vsHooks.isPixelShader = false;
        vsHooks.loadInput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                const SimdVal&, uint32_t col, mlir::Type scalarTy, SimdVal& out) -> bool {
            const DxilSigRecord* sig = vsLoaded.findInputSig(sigId);
            if (!sig) return false;
            IRHelpers lh{lb, lloc, simdWidth};
            bool wantInt = mlir::isa<mlir::IntegerType>(scalarTy);

            if (sig->systemValueType == 1) { // SV_VertexID
                mlir::Value v = wantInt ? vidVec : mlir::arith::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), vidVec).getResult();
                out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                return true;
            }
            if (sig->systemValueType == 2) { // SV_InstanceID
                mlir::Value v = wantInt ? iidVec : mlir::arith::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), iidVec).getResult();
                out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                return true;
            }

            std::string sigNameLower = toLowerAscii(sig->semanticName);
            for (UINT i = 0; i < desc.InputLayout.NumElements && desc.InputLayout.pInputElementDescs; ++i) {
                const auto& el = desc.InputLayout.pInputElementDescs[i];
                if (el.SemanticName && toLowerAscii(el.SemanticName) == sigNameLower &&
                    el.SemanticIndex == sig->semanticIndex) {
                    UINT slot = std::min<UINT>(el.InputSlot, 15);
                    mlir::Value idxVec = vidVec;
                    if (el.InputSlotClass == D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA) {
                        if (el.InstanceDataStepRate == 0) {
                            idxVec = lh.splatInt(i32Ty, 0);
                        } else if (el.InstanceDataStepRate == 1) {
                            idxVec = iidVec;
                        } else {
                            idxVec = mlir::arith::DivUIOp::create(
                                lb, lloc, iidVec, lh.splatInt(i32Ty, el.InstanceDataStepRate));
                        }
                    }
                    mlir::Value decoded = decodeVertexComponent(
                        lh, el, activeMask, vbBasePtrs[slot], vbStrideVecs[slot], vbSizeVecs[slot],
                        idxVec, elementByteOffsets[i], col, wantInt);
                    if (wantInt && scalarTy != i32Ty) {
                        decoded = mlir::arith::TruncIOp::create(lb, lloc, lh.vecTy(scalarTy), decoded);
                    } else if (!wantInt && scalarTy != lh.f32Ty()) {
                        decoded = mlir::arith::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), decoded);
                    }
                    out = {SimdVal::Kind::Vector, scalarTy, decoded, {}, {}};
                    return true;
                }
            }
            return false;
        };

        vsHooks.storeOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                                  const SimdVal& row, uint32_t col, const SimdVal& val, mlir::Value) -> bool {
            const DxilSigRecord* sig = vsLoaded.findOutputSig(sigId);
            if (!sig || val.kind != SimdVal::Kind::Vector) return true;
            IRHelpers sh{sb, sloc, simdWidth};
            mlir::Value vVec = val.vec;
            mlir::Type elemTy = llvm::cast<mlir::VectorType>(vVec.getType()).getElementType();
            if (elemTy != sh.f32Ty()) {
                if (mlir::isa<mlir::IntegerType>(elemTy)) {
                    if (elemTy.getIntOrFloatBitWidth() < 32) {
                        vVec = mlir::arith::ExtUIOp::create(sb, sloc, sh.vecTy(i32Ty), vVec);
                    }
                    vVec = mlir::arith::BitcastOp::create(sb, sloc, sh.vecTy(sh.f32Ty()), vVec);
                } else {
                    vVec = mlir::arith::ExtFOp::create(sb, sloc, sh.vecTy(sh.f32Ty()), vVec);
                }
            }

            auto storeAttrAtSlot = [&](uint32_t baseSlot) {
                auto rowConst = tryGetUniformConstUInt(row);
                if (rowConst.has_value()) {
                    uint32_t scalarSlot = baseSlot + (*rowConst) * 4 + col;
                    mlir::Value dst = sh.gepBytes(outAttrs, static_cast<int64_t>(scalarSlot * simdWidth * 4));
                    mlir::LLVM::StoreOp::create(sb, sloc, vVec, dst, /*alignment=*/4);
                } else {
                    for (uint32_t l = 0; l < simdWidth; ++l) {
                        mlir::Value rL = mlir::vector::ExtractOp::create(sb, sloc, row.vec, static_cast<int64_t>(l));
                        mlir::Value r32 = (rL.getType() == i32Ty)
                                              ? rL
                                              : mlir::arith::ExtUIOp::create(sb, sloc, i32Ty, rL).getResult();
                        mlir::Value slot32 = mlir::arith::AddIOp::create(
                            sb, sloc, sh.cInt(i32Ty, static_cast<int64_t>(baseSlot + col)),
                            mlir::arith::MulIOp::create(sb, sloc, r32, sh.cInt(i32Ty, 4)));
                        mlir::Value elemIdx = mlir::arith::AddIOp::create(
                            sb, sloc, mlir::arith::MulIOp::create(sb, sloc, slot32, sh.cInt(i32Ty, simdWidth)),
                            sh.cInt(i32Ty, static_cast<int64_t>(l)));
                        mlir::Value byteOff = mlir::arith::MulIOp::create(
                            sb, sloc, mlir::arith::ExtUIOp::create(sb, sloc, sh.i64Ty(), elemIdx), sh.cInt(sh.i64Ty(), 4));
                        mlir::Value vL = mlir::vector::ExtractOp::create(sb, sloc, vVec, static_cast<int64_t>(l));
                        mlir::LLVM::StoreOp::create(sb, sloc, vL, sh.gepBytesDyn(outAttrs, byteOff));
                    }
                }
            };

            if (sig->systemValueType == 3) { // SV_Position -> outPosXYZW[col * W + lane]
                if (col < 4) {
                    mlir::Value dst = sh.gepBytes(outPosXYZW, static_cast<int64_t>(col * simdWidth * 4));
                    mlir::LLVM::StoreOp::create(sb, sloc, vVec, dst, /*alignment=*/4);
                }
                if (!hasTess) return true;
            }

            if (!hasGS && !hasTess && sig->systemValueType == 6) { // SV_ClipDistance
                auto cit = clipSigBaseIdx.find(sigId);
                if (cit != clipSigBaseIdx.end()) {
                    uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (clipDistMask & (1u << idx))) {
                        mlir::Value dst = sh.gepBytes(outAttrs, static_cast<int64_t>((clipAttrOffset + idx) * simdWidth * 4));
                        mlir::LLVM::StoreOp::create(sb, sloc, vVec, dst, /*alignment=*/4);
                    }
                }
                return true;
            }
            if (!hasGS && !hasTess && sig->systemValueType == 7) { // SV_CullDistance
                auto cit = cullSigBaseIdx.find(sigId);
                if (cit != cullSigBaseIdx.end()) {
                    uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (cullDistMask & (1u << idx))) {
                        mlir::Value dst = sh.gepBytes(outAttrs, static_cast<int64_t>((cullAttrOffset + idx) * simdWidth * 4));
                        mlir::LLVM::StoreOp::create(sb, sloc, vVec, dst, /*alignment=*/4);
                    }
                }
                return true;
            }
            if (!hasGS && !hasTess && sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) { // SV_ViewportArrayIndex
                mlir::Value dst = sh.gepBytes(outAttrs, static_cast<int64_t>(vpIndexAttrOffset * simdWidth * 4));
                mlir::LLVM::StoreOp::create(sb, sloc, vVec, dst, /*alignment=*/4);
                return true;
            }
            if (!hasGS && !hasTess && sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) { // SV_ShadingRate
                mlir::Value dst = sh.gepBytes(outAttrs, static_cast<int64_t>(shadingRateAttrOffset * simdWidth * 4));
                mlir::LLVM::StoreOp::create(sb, sloc, vVec, dst, /*alignment=*/4);
                return true;
            }

            auto it = vsToPsMap.find(sigId);
            if (it != vsToPsMap.end() && it->second.psTargetRow >= 0) {
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol;
                storeAttrAtSlot(baseSlot);
            }
            return true;
        };

        mlir::Value vsRealMask = activeMask;
        if (!packetizeRegionWithHooks(b, loc, combinedMod, vsSrcFunc.getBody(), vsLoaded, vsEffectiveLayout,
                                      simdWidth, rootCtx, activeMask, vsRealMask, vsHooks, outError)) {
            return false;
        }
        mlir::func::ReturnOp::create(b, loc);
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
                    mlir::Value v0 = mlir::vector::ExtractOp::create(lb, lloc, vAxis->vec, 0);
                    if (v0.getType() != i32Ty) v0 = mlir::arith::ExtUIOp::create(lb, lloc, i32Ty, v0);
                    idx32 = mlir::arith::AddIOp::create(
                        lb, lloc, idx32, mlir::arith::MulIOp::create(lb, lloc, v0, lh.cInt(i32Ty, cpStride)));
                }
            }
            if (auto rc = tryGetUniformConstUInt(row)) {
                if (*rc != 0) {
                    idx32 = mlir::arith::AddIOp::create(lb, lloc, idx32, lh.cInt(i32Ty, static_cast<int64_t>((*rc) * 4)));
                }
            } else {
                mlir::Value r0 = mlir::vector::ExtractOp::create(lb, lloc, row.vec, 0);
                if (r0.getType() != i32Ty) r0 = mlir::arith::ExtUIOp::create(lb, lloc, i32Ty, r0);
                idx32 = mlir::arith::AddIOp::create(
                    lb, lloc, idx32, mlir::arith::MulIOp::create(lb, lloc, r0, lh.cInt(i32Ty, 4)));
            }
            mlir::Value byteOff = mlir::arith::MulIOp::create(
                lb, lloc, mlir::arith::ExtUIOp::create(lb, lloc, i64Ty, idx32), lh.cInt(i64Ty, 4));
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
                    outVal = mlir::arith::BitcastOp::create(lb, lloc, i32Ty, outVal);
                    if (scalarTy != i32Ty) outVal = mlir::arith::TruncIOp::create(lb, lloc, scalarTy, outVal);
                } else if (mlir::isa<mlir::FloatType>(scalarTy)) {
                    outVal = mlir::arith::TruncFOp::create(lb, lloc, scalarTy, outVal);
                }
            }
            return lh.splatScalar(outVal);
        };

        auto storeLane0ToF32Ptr = [](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t simdW,
                                     const SimdVal& val, mlir::Value realMask, mlir::Value dstPtr) {
            IRHelpers lh{lb, lloc, simdW};
            auto i32Ty = lh.i32Ty();
            auto f32Ty = lh.f32Ty();
            mlir::Value v0 = mlir::vector::ExtractOp::create(lb, lloc, val.vec, 0);
            if (v0.getType() != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(v0.getType())) {
                    if (v0.getType() != i32Ty) v0 = mlir::arith::ExtUIOp::create(lb, lloc, i32Ty, v0);
                    v0 = mlir::arith::BitcastOp::create(lb, lloc, f32Ty, v0);
                } else {
                    v0 = mlir::arith::ExtFOp::create(lb, lloc, f32Ty, v0);
                }
            }
            mlir::Value l0Active = mlir::vector::ExtractOp::create(lb, lloc, realMask, 0);
            mlir::scf::IfOp::create(lb, lloc, l0Active, [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                mlir::LLVM::StoreOp::create(sb, sloc, v0, dstPtr, /*alignment=*/4);
                mlir::scf::YieldOp::create(sb, sloc);
            });
        };

        // 1) @<vsSym>_hs(rootCtx, inCPAttrs, patchId, outCpId, outCPAttrs)
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
            auto f32Ty = h.f32Ty();
            auto fnTy = b.getFunctionType({ptrTy, ptrTy, i32Ty, i32Ty, ptrTy}, {});
            auto hsFn = mlir::func::FuncOp::create(b, loc, vsSym + "_hs", fnTy);
            mlir::Block* entry = hsFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value rootCtx = entry->getArgument(0);
            mlir::Value inCPAttrs = entry->getArgument(1);
            mlir::Value patchIdArg = entry->getArgument(2);
            mlir::Value outCpIdArg = entry->getArgument(3);
            mlir::Value outCPAttrs = entry->getArgument(4);

            mlir::Value activeMask = mlir::arith::CmpIOp::create(
                b, loc, mlir::arith::CmpIPredicate::eq, h.laneIota(), h.splatInt(i32Ty, 0));

            StageHooks hsHooks;
            hsHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
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
            hsHooks.storeOutput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                      const SimdVal& row, uint32_t col, const SimdVal& val,
                                      mlir::Value realMask) -> bool {
                auto it = hsToDsMap.find(sigId);
                if (it == hsToDsMap.end() || it->second.psTargetRow < 0) return true;
                uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, outCPAttrs, 0, nullptr, row, baseSlot);
                storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, ptr);
                return true;
            };
            hsHooks.loadOutputControlPointId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                out = IRHelpers{lb, lloc, simdWidth}.splatScalar(outCpIdArg);
                return true;
            };
            hsHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                out = IRHelpers{lb, lloc, simdWidth}.splatScalar(patchIdArg);
                return true;
            };

            mlir::Value hsRealMask = activeMask;
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

            mlir::Value activeMask = mlir::arith::CmpIOp::create(
                b, loc, mlir::arith::CmpIPredicate::eq, h.laneIota(), h.splatInt(i32Ty, 0));

            StageHooks hspcHooks;
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
                        mlir::Value r0 = mlir::vector::ExtractOp::create(lb, lloc, row.vec, 0);
                        if (r0.getType() != i32Ty) r0 = mlir::arith::ExtUIOp::create(lb, lloc, i32Ty, r0);
                        idx32 = mlir::arith::AddIOp::create(lb, lloc, idx32, r0);
                    }
                    mlir::Value byteOff = mlir::arith::MulIOp::create(
                        lb, lloc, mlir::arith::ExtUIOp::create(lb, lloc, lh.i64Ty(), idx32), lh.cInt(lh.i64Ty(), 4));
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
                out = IRHelpers{lb, lloc, simdWidth}.splatScalar(patchIdArg);
                return true;
            };

            mlir::Value hspcRealMask = activeMask;
            if (!packetizeRegionWithHooks(b, loc, combinedMod, hspcSrcFunc.getBody(), hsLoaded, hsEffectiveLayout,
                                          simdWidth, rootCtx, activeMask, hspcRealMask, hspcHooks, outError)) {
                return false;
            }
            mlir::func::ReturnOp::create(b, loc);
        }

        // 3) @<vsSym>_ds(rootCtx, inCPAttrs, inPatchConsts, inTessFactors, patchId, domU, domV, domW, outPosXYZW, outAttrs)
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
            auto fnTy = b.getFunctionType({ptrTy, ptrTy, ptrTy, ptrTy, i32Ty, f32Ty, f32Ty, f32Ty, ptrTy, ptrTy}, {});
            auto dsFn = mlir::func::FuncOp::create(b, loc, vsSym + "_ds", fnTy);
            mlir::Block* entry = dsFn.addEntryBlock();
            b.setInsertionPointToEnd(entry);

            mlir::Value rootCtx = entry->getArgument(0);
            mlir::Value inCPAttrs = entry->getArgument(1);
            mlir::Value inPatchConsts = entry->getArgument(2);
            mlir::Value inTessFactors = entry->getArgument(3);
            mlir::Value patchIdArg = entry->getArgument(4);
            mlir::Value domUArg = entry->getArgument(5);
            mlir::Value domVArg = entry->getArgument(6);
            mlir::Value domWArg = entry->getArgument(7);
            mlir::Value outPosXYZW = entry->getArgument(8);
            mlir::Value outAttrs = entry->getArgument(9);

            mlir::Value activeMask = mlir::arith::CmpIOp::create(
                b, loc, mlir::arith::CmpIPredicate::eq, h.laneIota(), h.splatInt(i32Ty, 0));

            StageHooks dsHooks;
            dsHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                              const SimdVal& row, uint32_t col, const SimdVal& cpIdx,
                                              mlir::Type scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = dsLoaded.findInputSig(sigId);
                if (!sig) return false;
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, inCPAttrs, numHSOutputScalars, &cpIdx, row, baseSlot);
                mlir::Value fVal = IRHelpers{lb, lloc, simdWidth}.loadScalar(f32Ty, ptr);
                out = convertLoadedF32ToSimd(lb, lloc, simdWidth, fVal, scalarTy);
                return true;
            };
            dsHooks.loadPatchConstant = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                            const SimdVal& row, uint32_t col,
                                            mlir::Type scalarTy, SimdVal& out) -> bool {
                const DxilSigRecord* sig = dsLoaded.findPatchConstSig(sigId);
                if (!sig) return false;
                IRHelpers lh{lb, lloc, simdWidth};
                if (sig->systemValueType == 25 || sig->systemValueType == 26) {
                    uint32_t baseIdx = (sig->systemValueType == 25) ? 0u : 4u;
                    mlir::Value idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseIdx));
                    if (auto rc = tryGetUniformConstUInt(row)) {
                        idx32 = lh.cInt(i32Ty, static_cast<int64_t>(baseIdx + *rc));
                    } else {
                        mlir::Value r0 = mlir::vector::ExtractOp::create(lb, lloc, row.vec, 0);
                        if (r0.getType() != i32Ty) r0 = mlir::arith::ExtUIOp::create(lb, lloc, i32Ty, r0);
                        idx32 = mlir::arith::AddIOp::create(lb, lloc, idx32, r0);
                    }
                    mlir::Value byteOff = mlir::arith::MulIOp::create(
                        lb, lloc, mlir::arith::ExtUIOp::create(lb, lloc, lh.i64Ty(), idx32), lh.cInt(lh.i64Ty(), 4));
                    mlir::Value fVal = lh.loadScalar(f32Ty, lh.gepBytesDyn(inTessFactors, byteOff));
                    out = convertLoadedF32ToSimd(lb, lloc, simdWidth, fVal, scalarTy);
                    return true;
                }
                uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, inPatchConsts, 0, nullptr, row, baseSlot);
                mlir::Value fVal = lh.loadScalar(f32Ty, ptr);
                out = convertLoadedF32ToSimd(lb, lloc, simdWidth, fVal, scalarTy);
                return true;
            };
            dsHooks.loadDomainLocation = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t comp,
                                             mlir::Type scalarTy, SimdVal& out) -> bool {
                IRHelpers lh{lb, lloc, simdWidth};
                mlir::Value v = (comp == 0) ? domUArg : ((comp == 1) ? domVArg : ((comp == 2) ? domWArg : lh.cFlt(0.0)));
                out = lh.splatScalar(v);
                return true;
            };
            dsHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
                out = IRHelpers{lb, lloc, simdWidth}.splatScalar(patchIdArg);
                return true;
            };
            dsHooks.storeOutput = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                      const SimdVal& row, uint32_t col, const SimdVal& val,
                                      mlir::Value realMask) -> bool {
                const DxilSigRecord* sig = dsLoaded.findOutputSig(sigId);
                if (!sig) return false;
                if (sig->systemValueType == 3) {
                    if (col < 4) {
                        mlir::Value dst = IRHelpers{lb, lloc, simdWidth}.gepBytes(outPosXYZW, static_cast<int64_t>(col * 4));
                        storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, dst);
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 6) {
                    auto cit = clipSigBaseIdx.find(sigId);
                    if (cit != clipSigBaseIdx.end()) {
                        uint32_t r = tryGetUniformConstUInt(row).value_or(0);
                        uint32_t idx = cit->second + r * sig->cols + col;
                        if (idx < 8 && (clipDistMask & (1u << idx))) {
                            mlir::Value dst = IRHelpers{lb, lloc, simdWidth}.gepBytes(
                                outAttrs, static_cast<int64_t>((clipAttrOffset + idx) * 4));
                            storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, dst);
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
                            mlir::Value dst = IRHelpers{lb, lloc, simdWidth}.gepBytes(
                                outAttrs, static_cast<int64_t>((cullAttrOffset + idx) * 4));
                            storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, dst);
                        }
                    }
                    return true;
                }
                if (!hasGS && sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                    mlir::Value dst = IRHelpers{lb, lloc, simdWidth}.gepBytes(
                        outAttrs, static_cast<int64_t>(vpIndexAttrOffset * 4));
                    storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, dst);
                    return true;
                }
                if (!hasGS && sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                    mlir::Value dst = IRHelpers{lb, lloc, simdWidth}.gepBytes(
                        outAttrs, static_cast<int64_t>(shadingRateAttrOffset * 4));
                    storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, dst);
                    return true;
                }
                auto it = dsToNextMap.find(sigId);
                if (it != dsToNextMap.end() && it->second.psTargetRow >= 0) {
                    uint32_t baseSlot = static_cast<uint32_t>(it->second.psTargetRow) * 4 + it->second.psStartCol + col;
                    mlir::Value ptr = computeScalarPtr(lb, lloc, simdWidth, outAttrs, 0, nullptr, row, baseSlot);
                    storeLane0ToF32Ptr(lb, lloc, simdWidth, val, realMask, ptr);
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

    // --- 1b. Emit @<vsSym>_gs(rootCtx, inPosXYZW, inAttrs, primId, gsInstId, outPosXYZW, outAttrs, outStripCounts, outTotalVerts, outNumStrips) ---
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
        auto fnTy = b.getFunctionType({ptrTy, ptrTy, ptrTy, i32Ty, i32Ty, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy}, {});
        auto gsFn = mlir::func::FuncOp::create(b, loc, vsSym + "_gs", fnTy);
        mlir::Block* entry = gsFn.addEntryBlock();
        b.setInsertionPointToEnd(entry);

        mlir::Value rootCtx = entry->getArgument(0);
        mlir::Value inPosXYZW = entry->getArgument(1);
        mlir::Value inAttrs = entry->getArgument(2);
        mlir::Value primIdArg = entry->getArgument(3);
        mlir::Value gsInstIdArg = entry->getArgument(4);
        mlir::Value outPosXYZW = entry->getArgument(5);
        mlir::Value outAttrs = entry->getArgument(6);
        mlir::Value outStripCounts = entry->getArgument(7);
        mlir::Value outTotalVerts = entry->getArgument(8);
        mlir::Value outNumStrips = entry->getArgument(9);

        // Run lane 0 only for deterministic per-primitive stream emission
        mlir::Value activeMask = mlir::arith::CmpIOp::create(
            b, loc, mlir::arith::CmpIPredicate::eq, h.laneIota(), h.splatInt(i32Ty, 0));

        uint32_t maxVerts = std::max<uint32_t>(1, gsLoaded.gsMaxVertexCount);
        uint32_t outAttrScalars = totalPostGSVertexAttrs;

        mlir::Value stagingPos = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, f32Ty, h.cInt(i32Ty, 4), 16);
        mlir::Value stagingAttrs = mlir::LLVM::AllocaOp::create(
            b, loc, ptrTy, f32Ty, h.cInt(i32Ty, std::max<uint32_t>(1, outAttrScalars)), 16);
        for (uint32_t c = 0; c < 4; ++c) {
            h.storeScalar(h.cFlt(0.0), h.gepBytes(stagingPos, static_cast<int64_t>(c * 4)));
        }
        for (uint32_t s = 0; s < outAttrScalars; ++s) {
            h.storeScalar(h.cFlt(0.0), h.gepBytes(stagingAttrs, static_cast<int64_t>(s * 4)));
        }

        mlir::Value emitCountPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, i32Ty, h.cInt(i32Ty, 1), 4);
        mlir::Value curStripPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, i32Ty, h.cInt(i32Ty, 1), 4);
        mlir::Value numStripsPtr = mlir::LLVM::AllocaOp::create(b, loc, ptrTy, i32Ty, h.cInt(i32Ty, 1), 4);
        h.storeScalar(h.cInt(i32Ty, 0), emitCountPtr);
        h.storeScalar(h.cInt(i32Ty, 0), curStripPtr);
        h.storeScalar(h.cInt(i32Ty, 0), numStripsPtr);

        StageHooks gsHooks;
        gsHooks.loadInputWithVertex = [&](mlir::OpBuilder& lb, mlir::Location lloc, uint32_t sigId,
                                          const SimdVal& row, uint32_t col, const SimdVal& vAxis,
                                          mlir::Type scalarTy, SimdVal& out) -> bool {
            const DxilSigRecord* sig = gsLoaded.findInputSig(sigId);
            if (!sig) return false;
            IRHelpers lh{lb, lloc, simdWidth};
            uint32_t vIdx = tryGetUniformConstUInt(vAxis).value_or(0);
            uint32_t r = tryGetUniformConstUInt(row).value_or(0);
            mlir::Value scalarVal;
            if (sig->systemValueType == 3) { // SV_Position
                scalarVal = lh.loadScalar(f32Ty, lh.gepBytes(inPosXYZW, static_cast<int64_t>((vIdx * 4 + col) * 4)));
            } else {
                uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                if (slot < numGSInputScalars) {
                    scalarVal = lh.loadScalar(
                        f32Ty, lh.gepBytes(inAttrs, static_cast<int64_t>((vIdx * numGSInputScalars + slot) * 4)));
                } else {
                    scalarVal = lh.cFlt(0.0);
                }
            }
            if (scalarTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(scalarTy)) {
                    scalarVal = mlir::arith::BitcastOp::create(lb, lloc, i32Ty, scalarVal);
                    if (scalarTy != i32Ty) {
                        scalarVal = mlir::arith::TruncIOp::create(lb, lloc, scalarTy, scalarVal);
                    }
                } else {
                    scalarVal = mlir::arith::BitcastOp::create(lb, lloc, scalarTy, scalarVal);
                }
            }
            mlir::Value vecVal = mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(scalarTy), scalarVal);
            out = {SimdVal::Kind::Vector, scalarTy, vecVal, {}, {}};
            return true;
        };

        gsHooks.loadPrimitiveId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
            IRHelpers lh{lb, lloc, simdWidth};
            mlir::Value v = primIdArg;
            if (scalarTy != i32Ty) v = mlir::arith::TruncIOp::create(lb, lloc, scalarTy, v);
            out = {SimdVal::Kind::Vector, scalarTy, mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(scalarTy), v), {}, {}};
            return true;
        };

        gsHooks.loadGSInstanceId = [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Type scalarTy, SimdVal& out) -> bool {
            IRHelpers lh{lb, lloc, simdWidth};
            mlir::Value v = gsInstIdArg;
            if (scalarTy != i32Ty) v = mlir::arith::TruncIOp::create(lb, lloc, scalarTy, v);
            out = {SimdVal::Kind::Vector, scalarTy, mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(scalarTy), v), {}, {}};
            return true;
        };

        gsHooks.storeOutput = [&](mlir::OpBuilder& sb, mlir::Location sloc, uint32_t sigId,
                                  const SimdVal& row, uint32_t col, const SimdVal& val, mlir::Value) -> bool {
            const DxilSigRecord* sig = gsLoaded.findOutputSig(sigId);
            if (!sig || val.kind != SimdVal::Kind::Vector) return true;
            IRHelpers sh{sb, sloc, simdWidth};
            mlir::Value lane0 = mlir::vector::ExtractOp::create(sb, sloc, val.vec, 0);
            mlir::Type elemTy = lane0.getType();
            if (elemTy != f32Ty) {
                if (mlir::isa<mlir::IntegerType>(elemTy)) {
                    if (elemTy.getIntOrFloatBitWidth() < 32) {
                        lane0 = mlir::arith::ExtUIOp::create(sb, sloc, i32Ty, lane0);
                    }
                    lane0 = mlir::arith::BitcastOp::create(sb, sloc, f32Ty, lane0);
                } else {
                    lane0 = mlir::arith::ExtFOp::create(sb, sloc, f32Ty, lane0);
                }
            }
            if (sig->systemValueType == 3) {
                if (col < 4) sh.storeScalar(lane0, sh.gepBytes(stagingPos, static_cast<int64_t>(col * 4)));
                return true;
            }
            uint32_t r = tryGetUniformConstUInt(row).value_or(0);
            if (sig->systemValueType == 6) {
                auto cit = clipSigBaseIdx.find(sigId);
                if (cit != clipSigBaseIdx.end()) {
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (clipDistMask & (1u << idx))) {
                        sh.storeScalar(lane0, sh.gepBytes(stagingAttrs, static_cast<int64_t>((clipAttrOffset + idx) * 4)));
                    }
                }
                return true;
            }
            if (sig->systemValueType == 7) {
                auto cit = cullSigBaseIdx.find(sigId);
                if (cit != cullSigBaseIdx.end()) {
                    uint32_t idx = cit->second + r * sig->cols + col;
                    if (idx < 8 && (cullDistMask & (1u << idx))) {
                        sh.storeScalar(lane0, sh.gepBytes(stagingAttrs, static_cast<int64_t>((cullAttrOffset + idx) * 4)));
                    }
                }
                return true;
            }
            if (sig->systemValueType == 5 && hasViewportArrayIndex && col == 0) {
                sh.storeScalar(lane0, sh.gepBytes(stagingAttrs, static_cast<int64_t>(vpIndexAttrOffset * 4)));
                return true;
            }
            if (sig->systemValueType == 29 && hasPerPrimShadingRate && col == 0) {
                sh.storeScalar(lane0, sh.gepBytes(stagingAttrs, static_cast<int64_t>(shadingRateAttrOffset * 4)));
                return true;
            }
            auto it = gsToPsMap.find(sigId);
            if (it != gsToPsMap.end() && it->second.psTargetRow >= 0) {
                uint32_t slot = (static_cast<uint32_t>(it->second.psTargetRow) + r) * 4 + it->second.psStartCol + col;
                if (slot < outAttrScalars) {
                    sh.storeScalar(lane0, sh.gepBytes(stagingAttrs, static_cast<int64_t>(slot * 4)));
                }
            }
            return true;
        };

        gsHooks.emitStream = [&](mlir::OpBuilder& eb, mlir::Location eloc, uint32_t streamId, mlir::Value realMask) -> bool {
            if (streamId != 0) return true;
            IRHelpers eh{eb, eloc, simdWidth};
            mlir::Value lane0Active = mlir::vector::ExtractOp::create(eb, eloc, realMask, 0);
            mlir::Value curCount = eh.loadScalar(i32Ty, emitCountPtr);
            mlir::Value hasRoom = mlir::arith::CmpIOp::create(
                eb, eloc, mlir::arith::CmpIPredicate::ult, curCount, eh.cInt(i32Ty, maxVerts));
            mlir::Value doEmit = mlir::arith::AndIOp::create(eb, eloc, lane0Active, hasRoom);
            mlir::scf::IfOp::create(eb, eloc, doEmit, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                IRHelpers nh{nb, nloc, simdWidth};
                mlir::Value idx64 = mlir::arith::ExtUIOp::create(nb, nloc, nh.i64Ty(), curCount);
                mlir::Value posBase = mlir::arith::MulIOp::create(nb, nloc, idx64, nh.cInt(nh.i64Ty(), 16));
                for (uint32_t c = 0; c < 4; ++c) {
                    mlir::Value pv = nh.loadScalar(f32Ty, nh.gepBytes(stagingPos, static_cast<int64_t>(c * 4)));
                    mlir::Value off = mlir::arith::AddIOp::create(nb, nloc, posBase, nh.cInt(nh.i64Ty(), static_cast<int64_t>(c * 4)));
                    nh.storeScalar(pv, nh.gepBytesDyn(outPosXYZW, off));
                }
                if (outAttrScalars > 0) {
                    mlir::Value attrBase = mlir::arith::MulIOp::create(
                        nb, nloc, idx64, nh.cInt(nh.i64Ty(), static_cast<int64_t>(outAttrScalars * 4)));
                    for (uint32_t s = 0; s < outAttrScalars; ++s) {
                        mlir::Value av = nh.loadScalar(f32Ty, nh.gepBytes(stagingAttrs, static_cast<int64_t>(s * 4)));
                        mlir::Value off = mlir::arith::AddIOp::create(nb, nloc, attrBase, nh.cInt(nh.i64Ty(), static_cast<int64_t>(s * 4)));
                        nh.storeScalar(av, nh.gepBytesDyn(outAttrs, off));
                    }
                }
                nh.storeScalar(mlir::arith::AddIOp::create(nb, nloc, curCount, nh.cInt(i32Ty, 1)), emitCountPtr);
                mlir::Value cs = nh.loadScalar(i32Ty, curStripPtr);
                nh.storeScalar(mlir::arith::AddIOp::create(nb, nloc, cs, nh.cInt(i32Ty, 1)), curStripPtr);
                mlir::scf::YieldOp::create(nb, nloc);
            });
            return true;
        };

        gsHooks.cutStream = [&](mlir::OpBuilder& cb, mlir::Location cloc, uint32_t streamId, mlir::Value realMask) -> bool {
            if (streamId != 0) return true;
            IRHelpers ch{cb, cloc, simdWidth};
            mlir::Value lane0Active = mlir::vector::ExtractOp::create(cb, cloc, realMask, 0);
            mlir::Value cs = ch.loadScalar(i32Ty, curStripPtr);
            mlir::Value hasVerts = mlir::arith::CmpIOp::create(
                cb, cloc, mlir::arith::CmpIPredicate::ugt, cs, ch.cInt(i32Ty, 0));
            mlir::Value doCut = mlir::arith::AndIOp::create(cb, cloc, lane0Active, hasVerts);
            mlir::scf::IfOp::create(cb, cloc, doCut, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                IRHelpers nh{nb, nloc, simdWidth};
                mlir::Value ns = nh.loadScalar(i32Ty, numStripsPtr);
                mlir::Value off = mlir::arith::MulIOp::create(
                    nb, nloc, mlir::arith::ExtUIOp::create(nb, nloc, nh.i64Ty(), ns), nh.cInt(nh.i64Ty(), 4));
                nh.storeScalar(cs, nh.gepBytesDyn(outStripCounts, off));
                nh.storeScalar(mlir::arith::AddIOp::create(nb, nloc, ns, nh.cInt(i32Ty, 1)), numStripsPtr);
                nh.storeScalar(nh.cInt(i32Ty, 0), curStripPtr);
                mlir::scf::YieldOp::create(nb, nloc);
            });
            return true;
        };

        mlir::Value gsRealMask = activeMask;
        if (!packetizeRegionWithHooks(b, loc, combinedMod, gsSrcFunc.getBody(), gsLoaded, gsEffectiveLayout,
                                      simdWidth, rootCtx, activeMask, gsRealMask, gsHooks, outError)) {
            return false;
        }

        // Flush any remaining vertices in current strip
        mlir::Value finalCS = h.loadScalar(i32Ty, curStripPtr);
        mlir::Value hasFinal = mlir::arith::CmpIOp::create(
            b, loc, mlir::arith::CmpIPredicate::ugt, finalCS, h.cInt(i32Ty, 0));
        mlir::scf::IfOp::create(b, loc, hasFinal, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
            IRHelpers nh{nb, nloc, simdWidth};
            mlir::Value ns = nh.loadScalar(i32Ty, numStripsPtr);
            mlir::Value off = mlir::arith::MulIOp::create(
                nb, nloc, mlir::arith::ExtUIOp::create(nb, nloc, nh.i64Ty(), ns), nh.cInt(nh.i64Ty(), 4));
            nh.storeScalar(finalCS, nh.gepBytesDyn(outStripCounts, off));
            nh.storeScalar(mlir::arith::AddIOp::create(nb, nloc, ns, nh.cInt(i32Ty, 1)), numStripsPtr);
            mlir::scf::YieldOp::create(nb, nloc);
        });

        h.storeScalar(h.loadScalar(i32Ty, emitCountPtr), outTotalVerts);
        h.storeScalar(h.loadScalar(i32Ty, numStripsPtr), outNumStrips);
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
        mlir::Value covMask32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, coverageMask)));
        mlir::Value isFront32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, isFrontFace)));
        mlir::Value primId32 = h.loadScalar(i32Ty, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, primitiveId)));

        mlir::Value covVec = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(i32Ty), covMask32);
        mlir::Value shifted = mlir::arith::ShRUIOp::create(b, loc, covVec, h.laneIota());
        mlir::Value bitOne = mlir::arith::AndIOp::create(b, loc, shifted, h.splatInt(i32Ty, 1));
        mlir::Value initRealMask = mlir::arith::CmpIOp::create(
            b, loc, mlir::arith::CmpIPredicate::ne, bitOne, h.splatInt(i32Ty, 0));

        // Load PixelBatchArgs vectors
        auto loadVecFromArgField = [&](size_t fieldOffset) -> mlir::Value {
            mlir::Value p = h.loadScalar(ptrTy, h.gepBytes(argsPtr, static_cast<int64_t>(fieldOffset)));
            return mlir::LLVM::LoadOp::create(b, loc, h.vecTy(f32Ty), p, /*alignment=*/4);
        };
        mlir::Value pixelX = loadVecFromArgField(offsetof(PixelBatchArgs, pixelX));
        mlir::Value pixelY = loadVecFromArgField(offsetof(PixelBatchArgs, pixelY));
        mlir::Value baryI = loadVecFromArgField(offsetof(PixelBatchArgs, baryI));
        mlir::Value baryJ = loadVecFromArgField(offsetof(PixelBatchArgs, baryJ));

        // Load quad offsets (W/4 elements)
        uint32_t numQuads = simdWidth / 4;
        mlir::Value quadOffPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, quadOffsets)));
        llvm::SmallVector<mlir::Value, 4> quadOffsets(numQuads);
        for (uint32_t q = 0; q < numQuads; ++q) {
            quadOffsets[q] = h.loadScalar(i32Ty, h.gepBytes(quadOffPtr, static_cast<int64_t>(q * 4)));
        }

        // Interpolate Z from triDepth[0..2]
        mlir::Value triDepthPtr = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, triDepth)));
        mlir::Value z0 = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(f32Ty), h.loadScalar(f32Ty, triDepthPtr));
        mlir::Value dz1 = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(f32Ty), h.loadScalar(f32Ty, h.gepBytes(triDepthPtr, 4)));
        mlir::Value dz2 = mlir::vector::BroadcastOp::create(b, loc, h.vecTy(f32Ty), h.loadScalar(f32Ty, h.gepBytes(triDepthPtr, 8)));
        mlir::Value interpZ = mlir::arith::AddFOp::create(
            b, loc, z0,
            mlir::arith::AddFOp::create(
                b, loc, mlir::arith::MulFOp::create(b, loc, baryI, dz1),
                mlir::arith::MulFOp::create(b, loc, baryJ, dz2)));
        if (!desc.RasterizerState.DepthClipEnable) {
            interpZ = h.clamp01(interpZ);
        }

        mlir::Value tileDepthBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, tileDepthSOA)));
        mlir::Value tileStencilBase = h.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, tileStencilSOA)));
        mlir::Value tileColorBases[8]{};
        for (uint32_t rt = 0; rt < 8; ++rt) {
            tileColorBases[rt] = h.loadScalar(
                ptrTy, h.gepBytes(argsPtr, static_cast<int64_t>(offsetof(PixelBatchArgs, tileColorSOA) + rt * 8)));
        }

        // Helper lambdas for quad-swizzled depth & stencil SOA access (zero branches on loads, masked on writes)
        auto loadDepthVec = [&](mlir::OpBuilder& db, mlir::Location dloc) -> mlir::Value {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value out = dh.splatFlt(1.0);
            auto v4f = mlir::VectorType::get({4}, f32Ty);
            for (uint32_t q = 0; q < numQuads; ++q) {
                mlir::Value qByte = mlir::arith::MulIOp::create(
                    db, dloc, mlir::arith::ExtUIOp::create(db, dloc, dh.i64Ty(), quadOffsets[q]), dh.cInt(dh.i64Ty(), 4));
                mlir::Value qp = dh.gepBytesDyn(tileDepthBase, qByte);
                mlir::Value q4 = dh.loadScalar(v4f, qp);
                for (uint32_t s = 0; s < 4; ++s) {
                    mlir::Value sc = mlir::vector::ExtractOp::create(db, dloc, q4, static_cast<int64_t>(s));
                    out = mlir::vector::InsertOp::create(db, dloc, sc, out, static_cast<int64_t>(q * 4 + s));
                }
            }
            return out;
        };

        auto storeDepthVec = [&](mlir::OpBuilder& db, mlir::Location dloc, mlir::Value maskVec, mlir::Value newZ, mlir::Value oldZ) {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value merged = mlir::arith::SelectOp::create(db, dloc, maskVec, newZ, oldZ);
            auto v4f = mlir::VectorType::get({4}, f32Ty);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::SmallVector<int64_t, 4> qIdx{q * 4, q * 4 + 1, q * 4 + 2, q * 4 + 3};
                mlir::Value qMask = mlir::vector::ShuffleOp::create(db, dloc, maskVec, maskVec, qIdx);
                mlir::Value anyQ = mlir::vector::ReductionOp::create(db, dloc, mlir::vector::CombiningKind::OR, qMask);
                mlir::scf::IfOp::create(db, dloc, anyQ, [&](mlir::OpBuilder& qb, mlir::Location qloc) {
                    IRHelpers qh{qb, qloc, simdWidth};
                    mlir::Value q4 = mlir::vector::ShuffleOp::create(qb, qloc, merged, merged, qIdx);
                    mlir::Value qByte = mlir::arith::MulIOp::create(
                        qb, qloc, mlir::arith::ExtUIOp::create(qb, qloc, qh.i64Ty(), quadOffsets[q]), qh.cInt(qh.i64Ty(), 4));
                    mlir::Value qp = qh.gepBytesDyn(tileDepthBase, qByte);
                    mlir::LLVM::StoreOp::create(qb, qloc, q4, qp);
                    mlir::scf::YieldOp::create(qb, qloc);
                });
            }
        };

        auto loadStencilVec = [&](mlir::OpBuilder& db, mlir::Location dloc) -> mlir::Value {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value out = dh.splatInt(dh.i8Ty(), 0);
            auto v4i8 = mlir::VectorType::get({4}, dh.i8Ty());
            for (uint32_t q = 0; q < numQuads; ++q) {
                mlir::Value q64 = mlir::arith::ExtUIOp::create(db, dloc, dh.i64Ty(), quadOffsets[q]);
                mlir::Value qp = dh.gepBytesDyn(tileStencilBase, q64);
                mlir::Value q4 = dh.loadScalar(v4i8, qp);
                for (uint32_t s = 0; s < 4; ++s) {
                    mlir::Value sc = mlir::vector::ExtractOp::create(db, dloc, q4, static_cast<int64_t>(s));
                    out = mlir::vector::InsertOp::create(db, dloc, sc, out, static_cast<int64_t>(q * 4 + s));
                }
            }
            return out;
        };

        auto storeStencilVec = [&](mlir::OpBuilder& db, mlir::Location dloc, mlir::Value maskVec, mlir::Value newS, mlir::Value oldS) {
            IRHelpers dh{db, dloc, simdWidth};
            mlir::Value merged = mlir::arith::SelectOp::create(db, dloc, maskVec, newS, oldS);
            for (uint32_t q = 0; q < numQuads; ++q) {
                llvm::SmallVector<int64_t, 4> qIdx{q * 4, q * 4 + 1, q * 4 + 2, q * 4 + 3};
                mlir::Value qMask = mlir::vector::ShuffleOp::create(db, dloc, maskVec, maskVec, qIdx);
                mlir::Value anyQ = mlir::vector::ReductionOp::create(db, dloc, mlir::vector::CombiningKind::OR, qMask);
                mlir::scf::IfOp::create(db, dloc, anyQ, [&](mlir::OpBuilder& qb, mlir::Location qloc) {
                    IRHelpers qh{qb, qloc, simdWidth};
                    mlir::Value q4 = mlir::vector::ShuffleOp::create(qb, qloc, merged, merged, qIdx);
                    mlir::Value q64 = mlir::arith::ExtUIOp::create(qb, qloc, qh.i64Ty(), quadOffsets[q]);
                    mlir::Value qp = qh.gepBytesDyn(tileStencilBase, q64);
                    mlir::LLVM::StoreOp::create(qb, qloc, q4, qp);
                    mlir::scf::YieldOp::create(qb, qloc);
                });
            }
        };

        auto runDepthStencil = [&](mlir::OpBuilder& db, mlir::Location dloc, mlir::Value curRealMask, mlir::Value candZ,
                                   mlir::Value candStencilRef8 = mlir::Value()) -> mlir::Value {
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
                oldStencil = loadStencilVec(db, dloc);
                if (candStencilRef8) {
                    ref8 = candStencilRef8;
                } else {
                    mlir::Value ref32 = dh.loadScalar(i32Ty, dh.gepBytes(rootCtx, offsetof(DriverRootContext, stencil_ref)));
                    mlir::Value refScalar8 = mlir::arith::TruncIOp::create(db, dloc, dh.i8Ty(), ref32);
                    ref8 = mlir::vector::BroadcastOp::create(db, dloc, dh.vecTy(dh.i8Ty()), refScalar8);
                }
                mlir::Value rMask8 = dh.splatInt(dh.i8Ty(), ds.StencilReadMask);
                writeMask8 = dh.splatInt(dh.i8Ty(), ds.StencilWriteMask);
                mlir::Value refMasked = mlir::arith::AndIOp::create(db, dloc, ref8, rMask8);
                mlir::Value bufMasked = mlir::arith::AndIOp::create(db, dloc, oldStencil, rMask8);
                mlir::Value isFrontBool = mlir::arith::CmpIOp::create(
                    db, dloc, mlir::arith::CmpIPredicate::ne, isFront32, dh.cInt(i32Ty, 0));
                mlir::Value frontPass = dh.evalStencilCmp(ds.FrontFace.StencilFunc, refMasked, bufMasked);
                mlir::Value backPass = dh.evalStencilCmp(ds.BackFace.StencilFunc, refMasked, bufMasked);
                stencilPassMask = mlir::arith::SelectOp::create(
                    db, dloc, mlir::vector::BroadcastOp::create(db, dloc, dh.vecTy(dh.i1Ty()), isFrontBool),
                    frontPass, backPass);
                nextMask = mlir::arith::AndIOp::create(db, dloc, nextMask, stencilPassMask);
            }

            mlir::Value zPassMask = dh.allTrueMask();
            if (depthOn) {
                mlir::Value oldZ = loadDepthVec(db, dloc);
                zPassMask = dh.evalDepthCmp(ds.DepthFunc, candZ, oldZ);
                nextMask = mlir::arith::AndIOp::create(db, dloc, nextMask, zPassMask);
                if (ds.DepthWriteMask == D3D12_DEPTH_WRITE_MASK_ALL) {
                    storeDepthVec(db, dloc, nextMask, candZ, oldZ);
                }
            }

            if (stencilOn && ds.StencilWriteMask != 0) {
                mlir::Value sFailMask = mlir::arith::AndIOp::create(
                    db, dloc, curRealMask, mlir::arith::XOrIOp::create(db, dloc, stencilPassMask, dh.allTrueMask()));
                mlir::Value zFailMask = mlir::arith::AndIOp::create(
                    db, dloc, mlir::arith::AndIOp::create(db, dloc, curRealMask, stencilPassMask),
                    mlir::arith::XOrIOp::create(db, dloc, zPassMask, dh.allTrueMask()));
                mlir::Value sUpdated = oldStencil;
                if (activeSOp.StencilFailOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = mlir::arith::SelectOp::create(
                        db, dloc, sFailMask, dh.applyStencilOp(activeSOp.StencilFailOp, oldStencil, ref8), sUpdated);
                }
                if (activeSOp.StencilDepthFailOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = mlir::arith::SelectOp::create(
                        db, dloc, zFailMask, dh.applyStencilOp(activeSOp.StencilDepthFailOp, oldStencil, ref8), sUpdated);
                }
                if (activeSOp.StencilPassOp != D3D12_STENCIL_OP_KEEP) {
                    sUpdated = mlir::arith::SelectOp::create(
                        db, dloc, nextMask, dh.applyStencilOp(activeSOp.StencilPassOp, oldStencil, ref8), sUpdated);
                }
                mlir::Value keepBits = dh.splatInt(dh.i8Ty(), static_cast<uint8_t>(~ds.StencilWriteMask));
                mlir::Value maskedNewS = mlir::arith::OrIOp::create(
                    db, dloc, mlir::arith::AndIOp::create(db, dloc, oldStencil, keepBits),
                    mlir::arith::AndIOp::create(db, dloc, sUpdated, writeMask8));
                storeStencilVec(db, dloc, curRealMask, maskedNewS, oldStencil);
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

        mlir::Value realMaskAfterEarly = initRealMask;
        if (canEarlyZ) {
            realMaskAfterEarly = runDepthStencil(b, loc, initRealMask, interpZ);
        }

        if (hasPS) {
            mlir::func::FuncOp psSrcFunc = psMlirMod->lookupSymbol<mlir::func::FuncOp>(psLoaded.entryName);
            if (!psSrcFunc || psSrcFunc.getBody().empty()) {
                outError = "Pixel shader entry function missing after MLIR SCF lift";
                return false;
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
                    mlir::Value triInvWPtr = ph.loadScalar(ptrTy, ph.gepBytes(argsPtr, offsetof(PixelBatchArgs, triInvW)));
                    mlir::Value iw0 = mlir::vector::BroadcastOp::create(pb, ploc, ph.vecTy(f32Ty), ph.loadScalar(f32Ty, triInvWPtr));
                    mlir::Value diw1 = mlir::vector::BroadcastOp::create(pb, ploc, ph.vecTy(f32Ty), ph.loadScalar(f32Ty, ph.gepBytes(triInvWPtr, 4)));
                    mlir::Value diw2 = mlir::vector::BroadcastOp::create(pb, ploc, ph.vecTy(f32Ty), ph.loadScalar(f32Ty, ph.gepBytes(triInvWPtr, 8)));
                    invWVec = mlir::arith::AddFOp::create(
                        pb, ploc, iw0,
                        mlir::arith::AddFOp::create(
                            pb, ploc, mlir::arith::MulFOp::create(pb, ploc, baryI, diw1),
                            mlir::arith::MulFOp::create(pb, ploc, baryJ, diw2)));
                    mlir::Value safeInvW = mlir::arith::SelectOp::create(
                        pb, ploc,
                        mlir::arith::CmpFOp::create(pb, ploc, mlir::arith::CmpFPredicate::OGT,
                                                    mlir::math::AbsFOp::create(pb, ploc, invWVec), ph.splatFlt(1e-20)),
                        invWVec, ph.splatFlt(1.0));
                    wVec = mlir::arith::DivFOp::create(pb, ploc, ph.splatFlt(1.0), safeInvW);
                }

                mlir::Value triAttrBasePtr = ph.loadScalar(ptrTy, ph.gepBytes(argsPtr, offsetof(PixelBatchArgs, triAttrBase)));
                mlir::Value triAttrFlatPtr = ph.loadScalar(ptrTy, h.gepBytes(argsPtr, offsetof(PixelBatchArgs, triAttrFlat)));

                // Cache interpolated scalar registers (max 16 * 4 = 64 scalars)
                std::array<mlir::Value, 64> interpScalars{};
                for (const auto& pi : psLoaded.inputSigs) {
                    if (pi.systemValueType != 0) continue;
                    for (uint32_t r = 0; r < pi.rows; ++r) {
                        for (uint32_t c = 0; c < pi.cols && (pi.startCol + c) < 4; ++c) {
                            uint32_t slot = (pi.startRow + r) * 4 + (pi.startCol + c);
                            if (slot >= 64) continue;
                            if (pi.interpMode == 1) { // nointerpolation
                                mlir::Value fv = ph.loadScalar(f32Ty, ph.gepBytes(triAttrFlatPtr, static_cast<int64_t>(slot * 4)));
                                interpScalars[slot] = mlir::vector::BroadcastOp::create(pb, ploc, ph.vecTy(f32Ty), fv);
                            } else {
                                int64_t baseOff = static_cast<int64_t>(slot * 3 * 4);
                                mlir::Value a0 = mlir::vector::BroadcastOp::create(
                                    pb, ploc, ph.vecTy(f32Ty), ph.loadScalar(f32Ty, ph.gepBytes(triAttrBasePtr, baseOff)));
                                mlir::Value da1 = mlir::vector::BroadcastOp::create(
                                    pb, ploc, ph.vecTy(f32Ty), ph.loadScalar(f32Ty, ph.gepBytes(triAttrBasePtr, baseOff + 4)));
                                mlir::Value da2 = mlir::vector::BroadcastOp::create(
                                    pb, ploc, ph.vecTy(f32Ty), ph.loadScalar(f32Ty, ph.gepBytes(triAttrBasePtr, baseOff + 8)));
                                mlir::Value lin = mlir::arith::AddFOp::create(
                                    pb, ploc, a0,
                                    mlir::arith::AddFOp::create(
                                        pb, ploc, mlir::arith::MulFOp::create(pb, ploc, baryI, da1),
                                        mlir::arith::MulFOp::create(pb, ploc, baryJ, da2)));
                                interpScalars[slot] = (pi.interpMode == 4)
                                                          ? lin
                                                          : mlir::arith::MulFOp::create(pb, ploc, lin, wVec).getResult();
                            }
                        }
                    }
                }

                // Allocate stack slots for PS outputs (SV_Target0..7 and optional SV_Depth) so writes inside scf.if work naturally
                std::array<std::array<mlir::Value, 4>, 8> rtOutPtrs{};
                for (uint32_t rt = 0; rt < 8; ++rt) {
                    if (rt < desc.NumRenderTargets && desc.RTVFormats[rt] != DXGI_FORMAT_UNKNOWN) {
                        for (uint32_t c = 0; c < 4; ++c) {
                            rtOutPtrs[rt][c] = mlir::LLVM::AllocaOp::create(
                                pb, ploc, ptrTy, ph.vecTy(f32Ty), ph.cInt(i32Ty, 1), std::max<uint32_t>(64, ph.W * 8));
                            ph.storeScalar(ph.splatFlt(c == 3 ? 1.0 : 0.0), rtOutPtrs[rt][c]);
                        }
                    }
                }
                mlir::Value outDepthPtr = mlir::LLVM::AllocaOp::create(
                    pb, ploc, ptrTy, ph.vecTy(f32Ty), ph.cInt(i32Ty, 1), std::max<uint32_t>(64, ph.W * 8));
                ph.storeScalar(interpZ, outDepthPtr);

                mlir::Value outStencilRefPtr;
                if (psWritesStencilRef) {
                    outStencilRefPtr = mlir::LLVM::AllocaOp::create(
                        pb, ploc, ptrTy, ph.vecTy(ph.i8Ty()), ph.cInt(i32Ty, 1), std::max<uint32_t>(64, ph.W * 8));
                    mlir::Value initRef32 = ph.loadScalar(i32Ty, ph.gepBytes(rootCtx, offsetof(DriverRootContext, stencil_ref)));
                    mlir::Value initRef8 = mlir::arith::TruncIOp::create(pb, ploc, ph.i8Ty(), initRef32);
                    ph.storeScalar(mlir::vector::BroadcastOp::create(pb, ploc, ph.vecTy(ph.i8Ty()), initRef8), outStencilRefPtr);
                }

                StageHooks psHooks;
                psHooks.isPixelShader = true;
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
                        mlir::Value vp32 = lh.loadScalar(
                            i32Ty, lh.gepBytes(argsPtr, offsetof(PixelBatchArgs, viewportArrayIndex)));
                        mlir::Value v = mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(i32Ty), vp32).getResult();
                        if (scalarTy != i32Ty) {
                            v = mlir::arith::TruncIOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 10) { // SV_PrimitiveID
                        mlir::Value v = mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(i32Ty), primId32).getResult();
                        if (scalarTy != i32Ty) {
                            v = mlir::arith::TruncIOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 28) { // SV_Barycentrics
                        mlir::Value v = lh.splatFlt(0.0);
                        if (sig->interpMode == 4) { // noperspective
                            mlir::Value b0 = mlir::arith::SubFOp::create(
                                lb, lloc, lh.splatFlt(1.0),
                                mlir::arith::AddFOp::create(lb, lloc, baryI, baryJ));
                            if (col == 0) v = b0;
                            else if (col == 1) v = baryI;
                            else if (col == 2) v = baryJ;
                        } else if (col < 3) { // perspective (linear) using original-primitive barycentrics
                            mlir::Value triOrigBaryPtr = lh.loadScalar(
                                ptrTy, lh.gepBytes(argsPtr, offsetof(PixelBatchArgs, triOrigBary)));
                            int64_t baseOff = static_cast<int64_t>(col * 3 * 4);
                            mlir::Value a0 = mlir::vector::BroadcastOp::create(
                                lb, lloc, lh.vecTy(f32Ty), lh.loadScalar(f32Ty, lh.gepBytes(triOrigBaryPtr, baseOff)));
                            mlir::Value da1 = mlir::vector::BroadcastOp::create(
                                lb, lloc, lh.vecTy(f32Ty), lh.loadScalar(f32Ty, lh.gepBytes(triOrigBaryPtr, baseOff + 4)));
                            mlir::Value da2 = mlir::vector::BroadcastOp::create(
                                lb, lloc, lh.vecTy(f32Ty), lh.loadScalar(f32Ty, lh.gepBytes(triOrigBaryPtr, baseOff + 8)));
                            mlir::Value lin = mlir::arith::AddFOp::create(
                                lb, lloc, a0,
                                mlir::arith::AddFOp::create(
                                    lb, lloc, mlir::arith::MulFOp::create(lb, lloc, baryI, da1),
                                    mlir::arith::MulFOp::create(lb, lloc, baryJ, da2)));
                            v = mlir::arith::MulFOp::create(lb, lloc, lin, wVec);
                        }
                        if (scalarTy != f32Ty) {
                            v = mlir::arith::TruncFOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 29) { // SV_ShadingRate
                        mlir::Value sr32 = lh.loadScalar(
                            i32Ty, lh.gepBytes(argsPtr, offsetof(PixelBatchArgs, shadingRate)));
                        mlir::Value v = mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(i32Ty), sr32).getResult();
                        if (scalarTy != i32Ty) {
                            v = mlir::arith::TruncIOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                        }
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    if (sig->systemValueType == 13) { // SV_IsFrontFace
                        mlir::Value ffCond = mlir::arith::CmpIOp::create(
                            lb, lloc, mlir::arith::CmpIPredicate::ne, isFront32, lh.cInt(i32Ty, 0));
                        mlir::Value v = (scalarTy == lh.i1Ty())
                                            ? mlir::vector::BroadcastOp::create(lb, lloc, lh.vecTy(lh.i1Ty()), ffCond).getResult()
                                            : mlir::vector::BroadcastOp::create(
                                                  lb, lloc, lh.vecTy(i32Ty),
                                                  mlir::arith::ExtUIOp::create(lb, lloc, i32Ty, ffCond))
                                                  .getResult();
                        out = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                        return true;
                    }
                    auto rowConst = tryGetUniformConstUInt(row);
                    mlir::Value v;
                    if (rowConst.has_value() || sig->rows <= 1) {
                        uint32_t r = rowConst.value_or(0);
                        uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                        v = (slot < 64 && interpScalars[slot]) ? interpScalars[slot] : lh.splatFlt(0.0);
                    } else {
                        uint32_t baseSlot = sig->startRow * 4 + sig->startCol + col;
                        v = (baseSlot < 64 && interpScalars[baseSlot]) ? interpScalars[baseSlot] : lh.splatFlt(0.0);
                        mlir::Type rElemTy = llvm::cast<mlir::VectorType>(row.vec.getType()).getElementType();
                        for (uint32_t r = 1; r < sig->rows; ++r) {
                            uint32_t slot = (sig->startRow + r) * 4 + sig->startCol + col;
                            if (slot >= 64 || !interpScalars[slot]) continue;
                            mlir::Value cmp = mlir::arith::CmpIOp::create(
                                lb, lloc, mlir::arith::CmpIPredicate::eq, row.vec, lh.splatInt(rElemTy, r));
                            v = mlir::arith::SelectOp::create(lb, lloc, cmp, interpScalars[slot], v);
                        }
                    }
                    if (scalarTy != f32Ty) {
                        if (mlir::isa<mlir::IntegerType>(scalarTy)) {
                            v = mlir::arith::BitcastOp::create(lb, lloc, lh.vecTy(i32Ty), v);
                            if (scalarTy != i32Ty) {
                                v = mlir::arith::TruncIOp::create(lb, lloc, lh.vecTy(scalarTy), v);
                            }
                        } else {
                            v = mlir::arith::BitcastOp::create(lb, lloc, lh.vecTy(scalarTy), v);
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
                                          : mlir::arith::TruncIOp::create(sb, sloc, sh.vecTy(sh.i8Ty()), vVec).getResult();
                        } else {
                            mlir::Value asI32 = mlir::arith::BitcastOp::create(sb, sloc, sh.vecTy(i32Ty), vVec);
                            refVec8 = mlir::arith::TruncIOp::create(sb, sloc, sh.vecTy(sh.i8Ty()), asI32);
                        }
                        mlir::Value prev = sh.loadScalar(sh.vecTy(sh.i8Ty()), outStencilRefPtr);
                        sh.storeScalar(mlir::arith::SelectOp::create(sb, sloc, writeMask, refVec8, prev), outStencilRefPtr);
                        return true;
                    }
                    if (et != f32Ty) {
                        vVec = mlir::isa<mlir::IntegerType>(et)
                                   ? mlir::arith::BitcastOp::create(sb, sloc, sh.vecTy(f32Ty), vVec).getResult()
                                   : mlir::arith::ExtFOp::create(sb, sloc, sh.vecTy(f32Ty), vVec).getResult();
                    }
                    if (sig->systemValueType == 17 || sig->systemValueType == 18 || sig->systemValueType == 19) { // SV_Depth*
                        mlir::Value prev = sh.loadScalar(sh.vecTy(f32Ty), outDepthPtr);
                        sh.storeScalar(mlir::arith::SelectOp::create(sb, sloc, writeMask, vVec, prev), outDepthPtr);
                        return true;
                    }
                    if (sig->systemValueType == 16 && sig->semanticIndex < 8 && col < 4) { // SV_TargetN
                        mlir::Value slotPtr = rtOutPtrs[sig->semanticIndex][col];
                        if (slotPtr) {
                            mlir::Value prev = sh.loadScalar(sh.vecTy(f32Ty), slotPtr);
                            sh.storeScalar(mlir::arith::SelectOp::create(sb, sloc, writeMask, vVec, prev), slotPtr);
                        }
                    }
                    return true;
                };

                if (!packetizeRegionWithHooks(pb, ploc, combinedMod, psSrcFunc.getBody(), psLoaded, psEffectiveLayout,
                                              simdWidth, rootCtx, execMask, curRealMask, psHooks, outError)) {
                    return;
                }

                // Late-Z if Early-Z was skipped
                if (!canEarlyZ) {
                    mlir::Value finalZ = ph.loadScalar(ph.vecTy(f32Ty), outDepthPtr);
                    mlir::Value finalStencilRef = psWritesStencilRef
                                                      ? ph.loadScalar(ph.vecTy(ph.i8Ty()), outStencilRefPtr)
                                                      : mlir::Value();
                    curRealMask = runDepthStencil(pb, ploc, curRealMask, finalZ, finalStencilRef);
                }

                // Fixed-function Output Merger Blending across active MRTs
                mlir::Value anyWrite = ph.orReduce(curRealMask);
                mlir::scf::IfOp::create(pb, ploc, anyWrite, [&](mlir::OpBuilder& ob, mlir::Location oloc) {
                    IRHelpers oh{ob, oloc, simdWidth};
                    mlir::Value blendFactors[4]{};
                    bool loadedBlendFactor = false;
                    auto ensureBlendFactor = [&]() {
                        if (loadedBlendFactor) return;
                        for (int c = 0; c < 4; ++c) {
                            mlir::Value bf = oh.loadScalar(
                                f32Ty, oh.gepBytes(rootCtx, static_cast<int64_t>(offsetof(DriverRootContext, blend_factor) + c * 4)));
                            blendFactors[c] = mlir::vector::BroadcastOp::create(ob, oloc, oh.vecTy(f32Ty), bf);
                        }
                        loadedBlendFactor = true;
                    };

                    for (uint32_t rt = 0; rt < desc.NumRenderTargets && rt < 8; ++rt) {
                        if (desc.RTVFormats[rt] == DXGI_FORMAT_UNKNOWN) continue;
                        const auto& rtBlend = desc.BlendState.RenderTarget[desc.BlendState.IndependentBlendEnable ? rt : 0];
                        uint8_t writeMask = rtBlend.RenderTargetWriteMask;
                        if (writeMask == 0) continue;

                        mlir::Value src[4];
                        for (int c = 0; c < 4; ++c) {
                            src[c] = rtOutPtrs[rt][c] ? oh.loadScalar(oh.vecTy(f32Ty), rtOutPtrs[rt][c]) : oh.splatFlt(c == 3 ? 1.0 : 0.0);
                            if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) {
                                src[c] = oh.clamp01(src[c]);
                            }
                        }

                        // Load current tile color for all W lanes (needed for partial quad masking and blending)
                        mlir::Value dst[4];
                        auto v4f = mlir::VectorType::get({4}, f32Ty);
                        for (int c = 0; c < 4; ++c) {
                            dst[c] = oh.splatFlt(0.0);
                            for (uint32_t q = 0; q < numQuads; ++q) {
                                mlir::Value pixIdx = mlir::arith::AddIOp::create(
                                    ob, oloc, quadOffsets[q], oh.cInt(i32Ty, c * 4096));
                                mlir::Value byteOff = mlir::arith::MulIOp::create(
                                    ob, oloc, mlir::arith::ExtUIOp::create(ob, oloc, oh.i64Ty(), pixIdx), oh.cInt(oh.i64Ty(), 4));
                                mlir::Value q4 = oh.loadScalar(v4f, oh.gepBytesDyn(tileColorBases[rt], byteOff));
                                for (uint32_t s = 0; s < 4; ++s) {
                                    mlir::Value sc = mlir::vector::ExtractOp::create(ob, oloc, q4, static_cast<int64_t>(s));
                                    dst[c] = mlir::vector::InsertOp::create(ob, oloc, sc, dst[c], static_cast<int64_t>(q * 4 + s));
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
                                mlir::Value sTerm = mlir::arith::MulFOp::create(ob, oloc, src[c], sf);
                                mlir::Value dTerm = mlir::arith::MulFOp::create(ob, oloc, dst[c], df);
                                outColor[c] = oh.evalBlendOp(rtBlend.BlendOp, sTerm, dTerm);
                                if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) outColor[c] = oh.clamp01(outColor[c]);
                            }
                            mlir::Value saf = oh.evalBlendFactorRGB(rtBlend.SrcBlendAlpha, src, dst, blendFactors, 3);
                            mlir::Value daf = oh.evalBlendFactorRGB(rtBlend.DestBlendAlpha, src, dst, blendFactors, 3);
                            outColor[3] = oh.evalBlendOp(
                                rtBlend.BlendOpAlpha,
                                mlir::arith::MulFOp::create(ob, oloc, src[3], saf),
                                mlir::arith::MulFOp::create(ob, oloc, dst[3], daf));
                            if (isUnormRenderTargetFormat(desc.RTVFormats[rt])) outColor[3] = oh.clamp01(outColor[3]);
                        }

                        // Masked SOA quad write-back respecting RenderTargetWriteMask & curRealMask
                        for (uint32_t q = 0; q < numQuads; ++q) {
                            llvm::SmallVector<int64_t, 4> qIdx{q * 4, q * 4 + 1, q * 4 + 2, q * 4 + 3};
                            mlir::Value qMask = mlir::vector::ShuffleOp::create(ob, oloc, curRealMask, curRealMask, qIdx);
                            mlir::Value anyQ = mlir::vector::ReductionOp::create(ob, oloc, mlir::vector::CombiningKind::OR, qMask);
                            mlir::scf::IfOp::create(ob, oloc, anyQ, [&](mlir::OpBuilder& qb, mlir::Location qloc) {
                                IRHelpers qh{qb, qloc, simdWidth};
                                for (int c = 0; c < 4; ++c) {
                                    if (!(writeMask & (1u << c))) continue;
                                    mlir::Value merged = mlir::arith::SelectOp::create(qb, qloc, curRealMask, outColor[c], dst[c]);
                                    mlir::Value q4 = mlir::vector::ShuffleOp::create(qb, qloc, merged, merged, qIdx);
                                    mlir::Value pixIdx = mlir::arith::AddIOp::create(
                                        qb, qloc, quadOffsets[q], qh.cInt(i32Ty, c * 4096));
                                    mlir::Value byteOff = mlir::arith::MulIOp::create(
                                        qb, qloc, mlir::arith::ExtUIOp::create(qb, qloc, qh.i64Ty(), pixIdx), qh.cInt(qh.i64Ty(), 4));
                                    mlir::LLVM::StoreOp::create(qb, qloc, q4, qh.gepBytesDyn(tileColorBases[rt], byteOff));
                                }
                                mlir::scf::YieldOp::create(qb, qloc);
                            });
                        }
                    }
                    mlir::scf::YieldOp::create(ob, oloc);
                });
            };

            if (canEarlyZ) {
                mlir::Value anySurvive = h.orReduce(realMaskAfterEarly);
                mlir::scf::IfOp::create(b, loc, anySurvive, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    emitPSBodyAndOM(nb, nloc, realMaskAfterEarly);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
                if (!outError.empty()) return false;
            } else {
                emitPSBodyAndOM(b, loc, realMaskAfterEarly);
                if (!outError.empty()) return false;
            }
        }

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
    outMeta.psWritesDepth = psWritesDepth ? 1 : 0;
    outMeta.psUsesDiscard = psUsesDiscard ? 1 : 0;
    outMeta.psHasSideEffects = psHasSideEffects ? 1 : 0;
    return true;
}

bool compileGraphicsPSO(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    CompiledGraphicsPSO& outPSO,
    std::string& outError) {
    uint32_t simdWidth = getTargetSimdWidth();
    uint64_t lo = 0, hi = 0;
    computeGraphicsPSOHash(desc, rootLayout, simdWidth, lo, hi);
    outPSO.cacheHashLo = lo;
    outPSO.cacheHashHi = hi;

    std::string hex = hashToHex128(lo, hi);
    std::string vsSym = "nacelle_vs_" + hex;
    std::string psSym = "nacelle_ps_" + hex;

    if (tryLoadGraphicsPSOFromCache(lo, hi, vsSym, psSym, outPSO)) {
        return true;
    }

    mlir::MLIRContext mlirCtx;
    mlir::OwningOpRef<mlir::ModuleOp> combinedMod;
    CachedPSOMeta meta;
    if (!synthesizeGraphicsPSOMLIR(
            desc, rootLayout, simdWidth, vsSym, psSym, mlirCtx, combinedMod, meta,
            outPSO.vsInputs, outPSO.vsOutputs, outPSO.psInputs, outPSO.psOutputs, outError)) {
        return false;
    }

    auto outLlvmCtx = std::make_unique<llvm::LLVMContext>();
    auto linkedMod = translateMLIRToLinkedLLVMIR(*combinedMod, *outLlvmCtx, outError);
    if (!linkedMod) return false;

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
    outPSO.psWritesDepth = meta.psWritesDepth != 0;
    outPSO.psUsesDiscard = meta.psUsesDiscard != 0;
    outPSO.psHasSideEffects = meta.psHasSideEffects != 0;

    return jitCompileGraphicsTier1AndScheduleTier2(
        std::move(linkedMod), std::move(outLlvmCtx), vsSym, psSym, meta, outPSO, outError);
}

} // namespace nacelle
