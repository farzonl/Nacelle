#include "simt_to_simd_pass.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <unordered_map>

#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Verifier.h>

namespace nacelle {

namespace {

static std::optional<uint64_t> getConstIntValue(mlir::Value v) {
    if (!v) return std::nullopt;
    if (auto c = v.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
            return ia.getValue().getZExtValue();
        }
        if (auto dea = llvm::dyn_cast<mlir::DenseElementsAttr>(c.getValue())) {
            if (dea.isSplat() && mlir::isa<mlir::IntegerType>(dea.getElementType())) {
                return dea.getSplatValue<llvm::APInt>().getZExtValue();
            }
        }
    }
    if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>()) {
        if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
            return ia.getValue().getZExtValue();
        }
        if (auto dea = llvm::dyn_cast<mlir::DenseElementsAttr>(c.getValue())) {
            if (dea.isSplat() && mlir::isa<mlir::IntegerType>(dea.getElementType())) {
                return dea.getSplatValue<llvm::APInt>().getZExtValue();
            }
        }
    }
    return std::nullopt;
}

static bool isUndefOrPoison(mlir::Value v) {
    return v && (v.getDefiningOp<mlir::LLVM::UndefOp>() ||
                 v.getDefiningOp<mlir::LLVM::PoisonOp>());
}

static bool extractConstStructInts(mlir::Value v, llvm::SmallVectorImpl<uint64_t>& out) {
    if (!v) return false;
    auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(v.getType());
    if (!st) return false;
    out.assign(st.getBody().size(), 0);

    llvm::SmallVector<std::pair<size_t, mlir::Value>, 8> inserts;
    mlir::Value cur = v;
    while (auto iv = cur.getDefiningOp<mlir::LLVM::InsertValueOp>()) {
        auto pos = iv.getPosition();
        if (pos.size() == 1 && static_cast<size_t>(pos[0]) < out.size()) {
            inserts.push_back({static_cast<size_t>(pos[0]), iv.getValue()});
        }
        cur = iv.getContainer();
    }

    bool recognized = !inserts.empty();
    if (cur.getDefiningOp<mlir::LLVM::ZeroOp>() || isUndefOrPoison(cur)) {
        recognized = true;
    } else if (auto cst = cur.getDefiningOp<mlir::LLVM::ConstantOp>()) {
        if (auto arr = llvm::dyn_cast<mlir::ArrayAttr>(cst.getValue())) {
            recognized = true;
            for (size_t i = 0; i < arr.size() && i < out.size(); ++i) {
                if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(arr[i])) {
                    out[i] = ia.getValue().getZExtValue();
                }
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

static bool isBarrierOp(mlir::Operation* op) {
    auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(op);
    if (!call) return false;
    auto callee = call.getCallee();
    if (!callee || !callee->starts_with("dx.op.barrier")) return false;
    if (call.getNumOperands() == 0) return true;
    uint64_t opNum = getConstIntValue(call.getOperand(0)).value_or(80);
    if (opNum == 80 && call.getNumOperands() > 1) {
        uint64_t mode = getConstIntValue(call.getOperand(1)).value_or(1);
        return (mode & 1u) != 0; // DXIL::BarrierMode::SYNC_THREAD_GROUP
    }
    if ((opNum == 244 || opNum == 245) && call.getNumOperands() > 2) {
        uint64_t sem = getConstIntValue(call.getOperand(2)).value_or(1);
        return (sem & 1u) != 0; // DXIL::BarrierSemanticFlags::GroupSync
    }
    return true;
}

static bool opContainsBarrier(mlir::Operation* op) {
    if (isBarrierOp(op)) return true;
    for (mlir::Region& r : op->getRegions()) {
        for (mlir::Block& bb : r) {
            for (mlir::Operation& child : bb) {
                if (opContainsBarrier(&child)) return true;
            }
        }
    }
    return false;
}

static bool regionContainsBarrier(mlir::Region& r) {
    for (mlir::Block& bb : r) {
        for (mlir::Operation& op : bb) {
            if (opContainsBarrier(&op)) return true;
        }
    }
    return false;
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

static mlir::LLVM::ICmpPredicate toLLVMIcmpPred(mlir::arith::CmpIPredicate p) {
    using AP = mlir::arith::CmpIPredicate;
    using LP = mlir::LLVM::ICmpPredicate;
    switch (p) {
    case AP::eq: return LP::eq;
    case AP::ne: return LP::ne;
    case AP::slt: return LP::slt;
    case AP::sle: return LP::sle;
    case AP::sgt: return LP::sgt;
    case AP::sge: return LP::sge;
    case AP::ult: return LP::ult;
    case AP::ule: return LP::ule;
    case AP::ugt: return LP::ugt;
    case AP::uge: return LP::uge;
    }
    return LP::eq;
}

class SimdPacketizer {
public:
    struct BufferDescInfo {
        mlir::Value descP;
        mlir::Value basePtr;
        mlir::Value byteSize;
        mlir::Value isReserved;
        mlir::Value okPtr;
        mlir::Value elemStride;
        mlir::Value stride64;
    };

    struct DescInfoCacheKey {
        mlir::Value descPtr;
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
        mlir::OpBuilder& builder,
        mlir::ModuleOp mod,
        const DxilLoadedShader& shader,
        const RootSignatureLayout& rootLayout,
        uint32_t simdWidth,
        mlir::Value rootCtxPtr,
        StageHooks& hooks,
        DescInfoCache* sharedDescCache = nullptr)
        : b_(builder),
          ctx_(builder.getContext()),
          mod_(mod),
          shader_(shader),
          rootLayout_(rootLayout),
          W_(simdWidth),
          rootCtxPtr_(rootCtxPtr),
          hooks_(hooks),
          descCache_(sharedDescCache ? sharedDescCache : &ownedDescCache_) {
        const char* strictFpEnv = std::getenv("NACELLE_STRICT_FP");
        strictFp_ = strictFpEnv && strictFpEnv[0] == '1';
    }

    mlir::OpBuilder& builder() { return b_; }
    uint32_t width() const { return W_; }

    mlir::Type ptrTy() const { return mlir::LLVM::LLVMPointerType::get(ctx_); }
    mlir::Type i1Ty() const { return b_.getI1Type(); }
    mlir::Type i8Ty() const { return b_.getI8Type(); }
    mlir::Type i32Ty() const { return b_.getI32Type(); }
    mlir::Type i64Ty() const { return b_.getI64Type(); }
    mlir::Type f32Ty() const { return b_.getF32Type(); }
    mlir::VectorType vecTy(mlir::Type elem) const { return mlir::VectorType::get({W_}, elem); }

    mlir::Value cInt(mlir::Location loc, mlir::Type ty, uint64_t val) {
        return mlir::LLVM::ConstantOp::create(b_, loc, ty, b_.getIntegerAttr(ty, val));
    }
    mlir::Value cFlt(mlir::Location loc, mlir::Type ty, double val) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return mlir::LLVM::ConstantOp::create(b_, loc, ft, b_.getFloatAttr(ft, val));
    }
    mlir::Value splatInt(mlir::Location loc, mlir::Type elemTy, uint64_t val) {
        auto vt = vecTy(elemTy);
        return mlir::LLVM::ConstantOp::create(
            b_, loc, vt, mlir::DenseElementsAttr::get(vt, b_.getIntegerAttr(elemTy, val)));
    }
    mlir::Value splatFlt(mlir::Location loc, mlir::Type elemTy, double val) {
        auto vt = vecTy(elemTy);
        return mlir::LLVM::ConstantOp::create(
            b_, loc, vt, mlir::DenseElementsAttr::get(vt, b_.getFloatAttr(elemTy, val)));
    }
    mlir::Value laneIota(mlir::Location loc) {
        llvm::SmallVector<int32_t, 16> vals;
        for (uint32_t i = 0; i < W_; ++i) vals.push_back(static_cast<int32_t>(i));
        auto vt = vecTy(i32Ty());
        return mlir::LLVM::ConstantOp::create(
            b_, loc, vt, mlir::DenseElementsAttr::get(vt, llvm::ArrayRef<int32_t>(vals)));
    }
    mlir::Value llvmExtract(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value vec, uint32_t idx) {
        return ::nacelle::llvmExtract(builder, loc, vec, idx);
    }
    mlir::Value llvmExtract(mlir::Location loc, mlir::Value vec, uint32_t idx) {
        return ::nacelle::llvmExtract(b_, loc, vec, idx);
    }
    mlir::Value llvmInsert(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value vec, mlir::Value elem, uint32_t idx) {
        return ::nacelle::llvmInsert(builder, loc, vec, elem, idx);
    }
    mlir::Value llvmInsert(mlir::Location loc, mlir::Value vec, mlir::Value elem, uint32_t idx) {
        return ::nacelle::llvmInsert(b_, loc, vec, elem, idx);
    }
    mlir::Value llvmSplat(mlir::OpBuilder& builder, mlir::Location loc, mlir::Type elemTy, mlir::Value scalar) {
        return ::nacelle::llvmSplat(builder, loc, elemTy, scalar, W_);
    }
    mlir::Value llvmSplat(mlir::Location loc, mlir::Type elemTy, mlir::Value scalar) {
        return ::nacelle::llvmSplat(b_, loc, elemTy, scalar, W_);
    }
    mlir::LLVM::FastmathFlags fpFmf() const {
        if (strictFp_) return mlir::LLVM::FastmathFlags::none;
        return mlir::LLVM::FastmathFlags::reassoc | mlir::LLVM::FastmathFlags::nsz |
               mlir::LLVM::FastmathFlags::arcp | mlir::LLVM::FastmathFlags::contract |
               mlir::LLVM::FastmathFlags::afn;
    }
    template <typename OpT, typename... Args>
    mlir::Value createFp(mlir::OpBuilder& builder, mlir::Location loc, Args&&... args) {
        auto op = OpT::create(builder, loc, std::forward<Args>(args)...);
        if (!strictFp_) {
            op.setFastmathFlagsAttr(mlir::LLVM::FastmathFlagsAttr::get(ctx_, fpFmf()));
        }
        return op.getResult();
    }
    template <typename OpT, typename... Args>
    mlir::Value createFp(mlir::Location loc, Args&&... args) {
        return createFp<OpT>(b_, loc, std::forward<Args>(args)...);
    }

    mlir::Value emitMathUnary(mlir::Location loc, llvm::StringRef baseName, mlir::Value vecArg) {
        auto vt = llvm::cast<mlir::VectorType>(vecArg.getType());
        mlir::Type elemTy = vt.getElementType();
        if (elemTy.isF16()) {
            mlir::Value f32Arg = mlir::LLVM::FPExtOp::create(b_, loc, vecTy(f32Ty()), vecArg);
            mlir::Value f32Res = emitMathUnary(loc, baseName, f32Arg);
            return mlir::LLVM::FPTruncOp::create(b_, loc, vt, f32Res);
        }
        if (!strictFp_ && elemTy.isF32() && (W_ == 4 || W_ == 8 || W_ == 16)) {
            std::string fnName = ("nacelle_math_v" + std::to_string(W_) + "_" + baseName + "f").str();
            mlir::func::FuncOp callee = mod_.lookupSymbol<mlir::func::FuncOp>(fnName);
            if (!callee) {
                mlir::OpBuilder mb(ctx_);
                mb.setInsertionPointToStart(mod_.getBody());
                auto fnTy = mb.getFunctionType({vt}, {vt});
                callee = mlir::func::FuncOp::create(mb, mod_.getLoc(), fnName, fnTy);
                callee.setPrivate();
            }
            return mlir::func::CallOp::create(b_, loc, callee, mlir::ValueRange{vecArg}).getResult(0);
        }
        if (baseName == "sin") return createFp<mlir::LLVM::SinOp>(loc, vt, vecArg);
        if (baseName == "cos") return createFp<mlir::LLVM::CosOp>(loc, vt, vecArg);
        if (baseName == "exp2") return createFp<mlir::LLVM::Exp2Op>(loc, vt, vecArg);
        if (baseName == "log2") return createFp<mlir::LLVM::Log2Op>(loc, vt, vecArg);
        // Scalar libm fallback for strict-FP or non-f32 transcendental ops.
        std::string libmName = (elemTy.isF32() ? (baseName + "f") : baseName).str();
        mlir::LLVM::LLVMFuncOp libmFn = getOrDeclareLibmFn(loc, libmName, elemTy);
        mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vt);
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value x = llvmExtract(loc, vecArg, l);
            mlir::Value r = mlir::LLVM::CallOp::create(b_, loc, libmFn, mlir::ValueRange{x}).getResult();
            outVec = llvmInsert(loc, outVec, r, l);
        }
        return outVec;
    }

    mlir::LLVM::LLVMFuncOp getOrDeclareLibmFn(mlir::Location loc, llvm::StringRef name, mlir::Type ty) {
        if (auto existing = mod_.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) return existing;
        mlir::OpBuilder mb(ctx_);
        mb.setInsertionPointToStart(mod_.getBody());
        auto fnTy = mlir::LLVM::LLVMFunctionType::get(ty, {ty});
        return mlir::LLVM::LLVMFuncOp::create(mb, loc, name, fnTy);
    }

    mlir::Value orReduce(mlir::Location loc, mlir::Value vecI1) {
        return mlir::LLVM::vector_reduce_or::create(b_, loc, i1Ty(), vecI1);
    }

    mlir::Value allTrueMask(mlir::Location loc) { return splatInt(loc, i1Ty(), 1); }
    mlir::Value allFalseMask(mlir::Location loc) { return splatInt(loc, i1Ty(), 0); }

    bool isMaskConstAllTrue(mlir::Value maskVec) const {
        if (!maskVec) return false;
        if (auto cst = maskVec.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(cst.getValue())) {
                return dense.getElementType().isInteger(1) && dense.isSplat() && dense.getSplatValue<bool>();
            }
        }
        return false;
    }

    bool isLaneMaskConstFalse(mlir::Value maskVec, uint32_t lane, uint32_t depth = 0) const {
        if (!maskVec || depth > 8) return false;
        if (auto cst = maskVec.getDefiningOp<mlir::arith::ConstantOp>()) {
            if (auto dense = llvm::dyn_cast<mlir::DenseElementsAttr>(cst.getValue())) {
                if (!dense.getElementType().isInteger(1)) return false;
                if (dense.isSplat()) return !dense.getSplatValue<bool>();
                if (static_cast<int64_t>(lane) < dense.getNumElements()) {
                    auto it = dense.value_begin<bool>();
                    std::advance(it, lane);
                    return !(*it);
                }
            }
            return false;
        }
        if (auto andOp = maskVec.getDefiningOp<mlir::arith::AndIOp>()) {
            return isLaneMaskConstFalse(andOp.getLhs(), lane, depth + 1) ||
                   isLaneMaskConstFalse(andOp.getRhs(), lane, depth + 1);
        }
        if (auto orOp = maskVec.getDefiningOp<mlir::arith::OrIOp>()) {
            return isLaneMaskConstFalse(orOp.getLhs(), lane, depth + 1) &&
                   isLaneMaskConstFalse(orOp.getRhs(), lane, depth + 1);
        }
        if (auto res = llvm::dyn_cast<mlir::OpResult>(maskVec)) {
            if (auto w = llvm::dyn_cast<mlir::scf::WhileOp>(res.getOwner())) {
                if (res.getResultNumber() == 1 && w.getInits().size() >= 2) {
                    return isLaneMaskConstFalse(w.getInits()[1], lane, depth + 1);
                }
            }
            if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(res.getOwner())) {
                if (!ifOp.getThenRegion().empty() && !ifOp.getElseRegion().empty()) {
                    auto ty = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
                    auto ey = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
                    unsigned rn = res.getResultNumber();
                    if (ty && ey && rn < ty.getNumOperands() && rn < ey.getNumOperands()) {
                        return isLaneMaskConstFalse(ty.getOperand(rn), lane, depth + 1) &&
                               isLaneMaskConstFalse(ey.getOperand(rn), lane, depth + 1);
                    }
                }
            }
        }
        if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(maskVec)) {
            if (mlir::Block* blk = barg.getOwner()) {
                if (auto w = llvm::dyn_cast_or_null<mlir::scf::WhileOp>(blk->getParentOp())) {
                    if (w.getInits().size() >= 2) {
                        if (barg.getArgNumber() == 1) {
                            return isLaneMaskConstFalse(w.getInits()[1], lane, depth + 1);
                        }
                        if (barg.getArgNumber() == 0) {
                            if (hooks_.isPixelShader && shader_.hasDerivativesOrImplicitSample) {
                                uint32_t qBase = lane & ~3u;
                                return isLaneMaskConstFalse(w.getInits()[0], qBase, depth + 1) &&
                                       isLaneMaskConstFalse(w.getInits()[0], qBase + 1, depth + 1) &&
                                       isLaneMaskConstFalse(w.getInits()[0], qBase + 2, depth + 1) &&
                                       isLaneMaskConstFalse(w.getInits()[0], qBase + 3, depth + 1);
                            }
                            return isLaneMaskConstFalse(w.getInits()[0], lane, depth + 1);
                        }
                    }
                }
            }
        }
        return false;
    }

    mlir::Value expandMaskQuads(mlir::Location loc, mlir::Value mask) {
        // For quad q (4 lanes: 4q..4q+3), any active lane makes all 4 lanes execute in execMask.
        llvm::SmallVector<int32_t, 16> s1, s2;
        for (uint32_t l = 0; l < W_; ++l) {
            s1.push_back(static_cast<int32_t>(l ^ 1u));
            s2.push_back(static_cast<int32_t>(l ^ 2u));
        }
        mlir::Value pair = mlir::LLVM::OrOp::create(
            b_, loc, mask, mlir::LLVM::ShuffleVectorOp::create(b_, loc, mask, mask, s1));
        return mlir::LLVM::OrOp::create(
            b_, loc, pair, mlir::LLVM::ShuffleVectorOp::create(b_, loc, pair, pair, s2));
    }

    SimdVal makeZeroSimd(mlir::Location loc, mlir::Type origTy) {
        if (mlir::isa<mlir::IntegerType>(origTy)) {
            SimdVal sv{SimdVal::Kind::Vector, origTy, splatInt(loc, origTy, 0), {}, {}};
            sv.pattern = SimdVal::LanePattern::Uniform;
            return sv;
        }
        if (mlir::isa<mlir::FloatType>(origTy)) {
            SimdVal sv{SimdVal::Kind::Vector, origTy, splatFlt(loc, origTy, 0.0), {}, {}};
            sv.pattern = SimdVal::LanePattern::Uniform;
            return sv;
        }
        if (mlir::isa<mlir::LLVM::LLVMPointerType>(origTy)) {
            auto zeroPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = origTy;
            sv.lanes.push_back(zeroPtr);
            return sv;
        }
        if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (mlir::Type fTy : st.getBody()) {
                sv.fields.push_back(makeZeroSimd(loc, fTy));
            }
            return sv;
        }
        if (auto at = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < at.getNumElements(); ++i) {
                sv.fields.push_back(makeZeroSimd(loc, at.getElementType()));
            }
            return sv;
        }
        if (auto vt = llvm::dyn_cast<mlir::VectorType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                sv.fields.push_back(makeZeroSimd(loc, vt.getElementType()));
            }
            return sv;
        }
        if (origTy.isIndex()) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = origTy;
            sv.lanes.push_back(mlir::arith::ConstantIndexOp::create(b_, loc, 0));
            return sv;
        }
        return {SimdVal::Kind::Vector, origTy, splatInt(loc, i32Ty(), 0), {}, {}};
    }

    SimdVal getMapped(mlir::Location loc, mlir::Value orig) {
        if (!orig) return {};
        auto it = valueMap_.find(orig);
        if (it != valueMap_.end()) return it->second;

        if (isUndefOrPoison(orig)) {
            return makeZeroSimd(loc, orig.getType());
        }
        if (auto c = orig.getDefiningOp<mlir::LLVM::ConstantOp>()) {
            mlir::Type ty = orig.getType();
            if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
                SimdVal sv{SimdVal::Kind::Vector, ty, splatInt(loc, ty, ia.getValue().getZExtValue()), {}, {}};
                sv.pattern = SimdVal::LanePattern::Uniform;
                return sv;
            }
            if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(c.getValue())) {
                SimdVal sv{SimdVal::Kind::Vector, ty, splatFlt(loc, ty, fa.getValueAsDouble()), {}, {}};
                sv.pattern = SimdVal::LanePattern::Uniform;
                return sv;
            }
            if (auto da = llvm::dyn_cast<mlir::DenseElementsAttr>(c.getValue())) {
                if (auto vt = llvm::dyn_cast<mlir::VectorType>(ty)) {
                    SimdVal sv;
                    sv.kind = SimdVal::Kind::Struct;
                    sv.origType = ty;
                    mlir::Type et = vt.getElementType();
                    if (mlir::isa<mlir::IntegerType>(et)) {
                        auto vals = da.getValues<llvm::APInt>();
                        auto itV = vals.begin();
                        for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                            uint64_t v = da.isSplat() ? (*vals.begin()).getZExtValue() : (*itV++).getZExtValue();
                            sv.fields.push_back({SimdVal::Kind::Vector, et, splatInt(loc, et, v), {}, {}});
                        }
                    } else if (mlir::isa<mlir::FloatType>(et)) {
                        auto vals = da.getValues<llvm::APFloat>();
                        auto itV = vals.begin();
                        for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                            double v = da.isSplat() ? (*vals.begin()).convertToDouble() : (*itV++).convertToDouble();
                            sv.fields.push_back({SimdVal::Kind::Vector, et, splatFlt(loc, et, v), {}, {}});
                        }
                    }
                    return sv;
                }
            }
        }
        if (auto c = orig.getDefiningOp<mlir::arith::ConstantOp>()) {
            mlir::Type ty = orig.getType();
            if (auto ia = llvm::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
                if (ty.isIndex()) {
                    SimdVal sv;
                    sv.kind = SimdVal::Kind::Lanes;
                    sv.origType = ty;
                    sv.lanes.push_back(mlir::arith::ConstantIndexOp::create(b_, loc, ia.getValue().getSExtValue()));
                    return sv;
                }
                SimdVal sv{SimdVal::Kind::Vector, ty, splatInt(loc, ty, ia.getValue().getZExtValue()), {}, {}};
                sv.pattern = SimdVal::LanePattern::Uniform;
                return sv;
            }
            if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(c.getValue())) {
                SimdVal sv{SimdVal::Kind::Vector, ty, splatFlt(loc, ty, fa.getValueAsDouble()), {}, {}};
                sv.pattern = SimdVal::LanePattern::Uniform;
                return sv;
            }
            if (auto da = llvm::dyn_cast<mlir::DenseElementsAttr>(c.getValue())) {
                if (auto vt = llvm::dyn_cast<mlir::VectorType>(ty)) {
                    SimdVal sv;
                    sv.kind = SimdVal::Kind::Struct;
                    sv.origType = ty;
                    mlir::Type et = vt.getElementType();
                    if (mlir::isa<mlir::IntegerType>(et)) {
                        auto vals = da.getValues<llvm::APInt>();
                        auto itV = vals.begin();
                        for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                            uint64_t v = da.isSplat() ? (*vals.begin()).getZExtValue() : (*itV++).getZExtValue();
                            sv.fields.push_back({SimdVal::Kind::Vector, et, splatInt(loc, et, v), {}, {}});
                        }
                    } else if (mlir::isa<mlir::FloatType>(et)) {
                        auto vals = da.getValues<llvm::APFloat>();
                        auto itV = vals.begin();
                        for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                            double v = da.isSplat() ? (*vals.begin()).convertToDouble() : (*itV++).convertToDouble();
                            sv.fields.push_back({SimdVal::Kind::Vector, et, splatFlt(loc, et, v), {}, {}});
                        }
                    }
                    return sv;
                }
            }
        }
        if (orig.getDefiningOp<mlir::LLVM::ZeroOp>()) {
            return makeZeroSimd(loc, orig.getType());
        }
        if (auto addrOp = orig.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
            mlir::Value newAddr = mlir::LLVM::AddressOfOp::create(
                b_, loc, addrOp.getRes().getType(), addrOp.getGlobalName());
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = addrOp.getRes().getType();
            sv.lanes.push_back(newAddr);
            return sv;
        }
        return makeZeroSimd(loc, orig.getType());
    }

    void setMapped(mlir::Value orig, const SimdVal& sv) { valueMap_[orig] = sv; }

    mlir::Value asVec(mlir::Location loc, const SimdVal& sv) {
        if (sv.kind == SimdVal::Kind::Vector) return sv.vec;
        if (sv.kind == SimdVal::Kind::Lanes && !sv.lanes.empty()) {
            if (sv.lanes.size() == 1) {
                return llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(sv.lanes[0].getType())).getElementType(), sv.lanes[0]);
            }
            mlir::Type elemTy = sv.lanes[0].getType();
            mlir::Value v = mlir::LLVM::isCompatibleType(elemTy)
                                ? mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy)))
                                : mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy)));
            for (uint32_t l = 0; l < W_; ++l) {
                v = llvmInsert(b_, loc, v, sv.lanes[l], static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            return v;
        }
        return nullptr;
    }

    mlir::Value laneValue(mlir::Location loc, const SimdVal& sv, uint32_t l) {
        if (sv.kind == SimdVal::Kind::Vector) {
            return llvmExtract(b_, loc, sv.vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
        }
        if (sv.kind == SimdVal::Kind::Lanes && !sv.lanes.empty()) {
            return sv.lanes.size() == 1 ? sv.lanes[0] : sv.lanes[l];
        }
        if (sv.kind == SimdVal::Kind::Handle && !sv.lanes.empty()) {
            return sv.lanes.size() == 1 ? sv.lanes[0] : sv.lanes[l];
        }
        if (sv.kind == SimdVal::Kind::Struct && sv.fields.size() == 1) {
            return laneValue(loc, sv.fields[0], l);
        }
        return nullptr;
    }

    SimdVal selectVal(mlir::Location loc, mlir::Value condVec, const SimdVal& t, const SimdVal& f) {
        if (t.kind == SimdVal::Kind::Vector && f.kind == SimdVal::Kind::Vector) {
            mlir::Value s = mlir::LLVM::SelectOp::create(b_, loc, condVec, t.vec, f.vec);
            return {SimdVal::Kind::Vector, t.origType, s, {}, {}};
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
                mlir::Value cL = llvmExtract(b_, loc, condVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value tL = laneValue(loc, t, l);
                mlir::Value fL = laneValue(loc, f, l);
                out.lanes.push_back(mlir::LLVM::SelectOp::create(b_, loc, cL, tL, fL));
            }
            return out;
        }
        if (t.kind == SimdVal::Kind::Struct && f.kind == SimdVal::Kind::Struct &&
            t.fields.size() == f.fields.size()) {
            SimdVal out = t;
            out.fields.clear();
            for (size_t i = 0; i < t.fields.size(); ++i) {
                out.fields.push_back(selectVal(loc, condVec, t.fields[i], f.fields[i]));
            }
            return out;
        }
        return t;
    }

    /// Ensures a SimdVal only contains Vector/Lanes/Handle values with unambiguous flat storage.
    void flattenSimdTypes(mlir::Type origTy, llvm::SmallVectorImpl<mlir::Type>& outVecTypes) {
        if (mlir::isa<mlir::IntegerType, mlir::FloatType>(origTy)) {
            outVecTypes.push_back(vecTy(origTy));
        } else if (origTy.isIndex()) {
            outVecTypes.push_back(vecTy(i64Ty()));
        } else if (mlir::isa<mlir::LLVM::LLVMPointerType>(origTy)) {
            outVecTypes.push_back(vecTy(ptrTy()));
        } else if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(origTy)) {
            for (mlir::Type f : st.getBody()) flattenSimdTypes(f, outVecTypes);
        } else if (auto at = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(origTy)) {
            for (unsigned i = 0; i < at.getNumElements(); ++i) flattenSimdTypes(at.getElementType(), outVecTypes);
        } else {
            outVecTypes.push_back(vecTy(i32Ty()));
        }
    }

    void packSimdToValues(mlir::Location loc, const SimdVal& sv, llvm::SmallVectorImpl<mlir::Value>& out) {
        if (sv.kind == SimdVal::Kind::Vector) {
            out.push_back(sv.vec);
        } else if (sv.origType && sv.origType.isIndex()) {
            mlir::Value vec64 = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i64Ty()));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value idxL = laneValue(loc, sv, l);
                mlir::Value i64L = idxL ? mlir::Value(mlir::arith::IndexCastOp::create(b_, loc, i64Ty(), idxL))
                                        : cInt(loc, i64Ty(), 0);
                vec64 = llvmInsert(b_, loc, vec64, i64L, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            out.push_back(vec64);
        } else if (sv.kind == SimdVal::Kind::Lanes || sv.kind == SimdVal::Kind::Handle) {
            if (sv.isUniform()) {
                out.push_back(llvmSplat(loc, ptrTy(), laneValue(loc, sv, 0)));
            } else {
                mlir::Value vec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(ptrTy()));
                for (uint32_t l = 0; l < W_; ++l) {
                    vec = llvmInsert(loc, vec, laneValue(loc, sv, l), l);
                }
                out.push_back(vec);
            }
        } else if (sv.kind == SimdVal::Kind::Struct) {
            for (const auto& f : sv.fields) packSimdToValues(loc, f, out);
        } else {
            out.push_back(splatInt(loc, i32Ty(), 0));
        }
    }

    SimdVal unpackSimdFromValues(mlir::Location loc, mlir::Type origTy, mlir::ValueRange vals, size_t& cursor, const SimdVal* proto = nullptr) {
        if (proto && proto->kind == SimdVal::Kind::Handle) {
            SimdVal h = *proto;
            h.lanes.clear();
            mlir::Value ptrVec = vals[cursor++];
            if (h.isUniform()) {
                mlir::Value p0 = llvmExtract(loc, ptrVec, 0);
                h.lanes.assign(W_, p0);
            } else {
                for (uint32_t l = 0; l < W_; ++l) {
                    h.lanes.push_back(llvmExtract(loc, ptrVec, l));
                }
            }
            return h;
        }
        if (mlir::isa<mlir::IntegerType, mlir::FloatType>(origTy)) {
            return {SimdVal::Kind::Vector, origTy, vals[cursor++], {}, {}};
        }
        if (origTy.isIndex()) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = origTy;
            mlir::Value vec64 = vals[cursor++];
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value i64L = llvmExtract(b_, loc, vec64, static_cast<uint32_t>(static_cast<int64_t>(l)));
                sv.lanes.push_back(mlir::arith::IndexCastOp::create(b_, loc, origTy, i64L));
            }
            return sv;
        }
        if (mlir::isa<mlir::LLVM::LLVMPointerType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = origTy;
            mlir::Value ptrVec = vals[cursor++];
            for (uint32_t l = 0; l < W_; ++l) {
                sv.lanes.push_back(llvmExtract(loc, ptrVec, l));
            }
            return sv;
        }
        if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < st.getBody().size(); ++i) {
                const SimdVal* fp = (proto && i < proto->fields.size()) ? &proto->fields[i] : nullptr;
                sv.fields.push_back(unpackSimdFromValues(loc, st.getBody()[i], vals, cursor, fp));
            }
            return sv;
        }
        if (auto at = llvm::dyn_cast<mlir::LLVM::LLVMArrayType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Struct;
            sv.origType = origTy;
            for (unsigned i = 0; i < at.getNumElements(); ++i) {
                const SimdVal* fp = (proto && i < proto->fields.size()) ? &proto->fields[i] : nullptr;
                sv.fields.push_back(unpackSimdFromValues(loc, at.getElementType(), vals, cursor, fp));
            }
            return sv;
        }
        return {SimdVal::Kind::Vector, origTy, vals[cursor++], {}, {}};
    }

    bool packetizeBlockRange(
        mlir::Block::iterator beginIt,
        mlir::Block::iterator endIt,
        mlir::Value execMask,
        mlir::Value& realMask,
        std::string& outError) {
        for (auto it = beginIt; it != endIt; ++it) {
            mlir::Operation& op = *it;
            if (op.hasTrait<mlir::OpTrait::IsTerminator>()) break;
            if (!packetizeOp(op, execMask, realMask, outError)) return false;
        }
        return true;
    }

    bool packetizeBlock(mlir::Block& srcBlock, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        return packetizeBlockRange(srcBlock.begin(), srcBlock.end(), execMask, realMask, outError);
    }

private:
    mlir::Value gepByteOffset(mlir::Location loc, mlir::Value basePtr, int64_t byteOff) {
        if (byteOff == 0) return basePtr;
        return mlir::LLVM::GEPOp::create(
            b_, loc, ptrTy(), i8Ty(), basePtr,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{byteOff});
    }

    mlir::Value gepDynByteOffset(mlir::Location loc, mlir::Value basePtr, mlir::Value byteOff) {
        return mlir::LLVM::GEPOp::create(
            b_, loc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{byteOff});
    }

    mlir::Value loadScalar(mlir::Location loc, mlir::Type ty, mlir::Value ptr) {
        return mlir::LLVM::LoadOp::create(b_, loc, ty, ptr);
    }
    mlir::Value loadInvariantScalar(mlir::Location loc, mlir::Type ty, mlir::Value ptr) {
        auto op = mlir::LLVM::LoadOp::create(b_, loc, ty, ptr);
        op.setInvariant(true);
        return op;
    }

    void storeScalar(mlir::Location loc, mlir::Value val, mlir::Value ptr) {
        mlir::LLVM::StoreOp::create(b_, loc, val, ptr);
    }

    mlir::Value createEntryAlloca(mlir::Location loc, mlir::Type elemTy, uint64_t count, unsigned align = 64) {
        mlir::Block* entryBlk = hooks_.entryBlock;
        if (!entryBlk) {
            mlir::Block* curBlk = b_.getInsertionBlock();
            mlir::Operation* parentOp = curBlk ? curBlk->getParentOp() : nullptr;
            while (parentOp && !llvm::isa<mlir::func::FuncOp>(parentOp)) {
                parentOp = parentOp->getParentOp();
            }
            if (auto fnOp = llvm::dyn_cast_or_null<mlir::func::FuncOp>(parentOp)) {
                entryBlk = &fnOp.getBody().front();
            }
        }
        if (entryBlk) {
            mlir::OpBuilder eb(entryBlk, entryBlk->begin());
            mlir::Value sz = mlir::LLVM::ConstantOp::create(eb, loc, eb.getI32Type(), eb.getIntegerAttr(eb.getI32Type(), count));
            return mlir::LLVM::AllocaOp::create(eb, loc, ptrTy(), elemTy, sz, align);
        }
        return mlir::LLVM::AllocaOp::create(b_, loc, ptrTy(), elemTy, cInt(loc, i32Ty(), count), align);
    }

    mlir::Value getOrCreateScratchAlloca(mlir::Location loc) {
        if (!scratchAlloca_) {
            // 16-byte aligned scratch slot in entry block for masked-off alloca lanes.
            scratchAlloca_ = createEntryAlloca(loc, i64Ty(), 2, 16);
        }
        return scratchAlloca_;
    }

    mlir::LLVM::LLVMFuncOp getOrDeclareRuntimeFn(
        mlir::Location loc, llvm::StringRef name, mlir::Type retTy, llvm::ArrayRef<mlir::Type> argTys) {
        if (auto existing = mod_.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
            return existing;
        }
        mlir::OpBuilder::InsertionGuard g(b_);
        b_.setInsertionPointToStart(mod_.getBody());
        auto fnTy = mlir::LLVM::LLVMFunctionType::get(
            retTy ? retTy : mlir::LLVM::LLVMVoidType::get(ctx_), argTys, false);
        return mlir::LLVM::LLVMFuncOp::create(b_, loc, name, fnTy);
    }

    bool packetizeOp(mlir::Operation& op, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = op.getLoc();

        // Constants / Undef / Poison / Casts
        if (llvm::isa<mlir::LLVM::ConstantOp, mlir::arith::ConstantOp,
                      mlir::LLVM::UndefOp, mlir::LLVM::PoisonOp,
                      mlir::LLVM::ZeroOp>(&op) ||
            op.getName().getStringRef() == "ub.poison") {
            setMapped(op.getResult(0), getMapped(loc, op.getResult(0)));
            return true;
        }
        if (auto addrOp = llvm::dyn_cast<mlir::LLVM::AddressOfOp>(&op)) {
            mlir::Value newAddr = mlir::LLVM::AddressOfOp::create(
                b_, loc, addrOp.getRes().getType(), addrOp.getGlobalName());
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = addrOp.getRes().getType();
            sv.lanes.push_back(newAddr);
            setMapped(addrOp.getRes(), sv);
            return true;
        }
        if (auto frz = llvm::dyn_cast<mlir::LLVM::FreezeOp>(&op)) {
            setMapped(frz.getRes(), getMapped(loc, frz.getVal()));
            return true;
        }

        // Control Flow: scf.if
        if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(&op)) {
            return packetizeScfIf(ifOp, execMask, realMask, outError);
        }
        // Control Flow: scf.index_switch
        if (auto swOp = llvm::dyn_cast<mlir::scf::IndexSwitchOp>(&op)) {
            return packetizeScfIndexSwitch(swOp, execMask, realMask, outError);
        }
        // Control Flow: scf.while
        if (auto whileOp = llvm::dyn_cast<mlir::scf::WhileOp>(&op)) {
            return packetizeScfWhile(whileOp, execMask, realMask, outError);
        }
        // Control Flow: scf.for (if canonicalizer produced one)
        if (auto forOp = llvm::dyn_cast<mlir::scf::ForOp>(&op)) {
            return packetizeScfFor(forOp, execMask, realMask, outError);
        }

        // Calls
        if (auto call = llvm::dyn_cast<mlir::LLVM::CallOp>(&op)) {
            return packetizeCall(call, execMask, realMask, outError);
        }

        // Vector element insert/extract/shuffle (DXIL long vectors)
        if (auto ee = llvm::dyn_cast<mlir::LLVM::ExtractElementOp>(&op)) {
            SimdVal vec = getMapped(loc, ee.getVector());
            if (vec.kind != SimdVal::Kind::Struct || vec.fields.empty()) {
                setMapped(ee.getRes(), makeZeroSimd(loc, ee.getType()));
                return true;
            }
            if (auto idxC = getConstIntValue(ee.getPosition())) {
                size_t idx = static_cast<size_t>(*idxC);
                setMapped(ee.getRes(), idx < vec.fields.size() ? vec.fields[idx] : makeZeroSimd(loc, ee.getType()));
                return true;
            }
            mlir::Value idxVec = asVec(loc, getMapped(loc, ee.getPosition()));
            mlir::Type idxTy = llvm::cast<mlir::VectorType>(idxVec.getType()).getElementType();
            mlir::Type elemTy = ee.getType();
            mlir::Value res = asVec(loc, vec.fields[0]);
            for (size_t i = 1; i < vec.fields.size(); ++i) {
                mlir::Value eq = mlir::LLVM::ICmpOp::create(
                    b_, loc, mlir::LLVM::ICmpPredicate::eq, idxVec, splatInt(loc, idxTy, i));
                res = mlir::LLVM::SelectOp::create(b_, loc, eq, asVec(loc, vec.fields[i]), res);
            }
            setMapped(ee.getRes(), {SimdVal::Kind::Vector, elemTy, res, {}, {}});
            return true;
        }
        if (auto ie = llvm::dyn_cast<mlir::LLVM::InsertElementOp>(&op)) {
            SimdVal vec = getMapped(loc, ie.getVector());
            if (vec.kind != SimdVal::Kind::Struct || vec.fields.empty()) {
                vec = makeZeroSimd(loc, ie.getType());
            }
            SimdVal val = getMapped(loc, ie.getValue());
            if (auto idxC = getConstIntValue(ie.getPosition())) {
                size_t idx = static_cast<size_t>(*idxC);
                if (idx < vec.fields.size()) vec.fields[idx] = val;
                setMapped(ie.getRes(), vec);
                return true;
            }
            mlir::Value idxVec = asVec(loc, getMapped(loc, ie.getPosition()));
            mlir::Type idxTy = llvm::cast<mlir::VectorType>(idxVec.getType()).getElementType();
            mlir::Value valVec = asVec(loc, val);
            for (size_t i = 0; i < vec.fields.size(); ++i) {
                mlir::Value eq = mlir::LLVM::ICmpOp::create(
                    b_, loc, mlir::LLVM::ICmpPredicate::eq, idxVec, splatInt(loc, idxTy, i));
                mlir::Value cur = asVec(loc, vec.fields[i]);
                vec.fields[i].vec = mlir::LLVM::SelectOp::create(b_, loc, eq, valVec, cur);
            }
            setMapped(ie.getRes(), vec);
            return true;
        }
        if (auto sv = llvm::dyn_cast<mlir::LLVM::ShuffleVectorOp>(&op)) {
            SimdVal v1 = getMapped(loc, sv.getV1());
            SimdVal v2 = getMapped(loc, sv.getV2());
            auto resVt = llvm::cast<mlir::VectorType>(sv.getType());
            mlir::Type elemTy = resVt.getElementType();
            size_t n1 = (v1.kind == SimdVal::Kind::Struct) ? v1.fields.size() : 0;
            size_t n2 = (v2.kind == SimdVal::Kind::Struct) ? v2.fields.size() : 0;
            SimdVal out;
            out.kind = SimdVal::Kind::Struct;
            out.origType = resVt;
            for (int32_t m : sv.getMask()) {
                if (m >= 0 && static_cast<size_t>(m) < n1) {
                    out.fields.push_back(v1.fields[static_cast<size_t>(m)]);
                } else if (m >= 0 && static_cast<size_t>(m) < n1 + n2) {
                    out.fields.push_back(v2.fields[static_cast<size_t>(m) - n1]);
                } else {
                    out.fields.push_back(makeZeroSimd(loc, elemTy));
                }
            }
            setMapped(sv.getRes(), out);
            return true;
        }

        // Aggregate insert/extract
        if (auto ev = llvm::dyn_cast<mlir::LLVM::ExtractValueOp>(&op)) {
            SimdVal agg = getMapped(loc, ev.getContainer());
            SimdVal cur = agg;
            for (int64_t idx : ev.getPosition()) {
                if (cur.kind == SimdVal::Kind::Struct && static_cast<size_t>(idx) < cur.fields.size()) {
                    SimdVal next = cur.fields[static_cast<size_t>(idx)];
                    cur = std::move(next);
                } else {
                    cur = makeZeroSimd(loc, ev.getType());
                    break;
                }
            }
            setMapped(ev.getRes(), cur);
            return true;
        }
        if (auto iv = llvm::dyn_cast<mlir::LLVM::InsertValueOp>(&op)) {
            SimdVal agg = getMapped(loc, iv.getContainer());
            SimdVal elem = getMapped(loc, iv.getValue());
            auto pos = iv.getPosition();
            std::function<void(SimdVal&, size_t)> setRec = [&](SimdVal& cur, size_t d) {
                if (cur.kind != SimdVal::Kind::Struct) return;
                size_t idx = static_cast<size_t>(pos[d]);
                if (idx >= cur.fields.size()) return;
                if (d + 1 == pos.size()) cur.fields[idx] = elem;
                else setRec(cur.fields[idx], d + 1);
            };
            setRec(agg, 0);
            setMapped(iv.getRes(), agg);
            return true;
        }

        // Stack Alloca / GEP / Load / Store / Atomic
        if (auto allocaOp = llvm::dyn_cast<mlir::LLVM::AllocaOp>(&op)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            sv.isAllocaPtr = true;
            mlir::Type elemTy = allocaOp.getElemType();
            uint64_t count = getConstIntValue(allocaOp.getArraySize()).value_or(1);
            mlir::Type slotTy = (count == 1) ? elemTy : mlir::Type(mlir::LLVM::LLVMArrayType::get(elemTy, count));
            uint32_t numWaves = std::max<uint32_t>(1u, hooks_.numWaves);
            mlir::Value baseAlloca = createEntryAlloca(loc, slotTy, static_cast<uint64_t>(numWaves) * W_, 64);
            if (hooks_.waveIdx && numWaves > 1u) {
                mlir::Value w32 = hooks_.waveIdx;
                mlir::Value waveBase = mlir::LLVM::MulOp::create(b_, loc, w32, cInt(loc, i32Ty(), W_));
                for (uint32_t l = 0; l < W_; ++l) {
                    mlir::Value slotIdx = mlir::LLVM::AddOp::create(b_, loc, waveBase, cInt(loc, i32Ty(), l));
                    sv.lanes.push_back(mlir::LLVM::GEPOp::create(
                        b_, loc, ptrTy(), slotTy, baseAlloca, mlir::ValueRange{slotIdx}));
                }
            } else {
                for (uint32_t l = 0; l < W_; ++l) {
                    sv.lanes.push_back(mlir::LLVM::GEPOp::create(
                        b_, loc, ptrTy(), slotTy, baseAlloca,
                        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(l)}));
                }
            }
            setMapped(allocaOp.getRes(), sv);
            return true;
        }
        if (auto gep = llvm::dyn_cast<mlir::LLVM::GEPOp>(&op)) {
            SimdVal base = getMapped(loc, gep.getBase());
            llvm::SmallVector<SimdVal, 4> idxVals;
            for (mlir::Value iv : gep.getDynamicIndices()) idxVals.push_back(getMapped(loc, iv));

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
            out.origType = gep.getType();
            out.isAllocaPtr = base.isAllocaPtr;
            if (uniform) {
                out.pattern = SimdVal::LanePattern::Uniform;
            } else if (base.isUniform() && idxVals.size() == 1 && idxVals[0].isUnitStride() &&
                       gep.getRawConstantIndices().size() == 1 &&
                       gep.getRawConstantIndices()[0] == mlir::LLVM::GEPOp::kDynamicIndex) {
                out.pattern = SimdVal::LanePattern::UnitStride;
            }
            auto rawConsts = gep.getRawConstantIndices();
            uint32_t lanesOut = uniform ? 1u : W_;
            for (uint32_t l = 0; l < lanesOut; ++l) {
                llvm::SmallVector<mlir::LLVM::GEPArg, 4> gepArgs;
                size_t dynIdx = 0;
                for (int32_t rc : rawConsts) {
                    if (rc == mlir::LLVM::GEPOp::kDynamicIndex) {
                        gepArgs.push_back(laneValue(loc, idxVals[dynIdx++], l));
                    } else {
                        gepArgs.push_back(rc);
                    }
                }
                mlir::Value lp = mlir::LLVM::GEPOp::create(
                    b_, loc, ptrTy(), gep.getElemType(), laneValue(loc, base, l), gepArgs,
                    gep.getNoWrapFlags());
                out.lanes.push_back(lp);
            }
            setMapped(gep.getRes(), out);
            return true;
        }
        if (auto loadOp = llvm::dyn_cast<mlir::LLVM::LoadOp>(&op)) {
            SimdVal addr = getMapped(loc, loadOp.getAddr());
            mlir::Type elemTy = loadOp.getResult().getType();
            uint64_t align = loadOp.getAlignment().value_or(1);
            if (addr.isUniform() && mlir::isa<mlir::IntegerType, mlir::FloatType>(elemTy)) {
                mlir::Value s = loadScalar(loc, elemTy, laneValue(loc, addr, 0));
                mlir::Value bvec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(elemTy)).getElementType(), s);
                SimdVal sv{SimdVal::Kind::Vector, elemTy, bvec, {}, {}};
                sv.pattern = SimdVal::LanePattern::Uniform;
                setMapped(loadOp.getResult(), std::move(sv));
                return true;
            }
            if (mlir::isa<mlir::IntegerType, mlir::FloatType>(elemTy)) {
                if (addr.isUnitStride() && !addr.isAllocaPtr && isMaskConstAllTrue(execMask)) {
                    mlir::Value v = mlir::LLVM::LoadOp::create(
                        b_, loc, vecTy(elemTy), laneValue(loc, addr, 0), static_cast<unsigned>(align));
                    setMapped(loadOp.getResult(), {SimdVal::Kind::Vector, elemTy, v, {}, {}});
                    return true;
                }
                if (addr.isAllocaPtr) {
                    mlir::Value v = mlir::LLVM::UndefOp::create(b_, loc, vecTy(elemTy));
                    mlir::Value scratch = getOrCreateScratchAlloca(loc);
                    for (uint32_t l = 0; l < W_; ++l) {
                        if (isLaneMaskConstFalse(execMask, l)) continue;
                        mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value pL = laneValue(loc, addr, l);
                        mlir::Value safePtr = mlir::LLVM::SelectOp::create(b_, loc, mL, pL, scratch);
                        mlir::Value s = loadScalar(loc, elemTy, safePtr);
                        v = llvmInsert(b_, loc, v, s, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    }
                    setMapped(loadOp.getResult(), {SimdVal::Kind::Vector, elemTy, v, {}, {}});
                    return true;
                }
                mlir::Value ptrs = asVec(loc, addr);
                mlir::Value undefVec = mlir::LLVM::UndefOp::create(b_, loc, vecTy(elemTy));
                mlir::Value v = mlir::LLVM::masked_gather::create(
                    b_, loc, vecTy(elemTy), ptrs, execMask, mlir::ValueRange{undefVec}, align);
                setMapped(loadOp.getResult(), {SimdVal::Kind::Vector, elemTy, v, {}, {}});
                return true;
            }
            if (auto vt = mlir::dyn_cast<mlir::VectorType>(elemTy)) {
                mlir::Type scalarTy = vt.getElementType();
                int64_t n = vt.getNumElements();
                mlir::Type ptrTy = loadOp.getAddr().getType();
                SimdVal res;
                res.kind = SimdVal::Kind::Struct;
                res.origType = vt;
                res.fields.resize(static_cast<size_t>(n));
                if (addr.isUniform()) {
                    for (int64_t i = 0; i < n; ++i) {
                        mlir::Value ep = mlir::LLVM::GEPOp::create(
                            b_, loc, ptrTy, scalarTy, addr.lanes[0],
                            mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
                        mlir::Value s = loadScalar(loc, scalarTy, ep);
                        mlir::Value bvec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(scalarTy)).getElementType(), s);
                        res.fields[static_cast<size_t>(i)] = {SimdVal::Kind::Vector, scalarTy, bvec, {}, {}};
                    }
                } else if (addr.isAllocaPtr && mlir::isa<mlir::IntegerType, mlir::FloatType>(scalarTy)) {
                    mlir::Value scratch = getOrCreateScratchAlloca(loc);
                    for (int64_t i = 0; i < n; ++i) {
                        mlir::Value v = mlir::LLVM::UndefOp::create(b_, loc, vecTy(scalarTy));
                        for (uint32_t l = 0; l < W_; ++l) {
                            if (isLaneMaskConstFalse(execMask, l)) continue;
                            mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                            mlir::Value pL = laneValue(loc, addr, l);
                            mlir::Value ep = (i == 0)
                                                 ? pL
                                                 : mlir::Value(mlir::LLVM::GEPOp::create(
                                                       b_, loc, ptrTy, scalarTy, pL,
                                                       mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)}));
                            mlir::Value safePtr = mlir::LLVM::SelectOp::create(b_, loc, mL, ep, scratch);
                            mlir::Value s = loadScalar(loc, scalarTy, safePtr);
                            v = llvmInsert(b_, loc, v, s, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        }
                        res.fields[static_cast<size_t>(i)] = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                    }
                } else if (mlir::isa<mlir::IntegerType, mlir::FloatType>(scalarTy)) {
                    mlir::Value ptrs = asVec(loc, addr);
                    mlir::Value undefVec = mlir::LLVM::UndefOp::create(b_, loc, vecTy(scalarTy));
                    for (int64_t i = 0; i < n; ++i) {
                        mlir::Value epVec = (i == 0)
                                                ? ptrs
                                                : mlir::Value(mlir::LLVM::GEPOp::create(
                                                      b_, loc, vecTy(ptrTy), scalarTy, ptrs,
                                                      mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)}));
                        mlir::Value v = mlir::LLVM::masked_gather::create(
                            b_, loc, vecTy(scalarTy), epVec, execMask, mlir::ValueRange{undefVec}, align);
                        res.fields[static_cast<size_t>(i)] = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                    }
                } else {
                    for (int64_t i = 0; i < n; ++i) {
                        mlir::Value v = mlir::LLVM::UndefOp::create(b_, loc, vecTy(scalarTy));
                        for (uint32_t l = 0; l < W_; ++l) {
                            if (isLaneMaskConstFalse(execMask, l)) continue;
                            mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                            mlir::Value pL = laneValue(loc, addr, l);
                            auto ifL = mlir::scf::IfOp::create(
                                b_, loc, mL,
                                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                    mlir::Value ep = mlir::LLVM::GEPOp::create(
                                        nb, nloc, ptrTy, scalarTy, pL,
                                        mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
                                    mlir::scf::YieldOp::create(
                                        nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, scalarTy, ep)});
                                },
                                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                    mlir::scf::YieldOp::create(
                                        nb, nloc, mlir::ValueRange{mlir::LLVM::UndefOp::create(nb, nloc, scalarTy)});
                                });
                            v = llvmInsert(b_, loc, v, ifL.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
                        }
                        res.fields[static_cast<size_t>(i)] = {SimdVal::Kind::Vector, scalarTy, v, {}, {}};
                    }
                }
                setMapped(loadOp.getResult(), res);
                return true;
            }
            SimdVal res;
            res.kind = SimdVal::Kind::Lanes;
            res.origType = elemTy;
            for (uint32_t l = 0; l < (addr.isUniform() ? 1u : W_); ++l) {
                res.lanes.push_back(loadScalar(loc, elemTy, laneValue(loc, addr, l)));
            }
            setMapped(loadOp.getResult(), res);
            return true;
        }
        if (auto storeOp = llvm::dyn_cast<mlir::LLVM::StoreOp>(&op)) {
            SimdVal val = getMapped(loc, storeOp.getValue());
            SimdVal addr = getMapped(loc, storeOp.getAddr());
            mlir::Type valTy = storeOp.getValue().getType();
            uint64_t align = storeOp.getAlignment().value_or(1);
            if (auto vt = mlir::dyn_cast<mlir::VectorType>(valTy)) {
                if (val.kind == SimdVal::Kind::Struct && val.fields.size() == static_cast<size_t>(vt.getNumElements())) {
                    mlir::Type scalarTy = vt.getElementType();
                    mlir::Type ptrTy = storeOp.getAddr().getType();
                    if (mlir::isa<mlir::IntegerType, mlir::FloatType>(scalarTy)) {
                        if (addr.isAllocaPtr) {
                            mlir::Value scratch = getOrCreateScratchAlloca(loc);
                            for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                                for (uint32_t l = 0; l < W_; ++l) {
                                    if (isLaneMaskConstFalse(execMask, l)) continue;
                                    mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                                    mlir::Value pL = laneValue(loc, addr, l);
                                    mlir::Value ep = (i == 0)
                                                         ? pL
                                                         : mlir::Value(mlir::LLVM::GEPOp::create(
                                                               b_, loc, ptrTy, scalarTy, pL,
                                                               mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)}));
                                    mlir::Value safePtr = mlir::LLVM::SelectOp::create(b_, loc, mL, ep, scratch);
                                    mlir::Value vL = laneValue(loc, val.fields[static_cast<size_t>(i)], l);
                                    storeScalar(loc, vL, safePtr);
                                }
                            }
                            return true;
                        }
                        mlir::Value ptrs = asVec(loc, addr);
                        for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                            mlir::Value valVec = asVec(loc, val.fields[static_cast<size_t>(i)]);
                            mlir::Value epVec = (i == 0)
                                                    ? ptrs
                                                    : mlir::Value(mlir::LLVM::GEPOp::create(
                                                          b_, loc, vecTy(ptrTy), scalarTy, ptrs,
                                                          mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)}));
                            mlir::LLVM::masked_scatter::create(b_, loc, valVec, epVec, execMask, align);
                        }
                        return true;
                    }
                    for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                        for (uint32_t l = 0; l < W_; ++l) {
                            if (isLaneMaskConstFalse(execMask, l)) continue;
                            mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                            mlir::Value pL = laneValue(loc, addr, l);
                            mlir::Value vL = laneValue(loc, val.fields[static_cast<size_t>(i)], l);
                            mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                mlir::Value ep = mlir::LLVM::GEPOp::create(
                                    nb, nloc, ptrTy, scalarTy, pL,
                                    mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
                                mlir::LLVM::StoreOp::create(nb, nloc, vL, ep);
                                mlir::scf::YieldOp::create(nb, nloc);
                            });
                        }
                    }
                    return true;
                }
            }
            if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::LLVM::LLVMPointerType>(valTy)) {
                if (addr.isUnitStride() && !addr.isAllocaPtr && isMaskConstAllTrue(execMask)) {
                    if (mlir::Value valVec = asVec(loc, val)) {
                        mlir::LLVM::StoreOp::create(
                            b_, loc, valVec, laneValue(loc, addr, 0), static_cast<unsigned>(align));
                        return true;
                    }
                }
                if (addr.isAllocaPtr) {
                    mlir::Value scratch = getOrCreateScratchAlloca(loc);
                    for (uint32_t l = 0; l < W_; ++l) {
                        if (isLaneMaskConstFalse(execMask, l)) continue;
                        mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value pL = laneValue(loc, addr, l);
                        mlir::Value safePtr = mlir::LLVM::SelectOp::create(b_, loc, mL, pL, scratch);
                        mlir::Value vL = laneValue(loc, val, l);
                        storeScalar(loc, vL, safePtr);
                    }
                    return true;
                }
                mlir::Value valVec = asVec(loc, val);
                mlir::Value ptrs = asVec(loc, addr);
                if (valVec && ptrs) {
                    mlir::LLVM::masked_scatter::create(b_, loc, valVec, ptrs, execMask, align);
                    return true;
                }
            }
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value pL = laneValue(loc, addr, l);
                mlir::Value vL = laneValue(loc, val, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::LLVM::StoreOp::create(nb, nloc, vL, pL);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            }
            return true;
        }
        if (auto rmw = llvm::dyn_cast<mlir::LLVM::AtomicRMWOp>(&op)) {
            SimdVal addr = getMapped(loc, rmw.getPtr());
            SimdVal val = getMapped(loc, rmw.getVal());
            mlir::Type ty = rmw.getRes().getType();
            mlir::Value outVec = mlir::isa<mlir::FloatType>(ty) ? splatFlt(loc, ty, 0.0) : splatInt(loc, ty, 0);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value pL = laneValue(loc, addr, l);
                mlir::Value vL = laneValue(loc, val, l);
                auto ifL = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value r;
                        if (mlir::isa<mlir::FloatType>(ty) && rmw.getBinOp() == mlir::LLVM::AtomicBinOp::xchg) {
                            mlir::Type intTy = nb.getIntegerType(ty.getIntOrFloatBitWidth());
                            mlir::Value vInt = mlir::LLVM::BitcastOp::create(nb, nloc, intTy, vL);
                            mlir::Value rInt = mlir::LLVM::AtomicRMWOp::create(
                                nb, nloc, mlir::LLVM::AtomicBinOp::xchg, pL, vInt, rmw.getOrdering());
                            r = mlir::LLVM::BitcastOp::create(nb, nloc, ty, rInt);
                        } else {
                            r = mlir::LLVM::AtomicRMWOp::create(nb, nloc, rmw.getBinOp(), pL, vL, rmw.getOrdering());
                        }
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{r});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc, mlir::ValueRange{mlir::LLVM::UndefOp::create(nb, nloc, ty)});
                    });
                outVec = llvmInsert(b_, loc, outVec, ifL.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(rmw.getRes(), {SimdVal::Kind::Vector, ty, outVec, {}, {}});
            return true;
        }
        if (auto cx = llvm::dyn_cast<mlir::LLVM::AtomicCmpXchgOp>(&op)) {
            SimdVal addr = getMapped(loc, cx.getPtr());
            SimdVal cmp = getMapped(loc, cx.getCmp());
            SimdVal val = getMapped(loc, cx.getVal());
            mlir::Type elemTy = cmp.origType;
            mlir::Value outValVec = mlir::isa<mlir::FloatType>(elemTy) ? splatFlt(loc, elemTy, 0.0) : splatInt(loc, elemTy, 0);
            mlir::Value outOkVec = splatInt(loc, i1Ty(), 0);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value pL = laneValue(loc, addr, l);
                mlir::Value cL = laneValue(loc, cmp, l);
                mlir::Value vL = laneValue(loc, val, l);
                auto ifL = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value cOp = cL;
                        mlir::Value vOp = vL;
                        if (mlir::isa<mlir::FloatType>(elemTy)) {
                            mlir::Type intTy = nb.getIntegerType(elemTy.getIntOrFloatBitWidth());
                            cOp = mlir::LLVM::BitcastOp::create(nb, nloc, intTy, cL);
                            vOp = mlir::LLVM::BitcastOp::create(nb, nloc, intTy, vL);
                        }
                        auto r = mlir::LLVM::AtomicCmpXchgOp::create(
                            nb, nloc, pL, cOp, vOp, cx.getSuccessOrdering(), cx.getFailureOrdering());
                        mlir::Value oldV = mlir::LLVM::ExtractValueOp::create(nb, nloc, r, llvm::ArrayRef<int64_t>{0});
                        if (mlir::isa<mlir::FloatType>(elemTy)) {
                            oldV = mlir::LLVM::BitcastOp::create(nb, nloc, elemTy, oldV);
                        }
                        mlir::Value ok = mlir::LLVM::ExtractValueOp::create(nb, nloc, r, llvm::ArrayRef<int64_t>{1});
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{oldV, ok});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc,
                            mlir::ValueRange{mlir::LLVM::UndefOp::create(nb, nloc, elemTy),
                                             mlir::LLVM::ConstantOp::create(nb, nloc, i1Ty(), nb.getIntegerAttr(i1Ty(), 0))});
                    });
                outValVec = llvmInsert(b_, loc, outValVec, ifL.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
                outOkVec = llvmInsert(b_, loc, outOkVec, ifL.getResult(1), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = cx.getRes().getType();
            st.fields.push_back({SimdVal::Kind::Vector, elemTy, outValVec, {}, {}});
            st.fields.push_back({SimdVal::Kind::Vector, i1Ty(), outOkVec, {}, {}});
            setMapped(cx.getRes(), st);
            return true;
        }
        if (auto f = llvm::dyn_cast<mlir::LLVM::FenceOp>(&op)) {
            mlir::LLVM::FenceOp::create(b_, loc, f.getOrdering(), f.getSyncscope().value_or(""));
            return true;
        }

        // Vectorizable LLVM & Arith ops
        if (packetizeVectorizableOp(op, execMask)) return true;

        // Unsupported pointer casts / PtrToInt / IntToPtr scalar fallback
        if (auto p2i = llvm::dyn_cast<mlir::LLVM::PtrToIntOp>(&op)) {
            SimdVal src = getMapped(loc, p2i.getArg());
            mlir::Type dstTy = p2i.getRes().getType();
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(dstTy));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value iv = mlir::LLVM::PtrToIntOp::create(b_, loc, dstTy, laneValue(loc, src, l));
                outVec = llvmInsert(b_, loc, outVec, iv, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(p2i.getRes(), {SimdVal::Kind::Vector, dstTy, outVec, {}, {}});
            return true;
        }
        if (auto i2p = llvm::dyn_cast<mlir::LLVM::IntToPtrOp>(&op)) {
            SimdVal src = getMapped(loc, i2p.getArg());
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            for (uint32_t l = 0; l < W_; ++l) {
                sv.lanes.push_back(mlir::LLVM::IntToPtrOp::create(b_, loc, ptrTy(), laneValue(loc, src, l)));
            }
            setMapped(i2p.getRes(), sv);
            return true;
        }
        if (llvm::isa<mlir::LLVM::LifetimeStartOp, mlir::LLVM::LifetimeEndOp>(&op)) {
            return true;
        }

        std::string opStr;
        llvm::raw_string_ostream os(opStr);
        op.print(os);
        outError = "Unsupported MLIR operation in SIMT packetizer: " + opStr;
        return false;
    }

    bool packetizeVectorizableOp(mlir::Operation& op, mlir::Value /*execMask*/) {
        // Unary and binary LLVM dialect ops and intrinsics that natively accept vector<W x T>.
        mlir::Location loc = op.getLoc();
        auto vecUnary = [&](auto tagFn) -> bool {
            mlir::Type resTy = op.getResult(0).getType();
            SimdVal aSv = getMapped(loc, op.getOperand(0));
            auto outPat = aSv.isUniform() ? SimdVal::LanePattern::Uniform : SimdVal::LanePattern::Divergent;
            if (auto vt = llvm::dyn_cast<mlir::VectorType>(resTy)) {
                mlir::Type resElem = vt.getElementType();
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = resTy;
                out.pattern = outPat;
                for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                    mlir::Value ai = (aSv.kind == SimdVal::Kind::Struct && static_cast<size_t>(i) < aSv.fields.size())
                                         ? asVec(loc, aSv.fields[static_cast<size_t>(i)])
                                         : asVec(loc, aSv);
                    mlir::Value r = tagFn(vecTy(resElem), ai);
                    SimdVal fsv{SimdVal::Kind::Vector, resElem, r, {}, {}};
                    fsv.pattern = outPat;
                    out.fields.push_back(std::move(fsv));
                }
                setMapped(op.getResult(0), out);
                return true;
            }
            mlir::Value a = asVec(loc, aSv);
            mlir::Value r = tagFn(vecTy(resTy), a);
            SimdVal sv{SimdVal::Kind::Vector, resTy, r, {}, {}};
            sv.pattern = outPat;
            setMapped(op.getResult(0), std::move(sv));
            return true;
        };
        auto vecBinary = [&](auto tagFn) -> bool {
            mlir::Type resTy = op.getResult(0).getType();
            SimdVal aSv = getMapped(loc, op.getOperand(0));
            SimdVal cSv = getMapped(loc, op.getOperand(1));
            SimdVal::LanePattern outPat = SimdVal::LanePattern::Divergent;
            if (aSv.isUniform() && cSv.isUniform()) {
                outPat = SimdVal::LanePattern::Uniform;
            } else if (llvm::isa<mlir::LLVM::AddOp, mlir::arith::AddIOp>(&op) &&
                       ((aSv.isUnitStride() && cSv.isUniform()) || (aSv.isUniform() && cSv.isUnitStride()))) {
                outPat = SimdVal::LanePattern::UnitStride;
            } else if (llvm::isa<mlir::LLVM::SubOp, mlir::arith::SubIOp>(&op) &&
                       aSv.isUnitStride() && cSv.isUniform()) {
                outPat = SimdVal::LanePattern::UnitStride;
            }
            if (auto vt = llvm::dyn_cast<mlir::VectorType>(resTy)) {
                mlir::Type resElem = vt.getElementType();
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = resTy;
                out.pattern = outPat;
                for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                    mlir::Value ai = (aSv.kind == SimdVal::Kind::Struct && static_cast<size_t>(i) < aSv.fields.size())
                                         ? asVec(loc, aSv.fields[static_cast<size_t>(i)])
                                         : asVec(loc, aSv);
                    mlir::Value ci = (cSv.kind == SimdVal::Kind::Struct && static_cast<size_t>(i) < cSv.fields.size())
                                         ? asVec(loc, cSv.fields[static_cast<size_t>(i)])
                                         : asVec(loc, cSv);
                    mlir::Value r = tagFn(vecTy(resElem), ai, ci);
                    SimdVal fsv{SimdVal::Kind::Vector, resElem, r, {}, {}};
                    fsv.pattern = outPat;
                    out.fields.push_back(std::move(fsv));
                }
                setMapped(op.getResult(0), out);
                return true;
            }
            mlir::Value a = asVec(loc, aSv);
            mlir::Value c = asVec(loc, cSv);
            mlir::Value r = tagFn(vecTy(resTy), a, c);
            SimdVal sv{SimdVal::Kind::Vector, resTy, r, {}, {}};
            sv.pattern = outPat;
            setMapped(op.getResult(0), std::move(sv));
            return true;
        };

        if (llvm::isa<mlir::LLVM::AddOp, mlir::arith::AddIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::AddOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SubOp, mlir::arith::SubIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SubOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::MulOp, mlir::arith::MulIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::MulOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SDivOp, mlir::arith::DivSIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SDivOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::UDivOp, mlir::arith::DivUIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::UDivOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SRemOp, mlir::arith::RemSIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SRemOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::URemOp, mlir::arith::RemUIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::URemOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::ShlOp, mlir::arith::ShLIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::ShlOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::LShrOp, mlir::arith::ShRUIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::LShrOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::AShrOp, mlir::arith::ShRSIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::AShrOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::AndOp, mlir::arith::AndIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::AndOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::OrOp, mlir::arith::OrIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::OrOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::XOrOp, mlir::arith::XOrIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::XOrOp::create(b_, loc, rt, a, c); });

        if (llvm::isa<mlir::LLVM::FAddOp, mlir::arith::AddFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::FAddOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FSubOp, mlir::arith::SubFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::FSubOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FMulOp, mlir::arith::MulFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::FMulOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FDivOp, mlir::arith::DivFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::FDivOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FRemOp, mlir::arith::RemFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::FRemOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FNegOp, mlir::arith::NegFOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return createFp<mlir::LLVM::FNegOp>(loc, rt, a); });

        if (llvm::isa<mlir::LLVM::ZExtOp, mlir::arith::ExtUIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::ZExtOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SExtOp, mlir::arith::ExtSIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::SExtOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::TruncOp, mlir::arith::TruncIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::TruncOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPTruncOp, mlir::arith::TruncFOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::FPTruncOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPExtOp, mlir::arith::ExtFOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::FPExtOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPToSIOp, mlir::arith::FPToSIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::FPToSIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPToUIOp, mlir::arith::FPToUIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::FPToUIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SIToFPOp, mlir::arith::SIToFPOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::SIToFPOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::UIToFPOp, mlir::arith::UIToFPOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::UIToFPOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::BitcastOp, mlir::arith::BitcastOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::BitcastOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::arith::IndexCastOp, mlir::arith::IndexCastUIOp>(&op)) {
            bool isUnsigned = llvm::isa<mlir::arith::IndexCastUIOp>(&op);
            SimdVal v = getMapped(loc, op.getOperand(0));
            if (!v.valid()) return false;
            mlir::Type resTy = op.getResult(0).getType();
            if (resTy.isIndex()) {
                SimdVal out;
                out.kind = SimdVal::Kind::Lanes;
                out.origType = resTy;
                uint32_t nLanes = (v.kind == SimdVal::Kind::Lanes && v.lanes.size() == 1) ? 1 : W_;
                for (uint32_t l = 0; l < nLanes; ++l) {
                    mlir::Value s = (v.kind == SimdVal::Kind::Lanes)
                                        ? v.lanes[l]
                                        : mlir::Value(llvmExtract(b_, loc, v.vec, static_cast<uint32_t>(static_cast<int64_t>(l))));
                    mlir::Value casted = isUnsigned
                                             ? mlir::Value(mlir::arith::IndexCastUIOp::create(b_, loc, resTy, s))
                                             : mlir::Value(mlir::arith::IndexCastOp::create(b_, loc, resTy, s));
                    out.lanes.push_back(casted);
                }
                setMapped(op.getResult(0), std::move(out));
                return true;
            }
            mlir::Type vecResTy = vecTy(resTy);
            if (v.kind == SimdVal::Kind::Lanes) {
                mlir::Value vec = llvm::cast<mlir::Value>(mlir::LLVM::UndefOp::create(b_, loc, vecResTy));
                for (uint32_t l = 0; l < W_; ++l) {
                    mlir::Value s = v.lanes.size() == 1 ? v.lanes[0] : v.lanes[l];
                    mlir::Value casted = isUnsigned
                                             ? mlir::Value(mlir::arith::IndexCastUIOp::create(b_, loc, resTy, s))
                                             : mlir::Value(mlir::arith::IndexCastOp::create(b_, loc, resTy, s));
                    vec = llvmInsert(b_, loc, vec, casted, static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
                SimdVal vecOut;
                vecOut.kind = SimdVal::Kind::Vector;
                vecOut.origType = resTy;
                vecOut.vec = vec;
                setMapped(op.getResult(0), std::move(vecOut));
                return true;
            }
            uint32_t srcBits = llvm::cast<mlir::VectorType>(v.vec.getType()).getElementType().getIntOrFloatBitWidth();
            uint32_t dstBits = resTy.getIntOrFloatBitWidth();
            mlir::Value out = v.vec;
            if (dstBits > srcBits) {
                out = isUnsigned ? mlir::Value(mlir::LLVM::ZExtOp::create(b_, loc, vecResTy, v.vec))
                                 : mlir::Value(mlir::LLVM::SExtOp::create(b_, loc, vecResTy, v.vec));
            } else if (dstBits < srcBits) {
                out = mlir::LLVM::TruncOp::create(b_, loc, vecResTy, v.vec);
            }
            SimdVal vecOut;
            vecOut.kind = SimdVal::Kind::Vector;
            vecOut.origType = resTy;
            vecOut.vec = out;
            setMapped(op.getResult(0), std::move(vecOut));
            return true;
        }

        if (auto icmp = llvm::dyn_cast<mlir::LLVM::ICmpOp>(&op)) {
            if (auto vt = llvm::dyn_cast<mlir::VectorType>(icmp.getRes().getType())) {
                SimdVal aSv = getMapped(loc, icmp.getLhs());
                SimdVal cSv = getMapped(loc, icmp.getRhs());
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = vt;
                for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                    mlir::Value ai = asVec(loc, aSv.fields[static_cast<size_t>(i)]);
                    mlir::Value ci = asVec(loc, cSv.fields[static_cast<size_t>(i)]);
                    mlir::Value r = mlir::LLVM::ICmpOp::create(b_, loc, vecTy(i1Ty()), icmp.getPredicate(), ai, ci);
                    out.fields.push_back({SimdVal::Kind::Vector, i1Ty(), r, {}, {}});
                }
                setMapped(icmp.getRes(), out);
                return true;
            }
            if (mlir::isa<mlir::LLVM::LLVMPointerType>(icmp.getLhs().getType())) {
                SimdVal a = getMapped(loc, icmp.getLhs());
                SimdVal c = getMapped(loc, icmp.getRhs());
                mlir::Value outVec = splatInt(loc, i1Ty(), 0);
                for (uint32_t l = 0; l < W_; ++l) {
                    mlir::Value r = mlir::LLVM::ICmpOp::create(
                        b_, loc, icmp.getPredicate(), laneValue(loc, a, l), laneValue(loc, c, l));
                    outVec = llvmInsert(b_, loc, outVec, r, static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
                setMapped(icmp.getRes(), {SimdVal::Kind::Vector, i1Ty(), outVec, {}, {}});
                return true;
            }
            mlir::Value a = asVec(loc, getMapped(loc, icmp.getLhs()));
            mlir::Value c = asVec(loc, getMapped(loc, icmp.getRhs()));
            mlir::Value r = mlir::LLVM::ICmpOp::create(b_, loc, vecTy(i1Ty()), icmp.getPredicate(), a, c);
            setMapped(icmp.getRes(), {SimdVal::Kind::Vector, i1Ty(), r, {}, {}});
            return true;
        }
        if (auto icmp = llvm::dyn_cast<mlir::arith::CmpIOp>(&op)) {
            if (icmp.getLhs().getType().isIndex()) {
                SimdVal va = getMapped(loc, icmp.getLhs());
                SimdVal vc = getMapped(loc, icmp.getRhs());
                if (!va.valid() || !vc.valid()) return false;
                mlir::Value outVec = splatInt(loc, i1Ty(), 0);
                for (uint32_t l = 0; l < W_; ++l) {
                    mlir::Value la = (va.kind == SimdVal::Kind::Lanes)
                                         ? (va.lanes.size() == 1 ? va.lanes[0] : va.lanes[l])
                                         : mlir::Value(llvmExtract(b_, loc, va.vec, static_cast<uint32_t>(static_cast<int64_t>(l))));
                    mlir::Value lc = (vc.kind == SimdVal::Kind::Lanes)
                                         ? (vc.lanes.size() == 1 ? vc.lanes[0] : vc.lanes[l])
                                         : mlir::Value(llvmExtract(b_, loc, vc.vec, static_cast<uint32_t>(static_cast<int64_t>(l))));
                    mlir::Value r = mlir::LLVM::ICmpOp::create(b_, loc, toLLVMIcmpPred(icmp.getPredicate()), la, lc);
                    outVec = llvmInsert(b_, loc, outVec, r, static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
                setMapped(icmp.getResult(), {SimdVal::Kind::Vector, i1Ty(), outVec, {}, {}});
                return true;
            }
            mlir::Value a = asVec(loc, getMapped(loc, icmp.getLhs()));
            mlir::Value c = asVec(loc, getMapped(loc, icmp.getRhs()));
            if (!a || !c) return false;
            mlir::Value r = mlir::LLVM::ICmpOp::create(b_, loc, toLLVMIcmpPred(icmp.getPredicate()), a, c);
            setMapped(icmp.getResult(), {SimdVal::Kind::Vector, i1Ty(), r, {}, {}});
            return true;
        }
        if (auto fcmp = llvm::dyn_cast<mlir::LLVM::FCmpOp>(&op)) {
            if (auto vt = llvm::dyn_cast<mlir::VectorType>(fcmp.getRes().getType())) {
                SimdVal aSv = getMapped(loc, fcmp.getLhs());
                SimdVal cSv = getMapped(loc, fcmp.getRhs());
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = vt;
                for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                    mlir::Value ai = asVec(loc, aSv.fields[static_cast<size_t>(i)]);
                    mlir::Value ci = asVec(loc, cSv.fields[static_cast<size_t>(i)]);
                    mlir::Value r = mlir::LLVM::FCmpOp::create(b_, loc, vecTy(i1Ty()), fcmp.getPredicate(), ai, ci);
                    out.fields.push_back({SimdVal::Kind::Vector, i1Ty(), r, {}, {}});
                }
                setMapped(fcmp.getRes(), out);
                return true;
            }
            mlir::Value a = asVec(loc, getMapped(loc, fcmp.getLhs()));
            mlir::Value c = asVec(loc, getMapped(loc, fcmp.getRhs()));
            mlir::Value r = mlir::LLVM::FCmpOp::create(b_, loc, vecTy(i1Ty()), fcmp.getPredicate(), a, c);
            setMapped(fcmp.getRes(), {SimdVal::Kind::Vector, i1Ty(), r, {}, {}});
            return true;
        }
        if (auto sel = llvm::dyn_cast<mlir::LLVM::SelectOp>(&op)) {
            SimdVal condSv = getMapped(loc, sel.getCondition());
            SimdVal t = getMapped(loc, sel.getTrueValue());
            SimdVal f = getMapped(loc, sel.getFalseValue());
            if (condSv.kind == SimdVal::Kind::Struct && t.kind == SimdVal::Kind::Struct && f.kind == SimdVal::Kind::Struct) {
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = sel.getType();
                for (size_t i = 0; i < condSv.fields.size() && i < t.fields.size() && i < f.fields.size(); ++i) {
                    out.fields.push_back(selectVal(loc, asVec(loc, condSv.fields[i]), t.fields[i], f.fields[i]));
                }
                setMapped(sel.getRes(), out);
                return true;
            }
            mlir::Value c = asVec(loc, condSv);
            setMapped(sel.getRes(), selectVal(loc, c, t, f));
            return true;
        }
        if (auto sel = llvm::dyn_cast<mlir::arith::SelectOp>(&op)) {
            mlir::Value c = asVec(loc, getMapped(loc, sel.getCondition()));
            SimdVal t = getMapped(loc, sel.getTrueValue());
            SimdVal f = getMapped(loc, sel.getFalseValue());
            setMapped(sel.getResult(), selectVal(loc, c, t, f));
            return true;
        }

        // Standard LLVM intrinsics (which DXIL/Clang might emit alongside dx.op.*)
        if (llvm::isa<mlir::LLVM::FAbsOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return createFp<mlir::LLVM::FAbsOp>(loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SqrtOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return createFp<mlir::LLVM::SqrtOp>(loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SinOp>(&op))
            return vecUnary([&](mlir::Type, mlir::Value a) { return emitMathUnary(loc, "sin", a); });
        if (llvm::isa<mlir::LLVM::CosOp>(&op))
            return vecUnary([&](mlir::Type, mlir::Value a) { return emitMathUnary(loc, "cos", a); });
        if (llvm::isa<mlir::LLVM::Exp2Op>(&op))
            return vecUnary([&](mlir::Type, mlir::Value a) { return emitMathUnary(loc, "exp2", a); });
        if (llvm::isa<mlir::LLVM::Log2Op>(&op))
            return vecUnary([&](mlir::Type, mlir::Value a) { return emitMathUnary(loc, "log2", a); });
        if (llvm::isa<mlir::LLVM::FFloorOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return createFp<mlir::LLVM::FFloorOp>(loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FCeilOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return createFp<mlir::LLVM::FCeilOp>(loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FTruncOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return createFp<mlir::LLVM::FTruncOp>(loc, rt, a); });
        if (llvm::isa<mlir::LLVM::MinNumOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::MinNumOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::MaxNumOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return createFp<mlir::LLVM::MaxNumOp>(loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::UMinOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::UMinOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::UMaxOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::UMaxOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SMinOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SMinOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SMaxOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SMaxOp::create(b_, loc, rt, a, c); });
        auto castIntVec = [&](mlir::Value v, mlir::Type dstElem) -> mlir::Value {
            mlir::Type srcElem = llvm::cast<mlir::VectorType>(v.getType()).getElementType();
            if (srcElem == dstElem) return v;
            return (srcElem.getIntOrFloatBitWidth() > dstElem.getIntOrFloatBitWidth())
                       ? mlir::Value(mlir::LLVM::TruncOp::create(b_, loc, vecTy(dstElem), v))
                       : mlir::Value(mlir::LLVM::ZExtOp::create(b_, loc, vecTy(dstElem), v));
        };
        if (llvm::isa<mlir::LLVM::CtPopOp>(&op)) {
            mlir::Type resElem = op.getResult(0).getType();
            mlir::Value a = asVec(loc, getMapped(loc, op.getOperand(0)));
            mlir::Value r = castIntVec(mlir::LLVM::CtPopOp::create(b_, loc, a.getType(), a), resElem);
            setMapped(op.getResult(0), {SimdVal::Kind::Vector, resElem, r, {}, {}});
            return true;
        }
        if (llvm::isa<mlir::LLVM::BitReverseOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::LLVM::BitReverseOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FMAOp, mlir::LLVM::FMulAddOp>(&op)) {
            mlir::Type resElem = op.getResult(0).getType();
            mlir::Value a = asVec(loc, getMapped(loc, op.getOperand(0)));
            mlir::Value x = asVec(loc, getMapped(loc, op.getOperand(1)));
            mlir::Value c = asVec(loc, getMapped(loc, op.getOperand(2)));
            mlir::Value r = createFp<mlir::LLVM::FMAOp>(loc, vecTy(resElem), a, x, c);
            setMapped(op.getResult(0), {SimdVal::Kind::Vector, resElem, r, {}, {}});
            return true;
        }
        if (auto ctlz = llvm::dyn_cast<mlir::LLVM::CountLeadingZerosOp>(&op)) {
            mlir::Type resElem = ctlz.getType();
            mlir::Value a = asVec(loc, getMapped(loc, ctlz.getIn()));
            mlir::Value r = castIntVec(mlir::LLVM::CountLeadingZerosOp::create(b_, loc, a.getType(), a, false), resElem);
            setMapped(ctlz.getRes(), {SimdVal::Kind::Vector, resElem, r, {}, {}});
            return true;
        }
        if (auto cttz = llvm::dyn_cast<mlir::LLVM::CountTrailingZerosOp>(&op)) {
            mlir::Type resElem = cttz.getType();
            mlir::Value a = asVec(loc, getMapped(loc, cttz.getIn()));
            mlir::Value r = castIntVec(mlir::LLVM::CountTrailingZerosOp::create(b_, loc, a.getType(), a, false), resElem);
            setMapped(cttz.getRes(), {SimdVal::Kind::Vector, resElem, r, {}, {}});
            return true;
        }
        if (auto fpc = llvm::dyn_cast<mlir::LLVM::IsFPClass>(&op)) {
            mlir::Value a = asVec(loc, getMapped(loc, fpc.getIn()));
            auto r = mlir::LLVM::IsFPClass::create(b_, loc, vecTy(i1Ty()), a, fpc.getBit());
            setMapped(fpc.getRes(), {SimdVal::Kind::Vector, i1Ty(), r.getRes(), {}, {}});
            return true;
        }
        return false;
    }

    bool packetizeScfIndexSwitch(mlir::scf::IndexSwitchOp swOp, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = swOp.getLoc();
        SimdVal argVal = getMapped(loc, swOp.getArg());
        if (!argVal.valid()) {
            outError = "Unmapped scf.index_switch arg";
            return false;
        }
        // Build vector<W x i64> for the switch selector
        mlir::Value argVec = splatInt(loc, i64Ty(), 0);
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value idxVal = (argVal.kind == SimdVal::Kind::Lanes)
                                     ? (argVal.lanes.size() == 1 ? argVal.lanes[0] : argVal.lanes[l])
                                     : mlir::Value(llvmExtract(b_, loc, argVal.vec, static_cast<uint32_t>(static_cast<int64_t>(l))));
            if (!idxVal) {
                outError = "Null idxVal in scf.index_switch";
                return false;
            }
            mlir::Value i64v = idxVal.getType().isIndex()
                                   ? mlir::Value(mlir::arith::IndexCastOp::create(b_, loc, i64Ty(), idxVal))
                                   : idxVal;
            argVec = llvmInsert(b_, loc, argVec, i64v, static_cast<uint32_t>(static_cast<int64_t>(l)));
        }

        llvm::SmallVector<SimdVal, 8> accVals;
        for (mlir::Value r : swOp.getResults()) {
            accVals.push_back(makeZeroSimd(loc, r.getType()));
        }

        auto runCaseBlock = [&](mlir::Block& srcBB, mlir::Value caseMask) -> bool {
            mlir::Value curReal = mlir::LLVM::AndOp::create(b_, loc, realMask, caseMask);
            mlir::Value curExec = mlir::LLVM::AndOp::create(b_, loc, execMask, caseMask);
            if (hooks_.isPixelShader && shader_.hasDerivativesOrImplicitSample) {
                curExec = expandMaskQuads(loc, curReal);
            }
            mlir::Value anyExec = orReduce(loc, curExec);
            llvm::SmallVector<mlir::Type, 8> packedTypes;
            for (mlir::Value r : swOp.getResults()) {
                flattenSimdTypes(r.getType(), packedTypes);
            }
            auto savedMap = valueMap_;
            if (!packedTypes.empty()) {
                auto guard = mlir::scf::IfOp::create(
                    b_, loc, anyExec,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                        sub.valueMap_ = valueMap_;
                        bool ok = sub.packetizeBlock(srcBB, curExec, curReal, outError);
                        llvm::SmallVector<mlir::Value, 8> yielded;
                        if (ok) {
                            if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(srcBB.getTerminator())) {
                                for (size_t i = 0; i < yo.getNumOperands(); ++i) {
                                    sub.packSimdToValues(nloc, sub.getMapped(nloc, yo.getOperand(i)), yielded);
                                }
                            }
                        } else {
                            for (mlir::Value r : swOp.getResults()) {
                                sub.packSimdToValues(nloc, sub.makeZeroSimd(nloc, r.getType()), yielded);
                            }
                        }
                        mlir::scf::YieldOp::create(nb, nloc, yielded);
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        llvm::SmallVector<mlir::Value, 8> zeros;
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                        for (mlir::Value r : swOp.getResults()) {
                            sub.packSimdToValues(nloc, sub.makeZeroSimd(nloc, r.getType()), zeros);
                        }
                        mlir::scf::YieldOp::create(nb, nloc, zeros);
                    });
                size_t cur = 0;
                for (size_t i = 0; i < swOp.getNumResults(); ++i) {
                    SimdVal caseRes = unpackSimdFromValues(loc, swOp.getResult(i).getType(), guard.getResults(), cur, &accVals[i]);
                    accVals[i] = selectVal(loc, caseMask, caseRes, accVals[i]);
                }
            } else {
                mlir::scf::IfOp::create(
                    b_, loc, anyExec,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                        sub.valueMap_ = valueMap_;
                        (void)sub.packetizeBlock(srcBB, curExec, curReal, outError);
                        mlir::scf::YieldOp::create(nb, nloc);
                    });
            }
            valueMap_ = std::move(savedMap);
            return outError.empty();
        };

        mlir::Value remainingMask = allTrueMask(loc);
        auto cases = swOp.getCases();
        for (size_t ci = 0; ci < cases.size(); ++ci) {
            mlir::Value cVec = splatInt(loc, i64Ty(), cases[ci]);
            mlir::Value eqMask = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, argVec, cVec);
            mlir::Value caseMask = mlir::LLVM::AndOp::create(b_, loc, remainingMask, eqMask);
            if (!runCaseBlock(swOp.getCaseRegions()[ci].front(), caseMask)) return false;
            mlir::Value notCase = mlir::LLVM::XOrOp::create(b_, loc, caseMask, allTrueMask(loc));
            remainingMask = mlir::LLVM::AndOp::create(b_, loc, remainingMask, notCase);
        }
        if (!swOp.getDefaultRegion().empty()) {
            if (!runCaseBlock(swOp.getDefaultRegion().front(), remainingMask)) return false;
        }
        for (size_t i = 0; i < swOp.getNumResults(); ++i) {
            setMapped(swOp.getResult(i), accVals[i]);
        }
        return true;
    }

    bool packetizeScfIf(mlir::scf::IfOp ifOp, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = ifOp.getLoc();
        mlir::Value condVec = asVec(loc, getMapped(loc, ifOp.getCondition()));
        mlir::Value realThen = mlir::LLVM::AndOp::create(b_, loc, realMask, condVec);
        mlir::Value notCond = mlir::LLVM::XOrOp::create(b_, loc, condVec, allTrueMask(loc));
        mlir::Value realElse = mlir::LLVM::AndOp::create(b_, loc, realMask, notCond);

        mlir::Value execThen = mlir::LLVM::AndOp::create(b_, loc, execMask, condVec);
        mlir::Value execElse = mlir::LLVM::AndOp::create(b_, loc, execMask, notCond);
        if (hooks_.isPixelShader && shader_.hasDerivativesOrImplicitSample) {
            execThen = expandMaskQuads(loc, realThen);
            execElse = expandMaskQuads(loc, realElse);
        }

        llvm::SmallVector<mlir::Type, 8> packedTypes;
        llvm::SmallVector<SimdVal, 8> yieldProtos;
        for (mlir::Value r : ifOp.getResults()) {
            SimdVal z = makeZeroSimd(loc, r.getType());
            yieldProtos.push_back(z);
            flattenSimdTypes(r.getType(), packedTypes);
        }
        // In pixel shader with discard inside branch, thread updated realMask out!
        bool hasRealMaskOut = hooks_.isPixelShader && shader_.usesDiscard;
        if (hasRealMaskOut) packedTypes.push_back(vecTy(i1Ty()));

        mlir::Value realMaskAfterThen = realMask;
        auto runBranch = [&](mlir::Block& srcBB, mlir::Value bExec, mlir::Value bReal,
                             llvm::SmallVectorImpl<mlir::Value>& outPacked) -> bool {
            mlir::Value curReal = bReal;
            auto savedMap = valueMap_;
            mlir::Value anyExec = orReduce(loc, bExec);
            if (!packedTypes.empty()) {
                auto guard = mlir::scf::IfOp::create(
                    b_, loc, anyExec,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                        sub.valueMap_ = valueMap_;
                        if (!sub.packetizeBlock(srcBB, bExec, curReal, outError)) {
                            for (mlir::Type pt : packedTypes) (void)pt;
                        }
                        llvm::SmallVector<mlir::Value, 8> yielded;
                        if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(srcBB.getTerminator())) {
                            for (size_t i = 0; i < yo.getNumOperands(); ++i) {
                                sub.packSimdToValues(nloc, sub.getMapped(nloc, yo.getOperand(i)), yielded);
                            }
                        }
                        if (hasRealMaskOut) yielded.push_back(curReal);
                        mlir::scf::YieldOp::create(nb, nloc, yielded);
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        llvm::SmallVector<mlir::Value, 8> zeros;
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                        for (mlir::Value r : ifOp.getResults()) {
                            sub.packSimdToValues(nloc, sub.makeZeroSimd(nloc, r.getType()), zeros);
                        }
                        if (hasRealMaskOut) zeros.push_back(bReal);
                        mlir::scf::YieldOp::create(nb, nloc, zeros);
                    });
                for (mlir::Value r : guard.getResults()) outPacked.push_back(r);
            } else {
                mlir::scf::IfOp::create(
                    b_, loc, anyExec,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                        sub.valueMap_ = valueMap_;
                        (void)sub.packetizeBlock(srcBB, bExec, curReal, outError);
                        mlir::scf::YieldOp::create(nb, nloc);
                    });
            }
            valueMap_ = std::move(savedMap);
            return outError.empty();
        };

        llvm::SmallVector<mlir::Value, 8> thenPacked, elsePacked;
        if (!runBranch(ifOp.getThenRegion().front(), execThen, realThen, thenPacked)) return false;

        bool hasElse = !ifOp.getElseRegion().empty();
        if (hasElse) {
            mlir::Block& elseBB = ifOp.getElseRegion().front();
            if (!runBranch(elseBB, execElse, realElse, elsePacked)) return false;
        } else if (hasRealMaskOut) {
            for (mlir::Value r : ifOp.getResults()) packSimdToValues(loc, makeZeroSimd(loc, r.getType()), elsePacked);
            elsePacked.push_back(realElse);
        }

        if (!ifOp.getResults().empty()) {
            size_t tCur = 0, eCur = 0;
            for (size_t i = 0; i < ifOp.getResults().size(); ++i) {
                mlir::Type origTy = ifOp.getResult(i).getType();
                SimdVal tv = unpackSimdFromValues(loc, origTy, thenPacked, tCur, &yieldProtos[i]);
                SimdVal ev = hasElse
                                 ? unpackSimdFromValues(loc, origTy, elsePacked, eCur, &yieldProtos[i])
                                 : makeZeroSimd(loc, origTy);
                setMapped(ifOp.getResult(i), selectVal(loc, condVec, tv, ev));
            }
        }
        if (hasRealMaskOut && !thenPacked.empty() && !elsePacked.empty()) {
            realMask = mlir::LLVM::OrOp::create(b_, loc, thenPacked.back(), elsePacked.back());
            (void)realMaskAfterThen;
        }
        return true;
    }

    bool packetizeScfWhile(mlir::scf::WhileOp whileOp, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = whileOp.getLoc();
        llvm::SmallVector<mlir::Type, 8> resTypes;
        llvm::SmallVector<mlir::Value, 8> initVals;
        llvm::SmallVector<SimdVal, 8> initProtos;
        llvm::SmallVector<SimdVal, 8> exitProtos;

        // First two loop-carried values: [execMask, aliveMask]
        resTypes.push_back(vecTy(i1Ty()));
        resTypes.push_back(vecTy(i1Ty()));
        initVals.push_back(execMask);
        initVals.push_back(realMask);

        for (mlir::Value iv : whileOp.getInits()) {
            SimdVal sv = getMapped(loc, iv);
            initProtos.push_back(sv);
            packSimdToValues(loc, sv, initVals);
        }

        // condArgs / result types (U) appear twice in After/results: [curU..., exitU...]
        for (mlir::Value r : whileOp.getResults()) {
            SimdVal z = makeZeroSimd(loc, r.getType());
            exitProtos.push_back(z);
            flattenSimdTypes(r.getType(), resTypes);
        }
        for (mlir::Value r : whileOp.getResults()) {
            flattenSimdTypes(r.getType(), resTypes);
        }
        for (const SimdVal& z : exitProtos) {
            packSimdToValues(loc, z, initVals);
        }

        mlir::Block& srcBeforeOuter = whileOp.getBefore().front();
        mlir::Block& srcAfterOuter = whileOp.getAfter().front();
        auto outerCondOp = llvm::cast<mlir::scf::ConditionOp>(srcBeforeOuter.getTerminator());

        auto newWhile = mlir::scf::WhileOp::create(
            b_, loc, resTypes, initVals,
            [&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::ValueRange bArgs) {
                SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                sub.valueMap_ = valueMap_;
                mlir::Value loopExec = bArgs[0];
                mlir::Value aliveMask = bArgs[1];
                size_t cur = 2;
                mlir::Block& srcBefore = whileOp.getBefore().front();
                for (unsigned i = 0; i < srcBefore.getNumArguments(); ++i) {
                    mlir::Type at = srcBefore.getArgument(i).getType();
                    sub.setMapped(srcBefore.getArgument(i),
                                  sub.unpackSimdFromValues(nloc, at, bArgs, cur, &initProtos[i]));
                }
                llvm::SmallVector<SimdVal, 8> prevExitU;
                for (unsigned j = 0; j < whileOp.getNumResults(); ++j) {
                    mlir::Type rt = whileOp.getResult(j).getType();
                    prevExitU.push_back(sub.unpackSimdFromValues(nloc, rt, bArgs, cur, &exitProtos[j]));
                }

                mlir::Value curReal = mlir::LLVM::AndOp::create(nb, nloc, loopExec, aliveMask);
                (void)sub.packetizeBlock(srcBefore, loopExec, curReal, outError);

                mlir::Value notLoopExec = mlir::LLVM::XOrOp::create(nb, nloc, loopExec, sub.allTrueMask(nloc));
                mlir::Value survived = mlir::LLVM::OrOp::create(nb, nloc, notLoopExec, curReal);
                mlir::Value aliveAfterBefore = mlir::LLVM::AndOp::create(nb, nloc, aliveMask, survived);

                auto condOp = llvm::cast<mlir::scf::ConditionOp>(srcBefore.getTerminator());
                mlir::Value condVec = sub.asVec(nloc, sub.getMapped(nloc, condOp.getCondition()));
                mlir::Value nextExec = mlir::LLVM::AndOp::create(nb, nloc, loopExec, condVec);
                nextExec = mlir::LLVM::AndOp::create(nb, nloc, nextExec, aliveAfterBefore);
                if (hooks_.isPixelShader && shader_.hasDerivativesOrImplicitSample) {
                    nextExec = sub.expandMaskQuads(nloc, nextExec);
                }
                mlir::Value anyActive = sub.orReduce(nloc, nextExec);

                llvm::SmallVector<mlir::Value, 8> condArgs;
                condArgs.push_back(nextExec);
                condArgs.push_back(aliveAfterBefore);

                llvm::SmallVector<SimdVal, 8> curUVals;
                for (unsigned j = 0; j < condOp.getArgs().size(); ++j) {
                    SimdVal cv = sub.getMapped(nloc, condOp.getArgs()[j]);
                    curUVals.push_back(cv);
                    sub.packSimdToValues(nloc, cv, condArgs);
                }
                for (unsigned j = 0; j < condOp.getArgs().size(); ++j) {
                    SimdVal updatedExit = (j < prevExitU.size())
                                              ? sub.selectVal(nloc, loopExec, curUVals[j], prevExitU[j])
                                              : curUVals[j];
                    sub.packSimdToValues(nloc, updatedExit, condArgs);
                }
                mlir::scf::ConditionOp::create(nb, nloc, anyActive, condArgs);
            },
            [&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::ValueRange aArgs) {
                SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                sub.valueMap_ = valueMap_;
                mlir::Value bodyExec = aArgs[0];
                mlir::Value bodyAlive = aArgs[1];
                size_t cur = 2;
                mlir::Block& srcAfter = whileOp.getAfter().front();
                llvm::SmallVector<SimdVal, 8> inStates;
                for (unsigned i = 0; i < srcAfter.getNumArguments(); ++i) {
                    mlir::Type at = srcAfter.getArgument(i).getType();
                    const SimdVal* proto = i < exitProtos.size() ? &exitProtos[i] : nullptr;
                    SimdVal sv = sub.unpackSimdFromValues(nloc, at, aArgs, cur, proto);
                    inStates.push_back(sv);
                    sub.setMapped(srcAfter.getArgument(i), sv);
                }
                llvm::SmallVector<SimdVal, 8> passedExitU;
                for (unsigned j = 0; j < whileOp.getNumResults(); ++j) {
                    mlir::Type rt = whileOp.getResult(j).getType();
                    passedExitU.push_back(sub.unpackSimdFromValues(nloc, rt, aArgs, cur, &exitProtos[j]));
                }

                mlir::Value bodyReal = mlir::LLVM::AndOp::create(nb, nloc, bodyExec, bodyAlive);
                (void)sub.packetizeBlock(srcAfter, bodyExec, bodyReal, outError);

                mlir::Value notBodyExec = mlir::LLVM::XOrOp::create(nb, nloc, bodyExec, sub.allTrueMask(nloc));
                mlir::Value survived = mlir::LLVM::OrOp::create(nb, nloc, notBodyExec, bodyReal);
                mlir::Value aliveAfterBody = mlir::LLVM::AndOp::create(nb, nloc, bodyAlive, survived);

                auto yieldOp = llvm::cast<mlir::scf::YieldOp>(srcAfter.getTerminator());
                llvm::SmallVector<mlir::Value, 8> nextArgs;
                nextArgs.push_back(bodyExec);
                nextArgs.push_back(aliveAfterBody);
                for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                    SimdVal nextV = sub.getMapped(nloc, yieldOp.getOperand(i));
                    bool canBlend = i < inStates.size() && inStates[i].origType == yieldOp.getOperand(i).getType();
                    SimdVal blended = canBlend ? sub.selectVal(nloc, bodyExec, nextV, inStates[i]) : nextV;
                    sub.packSimdToValues(nloc, blended, nextArgs);
                }
                for (const SimdVal& eu : passedExitU) {
                    sub.packSimdToValues(nloc, eu, nextArgs);
                }
                mlir::scf::YieldOp::create(nb, nloc, nextArgs);
            });
        if (!outError.empty()) return false;

        // `newWhile.getResult(1)` holds `aliveAfterBefore` (surviving lanes across iterations).
        realMask = newWhile.getResult(1);
        size_t cur = 2;
        for (mlir::Value ca : outerCondOp.getArgs()) {
            llvm::SmallVector<mlir::Type, 4> tmpTys;
            flattenSimdTypes(ca.getType(), tmpTys);
            cur += tmpTys.size();
        }
        for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
            mlir::Type rt = whileOp.getResult(i).getType();
            const SimdVal* proto = i < exitProtos.size() ? &exitProtos[i] : nullptr;
            setMapped(whileOp.getResult(i), unpackSimdFromValues(loc, rt, newWhile.getResults(), cur, proto));
        }
        (void)srcAfterOuter;
        return true;
    }

    bool packetizeScfFor(mlir::scf::ForOp forOp, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = forOp.getLoc();
        mlir::Value lb = llvmExtract(b_, loc, asVec(loc, getMapped(loc, forOp.getLowerBound())), static_cast<uint32_t>(0));
        mlir::Value ub = llvmExtract(b_, loc, asVec(loc, getMapped(loc, forOp.getUpperBound())), static_cast<uint32_t>(0));
        mlir::Value step = llvmExtract(b_, loc, asVec(loc, getMapped(loc, forOp.getStep())), static_cast<uint32_t>(0));
        llvm::SmallVector<mlir::Value, 4> inits;
        llvm::SmallVector<SimdVal, 4> protos;
        for (mlir::Value iv : forOp.getInitArgs()) {
            SimdVal sv = getMapped(loc, iv);
            protos.push_back(sv);
            packSimdToValues(loc, sv, inits);
        }
        auto nf = mlir::scf::ForOp::create(
            b_, loc, lb, ub, step, inits,
            [&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::Value iv, mlir::ValueRange iterArgs) {
                SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_, descCache_);
                sub.valueMap_ = valueMap_;
                mlir::Value ivVec = llvmSplat(nb, nloc, llvm::cast<mlir::VectorType>(sub.vecTy(iv.getType())).getElementType(), iv);
                sub.setMapped(forOp.getInductionVar(), {SimdVal::Kind::Vector, iv.getType(), ivVec, {}, {}});
                size_t cur = 0;
                for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); ++i) {
                    sub.setMapped(forOp.getRegionIterArgs()[i],
                                  sub.unpackSimdFromValues(nloc, forOp.getRegionIterArgs()[i].getType(), iterArgs, cur, &protos[i]));
                }
                mlir::Value loopReal = realMask;
                (void)sub.packetizeBlock(*forOp.getBody(), execMask, loopReal, outError);
                llvm::SmallVector<mlir::Value, 4> yVals;
                if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(forOp.getBody()->getTerminator())) {
                    for (mlir::Value v : yo.getOperands()) sub.packSimdToValues(nloc, sub.getMapped(nloc, v), yVals);
                }
                mlir::scf::YieldOp::create(nb, nloc, yVals);
            });
        if (!outError.empty()) return false;
        size_t cur = 0;
        for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
            setMapped(forOp.getResult(i), unpackSimdFromValues(loc, forOp.getResult(i).getType(), nf.getResults(), cur, &protos[i]));
        }
        return true;
    }

    bool packetizeCall(mlir::LLVM::CallOp call, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = call.getLoc();
        auto callee = call.getCallee();
        if (!callee) {
            outError = "Indirect call inside DXIL shader is not supported";
            return false;
        }
        if (*callee == "nacelle.groupshared.ptr") {
            uint64_t byteOff = getConstIntValue(call.getOperand(0)).value_or(0);
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            sv.lanes.push_back(gepByteOffset(loc, hooks_.groupSharedBase, static_cast<int64_t>(byteOff)));
            setMapped(call.getResult(), sv);
            return true;
        }
        if (callee->starts_with("dx.op.")) {
            uint32_t opNum = static_cast<uint32_t>(getConstIntValue(call.getOperand(0)).value_or(0));
            return packetizeDxOp(call, opNum, execMask, realMask, outError);
        }
        outError = "Unsupported call inside DXIL shader: " + callee->str();
        return false;
    }

    // --- Resource Handle Resolution ---
    SimdVal resolveStaticBinding(mlir::Location loc, uint32_t resClass, uint32_t space, uint32_t lowerBound,
                                 uint32_t resourceKind, uint32_t stride, mlir::Value indexVal, mlir::Type origTy,
                                 mlir::Value execMask) {
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
            h.lanes.push_back(gepByteOffset(loc, rootCtxPtr_, off));
            return h;
        }
        if (tgt.kind == RootBindingTarget::RootDescriptor) {
            int64_t off = static_cast<int64_t>(offsetof(DriverRootContext, inline_descriptors)) +
                          static_cast<int64_t>(tgt.rootParamIndex) * 64;
            h.lanes.push_back(gepByteOffset(loc, rootCtxPtr_, off));
            return h;
        }
        if (tgt.kind == RootBindingTarget::StaticSampler) {
            int64_t off = static_cast<int64_t>(offsetof(DriverRootContext, static_samplers)) +
                          static_cast<int64_t>(tgt.staticSamplerIndex) * 64;
            h.lanes.push_back(gepByteOffset(loc, rootCtxPtr_, off));
            return h;
        }
        if (tgt.kind == RootBindingTarget::DescriptorTableOffset) {
            int64_t slotOff = static_cast<int64_t>(offsetof(DriverRootContext, root_slots)) +
                              static_cast<int64_t>(tgt.rootParamIndex) * 8;
            mlir::Value tblBase = loadInvariantScalar(loc, ptrTy(), gepByteOffset(loc, rootCtxPtr_, slotOff));
            auto idxConst = getConstIntValue(indexVal);
            if (idxConst.has_value()) {
                int64_t regIdx = static_cast<int64_t>(*idxConst);
                int64_t tableIdx = static_cast<int64_t>(tgt.tableOffsetOrConstStart) +
                                   (regIdx - static_cast<int64_t>(lowerBound));
                h.lanes.push_back(gepByteOffset(loc, tblBase, tableIdx * 64));
                return h;
            }
            bool isSamp = (resClass == 3);
            int64_t heapBaseOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_base))
                                         : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_base));
            int64_t heapCntOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_count))
                                        : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_count));
            mlir::Value heapBase = loadInvariantScalar(loc, ptrTy(), gepByteOffset(loc, rootCtxPtr_, heapBaseOff));
            mlir::Value heapCnt = loadInvariantScalar(loc, i32Ty(), gepByteOffset(loc, rootCtxPtr_, heapCntOff));
            mlir::Value heapI64 = mlir::LLVM::PtrToIntOp::create(b_, loc, i64Ty(), heapBase);
            mlir::Value tblI64 = mlir::LLVM::PtrToIntOp::create(b_, loc, i64Ty(), tblBase);
            mlir::Value heapBytes = mlir::LLVM::MulOp::create(
                b_, loc, mlir::LLVM::ZExtOp::create(b_, loc, i64Ty(), heapCnt), cInt(loc, i64Ty(), 64));
            mlir::Value heapEnd = mlir::LLVM::AddOp::create(b_, loc, heapI64, heapBytes);
            mlir::Value heapNonNull = mlir::LLVM::ICmpOp::create(
                b_, loc, mlir::LLVM::ICmpPredicate::ne, heapI64, cInt(loc, i64Ty(), 0));
            mlir::Value tblInHeap = mlir::LLVM::AndOp::create(
                b_, loc, heapNonNull,
                mlir::LLVM::AndOp::create(
                    b_, loc,
                    mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::uge, tblI64, heapI64),
                    mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ult, tblI64, heapEnd)));

            mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            SimdVal idxSimd = getMapped(loc, indexVal);
            mlir::Value idxVec = asVec(loc, idxSimd);
            int64_t bias = static_cast<int64_t>(tgt.tableOffsetOrConstStart) - static_cast<int64_t>(lowerBound);
            for (uint32_t l = 0; l < W_; ++l) {
                if (execMask && isLaneMaskConstFalse(execMask, l)) {
                    h.lanes.push_back(nullPtr);
                    continue;
                }
                mlir::Value idxL = llvmExtract(b_, loc, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value idx64 = mlir::LLVM::SExtOp::create(b_, loc, i64Ty(), idxL);
                mlir::Value slot = mlir::LLVM::AddOp::create(b_, loc, idx64, cInt(loc, i64Ty(), bias));
                mlir::Value byteOff = mlir::LLVM::MulOp::create(b_, loc, slot, cInt(loc, i64Ty(), 64));
                mlir::Value candPtr = gepDynByteOffset(loc, tblBase, byteOff);

                mlir::Value slotNonNeg = mlir::LLVM::ICmpOp::create(
                    b_, loc, mlir::LLVM::ICmpPredicate::sge, slot, cInt(loc, i64Ty(), 0));
                mlir::Value candI64 = mlir::LLVM::PtrToIntOp::create(b_, loc, i64Ty(), candPtr);
                mlir::Value inHeap = mlir::LLVM::AndOp::create(
                    b_, loc,
                    mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::uge, candI64, heapI64),
                    mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ult, candI64, heapEnd));
                // If tblBase is inside the descriptor heap, enforce [heapBase, heapEnd);
                // otherwise allow non-negative slot offsets (e.g. standalone test tables).
                mlir::Value boundsOk = mlir::LLVM::SelectOp::create(b_, loc, tblInHeap, inHeap, slotNonNeg);
                if (execMask) {
                    mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    boundsOk = mlir::LLVM::AndOp::create(b_, loc, mL, boundsOk);
                }
                h.lanes.push_back(mlir::LLVM::SelectOp::create(b_, loc, boundsOk, candPtr, nullPtr));
            }
            return h;
        }

        // Unbound fallback -> null descriptor pointer (bounds checks return 0 safely)
        h.lanes.push_back(mlir::LLVM::ZeroOp::create(b_, loc, ptrTy()));
        return h;
    }

    mlir::Value maskDescPtrForLane(mlir::Location loc, const SimdVal& h, mlir::Value maskVec, uint32_t lane) {
        mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
        if (maskVec && isLaneMaskConstFalse(maskVec, lane)) {
            return nullPtr;
        }
        mlir::Value descP = laneValue(loc, h, lane);
        if (maskVec && h.lanes.size() > 1) {
            mlir::Value mL = llvmExtract(b_, loc, maskVec, static_cast<uint32_t>(static_cast<int64_t>(lane)));
            descP = mlir::LLVM::SelectOp::create(b_, loc, mL, descP, nullPtr);
        }
        return descP;
    }

    // Loads NacelleDescriptor fields safely (if descPtr is null, returns 0)
    mlir::Value loadDescBasePtr(mlir::Location loc, mlir::Value descPtr) {
        mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
        mlir::Value isNull = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, descPtr, nullPtr);
        return mlir::scf::IfOp::create(
                   b_, loc, isNull,
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy())});
                   },
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       auto p = mlir::LLVM::LoadOp::create(nb, nloc, ptrTy(), descPtr);
                       p.setInvariant(true);
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{p});
                   })
            .getResult(0);
    }

    mlir::Value loadDescU32Field(mlir::Location loc, mlir::Value descPtr, int64_t byteOff, uint32_t fallback = 0) {
        mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
        mlir::Value isNull = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, descPtr, nullPtr);
        return mlir::scf::IfOp::create(
                   b_, loc, isNull,
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), fallback))});
                   },
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::Value fp = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), descPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{byteOff});
                       auto val = mlir::LLVM::LoadOp::create(nb, nloc, i32Ty(), fp);
                       val.setInvariant(true);
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{val});
                   })
            .getResult(0);
    }

    mlir::Value loadDescU64Field(mlir::Location loc, mlir::Value descPtr, int64_t byteOff) {
        mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
        mlir::Value isNull = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, descPtr, nullPtr);
        return mlir::scf::IfOp::create(
                   b_, loc, isNull,
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, i64Ty(), nb.getIntegerAttr(i64Ty(), 0))});
                   },
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::Value fp = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), descPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{byteOff});
                       auto val = mlir::LLVM::LoadOp::create(nb, nloc, i64Ty(), fp);
                       val.setInvariant(true);
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{val});
                   })
            .getResult(0);
    }

    bool packetizeDxOp(mlir::LLVM::CallOp call, uint32_t op, mlir::Value execMask, mlir::Value& realMask, std::string& outError) {
        mlir::Location loc = call.getLoc();
        auto arg = [&](unsigned i) -> mlir::Value { return i < call.getNumOperands() ? call.getOperand(i) : mlir::Value(); };
        auto resTy = [&]() -> mlir::Type { return call.getNumResults() ? call.getResult().getType() : mlir::Type(); };

        // 1. Thread / Group system values (93..96)
        if (op == 93 || op == 94 || op == 95) {
            uint32_t comp = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            mlir::Value v = (op == 93) ? (comp < hooks_.threadId.size() ? hooks_.threadId[comp] : splatInt(loc, i32Ty(), 0))
                            : (op == 94) ? (comp < hooks_.groupId.size() ? hooks_.groupId[comp] : splatInt(loc, i32Ty(), 0))
                                         : (comp < hooks_.threadIdInGroup.size() ? hooks_.threadIdInGroup[comp] : splatInt(loc, i32Ty(), 0));
            SimdVal sv{SimdVal::Kind::Vector, i32Ty(), v, {}, {}};
            if (op == 94) {
                sv.pattern = SimdVal::LanePattern::Uniform;
            } else if ((op == 93 || op == 95) && comp == 0 && !hooks_.isPixelShader) {
                sv.pattern = SimdVal::LanePattern::UnitStride;
            }
            setMapped(call.getResult(), std::move(sv));
            return true;
        }
        if (op == 96) {
            mlir::Value v = hooks_.flatGroupId ? hooks_.flatGroupId : splatInt(loc, i32Ty(), 0);
            SimdVal sv{SimdVal::Kind::Vector, i32Ty(), v, {}, {}};
            sv.pattern = SimdVal::LanePattern::Uniform;
            setMapped(call.getResult(), std::move(sv));
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
            setMapped(call.getResult(), resolveStaticBinding(loc, resClass, space, lower, kind, stride, arg(3), resTy(), execMask));
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
                SimdVal rb = getMapped(loc, arg(1));
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
            setMapped(call.getResult(), resolveStaticBinding(loc, resClass, space, lower, kind, stride, arg(2), resTy(), execMask));
            return true;
        }
        if (op == 218) {
            // createHandleFromHeap(op, index, samplerHeap, nonUniform)
            bool isSamp = getConstIntValue(arg(2)).value_or(0) != 0;
            int64_t baseOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_base))
                                     : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_base));
            int64_t cntOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_count))
                                    : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_count));
            mlir::Value heapBase = loadInvariantScalar(loc, ptrTy(), gepByteOffset(loc, rootCtxPtr_, baseOff));
            mlir::Value heapCnt = loadInvariantScalar(loc, i32Ty(), gepByteOffset(loc, rootCtxPtr_, cntOff));

            SimdVal h;
            h.kind = SimdVal::Kind::Handle;
            h.origType = resTy();
            h.resClass = isSamp ? 3u : 0u;
            if (auto idxC = getConstIntValue(arg(1))) {
                mlir::Value ok = mlir::LLVM::ICmpOp::create(
                    b_, loc, mlir::LLVM::ICmpPredicate::ult, cInt(loc, i32Ty(), *idxC), heapCnt);
                mlir::Value p = mlir::scf::IfOp::create(
                                    b_, loc, ok,
                                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{
                                            mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), heapBase,
                                                                      llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>((*idxC) * 64)})});
                                    },
                                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy())});
                                    })
                                    .getResult(0);
                h.lanes.push_back(p);
                setMapped(call.getResult(), h);
                return true;
            }
            mlir::Value idxVec = asVec(loc, getMapped(loc, arg(1)));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value idxL = llvmExtract(b_, loc, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value ok = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ult, idxL, heapCnt);
                mlir::Value p = mlir::scf::IfOp::create(
                                    b_, loc, ok,
                                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                        mlir::Value idx64 = mlir::LLVM::ZExtOp::create(nb, nloc, i64Ty(), idxL);
                                        mlir::Value bOff = mlir::LLVM::MulOp::create(
                                            nb, nloc, idx64, mlir::LLVM::ConstantOp::create(nb, nloc, i64Ty(), nb.getIntegerAttr(i64Ty(), 64)));
                                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{
                                            mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), heapBase, mlir::ValueRange{bOff})});
                                    },
                                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy())});
                                    })
                                    .getResult(0);
                h.lanes.push_back(p);
            }
            setMapped(call.getResult(), h);
            return true;
        }
        if (op == 160) {
            // createHandleForLib(op, Resource) -> trace Resource load/GEP back to global symbol
            mlir::Value ptrVal = arg(1);
            if (auto loadOp = ptrVal.getDefiningOp<mlir::LLVM::LoadOp>()) {
                ptrVal = loadOp.getAddr();
            }
            mlir::Value indexVal;
            if (auto gepOp = ptrVal.getDefiningOp<mlir::LLVM::GEPOp>()) {
                auto dynIdx = gepOp.getDynamicIndices();
                if (!dynIdx.empty()) {
                    indexVal = dynIdx.back();
                } else {
                    auto rawIdx = gepOp.getRawConstantIndices();
                    if (!rawIdx.empty() && rawIdx.back() != mlir::LLVM::GEPOp::kDynamicIndex) {
                        indexVal = cInt(loc, i32Ty(), static_cast<uint64_t>(rawIdx.back()));
                    }
                }
                ptrVal = gepOp.getBase();
            }
            std::string symName;
            if (auto addrOf = ptrVal.getDefiningOp<mlir::LLVM::AddressOfOp>()) {
                symName = addrOf.getGlobalName().str();
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
            mlir::Value regIdx = indexVal;
            if (!regIdx) {
                regIdx = cInt(loc, i32Ty(), rec->lowerBound);
            } else {
                // GEP offset is relative to lowerBound
                if (auto cIdx = getConstIntValue(regIdx)) {
                    regIdx = cInt(loc, i32Ty(), rec->lowerBound + static_cast<uint32_t>(*cIdx));
                }
            }
            setMapped(call.getResult(),
                      resolveStaticBinding(loc, rec->resClass, rec->space, rec->lowerBound,
                                           rec->resourceKind, rec->structuredStride, regIdx, resTy(), execMask));
            return true;
        }
        if (op == 216) {
            // annotateHandle(op, handle, ResourceProperties{i32 kindFlags, i32 strideOrFmt})
            SimdVal h = getMapped(loc, arg(1));
            uint32_t kindFlags = 0, strideOrFmt = 0;
            llvm::SmallVector<uint64_t, 4> propFields;
            if (extractConstStructInts(arg(2), propFields) && propFields.size() >= 2) {
                kindFlags = static_cast<uint32_t>(propFields[0]);
                strideOrFmt = static_cast<uint32_t>(propFields[1]);
            } else {
                SimdVal rp = getMapped(loc, arg(2));
                if (rp.fields.size() >= 2) {
                    kindFlags = static_cast<uint32_t>(getConstIntValue(rp.fields[0].vec).value_or(0));
                    strideOrFmt = static_cast<uint32_t>(getConstIntValue(rp.fields[1].vec).value_or(0));
                }
            }
            if ((kindFlags & 0xFFu) != 0) h.resourceKind = kindFlags & 0xFFu;
            if ((kindFlags & 0xFFu) == 12u && strideOrFmt > 0) h.stride = strideOrFmt;
            if ((kindFlags & 0x1000u) != 0) h.resClass = 1u;
            setMapped(call.getResult(), h);
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
            SimdVal aSv = getMapped(loc, arg(1));
            SimdVal bSv = getMapped(loc, arg(2));
            mlir::Type resElem = call.getResult().getType();
            mlir::Type vT = vecTy(resElem);
            if (aSv.kind != SimdVal::Kind::Struct || bSv.kind != SimdVal::Kind::Struct || aSv.fields.empty()) {
                setMapped(call.getResult(), makeZeroSimd(loc, resElem));
                return true;
            }
            size_t n = std::min(aSv.fields.size(), bSv.fields.size());
            mlir::Value acc = mlir::LLVM::FMulOp::create(
                b_, loc, vT, asVec(loc, aSv.fields[0]), asVec(loc, bSv.fields[0]));
            for (size_t i = 1; i < n; ++i) {
                mlir::Value prod = mlir::LLVM::FMulOp::create(
                    b_, loc, vT, asVec(loc, aSv.fields[i]), asVec(loc, bSv.fields[i]));
                acc = mlir::LLVM::FAddOp::create(b_, loc, vT, acc, prod);
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, resElem, acc, {}, {}});
            return true;
        }
        if (op == 70) {
            // bufferUpdateCounter: extra_ptr holds atomic counter ptr if bound.
            // DXIL passes inc as i8 (+1 for IncrementCounter, -1 for DecrementCounter);
            // sign-extend from i8 and return pre-op value for increment, post-op value for decrement.
            SimdVal h = getMapped(loc, arg(1));
            int32_t inc = static_cast<int32_t>(static_cast<int8_t>(getConstIntValue(arg(2)).value_or(1)));
            mlir::Value outVec = splatInt(loc, i32Ty(), 0);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value descP = laneValue(loc, h, l);
                mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
                mlir::Value isValid = mlir::LLVM::AndOp::create(
                    b_, loc, mL, mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, descP, nullP));
                auto r = mlir::scf::IfOp::create(
                    b_, loc, isValid,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value ep = mlir::LLVM::GEPOp::create(
                            nb, nloc, ptrTy(), i8Ty(), descP,
                            llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(offsetof(NacelleDescriptor, extra_ptr))});
                        mlir::Value cptr = mlir::LLVM::LoadOp::create(nb, nloc, ptrTy(), ep);
                        mlir::Value cValid = mlir::LLVM::ICmpOp::create(
                            nb, nloc, mlir::LLVM::ICmpPredicate::ne, cptr, mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()));
                        auto cr = mlir::scf::IfOp::create(
                            nb, nloc, cValid,
                            [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                                mlir::Value incVal = mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), inc));
                                mlir::Value old = mlir::LLVM::AtomicRMWOp::create(
                                    ib, iloc, mlir::LLVM::AtomicBinOp::add, cptr, incVal,
                                    mlir::LLVM::AtomicOrdering::acq_rel);
                                mlir::Value retVal = (inc < 0)
                                    ? mlir::LLVM::AddOp::create(ib, iloc, old, incVal).getResult()
                                    : old;
                                mlir::scf::YieldOp::create(ib, iloc, mlir::ValueRange{retVal});
                            },
                            [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                                mlir::scf::YieldOp::create(ib, iloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 0))});
                            });
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{cr.getResult(0)});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), 0))});
                    });
                outVec = llvmInsert(b_, loc, outVec, r.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), outVec, {}, {}});
            return true;
        }
        if (op == 71) { // CheckAccessFullyMapped(status) -> i1
            mlir::Value stVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value mapped = mlir::LLVM::ICmpOp::create(
                b_, loc, mlir::LLVM::ICmpPredicate::ne, stVec, splatInt(loc, i32Ty(), 0));
            setMapped(call.getResult(), {SimdVal::Kind::Vector, b_.getI1Type(), mapped, {}, {}});
            return true;
        }

        // 5. Atomics (78, 79) & Memory Fences without GroupSync (80, 244, 245)
        if (op == 78 || op == 79) return packetizeAtomicOp(call, op == 79, realMask, outError);
        if (op == 80 || op == 244 || op == 245) {
            mlir::LLVM::FenceOp::create(b_, loc, mlir::LLVM::AtomicOrdering::acq_rel);
            return true;
        }

        // 6. Textures (60..67, 72, 73, 74, 81)
        if ((op >= 60 && op <= 67) || op == 72 || op == 73 || op == 74 || op == 81) {
            return packetizeTextureOp(call, op, execMask, realMask, outError);
        }

        // 7. VS / GS / HS / DS / PS Input/Output, Stream, & Discard
        if (op == 4) { // loadInput(sigId, row, col, gsVertexAxis)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal res;
            bool ok = false;
            if (hooks_.loadInputWithVertex) {
                SimdVal vAxis = (call.getNumOperands() > 4) ? getMapped(loc, arg(4)) : makeZeroSimd(loc, i32Ty());
                ok = hooks_.loadInputWithVertex(b_, loc, sigId, row, col, vAxis, resTy(), res);
            } else if (hooks_.loadInput) {
                ok = hooks_.loadInput(b_, loc, sigId, row, col, resTy(), res);
            }
            if (!ok) res = makeZeroSimd(loc, resTy());
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 5) { // storeOutput
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(loc, arg(4));
            if (hooks_.storeOutput) {
                if (!hooks_.storeOutput(b_, loc, sigId, row, col, val, realMask)) return false;
            }
            return true;
        }
        if (op == 97 || op == 98 || op == 99) { // emitStream / cutStream / emitThenCutStream
            uint32_t streamId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            if ((op == 97 || op == 99) && hooks_.emitStream) {
                if (!hooks_.emitStream(b_, loc, streamId, realMask)) return false;
            }
            if ((op == 98 || op == 99) && hooks_.cutStream) {
                if (!hooks_.cutStream(b_, loc, streamId, realMask)) return false;
            }
            return true;
        }
        if (op == 100) { // gsInstanceID
            SimdVal res;
            if (!hooks_.loadGSInstanceId || !hooks_.loadGSInstanceId(b_, loc, resTy(), res)) {
                res = makeZeroSimd(loc, resTy());
            }
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 103) { // loadOutputControlPoint(sigId, row, col, cpId)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal cpId = getMapped(loc, arg(4));
            SimdVal res;
            if (!hooks_.loadOutputControlPoint ||
                !hooks_.loadOutputControlPoint(b_, loc, sigId, row, col, cpId, resTy(), res)) {
                res = makeZeroSimd(loc, resTy());
            }
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 104) { // loadPatchConstant(sigId, row, col)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal res;
            if (!hooks_.loadPatchConstant ||
                !hooks_.loadPatchConstant(b_, loc, sigId, row, col, resTy(), res)) {
                res = makeZeroSimd(loc, resTy());
            }
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 105) { // domainLocation(col)
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal res;
            if (!hooks_.loadDomainLocation || !hooks_.loadDomainLocation(b_, loc, col, resTy(), res)) {
                res = makeZeroSimd(loc, resTy());
            }
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 106) { // storePatchConstant(sigId, row, col, val)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(loc, arg(4));
            if (hooks_.storePatchConstant) {
                if (!hooks_.storePatchConstant(b_, loc, sigId, row, col, val, realMask)) return false;
            }
            return true;
        }
        if (op == 107) { // outputControlPointID
            SimdVal res;
            if (!hooks_.loadOutputControlPointId || !hooks_.loadOutputControlPointId(b_, loc, resTy(), res)) {
                res = makeZeroSimd(loc, resTy());
            }
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 108) { // primitiveID
            SimdVal res;
            if (!hooks_.loadPrimitiveId || !hooks_.loadPrimitiveId(b_, loc, resTy(), res)) {
                res = makeZeroSimd(loc, resTy());
            }
            setMapped(call.getResult(), res);
            return true;
        }
        if (op == 82) { // discard(cond)
            mlir::Value condVec = asVec(loc, getMapped(loc, arg(1)));
            if (hooks_.onDiscard) {
                hooks_.onDiscard(b_, loc, condVec, execMask, realMask);
            } else {
                mlir::Value killed = mlir::LLVM::AndOp::create(b_, loc, execMask, condVec);
                mlir::Value keep = mlir::LLVM::XOrOp::create(b_, loc, killed, allTrueMask(loc));
                realMask = mlir::LLVM::AndOp::create(b_, loc, realMask, keep);
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
            SimdVal nv = getMapped(loc, arg(1));
            SimdVal np = getMapped(loc, arg(2));
            if (hooks_.setMeshOutputCounts) {
                if (!hooks_.setMeshOutputCounts(b_, loc, nv, np, realMask)) return false;
            }
            return true;
        }
        if (op == 169) { // EmitIndices(op, primIdx, v0, v1, v2)
            SimdVal pIdx = getMapped(loc, arg(1));
            SimdVal v0 = getMapped(loc, arg(2));
            SimdVal v1 = getMapped(loc, arg(3));
            SimdVal v2 = (call.getNumOperands() > 4) ? getMapped(loc, arg(4)) : makeZeroSimd(loc, i32Ty());
            if (hooks_.emitIndices) {
                if (!hooks_.emitIndices(b_, loc, pIdx, v0, v1, v2, realMask)) return false;
            }
            return true;
        }
        if (op == 170) { // GetMeshPayload(op) -> ptr
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = ptrTy();
            sv.lanes.push_back(hooks_.meshPayloadBase ? hooks_.meshPayloadBase
                                                      : mlir::Value(mlir::LLVM::ZeroOp::create(b_, loc, ptrTy())));
            setMapped(call.getResult(), sv);
            return true;
        }
        if (op == 171) { // StoreVertexOutput(op, sigId, row, col, val, vertIdx)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(loc, arg(4));
            SimdVal vIdx = getMapped(loc, arg(5));
            if (hooks_.storeVertexOutput) {
                if (!hooks_.storeVertexOutput(b_, loc, sigId, row, col, val, vIdx, realMask)) return false;
            }
            return true;
        }
        if (op == 172) { // StorePrimitiveOutput(op, sigId, row, col, val, primIdx)
            uint32_t sigId = static_cast<uint32_t>(getConstIntValue(arg(1)).value_or(0));
            SimdVal row = getMapped(loc, arg(2));
            uint32_t col = static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0));
            SimdVal val = getMapped(loc, arg(4));
            SimdVal pIdx = getMapped(loc, arg(5));
            if (hooks_.storePrimitiveOutput) {
                if (!hooks_.storePrimitiveOutput(b_, loc, sigId, row, col, val, pIdx, realMask)) return false;
            }
            return true;
        }
        if (op == 173) { // DispatchMesh(op, countX, countY, countZ, payloadPtr)
            SimdVal cx = getMapped(loc, arg(1));
            SimdVal cy = getMapped(loc, arg(2));
            SimdVal cz = getMapped(loc, arg(3));
            SimdVal pp = (call.getNumOperands() > 4) ? getMapped(loc, arg(4)) : makeZeroSimd(loc, ptrTy());
            if (hooks_.dispatchMesh) {
                if (!hooks_.dispatchMesh(b_, loc, cx, cy, cz, pp, realMask)) return false;
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
        mlir::Location loc,
        mlir::OpBuilder& nb,
        const SimdVal& h,
        mlir::Value descPtr,
        bool needElemStride,
        bool needStride64) {
        mlir::Block* entryBlock = getFuncEntryBlock();
        const bool canCache = (entryBlock != nullptr) && (descPtr.getParentBlock() == entryBlock);
        DescInfoCacheKey key{descPtr, h.isRootConstants, h.stride, needElemStride, needStride64};
        if (canCache) {
            for (const auto& entry : *descCache_) {
                if (entry.first == key) return entry.second;
            }
        }

        mlir::OpBuilder::InsertionGuard guard(nb);
        if (canCache) {
            if (auto* defOp = descPtr.getDefiningOp()) {
                nb.setInsertionPointAfter(defOp);
            } else {
                nb.setInsertionPointToStart(entryBlock);
            }
        }

        BufferDescInfo info;
        info.descP = descPtr;
        if (h.isRootConstants) {
            info.basePtr = descPtr;
            info.byteSize = mlir::LLVM::ConstantOp::create(nb, loc, i64Ty(), nb.getIntegerAttr(i64Ty(), UINT64_MAX));
            info.isReserved = mlir::LLVM::ConstantOp::create(nb, loc, i1Ty(), nb.getIntegerAttr(i1Ty(), 0));
            mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(nb, loc, ptrTy());
            info.okPtr = mlir::LLVM::ICmpOp::create(nb, loc, mlir::LLVM::ICmpPredicate::ne, descPtr, nullPtr);
            if (needElemStride) info.elemStride = mlir::LLVM::ConstantOp::create(nb, loc, i32Ty(), nb.getIntegerAttr(i32Ty(), 0));
            if (needStride64) info.stride64 = mlir::LLVM::ConstantOp::create(nb, loc, i64Ty(), nb.getIntegerAttr(i64Ty(), h.stride));
            return info;
        }

        bool loadEs = needElemStride || (needStride64 && h.stride == 0);
        mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(nb, loc, ptrTy());
        mlir::Value isNull = mlir::LLVM::ICmpOp::create(nb, loc, mlir::LLVM::ICmpPredicate::eq, descPtr, nullPtr);
        auto ifOp = mlir::scf::IfOp::create(
            nb, loc, isNull,
            [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                llvm::SmallVector<mlir::Value, 4> z = {
                    mlir::LLVM::ZeroOp::create(ib, iloc, ptrTy()),
                    mlir::LLVM::ConstantOp::create(ib, iloc, i64Ty(), ib.getIntegerAttr(i64Ty(), 0)),
                    mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 0))};
                if (loadEs) z.push_back(mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 0)));
                mlir::scf::YieldOp::create(ib, iloc, z);
            },
            [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                auto p = mlir::LLVM::LoadOp::create(ib, iloc, ptrTy(), descPtr);
                p.setInvariant(true);
                mlir::Value bsp = mlir::LLVM::GEPOp::create(
                    ib, iloc, ptrTy(), i8Ty(), descPtr,
                    llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(offsetof(NacelleDescriptor, byte_size))});
                auto bs = mlir::LLVM::LoadOp::create(ib, iloc, i64Ty(), bsp);
                bs.setInvariant(true);
                mlir::Value vfp = mlir::LLVM::GEPOp::create(
                    ib, iloc, ptrTy(), i8Ty(), descPtr,
                    llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(offsetof(NacelleDescriptor, view_flags))});
                auto vf = mlir::LLVM::LoadOp::create(ib, iloc, i32Ty(), vfp);
                vf.setInvariant(true);
                llvm::SmallVector<mlir::Value, 4> vals = {p, bs, vf};
                if (loadEs) {
                    mlir::Value esp = mlir::LLVM::GEPOp::create(
                        ib, iloc, ptrTy(), i8Ty(), descPtr,
                        llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(offsetof(NacelleDescriptor, elem_stride))});
                    auto es = mlir::LLVM::LoadOp::create(ib, iloc, i32Ty(), esp);
                    es.setInvariant(true);
                    vals.push_back(es);
                }
                mlir::scf::YieldOp::create(ib, iloc, vals);
            });
        info.basePtr = ifOp.getResult(0);
        info.byteSize = ifOp.getResult(1);
        mlir::Value vf = ifOp.getResult(2);
        mlir::Value flag = mlir::LLVM::AndOp::create(
            nb, loc, vf, mlir::LLVM::ConstantOp::create(nb, loc, i32Ty(), nb.getIntegerAttr(i32Ty(), kDescViewFlagReservedBuffer)));
        info.isReserved = mlir::LLVM::ICmpOp::create(
            nb, loc, mlir::LLVM::ICmpPredicate::ne, flag, mlir::LLVM::ConstantOp::create(nb, loc, i32Ty(), nb.getIntegerAttr(i32Ty(), 0)));
        info.okPtr = mlir::LLVM::ICmpOp::create(nb, loc, mlir::LLVM::ICmpPredicate::ne, info.basePtr, nullPtr);
        mlir::Value loadedEs = loadEs ? ifOp.getResult(3) : mlir::Value();
        if (needElemStride) info.elemStride = loadedEs;
        if (needStride64) {
            info.stride64 = (h.stride > 0)
                                ? mlir::Value(mlir::LLVM::ConstantOp::create(nb, loc, i64Ty(), nb.getIntegerAttr(i64Ty(), h.stride)))
                                : mlir::Value(mlir::LLVM::ZExtOp::create(nb, loc, i64Ty(), loadedEs));
        }
        if (canCache) {
            descCache_->push_back({key, info});
        }
        return info;
    }

    mlir::Value resolveReservedBufferElemPtr(
        mlir::OpBuilder& nb,
        mlir::Location nloc,
        mlir::Value descP,
        mlir::Value tileTablePtr,
        mlir::Value cOff,
        uint32_t elemBytes) {
        auto ci64 = [&](uint64_t v) { return mlir::LLVM::ConstantOp::create(nb, nloc, i64Ty(), nb.getIntegerAttr(i64Ty(), v)); };
        auto loadU32 = [&](int64_t fOff) -> mlir::Value {
            mlir::Value fp = mlir::LLVM::GEPOp::create(
                nb, nloc, ptrTy(), i8Ty(), descP,
                llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(fOff)});
            return mlir::LLVM::LoadOp::create(nb, nloc, i32Ty(), fp);
        };
        mlir::Value lo = mlir::LLVM::ZExtOp::create(nb, nloc, i64Ty(), loadU32(offsetof(NacelleDescriptor, row_pitch)));
        mlir::Value hi = mlir::LLVM::ZExtOp::create(nb, nloc, i64Ty(), loadU32(offsetof(NacelleDescriptor, depth_or_array)));
        mlir::Value viewOff = mlir::LLVM::OrOp::create(
            nb, nloc, mlir::LLVM::ShlOp::create(nb, nloc, hi, ci64(32)), lo);
        mlir::Value resOff = mlir::LLVM::AddOp::create(nb, nloc, viewOff, cOff);
        mlir::Value tileIdx = mlir::LLVM::LShrOp::create(nb, nloc, resOff, ci64(16));
        mlir::Value endTileIdx =
            (elemBytes > 1)
                ? mlir::Value(mlir::LLVM::LShrOp::create(
                      nb, nloc, mlir::LLVM::AddOp::create(nb, nloc, resOff, ci64(elemBytes - 1)), ci64(16)))
                : tileIdx;
        mlir::Value totalTiles64 =
            mlir::LLVM::ZExtOp::create(nb, nloc, i64Ty(), loadU32(offsetof(NacelleDescriptor, height)));
        mlir::Value validTile = mlir::LLVM::AndOp::create(
            nb, nloc,
            mlir::LLVM::ICmpOp::create(nb, nloc, mlir::LLVM::ICmpPredicate::ult, tileIdx, totalTiles64),
            mlir::LLVM::ICmpOp::create(nb, nloc, mlir::LLVM::ICmpPredicate::eq, tileIdx, endTileIdx));

        return mlir::scf::IfOp::create(
                   nb, nloc, validTile,
                   [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                       mlir::Value slotByteOff = mlir::LLVM::ShlOp::create(
                           tb, tloc, tileIdx, mlir::LLVM::ConstantOp::create(tb, tloc, i64Ty(), tb.getIntegerAttr(i64Ty(), 3)));
                       mlir::Value slotPtr = mlir::LLVM::GEPOp::create(
                           tb, tloc, ptrTy(), i8Ty(), tileTablePtr, mlir::ValueRange{slotByteOff});
                       mlir::Value tileBase = mlir::LLVM::LoadOp::create(tb, tloc, ptrTy(), slotPtr);
                       mlir::Value nullP = mlir::LLVM::ZeroOp::create(tb, tloc, ptrTy());
                       mlir::Value hasPage = mlir::LLVM::ICmpOp::create(
                           tb, tloc, mlir::LLVM::ICmpPredicate::ne, tileBase, nullP);
                       mlir::Value elemPtr =
                           mlir::scf::IfOp::create(
                               tb, tloc, hasPage,
                               [&](mlir::OpBuilder& pb, mlir::Location ploc) {
                                   mlir::Value inTile = mlir::LLVM::AndOp::create(
                                       pb, ploc, resOff, mlir::LLVM::ConstantOp::create(pb, ploc, i64Ty(), pb.getIntegerAttr(i64Ty(), 0xFFFF)));
                                   mlir::scf::YieldOp::create(
                                       pb, ploc,
                                       mlir::ValueRange{mlir::LLVM::GEPOp::create(
                                           pb, ploc, ptrTy(), i8Ty(), tileBase, mlir::ValueRange{inTile})});
                               },
                               [&](mlir::OpBuilder& pb, mlir::Location ploc) {
                                   mlir::scf::YieldOp::create(
                                       pb, ploc, mlir::ValueRange{mlir::LLVM::ZeroOp::create(pb, ploc, ptrTy())});
                               })
                               .getResult(0);
                       mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{elemPtr});
                   },
                   [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                       mlir::scf::YieldOp::create(
                           tb, tloc, mlir::ValueRange{mlir::LLVM::ZeroOp::create(tb, tloc, ptrTy())});
                   })
            .getResult(0);
    }

    std::pair<mlir::Value, mlir::Value> emitScalarBufferComponentLoad(
        mlir::Location loc,
        mlir::Type elemTy,
        uint32_t elemBytes,
        mlir::Value descP,
        mlir::Value basePtr,
        mlir::Value isReserved,
        mlir::Value inBounds,
        mlir::Value cOff) {
        auto loadIf = mlir::scf::IfOp::create(
            b_, loc, inBounds,
            [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                auto resIf = mlir::scf::IfOp::create(
                    nb, nloc, isReserved,
                    [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                        mlir::Value ep = resolveReservedBufferElemPtr(rb, rloc, descP, basePtr, cOff, elemBytes);
                        mlir::Value nullP = mlir::LLVM::ZeroOp::create(rb, rloc, ptrTy());
                        mlir::Value mapped = mlir::LLVM::ICmpOp::create(
                            rb, rloc, mlir::LLVM::ICmpPredicate::ne, ep, nullP);
                        mlir::Value v =
                            mlir::scf::IfOp::create(
                                rb, rloc, mapped,
                                [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                    mlir::scf::YieldOp::create(
                                        lb, lloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(lb, lloc, elemTy, ep)});
                                },
                                [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                    SimdPacketizer tmp(lb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                                    mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(lloc, elemTy, 0.0)
                                                                                       : tmp.cInt(lloc, elemTy, 0);
                                    mlir::scf::YieldOp::create(lb, lloc, mlir::ValueRange{z});
                                })
                                .getResult(0);
                        mlir::scf::YieldOp::create(rb, rloc, mlir::ValueRange{v, mapped});
                    },
                    [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                        mlir::Value ep = mlir::LLVM::GEPOp::create(
                            rb, rloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                        mlir::Value v = mlir::LLVM::LoadOp::create(rb, rloc, elemTy, ep);
                        mlir::Value t = mlir::LLVM::ConstantOp::create(rb, rloc, rb.getI1Type(), rb.getIntegerAttr(rb.getI1Type(), 1));
                        mlir::scf::YieldOp::create(rb, rloc, mlir::ValueRange{v, t});
                    });
                mlir::scf::YieldOp::create(nb, nloc, resIf.getResults());
            },
            [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(nloc, elemTy, 0.0)
                                                                   : tmp.cInt(nloc, elemTy, 0);
                mlir::Value t = mlir::LLVM::ConstantOp::create(nb, nloc, nb.getI1Type(), nb.getIntegerAttr(nb.getI1Type(), 1));
                mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z, t});
            });
        return {loadIf.getResult(0), loadIf.getResult(1)};
    }

    void emitScalarBufferComponentStore(
        mlir::Location loc,
        uint32_t elemBytes,
        mlir::Value descP,
        mlir::Value basePtr,
        mlir::Value isReserved,
        mlir::Value doStore,
        mlir::Value cOff,
        mlir::Value valScalar) {
        mlir::scf::IfOp::create(b_, loc, doStore, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
            mlir::scf::IfOp::create(
                nb, nloc, isReserved,
                [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                    mlir::Value ep = resolveReservedBufferElemPtr(rb, rloc, descP, basePtr, cOff, elemBytes);
                    mlir::Value nullP = mlir::LLVM::ZeroOp::create(rb, rloc, ptrTy());
                    mlir::Value mapped = mlir::LLVM::ICmpOp::create(
                        rb, rloc, mlir::LLVM::ICmpPredicate::ne, ep, nullP);
                    mlir::scf::IfOp::create(rb, rloc, mapped, [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                        mlir::LLVM::StoreOp::create(sb, sloc, valScalar, ep);
                        mlir::scf::YieldOp::create(sb, sloc);
                    });
                    mlir::scf::YieldOp::create(rb, rloc);
                },
                [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                    mlir::Value ep = mlir::LLVM::GEPOp::create(
                        rb, rloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                    mlir::LLVM::StoreOp::create(rb, rloc, valScalar, ep);
                    mlir::scf::YieldOp::create(rb, rloc);
                });
            mlir::scf::YieldOp::create(nb, nloc);
        });
    }

    std::pair<mlir::Value, mlir::Value> emitReservedElemLoad(
        mlir::OpBuilder& rb,
        mlir::Location rloc,
        mlir::Type elemTy,
        uint32_t elemBytes,
        mlir::Value descP,
        mlir::Value tileTablePtr,
        mlir::Value inBounds,
        mlir::Value cOff) {
        auto makeZero = [&](mlir::OpBuilder& zb, mlir::Location zloc) -> mlir::Value {
            if (mlir::isa<mlir::FloatType>(elemTy)) {
                auto ft = llvm::cast<mlir::FloatType>(elemTy);
                return mlir::LLVM::ConstantOp::create(zb, zloc, ft, mlir::FloatAttr::get(ft, mlir::FloatAttr::get(ft, 0.0).getValue()));
            }
            return mlir::LLVM::ConstantOp::create(zb, zloc, elemTy, zb.getIntegerAttr(elemTy, 0));
        };
        auto loadIf = mlir::scf::IfOp::create(
            rb, rloc, inBounds,
            [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                mlir::Value ep = resolveReservedBufferElemPtr(tb, tloc, descP, tileTablePtr, cOff, elemBytes);
                mlir::Value nullP = mlir::LLVM::ZeroOp::create(tb, tloc, ptrTy());
                mlir::Value mapped = mlir::LLVM::ICmpOp::create(
                    tb, tloc, mlir::LLVM::ICmpPredicate::ne, ep, nullP);
                mlir::Value v =
                    mlir::scf::IfOp::create(
                        tb, tloc, mapped,
                        [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                            mlir::scf::YieldOp::create(
                                lb, lloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(lb, lloc, elemTy, ep)});
                        },
                        [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                            mlir::scf::YieldOp::create(lb, lloc, mlir::ValueRange{makeZero(lb, lloc)});
                        })
                        .getResult(0);
                mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{v, mapped});
            },
            [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                mlir::Value t = mlir::LLVM::ConstantOp::create(tb, tloc, i1Ty(), tb.getIntegerAttr(i1Ty(), 1));
                mlir::scf::YieldOp::create(tb, tloc, mlir::ValueRange{makeZero(tb, tloc), t});
            });
        return {loadIf.getResult(0), loadIf.getResult(1)};
    }

    void emitReservedElemStore(
        mlir::OpBuilder& rb,
        mlir::Location rloc,
        uint32_t elemBytes,
        mlir::Value descP,
        mlir::Value tileTablePtr,
        mlir::Value doStore,
        mlir::Value cOff,
        mlir::Value valScalar) {
        mlir::scf::IfOp::create(rb, rloc, doStore, [&](mlir::OpBuilder& tb, mlir::Location tloc) {
            mlir::Value ep = resolveReservedBufferElemPtr(tb, tloc, descP, tileTablePtr, cOff, elemBytes);
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(tb, tloc, ptrTy());
            mlir::Value mapped = mlir::LLVM::ICmpOp::create(
                tb, tloc, mlir::LLVM::ICmpPredicate::ne, ep, nullP);
            mlir::scf::IfOp::create(tb, tloc, mapped, [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                mlir::LLVM::StoreOp::create(sb, sloc, valScalar, ep);
                mlir::scf::YieldOp::create(sb, sloc);
            });
            mlir::scf::YieldOp::create(tb, tloc);
        });
    }

    mlir::Value computeBufferByteOffsetScalar(
        mlir::Location loc, mlir::OpBuilder& nb, const SimdVal& h, mlir::Value stride64,
        mlir::Value idxL, mlir::Value elemOffL, bool isRawOp, bool hasElemOff) {
        mlir::Value idx64 = mlir::LLVM::ZExtOp::create(nb, loc, i64Ty(), idxL);
        if (isRawOp) {
            if (!hasElemOff) return idx64;
            mlir::Value e64 = mlir::LLVM::ZExtOp::create(nb, loc, i64Ty(), elemOffL);
            return mlir::LLVM::AddOp::create(
                nb, loc, mlir::LLVM::MulOp::create(nb, loc, idx64, stride64), e64);
        }
        if (hasElemOff) {
            mlir::Value e64 = mlir::LLVM::ZExtOp::create(nb, loc, i64Ty(), elemOffL);
            return mlir::LLVM::AddOp::create(
                nb, loc, mlir::LLVM::MulOp::create(nb, loc, idx64, stride64), e64);
        }
        if (h.resourceKind == 11) return idx64;
        return mlir::LLVM::MulOp::create(nb, loc, idx64, stride64);
    }

    bool tryGetUniformScalar(mlir::Location loc, mlir::Value origVal, mlir::Value& outScalar) {
        if (auto c = getConstIntValue(origVal)) {
            outScalar = cInt(loc, i32Ty(), static_cast<int64_t>(*c));
            return true;
        }
        SimdVal sv = getMapped(loc, origVal);
        if (sv.isUniform()) {
            outScalar = !sv.lanes.empty()
                            ? sv.lanes[0]
                            : mlir::Value(llvmExtract(b_, loc, sv.vec, static_cast<uint32_t>(0)));
            return true;
        }
        if (sv.kind == SimdVal::Kind::Vector) {
            if (auto shuf = sv.vec.getDefiningOp<mlir::LLVM::ShuffleVectorOp>()) {
                llvm::ArrayRef<int32_t> mask = shuf.getMask();
                if (!mask.empty() && llvm::all_of(mask, [](int32_t idx) { return idx == 0; })) {
                    if (auto ins = shuf.getV1().getDefiningOp<mlir::LLVM::InsertElementOp>()) {
                        if (getConstIntValue(ins.getPosition()) == 0) {
                            outScalar = ins.getValue();
                            return true;
                        }
                    }
                }
            }
            if (auto cst = sv.vec.getDefiningOp<mlir::LLVM::ConstantOp>()) {
                if (auto dea = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue())) {
                    if (dea.isSplat()) {
                        auto attr = mlir::cast<mlir::TypedAttr>(dea.getSplatValue<mlir::Attribute>());
                        outScalar = mlir::LLVM::ConstantOp::create(b_, loc, attr.getType(), attr);
                        return true;
                    }
                }
            }
        }
        return false;
    }

    std::pair<llvm::SmallVector<mlir::Value, 16>, mlir::Value> emitCoalescedUniformBufferLoad(
        mlir::Location loc,
        mlir::Type elemTy,
        uint32_t elemBytes,
        const BufferDescInfo& info,
        bool isRootConstants,
        bool nonStraddling,
        mlir::Value byteOff,
        unsigned totalComps,
        llvm::ArrayRef<unsigned> activeComps) {
        mlir::Value zeroVec = mlir::isa<mlir::FloatType>(elemTy) ? splatFlt(loc, elemTy, 0.0) : splatInt(loc, elemTy, 0);
        if (activeComps.empty()) {
            return {llvm::SmallVector<mlir::Value, 16>(totalComps, zeroVec), splatInt(loc, i32Ty(), 1)};
        }

        auto makeScalarZero = [&](mlir::OpBuilder& zb, mlir::Location zloc) -> mlir::Value {
            if (mlir::isa<mlir::FloatType>(elemTy)) {
                auto ft = llvm::cast<mlir::FloatType>(elemTy);
                return mlir::LLVM::ConstantOp::create(zb, zloc, ft, mlir::FloatAttr::get(ft, mlir::FloatAttr::get(ft, 0.0).getValue()));
            }
            return mlir::LLVM::ConstantOp::create(zb, zloc, elemTy, zb.getIntegerAttr(elemTy, 0));
        };

        auto compInBounds = [&](mlir::OpBuilder& zb, mlir::Location zloc, unsigned c) -> std::pair<mlir::Value, mlir::Value> {
            mlir::Value cOff = (c == 0)
                                   ? byteOff
                                   : mlir::Value(mlir::LLVM::AddOp::create(
                                         zb, zloc, byteOff,
                                         mlir::LLVM::ConstantOp::create(zb, zloc, i64Ty(), zb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes))));
            if (isRootConstants) return {cOff, info.okPtr};
            mlir::Value cEnd = mlir::LLVM::AddOp::create(
                zb, zloc, cOff, mlir::LLVM::ConstantOp::create(zb, zloc, i64Ty(), zb.getIntegerAttr(i64Ty(), elemBytes)));
            mlir::Value ok = mlir::LLVM::AndOp::create(
                zb, zloc, info.okPtr,
                mlir::LLVM::ICmpOp::create(zb, zloc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize));
            if (info.elemStride) {
                mlir::Value elemEnd = mlir::LLVM::ConstantOp::create(zb, zloc, i32Ty(), zb.getIntegerAttr(i32Ty(), (c + 1) * elemBytes));
                mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
                    zb, zloc, mlir::LLVM::ICmpPredicate::eq, info.elemStride,
                    mlir::LLVM::ConstantOp::create(zb, zloc, i32Ty(), zb.getIntegerAttr(i32Ty(), 0)));
                mlir::Value inElem = mlir::LLVM::ICmpOp::create(
                    zb, zloc, mlir::LLVM::ICmpPredicate::ule, elemEnd, info.elemStride);
                ok = mlir::LLVM::AndOp::create(zb, zloc, ok, mlir::LLVM::OrOp::create(zb, zloc, strideZero, inElem));
            }
            return {cOff, ok};
        };

        auto emitNormalScalarLoads = [&](mlir::OpBuilder& nb, mlir::Location nloc) -> llvm::SmallVector<mlir::Value, 16> {
            auto [lastOff, fullOk] = compInBounds(nb, nloc, activeComps.back());
            (void)lastOff;
            auto fullIf = mlir::scf::IfOp::create(
                nb, nloc, fullOk,
                [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                    mlir::Value baseElemPtr = mlir::LLVM::GEPOp::create(
                        tb, tloc, ptrTy(), i8Ty(), info.basePtr, mlir::ValueRange{byteOff});
                    llvm::SmallVector<mlir::Value, 16> vals;
                    vals.reserve(activeComps.size());
                    for (unsigned c : activeComps) {
                        mlir::Value ep = (c == 0)
                                             ? baseElemPtr
                                             : mlir::Value(mlir::LLVM::GEPOp::create(
                                                   tb, tloc, ptrTy(), i8Ty(), baseElemPtr,
                                                   llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(c * elemBytes)}));
                        vals.push_back(mlir::LLVM::LoadOp::create(tb, tloc, elemTy, ep));
                    }
                    mlir::scf::YieldOp::create(tb, tloc, vals);
                },
                [&](mlir::OpBuilder& eb, mlir::Location eloc) {
                    if (activeComps.size() == 1 || nonStraddling || isRootConstants) {
                        llvm::SmallVector<mlir::Value, 16> zeros(activeComps.size(), makeScalarZero(eb, eloc));
                        mlir::scf::YieldOp::create(eb, eloc, zeros);
                        return;
                    }
                    auto [firstOff, firstOk] = compInBounds(eb, eloc, activeComps.front());
                    auto partialIf = mlir::scf::IfOp::create(
                        eb, eloc, firstOk,
                        [&](mlir::OpBuilder& pb, mlir::Location ploc) {
                            llvm::SmallVector<mlir::Value, 16> vals;
                            vals.reserve(activeComps.size());
                            mlir::Value ep0 = mlir::LLVM::GEPOp::create(
                                pb, ploc, ptrTy(), i8Ty(), info.basePtr, mlir::ValueRange{firstOff});
                            vals.push_back(mlir::LLVM::LoadOp::create(pb, ploc, elemTy, ep0));
                            for (size_t idx = 1; idx + 1 < activeComps.size(); ++idx) {
                                auto [mOff, mOk] = compInBounds(pb, ploc, activeComps[idx]);
                                mlir::Value mv =
                                    mlir::scf::IfOp::create(
                                        pb, ploc, mOk,
                                        [&](mlir::OpBuilder& mb, mlir::Location mloc) {
                                            mlir::Value mep = mlir::LLVM::GEPOp::create(
                                                mb, mloc, ptrTy(), i8Ty(), info.basePtr, mlir::ValueRange{mOff});
                                            mlir::scf::YieldOp::create(
                                                mb, mloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(mb, mloc, elemTy, mep)});
                                        },
                                        [&](mlir::OpBuilder& mb, mlir::Location mloc) {
                                            mlir::scf::YieldOp::create(mb, mloc, mlir::ValueRange{makeScalarZero(mb, mloc)});
                                        })
                                        .getResult(0);
                                vals.push_back(mv);
                            }
                            vals.push_back(makeScalarZero(pb, ploc));
                            mlir::scf::YieldOp::create(pb, ploc, vals);
                        },
                        [&](mlir::OpBuilder& pb, mlir::Location ploc) {
                            llvm::SmallVector<mlir::Value, 16> zeros(activeComps.size(), makeScalarZero(pb, ploc));
                            mlir::scf::YieldOp::create(pb, ploc, zeros);
                        });
                    mlir::scf::YieldOp::create(eb, eloc, partialIf.getResults());
                });
            return llvm::SmallVector<mlir::Value, 16>(fullIf.getResults().begin(), fullIf.getResults().end());
        };

        llvm::SmallVector<mlir::Value, 16> outVecs(totalComps, zeroVec);
        if (isRootConstants) {
            auto scalars = emitNormalScalarLoads(b_, loc);
            for (size_t i = 0; i < activeComps.size(); ++i) {
                outVecs[activeComps[i]] = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(elemTy)).getElementType(), scalars[i]);
            }
            return {outVecs, splatInt(loc, i32Ty(), 1)};
        }

        auto emitPerCompReserved = [&](mlir::OpBuilder& rb, mlir::Location rloc) -> llvm::SmallVector<mlir::Value, 17> {
            mlir::Value allMapped = mlir::LLVM::ConstantOp::create(rb, rloc, i1Ty(), rb.getIntegerAttr(i1Ty(), 1));
            llvm::SmallVector<mlir::Value, 17> yielded;
            yielded.reserve(activeComps.size() + 1);
            for (unsigned c : activeComps) {
                auto [cOff, ok] = compInBounds(rb, rloc, c);
                auto [v, mapped] = emitReservedElemLoad(
                    rb, rloc, elemTy, elemBytes, info.descP, info.basePtr, ok, cOff);
                allMapped = mlir::LLVM::AndOp::create(rb, rloc, allMapped, mapped);
                yielded.push_back(v);
            }
            yielded.push_back(mlir::LLVM::ZExtOp::create(rb, rloc, i32Ty(), allMapped));
            return yielded;
        };

        auto ifRes = mlir::scf::IfOp::create(
            b_, loc, info.isReserved,
            [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                if (activeComps.size() <= 1) {
                    mlir::scf::YieldOp::create(rb, rloc, emitPerCompReserved(rb, rloc));
                    return;
                }
                auto [lastOff, fullOk] = compInBounds(rb, rloc, activeComps.back());
                (void)lastOff;
                uint32_t spanBytes = (activeComps.back() + 1u) * elemBytes;
                auto fastIf = mlir::scf::IfOp::create(
                    rb, rloc, fullOk,
                    [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                        mlir::Value spanPtr = resolveReservedBufferElemPtr(
                            tb, tloc, info.descP, info.basePtr, byteOff, spanBytes);
                        mlir::Value nullP = mlir::LLVM::ZeroOp::create(tb, tloc, ptrTy());
                        mlir::Value samePageMapped = mlir::LLVM::ICmpOp::create(
                            tb, tloc, mlir::LLVM::ICmpPredicate::ne, spanPtr, nullP);
                        auto innerIf = mlir::scf::IfOp::create(
                            tb, tloc, samePageMapped,
                            [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                llvm::SmallVector<mlir::Value, 17> vals;
                                vals.reserve(activeComps.size() + 1);
                                for (unsigned c : activeComps) {
                                    mlir::Value ep = (c == 0)
                                                         ? spanPtr
                                                         : mlir::Value(mlir::LLVM::GEPOp::create(
                                                               lb, lloc, ptrTy(), i8Ty(), spanPtr,
                                                               llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(c * elemBytes)}));
                                    vals.push_back(mlir::LLVM::LoadOp::create(lb, lloc, elemTy, ep));
                                }
                                vals.push_back(mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), 1)));
                                mlir::scf::YieldOp::create(lb, lloc, vals);
                            },
                            [&](mlir::OpBuilder& lb, mlir::Location lloc) {
                                mlir::scf::YieldOp::create(lb, lloc, emitPerCompReserved(lb, lloc));
                            });
                        mlir::scf::YieldOp::create(tb, tloc, innerIf.getResults());
                    },
                    [&](mlir::OpBuilder& tb, mlir::Location tloc) {
                        mlir::scf::YieldOp::create(tb, tloc, emitPerCompReserved(tb, tloc));
                    });
                mlir::scf::YieldOp::create(rb, rloc, fastIf.getResults());
            },
            [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                auto vals = emitNormalScalarLoads(rb, rloc);
                vals.push_back(mlir::LLVM::ConstantOp::create(rb, rloc, i32Ty(), rb.getIntegerAttr(i32Ty(), 1)));
                mlir::scf::YieldOp::create(rb, rloc, vals);
            });

        for (size_t i = 0; i < activeComps.size(); ++i) {
            outVecs[activeComps[i]] = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(elemTy)).getElementType(), ifRes.getResult(i));
        }
        mlir::Value statusVec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i32Ty())).getElementType(), ifRes.getResult(activeComps.size()));
        return {outVecs, statusVec};
    }

    mlir::Value computeBufferByteOffsetVec(
        mlir::Location loc, mlir::OpBuilder& nb, const SimdVal& h, mlir::Value stride64Vec,
        mlir::Value idxVec, mlir::Value elemOffVec, bool isRawOp, bool hasElemOff) {
        mlir::Value idx64Vec = mlir::LLVM::ZExtOp::create(nb, loc, vecTy(i64Ty()), idxVec);
        if (isRawOp) {
            if (!hasElemOff) return idx64Vec;
            mlir::Value e64Vec = mlir::LLVM::ZExtOp::create(nb, loc, vecTy(i64Ty()), elemOffVec);
            return mlir::LLVM::AddOp::create(
                nb, loc, mlir::LLVM::MulOp::create(nb, loc, idx64Vec, stride64Vec), e64Vec);
        }
        if (hasElemOff) {
            mlir::Value e64Vec = mlir::LLVM::ZExtOp::create(nb, loc, vecTy(i64Ty()), elemOffVec);
            return mlir::LLVM::AddOp::create(
                nb, loc, mlir::LLVM::MulOp::create(nb, loc, idx64Vec, stride64Vec), e64Vec);
        }
        if (h.resourceKind == 11) return idx64Vec;
        return mlir::LLVM::MulOp::create(nb, loc, idx64Vec, stride64Vec);
    }

    mlir::Value computeBufferBoundsMaskVec(
        mlir::Location loc, mlir::OpBuilder& nb, mlir::Value byteOffVec, mlir::Value byteSizeVec,
        mlir::Value okPtrVec, mlir::Value elemStrideOkVec, uint32_t c, uint32_t elemBytes) {
        mlir::Value endBytesSplat = mlir::LLVM::ConstantOp::create(nb, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecTy(i64Ty()), nb.getIntegerAttr(i64Ty(), uint64_t(c + 1) * elemBytes))).getType(), mlir::DenseElementsAttr::get(vecTy(i64Ty()), nb.getIntegerAttr(i64Ty(), uint64_t(c + 1) * elemBytes)));
        mlir::Value cEndVec = mlir::LLVM::AddOp::create(nb, loc, byteOffVec, endBytesSplat);
        mlir::Value inBoundsVec = mlir::LLVM::AndOp::create(
            nb, loc, okPtrVec,
            mlir::LLVM::ICmpOp::create(nb, loc, mlir::LLVM::ICmpPredicate::ule, cEndVec, byteSizeVec));
        if (elemStrideOkVec) {
            inBoundsVec = mlir::LLVM::AndOp::create(nb, loc, inBoundsVec, elemStrideOkVec);
        }
        return inBoundsVec;
    }

    std::pair<mlir::Value, mlir::Value> checkContiguousByteOffsets(
        mlir::Location loc, mlir::OpBuilder& nb, mlir::Value byteOffVec, uint32_t elemBytes) {
        mlir::Value lane0Off = llvmExtract(nb, loc, byteOffVec, static_cast<uint32_t>(0));
        if (W_ <= 1) {
            return {mlir::LLVM::ConstantOp::create(nb, loc, i1Ty(), nb.getIntegerAttr(i1Ty(), 1)), lane0Off};
        }
        llvm::SmallVector<int64_t, 16> steps;
        for (uint32_t l = 0; l < W_; ++l) {
            steps.push_back(static_cast<int64_t>(l) * elemBytes);
        }
        mlir::Value stepConst = mlir::LLVM::ConstantOp::create(nb, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecTy(i64Ty()), llvm::ArrayRef<int64_t>(steps))).getType(), mlir::DenseElementsAttr::get(vecTy(i64Ty()), llvm::ArrayRef<int64_t>(steps)));
        mlir::Value expectedVec = mlir::LLVM::AddOp::create(
            nb, loc, llvmSplat(nb, loc, llvm::cast<mlir::VectorType>(vecTy(i64Ty())).getElementType(), lane0Off), stepConst);
        mlir::Value eqVec = mlir::LLVM::ICmpOp::create(nb, loc, mlir::LLVM::ICmpPredicate::eq, byteOffVec, expectedVec);
        mlir::Value isContig = mlir::LLVM::vector_reduce_and::create(nb, loc, i1Ty(), eqVec);
        return {isContig, lane0Off};
    }

    mlir::Value computeScalarElemStrideOkVec(
        mlir::Location loc, mlir::OpBuilder& nb, mlir::Value elemStride, uint32_t c, uint32_t elemBytes) {
        if (!elemStride) return {};
        mlir::Value elemEnd = mlir::LLVM::ConstantOp::create(nb, loc, i32Ty(), nb.getIntegerAttr(i32Ty(), (c + 1) * elemBytes));
        mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
            nb, loc, mlir::LLVM::ICmpPredicate::eq, elemStride, mlir::LLVM::ConstantOp::create(nb, loc, i32Ty(), nb.getIntegerAttr(i32Ty(), 0)));
        mlir::Value inElem = mlir::LLVM::ICmpOp::create(
            nb, loc, mlir::LLVM::ICmpPredicate::ule, elemEnd, elemStride);
        mlir::Value ok = mlir::LLVM::OrOp::create(nb, loc, strideZero, inElem);
        return llvmSplat(nb, loc, llvm::cast<mlir::VectorType>(vecTy(i1Ty())).getElementType(), ok);
    }

    mlir::Value computeVecElemStrideOkVec(
        mlir::Location loc, mlir::OpBuilder& nb, mlir::Value elemStrideVec, uint32_t c, uint32_t elemBytes) {
        if (!elemStrideVec) return {};
        auto splatI32 = [&](uint32_t v) {
            return mlir::LLVM::ConstantOp::create(nb, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), v))).getType(), mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), v)));
        };
        mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
            nb, loc, mlir::LLVM::ICmpPredicate::eq, elemStrideVec, splatI32(0));
        mlir::Value inElem = mlir::LLVM::ICmpOp::create(
            nb, loc, mlir::LLVM::ICmpPredicate::ule, splatI32((c + 1) * elemBytes), elemStrideVec);
        return mlir::LLVM::OrOp::create(nb, loc, strideZero, inElem);
    }

    bool packetizeCBufferLoad(mlir::LLVM::CallOp call, bool isLegacy, std::string& outError) {
        mlir::Location loc = call.getLoc();
        SimdVal h = getMapped(loc, call.getOperand(1));
        mlir::Value idxArg = call.getOperand(2);
        auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(call.getResult().getType());
        if (!st) {
            outError = "cbufferLoad expected struct return type";
            return false;
        }
        const unsigned numFields = static_cast<unsigned>(st.getBody().size());
        mlir::Type elemTy = st.getBody()[0];
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);

        uint32_t usedMask = 0;
        bool allExtract = true;
        for (mlir::Operation* user : call.getResult().getUsers()) {
            if (auto ev = llvm::dyn_cast<mlir::LLVM::ExtractValueOp>(user)) {
                auto pos = ev.getPosition();
                if (!pos.empty() && pos[0] >= 0 && pos[0] < 32) {
                    usedMask |= (1u << static_cast<uint32_t>(pos[0]));
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

        mlir::Value uniformIdx;
        if (h.isUniform() && tryGetUniformScalar(loc, idxArg, uniformIdx)) {
            mlir::Value byteOff;
            if (auto idxConst = getConstIntValue(idxArg)) {
                byteOff = cInt(loc, i64Ty(), isLegacy ? (*idxConst * 16) : *idxConst);
            } else {
                mlir::Value idx64 = mlir::LLVM::ZExtOp::create(b_, loc, i64Ty(), uniformIdx);
                byteOff = isLegacy ? mlir::Value(mlir::LLVM::MulOp::create(b_, loc, idx64, cInt(loc, i64Ty(), 16))) : idx64;
            }
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, h.lanes[0], /*needElemStride=*/false, /*needStride64=*/false);
            auto [compVecs, _] = emitCoalescedUniformBufferLoad(
                loc, elemTy, elemBytes, info, h.isRootConstants, /*nonStraddling=*/false, byteOff, numFields, activeComps);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = st;
            for (unsigned f = 0; f < numFields; ++f) {
                outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, compVecs[f], {}, {}});
            }
            setMapped(call.getResult(), outSt);
            return true;
        }

        mlir::Value zeroElemVec = mlir::isa<mlir::FloatType>(elemTy) ? splatFlt(loc, elemTy, 0.0) : splatInt(loc, elemTy, 0);
        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        if (activeComps.empty()) {
            for (unsigned f = 0; f < numFields; ++f) {
                outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, zeroElemVec, {}, {}});
            }
            setMapped(call.getResult(), outSt);
            return true;
        }

        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        if (h.isUniform()) {
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, h.lanes[0], /*needElemStride=*/false, /*needStride64=*/false);
            mlir::Value idx64Vec = mlir::LLVM::ZExtOp::create(b_, loc, vecTy(i64Ty()), idxVec);
            mlir::Value off64Vec = isLegacy ? mlir::Value(mlir::LLVM::MulOp::create(b_, loc, idx64Vec, splatInt(loc, i64Ty(), 16))) : idx64Vec;

            auto emitGatherCBuffer = [&](mlir::OpBuilder& gb, mlir::Location gloc) -> llvm::SmallVector<mlir::Value, 8> {
                mlir::Value okPtrVec = llvmSplat(gb, gloc, llvm::cast<mlir::VectorType>(vecTy(i1Ty())).getElementType(), info.okPtr);
                mlir::Value byteSizeVec = h.isRootConstants
                                              ? mlir::Value()
                                              : mlir::Value(llvmSplat(gb, gloc, llvm::cast<mlir::VectorType>(vecTy(i64Ty())).getElementType(), info.byteSize));
                llvm::SmallVector<mlir::Value, 8> gathered;
                gathered.reserve(activeComps.size());
                for (unsigned f : activeComps) {
                    mlir::Value inBoundsVec = h.isRootConstants
                                                  ? okPtrVec
                                                  : computeBufferBoundsMaskVec(
                                                        gloc, gb, off64Vec, byteSizeVec, okPtrVec, mlir::Value(), f, elemBytes);
                    mlir::Value compBase = (f == 0)
                                               ? info.basePtr
                                               : mlir::Value(mlir::LLVM::GEPOp::create(
                                                     gb, gloc, ptrTy(), i8Ty(), info.basePtr,
                                                     llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(f * elemBytes)}));
                    mlir::Value ptrs = mlir::LLVM::GEPOp::create(
                        gb, gloc, vecTy(ptrTy()), i8Ty(), compBase, mlir::ValueRange{off64Vec});
                    gathered.push_back(mlir::LLVM::masked_gather::create(
                        gb, gloc, vecTy(elemTy), ptrs, inBoundsVec, mlir::ValueRange{zeroElemVec}, /*alignment=*/1));
                }
                return gathered;
            };

            llvm::SmallVector<mlir::Value, 8> activeVecs;
            if (h.isRootConstants) {
                activeVecs = emitGatherCBuffer(b_, loc);
            } else {
                auto ifRes = mlir::scf::IfOp::create(
                    b_, loc, info.isReserved,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value c0 = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(0));
                        mlir::Value c1 = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(1));
                        mlir::Value cW = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(W_));
                        llvm::SmallVector<mlir::Value, 8> initArgs(activeComps.size(), zeroElemVec);
                        auto loop = mlir::scf::ForOp::create(
                            nb, nloc, c0, cW, c1, initArgs,
                            [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Value lIdx, mlir::ValueRange iterArgs) {
                                mlir::Value offL = mlir::LLVM::ExtractElementOp::create(lb, lloc, off64Vec, lIdx);
                                llvm::SmallVector<mlir::Value, 8> nextArgs(iterArgs.begin(), iterArgs.end());
                                for (size_t i = 0; i < activeComps.size(); ++i) {
                                    unsigned f = activeComps[i];
                                    mlir::Value cOff = (f == 0)
                                                           ? offL
                                                           : mlir::Value(mlir::LLVM::AddOp::create(
                                                                 lb, lloc, offL,
                                                                 mlir::LLVM::ConstantOp::create(lb, lloc, i64Ty(), lb.getIntegerAttr(i64Ty(), uint64_t(f) * elemBytes))));
                                    mlir::Value cEnd = mlir::LLVM::AddOp::create(
                                        lb, lloc, cOff, mlir::LLVM::ConstantOp::create(lb, lloc, i64Ty(), lb.getIntegerAttr(i64Ty(), elemBytes)));
                                    mlir::Value inBounds = mlir::LLVM::AndOp::create(
                                        lb, lloc, info.okPtr,
                                        mlir::LLVM::ICmpOp::create(lb, lloc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize));
                                    auto [val, _] = emitReservedElemLoad(
                                        lb, lloc, elemTy, elemBytes, info.descP, info.basePtr, inBounds, cOff);
                                    nextArgs[i] = mlir::LLVM::InsertElementOp::create(lb, lloc, nextArgs[i], val, lIdx);
                                }
                                mlir::scf::YieldOp::create(lb, lloc, nextArgs);
                            });
                        mlir::scf::YieldOp::create(nb, nloc, loop.getResults());
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, emitGatherCBuffer(nb, nloc));
                    });
                activeVecs.assign(ifRes.getResults().begin(), ifRes.getResults().end());
            }

            llvm::SmallVector<mlir::Value, 8> fieldVecs(numFields, zeroElemVec);
            for (size_t i = 0; i < activeComps.size(); ++i) fieldVecs[activeComps[i]] = activeVecs[i];
            for (unsigned f = 0; f < numFields; ++f) {
                outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, fieldVecs[f], {}, {}});
            }
            setMapped(call.getResult(), outSt);
            return true;
        }

        llvm::SmallVector<mlir::Value, 8> fieldVecs(numFields, zeroElemVec);
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value ptr = laneValue(loc, h, l);
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, ptr, /*needElemStride=*/false, /*needStride64=*/false);
            mlir::Value idxL = llvmExtract(b_, loc, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
            mlir::Value idx64 = mlir::LLVM::ZExtOp::create(b_, loc, i64Ty(), idxL);
            mlir::Value off64 = isLegacy ? mlir::Value(mlir::LLVM::MulOp::create(b_, loc, idx64, cInt(loc, i64Ty(), 16))) : idx64;
            for (unsigned f : activeComps) {
                mlir::Value cOff = (f == 0) ? off64 : mlir::Value(mlir::LLVM::AddOp::create(b_, loc, off64, cInt(loc, i64Ty(), uint64_t(f) * elemBytes)));
                mlir::Value cEnd = mlir::LLVM::AddOp::create(b_, loc, cOff, cInt(loc, i64Ty(), elemBytes));
                mlir::Value inBounds = h.isRootConstants
                                           ? info.okPtr
                                           : mlir::Value(mlir::LLVM::AndOp::create(
                                                 b_, loc, info.okPtr,
                                                 mlir::LLVM::ICmpOp::create(
                                                     b_, loc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize)));
                auto [val, _] = emitScalarBufferComponentLoad(
                    loc, elemTy, elemBytes, ptr, info.basePtr, info.isReserved, inBounds, cOff);
                fieldVecs[f] = llvmInsert(b_, loc, fieldVecs[f], val, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
        }
        for (unsigned f = 0; f < numFields; ++f) {
            outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, fieldVecs[f], {}, {}});
        }
        setMapped(call.getResult(), outSt);
        return true;
    }

    std::pair<llvm::SmallVector<mlir::Value, 16>, mlir::Value> emitPacketizedBufferLoads(
        mlir::Location loc,
        const SimdVal& h,
        mlir::Value idxArg,
        mlir::Value elemOffArg,
        bool hasElemOff,
        bool isRawOp,
        mlir::Type elemTy,
        unsigned totalComps,
        llvm::ArrayRef<unsigned> activeComps) {
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);
        mlir::Value zeroElemVec = mlir::isa<mlir::FloatType>(elemTy) ? splatFlt(loc, elemTy, 0.0) : splatInt(loc, elemTy, 0);
        if (activeComps.empty()) {
            return {llvm::SmallVector<mlir::Value, 16>(totalComps, zeroElemVec), splatInt(loc, i32Ty(), 1)};
        }

        bool hasElemStride = (!isRawOp && !hasElemOff && h.resourceKind != 11);
        bool needsStride = hasElemOff || (!isRawOp && h.resourceKind != 11);
        auto elemOffConst = hasElemOff ? getConstIntValue(elemOffArg) : std::optional<uint64_t>{};
        unsigned lastActiveComp = activeComps.back();
        bool nonStraddling = hasElemOff && h.resourceKind == 12 && h.stride > 0 && elemOffConst.has_value() &&
                             (*elemOffConst + uint64_t(lastActiveComp + 1) * elemBytes <= h.stride);

        mlir::Value uniformIdx;
        mlir::Value uniformElemOff;
        bool isUniformIdx = tryGetUniformScalar(loc, idxArg, uniformIdx);
        bool isUniformElemOff = !hasElemOff || tryGetUniformScalar(loc, elemOffArg, uniformElemOff);
        if (h.isUniform() && isUniformIdx && isUniformElemOff) {
            if (!hasElemOff) uniformElemOff = cInt(loc, i32Ty(), 0);
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, h.lanes[0], hasElemStride, needsStride);
            mlir::Value byteOff = computeBufferByteOffsetScalar(
                loc, b_, h, info.stride64, uniformIdx, uniformElemOff, isRawOp, hasElemOff);
            return emitCoalescedUniformBufferLoad(
                loc, elemTy, elemBytes, info, h.isRootConstants, nonStraddling, byteOff, totalComps, activeComps);
        }

        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        mlir::Value elemOffVec = hasElemOff ? asVec(loc, getMapped(loc, elemOffArg)) : splatInt(loc, i32Ty(), 0);

        if (h.isUniform()) {
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, h.lanes[0], hasElemStride, needsStride);
            mlir::Value stride64Vec = needsStride
                                          ? mlir::Value(llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i64Ty())).getElementType(), info.stride64))
                                          : mlir::Value();

            auto ifRes = mlir::scf::IfOp::create(
                b_, loc, info.isReserved,
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value c0 = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(0));
                    mlir::Value c1 = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(1));
                    mlir::Value cW = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(W_));
                    llvm::SmallVector<mlir::Value, 17> initArgs(activeComps.size(), zeroElemVec);
                    initArgs.push_back(mlir::LLVM::ConstantOp::create(nb, nloc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), 1))).getType(), mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), 1))));
                    auto loop = mlir::scf::ForOp::create(
                        nb, nloc, c0, cW, c1, initArgs,
                        [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Value lIdx, mlir::ValueRange iterArgs) {
                            mlir::Value idxL = mlir::LLVM::ExtractElementOp::create(lb, lloc, idxVec, lIdx);
                            mlir::Value eL = hasElemOff
                                                 ? mlir::Value(mlir::LLVM::ExtractElementOp::create(lb, lloc, elemOffVec, lIdx))
                                                 : mlir::Value(mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), 0)));
                            mlir::Value byteOff = computeBufferByteOffsetScalar(
                                lloc, lb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                            mlir::Value laneMapped = mlir::LLVM::ConstantOp::create(lb, lloc, i1Ty(), lb.getIntegerAttr(i1Ty(), 1));
                            llvm::SmallVector<mlir::Value, 17> nextArgs(iterArgs.begin(), iterArgs.end());
                            for (size_t i = 0; i < activeComps.size(); ++i) {
                                unsigned c = activeComps[i];
                                mlir::Value cOff = (c == 0)
                                                       ? byteOff
                                                       : mlir::Value(mlir::LLVM::AddOp::create(
                                                             lb, lloc, byteOff,
                                                             mlir::LLVM::ConstantOp::create(lb, lloc, i64Ty(), lb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes))));
                                mlir::Value cEnd = mlir::LLVM::AddOp::create(
                                    lb, lloc, cOff, mlir::LLVM::ConstantOp::create(lb, lloc, i64Ty(), lb.getIntegerAttr(i64Ty(), elemBytes)));
                                mlir::Value inBounds = mlir::LLVM::AndOp::create(
                                    lb, lloc, info.okPtr,
                                    mlir::LLVM::ICmpOp::create(lb, lloc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize));
                                if (info.elemStride) {
                                    mlir::Value elemEnd = mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), (c + 1) * elemBytes));
                                    mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
                                        lb, lloc, mlir::LLVM::ICmpPredicate::eq, info.elemStride,
                                        mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), 0)));
                                    mlir::Value inElem = mlir::LLVM::ICmpOp::create(
                                        lb, lloc, mlir::LLVM::ICmpPredicate::ule, elemEnd, info.elemStride);
                                    inBounds = mlir::LLVM::AndOp::create(
                                        lb, lloc, inBounds, mlir::LLVM::OrOp::create(lb, lloc, strideZero, inElem));
                                }
                                auto [val, compMapped] = emitReservedElemLoad(
                                    lb, lloc, elemTy, elemBytes, info.descP, info.basePtr, inBounds, cOff);
                                laneMapped = mlir::LLVM::AndOp::create(lb, lloc, laneMapped, compMapped);
                                nextArgs[i] = mlir::LLVM::InsertElementOp::create(lb, lloc, nextArgs[i], val, lIdx);
                            }
                            mlir::Value laneStatus = mlir::LLVM::ZExtOp::create(lb, lloc, i32Ty(), laneMapped);
                            nextArgs[activeComps.size()] = mlir::LLVM::InsertElementOp::create(lb, lloc, nextArgs[activeComps.size()], laneStatus, lIdx);
                            mlir::scf::YieldOp::create(lb, lloc, nextArgs);
                        });
                    mlir::scf::YieldOp::create(nb, nloc, loop.getResults());
                },
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value byteOffVec = computeBufferByteOffsetVec(
                        nloc, nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                    mlir::Value byteSizeVec = llvmSplat(nb, nloc, llvm::cast<mlir::VectorType>(vecTy(i64Ty())).getElementType(), info.byteSize);
                    mlir::Value okPtrVec = llvmSplat(nb, nloc, llvm::cast<mlir::VectorType>(vecTy(i1Ty())).getElementType(), info.okPtr);

                    llvm::SmallVector<mlir::Value, 16> inBounds(activeComps.size());
                    if (nonStraddling) {
                        mlir::Value esOk = computeScalarElemStrideOkVec(nloc, nb, info.elemStride, lastActiveComp, elemBytes);
                        mlir::Value sharedMask = computeBufferBoundsMaskVec(
                            nloc, nb, byteOffVec, byteSizeVec, okPtrVec, esOk, lastActiveComp, elemBytes);
                        for (size_t i = 0; i < activeComps.size(); ++i) inBounds[i] = sharedMask;
                    } else {
                        for (size_t i = 0; i < activeComps.size(); ++i) {
                            unsigned c = activeComps[i];
                            mlir::Value esOk = computeScalarElemStrideOkVec(nloc, nb, info.elemStride, c, elemBytes);
                            inBounds[i] = computeBufferBoundsMaskVec(
                                nloc, nb, byteOffVec, byteSizeVec, okPtrVec, esOk, c, elemBytes);
                        }
                    }

                    auto emitGatherLoads = [&](mlir::OpBuilder& gb, mlir::Location gloc) -> llvm::SmallVector<mlir::Value, 16> {
                        llvm::SmallVector<mlir::Value, 16> gathered;
                        gathered.reserve(activeComps.size());
                        for (size_t i = 0; i < activeComps.size(); ++i) {
                            unsigned c = activeComps[i];
                            mlir::Value compBase = (c == 0)
                                                       ? info.basePtr
                                                       : mlir::Value(mlir::LLVM::GEPOp::create(
                                                             gb, gloc, ptrTy(), i8Ty(), info.basePtr,
                                                             llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(c * elemBytes)}));
                            mlir::Value ptrs = mlir::LLVM::GEPOp::create(
                                gb, gloc, vecTy(ptrTy()), i8Ty(), compBase, mlir::ValueRange{byteOffVec});
                            gathered.push_back(mlir::LLVM::masked_gather::create(
                                gb, gloc, vecTy(elemTy), ptrs, inBounds[i], mlir::ValueRange{zeroElemVec}, /*alignment=*/1));
                        }
                        return gathered;
                    };

                    bool canBeContig = !(W_ > 1 && ((hasElemOff && h.stride > elemBytes && isUniformElemOff) || activeComps.size() > 4));
                    llvm::SmallVector<mlir::Value, 17> out;
                    out.reserve(activeComps.size() + 1);
                    if (canBeContig) {
                        auto [isContig, lane0Off] = checkContiguousByteOffsets(nloc, nb, byteOffVec, elemBytes);
                        auto loadIfContig = mlir::scf::IfOp::create(
                            nb, nloc, isContig,
                            [&](mlir::OpBuilder& cb, mlir::Location cloc) {
                                mlir::Value baseLane0Ptr = mlir::LLVM::GEPOp::create(
                                    cb, cloc, ptrTy(), i8Ty(), info.basePtr, mlir::ValueRange{lane0Off});
                                llvm::SmallVector<mlir::Value, 16> contigOut;
                                contigOut.reserve(activeComps.size());
                                for (size_t i = 0; i < activeComps.size(); ++i) {
                                    unsigned c = activeComps[i];
                                    mlir::Value ptr0 = (c == 0)
                                                           ? baseLane0Ptr
                                                           : mlir::Value(mlir::LLVM::GEPOp::create(
                                                                 cb, cloc, ptrTy(), i8Ty(), baseLane0Ptr,
                                                                 llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(c * elemBytes)}));
                                    contigOut.push_back(mlir::LLVM::MaskedLoadOp::create(
                                        cb, cloc, vecTy(elemTy), ptr0, inBounds[i], zeroElemVec, /*alignment=*/1));
                                }
                                mlir::scf::YieldOp::create(cb, cloc, contigOut);
                            },
                            [&](mlir::OpBuilder& cb, mlir::Location cloc) {
                                mlir::scf::YieldOp::create(cb, cloc, emitGatherLoads(cb, cloc));
                            });
                        out.assign(loadIfContig.getResults().begin(), loadIfContig.getResults().end());
                    } else {
                        out = emitGatherLoads(nb, nloc);
                    }
                    out.push_back(mlir::LLVM::ConstantOp::create(nb, nloc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), 1))).getType(), mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), 1))));
                    mlir::scf::YieldOp::create(nb, nloc, out);
                });

            llvm::SmallVector<mlir::Value, 16> compVecs(totalComps, zeroElemVec);
            for (size_t i = 0; i < activeComps.size(); ++i) {
                compVecs[activeComps[i]] = ifRes.getResult(i);
            }
            return {compVecs, ifRes.getResult(activeComps.size())};
        }

        llvm::SmallVector<BufferDescInfo, 16> descInfos(W_);
        mlir::Value basePtrVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(ptrTy()));
        mlir::Value byteSizeVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i64Ty()));
        mlir::Value okPtrVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i1Ty()));
        mlir::Value isReservedVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i1Ty()));
        mlir::Value stride64Vec = needsStride ? mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i64Ty()))) : mlir::Value();
        mlir::Value elemStrideVec = hasElemStride ? mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i32Ty()))) : mlir::Value();

        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value descP = laneValue(loc, h, l);
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, descP, hasElemStride, needsStride);
            descInfos[l] = info;
            basePtrVec = llvmInsert(b_, loc, basePtrVec, info.basePtr, static_cast<uint32_t>(static_cast<int64_t>(l)));
            byteSizeVec = llvmInsert(b_, loc, byteSizeVec, info.byteSize, static_cast<uint32_t>(static_cast<int64_t>(l)));
            okPtrVec = llvmInsert(b_, loc, okPtrVec, info.okPtr, static_cast<uint32_t>(static_cast<int64_t>(l)));
            isReservedVec = llvmInsert(b_, loc, isReservedVec, info.isReserved, static_cast<uint32_t>(static_cast<int64_t>(l)));
            if (hasElemStride) {
                elemStrideVec = llvmInsert(b_, loc, elemStrideVec, info.elemStride, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            if (needsStride) {
                stride64Vec = llvmInsert(b_, loc, stride64Vec, info.stride64, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
        }
        mlir::Value anyReserved = orReduce(loc, isReservedVec);
        auto ifRes = mlir::scf::IfOp::create(
            b_, loc, anyReserved,
            [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                llvm::SmallVector<mlir::Value, 17> activeVecs(
                    activeComps.size(), mlir::LLVM::PoisonOp::create(nb, nloc, vecTy(elemTy)));
                mlir::Value statusVec = tmp.splatInt(nloc, i32Ty(), 1);
                for (uint32_t l = 0; l < W_; ++l) {
                    const BufferDescInfo& info = descInfos[l];
                    mlir::Value idxL = llvmExtract(nb, nloc, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::Value eL = llvmExtract(nb, nloc, elemOffVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::Value byteOff = tmp.computeBufferByteOffsetScalar(
                        nloc, nb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                    mlir::Value laneMapped = tmp.cInt(nloc, i1Ty(), 1);
                    for (size_t i = 0; i < activeComps.size(); ++i) {
                        unsigned c = activeComps[i];
                        mlir::Value cOff = (c == 0)
                                               ? byteOff
                                               : mlir::Value(mlir::LLVM::AddOp::create(
                                                     nb, nloc, byteOff, tmp.cInt(nloc, i64Ty(), uint64_t(c) * elemBytes)));
                        mlir::Value cEnd = mlir::LLVM::AddOp::create(nb, nloc, cOff, tmp.cInt(nloc, i64Ty(), elemBytes));
                        mlir::Value inBounds = mlir::LLVM::AndOp::create(
                            nb, nloc, info.okPtr,
                            mlir::LLVM::ICmpOp::create(nb, nloc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize));
                        if (info.elemStride) {
                            mlir::Value elemEnd = tmp.cInt(nloc, i32Ty(), (c + 1) * elemBytes);
                            mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
                                nb, nloc, mlir::LLVM::ICmpPredicate::eq, info.elemStride, tmp.cInt(nloc, i32Ty(), 0));
                            mlir::Value inElem = mlir::LLVM::ICmpOp::create(
                                nb, nloc, mlir::LLVM::ICmpPredicate::ule, elemEnd, info.elemStride);
                            inBounds = mlir::LLVM::AndOp::create(
                                nb, nloc, inBounds, mlir::LLVM::OrOp::create(nb, nloc, strideZero, inElem));
                        }
                        auto [val, compMapped] = tmp.emitScalarBufferComponentLoad(
                            nloc, elemTy, elemBytes, info.descP, info.basePtr, info.isReserved, inBounds, cOff);
                        laneMapped = mlir::LLVM::AndOp::create(nb, nloc, laneMapped, compMapped);
                        activeVecs[i] = llvmInsert(nb, nloc, activeVecs[i], val, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    }
                    mlir::Value laneStatus = mlir::LLVM::ZExtOp::create(nb, nloc, i32Ty(), laneMapped);
                    statusVec = llvmInsert(nb, nloc, statusVec, laneStatus, static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
                activeVecs.push_back(statusVec);
                mlir::scf::YieldOp::create(nb, nloc, activeVecs);
            },
            [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                mlir::Value byteOffVec = computeBufferByteOffsetVec(
                    nloc, nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                mlir::Value sharedMask;
                if (nonStraddling) {
                    mlir::Value esOk = computeVecElemStrideOkVec(nloc, nb, elemStrideVec, lastActiveComp, elemBytes);
                    sharedMask = computeBufferBoundsMaskVec(
                        nloc, nb, byteOffVec, byteSizeVec, okPtrVec, esOk, lastActiveComp, elemBytes);
                }
                llvm::SmallVector<mlir::Value, 17> out;
                out.reserve(activeComps.size() + 1);
                for (size_t i = 0; i < activeComps.size(); ++i) {
                    unsigned c = activeComps[i];
                    mlir::Value inBoundsVec = sharedMask;
                    if (!inBoundsVec) {
                        mlir::Value esOk = computeVecElemStrideOkVec(nloc, nb, elemStrideVec, c, elemBytes);
                        inBoundsVec = computeBufferBoundsMaskVec(
                            nloc, nb, byteOffVec, byteSizeVec, okPtrVec, esOk, c, elemBytes);
                    }
                    mlir::Value cOffVec =
                        (c == 0)
                            ? byteOffVec
                            : mlir::Value(mlir::LLVM::AddOp::create(
                                  nb, nloc, byteOffVec,
                                  mlir::LLVM::ConstantOp::create(nb, nloc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(
                                          vecTy(i64Ty()), nb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes))).getType(), mlir::DenseElementsAttr::get(
                                          vecTy(i64Ty()), nb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes)))));
                    mlir::Value ptrs = mlir::LLVM::GEPOp::create(
                        nb, nloc, vecTy(ptrTy()), i8Ty(), basePtrVec, mlir::ValueRange{cOffVec});
                    out.push_back(mlir::LLVM::masked_gather::create(
                        nb, nloc, vecTy(elemTy), ptrs, inBoundsVec, mlir::ValueRange{zeroElemVec}, /*alignment=*/1));
                }
                out.push_back(mlir::LLVM::ConstantOp::create(nb, nloc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), 1))).getType(), mlir::DenseElementsAttr::get(vecTy(i32Ty()), nb.getIntegerAttr(i32Ty(), 1))));
                mlir::scf::YieldOp::create(nb, nloc, out);
            });

        llvm::SmallVector<mlir::Value, 16> compVecs(totalComps, zeroElemVec);
        for (size_t i = 0; i < activeComps.size(); ++i) {
            compVecs[activeComps[i]] = ifRes.getResult(i);
        }
        return {compVecs, ifRes.getResult(activeComps.size())};
    }

    void emitPacketizedBufferStores(
        mlir::Location loc,
        const SimdVal& h,
        mlir::Value idxArg,
        mlir::Value elemOffArg,
        bool hasElemOff,
        bool isRawOp,
        mlir::Value realMask,
        mlir::Type elemTy,
        llvm::ArrayRef<unsigned> activeComps,
        llvm::ArrayRef<mlir::Value> activeValVecs) {
        if (activeComps.empty()) return;
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);
        bool hasElemStride = (!isRawOp && !hasElemOff && h.resourceKind != 11);
        bool needsStride = hasElemOff || (!isRawOp && h.resourceKind != 11);
        auto elemOffConst = hasElemOff ? getConstIntValue(elemOffArg) : std::optional<uint64_t>{};
        unsigned lastActiveComp = activeComps.back();
        bool nonStraddling = hasElemOff && h.resourceKind == 12 && h.stride > 0 && elemOffConst.has_value() &&
                             (*elemOffConst + uint64_t(lastActiveComp + 1) * elemBytes <= h.stride);

        mlir::Value dummyUniformElemOff;
        bool isUniformElemOff = !hasElemOff || tryGetUniformScalar(loc, elemOffArg, dummyUniformElemOff);
        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        mlir::Value elemOffVec = hasElemOff ? asVec(loc, getMapped(loc, elemOffArg)) : splatInt(loc, i32Ty(), 0);

        if (h.isUniform()) {
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, h.lanes[0], hasElemStride, needsStride);
            mlir::Value stride64Vec = needsStride
                                          ? mlir::Value(llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i64Ty())).getElementType(), info.stride64))
                                          : mlir::Value();

            mlir::scf::IfOp::create(
                b_, loc, info.isReserved,
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value c0 = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(0));
                    mlir::Value c1 = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(1));
                    mlir::Value cW = mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getI32IntegerAttr(W_));
                    mlir::scf::ForOp::create(
                        nb, nloc, c0, cW, c1, mlir::ValueRange{},
                        [&](mlir::OpBuilder& lb, mlir::Location lloc, mlir::Value lIdx, mlir::ValueRange) {
                            mlir::Value mL = mlir::LLVM::ExtractElementOp::create(lb, lloc, realMask, lIdx);
                            mlir::Value okPtr = mlir::LLVM::AndOp::create(lb, lloc, mL, info.okPtr);
                            mlir::Value idxL = mlir::LLVM::ExtractElementOp::create(lb, lloc, idxVec, lIdx);
                            mlir::Value eL = hasElemOff
                                                 ? mlir::Value(mlir::LLVM::ExtractElementOp::create(lb, lloc, elemOffVec, lIdx))
                                                 : mlir::Value(mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), 0)));
                            mlir::Value byteOff = computeBufferByteOffsetScalar(
                                lloc, lb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                            for (size_t i = 0; i < activeComps.size(); ++i) {
                                unsigned c = activeComps[i];
                                mlir::Value vL = mlir::LLVM::ExtractElementOp::create(lb, lloc, activeValVecs[i], lIdx);
                                mlir::Value cOff = (c == 0)
                                                       ? byteOff
                                                       : mlir::Value(mlir::LLVM::AddOp::create(
                                                             lb, lloc, byteOff,
                                                             mlir::LLVM::ConstantOp::create(lb, lloc, i64Ty(), lb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes))));
                                mlir::Value cEnd = mlir::LLVM::AddOp::create(
                                    lb, lloc, cOff, mlir::LLVM::ConstantOp::create(lb, lloc, i64Ty(), lb.getIntegerAttr(i64Ty(), elemBytes)));
                                mlir::Value doStore = mlir::LLVM::AndOp::create(
                                    lb, lloc, okPtr,
                                    mlir::LLVM::ICmpOp::create(lb, lloc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize));
                                if (info.elemStride) {
                                    mlir::Value elemEnd = mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), (c + 1) * elemBytes));
                                    mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
                                        lb, lloc, mlir::LLVM::ICmpPredicate::eq, info.elemStride,
                                        mlir::LLVM::ConstantOp::create(lb, lloc, i32Ty(), lb.getIntegerAttr(i32Ty(), 0)));
                                    mlir::Value inElem = mlir::LLVM::ICmpOp::create(
                                        lb, lloc, mlir::LLVM::ICmpPredicate::ule, elemEnd, info.elemStride);
                                    doStore = mlir::LLVM::AndOp::create(
                                        lb, lloc, doStore, mlir::LLVM::OrOp::create(lb, lloc, strideZero, inElem));
                                }
                                emitReservedElemStore(lb, lloc, elemBytes, info.descP, info.basePtr, doStore, cOff, vL);
                            }
                            mlir::scf::YieldOp::create(lb, lloc);
                        });
                    mlir::scf::YieldOp::create(nb, nloc);
                },
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value byteOffVec = computeBufferByteOffsetVec(
                        nloc, nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                    mlir::Value byteSizeVec = llvmSplat(nb, nloc, llvm::cast<mlir::VectorType>(vecTy(i64Ty())).getElementType(), info.byteSize);
                    mlir::Value activeOkVec = mlir::LLVM::AndOp::create(
                        nb, nloc, realMask, llvmSplat(nb, nloc, llvm::cast<mlir::VectorType>(vecTy(i1Ty())).getElementType(), info.okPtr));

                    llvm::SmallVector<mlir::Value, 16> doStores(activeComps.size());
                    if (nonStraddling) {
                        mlir::Value esOk = computeScalarElemStrideOkVec(nloc, nb, info.elemStride, lastActiveComp, elemBytes);
                        mlir::Value sharedMask = computeBufferBoundsMaskVec(
                            nloc, nb, byteOffVec, byteSizeVec, activeOkVec, esOk, lastActiveComp, elemBytes);
                        for (size_t i = 0; i < activeComps.size(); ++i) doStores[i] = sharedMask;
                    } else {
                        for (size_t i = 0; i < activeComps.size(); ++i) {
                            unsigned c = activeComps[i];
                            mlir::Value esOk = computeScalarElemStrideOkVec(nloc, nb, info.elemStride, c, elemBytes);
                            doStores[i] = computeBufferBoundsMaskVec(
                                nloc, nb, byteOffVec, byteSizeVec, activeOkVec, esOk, c, elemBytes);
                        }
                    }

                    auto emitScatterStores = [&](mlir::OpBuilder& sb, mlir::Location sloc) {
                        for (size_t i = 0; i < activeComps.size(); ++i) {
                            unsigned c = activeComps[i];
                            mlir::Value compBase = (c == 0)
                                                       ? info.basePtr
                                                       : mlir::Value(mlir::LLVM::GEPOp::create(
                                                             sb, sloc, ptrTy(), i8Ty(), info.basePtr,
                                                             llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(c * elemBytes)}));
                            mlir::Value ptrs = mlir::LLVM::GEPOp::create(
                                sb, sloc, vecTy(ptrTy()), i8Ty(), compBase, mlir::ValueRange{byteOffVec});
                            mlir::LLVM::masked_scatter::create(
                                sb, sloc, activeValVecs[i], ptrs, doStores[i], /*alignment=*/1);
                        }
                    };

                    bool canBeContig = !(W_ > 1 && ((hasElemOff && h.stride > elemBytes && isUniformElemOff) || activeComps.size() > 4));
                    if (canBeContig) {
                        auto [isContig, lane0Off] = checkContiguousByteOffsets(nloc, nb, byteOffVec, elemBytes);
                        mlir::scf::IfOp::create(
                            nb, nloc, isContig,
                            [&](mlir::OpBuilder& cb, mlir::Location cloc) {
                                mlir::Value baseLane0Ptr = mlir::LLVM::GEPOp::create(
                                    cb, cloc, ptrTy(), i8Ty(), info.basePtr, mlir::ValueRange{lane0Off});
                                for (size_t i = 0; i < activeComps.size(); ++i) {
                                    unsigned c = activeComps[i];
                                    mlir::Value ptr0 = (c == 0)
                                                           ? baseLane0Ptr
                                                           : mlir::Value(mlir::LLVM::GEPOp::create(
                                                                 cb, cloc, ptrTy(), i8Ty(), baseLane0Ptr,
                                                                 llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(c * elemBytes)}));
                                    mlir::LLVM::MaskedStoreOp::create(
                                        cb, cloc, activeValVecs[i], ptr0, doStores[i], /*alignment=*/1);
                                }
                                mlir::scf::YieldOp::create(cb, cloc);
                            },
                            [&](mlir::OpBuilder& cb, mlir::Location cloc) {
                                emitScatterStores(cb, cloc);
                                mlir::scf::YieldOp::create(cb, cloc);
                            });
                    } else {
                        emitScatterStores(nb, nloc);
                    }
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            return;
        }

        llvm::SmallVector<BufferDescInfo, 16> descInfos(W_);
        mlir::Value basePtrVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(ptrTy()));
        mlir::Value byteSizeVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i64Ty()));
        mlir::Value okPtrVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i1Ty()));
        mlir::Value isReservedVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i1Ty()));
        mlir::Value stride64Vec = needsStride ? mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i64Ty()))) : mlir::Value();
        mlir::Value elemStrideVec = hasElemStride ? mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i32Ty()))) : mlir::Value();

        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value descP = laneValue(loc, h, l);
            BufferDescInfo info = loadBufferDescInfo(loc, b_, h, descP, hasElemStride, needsStride);
            descInfos[l] = info;
            basePtrVec = llvmInsert(b_, loc, basePtrVec, info.basePtr, static_cast<uint32_t>(static_cast<int64_t>(l)));
            byteSizeVec = llvmInsert(b_, loc, byteSizeVec, info.byteSize, static_cast<uint32_t>(static_cast<int64_t>(l)));
            okPtrVec = llvmInsert(b_, loc, okPtrVec, info.okPtr, static_cast<uint32_t>(static_cast<int64_t>(l)));
            isReservedVec = llvmInsert(b_, loc, isReservedVec, info.isReserved, static_cast<uint32_t>(static_cast<int64_t>(l)));
            if (hasElemStride) {
                elemStrideVec = llvmInsert(b_, loc, elemStrideVec, info.elemStride, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            if (needsStride) {
                stride64Vec = llvmInsert(b_, loc, stride64Vec, info.stride64, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
        }
        mlir::Value anyReserved = orReduce(loc, isReservedVec);
        mlir::Value activeOkVec = mlir::LLVM::AndOp::create(b_, loc, realMask, okPtrVec);

        mlir::scf::IfOp::create(
            b_, loc, anyReserved,
            [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                for (uint32_t l = 0; l < W_; ++l) {
                    const BufferDescInfo& info = descInfos[l];
                    mlir::Value mL = llvmExtract(nb, nloc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::Value okPtr = mlir::LLVM::AndOp::create(nb, nloc, mL, info.okPtr);
                    mlir::Value idxL = llvmExtract(nb, nloc, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::Value eL = llvmExtract(nb, nloc, elemOffVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                    mlir::Value byteOff = tmp.computeBufferByteOffsetScalar(
                        nloc, nb, h, info.stride64, idxL, eL, isRawOp, hasElemOff);
                    for (size_t i = 0; i < activeComps.size(); ++i) {
                        unsigned c = activeComps[i];
                        mlir::Value vL = llvmExtract(nb, nloc, activeValVecs[i], static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value cOff = (c == 0)
                                               ? byteOff
                                               : mlir::Value(mlir::LLVM::AddOp::create(
                                                     nb, nloc, byteOff, tmp.cInt(nloc, i64Ty(), uint64_t(c) * elemBytes)));
                        mlir::Value cEnd = mlir::LLVM::AddOp::create(nb, nloc, cOff, tmp.cInt(nloc, i64Ty(), elemBytes));
                        mlir::Value doStore = mlir::LLVM::AndOp::create(
                            nb, nloc, okPtr,
                            mlir::LLVM::ICmpOp::create(nb, nloc, mlir::LLVM::ICmpPredicate::ule, cEnd, info.byteSize));
                        if (info.elemStride) {
                            mlir::Value elemEnd = tmp.cInt(nloc, i32Ty(), (c + 1) * elemBytes);
                            mlir::Value strideZero = mlir::LLVM::ICmpOp::create(
                                nb, nloc, mlir::LLVM::ICmpPredicate::eq, info.elemStride, tmp.cInt(nloc, i32Ty(), 0));
                            mlir::Value inElem = mlir::LLVM::ICmpOp::create(
                                nb, nloc, mlir::LLVM::ICmpPredicate::ule, elemEnd, info.elemStride);
                            doStore = mlir::LLVM::AndOp::create(
                                nb, nloc, doStore, mlir::LLVM::OrOp::create(nb, nloc, strideZero, inElem));
                        }
                        tmp.emitScalarBufferComponentStore(
                            nloc, elemBytes, info.descP, info.basePtr, info.isReserved, doStore, cOff, vL);
                    }
                }
                mlir::scf::YieldOp::create(nb, nloc);
            },
            [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                mlir::Value byteOffVec = computeBufferByteOffsetVec(
                    nloc, nb, h, stride64Vec, idxVec, elemOffVec, isRawOp, hasElemOff);
                mlir::Value sharedMask;
                if (nonStraddling) {
                    mlir::Value esOk = computeVecElemStrideOkVec(nloc, nb, elemStrideVec, lastActiveComp, elemBytes);
                    sharedMask = computeBufferBoundsMaskVec(
                        nloc, nb, byteOffVec, byteSizeVec, activeOkVec, esOk, lastActiveComp, elemBytes);
                }
                for (size_t i = 0; i < activeComps.size(); ++i) {
                    unsigned c = activeComps[i];
                    mlir::Value doStoreVec = sharedMask;
                    if (!doStoreVec) {
                        mlir::Value esOk = computeVecElemStrideOkVec(nloc, nb, elemStrideVec, c, elemBytes);
                        doStoreVec = computeBufferBoundsMaskVec(
                            nloc, nb, byteOffVec, byteSizeVec, activeOkVec, esOk, c, elemBytes);
                    }
                    mlir::Value cOffVec =
                        (c == 0)
                            ? byteOffVec
                            : mlir::Value(mlir::LLVM::AddOp::create(
                                  nb, nloc, byteOffVec,
                                  mlir::LLVM::ConstantOp::create(nb, nloc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(
                                          vecTy(i64Ty()), nb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes))).getType(), mlir::DenseElementsAttr::get(
                                          vecTy(i64Ty()), nb.getIntegerAttr(i64Ty(), uint64_t(c) * elemBytes)))));
                    mlir::Value ptrs = mlir::LLVM::GEPOp::create(
                        nb, nloc, vecTy(ptrTy()), i8Ty(), basePtrVec, mlir::ValueRange{cOffVec});
                    mlir::LLVM::masked_scatter::create(
                        nb, nloc, activeValVecs[i], ptrs, doStoreVec, /*alignment=*/1);
                }
                mlir::scf::YieldOp::create(nb, nloc);
            });
    }

    bool packetizeBufferLoad(mlir::LLVM::CallOp call, bool isRawOp, std::string& outError) {
        mlir::Location loc = call.getLoc();
        SimdVal h = getMapped(loc, call.getOperand(1));
        mlir::Value idxArg = call.getOperand(2);
        mlir::Value elemOffArg = call.getOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);
        uint32_t mask = isRawOp ? static_cast<uint32_t>(getConstIntValue(call.getOperand(4)).value_or(15)) : 15u;

        auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(call.getResult().getType());
        if (!st || st.getBody().size() < 5) {
            outError = "bufferLoad expected ResRet struct";
            return false;
        }
        mlir::Type elemTy = st.getBody()[0];

        uint32_t usedCompMask = 0;
        bool statusUsed = false;
        bool allExtract = true;
        for (mlir::Operation* user : call.getResult().getUsers()) {
            if (auto ev = llvm::dyn_cast<mlir::LLVM::ExtractValueOp>(user)) {
                auto pos = ev.getPosition();
                if (!pos.empty()) {
                    if (pos[0] >= 0 && pos[0] < 4) {
                        usedCompMask |= (1u << static_cast<uint32_t>(pos[0]));
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
            loc, h, idxArg, elemOffArg, hasElemOff, isRawOp, elemTy, /*totalComps=*/4, activeComps);

        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        for (unsigned c = 0; c < 4; ++c) {
            outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, compVecs[c], {}, {}});
        }
        outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), statusVec, {}, {}});
        setMapped(call.getResult(), outSt);
        return true;
    }

    bool packetizeBufferStore(mlir::LLVM::CallOp call, bool isRawOp, mlir::Value realMask, std::string&) {
        mlir::Location loc = call.getLoc();
        SimdVal h = getMapped(loc, call.getOperand(1));
        mlir::Value idxArg = call.getOperand(2);
        mlir::Value elemOffArg = call.getOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);
        uint32_t mask = static_cast<uint32_t>(getConstIntValue(call.getOperand(8)).value_or(15));

        llvm::SmallVector<unsigned, 4> activeComps;
        llvm::SmallVector<mlir::Value, 4> activeValVecs;
        mlir::Type elemTy = i32Ty();
        for (unsigned c = 0; c < 4; ++c) {
            if (mask & (1u << c)) {
                mlir::Value v = asVec(loc, getMapped(loc, call.getOperand(4 + c)));
                elemTy = llvm::cast<mlir::VectorType>(v.getType()).getElementType();
                activeComps.push_back(c);
                activeValVecs.push_back(v);
            }
        }
        emitPacketizedBufferStores(
            loc, h, idxArg, elemOffArg, hasElemOff, isRawOp, realMask, elemTy, activeComps, activeValVecs);
        return true;
    }

    bool packetizeBufferVectorLoad(mlir::LLVM::CallOp call, std::string& outError) {
        mlir::Location loc = call.getLoc();
        SimdVal h = getMapped(loc, call.getOperand(1));
        mlir::Value idxArg = call.getOperand(2);
        mlir::Value elemOffArg = call.getOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);

        auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(call.getResult().getType());
        auto vt = st ? llvm::dyn_cast<mlir::VectorType>(st.getBody()[0]) : mlir::VectorType();
        if (!st || !vt || st.getBody().size() < 2) {
            outError = "rawBufferVectorLoad expected ResRet struct with vector element";
            return false;
        }
        mlir::Type elemTy = vt.getElementType();
        unsigned numElems = static_cast<unsigned>(vt.getNumElements());

        llvm::SmallVector<bool, 16> elemUsed(numElems, false);
        bool statusUsed = false;
        bool canPrune = true;
        for (mlir::Operation* user : call.getResult().getUsers()) {
            auto ev = llvm::dyn_cast<mlir::LLVM::ExtractValueOp>(user);
            if (!ev || ev.getPosition().empty()) {
                canPrune = false;
                break;
            }
            int64_t fieldIdx = ev.getPosition()[0];
            if (fieldIdx == 1) {
                statusUsed = true;
            } else if (fieldIdx == 0) {
                for (mlir::Operation* vUser : ev.getResult().getUsers()) {
                    auto ee = llvm::dyn_cast<mlir::LLVM::ExtractElementOp>(vUser);
                    auto idxC = ee ? getConstIntValue(ee.getPosition()) : std::optional<uint64_t>{};
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
            loc, h, idxArg, elemOffArg, hasElemOff, /*isRawOp=*/true, elemTy, numElems, activeComps);

        SimdVal vecVal;
        vecVal.kind = SimdVal::Kind::Struct;
        vecVal.origType = vt;
        for (unsigned c = 0; c < numElems; ++c) {
            vecVal.fields.push_back({SimdVal::Kind::Vector, elemTy, compVecs[c], {}, {}});
        }
        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        outSt.fields.push_back(std::move(vecVal));
        outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), statusVec, {}, {}});
        setMapped(call.getResult(), outSt);
        return true;
    }

    bool packetizeBufferVectorStore(mlir::LLVM::CallOp call, mlir::Value realMask, std::string& outError) {
        mlir::Location loc = call.getLoc();
        SimdVal h = getMapped(loc, call.getOperand(1));
        mlir::Value idxArg = call.getOperand(2);
        mlir::Value elemOffArg = call.getOperand(3);
        bool hasElemOff = !isUndefOrPoison(elemOffArg);
        SimdVal valSv = getMapped(loc, call.getOperand(4));
        if (valSv.kind != SimdVal::Kind::Struct || valSv.fields.empty()) {
            outError = "rawBufferVectorStore expected vector value";
            return false;
        }
        unsigned numElems = static_cast<unsigned>(valSv.fields.size());
        llvm::SmallVector<unsigned, 16> activeComps;
        llvm::SmallVector<mlir::Value, 16> activeValVecs;
        activeComps.reserve(numElems);
        activeValVecs.reserve(numElems);
        for (unsigned c = 0; c < numElems; ++c) {
            activeComps.push_back(c);
            activeValVecs.push_back(asVec(loc, valSv.fields[c]));
        }
        mlir::Type elemTy = llvm::cast<mlir::VectorType>(activeValVecs[0].getType()).getElementType();
        emitPacketizedBufferStores(
            loc, h, idxArg, elemOffArg, hasElemOff, /*isRawOp=*/true, realMask, elemTy, activeComps, activeValVecs);
        return true;
    }

    bool packetizeAtomicOp(mlir::LLVM::CallOp call, bool isCmpXchg, mlir::Value realMask, std::string&) {
        mlir::Location loc = call.getLoc();
        // atomicBinOp(op, handle, atomicOp, off0, off1, off2, newVal)
        // atomicCompareExchange(op, handle, off0, off1, off2, cmpVal, newVal)
        SimdVal h = getMapped(loc, call.getOperand(1));
        uint32_t atomicOp = isCmpXchg ? 0 : static_cast<uint32_t>(getConstIntValue(call.getOperand(2)).value_or(0));
        unsigned offStart = isCmpXchg ? 2u : 3u;
        mlir::Value off0 = asVec(loc, getMapped(loc, call.getOperand(offStart)));
        bool hasOff1 = !isUndefOrPoison(call.getOperand(offStart + 1));
        mlir::Value off1 = hasOff1 ? asVec(loc, getMapped(loc, call.getOperand(offStart + 1))) : splatInt(loc, i32Ty(), 0);
        mlir::Value cmpValVec = isCmpXchg ? asVec(loc, getMapped(loc, call.getOperand(5))) : mlir::Value();
        mlir::Value newValVec = asVec(loc, getMapped(loc, call.getOperand(6)));

        auto toBinOp = [](uint32_t dxAtomic) -> mlir::LLVM::AtomicBinOp {
            switch (dxAtomic) {
            case 0: return mlir::LLVM::AtomicBinOp::add;
            case 1: return mlir::LLVM::AtomicBinOp::_and;
            case 2: return mlir::LLVM::AtomicBinOp::_or;
            case 3: return mlir::LLVM::AtomicBinOp::_xor;
            case 4: return mlir::LLVM::AtomicBinOp::min;
            case 5: return mlir::LLVM::AtomicBinOp::max;
            case 6: return mlir::LLVM::AtomicBinOp::umin;
            case 7: return mlir::LLVM::AtomicBinOp::umax;
            case 8: return mlir::LLVM::AtomicBinOp::xchg;
            default: return mlir::LLVM::AtomicBinOp::add;
            }
        };

        mlir::Type resTy = call.getResult().getType();
        uint32_t bitWidth = resTy.getIntOrFloatBitWidth();
        uint32_t elemBytes = std::max<uint32_t>(1u, bitWidth / 8u);
        bool isFloat = mlir::isa<mlir::FloatType>(resTy);
        mlir::Type intTy = b_.getIntegerType(bitWidth);
        const bool isTexAtomic = (h.resourceKind >= 1 && h.resourceKind <= 9);
        const bool isRawBufAtomic = (h.resourceKind == 11 || h.resourceKind == 12);
        const bool needAtomicStride = !isTexAtomic && (hasOff1 || !isRawBufAtomic);

        std::optional<BufferDescInfo> uniformBufInfo;
        if (!isTexAtomic && h.isUniform()) {
            uniformBufInfo = loadBufferDescInfo(loc, b_, h, h.lanes[0], /*needElemStride=*/false, needAtomicStride);
        }

        mlir::Value outVec = isFloat ? splatFlt(loc, resTy, 0.0) : splatInt(loc, resTy, 0);
        for (uint32_t l = 0; l < W_; ++l) {
            if (isLaneMaskConstFalse(realMask, l)) continue;
            mlir::Value mL = llvmExtract(b_, loc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
            mlir::Value descP;
            mlir::Value basePtr;
            mlir::Value isReserved;
            mlir::Value byteSize;
            mlir::Value okPtr;
            mlir::Value o0L = llvmExtract(b_, loc, off0, static_cast<uint32_t>(static_cast<int64_t>(l)));
            mlir::Value o1L = llvmExtract(b_, loc, off1, static_cast<uint32_t>(static_cast<int64_t>(l)));
            mlir::Value byteOff;
            if (isTexAtomic) {
                descP = maskDescPtrForLane(loc, h, realMask, l);
                basePtr = loadDescBasePtr(loc, descP);
                isReserved = cInt(loc, i1Ty(), 0);
                byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
                mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
                okPtr = mlir::LLVM::AndOp::create(
                    b_, loc, mL, mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP));
                // RWTexture2D atomic: off0=x, off1=y
                mlir::Value rp = loadDescU32Field(loc, descP, offsetof(NacelleDescriptor, row_pitch), 0);
                mlir::Value y64 = mlir::LLVM::MulOp::create(
                    b_, loc, mlir::LLVM::ZExtOp::create(b_, loc, i64Ty(), o1L),
                    mlir::LLVM::ZExtOp::create(b_, loc, i64Ty(), rp));
                mlir::Value x64 = mlir::LLVM::MulOp::create(
                    b_, loc, mlir::LLVM::ZExtOp::create(b_, loc, i64Ty(), o0L), cInt(loc, i64Ty(), elemBytes));
                byteOff = mlir::LLVM::AddOp::create(b_, loc, y64, x64);
            } else {
                BufferDescInfo info = uniformBufInfo.has_value()
                                          ? *uniformBufInfo
                                          : loadBufferDescInfo(loc, b_, h, maskDescPtrForLane(loc, h, realMask, l), /*needElemStride=*/false, needAtomicStride);
                descP = info.descP;
                basePtr = info.basePtr;
                isReserved = info.isReserved;
                byteSize = info.byteSize;
                okPtr = mlir::LLVM::AndOp::create(b_, loc, mL, info.okPtr);
                byteOff = computeBufferByteOffsetScalar(loc, b_, h, info.stride64, o0L, o1L, isRawBufAtomic, hasOff1);
            }
            mlir::Value cEnd = mlir::LLVM::AddOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), elemBytes));
            mlir::Value doOp = mlir::LLVM::AndOp::create(
                b_, loc, okPtr,
                mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ule, cEnd, byteSize));
            mlir::Value newL = llvmExtract(b_, loc, newValVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
            mlir::Value cmpL = isCmpXchg ? llvmExtract(b_, loc, cmpValVec, static_cast<uint32_t>(static_cast<int64_t>(l))) : mlir::Value();

            auto makeZeroRes = [&](mlir::OpBuilder& zb, mlir::Location zloc) -> mlir::Value {
                if (isFloat) {
                    auto ft = llvm::cast<mlir::FloatType>(resTy);
                    return mlir::LLVM::ConstantOp::create(zb, zloc, ft, mlir::FloatAttr::get(ft, mlir::FloatAttr::get(ft, 0.0).getValue()));
                }
                return mlir::LLVM::ConstantOp::create(zb, zloc, resTy, zb.getIntegerAttr(resTy, 0));
            };

            mlir::Value resL = mlir::scf::IfOp::create(
                                   b_, loc, doOp,
                                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                       mlir::Value ep =
                                           mlir::scf::IfOp::create(
                                               nb, nloc, isReserved,
                                               [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                                                   mlir::scf::YieldOp::create(
                                                       rb, rloc,
                                                       mlir::ValueRange{resolveReservedBufferElemPtr(
                                                           rb, rloc, descP, basePtr, byteOff, elemBytes)});
                                               },
                                               [&](mlir::OpBuilder& rb, mlir::Location rloc) {
                                                   mlir::scf::YieldOp::create(
                                                       rb, rloc,
                                                       mlir::ValueRange{mlir::LLVM::GEPOp::create(
                                                           rb, rloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{byteOff})});
                                               })
                                               .getResult(0);
                                       mlir::Value epValid = mlir::LLVM::ICmpOp::create(
                                           nb, nloc, mlir::LLVM::ICmpPredicate::ne, ep,
                                           mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()));
                                       mlir::Value r =
                                           mlir::scf::IfOp::create(
                                               nb, nloc, epValid,
                                               [&](mlir::OpBuilder& ab, mlir::Location aloc) {
                                                   mlir::Value ar;
                                                   if (isCmpXchg) {
                                                       mlir::Value cOp = isFloat ? mlir::Value(mlir::LLVM::BitcastOp::create(ab, aloc, intTy, cmpL)) : cmpL;
                                                       mlir::Value nOp = isFloat ? mlir::Value(mlir::LLVM::BitcastOp::create(ab, aloc, intTy, newL)) : newL;
                                                       auto cx = mlir::LLVM::AtomicCmpXchgOp::create(
                                                           ab, aloc, ep, cOp, nOp,
                                                           mlir::LLVM::AtomicOrdering::seq_cst,
                                                           mlir::LLVM::AtomicOrdering::acquire);
                                                       ar = mlir::LLVM::ExtractValueOp::create(ab, aloc, cx, llvm::ArrayRef<int64_t>{0});
                                                       if (isFloat) ar = mlir::LLVM::BitcastOp::create(ab, aloc, resTy, ar);
                                                   } else if (isFloat && toBinOp(atomicOp) == mlir::LLVM::AtomicBinOp::xchg) {
                                                       mlir::Value nOp = mlir::LLVM::BitcastOp::create(ab, aloc, intTy, newL);
                                                       ar = mlir::LLVM::AtomicRMWOp::create(
                                                           ab, aloc, mlir::LLVM::AtomicBinOp::xchg, ep, nOp, mlir::LLVM::AtomicOrdering::seq_cst);
                                                       ar = mlir::LLVM::BitcastOp::create(ab, aloc, resTy, ar);
                                                   } else {
                                                       ar = mlir::LLVM::AtomicRMWOp::create(
                                                           ab, aloc, toBinOp(atomicOp), ep, newL, mlir::LLVM::AtomicOrdering::seq_cst);
                                                   }
                                                   mlir::scf::YieldOp::create(ab, aloc, mlir::ValueRange{ar});
                                               },
                                               [&](mlir::OpBuilder& ab, mlir::Location aloc) {
                                                   mlir::scf::YieldOp::create(ab, aloc, mlir::ValueRange{makeZeroRes(ab, aloc)});
                                               })
                                               .getResult(0);
                                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{r});
                                   },
                                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{makeZeroRes(nb, nloc)});
                                   })
                                   .getResult(0);
            outVec = llvmInsert(b_, loc, outVec, resL, static_cast<uint32_t>(static_cast<int64_t>(l)));
        }
        setMapped(call.getResult(), {SimdVal::Kind::Vector, resTy, outVec, {}, {}});
        return true;
    }

    std::pair<mlir::Value, mlir::Value> computeQuadFiniteDiff(mlir::Location loc, mlir::Value vec, bool fine) {
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
        mlir::Value ax = mlir::LLVM::ShuffleVectorOp::create(b_, loc, vec, vec, loX);
        mlir::Value bx = mlir::LLVM::ShuffleVectorOp::create(b_, loc, vec, vec, hiX);
        mlir::Value ay = mlir::LLVM::ShuffleVectorOp::create(b_, loc, vec, vec, loY);
        mlir::Value by = mlir::LLVM::ShuffleVectorOp::create(b_, loc, vec, vec, hiY);
        return {mlir::LLVM::FSubOp::create(b_, loc, bx, ax), mlir::LLVM::FSubOp::create(b_, loc, by, ay)};
    }

    bool packetizeTextureOp(mlir::LLVM::CallOp call, uint32_t op, mlir::Value execMask, mlir::Value realMask, std::string&) {
        mlir::Location loc = call.getLoc();
        auto vecOrZero = [&](unsigned argIdx, mlir::Type ty, double z = 0.0) -> mlir::Value {
            mlir::Value v = call.getOperand(argIdx);
            if (isUndefOrPoison(v)) return mlir::isa<mlir::FloatType>(ty) ? splatFlt(loc, ty, z) : splatInt(loc, ty, static_cast<uint64_t>(z));
            return asVec(loc, getMapped(loc, v));
        };

        if (op == 72) { // getDimensions(op, handle, mip)
            SimdVal h = getMapped(loc, call.getOperand(1));
            mlir::Value mipVec = vecOrZero(2, i32Ty(), 0.0);
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_tex_get_dims", nullptr, {ptrTy(), i32Ty(), ptrTy()});
            mlir::Value tmp4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(i32Ty(), 4), 1, 16);
            llvm::SmallVector<mlir::Value, 4> dims(4, splatInt(loc, i32Ty(), 0));
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value m = llvmExtract(b_, loc, mipVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::LLVM::CallOp::create(b_, loc, fn, mlir::ValueRange{laneValue(loc, h, l), m, tmp4});
                for (unsigned c = 0; c < 4; ++c) {
                    mlir::Value ep = gepByteOffset(loc, tmp4, c * 4);
                    dims[c] = llvmInsert(b_, loc, dims[c], loadScalar(loc, i32Ty(), ep), static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            for (unsigned c = 0; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), dims[c], {}, {}});
            setMapped(call.getResult(), st);
            return true;
        }

        if (op == 67) { // textureStore(op, uav, c0, c1, c2, v0, v1, v2, v3, mask)
            SimdVal h = getMapped(loc, call.getOperand(1));
            mlir::Value x = vecOrZero(2, i32Ty(), 0.0);
            mlir::Value y = vecOrZero(3, i32Ty(), 0.0);
            mlir::Value z = vecOrZero(4, i32Ty(), 0.0);
            mlir::Value r = vecOrZero(5, f32Ty(), 0.0);
            mlir::Value g = vecOrZero(6, f32Ty(), 0.0);
            mlir::Value b = vecOrZero(7, f32Ty(), 0.0);
            mlir::Value a = vecOrZero(8, f32Ty(), 0.0);
            // If uav stores uint/int values, bitcast? Actually DXIL textureStore.i32 passes i32. Convert if needed.
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_tex_store", nullptr,
                                            {ptrTy(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty()});
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(realMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, realMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value dp = laneValue(loc, h, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    auto ext = [&](mlir::Value v) {
                        mlir::Value ev = llvmExtract(nb, nloc, v, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        if (auto it = llvm::dyn_cast<mlir::IntegerType>(ev.getType())) {
                            if (it.getWidth() < 32) {
                                ev = mlir::LLVM::ZExtOp::create(nb, nloc, nb.getI32Type(), ev);
                            } else if (it.getWidth() > 32) {
                                ev = mlir::LLVM::TruncOp::create(nb, nloc, nb.getI32Type(), ev);
                            }
                            ev = mlir::LLVM::BitcastOp::create(nb, nloc, nb.getF32Type(), ev);
                        }
                        return ev;
                    };
                    mlir::LLVM::CallOp::create(
                        nb, nloc, fn,
                        mlir::ValueRange{
                            dp,
                            llvmExtract(nb, nloc, x, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, y, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, z, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            ext(r), ext(g), ext(b), ext(a)});
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            }
            return true;
        }

        if (op == 66) { // textureLoad(op, srv, mip, x, y, z, off0, off1, off2)
            SimdVal h = getMapped(loc, call.getOperand(1));
            mlir::Value mip = vecOrZero(2, i32Ty(), 0.0);
            mlir::Value x = mlir::LLVM::AddOp::create(b_, loc, vecOrZero(3, i32Ty(), 0.0), vecOrZero(6, i32Ty(), 0.0));
            mlir::Value y = mlir::LLVM::AddOp::create(b_, loc, vecOrZero(4, i32Ty(), 0.0), vecOrZero(7, i32Ty(), 0.0));
            mlir::Value z = mlir::LLVM::AddOp::create(b_, loc, vecOrZero(5, i32Ty(), 0.0), vecOrZero(8, i32Ty(), 0.0));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_tex_load", i32Ty(), {ptrTy(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            mlir::Value out4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 4), 1, 16);
            llvm::SmallVector<mlir::Value, 4> rgba(4, splatFlt(loc, f32Ty(), 0.0));
            mlir::Value statusVec = splatInt(loc, i32Ty(), 1);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value dp = laneValue(loc, h, l);
                auto stIf = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        auto c = mlir::LLVM::CallOp::create(
                            nb, nloc, fn,
                            mlir::ValueRange{
                                dp,
                                llvmExtract(nb, nloc, x, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, y, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, z, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, mip, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                out4});
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult()});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), 1))});
                    });
                statusVec = llvmInsert(b_, loc, statusVec, stIf.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
                for (unsigned c = 0; c < 4; ++c) {
                    mlir::Value ep = gepByteOffset(loc, out4, c * 4);
                    rgba[c] = llvmInsert(b_, loc, rgba[c], loadScalar(loc, f32Ty(), ep), static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            mlir::Type resElem = llvm::cast<mlir::LLVM::LLVMStructType>(st.origType).getBody()[0];
            for (unsigned c = 0; c < 4; ++c) {
                mlir::Value v = rgba[c];
                if (auto it = llvm::dyn_cast<mlir::IntegerType>(resElem)) {
                    v = mlir::LLVM::BitcastOp::create(b_, loc, vecTy(i32Ty()), v);
                    if (it.getWidth() < 32) {
                        v = mlir::LLVM::TruncOp::create(b_, loc, vecTy(resElem), v);
                    } else if (it.getWidth() > 32) {
                        v = mlir::LLVM::ZExtOp::create(b_, loc, vecTy(resElem), v);
                    }
                }
                st.fields.push_back({SimdVal::Kind::Vector, resElem, v, {}, {}});
            }
            st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), statusVec, {}, {}});
            setMapped(call.getResult(), st);
            return true;
        }

        if (op == 73 || op == 74) { // textureGather / textureGatherCmp
            bool isCmp = (op == 74);
            SimdVal texH = getMapped(loc, call.getOperand(1));
            SimdVal sampH = getMapped(loc, call.getOperand(2));
            mlir::Value u = vecOrZero(3, f32Ty(), 0.0);
            mlir::Value v = vecOrZero(4, f32Ty(), 0.0);
            mlir::Value w = vecOrZero(5, f32Ty(), 0.0);
            mlir::Value offX = vecOrZero(7, i32Ty(), 0.0);
            mlir::Value offY = vecOrZero(8, i32Ty(), 0.0);
            mlir::Value comp = vecOrZero(9, i32Ty(), 0.0);
            mlir::Value cmpV = isCmp ? vecOrZero(10, f32Ty(), 0.0) : mlir::Value();
            auto fn = isCmp
                          ? getOrDeclareRuntimeFn(
                                loc, "nacelle_tex_gather_cmp", i32Ty(),
                                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), ptrTy()})
                          : getOrDeclareRuntimeFn(
                                loc, "nacelle_tex_gather", i32Ty(),
                                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            mlir::Value out4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 4), 1, 16);
            llvm::SmallVector<mlir::Value, 4> rgba(4, splatFlt(loc, f32Ty(), 0.0));
            mlir::Value statusVec = splatInt(loc, i32Ty(), 1);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::Value sp = laneValue(loc, sampH, l);
                auto stIf = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        llvm::SmallVector<mlir::Value, 10> args{
                            tp, sp,
                            llvmExtract(nb, nloc, u, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, v, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, w, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, offX, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, offY, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, comp, static_cast<uint32_t>(static_cast<int64_t>(l)))};
                        if (isCmp) args.push_back(llvmExtract(nb, nloc, cmpV, static_cast<uint32_t>(static_cast<int64_t>(l))));
                        args.push_back(out4);
                        auto c = mlir::LLVM::CallOp::create(nb, nloc, fn, args);
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult()});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), 1))});
                    });
                statusVec = llvmInsert(b_, loc, statusVec, stIf.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
                for (unsigned c = 0; c < 4; ++c) {
                    mlir::Value ep = gepByteOffset(loc, out4, c * 4);
                    rgba[c] = llvmInsert(b_, loc, rgba[c], loadScalar(loc, f32Ty(), ep), static_cast<uint32_t>(static_cast<int64_t>(l)));
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            for (unsigned c = 0; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), rgba[c], {}, {}});
            st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), statusVec, {}, {}});
            setMapped(call.getResult(), st);
            return true;
        }

        if (op == 81) { // calculateLOD(op, srv, samp, coord0, coord1, coord2, clamped) -> float
            SimdVal texH = getMapped(loc, call.getOperand(1));
            SimdVal sampH = getMapped(loc, call.getOperand(2));
            mlir::Value u = vecOrZero(3, f32Ty(), 0.0);
            mlir::Value v = vecOrZero(4, f32Ty(), 0.0);
            mlir::Value w = vecOrZero(5, f32Ty(), 0.0);
            mlir::Value clampedVec = asVec(loc, getMapped(loc, call.getOperand(6)));

            auto projFn = getOrDeclareRuntimeFn(
                loc, "nacelle_tex_project_lod_coords", nullptr,
                {ptrTy(), f32Ty(), f32Ty(), f32Ty(), ptrTy()});
            mlir::Value out3 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 3), 1, 16);
            mlir::Value su = splatFlt(loc, f32Ty(), 0.0);
            mlir::Value sv = su;
            mlir::Value sw = su;
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::LLVM::CallOp::create(
                        nb, nloc, projFn,
                        mlir::ValueRange{
                            tp,
                            llvmExtract(nb, nloc, u, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, v, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            llvmExtract(nb, nloc, w, static_cast<uint32_t>(static_cast<int64_t>(l))),
                            out3});
                    mlir::scf::YieldOp::create(nb, nloc);
                });
                su = llvmInsert(b_, loc, su, loadScalar(loc, f32Ty(), out3), static_cast<uint32_t>(static_cast<int64_t>(l)));
                sv = llvmInsert(b_, loc, sv, loadScalar(loc, f32Ty(), gepByteOffset(loc, out3, 4)), static_cast<uint32_t>(static_cast<int64_t>(l)));
                sw = llvmInsert(b_, loc, sw, loadScalar(loc, f32Ty(), gepByteOffset(loc, out3, 8)), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }

            mlir::Value ddxU = splatFlt(loc, f32Ty(), 0.0), ddxV = ddxU, ddxW = ddxU;
            mlir::Value ddyU = ddxU, ddyV = ddxU, ddyW = ddxU;
            if (hooks_.isPixelShader) {
                auto [ux, uy] = computeQuadFiniteDiff(loc, su, false);
                auto [vx, vy] = computeQuadFiniteDiff(loc, sv, false);
                auto [wx, wy] = computeQuadFiniteDiff(loc, sw, false);
                ddxU = ux; ddyU = uy; ddxV = vx; ddyV = vy; ddxW = wx; ddyW = wy;
            }

            auto lodFn = getOrDeclareRuntimeFn(
                loc, "nacelle_tex_calc_lod", f32Ty(),
                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), i32Ty()});
            mlir::Value resVec = splatFlt(loc, f32Ty(), 0.0);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::Value sp = laneValue(loc, sampH, l);
                auto r = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value c1 = llvmExtract(nb, nloc, clampedVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                        mlir::Value c32 = mlir::LLVM::ZExtOp::create(nb, nloc, nb.getI32Type(), c1);
                        auto c = mlir::LLVM::CallOp::create(
                            nb, nloc, lodFn,
                            mlir::ValueRange{
                                tp, sp,
                                llvmExtract(nb, nloc, ddxU, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, ddxV, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, ddxW, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, ddyU, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, ddyV, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, ddyW, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                c32});
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult()});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, nb.getF32Type(), mlir::FloatAttr::get(nb.getF32Type(), llvm::APFloat(0.0f)))});
                    });
                resVec = llvmInsert(b_, loc, resVec, r.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, f32Ty(), resVec, {}, {}});
            return true;
        }

        if (op == 64 || op == 65) { // sampleCmp / sampleCmpLevelZero
            SimdVal texH = getMapped(loc, call.getOperand(1));
            SimdVal sampH = getMapped(loc, call.getOperand(2));
            mlir::Value u = vecOrZero(3, f32Ty(), 0.0);
            mlir::Value v = vecOrZero(4, f32Ty(), 0.0);
            mlir::Value w = vecOrZero(5, f32Ty(), 0.0);
            mlir::Value offX = vecOrZero(7, i32Ty(), 0.0);
            mlir::Value offY = vecOrZero(8, i32Ty(), 0.0);
            mlir::Value cmpV = vecOrZero(10, f32Ty(), 0.0);
            uint32_t isL0 = (op == 65) ? 1u : 0u;
            mlir::Value expLod = splatFlt(loc, f32Ty(), 0.0);
            if (op == 64 && hooks_.isPixelShader) {
                auto [dux, duy] = computeQuadFiniteDiff(loc, u, true);
                auto [dvx, dvy] = computeQuadFiniteDiff(loc, v, true);
                mlir::Value d2 = mlir::LLVM::MaxNumOp::create(
                    b_, loc,
                    mlir::LLVM::FAddOp::create(b_, loc, mlir::LLVM::FMulOp::create(b_, loc, dux, dux), mlir::LLVM::FMulOp::create(b_, loc, dvx, dvx)),
                    mlir::LLVM::FAddOp::create(b_, loc, mlir::LLVM::FMulOp::create(b_, loc, duy, duy), mlir::LLVM::FMulOp::create(b_, loc, dvy, dvy)));
                // Multiply by approx tex size or let explicit lod stay ~0 if constant; good approximation
                expLod = mlir::LLVM::FMulOp::create(
                    b_, loc, splatFlt(loc, f32Ty(), 0.5),
                    emitMathUnary(loc, "log2", createFp<mlir::LLVM::MaxNumOp>(loc, vecTy(f32Ty()), d2, splatFlt(loc, f32Ty(), 1e-12))));
                expLod = mlir::LLVM::MaxNumOp::create(b_, loc, expLod, splatFlt(loc, f32Ty(), 0.0));
            }
            auto fn = getOrDeclareRuntimeFn(
                loc, "nacelle_tex_sample_cmp", f32Ty(),
                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            mlir::Value statusPtr = createEntryAlloca(loc, i32Ty(), 1, 4);
            mlir::Value resVec = splatFlt(loc, f32Ty(), 0.0);
            mlir::Value statusVec = splatInt(loc, i32Ty(), 1);
            for (uint32_t l = 0; l < W_; ++l) {
                if (isLaneMaskConstFalse(execMask, l)) continue;
                mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::Value sp = laneValue(loc, sampH, l);
                auto r = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        auto c = mlir::LLVM::CallOp::create(
                            nb, nloc, fn,
                            mlir::ValueRange{
                                tp, sp,
                                llvmExtract(nb, nloc, u, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, v, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, w, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, cmpV, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, expLod, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), isL0)),
                                llvmExtract(nb, nloc, offX, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                llvmExtract(nb, nloc, offY, static_cast<uint32_t>(static_cast<int64_t>(l))),
                                statusPtr});
                        mlir::Value stVal = mlir::LLVM::LoadOp::create(nb, nloc, i32Ty(), statusPtr);
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult(), stVal});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc,
                            mlir::ValueRange{
                                mlir::LLVM::ConstantOp::create(nb, nloc, nb.getF32Type(), mlir::FloatAttr::get(nb.getF32Type(), llvm::APFloat(0.0f))),
                                mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), 1))});
                    });
                resVec = llvmInsert(b_, loc, resVec, r.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
                statusVec = llvmInsert(b_, loc, statusVec, r.getResult(1), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), resVec, {}, {}});
            for (unsigned c = 1; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), splatFlt(loc, f32Ty(), 0.0), {}, {}});
            st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), statusVec, {}, {}});
            setMapped(call.getResult(), st);
            return true;
        }

        // Sample (60), SampleBias (61), SampleLevel (62), SampleGrad (63)
        SimdVal texH = getMapped(loc, call.getOperand(1));
        SimdVal sampH = getMapped(loc, call.getOperand(2));
        mlir::Value u = vecOrZero(3, f32Ty(), 0.0);
        mlir::Value v = vecOrZero(4, f32Ty(), 0.0);
        mlir::Value w = vecOrZero(5, f32Ty(), 0.0);
        mlir::Value c3 = vecOrZero(6, f32Ty(), 0.0);
        mlir::Value offX = vecOrZero(7, i32Ty(), 0.0);
        mlir::Value offY = vecOrZero(8, i32Ty(), 0.0);
        mlir::Value offZ = vecOrZero(9, i32Ty(), 0.0);

        uint32_t lodMode = 0;
        mlir::Value ddxU = splatFlt(loc, f32Ty(), 0.0), ddxV = ddxU, ddxW = ddxU, ddyU = ddxU, ddyV = ddxU, ddyW = ddxU;
        mlir::Value bias = ddxU, expLod = ddxU, clampLod = splatFlt(loc, f32Ty(), -1.0);

        if (op == 60 || op == 61) {
            lodMode = (op == 61) ? 1u : 0u;
            if (hooks_.isPixelShader) {
                auto [ux, uy] = computeQuadFiniteDiff(loc, u, false);
                auto [vx, vy] = computeQuadFiniteDiff(loc, v, false);
                auto [wx, wy] = computeQuadFiniteDiff(loc, w, false);
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
            loc, "nacelle_tex_sample", i32Ty(),
            {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(),
             f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(),
             f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), ptrTy()});
        mlir::Value out4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 4), 1, 16);
        llvm::SmallVector<mlir::Value, 4> rgba(4, splatFlt(loc, f32Ty(), 0.0));
        mlir::Value statusVec = splatInt(loc, i32Ty(), 1);
        for (uint32_t l = 0; l < W_; ++l) {
            if (isLaneMaskConstFalse(execMask, l)) continue;
            mlir::Value mL = llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
            mlir::Value tp = laneValue(loc, texH, l);
            mlir::Value sp = laneValue(loc, sampH, l);
            auto stIf = mlir::scf::IfOp::create(
                b_, loc, mL,
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    auto ex = [&](mlir::Value vv) { return llvmExtract(nb, nloc, vv, static_cast<uint32_t>(static_cast<int64_t>(l))); };
                    auto c = mlir::LLVM::CallOp::create(
                        nb, nloc, fn,
                        mlir::ValueRange{
                            tp, sp, ex(u), ex(v), ex(w), ex(c3),
                            ex(ddxU), ex(ddxV), ex(ddxW), ex(ddyU), ex(ddyV), ex(ddyW),
                            ex(bias), ex(expLod),
                            mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), lodMode)),
                            ex(offX), ex(offY), ex(offZ), ex(clampLod), out4});
                    mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult()});
                },
                [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::scf::YieldOp::create(
                        nb, nloc, mlir::ValueRange{mlir::LLVM::ConstantOp::create(nb, nloc, i32Ty(), nb.getIntegerAttr(i32Ty(), 1))});
                });
            statusVec = llvmInsert(b_, loc, statusVec, stIf.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
            for (unsigned c = 0; c < 4; ++c) {
                mlir::Value ep = gepByteOffset(loc, out4, c * 4);
                rgba[c] = llvmInsert(b_, loc, rgba[c], loadScalar(loc, f32Ty(), ep), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
        }
        SimdVal st;
        st.kind = SimdVal::Kind::Struct;
        st.origType = call.getResult().getType();
        for (unsigned c = 0; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), rgba[c], {}, {}});
        st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), statusVec, {}, {}});
        setMapped(call.getResult(), st);
        return true;
    }

    bool packetizeWaveOp(mlir::LLVM::CallOp call, uint32_t op, mlir::Value activeMask, std::string&) {
        mlir::Location loc = call.getLoc();
        auto arg = [&](unsigned i) { return call.getOperand(i); };

        if (op == 111) { // waveGetLaneIndex
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), laneIota(loc), {}, {}});
            return true;
        }
        if (op == 112) { // waveGetLaneCount
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), W_), {}, {}});
            return true;
        }

        // Convert activeMask (vector<Wxi1>) into 4 x i32 ballot words covering up to 128 lanes.
        auto buildBallotWords = [&](mlir::Value m) -> std::array<mlir::Value, 4> {
            mlir::Value zero = cInt(loc, i32Ty(), 0);
            if (W_ <= 32) {
                mlir::Value bits = mlir::LLVM::ZExtOp::create(b_, loc, vecTy(i32Ty()), m);
                mlir::Value shl = mlir::LLVM::ShlOp::create(b_, loc, bits, laneIota(loc));
                mlir::Value w0 = mlir::LLVM::vector_reduce_or::create(b_, loc, i32Ty(), shl);
                return {w0, zero, zero, zero};
            }
            mlir::VectorType v32i32 = mlir::VectorType::get({32}, i32Ty());
            llvm::SmallVector<int32_t, 32> iotaVals(32);
            for (int32_t i = 0; i < 32; ++i) iotaVals[i] = i;
            mlir::Value iota32 = mlir::LLVM::ConstantOp::create(
                b_, loc, v32i32,
                mlir::DenseElementsAttr::get(llvm::cast<mlir::ShapedType>(v32i32), llvm::ArrayRef<int32_t>(iotaVals)));
            std::array<mlir::Value, 4> words{zero, zero, zero, zero};
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (uint32_t w = 0; w < numWords; ++w) {
                llvm::SmallVector<int32_t, 32> sliceIdx(32);
                for (int32_t i = 0; i < 32; ++i) sliceIdx[i] = static_cast<int32_t>(w * 32u) + i;
                mlir::Value slice = mlir::LLVM::ShuffleVectorOp::create(b_, loc, m, m, sliceIdx);
                mlir::Value bits = mlir::LLVM::ZExtOp::create(b_, loc, v32i32, slice);
                mlir::Value shl = mlir::LLVM::ShlOp::create(b_, loc, bits, iota32);
                words[w] = mlir::LLVM::vector_reduce_or::create(b_, loc, i32Ty(), shl);
            }
            return words;
        };

        auto buildFirstActiveLane = [&](mlir::Value m) -> mlir::Value {
            auto words = buildBallotWords(m);
            if (W_ <= 32) {
                return mlir::LLVM::UMinOp::create(
                    b_, loc, mlir::LLVM::CountTrailingZerosOp::create(b_, loc, words[0], false), cInt(loc, i32Ty(), W_ - 1));
            }
            mlir::Value firstIdx = cInt(loc, i32Ty(), W_ - 1);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (int w = static_cast<int>(numWords) - 1; w >= 0; --w) {
                mlir::Value nz = mlir::LLVM::ICmpOp::create(
                    b_, loc, mlir::LLVM::ICmpPredicate::ne, words[w], cInt(loc, i32Ty(), 0));
                mlir::Value idx = mlir::LLVM::AddOp::create(
                    b_, loc, cInt(loc, i32Ty(), static_cast<uint64_t>(w) * 32u),
                    mlir::LLVM::CountTrailingZerosOp::create(b_, loc, words[w], false));
                firstIdx = mlir::LLVM::SelectOp::create(b_, loc, nz, idx, firstIdx);
            }
            return firstIdx;
        };

        if (op == 110) { // waveIsFirstLane
            mlir::Value firstIdx = buildFirstActiveLane(activeMask);
            mlir::Value firstVec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i32Ty())).getElementType(), firstIdx);
            mlir::Value isFirst = mlir::LLVM::ICmpOp::create(
                b_, loc, mlir::LLVM::ICmpPredicate::eq, laneIota(loc), firstVec);
            isFirst = mlir::LLVM::AndOp::create(b_, loc, isFirst, activeMask);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i1Ty(), isFirst, {}, {}});
            return true;
        }
        if (op == 113 || op == 114) { // waveAnyTrue / waveAllTrue
            mlir::Value condVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value scalarRes;
            if (op == 113) {
                mlir::Value actCond = mlir::LLVM::AndOp::create(b_, loc, condVec, activeMask);
                scalarRes = mlir::LLVM::vector_reduce_or::create(b_, loc, i1Ty(), actCond);
            } else {
                // !active || cond across all lanes
                mlir::Value inact = mlir::LLVM::XOrOp::create(b_, loc, activeMask, allTrueMask(loc));
                mlir::Value impl = mlir::LLVM::OrOp::create(b_, loc, condVec, inact);
                scalarRes = mlir::LLVM::vector_reduce_and::create(b_, loc, i1Ty(), impl);
            }
            mlir::Value bvec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i1Ty())).getElementType(), scalarRes);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i1Ty(), bvec, {}, {}});
            return true;
        }
        if (op == 116) { // waveActiveBallot(cond) -> %dx.types.fouri32
            mlir::Value condVec = mlir::LLVM::AndOp::create(b_, loc, asVec(loc, getMapped(loc, arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            for (int i = 0; i < 4; ++i) {
                st.fields.push_back({SimdVal::Kind::Vector, i32Ty(),
                                     llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i32Ty())).getElementType(), words[i]), {}, {}});
            }
            setMapped(call.getResult(), st);
            return true;
        }
        if (op == 118) { // waveReadLaneFirst(val)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value firstIdx = buildFirstActiveLane(activeMask);
            mlir::Value scalar = mlir::LLVM::ExtractElementOp::create(b_, loc, valVec, firstIdx);
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(et)).getElementType(), scalar), {}, {}});
            return true;
        }
        if (op == 117) { // waveReadLaneAt(val, lane)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value laneVec = asVec(loc, getMapped(loc, arg(2)));
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(et));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value li = llvmExtract(b_, loc, laneVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value clamped = mlir::LLVM::UMinOp::create(b_, loc, li, cInt(loc, i32Ty(), W_ - 1));
                mlir::Value s = mlir::LLVM::ExtractElementOp::create(b_, loc, valVec, clamped);
                outVec = llvmInsert(b_, loc, outVec, s, static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, outVec, {}, {}});
            return true;
        }
        if (op == 115) { // waveActiveAllEqual(val)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value firstIdx = buildFirstActiveLane(activeMask);
            mlir::Value firstScalar = mlir::LLVM::ExtractElementOp::create(b_, loc, valVec, firstIdx);
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            mlir::Value firstVec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(et)).getElementType(), firstScalar);
            mlir::Value eq = mlir::isa<mlir::FloatType>(et)
                                 ? mlir::LLVM::FCmpOp::create(b_, loc, mlir::LLVM::FCmpPredicate::oeq, valVec, firstVec).getResult()
                                 : mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, valVec, firstVec).getResult();
            mlir::Value inact = mlir::LLVM::XOrOp::create(b_, loc, activeMask, allTrueMask(loc));
            mlir::Value allEq = mlir::LLVM::vector_reduce_and::create(
                b_, loc, i1Ty(), mlir::LLVM::OrOp::create(b_, loc, eq, inact));
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i1Ty(), llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i1Ty())).getElementType(), allEq), {}, {}});
            return true;
        }
        if (op == 119 || op == 120) { // waveActiveOp / waveActiveBit
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            uint32_t sop = (op == 119) ? static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0)) : 0u;
            bool isFloat = mlir::isa<mlir::FloatType>(et);

            mlir::Value identity;
            enum class WaveRedKind { And, Or, Xor, Add, Mul, MinF, MaxF, MinUI, MaxUI, MinSI, MaxSI };
            WaveRedKind rk = WaveRedKind::Add;
            if (op == 120) {
                if (kind == 0) { // AND
                    rk = WaveRedKind::And;
                    identity = splatInt(loc, et, ~uint64_t(0));
                } else if (kind == 1) { // OR
                    rk = WaveRedKind::Or;
                    identity = splatInt(loc, et, 0);
                } else { // XOR
                    rk = WaveRedKind::Xor;
                    identity = splatInt(loc, et, 0);
                }
            } else if (kind == 0) { // SUM
                rk = WaveRedKind::Add;
                identity = isFloat ? splatFlt(loc, et, 0.0) : splatInt(loc, et, 0);
            } else if (kind == 1) { // PROD
                rk = WaveRedKind::Mul;
                identity = isFloat ? splatFlt(loc, et, 1.0) : splatInt(loc, et, 1);
            } else if (kind == 2) { // MIN
                if (isFloat) {
                    rk = WaveRedKind::MinF;
                    identity = splatFlt(loc, et, std::numeric_limits<double>::infinity());
                } else if (sop != 0) {
                    rk = WaveRedKind::MinUI;
                    identity = splatInt(loc, et, ~uint64_t(0));
                } else {
                    rk = WaveRedKind::MinSI;
                    unsigned bw = et.getIntOrFloatBitWidth();
                    uint64_t sMax = (bw >= 64) ? uint64_t(INT64_MAX) : ((uint64_t(1) << (bw - 1)) - 1);
                    identity = splatInt(loc, et, sMax);
                }
            } else { // MAX
                if (isFloat) {
                    rk = WaveRedKind::MaxF;
                    identity = splatFlt(loc, et, -std::numeric_limits<double>::infinity());
                } else if (sop != 0) {
                    rk = WaveRedKind::MaxUI;
                    identity = splatInt(loc, et, 0);
                } else {
                    rk = WaveRedKind::MaxSI;
                    unsigned bw = et.getIntOrFloatBitWidth();
                    uint64_t sMin = (bw >= 64) ? uint64_t(INT64_MIN) : (uint64_t(1) << (bw - 1));
                    identity = splatInt(loc, et, sMin);
                }
            }
            mlir::Value masked = mlir::LLVM::SelectOp::create(b_, loc, activeMask, valVec, identity);
            mlir::Value reduced;
            switch (rk) {
            case WaveRedKind::And: reduced = mlir::LLVM::vector_reduce_and::create(b_, loc, et, masked); break;
            case WaveRedKind::Or: reduced = mlir::LLVM::vector_reduce_or::create(b_, loc, et, masked); break;
            case WaveRedKind::Xor: reduced = mlir::LLVM::vector_reduce_xor::create(b_, loc, et, masked); break;
            case WaveRedKind::Add:
                reduced = isFloat
                              ? mlir::Value(createFp<mlir::LLVM::vector_reduce_fadd>(b_, loc, et, cFlt(loc, et, 0.0), masked))
                              : mlir::Value(mlir::LLVM::vector_reduce_add::create(b_, loc, et, masked));
                break;
            case WaveRedKind::Mul:
                reduced = isFloat
                              ? mlir::Value(createFp<mlir::LLVM::vector_reduce_fmul>(b_, loc, et, cFlt(loc, et, 1.0), masked))
                              : mlir::Value(mlir::LLVM::vector_reduce_mul::create(b_, loc, et, masked));
                break;
            case WaveRedKind::MinF: reduced = mlir::LLVM::vector_reduce_fmin::create(b_, loc, et, masked); break;
            case WaveRedKind::MaxF: reduced = mlir::LLVM::vector_reduce_fmax::create(b_, loc, et, masked); break;
            case WaveRedKind::MinUI: reduced = mlir::LLVM::vector_reduce_umin::create(b_, loc, et, masked); break;
            case WaveRedKind::MaxUI: reduced = mlir::LLVM::vector_reduce_umax::create(b_, loc, et, masked); break;
            case WaveRedKind::MinSI: reduced = mlir::LLVM::vector_reduce_smin::create(b_, loc, et, masked); break;
            case WaveRedKind::MaxSI: reduced = mlir::LLVM::vector_reduce_smax::create(b_, loc, et, masked); break;
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(et)).getElementType(), reduced), {}, {}});
            return true;
        }
        if (op == 121) { // wavePrefixOp (exclusive prefix sum or prod)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            bool isMul = (kind == 1);
            bool isFloat = mlir::isa<mlir::FloatType>(et);
            mlir::Value idScalar = isFloat ? cFlt(loc, et, isMul ? 1.0 : 0.0) : cInt(loc, et, isMul ? 1 : 0);
            mlir::Value idVec = isFloat ? splatFlt(loc, et, isMul ? 1.0 : 0.0) : splatInt(loc, et, isMul ? 1 : 0);
            mlir::Value contrib = mlir::LLVM::SelectOp::create(b_, loc, activeMask, valVec, idVec);
            mlir::Value acc = idScalar;
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(et));
            for (uint32_t l = 0; l < W_; ++l) {
                outVec = llvmInsert(b_, loc, outVec, acc, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value vL = llvmExtract(b_, loc, contrib, static_cast<uint32_t>(static_cast<int64_t>(l)));
                if (isFloat) {
                    acc = isMul ? mlir::LLVM::FMulOp::create(b_, loc, acc, vL).getResult()
                                : mlir::LLVM::FAddOp::create(b_, loc, acc, vL).getResult();
                } else {
                    acc = isMul ? mlir::LLVM::MulOp::create(b_, loc, acc, vL).getResult()
                                : mlir::LLVM::AddOp::create(b_, loc, acc, vL).getResult();
                }
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, outVec, {}, {}});
            return true;
        }
        if (op == 123) { // quadOp(val, kind): 0=ReadAcrossX, 1=ReadAcrossY, 2=ReadAcrossDiagonal
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            uint32_t xorMask = (kind == 0) ? 1u : (kind == 1 ? 2u : 3u);
            llvm::SmallVector<int32_t, 16> indices;
            indices.reserve(W_);
            for (uint32_t l = 0; l < W_; ++l) {
                uint32_t peer = l ^ xorMask;
                indices.push_back(peer < W_ ? static_cast<int32_t>(peer) : static_cast<int32_t>(l));
            }
            mlir::Value res = mlir::LLVM::ShuffleVectorOp::create(b_, loc, valVec, valVec, indices);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, res, {}, {}});
            return true;
        }
        if (op == 135) { // waveAllBitCount(cond) -> i32
            mlir::Value condVec = mlir::LLVM::AndOp::create(b_, loc, asVec(loc, getMapped(loc, arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            mlir::Value cnt = mlir::LLVM::CtPopOp::create(b_, loc, words[0]);
            for (uint32_t w = 1; w < numWords; ++w) {
                mlir::Value wc = mlir::LLVM::CtPopOp::create(b_, loc, words[w]);
                cnt = mlir::LLVM::AddOp::create(b_, loc, cnt, wc);
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), llvmSplat(loc, i32Ty(), cnt), {}, {}});
            return true;
        }
        if (op == 136) { // wavePrefixBitCount(cond) -> i32
            mlir::Value condVec = mlir::LLVM::AndOp::create(b_, loc, asVec(loc, getMapped(loc, arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            if (W_ <= 32) {
                mlir::Value ballotVec = llvmSplat(loc, i32Ty(), words[0]);
                llvm::SmallVector<int32_t, 32> laneMasks;
                laneMasks.reserve(W_);
                for (uint32_t l = 0; l < W_; ++l) {
                    laneMasks.push_back(static_cast<int32_t>((uint32_t(1) << l) - 1u));
                }
                mlir::Value maskVec = mlir::LLVM::ConstantOp::create(
                    b_, loc, vecTy(i32Ty()),
                    mlir::DenseElementsAttr::get(llvm::cast<mlir::ShapedType>(vecTy(i32Ty())), llvm::ArrayRef<int32_t>(laneMasks)));
                mlir::Value masked = mlir::LLVM::AndOp::create(b_, loc, ballotVec, maskVec);
                mlir::Value cntVec = mlir::LLVM::CtPopOp::create(b_, loc, masked);
                setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), cntVec, {}, {}});
                return true;
            }
            mlir::VectorType v32i32 = mlir::VectorType::get({32}, i32Ty());
            llvm::SmallVector<int32_t, 32> laneMasks32(32);
            for (int32_t l = 0; l < 32; ++l) {
                laneMasks32[l] = static_cast<int32_t>((uint32_t(1) << l) - 1u);
            }
            mlir::Value maskVec32 = mlir::LLVM::ConstantOp::create(
                b_, loc, v32i32,
                mlir::DenseElementsAttr::get(llvm::cast<mlir::ShapedType>(v32i32), llvm::ArrayRef<int32_t>(laneMasks32)));
            mlir::Value p32 = mlir::LLVM::PoisonOp::create(b_, loc, v32i32);
            llvm::SmallVector<int32_t, 32> zeros32(32, 0);
            auto splat32 = [&](mlir::Value s) {
                mlir::Value ins = llvmInsert(b_, loc, p32, s, 0);
                return mlir::LLVM::ShuffleVectorOp::create(b_, loc, ins, p32, zeros32);
            };
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i32Ty()));
            mlir::Value prefixSum = cInt(loc, i32Ty(), 0);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (uint32_t w = 0; w < numWords; ++w) {
                mlir::Value bVec = splat32(words[w]);
                mlir::Value inWord = mlir::LLVM::CtPopOp::create(b_, loc, mlir::LLVM::AndOp::create(b_, loc, bVec, maskVec32));
                mlir::Value chunk = mlir::LLVM::AddOp::create(b_, loc, inWord, splat32(prefixSum));
                for (uint32_t i = 0; i < 32u && (w * 32u + i) < W_; ++i) {
                    outVec = llvmInsert(loc, outVec, llvmExtract(loc, chunk, i), w * 32u + i);
                }
                if (w + 1 < numWords) {
                    prefixSum = mlir::LLVM::AddOp::create(b_, loc, prefixSum, mlir::LLVM::CtPopOp::create(b_, loc, words[w]));
                }
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), outVec, {}, {}});
            return true;
        }
        if (op == 166) { // waveMultiPrefixOp(value, mask0, mask1, mask2, mask3, op, sop)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            std::array<mlir::Value, 4> maskVecs{
                asVec(loc, getMapped(loc, arg(2))),
                asVec(loc, getMapped(loc, arg(3))),
                asVec(loc, getMapped(loc, arg(4))),
                asVec(loc, getMapped(loc, arg(5))),
            };
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(6)).value_or(0)); // 0=Sum, 1=BitAnd, 2=BitOr, 3=BitXor, 4=Prod
            bool isFloat = mlir::isa<mlir::FloatType>(et);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);

            std::string fnName = "__nacelle_wave_multi_prefix_" + std::to_string(kind) + "_" +
                                 (isFloat ? "f" : "i") + std::to_string(et.getIntOrFloatBitWidth()) +
                                 "_w" + std::to_string(W_);
            auto helperFn = mod_.lookupSymbol<mlir::func::FuncOp>(fnName);
            if (!helperFn) {
                llvm::SmallVector<mlir::Type, 6> argTypes;
                argTypes.push_back(vecTy(et));
                for (uint32_t w = 0; w < numWords; ++w) {
                    argTypes.push_back(vecTy(i32Ty()));
                }
                argTypes.push_back(vecTy(i1Ty()));
                auto fnTy = b_.getFunctionType(argTypes, {vecTy(et)});

                mlir::OpBuilder::InsertionGuard guard(b_);
                b_.setInsertionPointToEnd(&mod_.getBodyRegion().front());
                helperFn = mlir::func::FuncOp::create(b_, loc, fnName, fnTy);
                helperFn.setPrivate();
                mlir::Block* entry = helperFn.addEntryBlock();
                b_.setInsertionPointToStart(entry);

                mlir::Value fnValVec = entry->getArgument(0);
                llvm::SmallVector<mlir::Value, 4> fnMaskVecs;
                for (uint32_t w = 0; w < numWords; ++w) {
                    fnMaskVecs.push_back(entry->getArgument(1 + w));
                }
                mlir::Value fnActiveMask = entry->getArgument(1 + numWords);

                mlir::Value idScalar;
                if (kind == 1) {
                    idScalar = cInt(loc, et, ~uint64_t(0));
                } else if (kind == 4) {
                    idScalar = isFloat ? cFlt(loc, et, 1.0) : cInt(loc, et, 1);
                } else {
                    idScalar = isFloat ? cFlt(loc, et, 0.0) : cInt(loc, et, 0);
                }

                mlir::Value c0 = mlir::LLVM::ConstantOp::create(b_, loc, i32Ty(), b_.getI32IntegerAttr(0));
                mlir::Value c1 = mlir::LLVM::ConstantOp::create(b_, loc, i32Ty(), b_.getI32IntegerAttr(1));
                mlir::Value cW = mlir::LLVM::ConstantOp::create(b_, loc, i32Ty(), b_.getI32IntegerAttr(W_));
                mlir::Value initOut = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(et));

                auto outer = mlir::scf::ForOp::create(
                    b_, loc, c0, cW, c1, mlir::ValueRange{initOut},
                    [&](mlir::OpBuilder& ob, mlir::Location oloc, mlir::Value lIdx, mlir::ValueRange outerArgs) {
                        mlir::Value curOut = outerArgs[0];
                        llvm::SmallVector<mlir::Value, 4> maskL(numWords);
                        for (uint32_t w = 0; w < numWords; ++w) {
                            maskL[w] = mlir::LLVM::ExtractElementOp::create(ob, oloc, fnMaskVecs[w], lIdx);
                        }
                        auto inner = mlir::scf::ForOp::create(
                            ob, oloc, c0, lIdx, c1, mlir::ValueRange{idScalar},
                            [&](mlir::OpBuilder& ib, mlir::Location iloc, mlir::Value jIdx, mlir::ValueRange innerArgs) {
                                mlir::Value acc = innerArgs[0];
                                mlir::Value j32 = jIdx;
                                mlir::Value maskWordL = maskL[0];
                                if (numWords > 1) {
                                    mlir::Value wIdx = mlir::LLVM::LShrOp::create(
                                        ib, iloc, j32, mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 5)));
                                    for (uint32_t w = 1; w < numWords; ++w) {
                                        mlir::Value isW = mlir::LLVM::ICmpOp::create(
                                            ib, iloc, mlir::LLVM::ICmpPredicate::eq, wIdx,
                                            mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), w)));
                                        maskWordL = mlir::LLVM::SelectOp::create(ib, iloc, isW, maskL[w], maskWordL);
                                    }
                                }
                                mlir::Value bitIdx = mlir::LLVM::AndOp::create(
                                    ib, iloc, j32, mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 31)));
                                mlir::Value shifted = mlir::LLVM::LShrOp::create(ib, iloc, maskWordL, bitIdx);
                                mlir::Value bitJ = mlir::LLVM::AndOp::create(
                                    ib, iloc, shifted, mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 1)));
                                mlir::Value inGroup = mlir::LLVM::ICmpOp::create(
                                    ib, iloc, mlir::LLVM::ICmpPredicate::ne, bitJ,
                                    mlir::LLVM::ConstantOp::create(ib, iloc, i32Ty(), ib.getIntegerAttr(i32Ty(), 0)));
                                mlir::Value actJ = mlir::LLVM::ExtractElementOp::create(ib, iloc, fnActiveMask, jIdx);
                                mlir::Value take = mlir::LLVM::AndOp::create(ib, iloc, inGroup, actJ);
                                mlir::Value valJ = mlir::LLVM::ExtractElementOp::create(ib, iloc, fnValVec, jIdx);

                                mlir::Value cand;
                                if (kind == 0) {
                                    cand = isFloat ? mlir::LLVM::FAddOp::create(ib, iloc, acc, valJ).getResult()
                                                   : mlir::LLVM::AddOp::create(ib, iloc, acc, valJ).getResult();
                                } else if (kind == 1) {
                                    cand = mlir::LLVM::AndOp::create(ib, iloc, acc, valJ);
                                } else if (kind == 2) {
                                    cand = mlir::LLVM::OrOp::create(ib, iloc, acc, valJ);
                                } else if (kind == 3) {
                                    cand = mlir::LLVM::XOrOp::create(ib, iloc, acc, valJ);
                                } else {
                                    cand = isFloat ? mlir::LLVM::FMulOp::create(ib, iloc, acc, valJ).getResult()
                                                   : mlir::LLVM::MulOp::create(ib, iloc, acc, valJ).getResult();
                                }
                                mlir::Value nextAcc = mlir::LLVM::SelectOp::create(ib, iloc, take, cand, acc);
                                mlir::scf::YieldOp::create(ib, iloc, mlir::ValueRange{nextAcc});
                            });
                        mlir::Value nextOut = mlir::LLVM::InsertElementOp::create(ob, oloc, curOut, inner.getResult(0), lIdx);
                        mlir::scf::YieldOp::create(ob, oloc, mlir::ValueRange{nextOut});
                    });
                mlir::func::ReturnOp::create(b_, loc, mlir::ValueRange{outer.getResult(0)});
            }

            llvm::SmallVector<mlir::Value, 6> callArgs;
            callArgs.push_back(valVec);
            for (uint32_t w = 0; w < numWords; ++w) {
                callArgs.push_back(maskVecs[w]);
            }
            callArgs.push_back(activeMask);
            auto callRes = mlir::func::CallOp::create(b_, loc, helperFn, callArgs);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, callRes.getResult(0), {}, {}});
            return true;
        }
        return false;
    }

    bool packetizeDxrOp(mlir::LLVM::CallOp call, uint32_t op, mlir::Value execMask, std::string& outError) {
        mlir::Location loc = call.getLoc();
        auto arg = [&](unsigned i) -> mlir::Value {
            return i < call.getNumOperands() ? call.getOperand(i) : mlir::Value();
        };
        auto laneVec = [&](mlir::Value v) -> mlir::Value {
            return asVec(loc, getMapped(loc, v));
        };
        auto activeAt = [&](uint32_t l) -> mlir::Value {
            return llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
        };

        auto runPerActiveLaneVoid = [&](auto&& emitCall) {
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value act = activeAt(l);
                mlir::scf::IfOp::create(b_, loc, act, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    emitCall(nb, nloc, l);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            }
        };

        auto runPerActiveLaneScalar = [&](mlir::Type elemTy, mlir::Value fallbackVal, auto&& emitCall) {
            mlir::Value outVec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(elemTy)).getElementType(), fallbackVal);
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value act = activeAt(l);
                auto ifOp = mlir::scf::IfOp::create(
                    b_, loc, act,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value val = emitCall(nb, nloc, l);
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{val});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{fallbackVal});
                    });
                outVec = llvmInsert(b_, loc, outVec, ifOp.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, elemTy, outVec, {}, {}});
        };

        // DispatchRaysIndex (145) & DispatchRaysDimensions (146)
        if (op == 145 || op == 146) {
            const auto& arr = (op == 145) ? hooks_.threadId : hooks_.dispatchRaysDim;
            auto bcastI32 = [&](mlir::Value scalar) -> mlir::Value {
                return llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(i32Ty())).getElementType(), scalar);
            };
            if (auto comp = getConstIntValue(arg(1))) {
                const uint32_t c = static_cast<uint32_t>(*comp);
                mlir::Value v = (c < 3 && arr[c]) ? arr[c] : bcastI32(cInt(loc, i32Ty(), 0));
                setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), v, {}, {}});
                return true;
            }
            mlir::Value compVec = mlir::LLVM::SExtOp::create(b_, loc, vecTy(i32Ty()), laneVec(arg(1)));
            mlir::Value v0 = arr[0] ? arr[0] : bcastI32(cInt(loc, i32Ty(), 0));
            mlir::Value v1 = arr[1] ? arr[1] : bcastI32(cInt(loc, i32Ty(), 0));
            mlir::Value v2 = arr[2] ? arr[2] : bcastI32(cInt(loc, i32Ty(), 0));
            mlir::Value is1 = mlir::LLVM::ICmpOp::create(
                b_, loc, mlir::LLVM::ICmpPredicate::eq, compVec, bcastI32(cInt(loc, i32Ty(), 1)));
            mlir::Value is2 = mlir::LLVM::ICmpOp::create(
                b_, loc, mlir::LLVM::ICmpPredicate::eq, compVec, bcastI32(cInt(loc, i32Ty(), 2)));
            mlir::Value sel01 = mlir::LLVM::SelectOp::create(b_, loc, is1, v1, v0);
            mlir::Value res = mlir::LLVM::SelectOp::create(b_, loc, is2, v2, sel01);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), res, {}, {}});
            return true;
        }

        // TraceRay (157)
        if (op == 157) {
            SimdVal tlasHandle = getMapped(loc, arg(1));
            mlir::Value uniformTlasPtr;
            if (tlasHandle.lanes.size() == 1) {
                uniformTlasPtr = loadDescBasePtr(loc, tlasHandle.lanes[0]);
            }
            mlir::Value flagsVec = laneVec(arg(2));
            mlir::Value maskVec = laneVec(arg(3));
            mlir::Value rayContribVec = laneVec(arg(4));
            mlir::Value geomMultVec = laneVec(arg(5));
            mlir::Value missIdxVec = laneVec(arg(6));
            mlir::Value oxVec = laneVec(arg(7));
            mlir::Value oyVec = laneVec(arg(8));
            mlir::Value ozVec = laneVec(arg(9));
            mlir::Value tMinVec = laneVec(arg(10));
            mlir::Value dxVec = laneVec(arg(11));
            mlir::Value dyVec = laneVec(arg(12));
            mlir::Value dzVec = laneVec(arg(13));
            mlir::Value tMaxVec = laneVec(arg(14));
            SimdVal payloadVal = getMapped(loc, arg(15));

            auto fn = getOrDeclareRuntimeFn(
                loc, "nacelle_dxr_trace_ray", nullptr,
                {ptrTy(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), i32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty(), ptrTy()});

            runPerActiveLaneVoid([&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value tlasPtr = uniformTlasPtr;
                if (!tlasPtr) {
                    mlir::Value descPtr = l < tlasHandle.lanes.size()
                                              ? tlasHandle.lanes[l]
                                              : mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()).getResult();
                    tlasPtr = mlir::LLVM::LoadOp::create(nb, nloc, ptrTy(), descPtr).getResult();
                }
                mlir::Value pPtr = l < payloadVal.lanes.size()
                                       ? payloadVal.lanes[l]
                                       : (!payloadVal.lanes.empty()
                                              ? payloadVal.lanes[0]
                                              : mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()).getResult());
                auto ext = [&](mlir::Value v) -> mlir::Value {
                    return llvmExtract(nb, nloc, v, static_cast<uint32_t>(static_cast<int64_t>(l)));
                };
                mlir::LLVM::CallOp::create(
                    nb, nloc, fn,
                    mlir::ValueRange{tlasPtr, ext(flagsVec), ext(maskVec), ext(rayContribVec),
                                     ext(geomMultVec), ext(missIdxVec), ext(oxVec), ext(oyVec),
                                     ext(ozVec), ext(tMinVec), ext(dxVec), ext(dyVec),
                                     ext(dzVec), ext(tMaxVec), pPtr});
            });
            return true;
        }

        // ReportHit (158)
        if (op == 158) {
            mlir::Value tHitVec = laneVec(arg(1));
            mlir::Value hitKindVec = laneVec(arg(2));
            SimdVal attrVal = getMapped(loc, arg(3));
            int32_t attrBytes = 8;
            if (auto allocaOp = arg(3).getDefiningOp<mlir::LLVM::AllocaOp>()) {
                mlir::DataLayout dl(mod_);
                attrBytes = static_cast<int32_t>(dl.getTypeSize(allocaOp.getElemType()));
            }
            mlir::Value attrBytesConst = cInt(loc, i32Ty(), static_cast<int64_t>(attrBytes));
            auto fn = getOrDeclareRuntimeFn(
                loc, "nacelle_dxr_report_hit", i32Ty(), {f32Ty(), i32Ty(), ptrTy(), i32Ty()});
            runPerActiveLaneScalar(i1Ty(), cInt(loc, i1Ty(), 0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value t = llvmExtract(nb, nloc, tHitVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value k = llvmExtract(nb, nloc, hitKindVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value aPtr = l < attrVal.lanes.size()
                                       ? attrVal.lanes[l]
                                       : (!attrVal.lanes.empty()
                                              ? attrVal.lanes[0]
                                              : mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()).getResult());
                mlir::Value r = mlir::LLVM::CallOp::create(
                    nb, nloc, fn, mlir::ValueRange{t, k, aPtr, attrBytesConst}).getResult();
                return mlir::LLVM::TruncOp::create(nb, nloc, i1Ty(), r).getResult();
            });
            return true;
        }

        // CallShader (159)
        if (op == 159) {
            mlir::Value idxVec = laneVec(arg(1));
            SimdVal paramVal = getMapped(loc, arg(2));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_dxr_call_shader", nullptr, {i32Ty(), ptrTy()});
            runPerActiveLaneVoid([&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value idx = llvmExtract(nb, nloc, idxVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value pPtr = l < paramVal.lanes.size()
                                       ? paramVal.lanes[l]
                                       : (!paramVal.lanes.empty()
                                              ? paramVal.lanes[0]
                                              : mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()).getResult());
                mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{idx, pPtr});
            });
            return true;
        }

        // IgnoreHit (155) & AcceptHitAndEndSearch (156)
        if (op == 155 || op == 156) {
            const char* fnName = (op == 155) ? "nacelle_dxr_ignore_hit" : "nacelle_dxr_accept_hit_and_end_search";
            auto fn = getOrDeclareRuntimeFn(loc, fnName, nullptr, {});
            runPerActiveLaneVoid([&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t) {
                mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{});
            });
            return true;
        }

        // Scalar float state queries: RayTMin (153), RayTCurrent (154)
        if (op == 153 || op == 154) {
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_dxr_state_f32", f32Ty(), {i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(loc, f32Ty(), 0.0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t) {
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst}).getResult();
            });
            return true;
        }

        // Vector float state queries: WorldRayOrigin (147), WorldRayDirection (148),
        // ObjectRayOrigin (149), ObjectRayDirection (150)
        if (op >= 147 && op <= 150) {
            mlir::Value compVec = mlir::LLVM::SExtOp::create(b_, loc, vecTy(i32Ty()), laneVec(arg(1)));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_dxr_state_vec", f32Ty(), {i32Ty(), i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(loc, f32Ty(), 0.0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value c = llvmExtract(nb, nloc, compVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, c}).getResult();
            });
            return true;
        }

        // Matrix float state queries: ObjectToWorld (151), WorldToObject (152)
        if (op == 151 || op == 152) {
            mlir::Value rowVec = laneVec(arg(1));
            mlir::Value colVec = mlir::LLVM::SExtOp::create(b_, loc, vecTy(i32Ty()), laneVec(arg(2)));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_dxr_state_mat", f32Ty(), {i32Ty(), i32Ty(), i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(loc, f32Ty(), 0.0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value r = llvmExtract(nb, nloc, rowVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value c = llvmExtract(nb, nloc, colVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, r, c}).getResult();
            });
            return true;
        }

        // Scalar i32 state queries: InstanceID (141), InstanceIndex (142), HitKind (143),
        // RayFlags (144), PrimitiveIndex (161), GeometryIndex (213)
        if (op == 141 || op == 142 || op == 143 || op == 144 || op == 161 || op == 213) {
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_dxr_state_i32", i32Ty(), {i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(i32Ty(), cInt(loc, i32Ty(), 0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t) {
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst}).getResult();
            });
            return true;
        }

        outError = "Unsupported DXR opcode: " + std::to_string(op);
        return false;
    }

    bool packetizeRayQueryOp(mlir::LLVM::CallOp call, uint32_t op, mlir::Value execMask) {
        mlir::Location loc = call.getLoc();
        auto arg = [&](unsigned i) -> mlir::Value {
            return i < call.getNumOperands() ? call.getOperand(i) : mlir::Value();
        };
        auto laneVec = [&](mlir::Value v) -> mlir::Value {
            return asVec(loc, getMapped(loc, v));
        };
        auto laneAt = [&](mlir::Value vec, uint32_t l) -> mlir::Value {
            return llvmExtract(b_, loc, vec, static_cast<uint32_t>(static_cast<int64_t>(l)));
        };
        auto activeAt = [&](uint32_t l) -> mlir::Value {
            return llvmExtract(b_, loc, execMask, static_cast<uint32_t>(static_cast<int64_t>(l)));
        };

        auto runPerActiveLaneVoid = [&](auto&& emitCall) {
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value act = activeAt(l);
                mlir::scf::IfOp::create(b_, loc, act, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    emitCall(nb, nloc, l);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            }
        };

        auto runPerActiveLaneScalar = [&](mlir::Type elemTy, mlir::Value fallbackVal, auto&& emitCall) {
            mlir::Value outVec = llvmSplat(b_, loc, llvm::cast<mlir::VectorType>(vecTy(elemTy)).getElementType(), fallbackVal);
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value act = activeAt(l);
                auto ifOp = mlir::scf::IfOp::create(
                    b_, loc, act,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value val = emitCall(nb, nloc, l);
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{val});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{fallbackVal});
                    });
                outVec = llvmInsert(b_, loc, outVec, ifOp.getResult(0), static_cast<uint32_t>(static_cast<int64_t>(l)));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, elemTy, outVec, {}, {}});
        };

        // AllocateRayQuery (178, 258)
        if (op == 178 || op == 258) {
            mlir::Value flagsVec = laneVec(arg(1));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_alloc", i32Ty(), {i32Ty()});
            runPerActiveLaneScalar(i32Ty(), cInt(loc, i32Ty(), 0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value fl = llvmExtract(nb, nloc, flagsVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{fl}).getResult();
            });
            return true;
        }

        // RayQuery_TraceRayInline (179)
        if (op == 179) {
            mlir::Value qVec = laneVec(arg(1));
            SimdVal tlasHandle = getMapped(loc, arg(2));
            mlir::Value uniformTlasPtr;
            if (tlasHandle.lanes.size() == 1) {
                uniformTlasPtr = loadDescBasePtr(loc, tlasHandle.lanes[0]);
            }
            mlir::Value flagsVec = laneVec(arg(3));
            mlir::Value maskVec = laneVec(arg(4));
            mlir::Value oxVec = laneVec(arg(5));
            mlir::Value oyVec = laneVec(arg(6));
            mlir::Value ozVec = laneVec(arg(7));
            mlir::Value tMinVec = laneVec(arg(8));
            mlir::Value dxVec = laneVec(arg(9));
            mlir::Value dyVec = laneVec(arg(10));
            mlir::Value dzVec = laneVec(arg(11));
            mlir::Value tMaxVec = laneVec(arg(12));

            auto fn = getOrDeclareRuntimeFn(
                loc, "nacelle_rayquery_trace_inline", nullptr,
                {i32Ty(), ptrTy(), i32Ty(), i32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty(),
                 f32Ty(), f32Ty(), f32Ty(), f32Ty()});

            runPerActiveLaneVoid([&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value tlasPtr = uniformTlasPtr;
                if (!tlasPtr) {
                    mlir::Value descPtr = l < tlasHandle.lanes.size()
                                              ? tlasHandle.lanes[l]
                                              : mlir::LLVM::ZeroOp::create(nb, nloc, ptrTy()).getResult();
                    tlasPtr = mlir::LLVM::LoadOp::create(nb, nloc, ptrTy(), descPtr).getResult();
                }
                auto ext = [&](mlir::Value v) -> mlir::Value {
                    return llvmExtract(nb, nloc, v, static_cast<uint32_t>(static_cast<int64_t>(l)));
                };
                mlir::LLVM::CallOp::create(
                    nb, nloc, fn,
                    mlir::ValueRange{ext(qVec), tlasPtr, ext(flagsVec), ext(maskVec),
                                     ext(oxVec), ext(oyVec), ext(ozVec), ext(tMinVec),
                                     ext(dxVec), ext(dyVec), ext(dzVec), ext(tMaxVec)});
            });
            return true;
        }

        // RayQuery_Proceed (180)
        if (op == 180) {
            mlir::Value qVec = laneVec(arg(1));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_proceed", i32Ty(), {i32Ty()});
            runPerActiveLaneScalar(i1Ty(), cInt(loc, i1Ty(), 0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value r = mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{q}).getResult();
                return mlir::LLVM::TruncOp::create(nb, nloc, i1Ty(), r).getResult();
            });
            return true;
        }

        // RayQuery_Abort (181) & RayQuery_CommitNonOpaqueTriangleHit (182)
        if (op == 181 || op == 182) {
            mlir::Value qVec = laneVec(arg(1));
            const char* fnName = (op == 181) ? "nacelle_rayquery_abort" : "nacelle_rayquery_commit_triangle";
            auto fn = getOrDeclareRuntimeFn(loc, fnName, nullptr, {i32Ty()});
            runPerActiveLaneVoid([&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{q});
            });
            return true;
        }

        // RayQuery_CommitProceduralPrimitiveHit (183)
        if (op == 183) {
            mlir::Value qVec = laneVec(arg(1));
            mlir::Value tVec = laneVec(arg(2));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_commit_procedural", nullptr, {i32Ty(), f32Ty()});
            runPerActiveLaneVoid([&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value t = llvmExtract(nb, nloc, tVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{q, t});
            });
            return true;
        }

        // Scalar float state queries: RayTMin (198), CandidateTriangleRayT (199), CommittedRayT (200)
        if (op == 198 || op == 199 || op == 200) {
            mlir::Value qVec = laneVec(arg(1));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_state_f32", f32Ty(), {i32Ty(), i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(loc, f32Ty(), 0.0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, q}).getResult();
            });
            return true;
        }

        // Vector float state queries: Barycentrics (193, 194), WorldRay* (196, 197), ObjectRay* (205, 206, 211, 212)
        if (op == 193 || op == 194 || op == 196 || op == 197 ||
            op == 205 || op == 206 || op == 211 || op == 212) {
            mlir::Value qVec = laneVec(arg(1));
            mlir::Value compVec = mlir::LLVM::SExtOp::create(b_, loc, vecTy(i32Ty()), laneVec(arg(2)));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_state_vec", f32Ty(), {i32Ty(), i32Ty(), i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(loc, f32Ty(), 0.0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value c = llvmExtract(nb, nloc, compVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, q, c}).getResult();
            });
            return true;
        }

        // Matrix float state queries: ObjectToWorld3x4 / WorldToObject3x4 (186..189)
        if (op >= 186 && op <= 189) {
            mlir::Value qVec = laneVec(arg(1));
            mlir::Value rowVec = laneVec(arg(2));
            mlir::Value colVec = mlir::LLVM::SExtOp::create(b_, loc, vecTy(i32Ty()), laneVec(arg(3)));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_state_mat", f32Ty(), {i32Ty(), i32Ty(), i32Ty(), i32Ty()});
            mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
            runPerActiveLaneScalar(f32Ty(), cFlt(loc, f32Ty(), 0.0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value r = llvmExtract(nb, nloc, rowVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value c = llvmExtract(nb, nloc, colVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, q, r, c}).getResult();
            });
            return true;
        }

        // Scalar i32 / i1 state queries (184, 185, 190..192, 195, 201..204, 207..210, 214, 215)
        mlir::Value qVec = laneVec(arg(1));
        auto fn = getOrDeclareRuntimeFn(loc, "nacelle_rayquery_state_i32", i32Ty(), {i32Ty(), i32Ty()});
        mlir::Value opConst = cInt(loc, i32Ty(), static_cast<int64_t>(op));
        const bool isBool = call.getResult().getType().isInteger(1);
        if (isBool) {
            runPerActiveLaneScalar(i1Ty(), cInt(loc, i1Ty(), 0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                mlir::Value r = mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, q}).getResult();
                return mlir::LLVM::TruncOp::create(nb, nloc, i1Ty(), r).getResult();
            });
        } else {
            runPerActiveLaneScalar(i32Ty(), cInt(loc, i32Ty(), 0), [&](mlir::OpBuilder& nb, mlir::Location nloc, uint32_t l) {
                mlir::Value q = llvmExtract(nb, nloc, qVec, static_cast<uint32_t>(static_cast<int64_t>(l)));
                return mlir::LLVM::CallOp::create(nb, nloc, fn, mlir::ValueRange{opConst, q}).getResult();
            });
        }
        return true;
    }

    bool packetizeMathOp(mlir::LLVM::CallOp call, uint32_t op, std::string& outError) {
        if (call.getNumResults() == 0) {
            outError = "Unsupported dx.op opcode: " + std::to_string(op);
            return false;
        }
        mlir::Location loc = call.getLoc();
        auto fullResTy = call.getResult().getType();
        auto vt = llvm::dyn_cast<mlir::VectorType>(fullResTy);
        auto resTy = vt ? vt.getElementType() : fullResTy;
        auto vT = vecTy(resTy);

        if (vt) {
            SimdVal outVecStruct;
            outVecStruct.kind = SimdVal::Kind::Struct;
            outVecStruct.origType = fullResTy;
            unsigned n = static_cast<unsigned>(vt.getNumElements());
            for (unsigned elemIdx = 0; elemIdx < n; ++elemIdx) {
                mlir::Value elemOut;
                auto argV = [&](unsigned i) {
                    SimdVal sv = getMapped(loc, call.getOperand(i));
                    if (sv.kind == SimdVal::Kind::Struct && elemIdx < sv.fields.size()) {
                        return asVec(loc, sv.fields[elemIdx]);
                    }
                    return asVec(loc, sv);
                };
                auto store = [&](mlir::Value r) {
                    elemOut = r;
                    return true;
                };
                auto flushDenormF = [&](mlir::Value v) -> mlir::Value {
                    if (!resTy.isF32()) return v;
                    mlir::Value bits = mlir::LLVM::BitcastOp::create(b_, loc, vecTy(i32Ty()), v);
                    mlir::Value absBits = mlir::LLVM::AndOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x7FFFFFFFu));
                    mlir::Value isDenorm = mlir::LLVM::ICmpOp::create(
                        b_, loc, mlir::LLVM::ICmpPredicate::ult, absBits, splatInt(loc, i32Ty(), 0x00800000u));
                    mlir::Value signZero = mlir::LLVM::AndOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x80000000u));
                    mlir::Value flushed = mlir::LLVM::SelectOp::create(b_, loc, isDenorm, signZero, bits);
                    return mlir::LLVM::BitcastOp::create(b_, loc, vT, flushed);
                };
                if (!packetizeScalarMathSwitch(call, op, loc, resTy, vT, argV, store, flushDenormF, outError)) {
                    return false;
                }
                outVecStruct.fields.push_back({SimdVal::Kind::Vector, resTy, elemOut, {}, {}});
            }
            setMapped(call.getResult(), outVecStruct);
            return true;
        }

        auto argV = [&](unsigned i) { return asVec(loc, getMapped(loc, call.getOperand(i))); };
        auto store = [&](mlir::Value r) {
            setMapped(call.getResult(), {SimdVal::Kind::Vector, resTy, r, {}, {}});
            return true;
        };

        auto flushDenormF = [&](mlir::Value v) -> mlir::Value {
            if (!resTy.isF32()) return v;
            mlir::Value bits = mlir::LLVM::BitcastOp::create(b_, loc, vecTy(i32Ty()), v);
            mlir::Value absBits = mlir::LLVM::AndOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x7FFFFFFFu));
            mlir::Value isDenorm = mlir::LLVM::ICmpOp::create(
                b_, loc, mlir::LLVM::ICmpPredicate::ult, absBits, splatInt(loc, i32Ty(), 0x00800000u));
            mlir::Value signZero = mlir::LLVM::AndOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x80000000u));
            mlir::Value flushed = mlir::LLVM::SelectOp::create(b_, loc, isDenorm, signZero, bits);
            return mlir::LLVM::BitcastOp::create(b_, loc, vT, flushed);
        };

        return packetizeScalarMathSwitch(call, op, loc, resTy, vT, argV, store, flushDenormF, outError);
    }

    template <typename ArgVFn, typename StoreFn, typename FlushFn>
    bool packetizeScalarMathSwitch(
        mlir::LLVM::CallOp call, uint32_t op, mlir::Location loc, mlir::Type resTy, mlir::Type vT,
        ArgVFn&& argV, StoreFn&& store, FlushFn&& flushDenormF, std::string& outError) {
        switch (op) {
        case 6: return store(createFp<mlir::LLVM::FAbsOp>(loc, vT, argV(1)));
        case 7: { // Saturate
            mlir::Value lo = createFp<mlir::LLVM::MaxNumOp>(loc, vT, argV(1), splatFlt(loc, resTy, 0.0));
            return store(createFp<mlir::LLVM::MinNumOp>(loc, vT, lo, splatFlt(loc, resTy, 1.0)));
        }
        case 8: return store(mlir::LLVM::IsFPClass::create(b_, loc, vT, argV(1), static_cast<uint32_t>(llvm::fcNan)));
        case 9: return store(mlir::LLVM::IsFPClass::create(b_, loc, vT, argV(1), static_cast<uint32_t>(llvm::fcInf)));
        case 10: return store(mlir::LLVM::IsFPClass::create(
            b_, loc, vT, argV(1), static_cast<uint32_t>(llvm::fcNormal | llvm::fcSubnormal | llvm::fcZero)));
        case 11: return store(mlir::LLVM::IsFPClass::create(b_, loc, vT, argV(1), static_cast<uint32_t>(llvm::fcNormal)));
        case 12: return store(emitMathUnary(loc, "cos", flushDenormF(argV(1))));
        case 13: return store(emitMathUnary(loc, "sin", flushDenormF(argV(1))));
        case 14: return store(emitMathUnary(loc, "tan", flushDenormF(argV(1))));
        case 15: return store(emitMathUnary(loc, "acos", flushDenormF(argV(1))));
        case 16: return store(emitMathUnary(loc, "asin", flushDenormF(argV(1))));
        case 17: return store(emitMathUnary(loc, "atan", flushDenormF(argV(1))));
        case 18: return store(emitMathUnary(loc, "cosh", flushDenormF(argV(1))));
        case 19: return store(emitMathUnary(loc, "sinh", flushDenormF(argV(1))));
        case 20: return store(emitMathUnary(loc, "tanh", flushDenormF(argV(1))));
        case 21: return store(emitMathUnary(loc, "exp2", flushDenormF(argV(1)))); // DXIL Exp is base-2!
        case 22: { // Frc = x - floor(x)
            mlir::Value x = argV(1);
            return store(createFp<mlir::LLVM::FSubOp>(loc, vT, x, createFp<mlir::LLVM::FFloorOp>(loc, vT, x)));
        }
        case 23: return store(emitMathUnary(loc, "log2", flushDenormF(argV(1)))); // DXIL Log is base-2!
        case 24: return store(createFp<mlir::LLVM::SqrtOp>(loc, vT, argV(1)));
        case 25: { // Rsqrt = 1.0 / sqrt(x)
            mlir::Value sq = createFp<mlir::LLVM::SqrtOp>(loc, vT, argV(1));
            return store(createFp<mlir::LLVM::FDivOp>(loc, vT, splatFlt(loc, resTy, 1.0), sq));
        }
        case 26: return store(createFp<mlir::LLVM::RoundEvenOp>(loc, vT, argV(1)));
        case 27: return store(createFp<mlir::LLVM::FFloorOp>(loc, vT, argV(1)));
        case 28: return store(createFp<mlir::LLVM::FCeilOp>(loc, vT, argV(1)));
        case 29: return store(createFp<mlir::LLVM::FTruncOp>(loc, vT, argV(1)));
        case 30: return store(mlir::LLVM::BitReverseOp::create(b_, loc, vT, argV(1)));
        case 31: { // Countbits: population count, result is always i32 in DXIL
            mlir::Value x = argV(1);
            mlir::Type argTy = llvm::cast<mlir::VectorType>(x.getType()).getElementType();
            mlir::Value pop = mlir::LLVM::CtPopOp::create(b_, loc, vecTy(argTy), x);
            if (argTy != resTy) {
                pop = (argTy.getIntOrFloatBitWidth() > resTy.getIntOrFloatBitWidth())
                          ? mlir::Value(mlir::LLVM::TruncOp::create(b_, loc, vT, pop))
                          : mlir::Value(mlir::LLVM::ZExtOp::create(b_, loc, vT, pop));
            }
            return store(pop);
        }
        case 32: { // FirstbitLo: index of lowest set bit, or -1 if 0
            mlir::Value x = argV(1);
            mlir::Type argTy = llvm::cast<mlir::VectorType>(x.getType()).getElementType();
            mlir::Value tz = mlir::LLVM::CountTrailingZerosOp::create(b_, loc, vecTy(argTy), x, false);
            if (argTy != resTy) {
                tz = (argTy.getIntOrFloatBitWidth() > resTy.getIntOrFloatBitWidth())
                         ? mlir::Value(mlir::LLVM::TruncOp::create(b_, loc, vT, tz))
                         : mlir::Value(mlir::LLVM::ZExtOp::create(b_, loc, vT, tz));
            }
            mlir::Value isZero = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, x, splatInt(loc, argTy, 0));
            return store(mlir::LLVM::SelectOp::create(b_, loc, isZero, splatInt(loc, resTy, ~uint64_t(0)), tz));
        }
        case 33: // FirstbitHi: clz(x), or -1 if 0 (DXIL returns leading-zero count from MSB; DXC emits bitWidth-1 - res)
        case 34: { // FirstbitSHi: clz(x < 0 ? ~x : x), or -1 if 0 or -1
            mlir::Value x = argV(1);
            mlir::Type argTy = llvm::cast<mlir::VectorType>(x.getType()).getElementType();
            if (op == 34) {
                mlir::Value isNeg = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::slt, x, splatInt(loc, argTy, 0));
                mlir::Value inv = mlir::LLVM::XOrOp::create(b_, loc, x, splatInt(loc, argTy, ~uint64_t(0)));
                x = mlir::LLVM::SelectOp::create(b_, loc, isNeg, inv, x);
            }
            mlir::Value lz = mlir::LLVM::CountLeadingZerosOp::create(b_, loc, vecTy(argTy), x, false);
            if (argTy != resTy) {
                lz = (argTy.getIntOrFloatBitWidth() > resTy.getIntOrFloatBitWidth())
                         ? mlir::Value(mlir::LLVM::TruncOp::create(b_, loc, vT, lz))
                         : mlir::Value(mlir::LLVM::ZExtOp::create(b_, loc, vT, lz));
            }
            mlir::Value isZero = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, x, splatInt(loc, argTy, 0));
            return store(mlir::LLVM::SelectOp::create(b_, loc, isZero, splatInt(loc, resTy, ~uint64_t(0)), lz));
        }
        case 35: return store(createFp<mlir::LLVM::MaxNumOp>(loc, vT, argV(1), argV(2)));
        case 36: return store(createFp<mlir::LLVM::MinNumOp>(loc, vT, argV(1), argV(2)));
        case 37: return store(mlir::LLVM::SMaxOp::create(b_, loc, vT, argV(1), argV(2)));
        case 38: return store(mlir::LLVM::SMinOp::create(b_, loc, vT, argV(1), argV(2)));
        case 39: return store(mlir::LLVM::UMaxOp::create(b_, loc, vT, argV(1), argV(2)));
        case 40: return store(mlir::LLVM::UMinOp::create(b_, loc, vT, argV(1), argV(2)));
        case 44: { // UAddc(a, b) -> { sum, carry }
            auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(resTy);
            if (!st || st.getBody().size() < 2) {
                outError = "UAddc expected { T, i1 } struct return type";
                return false;
            }
            mlir::Type elemTy = st.getBody()[0];
            mlir::Value a = argV(1);
            mlir::Value b = argV(2);
            mlir::Value sum = mlir::LLVM::AddOp::create(b_, loc, a, b);
            mlir::Value carry = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ult, sum, a);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, sum, {}, {}});
            outSt.fields.push_back({SimdVal::Kind::Vector, i1Ty(), carry, {}, {}});
            setMapped(call.getResult(), outSt);
            return true;
        }
        case 46:
        case 47: return store(createFp<mlir::LLVM::FMAOp>(loc, vT, argV(1), argV(2), argV(3)));
        case 48:
        case 49: {
            mlir::Value prod = mlir::LLVM::MulOp::create(b_, loc, vT, argV(1), argV(2));
            return store(mlir::LLVM::AddOp::create(b_, loc, vT, prod, argV(3)));
        }
        case 54: // Dot2
        case 55: // Dot3
        case 56: { // Dot4
            unsigned n = (op == 54) ? 2u : (op == 55 ? 3u : 4u);
            mlir::Value sum = mlir::LLVM::FMulOp::create(b_, loc, vT, argV(1), argV(1 + n));
            for (unsigned c = 1; c < n; ++c) {
                mlir::Value p = mlir::LLVM::FMulOp::create(b_, loc, vT, argV(1 + c), argV(1 + n + c));
                sum = mlir::LLVM::FAddOp::create(b_, loc, vT, sum, p);
            }
            return store(sum);
        }
        case 83: // DerivCoarseX
        case 84: // DerivCoarseY
        case 85: // DerivFineX
        case 86: { // DerivFineY
            if (!hooks_.isPixelShader) return store(splatFlt(loc, resTy, 0.0));
            bool fine = (op == 85 || op == 86);
            auto [dx, dy] = computeQuadFiniteDiff(loc, argV(1), fine);
            return store((op == 83 || op == 85) ? dx : dy);
        }
        case 101: { // MakeDouble(lo, hi) -> f64
            mlir::Value lo64 = mlir::LLVM::ZExtOp::create(b_, loc, vecTy(i64Ty()), argV(1));
            mlir::Value hi64 = mlir::LLVM::ZExtOp::create(b_, loc, vecTy(i64Ty()), argV(2));
            mlir::Value hiShift = mlir::LLVM::ShlOp::create(b_, loc, hi64, splatInt(loc, i64Ty(), 32));
            mlir::Value bits = mlir::LLVM::OrOp::create(b_, loc, lo64, hiShift);
            return store(mlir::LLVM::BitcastOp::create(b_, loc, vecTy(b_.getF64Type()), bits));
        }
        case 102: { // SplitDouble(val) -> { i32 lo, i32 hi }
            auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(resTy);
            if (!st || st.getBody().size() < 2) {
                outError = "SplitDouble expected { i32, i32 } struct return type";
                return false;
            }
            mlir::Value bits = mlir::LLVM::BitcastOp::create(b_, loc, vecTy(i64Ty()), argV(1));
            mlir::Value lo = mlir::LLVM::TruncOp::create(b_, loc, vecTy(i32Ty()), bits);
            mlir::Value hi = mlir::LLVM::TruncOp::create(
                b_, loc, vecTy(i32Ty()),
                mlir::LLVM::LShrOp::create(b_, loc, bits, splatInt(loc, i64Ty(), 32)));
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), lo, {}, {}});
            outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), hi, {}, {}});
            setMapped(call.getResult(), outSt);
            return true;
        }
        case 130: { // LegacyF32ToF16(val) -> i32
            mlir::Value f16v = mlir::LLVM::FPTruncOp::create(b_, loc, vecTy(b_.getF16Type()), argV(1));
            mlir::Value i16v = mlir::LLVM::BitcastOp::create(b_, loc, vecTy(b_.getI16Type()), f16v);
            return store(mlir::LLVM::ZExtOp::create(b_, loc, vecTy(i32Ty()), i16v));
        }
        case 131: { // LegacyF16ToF32(val) -> f32
            mlir::Value i16v = mlir::LLVM::TruncOp::create(b_, loc, vecTy(b_.getI16Type()), argV(1));
            mlir::Value f16v = mlir::LLVM::BitcastOp::create(b_, loc, vecTy(b_.getF16Type()), i16v);
            return store(mlir::LLVM::FPExtOp::create(b_, loc, vecTy(f32Ty()), f16v));
        }
        case 162: { // Dot2AddHalf(acc, ax, ay, bx, by) -> f32
            mlir::Value acc = argV(1);
            mlir::Value ax = mlir::LLVM::FPExtOp::create(b_, loc, vecTy(f32Ty()), argV(2));
            mlir::Value ay = mlir::LLVM::FPExtOp::create(b_, loc, vecTy(f32Ty()), argV(3));
            mlir::Value bx = mlir::LLVM::FPExtOp::create(b_, loc, vecTy(f32Ty()), argV(4));
            mlir::Value by = mlir::LLVM::FPExtOp::create(b_, loc, vecTy(f32Ty()), argV(5));
            mlir::Value px = mlir::LLVM::FMulOp::create(b_, loc, ax, bx);
            mlir::Value py = mlir::LLVM::FMulOp::create(b_, loc, ay, by);
            mlir::Value sum = mlir::LLVM::FAddOp::create(b_, loc, acc, mlir::LLVM::FAddOp::create(b_, loc, px, py));
            return store(sum);
        }
        case 163: // Dot4AddI8Packed(acc, a, b)
        case 164: { // Dot4AddU8Packed(acc, a, b)
            bool isSigned = (op == 163);
            mlir::Value acc = argV(1);
            mlir::Value a = argV(2);
            mlir::Value b = argV(3);
            mlir::Value maskFF = splatInt(loc, i32Ty(), 0xFF);
            mlir::Value c24 = splatInt(loc, i32Ty(), 24);
            for (unsigned i = 0; i < 4; ++i) {
                mlir::Value ai, bi;
                if (isSigned) {
                    mlir::Value shlAmt = splatInt(loc, i32Ty(), 24 - i * 8);
                    ai = mlir::LLVM::AShrOp::create(b_, loc, mlir::LLVM::ShlOp::create(b_, loc, a, shlAmt), c24);
                    bi = mlir::LLVM::AShrOp::create(b_, loc, mlir::LLVM::ShlOp::create(b_, loc, b, shlAmt), c24);
                } else {
                    mlir::Value shrAmt = splatInt(loc, i32Ty(), i * 8);
                    ai = mlir::LLVM::AndOp::create(b_, loc, mlir::LLVM::LShrOp::create(b_, loc, a, shrAmt), maskFF);
                    bi = mlir::LLVM::AndOp::create(b_, loc, mlir::LLVM::LShrOp::create(b_, loc, b, shrAmt), maskFF);
                }
                acc = mlir::LLVM::AddOp::create(b_, loc, acc, mlir::LLVM::MulOp::create(b_, loc, ai, bi));
            }
            return store(acc);
        }
        case 219: { // Unpack4x8(unpackMode, pk) -> %dx.types.fouri32 / fouri16
            auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(resTy);
            if (!st || st.getBody().size() < 4) {
                outError = "Unpack4x8 expected 4-element struct return type";
                return false;
            }
            uint32_t mode = static_cast<uint32_t>(getConstIntValue(call.getOperand(1)).value_or(0));
            mlir::Value pk = argV(2);
            mlir::Type elemTy = st.getBody()[0];
            bool needTrunc = elemTy.getIntOrFloatBitWidth() < 32;
            mlir::Value maskFF = splatInt(loc, i32Ty(), 0xFF);
            mlir::Value c24 = splatInt(loc, i32Ty(), 24);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            for (unsigned i = 0; i < 4; ++i) {
                mlir::Value comp;
                if (mode == 1) { // Signed
                    mlir::Value shlAmt = splatInt(loc, i32Ty(), 24 - i * 8);
                    comp = mlir::LLVM::AShrOp::create(b_, loc, mlir::LLVM::ShlOp::create(b_, loc, pk, shlAmt), c24);
                } else { // Unsigned
                    mlir::Value shrAmt = splatInt(loc, i32Ty(), i * 8);
                    comp = mlir::LLVM::AndOp::create(b_, loc, mlir::LLVM::LShrOp::create(b_, loc, pk, shrAmt), maskFF);
                }
                if (needTrunc) {
                    comp = mlir::LLVM::TruncOp::create(b_, loc, vecTy(elemTy), comp);
                }
                outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, comp, {}, {}});
            }
            setMapped(call.getResult(), outSt);
            return true;
        }
        case 220: { // Pack4x8(packMode, x, y, z, w) -> i32
            uint32_t mode = static_cast<uint32_t>(getConstIntValue(call.getOperand(1)).value_or(0));
            mlir::Value maskFF = splatInt(loc, i32Ty(), 0xFF);
            mlir::Value acc = splatInt(loc, i32Ty(), 0);
            for (unsigned i = 0; i < 4; ++i) {
                mlir::Value c = argV(2 + i);
                mlir::Type cet = llvm::cast<mlir::VectorType>(c.getType()).getElementType();
                if (cet.getIntOrFloatBitWidth() < 32) {
                    c = mlir::LLVM::SExtOp::create(b_, loc, vecTy(i32Ty()), c);
                }
                if (mode == 1) { // U8Clamp [0, 255]
                    c = mlir::LLVM::SMaxOp::create(b_, loc, splatInt(loc, i32Ty(), 0),
                                                     mlir::LLVM::SMinOp::create(b_, loc, splatInt(loc, i32Ty(), 255), c));
                } else if (mode == 2) { // S8Clamp [-128, 127]
                    c = mlir::LLVM::SMaxOp::create(b_, loc, splatInt(loc, i32Ty(), -128),
                                                     mlir::LLVM::SMinOp::create(b_, loc, splatInt(loc, i32Ty(), 127), c));
                }
                c = mlir::LLVM::AndOp::create(b_, loc, c, maskFF);
                if (i > 0) {
                    c = mlir::LLVM::ShlOp::create(b_, loc, c, splatInt(loc, i32Ty(), i * 8));
                }
                acc = mlir::LLVM::OrOp::create(b_, loc, acc, c);
            }
            return store(acc);
        }
        default:
            outError = "Unsupported dx.op opcode: " + std::to_string(op);
            return false;
        }
    }

    mlir::OpBuilder& b_;
    mlir::MLIRContext* ctx_;
    mlir::ModuleOp mod_;
    const DxilLoadedShader& shader_;
    const RootSignatureLayout& rootLayout_;
    uint32_t W_;
    mlir::Value rootCtxPtr_;
    StageHooks& hooks_;
    mlir::Value scratchAlloca_;
    bool strictFp_ = false;
    DescInfoCache ownedDescCache_;
    DescInfoCache* descCache_ = nullptr;

    mlir::Block* getFuncEntryBlock() const {
        mlir::Block* cur = b_.getInsertionBlock();
        while (cur && cur->getParentOp() && !llvm::isa<mlir::func::FuncOp>(cur->getParentOp())) {
            cur = cur->getParentOp()->getBlock();
        }
        if (cur && cur->getParentOp()) {
            if (auto fn = llvm::dyn_cast<mlir::func::FuncOp>(cur->getParentOp())) {
                return &fn.getBody().front();
            }
        }
        return nullptr;
    }

public:
    llvm::DenseMap<mlir::Value, SimdVal> valueMap_;
};

// --- Barrier Wave-Loop Fission Helpers ---

struct SpillSlot {
    mlir::Value arrPtr; // `!llvm.ptr` pointing to [numWaves x packedStructOrArray]
    mlir::Type origTy;
    mlir::Type packedValTy;
    bool isHandle = false;
    SimdVal handleProto;
};

static mlir::Value buildFlatTidAndMask(
    mlir::OpBuilder& b, mlir::Location loc, uint32_t W,
    mlir::Value waveIdx, mlir::Value totalThreads, mlir::Value& outActiveMask) {
    mlir::Type i32Ty = b.getI32Type();
    mlir::VectorType vI32 = mlir::VectorType::get({W}, i32Ty);
    if (auto wConst = getConstIntValue(waveIdx)) {
        if (auto totConst = getConstIntValue(totalThreads)) {
            uint64_t baseVal = (*wConst) * W;
            llvm::SmallVector<int32_t, 16> tids;
            llvm::SmallVector<bool, 16> maskBits;
            tids.reserve(W);
            maskBits.reserve(W);
            for (uint32_t i = 0; i < W; ++i) {
                uint64_t tid = baseVal + i;
                tids.push_back(static_cast<int32_t>(tid));
                maskBits.push_back(tid < *totConst);
            }
            mlir::VectorType vI1 = mlir::VectorType::get({W}, b.getI1Type());
            outActiveMask = mlir::LLVM::ConstantOp::create(b, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vI1, llvm::ArrayRef<bool>(maskBits))).getType(), mlir::DenseElementsAttr::get(vI1, llvm::ArrayRef<bool>(maskBits)));
            return mlir::LLVM::ConstantOp::create(b, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vI32, llvm::ArrayRef<int32_t>(tids))).getType(), mlir::DenseElementsAttr::get(vI32, llvm::ArrayRef<int32_t>(tids)));
        }
    }
    llvm::SmallVector<int32_t, 16> iota;
    for (uint32_t i = 0; i < W; ++i) iota.push_back(static_cast<int32_t>(i));
    mlir::Value iotaVec = mlir::LLVM::ConstantOp::create(b, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vI32, llvm::ArrayRef<int32_t>(iota))).getType(), mlir::DenseElementsAttr::get(vI32, llvm::ArrayRef<int32_t>(iota)));
    mlir::Value w32 = waveIdx;
    mlir::Value base = mlir::LLVM::MulOp::create(
        b, loc, w32, mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getIntegerAttr(i32Ty, W)));
    mlir::Value baseVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vI32).getElementType(), base, W);
    mlir::Value flatTid = mlir::LLVM::AddOp::create(b, loc, baseVec, iotaVec);
    mlir::Value totVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vI32).getElementType(), totalThreads, W);
    outActiveMask = mlir::LLVM::ICmpOp::create(
        b, loc, mlir::LLVM::ICmpPredicate::ult, flatTid, totVec);
    return flatTid;
}

static void setupComputeThreadIds(
    mlir::OpBuilder& b, mlir::Location loc, uint32_t W, const uint32_t tgs[3],
    mlir::Value flatTid, mlir::Value gx, mlir::Value gy, mlir::Value gz,
    StageHooks& hooks) {
    mlir::Type i32Ty = b.getI32Type();
    mlir::VectorType vI32 = mlir::VectorType::get({W}, i32Ty);
    auto splat = [&](uint32_t val) {
        return mlir::LLVM::ConstantOp::create(b, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vI32, b.getIntegerAttr(i32Ty, val))).getType(), mlir::DenseElementsAttr::get(vI32, b.getIntegerAttr(i32Ty, val)));
    };
    mlir::Value zeroVec = splat(0);
    mlir::Value tgsX = splat(tgs[0]), tgsY = splat(tgs[1]), tgsZ = splat(tgs[2]);
    mlir::Value xy = splat(tgs[0] * tgs[1]);

    mlir::Value lx = (tgs[0] == 1) ? zeroVec : mlir::Value(mlir::LLVM::URemOp::create(b, loc, flatTid, tgsX));
    mlir::Value qx = (tgs[0] == 1) ? flatTid : mlir::Value(mlir::LLVM::UDivOp::create(b, loc, flatTid, tgsX));
    mlir::Value ly = (tgs[1] == 1) ? zeroVec : mlir::Value(mlir::LLVM::URemOp::create(b, loc, qx, tgsY));
    mlir::Value lz = (tgs[2] == 1) ? zeroVec : mlir::Value(mlir::LLVM::UDivOp::create(b, loc, flatTid, xy));

    mlir::Value gxVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vI32).getElementType(), gx, W);
    mlir::Value gyVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vI32).getElementType(), gy, W);
    mlir::Value gzVec = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vI32).getElementType(), gz, W);

    hooks.groupId = {gxVec, gyVec, gzVec};
    hooks.threadIdInGroup = {lx, ly, lz};
    hooks.flatGroupId = (tgs[0] == 1 && tgs[1] == 1 && tgs[2] == 1) ? zeroVec : flatTid;
    hooks.threadId = {
        (tgs[0] == 1) ? gxVec
                      : mlir::Value(mlir::LLVM::AddOp::create(b, loc, mlir::LLVM::MulOp::create(b, loc, gxVec, tgsX), lx)),
        (tgs[1] == 1) ? gyVec
                      : mlir::Value(mlir::LLVM::AddOp::create(b, loc, mlir::LLVM::MulOp::create(b, loc, gyVec, tgsY), ly)),
        (tgs[2] == 1) ? gzVec
                      : mlir::Value(mlir::LLVM::AddOp::create(b, loc, mlir::LLVM::MulOp::create(b, loc, gzVec, tgsZ), lz))};
}

// Collects all SSA values used inside `[segBegin, segEnd)` that are already tracked in `spills`,
// and all values defined inside `[segBegin, segEnd)` that are used strictly after `segEnd`.
static void analyzeSegmentLiveness(
    mlir::Block::iterator segBegin, mlir::Block::iterator segEnd,
    const llvm::DenseMap<mlir::Value, SpillSlot>& spills,
    llvm::SmallVectorImpl<mlir::Value>& needReload,
    llvm::SmallVectorImpl<mlir::Value>& needSpill) {
    llvm::DenseSet<mlir::Operation*> inSegOps;
    for (auto it = segBegin; it != segEnd; ++it) inSegOps.insert(&*it);

    llvm::DenseSet<mlir::Value> seenReload;
    for (auto it = segBegin; it != segEnd; ++it) {
        it->walk([&](mlir::Operation* inner) {
            for (mlir::Value opnd : inner->getOperands()) {
                if (spills.count(opnd) && seenReload.insert(opnd).second) {
                    needReload.push_back(opnd);
                }
            }
        });
    }

    auto isUsedAfterSegEnd = [&](mlir::Value v) -> bool {
        for (mlir::Operation* user : v.getUsers()) {
            // Walk up to the enclosing op inside segEnd->getBlock()
            mlir::Operation* anc = user;
            mlir::Block* parentBlk = segBegin->getBlock();
            while (anc && anc->getBlock() != parentBlk) anc = anc->getParentOp();
            if (!anc) return true;
            if (inSegOps.count(anc)) continue;
            // Must be strictly after segBegin..segEnd or be a terminator of an outer region
            if (&*segBegin != anc && !anc->isBeforeInBlock(&*segBegin)) return true;
        }
        return false;
    };

    for (auto it = segBegin; it != segEnd; ++it) {
        for (mlir::Value res : it->getResults()) {
            if (!spills.count(res) && isUsedAfterSegEnd(res)) {
                needSpill.push_back(res);
            }
        }
    }
}

static bool packetizeComputeBlockWithBarriers(
    mlir::OpBuilder& b, mlir::Location loc, mlir::ModuleOp mod,
    mlir::Block& srcBlock, const DxilLoadedShader& shader, const RootSignatureLayout& rootLayout,
    uint32_t W, uint32_t numWavesVal, mlir::Value rootCtxPtr, mlir::Value gsBase,
    mlir::Value gx, mlir::Value gy, mlir::Value gz,
    mlir::Value totalThreads, mlir::Value numWaves,
    mlir::Value waveMaskArr, // [numWaves x vector<W x i1>] current active mask per wave
    llvm::DenseMap<mlir::Value, SpillSlot>& spills,
    mlir::Operation* entryAllocaAnchor,
    const StageHooks& baseHooks,
    std::string& outError) {
    mlir::Type i32Ty = b.getI32Type();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(b.getContext());

    auto allocSpillSlot = [&](mlir::Type origTy, const SimdVal* proto = nullptr) -> SpillSlot {
        SpillSlot slot;
        slot.origTy = origTy;
        if (proto && proto->kind == SimdVal::Kind::Handle) {
            slot.isHandle = true;
            slot.handleProto = *proto;
        }
        mlir::OpBuilder eb(entryAllocaAnchor);
        StageHooks unusedHooks;
        SimdPacketizer helper(eb, mod, shader, rootLayout, W, rootCtxPtr, unusedHooks);
        llvm::SmallVector<mlir::Type, 4> flatTys;
        helper.flattenSimdTypes(origTy, flatTys);
        slot.packedValTy = (flatTys.size() == 1) ? flatTys[0] : mlir::LLVM::LLVMStructType::getLiteral(b.getContext(), flatTys);
        mlir::Value n32 = mlir::LLVM::ConstantOp::create(eb, loc, i32Ty, eb.getIntegerAttr(i32Ty, numWavesVal));
        uint32_t vecAlign = std::max<uint32_t>(64, W * 8);
        slot.arrPtr = mlir::LLVM::AllocaOp::create(eb, loc, ptrTy, slot.packedValTy, n32, vecAlign);
        return slot;
    };

    auto storeSpillAtWave = [&](mlir::OpBuilder& wb, SimdPacketizer& pkt, const SpillSlot& slot,
                                mlir::Value waveIdx, const SimdVal& sv) {
        llvm::SmallVector<mlir::Value, 4> packed;
        pkt.packSimdToValues(loc, sv, packed);
        mlir::Value valToStore;
        if (packed.size() == 1) {
            valToStore = packed[0];
        } else {
            valToStore = mlir::LLVM::UndefOp::create(wb, loc, slot.packedValTy);
            for (size_t i = 0; i < packed.size(); ++i) {
                valToStore = mlir::LLVM::InsertValueOp::create(
                    wb, loc, valToStore, packed[i], llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)});
            }
        }
        mlir::Value w32 = waveIdx;
        mlir::Value wp = mlir::LLVM::GEPOp::create(wb, loc, ptrTy, slot.packedValTy, slot.arrPtr, mlir::ValueRange{w32});
        mlir::LLVM::StoreOp::create(wb, loc, valToStore, wp, 1u);
    };

    auto loadSpillAtWave = [&](mlir::OpBuilder& wb, SimdPacketizer& pkt, const SpillSlot& slot,
                               mlir::Value waveIdx) -> SimdVal {
        mlir::Value w32 = waveIdx;
        mlir::Value wp = mlir::LLVM::GEPOp::create(wb, loc, ptrTy, slot.packedValTy, slot.arrPtr, mlir::ValueRange{w32});
        mlir::Value loaded = mlir::LLVM::LoadOp::create(wb, loc, slot.packedValTy, wp, 1u);
        llvm::SmallVector<mlir::Value, 4> flatVals;
        if (auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(slot.packedValTy); st && slot.isHandle == false && !mlir::isa<mlir::LLVM::LLVMStructType>(slot.origTy) && !mlir::isa<mlir::LLVM::LLVMArrayType>(slot.origTy)) {
            flatVals.push_back(loaded);
        } else if (auto st2 = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(slot.packedValTy)) {
            for (unsigned i = 0; i < st2.getBody().size(); ++i) {
                flatVals.push_back(mlir::LLVM::ExtractValueOp::create(
                    wb, loc, loaded, llvm::ArrayRef<int64_t>{static_cast<int64_t>(i)}));
            }
        } else {
            flatVals.push_back(loaded);
        }
        size_t cur = 0;
        return pkt.unpackSimdFromValues(loc, slot.origTy, flatVals, cur, slot.isHandle ? &slot.handleProto : nullptr);
    };

    mlir::Value c0 = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
    mlir::Value c1 = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(1));

    auto emitWaveSegment = [&](mlir::Block::iterator segBegin, mlir::Block::iterator segEnd) -> bool {
        if (segBegin == segEnd) return true;
        llvm::SmallVector<mlir::Value, 8> needReload, needSpill;
        analyzeSegmentLiveness(segBegin, segEnd, spills, needReload, needSpill);

        for (mlir::Value v : needSpill) {
            spills[v] = allocSpillSlot(v.getType());
        }

        mlir::scf::ForOp::create(
            b, loc, c0, numWaves, c1, mlir::ValueRange{},
            [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                mlir::Value baseMask;
                mlir::Value flatTid = buildFlatTidAndMask(wb, wloc, W, waveIdx, totalThreads, baseMask);
                mlir::Value w32 = waveIdx;
                mlir::Value mp = mlir::LLVM::GEPOp::create(
                    wb, wloc, ptrTy, mlir::VectorType::get({W}, wb.getI1Type()), waveMaskArr, mlir::ValueRange{w32});
                mlir::Value activeMask = mlir::LLVM::LoadOp::create(
                    wb, wloc, mlir::VectorType::get({W}, wb.getI1Type()), mp, 1u);

                StageHooks hooks = baseHooks;
                hooks.entryBlock = entryAllocaAnchor ? entryAllocaAnchor->getBlock() : baseHooks.entryBlock;
                hooks.groupSharedBase = gsBase;
                hooks.waveIdx = waveIdx;
                hooks.numWaves = numWavesVal;
                setupComputeThreadIds(wb, wloc, W, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
                SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, hooks);
                for (mlir::Value rv : needReload) {
                    pkt.valueMap_[rv] = loadSpillAtWave(wb, pkt, spills[rv], waveIdx);
                }
                mlir::Value realMask = activeMask;
                if (!pkt.packetizeBlockRange(segBegin, segEnd, activeMask, realMask, outError)) {
                    mlir::scf::YieldOp::create(wb, wloc);
                    return;
                }
                for (mlir::Value sv : needSpill) {
                    const auto& mapped = pkt.valueMap_[sv];
                    // Update proto if Handle metadata discovered inside this segment
                    if (mapped.kind == SimdVal::Kind::Handle) {
                        spills[sv].isHandle = true;
                        spills[sv].handleProto = mapped;
                    }
                    storeSpillAtWave(wb, pkt, spills[sv], waveIdx, mapped);
                }
                mlir::scf::YieldOp::create(wb, wloc);
            });
        return outError.empty();
    };

    auto it = srcBlock.begin();
    auto segBegin = it;
    while (it != srcBlock.end() && !it->hasTrait<mlir::OpTrait::IsTerminator>()) {
        mlir::Operation* op = &*it;
        if (isBarrierOp(op)) {
            if (!emitWaveSegment(segBegin, it)) return false;
            mlir::LLVM::FenceOp::create(b, loc, mlir::LLVM::AtomicOrdering::acq_rel);
            ++it;
            segBegin = it;
            continue;
        }
        if (opContainsBarrier(op)) {
            if (!emitWaveSegment(segBegin, it)) return false;

            if (auto ifOp = llvm::dyn_cast<mlir::scf::IfOp>(op)) {
                // Evaluate uniform condition at wave 0 lane 0
                StageHooks dummyHooks;
                SimdPacketizer condPkt(b, mod, shader, rootLayout, W, rootCtxPtr, dummyHooks);
                SimdVal cVal = spills.count(ifOp.getCondition())
                                   ? loadSpillAtWave(b, condPkt, spills[ifOp.getCondition()], c0)
                                   : condPkt.getMapped(loc, ifOp.getCondition());
                mlir::Value condScalar = llvmExtract(b, loc, condPkt.asVec(loc, cVal), static_cast<uint32_t>(0));
                for (mlir::Value r : ifOp.getResults()) spills[r] = allocSpillSlot(r.getType());

                mlir::scf::IfOp::create(
                    b, loc, condScalar,
                    [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                        if (!packetizeComputeBlockWithBarriers(
                                ib, iloc, mod, ifOp.getThenRegion().front(), shader, rootLayout, W, numWavesVal,
                                rootCtxPtr, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
                                entryAllocaAnchor, baseHooks, outError)) {
                            mlir::scf::YieldOp::create(ib, iloc);
                            return;
                        }
                        if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(ifOp.getThenRegion().front().getTerminator())) {
                            if (yo.getNumOperands() > 0) {
                                mlir::scf::ForOp::create(ib, iloc, c0, numWaves, c1, mlir::ValueRange{},
                                                         [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                                                             StageHooks dh;
                                                             SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, dh);
                                                             for (unsigned k = 0; k < yo.getNumOperands(); ++k) {
                                                                 SimdVal val = spills.count(yo.getOperand(k))
                                                                                   ? loadSpillAtWave(wb, pkt, spills[yo.getOperand(k)], waveIdx)
                                                                                   : pkt.getMapped(wloc, yo.getOperand(k));
                                                                 if (val.kind == SimdVal::Kind::Handle) {
                                                                     spills[ifOp.getResult(k)].isHandle = true;
                                                                     spills[ifOp.getResult(k)].handleProto = val;
                                                                 }
                                                                 storeSpillAtWave(wb, pkt, spills[ifOp.getResult(k)], waveIdx, val);
                                                             }
                                                             mlir::scf::YieldOp::create(wb, wloc);
                                                         });
                            }
                        }
                        mlir::scf::YieldOp::create(ib, iloc);
                    },
                    [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                        if (!ifOp.getElseRegion().empty()) {
                            if (!packetizeComputeBlockWithBarriers(
                                    ib, iloc, mod, ifOp.getElseRegion().front(), shader, rootLayout, W, numWavesVal,
                                    rootCtxPtr, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
                                    entryAllocaAnchor, baseHooks, outError)) {
                                mlir::scf::YieldOp::create(ib, iloc);
                                return;
                            }
                            if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(ifOp.getElseRegion().front().getTerminator())) {
                                if (yo.getNumOperands() > 0) {
                                    mlir::scf::ForOp::create(ib, iloc, c0, numWaves, c1, mlir::ValueRange{},
                                                             [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                                                                 StageHooks dh;
                                                                 SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, dh);
                                                                 for (unsigned k = 0; k < yo.getNumOperands(); ++k) {
                                                                     SimdVal val = spills.count(yo.getOperand(k))
                                                                                       ? loadSpillAtWave(wb, pkt, spills[yo.getOperand(k)], waveIdx)
                                                                                       : pkt.getMapped(wloc, yo.getOperand(k));
                                                                     if (val.kind == SimdVal::Kind::Handle) {
                                                                         spills[ifOp.getResult(k)].isHandle = true;
                                                                         spills[ifOp.getResult(k)].handleProto = val;
                                                                     }
                                                                     storeSpillAtWave(wb, pkt, spills[ifOp.getResult(k)], waveIdx, val);
                                                                 }
                                                                 mlir::scf::YieldOp::create(wb, wloc);
                                                             });
                                }
                            }
                        }
                        mlir::scf::YieldOp::create(ib, iloc);
                    });
                if (!outError.empty()) return false;
            } else if (auto whileOp = llvm::dyn_cast<mlir::scf::WhileOp>(op)) {
                mlir::Block& srcBefore = whileOp.getBefore().front();
                mlir::Block& srcAfter = whileOp.getAfter().front();

                llvm::SmallVector<SpillSlot, 8> beforeSlots, afterSlots;
                for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
                    mlir::Value iv = whileOp.getInits()[i];
                    const SimdVal* proto = (spills.count(iv) && spills[iv].isHandle) ? &spills[iv].handleProto : nullptr;
                    SpillSlot bs = allocSpillSlot(iv.getType(), proto);
                    beforeSlots.push_back(bs);
                    spills[srcBefore.getArgument(i)] = bs;
                }
                for (unsigned i = 0; i < srcAfter.getNumArguments(); ++i) {
                    SpillSlot as = allocSpillSlot(srcAfter.getArgument(i).getType());
                    afterSlots.push_back(as);
                    spills[srcAfter.getArgument(i)] = as;
                }
                for (unsigned i = 0; i < whileOp.getNumResults(); ++i) {
                    spills[whileOp.getResult(i)] = afterSlots[i];
                }

                // Initialize beforeSlots from whileOp.getInits() and zero-init afterSlots
                mlir::scf::ForOp::create(
                    b, loc, c0, numWaves, c1, mlir::ValueRange{},
                    [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                        StageHooks dh;
                        SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, dh);
                        for (unsigned i = 0; i < whileOp.getInits().size(); ++i) {
                            mlir::Value iv = whileOp.getInits()[i];
                            SimdVal sv = spills.count(iv) ? loadSpillAtWave(wb, pkt, spills[iv], waveIdx) : pkt.getMapped(wloc, iv);
                            if (sv.kind == SimdVal::Kind::Handle) {
                                beforeSlots[i].isHandle = true;
                                beforeSlots[i].handleProto = sv;
                                spills[srcBefore.getArgument(i)] = beforeSlots[i];
                            }
                            storeSpillAtWave(wb, pkt, beforeSlots[i], waveIdx, sv);
                        }
                        for (unsigned i = 0; i < afterSlots.size(); ++i) {
                            storeSpillAtWave(wb, pkt, afterSlots[i], waveIdx, pkt.makeZeroSimd(wloc, afterSlots[i].origTy));
                        }
                        mlir::scf::YieldOp::create(wb, wloc);
                    });

                // Save outer wave masks on stack
                mlir::VectorType maskVecTy = mlir::VectorType::get({W}, b.getI1Type());
                mlir::Value savedWaveMaskArr;
                {
                    mlir::OpBuilder eb(entryAllocaAnchor);
                    savedWaveMaskArr = mlir::LLVM::AllocaOp::create(
                        eb, loc, ptrTy, maskVecTy,
                        mlir::LLVM::ConstantOp::create(eb, loc, i32Ty, eb.getIntegerAttr(i32Ty, numWavesVal)),
                        std::max<uint32_t>(64, W * 8));
                }
                mlir::scf::ForOp::create(
                    b, loc, c0, numWaves, c1, mlir::ValueRange{},
                    [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                        mlir::Value w32 = waveIdx;
                        mlir::Value srcP = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                        mlir::Value dstP = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, savedWaveMaskArr, mlir::ValueRange{w32});
                        mlir::LLVM::StoreOp::create(wb, wloc, mlir::LLVM::LoadOp::create(wb, wloc, maskVecTy, srcP, 1u), dstP, 1u);
                        mlir::scf::YieldOp::create(wb, wloc);
                    });

                mlir::scf::WhileOp::create(
                    b, loc, mlir::TypeRange{}, mlir::ValueRange{},
                    [&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::ValueRange) {
                        if (!packetizeComputeBlockWithBarriers(
                                nb, nloc, mod, srcBefore, shader, rootLayout, W, numWavesVal, rootCtxPtr, gsBase,
                                gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills, entryAllocaAnchor, baseHooks, outError)) {
                            mlir::Value cFalse = mlir::LLVM::ConstantOp::create(nb, nloc, nb.getI1Type(), nb.getIntegerAttr(nb.getI1Type(), 0));
                            mlir::scf::ConditionOp::create(nb, nloc, cFalse, mlir::ValueRange{});
                            return;
                        }

                        auto condOp = llvm::cast<mlir::scf::ConditionOp>(srcBefore.getTerminator());
                        for (unsigned i = 0; i < condOp.getArgs().size(); ++i) {
                            mlir::Value ca = condOp.getArgs()[i];
                            if (spills.count(ca) && spills[ca].isHandle && !afterSlots[i].isHandle) {
                                afterSlots[i].isHandle = true;
                                afterSlots[i].handleProto = spills[ca].handleProto;
                                spills[srcAfter.getArgument(i)] = afterSlots[i];
                                if (i < whileOp.getNumResults()) spills[whileOp.getResult(i)] = afterSlots[i];
                            }
                        }

                        mlir::Value cFalse = mlir::LLVM::ConstantOp::create(nb, nloc, nb.getI1Type(), nb.getIntegerAttr(nb.getI1Type(), 0));
                        auto anyLoop = mlir::scf::ForOp::create(
                            nb, nloc, c0, numWaves, c1, mlir::ValueRange{cFalse},
                            [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange acc) {
                                mlir::Value baseMask;
                                mlir::Value flatTid = buildFlatTidAndMask(wb, wloc, W, waveIdx, totalThreads, baseMask);
                                mlir::Value w32 = waveIdx;
                                mlir::Value mp = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                                mlir::Value curMask = mlir::LLVM::LoadOp::create(wb, wloc, maskVecTy, mp, 1u);

                                StageHooks hooks = baseHooks;
                                hooks.entryBlock = entryAllocaAnchor ? entryAllocaAnchor->getBlock() : nullptr;
                                hooks.groupSharedBase = gsBase;
                                hooks.waveIdx = waveIdx;
                                hooks.numWaves = numWavesVal;
                                setupComputeThreadIds(wb, wloc, W, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
                                SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, hooks);

                                mlir::Value condOpnd = condOp.getCondition();
                                SimdVal condSimd = spills.count(condOpnd)
                                                       ? loadSpillAtWave(wb, pkt, spills[condOpnd], waveIdx)
                                                       : pkt.getMapped(wloc, condOpnd);
                                mlir::Value condVec = pkt.asVec(wloc, condSimd);
                                mlir::Value nextMask = mlir::LLVM::AndOp::create(wb, wloc, curMask, condVec);
                                mlir::LLVM::StoreOp::create(wb, wloc, nextMask, mp, 1u);

                                for (unsigned i = 0; i < condOp.getArgs().size(); ++i) {
                                    mlir::Value ca = condOp.getArgs()[i];
                                    SimdVal cv = spills.count(ca)
                                                     ? loadSpillAtWave(wb, pkt, spills[ca], waveIdx)
                                                     : pkt.getMapped(wloc, ca);
                                    if (cv.kind == SimdVal::Kind::Handle && !afterSlots[i].isHandle) {
                                        afterSlots[i].isHandle = true;
                                        afterSlots[i].handleProto = cv;
                                        spills[srcAfter.getArgument(i)] = afterSlots[i];
                                        if (i < whileOp.getNumResults()) spills[whileOp.getResult(i)] = afterSlots[i];
                                    }
                                    SimdVal prevAfter = loadSpillAtWave(wb, pkt, afterSlots[i], waveIdx);
                                    storeSpillAtWave(wb, pkt, afterSlots[i], waveIdx, pkt.selectVal(wloc, curMask, cv, prevAfter));
                                }
                                mlir::Value waveAny = pkt.orReduce(wloc, nextMask);
                                mlir::Value nextAny = mlir::LLVM::OrOp::create(wb, wloc, acc[0], waveAny);
                                mlir::scf::YieldOp::create(wb, wloc, mlir::ValueRange{nextAny});
                            });
                        mlir::scf::ConditionOp::create(nb, nloc, anyLoop.getResult(0), mlir::ValueRange{});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::ValueRange) {
                        if (!packetizeComputeBlockWithBarriers(
                                nb, nloc, mod, srcAfter, shader, rootLayout, W, numWavesVal, rootCtxPtr, gsBase,
                                gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills, entryAllocaAnchor, baseHooks, outError)) {
                            mlir::scf::YieldOp::create(nb, nloc);
                            return;
                        }
                        if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(srcAfter.getTerminator())) {
                            mlir::scf::ForOp::create(
                                nb, nloc, c0, numWaves, c1, mlir::ValueRange{},
                                [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                                    StageHooks dh;
                                    SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, dh);
                                    mlir::Value w32 = waveIdx;
                                    mlir::Value mp = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                                    mlir::Value m = mlir::LLVM::LoadOp::create(wb, wloc, maskVecTy, mp, 1u);
                                    for (unsigned i = 0; i < yo.getNumOperands(); ++i) {
                                        mlir::Value yv = yo.getOperand(i);
                                        SimdVal nextV = spills.count(yv) ? loadSpillAtWave(wb, pkt, spills[yv], waveIdx) : pkt.getMapped(wloc, yv);
                                        if (nextV.kind == SimdVal::Kind::Handle && !beforeSlots[i].isHandle) {
                                            beforeSlots[i].isHandle = true;
                                            beforeSlots[i].handleProto = nextV;
                                            spills[srcBefore.getArgument(i)] = beforeSlots[i];
                                        }
                                        SimdVal prevV = loadSpillAtWave(wb, pkt, beforeSlots[i], waveIdx);
                                        storeSpillAtWave(wb, pkt, beforeSlots[i], waveIdx, pkt.selectVal(wloc, m, nextV, prevV));
                                    }
                                    mlir::scf::YieldOp::create(wb, wloc);
                                });
                        }
                        mlir::scf::YieldOp::create(nb, nloc);
                    });

                // Restore outer wave masks
                mlir::scf::ForOp::create(
                    b, loc, c0, numWaves, c1, mlir::ValueRange{},
                    [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                        mlir::Value w32 = waveIdx;
                        mlir::Value srcP = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, savedWaveMaskArr, mlir::ValueRange{w32});
                        mlir::Value dstP = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                        mlir::LLVM::StoreOp::create(wb, wloc, mlir::LLVM::LoadOp::create(wb, wloc, maskVecTy, srcP, 1u), dstP, 1u);
                        mlir::scf::YieldOp::create(wb, wloc);
                    });
                if (!outError.empty()) return false;
            } else {
                outError = "Unsupported structured control-flow op containing barrier";
                return false;
            }
            ++it;
            segBegin = it;
            continue;
        }
        ++it;
    }

    if (!emitWaveSegment(segBegin, it)) return false;
    return true;
}

} // namespace

bool packetizeThreadGroupRegion(
    mlir::OpBuilder& b,
    mlir::Location loc,
    mlir::ModuleOp mod,
    mlir::Region& srcRegion,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    mlir::Value rootCtx,
    mlir::Value gsBase,
    mlir::Value gx,
    mlir::Value gy,
    mlir::Value gz,
    mlir::Block* entry,
    const StageHooks& baseHooks,
    std::string& outError) {
    if (srcRegion.empty()) return true;
    mlir::MLIRContext* ctx = mod.getContext();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type i32Ty = b.getI32Type();

    const uint32_t totalThreadsVal =
        std::max(1u, shader.threadGroupSize[0] * shader.threadGroupSize[1] * shader.threadGroupSize[2]);
    const uint32_t numWavesVal = (totalThreadsVal + simdWidth - 1) / simdWidth;
    mlir::Value totalThreads = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getIntegerAttr(i32Ty, totalThreadsVal));
    mlir::Value c0 = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
    mlir::Value c1 = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(1));
    mlir::Value numWaves = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(numWavesVal));

    if (!regionContainsBarrier(srcRegion)) {
        auto emitWaveBody = [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx) {
            mlir::Value activeMask;
            mlir::Value flatTid = buildFlatTidAndMask(wb, wloc, simdWidth, waveIdx, totalThreads, activeMask);
            StageHooks hooks = baseHooks;
            hooks.entryBlock = entry;
            hooks.groupSharedBase = gsBase;
            hooks.waveIdx = mlir::Value();
            hooks.numWaves = 1;
            setupComputeThreadIds(wb, wloc, simdWidth, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
            SimdPacketizer pkt(wb, mod, shader, rootLayout, simdWidth, rootCtx, hooks);
            mlir::Value realMask = activeMask;
            (void)pkt.packetizeBlock(srcRegion.front(), activeMask, realMask, outError);
        };
        if (numWavesVal == 1) {
            emitWaveBody(b, loc, c0);
            b.setInsertionPointToEnd(entry);
        } else {
            mlir::scf::ForOp::create(
                b, loc, c0, numWaves, c1, mlir::ValueRange{},
                [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                    emitWaveBody(wb, wloc, waveIdx);
                    mlir::scf::YieldOp::create(wb, wloc);
                });
        }
    } else {
        mlir::VectorType maskVecTy = mlir::VectorType::get({simdWidth}, b.getI1Type());
        mlir::Value waveMaskArr = mlir::LLVM::AllocaOp::create(
            b, loc, ptrTy, maskVecTy, mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getIntegerAttr(i32Ty, numWavesVal)),
            std::max<uint32_t>(64, simdWidth * 8));
        mlir::Operation* entryAllocaAnchor = waveMaskArr.getDefiningOp();
        // Initialize per-wave active masks
        mlir::scf::ForOp::create(
            b, loc, c0, numWaves, c1, mlir::ValueRange{},
            [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                mlir::Value activeMask;
                (void)buildFlatTidAndMask(wb, wloc, simdWidth, waveIdx, totalThreads, activeMask);
                mlir::Value w32 = waveIdx;
                mlir::Value mp = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                mlir::LLVM::StoreOp::create(wb, wloc, activeMask, mp, 1u);
                mlir::scf::YieldOp::create(wb, wloc);
            });

        llvm::DenseMap<mlir::Value, SpillSlot> spills;
        (void)packetizeComputeBlockWithBarriers(
            b, loc, mod, srcRegion.front(), shader, rootLayout, simdWidth, numWavesVal,
            rootCtx, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
            entryAllocaAnchor, baseHooks, outError);
    }
    return outError.empty();
}

bool packetizeComputeEntry(
    mlir::ModuleOp mod,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError) {
    auto srcFunc = mod.lookupSymbol<mlir::func::FuncOp>(shader.entryName);
    if (!srcFunc || srcFunc.getBody().empty()) {
        outError = "Source compute func.func not found: " + shader.entryName;
        return false;
    }

    mlir::MLIRContext* ctx = mod.getContext();
    mlir::OpBuilder b(mod.getBody(), mod.getBody()->end());
    mlir::Location loc = srcFunc.getLoc();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type i32Ty = b.getI32Type();

    // void fn(ptr rootCtx, ptr gsBase, i32 gx, i32 gy, i32 gz, i32 ngx, i32 ngy, i32 ngz)
    auto fnTy = b.getFunctionType({ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty, i32Ty}, {});
    auto dstFunc = mlir::func::FuncOp::create(b, loc, symbolName, fnTy);
    dstFunc.setArgAttr(0, "llvm.noalias", b.getUnitAttr());
    dstFunc.setArgAttr(0, "llvm.readonly", b.getUnitAttr());
    mlir::Block* entry = dstFunc.addEntryBlock();
    b.setInsertionPointToStart(entry);

    mlir::Value rootCtx = entry->getArgument(0);
    mlir::Value gsBase = entry->getArgument(1);
    mlir::Value gx = entry->getArgument(2);
    mlir::Value gy = entry->getArgument(3);
    mlir::Value gz = entry->getArgument(4);

    StageHooks baseHooks;
    if (!packetizeThreadGroupRegion(b, loc, mod, srcFunc.getBody(), shader, rootLayout,
                                    simdWidth, rootCtx, gsBase, gx, gy, gz, entry, baseHooks, outError)) {
        return false;
    }
    mlir::func::ReturnOp::create(b, loc);

    srcFunc.erase();
    if (auto helper = mod.lookupSymbol<mlir::LLVM::LLVMFuncOp>("nacelle.groupshared.ptr")) {
        if (helper.use_empty()) helper.erase();
    }
    llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 16> deadDxOps;
    for (auto fnOp : mod.getOps<mlir::LLVM::LLVMFuncOp>()) {
        if (fnOp.isDeclaration() && fnOp.getName().starts_with("dx.op.") && fnOp.use_empty()) {
            deadDxOps.push_back(fnOp);
        }
    }
    for (auto fnOp : deadDxOps) fnOp.erase();

    if (mlir::failed(mlir::verify(mod))) {
        outError = "MLIR verification failed after SIMT-to-SIMD packetization";
        return false;
    }
    return true;
}

bool packetizeRegionWithHooks(
    mlir::OpBuilder& builder,
    mlir::Location loc,
    mlir::ModuleOp mod,
    mlir::Region& srcRegion,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    mlir::Value rootCtxPtr,
    mlir::Value initExecMask,
    mlir::Value& inoutRealMask,
    StageHooks& hooks,
    std::string& outError) {
    (void)loc;
    if (srcRegion.empty()) return true;
    SimdPacketizer pkt(builder, mod, shader, rootLayout, simdWidth, rootCtxPtr, hooks);
    return pkt.packetizeBlock(srcRegion.front(), initExecMask, inoutRealMask, outError);
}

bool packetizeDxrEntry(
    mlir::ModuleOp mod,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError) {
    auto srcFunc = mod.lookupSymbol<mlir::func::FuncOp>(shader.entryName);
    if (!srcFunc || srcFunc.getBody().empty()) {
        outError = "Source DXR func.func not found: " + shader.entryName;
        return false;
    }

    mlir::MLIRContext* ctx = mod.getContext();
    mlir::OpBuilder b(mod.getBody(), mod.getBody()->end());
    mlir::Location loc = srcFunc.getLoc();
    mlir::Type ptrTy = mlir::LLVM::LLVMPointerType::get(ctx);
    mlir::Type i32Ty = b.getI32Type();
    mlir::Type i1Ty = b.getI1Type();
    mlir::VectorType vecI32Ty = mlir::VectorType::get({simdWidth}, i32Ty);
    mlir::VectorType vecI1Ty = mlir::VectorType::get({simdWidth}, i1Ty);

    // void fn(ptr rootCtx, i32 activeMask, ptr rxPtr, ptr ryPtr, ptr rzPtr,
    //         i32 dimX, i32 dimY, i32 dimZ, ptr payloadPtr, ptr attrPtr)
    auto fnTy = b.getFunctionType(
        {ptrTy, i32Ty, ptrTy, ptrTy, ptrTy, i32Ty, i32Ty, i32Ty, ptrTy, ptrTy}, {});
    auto dstFunc = mlir::func::FuncOp::create(b, loc, symbolName, fnTy);
    mlir::Block* entry = dstFunc.addEntryBlock();
    b.setInsertionPointToStart(entry);

    mlir::Value rootCtx = entry->getArgument(0);
    mlir::Value activeMaskArg = entry->getArgument(1);
    mlir::Value rxPtr = entry->getArgument(2);
    mlir::Value ryPtr = entry->getArgument(3);
    mlir::Value rzPtr = entry->getArgument(4);
    mlir::Value dimX = entry->getArgument(5);
    mlir::Value dimY = entry->getArgument(6);
    mlir::Value dimZ = entry->getArgument(7);
    mlir::Value payloadPtr = entry->getArgument(8);
    mlir::Value attrPtr = entry->getArgument(9);

    mlir::Value c0 = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getIntegerAttr(i32Ty, 0));
    mlir::Value c1 = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getIntegerAttr(i32Ty, 1));
    mlir::Value initExecMask = mlir::LLVM::ConstantOp::create(b, loc, llvm::cast<mlir::TypedAttr>(mlir::DenseElementsAttr::get(vecI1Ty, b.getBoolAttr(false))).getType(), mlir::DenseElementsAttr::get(vecI1Ty, b.getBoolAttr(false)));
    for (uint32_t l = 0; l < simdWidth; ++l) {
        mlir::Value shift = mlir::LLVM::ConstantOp::create(b, loc, i32Ty, b.getIntegerAttr(i32Ty, l));
        mlir::Value shifted = mlir::LLVM::LShrOp::create(b, loc, activeMaskArg, shift);
        mlir::Value bit = mlir::LLVM::AndOp::create(b, loc, shifted, c1);
        mlir::Value cond = mlir::LLVM::ICmpOp::create(
            b, loc, mlir::LLVM::ICmpPredicate::ne, bit, c0);
        initExecMask = llvmInsert(b, loc, initExecMask, cond, static_cast<uint32_t>(static_cast<int64_t>(l)));
    }

    StageHooks hooks;
    hooks.entryBlock = entry;
    hooks.threadId[0] = mlir::LLVM::LoadOp::create(b, loc, vecI32Ty, rxPtr, 4u);
    hooks.threadId[1] = mlir::LLVM::LoadOp::create(b, loc, vecI32Ty, ryPtr, 4u);
    hooks.threadId[2] = mlir::LLVM::LoadOp::create(b, loc, vecI32Ty, rzPtr, 4u);
    hooks.dispatchRaysDim[0] = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vecI32Ty).getElementType(), dimX, simdWidth);
    hooks.dispatchRaysDim[1] = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vecI32Ty).getElementType(), dimY, simdWidth);
    hooks.dispatchRaysDim[2] = llvmSplat(b, loc, llvm::cast<mlir::VectorType>(vecI32Ty).getElementType(), dimZ, simdWidth);

    SimdPacketizer pkt(b, mod, shader, rootLayout, simdWidth, rootCtx, hooks);
    for (unsigned a = 0; a < srcFunc.getNumArguments() && a < 2; ++a) {
        mlir::Value ptrArg = (a == 0) ? payloadPtr : attrPtr;
        llvm::SmallVector<mlir::Value, 8> lanes(simdWidth, ptrArg);
        pkt.setMapped(srcFunc.getArgument(a), {SimdVal::Kind::Lanes, ptrTy, mlir::Value(), std::move(lanes), {}});
    }

    mlir::Value realMask = initExecMask;
    if (!pkt.packetizeBlock(srcFunc.getBody().front(), initExecMask, realMask, outError)) {
        return false;
    }

    mlir::func::ReturnOp::create(b, loc);
    srcFunc.erase();

    llvm::SmallVector<mlir::LLVM::LLVMFuncOp, 16> deadDxOps;
    for (auto fnOp : mod.getOps<mlir::LLVM::LLVMFuncOp>()) {
        if (fnOp.isDeclaration() && fnOp.getName().starts_with("dx.op.") && fnOp.use_empty()) {
            deadDxOps.push_back(fnOp);
        }
    }
    for (auto fnOp : deadDxOps) fnOp.erase();

    if (mlir::failed(mlir::verify(mod))) {
        outError = "MLIR verification failed after DXR SIMT-to-SIMD packetization";
        return false;
    }
    return true;
}

} // namespace nacelle
