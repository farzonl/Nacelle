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
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/Dialect/Vector/IR/VectorOps.h>
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
                 v.getDefiningOp<mlir::LLVM::PoisonOp>() ||
                 v.getDefiningOp<mlir::ub::PoisonOp>());
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

class SimdPacketizer {
public:
    SimdPacketizer(
        mlir::OpBuilder& builder,
        mlir::ModuleOp mod,
        const DxilLoadedShader& shader,
        const RootSignatureLayout& rootLayout,
        uint32_t simdWidth,
        mlir::Value rootCtxPtr,
        StageHooks& hooks)
        : b_(builder),
          ctx_(builder.getContext()),
          mod_(mod),
          shader_(shader),
          rootLayout_(rootLayout),
          W_(simdWidth),
          rootCtxPtr_(rootCtxPtr),
          hooks_(hooks) {}

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
        return mlir::arith::ConstantIntOp::create(b_, loc, ty, val);
    }
    mlir::Value cFlt(mlir::Location loc, mlir::Type ty, double val) {
        auto ft = llvm::cast<mlir::FloatType>(ty);
        return mlir::arith::ConstantFloatOp::create(
            b_, loc, ft, mlir::FloatAttr::get(ft, val).getValue());
    }
    mlir::Value splatInt(mlir::Location loc, mlir::Type elemTy, uint64_t val) {
        auto vt = vecTy(elemTy);
        return mlir::arith::ConstantOp::create(
            b_, loc, mlir::DenseElementsAttr::get(vt, b_.getIntegerAttr(elemTy, val)));
    }
    mlir::Value splatFlt(mlir::Location loc, mlir::Type elemTy, double val) {
        auto vt = vecTy(elemTy);
        return mlir::arith::ConstantOp::create(
            b_, loc, mlir::DenseElementsAttr::get(vt, b_.getFloatAttr(elemTy, val)));
    }
    mlir::Value laneIota(mlir::Location loc) {
        llvm::SmallVector<int32_t, 16> vals;
        for (uint32_t i = 0; i < W_; ++i) vals.push_back(static_cast<int32_t>(i));
        return mlir::arith::ConstantOp::create(
            b_, loc, mlir::DenseElementsAttr::get(vecTy(i32Ty()), llvm::ArrayRef<int32_t>(vals)));
    }

    mlir::Value orReduce(mlir::Location loc, mlir::Value vecI1) {
        return mlir::vector::ReductionOp::create(
            b_, loc, mlir::vector::CombiningKind::OR, vecI1);
    }

    mlir::Value allTrueMask(mlir::Location loc) { return splatInt(loc, i1Ty(), 1); }
    mlir::Value allFalseMask(mlir::Location loc) { return splatInt(loc, i1Ty(), 0); }

    mlir::Value expandMaskQuads(mlir::Location loc, mlir::Value mask) {
        // For quad q (4 lanes: 4q..4q+3), any active lane makes all 4 lanes execute in execMask.
        llvm::SmallVector<int64_t, 16> s1, s2;
        for (uint32_t l = 0; l < W_; ++l) {
            s1.push_back(static_cast<int64_t>(l ^ 1u));
            s2.push_back(static_cast<int64_t>(l ^ 2u));
        }
        mlir::Value pair = mlir::arith::OrIOp::create(
            b_, loc, mask, mlir::vector::ShuffleOp::create(b_, loc, mask, mask, s1));
        return mlir::arith::OrIOp::create(
            b_, loc, pair, mlir::vector::ShuffleOp::create(b_, loc, pair, pair, s2));
    }

    SimdVal makeZeroSimd(mlir::Location loc, mlir::Type origTy) {
        if (mlir::isa<mlir::IntegerType>(origTy)) {
            return {SimdVal::Kind::Vector, origTy, splatInt(loc, origTy, 0), {}, {}};
        }
        if (mlir::isa<mlir::FloatType>(origTy)) {
            return {SimdVal::Kind::Vector, origTy, splatFlt(loc, origTy, 0.0), {}, {}};
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
                return {SimdVal::Kind::Vector, ty, splatInt(loc, ty, ia.getValue().getZExtValue()), {}, {}};
            }
            if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(c.getValue())) {
                return {SimdVal::Kind::Vector, ty, splatFlt(loc, ty, fa.getValueAsDouble()), {}, {}};
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
                return {SimdVal::Kind::Vector, ty, splatInt(loc, ty, ia.getValue().getZExtValue()), {}, {}};
            }
            if (auto fa = llvm::dyn_cast<mlir::FloatAttr>(c.getValue())) {
                return {SimdVal::Kind::Vector, ty, splatFlt(loc, ty, fa.getValueAsDouble()), {}, {}};
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
                return mlir::vector::BroadcastOp::create(b_, loc, vecTy(sv.lanes[0].getType()), sv.lanes[0]);
            }
            mlir::Type elemTy = sv.lanes[0].getType();
            mlir::Value v = mlir::LLVM::isCompatibleType(elemTy)
                                ? mlir::Value(mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy)))
                                : mlir::Value(mlir::ub::PoisonOp::create(b_, loc, vecTy(elemTy)));
            for (uint32_t l = 0; l < W_; ++l) {
                v = mlir::vector::InsertOp::create(b_, loc, sv.lanes[l], v, static_cast<int64_t>(l));
            }
            return v;
        }
        return nullptr;
    }

    mlir::Value laneValue(mlir::Location loc, const SimdVal& sv, uint32_t l) {
        if (sv.kind == SimdVal::Kind::Vector) {
            return mlir::vector::ExtractOp::create(b_, loc, sv.vec, static_cast<int64_t>(l));
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
            mlir::Value s = mlir::arith::SelectOp::create(b_, loc, condVec, t.vec, f.vec);
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
                mlir::Value cL = mlir::vector::ExtractOp::create(b_, loc, condVec, static_cast<int64_t>(l));
                mlir::Value tL = laneValue(loc, t, l);
                mlir::Value fL = laneValue(loc, f, l);
                out.lanes.push_back(mlir::arith::SelectOp::create(b_, loc, cL, tL, fL));
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
            outVecTypes.push_back(mlir::LLVM::LLVMArrayType::get(ptrTy(), W_));
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
                vec64 = mlir::vector::InsertOp::create(b_, loc, i64L, vec64, static_cast<int64_t>(l));
            }
            out.push_back(vec64);
        } else if (sv.kind == SimdVal::Kind::Lanes || sv.kind == SimdVal::Kind::Handle) {
            mlir::Type arrTy = mlir::LLVM::LLVMArrayType::get(ptrTy(), W_);
            mlir::Value arr = mlir::LLVM::UndefOp::create(b_, loc, arrTy);
            for (uint32_t l = 0; l < W_; ++l) {
                arr = mlir::LLVM::InsertValueOp::create(b_, loc, arr, laneValue(loc, sv, l),
                                                        llvm::ArrayRef<int64_t>{static_cast<int64_t>(l)});
            }
            out.push_back(arr);
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
            mlir::Value arr = vals[cursor++];
            for (uint32_t l = 0; l < W_; ++l) {
                h.lanes.push_back(mlir::LLVM::ExtractValueOp::create(
                    b_, loc, arr, llvm::ArrayRef<int64_t>{static_cast<int64_t>(l)}));
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
                mlir::Value i64L = mlir::vector::ExtractOp::create(b_, loc, vec64, static_cast<int64_t>(l));
                sv.lanes.push_back(mlir::arith::IndexCastOp::create(b_, loc, origTy, i64L));
            }
            return sv;
        }
        if (mlir::isa<mlir::LLVM::LLVMPointerType>(origTy)) {
            SimdVal sv;
            sv.kind = SimdVal::Kind::Lanes;
            sv.origType = origTy;
            mlir::Value arr = vals[cursor++];
            for (uint32_t l = 0; l < W_; ++l) {
                sv.lanes.push_back(mlir::LLVM::ExtractValueOp::create(
                    b_, loc, arr, llvm::ArrayRef<int64_t>{static_cast<int64_t>(l)}));
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

    void storeScalar(mlir::Location loc, mlir::Value val, mlir::Value ptr) {
        mlir::LLVM::StoreOp::create(b_, loc, val, ptr);
    }

    mlir::Value createEntryAlloca(mlir::Location loc, mlir::Type elemTy, uint64_t count, unsigned align = 64) {
        mlir::Block* curBlk = b_.getInsertionBlock();
        mlir::Operation* parentOp = curBlk ? curBlk->getParentOp() : nullptr;
        while (parentOp && !llvm::isa<mlir::func::FuncOp>(parentOp)) {
            parentOp = parentOp->getParentOp();
        }
        if (auto fnOp = llvm::dyn_cast_or_null<mlir::func::FuncOp>(parentOp)) {
            mlir::Block& entryBlk = fnOp.getBody().front();
            mlir::OpBuilder eb(&entryBlk, entryBlk.begin());
            mlir::Value sz = mlir::arith::ConstantIntOp::create(eb, loc, eb.getI32Type(), count);
            return mlir::LLVM::AllocaOp::create(eb, loc, ptrTy(), elemTy, sz, align);
        }
        return mlir::LLVM::AllocaOp::create(b_, loc, ptrTy(), elemTy, cInt(loc, i32Ty(), count), align);
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
                      mlir::ub::PoisonOp, mlir::LLVM::ZeroOp>(&op)) {
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
                mlir::Value eq = mlir::arith::CmpIOp::create(
                    b_, loc, mlir::arith::CmpIPredicate::eq, idxVec, splatInt(loc, idxTy, i));
                res = mlir::arith::SelectOp::create(b_, loc, eq, asVec(loc, vec.fields[i]), res);
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
                mlir::Value eq = mlir::arith::CmpIOp::create(
                    b_, loc, mlir::arith::CmpIPredicate::eq, idxVec, splatInt(loc, idxTy, i));
                mlir::Value cur = asVec(loc, vec.fields[i]);
                vec.fields[i].vec = mlir::arith::SelectOp::create(b_, loc, eq, valVec, cur);
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
            mlir::Type elemTy = allocaOp.getElemType();
            uint64_t count = getConstIntValue(allocaOp.getArraySize()).value_or(1);
            mlir::Type slotTy = (count == 1) ? elemTy : mlir::Type(mlir::LLVM::LLVMArrayType::get(elemTy, count));
            uint32_t numWaves = std::max<uint32_t>(1u, hooks_.numWaves);
            mlir::Value baseAlloca = createEntryAlloca(loc, slotTy, static_cast<uint64_t>(numWaves) * W_, 64);
            if (hooks_.waveIdx && numWaves > 1u) {
                mlir::Value w32 = mlir::arith::IndexCastOp::create(b_, loc, i32Ty(), hooks_.waveIdx);
                mlir::Value waveBase = mlir::arith::MulIOp::create(b_, loc, w32, cInt(loc, i32Ty(), W_));
                for (uint32_t l = 0; l < W_; ++l) {
                    mlir::Value slotIdx = mlir::arith::AddIOp::create(b_, loc, waveBase, cInt(loc, i32Ty(), l));
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
                    if (iv.kind == SimdVal::Kind::Vector && !iv.vec.getDefiningOp<mlir::arith::ConstantOp>()) {
                        // Safe conservatively to expand if any dynamic index might vary
                        uniform = false;
                        break;
                    }
                }
            }

            SimdVal out;
            out.kind = SimdVal::Kind::Lanes;
            out.origType = gep.getType();
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
            if (addr.isUniform() && mlir::isa<mlir::IntegerType, mlir::FloatType>(elemTy)) {
                mlir::Value s = loadScalar(loc, elemTy, addr.lanes[0]);
                mlir::Value bvec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(elemTy), s);
                setMapped(loadOp.getResult(), {SimdVal::Kind::Vector, elemTy, bvec, {}, {}});
                return true;
            }
            if (mlir::isa<mlir::IntegerType, mlir::FloatType>(elemTy)) {
                mlir::Value v = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy));
                for (uint32_t l = 0; l < W_; ++l) {
                    mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
                    mlir::Value pL = laneValue(loc, addr, l);
                    auto ifL = mlir::scf::IfOp::create(
                        b_, loc, mL,
                        [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                            mlir::scf::YieldOp::create(
                                nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, elemTy, pL)});
                        },
                        [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                            mlir::scf::YieldOp::create(
                                nb, nloc, mlir::ValueRange{mlir::LLVM::UndefOp::create(nb, nloc, elemTy)});
                        });
                    v = mlir::vector::InsertOp::create(b_, loc, ifL.getResult(0), v, static_cast<int64_t>(l));
                }
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
                for (int64_t i = 0; i < n; ++i) {
                    if (addr.isUniform()) {
                        mlir::Value ep = mlir::LLVM::GEPOp::create(
                            b_, loc, ptrTy, scalarTy, addr.lanes[0],
                            mlir::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(i)});
                        mlir::Value s = loadScalar(loc, scalarTy, ep);
                        mlir::Value bvec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(scalarTy), s);
                        res.fields[static_cast<size_t>(i)] = {SimdVal::Kind::Vector, scalarTy, bvec, {}, {}};
                    } else {
                        mlir::Value v = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(scalarTy));
                        for (uint32_t l = 0; l < W_; ++l) {
                            mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
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
                            v = mlir::vector::InsertOp::create(b_, loc, ifL.getResult(0), v, static_cast<int64_t>(l));
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
            if (auto vt = mlir::dyn_cast<mlir::VectorType>(storeOp.getValue().getType())) {
                if (val.kind == SimdVal::Kind::Struct && val.fields.size() == static_cast<size_t>(vt.getNumElements())) {
                    mlir::Type scalarTy = vt.getElementType();
                    mlir::Type ptrTy = storeOp.getAddr().getType();
                    for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                        for (uint32_t l = 0; l < W_; ++l) {
                            mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
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
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
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
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(ty));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
                mlir::Value pL = laneValue(loc, addr, l);
                mlir::Value vL = laneValue(loc, val, l);
                auto ifL = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value r;
                        if (mlir::isa<mlir::FloatType>(ty) && rmw.getBinOp() == mlir::LLVM::AtomicBinOp::xchg) {
                            mlir::Type intTy = nb.getIntegerType(ty.getIntOrFloatBitWidth());
                            mlir::Value vInt = mlir::arith::BitcastOp::create(nb, nloc, intTy, vL);
                            mlir::Value rInt = mlir::LLVM::AtomicRMWOp::create(
                                nb, nloc, mlir::LLVM::AtomicBinOp::xchg, pL, vInt, rmw.getOrdering());
                            r = mlir::arith::BitcastOp::create(nb, nloc, ty, rInt);
                        } else {
                            r = mlir::LLVM::AtomicRMWOp::create(nb, nloc, rmw.getBinOp(), pL, vL, rmw.getOrdering());
                        }
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{r});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc, mlir::ValueRange{mlir::LLVM::UndefOp::create(nb, nloc, ty)});
                    });
                outVec = mlir::vector::InsertOp::create(b_, loc, ifL.getResult(0), outVec, static_cast<int64_t>(l));
            }
            setMapped(rmw.getRes(), {SimdVal::Kind::Vector, ty, outVec, {}, {}});
            return true;
        }
        if (auto cx = llvm::dyn_cast<mlir::LLVM::AtomicCmpXchgOp>(&op)) {
            SimdVal addr = getMapped(loc, cx.getPtr());
            SimdVal cmp = getMapped(loc, cx.getCmp());
            SimdVal val = getMapped(loc, cx.getVal());
            mlir::Type elemTy = cmp.origType;
            mlir::Value outValVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy));
            mlir::Value outOkVec = splatInt(loc, i1Ty(), 0);
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
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
                            cOp = mlir::arith::BitcastOp::create(nb, nloc, intTy, cL);
                            vOp = mlir::arith::BitcastOp::create(nb, nloc, intTy, vL);
                        }
                        auto r = mlir::LLVM::AtomicCmpXchgOp::create(
                            nb, nloc, pL, cOp, vOp, cx.getSuccessOrdering(), cx.getFailureOrdering());
                        mlir::Value oldV = mlir::LLVM::ExtractValueOp::create(nb, nloc, r, llvm::ArrayRef<int64_t>{0});
                        if (mlir::isa<mlir::FloatType>(elemTy)) {
                            oldV = mlir::arith::BitcastOp::create(nb, nloc, elemTy, oldV);
                        }
                        mlir::Value ok = mlir::LLVM::ExtractValueOp::create(nb, nloc, r, llvm::ArrayRef<int64_t>{1});
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{oldV, ok});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(
                            nb, nloc,
                            mlir::ValueRange{mlir::LLVM::UndefOp::create(nb, nloc, elemTy),
                                             mlir::arith::ConstantIntOp::create(nb, nloc, i1Ty(), 0)});
                    });
                outValVec = mlir::vector::InsertOp::create(b_, loc, ifL.getResult(0), outValVec, static_cast<int64_t>(l));
                outOkVec = mlir::vector::InsertOp::create(b_, loc, ifL.getResult(1), outOkVec, static_cast<int64_t>(l));
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
                outVec = mlir::vector::InsertOp::create(b_, loc, iv, outVec, static_cast<int64_t>(l));
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
            if (auto vt = llvm::dyn_cast<mlir::VectorType>(resTy)) {
                mlir::Type resElem = vt.getElementType();
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = resTy;
                for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                    mlir::Value ai = (aSv.kind == SimdVal::Kind::Struct && static_cast<size_t>(i) < aSv.fields.size())
                                         ? asVec(loc, aSv.fields[static_cast<size_t>(i)])
                                         : asVec(loc, aSv);
                    mlir::Value r = tagFn(vecTy(resElem), ai);
                    out.fields.push_back({SimdVal::Kind::Vector, resElem, r, {}, {}});
                }
                setMapped(op.getResult(0), out);
                return true;
            }
            mlir::Value a = asVec(loc, aSv);
            mlir::Value r = tagFn(vecTy(resTy), a);
            setMapped(op.getResult(0), {SimdVal::Kind::Vector, resTy, r, {}, {}});
            return true;
        };
        auto vecBinary = [&](auto tagFn) -> bool {
            mlir::Type resTy = op.getResult(0).getType();
            SimdVal aSv = getMapped(loc, op.getOperand(0));
            SimdVal cSv = getMapped(loc, op.getOperand(1));
            if (auto vt = llvm::dyn_cast<mlir::VectorType>(resTy)) {
                mlir::Type resElem = vt.getElementType();
                SimdVal out;
                out.kind = SimdVal::Kind::Struct;
                out.origType = resTy;
                for (int64_t i = 0; i < vt.getNumElements(); ++i) {
                    mlir::Value ai = (aSv.kind == SimdVal::Kind::Struct && static_cast<size_t>(i) < aSv.fields.size())
                                         ? asVec(loc, aSv.fields[static_cast<size_t>(i)])
                                         : asVec(loc, aSv);
                    mlir::Value ci = (cSv.kind == SimdVal::Kind::Struct && static_cast<size_t>(i) < cSv.fields.size())
                                         ? asVec(loc, cSv.fields[static_cast<size_t>(i)])
                                         : asVec(loc, cSv);
                    mlir::Value r = tagFn(vecTy(resElem), ai, ci);
                    out.fields.push_back({SimdVal::Kind::Vector, resElem, r, {}, {}});
                }
                setMapped(op.getResult(0), out);
                return true;
            }
            mlir::Value a = asVec(loc, aSv);
            mlir::Value c = asVec(loc, cSv);
            mlir::Value r = tagFn(vecTy(resTy), a, c);
            setMapped(op.getResult(0), {SimdVal::Kind::Vector, resTy, r, {}, {}});
            return true;
        };

        if (llvm::isa<mlir::LLVM::AddOp, mlir::arith::AddIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::AddIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SubOp, mlir::arith::SubIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::SubIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::MulOp, mlir::arith::MulIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MulIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SDivOp, mlir::arith::DivSIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SDivOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::UDivOp, mlir::arith::DivUIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::UDivOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SRemOp, mlir::arith::RemSIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::SRemOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::URemOp, mlir::arith::RemUIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::LLVM::URemOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::ShlOp, mlir::arith::ShLIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::ShLIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::LShrOp, mlir::arith::ShRUIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::ShRUIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::AShrOp, mlir::arith::ShRSIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::ShRSIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::AndOp, mlir::arith::AndIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::AndIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::OrOp, mlir::arith::OrIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::OrIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::XOrOp, mlir::arith::XOrIOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::XOrIOp::create(b_, loc, rt, a, c); });

        if (llvm::isa<mlir::LLVM::FAddOp, mlir::arith::AddFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::AddFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FSubOp, mlir::arith::SubFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::SubFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FMulOp, mlir::arith::MulFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MulFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FDivOp, mlir::arith::DivFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::DivFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FRemOp, mlir::arith::RemFOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::RemFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::FNegOp, mlir::arith::NegFOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::NegFOp::create(b_, loc, rt, a); });

        if (llvm::isa<mlir::LLVM::ZExtOp, mlir::arith::ExtUIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::ExtUIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SExtOp, mlir::arith::ExtSIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::ExtSIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::TruncOp, mlir::arith::TruncIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::TruncIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPTruncOp, mlir::arith::TruncFOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::TruncFOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPExtOp, mlir::arith::ExtFOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::ExtFOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPToSIOp, mlir::arith::FPToSIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::FPToSIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FPToUIOp, mlir::arith::FPToUIOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::FPToUIOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SIToFPOp, mlir::arith::SIToFPOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::SIToFPOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::UIToFPOp, mlir::arith::UIToFPOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::UIToFPOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::BitcastOp, mlir::arith::BitcastOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::arith::BitcastOp::create(b_, loc, rt, a); });
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
                                        : mlir::Value(mlir::vector::ExtractOp::create(b_, loc, v.vec, static_cast<int64_t>(l)));
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
                    vec = mlir::vector::InsertOp::create(b_, loc, casted, vec, static_cast<int64_t>(l));
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
                out = isUnsigned ? mlir::Value(mlir::arith::ExtUIOp::create(b_, loc, vecResTy, v.vec))
                                 : mlir::Value(mlir::arith::ExtSIOp::create(b_, loc, vecResTy, v.vec));
            } else if (dstBits < srcBits) {
                out = mlir::arith::TruncIOp::create(b_, loc, vecResTy, v.vec);
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
                    outVec = mlir::vector::InsertOp::create(b_, loc, r, outVec, static_cast<int64_t>(l));
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
                                         : mlir::Value(mlir::vector::ExtractOp::create(b_, loc, va.vec, static_cast<int64_t>(l)));
                    mlir::Value lc = (vc.kind == SimdVal::Kind::Lanes)
                                         ? (vc.lanes.size() == 1 ? vc.lanes[0] : vc.lanes[l])
                                         : mlir::Value(mlir::vector::ExtractOp::create(b_, loc, vc.vec, static_cast<int64_t>(l)));
                    mlir::Value r = mlir::arith::CmpIOp::create(b_, loc, icmp.getPredicate(), la, lc);
                    outVec = mlir::vector::InsertOp::create(b_, loc, r, outVec, static_cast<int64_t>(l));
                }
                setMapped(icmp.getResult(), {SimdVal::Kind::Vector, i1Ty(), outVec, {}, {}});
                return true;
            }
            mlir::Value a = asVec(loc, getMapped(loc, icmp.getLhs()));
            mlir::Value c = asVec(loc, getMapped(loc, icmp.getRhs()));
            if (!a || !c) return false;
            mlir::Value r = mlir::arith::CmpIOp::create(b_, loc, icmp.getPredicate(), a, c);
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
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::AbsFOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SqrtOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::SqrtOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::SinOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::SinOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::CosOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::CosOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::Exp2Op>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::Exp2Op::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::Log2Op>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::Log2Op::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FFloorOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::FloorOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FCeilOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::CeilOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::FTruncOp>(&op))
            return vecUnary([&](mlir::Type rt, mlir::Value a) { return mlir::math::TruncOp::create(b_, loc, rt, a); });
        if (llvm::isa<mlir::LLVM::MinNumOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MinimumFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::MaxNumOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MaximumFOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::UMinOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MinUIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::UMaxOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MaxUIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SMinOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MinSIOp::create(b_, loc, rt, a, c); });
        if (llvm::isa<mlir::LLVM::SMaxOp>(&op))
            return vecBinary([&](mlir::Type rt, mlir::Value a, mlir::Value c) { return mlir::arith::MaxSIOp::create(b_, loc, rt, a, c); });
        auto castIntVec = [&](mlir::Value v, mlir::Type dstElem) -> mlir::Value {
            mlir::Type srcElem = llvm::cast<mlir::VectorType>(v.getType()).getElementType();
            if (srcElem == dstElem) return v;
            return (srcElem.getIntOrFloatBitWidth() > dstElem.getIntOrFloatBitWidth())
                       ? mlir::Value(mlir::arith::TruncIOp::create(b_, loc, vecTy(dstElem), v))
                       : mlir::Value(mlir::arith::ExtUIOp::create(b_, loc, vecTy(dstElem), v));
        };
        if (llvm::isa<mlir::LLVM::CtPopOp>(&op)) {
            mlir::Type resElem = op.getResult(0).getType();
            mlir::Value a = asVec(loc, getMapped(loc, op.getOperand(0)));
            mlir::Value r = castIntVec(mlir::math::CtPopOp::create(b_, loc, a.getType(), a), resElem);
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
            mlir::Value r = mlir::math::FmaOp::create(b_, loc, vecTy(resElem), a, x, c);
            setMapped(op.getResult(0), {SimdVal::Kind::Vector, resElem, r, {}, {}});
            return true;
        }
        if (auto ctlz = llvm::dyn_cast<mlir::LLVM::CountLeadingZerosOp>(&op)) {
            mlir::Type resElem = ctlz.getType();
            mlir::Value a = asVec(loc, getMapped(loc, ctlz.getIn()));
            mlir::Value r = castIntVec(mlir::math::CountLeadingZerosOp::create(b_, loc, a.getType(), a), resElem);
            setMapped(ctlz.getRes(), {SimdVal::Kind::Vector, resElem, r, {}, {}});
            return true;
        }
        if (auto cttz = llvm::dyn_cast<mlir::LLVM::CountTrailingZerosOp>(&op)) {
            mlir::Type resElem = cttz.getType();
            mlir::Value a = asVec(loc, getMapped(loc, cttz.getIn()));
            mlir::Value r = castIntVec(mlir::math::CountTrailingZerosOp::create(b_, loc, a.getType(), a), resElem);
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
                                     : mlir::Value(mlir::vector::ExtractOp::create(b_, loc, argVal.vec, static_cast<int64_t>(l)));
            if (!idxVal) {
                outError = "Null idxVal in scf.index_switch";
                return false;
            }
            mlir::Value i64v = idxVal.getType().isIndex()
                                   ? mlir::Value(mlir::arith::IndexCastOp::create(b_, loc, i64Ty(), idxVal))
                                   : idxVal;
            argVec = mlir::vector::InsertOp::create(b_, loc, i64v, argVec, static_cast<int64_t>(l)).getResult();
        }

        llvm::SmallVector<SimdVal, 8> accVals;
        for (mlir::Value r : swOp.getResults()) {
            accVals.push_back(makeZeroSimd(loc, r.getType()));
        }

        auto runCaseBlock = [&](mlir::Block& srcBB, mlir::Value caseMask) -> bool {
            mlir::Value curReal = mlir::arith::AndIOp::create(b_, loc, realMask, caseMask);
            mlir::Value curExec = mlir::arith::AndIOp::create(b_, loc, execMask, caseMask);
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
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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
            mlir::Value eqMask = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::eq, argVec, cVec);
            mlir::Value caseMask = mlir::arith::AndIOp::create(b_, loc, remainingMask, eqMask);
            if (!runCaseBlock(swOp.getCaseRegions()[ci].front(), caseMask)) return false;
            mlir::Value notCase = mlir::arith::XOrIOp::create(b_, loc, caseMask, allTrueMask(loc));
            remainingMask = mlir::arith::AndIOp::create(b_, loc, remainingMask, notCase);
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
        mlir::Value realThen = mlir::arith::AndIOp::create(b_, loc, realMask, condVec);
        mlir::Value notCond = mlir::arith::XOrIOp::create(b_, loc, condVec, allTrueMask(loc));
        mlir::Value realElse = mlir::arith::AndIOp::create(b_, loc, realMask, notCond);

        mlir::Value execThen = mlir::arith::AndIOp::create(b_, loc, execMask, condVec);
        mlir::Value execElse = mlir::arith::AndIOp::create(b_, loc, execMask, notCond);
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
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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
                        SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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
            realMask = mlir::arith::OrIOp::create(b_, loc, thenPacked.back(), elsePacked.back());
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
                SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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

                mlir::Value curReal = mlir::arith::AndIOp::create(nb, nloc, loopExec, aliveMask);
                (void)sub.packetizeBlock(srcBefore, loopExec, curReal, outError);

                mlir::Value notLoopExec = mlir::arith::XOrIOp::create(nb, nloc, loopExec, sub.allTrueMask(nloc));
                mlir::Value survived = mlir::arith::OrIOp::create(nb, nloc, notLoopExec, curReal);
                mlir::Value aliveAfterBefore = mlir::arith::AndIOp::create(nb, nloc, aliveMask, survived);

                auto condOp = llvm::cast<mlir::scf::ConditionOp>(srcBefore.getTerminator());
                mlir::Value condVec = sub.asVec(nloc, sub.getMapped(nloc, condOp.getCondition()));
                mlir::Value nextExec = mlir::arith::AndIOp::create(nb, nloc, loopExec, condVec);
                nextExec = mlir::arith::AndIOp::create(nb, nloc, nextExec, aliveAfterBefore);
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
                SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
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

                mlir::Value bodyReal = mlir::arith::AndIOp::create(nb, nloc, bodyExec, bodyAlive);
                (void)sub.packetizeBlock(srcAfter, bodyExec, bodyReal, outError);

                mlir::Value notBodyExec = mlir::arith::XOrIOp::create(nb, nloc, bodyExec, sub.allTrueMask(nloc));
                mlir::Value survived = mlir::arith::OrIOp::create(nb, nloc, notBodyExec, bodyReal);
                mlir::Value aliveAfterBody = mlir::arith::AndIOp::create(nb, nloc, bodyAlive, survived);

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
        mlir::Value lb = mlir::vector::ExtractOp::create(b_, loc, asVec(loc, getMapped(loc, forOp.getLowerBound())), 0);
        mlir::Value ub = mlir::vector::ExtractOp::create(b_, loc, asVec(loc, getMapped(loc, forOp.getUpperBound())), 0);
        mlir::Value step = mlir::vector::ExtractOp::create(b_, loc, asVec(loc, getMapped(loc, forOp.getStep())), 0);
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
                SimdPacketizer sub(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                sub.valueMap_ = valueMap_;
                mlir::Value ivVec = mlir::vector::BroadcastOp::create(nb, nloc, sub.vecTy(iv.getType()), iv);
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
                                 uint32_t resourceKind, uint32_t stride, mlir::Value indexVal, mlir::Type origTy) {
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
            mlir::Value tblBase = loadScalar(loc, ptrTy(), gepByteOffset(loc, rootCtxPtr_, slotOff));
            auto idxConst = getConstIntValue(indexVal);
            if (idxConst.has_value()) {
                int64_t regIdx = static_cast<int64_t>(*idxConst);
                int64_t tableIdx = static_cast<int64_t>(tgt.tableOffsetOrConstStart) +
                                   (regIdx - static_cast<int64_t>(lowerBound));
                h.lanes.push_back(gepByteOffset(loc, tblBase, tableIdx * 64));
                return h;
            }
            SimdVal idxSimd = getMapped(loc, indexVal);
            mlir::Value idxVec = asVec(loc, idxSimd);
            int64_t bias = static_cast<int64_t>(tgt.tableOffsetOrConstStart) - static_cast<int64_t>(lowerBound);
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
                mlir::Value idx64 = mlir::arith::ExtSIOp::create(b_, loc, i64Ty(), idxL);
                mlir::Value slot = mlir::arith::AddIOp::create(b_, loc, idx64, cInt(loc, i64Ty(), bias));
                mlir::Value byteOff = mlir::arith::MulIOp::create(b_, loc, slot, cInt(loc, i64Ty(), 64));
                h.lanes.push_back(gepDynByteOffset(loc, tblBase, byteOff));
            }
            return h;
        }

        // Unbound fallback -> null descriptor pointer (bounds checks return 0 safely)
        h.lanes.push_back(mlir::LLVM::ZeroOp::create(b_, loc, ptrTy()));
        return h;
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
                       mlir::Value p = mlir::LLVM::LoadOp::create(nb, nloc, ptrTy(), descPtr);
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
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(nb, nloc, i32Ty(), fallback)});
                   },
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::Value fp = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), descPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{byteOff});
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, i32Ty(), fp)});
                   })
            .getResult(0);
    }

    mlir::Value loadDescU64Field(mlir::Location loc, mlir::Value descPtr, int64_t byteOff) {
        mlir::Value nullPtr = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
        mlir::Value isNull = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, descPtr, nullPtr);
        return mlir::scf::IfOp::create(
                   b_, loc, isNull,
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(nb, nloc, i64Ty(), 0)});
                   },
                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                       mlir::Value fp = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), descPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{byteOff});
                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, i64Ty(), fp)});
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
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), v, {}, {}});
            return true;
        }
        if (op == 96) {
            mlir::Value v = hooks_.flatGroupId ? hooks_.flatGroupId : splatInt(loc, i32Ty(), 0);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), v, {}, {}});
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
            setMapped(call.getResult(), resolveStaticBinding(loc, resClass, space, lower, kind, stride, arg(3), resTy()));
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
            setMapped(call.getResult(), resolveStaticBinding(loc, resClass, space, lower, kind, stride, arg(2), resTy()));
            return true;
        }
        if (op == 218) {
            // createHandleFromHeap(op, index, samplerHeap, nonUniform)
            bool isSamp = getConstIntValue(arg(2)).value_or(0) != 0;
            int64_t baseOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_base))
                                     : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_base));
            int64_t cntOff = isSamp ? static_cast<int64_t>(offsetof(DriverRootContext, sampler_heap_count))
                                    : static_cast<int64_t>(offsetof(DriverRootContext, resource_heap_count));
            mlir::Value heapBase = loadScalar(loc, ptrTy(), gepByteOffset(loc, rootCtxPtr_, baseOff));
            mlir::Value heapCnt = loadScalar(loc, i32Ty(), gepByteOffset(loc, rootCtxPtr_, cntOff));

            SimdVal h;
            h.kind = SimdVal::Kind::Handle;
            h.origType = resTy();
            h.resClass = isSamp ? 3u : 0u;
            if (auto idxC = getConstIntValue(arg(1))) {
                mlir::Value ok = mlir::arith::CmpIOp::create(
                    b_, loc, mlir::arith::CmpIPredicate::ult, cInt(loc, i32Ty(), *idxC), heapCnt);
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
                mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
                mlir::Value ok = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ult, idxL, heapCnt);
                mlir::Value p = mlir::scf::IfOp::create(
                                    b_, loc, ok,
                                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                        mlir::Value idx64 = mlir::arith::ExtUIOp::create(nb, nloc, i64Ty(), idxL);
                                        mlir::Value bOff = mlir::arith::MulIOp::create(
                                            nb, nloc, idx64, mlir::arith::ConstantIntOp::create(nb, nloc, i64Ty(), 64));
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
            mlir::Value acc = mlir::arith::MulFOp::create(
                b_, loc, vT, asVec(loc, aSv.fields[0]), asVec(loc, bSv.fields[0]));
            for (size_t i = 1; i < n; ++i) {
                mlir::Value prod = mlir::arith::MulFOp::create(
                    b_, loc, vT, asVec(loc, aSv.fields[i]), asVec(loc, bSv.fields[i]));
                acc = mlir::arith::AddFOp::create(b_, loc, vT, acc, prod);
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
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
                mlir::Value descP = laneValue(loc, h, l);
                mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
                mlir::Value isValid = mlir::arith::AndIOp::create(
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
                                mlir::Value incVal = mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), inc);
                                mlir::Value old = mlir::LLVM::AtomicRMWOp::create(
                                    ib, iloc, mlir::LLVM::AtomicBinOp::add, cptr, incVal,
                                    mlir::LLVM::AtomicOrdering::acq_rel);
                                mlir::Value retVal = (inc < 0)
                                    ? mlir::arith::AddIOp::create(ib, iloc, old, incVal).getResult()
                                    : old;
                                mlir::scf::YieldOp::create(ib, iloc, mlir::ValueRange{retVal});
                            },
                            [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                                mlir::scf::YieldOp::create(ib, iloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), 0)});
                            });
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{cr.getResult(0)});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(nb, nloc, i32Ty(), 0)});
                    });
                outVec = mlir::vector::InsertOp::create(b_, loc, r.getResult(0), outVec, static_cast<int64_t>(l));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), outVec, {}, {}});
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
                mlir::Value killed = mlir::arith::AndIOp::create(b_, loc, execMask, condVec);
                mlir::Value keep = mlir::arith::XOrIOp::create(b_, loc, killed, allTrueMask(loc));
                realMask = mlir::arith::AndIOp::create(b_, loc, realMask, keep);
            }
            return true;
        }

        // 8. Wave & Quad Intrinsics (110..121, 123, 135, 136, 166)
        if ((op >= 110 && op <= 121) || op == 123 || op == 135 || op == 136 || op == 166) {
            return packetizeWaveOp(call, op, realMask, outError);
        }

        // 9. Math, Bitwise & Quad Derivatives (6..56, 83..86)
        return packetizeMathOp(call, op, outError);
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
        mlir::Type elemTy = st.getBody()[0];
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);

        auto idxConst = getConstIntValue(idxArg);
        if (h.isUniform() && idxConst.has_value()) {
            uint64_t byteOff = isLegacy ? (*idxConst * 16) : *idxConst;
            mlir::Value ptr = h.lanes[0];
            mlir::Value dataBase = h.isRootConstants ? ptr : loadDescBasePtr(loc, ptr);
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value isNull = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::eq, dataBase, nullP);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = st;
            for (unsigned f = 0; f < st.getBody().size(); ++f) {
                mlir::Value val = mlir::scf::IfOp::create(
                                      b_, loc, isNull,
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                                          mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(nloc, elemTy, 0.0) : tmp.cInt(nloc, elemTy, 0);
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z});
                                      },
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          mlir::Value ep = mlir::LLVM::GEPOp::create(
                                              nb, nloc, ptrTy(), i8Ty(), dataBase,
                                              llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int64_t>(byteOff + f * elemBytes)});
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, elemTy, ep)});
                                      })
                                      .getResult(0);
                mlir::Value bvec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(elemTy), val);
                outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, bvec, {}, {}});
            }
            setMapped(call.getResult(), outSt);
            return true;
        }

        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        const unsigned numFields = static_cast<unsigned>(st.getBody().size());
        llvm::SmallVector<mlir::Value, 8> fieldVecs(numFields, mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy)));
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value ptr = laneValue(loc, h, l);
            mlir::Value dataBase = h.isRootConstants ? ptr : loadDescBasePtr(loc, ptr);
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value ok = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, dataBase, nullP);
            mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
            mlir::Value idx64 = mlir::arith::ExtUIOp::create(b_, loc, i64Ty(), idxL);
            mlir::Value off64 = isLegacy ? mlir::arith::MulIOp::create(b_, loc, idx64, cInt(loc, i64Ty(), 16)) : idx64;
            for (unsigned f = 0; f < st.getBody().size(); ++f) {
                mlir::Value val = mlir::scf::IfOp::create(
                                      b_, loc, ok,
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          mlir::Value tot = mlir::arith::AddIOp::create(
                                              nb, nloc, off64, mlir::arith::ConstantIntOp::create(nb, nloc, i64Ty(), f * elemBytes));
                                          mlir::Value ep = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), dataBase, mlir::ValueRange{tot});
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, elemTy, ep)});
                                      },
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                                          mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(nloc, elemTy, 0.0) : tmp.cInt(nloc, elemTy, 0);
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z});
                                      })
                                      .getResult(0);
                fieldVecs[f] = mlir::vector::InsertOp::create(b_, loc, val, fieldVecs[f], static_cast<int64_t>(l));
            }
        }
        for (unsigned f = 0; f < numFields; ++f) {
            outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, fieldVecs[f], {}, {}});
        }
        setMapped(call.getResult(), outSt);
        return true;
    }

    // Computes per-lane buffer element base byte offset
    mlir::Value computeBufferByteOffset(
        mlir::Location loc, mlir::OpBuilder& nb, const SimdVal& h, mlir::Value descPtr,
        mlir::Value idxL, mlir::Value elemOffL, bool isRawOp, bool hasElemOff) {
        auto ci = [&](uint64_t v) { return mlir::arith::ConstantIntOp::create(nb, loc, i64Ty(), v); };
        auto loadStride64 = [&]() -> mlir::Value {
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(nb, loc, ptrTy());
            mlir::Value isNull = mlir::LLVM::ICmpOp::create(nb, loc, mlir::LLVM::ICmpPredicate::eq, descPtr, nullP);
            mlir::Value s32 = mlir::scf::IfOp::create(
                                  nb, loc, isNull,
                                  [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                                      mlir::scf::YieldOp::create(
                                          ib, iloc, mlir::ValueRange{mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), 0)});
                                  },
                                  [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                                      mlir::Value sp = mlir::LLVM::GEPOp::create(
                                          ib, iloc, ptrTy(), i8Ty(), descPtr,
                                          llvm::ArrayRef<mlir::LLVM::GEPArg>{static_cast<int32_t>(offsetof(NacelleDescriptor, elem_stride))});
                                      mlir::scf::YieldOp::create(
                                          ib, iloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(ib, iloc, i32Ty(), sp)});
                                  })
                                  .getResult(0);
            return mlir::arith::ExtUIOp::create(nb, loc, i64Ty(), s32);
        };
        mlir::Value idx64 = mlir::arith::ExtUIOp::create(nb, loc, i64Ty(), idxL);
        if (isRawOp) {
            if (!hasElemOff) {
                return idx64; // ByteAddressBuffer: index IS byte offset
            }
            // StructuredBuffer: idx * stride + elemOff
            mlir::Value strideVal = (h.stride > 0) ? ci(h.stride) : loadStride64();
            mlir::Value e64 = mlir::arith::ExtUIOp::create(nb, loc, i64Ty(), elemOffL);
            return mlir::arith::AddIOp::create(nb, loc, mlir::arith::MulIOp::create(nb, loc, idx64, strideVal), e64);
        }
        // Legacy bufferLoad/bufferStore (68/69)
        if (hasElemOff) {
            mlir::Value strideVal = (h.stride > 0) ? ci(h.stride) : loadStride64();
            mlir::Value e64 = mlir::arith::ExtUIOp::create(nb, loc, i64Ty(), elemOffL);
            return mlir::arith::AddIOp::create(nb, loc, mlir::arith::MulIOp::create(nb, loc, idx64, strideVal), e64);
        }
        if (h.resourceKind == 11) {
            return idx64; // RawBuffer
        }
        // Typed buffer: idx * desc.elem_stride
        return mlir::arith::MulIOp::create(nb, loc, idx64, loadStride64());
    }

    bool tryGetUniformScalar(mlir::Location loc, mlir::Value origVal, mlir::Value& outScalar) {
        if (auto c = getConstIntValue(origVal)) {
            outScalar = cInt(loc, i32Ty(), static_cast<int64_t>(*c));
            return true;
        }
        SimdVal sv = getMapped(loc, origVal);
        if (sv.isUniform()) {
            outScalar = sv.lanes[0];
            return true;
        }
        if (sv.kind == SimdVal::Kind::Vector) {
            if (auto bcast = sv.vec.getDefiningOp<mlir::vector::BroadcastOp>()) {
                if (!mlir::isa<mlir::VectorType>(bcast.getSource().getType())) {
                    outScalar = bcast.getSource();
                    return true;
                }
            }
            if (auto cst = sv.vec.getDefiningOp<mlir::arith::ConstantOp>()) {
                if (auto dea = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue())) {
                    if (dea.isSplat()) {
                        outScalar = mlir::arith::ConstantOp::create(
                            b_, loc, mlir::cast<mlir::TypedAttr>(dea.getSplatValue<mlir::Attribute>()));
                        return true;
                    }
                }
            }
        }
        return false;
    }

    mlir::Value emitUniformBufferComponentLoad(
        mlir::Location loc,
        mlir::Type elemTy,
        uint32_t elemBytes,
        mlir::Value basePtr,
        mlir::Value byteSize,
        mlir::Value okPtr,
        mlir::Value byteOff,
        mlir::Value elemStride,
        unsigned c) {
        mlir::Value cOff = mlir::arith::AddIOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), c * elemBytes));
        mlir::Value cEnd = mlir::arith::AddIOp::create(b_, loc, cOff, cInt(loc, i64Ty(), elemBytes));
        mlir::Value inBounds = mlir::arith::AndIOp::create(
            b_, loc, okPtr,
            mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, cEnd, byteSize));
        if (elemStride) {
            mlir::Value elemEnd = cInt(loc, i32Ty(), (c + 1) * elemBytes);
            mlir::Value strideZero = mlir::arith::CmpIOp::create(
                b_, loc, mlir::arith::CmpIPredicate::eq, elemStride, cInt(loc, i32Ty(), 0));
            mlir::Value inElem = mlir::arith::CmpIOp::create(
                b_, loc, mlir::arith::CmpIPredicate::ule, elemEnd, elemStride);
            inBounds = mlir::arith::AndIOp::create(
                b_, loc, inBounds, mlir::arith::OrIOp::create(b_, loc, strideZero, inElem));
        }
        mlir::Value val = mlir::scf::IfOp::create(
                              b_, loc, inBounds,
                              [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                  mlir::Value ep = mlir::LLVM::GEPOp::create(
                                      nb, nloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                                  mlir::scf::YieldOp::create(
                                      nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, elemTy, ep)});
                              },
                              [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                  SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                                  mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(nloc, elemTy, 0.0)
                                                                                     : tmp.cInt(nloc, elemTy, 0);
                                  mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z});
                              })
                              .getResult(0);
        return mlir::vector::BroadcastOp::create(b_, loc, vecTy(elemTy), val);
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
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);

        mlir::Value uniformIdx;
        mlir::Value uniformElemOff;
        if (h.isUniform() && tryGetUniformScalar(loc, idxArg, uniformIdx) &&
            (!hasElemOff || tryGetUniformScalar(loc, elemOffArg, uniformElemOff))) {
            if (!hasElemOff) uniformElemOff = cInt(loc, i32Ty(), 0);
            mlir::Value descP = h.lanes[0];
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value elemStride = (!isRawOp && !hasElemOff)
                                         ? loadDescU32Field(loc, descP, offsetof(NacelleDescriptor, elem_stride), 0)
                                         : mlir::Value();
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP);
            mlir::Value byteOff = computeBufferByteOffset(loc, b_, h, descP, uniformIdx, uniformElemOff, isRawOp, hasElemOff);

            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = st;
            for (unsigned c = 0; c < 4; ++c) {
                if ((mask & (1u << c)) == 0) {
                    mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? cFlt(loc, elemTy, 0.0) : cInt(loc, elemTy, 0);
                    outSt.fields.push_back({SimdVal::Kind::Vector, elemTy,
                                            mlir::vector::BroadcastOp::create(b_, loc, vecTy(elemTy), z), {}, {}});
                    continue;
                }
                mlir::Value bvec = emitUniformBufferComponentLoad(
                    loc, elemTy, elemBytes, basePtr, byteSize, okPtr, byteOff, elemStride, c);
                outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, bvec, {}, {}});
            }
            outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
            setMapped(call.getResult(), outSt);
            return true;
        }

        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        mlir::Value elemOffVec = hasElemOff ? asVec(loc, getMapped(loc, elemOffArg)) : splatInt(loc, i32Ty(), 0);

        llvm::SmallVector<mlir::Value, 4> compVecs(4, mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy)));
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value descP = laneValue(loc, h, l);
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value elemStride = (!isRawOp && !hasElemOff)
                                         ? loadDescU32Field(loc, descP, offsetof(NacelleDescriptor, elem_stride), 0)
                                         : mlir::Value();
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP);
            mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
            mlir::Value eL = mlir::vector::ExtractOp::create(b_, loc, elemOffVec, static_cast<int64_t>(l));
            mlir::Value byteOff = computeBufferByteOffset(loc, b_, h, descP, idxL, eL, isRawOp, hasElemOff);

            for (unsigned c = 0; c < 4; ++c) {
                if ((mask & (1u << c)) == 0) {
                    mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? cFlt(loc, elemTy, 0.0) : cInt(loc, elemTy, 0);
                    compVecs[c] = mlir::vector::InsertOp::create(b_, loc, z, compVecs[c], static_cast<int64_t>(l));
                    continue;
                }
                mlir::Value cOff = mlir::arith::AddIOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), c * elemBytes));
                mlir::Value cEnd = mlir::arith::AddIOp::create(b_, loc, cOff, cInt(loc, i64Ty(), elemBytes));
                mlir::Value inBounds = mlir::arith::AndIOp::create(
                    b_, loc, okPtr,
                    mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, cEnd, byteSize));
                if (elemStride) {
                    mlir::Value elemEnd = cInt(loc, i32Ty(), (c + 1) * elemBytes);
                    mlir::Value strideZero = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::eq, elemStride, cInt(loc, i32Ty(), 0));
                    mlir::Value inElem = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, elemEnd, elemStride);
                    inBounds = mlir::arith::AndIOp::create(
                        b_, loc, inBounds, mlir::arith::OrIOp::create(b_, loc, strideZero, inElem));
                }
                mlir::Value val = mlir::scf::IfOp::create(
                                      b_, loc, inBounds,
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          mlir::Value ep = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, elemTy, ep)});
                                      },
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                                          mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(nloc, elemTy, 0.0) : tmp.cInt(nloc, elemTy, 0);
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z});
                                      })
                                      .getResult(0);
                compVecs[c] = mlir::vector::InsertOp::create(b_, loc, val, compVecs[c], static_cast<int64_t>(l));
            }
        }

        SimdVal outSt;
        outSt.kind = SimdVal::Kind::Struct;
        outSt.origType = st;
        for (unsigned c = 0; c < 4; ++c) outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, compVecs[c], {}, {}});
        outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
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

        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        mlir::Value elemOffVec = hasElemOff ? asVec(loc, getMapped(loc, elemOffArg)) : splatInt(loc, i32Ty(), 0);
        llvm::SmallVector<mlir::Value, 4> valVecs(4);
        mlir::Type elemTy = i32Ty();
        uint32_t elemBytes = 4;
        for (unsigned c = 0; c < 4; ++c) {
            if (mask & (1u << c)) {
                valVecs[c] = asVec(loc, getMapped(loc, call.getOperand(4 + c)));
                elemTy = llvm::cast<mlir::VectorType>(valVecs[c].getType()).getElementType();
                elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);
            }
        }

        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
            mlir::Value descP = laneValue(loc, h, l);
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value elemStride = (!isRawOp && !hasElemOff)
                                         ? loadDescU32Field(loc, descP, offsetof(NacelleDescriptor, elem_stride), 0)
                                         : mlir::Value();
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::arith::AndIOp::create(
                b_, loc, mL, mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP));
            mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
            mlir::Value eL = mlir::vector::ExtractOp::create(b_, loc, elemOffVec, static_cast<int64_t>(l));
            mlir::Value byteOff = computeBufferByteOffset(loc, b_, h, descP, idxL, eL, isRawOp, hasElemOff);

            for (unsigned c = 0; c < 4; ++c) {
                if ((mask & (1u << c)) == 0) continue;
                mlir::Value vL = mlir::vector::ExtractOp::create(b_, loc, valVecs[c], static_cast<int64_t>(l));
                mlir::Value cOff = mlir::arith::AddIOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), c * elemBytes));
                mlir::Value cEnd = mlir::arith::AddIOp::create(b_, loc, cOff, cInt(loc, i64Ty(), elemBytes));
                mlir::Value doStore = mlir::arith::AndIOp::create(
                    b_, loc, okPtr,
                    mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, cEnd, byteSize));
                if (elemStride) {
                    mlir::Value elemEnd = cInt(loc, i32Ty(), (c + 1) * elemBytes);
                    mlir::Value strideZero = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::eq, elemStride, cInt(loc, i32Ty(), 0));
                    mlir::Value inElem = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, elemEnd, elemStride);
                    doStore = mlir::arith::AndIOp::create(
                        b_, loc, doStore, mlir::arith::OrIOp::create(b_, loc, strideZero, inElem));
                }
                mlir::scf::IfOp::create(b_, loc, doStore, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::Value ep = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                    mlir::LLVM::StoreOp::create(nb, nloc, vL, ep);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            }
        }
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
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);

        mlir::Value uniformIdx;
        mlir::Value uniformElemOff;
        if (h.isUniform() && tryGetUniformScalar(loc, idxArg, uniformIdx) &&
            (!hasElemOff || tryGetUniformScalar(loc, elemOffArg, uniformElemOff))) {
            if (!hasElemOff) uniformElemOff = cInt(loc, i32Ty(), 0);
            mlir::Value descP = h.lanes[0];
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP);
            mlir::Value byteOff = computeBufferByteOffset(loc, b_, h, descP, uniformIdx, uniformElemOff, /*isRawOp=*/true, hasElemOff);

            SimdVal vecVal;
            vecVal.kind = SimdVal::Kind::Struct;
            vecVal.origType = vt;
            for (unsigned c = 0; c < numElems; ++c) {
                mlir::Value bvec = emitUniformBufferComponentLoad(
                    loc, elemTy, elemBytes, basePtr, byteSize, okPtr, byteOff, mlir::Value(), c);
                vecVal.fields.push_back({SimdVal::Kind::Vector, elemTy, bvec, {}, {}});
            }
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = st;
            outSt.fields.push_back(std::move(vecVal));
            outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
            setMapped(call.getResult(), outSt);
            return true;
        }

        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        mlir::Value elemOffVec = hasElemOff ? asVec(loc, getMapped(loc, elemOffArg)) : splatInt(loc, i32Ty(), 0);

        llvm::SmallVector<mlir::Value, 16> compVecs(numElems, mlir::LLVM::PoisonOp::create(b_, loc, vecTy(elemTy)));
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value descP = laneValue(loc, h, l);
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP);
            mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
            mlir::Value eL = mlir::vector::ExtractOp::create(b_, loc, elemOffVec, static_cast<int64_t>(l));
            mlir::Value byteOff = computeBufferByteOffset(loc, b_, h, descP, idxL, eL, /*isRawOp=*/true, hasElemOff);

            for (unsigned c = 0; c < numElems; ++c) {
                mlir::Value cOff = mlir::arith::AddIOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), c * elemBytes));
                mlir::Value cEnd = mlir::arith::AddIOp::create(b_, loc, cOff, cInt(loc, i64Ty(), elemBytes));
                mlir::Value inBounds = mlir::arith::AndIOp::create(
                    b_, loc, okPtr,
                    mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, cEnd, byteSize));
                mlir::Value val = mlir::scf::IfOp::create(
                                      b_, loc, inBounds,
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          mlir::Value ep = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::LLVM::LoadOp::create(nb, nloc, elemTy, ep)});
                                      },
                                      [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                          SimdPacketizer tmp(nb, mod_, shader_, rootLayout_, W_, rootCtxPtr_, hooks_);
                                          mlir::Value z = mlir::isa<mlir::FloatType>(elemTy) ? tmp.cFlt(nloc, elemTy, 0.0) : tmp.cInt(nloc, elemTy, 0);
                                          mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z});
                                      })
                                      .getResult(0);
                compVecs[c] = mlir::vector::InsertOp::create(b_, loc, val, compVecs[c], static_cast<int64_t>(l));
            }
        }

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
        outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
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
        mlir::Value firstVec = asVec(loc, valSv.fields[0]);
        mlir::Type elemTy = llvm::cast<mlir::VectorType>(firstVec.getType()).getElementType();
        uint32_t elemBytes = std::max<uint32_t>(1, elemTy.getIntOrFloatBitWidth() / 8);
        mlir::Value idxVec = asVec(loc, getMapped(loc, idxArg));
        mlir::Value elemOffVec = hasElemOff ? asVec(loc, getMapped(loc, elemOffArg)) : splatInt(loc, i32Ty(), 0);

        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
            mlir::Value descP = laneValue(loc, h, l);
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::arith::AndIOp::create(
                b_, loc, mL, mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP));
            mlir::Value idxL = mlir::vector::ExtractOp::create(b_, loc, idxVec, static_cast<int64_t>(l));
            mlir::Value eL = mlir::vector::ExtractOp::create(b_, loc, elemOffVec, static_cast<int64_t>(l));
            mlir::Value byteOff = computeBufferByteOffset(loc, b_, h, descP, idxL, eL, /*isRawOp=*/true, hasElemOff);

            for (unsigned c = 0; c < numElems; ++c) {
                mlir::Value vL = mlir::vector::ExtractOp::create(b_, loc, asVec(loc, valSv.fields[c]), static_cast<int64_t>(l));
                mlir::Value cOff = mlir::arith::AddIOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), c * elemBytes));
                mlir::Value cEnd = mlir::arith::AddIOp::create(b_, loc, cOff, cInt(loc, i64Ty(), elemBytes));
                mlir::Value doStore = mlir::arith::AndIOp::create(
                    b_, loc, okPtr,
                    mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, cEnd, byteSize));
                mlir::scf::IfOp::create(
                    b_, loc, doStore,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value ep = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{cOff});
                        mlir::LLVM::StoreOp::create(nb, nloc, vL, ep);
                        mlir::scf::YieldOp::create(nb, nloc);
                    });
            }
        }
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

        mlir::Value outVec = isFloat ? splatFlt(loc, resTy, 0.0) : splatInt(loc, resTy, 0);
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
            mlir::Value descP = laneValue(loc, h, l);
            mlir::Value basePtr = loadDescBasePtr(loc, descP);
            mlir::Value byteSize = loadDescU64Field(loc, descP, offsetof(NacelleDescriptor, byte_size));
            mlir::Value nullP = mlir::LLVM::ZeroOp::create(b_, loc, ptrTy());
            mlir::Value okPtr = mlir::arith::AndIOp::create(
                b_, loc, mL, mlir::LLVM::ICmpOp::create(b_, loc, mlir::LLVM::ICmpPredicate::ne, basePtr, nullP));
            mlir::Value o0L = mlir::vector::ExtractOp::create(b_, loc, off0, static_cast<int64_t>(l));
            mlir::Value o1L = mlir::vector::ExtractOp::create(b_, loc, off1, static_cast<int64_t>(l));
            mlir::Value byteOff;
            if (h.resourceKind >= 1 && h.resourceKind <= 9) {
                // RWTexture2D atomic: off0=x, off1=y
                mlir::Value rp = loadDescU32Field(loc, descP, offsetof(NacelleDescriptor, row_pitch), 0);
                mlir::Value y64 = mlir::arith::MulIOp::create(
                    b_, loc, mlir::arith::ExtUIOp::create(b_, loc, i64Ty(), o1L),
                    mlir::arith::ExtUIOp::create(b_, loc, i64Ty(), rp));
                mlir::Value x64 = mlir::arith::MulIOp::create(
                    b_, loc, mlir::arith::ExtUIOp::create(b_, loc, i64Ty(), o0L), cInt(loc, i64Ty(), elemBytes));
                byteOff = mlir::arith::AddIOp::create(b_, loc, y64, x64);
            } else {
                byteOff = computeBufferByteOffset(loc, b_, h, descP, o0L, o1L, h.resourceKind == 11 || h.resourceKind == 12, hasOff1);
            }
            mlir::Value cEnd = mlir::arith::AddIOp::create(b_, loc, byteOff, cInt(loc, i64Ty(), elemBytes));
            mlir::Value doOp = mlir::arith::AndIOp::create(
                b_, loc, okPtr,
                mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ule, cEnd, byteSize));
            mlir::Value newL = mlir::vector::ExtractOp::create(b_, loc, newValVec, static_cast<int64_t>(l));
            mlir::Value cmpL = isCmpXchg ? mlir::vector::ExtractOp::create(b_, loc, cmpValVec, static_cast<int64_t>(l)) : mlir::Value();

            mlir::Value resL = mlir::scf::IfOp::create(
                                   b_, loc, doOp,
                                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                       mlir::Value ep = mlir::LLVM::GEPOp::create(nb, nloc, ptrTy(), i8Ty(), basePtr, mlir::ValueRange{byteOff});
                                       mlir::Value r;
                                       if (isCmpXchg) {
                                           mlir::Value cOp = isFloat ? mlir::Value(mlir::arith::BitcastOp::create(nb, nloc, intTy, cmpL)) : cmpL;
                                           mlir::Value nOp = isFloat ? mlir::Value(mlir::arith::BitcastOp::create(nb, nloc, intTy, newL)) : newL;
                                           auto cx = mlir::LLVM::AtomicCmpXchgOp::create(
                                               nb, nloc, ep, cOp, nOp,
                                               mlir::LLVM::AtomicOrdering::seq_cst,
                                               mlir::LLVM::AtomicOrdering::acquire);
                                           r = mlir::LLVM::ExtractValueOp::create(nb, nloc, cx, llvm::ArrayRef<int64_t>{0});
                                           if (isFloat) r = mlir::arith::BitcastOp::create(nb, nloc, resTy, r);
                                       } else if (isFloat && toBinOp(atomicOp) == mlir::LLVM::AtomicBinOp::xchg) {
                                           mlir::Value nOp = mlir::arith::BitcastOp::create(nb, nloc, intTy, newL);
                                           r = mlir::LLVM::AtomicRMWOp::create(
                                               nb, nloc, mlir::LLVM::AtomicBinOp::xchg, ep, nOp, mlir::LLVM::AtomicOrdering::seq_cst);
                                           r = mlir::arith::BitcastOp::create(nb, nloc, resTy, r);
                                       } else {
                                           r = mlir::LLVM::AtomicRMWOp::create(
                                               nb, nloc, toBinOp(atomicOp), ep, newL, mlir::LLVM::AtomicOrdering::seq_cst);
                                       }
                                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{r});
                                   },
                                   [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                                       mlir::Value z;
                                       if (isFloat) {
                                           auto ft = llvm::cast<mlir::FloatType>(resTy);
                                           z = mlir::arith::ConstantFloatOp::create(nb, nloc, ft, mlir::FloatAttr::get(ft, 0.0).getValue());
                                       } else {
                                           z = mlir::arith::ConstantIntOp::create(nb, nloc, resTy, 0);
                                       }
                                       mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{z});
                                   })
                                   .getResult(0);
            outVec = mlir::vector::InsertOp::create(b_, loc, resL, outVec, static_cast<int64_t>(l));
        }
        setMapped(call.getResult(), {SimdVal::Kind::Vector, resTy, outVec, {}, {}});
        return true;
    }

    std::pair<mlir::Value, mlir::Value> computeQuadFiniteDiff(mlir::Location loc, mlir::Value vec, bool fine) {
        llvm::SmallVector<int64_t, 16> loX, hiX, loY, hiY;
        for (uint32_t q = 0; q < W_; q += 4) {
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
        mlir::Value ax = mlir::vector::ShuffleOp::create(b_, loc, vec, vec, loX);
        mlir::Value bx = mlir::vector::ShuffleOp::create(b_, loc, vec, vec, hiX);
        mlir::Value ay = mlir::vector::ShuffleOp::create(b_, loc, vec, vec, loY);
        mlir::Value by = mlir::vector::ShuffleOp::create(b_, loc, vec, vec, hiY);
        return {mlir::arith::SubFOp::create(b_, loc, bx, ax), mlir::arith::SubFOp::create(b_, loc, by, ay)};
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
                mlir::Value m = mlir::vector::ExtractOp::create(b_, loc, mipVec, static_cast<int64_t>(l));
                mlir::LLVM::CallOp::create(b_, loc, fn, mlir::ValueRange{laneValue(loc, h, l), m, tmp4});
                for (unsigned c = 0; c < 4; ++c) {
                    mlir::Value ep = gepByteOffset(loc, tmp4, c * 4);
                    dims[c] = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, i32Ty(), ep), dims[c], static_cast<int64_t>(l));
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
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, realMask, static_cast<int64_t>(l));
                mlir::Value dp = laneValue(loc, h, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    auto ext = [&](mlir::Value v) {
                        mlir::Value ev = mlir::vector::ExtractOp::create(nb, nloc, v, static_cast<int64_t>(l));
                        if (auto it = llvm::dyn_cast<mlir::IntegerType>(ev.getType())) {
                            if (it.getWidth() < 32) {
                                ev = mlir::arith::ExtUIOp::create(nb, nloc, nb.getI32Type(), ev);
                            } else if (it.getWidth() > 32) {
                                ev = mlir::arith::TruncIOp::create(nb, nloc, nb.getI32Type(), ev);
                            }
                            ev = mlir::arith::BitcastOp::create(nb, nloc, nb.getF32Type(), ev);
                        }
                        return ev;
                    };
                    mlir::LLVM::CallOp::create(
                        nb, nloc, fn,
                        mlir::ValueRange{
                            dp,
                            mlir::vector::ExtractOp::create(nb, nloc, x, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, y, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, z, static_cast<int64_t>(l)),
                            ext(r), ext(g), ext(b), ext(a)});
                    mlir::scf::YieldOp::create(nb, nloc);
                });
            }
            return true;
        }

        if (op == 66) { // textureLoad(op, srv, mip, x, y, z, off0, off1, off2)
            SimdVal h = getMapped(loc, call.getOperand(1));
            mlir::Value mip = vecOrZero(2, i32Ty(), 0.0);
            mlir::Value x = mlir::arith::AddIOp::create(b_, loc, vecOrZero(3, i32Ty(), 0.0), vecOrZero(6, i32Ty(), 0.0));
            mlir::Value y = mlir::arith::AddIOp::create(b_, loc, vecOrZero(4, i32Ty(), 0.0), vecOrZero(7, i32Ty(), 0.0));
            mlir::Value z = mlir::arith::AddIOp::create(b_, loc, vecOrZero(5, i32Ty(), 0.0), vecOrZero(8, i32Ty(), 0.0));
            auto fn = getOrDeclareRuntimeFn(loc, "nacelle_tex_load", nullptr, {ptrTy(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            mlir::Value out4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 4), 1, 16);
            llvm::SmallVector<mlir::Value, 4> rgba(4, splatFlt(loc, f32Ty(), 0.0));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
                mlir::Value dp = laneValue(loc, h, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::LLVM::CallOp::create(
                        nb, nloc, fn,
                        mlir::ValueRange{
                            dp,
                            mlir::vector::ExtractOp::create(nb, nloc, x, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, y, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, z, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, mip, static_cast<int64_t>(l)),
                            out4});
                    mlir::scf::YieldOp::create(nb, nloc);
                });
                for (unsigned c = 0; c < 4; ++c) {
                    mlir::Value ep = gepByteOffset(loc, out4, c * 4);
                    rgba[c] = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, f32Ty(), ep), rgba[c], static_cast<int64_t>(l));
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            mlir::Type resElem = llvm::cast<mlir::LLVM::LLVMStructType>(st.origType).getBody()[0];
            for (unsigned c = 0; c < 4; ++c) {
                mlir::Value v = rgba[c];
                if (auto it = llvm::dyn_cast<mlir::IntegerType>(resElem)) {
                    v = mlir::arith::BitcastOp::create(b_, loc, vecTy(i32Ty()), v);
                    if (it.getWidth() < 32) {
                        v = mlir::arith::TruncIOp::create(b_, loc, vecTy(resElem), v);
                    } else if (it.getWidth() > 32) {
                        v = mlir::arith::ExtUIOp::create(b_, loc, vecTy(resElem), v);
                    }
                }
                st.fields.push_back({SimdVal::Kind::Vector, resElem, v, {}, {}});
            }
            st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
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
                                loc, "nacelle_tex_gather_cmp", nullptr,
                                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), ptrTy()})
                          : getOrDeclareRuntimeFn(
                                loc, "nacelle_tex_gather", nullptr,
                                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), ptrTy()});
            mlir::Value out4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 4), 1, 16);
            llvm::SmallVector<mlir::Value, 4> rgba(4, splatFlt(loc, f32Ty(), 0.0));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::Value sp = laneValue(loc, sampH, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    llvm::SmallVector<mlir::Value, 10> args{
                        tp, sp,
                        mlir::vector::ExtractOp::create(nb, nloc, u, static_cast<int64_t>(l)),
                        mlir::vector::ExtractOp::create(nb, nloc, v, static_cast<int64_t>(l)),
                        mlir::vector::ExtractOp::create(nb, nloc, w, static_cast<int64_t>(l)),
                        mlir::vector::ExtractOp::create(nb, nloc, offX, static_cast<int64_t>(l)),
                        mlir::vector::ExtractOp::create(nb, nloc, offY, static_cast<int64_t>(l)),
                        mlir::vector::ExtractOp::create(nb, nloc, comp, static_cast<int64_t>(l))};
                    if (isCmp) args.push_back(mlir::vector::ExtractOp::create(nb, nloc, cmpV, static_cast<int64_t>(l)));
                    args.push_back(out4);
                    mlir::LLVM::CallOp::create(nb, nloc, fn, args);
                    mlir::scf::YieldOp::create(nb, nloc);
                });
                for (unsigned c = 0; c < 4; ++c) {
                    mlir::Value ep = gepByteOffset(loc, out4, c * 4);
                    rgba[c] = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, f32Ty(), ep), rgba[c], static_cast<int64_t>(l));
                }
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            for (unsigned c = 0; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), rgba[c], {}, {}});
            st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
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
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                    mlir::LLVM::CallOp::create(
                        nb, nloc, projFn,
                        mlir::ValueRange{
                            tp,
                            mlir::vector::ExtractOp::create(nb, nloc, u, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, v, static_cast<int64_t>(l)),
                            mlir::vector::ExtractOp::create(nb, nloc, w, static_cast<int64_t>(l)),
                            out3});
                    mlir::scf::YieldOp::create(nb, nloc);
                });
                su = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, f32Ty(), out3), su, static_cast<int64_t>(l));
                sv = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, f32Ty(), gepByteOffset(loc, out3, 4)), sv, static_cast<int64_t>(l));
                sw = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, f32Ty(), gepByteOffset(loc, out3, 8)), sw, static_cast<int64_t>(l));
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
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::Value sp = laneValue(loc, sampH, l);
                auto r = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::Value c1 = mlir::vector::ExtractOp::create(nb, nloc, clampedVec, static_cast<int64_t>(l));
                        mlir::Value c32 = mlir::arith::ExtUIOp::create(nb, nloc, nb.getI32Type(), c1);
                        auto c = mlir::LLVM::CallOp::create(
                            nb, nloc, lodFn,
                            mlir::ValueRange{
                                tp, sp,
                                mlir::vector::ExtractOp::create(nb, nloc, ddxU, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, ddxV, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, ddxW, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, ddyU, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, ddyV, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, ddyW, static_cast<int64_t>(l)),
                                c32});
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult()});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantFloatOp::create(nb, nloc, nb.getF32Type(), llvm::APFloat(0.0f))});
                    });
                resVec = mlir::vector::InsertOp::create(b_, loc, r.getResult(0), resVec, static_cast<int64_t>(l));
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
                mlir::Value d2 = mlir::arith::MaxNumFOp::create(
                    b_, loc,
                    mlir::arith::AddFOp::create(b_, loc, mlir::arith::MulFOp::create(b_, loc, dux, dux), mlir::arith::MulFOp::create(b_, loc, dvx, dvx)),
                    mlir::arith::AddFOp::create(b_, loc, mlir::arith::MulFOp::create(b_, loc, duy, duy), mlir::arith::MulFOp::create(b_, loc, dvy, dvy)));
                // Multiply by approx tex size or let explicit lod stay ~0 if constant; good approximation
                expLod = mlir::arith::MulFOp::create(
                    b_, loc, splatFlt(loc, f32Ty(), 0.5),
                    mlir::math::Log2Op::create(b_, loc, mlir::arith::MaxNumFOp::create(b_, loc, d2, splatFlt(loc, f32Ty(), 1e-12))));
                expLod = mlir::arith::MaxNumFOp::create(b_, loc, expLod, splatFlt(loc, f32Ty(), 0.0));
            }
            auto fn = getOrDeclareRuntimeFn(
                loc, "nacelle_tex_sample_cmp", f32Ty(),
                {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty()});
            mlir::Value resVec = splatFlt(loc, f32Ty(), 0.0);
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
                mlir::Value tp = laneValue(loc, texH, l);
                mlir::Value sp = laneValue(loc, sampH, l);
                auto r = mlir::scf::IfOp::create(
                    b_, loc, mL,
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        auto c = mlir::LLVM::CallOp::create(
                            nb, nloc, fn,
                            mlir::ValueRange{
                                tp, sp,
                                mlir::vector::ExtractOp::create(nb, nloc, u, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, v, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, w, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, cmpV, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, expLod, static_cast<int64_t>(l)),
                                mlir::arith::ConstantIntOp::create(nb, nloc, i32Ty(), isL0),
                                mlir::vector::ExtractOp::create(nb, nloc, offX, static_cast<int64_t>(l)),
                                mlir::vector::ExtractOp::create(nb, nloc, offY, static_cast<int64_t>(l))});
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{c.getResult()});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                        mlir::scf::YieldOp::create(nb, nloc, mlir::ValueRange{mlir::arith::ConstantFloatOp::create(nb, nloc, nb.getF32Type(), llvm::APFloat(0.0f))});
                    });
                resVec = mlir::vector::InsertOp::create(b_, loc, r.getResult(0), resVec, static_cast<int64_t>(l));
            }
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), resVec, {}, {}});
            for (unsigned c = 1; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), splatFlt(loc, f32Ty(), 0.0), {}, {}});
            st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
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
            loc, "nacelle_tex_sample", nullptr,
            {ptrTy(), ptrTy(), f32Ty(), f32Ty(), f32Ty(), f32Ty(),
             f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(), f32Ty(),
             f32Ty(), f32Ty(), i32Ty(), i32Ty(), i32Ty(), i32Ty(), f32Ty(), ptrTy()});
        mlir::Value out4 = createEntryAlloca(loc, mlir::LLVM::LLVMArrayType::get(f32Ty(), 4), 1, 16);
        llvm::SmallVector<mlir::Value, 4> rgba(4, splatFlt(loc, f32Ty(), 0.0));
        for (uint32_t l = 0; l < W_; ++l) {
            mlir::Value mL = mlir::vector::ExtractOp::create(b_, loc, execMask, static_cast<int64_t>(l));
            mlir::Value tp = laneValue(loc, texH, l);
            mlir::Value sp = laneValue(loc, sampH, l);
            mlir::scf::IfOp::create(b_, loc, mL, [&](mlir::OpBuilder& nb, mlir::Location nloc) {
                auto ex = [&](mlir::Value vv) { return mlir::vector::ExtractOp::create(nb, nloc, vv, static_cast<int64_t>(l)); };
                mlir::LLVM::CallOp::create(
                    nb, nloc, fn,
                    mlir::ValueRange{
                        tp, sp, ex(u), ex(v), ex(w), ex(c3),
                        ex(ddxU), ex(ddxV), ex(ddxW), ex(ddyU), ex(ddyV), ex(ddyW),
                        ex(bias), ex(expLod),
                        mlir::arith::ConstantIntOp::create(nb, nloc, i32Ty(), lodMode),
                        ex(offX), ex(offY), ex(offZ), ex(clampLod), out4});
                mlir::scf::YieldOp::create(nb, nloc);
            });
            for (unsigned c = 0; c < 4; ++c) {
                mlir::Value ep = gepByteOffset(loc, out4, c * 4);
                rgba[c] = mlir::vector::InsertOp::create(b_, loc, loadScalar(loc, f32Ty(), ep), rgba[c], static_cast<int64_t>(l));
            }
        }
        SimdVal st;
        st.kind = SimdVal::Kind::Struct;
        st.origType = call.getResult().getType();
        for (unsigned c = 0; c < 4; ++c) st.fields.push_back({SimdVal::Kind::Vector, f32Ty(), rgba[c], {}, {}});
        st.fields.push_back({SimdVal::Kind::Vector, i32Ty(), splatInt(loc, i32Ty(), 0), {}, {}});
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
                mlir::Value bits = mlir::arith::ExtUIOp::create(b_, loc, vecTy(i32Ty()), m);
                mlir::Value shl = mlir::arith::ShLIOp::create(b_, loc, bits, laneIota(loc));
                mlir::Value w0 = mlir::vector::ReductionOp::create(b_, loc, mlir::vector::CombiningKind::OR, shl);
                return {w0, zero, zero, zero};
            }
            mlir::VectorType v32i32 = mlir::VectorType::get({32}, i32Ty());
            llvm::SmallVector<int32_t, 32> iotaVals(32);
            for (int32_t i = 0; i < 32; ++i) iotaVals[i] = i;
            mlir::Value iota32 = mlir::arith::ConstantOp::create(
                b_, loc, mlir::DenseElementsAttr::get(llvm::cast<mlir::ShapedType>(v32i32), llvm::ArrayRef<int32_t>(iotaVals)));
            std::array<mlir::Value, 4> words{zero, zero, zero, zero};
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (uint32_t w = 0; w < numWords; ++w) {
                mlir::Value slice = mlir::vector::ExtractStridedSliceOp::create(
                    b_, loc, m, llvm::ArrayRef<int64_t>{static_cast<int64_t>(w * 32u)},
                    llvm::ArrayRef<int64_t>{32}, llvm::ArrayRef<int64_t>{1});
                mlir::Value bits = mlir::arith::ExtUIOp::create(b_, loc, v32i32, slice);
                mlir::Value shl = mlir::arith::ShLIOp::create(b_, loc, bits, iota32);
                words[w] = mlir::vector::ReductionOp::create(b_, loc, mlir::vector::CombiningKind::OR, shl);
            }
            return words;
        };

        auto buildFirstActiveLane = [&](mlir::Value m) -> mlir::Value {
            auto words = buildBallotWords(m);
            if (W_ <= 32) {
                return mlir::arith::MinUIOp::create(
                    b_, loc, mlir::math::CountTrailingZerosOp::create(b_, loc, words[0]), cInt(loc, i32Ty(), W_ - 1));
            }
            mlir::Value firstIdx = cInt(loc, i32Ty(), W_ - 1);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (int w = static_cast<int>(numWords) - 1; w >= 0; --w) {
                mlir::Value nz = mlir::arith::CmpIOp::create(
                    b_, loc, mlir::arith::CmpIPredicate::ne, words[w], cInt(loc, i32Ty(), 0));
                mlir::Value idx = mlir::arith::AddIOp::create(
                    b_, loc, cInt(loc, i32Ty(), static_cast<uint64_t>(w) * 32u),
                    mlir::math::CountTrailingZerosOp::create(b_, loc, words[w]));
                firstIdx = mlir::arith::SelectOp::create(b_, loc, nz, idx, firstIdx);
            }
            return firstIdx;
        };

        if (op == 110) { // waveIsFirstLane
            mlir::Value firstIdx = buildFirstActiveLane(activeMask);
            mlir::Value firstVec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(i32Ty()), firstIdx);
            mlir::Value isFirst = mlir::arith::CmpIOp::create(
                b_, loc, mlir::arith::CmpIPredicate::eq, laneIota(loc), firstVec);
            isFirst = mlir::arith::AndIOp::create(b_, loc, isFirst, activeMask);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i1Ty(), isFirst, {}, {}});
            return true;
        }
        if (op == 113 || op == 114) { // waveAnyTrue / waveAllTrue
            mlir::Value condVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value scalarRes;
            if (op == 113) {
                mlir::Value actCond = mlir::arith::AndIOp::create(b_, loc, condVec, activeMask);
                scalarRes = mlir::vector::ReductionOp::create(b_, loc, mlir::vector::CombiningKind::OR, actCond);
            } else {
                // !active || cond across all lanes
                mlir::Value inact = mlir::arith::XOrIOp::create(b_, loc, activeMask, allTrueMask(loc));
                mlir::Value impl = mlir::arith::OrIOp::create(b_, loc, condVec, inact);
                scalarRes = mlir::vector::ReductionOp::create(b_, loc, mlir::vector::CombiningKind::AND, impl);
            }
            mlir::Value bvec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(i1Ty()), scalarRes);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i1Ty(), bvec, {}, {}});
            return true;
        }
        if (op == 116) { // waveActiveBallot(cond) -> %dx.types.fouri32
            mlir::Value condVec = mlir::arith::AndIOp::create(b_, loc, asVec(loc, getMapped(loc, arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            SimdVal st;
            st.kind = SimdVal::Kind::Struct;
            st.origType = call.getResult().getType();
            for (int i = 0; i < 4; ++i) {
                st.fields.push_back({SimdVal::Kind::Vector, i32Ty(),
                                     mlir::vector::BroadcastOp::create(b_, loc, vecTy(i32Ty()), words[i]), {}, {}});
            }
            setMapped(call.getResult(), st);
            return true;
        }
        if (op == 118) { // waveReadLaneFirst(val)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value firstIdx = buildFirstActiveLane(activeMask);
            mlir::Value scalar = mlir::vector::ExtractOp::create(
                b_, loc, valVec, mlir::OpFoldResult{mlir::arith::IndexCastOp::create(b_, loc, b_.getIndexType(), firstIdx)});
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, mlir::vector::BroadcastOp::create(b_, loc, vecTy(et), scalar), {}, {}});
            return true;
        }
        if (op == 117) { // waveReadLaneAt(val, lane)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value laneVec = asVec(loc, getMapped(loc, arg(2)));
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(et));
            for (uint32_t l = 0; l < W_; ++l) {
                mlir::Value li = mlir::vector::ExtractOp::create(b_, loc, laneVec, static_cast<int64_t>(l));
                mlir::Value clamped = mlir::arith::MinUIOp::create(b_, loc, li, cInt(loc, i32Ty(), W_ - 1));
                mlir::Value idx = mlir::arith::IndexCastOp::create(b_, loc, b_.getIndexType(), clamped);
                mlir::Value s = mlir::vector::ExtractOp::create(b_, loc, valVec, mlir::OpFoldResult{idx});
                outVec = mlir::vector::InsertOp::create(b_, loc, s, outVec, static_cast<int64_t>(l));
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, outVec, {}, {}});
            return true;
        }
        if (op == 115) { // waveActiveAllEqual(val)
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Value firstIdx = buildFirstActiveLane(activeMask);
            mlir::Value firstScalar = mlir::vector::ExtractOp::create(
                b_, loc, valVec, mlir::OpFoldResult{mlir::arith::IndexCastOp::create(b_, loc, b_.getIndexType(), firstIdx)});
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            mlir::Value firstVec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(et), firstScalar);
            mlir::Value eq = mlir::isa<mlir::FloatType>(et)
                                 ? mlir::arith::CmpFOp::create(b_, loc, mlir::arith::CmpFPredicate::OEQ, valVec, firstVec).getResult()
                                 : mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::eq, valVec, firstVec).getResult();
            mlir::Value inact = mlir::arith::XOrIOp::create(b_, loc, activeMask, allTrueMask(loc));
            mlir::Value allEq = mlir::vector::ReductionOp::create(
                b_, loc, mlir::vector::CombiningKind::AND, mlir::arith::OrIOp::create(b_, loc, eq, inact));
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i1Ty(), mlir::vector::BroadcastOp::create(b_, loc, vecTy(i1Ty()), allEq), {}, {}});
            return true;
        }
        if (op == 119 || op == 120) { // waveActiveOp / waveActiveBit
            mlir::Value valVec = asVec(loc, getMapped(loc, arg(1)));
            mlir::Type et = llvm::cast<mlir::VectorType>(valVec.getType()).getElementType();
            uint32_t kind = static_cast<uint32_t>(getConstIntValue(arg(2)).value_or(0));
            uint32_t sop = (op == 119) ? static_cast<uint32_t>(getConstIntValue(arg(3)).value_or(0)) : 0u;
            bool isFloat = mlir::isa<mlir::FloatType>(et);

            mlir::Value identity;
            mlir::vector::CombiningKind ck = mlir::vector::CombiningKind::ADD;
            if (op == 120) {
                if (kind == 0) { // AND
                    ck = mlir::vector::CombiningKind::AND;
                    identity = splatInt(loc, et, ~uint64_t(0));
                } else if (kind == 1) { // OR
                    ck = mlir::vector::CombiningKind::OR;
                    identity = splatInt(loc, et, 0);
                } else { // XOR
                    ck = mlir::vector::CombiningKind::XOR;
                    identity = splatInt(loc, et, 0);
                }
            } else if (kind == 0) { // SUM
                ck = mlir::vector::CombiningKind::ADD;
                identity = isFloat ? splatFlt(loc, et, 0.0) : splatInt(loc, et, 0);
            } else if (kind == 1) { // PROD
                ck = mlir::vector::CombiningKind::MUL;
                identity = isFloat ? splatFlt(loc, et, 1.0) : splatInt(loc, et, 1);
            } else if (kind == 2) { // MIN
                if (isFloat) {
                    ck = mlir::vector::CombiningKind::MINIMUMF;
                    identity = splatFlt(loc, et, std::numeric_limits<double>::infinity());
                } else if (sop != 0) {
                    ck = mlir::vector::CombiningKind::MINUI;
                    identity = splatInt(loc, et, ~uint64_t(0));
                } else {
                    ck = mlir::vector::CombiningKind::MINSI;
                    unsigned bw = et.getIntOrFloatBitWidth();
                    uint64_t sMax = (bw >= 64) ? uint64_t(INT64_MAX) : ((uint64_t(1) << (bw - 1)) - 1);
                    identity = splatInt(loc, et, sMax);
                }
            } else { // MAX
                if (isFloat) {
                    ck = mlir::vector::CombiningKind::MAXIMUMF;
                    identity = splatFlt(loc, et, -std::numeric_limits<double>::infinity());
                } else if (sop != 0) {
                    ck = mlir::vector::CombiningKind::MAXUI;
                    identity = splatInt(loc, et, 0);
                } else {
                    ck = mlir::vector::CombiningKind::MAXSI;
                    unsigned bw = et.getIntOrFloatBitWidth();
                    uint64_t sMin = (bw >= 64) ? uint64_t(INT64_MIN) : (uint64_t(1) << (bw - 1));
                    identity = splatInt(loc, et, sMin);
                }
            }
            mlir::Value masked = mlir::arith::SelectOp::create(b_, loc, activeMask, valVec, identity);
            mlir::Value reduced = mlir::vector::ReductionOp::create(b_, loc, ck, masked);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, mlir::vector::BroadcastOp::create(b_, loc, vecTy(et), reduced), {}, {}});
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
            mlir::Value contrib = mlir::arith::SelectOp::create(b_, loc, activeMask, valVec, idVec);
            mlir::Value acc = idScalar;
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(et));
            for (uint32_t l = 0; l < W_; ++l) {
                outVec = mlir::vector::InsertOp::create(b_, loc, acc, outVec, static_cast<int64_t>(l));
                mlir::Value vL = mlir::vector::ExtractOp::create(b_, loc, contrib, static_cast<int64_t>(l));
                if (isFloat) {
                    acc = isMul ? mlir::arith::MulFOp::create(b_, loc, acc, vL).getResult()
                                : mlir::arith::AddFOp::create(b_, loc, acc, vL).getResult();
                } else {
                    acc = isMul ? mlir::arith::MulIOp::create(b_, loc, acc, vL).getResult()
                                : mlir::arith::AddIOp::create(b_, loc, acc, vL).getResult();
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
            llvm::SmallVector<int64_t, 16> indices;
            indices.reserve(W_);
            for (uint32_t l = 0; l < W_; ++l) {
                uint32_t peer = l ^ xorMask;
                indices.push_back(peer < W_ ? static_cast<int64_t>(peer) : static_cast<int64_t>(l));
            }
            mlir::Value res = mlir::vector::ShuffleOp::create(b_, loc, valVec, valVec, indices);
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, res, {}, {}});
            return true;
        }
        if (op == 135) { // waveAllBitCount(cond) -> i32
            mlir::Value condVec = mlir::arith::AndIOp::create(b_, loc, asVec(loc, getMapped(loc, arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            mlir::Value cnt = mlir::math::CtPopOp::create(b_, loc, words[0]);
            for (uint32_t w = 1; w < numWords; ++w) {
                mlir::Value wc = mlir::math::CtPopOp::create(b_, loc, words[w]);
                cnt = mlir::arith::AddIOp::create(b_, loc, cnt, wc);
            }
            setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), mlir::vector::BroadcastOp::create(b_, loc, vecTy(i32Ty()), cnt), {}, {}});
            return true;
        }
        if (op == 136) { // wavePrefixBitCount(cond) -> i32
            mlir::Value condVec = mlir::arith::AndIOp::create(b_, loc, asVec(loc, getMapped(loc, arg(1))), activeMask);
            auto words = buildBallotWords(condVec);
            if (W_ <= 32) {
                mlir::Value ballotVec = mlir::vector::BroadcastOp::create(b_, loc, vecTy(i32Ty()), words[0]);
                llvm::SmallVector<int32_t, 32> laneMasks;
                laneMasks.reserve(W_);
                for (uint32_t l = 0; l < W_; ++l) {
                    laneMasks.push_back(static_cast<int32_t>((uint32_t(1) << l) - 1u));
                }
                mlir::Value maskVec = mlir::arith::ConstantOp::create(
                    b_, loc, mlir::DenseElementsAttr::get(llvm::cast<mlir::ShapedType>(vecTy(i32Ty())), llvm::ArrayRef<int32_t>(laneMasks)));
                mlir::Value masked = mlir::arith::AndIOp::create(b_, loc, ballotVec, maskVec);
                mlir::Value cntVec = mlir::math::CtPopOp::create(b_, loc, masked);
                setMapped(call.getResult(), {SimdVal::Kind::Vector, i32Ty(), cntVec, {}, {}});
                return true;
            }
            mlir::VectorType v32i32 = mlir::VectorType::get({32}, i32Ty());
            llvm::SmallVector<int32_t, 32> laneMasks32(32);
            for (int32_t l = 0; l < 32; ++l) {
                laneMasks32[l] = static_cast<int32_t>((uint32_t(1) << l) - 1u);
            }
            mlir::Value maskVec32 = mlir::arith::ConstantOp::create(
                b_, loc, mlir::DenseElementsAttr::get(llvm::cast<mlir::ShapedType>(v32i32), llvm::ArrayRef<int32_t>(laneMasks32)));
            mlir::Value outVec = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(i32Ty()));
            mlir::Value prefixSum = cInt(loc, i32Ty(), 0);
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);
            for (uint32_t w = 0; w < numWords; ++w) {
                mlir::Value bVec = mlir::vector::BroadcastOp::create(b_, loc, v32i32, words[w]);
                mlir::Value inWord = mlir::math::CtPopOp::create(b_, loc, mlir::arith::AndIOp::create(b_, loc, bVec, maskVec32));
                mlir::Value chunk = mlir::arith::AddIOp::create(
                    b_, loc, inWord, mlir::vector::BroadcastOp::create(b_, loc, v32i32, prefixSum));
                outVec = mlir::vector::InsertStridedSliceOp::create(
                    b_, loc, chunk, outVec, llvm::ArrayRef<int64_t>{static_cast<int64_t>(w * 32u)},
                    llvm::ArrayRef<int64_t>{1});
                if (w + 1 < numWords) {
                    prefixSum = mlir::arith::AddIOp::create(b_, loc, prefixSum, mlir::math::CtPopOp::create(b_, loc, words[w]));
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

            mlir::Value idScalar;
            if (kind == 1) {
                idScalar = cInt(loc, et, ~uint64_t(0));
            } else if (kind == 4) {
                idScalar = isFloat ? cFlt(loc, et, 1.0) : cInt(loc, et, 1);
            } else {
                idScalar = isFloat ? cFlt(loc, et, 0.0) : cInt(loc, et, 0);
            }

            mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b_, loc, 0);
            mlir::Value c1 = mlir::arith::ConstantIndexOp::create(b_, loc, 1);
            mlir::Value cW = mlir::arith::ConstantIndexOp::create(b_, loc, W_);
            mlir::Value initOut = mlir::LLVM::PoisonOp::create(b_, loc, vecTy(et));
            uint32_t numWords = std::min<uint32_t>(4u, (W_ + 31u) / 32u);

            auto outer = mlir::scf::ForOp::create(
                b_, loc, c0, cW, c1, mlir::ValueRange{initOut},
                [&](mlir::OpBuilder& ob, mlir::Location oloc, mlir::Value lIdx, mlir::ValueRange outerArgs) {
                    mlir::Value curOut = outerArgs[0];
                    llvm::SmallVector<mlir::Value, 4> maskL(numWords);
                    for (uint32_t w = 0; w < numWords; ++w) {
                        maskL[w] = mlir::vector::ExtractOp::create(ob, oloc, maskVecs[w], mlir::OpFoldResult{lIdx});
                    }
                    auto inner = mlir::scf::ForOp::create(
                        ob, oloc, c0, lIdx, c1, mlir::ValueRange{idScalar},
                        [&](mlir::OpBuilder& ib, mlir::Location iloc, mlir::Value jIdx, mlir::ValueRange innerArgs) {
                            mlir::Value acc = innerArgs[0];
                            mlir::Value j32 = mlir::arith::IndexCastOp::create(ib, iloc, i32Ty(), jIdx);
                            mlir::Value maskWordL = maskL[0];
                            if (numWords > 1) {
                                mlir::Value wIdx = mlir::arith::ShRUIOp::create(
                                    ib, iloc, j32, mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), 5));
                                for (uint32_t w = 1; w < numWords; ++w) {
                                    mlir::Value isW = mlir::arith::CmpIOp::create(
                                        ib, iloc, mlir::arith::CmpIPredicate::eq, wIdx,
                                        mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), w));
                                    maskWordL = mlir::arith::SelectOp::create(ib, iloc, isW, maskL[w], maskWordL);
                                }
                            }
                            mlir::Value bitIdx = mlir::arith::AndIOp::create(
                                ib, iloc, j32, mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), 31));
                            mlir::Value shifted = mlir::arith::ShRUIOp::create(ib, iloc, maskWordL, bitIdx);
                            mlir::Value bitJ = mlir::arith::AndIOp::create(
                                ib, iloc, shifted, mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), 1));
                            mlir::Value inGroup = mlir::arith::CmpIOp::create(
                                ib, iloc, mlir::arith::CmpIPredicate::ne, bitJ,
                                mlir::arith::ConstantIntOp::create(ib, iloc, i32Ty(), 0));
                            mlir::Value actJ = mlir::vector::ExtractOp::create(ib, iloc, activeMask, mlir::OpFoldResult{jIdx});
                            mlir::Value take = mlir::arith::AndIOp::create(ib, iloc, inGroup, actJ);
                            mlir::Value valJ = mlir::vector::ExtractOp::create(ib, iloc, valVec, mlir::OpFoldResult{jIdx});

                            mlir::Value cand;
                            if (kind == 0) {
                                cand = isFloat ? mlir::arith::AddFOp::create(ib, iloc, acc, valJ).getResult()
                                               : mlir::arith::AddIOp::create(ib, iloc, acc, valJ).getResult();
                            } else if (kind == 1) {
                                cand = mlir::arith::AndIOp::create(ib, iloc, acc, valJ);
                            } else if (kind == 2) {
                                cand = mlir::arith::OrIOp::create(ib, iloc, acc, valJ);
                            } else if (kind == 3) {
                                cand = mlir::arith::XOrIOp::create(ib, iloc, acc, valJ);
                            } else {
                                cand = isFloat ? mlir::arith::MulFOp::create(ib, iloc, acc, valJ).getResult()
                                               : mlir::arith::MulIOp::create(ib, iloc, acc, valJ).getResult();
                            }
                            mlir::Value nextAcc = mlir::arith::SelectOp::create(ib, iloc, take, cand, acc);
                            mlir::scf::YieldOp::create(ib, iloc, mlir::ValueRange{nextAcc});
                        });
                    mlir::Value nextOut = mlir::vector::InsertOp::create(
                        ob, oloc, inner.getResult(0), curOut, mlir::OpFoldResult{lIdx});
                    mlir::scf::YieldOp::create(ob, oloc, mlir::ValueRange{nextOut});
                });
            setMapped(call.getResult(), {SimdVal::Kind::Vector, et, outer.getResult(0), {}, {}});
            return true;
        }
        return false;
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
                    mlir::Value bits = mlir::arith::BitcastOp::create(b_, loc, vecTy(i32Ty()), v);
                    mlir::Value absBits = mlir::arith::AndIOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x7FFFFFFFu));
                    mlir::Value isDenorm = mlir::arith::CmpIOp::create(
                        b_, loc, mlir::arith::CmpIPredicate::ult, absBits, splatInt(loc, i32Ty(), 0x00800000u));
                    mlir::Value signZero = mlir::arith::AndIOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x80000000u));
                    mlir::Value flushed = mlir::arith::SelectOp::create(b_, loc, isDenorm, signZero, bits);
                    return mlir::arith::BitcastOp::create(b_, loc, vT, flushed);
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
            mlir::Value bits = mlir::arith::BitcastOp::create(b_, loc, vecTy(i32Ty()), v);
            mlir::Value absBits = mlir::arith::AndIOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x7FFFFFFFu));
            mlir::Value isDenorm = mlir::arith::CmpIOp::create(
                b_, loc, mlir::arith::CmpIPredicate::ult, absBits, splatInt(loc, i32Ty(), 0x00800000u));
            mlir::Value signZero = mlir::arith::AndIOp::create(b_, loc, bits, splatInt(loc, i32Ty(), 0x80000000u));
            mlir::Value flushed = mlir::arith::SelectOp::create(b_, loc, isDenorm, signZero, bits);
            return mlir::arith::BitcastOp::create(b_, loc, vT, flushed);
        };

        return packetizeScalarMathSwitch(call, op, loc, resTy, vT, argV, store, flushDenormF, outError);
    }

    template <typename ArgVFn, typename StoreFn, typename FlushFn>
    bool packetizeScalarMathSwitch(
        mlir::LLVM::CallOp call, uint32_t op, mlir::Location loc, mlir::Type resTy, mlir::Type vT,
        ArgVFn&& argV, StoreFn&& store, FlushFn&& flushDenormF, std::string& outError) {
        switch (op) {
        case 6: return store(mlir::math::AbsFOp::create(b_, loc, vT, argV(1)));
        case 7: { // Saturate
            mlir::Value lo = mlir::arith::MaxNumFOp::create(b_, loc, vT, argV(1), splatFlt(loc, resTy, 0.0));
            return store(mlir::arith::MinNumFOp::create(b_, loc, vT, lo, splatFlt(loc, resTy, 1.0)));
        }
        case 8: return store(mlir::math::IsNaNOp::create(b_, loc, vT, argV(1)));
        case 9: return store(mlir::math::IsInfOp::create(b_, loc, vT, argV(1)));
        case 10: return store(mlir::math::IsFiniteOp::create(b_, loc, vT, argV(1)));
        case 11: return store(mlir::math::IsNormalOp::create(b_, loc, vT, argV(1)));
        case 12: return store(mlir::math::CosOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 13: return store(mlir::math::SinOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 14: return store(mlir::math::TanOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 15: return store(mlir::math::AcosOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 16: return store(mlir::math::AsinOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 17: return store(mlir::math::AtanOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 18: return store(mlir::math::CoshOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 19: return store(mlir::math::SinhOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 20: return store(mlir::math::TanhOp::create(b_, loc, vT, flushDenormF(argV(1))));
        case 21: return store(mlir::math::Exp2Op::create(b_, loc, vT, flushDenormF(argV(1)))); // DXIL Exp is base-2!
        case 22: { // Frc = x - floor(x)
            mlir::Value x = argV(1);
            return store(mlir::arith::SubFOp::create(b_, loc, vT, x, mlir::math::FloorOp::create(b_, loc, vT, x)));
        }
        case 23: return store(mlir::math::Log2Op::create(b_, loc, vT, flushDenormF(argV(1)))); // DXIL Log is base-2!
        case 24: return store(mlir::math::SqrtOp::create(b_, loc, vT, argV(1)));
        case 25: { // Rsqrt = 1.0 / sqrt(x)
            mlir::Value sq = mlir::math::SqrtOp::create(b_, loc, vT, argV(1));
            return store(mlir::arith::DivFOp::create(b_, loc, vT, splatFlt(loc, resTy, 1.0), sq));
        }
        case 26: return store(mlir::math::RoundEvenOp::create(b_, loc, vT, argV(1)));
        case 27: return store(mlir::math::FloorOp::create(b_, loc, vT, argV(1)));
        case 28: return store(mlir::math::CeilOp::create(b_, loc, vT, argV(1)));
        case 29: return store(mlir::math::TruncOp::create(b_, loc, vT, argV(1)));
        case 30: return store(mlir::LLVM::BitReverseOp::create(b_, loc, vT, argV(1)));
        case 31: { // Countbits: population count, result is always i32 in DXIL
            mlir::Value x = argV(1);
            mlir::Type argTy = llvm::cast<mlir::VectorType>(x.getType()).getElementType();
            mlir::Value pop = mlir::math::CtPopOp::create(b_, loc, vecTy(argTy), x);
            if (argTy != resTy) {
                pop = (argTy.getIntOrFloatBitWidth() > resTy.getIntOrFloatBitWidth())
                          ? mlir::Value(mlir::arith::TruncIOp::create(b_, loc, vT, pop))
                          : mlir::Value(mlir::arith::ExtUIOp::create(b_, loc, vT, pop));
            }
            return store(pop);
        }
        case 32: { // FirstbitLo: index of lowest set bit, or -1 if 0
            mlir::Value x = argV(1);
            mlir::Type argTy = llvm::cast<mlir::VectorType>(x.getType()).getElementType();
            mlir::Value tz = mlir::math::CountTrailingZerosOp::create(b_, loc, vecTy(argTy), x);
            if (argTy != resTy) {
                tz = (argTy.getIntOrFloatBitWidth() > resTy.getIntOrFloatBitWidth())
                         ? mlir::Value(mlir::arith::TruncIOp::create(b_, loc, vT, tz))
                         : mlir::Value(mlir::arith::ExtUIOp::create(b_, loc, vT, tz));
            }
            mlir::Value isZero = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::eq, x, splatInt(loc, argTy, 0));
            return store(mlir::arith::SelectOp::create(b_, loc, isZero, splatInt(loc, resTy, ~uint64_t(0)), tz));
        }
        case 33: // FirstbitHi: clz(x), or -1 if 0 (DXIL returns leading-zero count from MSB; DXC emits bitWidth-1 - res)
        case 34: { // FirstbitSHi: clz(x < 0 ? ~x : x), or -1 if 0 or -1
            mlir::Value x = argV(1);
            mlir::Type argTy = llvm::cast<mlir::VectorType>(x.getType()).getElementType();
            if (op == 34) {
                mlir::Value isNeg = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::slt, x, splatInt(loc, argTy, 0));
                mlir::Value inv = mlir::arith::XOrIOp::create(b_, loc, x, splatInt(loc, argTy, ~uint64_t(0)));
                x = mlir::arith::SelectOp::create(b_, loc, isNeg, inv, x);
            }
            mlir::Value lz = mlir::math::CountLeadingZerosOp::create(b_, loc, vecTy(argTy), x);
            if (argTy != resTy) {
                lz = (argTy.getIntOrFloatBitWidth() > resTy.getIntOrFloatBitWidth())
                         ? mlir::Value(mlir::arith::TruncIOp::create(b_, loc, vT, lz))
                         : mlir::Value(mlir::arith::ExtUIOp::create(b_, loc, vT, lz));
            }
            mlir::Value isZero = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::eq, x, splatInt(loc, argTy, 0));
            return store(mlir::arith::SelectOp::create(b_, loc, isZero, splatInt(loc, resTy, ~uint64_t(0)), lz));
        }
        case 35: return store(mlir::arith::MaxNumFOp::create(b_, loc, vT, argV(1), argV(2)));
        case 36: return store(mlir::arith::MinNumFOp::create(b_, loc, vT, argV(1), argV(2)));
        case 37: return store(mlir::arith::MaxSIOp::create(b_, loc, vT, argV(1), argV(2)));
        case 38: return store(mlir::arith::MinSIOp::create(b_, loc, vT, argV(1), argV(2)));
        case 39: return store(mlir::arith::MaxUIOp::create(b_, loc, vT, argV(1), argV(2)));
        case 40: return store(mlir::arith::MinUIOp::create(b_, loc, vT, argV(1), argV(2)));
        case 44: { // UAddc(a, b) -> { sum, carry }
            auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(resTy);
            if (!st || st.getBody().size() < 2) {
                outError = "UAddc expected { T, i1 } struct return type";
                return false;
            }
            mlir::Type elemTy = st.getBody()[0];
            mlir::Value a = argV(1);
            mlir::Value b = argV(2);
            mlir::Value sum = mlir::arith::AddIOp::create(b_, loc, a, b);
            mlir::Value carry = mlir::arith::CmpIOp::create(b_, loc, mlir::arith::CmpIPredicate::ult, sum, a);
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            outSt.fields.push_back({SimdVal::Kind::Vector, elemTy, sum, {}, {}});
            outSt.fields.push_back({SimdVal::Kind::Vector, i1Ty(), carry, {}, {}});
            setMapped(call.getResult(), outSt);
            return true;
        }
        case 46:
        case 47: return store(mlir::math::FmaOp::create(b_, loc, vT, argV(1), argV(2), argV(3)));
        case 48:
        case 49: {
            mlir::Value prod = mlir::arith::MulIOp::create(b_, loc, vT, argV(1), argV(2));
            return store(mlir::arith::AddIOp::create(b_, loc, vT, prod, argV(3)));
        }
        case 54: // Dot2
        case 55: // Dot3
        case 56: { // Dot4
            unsigned n = (op == 54) ? 2u : (op == 55 ? 3u : 4u);
            mlir::Value sum = mlir::arith::MulFOp::create(b_, loc, vT, argV(1), argV(1 + n));
            for (unsigned c = 1; c < n; ++c) {
                mlir::Value p = mlir::arith::MulFOp::create(b_, loc, vT, argV(1 + c), argV(1 + n + c));
                sum = mlir::arith::AddFOp::create(b_, loc, vT, sum, p);
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
            mlir::Value lo64 = mlir::arith::ExtUIOp::create(b_, loc, vecTy(i64Ty()), argV(1));
            mlir::Value hi64 = mlir::arith::ExtUIOp::create(b_, loc, vecTy(i64Ty()), argV(2));
            mlir::Value hiShift = mlir::arith::ShLIOp::create(b_, loc, hi64, splatInt(loc, i64Ty(), 32));
            mlir::Value bits = mlir::arith::OrIOp::create(b_, loc, lo64, hiShift);
            return store(mlir::arith::BitcastOp::create(b_, loc, vecTy(b_.getF64Type()), bits));
        }
        case 102: { // SplitDouble(val) -> { i32 lo, i32 hi }
            auto st = llvm::dyn_cast<mlir::LLVM::LLVMStructType>(resTy);
            if (!st || st.getBody().size() < 2) {
                outError = "SplitDouble expected { i32, i32 } struct return type";
                return false;
            }
            mlir::Value bits = mlir::arith::BitcastOp::create(b_, loc, vecTy(i64Ty()), argV(1));
            mlir::Value lo = mlir::arith::TruncIOp::create(b_, loc, vecTy(i32Ty()), bits);
            mlir::Value hi = mlir::arith::TruncIOp::create(
                b_, loc, vecTy(i32Ty()),
                mlir::arith::ShRUIOp::create(b_, loc, bits, splatInt(loc, i64Ty(), 32)));
            SimdVal outSt;
            outSt.kind = SimdVal::Kind::Struct;
            outSt.origType = resTy;
            outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), lo, {}, {}});
            outSt.fields.push_back({SimdVal::Kind::Vector, i32Ty(), hi, {}, {}});
            setMapped(call.getResult(), outSt);
            return true;
        }
        case 130: { // LegacyF32ToF16(val) -> i32
            mlir::Value f16v = mlir::arith::TruncFOp::create(b_, loc, vecTy(b_.getF16Type()), argV(1));
            mlir::Value i16v = mlir::arith::BitcastOp::create(b_, loc, vecTy(b_.getI16Type()), f16v);
            return store(mlir::arith::ExtUIOp::create(b_, loc, vecTy(i32Ty()), i16v));
        }
        case 131: { // LegacyF16ToF32(val) -> f32
            mlir::Value i16v = mlir::arith::TruncIOp::create(b_, loc, vecTy(b_.getI16Type()), argV(1));
            mlir::Value f16v = mlir::arith::BitcastOp::create(b_, loc, vecTy(b_.getF16Type()), i16v);
            return store(mlir::arith::ExtFOp::create(b_, loc, vecTy(f32Ty()), f16v));
        }
        case 162: { // Dot2AddHalf(acc, ax, ay, bx, by) -> f32
            mlir::Value acc = argV(1);
            mlir::Value ax = mlir::arith::ExtFOp::create(b_, loc, vecTy(f32Ty()), argV(2));
            mlir::Value ay = mlir::arith::ExtFOp::create(b_, loc, vecTy(f32Ty()), argV(3));
            mlir::Value bx = mlir::arith::ExtFOp::create(b_, loc, vecTy(f32Ty()), argV(4));
            mlir::Value by = mlir::arith::ExtFOp::create(b_, loc, vecTy(f32Ty()), argV(5));
            mlir::Value px = mlir::arith::MulFOp::create(b_, loc, ax, bx);
            mlir::Value py = mlir::arith::MulFOp::create(b_, loc, ay, by);
            mlir::Value sum = mlir::arith::AddFOp::create(b_, loc, acc, mlir::arith::AddFOp::create(b_, loc, px, py));
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
                    ai = mlir::arith::ShRSIOp::create(b_, loc, mlir::arith::ShLIOp::create(b_, loc, a, shlAmt), c24);
                    bi = mlir::arith::ShRSIOp::create(b_, loc, mlir::arith::ShLIOp::create(b_, loc, b, shlAmt), c24);
                } else {
                    mlir::Value shrAmt = splatInt(loc, i32Ty(), i * 8);
                    ai = mlir::arith::AndIOp::create(b_, loc, mlir::arith::ShRUIOp::create(b_, loc, a, shrAmt), maskFF);
                    bi = mlir::arith::AndIOp::create(b_, loc, mlir::arith::ShRUIOp::create(b_, loc, b, shrAmt), maskFF);
                }
                acc = mlir::arith::AddIOp::create(b_, loc, acc, mlir::arith::MulIOp::create(b_, loc, ai, bi));
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
                    comp = mlir::arith::ShRSIOp::create(b_, loc, mlir::arith::ShLIOp::create(b_, loc, pk, shlAmt), c24);
                } else { // Unsigned
                    mlir::Value shrAmt = splatInt(loc, i32Ty(), i * 8);
                    comp = mlir::arith::AndIOp::create(b_, loc, mlir::arith::ShRUIOp::create(b_, loc, pk, shrAmt), maskFF);
                }
                if (needTrunc) {
                    comp = mlir::arith::TruncIOp::create(b_, loc, vecTy(elemTy), comp);
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
                    c = mlir::arith::ExtSIOp::create(b_, loc, vecTy(i32Ty()), c);
                }
                if (mode == 1) { // U8Clamp [0, 255]
                    c = mlir::arith::MaxSIOp::create(b_, loc, splatInt(loc, i32Ty(), 0),
                                                     mlir::arith::MinSIOp::create(b_, loc, splatInt(loc, i32Ty(), 255), c));
                } else if (mode == 2) { // S8Clamp [-128, 127]
                    c = mlir::arith::MaxSIOp::create(b_, loc, splatInt(loc, i32Ty(), -128),
                                                     mlir::arith::MinSIOp::create(b_, loc, splatInt(loc, i32Ty(), 127), c));
                }
                c = mlir::arith::AndIOp::create(b_, loc, c, maskFF);
                if (i > 0) {
                    c = mlir::arith::ShLIOp::create(b_, loc, c, splatInt(loc, i32Ty(), i * 8));
                }
                acc = mlir::arith::OrIOp::create(b_, loc, acc, c);
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
    llvm::SmallVector<int32_t, 16> iota;
    for (uint32_t i = 0; i < W; ++i) iota.push_back(static_cast<int32_t>(i));
    mlir::Value iotaVec = mlir::arith::ConstantOp::create(
        b, loc, mlir::DenseElementsAttr::get(vI32, llvm::ArrayRef<int32_t>(iota)));
    mlir::Value w32 = mlir::arith::IndexCastOp::create(b, loc, i32Ty, waveIdx);
    mlir::Value base = mlir::arith::MulIOp::create(
        b, loc, w32, mlir::arith::ConstantIntOp::create(b, loc, i32Ty, W));
    mlir::Value baseVec = mlir::vector::BroadcastOp::create(b, loc, vI32, base);
    mlir::Value flatTid = mlir::arith::AddIOp::create(b, loc, baseVec, iotaVec);
    mlir::Value totVec = mlir::vector::BroadcastOp::create(b, loc, vI32, totalThreads);
    outActiveMask = mlir::arith::CmpIOp::create(
        b, loc, mlir::arith::CmpIPredicate::ult, flatTid, totVec);
    return flatTid;
}

static void setupComputeThreadIds(
    mlir::OpBuilder& b, mlir::Location loc, uint32_t W, const uint32_t tgs[3],
    mlir::Value flatTid, mlir::Value gx, mlir::Value gy, mlir::Value gz,
    StageHooks& hooks) {
    mlir::Type i32Ty = b.getI32Type();
    mlir::VectorType vI32 = mlir::VectorType::get({W}, i32Ty);
    auto splat = [&](uint32_t val) {
        return mlir::arith::ConstantOp::create(
            b, loc, mlir::DenseElementsAttr::get(vI32, b.getIntegerAttr(i32Ty, val)));
    };
    mlir::Value tgsX = splat(tgs[0]), tgsY = splat(tgs[1]), tgsZ = splat(tgs[2]);
    mlir::Value xy = splat(tgs[0] * tgs[1]);

    mlir::Value lx = mlir::arith::RemUIOp::create(b, loc, flatTid, tgsX);
    mlir::Value qx = mlir::arith::DivUIOp::create(b, loc, flatTid, tgsX);
    mlir::Value ly = mlir::arith::RemUIOp::create(b, loc, qx, tgsY);
    mlir::Value lz = mlir::arith::DivUIOp::create(b, loc, flatTid, xy);

    mlir::Value gxVec = mlir::vector::BroadcastOp::create(b, loc, vI32, gx);
    mlir::Value gyVec = mlir::vector::BroadcastOp::create(b, loc, vI32, gy);
    mlir::Value gzVec = mlir::vector::BroadcastOp::create(b, loc, vI32, gz);

    hooks.groupId = {gxVec, gyVec, gzVec};
    hooks.threadIdInGroup = {lx, ly, lz};
    hooks.flatGroupId = flatTid;
    hooks.threadId = {
        mlir::arith::AddIOp::create(b, loc, mlir::arith::MulIOp::create(b, loc, gxVec, tgsX), lx),
        mlir::arith::AddIOp::create(b, loc, mlir::arith::MulIOp::create(b, loc, gyVec, tgsY), ly),
        mlir::arith::AddIOp::create(b, loc, mlir::arith::MulIOp::create(b, loc, gzVec, tgsZ), lz)};
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
        mlir::Value n32 = mlir::arith::ConstantIntOp::create(eb, loc, i32Ty, numWavesVal);
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
        mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, loc, i32Ty, waveIdx);
        mlir::Value wp = mlir::LLVM::GEPOp::create(wb, loc, ptrTy, slot.packedValTy, slot.arrPtr, mlir::ValueRange{w32});
        mlir::LLVM::StoreOp::create(wb, loc, valToStore, wp, 1u);
    };

    auto loadSpillAtWave = [&](mlir::OpBuilder& wb, SimdPacketizer& pkt, const SpillSlot& slot,
                               mlir::Value waveIdx) -> SimdVal {
        mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, loc, i32Ty, waveIdx);
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

    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, loc, 0);
    mlir::Value c1 = mlir::arith::ConstantIndexOp::create(b, loc, 1);

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
                mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, wloc, i32Ty, waveIdx);
                mlir::Value mp = mlir::LLVM::GEPOp::create(
                    wb, wloc, ptrTy, mlir::VectorType::get({W}, wb.getI1Type()), waveMaskArr, mlir::ValueRange{w32});
                mlir::Value activeMask = mlir::LLVM::LoadOp::create(
                    wb, wloc, mlir::VectorType::get({W}, wb.getI1Type()), mp, 1u);

                StageHooks hooks;
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
                mlir::Value condScalar = mlir::vector::ExtractOp::create(b, loc, condPkt.asVec(loc, cVal), 0);
                for (mlir::Value r : ifOp.getResults()) spills[r] = allocSpillSlot(r.getType());

                mlir::scf::IfOp::create(
                    b, loc, condScalar,
                    [&](mlir::OpBuilder& ib, mlir::Location iloc) {
                        if (!packetizeComputeBlockWithBarriers(
                                ib, iloc, mod, ifOp.getThenRegion().front(), shader, rootLayout, W, numWavesVal,
                                rootCtxPtr, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills,
                                entryAllocaAnchor, outError)) {
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
                                    entryAllocaAnchor, outError)) {
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
                        mlir::arith::ConstantIntOp::create(eb, loc, i32Ty, numWavesVal),
                        std::max<uint32_t>(64, W * 8));
                }
                mlir::scf::ForOp::create(
                    b, loc, c0, numWaves, c1, mlir::ValueRange{},
                    [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                        mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, wloc, i32Ty, waveIdx);
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
                                gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills, entryAllocaAnchor, outError)) {
                            mlir::Value cFalse = mlir::arith::ConstantIntOp::create(nb, nloc, nb.getI1Type(), 0);
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

                        mlir::Value cFalse = mlir::arith::ConstantIntOp::create(nb, nloc, nb.getI1Type(), 0);
                        auto anyLoop = mlir::scf::ForOp::create(
                            nb, nloc, c0, numWaves, c1, mlir::ValueRange{cFalse},
                            [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange acc) {
                                mlir::Value baseMask;
                                mlir::Value flatTid = buildFlatTidAndMask(wb, wloc, W, waveIdx, totalThreads, baseMask);
                                mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, wloc, i32Ty, waveIdx);
                                mlir::Value mp = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                                mlir::Value curMask = mlir::LLVM::LoadOp::create(wb, wloc, maskVecTy, mp, 1u);

                                StageHooks hooks;
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
                                mlir::Value nextMask = mlir::arith::AndIOp::create(wb, wloc, curMask, condVec);
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
                                mlir::Value nextAny = mlir::arith::OrIOp::create(wb, wloc, acc[0], waveAny);
                                mlir::scf::YieldOp::create(wb, wloc, mlir::ValueRange{nextAny});
                            });
                        mlir::scf::ConditionOp::create(nb, nloc, anyLoop.getResult(0), mlir::ValueRange{});
                    },
                    [&](mlir::OpBuilder& nb, mlir::Location nloc, mlir::ValueRange) {
                        if (!packetizeComputeBlockWithBarriers(
                                nb, nloc, mod, srcAfter, shader, rootLayout, W, numWavesVal, rootCtxPtr, gsBase,
                                gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills, entryAllocaAnchor, outError)) {
                            mlir::scf::YieldOp::create(nb, nloc);
                            return;
                        }
                        if (auto yo = llvm::dyn_cast_or_null<mlir::scf::YieldOp>(srcAfter.getTerminator())) {
                            mlir::scf::ForOp::create(
                                nb, nloc, c0, numWaves, c1, mlir::ValueRange{},
                                [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                                    StageHooks dh;
                                    SimdPacketizer pkt(wb, mod, shader, rootLayout, W, rootCtxPtr, dh);
                                    mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, wloc, i32Ty, waveIdx);
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
                        mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, wloc, i32Ty, waveIdx);
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
    mlir::Block* entry = dstFunc.addEntryBlock();
    b.setInsertionPointToStart(entry);

    mlir::Value rootCtx = entry->getArgument(0);
    mlir::Value gsBase = entry->getArgument(1);
    mlir::Value gx = entry->getArgument(2);
    mlir::Value gy = entry->getArgument(3);
    mlir::Value gz = entry->getArgument(4);

    const uint32_t totalThreadsVal = shader.threadGroupSize[0] * shader.threadGroupSize[1] * shader.threadGroupSize[2];
    const uint32_t numWavesVal = (totalThreadsVal + simdWidth - 1) / simdWidth;
    mlir::Value totalThreads = mlir::arith::ConstantIntOp::create(b, loc, i32Ty, totalThreadsVal);
    mlir::Value c0 = mlir::arith::ConstantIndexOp::create(b, loc, 0);
    mlir::Value c1 = mlir::arith::ConstantIndexOp::create(b, loc, 1);
    mlir::Value numWaves = mlir::arith::ConstantIndexOp::create(b, loc, numWavesVal);

    if (!regionContainsBarrier(srcFunc.getBody())) {
        mlir::scf::ForOp::create(
            b, loc, c0, numWaves, c1, mlir::ValueRange{},
            [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                mlir::Value activeMask;
                mlir::Value flatTid = buildFlatTidAndMask(wb, wloc, simdWidth, waveIdx, totalThreads, activeMask);
                StageHooks hooks;
                hooks.groupSharedBase = gsBase;
                hooks.waveIdx = waveIdx;
                hooks.numWaves = numWavesVal;
                setupComputeThreadIds(wb, wloc, simdWidth, shader.threadGroupSize, flatTid, gx, gy, gz, hooks);
                SimdPacketizer pkt(wb, mod, shader, rootLayout, simdWidth, rootCtx, hooks);
                mlir::Value realMask = activeMask;
                (void)pkt.packetizeBlock(srcFunc.getBody().front(), activeMask, realMask, outError);
                mlir::scf::YieldOp::create(wb, wloc);
            });
    } else {
        mlir::VectorType maskVecTy = mlir::VectorType::get({simdWidth}, b.getI1Type());
        mlir::Value waveMaskArr = mlir::LLVM::AllocaOp::create(
            b, loc, ptrTy, maskVecTy, mlir::arith::ConstantIntOp::create(b, loc, i32Ty, numWavesVal),
            std::max<uint32_t>(64, simdWidth * 8));
        mlir::Operation* entryAllocaAnchor = waveMaskArr.getDefiningOp();
        // Initialize per-wave active masks
        mlir::scf::ForOp::create(
            b, loc, c0, numWaves, c1, mlir::ValueRange{},
            [&](mlir::OpBuilder& wb, mlir::Location wloc, mlir::Value waveIdx, mlir::ValueRange) {
                mlir::Value activeMask;
                (void)buildFlatTidAndMask(wb, wloc, simdWidth, waveIdx, totalThreads, activeMask);
                mlir::Value w32 = mlir::arith::IndexCastOp::create(wb, wloc, i32Ty, waveIdx);
                mlir::Value mp = mlir::LLVM::GEPOp::create(wb, wloc, ptrTy, maskVecTy, waveMaskArr, mlir::ValueRange{w32});
                mlir::LLVM::StoreOp::create(wb, wloc, activeMask, mp, 1u);
                mlir::scf::YieldOp::create(wb, wloc);
            });

        llvm::DenseMap<mlir::Value, SpillSlot> spills;
        (void)packetizeComputeBlockWithBarriers(
            b, loc, mod, srcFunc.getBody().front(), shader, rootLayout, simdWidth, numWavesVal,
            rootCtx, gsBase, gx, gy, gz, totalThreads, numWaves, waveMaskArr, spills, entryAllocaAnchor, outError);
    }

    if (!outError.empty()) return false;
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

} // namespace nacelle
