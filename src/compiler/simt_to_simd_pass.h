#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Value.h>

#include "dxil_loader.h"

namespace nacelle {

/// Vector-packetized or aggregate representation of a scalar DXIL SSA value across W lanes.
struct SimdVal {
    enum class Kind : uint8_t {
        Invalid = 0,
        Vector, // `vec` has type vector<W x ScalarTy>
        Lanes,  // `lanes` has 1 (uniform) or W (divergent) scalar values (used for pointers / structs)
        Struct, // `fields` recursively hold struct members
        Handle  // Resolved DXIL resource/sampler handle
    };

    Kind kind = Kind::Invalid;
    mlir::Type origType;
    mlir::Value vec;
    llvm::SmallVector<mlir::Value, 16> lanes;
    std::vector<SimdVal> fields;

    // Handle metadata (for Kind::Handle)
    uint32_t resClass = 0;      // 0=SRV, 1=UAV, 2=CBV, 3=Sampler
    uint32_t resourceKind = 0;  // DXIL ResourceKind
    uint32_t stride = 0;
    bool isRootConstants = false;
    bool isUniform() const { return kind == Kind::Lanes && lanes.size() == 1; }
    bool valid() const { return kind != Kind::Invalid; }
};

/// Stage-specific bindings supplied to the SIMT packetizer (Compute, Vertex, or Pixel).
struct StageHooks {
    bool isPixelShader = false;
    llvm::SmallVector<mlir::Value, 3> threadId;        // vector<W x i32>
    llvm::SmallVector<mlir::Value, 3> groupId;         // vector<W x i32>
    llvm::SmallVector<mlir::Value, 3> threadIdInGroup; // vector<W x i32>
    mlir::Value flatGroupId;                           // vector<W x i32>
    mlir::Value groupSharedBase;                       // !llvm.ptr

    std::function<bool(mlir::OpBuilder&, mlir::Location, uint32_t sigId,
                       const SimdVal& row, uint32_t col, mlir::Type scalarTy, SimdVal& out)> loadInput;
    std::function<bool(mlir::OpBuilder&, mlir::Location, uint32_t sigId,
                       const SimdVal& row, uint32_t col, const SimdVal& val, mlir::Value realMask)> storeOutput;
    std::function<void(mlir::OpBuilder&, mlir::Location, mlir::Value condVec,
                       mlir::Value execMask, mlir::Value& realMask)> onDiscard;
};

/// Packetizes a lifted Compute entry function (`func.func @entryName`) into
/// `func.func @nacelle_cs_entry(ptr %rootCtx, ptr %gsBase, i32 %gx, i32 %gy, i32 %gz, i32 %ngx, i32 %ngy, i32 %ngz)`.
bool packetizeComputeEntry(
    mlir::ModuleOp mod,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError);

/// Packetizes a barrier-free shader body (e.g. VS or PS) inside the caller's builder scope,
/// initializing `rootCtxPtr`, `execMask` (`vector<Wxi1>`), and `realMask` (`vector<Wxi1>`).
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
    std::string& outError);

} // namespace nacelle
