#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Module.h>

#include "dxil_loader.h"
#include "shader_compiler.h"
#include "structured_cfg.h"

namespace nacelle {

/// Vector-packetized or aggregate representation of a scalar DXIL SSA value across W lanes.
struct SimdVal {
    enum class Kind : uint8_t {
        Invalid = 0,
        Vector, // `vec` has type <W x ScalarTy>
        Lanes,  // `lanes` has 1 (uniform) or W (divergent) scalar values (used for pointers / structs)
        Struct, // `fields` recursively hold struct members
        Handle  // Resolved DXIL resource/sampler handle
    };

    enum class LanePattern : uint8_t {
        Divergent = 0,
        Uniform,   // All W lanes hold the same value
        UnitStride // Lane l holds base + l (e.g. threadId.x + uniform)
    };

    Kind kind = Kind::Invalid;
    llvm::Type* origType = nullptr;
    llvm::Value* vec = nullptr;
    llvm::SmallVector<llvm::Value*, 16> lanes;
    std::vector<SimdVal> fields;

    // Handle metadata (for Kind::Handle)
    uint32_t resClass = 0;      // 0=SRV, 1=UAV, 2=CBV, 3=Sampler
    uint32_t resourceKind = 0;  // DXIL ResourceKind
    uint32_t stride = 0;
    bool isRootConstants = false;
    bool isAllocaPtr = false;
    LanePattern pattern = LanePattern::Divergent;
    bool isUniform() const {
        return pattern == LanePattern::Uniform ||
               ((kind == Kind::Lanes || kind == Kind::Handle) && lanes.size() == 1);
    }
    bool isUnitStride() const { return pattern == LanePattern::UnitStride; }
    bool valid() const { return kind != Kind::Invalid; }

    static SimdVal makeVector(llvm::Value* v, llvm::Type* origT, LanePattern pat = LanePattern::Divergent) {
        SimdVal sv;
        sv.kind = Kind::Vector;
        sv.origType = origT;
        sv.vec = v;
        sv.pattern = pat;
        return sv;
    }

    static SimdVal makeLanes(llvm::SmallVector<llvm::Value*, 16> l, llvm::Type* origT) {
        SimdVal sv;
        sv.kind = Kind::Lanes;
        sv.origType = origT;
        sv.lanes = std::move(l);
        return sv;
    }
};

/// Stage-specific bindings supplied to the SIMT packetizer (Compute, Vertex, or Pixel).
struct StageHooks {
    bool isPixelShader = false;
    llvm::SmallVector<llvm::Value*, 3> threadId;        // <W x i32>
    llvm::SmallVector<llvm::Value*, 3> dispatchRaysDim; // <W x i32>
    llvm::SmallVector<llvm::Value*, 3> groupId;         // <W x i32>
    llvm::SmallVector<llvm::Value*, 3> threadIdInGroup; // <W x i32>
    llvm::Value* flatGroupId = nullptr;                 // <W x i32>
    llvm::Value* groupSharedBase = nullptr;             // ptr
    llvm::Value* meshPayloadBase = nullptr;             // ptr (for Mesh Shader GetMeshPayload)
    llvm::BasicBlock* entryBlock = nullptr;             // function entry block for hoisted allocas
    llvm::Value* waveIdx = nullptr;                     // i32 (optional, set in multi-wave compute/MS/AS)
    uint32_t numWaves = 1;

    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId,
                       const SimdVal& row, uint32_t col, llvm::Type* scalarTy, SimdVal& out)> loadInput;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId,
                       const SimdVal& row, uint32_t col, const SimdVal& vertexAxis,
                       llvm::Type* scalarTy, SimdVal& out)> loadInputWithVertex;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId,
                       const SimdVal& row, uint32_t col, const SimdVal& val, llvm::Value* realMask)> storeOutput;
    std::function<void(llvm::IRBuilder<>&, llvm::Value* condVec,
                       llvm::Value* execMask, llvm::Value*& realMask)> onDiscard;
    std::function<bool(llvm::IRBuilder<>&, uint32_t streamId, llvm::Value* realMask)> emitStream;
    std::function<bool(llvm::IRBuilder<>&, uint32_t streamId, llvm::Value* realMask)> cutStream;
    std::function<bool(llvm::IRBuilder<>&, llvm::Type* scalarTy, SimdVal& out)> loadPrimitiveId;
    std::function<bool(llvm::IRBuilder<>&, llvm::Type* scalarTy, SimdVal& out)> loadGSInstanceId;
    std::function<bool(llvm::IRBuilder<>&, llvm::Type* scalarTy, SimdVal& out)> loadOutputControlPointId;
    std::function<bool(llvm::IRBuilder<>&, uint32_t col, llvm::Type* scalarTy, SimdVal& out)> loadDomainLocation;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId,
                       const SimdVal& row, uint32_t col, const SimdVal& cpId,
                       llvm::Type* scalarTy, SimdVal& out)> loadOutputControlPoint;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId,
                       const SimdVal& row, uint32_t col, llvm::Type* scalarTy, SimdVal& out)> loadPatchConstant;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId,
                       const SimdVal& row, uint32_t col, const SimdVal& val, llvm::Value* realMask)> storePatchConstant;
    std::function<bool(llvm::IRBuilder<>&, const SimdVal& countX, const SimdVal& countY,
                       const SimdVal& countZ, const SimdVal& payloadPtr, llvm::Value* realMask)> dispatchMesh;
    std::function<bool(llvm::IRBuilder<>&, const SimdVal& numVerts, const SimdVal& numPrims,
                       llvm::Value* realMask)> setMeshOutputCounts;
    std::function<bool(llvm::IRBuilder<>&, const SimdVal& primIdx, const SimdVal& v0,
                       const SimdVal& v1, const SimdVal& v2, llvm::Value* realMask)> emitIndices;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId, const SimdVal& row, uint32_t col,
                       const SimdVal& val, const SimdVal& vertIdx, llvm::Value* realMask)> storeVertexOutput;
    std::function<bool(llvm::IRBuilder<>&, uint32_t sigId, const SimdVal& row, uint32_t col,
                       const SimdVal& val, const SimdVal& primIdx, llvm::Value* realMask)> storePrimitiveOutput;
};

/// Packetizes a threadgroup-based shader body (Compute, Amplification, or Mesh) across all waves in
/// the threadgroup, supporting barriers, groupshared memory, and stage-specific `baseHooks`.
bool packetizeThreadGroupRegion(
    llvm::IRBuilder<>& builder,
    llvm::Module& mod,
    const SeqNode& srcRegion,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    llvm::Value* rootCtxPtr,
    llvm::Value* gsBase,
    llvm::Value* gx,
    llvm::Value* gy,
    llvm::Value* gz,
    llvm::BasicBlock* entryBlock,
    const StageHooks& baseHooks,
    std::string& outError);

/// Packetizes a lifted Compute entry function into
/// `define void @symbolName(ptr %rootCtx, ptr %gsBase, i32 %gx, i32 %gy, i32 %gz, i32 %ngx, i32 %ngy, i32 %ngz)`.
bool packetizeComputeEntry(
    llvm::Module& mod,
    const StructuredFunction& srcTree,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError);

/// Packetizes a barrier-free shader body (e.g. VS or PS) inside the caller's builder scope,
/// initializing `rootCtxPtr`, `execMask` (`<W x i1>`), and `realMask` (`<W x i1>`).
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
    std::string& outError);

bool packetizeDxrEntry(
    llvm::Module& mod,
    const StructuredFunction& srcTree,
    const DxilLoadedShader& shader,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& symbolName,
    std::string& outError);

} // namespace nacelle
