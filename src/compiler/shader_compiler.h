#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../api/d3d12_types.h"

namespace nacelle {

/// Signature element metadata extracted from DXContainer ISGN/OSGN.
struct ShaderSigElement {
    std::string semanticName;
    uint32_t semanticIndex = 0;
    uint32_t systemValueType = 0; // D3D_NAME / dxbc::SemanticKind
    uint32_t componentType = 0;   // 1=uint32, 2=sint32, 3=float32
    uint32_t registerIndex = 0;
    uint8_t mask = 0;             // 4-bit component mask
    uint8_t rwMask = 0;
    uint8_t stream = 0;
};

/// Resource binding entry in DXIL !dx.resources mapped to a Root Signature slot.
struct RootBindingTarget {
    enum Kind : uint8_t {
        Invalid = 0,
        RootConstants,
        RootDescriptor,
        DescriptorTableOffset,
        StaticSampler
    };
    Kind kind = Invalid;
    uint32_t rootParamIndex = 0;
    uint32_t tableOffsetOrConstStart = 0; // Descriptor offset in table, or DWORD offset in root constants
    uint32_t staticSamplerIndex = 0;
};

/// Root Signature layout metadata used by the compiler to resolve resource handles.
struct RootSignatureLayout {
    struct Param {
        D3D12_ROOT_PARAMETER_TYPE type = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL;
        uint32_t shaderRegister = 0;
        uint32_t registerSpace = 0;
        uint32_t num32BitValues = 0;
        uint32_t constDwordOffset = 0;
        struct Range {
            D3D12_DESCRIPTOR_RANGE_TYPE type = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
            uint32_t numDescriptors = 1;
            uint32_t baseShaderRegister = 0;
            uint32_t registerSpace = 0;
            uint32_t offsetInTable = 0;
        };
        std::vector<Range> ranges;
    };
    struct StaticSampler {
        uint32_t shaderRegister = 0;
        uint32_t registerSpace = 0;
        D3D12_SHADER_VISIBILITY visibility = D3D12_SHADER_VISIBILITY_ALL;
        D3D12_SAMPLER_DESC desc{};
    };
    std::vector<Param> params;
    std::vector<StaticSampler> staticSamplers;
    uint64_t hashLo = 0;
    uint64_t hashHi = 0;

    RootBindingTarget resolveBinding(uint32_t resClass /*0=SRV,1=UAV,2=CBV,3=Sampler*/,
                                     uint32_t space,
                                     uint32_t reg) const;
};

/// Compute kernel entry signature.
using CompiledComputeFn = void (*)(
    const DriverRootContext* rootCtx,
    uint8_t* groupSharedScratch,
    uint32_t groupX, uint32_t groupY, uint32_t groupZ,
    uint32_t numGroupsX, uint32_t numGroupsY, uint32_t numGroupsZ);

/// Vertex shader SIMD batch kernel signature.
using CompiledVertexFn = void (*)(
    const DriverRootContext* rootCtx,
    const uint32_t* vertexIds,
    const uint32_t* instanceIds,
    uint32_t activeCount,
    float* outPosXYZW,  // [4 * W] SOA: comp * W + lane
    float* outAttrs);   // [numOutputScalars * W] SOA

/// Pixel Uber-PSO quad batch kernel arguments.
struct PixelBatchArgs {
    const DriverRootContext* rootCtx = nullptr;
    const float* triAttrBase = nullptr; // [numPSInputRegs * 4 * 3]: for each scalar (a0/w0, a1/w1 - a0/w0, a2/w2 - a0/w0)
    const float* triAttrFlat = nullptr; // [numPSInputRegs * 4]: un-divided v0 attribute for flat/nointerpolation
    const float* triInvW = nullptr;     // [3]: (1/w0, 1/w1 - 1/w0, 1/w2 - 1/w0)
    const float* triOrigBary = nullptr; // [9]: for k in 0..2, (b0/w0, b1/w1 - b0/w0, b2/w2 - b0/w0)
    const uint32_t* quadOffsets = nullptr; // [W/4]: pixel base index (quadIdx * 4) inside 64x64 tile SOA
    const float* pixelX = nullptr;      // [W]
    const float* pixelY = nullptr;      // [W]
    const float* baryI = nullptr;       // [W]
    const float* baryJ = nullptr;       // [W]
    const float* triDepth = nullptr;    // [3]: (z0, z1 - z0, z2 - z0)
    uint32_t coverageMask = 0;
    uint32_t isFrontFace = 1;
    uint32_t primitiveId = 0;
    uint32_t viewportArrayIndex = 0;
    uint32_t shadingRate = 0;
    float* tileColorSOA[8]{};           // Each points to 4 * 4096 floats: plane c * 4096 + pixIdx
    float* tileDepthSOA = nullptr;      // 4096 floats
    uint8_t* tileStencilSOA = nullptr;  // 4096 uint8_t
};

using CompiledPixelQuadFn = void (*)(const PixelBatchArgs* args);

/// Geometry Uber-PSO kernel: executes one input primitive (with lane 0 active).
using CompiledGeometryFn = void (*)(
    const DriverRootContext* rootCtx,
    const float* inVertPosXYZW,   // [numInVerts * 4]
    const float* inVertAttrs,     // [numInVerts * numGSInputScalars]
    uint32_t primId,
    uint32_t gsInstanceId,
    float* outVertPosXYZW,        // [maxVertexCount * 4]
    float* outVertAttrs,          // [maxVertexCount * numOutScalars]
    uint32_t* outStripVertCounts, // [maxVertexCount]
    uint32_t* outTotalVerts,      // [1]
    uint32_t* outNumStrips);      // [1]

/// Hull shader control-point kernel: executes one output control point (lane 0 active).
using CompiledHullCPFn = void (*)(
    const DriverRootContext* rootCtx,
    const float* inCPAttrs,  // [hsInputCPCount * numHSInputScalars]
    uint32_t patchId,
    uint32_t outCpId,
    float* outCPAttrs);      // [numHSOutputScalars]

/// Hull shader patch-constant kernel: executes once per patch (lane 0 active).
using CompiledHullPCFn = void (*)(
    const DriverRootContext* rootCtx,
    const float* inCPAttrs,   // [hsInputCPCount * numHSInputScalars]
    const float* outCPAttrs,  // [hsOutputCPCount * numHSOutputScalars]
    uint32_t patchId,
    float* outPatchConsts,    // [numPatchConstScalars]
    float* outTessFactors);   // [6]: 4 edge + 2 inside

/// Domain shader kernel: executes one tessellated vertex (lane 0 active).
using CompiledDomainFn = void (*)(
    const DriverRootContext* rootCtx,
    const float* inCPAttrs,     // [hsOutputCPCount * numHSOutputScalars]
    const float* inPatchConsts, // [numPatchConstScalars]
    const float* inTessFactors, // [6]
    uint32_t patchId,
    float domU,
    float domV,
    float domW,
    float* outPosXYZW,          // [4]
    float* outAttrs);           // [numDSOutScalars]

/// Amplification shader threadgroup kernel arguments.
struct AmplificationThreadGroupArgs {
    const DriverRootContext* rootCtx = nullptr;
    uint8_t* groupSharedScratch = nullptr;
    uint32_t gx = 0;
    uint32_t gy = 0;
    uint32_t gz = 0;
    uint32_t ngx = 1;
    uint32_t ngy = 1;
    uint32_t ngz = 1;
    uint32_t* outDispatchDims = nullptr; // [4]: countX, countY, countZ, dispatchedFlag
    void* outPayload = nullptr;
};

/// Mesh shader threadgroup kernel arguments.
struct MeshThreadGroupArgs {
    const DriverRootContext* rootCtx = nullptr;
    uint8_t* groupSharedScratch = nullptr;
    const void* payloadPtr = nullptr;
    uint32_t gx = 0;
    uint32_t gy = 0;
    uint32_t gz = 0;
    uint32_t ngx = 1;
    uint32_t ngy = 1;
    uint32_t ngz = 1;
    uint32_t* outCounts = nullptr;      // [2]: numVertices, numPrimitives
    float* outVertPosXYZW = nullptr;    // [maxVerts * 4]
    float* outVertAttrs = nullptr;      // [maxVerts * totalVSScalarAttrs]
    uint32_t* outPrimIndices = nullptr; // [maxPrims * 3]
    float* outPrimAttrs = nullptr;      // [maxPrims * totalVSScalarAttrs]
    uint8_t* outPrimCull = nullptr;     // [maxPrims]
    uint32_t* outPrimId = nullptr;      // [maxPrims]
};

using CompiledAmplificationFn = void (*)(const AmplificationThreadGroupArgs* args);
using CompiledMeshFn = void (*)(const MeshThreadGroupArgs* args);

/// Optional Amplification & Mesh shader bytecodes for pipeline state stream PSOs.
struct MeshShaderStages {
    D3D12_SHADER_BYTECODE AS{};
    D3D12_SHADER_BYTECODE MS{};
};

struct PsoUpdateBox;

/// Compiled Compute PSO artifact.
struct CompiledComputePSO {
    CompiledComputePSO() = default;
    CompiledComputePSO(const CompiledComputePSO&) = delete;
    CompiledComputePSO& operator=(const CompiledComputePSO&) = delete;
    CompiledComputePSO(CompiledComputePSO&&) = delete;
    CompiledComputePSO& operator=(CompiledComputePSO&&) = delete;
    ~CompiledComputePSO();

    std::atomic<CompiledComputeFn> fn{nullptr};
    uint32_t threadGroupSize[3]{1, 1, 1};
    uint32_t groupSharedBytes = 0;
    uint32_t simdWidth = 4;
    uint64_t cacheHashLo = 0;
    uint64_t cacheHashHi = 0;
    std::string symbolName;
    std::shared_ptr<PsoUpdateBox> asyncBox;
};

/// Compiled Graphics PSO artifact.
struct CompiledGraphicsPSO {
    CompiledGraphicsPSO() = default;
    CompiledGraphicsPSO(const CompiledGraphicsPSO&) = delete;
    CompiledGraphicsPSO& operator=(const CompiledGraphicsPSO&) = delete;
    CompiledGraphicsPSO(CompiledGraphicsPSO&&) = delete;
    CompiledGraphicsPSO& operator=(CompiledGraphicsPSO&&) = delete;
    ~CompiledGraphicsPSO();

    std::atomic<CompiledVertexFn> vsFn{nullptr};
    std::atomic<CompiledPixelQuadFn> psFn{nullptr};
    std::atomic<CompiledGeometryFn> gsFn{nullptr};
    std::atomic<CompiledHullCPFn> hsFn{nullptr};
    std::atomic<CompiledHullPCFn> hsPcFn{nullptr};
    std::atomic<CompiledDomainFn> dsFn{nullptr};
    std::atomic<CompiledAmplificationFn> asFn{nullptr};
    std::atomic<CompiledMeshFn> msFn{nullptr};
    bool hasGS = false;
    uint32_t gsMaxVertexCount = 0;
    uint32_t gsOutputTopology = 5;
    uint32_t gsInstanceCount = 1;
    uint32_t numGSInputScalars = 0;
    bool hasTess = false;
    uint32_t hsInputCPCount = 0;
    uint32_t hsOutputCPCount = 0;
    uint32_t tessDomain = 0;
    uint32_t tessPartitioning = 0;
    uint32_t tessOutputPrimitive = 0;
    float hsMaxTessFactor = 64.0f;
    uint32_t numHSInputScalars = 0;
    uint32_t numHSOutputScalars = 0;
    uint32_t numPatchConstScalars = 0;
    bool hasMS = false;
    bool hasAS = false;
    uint32_t msThreadGroupSize[3]{1, 1, 1};
    uint32_t msGroupSharedBytes = 0;
    uint32_t msMaxVertexCount = 0;
    uint32_t msMaxPrimitiveCount = 0;
    uint32_t msOutputTopology = 2;     // 1=line, 2=triangle
    uint32_t msPayloadBytes = 0;
    uint64_t msPrimAttrMask = 0;       // Bitmask of scalar slots (0..63) written per-primitive
    uint32_t asThreadGroupSize[3]{1, 1, 1};
    uint32_t asGroupSharedBytes = 0;
    uint32_t asPayloadBytes = 0;
    std::string gsSymbolName;
    uint32_t simdWidth = 4;
    uint32_t numVSOutputRegs = 0;      // Including SV_Position
    int32_t posOutputReg = 0;          // Register index of SV_Position in OSGN
    uint32_t numPSInputRegs = 0;       // Max PS input register count
    uint32_t flatInterpMask = 0;       // Bitmask of VS->PS output regs using nointerpolation
    uint32_t clipDistMask = 0;         // Bitmask of active SV_ClipDistance scalar components (0..7)
    uint32_t cullDistMask = 0;         // Bitmask of active SV_CullDistance scalar components (0..7)
    uint32_t clipAttrOffset = 0;       // Starting scalar slot in outAttrs/attrBaseArena for SV_ClipDistance
    uint32_t cullAttrOffset = 0;       // Starting scalar slot in outAttrs/attrBaseArena for SV_CullDistance
    bool hasViewportArrayIndex = false;
    uint32_t vpIndexAttrOffset = 0;    // Scalar slot in outAttrs/attrBaseArena for SV_ViewportArrayIndex
    bool hasPerPrimShadingRate = false;
    uint32_t shadingRateAttrOffset = 0;// Scalar slot in outAttrs/attrBaseArena for SV_ShadingRate
    uint32_t totalVSScalarAttrs = 0;   // Total scalar attributes emitted by VS into outAttrs
    bool psWritesDepth = false;
    bool psUsesDiscard = false;
    bool psHasSideEffects = false;
    std::vector<ShaderSigElement> vsInputs;
    std::vector<ShaderSigElement> vsOutputs;
    std::vector<ShaderSigElement> psInputs;
    std::vector<ShaderSigElement> psOutputs;
    uint64_t cacheHashLo = 0;
    uint64_t cacheHashHi = 0;
    std::string vsSymbolName;
    std::string psSymbolName;
    std::shared_ptr<PsoUpdateBox> asyncBox;
};

/// Returns the active SIMD lane width (4 on ARM64 NEON, 8 on AVX2, 16 on AVX-512, or NACELLE_SIMD_WIDTH / override).
uint32_t getTargetSimdWidth();
void setSimdWidthOverride(uint32_t width); // 0 restores default / env behavior

/// Compiles a Compute PSO (Tier-1 fast compile or cache hit, optionally scheduling Tier-2 -O3 promotion).
bool compileComputePSO(
    const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    CompiledComputePSO& outPSO,
    std::string& outError);

/// Compiles a Graphics or Mesh PSO (VS SIMD batch or AS/MS threadgroups + Pixel Uber-PSO).
bool compileGraphicsPSO(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    CompiledGraphicsPSO& outPSO,
    std::string& outError,
    const MeshShaderStages& meshStages = {});

/// Waits for any pending background Tier-2 (-O3) JIT optimizations (useful for deterministic teardown/testing).
void waitForAllAsyncJITTasks();

} // namespace nacelle
