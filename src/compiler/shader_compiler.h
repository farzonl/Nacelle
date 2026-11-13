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
    const uint32_t* quadOffsets = nullptr; // [W/4]: pixel base index (quadIdx * 4) inside 64x64 tile SOA
    const float* pixelX = nullptr;      // [W]
    const float* pixelY = nullptr;      // [W]
    const float* baryI = nullptr;       // [W]
    const float* baryJ = nullptr;       // [W]
    const float* triDepth = nullptr;    // [3]: (z0, z1 - z0, z2 - z0)
    uint32_t coverageMask = 0;
    uint32_t isFrontFace = 1;
    float* tileColorSOA[8]{};           // Each points to 4 * 4096 floats: plane c * 4096 + pixIdx
    float* tileDepthSOA = nullptr;      // 4096 floats
    uint8_t* tileStencilSOA = nullptr;  // 4096 uint8_t
};

using CompiledPixelQuadFn = void (*)(const PixelBatchArgs* args);

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
    uint32_t simdWidth = 4;
    uint32_t numVSOutputRegs = 0;      // Including SV_Position
    int32_t posOutputReg = 0;          // Register index of SV_Position in OSGN
    uint32_t numPSInputRegs = 0;       // Max PS input register count
    uint32_t flatInterpMask = 0;       // Bitmask of VS->PS output regs using nointerpolation
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

/// Compiles a Graphics PSO (VS SIMD batch + Pixel Uber-PSO).
bool compileGraphicsPSO(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    CompiledGraphicsPSO& outPSO,
    std::string& outError);

/// Waits for any pending background Tier-2 (-O3) JIT optimizations (useful for deterministic teardown/testing).
void waitForAllAsyncJITTasks();

} // namespace nacelle
