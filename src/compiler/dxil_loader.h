#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "shader_compiler.h"

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace nacelle {

struct DxilResourceRecord {
    uint32_t resClass = 0;         // 0=SRV, 1=UAV, 2=CBV, 3=Sampler
    uint32_t rangeId = 0;
    uint32_t space = 0;
    uint32_t lowerBound = 0;
    uint32_t rangeSize = 1;        // UINT32_MAX if unbounded
    uint32_t resourceKind = 0;     // DXIL ResourceKind (1=Tex1D, 2=Tex2D, ..., 10=TypedBuffer, 11=RawBuffer, 12=StructuredBuffer, 13=CBuffer, 14=Sampler)
    uint32_t structuredStride = 0; // Bytes for StructuredBuffer, byte size for CBuffer
};

struct DxilSigRecord {
    uint32_t sigId = 0;
    std::string semanticName;
    uint32_t semanticIndex = 0;
    uint32_t systemValueType = 0; // DXIL SemanticKind (0=Arbitrary, 1=VertexID, 2=InstanceID, 3=Position, 8=IsFrontFace, 16=Target, 17=Depth, ...)
    uint32_t componentType = 9;   // DXIL ComponentType (1=I1, 4=I32, 5=U32, 9=F32)
    uint32_t interpMode = 2;      // 1=Constant(nointerpolation), 2=Linear, 4=LinearNoperspective
    uint32_t rows = 1;
    uint32_t cols = 4;
    uint32_t startRow = 0;
    uint32_t startCol = 0;
};

struct DxilLoadedShader {
    std::unique_ptr<llvm::Module> module;
    std::string entryName;
    std::string stage; // "cs", "vs", "ps"
    uint32_t smMajor = 6;
    uint32_t smMinor = 0;
    uint32_t threadGroupSize[3]{1, 1, 1};
    uint32_t groupSharedBytes = 0;
    uint8_t shaderHash[16]{};

    std::vector<DxilResourceRecord> resources;
    std::vector<DxilSigRecord> inputSigs;
    std::vector<DxilSigRecord> outputSigs;

    bool hasBarrier = false;
    bool hasDerivativesOrImplicitSample = false;
    bool usesDiscard = false;
    bool writesDepth = false;
    bool hasSideEffects = false;

    RootSignatureLayout embeddedRootSig;
    bool hasEmbeddedRootSig = false;

    const DxilResourceRecord* findResource(uint32_t resClass, uint32_t rangeId) const;
    const DxilSigRecord* findInputSig(uint32_t sigId) const;
    const DxilSigRecord* findOutputSig(uint32_t sigId) const;
};

/// Returns the host TargetTriple + DataLayout string cached from LLJIT host detection.
const std::string& getHostDataLayoutString();
const std::string& getHostTargetTripleString();

/// Unpacks a D3D12_SHADER_BYTECODE DXContainer or raw DXIL bitcode into upgraded LLVM 23 IR.
bool loadAndUpgradeDXIL(
    const D3D12_SHADER_BYTECODE& bytecode,
    llvm::LLVMContext& llvmCtx,
    DxilLoadedShader& outShader,
    std::string& outError);

} // namespace nacelle
