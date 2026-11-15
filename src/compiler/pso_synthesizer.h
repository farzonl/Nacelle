#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "common/platform.h"
#include "jit_engine.h"
#include "shader_compiler.h"

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace nacelle {

struct CachedVSMeta;
struct DxilSigRecord;

/// Synthesizes a standalone Vertex Stage LLVM module (`vsSym`) for split VS/PS caching.
bool synthesizeVertexStageLLVM(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    llvm::LLVMContext& llvmCtx,
    std::unique_ptr<llvm::Module>& outModule,
    CachedVSMeta& outVSMeta,
    std::vector<ShaderSigElement>& outVSInputs,
    std::vector<ShaderSigElement>& outVSOutputs,
    std::vector<DxilSigRecord>& outVSOutputRecs,
    std::string& outError);

/// Synthesizes a combined Graphics PSO LLVM module containing the entrypoints
/// `vsSym` (or mesh/amplification pipeline) and `psSym`.
bool synthesizeGraphicsPSOLLVM(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
    const RootSignatureLayout& rootLayout,
    uint32_t simdWidth,
    const std::string& vsSym,
    const std::string& psSym,
    llvm::LLVMContext& llvmCtx,
    std::unique_ptr<llvm::Module>& outModule,
    CachedPSOMeta& outMeta,
    std::vector<ShaderSigElement>& outVSInputs,
    std::vector<ShaderSigElement>& outVSOutputs,
    std::vector<ShaderSigElement>& outPSInputs,
    std::vector<ShaderSigElement>& outPSOutputs,
    std::string& outError,
    const MeshShaderStages& meshStages = {},
    bool skipVertexStage = false,
    const CachedVSMeta* precompiledVSMeta = nullptr,
    const std::vector<DxilSigRecord>* precompiledVSOutputRecs = nullptr);

} // namespace nacelle
