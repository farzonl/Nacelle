#pragma once

#include <string>
#include <vector>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "jit_engine.h"
#include "shader_compiler.h"

namespace nacelle {

/// Synthesizes standalone Vertex Stage MLIR (`@<vsSym>`) with canonical VS output layout.
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
    std::string& outError);

/// Synthesizes combined Vertex/Mesh + Pixel Uber-PSO MLIR (`@<vsSym>` / `@<vsSym>_ms` / `@<vsSym>_as` and `@<psSym>`) for a graphics PSO.
/// When `skipVS` is true (for decoupled VS stage cache hits on `!hasMS && !hasTess && !hasGS` PSOs), only `@<psSym>`
/// is emitted into `outCombinedMod` using `precompiledVSMeta` and `precompiledVSOutputRecs`.
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
    const MeshShaderStages& meshStages = {},
    bool skipVS = false,
    const CachedVSMeta* precompiledVSMeta = nullptr,
    const std::vector<DxilSigRecord>* precompiledVSOutputRecs = nullptr);

} // namespace nacelle
