#pragma once

#include <string>
#include <vector>

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/OwningOpRef.h>

#include "jit_engine.h"
#include "shader_compiler.h"

namespace nacelle {

/// Synthesizes combined Vertex/Mesh + Pixel Uber-PSO MLIR (`@<vsSym>` / `@<vsSym>_ms` / `@<vsSym>_as` and `@<psSym>`) for a graphics PSO.
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
    const MeshShaderStages& meshStages = {});

} // namespace nacelle
