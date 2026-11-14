#pragma once

#include "../common/platform.h"
#include "../api/d3d12_types.h"

#include <BS_thread_pool.hpp>
#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nacelle {

using DxrShaderFn = void (*)(
    const DriverRootContext* rootCtx,
    uint32_t activeMask,
    const uint32_t* rayIdxX,
    const uint32_t* rayIdxY,
    const uint32_t* rayIdxZ,
    uint32_t dimX,
    uint32_t dimY,
    uint32_t dimZ,
    void* payloadPtr,
    void* attrPtr);

struct CompiledDxrShader {
    std::string exportName;
    uint32_t shaderKind = 0; // 7=RayGen, 8=Intersection, 9=AnyHit, 10=ClosestHit, 11=Miss, 12=Callable
    DxrShaderFn fn = nullptr;
};

struct CompiledDxrHitGroup {
    std::string exportName;
    uint32_t closestHitIdx = UINT32_MAX;
    uint32_t anyHitIdx = UINT32_MAX;
    uint32_t intersectionIdx = UINT32_MAX;
};

struct NacelleShaderIdentifier {
    static constexpr uint64_t kMagic = 0x4E4143454C4C4552ULL; // 'NACELLER'
    uint64_t magic = kMagic;
    uint64_t stateObjPtr = 0;
    uint32_t entryIndex = 0;
    uint32_t isHitGroup = 0;
    uint8_t reserved[8]{};
};
static_assert(sizeof(NacelleShaderIdentifier) == D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES,
              "Shader identifier must match D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES (32 bytes)");

struct CompiledDxrStateObject {
    std::vector<CompiledDxrShader> shaders;
    std::vector<CompiledDxrHitGroup> hitGroups;
    std::unordered_map<std::string, NacelleShaderIdentifier> identifiers;
    uint32_t maxRecursionDepth = 1;
    uint32_t maxPayloadBytes = 64;
    uint32_t maxAttrBytes = 32;
    uint32_t simdWidth = 4;
};

void executeDispatchRays(
    BS::thread_pool<>& pool,
    const D3D12_DISPATCH_RAYS_DESC& desc,
    const DriverRootContext& computeCtx,
    const CompiledDxrStateObject& stateObj);

} // namespace nacelle

extern "C" {
void nacelle_dxr_trace_ray(
    const void* tlasPtr,
    int32_t rayFlags,
    int32_t instanceMask,
    int32_t rayContributionToHitGroupIndex,
    int32_t multiplierForGeometryContribution,
    int32_t missShaderIndex,
    float ox,
    float oy,
    float oz,
    float tMin,
    float dx,
    float dy,
    float dz,
    float tMax,
    void* payloadPtr);
int32_t nacelle_dxr_state_i32(int32_t op);
float nacelle_dxr_state_f32(int32_t op);
float nacelle_dxr_state_vec(int32_t op, int32_t comp);
float nacelle_dxr_state_mat(int32_t op, int32_t row, int32_t col);
void nacelle_dxr_ignore_hit();
void nacelle_dxr_accept_hit_and_end_search();
int32_t nacelle_dxr_report_hit(float tHit, int32_t hitKind, const void* attrPtr, int32_t attrBytes);
void nacelle_dxr_call_shader(int32_t shaderIndex, void* paramPtr);
}
