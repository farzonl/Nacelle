#pragma once

#include "../common/platform.h"
#include <cstdint>

namespace nacelle {

void getRaytracingAccelerationStructurePrebuildInfo(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* inputs,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* outInfo);

void buildRaytracingAccelerationStructure(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc);

void copyRaytracingAccelerationStructure(
    D3D12_GPU_VIRTUAL_ADDRESS destAddr,
    D3D12_GPU_VIRTUAL_ADDRESS srcAddr,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE mode);

struct PipelineHitResult {
    bool hit = false;
    bool isProcedural = false;
    float t = 0.0f;
    float baryU = 0.0f;
    float baryV = 0.0f;
    uint32_t hitKind = 254;
    uint32_t instanceIndex = 0;
    uint32_t instanceID = 0;
    uint32_t geometryIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t instanceContributionToHitGroupIndex = 0;
    float objOrigin[3]{};
    float objDir[3]{};
    float objToWorld[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    float worldToObj[12]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
};

// Return values for DxrAnyHitFn: 0 = accept hit, 1 = ignore hit, 2 = accept and end search
using DxrAnyHitFn = int32_t (*)(void* userData, const PipelineHitResult& candidate);
using DxrIntersectionFn = void (*)(
    void* userData,
    const PipelineHitResult& candidate,
    float& inOutCommittedT,
    bool& outAccepted,
    bool& outEndSearch,
    uint32_t& outHitKind,
    float& outBaryU,
    float& outBaryV);

PipelineHitResult traceRayForPipeline(
    const void* tlasPtr,
    uint32_t rayFlags,
    uint32_t instanceMask,
    float ox,
    float oy,
    float oz,
    float tMin,
    float dx,
    float dy,
    float dz,
    float tMax,
    DxrAnyHitFn onAnyHit,
    DxrIntersectionFn onIntersection,
    void* userData);

} // namespace nacelle

extern "C" {
int32_t nacelle_rayquery_alloc(int32_t constRayFlags);
void nacelle_rayquery_trace_inline(
    int32_t handle,
    const void* tlasPtr,
    int32_t rayFlags,
    int32_t instanceMask,
    float ox,
    float oy,
    float oz,
    float tMin,
    float dx,
    float dy,
    float dz,
    float tMax);
int32_t nacelle_rayquery_proceed(int32_t handle);
void nacelle_rayquery_abort(int32_t handle);
void nacelle_rayquery_commit_triangle(int32_t handle);
void nacelle_rayquery_commit_procedural(int32_t handle, float tHit);
int32_t nacelle_rayquery_state_i32(int32_t op, int32_t handle);
float nacelle_rayquery_state_f32(int32_t op, int32_t handle);
float nacelle_rayquery_state_vec(int32_t op, int32_t handle, int32_t comp);
float nacelle_rayquery_state_mat(int32_t op, int32_t handle, int32_t row, int32_t col);
}
