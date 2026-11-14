#pragma once

#include <d3d12.h>
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
