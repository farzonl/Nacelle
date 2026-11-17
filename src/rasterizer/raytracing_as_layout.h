#pragma once

#include <stdint.h>

namespace nacelle {

constexpr uint32_t kBlasMagic = 0x424C4153u; // 'BLAS'
constexpr uint32_t kTlasMagic = 0x544C4153u; // 'TLAS'
constexpr uint32_t kInvalidNode = 0xFFFFFFFFu;

// Ray flags (D3D12_RAY_FLAGS / HLSL RAY_FLAG)
constexpr uint32_t kRayFlagForceOpaque = 0x01u;
constexpr uint32_t kRayFlagForceNonOpaque = 0x02u;
constexpr uint32_t kRayFlagAcceptFirstHitAndEndSearch = 0x04u;
constexpr uint32_t kRayFlagCullBackFacingTriangles = 0x10u;
constexpr uint32_t kRayFlagCullFrontFacingTriangles = 0x20u;
constexpr uint32_t kRayFlagCullOpaque = 0x40u;
constexpr uint32_t kRayFlagCullNonOpaque = 0x80u;
constexpr uint32_t kRayFlagSkipTriangles = 0x100u;
constexpr uint32_t kRayFlagSkipProceduralPrimitives = 0x200u;

// D3D12_RAYTRACING_INSTANCE_FLAGS
constexpr uint32_t kInstFlagTriangleCullDisable = 0x1u;
constexpr uint32_t kInstFlagTriangleFrontCCW = 0x2u;
constexpr uint32_t kInstFlagForceOpaque = 0x4u;
constexpr uint32_t kInstFlagForceNonOpaque = 0x8u;
constexpr uint32_t kInstFlagIdentityTransform = 0x80000000u;

// D3D12_RAYTRACING_GEOMETRY_FLAGS
constexpr uint32_t kGeomFlagOpaque = 0x1u;

// Committed / Candidate enums
constexpr uint32_t kCommittedNothing = 0u;
constexpr uint32_t kCommittedTriangleHit = 1u;
constexpr uint32_t kCommittedProceduralHit = 2u;
constexpr uint32_t kCandidateNonOpaqueTriangle = 0u;
constexpr uint32_t kCandidateProceduralPrimitive = 1u;

struct BvhNode {
    float minB[3];
    float maxB[3];
    uint32_t leftOrFirst;
    uint32_t count; // 0 => internal node (children: leftOrFirst, leftOrFirst + 1); >0 => leaf
};

struct BlasTriangle {
    float v0[3];
    float v1[3];
    float v2[3];
    uint32_t geometryIndex;
    uint32_t primitiveIndex;
    uint32_t geomFlags;
};

struct BlasAabb {
    float minB[3];
    float maxB[3];
    uint32_t geometryIndex;
    uint32_t primitiveIndex;
    uint32_t geomFlags;
};

struct BlasHeader {
    uint32_t magic;
    uint32_t totalBytes;
    uint32_t triRoot;
    uint32_t aabbRoot;
    uint32_t nodeCount;
    uint32_t triCount;
    uint32_t aabbCount;
    uint32_t nodesOffset;
    uint32_t trisOffset;
    uint32_t aabbsOffset;
    float boundsMin[3];
    float boundsMax[3];
};

struct TlasInstance {
    float objectToWorld[12];
    float worldToObject[12];
    float minB[3];
    float maxB[3];
    uint32_t instanceID;
    uint32_t instanceMask;
    uint32_t instanceContributionToHitGroupIndex;
    uint32_t flags;
    uint32_t instanceIndex;
    uint64_t blasAddress;
};

struct TlasHeader {
    uint32_t magic;
    uint32_t totalBytes;
    uint32_t rootNode;
    uint32_t nodeCount;
    uint32_t instanceCount;
    uint32_t nodesOffset;
    uint32_t instancesOffset;
};

} // namespace nacelle
