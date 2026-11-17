#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "rasterizer/raytracing_bvh.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace nacelle {
namespace {

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
constexpr uint32_t kInstFlagIdentityTransform = 0x80000000u;

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

constexpr size_t alignUp(size_t v, size_t a) {
    return (v + a - 1) & ~(a - 1);
}

float halfToFloat(uint16_t h) {
    uint32_t sign = (static_cast<uint32_t>(h & 0x8000u)) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x03FFu;
    uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 127 - 14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03FFu;
            bits = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

void readVertexPos(const uint8_t* ptr, DXGI_FORMAT fmt, float out[3]) {
    out[0] = 0.0f;
    out[1] = 0.0f;
    out[2] = 0.0f;
    switch (fmt) {
    case DXGI_FORMAT_R32G32B32_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: {
        const auto* f = reinterpret_cast<const float*>(ptr);
        out[0] = f[0];
        out[1] = f[1];
        out[2] = f[2];
        break;
    }
    case DXGI_FORMAT_R32G32_FLOAT: {
        const auto* f = reinterpret_cast<const float*>(ptr);
        out[0] = f[0];
        out[1] = f[1];
        break;
    }
    case DXGI_FORMAT_R16G16B16A16_FLOAT: {
        const auto* h = reinterpret_cast<const uint16_t*>(ptr);
        out[0] = halfToFloat(h[0]);
        out[1] = halfToFloat(h[1]);
        out[2] = halfToFloat(h[2]);
        break;
    }
    case DXGI_FORMAT_R16G16_FLOAT: {
        const auto* h = reinterpret_cast<const uint16_t*>(ptr);
        out[0] = halfToFloat(h[0]);
        out[1] = halfToFloat(h[1]);
        break;
    }
    default:
        break;
    }
}

void transformPoint3x4(const float m[12], const float in[3], float out[3]) {
    float x = in[0], y = in[1], z = in[2];
    out[0] = m[0] * x + m[1] * y + m[2] * z + m[3];
    out[1] = m[4] * x + m[5] * y + m[6] * z + m[7];
    out[2] = m[8] * x + m[9] * y + m[10] * z + m[11];
}

void transformVec3x3(const float m[12], const float in[3], float out[3]) {
    float x = in[0], y = in[1], z = in[2];
    out[0] = m[0] * x + m[1] * y + m[2] * z;
    out[1] = m[4] * x + m[5] * y + m[6] * z;
    out[2] = m[8] * x + m[9] * y + m[10] * z;
}

void invertTransform3x4(const float m[12], float inv[12]) {
    float a00 = m[0], a01 = m[1], a02 = m[2];
    float a10 = m[4], a11 = m[5], a12 = m[6];
    float a20 = m[8], a21 = m[9], a22 = m[10];

    float c00 = a11 * a22 - a12 * a21;
    float c01 = a12 * a20 - a10 * a22;
    float c02 = a10 * a21 - a11 * a20;
    float det = a00 * c00 + a01 * c01 + a02 * c02;
    if (std::fabs(det) < 1e-30f) {
        std::memset(inv, 0, 12 * sizeof(float));
        inv[0] = inv[5] = inv[10] = 1.0f;
        return;
    }
    float invDet = 1.0f / det;
    inv[0] = c00 * invDet;
    inv[1] = (a02 * a21 - a01 * a22) * invDet;
    inv[2] = (a01 * a12 - a02 * a11) * invDet;
    inv[4] = c01 * invDet;
    inv[5] = (a00 * a22 - a02 * a20) * invDet;
    inv[6] = (a02 * a10 - a00 * a12) * invDet;
    inv[8] = c02 * invDet;
    inv[9] = (a01 * a20 - a00 * a21) * invDet;
    inv[10] = (a00 * a11 - a01 * a10) * invDet;

    float tx = m[3], ty = m[7], tz = m[11];
    inv[3] = -(inv[0] * tx + inv[1] * ty + inv[2] * tz);
    inv[7] = -(inv[4] * tx + inv[5] * ty + inv[6] * tz);
    inv[11] = -(inv[8] * tx + inv[9] * ty + inv[10] * tz);
}

template <typename PrimT, typename BoundsFn>
uint32_t buildBvhRecursive(
    std::vector<PrimT>& prims,
    std::vector<BvhNode>& nodes,
    uint32_t first,
    uint32_t count,
    BoundsFn&& getBounds) {
    uint32_t nodeIdx = static_cast<uint32_t>(nodes.size());
    nodes.push_back(BvhNode{});

    float minB[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    float maxB[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
    for (uint32_t i = 0; i < count; ++i) {
        float pMin[3], pMax[3];
        getBounds(prims[first + i], pMin, pMax);
        for (int a = 0; a < 3; ++a) {
            minB[a] = std::min(minB[a], pMin[a]);
            maxB[a] = std::max(maxB[a], pMax[a]);
        }
    }

    for (int a = 0; a < 3; ++a) {
        nodes[nodeIdx].minB[a] = minB[a];
        nodes[nodeIdx].maxB[a] = maxB[a];
    }

    if (count <= 2) {
        nodes[nodeIdx].leftOrFirst = first;
        nodes[nodeIdx].count = count;
        return nodeIdx;
    }

    float ext[3] = {maxB[0] - minB[0], maxB[1] - minB[1], maxB[2] - minB[2]};
    int axis = 0;
    if (ext[1] > ext[axis]) axis = 1;
    if (ext[2] > ext[axis]) axis = 2;

    if (ext[axis] < 1e-6f) {
        nodes[nodeIdx].leftOrFirst = first;
        nodes[nodeIdx].count = count;
        return nodeIdx;
    }

    float splitPos = 0.5f * (minB[axis] + maxB[axis]);
    auto midIt = std::partition(prims.begin() + first, prims.begin() + first + count, [&](const PrimT& p) {
        float pMin[3], pMax[3];
        getBounds(p, pMin, pMax);
        return 0.5f * (pMin[axis] + pMax[axis]) < splitPos;
    });
    uint32_t leftCount = static_cast<uint32_t>(midIt - (prims.begin() + first));
    if (leftCount == 0 || leftCount == count) {
        leftCount = count / 2;
        std::nth_element(prims.begin() + first, prims.begin() + first + leftCount, prims.begin() + first + count,
                         [&](const PrimT& a, const PrimT& b) {
                             float aMin[3], aMax[3], bMin[3], bMax[3];
                             getBounds(a, aMin, aMax);
                             getBounds(b, bMin, bMax);
                             return (aMin[axis] + aMax[axis]) < (bMin[axis] + bMax[axis]);
                         });
    }

    uint32_t leftChild = static_cast<uint32_t>(nodes.size());
    nodes[nodeIdx].leftOrFirst = leftChild;
    nodes[nodeIdx].count = 0;

    // Reserve two contiguous child slots so children sit at leftChild and leftChild + 1.
    nodes.resize(leftChild + 2);
    auto fillSubtree = [&](uint32_t slot, uint32_t subFirst, uint32_t subCount) {
        if (subCount <= 2) {
            float sMin[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            float sMax[3] = {-std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(), -std::numeric_limits<float>::max()};
            for (uint32_t i = 0; i < subCount; ++i) {
                float pMin[3], pMax[3];
                getBounds(prims[subFirst + i], pMin, pMax);
                for (int a = 0; a < 3; ++a) {
                    sMin[a] = std::min(sMin[a], pMin[a]);
                    sMax[a] = std::max(sMax[a], pMax[a]);
                }
            }
            for (int a = 0; a < 3; ++a) {
                nodes[slot].minB[a] = sMin[a];
                nodes[slot].maxB[a] = sMax[a];
            }
            nodes[slot].leftOrFirst = subFirst;
            nodes[slot].count = subCount;
            return;
        }
        uint32_t built = buildBvhRecursive(prims, nodes, subFirst, subCount, getBounds);
        // Copy root of built subtree into the reserved slot if buildBvhRecursive appended a new node.
        nodes[slot] = nodes[built];
    };

    fillSubtree(leftChild, first, leftCount);
    fillSubtree(leftChild + 1, first + leftCount, count - leftCount);
    return nodeIdx;
}

const D3D12_RAYTRACING_GEOMETRY_DESC* getGeomDesc(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* inputs,
    UINT index) {
    if (inputs->DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS) {
        return inputs->ppGeometryDescs[index];
    }
    return &inputs->pGeometryDescs[index];
}

void countBlasPrimitives(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* inputs,
    uint32_t& outTriCount,
    uint32_t& outAabbCount) {
    outTriCount = 0;
    outAabbCount = 0;
    for (UINT g = 0; g < inputs->NumDescs; ++g) {
        const auto* gd = getGeomDesc(inputs, g);
        if (!gd) continue;
        if (gd->Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) {
            uint32_t count = gd->Triangles.IndexCount > 0
                                 ? gd->Triangles.IndexCount / 3
                                 : gd->Triangles.VertexCount / 3;
            outTriCount += count;
        } else if (gd->Type == D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS) {
            outAabbCount += static_cast<uint32_t>(gd->AABBs.AABBCount);
        }
    }
}

void buildBlas(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc) {
    const auto& inputs = desc->Inputs;
    std::vector<BlasTriangle> tris;
    std::vector<BlasAabb> aabbs;

    for (UINT g = 0; g < inputs.NumDescs; ++g) {
        const auto* gd = getGeomDesc(&inputs, g);
        if (!gd) continue;
        const uint32_t geomFlags = static_cast<uint32_t>(gd->Flags);

        if (gd->Type == D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES) {
            const auto& triDesc = gd->Triangles;
            const auto* vb = reinterpret_cast<const uint8_t*>(triDesc.VertexBuffer.StartAddress);
            const uint64_t vStride = triDesc.VertexBuffer.StrideInBytes;
            const bool hasTransform = triDesc.Transform3x4 != 0;
            float xform[12]{};
            if (hasTransform) {
                std::memcpy(xform, reinterpret_cast<const void*>(triDesc.Transform3x4), sizeof(xform));
            }

            const uint32_t triNum = triDesc.IndexCount > 0
                                        ? triDesc.IndexCount / 3
                                        : triDesc.VertexCount / 3;
            const auto* ib = reinterpret_cast<const uint8_t*>(triDesc.IndexBuffer);
            for (uint32_t t = 0; t < triNum; ++t) {
                uint32_t i0 = t * 3 + 0;
                uint32_t i1 = t * 3 + 1;
                uint32_t i2 = t * 3 + 2;
                if (triDesc.IndexCount > 0 && ib) {
                    if (triDesc.IndexFormat == DXGI_FORMAT_R32_UINT) {
                        const auto* idx32 = reinterpret_cast<const uint32_t*>(ib);
                        i0 = idx32[t * 3 + 0];
                        i1 = idx32[t * 3 + 1];
                        i2 = idx32[t * 3 + 2];
                    } else if (triDesc.IndexFormat == DXGI_FORMAT_R16_UINT) {
                        const auto* idx16 = reinterpret_cast<const uint16_t*>(ib);
                        i0 = idx16[t * 3 + 0];
                        i1 = idx16[t * 3 + 1];
                        i2 = idx16[t * 3 + 2];
                    }
                }
                BlasTriangle bt{};
                readVertexPos(vb + i0 * vStride, triDesc.VertexFormat, bt.v0);
                readVertexPos(vb + i1 * vStride, triDesc.VertexFormat, bt.v1);
                readVertexPos(vb + i2 * vStride, triDesc.VertexFormat, bt.v2);
                if (hasTransform) {
                    float tv0[3], tv1[3], tv2[3];
                    transformPoint3x4(xform, bt.v0, tv0);
                    transformPoint3x4(xform, bt.v1, tv1);
                    transformPoint3x4(xform, bt.v2, tv2);
                    std::memcpy(bt.v0, tv0, sizeof(tv0));
                    std::memcpy(bt.v1, tv1, sizeof(tv1));
                    std::memcpy(bt.v2, tv2, sizeof(tv2));
                }
                bt.geometryIndex = g;
                bt.primitiveIndex = t;
                bt.geomFlags = geomFlags;
                tris.push_back(bt);
            }
        } else if (gd->Type == D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS) {
            const auto& aabbDesc = gd->AABBs;
            const auto* ab = reinterpret_cast<const uint8_t*>(aabbDesc.AABBs.StartAddress);
            const uint64_t aStride = aabbDesc.AABBs.StrideInBytes > 0
                                         ? aabbDesc.AABBs.StrideInBytes
                                         : sizeof(D3D12_RAYTRACING_AABB);
            for (uint64_t a = 0; a < aabbDesc.AABBCount; ++a) {
                const auto* src = reinterpret_cast<const D3D12_RAYTRACING_AABB*>(ab + a * aStride);
                BlasAabb ba{};
                ba.minB[0] = src->MinX;
                ba.minB[1] = src->MinY;
                ba.minB[2] = src->MinZ;
                ba.maxB[0] = src->MaxX;
                ba.maxB[1] = src->MaxY;
                ba.maxB[2] = src->MaxZ;
                ba.geometryIndex = g;
                ba.primitiveIndex = static_cast<uint32_t>(a);
                ba.geomFlags = geomFlags;
                aabbs.push_back(ba);
            }
        }
    }

    std::vector<BvhNode> nodes;
    uint32_t triRoot = kInvalidNode;
    uint32_t aabbRoot = kInvalidNode;

    if (!tris.empty()) {
        triRoot = buildBvhRecursive(tris, nodes, 0, static_cast<uint32_t>(tris.size()),
                                    [](const BlasTriangle& t, float outMin[3], float outMax[3]) {
                                        for (int a = 0; a < 3; ++a) {
                                            outMin[a] = std::min({t.v0[a], t.v1[a], t.v2[a]});
                                            outMax[a] = std::max({t.v0[a], t.v1[a], t.v2[a]});
                                        }
                                    });
    }
    if (!aabbs.empty()) {
        aabbRoot = buildBvhRecursive(aabbs, nodes, 0, static_cast<uint32_t>(aabbs.size()),
                                     [](const BlasAabb& b, float outMin[3], float outMax[3]) {
                                         for (int a = 0; a < 3; ++a) {
                                             outMin[a] = b.minB[a];
                                             outMax[a] = b.maxB[a];
                                         }
                                     });
    }

    BlasHeader hdr{};
    hdr.magic = kBlasMagic;
    hdr.triRoot = triRoot;
    hdr.aabbRoot = aabbRoot;
    hdr.nodeCount = static_cast<uint32_t>(nodes.size());
    hdr.triCount = static_cast<uint32_t>(tris.size());
    hdr.aabbCount = static_cast<uint32_t>(aabbs.size());
    hdr.nodesOffset = static_cast<uint32_t>(alignUp(sizeof(BlasHeader), 16));
    hdr.trisOffset = static_cast<uint32_t>(alignUp(hdr.nodesOffset + nodes.size() * sizeof(BvhNode), 16));
    hdr.aabbsOffset = static_cast<uint32_t>(alignUp(hdr.trisOffset + tris.size() * sizeof(BlasTriangle), 16));
    hdr.totalBytes = static_cast<uint32_t>(alignUp(hdr.aabbsOffset + aabbs.size() * sizeof(BlasAabb), 16));

    for (int a = 0; a < 3; ++a) {
        hdr.boundsMin[a] = std::numeric_limits<float>::max();
        hdr.boundsMax[a] = -std::numeric_limits<float>::max();
    }
    if (triRoot != kInvalidNode) {
        for (int a = 0; a < 3; ++a) {
            hdr.boundsMin[a] = std::min(hdr.boundsMin[a], nodes[triRoot].minB[a]);
            hdr.boundsMax[a] = std::max(hdr.boundsMax[a], nodes[triRoot].maxB[a]);
        }
    }
    if (aabbRoot != kInvalidNode) {
        for (int a = 0; a < 3; ++a) {
            hdr.boundsMin[a] = std::min(hdr.boundsMin[a], nodes[aabbRoot].minB[a]);
            hdr.boundsMax[a] = std::max(hdr.boundsMax[a], nodes[aabbRoot].maxB[a]);
        }
    }
    if (triRoot == kInvalidNode && aabbRoot == kInvalidNode) {
        for (int a = 0; a < 3; ++a) {
            hdr.boundsMin[a] = 0.0f;
            hdr.boundsMax[a] = 0.0f;
        }
    }

    auto* dst = reinterpret_cast<uint8_t*>(desc->DestAccelerationStructureData);
    std::memcpy(dst, &hdr, sizeof(hdr));
    if (!nodes.empty()) {
        std::memcpy(dst + hdr.nodesOffset, nodes.data(), nodes.size() * sizeof(BvhNode));
    }
    if (!tris.empty()) {
        std::memcpy(dst + hdr.trisOffset, tris.data(), tris.size() * sizeof(BlasTriangle));
    }
    if (!aabbs.empty()) {
        std::memcpy(dst + hdr.aabbsOffset, aabbs.data(), aabbs.size() * sizeof(BlasAabb));
    }
}

void buildTlas(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc) {
    const auto& inputs = desc->Inputs;
    std::vector<TlasInstance> instances;
    instances.reserve(inputs.NumDescs);

    for (UINT i = 0; i < inputs.NumDescs; ++i) {
        const D3D12_RAYTRACING_INSTANCE_DESC* srcInst = nullptr;
        if (inputs.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS) {
            const auto* ptrs = reinterpret_cast<const D3D12_GPU_VIRTUAL_ADDRESS*>(inputs.InstanceDescs);
            srcInst = reinterpret_cast<const D3D12_RAYTRACING_INSTANCE_DESC*>(ptrs[i]);
        } else {
            const auto* arr = reinterpret_cast<const D3D12_RAYTRACING_INSTANCE_DESC*>(inputs.InstanceDescs);
            srcInst = &arr[i];
        }
        if (!srcInst || srcInst->InstanceMask == 0 || srcInst->AccelerationStructure == 0) {
            continue;
        }
        const auto* blas = reinterpret_cast<const BlasHeader*>(srcInst->AccelerationStructure);
        if (blas->magic != kBlasMagic) continue;

        TlasInstance inst{};
        std::memcpy(inst.objectToWorld, srcInst->Transform, sizeof(inst.objectToWorld));
        static constexpr float kIdent[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        const bool isIdent = std::memcmp(inst.objectToWorld, kIdent, sizeof(kIdent)) == 0;
        if (isIdent) {
            std::memcpy(inst.worldToObject, kIdent, sizeof(kIdent));
        } else {
            invertTransform3x4(inst.objectToWorld, inst.worldToObject);
        }
        inst.instanceID = srcInst->InstanceID;
        inst.instanceMask = srcInst->InstanceMask;
        inst.instanceContributionToHitGroupIndex = srcInst->InstanceContributionToHitGroupIndex;
        inst.flags = srcInst->Flags | (isIdent ? kInstFlagIdentityTransform : 0u);
        inst.instanceIndex = i;
        inst.blasAddress = srcInst->AccelerationStructure;

        for (int a = 0; a < 3; ++a) {
            inst.minB[a] = std::numeric_limits<float>::max();
            inst.maxB[a] = -std::numeric_limits<float>::max();
        }
        for (int c = 0; c < 8; ++c) {
            float localPt[3] = {
                (c & 1) ? blas->boundsMax[0] : blas->boundsMin[0],
                (c & 2) ? blas->boundsMax[1] : blas->boundsMin[1],
                (c & 4) ? blas->boundsMax[2] : blas->boundsMin[2],
            };
            float worldPt[3];
            transformPoint3x4(inst.objectToWorld, localPt, worldPt);
            for (int a = 0; a < 3; ++a) {
                inst.minB[a] = std::min(inst.minB[a], worldPt[a]);
                inst.maxB[a] = std::max(inst.maxB[a], worldPt[a]);
            }
        }
        instances.push_back(inst);
    }

    std::vector<BvhNode> nodes;
    uint32_t rootNode = kInvalidNode;
    if (!instances.empty()) {
        rootNode = buildBvhRecursive(instances, nodes, 0, static_cast<uint32_t>(instances.size()),
                                     [](const TlasInstance& inst, float outMin[3], float outMax[3]) {
                                         for (int a = 0; a < 3; ++a) {
                                             outMin[a] = inst.minB[a];
                                             outMax[a] = inst.maxB[a];
                                         }
                                     });
    }

    TlasHeader hdr{};
    hdr.magic = kTlasMagic;
    hdr.rootNode = rootNode;
    hdr.nodeCount = static_cast<uint32_t>(nodes.size());
    hdr.instanceCount = static_cast<uint32_t>(instances.size());
    hdr.nodesOffset = static_cast<uint32_t>(alignUp(sizeof(TlasHeader), 16));
    hdr.instancesOffset = static_cast<uint32_t>(alignUp(hdr.nodesOffset + nodes.size() * sizeof(BvhNode), 16));
    hdr.totalBytes = static_cast<uint32_t>(alignUp(hdr.instancesOffset + instances.size() * sizeof(TlasInstance), 16));

    auto* dst = reinterpret_cast<uint8_t*>(desc->DestAccelerationStructureData);
    std::memcpy(dst, &hdr, sizeof(hdr));
    if (!nodes.empty()) {
        std::memcpy(dst + hdr.nodesOffset, nodes.data(), nodes.size() * sizeof(BvhNode));
    }
    if (!instances.empty()) {
        std::memcpy(dst + hdr.instancesOffset, instances.data(), instances.size() * sizeof(TlasInstance));
    }
}

inline void computeInvDir(const float dir[3], float invDir[3]) {
    for (int a = 0; a < 3; ++a) {
        const float d = std::fabs(dir[a]) >= 1e-20f ? dir[a] : std::copysign(1e-20f, dir[a]);
        invDir[a] = 1.0f / d;
    }
}

inline bool intersectRayAabbFast(
    const float origin[3],
    const float invDir[3],
    float tMin,
    float tMax,
    const float bMin[3],
    const float bMax[3]) {
    for (int a = 0; a < 3; ++a) {
        const float t0 = (bMin[a] - origin[a]) * invDir[a];
        const float t1 = (bMax[a] - origin[a]) * invDir[a];
        const float tNear = t0 < t1 ? t0 : t1;
        const float tFar = t0 > t1 ? t0 : t1;
        tMin = tNear > tMin ? tNear : tMin;
        tMax = tFar < tMax ? tFar : tMax;
        if (tMax < tMin) return false;
    }
    return true;
}

bool intersectRayAabb(
    const float origin[3],
    const float dir[3],
    float tMin,
    float tMax,
    const float bMin[3],
    const float bMax[3]) {
    float invDir[3];
    computeInvDir(dir, invDir);
    return intersectRayAabbFast(origin, invDir, tMin, tMax, bMin, bMax);
}

bool intersectRayTriangle(
    const float orig[3],
    const float dir[3],
    float tMin,
    float tMax,
    const BlasTriangle& tri,
    float& outT,
    float& outU,
    float& outV) {
    float e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
    float e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};

    // pvec = cross(dir, e2)
    float pvec[3] = {
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    };
    float det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
    if (std::fabs(det) < 1e-30f) return false;
    float invDet = 1.0f / det;

    float tvec[3] = {orig[0] - tri.v0[0], orig[1] - tri.v0[1], orig[2] - tri.v0[2]};
    float u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    // qvec = cross(tvec, e1)
    float qvec[3] = {
        tvec[1] * e1[2] - tvec[2] * e1[1],
        tvec[2] * e1[0] - tvec[0] * e1[2],
        tvec[0] * e1[1] - tvec[1] * e1[0],
    };
    float v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    float t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * invDet;
    if (t < tMin || t > tMax) return false;

    outT = t;
    outU = u;
    outV = v;
    return true;
}

struct HitInfo {
    uint32_t status = kCommittedNothing;
    float t = 0.0f;
    float baryU = 0.0f;
    float baryV = 0.0f;
    uint32_t instanceIndex = 0;
    uint32_t instanceID = 0;
    uint32_t geometryIndex = 0;
    uint32_t primitiveIndex = 0;
    uint32_t instanceContributionToHitGroupIndex = 0;
    bool frontFace = false;
    bool frontFaceValid = false;
    bool nonOpaque = false;
    const TlasInstance* inst = nullptr;
    const BlasTriangle* tri = nullptr;
};

struct RayQueryState {
    uint32_t constRayFlags = 0;
    uint32_t rayFlags = 0;
    uint32_t instanceMask = 0xFF;
    float worldOrigin[3]{};
    float worldDir[3]{};
    float invWorldDir[3]{};
    float tMin = 0.0f;
    float tMax = 0.0f;

    const TlasHeader* tlas = nullptr;
    const BvhNode* tlasNodes = nullptr;
    const TlasInstance* tlasInstances = nullptr;

    bool finished = true;
    HitInfo committed{};
    HitInfo candidate{};
    uint32_t candidateType = kCandidateNonOpaqueTriangle;

    // Iterative TLAS + BLAS traversal state for Proceed()
    uint32_t tlasStack[32]{};
    int32_t tlasSp = 0;
    uint32_t instCursor = 0;
    uint32_t instEnd = 0;

    const TlasInstance* curInst = nullptr;
    const BlasHeader* curBlas = nullptr;
    const BvhNode* blasNodes = nullptr;
    const BlasTriangle* blasTris = nullptr;
    const BlasAabb* blasAabbs = nullptr;
    float curObjOrigin[3]{};
    float curObjDir[3]{};
    float curInvObjDir[3]{};

    uint32_t blasStack[32]{};
    int32_t blasSp = 0;
    bool traversingTris = false;
    bool traversingAabbs = false;
    uint32_t primCursor = 0;
    uint32_t primEnd = 0;
};

constexpr int32_t kMaxRayQueries = 256;
thread_local std::array<RayQueryState, kMaxRayQueries> g_rayQueries{};
thread_local uint32_t g_nextRayQuerySlot = 0;

inline RayQueryState* getQuery(int32_t handle) {
    if (handle < 0 || handle >= kMaxRayQueries) return nullptr;
    return &g_rayQueries[static_cast<size_t>(handle)];
}

inline bool computeTriFrontFace(const TlasInstance* inst, const BlasTriangle* tri, const float worldDir[3]) {
    if (!inst || !tri) return true;
    float e1[3] = {tri->v1[0] - tri->v0[0], tri->v1[1] - tri->v0[1], tri->v1[2] - tri->v0[2]};
    float e2[3] = {tri->v2[0] - tri->v0[0], tri->v2[1] - tri->v0[1], tri->v2[2] - tri->v0[2]};
    float we1[3], we2[3];
    transformVec3x3(inst->objectToWorld, e1, we1);
    transformVec3x3(inst->objectToWorld, e2, we2);
    float wn[3] = {
        we1[1] * we2[2] - we1[2] * we2[1],
        we1[2] * we2[0] - we1[0] * we2[2],
        we1[0] * we2[1] - we1[1] * we2[0],
    };
    float facingDot = worldDir[0] * wn[0] + worldDir[1] * wn[1] + worldDir[2] * wn[2];
    bool frontFace = facingDot < 0.0f;
    if (inst->flags & D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_FRONT_COUNTERCLOCKWISE) {
        frontFace = !frontFace;
    }
    return frontFace;
}

inline void beginInstance(RayQueryState& rq, const TlasInstance* inst) {
    rq.curInst = inst;
    rq.curBlas = reinterpret_cast<const BlasHeader*>(inst->blasAddress);
    if (!rq.curBlas || rq.curBlas->magic != kBlasMagic) {
        rq.curInst = nullptr;
        return;
    }
    const auto* base = reinterpret_cast<const uint8_t*>(rq.curBlas);
    rq.blasNodes = reinterpret_cast<const BvhNode*>(base + rq.curBlas->nodesOffset);
    rq.blasTris = reinterpret_cast<const BlasTriangle*>(base + rq.curBlas->trisOffset);
    rq.blasAabbs = reinterpret_cast<const BlasAabb*>(base + rq.curBlas->aabbsOffset);
    transformPoint3x4(inst->worldToObject, rq.worldOrigin, rq.curObjOrigin);
    transformVec3x3(inst->worldToObject, rq.worldDir, rq.curObjDir);
    computeInvDir(rq.curObjDir, rq.curInvObjDir);

    rq.blasSp = 0;
    rq.primCursor = 0;
    rq.primEnd = 0;
    rq.traversingTris = false;
    rq.traversingAabbs = false;

    if (!(rq.rayFlags & kRayFlagSkipTriangles) && rq.curBlas->triRoot != kInvalidNode) {
        rq.traversingTris = true;
        rq.blasStack[rq.blasSp++] = rq.curBlas->triRoot;
    } else if (!(rq.rayFlags & kRayFlagSkipProceduralPrimitives) && rq.curBlas->aabbRoot != kInvalidNode) {
        rq.traversingAabbs = true;
        rq.blasStack[rq.blasSp++] = rq.curBlas->aabbRoot;
    } else {
        rq.curInst = nullptr;
    }
}

inline bool resolveOpaque(uint32_t rayFlags, uint32_t instFlags, uint32_t geomFlags) {
    if (rayFlags & kRayFlagForceOpaque) return true;
    if (rayFlags & kRayFlagForceNonOpaque) return false;
    if (instFlags & D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_OPAQUE) return true;
    if (instFlags & D3D12_RAYTRACING_INSTANCE_FLAG_FORCE_NON_OPAQUE) return false;
    return (geomFlags & D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE) != 0;
}

float readMatrixElem(const float m[12], int32_t row, int32_t col) {
    if (row < 0 || row >= 3 || col < 0 || col >= 4) return 0.0f;
    return m[row * 4 + col];
}

} // namespace

void getRaytracingAccelerationStructurePrebuildInfo(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* inputs,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* outInfo) {
    if (!outInfo) return;
    std::memset(outInfo, 0, sizeof(*outInfo));
    if (!inputs) return;

    size_t resultBytes = 256;
    if (inputs->Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        uint32_t triCount = 0;
        uint32_t aabbCount = 0;
        countBlasPrimitives(inputs, triCount, aabbCount);
        size_t maxNodes = 2ull * (triCount + aabbCount) + 8ull;
        resultBytes = alignUp(sizeof(BlasHeader), 16) +
                      alignUp(maxNodes * sizeof(BvhNode), 16) +
                      alignUp(triCount * sizeof(BlasTriangle), 16) +
                      alignUp(aabbCount * sizeof(BlasAabb), 16);
    } else {
        size_t instCount = inputs->NumDescs;
        size_t maxNodes = 2ull * instCount + 4ull;
        resultBytes = alignUp(sizeof(TlasHeader), 16) +
                      alignUp(maxNodes * sizeof(BvhNode), 16) +
                      alignUp(instCount * sizeof(TlasInstance), 16);
    }

    outInfo->ResultDataMaxSizeInBytes = alignUp(std::max<size_t>(256, resultBytes), 256);
    outInfo->ScratchDataSizeInBytes = 256;
    outInfo->UpdateScratchDataSizeInBytes = 256;
}

void buildRaytracingAccelerationStructure(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* desc) {
    if (!desc || desc->DestAccelerationStructureData == 0) return;
    if (desc->Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        buildBlas(desc);
    } else {
        buildTlas(desc);
    }
}

void copyRaytracingAccelerationStructure(
    D3D12_GPU_VIRTUAL_ADDRESS destAddr,
    D3D12_GPU_VIRTUAL_ADDRESS srcAddr,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE /*mode*/) {
    if (!destAddr || !srcAddr || destAddr == srcAddr) return;
    const auto* hdr = reinterpret_cast<const uint32_t*>(srcAddr);
    uint32_t bytes = 0;
    if (hdr[0] == kBlasMagic) {
        bytes = reinterpret_cast<const BlasHeader*>(srcAddr)->totalBytes;
    } else if (hdr[0] == kTlasMagic) {
        bytes = reinterpret_cast<const TlasHeader*>(srcAddr)->totalBytes;
    }
    if (bytes > 0) {
        std::memcpy(reinterpret_cast<void*>(destAddr), reinterpret_cast<const void*>(srcAddr), bytes);
    }
}

} // namespace nacelle

extern "C" {

int32_t nacelle_rayquery_alloc(int32_t constRayFlags) {
    uint32_t slot = nacelle::g_nextRayQuerySlot % static_cast<uint32_t>(nacelle::kMaxRayQueries);
    ++nacelle::g_nextRayQuerySlot;
    auto& rq = nacelle::g_rayQueries[slot];
    rq.constRayFlags = static_cast<uint32_t>(constRayFlags);
    rq.finished = true;
    rq.committed.status = nacelle::kCommittedNothing;
    rq.committed.inst = nullptr;
    rq.committed.tri = nullptr;
    return static_cast<int32_t>(slot);
}

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
    float tMax) {
    auto* rq = nacelle::getQuery(handle);
    if (!rq) return;

    rq->rayFlags = rq->constRayFlags | static_cast<uint32_t>(rayFlags);
    rq->instanceMask = static_cast<uint32_t>(instanceMask) & 0xFFu;
    rq->worldOrigin[0] = ox;
    rq->worldOrigin[1] = oy;
    rq->worldOrigin[2] = oz;
    rq->worldDir[0] = dx;
    rq->worldDir[1] = dy;
    rq->worldDir[2] = dz;
    nacelle::computeInvDir(rq->worldDir, rq->invWorldDir);
    rq->tMin = tMin;
    rq->tMax = tMax;
    rq->committed.status = nacelle::kCommittedNothing;
    rq->committed.t = tMax;
    rq->committed.inst = nullptr;
    rq->committed.tri = nullptr;
    rq->committed.frontFace = false;
    rq->committed.frontFaceValid = false;
    rq->tlasSp = 0;
    rq->instCursor = 0;
    rq->instEnd = 0;
    rq->curInst = nullptr;
    rq->blasSp = 0;
    rq->traversingTris = false;
    rq->traversingAabbs = false;
    rq->primCursor = 0;
    rq->primEnd = 0;

    const auto* tlas = static_cast<const nacelle::TlasHeader*>(tlasPtr);
    if (!tlas || tlas->magic != nacelle::kTlasMagic || tlas->rootNode == nacelle::kInvalidNode || tMax < tMin) {
        rq->finished = true;
        return;
    }

    const auto* base = static_cast<const uint8_t*>(tlasPtr);
    rq->tlas = tlas;
    rq->tlasNodes = reinterpret_cast<const nacelle::BvhNode*>(base + tlas->nodesOffset);
    rq->tlasInstances = reinterpret_cast<const nacelle::TlasInstance*>(base + tlas->instancesOffset);
    rq->finished = false;
    rq->tlasStack[rq->tlasSp++] = tlas->rootNode;
}

int32_t nacelle_rayquery_proceed(int32_t handle) {
    auto* rq = nacelle::getQuery(handle);
    if (!rq || rq->finished) return 0;

    const bool needCullCheck =
        (rq->rayFlags & (nacelle::kRayFlagCullBackFacingTriangles | nacelle::kRayFlagCullFrontFacingTriangles)) != 0;

    while (true) {
        // 1. Inside an active BLAS instance
        if (rq->curInst) {
            if (rq->traversingTris) {
                while (rq->primCursor < rq->primEnd) {
                    const auto& tri = rq->blasTris[rq->primCursor++];
                    float t = 0.0f, u = 0.0f, v = 0.0f;
                    if (!nacelle::intersectRayTriangle(rq->curObjOrigin, rq->curObjDir, rq->tMin, rq->committed.t, tri, t, u, v)) {
                        continue;
                    }

                    bool frontFace = false;
                    bool frontFaceValid = false;
                    if (needCullCheck && !(rq->curInst->flags & D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE)) {
                        frontFace = nacelle::computeTriFrontFace(rq->curInst, &tri, rq->worldDir);
                        frontFaceValid = true;
                        if ((rq->rayFlags & nacelle::kRayFlagCullBackFacingTriangles) && !frontFace) continue;
                        if ((rq->rayFlags & nacelle::kRayFlagCullFrontFacingTriangles) && frontFace) continue;
                    }

                    bool isOpaque = nacelle::resolveOpaque(rq->rayFlags, rq->curInst->flags, tri.geomFlags);
                    if (isOpaque && (rq->rayFlags & nacelle::kRayFlagCullOpaque)) continue;
                    if (!isOpaque && (rq->rayFlags & nacelle::kRayFlagCullNonOpaque)) continue;

                    nacelle::HitInfo& dst = isOpaque ? rq->committed : rq->candidate;
                    dst.status = nacelle::kCommittedTriangleHit;
                    dst.t = t;
                    dst.baryU = u;
                    dst.baryV = v;
                    dst.frontFace = frontFace;
                    dst.frontFaceValid = frontFaceValid;
                    dst.instanceIndex = rq->curInst->instanceIndex;
                    dst.instanceID = rq->curInst->instanceID;
                    dst.geometryIndex = tri.geometryIndex;
                    dst.primitiveIndex = tri.primitiveIndex;
                    dst.instanceContributionToHitGroupIndex = rq->curInst->instanceContributionToHitGroupIndex;
                    dst.nonOpaque = !isOpaque;
                    dst.inst = rq->curInst;
                    dst.tri = &tri;

                    if (isOpaque) {
                        if (rq->rayFlags & nacelle::kRayFlagAcceptFirstHitAndEndSearch) {
                            rq->finished = true;
                            return 0;
                        }
                    } else {
                        rq->candidateType = nacelle::kCandidateNonOpaqueTriangle;
                        return 1;
                    }
                }

                if (rq->blasSp > 0) {
                    uint32_t nodeIdx = rq->blasStack[--rq->blasSp];
                    const auto& node = rq->blasNodes[nodeIdx];
                    if (nacelle::intersectRayAabbFast(rq->curObjOrigin, rq->curInvObjDir, rq->tMin, rq->committed.t, node.minB, node.maxB)) {
                        if (node.count > 0) {
                            rq->primCursor = node.leftOrFirst;
                            rq->primEnd = node.leftOrFirst + node.count;
                        } else {
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst + 1;
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst;
                        }
                    }
                    continue;
                }

                // Finished triangles; switch to procedural AABBs if present
                rq->traversingTris = false;
                if (!(rq->rayFlags & nacelle::kRayFlagSkipProceduralPrimitives) && rq->curBlas->aabbRoot != nacelle::kInvalidNode) {
                    rq->traversingAabbs = true;
                    rq->blasStack[rq->blasSp++] = rq->curBlas->aabbRoot;
                    continue;
                }
            }

            if (rq->traversingAabbs) {
                while (rq->primCursor < rq->primEnd) {
                    const auto& aabb = rq->blasAabbs[rq->primCursor++];
                    if (!nacelle::intersectRayAabbFast(rq->curObjOrigin, rq->curInvObjDir, rq->tMin, rq->committed.t, aabb.minB, aabb.maxB)) {
                        continue;
                    }
                    bool isOpaque = nacelle::resolveOpaque(rq->rayFlags, rq->curInst->flags, aabb.geomFlags);
                    if (isOpaque && (rq->rayFlags & nacelle::kRayFlagCullOpaque)) continue;
                    if (!isOpaque && (rq->rayFlags & nacelle::kRayFlagCullNonOpaque)) continue;

                    nacelle::HitInfo& cand = rq->candidate;
                    cand.status = nacelle::kCommittedProceduralHit;
                    cand.t = rq->committed.t;
                    cand.baryU = 0.0f;
                    cand.baryV = 0.0f;
                    cand.frontFace = false;
                    cand.frontFaceValid = true;
                    cand.instanceIndex = rq->curInst->instanceIndex;
                    cand.instanceID = rq->curInst->instanceID;
                    cand.geometryIndex = aabb.geometryIndex;
                    cand.primitiveIndex = aabb.primitiveIndex;
                    cand.instanceContributionToHitGroupIndex = rq->curInst->instanceContributionToHitGroupIndex;
                    cand.nonOpaque = !isOpaque;
                    cand.inst = rq->curInst;
                    cand.tri = nullptr;
                    rq->candidateType = nacelle::kCandidateProceduralPrimitive;
                    return 1;
                }

                if (rq->blasSp > 0) {
                    uint32_t nodeIdx = rq->blasStack[--rq->blasSp];
                    const auto& node = rq->blasNodes[nodeIdx];
                    if (nacelle::intersectRayAabbFast(rq->curObjOrigin, rq->curInvObjDir, rq->tMin, rq->committed.t, node.minB, node.maxB)) {
                        if (node.count > 0) {
                            rq->primCursor = node.leftOrFirst;
                            rq->primEnd = node.leftOrFirst + node.count;
                        } else {
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst + 1;
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst;
                        }
                    }
                    continue;
                }
                rq->traversingAabbs = false;
            }

            rq->curInst = nullptr;
        }

        // 2. Advance TLAS instance leaf cursor
        if (rq->instCursor < rq->instEnd) {
            const auto& inst = rq->tlasInstances[rq->instCursor++];
            if ((inst.instanceMask & rq->instanceMask) != 0 &&
                nacelle::intersectRayAabbFast(rq->worldOrigin, rq->invWorldDir, rq->tMin, rq->committed.t, inst.minB, inst.maxB)) {
                nacelle::beginInstance(*rq, &inst);
            }
            continue;
        }

        // 3. Pop next TLAS BVH node
        if (rq->tlasSp > 0) {
            uint32_t nodeIdx = rq->tlasStack[--rq->tlasSp];
            const auto& node = rq->tlasNodes[nodeIdx];
            if (nacelle::intersectRayAabbFast(rq->worldOrigin, rq->invWorldDir, rq->tMin, rq->committed.t, node.minB, node.maxB)) {
                if (node.count > 0) {
                    rq->instCursor = node.leftOrFirst;
                    rq->instEnd = node.leftOrFirst + node.count;
                } else {
                    rq->tlasStack[rq->tlasSp++] = node.leftOrFirst + 1;
                    rq->tlasStack[rq->tlasSp++] = node.leftOrFirst;
                }
            }
            continue;
        }

        rq->finished = true;
        return 0;
    }
}

void nacelle_rayquery_abort(int32_t handle) {
    if (auto* rq = nacelle::getQuery(handle)) {
        rq->finished = true;
    }
}

void nacelle_rayquery_commit_triangle(int32_t handle) {
    auto* rq = nacelle::getQuery(handle);
    if (!rq || rq->candidateType != nacelle::kCandidateNonOpaqueTriangle) return;
    if (rq->candidate.t >= rq->tMin && rq->candidate.t <= rq->committed.t) {
        rq->committed = rq->candidate;
        if (rq->rayFlags & nacelle::kRayFlagAcceptFirstHitAndEndSearch) {
            rq->finished = true;
        }
    }
}

void nacelle_rayquery_commit_procedural(int32_t handle, float tHit) {
    auto* rq = nacelle::getQuery(handle);
    if (!rq || rq->candidateType != nacelle::kCandidateProceduralPrimitive) return;
    if (tHit >= rq->tMin && tHit <= rq->committed.t) {
        rq->committed = rq->candidate;
        rq->committed.status = nacelle::kCommittedProceduralHit;
        rq->committed.t = tHit;
        if (rq->rayFlags & nacelle::kRayFlagAcceptFirstHitAndEndSearch) {
            rq->finished = true;
        }
    }
}

int32_t nacelle_rayquery_state_i32(int32_t op, int32_t handle) {
    const auto* rq = nacelle::getQuery(handle);
    if (!rq) return 0;
    switch (op) {
    case 184: return static_cast<int32_t>(rq->committed.status);
    case 185: return static_cast<int32_t>(rq->candidateType);
    case 190: return rq->candidate.nonOpaque ? 1 : 0;
    case 191:
        return (rq->candidate.frontFaceValid
                    ? rq->candidate.frontFace
                    : nacelle::computeTriFrontFace(rq->candidate.inst, rq->candidate.tri, rq->worldDir))
                   ? 1
                   : 0;
    case 192:
        return (rq->committed.frontFaceValid
                    ? rq->committed.frontFace
                    : nacelle::computeTriFrontFace(rq->committed.inst, rq->committed.tri, rq->worldDir))
                   ? 1
                   : 0;
    case 195: return static_cast<int32_t>(rq->rayFlags);
    case 201: return static_cast<int32_t>(rq->candidate.instanceIndex);
    case 202: return static_cast<int32_t>(rq->candidate.instanceID);
    case 203: return static_cast<int32_t>(rq->candidate.geometryIndex);
    case 204: return static_cast<int32_t>(rq->candidate.primitiveIndex);
    case 207: return static_cast<int32_t>(rq->committed.instanceIndex);
    case 208: return static_cast<int32_t>(rq->committed.instanceID);
    case 209: return static_cast<int32_t>(rq->committed.geometryIndex);
    case 210: return static_cast<int32_t>(rq->committed.primitiveIndex);
    case 214: return static_cast<int32_t>(rq->candidate.instanceContributionToHitGroupIndex);
    case 215: return static_cast<int32_t>(rq->committed.instanceContributionToHitGroupIndex);
    default: return 0;
    }
}

float nacelle_rayquery_state_f32(int32_t op, int32_t handle) {
    const auto* rq = nacelle::getQuery(handle);
    if (!rq) return 0.0f;
    switch (op) {
    case 198: return rq->tMin;
    case 199: return rq->candidate.t;
    case 200: return rq->committed.t;
    default: return 0.0f;
    }
}

float nacelle_rayquery_state_vec(int32_t op, int32_t handle, int32_t comp) {
    const auto* rq = nacelle::getQuery(handle);
    if (!rq || comp < 0) return 0.0f;
    switch (op) {
    case 193: return comp == 0 ? rq->candidate.baryU : (comp == 1 ? rq->candidate.baryV : 0.0f);
    case 194: return comp == 0 ? rq->committed.baryU : (comp == 1 ? rq->committed.baryV : 0.0f);
    case 196: return comp < 3 ? rq->worldOrigin[comp] : 0.0f;
    case 197: return comp < 3 ? rq->worldDir[comp] : 0.0f;
    case 205:
    case 211: {
        if (comp >= 3) return 0.0f;
        const auto* inst = (op == 205) ? rq->candidate.inst : rq->committed.inst;
        if (!inst) return 0.0f;
        float out[3];
        nacelle::transformPoint3x4(inst->worldToObject, rq->worldOrigin, out);
        return out[comp];
    }
    case 206:
    case 212: {
        if (comp >= 3) return 0.0f;
        const auto* inst = (op == 206) ? rq->candidate.inst : rq->committed.inst;
        if (!inst) return 0.0f;
        float out[3];
        nacelle::transformVec3x3(inst->worldToObject, rq->worldDir, out);
        return out[comp];
    }
    default: return 0.0f;
    }
}

float nacelle_rayquery_state_mat(int32_t op, int32_t handle, int32_t row, int32_t col) {
    const auto* rq = nacelle::getQuery(handle);
    if (!rq) return 0.0f;
    static constexpr float kIdentity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    switch (op) {
    case 186: return nacelle::readMatrixElem(rq->candidate.inst ? rq->candidate.inst->objectToWorld : kIdentity, row, col);
    case 187: return nacelle::readMatrixElem(rq->candidate.inst ? rq->candidate.inst->worldToObject : kIdentity, row, col);
    case 188: return nacelle::readMatrixElem(rq->committed.inst ? rq->committed.inst->objectToWorld : kIdentity, row, col);
    case 189: return nacelle::readMatrixElem(rq->committed.inst ? rq->committed.inst->worldToObject : kIdentity, row, col);
    default: return 0.0f;
    }
}

void nacelle_rayquery_alloc_pkt(uint32_t activeMask, const int32_t* constRayFlags, int32_t* outHandles) {
    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        outHandles[l] = nacelle_rayquery_alloc(constRayFlags[l]);
    }
}

void nacelle_rayquery_trace_inline_pkt(
    uint32_t activeMask,
    const int32_t* handles,
    const void* tlasPtr,
    const int32_t* rayFlags,
    const int32_t* instanceMask,
    const float* ox,
    const float* oy,
    const float* oz,
    const float* tMin,
    const float* dx,
    const float* dy,
    const float* dz,
    const float* tMax) {
    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        nacelle_rayquery_trace_inline(
            handles[l], tlasPtr, rayFlags[l], instanceMask[l],
            ox[l], oy[l], oz[l], tMin[l], dx[l], dy[l], dz[l], tMax[l]);
    }
}

uint32_t nacelle_rayquery_proceed_pkt(uint32_t activeMask, const int32_t* handles) {
    uint32_t retMask = 0;
    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        if (nacelle_rayquery_proceed(handles[l]) != 0) {
            retMask |= (1u << l);
        }
    }
    return retMask;
}

void nacelle_rayquery_state_i32_pkt(int32_t op, uint32_t activeMask, const int32_t* handles, int32_t* outVals) {
    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        outVals[l] = nacelle_rayquery_state_i32(op, handles[l]);
    }
}

void nacelle_rayquery_state_f32_pkt(int32_t op, uint32_t activeMask, const int32_t* handles, float* outVals) {
    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        outVals[l] = nacelle_rayquery_state_f32(op, handles[l]);
    }
}

void nacelle_rayquery_state_vec_pkt(
    int32_t op, uint32_t activeMask, const int32_t* handles, const int32_t* comps, float* outVals) {
    uint32_t m = activeMask;
    while (m) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
        m &= m - 1;
        outVals[l] = nacelle_rayquery_state_vec(op, handles[l], comps[l]);
    }
}

} // extern "C"

namespace nacelle {

namespace {

PipelineHitResult hitInfoToPipelineResult(
    const HitInfo& info,
    const float worldOrigin[3],
    const float worldDir[3],
    bool isHit,
    bool isProcedural,
    uint32_t customHitKind) {
    PipelineHitResult r{};
    r.hit = isHit;
    r.isProcedural = isProcedural;
    r.t = info.t;
    r.baryU = info.baryU;
    r.baryV = info.baryV;
    if (isProcedural) {
        r.hitKind = customHitKind;
    } else if (isHit) {
        bool ff = info.frontFaceValid ? info.frontFace : computeTriFrontFace(info.inst, info.tri, worldDir);
        r.hitKind = ff ? 254u : 255u;
    } else {
        r.hitKind = 254u;
    }
    r.instanceIndex = info.instanceIndex;
    r.instanceID = info.instanceID;
    r.geometryIndex = info.geometryIndex;
    r.primitiveIndex = info.primitiveIndex;
    r.instanceContributionToHitGroupIndex = info.instanceContributionToHitGroupIndex;
    if (info.inst) {
        r.objToWorldPtr = info.inst->objectToWorld;
        r.worldToObjPtr = info.inst->worldToObject;
    }
    return r;
}

} // namespace

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
    void* userData) {
    if (!onAnyHit && !onIntersection) {
        PipelineHitResult best{};
        best.hit = false;
        best.t = tMax;

        const auto* tlas = static_cast<const TlasHeader*>(tlasPtr);
        if (!tlas || tlas->magic != kTlasMagic || tlas->rootNode == kInvalidNode || tMax < tMin) {
            return best;
        }

        const float worldOrigin[3] = {ox, oy, oz};
        const float worldDir[3] = {dx, dy, dz};
        float invWorldDir[3];
        computeInvDir(worldDir, invWorldDir);

        const auto* tlasBase = static_cast<const uint8_t*>(tlasPtr);
        const auto* tlasNodes = reinterpret_cast<const BvhNode*>(tlasBase + tlas->nodesOffset);
        const auto* tlasInstances = reinterpret_cast<const TlasInstance*>(tlasBase + tlas->instancesOffset);

        const uint32_t mask = instanceMask & 0xFFu;
        const bool needCullCheck = (rayFlags & (kRayFlagCullBackFacingTriangles | kRayFlagCullFrontFacingTriangles)) != 0;
        const bool acceptFirst = (rayFlags & kRayFlagAcceptFirstHitAndEndSearch) != 0;
        const bool skipTris = (rayFlags & kRayFlagSkipTriangles) != 0;

        uint32_t tlasStack[32];
        int32_t tlasSp = 0;
        tlasStack[tlasSp++] = tlas->rootNode;

        const TlasInstance* bestInst = nullptr;
        const BlasTriangle* bestTri = nullptr;

        while (tlasSp > 0) {
            const uint32_t tNodeIdx = tlasStack[--tlasSp];
            const auto& tNode = tlasNodes[tNodeIdx];
            if (!intersectRayAabbFast(worldOrigin, invWorldDir, tMin, best.t, tNode.minB, tNode.maxB)) {
                continue;
            }
            if (tNode.count == 0) {
                tlasStack[tlasSp++] = tNode.leftOrFirst + 1;
                tlasStack[tlasSp++] = tNode.leftOrFirst;
                continue;
            }

            for (uint32_t ii = 0; ii < tNode.count; ++ii) {
                const auto& inst = tlasInstances[tNode.leftOrFirst + ii];
                if ((inst.instanceMask & mask) == 0 ||
                    !intersectRayAabbFast(worldOrigin, invWorldDir, tMin, best.t, inst.minB, inst.maxB)) {
                    continue;
                }
                const auto* blas = reinterpret_cast<const BlasHeader*>(inst.blasAddress);
                if (!blas || blas->magic != kBlasMagic || skipTris || blas->triRoot == kInvalidNode) {
                    continue;
                }

                const auto* blasBase = reinterpret_cast<const uint8_t*>(blas);
                const auto* blasNodes = reinterpret_cast<const BvhNode*>(blasBase + blas->nodesOffset);
                const auto* blasTris = reinterpret_cast<const BlasTriangle*>(blasBase + blas->trisOffset);

                float objOrigin[3], objDir[3], invObjDir[3];
                if (inst.flags & kInstFlagIdentityTransform) {
                    objOrigin[0] = worldOrigin[0]; objOrigin[1] = worldOrigin[1]; objOrigin[2] = worldOrigin[2];
                    objDir[0] = worldDir[0]; objDir[1] = worldDir[1]; objDir[2] = worldDir[2];
                    invObjDir[0] = invWorldDir[0]; invObjDir[1] = invWorldDir[1]; invObjDir[2] = invWorldDir[2];
                } else {
                    transformPoint3x4(inst.worldToObject, worldOrigin, objOrigin);
                    transformVec3x3(inst.worldToObject, worldDir, objDir);
                    computeInvDir(objDir, invObjDir);
                }

                uint32_t blasStack[32];
                int32_t blasSp = 0;
                blasStack[blasSp++] = blas->triRoot;

                while (blasSp > 0) {
                    const uint32_t bNodeIdx = blasStack[--blasSp];
                    const auto& bNode = blasNodes[bNodeIdx];
                    if (!intersectRayAabbFast(objOrigin, invObjDir, tMin, best.t, bNode.minB, bNode.maxB)) {
                        continue;
                    }
                    if (bNode.count == 0) {
                        blasStack[blasSp++] = bNode.leftOrFirst + 1;
                        blasStack[blasSp++] = bNode.leftOrFirst;
                        continue;
                    }
                    for (uint32_t ti = 0; ti < bNode.count; ++ti) {
                        const auto& tri = blasTris[bNode.leftOrFirst + ti];
                        float t = 0.0f, u = 0.0f, v = 0.0f;
                        if (!intersectRayTriangle(objOrigin, objDir, tMin, best.t, tri, t, u, v)) {
                            continue;
                        }
                        if (needCullCheck && !(inst.flags & D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE)) {
                            const bool ff = computeTriFrontFace(&inst, &tri, worldDir);
                            if ((rayFlags & kRayFlagCullBackFacingTriangles) && !ff) continue;
                            if ((rayFlags & kRayFlagCullFrontFacingTriangles) && ff) continue;
                        }
                        const bool isOpaque = resolveOpaque(rayFlags, inst.flags, tri.geomFlags);
                        if (isOpaque && (rayFlags & kRayFlagCullOpaque)) continue;
                        if (!isOpaque && (rayFlags & kRayFlagCullNonOpaque)) continue;

                        best.hit = true;
                        best.isProcedural = false;
                        best.t = t;
                        best.baryU = u;
                        best.baryV = v;
                        best.instanceIndex = inst.instanceIndex;
                        best.instanceID = inst.instanceID;
                        best.geometryIndex = tri.geometryIndex;
                        best.primitiveIndex = tri.primitiveIndex;
                        best.instanceContributionToHitGroupIndex = inst.instanceContributionToHitGroupIndex;
                        bestInst = &inst;
                        bestTri = &tri;

                        if (acceptFirst) {
                            tlasSp = 0;
                            blasSp = 0;
                            break;
                        }
                    }
                }
                if (acceptFirst && best.hit) break;
            }
        }

        if (best.hit && bestInst) {
            best.hitKind = computeTriFrontFace(bestInst, bestTri, worldDir) ? 254u : 255u;
            best.objToWorldPtr = bestInst->objectToWorld;
            best.worldToObjPtr = bestInst->worldToObject;
        }
        return best;
    }

    const int32_t h = nacelle_rayquery_alloc(0);
    nacelle_rayquery_trace_inline(
        h, tlasPtr, static_cast<int32_t>(rayFlags), static_cast<int32_t>(instanceMask),
        ox, oy, oz, tMin, dx, dy, dz, tMax);

    auto* rq = getQuery(h);
    uint32_t committedCustomHitKind = 0;

    while (nacelle_rayquery_proceed(h)) {
        if (rq->candidateType == kCandidateNonOpaqueTriangle) {
            PipelineHitResult cand = hitInfoToPipelineResult(rq->candidate, rq->worldOrigin, rq->worldDir, true, false, 0);
            int32_t act = onAnyHit ? onAnyHit(userData, cand) : 0;
            if (act == 0 || act == 2) {
                nacelle_rayquery_commit_triangle(h);
                if (act == 2) {
                    nacelle_rayquery_abort(h);
                    break;
                }
            }
        } else if (rq->candidateType == kCandidateProceduralPrimitive && onIntersection) {
            PipelineHitResult cand = hitInfoToPipelineResult(rq->candidate, rq->worldOrigin, rq->worldDir, true, true, 0);
            float committedT = rq->committed.t;
            bool accepted = false;
            bool endSearch = false;
            uint32_t hitKind = 0;
            float baryU = 0.0f;
            float baryV = 0.0f;
            onIntersection(userData, cand, committedT, accepted, endSearch, hitKind, baryU, baryV);
            if (accepted && committedT >= rq->tMin && committedT <= rq->committed.t) {
                nacelle_rayquery_commit_procedural(h, committedT);
                rq->committed.baryU = baryU;
                rq->committed.baryV = baryV;
                committedCustomHitKind = hitKind;
            }
            if (endSearch) {
                nacelle_rayquery_abort(h);
                break;
            }
        }
    }

    const bool hasHit = rq->committed.status != kCommittedNothing;
    const bool isProc = rq->committed.status == kCommittedProceduralHit;
    return hitInfoToPipelineResult(rq->committed, rq->worldOrigin, rq->worldDir, hasHit, isProc, committedCustomHitKind);
}

void traceRayPacketForPipeline(
    uint32_t activeMask,
    const void* tlasPtr,
    const uint32_t* rayFlags,
    const uint32_t* instanceMask,
    const float* ox,
    const float* oy,
    const float* oz,
    const float* tMin,
    const float* dx,
    const float* dy,
    const float* dz,
    const float* tMax,
    PipelineHitResult* outHits) {
    if (activeMask == 0) return;

    const auto* tlas = static_cast<const TlasHeader*>(tlasPtr);
    if (!tlas || tlas->magic != kTlasMagic || tlas->rootNode == kInvalidNode) {
        uint32_t m = activeMask;
        while (m) {
            const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
            m &= m - 1;
            outHits[l].hit = false;
            outHits[l].t = tMax[l];
        }
        return;
    }

    float invDx[16], invDy[16], invDz[16];
    uint32_t validMask = 0;
    uint32_t mInit = activeMask;
    while (mInit) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(mInit));
        mInit &= mInit - 1;
        outHits[l].hit = false;
        outHits[l].t = tMax[l];
        if (tMax[l] >= tMin[l]) {
            validMask |= (1u << l);
            const float ddx = std::fabs(dx[l]) >= 1e-20f ? dx[l] : std::copysign(1e-20f, dx[l]);
            const float ddy = std::fabs(dy[l]) >= 1e-20f ? dy[l] : std::copysign(1e-20f, dy[l]);
            const float ddz = std::fabs(dz[l]) >= 1e-20f ? dz[l] : std::copysign(1e-20f, dz[l]);
            invDx[l] = 1.0f / ddx;
            invDy[l] = 1.0f / ddy;
            invDz[l] = 1.0f / ddz;
        }
    }
    if (validMask == 0) return;

    const auto* tlasBase = static_cast<const uint8_t*>(tlasPtr);
    const auto* tlasNodes = reinterpret_cast<const BvhNode*>(tlasBase + tlas->nodesOffset);
    const auto* tlasInstances = reinterpret_cast<const TlasInstance*>(tlasBase + tlas->instancesOffset);

    const TlasInstance* bestInst[16]{};
    const BlasTriangle* bestTri[16]{};

    auto testAabbMask = [&](uint32_t mask, const float bMin[3], const float bMax[3],
                            const float* ax, const float* ay, const float* az,
                            const float* iax, const float* iay, const float* iaz) -> uint32_t {
        uint32_t hitMask = 0;
        uint32_t m = mask;
        while (m) {
            const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
            m &= m - 1;
            const float orig[3] = {ax[l], ay[l], az[l]};
            const float idir[3] = {iax[l], iay[l], iaz[l]};
            if (intersectRayAabbFast(orig, idir, tMin[l], outHits[l].t, bMin, bMax)) {
                hitMask |= (1u << l);
            }
        }
        return hitMask;
    };

    uint32_t tlasStack[32];
    int32_t tlasSp = 0;
    tlasStack[tlasSp++] = tlas->rootNode;

    while (tlasSp > 0) {
        const uint32_t tNodeIdx = tlasStack[--tlasSp];
        const auto& tNode = tlasNodes[tNodeIdx];
        const uint32_t nodeHitMask = testAabbMask(validMask, tNode.minB, tNode.maxB, ox, oy, oz, invDx, invDy, invDz);
        if (nodeHitMask == 0) continue;

        if (tNode.count == 0) {
            tlasStack[tlasSp++] = tNode.leftOrFirst + 1;
            tlasStack[tlasSp++] = tNode.leftOrFirst;
            continue;
        }

        for (uint32_t ii = 0; ii < tNode.count; ++ii) {
            const auto& inst = tlasInstances[tNode.leftOrFirst + ii];
            uint32_t instHitMask = 0;
            uint32_t m = nodeHitMask;
            while (m) {
                const uint32_t l = static_cast<uint32_t>(std::countr_zero(m));
                m &= m - 1;
                if ((inst.instanceMask & (instanceMask[l] & 0xFFu)) != 0) {
                    const float orig[3] = {ox[l], oy[l], oz[l]};
                    const float idir[3] = {invDx[l], invDy[l], invDz[l]};
                    if (intersectRayAabbFast(orig, idir, tMin[l], outHits[l].t, inst.minB, inst.maxB)) {
                        instHitMask |= (1u << l);
                    }
                }
            }
            if (instHitMask == 0) continue;

            const auto* blas = reinterpret_cast<const BlasHeader*>(inst.blasAddress);
            if (!blas || blas->magic != kBlasMagic || blas->triRoot == kInvalidNode) continue;

            const auto* blasBase = reinterpret_cast<const uint8_t*>(blas);
            const auto* blasNodes = reinterpret_cast<const BvhNode*>(blasBase + blas->nodesOffset);
            const auto* blasTris = reinterpret_cast<const BlasTriangle*>(blasBase + blas->trisOffset);

            const bool isIdent = (inst.flags & kInstFlagIdentityTransform) != 0;
            float objOx[16], objOy[16], objOz[16];
            float objDx[16], objDy[16], objDz[16];
            float invObjDx[16], invObjDy[16], invObjDz[16];

            if (!isIdent) {
                uint32_t lm = instHitMask;
                while (lm) {
                    const uint32_t l = static_cast<uint32_t>(std::countr_zero(lm));
                    lm &= lm - 1;
                    const float wo[3] = {ox[l], oy[l], oz[l]};
                    const float wd[3] = {dx[l], dy[l], dz[l]};
                    float oo[3], od[3], iod[3];
                    transformPoint3x4(inst.worldToObject, wo, oo);
                    transformVec3x3(inst.worldToObject, wd, od);
                    computeInvDir(od, iod);
                    objOx[l] = oo[0]; objOy[l] = oo[1]; objOz[l] = oo[2];
                    objDx[l] = od[0]; objDy[l] = od[1]; objDz[l] = od[2];
                    invObjDx[l] = iod[0]; invObjDy[l] = iod[1]; invObjDz[l] = iod[2];
                }
            }

            const float* bx = isIdent ? ox : objOx;
            const float* by = isIdent ? oy : objOy;
            const float* bz = isIdent ? oz : objOz;
            const float* bdx = isIdent ? dx : objDx;
            const float* bdy = isIdent ? dy : objDy;
            const float* bdz = isIdent ? dz : objDz;
            const float* ibx = isIdent ? invDx : invObjDx;
            const float* iby = isIdent ? invDy : invObjDy;
            const float* ibz = isIdent ? invDz : invObjDz;

            uint32_t blasStack[32];
            int32_t blasSp = 0;
            blasStack[blasSp++] = blas->triRoot;

            while (blasSp > 0) {
                const uint32_t bNodeIdx = blasStack[--blasSp];
                const auto& bNode = blasNodes[bNodeIdx];
                const uint32_t bHitMask = testAabbMask(instHitMask, bNode.minB, bNode.maxB, bx, by, bz, ibx, iby, ibz);
                if (bHitMask == 0) continue;

                if (bNode.count > 0) {
                    for (uint32_t ti = 0; ti < bNode.count; ++ti) {
                        const auto& tri = blasTris[bNode.leftOrFirst + ti];
                        uint32_t tm = bHitMask;
                        while (tm) {
                            const uint32_t l = static_cast<uint32_t>(std::countr_zero(tm));
                            tm &= tm - 1;
                            const uint32_t rf = rayFlags[l];
                            if (rf & kRayFlagSkipTriangles) continue;
                            const float orig[3] = {bx[l], by[l], bz[l]};
                            const float dir[3] = {bdx[l], bdy[l], bdz[l]};
                            float t = 0.0f, u = 0.0f, v = 0.0f;
                            if (!intersectRayTriangle(orig, dir, tMin[l], outHits[l].t, tri, t, u, v)) continue;

                            if ((rf & (kRayFlagCullBackFacingTriangles | kRayFlagCullFrontFacingTriangles)) &&
                                !(inst.flags & D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE)) {
                                const float wdir[3] = {dx[l], dy[l], dz[l]};
                                const bool ff = computeTriFrontFace(&inst, &tri, wdir);
                                if ((rf & kRayFlagCullBackFacingTriangles) && !ff) continue;
                                if ((rf & kRayFlagCullFrontFacingTriangles) && ff) continue;
                            }
                            const bool isOpaque = resolveOpaque(rf, inst.flags, tri.geomFlags);
                            if (isOpaque && (rf & kRayFlagCullOpaque)) continue;
                            if (!isOpaque && (rf & kRayFlagCullNonOpaque)) continue;

                            outHits[l].hit = true;
                            outHits[l].isProcedural = false;
                            outHits[l].t = t;
                            outHits[l].baryU = u;
                            outHits[l].baryV = v;
                            outHits[l].instanceIndex = inst.instanceIndex;
                            outHits[l].instanceID = inst.instanceID;
                            outHits[l].geometryIndex = tri.geometryIndex;
                            outHits[l].primitiveIndex = tri.primitiveIndex;
                            outHits[l].instanceContributionToHitGroupIndex = inst.instanceContributionToHitGroupIndex;
                            bestInst[l] = &inst;
                            bestTri[l] = &tri;
                        }
                    }
                } else {
                    blasStack[blasSp++] = bNode.leftOrFirst + 1;
                    blasStack[blasSp++] = bNode.leftOrFirst;
                }
            }
        }
    }

    uint32_t fm = validMask;
    while (fm) {
        const uint32_t l = static_cast<uint32_t>(std::countr_zero(fm));
        fm &= fm - 1;
        if (outHits[l].hit && bestInst[l]) {
            const float wdir[3] = {dx[l], dy[l], dz[l]};
            outHits[l].hitKind = computeTriFrontFace(bestInst[l], bestTri[l], wdir) ? 254u : 255u;
            outHits[l].objToWorldPtr = bestInst[l]->objectToWorld;
            outHits[l].worldToObjPtr = bestInst[l]->worldToObject;
        }
    }
}

} // namespace nacelle
