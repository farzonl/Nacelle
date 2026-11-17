#include "../rasterizer/raytracing_as_layout.h"
#include <stddef.h>
#include <stdint.h>

namespace {

using namespace nacelle;

struct BC_HitInfo {
    uint32_t status;
    float t;
    float baryU;
    float baryV;
    uint32_t instanceIndex;
    uint32_t instanceID;
    uint32_t geometryIndex;
    uint32_t primitiveIndex;
    uint32_t instanceContributionToHitGroupIndex;
    uint8_t frontFace;
    uint8_t frontFaceValid;
    uint8_t nonOpaque;
    uint8_t pad;
    const TlasInstance* inst;
    const BlasTriangle* tri;
};

struct BC_RayQueryState {
    uint32_t constRayFlags;
    uint32_t rayFlags;
    uint32_t instanceMask;
    float worldOrigin[3];
    float worldDir[3];
    float invWorldDir[3];
    float tMin;
    float tMax;

    const TlasHeader* tlas;
    const BvhNode* tlasNodes;
    const TlasInstance* tlasInstances;

    uint32_t finished;
    BC_HitInfo committed;
    BC_HitInfo candidate;
    uint32_t candidateType;

    uint32_t tlasStack[32];
    int32_t tlasSp;
    uint32_t instCursor;
    uint32_t instEnd;

    const TlasInstance* curInst;
    const BlasHeader* curBlas;
    const BvhNode* blasNodes;
    const BlasTriangle* blasTris;
    const BlasAabb* blasAabbs;
    float curObjOrigin[3];
    float curObjDir[3];
    float curInvObjDir[3];

    uint32_t blasStack[32];
    int32_t blasSp;
    uint8_t traversingTris;
    uint8_t traversingAabbs;
    uint16_t pad2;
    uint32_t primCursor;
    uint32_t primEnd;
};

static_assert(sizeof(BC_RayQueryState) <= 640, "BC_RayQueryState exceeds stack slot size");

inline void computeInvDir(const float dir[3], float invDir[3]) {
    for (int a = 0; a < 3; ++a) {
        const float absD = __builtin_fabsf(dir[a]);
        const float d = absD >= 1e-20f ? dir[a] : __builtin_copysignf(1e-20f, dir[a]);
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

inline void transformPoint3x4(const float m[12], const float in[3], float out[3]) {
    const float x = in[0], y = in[1], z = in[2];
    out[0] = m[0] * x + m[1] * y + m[2] * z + m[3];
    out[1] = m[4] * x + m[5] * y + m[6] * z + m[7];
    out[2] = m[8] * x + m[9] * y + m[10] * z + m[11];
}

inline void transformVec3x3(const float m[12], const float in[3], float out[3]) {
    const float x = in[0], y = in[1], z = in[2];
    out[0] = m[0] * x + m[1] * y + m[2] * z;
    out[1] = m[4] * x + m[5] * y + m[6] * z;
    out[2] = m[8] * x + m[9] * y + m[10] * z;
}

inline bool intersectRayTriangle(
    const float orig[3],
    const float dir[3],
    float tMin,
    float tMax,
    const BlasTriangle& tri,
    float& outT,
    float& outU,
    float& outV) {
    const float e1[3] = {tri.v1[0] - tri.v0[0], tri.v1[1] - tri.v0[1], tri.v1[2] - tri.v0[2]};
    const float e2[3] = {tri.v2[0] - tri.v0[0], tri.v2[1] - tri.v0[1], tri.v2[2] - tri.v0[2]};
    const float pvec[3] = {
        dir[1] * e2[2] - dir[2] * e2[1],
        dir[2] * e2[0] - dir[0] * e2[2],
        dir[0] * e2[1] - dir[1] * e2[0],
    };
    const float det = e1[0] * pvec[0] + e1[1] * pvec[1] + e1[2] * pvec[2];
    if (__builtin_fabsf(det) < 1e-30f) return false;
    const float invDet = 1.0f / det;

    const float tvec[3] = {orig[0] - tri.v0[0], orig[1] - tri.v0[1], orig[2] - tri.v0[2]};
    const float u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    const float qvec[3] = {
        tvec[1] * e1[2] - tvec[2] * e1[1],
        tvec[2] * e1[0] - tvec[0] * e1[2],
        tvec[0] * e1[1] - tvec[1] * e1[0],
    };
    const float v = (dir[0] * qvec[0] + dir[1] * qvec[1] + dir[2] * qvec[2]) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    const float t = (e2[0] * qvec[0] + e2[1] * qvec[1] + e2[2] * qvec[2]) * invDet;
    if (t < tMin || t > tMax) return false;

    outT = t;
    outU = u;
    outV = v;
    return true;
}

inline bool computeTriFrontFace(const TlasInstance* inst, const BlasTriangle* tri, const float worldDir[3]) {
    if (!inst || !tri) return true;
    const float e1[3] = {tri->v1[0] - tri->v0[0], tri->v1[1] - tri->v0[1], tri->v1[2] - tri->v0[2]};
    const float e2[3] = {tri->v2[0] - tri->v0[0], tri->v2[1] - tri->v0[1], tri->v2[2] - tri->v0[2]};
    float we1[3], we2[3];
    transformVec3x3(inst->objectToWorld, e1, we1);
    transformVec3x3(inst->objectToWorld, e2, we2);
    const float wn[3] = {
        we1[1] * we2[2] - we1[2] * we2[1],
        we1[2] * we2[0] - we1[0] * we2[2],
        we1[0] * we2[1] - we1[1] * we2[0],
    };
    const float facingDot = worldDir[0] * wn[0] + worldDir[1] * wn[1] + worldDir[2] * wn[2];
    bool frontFace = facingDot < 0.0f;
    if (inst->flags & kInstFlagTriangleFrontCCW) {
        frontFace = !frontFace;
    }
    return frontFace;
}

inline bool resolveOpaque(uint32_t rayFlags, uint32_t instFlags, uint32_t geomFlags) {
    if (rayFlags & kRayFlagForceOpaque) return true;
    if (rayFlags & kRayFlagForceNonOpaque) return false;
    if (instFlags & kInstFlagForceOpaque) return true;
    if (instFlags & kInstFlagForceNonOpaque) return false;
    return (geomFlags & kGeomFlagOpaque) != 0;
}

inline void beginInstance(BC_RayQueryState& rq, const TlasInstance* inst) {
    rq.curInst = inst;
    rq.curBlas = reinterpret_cast<const BlasHeader*>(static_cast<uintptr_t>(inst->blasAddress));
    if (!rq.curBlas || rq.curBlas->magic != kBlasMagic) {
        rq.curInst = nullptr;
        return;
    }
    const auto* base = reinterpret_cast<const uint8_t*>(rq.curBlas);
    rq.blasNodes = reinterpret_cast<const BvhNode*>(base + rq.curBlas->nodesOffset);
    rq.blasTris = reinterpret_cast<const BlasTriangle*>(base + rq.curBlas->trisOffset);
    rq.blasAabbs = reinterpret_cast<const BlasAabb*>(base + rq.curBlas->aabbsOffset);
    if (inst->flags & kInstFlagIdentityTransform) {
        rq.curObjOrigin[0] = rq.worldOrigin[0]; rq.curObjOrigin[1] = rq.worldOrigin[1]; rq.curObjOrigin[2] = rq.worldOrigin[2];
        rq.curObjDir[0] = rq.worldDir[0]; rq.curObjDir[1] = rq.worldDir[1]; rq.curObjDir[2] = rq.worldDir[2];
        rq.curInvObjDir[0] = rq.invWorldDir[0]; rq.curInvObjDir[1] = rq.invWorldDir[1]; rq.curInvObjDir[2] = rq.invWorldDir[2];
    } else {
        transformPoint3x4(inst->worldToObject, rq.worldOrigin, rq.curObjOrigin);
        transformVec3x3(inst->worldToObject, rq.worldDir, rq.curObjDir);
        computeInvDir(rq.curObjDir, rq.curInvObjDir);
    }

    rq.blasSp = 0;
    rq.primCursor = 0;
    rq.primEnd = 0;
    rq.traversingTris = 0;
    rq.traversingAabbs = 0;

    if (!(rq.rayFlags & kRayFlagSkipTriangles) && rq.curBlas->triRoot != kInvalidNode) {
        rq.traversingTris = 1;
        rq.blasStack[rq.blasSp++] = rq.curBlas->triRoot;
    } else if (!(rq.rayFlags & kRayFlagSkipProceduralPrimitives) && rq.curBlas->aabbRoot != kInvalidNode) {
        rq.traversingAabbs = 1;
        rq.blasStack[rq.blasSp++] = rq.curBlas->aabbRoot;
    } else {
        rq.curInst = nullptr;
    }
}

inline float readMatrixElem(const float m[12], int32_t row, int32_t col) {
    if (row < 0 || row >= 3 || col < 0 || col >= 4) return 0.0f;
    return m[row * 4 + col];
}

} // namespace

extern "C" {

void nacelle_rqbc_init(void* rqPtr, int32_t constRayFlags) {
    auto* rq = static_cast<BC_RayQueryState*>(rqPtr);
    rq->constRayFlags = static_cast<uint32_t>(constRayFlags);
    rq->finished = 1;
    rq->committed.status = kCommittedNothing;
    rq->committed.inst = nullptr;
    rq->committed.tri = nullptr;
}

void nacelle_rqbc_trace_inline(
    void* rqPtr,
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
    auto* rq = static_cast<BC_RayQueryState*>(rqPtr);
    rq->rayFlags = rq->constRayFlags | static_cast<uint32_t>(rayFlags);
    rq->instanceMask = static_cast<uint32_t>(instanceMask) & 0xFFu;
    rq->worldOrigin[0] = ox;
    rq->worldOrigin[1] = oy;
    rq->worldOrigin[2] = oz;
    rq->worldDir[0] = dx;
    rq->worldDir[1] = dy;
    rq->worldDir[2] = dz;
    computeInvDir(rq->worldDir, rq->invWorldDir);
    rq->tMin = tMin;
    rq->tMax = tMax;
    rq->committed.status = kCommittedNothing;
    rq->committed.t = tMax;
    rq->committed.inst = nullptr;
    rq->committed.tri = nullptr;
    rq->committed.frontFace = 0;
    rq->committed.frontFaceValid = 0;
    rq->tlasSp = 0;
    rq->instCursor = 0;
    rq->instEnd = 0;
    rq->curInst = nullptr;
    rq->blasSp = 0;
    rq->traversingTris = 0;
    rq->traversingAabbs = 0;
    rq->primCursor = 0;
    rq->primEnd = 0;

    const auto* tlas = static_cast<const TlasHeader*>(tlasPtr);
    if (!tlas || tlas->magic != kTlasMagic || tlas->rootNode == kInvalidNode || tMax < tMin) {
        rq->finished = 1;
        return;
    }

    const auto* base = static_cast<const uint8_t*>(tlasPtr);
    rq->tlas = tlas;
    rq->tlasNodes = reinterpret_cast<const BvhNode*>(base + tlas->nodesOffset);
    rq->tlasInstances = reinterpret_cast<const TlasInstance*>(base + tlas->instancesOffset);
    rq->finished = 0;
    rq->tlasStack[rq->tlasSp++] = tlas->rootNode;
}

__attribute__((noinline)) int32_t nacelle_rqbc_proceed(void* rqPtr) {
    auto* rq = static_cast<BC_RayQueryState*>(rqPtr);
    if (rq->finished) return 0;

    const bool needCullCheck =
        (rq->rayFlags & (kRayFlagCullBackFacingTriangles | kRayFlagCullFrontFacingTriangles)) != 0;

    while (true) {
        if (rq->curInst) {
            if (rq->traversingTris) {
                while (true) {
                    while (rq->primCursor < rq->primEnd) {
                        const auto& tri = rq->blasTris[rq->primCursor++];
                        float t = 0.0f, u = 0.0f, v = 0.0f;
                        if (!intersectRayTriangle(rq->curObjOrigin, rq->curObjDir, rq->tMin, rq->committed.t, tri, t, u, v)) {
                            continue;
                        }

                        uint8_t frontFace = 0;
                        uint8_t frontFaceValid = 0;
                        if (needCullCheck && !(rq->curInst->flags & kInstFlagTriangleCullDisable)) {
                            frontFace = computeTriFrontFace(rq->curInst, &tri, rq->worldDir) ? 1 : 0;
                            frontFaceValid = 1;
                            if ((rq->rayFlags & kRayFlagCullBackFacingTriangles) && !frontFace) continue;
                            if ((rq->rayFlags & kRayFlagCullFrontFacingTriangles) && frontFace) continue;
                        }

                        const bool isOpaque = resolveOpaque(rq->rayFlags, rq->curInst->flags, tri.geomFlags);
                        if (isOpaque && (rq->rayFlags & kRayFlagCullOpaque)) continue;
                        if (!isOpaque && (rq->rayFlags & kRayFlagCullNonOpaque)) continue;

                        BC_HitInfo& dst = isOpaque ? rq->committed : rq->candidate;
                        dst.status = kCommittedTriangleHit;
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
                        dst.nonOpaque = isOpaque ? 0 : 1;
                        dst.inst = rq->curInst;
                        dst.tri = &tri;

                        if (isOpaque) {
                            if (rq->rayFlags & kRayFlagAcceptFirstHitAndEndSearch) {
                                rq->finished = 1;
                                return 0;
                            }
                        } else {
                            rq->candidateType = kCandidateNonOpaqueTriangle;
                            return 1;
                        }
                    }

                    bool popped = false;
                    while (rq->blasSp > 0) {
                        const uint32_t nodeIdx = rq->blasStack[--rq->blasSp];
                        const auto& node = rq->blasNodes[nodeIdx];
                        if (intersectRayAabbFast(rq->curObjOrigin, rq->curInvObjDir, rq->tMin, rq->committed.t, node.minB, node.maxB)) {
                            if (node.count > 0) {
                                rq->primCursor = node.leftOrFirst;
                                rq->primEnd = node.leftOrFirst + node.count;
                                popped = true;
                                break;
                            }
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst + 1;
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst;
                        }
                    }
                    if (!popped) break;
                }

                rq->traversingTris = 0;
                if (!(rq->rayFlags & kRayFlagSkipProceduralPrimitives) && rq->curBlas->aabbRoot != kInvalidNode) {
                    rq->traversingAabbs = 1;
                    rq->blasStack[rq->blasSp++] = rq->curBlas->aabbRoot;
                }
            }

            if (rq->traversingAabbs) {
                while (true) {
                    while (rq->primCursor < rq->primEnd) {
                        const auto& aabb = rq->blasAabbs[rq->primCursor++];
                        if (!intersectRayAabbFast(rq->curObjOrigin, rq->curInvObjDir, rq->tMin, rq->committed.t, aabb.minB, aabb.maxB)) {
                            continue;
                        }
                        const bool isOpaque = resolveOpaque(rq->rayFlags, rq->curInst->flags, aabb.geomFlags);
                        if (isOpaque && (rq->rayFlags & kRayFlagCullOpaque)) continue;
                        if (!isOpaque && (rq->rayFlags & kRayFlagCullNonOpaque)) continue;

                        BC_HitInfo& cand = rq->candidate;
                        cand.status = kCommittedProceduralHit;
                        cand.t = rq->committed.t;
                        cand.baryU = 0.0f;
                        cand.baryV = 0.0f;
                        cand.frontFace = 0;
                        cand.frontFaceValid = 1;
                        cand.instanceIndex = rq->curInst->instanceIndex;
                        cand.instanceID = rq->curInst->instanceID;
                        cand.geometryIndex = aabb.geometryIndex;
                        cand.primitiveIndex = aabb.primitiveIndex;
                        cand.instanceContributionToHitGroupIndex = rq->curInst->instanceContributionToHitGroupIndex;
                        cand.nonOpaque = isOpaque ? 0 : 1;
                        cand.inst = rq->curInst;
                        cand.tri = nullptr;
                        rq->candidateType = kCandidateProceduralPrimitive;
                        return 1;
                    }

                    bool popped = false;
                    while (rq->blasSp > 0) {
                        const uint32_t nodeIdx = rq->blasStack[--rq->blasSp];
                        const auto& node = rq->blasNodes[nodeIdx];
                        if (intersectRayAabbFast(rq->curObjOrigin, rq->curInvObjDir, rq->tMin, rq->committed.t, node.minB, node.maxB)) {
                            if (node.count > 0) {
                                rq->primCursor = node.leftOrFirst;
                                rq->primEnd = node.leftOrFirst + node.count;
                                popped = true;
                                break;
                            }
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst + 1;
                            rq->blasStack[rq->blasSp++] = node.leftOrFirst;
                        }
                    }
                    if (!popped) break;
                }
                rq->traversingAabbs = 0;
            }

            rq->curInst = nullptr;
        }

        bool foundInst = false;
        while (rq->instCursor < rq->instEnd) {
            const auto& inst = rq->tlasInstances[rq->instCursor++];
            if ((inst.instanceMask & rq->instanceMask) != 0 &&
                intersectRayAabbFast(rq->worldOrigin, rq->invWorldDir, rq->tMin, rq->committed.t, inst.minB, inst.maxB)) {
                beginInstance(*rq, &inst);
                if (rq->curInst) {
                    foundInst = true;
                    break;
                }
            }
        }
        if (foundInst) continue;

        bool foundLeaf = false;
        while (rq->tlasSp > 0) {
            const uint32_t nodeIdx = rq->tlasStack[--rq->tlasSp];
            const auto& node = rq->tlasNodes[nodeIdx];
            if (intersectRayAabbFast(rq->worldOrigin, rq->invWorldDir, rq->tMin, rq->committed.t, node.minB, node.maxB)) {
                if (node.count > 0) {
                    rq->instCursor = node.leftOrFirst;
                    rq->instEnd = node.leftOrFirst + node.count;
                    foundLeaf = true;
                    break;
                }
                rq->tlasStack[rq->tlasSp++] = node.leftOrFirst + 1;
                rq->tlasStack[rq->tlasSp++] = node.leftOrFirst;
            }
        }
        if (foundLeaf) continue;

        rq->finished = 1;
        return 0;
    }
}

void nacelle_rqbc_abort(void* rqPtr) {
    static_cast<BC_RayQueryState*>(rqPtr)->finished = 1;
}

void nacelle_rqbc_commit_triangle(void* rqPtr) {
    auto* rq = static_cast<BC_RayQueryState*>(rqPtr);
    if (rq->candidateType != kCandidateNonOpaqueTriangle) return;
    if (rq->candidate.t >= rq->tMin && rq->candidate.t <= rq->committed.t) {
        rq->committed = rq->candidate;
        if (rq->rayFlags & kRayFlagAcceptFirstHitAndEndSearch) {
            rq->finished = 1;
        }
    }
}

void nacelle_rqbc_commit_procedural(void* rqPtr, float tHit) {
    auto* rq = static_cast<BC_RayQueryState*>(rqPtr);
    if (rq->candidateType != kCandidateProceduralPrimitive) return;
    if (tHit >= rq->tMin && tHit <= rq->committed.t) {
        rq->committed = rq->candidate;
        rq->committed.status = kCommittedProceduralHit;
        rq->committed.t = tHit;
        if (rq->rayFlags & kRayFlagAcceptFirstHitAndEndSearch) {
            rq->finished = 1;
        }
    }
}

int32_t nacelle_rqbc_state_i32(int32_t op, const void* rqPtr) {
    const auto* rq = static_cast<const BC_RayQueryState*>(rqPtr);
    switch (op) {
    case 184: return static_cast<int32_t>(rq->committed.status);
    case 185: return static_cast<int32_t>(rq->candidateType);
    case 190: return rq->candidate.nonOpaque ? 1 : 0;
    case 191:
        return (rq->candidate.frontFaceValid
                    ? (rq->candidate.frontFace != 0)
                    : computeTriFrontFace(rq->candidate.inst, rq->candidate.tri, rq->worldDir))
                   ? 1
                   : 0;
    case 192:
        return (rq->committed.frontFaceValid
                    ? (rq->committed.frontFace != 0)
                    : computeTriFrontFace(rq->committed.inst, rq->committed.tri, rq->worldDir))
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

float nacelle_rqbc_state_f32(int32_t op, const void* rqPtr) {
    const auto* rq = static_cast<const BC_RayQueryState*>(rqPtr);
    switch (op) {
    case 198: return rq->tMin;
    case 199: return rq->candidate.t;
    case 200: return rq->committed.t;
    default: return 0.0f;
    }
}

float nacelle_rqbc_state_vec(int32_t op, const void* rqPtr, int32_t comp) {
    const auto* rq = static_cast<const BC_RayQueryState*>(rqPtr);
    if (comp < 0) return 0.0f;
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
        transformPoint3x4(inst->worldToObject, rq->worldOrigin, out);
        return out[comp];
    }
    case 206:
    case 212: {
        if (comp >= 3) return 0.0f;
        const auto* inst = (op == 206) ? rq->candidate.inst : rq->committed.inst;
        if (!inst) return 0.0f;
        float out[3];
        transformVec3x3(inst->worldToObject, rq->worldDir, out);
        return out[comp];
    }
    default: return 0.0f;
    }
}

float nacelle_rqbc_state_mat(int32_t op, const void* rqPtr, int32_t row, int32_t col) {
    const auto* rq = static_cast<const BC_RayQueryState*>(rqPtr);
    static constexpr float kIdentity[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    switch (op) {
    case 186: return readMatrixElem(rq->candidate.inst ? rq->candidate.inst->objectToWorld : kIdentity, row, col);
    case 187: return readMatrixElem(rq->candidate.inst ? rq->candidate.inst->worldToObject : kIdentity, row, col);
    case 188: return readMatrixElem(rq->committed.inst ? rq->committed.inst->objectToWorld : kIdentity, row, col);
    case 189: return readMatrixElem(rq->committed.inst ? rq->committed.inst->worldToObject : kIdentity, row, col);
    default: return 0.0f;
    }
}

} // extern "C"
