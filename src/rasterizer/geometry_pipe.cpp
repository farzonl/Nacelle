#include "raster_types.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "../api/pipeline.h"
#include "../api/resource.h"
#include "../common/fp_env.h"

namespace nacelle {
namespace {

constexpr uint32_t kMaxSimdWidth = 16;
constexpr uint32_t kChunkPrims = 1024;
constexpr uint32_t kMaxChunkVerts = kChunkPrims * 3 + kMaxSimdWidth;
constexpr uint32_t kVtxCacheSize = 2048; // Power of two; 20 KB total fits in L1d
constexpr uint32_t kVtxCacheMask = kVtxCacheSize - 1u;
constexpr uint64_t kParallelPrimThreshold = 512;
constexpr uint64_t kMaxFusedBinMatrixEntries = 524288;

inline uint32_t readIndex(const D3D12_INDEX_BUFFER_VIEW& ibv, uint32_t idxPos, bool& isCut) {
    isCut = false;
    if (!ibv.BufferLocation) return 0;
    const auto* base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ibv.BufferLocation));
    if (ibv.Format == DXGI_FORMAT_R16_UINT) {
        if (static_cast<uint64_t>(idxPos) * 2 + 2 > ibv.SizeInBytes) return 0;
        uint16_t v = 0;
        std::memcpy(&v, base + static_cast<size_t>(idxPos) * 2, 2);
        if (v == 0xFFFFu) isCut = true;
        return v;
    } else {
        if (static_cast<uint64_t>(idxPos) * 4 + 4 > ibv.SizeInBytes) return 0;
        uint32_t v = 0;
        std::memcpy(&v, base + static_cast<size_t>(idxPos) * 4, 4);
        if (v == 0xFFFFFFFFu) isCut = true;
        return v;
    }
}

struct FastIndexReader {
    const uint8_t* base = nullptr;
    uint64_t maxElems = 0;
    bool is16Bit = false;

    void init(const D3D12_INDEX_BUFFER_VIEW& ibv) {
        base = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(ibv.BufferLocation));
        is16Bit = (ibv.Format == DXGI_FORMAT_R16_UINT);
        maxElems = base ? (ibv.SizeInBytes / (is16Bit ? 2u : 4u)) : 0u;
    }

    inline uint32_t readNoCut(uint32_t idxPos) const {
        if (idxPos >= maxElems) return 0;
        if (is16Bit) {
            uint16_t v = 0;
            std::memcpy(&v, base + static_cast<size_t>(idxPos) * 2, 2);
            return v;
        }
        uint32_t v = 0;
        std::memcpy(&v, base + static_cast<size_t>(idxPos) * 4, 4);
        return v;
    }
};

struct ChunkBinPair {
    uint32_t tileIdx;
    uint32_t localTriIdx;
};

struct TriangleEmitTarget {
    NoInitVec<SetupTriangle>* triangles = nullptr;
    NoInitVec<float>* attrBaseArena = nullptr;
    NoInitVec<float>* attrFlatArena = nullptr;
    NoInitVec<float>* origBaryArena = nullptr;
    std::vector<uint32_t>* tileCounts = nullptr;
    std::vector<ChunkBinPair>* binPairs = nullptr;
    uint32_t numTilesX = 0;
    uint32_t numTilesY = 0;

    inline void recordBinIfActive(const SetupTriangle& tri, uint32_t localTriIdx) {
        if (!tileCounts || !binPairs) return;
        if (numTilesX == 0 || numTilesY == 0 || tri.minX >= tri.maxX || tri.minY >= tri.maxY) return;
        uint32_t tx0 = static_cast<uint32_t>(std::max<int32_t>(0, tri.minX)) / kTileSize;
        uint32_t ty0 = static_cast<uint32_t>(std::max<int32_t>(0, tri.minY)) / kTileSize;
        uint32_t tx1 = std::min<uint32_t>(
            numTilesX - 1, static_cast<uint32_t>(std::max<int32_t>(0, tri.maxX - 1)) / kTileSize);
        uint32_t ty1 = std::min<uint32_t>(
            numTilesY - 1, static_cast<uint32_t>(std::max<int32_t>(0, tri.maxY - 1)) / kTileSize);
        for (uint32_t ty = ty0; ty <= ty1; ++ty) {
            uint32_t rowOff = ty * numTilesX;
            for (uint32_t tx = tx0; tx <= tx1; ++tx) {
                uint32_t tIdx = rowOff + tx;
                ++(*tileCounts)[tIdx];
                binPairs->push_back(ChunkBinPair{tIdx, localTriIdx});
            }
        }
    }
};

struct ChunkOutput {
    NoInitVec<SetupTriangle> triangles;
    NoInitVec<float> attrBase;
    NoInitVec<float> attrFlat;
    NoInitVec<float> origBary;
    std::vector<uint32_t> tileCounts;
    std::vector<ChunkBinPair> binPairs;

    void reset(size_t numTiles, bool enableFusedBinning) {
        triangles.clear();
        attrBase.clear();
        attrFlat.clear();
        origBary.clear();
        binPairs.clear();
        if (enableFusedBinning && numTiles > 0) {
            tileCounts.assign(numTiles, 0u);
        } else {
            tileCounts.clear();
        }
    }
};

struct alignas(64) ChunkTransientScratch {
    uint32_t cacheVId[kVtxCacheSize]{};
    uint32_t cacheGen[kVtxCacheSize]{};
    uint16_t cacheLocalIdx[kVtxCacheSize]{};
    uint32_t curGen = 1;

    alignas(64) uint32_t uniqueVIds[kMaxChunkVerts]{};
    alignas(64) uint32_t uniqueIIds[kMaxChunkVerts]{};
    uint16_t localTriVerts[kChunkPrims * 3]{};
    uint32_t localPrimIds[kChunkPrims]{};

    alignas(64) float posSOA[kMaxChunkVerts * 4]{};
    std::vector<float> attrsSOA;

    uint8_t outcodes[kMaxChunkVerts]{};
    float invW[kMaxChunkVerts]{};
    float screenX[kMaxChunkVerts]{};
    float screenY[kMaxChunkVerts]{};
    float screenZ[kMaxChunkVerts]{};
    int64_t fixedX[kMaxChunkVerts]{};
    int64_t fixedY[kMaxChunkVerts]{};

    inline uint32_t advanceGen() {
        if (++curGen == 0) {
            std::memset(cacheGen, 0, sizeof(cacheGen));
            curGen = 1;
        }
        return curGen;
    }
};

struct ClipVertex {
    float clipX;
    float clipY;
    float clipZ;
    float clipW;
    float origBary[3];
};

struct ClipPoly {
    static constexpr uint32_t kMaxVerts = 24;
    ClipVertex v[kMaxVerts];
    uint32_t count = 0;
};

// Clip polygon against one homogeneous plane f(v) >= 0 using lightweight 28-byte ClipVertex.
template <typename EvalFn>
inline void clipClipPolyPlane(const ClipPoly& inPoly, ClipPoly& outPoly, EvalFn&& eval) {
    outPoly.count = 0;
    const uint32_t n = inPoly.count;
    if (n == 0) return;

    ClipVertex prev = inPoly.v[n - 1];
    float prevD = eval(prev);
    bool prevIn = (prevD >= 0.0f);

    for (uint32_t i = 0; i < n; ++i) {
        const ClipVertex& curr = inPoly.v[i];
        float currD = eval(curr);
        bool currIn = (currD >= 0.0f);

        if (currIn != prevIn && outPoly.count < ClipPoly::kMaxVerts) {
            float denom = prevD - currD;
            float t = (std::fabs(denom) > 1e-20f) ? (prevD / denom) : 0.0f;
            t = std::clamp(t, 0.0f, 1.0f);
            ClipVertex& interp = outPoly.v[outPoly.count++];
            interp.clipX = prev.clipX + t * (curr.clipX - prev.clipX);
            interp.clipY = prev.clipY + t * (curr.clipY - prev.clipY);
            interp.clipZ = prev.clipZ + t * (curr.clipZ - prev.clipZ);
            interp.clipW = prev.clipW + t * (curr.clipW - prev.clipW);
            interp.origBary[0] = prev.origBary[0] + t * (curr.origBary[0] - prev.origBary[0]);
            interp.origBary[1] = prev.origBary[1] + t * (curr.origBary[1] - prev.origBary[1]);
            interp.origBary[2] = prev.origBary[2] + t * (curr.origBary[2] - prev.origBary[2]);
        }
        if (currIn && outPoly.count < ClipPoly::kMaxVerts) {
            outPoly.v[outPoly.count++] = curr;
        }
        prev = curr;
        prevD = currD;
        prevIn = currIn;
    }
}

uint32_t combineShadingRate(
    D3D12_SHADING_RATE baseRate,
    uint32_t primRate,
    D3D12_SHADING_RATE_COMBINER combiner) {
    uint32_t a = static_cast<uint32_t>(baseRate);
    uint32_t b = primRate;
    uint32_t ax = (a >> 2) & 0x3u, ay = a & 0x3u;
    uint32_t bx = (b >> 2) & 0x3u, by = b & 0x3u;
    switch (combiner) {
    case D3D12_SHADING_RATE_COMBINER_OVERRIDE:
        return b;
    case D3D12_SHADING_RATE_COMBINER_MIN:
        return (std::min(ax, bx) << 2) | std::min(ay, by);
    case D3D12_SHADING_RATE_COMBINER_MAX:
        return (std::max(ax, bx) << 2) | std::max(ay, by);
    case D3D12_SHADING_RATE_COMBINER_SUM:
        return (std::min(2u, ax + bx) << 2) | std::min(2u, ay + by);
    case D3D12_SHADING_RATE_COMBINER_PASSTHROUGH:
    default:
        return a;
    }
}

inline float evalDepthBias(
    const D3D12_RASTERIZER_DESC& rast,
    int64_t posArea256,
    float sx0, float sy0, float sz0,
    float sx1, float sy1, float sz1,
    float sx2, float sy2, float sz2) {
    if (rast.DepthBias == 0 && rast.SlopeScaledDepthBias == 0.0f) {
        return 0.0f;
    }
    constexpr float r = 1.0f / 16777216.0f; // UNORM24 / float32 unit
    float biasTerm = static_cast<float>(rast.DepthBias) * r;
    float slopeTerm = 0.0f;
    if (rast.SlopeScaledDepthBias != 0.0f) {
        float d = static_cast<float>(posArea256) / 65536.0f;
        float dzdx = ((sz1 - sz0) * (sy2 - sy0) - (sz2 - sz0) * (sy1 - sy0)) / d;
        float dzdy = ((sx1 - sx0) * (sz2 - sz0) - (sx2 - sx0) * (sz1 - sz0)) / d;
        float maxSlope = std::max(std::fabs(dzdx), std::fabs(dzdy));
        slopeTerm = rast.SlopeScaledDepthBias * maxSlope;
    }
    float depthBiasVal = biasTerm + slopeTerm;
    if (rast.DepthBiasClamp > 0.0f) {
        depthBiasVal = std::min(depthBiasVal, rast.DepthBiasClamp);
    } else if (rast.DepthBiasClamp < 0.0f) {
        depthBiasVal = std::max(depthBiasVal, rast.DepthBiasClamp);
    }
    return depthBiasVal;
}

template <typename GetTriAttrsFn>
void setupClippedBaryTriangle(
    const ClipVertex& v0,
    const ClipVertex& v1,
    const ClipVertex& v2,
    GetTriAttrsFn&& getTriAttrs,
    uint32_t primitiveId,
    uint32_t viewportIndex,
    uint32_t shadingRate,
    const D3D12_VIEWPORT& vp,
    const D3D12_RECT& scissor,
    const D3D12_RASTERIZER_DESC& rast,
    const RasterScene& sceneCfg,
    TriangleEmitTarget& target,
    bool unclipped) {
    if (v0.clipW <= 1e-7f || v1.clipW <= 1e-7f || v2.clipW <= 1e-7f) return;

    float iw0 = 1.0f / v0.clipW;
    float iw1 = 1.0f / v1.clipW;
    float iw2 = 1.0f / v2.clipW;

    float ndcX0 = v0.clipX * iw0, ndcY0 = v0.clipY * iw0, ndcZ0 = v0.clipZ * iw0;
    float ndcX1 = v1.clipX * iw1, ndcY1 = v1.clipY * iw1, ndcZ1 = v1.clipZ * iw1;
    float ndcX2 = v2.clipX * iw2, ndcY2 = v2.clipY * iw2, ndcZ2 = v2.clipZ * iw2;

    float sx0 = vp.TopLeftX + (ndcX0 * 0.5f + 0.5f) * vp.Width;
    float sy0 = vp.TopLeftY + (-ndcY0 * 0.5f + 0.5f) * vp.Height;
    float sz0 = vp.MinDepth + ndcZ0 * (vp.MaxDepth - vp.MinDepth);

    float sx1 = vp.TopLeftX + (ndcX1 * 0.5f + 0.5f) * vp.Width;
    float sy1 = vp.TopLeftY + (-ndcY1 * 0.5f + 0.5f) * vp.Height;
    float sz1 = vp.MinDepth + ndcZ1 * (vp.MaxDepth - vp.MinDepth);

    float sx2 = vp.TopLeftX + (ndcX2 * 0.5f + 0.5f) * vp.Width;
    float sy2 = vp.TopLeftY + (-ndcY2 * 0.5f + 0.5f) * vp.Height;
    float sz2 = vp.MinDepth + ndcZ2 * (vp.MaxDepth - vp.MinDepth);

    int64_t fx0 = static_cast<int64_t>(std::nearbyintf(sx0 * 256.0f));
    int64_t fy0 = static_cast<int64_t>(std::nearbyintf(sy0 * 256.0f));
    int64_t fx1 = static_cast<int64_t>(std::nearbyintf(sx1 * 256.0f));
    int64_t fy1 = static_cast<int64_t>(std::nearbyintf(sy1 * 256.0f));
    int64_t fx2 = static_cast<int64_t>(std::nearbyintf(sx2 * 256.0f));
    int64_t fy2 = static_cast<int64_t>(std::nearbyintf(sy2 * 256.0f));

    int64_t signedArea256 = (fx1 - fx0) * (fy2 - fy0) - (fy1 - fy0) * (fx2 - fx0);
    if (signedArea256 == 0) return;

    bool isCW = (signedArea256 > 0);
    bool isFront = rast.FrontCounterClockwise ? !isCW : isCW;

    if (rast.CullMode == D3D12_CULL_MODE_BACK && !isFront) return;
    if (rast.CullMode == D3D12_CULL_MODE_FRONT && isFront) return;

    int64_t minFx = std::min({fx0, fx1, fx2});
    int64_t maxFx = std::max({fx0, fx1, fx2});
    int64_t minFy = std::min({fy0, fy1, fy2});
    int64_t maxFy = std::max({fy0, fy1, fy2});

    int32_t minX = std::max<int32_t>(scissor.left, static_cast<int32_t>((minFx - 128 + 255) >> 8));
    int32_t maxX = std::min<int32_t>(scissor.right, static_cast<int32_t>((maxFx - 128) >> 8) + 1);
    int32_t minY = std::max<int32_t>(scissor.top, static_cast<int32_t>((minFy - 128 + 255) >> 8));
    int32_t maxY = std::min<int32_t>(scissor.bottom, static_cast<int32_t>((maxFy - 128) >> 8) + 1);

    if (minX >= maxX || minY >= maxY) return;

    int64_t vxA[3] = {fx1, fx2, fx0};
    int64_t vyA[3] = {fy1, fy2, fy0};
    int64_t vxB[3] = {fx2, fx0, fx1};
    int64_t vyB[3] = {fy2, fy0, fy1};

    SetupTriangle tri{};
    tri.minX = minX;
    tri.maxX = maxX;
    tri.minY = minY;
    tri.maxY = maxY;
    tri.isFrontFace = isFront ? 1u : 0u;
    tri.primitiveId = primitiveId;
    tri.viewportIndex = viewportIndex;
    tri.shadingRate = shadingRate;

    int64_t signFlip = isCW ? 1 : -1;
    int64_t posArea256 = signedArea256 * signFlip;
    tri.invArea = 1.0f / static_cast<float>(posArea256);

    for (int k = 0; k < 3; ++k) {
        int64_t dx = (vxB[k] - vxA[k]) * signFlip;
        int64_t dy = (vyB[k] - vyA[k]) * signFlip;
        tri.stepX[k] = -256 * dy;
        tri.stepY[k] = 256 * dx;
        tri.base[k] = dx * (128 - vyA[k]) - dy * (128 - vxA[k]);
        bool isTopLeft = (dy < 0) || (dy == 0 && dx > 0);
        tri.bias[k] = isTopLeft ? 0 : -1;
    }

    float depthBiasVal = evalDepthBias(rast, posArea256, sx0, sy0, sz0, sx1, sy1, sz1, sx2, sy2, sz2);

    tri.invW[0] = iw0;
    tri.invW[1] = iw1 - iw0;
    tri.invW[2] = iw2 - iw0;

    float z0 = sz0 + depthBiasVal;
    float z1 = sz1 + depthBiasVal;
    float z2 = sz2 + depthBiasVal;
    tri.depth[0] = z0;
    tri.depth[1] = z1 - z0;
    tri.depth[2] = z2 - z0;

    if (sceneCfg.needsOrigBary) {
        size_t obOff = target.origBaryArena->size();
        target.origBaryArena->resize(obOff + 9);
        float* obDst = target.origBaryArena->data() + obOff;
        for (int k = 0; k < 3; ++k) {
            float b0 = unclipped ? (k == 0 ? 1.0f : 0.0f) : v0.origBary[k];
            float b1 = unclipped ? (k == 1 ? 1.0f : 0.0f) : v1.origBary[k];
            float b2 = unclipped ? (k == 2 ? 1.0f : 0.0f) : v2.origBary[k];
            float b0w = b0 * iw0;
            float b1w = b1 * iw1;
            float b2w = b2 * iw2;
            obDst[k * 3 + 0] = b0w;
            obDst[k * 3 + 1] = b1w - b0w;
            obDst[k * 3 + 2] = b2w - b0w;
        }
    }

    const uint32_t numScalars = sceneCfg.numPSInputScalars;
    if (numScalars > 0) {
        size_t baseOff = target.attrBaseArena->size();
        target.attrBaseArena->resize(baseOff + static_cast<size_t>(numScalars) * 3);
        float* baseDst = target.attrBaseArena->data() + baseOff;

        float* flatDst = nullptr;
        if (sceneCfg.needsFlatAttrs) {
            size_t flatOff = target.attrFlatArena->size();
            target.attrFlatArena->resize(flatOff + numScalars);
            flatDst = target.attrFlatArena->data() + flatOff;
        }

        for (uint32_t s = 0; s < numScalars; ++s) {
            int16_t srcSlot = sceneCfg.psInputFromPreRastSlot[s];
            float raw0 = 0.0f, raw1 = 0.0f, raw2 = 0.0f;
            float orig0 = 0.0f, orig1 = 0.0f, orig2 = 0.0f;
            if (srcSlot >= 0 && static_cast<uint32_t>(srcSlot) < kMaxPSInputScalars) {
                getTriAttrs(static_cast<uint32_t>(srcSlot), orig0, orig1, orig2);
                if (unclipped) {
                    raw0 = orig0;
                    raw1 = orig1;
                    raw2 = orig2;
                } else {
                    raw0 = v0.origBary[0] * orig0 + v0.origBary[1] * orig1 + v0.origBary[2] * orig2;
                    raw1 = v1.origBary[0] * orig0 + v1.origBary[1] * orig1 + v1.origBary[2] * orig2;
                    raw2 = v2.origBary[0] * orig0 + v2.origBary[1] * orig1 + v2.origBary[2] * orig2;
                }
            }
            float a0 = raw0 * iw0;
            float a1 = raw1 * iw1;
            float a2 = raw2 * iw2;
            baseDst[s * 3 + 0] = a0;
            baseDst[s * 3 + 1] = a1 - a0;
            baseDst[s * 3 + 2] = a2 - a0;
            if (flatDst) {
                flatDst[s] = orig0;
            }
        }
    }

    uint32_t localTriIdx = static_cast<uint32_t>(target.triangles->size());
    target.triangles->push_back(tri);
    target.recordBinIfActive(tri, localTriIdx);
}

template <typename GetTriAttrsFn>
void clipAndEmitBaryTriangle(
    const ClipVertex& cv0,
    const ClipVertex& cv1,
    const ClipVertex& cv2,
    GetTriAttrsFn&& getTriAttrs,
    uint32_t primitiveId,
    uint32_t vpIdx,
    uint32_t shadingRate,
    const D3D12_VIEWPORT& vp,
    const D3D12_RECT& scissor,
    const D3D12_RASTERIZER_DESC& rast,
    const RasterScene& sceneCfg,
    TriangleEmitTarget& target) {
    uint8_t outMask = 0;
    auto markOut = [&](const ClipVertex& v) {
        if (v.clipW < 1e-6f) outMask |= 0x01u;
        if (v.clipX < -v.clipW) outMask |= 0x02u;
        if (v.clipX >  v.clipW) outMask |= 0x04u;
        if (v.clipY < -v.clipW) outMask |= 0x08u;
        if (v.clipY >  v.clipW) outMask |= 0x10u;
        if (v.clipZ <  0.0f)    outMask |= 0x20u;
        if (v.clipZ >  v.clipW) outMask |= 0x40u;
    };
    markOut(cv0);
    markOut(cv1);
    markOut(cv2);

    if (outMask == 0 && sceneCfg.clipDistMask == 0) {
        setupClippedBaryTriangle(
            cv0, cv1, cv2, getTriAttrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, sceneCfg, target, /*unclipped=*/true);
        return;
    }

    ClipPoly polyBufA, polyBufB;
    polyBufA.v[0] = cv0;
    polyBufA.v[1] = cv1;
    polyBufA.v[2] = cv2;
    polyBufA.count = 3;

    ClipPoly* cur = &polyBufA;
    ClipPoly* nxt = &polyBufB;
    auto runPlane = [&](uint8_t bit, auto&& eval) {
        if ((outMask & bit) == 0) return true;
        clipClipPolyPlane(*cur, *nxt, eval);
        std::swap(cur, nxt);
        return cur->count >= 3;
    };

    if (!runPlane(0x01u, [](const ClipVertex& v) { return v.clipW - 1e-6f; })) return;
    if (!runPlane(0x02u, [](const ClipVertex& v) { return v.clipX + v.clipW; })) return;
    if (!runPlane(0x04u, [](const ClipVertex& v) { return v.clipW - v.clipX; })) return;
    if (!runPlane(0x08u, [](const ClipVertex& v) { return v.clipY + v.clipW; })) return;
    if (!runPlane(0x10u, [](const ClipVertex& v) { return v.clipW - v.clipY; })) return;
    if (!runPlane(0x20u, [](const ClipVertex& v) { return v.clipZ; })) return;
    if (!runPlane(0x40u, [](const ClipVertex& v) { return v.clipW - v.clipZ; })) return;

    if (sceneCfg.clipDistMask != 0) {
        for (uint32_t k = 0; k < 8; ++k) {
            if (!(sceneCfg.clipDistMask & (1u << k))) continue;
            uint32_t s = sceneCfg.clipAttrOffset + k;
            if (s >= kMaxPSInputScalars) continue;
            float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f;
            getTriAttrs(s, a0, a1, a2);
            if (a0 >= 0.0f && a1 >= 0.0f && a2 >= 0.0f) continue;
            if (a0 < 0.0f && a1 < 0.0f && a2 < 0.0f) return;
            clipClipPolyPlane(*cur, *nxt, [a0, a1, a2](const ClipVertex& v) {
                return v.origBary[0] * a0 + v.origBary[1] * a1 + v.origBary[2] * a2;
            });
            std::swap(cur, nxt);
            if (cur->count < 3) return;
        }
    }

    for (uint32_t k = 1; k + 1 < cur->count; ++k) {
        setupClippedBaryTriangle(
            cur->v[0], cur->v[k], cur->v[k + 1], getTriAttrs,
            primitiveId, vpIdx, shadingRate, vp, scissor, rast, sceneCfg, target, /*unclipped=*/false);
    }
}

void processTriangle(
    const ShadedVertex& v0,
    const ShadedVertex& v1,
    const ShadedVertex& v2,
    uint32_t primitiveId,
    const D3D12_RASTERIZER_DESC& rast,
    const RasterScene& sceneCfg,
    TriangleEmitTarget& target,
    std::vector<ShadedVertex>& /*polyA*/,
    std::vector<ShadedVertex>& /*polyB*/) {
    uint32_t rawVpIdx = 0;
    if (sceneCfg.hasViewportArrayIndex && sceneCfg.vpIndexAttrOffset < kMaxPSInputScalars) {
        std::memcpy(&rawVpIdx, &v0.attrs[sceneCfg.vpIndexAttrOffset], sizeof(rawVpIdx));
    }
    uint32_t vpIdx = (rawVpIdx < sceneCfg.numViewports) ? rawVpIdx : 0u;
    uint32_t scIdx = (rawVpIdx < sceneCfg.numScissors) ? rawVpIdx : 0u;
    const D3D12_VIEWPORT& vp = sceneCfg.viewports[vpIdx];
    const D3D12_RECT& scissor = sceneCfg.scissors[scIdx];

    uint32_t primSR = 0;
    if (sceneCfg.hasPerPrimShadingRate && sceneCfg.shadingRateAttrOffset < kMaxPSInputScalars) {
        std::memcpy(&primSR, &v0.attrs[sceneCfg.shadingRateAttrOffset], sizeof(primSR));
    }
    uint32_t shadingRate = combineShadingRate(sceneCfg.drawShadingRate, primSR, sceneCfg.shadingRateCombiner);

    if (sceneCfg.cullDistMask != 0) {
        for (uint32_t k = 0; k < 8; ++k) {
            if (!(sceneCfg.cullDistMask & (1u << k))) continue;
            uint32_t s = sceneCfg.cullAttrOffset + k;
            if (s < kMaxPSInputScalars && v0.attrs[s] < 0.0f && v1.attrs[s] < 0.0f && v2.attrs[s] < 0.0f) {
                return;
            }
        }
    }

    auto allOut = [&](auto&& pred) {
        return pred(v0) && pred(v1) && pred(v2);
    };
    if (allOut([](const ShadedVertex& v) { return v.clipX < -v.clipW; }) ||
        allOut([](const ShadedVertex& v) { return v.clipX > v.clipW; }) ||
        allOut([](const ShadedVertex& v) { return v.clipY < -v.clipW; }) ||
        allOut([](const ShadedVertex& v) { return v.clipY > v.clipW; }) ||
        allOut([](const ShadedVertex& v) { return v.clipZ < 0.0f; }) ||
        allOut([](const ShadedVertex& v) { return v.clipZ > v.clipW; }) ||
        allOut([](const ShadedVertex& v) { return v.clipW <= 0.0f; })) {
        return;
    }

    auto getTriAttrs = [&](uint32_t srcSlot, float& a0, float& a1, float& a2) {
        a0 = v0.attrs[srcSlot];
        a1 = v1.attrs[srcSlot];
        a2 = v2.attrs[srcSlot];
    };

    ClipVertex cv0{v0.clipX, v0.clipY, v0.clipZ, v0.clipW, {1.0f, 0.0f, 0.0f}};
    ClipVertex cv1{v1.clipX, v1.clipY, v1.clipZ, v1.clipW, {0.0f, 1.0f, 0.0f}};
    ClipVertex cv2{v2.clipX, v2.clipY, v2.clipZ, v2.clipW, {0.0f, 0.0f, 1.0f}};

    auto insideGuardBand = [&](const ShadedVertex& v) {
        if (!(v.clipW > 1e-6f && v.clipZ >= 0.0f && v.clipZ <= v.clipW)) {
            return false;
        }
        float iw = 1.0f / v.clipW;
        float sx = vp.TopLeftX + (v.clipX * iw * 0.5f + 0.5f) * vp.Width;
        float sy = vp.TopLeftY + (-v.clipY * iw * 0.5f + 0.5f) * vp.Height;
        if (!(sx >= -16384.0f && sx <= 16384.0f && sy >= -16384.0f && sy <= 16384.0f)) {
            return false;
        }
        if (sceneCfg.clipDistMask != 0) {
            for (uint32_t k = 0; k < 8; ++k) {
                if (!(sceneCfg.clipDistMask & (1u << k))) continue;
                uint32_t s = sceneCfg.clipAttrOffset + k;
                if (s < kMaxPSInputScalars && v.attrs[s] < 0.0f) return false;
            }
        }
        return true;
    };
    if (insideGuardBand(v0) && insideGuardBand(v1) && insideGuardBand(v2)) {
        setupClippedBaryTriangle(
            cv0, cv1, cv2, getTriAttrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, sceneCfg, target, /*unclipped=*/true);
        return;
    }

    clipAndEmitBaryTriangle(
        cv0, cv1, cv2, getTriAttrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, sceneCfg, target);
}

void executeVertexTriangleChunk(
    uint64_t globalPrimStart,
    uint64_t globalPrimEnd,
    uint32_t primsPerInstance,
    bool isStrip,
    const uint32_t* prebuiltIndexedStripVIds,
    const FastIndexReader& ibReader,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw,
    CompiledVertexFn vsFn,
    uint32_t W,
    const D3D12_RASTERIZER_DESC& rast,
    const RasterScene& sceneCfg,
    TriangleEmitTarget& target) {
    if (globalPrimStart >= globalPrimEnd || primsPerInstance == 0) return;
    ScopedFlushDenormals flushDenorms;
    thread_local ChunkTransientScratch scratch;

    const uint32_t chunkPrimCount = static_cast<uint32_t>(globalPrimEnd - globalPrimStart);
    const uint32_t vsAttrScalars = sceneCfg.preRastScalarAttrs;
    const uint32_t vsAttrAllocScalars = std::max<uint32_t>(1u, vsAttrScalars);

    uint32_t instIdx = static_cast<uint32_t>(globalPrimStart / primsPerInstance);
    uint32_t primInInst = static_cast<uint32_t>(globalPrimStart % primsPerInstance);
    uint32_t curInstId = draw.startInstanceLocation + instIdx;

    uint32_t numUnique = 0;
    if (!draw.indexed && !isStrip) {
        // Non-indexed TRIANGLELIST: every vertex is unique and sequential; bypass vertex cache completely.
        for (uint32_t lp = 0; lp < chunkPrimCount; ++lp) {
            uint32_t baseV = draw.startIndexOrVertex + primInInst * 3u;
            uint16_t uBase = static_cast<uint16_t>(lp * 3u);
            scratch.uniqueVIds[uBase + 0] = baseV + 0u;
            scratch.uniqueVIds[uBase + 1] = baseV + 1u;
            scratch.uniqueVIds[uBase + 2] = baseV + 2u;
            scratch.uniqueIIds[uBase + 0] = curInstId;
            scratch.uniqueIIds[uBase + 1] = curInstId;
            scratch.uniqueIIds[uBase + 2] = curInstId;
            scratch.localTriVerts[lp * 3u + 0] = uBase + 0u;
            scratch.localTriVerts[lp * 3u + 1] = uBase + 1u;
            scratch.localTriVerts[lp * 3u + 2] = uBase + 2u;
            scratch.localPrimIds[lp] = primInInst;

            if (++primInInst == primsPerInstance) {
                primInInst = 0;
                ++instIdx;
                curInstId = draw.startInstanceLocation + instIdx;
            }
        }
        numUnique = chunkPrimCount * 3u;
    } else {
        // Indexed or TriangleStrip: L1-resident 2048-entry generation-tagged post-transform vertex cache.
        uint32_t gen = scratch.advanceGen();
        auto lookupOrInsert = [&](uint32_t vid, uint32_t iid) -> uint16_t {
            uint32_t slot = (vid * 2654435761u) & kVtxCacheMask;
            if (scratch.cacheGen[slot] == gen && scratch.cacheVId[slot] == vid) {
                return scratch.cacheLocalIdx[slot];
            }
            uint16_t uIdx = static_cast<uint16_t>(numUnique++);
            scratch.uniqueVIds[uIdx] = vid;
            scratch.uniqueIIds[uIdx] = iid;
            scratch.cacheGen[slot] = gen;
            scratch.cacheVId[slot] = vid;
            scratch.cacheLocalIdx[slot] = uIdx;
            return uIdx;
        };

        for (uint32_t lp = 0; lp < chunkPrimCount; ++lp) {
            uint32_t v0 = 0, v1 = 0, v2 = 0;
            if (prebuiltIndexedStripVIds) {
                const uint32_t* triSrc = prebuiltIndexedStripVIds + static_cast<size_t>(primInInst) * 3u;
                v0 = triSrc[0];
                v1 = triSrc[1];
                v2 = triSrc[2];
            } else if (!draw.indexed) {
                // Non-indexed TRIANGLESTRIP
                bool odd = (primInInst & 1u) != 0;
                v0 = draw.startIndexOrVertex + primInInst;
                v1 = draw.startIndexOrVertex + primInInst + (odd ? 2u : 1u);
                v2 = draw.startIndexOrVertex + primInInst + (odd ? 1u : 2u);
            } else {
                // Indexed TRIANGLELIST
                uint32_t ibOff = draw.startIndexOrVertex + primInInst * 3u;
                int32_t baseVtx = draw.baseVertexLocation;
                v0 = static_cast<uint32_t>(static_cast<int32_t>(ibReader.readNoCut(ibOff + 0u)) + baseVtx);
                v1 = static_cast<uint32_t>(static_cast<int32_t>(ibReader.readNoCut(ibOff + 1u)) + baseVtx);
                v2 = static_cast<uint32_t>(static_cast<int32_t>(ibReader.readNoCut(ibOff + 2u)) + baseVtx);
            }

            scratch.localTriVerts[lp * 3u + 0] = lookupOrInsert(v0, curInstId);
            scratch.localTriVerts[lp * 3u + 1] = lookupOrInsert(v1, curInstId);
            scratch.localTriVerts[lp * 3u + 2] = lookupOrInsert(v2, curInstId);
            scratch.localPrimIds[lp] = primInInst;

            if (++primInInst == primsPerInstance) {
                primInInst = 0;
                ++instIdx;
                curInstId = draw.startInstanceLocation + instIdx;
                gen = scratch.advanceGen(); // Invalidate post-transform cache across instance boundary in O(1)
            }
        }
    }

    if (numUnique == 0) return;

    const uint32_t numBatches = (numUnique + W - 1u) / W;
    const size_t neededAttrFloats = static_cast<size_t>(numBatches) * vsAttrAllocScalars * W;
    if (scratch.attrsSOA.size() < neededAttrFloats) {
        scratch.attrsSOA.resize(neededAttrFloats);
    }

    // Pad tail lanes of the final SIMD packet for contiguous W-wide vector loads in @nacelle_vs_*
    const uint32_t paddedEnd = numBatches * W;
    const uint32_t lastBase = (numBatches - 1u) * W;
    for (uint32_t u = numUnique; u < paddedEnd; ++u) {
        scratch.uniqueVIds[u] = scratch.uniqueVIds[lastBase];
        scratch.uniqueIIds[u] = scratch.uniqueIIds[lastBase];
    }

    // 1. Shade unique vertices in W-wide SIMD packets (outputs stay in SOA!)
    for (uint32_t b = 0; b < numBatches; ++b) {
        uint32_t base = b * W;
        uint32_t active = std::min<uint32_t>(W, numUnique - base);
        vsFn(
            &state.rootCtx,
            &scratch.uniqueVIds[base],
            &scratch.uniqueIIds[base],
            active,
            &scratch.posSOA[base * 4u],
            &scratch.attrsSOA[static_cast<size_t>(b) * vsAttrAllocScalars * W]);
    }

    // 2. Fused once-per-unique-vertex post-VS epilogue: frustum outcodes, guard-band check, 1/w, viewport & 24.8 snap
    const bool specialPerPrimState =
        sceneCfg.hasViewportArrayIndex || sceneCfg.hasPerPrimShadingRate ||
        (sceneCfg.clipDistMask != 0) || (sceneCfg.cullDistMask != 0);
    const D3D12_VIEWPORT& vp0 = sceneCfg.viewports[0];
    const D3D12_RECT& scissor0 = sceneCfg.scissors[0];

    const float vpHalfW = vp0.Width * 0.5f;
    const float vpHalfH = vp0.Height * 0.5f;
    const float vpCenterX = vp0.TopLeftX + vpHalfW;
    const float vpCenterY = vp0.TopLeftY + vpHalfH;
    const float vpDepthRange = vp0.MaxDepth - vp0.MinDepth;

    for (uint32_t b = 0; b < numBatches; ++b) {
        uint32_t base = b * W;
        uint32_t active = std::min<uint32_t>(W, numUnique - base);
        const float* __restrict posX = &scratch.posSOA[base * 4u + 0u * W];
        const float* __restrict posY = &scratch.posSOA[base * 4u + 1u * W];
        const float* __restrict posZ = &scratch.posSOA[base * 4u + 2u * W];
        const float* __restrict posW = &scratch.posSOA[base * 4u + 3u * W];
        uint8_t* __restrict outcodesDst = &scratch.outcodes[base];
        float* __restrict invWDst = &scratch.invW[base];
        float* __restrict screenXDst = &scratch.screenX[base];
        float* __restrict screenYDst = &scratch.screenY[base];
        float* __restrict screenZDst = &scratch.screenZ[base];
        int64_t* __restrict fixedXDst = &scratch.fixedX[base];
        int64_t* __restrict fixedYDst = &scratch.fixedY[base];

        if (!specialPerPrimState) {
            // Branchless SIMD-friendly epilogue over W-wide SOA packet (tail lanes are padded).
            for (uint32_t l = 0; l < W; ++l) {
                float cx = posX[l];
                float cy = posY[l];
                float cz = posZ[l];
                float cw = posW[l];

                uint8_t oc = 0;
                oc |= (cx < -cw) ? 0x01u : 0u;
                oc |= (cx >  cw) ? 0x02u : 0u;
                oc |= (cy < -cw) ? 0x04u : 0u;
                oc |= (cy >  cw) ? 0x08u : 0u;
                oc |= (cz < 0.0f) ? 0x10u : 0u;
                oc |= (cz >  cw) ? 0x20u : 0u;
                oc |= (!(cw > 0.0f)) ? 0x40u : 0u;

                bool validW = (cw > 1e-6f);
                float safeW = validW ? cw : 1.0f;
                float iw = 1.0f / safeW;
                float ndcX = cx * iw;
                float ndcY = cy * iw;
                float ndcZ = cz * iw;
                float sx = vpCenterX + ndcX * vpHalfW;
                float sy = vpCenterY - ndcY * vpHalfH;
                float sz = vp0.MinDepth + ndcZ * vpDepthRange;

                bool inGuard = validW && (cz >= 0.0f) && (cz <= cw) &&
                               (sx >= -16384.0f) && (sx <= 16384.0f) &&
                               (sy >= -16384.0f) && (sy <= 16384.0f);
                oc |= inGuard ? 0x80u : 0u;

                float scaledX = inGuard ? (sx * 256.0f) : 0.0f;
                float scaledY = inGuard ? (sy * 256.0f) : 0.0f;
                int32_t fx32 = static_cast<int32_t>(scaledX + (scaledX >= 0.0f ? 0.5f : -0.5f));
                int32_t fy32 = static_cast<int32_t>(scaledY + (scaledY >= 0.0f ? 0.5f : -0.5f));

                outcodesDst[l] = oc;
                invWDst[l] = iw;
                screenXDst[l] = sx;
                screenYDst[l] = sy;
                screenZDst[l] = sz;
                fixedXDst[l] = static_cast<int64_t>(fx32);
                fixedYDst[l] = static_cast<int64_t>(fy32);
            }
        } else {
            for (uint32_t l = 0; l < active; ++l) {
                float cx = posX[l];
                float cy = posY[l];
                float cz = posZ[l];
                float cw = posW[l];

                uint8_t oc = 0;
                if (cx < -cw) oc |= 0x01u;
                if (cx >  cw) oc |= 0x02u;
                if (cy < -cw) oc |= 0x04u;
                if (cy >  cw) oc |= 0x08u;
                if (cz < 0.0f) oc |= 0x10u;
                if (cz >  cw) oc |= 0x20u;
                if (!(cw > 0.0f)) oc |= 0x40u;
                outcodesDst[l] = oc;
            }
        }
    }

    // 3. Triangle assembly + early culling BEFORE touching attribute SOA
    const bool cullBack = (rast.CullMode == D3D12_CULL_MODE_BACK);
    const bool cullFront = (rast.CullMode == D3D12_CULL_MODE_FRONT);
    const bool frontCCW = (rast.FrontCounterClockwise != FALSE);
    const uint32_t defaultShadingRate =
        combineShadingRate(sceneCfg.drawShadingRate, 0u, sceneCfg.shadingRateCombiner);
    const uint32_t numPSScalars = sceneCfg.numPSInputScalars;
    const bool needsFlat = sceneCfg.needsFlatAttrs;
    const bool needsOrigBary = sceneCfg.needsOrigBary;
    const size_t batchAttrStride = static_cast<size_t>(vsAttrAllocScalars) * W;

    auto unpackFallbackVertex = [&](uint32_t u, ShadedVertex& sv) {
        uint32_t b = u / W;
        uint32_t l = u % W;
        const float* pBase = &scratch.posSOA[b * W * 4u];
        sv.clipX = pBase[0u * W + l];
        sv.clipY = pBase[1u * W + l];
        sv.clipZ = pBase[2u * W + l];
        sv.clipW = pBase[3u * W + l];
        const float* aBase = &scratch.attrsSOA[static_cast<size_t>(b) * batchAttrStride + l];
        for (uint32_t s = 0; s < vsAttrScalars; ++s) {
            sv.attrs[s] = aBase[static_cast<size_t>(s) * W];
        }
    };

    for (uint32_t lp = 0; lp < chunkPrimCount; ++lp) {
        uint32_t u0 = scratch.localTriVerts[lp * 3u + 0];
        uint32_t u1 = scratch.localTriVerts[lp * 3u + 1];
        uint32_t u2 = scratch.localTriVerts[lp * 3u + 2];

        uint8_t oc0 = scratch.outcodes[u0];
        uint8_t oc1 = scratch.outcodes[u1];
        uint8_t oc2 = scratch.outcodes[u2];

        // Trivial frustum reject across all 6 homogeneous planes + w <= 0
        if (!sceneCfg.clipDistMask && !sceneCfg.cullDistMask && ((oc0 & oc1 & oc2 & 0x7Fu) != 0)) {
            continue;
        }

        if (!specialPerPrimState && ((oc0 & oc1 & oc2 & 0x80u) != 0)) {
            // Fast guard-band path: zero ShadedVertex copies, cull before reading attributes!
            int64_t fx0 = scratch.fixedX[u0], fy0 = scratch.fixedY[u0];
            int64_t fx1 = scratch.fixedX[u1], fy1 = scratch.fixedY[u1];
            int64_t fx2 = scratch.fixedX[u2], fy2 = scratch.fixedY[u2];

            int64_t signedArea256 = (fx1 - fx0) * (fy2 - fy0) - (fy1 - fy0) * (fx2 - fx0);
            if (signedArea256 == 0) continue;

            bool isCW = (signedArea256 > 0);
            bool isFront = frontCCW ? !isCW : isCW;
            if (cullBack && !isFront) continue;
            if (cullFront && isFront) continue;

            int64_t minFx = std::min({fx0, fx1, fx2});
            int64_t maxFx = std::max({fx0, fx1, fx2});
            int64_t minFy = std::min({fy0, fy1, fy2});
            int64_t maxFy = std::max({fy0, fy1, fy2});

            int32_t minX = std::max<int32_t>(scissor0.left, static_cast<int32_t>((minFx - 128 + 255) >> 8));
            int32_t maxX = std::min<int32_t>(scissor0.right, static_cast<int32_t>((maxFx - 128) >> 8) + 1);
            int32_t minY = std::max<int32_t>(scissor0.top, static_cast<int32_t>((minFy - 128 + 255) >> 8));
            int32_t maxY = std::min<int32_t>(scissor0.bottom, static_cast<int32_t>((maxFy - 128) >> 8) + 1);
            if (minX >= maxX || minY >= maxY) continue;

            SetupTriangle tri{};
            tri.minX = minX;
            tri.maxX = maxX;
            tri.minY = minY;
            tri.maxY = maxY;
            tri.isFrontFace = isFront ? 1u : 0u;
            tri.primitiveId = scratch.localPrimIds[lp];
            tri.viewportIndex = 0u;
            tri.shadingRate = defaultShadingRate;

            int64_t signFlip = isCW ? 1 : -1;
            int64_t posArea256 = signedArea256 * signFlip;
            tri.invArea = 1.0f / static_cast<float>(posArea256);

            int64_t vxA[3] = {fx1, fx2, fx0};
            int64_t vyA[3] = {fy1, fy2, fy0};
            int64_t vxB[3] = {fx2, fx0, fx1};
            int64_t vyB[3] = {fy2, fy0, fy1};
            for (int k = 0; k < 3; ++k) {
                int64_t dx = (vxB[k] - vxA[k]) * signFlip;
                int64_t dy = (vyB[k] - vyA[k]) * signFlip;
                tri.stepX[k] = -256 * dy;
                tri.stepY[k] = 256 * dx;
                tri.base[k] = dx * (128 - vyA[k]) - dy * (128 - vxA[k]);
                bool isTopLeft = (dy < 0) || (dy == 0 && dx > 0);
                tri.bias[k] = isTopLeft ? 0 : -1;
            }

            float iw0 = scratch.invW[u0];
            float iw1 = scratch.invW[u1];
            float iw2 = scratch.invW[u2];
            tri.invW[0] = iw0;
            tri.invW[1] = iw1 - iw0;
            tri.invW[2] = iw2 - iw0;

            float sz0 = scratch.screenZ[u0];
            float sz1 = scratch.screenZ[u1];
            float sz2 = scratch.screenZ[u2];
            float depthBiasVal = evalDepthBias(
                rast, posArea256,
                scratch.screenX[u0], scratch.screenY[u0], sz0,
                scratch.screenX[u1], scratch.screenY[u1], sz1,
                scratch.screenX[u2], scratch.screenY[u2], sz2);
            float z0 = sz0 + depthBiasVal;
            float z1 = sz1 + depthBiasVal;
            float z2 = sz2 + depthBiasVal;
            tri.depth[0] = z0;
            tri.depth[1] = z1 - z0;
            tri.depth[2] = z2 - z0;

            if (needsOrigBary) {
                size_t obOff = target.origBaryArena->size();
                target.origBaryArena->resize(obOff + 9);
                float* obDst = target.origBaryArena->data() + obOff;
                obDst[0] = iw0;  obDst[1] = -iw0; obDst[2] = -iw0;
                obDst[3] = 0.0f; obDst[4] = iw1;  obDst[5] = 0.0f;
                obDst[6] = 0.0f; obDst[7] = 0.0f; obDst[8] = iw2;
            }

            if (numPSScalars > 0) {
                size_t baseOff = target.attrBaseArena->size();
                target.attrBaseArena->resize(baseOff + static_cast<size_t>(numPSScalars) * 3);
                float* baseDst = target.attrBaseArena->data() + baseOff;

                float* flatDst = nullptr;
                if (needsFlat) {
                    size_t flatOff = target.attrFlatArena->size();
                    target.attrFlatArena->resize(flatOff + numPSScalars);
                    flatDst = target.attrFlatArena->data() + flatOff;
                }

                const float* aPtr0 = &scratch.attrsSOA[static_cast<size_t>(u0 / W) * batchAttrStride + (u0 % W)];
                const float* aPtr1 = &scratch.attrsSOA[static_cast<size_t>(u1 / W) * batchAttrStride + (u1 % W)];
                const float* aPtr2 = &scratch.attrsSOA[static_cast<size_t>(u2 / W) * batchAttrStride + (u2 % W)];

                for (uint32_t s = 0; s < numPSScalars; ++s) {
                    int16_t vsSlot = sceneCfg.psInputFromPreRastSlot[s];
                    float raw0 = 0.0f, raw1 = 0.0f, raw2 = 0.0f;
                    if (vsSlot >= 0 && static_cast<uint32_t>(vsSlot) < vsAttrScalars) {
                        size_t slotOff = static_cast<size_t>(vsSlot) * W;
                        raw0 = aPtr0[slotOff];
                        raw1 = aPtr1[slotOff];
                        raw2 = aPtr2[slotOff];
                    }
                    float a0 = raw0 * iw0;
                    float a1 = raw1 * iw1;
                    float a2 = raw2 * iw2;
                    baseDst[s * 3 + 0] = a0;
                    baseDst[s * 3 + 1] = a1 - a0;
                    baseDst[s * 3 + 2] = a2 - a0;
                    if (flatDst) {
                        flatDst[s] = raw0;
                    }
                }
            }

            uint32_t localTriIdx = static_cast<uint32_t>(target.triangles->size());
            target.triangles->push_back(tri);
            target.recordBinIfActive(tri, localTriIdx);
            continue;
        }

        // Guard-band straddling or user clip/cull/viewport-array fallback
        if (!specialPerPrimState) {
            auto loadClipVert = [&](uint32_t u, float b0, float b1, float b2) -> ClipVertex {
                const float* pBase = &scratch.posSOA[(u / W) * W * 4u + (u % W)];
                return ClipVertex{pBase[0u * W], pBase[1u * W], pBase[2u * W], pBase[3u * W], {b0, b1, b2}};
            };
            ClipVertex cv0 = loadClipVert(u0, 1.0f, 0.0f, 0.0f);
            ClipVertex cv1 = loadClipVert(u1, 0.0f, 1.0f, 0.0f);
            ClipVertex cv2 = loadClipVert(u2, 0.0f, 0.0f, 1.0f);
            const float* aPtr0 = &scratch.attrsSOA[static_cast<size_t>(u0 / W) * batchAttrStride + (u0 % W)];
            const float* aPtr1 = &scratch.attrsSOA[static_cast<size_t>(u1 / W) * batchAttrStride + (u1 % W)];
            const float* aPtr2 = &scratch.attrsSOA[static_cast<size_t>(u2 / W) * batchAttrStride + (u2 % W)];
            auto getTriAttrs = [&](uint32_t vsSlot, float& a0, float& a1, float& a2) {
                if (vsSlot < vsAttrScalars) {
                    size_t slotOff = static_cast<size_t>(vsSlot) * W;
                    a0 = aPtr0[slotOff];
                    a1 = aPtr1[slotOff];
                    a2 = aPtr2[slotOff];
                } else {
                    a0 = 0.0f;
                    a1 = 0.0f;
                    a2 = 0.0f;
                }
            };
            clipAndEmitBaryTriangle(
                cv0, cv1, cv2, getTriAttrs, scratch.localPrimIds[lp], 0u, defaultShadingRate,
                vp0, scissor0, rast, sceneCfg, target);
            continue;
        }

        ShadedVertex sv0, sv1, sv2;
        unpackFallbackVertex(u0, sv0);
        unpackFallbackVertex(u1, sv1);
        unpackFallbackVertex(u2, sv2);
        std::vector<ShadedVertex> unusedA, unusedB;
        processTriangle(sv0, sv1, sv2, scratch.localPrimIds[lp], rast, sceneCfg, target, unusedA, unusedB);
    }
}

void setupClippedLine(
    const ShadedVertex& v0,
    const ShadedVertex& v1,
    const float* flatAttrsV0,
    uint32_t primitiveId,
    uint32_t viewportIndex,
    uint32_t shadingRate,
    const D3D12_VIEWPORT& vp,
    const D3D12_RECT& scissor,
    const D3D12_RASTERIZER_DESC& rast,
    RasterScene& scene,
    bool unclipped = false) {
    if (v0.clipW <= 1e-7f || v1.clipW <= 1e-7f) return;

    float iw0 = 1.0f / v0.clipW;
    float iw1 = 1.0f / v1.clipW;

    float ndcX0 = v0.clipX * iw0, ndcY0 = v0.clipY * iw0, ndcZ0 = v0.clipZ * iw0;
    float ndcX1 = v1.clipX * iw1, ndcY1 = v1.clipY * iw1, ndcZ1 = v1.clipZ * iw1;

    float sx0 = vp.TopLeftX + (ndcX0 * 0.5f + 0.5f) * vp.Width;
    float sy0 = vp.TopLeftY + (-ndcY0 * 0.5f + 0.5f) * vp.Height;
    float sz0 = vp.MinDepth + ndcZ0 * (vp.MaxDepth - vp.MinDepth);

    float sx1 = vp.TopLeftX + (ndcX1 * 0.5f + 0.5f) * vp.Width;
    float sy1 = vp.TopLeftY + (-ndcY1 * 0.5f + 0.5f) * vp.Height;
    float sz1 = vp.MinDepth + ndcZ1 * (vp.MaxDepth - vp.MinDepth);

    int64_t fx0 = std::llround(static_cast<double>(sx0) * 256.0);
    int64_t fy0 = std::llround(static_cast<double>(sy0) * 256.0);
    int64_t fx1 = std::llround(static_cast<double>(sx1) * 256.0);
    int64_t fy1 = std::llround(static_cast<double>(sy1) * 256.0);
    if (fx0 == fx1 && fy0 == fy1) return;

    float lx0 = static_cast<float>(fx0) * (1.0f / 256.0f);
    float ly0 = static_cast<float>(fy0) * (1.0f / 256.0f);
    float lx1 = static_cast<float>(fx1) * (1.0f / 256.0f);
    float ly1 = static_cast<float>(fy1) * (1.0f / 256.0f);

    int32_t minX = std::max<int32_t>(scissor.left, static_cast<int32_t>(std::floor(std::min(lx0, lx1))) - 1);
    int32_t maxX = std::min<int32_t>(scissor.right, static_cast<int32_t>(std::ceil(std::max(lx0, lx1))) + 2);
    int32_t minY = std::max<int32_t>(scissor.top, static_cast<int32_t>(std::floor(std::min(ly0, ly1))) - 1);
    int32_t maxY = std::min<int32_t>(scissor.bottom, static_cast<int32_t>(std::ceil(std::max(ly0, ly1))) + 2);
    if (minX >= maxX || minY >= maxY) return;

    float depthBiasVal = 0.0f;
    if (rast.DepthBias != 0) {
        depthBiasVal = static_cast<float>(rast.DepthBias) * (1.0f / 16777216.0f);
        if (rast.DepthBiasClamp > 0.0f) {
            depthBiasVal = std::min(depthBiasVal, rast.DepthBiasClamp);
        } else if (rast.DepthBiasClamp < 0.0f) {
            depthBiasVal = std::max(depthBiasVal, rast.DepthBiasClamp);
        }
    }

    SetupTriangle tri{};
    tri.isLine = true;
    tri.lineX0 = lx0;
    tri.lineY0 = ly0;
    tri.lineX1 = lx1;
    tri.lineY1 = ly1;
    tri.minX = minX;
    tri.maxX = maxX;
    tri.minY = minY;
    tri.maxY = maxY;
    tri.isFrontFace = 1u;
    tri.primitiveId = primitiveId;
    tri.viewportIndex = viewportIndex;
    tri.shadingRate = shadingRate;

    tri.invW[0] = iw0;
    tri.invW[1] = iw1 - iw0;
    tri.invW[2] = 0.0f;

    float z0 = sz0 + depthBiasVal;
    float z1 = sz1 + depthBiasVal;
    tri.depth[0] = z0;
    tri.depth[1] = z1 - z0;
    tri.depth[2] = 0.0f;

    if (scene.needsOrigBary) {
        size_t obOff = scene.origBaryArena.size();
        scene.origBaryArena.resize(obOff + 9);
        float* obDst = scene.origBaryArena.data() + obOff;
        for (int k = 0; k < 2; ++k) {
            float b0 = unclipped ? (k == 0 ? 1.0f : 0.0f) : v0.origBary[k];
            float b1 = unclipped ? (k == 1 ? 1.0f : 0.0f) : v1.origBary[k];
            float b0w = b0 * iw0;
            float b1w = b1 * iw1;
            obDst[k * 3 + 0] = b0w;
            obDst[k * 3 + 1] = b1w - b0w;
            obDst[k * 3 + 2] = 0.0f;
        }
        obDst[6] = 0.0f;
        obDst[7] = 0.0f;
        obDst[8] = 0.0f;
    }

    uint32_t numScalars = scene.numPSInputScalars;
    if (numScalars > 0) {
        size_t baseOff = scene.attrBaseArena.size();
        scene.attrBaseArena.resize(baseOff + static_cast<size_t>(numScalars) * 3);
        float* baseDst = scene.attrBaseArena.data() + baseOff;

        float* flatDst = nullptr;
        if (scene.needsFlatAttrs) {
            size_t flatOff = scene.attrFlatArena.size();
            scene.attrFlatArena.resize(flatOff + numScalars);
            flatDst = scene.attrFlatArena.data() + flatOff;
        }

        for (uint32_t s = 0; s < numScalars; ++s) {
            int16_t srcSlot = scene.psInputFromPreRastSlot[s];
            float raw0 = (srcSlot >= 0 && static_cast<uint32_t>(srcSlot) < kMaxPSInputScalars) ? v0.attrs[srcSlot] : 0.0f;
            float raw1 = (srcSlot >= 0 && static_cast<uint32_t>(srcSlot) < kMaxPSInputScalars) ? v1.attrs[srcSlot] : 0.0f;
            float a0 = raw0 * iw0;
            float a1 = raw1 * iw1;
            baseDst[s * 3 + 0] = a0;
            baseDst[s * 3 + 1] = a1 - a0;
            baseDst[s * 3 + 2] = 0.0f;
            if (flatDst) {
                flatDst[s] = (srcSlot >= 0 && static_cast<uint32_t>(srcSlot) < kMaxPSInputScalars)
                                 ? flatAttrsV0[srcSlot]
                                 : 0.0f;
            }
        }
    }

    scene.triangles.push_back(tri);
}

void processLine(
    const ShadedVertex& v0,
    const ShadedVertex& v1,
    uint32_t primitiveId,
    const D3D12_VIEWPORT& /*defaultVp*/,
    const D3D12_RASTERIZER_DESC& rast,
    RasterScene& scene) {
    uint32_t numScalars = scene.preRastScalarAttrs;

    uint32_t rawVpIdx = 0;
    if (scene.hasViewportArrayIndex && scene.vpIndexAttrOffset < kMaxPSInputScalars) {
        std::memcpy(&rawVpIdx, &v0.attrs[scene.vpIndexAttrOffset], sizeof(rawVpIdx));
    }
    uint32_t vpIdx = (rawVpIdx < scene.numViewports) ? rawVpIdx : 0u;
    uint32_t scIdx = (rawVpIdx < scene.numScissors) ? rawVpIdx : 0u;
    const D3D12_VIEWPORT& vp = scene.viewports[vpIdx];
    const D3D12_RECT& scissor = scene.scissors[scIdx];

    uint32_t primSR = 0;
    if (scene.hasPerPrimShadingRate && scene.shadingRateAttrOffset < kMaxPSInputScalars) {
        std::memcpy(&primSR, &v0.attrs[scene.shadingRateAttrOffset], sizeof(primSR));
    }
    uint32_t shadingRate = combineShadingRate(scene.drawShadingRate, primSR, scene.shadingRateCombiner);

    if (scene.cullDistMask != 0) {
        for (uint32_t k = 0; k < 8; ++k) {
            if (!(scene.cullDistMask & (1u << k))) continue;
            uint32_t s = scene.cullAttrOffset + k;
            if (s < kMaxPSInputScalars && v0.attrs[s] < 0.0f && v1.attrs[s] < 0.0f) {
                return;
            }
        }
    }

    auto insideGuardBand = [&](const ShadedVertex& v) {
        if (!(v.clipW > 1e-6f && (!rast.DepthClipEnable || (v.clipZ >= 0.0f && v.clipZ <= v.clipW)))) {
            return false;
        }
        float iw = 1.0f / v.clipW;
        float sx = vp.TopLeftX + (v.clipX * iw * 0.5f + 0.5f) * vp.Width;
        float sy = vp.TopLeftY + (-v.clipY * iw * 0.5f + 0.5f) * vp.Height;
        if (!(sx >= -16384.0f && sx <= 16384.0f && sy >= -16384.0f && sy <= 16384.0f)) {
            return false;
        }
        if (scene.clipDistMask != 0) {
            for (uint32_t k = 0; k < 8; ++k) {
                if (!(scene.clipDistMask & (1u << k))) continue;
                uint32_t s = scene.clipAttrOffset + k;
                if (s < kMaxPSInputScalars && v.attrs[s] < 0.0f) return false;
            }
        }
        return true;
    };
    if (insideGuardBand(v0) && insideGuardBand(v1)) {
        setupClippedLine(v0, v1, v0.attrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, scene, /*unclipped=*/true);
        return;
    }

    ShadedVertex a = v0;
    ShadedVertex b = v1;
    a.origBary[0] = 1.0f; a.origBary[1] = 0.0f; a.origBary[2] = 0.0f;
    b.origBary[0] = 0.0f; b.origBary[1] = 1.0f; b.origBary[2] = 0.0f;

    auto clipSegPlane = [&](auto&& eval) -> bool {
        float d0 = eval(a);
        float d1 = eval(b);
        if (d0 < 0.0f && d1 < 0.0f) return false;
        if (d0 >= 0.0f && d1 >= 0.0f) return true;
        float denom = d0 - d1;
        float t = (std::fabs(denom) > 1e-20f) ? std::clamp(d0 / denom, 0.0f, 1.0f) : 0.0f;
        ShadedVertex interp;
        interp.clipX = a.clipX + t * (b.clipX - a.clipX);
        interp.clipY = a.clipY + t * (b.clipY - a.clipY);
        interp.clipZ = a.clipZ + t * (b.clipZ - a.clipZ);
        interp.clipW = a.clipW + t * (b.clipW - a.clipW);
        for (int k = 0; k < 3; ++k) {
            interp.origBary[k] = a.origBary[k] + t * (b.origBary[k] - a.origBary[k]);
        }
        for (uint32_t s = 0; s < numScalars; ++s) {
            interp.attrs[s] = a.attrs[s] + t * (b.attrs[s] - a.attrs[s]);
        }
        if (d0 < 0.0f) {
            a = interp;
        } else {
            b = interp;
        }
        return true;
    };

    if (!clipSegPlane([](const ShadedVertex& v) { return v.clipW - 1e-6f; })) return;
    if (!clipSegPlane([](const ShadedVertex& v) { return v.clipX + v.clipW; })) return;
    if (!clipSegPlane([](const ShadedVertex& v) { return v.clipW - v.clipX; })) return;
    if (!clipSegPlane([](const ShadedVertex& v) { return v.clipY + v.clipW; })) return;
    if (!clipSegPlane([](const ShadedVertex& v) { return v.clipW - v.clipY; })) return;
    if (rast.DepthClipEnable) {
        if (!clipSegPlane([](const ShadedVertex& v) { return v.clipZ; })) return;
        if (!clipSegPlane([](const ShadedVertex& v) { return v.clipW - v.clipZ; })) return;
    }
    if (scene.clipDistMask != 0) {
        for (uint32_t k = 0; k < 8; ++k) {
            if (!(scene.clipDistMask & (1u << k))) continue;
            uint32_t s = scene.clipAttrOffset + k;
            if (s >= kMaxPSInputScalars) continue;
            if (!clipSegPlane([s](const ShadedVertex& v) { return v.attrs[s]; })) return;
        }
    }

    setupClippedLine(a, b, v0.attrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, scene, /*unclipped=*/false);
}

bool initRasterSceneHeader(
    const GraphicsExecutionState& state,
    const CompiledGraphicsPSO& compiled,
    RasterScene& outScene,
    D3D12_VIEWPORT& outDefaultVp) {
    uint32_t fbW = 0;
    uint32_t fbH = 0;
    for (uint32_t i = 0; i < state.numRTVs; ++i) {
        if (state.rtvs[i].resource) {
            const auto& rd = state.rtvs[i].resource->resourceDesc();
            uint32_t mip = state.rtvs[i].mip_slice;
            fbW = std::max<uint32_t>(fbW, std::max<uint32_t>(1u, static_cast<uint32_t>(rd.Width >> mip)));
            fbH = std::max<uint32_t>(fbH, std::max<uint32_t>(1u, rd.Height >> mip));
        }
    }
    if (state.hasDSV && state.dsv.resource) {
        const auto& dd = state.dsv.resource->resourceDesc();
        uint32_t mip = state.dsv.mip_slice;
        fbW = std::max<uint32_t>(fbW, std::max<uint32_t>(1u, static_cast<uint32_t>(dd.Width >> mip)));
        fbH = std::max<uint32_t>(fbH, std::max<uint32_t>(1u, dd.Height >> mip));
    }
    if (fbW == 0 || fbH == 0) {
        return false;
    }

    outDefaultVp = D3D12_VIEWPORT{0.0f, 0.0f, static_cast<float>(fbW), static_cast<float>(fbH), 0.0f, 1.0f};
    if (state.numViewports > 0) {
        outDefaultVp = state.viewports[0];
    }

    D3D12_RECT defaultScissor{0, 0, static_cast<LONG>(fbW), static_cast<LONG>(fbH)};
    outScene.numViewports = std::max<uint32_t>(1u, std::min<uint32_t>(state.numViewports, 16u));
    outScene.numScissors = std::max<uint32_t>(1u, std::min<uint32_t>(state.numScissors, 16u));
    for (uint32_t i = 0; i < 16; ++i) {
        outScene.viewports[i] = (i < state.numViewports) ? state.viewports[i] : outDefaultVp;
        if (i < state.numScissors) {
            outScene.scissors[i].left = std::max<LONG>(0, state.scissors[i].left);
            outScene.scissors[i].top = std::max<LONG>(0, state.scissors[i].top);
            outScene.scissors[i].right = std::min<LONG>(static_cast<LONG>(fbW), state.scissors[i].right);
            outScene.scissors[i].bottom = std::min<LONG>(static_cast<LONG>(fbH), state.scissors[i].bottom);
        } else {
            outScene.scissors[i] = (state.numScissors > 0) ? outScene.scissors[0] : defaultScissor;
        }
    }
    bool anyNonEmptyScissor = false;
    for (uint32_t i = 0; i < outScene.numScissors; ++i) {
        if (outScene.scissors[i].left < outScene.scissors[i].right &&
            outScene.scissors[i].top < outScene.scissors[i].bottom) {
            anyNonEmptyScissor = true;
            break;
        }
    }
    if (!anyNonEmptyScissor) {
        return false;
    }

    outScene.fbWidth = fbW;
    outScene.fbHeight = fbH;
    outScene.numTilesX = (fbW + kTileSize - 1) / kTileSize;
    outScene.numTilesY = (fbH + kTileSize - 1) / kTileSize;
    outScene.numPSInputScalars = std::min<uint32_t>(compiled.numPSInputRegs * 4u, kMaxPSInputScalars);
    outScene.preRastScalarAttrs = std::min<uint32_t>(
        std::max(compiled.totalVSScalarAttrs, compiled.numPSInputRegs * 4u), kMaxPSInputScalars);
    outScene.clipDistMask = compiled.clipDistMask;
    outScene.cullDistMask = compiled.cullDistMask;
    outScene.clipAttrOffset = compiled.clipAttrOffset;
    outScene.cullAttrOffset = compiled.cullAttrOffset;
    outScene.hasViewportArrayIndex = compiled.hasViewportArrayIndex;
    outScene.vpIndexAttrOffset = compiled.vpIndexAttrOffset;
    outScene.hasPerPrimShadingRate = compiled.hasPerPrimShadingRate;
    outScene.shadingRateAttrOffset = compiled.shadingRateAttrOffset;
    outScene.needsFlatAttrs = (outScene.numPSInputScalars > 0) &&
                              (((compiled.flatInterpMask & 0xFFFFu) != 0) || (compiled.msPrimAttrMask != 0));
    outScene.needsOrigBary = compiled.psUsesPerspBarycentrics;
    outScene.binsAlreadyPopulated = false;
    outScene.drawShadingRate = state.shadingRate;
    outScene.shadingRateCombiner = state.shadingRateCombiners[0];
    outScene.psInputFromPreRastSlot = compiled.psInputFromVSSlot;
    outScene.triangles.clear();
    outScene.attrBaseArena.clear();
    outScene.attrFlatArena.clear();
    outScene.origBaryArena.clear();
    return true;
}

} // namespace

bool buildRasterScene(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw,
    RasterScene& outScene,
    std::vector<TileBin>* optOutBins) {
    outScene.binsAlreadyPopulated = false;
    if (!state.pso || state.pso->isCompute()) return false;
    const auto& compiled = state.pso->graphicsPSO();
    if (!compiled) return false;

    CompiledVertexFn vsFn = compiled->vsFn.load(std::memory_order_acquire);
    if (!vsFn) {
        waitForAllAsyncJITTasks();
        vsFn = compiled->vsFn.load(std::memory_order_acquire);
    }
    if (!vsFn) return false;
    ScopedFlushDenormals flushDenorms;

    // Determine framebuffer dimensions from bound RTVs or DSV
    uint32_t fbW = 0;
    uint32_t fbH = 0;
    for (uint32_t i = 0; i < state.numRTVs; ++i) {
        if (state.rtvs[i].resource) {
            const auto& rd = state.rtvs[i].resource->resourceDesc();
            uint32_t mip = state.rtvs[i].mip_slice;
            fbW = std::max<uint32_t>(fbW, std::max<uint32_t>(1u, static_cast<uint32_t>(rd.Width >> mip)));
            fbH = std::max<uint32_t>(fbH, std::max<uint32_t>(1u, rd.Height >> mip));
        }
    }
    if (state.hasDSV && state.dsv.resource) {
        const auto& dd = state.dsv.resource->resourceDesc();
        uint32_t mip = state.dsv.mip_slice;
        fbW = std::max<uint32_t>(fbW, std::max<uint32_t>(1u, static_cast<uint32_t>(dd.Width >> mip)));
        fbH = std::max<uint32_t>(fbH, std::max<uint32_t>(1u, dd.Height >> mip));
    }
    if (fbW == 0 || fbH == 0 || draw.countPerInstance == 0 || draw.instanceCount == 0) {
        return false;
    }

    D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(fbW), static_cast<float>(fbH), 0.0f, 1.0f};
    if (state.numViewports > 0) {
        vp = state.viewports[0];
    }

    D3D12_RECT defaultScissor{0, 0, static_cast<LONG>(fbW), static_cast<LONG>(fbH)};
    outScene.numViewports = std::max<uint32_t>(1u, std::min<uint32_t>(state.numViewports, 16u));
    outScene.numScissors = std::max<uint32_t>(1u, std::min<uint32_t>(state.numScissors, 16u));
    for (uint32_t i = 0; i < 16; ++i) {
        outScene.viewports[i] = (i < state.numViewports) ? state.viewports[i] : vp;
        if (i < state.numScissors) {
            outScene.scissors[i].left = std::max<LONG>(0, state.scissors[i].left);
            outScene.scissors[i].top = std::max<LONG>(0, state.scissors[i].top);
            outScene.scissors[i].right = std::min<LONG>(static_cast<LONG>(fbW), state.scissors[i].right);
            outScene.scissors[i].bottom = std::min<LONG>(static_cast<LONG>(fbH), state.scissors[i].bottom);
        } else {
            outScene.scissors[i] = (state.numScissors > 0) ? outScene.scissors[0] : defaultScissor;
        }
    }
    bool anyNonEmptyScissor = false;
    for (uint32_t i = 0; i < outScene.numScissors; ++i) {
        if (outScene.scissors[i].left < outScene.scissors[i].right &&
            outScene.scissors[i].top < outScene.scissors[i].bottom) {
            anyNonEmptyScissor = true;
            break;
        }
    }
    if (!anyNonEmptyScissor) {
        return false;
    }

    const bool hasTess = compiled->hasTess;
    const bool hasGS = compiled->hasGS;

    outScene.fbWidth = fbW;
    outScene.fbHeight = fbH;
    outScene.numTilesX = (fbW + kTileSize - 1) / kTileSize;
    outScene.numTilesY = (fbH + kTileSize - 1) / kTileSize;
    outScene.numPSInputScalars = std::min<uint32_t>(compiled->numPSInputRegs * 4u, kMaxPSInputScalars);
    outScene.clipDistMask = compiled->clipDistMask;
    outScene.cullDistMask = compiled->cullDistMask;
    outScene.clipAttrOffset = compiled->clipAttrOffset;
    outScene.cullAttrOffset = compiled->cullAttrOffset;
    outScene.hasViewportArrayIndex = compiled->hasViewportArrayIndex;
    outScene.vpIndexAttrOffset = compiled->vpIndexAttrOffset;
    outScene.hasPerPrimShadingRate = compiled->hasPerPrimShadingRate;
    outScene.shadingRateAttrOffset = compiled->shadingRateAttrOffset;
    outScene.needsFlatAttrs = (outScene.numPSInputScalars > 0) && ((compiled->flatInterpMask & 0xFFFFu) != 0);
    outScene.needsOrigBary = compiled->psUsesPerspBarycentrics;
    outScene.drawShadingRate = state.shadingRate;
    outScene.shadingRateCombiner = state.shadingRateCombiners[0];
    outScene.psInputFromPreRastSlot = compiled->psInputFromVSSlot;

    if (!hasTess && !hasGS) {
        outScene.preRastScalarAttrs = std::min<uint32_t>(compiled->totalVSScalarAttrs, kMaxPSInputScalars);
    } else {
        uint32_t preRastMax = compiled->numPSInputRegs * 4u;
        for (uint32_t k = 0; k < 8; ++k) {
            if (compiled->clipDistMask & (1u << k)) {
                preRastMax = std::max(preRastMax, compiled->clipAttrOffset + k + 1u);
            }
            if (compiled->cullDistMask & (1u << k)) {
                preRastMax = std::max(preRastMax, compiled->cullAttrOffset + k + 1u);
            }
        }
        if (compiled->hasViewportArrayIndex) {
            preRastMax = std::max(preRastMax, compiled->vpIndexAttrOffset + 1u);
        }
        if (compiled->hasPerPrimShadingRate) {
            preRastMax = std::max(preRastMax, compiled->shadingRateAttrOffset + 1u);
        }
        outScene.preRastScalarAttrs = std::min<uint32_t>(preRastMax, kMaxPSInputScalars);
    }

    outScene.triangles.clear();
    outScene.attrBaseArena.clear();
    outScene.attrFlatArena.clear();
    outScene.origBaryArena.clear();

    const uint32_t W = std::clamp<uint32_t>(compiled->simdWidth, 1u, kMaxSimdWidth);
    const D3D12_RASTERIZER_DESC& rast = state.pso->graphicsDesc().RasterizerState;

    const bool isLineTopo = (state.topology == D3D_PRIMITIVE_TOPOLOGY_LINELIST ||
                             state.topology == D3D_PRIMITIVE_TOPOLOGY_LINESTRIP);

    // =========================================================================
    // Fast-path VS -> Rasterizer pipeline (!hasTess && !hasGS && !isLineTopo):
    // Chunked L1-resident vertex shading, post-transform cache, fused post-VS
    // epilogue, early culling before attribute setup, and multi-threaded
    // deterministic geometry + ordered tile binning.
    // =========================================================================
    if (!hasTess && !hasGS && !isLineTopo) {
        const bool isStrip = (state.topology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP ||
                              state.topology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ);
        uint32_t primsPerInstance = 0;
        thread_local std::vector<uint32_t> tlIndexedStripVIds;
        auto& indexedStripVIds = tlIndexedStripVIds;
        indexedStripVIds.clear();

        if (!isStrip) {
            primsPerInstance = draw.countPerInstance / 3u;
        } else if (!draw.indexed) {
            primsPerInstance = (draw.countPerInstance >= 3u) ? (draw.countPerInstance - 2u) : 0u;
        } else {
            // Indexed strip: scan index buffer once to resolve strip-cut restarts and winding parity.
            uint32_t stripRunVerts[2]{};
            uint32_t runLen = 0;
            indexedStripVIds.reserve((draw.countPerInstance >= 3u ? (draw.countPerInstance - 2u) : 0u) * 3u);
            for (uint32_t i = 0; i < draw.countPerInstance; ++i) {
                bool cut = false;
                uint32_t r = readIndex(state.indexBuffer, draw.startIndexOrVertex + i, cut);
                if (cut) {
                    runLen = 0;
                    continue;
                }
                uint32_t vid = static_cast<uint32_t>(static_cast<int32_t>(r) + draw.baseVertexLocation);
                if (runLen < 2u) {
                    stripRunVerts[runLen++] = vid;
                } else {
                    bool odd = ((runLen - 2u) & 1u) != 0;
                    uint32_t a = stripRunVerts[0];
                    uint32_t b = odd ? vid : stripRunVerts[1];
                    uint32_t c = odd ? stripRunVerts[1] : vid;
                    indexedStripVIds.push_back(a);
                    indexedStripVIds.push_back(b);
                    indexedStripVIds.push_back(c);
                    stripRunVerts[0] = stripRunVerts[1];
                    stripRunVerts[1] = vid;
                    ++runLen;
                }
            }
            primsPerInstance = static_cast<uint32_t>(indexedStripVIds.size() / 3u);
        }

        if (primsPerInstance == 0) return false;
        const uint64_t totalPrims = static_cast<uint64_t>(primsPerInstance) * draw.instanceCount;

        FastIndexReader ibReader;
        if (draw.indexed && !isStrip) {
            ibReader.init(state.indexBuffer);
        }
        const uint32_t* stripPtr = indexedStripVIds.empty() ? nullptr : indexedStripVIds.data();

        const char* forceSingleEnv = std::getenv("NACELLE_SINGLE_THREAD_GEOM");
        const bool allowParallel = !(forceSingleEnv && forceSingleEnv[0] == '1');

        if (!allowParallel || totalPrims < kParallelPrimThreshold || pool.get_thread_count() <= 1) {
            TriangleEmitTarget directTarget;
            directTarget.triangles = &outScene.triangles;
            directTarget.attrBaseArena = &outScene.attrBaseArena;
            directTarget.attrFlatArena = &outScene.attrFlatArena;
            directTarget.origBaryArena = &outScene.origBaryArena;

            for (uint64_t start = 0; start < totalPrims; start += kChunkPrims) {
                uint64_t end = std::min<uint64_t>(totalPrims, start + kChunkPrims);
                executeVertexTriangleChunk(
                    start, end, primsPerInstance, isStrip, stripPtr, ibReader,
                    state, draw, vsFn, W, rast, outScene, directTarget);
            }
            return !outScene.triangles.empty();
        }

        // Parallel chunked geometry + lock-free deterministic tile binning
        const uint32_t numChunks = static_cast<uint32_t>((totalPrims + kChunkPrims - 1u) / kChunkPrims);
        const size_t numTiles = static_cast<size_t>(outScene.numTilesX) * outScene.numTilesY;
        const bool fuseBinning = (optOutBins != nullptr) && (numTiles > 0) &&
                                 (static_cast<uint64_t>(numChunks) * numTiles <= kMaxFusedBinMatrixEntries);

        thread_local std::vector<ChunkOutput> tlChunkOutputs;
        auto& chunkOutputs = tlChunkOutputs;
        if (chunkOutputs.size() < numChunks) {
            chunkOutputs.resize(numChunks);
        }
        for (uint32_t c = 0; c < numChunks; ++c) {
            chunkOutputs[c].reset(numTiles, fuseBinning);
        }

        pool.detach_loop<uint32_t>(0, numChunks, [&](uint32_t c) {
            uint64_t start = static_cast<uint64_t>(c) * kChunkPrims;
            uint64_t end = std::min<uint64_t>(totalPrims, start + kChunkPrims);
            ChunkOutput& co = chunkOutputs[c];
            TriangleEmitTarget chunkTarget;
            chunkTarget.triangles = &co.triangles;
            chunkTarget.attrBaseArena = &co.attrBase;
            chunkTarget.attrFlatArena = &co.attrFlat;
            chunkTarget.origBaryArena = &co.origBary;
            if (fuseBinning) {
                chunkTarget.tileCounts = &co.tileCounts;
                chunkTarget.binPairs = &co.binPairs;
                chunkTarget.numTilesX = outScene.numTilesX;
                chunkTarget.numTilesY = outScene.numTilesY;
            }
            executeVertexTriangleChunk(
                start, end, primsPerInstance, isStrip, stripPtr, ibReader,
                state, draw, vsFn, W, rast, outScene, chunkTarget);
        });
        pool.wait();

        thread_local std::vector<size_t> tlChunkTriOffsets;
        auto& chunkTriOffsets = tlChunkTriOffsets;
        chunkTriOffsets.resize(numChunks + 1);
        chunkTriOffsets[0] = 0;
        for (uint32_t c = 0; c < numChunks; ++c) {
            chunkTriOffsets[c + 1] = chunkTriOffsets[c] + chunkOutputs[c].triangles.size();
        }
        const size_t totalTris = chunkTriOffsets[numChunks];
        if (totalTris == 0) {
            if (fuseBinning) {
                optOutBins->resize(numTiles);
                for (auto& b : *optOutBins) b.triIndices.clear();
                outScene.binsAlreadyPopulated = true;
            }
            return false;
        }

        const size_t psScalars = static_cast<size_t>(outScene.numPSInputScalars);
        outScene.triangles.resize(totalTris);
        if (psScalars > 0) {
            outScene.attrBaseArena.resize(totalTris * psScalars * 3);
            if (outScene.needsFlatAttrs) {
                outScene.attrFlatArena.resize(totalTris * psScalars);
            }
        }
        if (outScene.needsOrigBary) {
            outScene.origBaryArena.resize(totalTris * 9);
        }

        thread_local std::vector<uint32_t> tlChunkTileCursors;
        auto& chunkTileCursors = tlChunkTileCursors;
        if (fuseBinning) {
            optOutBins->resize(numTiles);
            chunkTileCursors.resize(static_cast<size_t>(numChunks) * numTiles);
            for (size_t t = 0; t < numTiles; ++t) {
                uint32_t sum = 0;
                for (uint32_t c = 0; c < numChunks; ++c) {
                    chunkTileCursors[static_cast<size_t>(c) * numTiles + t] = sum;
                    sum += chunkOutputs[c].tileCounts[t];
                }
                (*optOutBins)[t].triIndices.resize(sum);
            }
        }

        // Parallel scatter of triangles, arenas, and ordered tile indices into disjoint slices
        pool.detach_loop<uint32_t>(0, numChunks, [&](uint32_t c) {
            const ChunkOutput& co = chunkOutputs[c];
            const size_t count = co.triangles.size();
            if (count == 0) return;

            const size_t triBase = chunkTriOffsets[c];
            std::memcpy(
                outScene.triangles.data() + triBase,
                co.triangles.data(),
                count * sizeof(SetupTriangle));

            if (psScalars > 0) {
                std::memcpy(
                    outScene.attrBaseArena.data() + triBase * psScalars * 3,
                    co.attrBase.data(),
                    count * psScalars * 3 * sizeof(float));
                if (outScene.needsFlatAttrs) {
                    std::memcpy(
                        outScene.attrFlatArena.data() + triBase * psScalars,
                        co.attrFlat.data(),
                        count * psScalars * sizeof(float));
                }
            }
            if (outScene.needsOrigBary) {
                std::memcpy(
                    outScene.origBaryArena.data() + triBase * 9,
                    co.origBary.data(),
                    count * 9 * sizeof(float));
            }

            if (fuseBinning && !co.binPairs.empty()) {
                uint32_t* cursors = &chunkTileCursors[static_cast<size_t>(c) * numTiles];
                const uint32_t baseIdx32 = static_cast<uint32_t>(triBase);
                for (const ChunkBinPair& bp : co.binPairs) {
                    uint32_t dstPos = cursors[bp.tileIdx]++;
                    (*optOutBins)[bp.tileIdx].triIndices[dstPos] = baseIdx32 + bp.localTriIdx;
                }
            }
        });
        pool.wait();

        if (fuseBinning) {
            outScene.binsAlreadyPopulated = true;
        }
        return true;
    }

    // =========================================================================
    // Extended geometry pipeline (Tessellation HS/DS and/or Geometry Shader GS):
    // W-wide SIMD packetized execution across VS, HS, DS, and GS stages.
    // =========================================================================
    CompiledHullCPFn hsFn = hasTess ? compiled->hsFn.load(std::memory_order_acquire) : nullptr;
    CompiledHullPCFn hsPcFn = hasTess ? compiled->hsPcFn.load(std::memory_order_acquire) : nullptr;
    CompiledDomainFn dsFn = hasTess ? compiled->dsFn.load(std::memory_order_acquire) : nullptr;
    CompiledGeometryFn gsFn = hasGS ? compiled->gsFn.load(std::memory_order_acquire) : nullptr;
    if ((hasTess && (!hsFn || !hsPcFn || !dsFn)) || (hasGS && !gsFn)) {
        waitForAllAsyncJITTasks();
        hsFn = hasTess ? compiled->hsFn.load(std::memory_order_acquire) : nullptr;
        hsPcFn = hasTess ? compiled->hsPcFn.load(std::memory_order_acquire) : nullptr;
        dsFn = hasTess ? compiled->dsFn.load(std::memory_order_acquire) : nullptr;
        gsFn = hasGS ? compiled->gsFn.load(std::memory_order_acquire) : nullptr;
    }

    const uint32_t vsOutScalars =
        hasTess ? std::min<uint32_t>(compiled->numHSInputScalars, kMaxPSInputScalars)
                : (hasGS ? std::min<uint32_t>(compiled->numGSInputScalars, kMaxPSInputScalars)
                         : outScene.preRastScalarAttrs);

    uint32_t vertsPerPrim = 3;
    bool isStrip = false;
    if (hasTess) {
        if (state.topology >= D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST &&
            state.topology <= D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST) {
            vertsPerPrim = static_cast<uint32_t>(state.topology - D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST + 1);
        } else {
            vertsPerPrim = std::max<uint32_t>(1, compiled->hsInputCPCount);
        }
    } else {
        switch (state.topology) {
        case D3D_PRIMITIVE_TOPOLOGY_POINTLIST:
            vertsPerPrim = 1;
            break;
        case D3D_PRIMITIVE_TOPOLOGY_LINELIST:
            vertsPerPrim = 2;
            break;
        case D3D_PRIMITIVE_TOPOLOGY_LINESTRIP:
            vertsPerPrim = 2;
            isStrip = true;
            break;
        case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP:
        case D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ:
            vertsPerPrim = 3;
            isStrip = true;
            break;
        default:
            vertsPerPrim = 3;
            break;
        }
    }

    std::vector<uint32_t> triVerts;
    std::vector<uint32_t> uniqueVIds;
    std::unordered_map<uint32_t, uint32_t> vIdToUnique;

    auto internVertex = [&](uint32_t vid) -> uint32_t {
        auto [it, inserted] = vIdToUnique.emplace(vid, static_cast<uint32_t>(uniqueVIds.size()));
        if (inserted) uniqueVIds.push_back(vid);
        return it->second;
    };

    if (!draw.indexed) {
        uint32_t n = draw.countPerInstance;
        if (!isStrip) {
            triVerts.reserve((n / vertsPerPrim) * vertsPerPrim);
            for (uint32_t i = 0; i + vertsPerPrim - 1 < n; i += vertsPerPrim) {
                for (uint32_t k = 0; k < vertsPerPrim; ++k) {
                    triVerts.push_back(internVertex(draw.startIndexOrVertex + i + k));
                }
            }
        } else if (vertsPerPrim == 2) {
            for (uint32_t i = 0; i + 1 < n; ++i) {
                triVerts.push_back(internVertex(draw.startIndexOrVertex + i));
                triVerts.push_back(internVertex(draw.startIndexOrVertex + i + 1));
            }
        } else {
            for (uint32_t i = 0; i + 2 < n; ++i) {
                uint32_t a = internVertex(draw.startIndexOrVertex + i + 0);
                uint32_t b = internVertex(draw.startIndexOrVertex + ((i & 1u) ? (i + 2) : (i + 1)));
                uint32_t c = internVertex(draw.startIndexOrVertex + ((i & 1u) ? (i + 1) : (i + 2)));
                triVerts.push_back(a);
                triVerts.push_back(b);
                triVerts.push_back(c);
            }
        }
    } else {
        uint32_t n = draw.countPerInstance;
        if (!isStrip) {
            triVerts.reserve((n / vertsPerPrim) * vertsPerPrim);
            for (uint32_t i = 0; i + vertsPerPrim - 1 < n; i += vertsPerPrim) {
                for (uint32_t k = 0; k < vertsPerPrim; ++k) {
                    bool cut = false;
                    uint32_t r = readIndex(state.indexBuffer, draw.startIndexOrVertex + i + k, cut);
                    triVerts.push_back(internVertex(static_cast<uint32_t>(static_cast<int32_t>(r) + draw.baseVertexLocation)));
                }
            }
        } else {
            std::vector<uint32_t> stripRun;
            for (uint32_t i = 0; i < n; ++i) {
                bool cut = false;
                uint32_t r = readIndex(state.indexBuffer, draw.startIndexOrVertex + i, cut);
                if (cut) {
                    stripRun.clear();
                    continue;
                }
                stripRun.push_back(static_cast<uint32_t>(static_cast<int32_t>(r) + draw.baseVertexLocation));
                if (vertsPerPrim == 2 && stripRun.size() >= 2) {
                    size_t m = stripRun.size();
                    triVerts.push_back(internVertex(stripRun[m - 2]));
                    triVerts.push_back(internVertex(stripRun[m - 1]));
                } else if (vertsPerPrim == 3 && stripRun.size() >= 3) {
                    size_t m = stripRun.size();
                    bool odd = ((m - 3) & 1u) != 0;
                    uint32_t a = internVertex(stripRun[m - 3]);
                    uint32_t b = internVertex(stripRun[odd ? (m - 1) : (m - 2)]);
                    uint32_t c = internVertex(stripRun[odd ? (m - 2) : (m - 1)]);
                    triVerts.push_back(a);
                    triVerts.push_back(b);
                    triVerts.push_back(c);
                }
            }
        }
    }

    if (triVerts.empty() || uniqueVIds.empty()) return false;

    TriangleEmitTarget sceneTarget;
    sceneTarget.triangles = &outScene.triangles;
    sceneTarget.attrBaseArena = &outScene.attrBaseArena;
    sceneTarget.attrFlatArena = &outScene.attrFlatArena;
    sceneTarget.origBaryArena = &outScene.origBaryArena;

    std::vector<ShadedVertex> shadedVerts(uniqueVIds.size());
    alignas(64) uint32_t batchVIds[kMaxSimdWidth]{};
    alignas(64) uint32_t batchIIds[kMaxSimdWidth]{};
    alignas(64) float outPosXYZW[4 * kMaxSimdWidth]{};
    std::vector<float> outAttrs(std::max<uint32_t>(1, vsOutScalars) * W, 0.0f);
    std::vector<ShadedVertex> polyA, polyB;

    // Reusable Geometry Shader batch buffers (hoisted outside primitive/patch loops)
    const uint32_t gsMaxVerts = std::max<uint32_t>(1, compiled->gsMaxVertexCount);
    const uint32_t gsInstCount = std::max<uint32_t>(1, compiled->gsInstanceCount);
    const uint32_t gsOutScalars = outScene.preRastScalarAttrs;
    std::vector<float> gsInPosSOA;
    std::vector<float> gsInAttrsSOA;
    std::vector<float> gsOutPos;
    std::vector<float> gsOutAttrs;
    std::vector<uint32_t> gsStripCounts;
    std::vector<uint32_t> gsInstTotalVerts;
    std::vector<uint32_t> gsInstNumStrips;
    if (hasGS && gsFn) {
        gsOutPos.resize(static_cast<size_t>(gsInstCount) * W * gsMaxVerts * 4, 0.0f);
        gsOutAttrs.resize(std::max<size_t>(1, static_cast<size_t>(gsInstCount) * W * gsMaxVerts * gsOutScalars), 0.0f);
        gsStripCounts.resize(static_cast<size_t>(gsInstCount) * W * gsMaxVerts, 0u);
        gsInstTotalVerts.resize(static_cast<size_t>(gsInstCount) * kMaxSimdWidth, 0u);
        gsInstNumStrips.resize(static_cast<size_t>(gsInstCount) * kMaxSimdWidth, 0u);
    }

    auto runGSBatch = [&](const ShadedVertex* batchPrims, uint32_t primVertCount,
                          const uint32_t* primIdList, uint32_t activePrims, uint32_t inAttrScalars) {
        if (activePrims == 0 || !gsFn) return;
        gsInPosSOA.assign(static_cast<size_t>(primVertCount) * 4 * W, 0.0f);
        gsInAttrsSOA.assign(std::max<size_t>(1, static_cast<size_t>(primVertCount) * inAttrScalars * W), 0.0f);
        alignas(64) uint32_t paddedPrimIds[kMaxSimdWidth]{};

        for (uint32_t l = 0; l < activePrims; ++l) {
            paddedPrimIds[l] = primIdList[l];
            for (uint32_t k = 0; k < primVertCount; ++k) {
                const ShadedVertex& sv = batchPrims[l * primVertCount + k];
                gsInPosSOA[(k * 4 + 0) * W + l] = sv.clipX;
                gsInPosSOA[(k * 4 + 1) * W + l] = sv.clipY;
                gsInPosSOA[(k * 4 + 2) * W + l] = sv.clipZ;
                gsInPosSOA[(k * 4 + 3) * W + l] = sv.clipW;
                for (uint32_t s = 0; s < inAttrScalars; ++s) {
                    gsInAttrsSOA[(k * inAttrScalars + s) * W + l] = sv.attrs[s];
                }
            }
        }

        const size_t posInstStride = static_cast<size_t>(W) * gsMaxVerts * 4;
        const size_t attrInstStride = static_cast<size_t>(W) * gsMaxVerts * gsOutScalars;
        const size_t stripInstStride = static_cast<size_t>(W) * gsMaxVerts;

        for (uint32_t gInst = 0; gInst < gsInstCount; ++gInst) {
            alignas(64) uint32_t tvVec[kMaxSimdWidth]{};
            alignas(64) uint32_t nsVec[kMaxSimdWidth]{};
            gsFn(
                &state.rootCtx,
                gsInPosSOA.data(),
                gsInAttrsSOA.data(),
                paddedPrimIds,
                gInst,
                activePrims,
                &gsOutPos[gInst * posInstStride],
                gsOutScalars > 0 ? &gsOutAttrs[gInst * attrInstStride] : gsOutAttrs.data(),
                &gsStripCounts[gInst * stripInstStride],
                tvVec,
                nsVec);
            std::memcpy(&gsInstTotalVerts[gInst * kMaxSimdWidth], tvVec, sizeof(tvVec));
            std::memcpy(&gsInstNumStrips[gInst * kMaxSimdWidth], nsVec, sizeof(nsVec));
        }

        // Emit primitives in exact D3D12 (inputPrimitive, gsInstance) stream order
        for (uint32_t l = 0; l < activePrims; ++l) {
            uint32_t primId = paddedPrimIds[l];
            for (uint32_t gInst = 0; gInst < gsInstCount; ++gInst) {
                uint32_t numStrips = gsInstNumStrips[gInst * kMaxSimdWidth + l];
                const float* lanePos = &gsOutPos[gInst * posInstStride + static_cast<size_t>(l) * gsMaxVerts * 4];
                const float* laneAttrs =
                    gsOutScalars > 0
                        ? &gsOutAttrs[gInst * attrInstStride + static_cast<size_t>(l) * gsMaxVerts * gsOutScalars]
                        : nullptr;
                const uint32_t* laneStrips =
                    &gsStripCounts[gInst * stripInstStride + static_cast<size_t>(l) * gsMaxVerts];

                auto makeGsVert = [&](uint32_t vIdx) {
                    ShadedVertex sv;
                    sv.clipX = lanePos[vIdx * 4 + 0];
                    sv.clipY = lanePos[vIdx * 4 + 1];
                    sv.clipZ = lanePos[vIdx * 4 + 2];
                    sv.clipW = lanePos[vIdx * 4 + 3];
                    if (laneAttrs) {
                        for (uint32_t s = 0; s < gsOutScalars; ++s) {
                            sv.attrs[s] = laneAttrs[vIdx * gsOutScalars + s];
                        }
                    }
                    return sv;
                };

                uint32_t baseV = 0;
                for (uint32_t s = 0; s < numStrips; ++s) {
                    uint32_t sc = laneStrips[s];
                    if (compiled->gsOutputTopology == 3 && sc >= 2) { // LineStrip
                        for (uint32_t k = 0; k + 1 < sc; ++k) {
                            processLine(makeGsVert(baseV + k), makeGsVert(baseV + k + 1), primId, vp, rast, outScene);
                        }
                    } else if (compiled->gsOutputTopology == 5 && sc >= 3) { // TriangleStrip
                        for (uint32_t k = 0; k + 2 < sc; ++k) {
                            uint32_t i0 = baseV + k;
                            uint32_t i1 = baseV + k + ((k & 1u) ? 2u : 1u);
                            uint32_t i2 = baseV + k + ((k & 1u) ? 1u : 2u);
                            processTriangle(
                                makeGsVert(i0), makeGsVert(i1), makeGsVert(i2),
                                primId, rast, outScene, sceneTarget, polyA, polyB);
                        }
                    }
                    baseV += sc;
                }
            }
        }
    };

    outScene.triangles.reserve((triVerts.size() / vertsPerPrim) * draw.instanceCount);

    for (uint32_t inst = 0; inst < draw.instanceCount; ++inst) {
        uint32_t instId = draw.startInstanceLocation + inst;
        for (size_t base = 0; base < uniqueVIds.size(); base += W) {
            uint32_t active = static_cast<uint32_t>(std::min<size_t>(W, uniqueVIds.size() - base));
            for (uint32_t l = 0; l < active; ++l) {
                batchVIds[l] = uniqueVIds[base + l];
                batchIIds[l] = instId;
            }
            for (uint32_t l = active; l < W; ++l) {
                batchVIds[l] = uniqueVIds[base];
                batchIIds[l] = instId;
            }

            vsFn(&state.rootCtx, batchVIds, batchIIds, active, outPosXYZW, outAttrs.data());

            for (uint32_t l = 0; l < active; ++l) {
                ShadedVertex& sv = shadedVerts[base + l];
                sv.clipX = outPosXYZW[0 * W + l];
                sv.clipY = outPosXYZW[1 * W + l];
                sv.clipZ = outPosXYZW[2 * W + l];
                sv.clipW = outPosXYZW[3 * W + l];
                for (uint32_t s = 0; s < vsOutScalars; ++s) {
                    sv.attrs[s] = outAttrs[s * W + l];
                }
            }
        }

        if (hasTess && hsFn && hsPcFn && dsFn) {
            uint32_t inCPCount = vertsPerPrim;
            uint32_t outCPCount = std::max<uint32_t>(1, compiled->hsOutputCPCount);
            uint32_t inAttrScalars = std::max<uint32_t>(1, compiled->numHSInputScalars);
            uint32_t outAttrScalars = std::max<uint32_t>(1, compiled->numHSOutputScalars);
            uint32_t pcScalars = std::max<uint32_t>(1, compiled->numPatchConstScalars);
            uint32_t dsOutScalars =
                hasGS ? std::min<uint32_t>(compiled->numGSInputScalars, kMaxPSInputScalars)
                      : outScene.preRastScalarAttrs;

            std::vector<float> inCPAttrs(inCPCount * inAttrScalars, 0.0f);
            std::vector<float> outCPAttrs(outCPCount * outAttrScalars, 0.0f);
            std::vector<float> hsOutCPSOA(outAttrScalars * W, 0.0f);
            std::vector<float> patchConsts(pcScalars, 0.0f);
            float tessFactors[6]{};
            alignas(64) float dsPosSOA[4 * kMaxSimdWidth]{};
            std::vector<float> dsAttrsSOA(std::max<uint32_t>(1, dsOutScalars) * W, 0.0f);
            std::vector<float> patchDomU, patchDomV, patchDomW;
            std::vector<ShadedVertex> patchDSVerts;
            std::vector<uint32_t> triRowOffsets;
            std::vector<ShadedVertex> gsPrimBuf;
            std::vector<uint32_t> gsPrimIdBuf;

            auto partitionTF = [&](float tf) -> uint32_t {
                float clamped = std::clamp(tf, 1.0f, std::max(1.0f, compiled->hsMaxTessFactor));
                switch (compiled->tessPartitioning) {
                case 2: { // pow2
                    float p2 = std::pow(2.0f, std::ceil(std::log2(clamped)));
                    return static_cast<uint32_t>(std::max(1.0f, p2));
                }
                case 3: { // fractional_odd
                    uint32_t c = static_cast<uint32_t>(std::ceil(clamped));
                    return (c > 1 && (c % 2u) == 0u) ? (c + 1u) : std::max<uint32_t>(1, c);
                }
                case 4: { // fractional_even
                    uint32_t c = static_cast<uint32_t>(std::ceil(clamped));
                    return ((c % 2u) != 0u) ? (c + 1u) : std::max<uint32_t>(2, c);
                }
                default:
                    return static_cast<uint32_t>(std::max(1.0f, std::ceil(clamped)));
                }
            };

            uint32_t numPatches = static_cast<uint32_t>(triVerts.size() / vertsPerPrim);
            for (uint32_t p = 0; p < numPatches; ++p) {
                std::fill(inCPAttrs.begin(), inCPAttrs.end(), 0.0f);
                for (uint32_t k = 0; k < inCPCount; ++k) {
                    const ShadedVertex& sv = shadedVerts[triVerts[p * vertsPerPrim + k]];
                    uint32_t copyCount = std::min<uint32_t>(vsOutScalars, inAttrScalars);
                    for (uint32_t s = 0; s < copyCount; ++s) {
                        inCPAttrs[k * inAttrScalars + s] = sv.attrs[s];
                    }
                }
                std::fill(outCPAttrs.begin(), outCPAttrs.end(), 0.0f);
                for (uint32_t baseCp = 0; baseCp < outCPCount; baseCp += W) {
                    uint32_t activeCp = std::min<uint32_t>(W, outCPCount - baseCp);
                    std::fill(hsOutCPSOA.begin(), hsOutCPSOA.end(), 0.0f);
                    hsFn(&state.rootCtx, inCPAttrs.data(), p, baseCp, activeCp, hsOutCPSOA.data());
                    for (uint32_t l = 0; l < activeCp; ++l) {
                        float* dst = &outCPAttrs[(baseCp + l) * outAttrScalars];
                        for (uint32_t s = 0; s < outAttrScalars; ++s) {
                            dst[s] = hsOutCPSOA[s * W + l];
                        }
                    }
                }

                std::fill(patchConsts.begin(), patchConsts.end(), 0.0f);
                std::fill(std::begin(tessFactors), std::end(tessFactors), 0.0f);
                hsPcFn(&state.rootCtx, inCPAttrs.data(), outCPAttrs.data(), p, patchConsts.data(), tessFactors);

                auto evalDSAllPoints = [&]() {
                    const size_t totalPts = patchDomU.size();
                    patchDSVerts.resize(totalPts);
                    alignas(64) float uVec[kMaxSimdWidth]{};
                    alignas(64) float vVec[kMaxSimdWidth]{};
                    alignas(64) float wVec[kMaxSimdWidth]{};

                    for (size_t base = 0; base < totalPts; base += W) {
                        uint32_t active = static_cast<uint32_t>(std::min<size_t>(W, totalPts - base));
                        for (uint32_t l = 0; l < active; ++l) {
                            uVec[l] = patchDomU[base + l];
                            vVec[l] = patchDomV[base + l];
                            wVec[l] = patchDomW[base + l];
                        }
                        for (uint32_t l = active; l < W; ++l) {
                            uVec[l] = uVec[0];
                            vVec[l] = vVec[0];
                            wVec[l] = wVec[0];
                        }
                        std::memset(dsPosSOA, 0, sizeof(dsPosSOA));
                        std::fill(dsAttrsSOA.begin(), dsAttrsSOA.end(), 0.0f);
                        dsFn(
                            &state.rootCtx, outCPAttrs.data(), patchConsts.data(), tessFactors,
                            p, uVec, vVec, wVec, active, dsPosSOA, dsAttrsSOA.data());
                        for (uint32_t l = 0; l < active; ++l) {
                            ShadedVertex& sv = patchDSVerts[base + l];
                            sv = ShadedVertex{};
                            sv.clipX = dsPosSOA[0 * W + l];
                            sv.clipY = dsPosSOA[1 * W + l];
                            sv.clipZ = dsPosSOA[2 * W + l];
                            sv.clipW = dsPosSOA[3 * W + l];
                            for (uint32_t s = 0; s < dsOutScalars; ++s) {
                                sv.attrs[s] = dsAttrsSOA[s * W + l];
                            }
                        }
                    }
                };

                gsPrimBuf.clear();
                gsPrimIdBuf.clear();
                auto flushGSBuf = [&](uint32_t primVCount) {
                    if (gsPrimIdBuf.empty()) return;
                    runGSBatch(gsPrimBuf.data(), primVCount, gsPrimIdBuf.data(),
                               static_cast<uint32_t>(gsPrimIdBuf.size()), dsOutScalars);
                    gsPrimBuf.clear();
                    gsPrimIdBuf.clear();
                };

                auto emitTri = [&](const ShadedVertex& a, const ShadedVertex& b, const ShadedVertex& c) {
                    if (hasGS && gsFn) {
                        gsPrimBuf.push_back(a);
                        gsPrimBuf.push_back(b);
                        gsPrimBuf.push_back(c);
                        gsPrimIdBuf.push_back(p);
                        if (gsPrimIdBuf.size() == W) flushGSBuf(3);
                    } else {
                        processTriangle(a, b, c, p, rast, outScene, sceneTarget, polyA, polyB);
                    }
                };
                auto emitPoint = [&](const ShadedVertex& v) {
                    if (hasGS && gsFn) {
                        gsPrimBuf.push_back(v);
                        gsPrimIdBuf.push_back(p);
                        if (gsPrimIdBuf.size() == W) flushGSBuf(1);
                    }
                };
                const bool pointOut = (compiled->tessOutputPrimitive == 1);

                patchDomU.clear();
                patchDomV.clear();
                patchDomW.clear();

                if (compiled->tessDomain == 1) { // Isoline
                    if (!(tessFactors[0] > 0.0f) || !(tessFactors[1] > 0.0f)) continue;
                    uint32_t numLines = partitionTF(tessFactors[0]);
                    uint32_t numSegs = partitionTF(tessFactors[1]);
                    size_t totalPts = static_cast<size_t>(numLines) * (numSegs + 1);
                    patchDomU.reserve(totalPts);
                    patchDomV.reserve(totalPts);
                    patchDomW.reserve(totalPts);
                    for (uint32_t li = 0; li < numLines; ++li) {
                        float v = static_cast<float>(li) / static_cast<float>(numLines);
                        for (uint32_t si = 0; si <= numSegs; ++si) {
                            float u = static_cast<float>(si) / static_cast<float>(numSegs);
                            patchDomU.push_back(u);
                            patchDomV.push_back(v);
                            patchDomW.push_back(0.0f);
                        }
                    }
                    evalDSAllPoints();
                    if (pointOut) {
                        for (const auto& sv : patchDSVerts) emitPoint(sv);
                        flushGSBuf(1);
                    }
                } else if (compiled->tessDomain == 2) { // Tri
                    if (!(tessFactors[0] > 0.0f) || !(tessFactors[1] > 0.0f) || !(tessFactors[2] > 0.0f)) continue;
                    uint32_t n = std::max({partitionTF(tessFactors[0]), partitionTF(tessFactors[1]),
                                           partitionTF(tessFactors[2]), partitionTF(tessFactors[4])});
                    triRowOffsets.resize(n + 1);
                    uint32_t running = 0;
                    for (uint32_t i = 0; i <= n; ++i) {
                        triRowOffsets[i] = running;
                        running += (n - i + 1);
                    }
                    patchDomU.reserve(running);
                    patchDomV.reserve(running);
                    patchDomW.reserve(running);
                    for (uint32_t i = 0; i <= n; ++i) {
                        for (uint32_t j = 0; j <= n - i; ++j) {
                            float u = static_cast<float>(i) / static_cast<float>(n);
                            float v = static_cast<float>(j) / static_cast<float>(n);
                            float w = std::max(0.0f, 1.0f - u - v);
                            patchDomU.push_back(u);
                            patchDomV.push_back(v);
                            patchDomW.push_back(w);
                        }
                    }
                    evalDSAllPoints();
                    if (pointOut) {
                        for (const auto& sv : patchDSVerts) emitPoint(sv);
                        flushGSBuf(1);
                    } else {
                        bool cw = (compiled->tessOutputPrimitive == 3);
                        auto rowAt = [&](uint32_t i, uint32_t j) -> const ShadedVertex& {
                            return patchDSVerts[triRowOffsets[i] + j];
                        };
                        for (uint32_t i = 0; i < n; ++i) {
                            for (uint32_t j = 0; j < n - i; ++j) {
                                if (cw) {
                                    emitTri(rowAt(i, j), rowAt(i, j + 1), rowAt(i + 1, j));
                                } else {
                                    emitTri(rowAt(i, j), rowAt(i + 1, j), rowAt(i, j + 1));
                                }
                                if (i + j + 1 < n) {
                                    if (cw) {
                                        emitTri(rowAt(i + 1, j), rowAt(i, j + 1), rowAt(i + 1, j + 1));
                                    } else {
                                        emitTri(rowAt(i + 1, j), rowAt(i + 1, j + 1), rowAt(i, j + 1));
                                    }
                                }
                            }
                        }
                        flushGSBuf(3);
                    }
                } else if (compiled->tessDomain == 3) { // Quad
                    if (!(tessFactors[0] > 0.0f) || !(tessFactors[1] > 0.0f) ||
                        !(tessFactors[2] > 0.0f) || !(tessFactors[3] > 0.0f)) {
                        continue;
                    }
                    uint32_t nu = std::max({partitionTF(tessFactors[1]), partitionTF(tessFactors[3]), partitionTF(tessFactors[4])});
                    uint32_t nv = std::max({partitionTF(tessFactors[0]), partitionTF(tessFactors[2]), partitionTF(tessFactors[5])});
                    size_t totalPts = static_cast<size_t>(nv + 1) * (nu + 1);
                    patchDomU.reserve(totalPts);
                    patchDomV.reserve(totalPts);
                    patchDomW.reserve(totalPts);
                    for (uint32_t j = 0; j <= nv; ++j) {
                        float v = static_cast<float>(j) / static_cast<float>(nv);
                        for (uint32_t i = 0; i <= nu; ++i) {
                            float u = static_cast<float>(i) / static_cast<float>(nu);
                            patchDomU.push_back(u);
                            patchDomV.push_back(v);
                            patchDomW.push_back(0.0f);
                        }
                    }
                    evalDSAllPoints();
                    if (pointOut) {
                        for (const auto& sv : patchDSVerts) emitPoint(sv);
                        flushGSBuf(1);
                    } else {
                        bool cw = (compiled->tessOutputPrimitive == 3);
                        for (uint32_t j = 0; j < nv; ++j) {
                            for (uint32_t i = 0; i < nu; ++i) {
                                const ShadedVertex& v00 = patchDSVerts[j * (nu + 1) + i];
                                const ShadedVertex& v10 = patchDSVerts[j * (nu + 1) + (i + 1)];
                                const ShadedVertex& v01 = patchDSVerts[(j + 1) * (nu + 1) + i];
                                const ShadedVertex& v11 = patchDSVerts[(j + 1) * (nu + 1) + (i + 1)];
                                if (cw) {
                                    emitTri(v00, v11, v10);
                                    emitTri(v00, v01, v11);
                                } else {
                                    emitTri(v00, v10, v11);
                                    emitTri(v00, v11, v01);
                                }
                            }
                        }
                        flushGSBuf(3);
                    }
                }
            }
        } else if (hasGS && gsFn) {
            uint32_t numPrims = static_cast<uint32_t>(triVerts.size() / vertsPerPrim);
            std::vector<ShadedVertex> batchPrims(static_cast<size_t>(W) * vertsPerPrim);
            uint32_t batchPrimIds[kMaxSimdWidth]{};
            for (uint32_t baseP = 0; baseP < numPrims; baseP += W) {
                uint32_t activeP = std::min<uint32_t>(W, numPrims - baseP);
                for (uint32_t l = 0; l < activeP; ++l) {
                    uint32_t p = baseP + l;
                    batchPrimIds[l] = p;
                    for (uint32_t k = 0; k < vertsPerPrim; ++k) {
                        batchPrims[l * vertsPerPrim + k] = shadedVerts[triVerts[p * vertsPerPrim + k]];
                    }
                }
                runGSBatch(batchPrims.data(), vertsPerPrim, batchPrimIds, activeP, vsOutScalars);
            }
        } else if (vertsPerPrim == 2) {
            for (size_t t = 0; t + 1 < triVerts.size(); t += 2) {
                processLine(
                    shadedVerts[triVerts[t + 0]],
                    shadedVerts[triVerts[t + 1]],
                    static_cast<uint32_t>(t / 2),
                    vp,
                    rast,
                    outScene);
            }
        }
    }

    return !outScene.triangles.empty();
}

bool buildMeshRasterScene(
    BS::thread_pool<>& /*pool*/,
    const GraphicsExecutionState& state,
    const MeshDispatchParams& dispatch,
    RasterScene& outScene) {
    if (!state.pso || state.pso->isCompute()) return false;
    if (dispatch.groupCountX == 0 || dispatch.groupCountY == 0 || dispatch.groupCountZ == 0) {
        return false;
    }
    const auto& compiled = state.pso->graphicsPSO();
    if (!compiled || !compiled->hasMS) return false;

    CompiledMeshFn msFn = compiled->msFn.load(std::memory_order_acquire);
    CompiledAmplificationFn asFn = compiled->hasAS ? compiled->asFn.load(std::memory_order_acquire) : nullptr;
    if (!msFn || (compiled->hasAS && !asFn)) {
        waitForAllAsyncJITTasks();
        msFn = compiled->msFn.load(std::memory_order_acquire);
        if (compiled->hasAS) {
            asFn = compiled->asFn.load(std::memory_order_acquire);
        }
    }
    if (!msFn || (compiled->hasAS && !asFn)) return false;
    ScopedFlushDenormals flushDenorms;

    D3D12_VIEWPORT vp{};
    if (!initRasterSceneHeader(state, *compiled, outScene, vp)) {
        return false;
    }

    struct PendingMeshDispatch {
        uint32_t gx = 0;
        uint32_t gy = 0;
        uint32_t gz = 0;
        std::vector<uint8_t> payload;
    };
    std::vector<PendingMeshDispatch> meshDispatches;

    const uint32_t payloadBytes = std::max<uint32_t>({64u, compiled->asPayloadBytes, compiled->msPayloadBytes});
    if (compiled->hasAS) {
        std::vector<uint8_t> asScratch(std::max<uint32_t>(64u, compiled->asGroupSharedBytes), 0);
        for (uint32_t az = 0; az < dispatch.groupCountZ; ++az) {
            for (uint32_t ay = 0; ay < dispatch.groupCountY; ++ay) {
                for (uint32_t ax = 0; ax < dispatch.groupCountX; ++ax) {
                    if (compiled->asGroupSharedBytes > 0) {
                        std::memset(asScratch.data(), 0, compiled->asGroupSharedBytes);
                    }
                    uint32_t outDims[4]{0, 0, 0, 0};
                    std::vector<uint8_t> payloadBuf(payloadBytes, 0);

                    AmplificationThreadGroupArgs asArgs{};
                    asArgs.rootCtx = &state.rootCtx;
                    asArgs.groupSharedScratch = asScratch.data();
                    asArgs.gx = ax;
                    asArgs.gy = ay;
                    asArgs.gz = az;
                    asArgs.ngx = dispatch.groupCountX;
                    asArgs.ngy = dispatch.groupCountY;
                    asArgs.ngz = dispatch.groupCountZ;
                    asArgs.outDispatchDims = outDims;
                    asArgs.outPayload = payloadBuf.data();
                    asFn(&asArgs);

                    if (outDims[3] != 0 && outDims[0] > 0 && outDims[1] > 0 && outDims[2] > 0) {
                        meshDispatches.push_back(PendingMeshDispatch{
                            outDims[0], outDims[1], outDims[2], std::move(payloadBuf)});
                    }
                }
            }
        }
    } else {
        PendingMeshDispatch single{};
        single.gx = dispatch.groupCountX;
        single.gy = dispatch.groupCountY;
        single.gz = dispatch.groupCountZ;
        single.payload.assign(payloadBytes, 0);
        meshDispatches.push_back(std::move(single));
    }

    if (meshDispatches.empty()) return false;

    const D3D12_RASTERIZER_DESC& rast = state.pso->graphicsDesc().RasterizerState;
    const uint32_t maxVerts = std::max<uint32_t>(1u, compiled->msMaxVertexCount);
    const uint32_t maxPrims = std::max<uint32_t>(1u, compiled->msMaxPrimitiveCount);
    const uint32_t attrStride = compiled->totalVSScalarAttrs;
    const uint32_t copyScalars = std::min<uint32_t>(attrStride, kMaxPSInputScalars);
    const uint64_t primAttrMask = compiled->msPrimAttrMask;

    std::vector<uint8_t> msScratch(std::max<uint32_t>(64u, compiled->msGroupSharedBytes), 0);
    std::vector<float> outVertPosXYZW(maxVerts * 4u, 0.0f);
    std::vector<float> outVertAttrs(std::max<size_t>(1, static_cast<size_t>(maxVerts) * attrStride), 0.0f);
    std::vector<uint32_t> outPrimIndices(maxPrims * 3u, 0u);
    std::vector<float> outPrimAttrs(std::max<size_t>(1, static_cast<size_t>(maxPrims) * attrStride), 0.0f);
    std::vector<uint8_t> outPrimCull(maxPrims, 0u);
    std::vector<uint32_t> outPrimId(maxPrims, 0u);
    std::vector<ShadedVertex> polyA, polyB;

    auto makeMeshVertex = [&](uint32_t vIdx) -> ShadedVertex {
        ShadedVertex sv;
        sv.clipX = outVertPosXYZW[vIdx * 4u + 0];
        sv.clipY = outVertPosXYZW[vIdx * 4u + 1];
        sv.clipZ = outVertPosXYZW[vIdx * 4u + 2];
        sv.clipW = outVertPosXYZW[vIdx * 4u + 3];
        for (uint32_t s = 0; s < copyScalars; ++s) {
            sv.attrs[s] = outVertAttrs[static_cast<size_t>(vIdx) * attrStride + s];
        }
        return sv;
    };

    auto applyPrimAttrs = [&](ShadedVertex& v0, uint32_t pIdx) {
        if (primAttrMask == 0 || copyScalars == 0) return;
        const uint32_t limit = std::min<uint32_t>(copyScalars, 64u);
        for (uint32_t s = 0; s < limit; ++s) {
            if ((primAttrMask & (1ull << s)) != 0) {
                v0.attrs[s] = outPrimAttrs[static_cast<size_t>(pIdx) * attrStride + s];
            }
        }
    };

    TriangleEmitTarget meshTarget;
    meshTarget.triangles = &outScene.triangles;
    meshTarget.attrBaseArena = &outScene.attrBaseArena;
    meshTarget.attrFlatArena = &outScene.attrFlatArena;
    meshTarget.origBaryArena = &outScene.origBaryArena;

    for (const auto& md : meshDispatches) {
        for (uint32_t gz = 0; gz < md.gz; ++gz) {
            for (uint32_t gy = 0; gy < md.gy; ++gy) {
                for (uint32_t gx = 0; gx < md.gx; ++gx) {
                    if (compiled->msGroupSharedBytes > 0) {
                        std::memset(msScratch.data(), 0, compiled->msGroupSharedBytes);
                    }
                    uint32_t outCounts[2]{0, 0};
                    std::fill(outVertPosXYZW.begin(), outVertPosXYZW.end(), 0.0f);
                    std::fill(outVertAttrs.begin(), outVertAttrs.end(), 0.0f);
                    std::fill(outPrimIndices.begin(), outPrimIndices.end(), 0u);
                    std::fill(outPrimAttrs.begin(), outPrimAttrs.end(), 0.0f);
                    std::fill(outPrimCull.begin(), outPrimCull.end(), 0u);
                    for (uint32_t p = 0; p < maxPrims; ++p) {
                        outPrimId[p] = p;
                    }

                    MeshThreadGroupArgs msArgs{};
                    msArgs.rootCtx = &state.rootCtx;
                    msArgs.groupSharedScratch = msScratch.data();
                    msArgs.payloadPtr = md.payload.data();
                    msArgs.gx = gx;
                    msArgs.gy = gy;
                    msArgs.gz = gz;
                    msArgs.ngx = md.gx;
                    msArgs.ngy = md.gy;
                    msArgs.ngz = md.gz;
                    msArgs.outCounts = outCounts;
                    msArgs.outVertPosXYZW = outVertPosXYZW.data();
                    msArgs.outVertAttrs = outVertAttrs.data();
                    msArgs.outPrimIndices = outPrimIndices.data();
                    msArgs.outPrimAttrs = outPrimAttrs.data();
                    msArgs.outPrimCull = outPrimCull.data();
                    msArgs.outPrimId = outPrimId.data();
                    msFn(&msArgs);

                    const uint32_t nv = std::min<uint32_t>(outCounts[0], compiled->msMaxVertexCount);
                    const uint32_t np = std::min<uint32_t>(outCounts[1], compiled->msMaxPrimitiveCount);
                    if (nv == 0 || np == 0) continue;

                    if (compiled->msOutputTopology == 1) { // line
                        for (uint32_t p = 0; p < np; ++p) {
                            if (outPrimCull[p] != 0) continue;
                            uint32_t i0 = outPrimIndices[p * 3u + 0];
                            uint32_t i1 = outPrimIndices[p * 3u + 1];
                            if (i0 >= nv || i1 >= nv) continue;
                            ShadedVertex v0 = makeMeshVertex(i0);
                            ShadedVertex v1 = makeMeshVertex(i1);
                            applyPrimAttrs(v0, p);
                            processLine(v0, v1, outPrimId[p], vp, rast, outScene);
                        }
                    } else { // triangle
                        for (uint32_t p = 0; p < np; ++p) {
                            if (outPrimCull[p] != 0) continue;
                            uint32_t i0 = outPrimIndices[p * 3u + 0];
                            uint32_t i1 = outPrimIndices[p * 3u + 1];
                            uint32_t i2 = outPrimIndices[p * 3u + 2];
                            if (i0 >= nv || i1 >= nv || i2 >= nv) continue;
                            ShadedVertex v0 = makeMeshVertex(i0);
                            ShadedVertex v1 = makeMeshVertex(i1);
                            ShadedVertex v2 = makeMeshVertex(i2);
                            applyPrimAttrs(v0, p);
                            processTriangle(v0, v1, v2, outPrimId[p], rast, outScene, meshTarget, polyA, polyB);
                        }
                    }
                }
            }
        }
    }

    return !outScene.triangles.empty();
}

} // namespace nacelle
