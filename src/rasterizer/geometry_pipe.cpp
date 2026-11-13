#include "raster_types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "../api/pipeline.h"
#include "../api/resource.h"
#include "../common/fp_env.h"

namespace nacelle {
namespace {

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

// Clip polygon against one homogeneous plane f(v) >= 0.
template <typename EvalFn>
void clipPolygonPlane(
    const std::vector<ShadedVertex>& inPoly,
    std::vector<ShadedVertex>& outPoly,
    uint32_t numScalars,
    EvalFn&& eval) {
    outPoly.clear();
    if (inPoly.empty()) return;

    const size_t n = inPoly.size();
    ShadedVertex prev = inPoly[n - 1];
    float prevD = eval(prev);
    bool prevIn = (prevD >= 0.0f);

    for (size_t i = 0; i < n; ++i) {
        const ShadedVertex& curr = inPoly[i];
        float currD = eval(curr);
        bool currIn = (currD >= 0.0f);

        if (currIn != prevIn) {
            float denom = prevD - currD;
            float t = (std::fabs(denom) > 1e-20f) ? (prevD / denom) : 0.0f;
            t = std::clamp(t, 0.0f, 1.0f);
            ShadedVertex interp;
            interp.clipX = prev.clipX + t * (curr.clipX - prev.clipX);
            interp.clipY = prev.clipY + t * (curr.clipY - prev.clipY);
            interp.clipZ = prev.clipZ + t * (curr.clipZ - prev.clipZ);
            interp.clipW = prev.clipW + t * (curr.clipW - prev.clipW);
            for (uint32_t s = 0; s < numScalars; ++s) {
                interp.attrs[s] = prev.attrs[s] + t * (curr.attrs[s] - prev.attrs[s]);
            }
            outPoly.push_back(interp);
        }
        if (currIn) {
            outPoly.push_back(curr);
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

void setupClippedTriangle(
    const ShadedVertex& v0,
    const ShadedVertex& v1,
    const ShadedVertex& v2,
    const float* flatAttrsV0,
    uint32_t primitiveId,
    uint32_t viewportIndex,
    uint32_t shadingRate,
    const D3D12_VIEWPORT& vp,
    const D3D12_RECT& scissor,
    const D3D12_RASTERIZER_DESC& rast,
    RasterScene& scene) {
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

    // Snap to 24.8 fixed-point subpixel coordinates
    int64_t fx0 = std::llround(static_cast<double>(sx0) * 256.0);
    int64_t fy0 = std::llround(static_cast<double>(sy0) * 256.0);
    int64_t fx1 = std::llround(static_cast<double>(sx1) * 256.0);
    int64_t fy1 = std::llround(static_cast<double>(sy1) * 256.0);
    int64_t fx2 = std::llround(static_cast<double>(sx2) * 256.0);
    int64_t fy2 = std::llround(static_cast<double>(sy2) * 256.0);

    // 2x signed area in 24.8^2 units. In Y-down screen space, > 0 is clockwise (D3D12 default front face).
    int64_t signedArea256 = (fx1 - fx0) * (fy2 - fy0) - (fy1 - fy0) * (fx2 - fx0);
    if (signedArea256 == 0) return;

    bool isCW = (signedArea256 > 0);
    bool isFront = rast.FrontCounterClockwise ? !isCW : isCW;

    if (rast.CullMode == D3D12_CULL_MODE_BACK && !isFront) return;
    if (rast.CullMode == D3D12_CULL_MODE_FRONT && isFront) return;

    // Bounding box in integer pixels
    int64_t minFx = std::min({fx0, fx1, fx2});
    int64_t maxFx = std::max({fx0, fx1, fx2});
    int64_t minFy = std::min({fy0, fy1, fy2});
    int64_t maxFy = std::max({fy0, fy1, fy2});

    int32_t minX = std::max<int32_t>(scissor.left, static_cast<int32_t>((minFx - 128 + 255) >> 8));
    int32_t maxX = std::min<int32_t>(scissor.right, static_cast<int32_t>((maxFx - 128) >> 8) + 1);
    int32_t minY = std::max<int32_t>(scissor.top, static_cast<int32_t>((minFy - 128 + 255) >> 8));
    int32_t maxY = std::min<int32_t>(scissor.bottom, static_cast<int32_t>((maxFy - 128) >> 8) + 1);

    if (minX >= maxX || minY >= maxY) return;

    // Edge equations: E(px, py) = (b.x - a.x)*(py*256+128 - a.y) - (b.y - a.y)*(px*256+128 - a.x)
    // Edge 0: v1 -> v2, Edge 1: v2 -> v0, Edge 2: v0 -> v1
    int64_t vxA[3] = {fx1, fx2, fx0};
    int64_t vyA[3] = {fy1, fy2, fy0};
    int64_t vxB[3] = {fx2, fx0, fx1};
    int64_t vyB[3] = {fy2, fy0, fy1};

    SetupTriangle tri;
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
        // E = dx * (py*256 + 128 - vyA) - dy * (px*256 + 128 - vxA)
        //   = (-256 * dy) * px + (256 * dx) * py + (dx*(128 - vyA) - dy*(128 - vxA))
        tri.stepX[k] = -256 * dy;
        tri.stepY[k] = 256 * dx;
        tri.base[k] = dx * (128 - vyA[k]) - dy * (128 - vxA[k]);
        bool isTopLeft = (dy < 0) || (dy == 0 && dx > 0);
        tri.bias[k] = isTopLeft ? 0 : -1;
    }

    // Optional depth bias
    float depthBiasVal = 0.0f;
    if (rast.DepthBias != 0 || rast.SlopeScaledDepthBias != 0.0f) {
        float r = 1.0f / 16777216.0f; // UNORM24 / float32 unit
        float biasTerm = static_cast<float>(rast.DepthBias) * r;
        float slopeTerm = 0.0f;
        if (rast.SlopeScaledDepthBias != 0.0f) {
            float d = static_cast<float>(posArea256) / 65536.0f;
            float dzdx = ((sz1 - sz0) * (sy2 - sy0) - (sz2 - sz0) * (sy1 - sy0)) / d;
            float dzdy = ((sx1 - sx0) * (sz2 - sz0) - (sx2 - sx0) * (sz1 - sz0)) / d;
            float maxSlope = std::max(std::fabs(dzdx), std::fabs(dzdy));
            slopeTerm = rast.SlopeScaledDepthBias * maxSlope;
        }
        depthBiasVal = biasTerm + slopeTerm;
        if (rast.DepthBiasClamp > 0.0f) {
            depthBiasVal = std::min(depthBiasVal, rast.DepthBiasClamp);
        } else if (rast.DepthBiasClamp < 0.0f) {
            depthBiasVal = std::max(depthBiasVal, rast.DepthBiasClamp);
        }
    }

    tri.invW[0] = iw0;
    tri.invW[1] = iw1 - iw0;
    tri.invW[2] = iw2 - iw0;

    float z0 = sz0 + depthBiasVal;
    float z1 = sz1 + depthBiasVal;
    float z2 = sz2 + depthBiasVal;
    tri.depth[0] = z0;
    tri.depth[1] = z1 - z0;
    tri.depth[2] = z2 - z0;

    uint32_t numScalars = scene.numPSInputScalars;
    tri.attrBaseOffset = static_cast<uint32_t>(scene.attrBaseArena.size());
    tri.attrFlatOffset = static_cast<uint32_t>(scene.attrFlatArena.size());

    if (numScalars > 0) {
        scene.attrBaseArena.resize(tri.attrBaseOffset + numScalars * 3);
        scene.attrFlatArena.resize(tri.attrFlatOffset + numScalars);
        float* baseDst = scene.attrBaseArena.data() + tri.attrBaseOffset;
        float* flatDst = scene.attrFlatArena.data() + tri.attrFlatOffset;
        for (uint32_t s = 0; s < numScalars; ++s) {
            float a0 = v0.attrs[s] * iw0;
            float a1 = v1.attrs[s] * iw1;
            float a2 = v2.attrs[s] * iw2;
            baseDst[s * 3 + 0] = a0;
            baseDst[s * 3 + 1] = a1 - a0;
            baseDst[s * 3 + 2] = a2 - a0;
            flatDst[s] = flatAttrsV0[s];
        }
    }

    scene.triangles.push_back(tri);
}

void processTriangle(
    const ShadedVertex& v0,
    const ShadedVertex& v1,
    const ShadedVertex& v2,
    uint32_t primitiveId,
    const D3D12_VIEWPORT& /*defaultVp*/,
    const D3D12_RASTERIZER_DESC& rast,
    RasterScene& scene,
    std::vector<ShadedVertex>& polyA,
    std::vector<ShadedVertex>& polyB) {
    uint32_t numScalars = scene.numPSInputScalars;

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

    // SV_CullDistance: cull entire primitive if all vertices have < 0 for any active cull component
    if (scene.cullDistMask != 0) {
        for (uint32_t k = 0; k < 8; ++k) {
            if (!(scene.cullDistMask & (1u << k))) continue;
            uint32_t s = scene.cullAttrOffset + k;
            if (s < kMaxPSInputScalars && v0.attrs[s] < 0.0f && v1.attrs[s] < 0.0f && v2.attrs[s] < 0.0f) {
                return;
            }
        }
    }

    // Fast trivial reject against frustum planes
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

    // Fast trivial accept (only if no user clip planes are negative)
    auto inside = [&](const ShadedVertex& v) {
        if (!(v.clipW > 1e-6f &&
              v.clipX >= -v.clipW && v.clipX <= v.clipW &&
              v.clipY >= -v.clipW && v.clipY <= v.clipW &&
              v.clipZ >= 0.0f && v.clipZ <= v.clipW)) {
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
    if (inside(v0) && inside(v1) && inside(v2)) {
        setupClippedTriangle(v0, v1, v2, v0.attrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, scene);
        return;
    }

    // Homogeneous 6-plane + user clip-distance Sutherland-Hodgman clipping
    polyA.clear();
    polyA.push_back(v0);
    polyA.push_back(v1);
    polyA.push_back(v2);

    clipPolygonPlane(polyA, polyB, numScalars, [](const ShadedVertex& v) { return v.clipW - 1e-6f; });
    if (polyB.size() < 3) return;
    clipPolygonPlane(polyB, polyA, numScalars, [](const ShadedVertex& v) { return v.clipX + v.clipW; });
    if (polyA.size() < 3) return;
    clipPolygonPlane(polyA, polyB, numScalars, [](const ShadedVertex& v) { return v.clipW - v.clipX; });
    if (polyB.size() < 3) return;
    clipPolygonPlane(polyB, polyA, numScalars, [](const ShadedVertex& v) { return v.clipY + v.clipW; });
    if (polyA.size() < 3) return;
    clipPolygonPlane(polyA, polyB, numScalars, [](const ShadedVertex& v) { return v.clipW - v.clipY; });
    if (polyB.size() < 3) return;
    clipPolygonPlane(polyB, polyA, numScalars, [](const ShadedVertex& v) { return v.clipZ; });
    if (polyA.size() < 3) return;
    clipPolygonPlane(polyA, polyB, numScalars, [](const ShadedVertex& v) { return v.clipW - v.clipZ; });
    if (polyB.size() < 3) return;

    if (scene.clipDistMask != 0) {
        for (uint32_t k = 0; k < 8; ++k) {
            if (!(scene.clipDistMask & (1u << k))) continue;
            uint32_t s = scene.clipAttrOffset + k;
            if (s >= kMaxPSInputScalars) continue;
            clipPolygonPlane(polyB, polyA, numScalars, [s](const ShadedVertex& v) { return v.attrs[s]; });
            std::swap(polyA, polyB);
            if (polyB.size() < 3) return;
        }
    }

    for (size_t k = 1; k + 1 < polyB.size(); ++k) {
        setupClippedTriangle(polyB[0], polyB[k], polyB[k + 1], v0.attrs, primitiveId, vpIdx, shadingRate, vp, scissor, rast, scene);
    }
}

} // namespace

bool buildRasterScene(
    BS::thread_pool<>& /*pool*/,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw,
    RasterScene& outScene) {
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

    outScene.fbWidth = fbW;
    outScene.fbHeight = fbH;
    outScene.numTilesX = (fbW + kTileSize - 1) / kTileSize;
    outScene.numTilesY = (fbH + kTileSize - 1) / kTileSize;
    outScene.numPSInputScalars = std::min<uint32_t>(
        std::max(compiled->totalVSScalarAttrs, compiled->numPSInputRegs * 4), kMaxPSInputScalars);
    outScene.clipDistMask = compiled->clipDistMask;
    outScene.cullDistMask = compiled->cullDistMask;
    outScene.clipAttrOffset = compiled->clipAttrOffset;
    outScene.cullAttrOffset = compiled->cullAttrOffset;
    outScene.hasViewportArrayIndex = compiled->hasViewportArrayIndex;
    outScene.vpIndexAttrOffset = compiled->vpIndexAttrOffset;
    outScene.hasPerPrimShadingRate = compiled->hasPerPrimShadingRate;
    outScene.shadingRateAttrOffset = compiled->shadingRateAttrOffset;
    outScene.drawShadingRate = state.shadingRate;
    outScene.shadingRateCombiner = state.shadingRateCombiners[0];
    outScene.triangles.clear();
    outScene.attrBaseArena.clear();
    outScene.attrFlatArena.clear();

    const uint32_t W = std::max<uint32_t>(1, compiled->simdWidth);
    const uint32_t numScalars = outScene.numPSInputScalars;
    const bool hasTess = compiled->hasTess;
    const bool hasGS = compiled->hasGS;
    CompiledHullCPFn hsFn = hasTess ? compiled->hsFn.load(std::memory_order_acquire) : nullptr;
    CompiledHullPCFn hsPcFn = hasTess ? compiled->hsPcFn.load(std::memory_order_acquire) : nullptr;
    CompiledDomainFn dsFn = hasTess ? compiled->dsFn.load(std::memory_order_acquire) : nullptr;
    CompiledGeometryFn gsFn = hasGS ? compiled->gsFn.load(std::memory_order_acquire) : nullptr;
    const uint32_t vsOutScalars =
        hasTess ? std::min<uint32_t>(compiled->numHSInputScalars, kMaxPSInputScalars)
                : (hasGS ? std::min<uint32_t>(compiled->numGSInputScalars, kMaxPSInputScalars) : numScalars);

    uint32_t vertsPerPrim = 3;
    bool isStrip = false;
    if (hasTess) {
        if (state.topology >= D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST &&
            state.topology <= D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST) {
            vertsPerPrim = static_cast<uint32_t>(state.topology - D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST + 1);
        } else {
            vertsPerPrim = std::max<uint32_t>(1, compiled->hsInputCPCount);
        }
    } else if (hasGS) {
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
    } else {
        isStrip = (state.topology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP ||
                   state.topology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ);
    }

    // Build primitive vertex index list (referencing unique vertex indices)
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

    const D3D12_RASTERIZER_DESC& rast = state.pso->graphicsDesc().RasterizerState;
    std::vector<ShadedVertex> shadedVerts(uniqueVIds.size());
    std::vector<uint32_t> batchVIds(W, 0);
    std::vector<uint32_t> batchIIds(W, 0);
    std::vector<float> outPosXYZW(4 * W, 0.0f);
    std::vector<float> outAttrs(std::max<uint32_t>(1, vsOutScalars) * W, 0.0f);
    std::vector<ShadedVertex> polyA, polyB;

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

            vsFn(&state.rootCtx, batchVIds.data(), batchIIds.data(), active, outPosXYZW.data(), outAttrs.data());

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

        auto runGSOnPrim = [&](const ShadedVertex* primVerts, uint32_t primVertCount, uint32_t primId,
                               uint32_t inAttrScalars) {
            uint32_t maxVerts = std::max<uint32_t>(1, compiled->gsMaxVertexCount);
            std::vector<float> gsInPos(primVertCount * 4, 0.0f);
            std::vector<float> gsInAttrs(std::max<uint32_t>(1, primVertCount * inAttrScalars), 0.0f);
            std::vector<float> gsOutPos(maxVerts * 4, 0.0f);
            std::vector<float> gsOutAttrs(std::max<uint32_t>(1, maxVerts * numScalars), 0.0f);
            std::vector<uint32_t> gsStripCounts(maxVerts, 0);

            auto makeGsVert = [&](uint32_t vIdx) {
                ShadedVertex sv;
                sv.clipX = gsOutPos[vIdx * 4 + 0];
                sv.clipY = gsOutPos[vIdx * 4 + 1];
                sv.clipZ = gsOutPos[vIdx * 4 + 2];
                sv.clipW = gsOutPos[vIdx * 4 + 3];
                for (uint32_t s = 0; s < numScalars; ++s) {
                    sv.attrs[s] = gsOutAttrs[vIdx * numScalars + s];
                }
                return sv;
            };

            for (uint32_t k = 0; k < primVertCount; ++k) {
                const ShadedVertex& sv = primVerts[k];
                gsInPos[k * 4 + 0] = sv.clipX;
                gsInPos[k * 4 + 1] = sv.clipY;
                gsInPos[k * 4 + 2] = sv.clipZ;
                gsInPos[k * 4 + 3] = sv.clipW;
                for (uint32_t s = 0; s < inAttrScalars; ++s) {
                    gsInAttrs[k * inAttrScalars + s] = sv.attrs[s];
                }
            }
            for (uint32_t gInst = 0; gInst < std::max<uint32_t>(1, compiled->gsInstanceCount); ++gInst) {
                uint32_t totalVerts = 0;
                uint32_t numStrips = 0;
                gsFn(&state.rootCtx, gsInPos.data(), gsInAttrs.data(), primId, gInst,
                     gsOutPos.data(), gsOutAttrs.data(), gsStripCounts.data(), &totalVerts, &numStrips);
                uint32_t baseV = 0;
                for (uint32_t s = 0; s < numStrips; ++s) {
                    uint32_t sc = gsStripCounts[s];
                    if (compiled->gsOutputTopology == 5 && sc >= 3) { // TriangleStrip
                        for (uint32_t k = 0; k + 2 < sc; ++k) {
                            uint32_t i0 = baseV + k;
                            uint32_t i1 = baseV + k + ((k & 1u) ? 2u : 1u);
                            uint32_t i2 = baseV + k + ((k & 1u) ? 1u : 2u);
                            processTriangle(makeGsVert(i0), makeGsVert(i1), makeGsVert(i2), primId, vp, rast, outScene, polyA, polyB);
                        }
                    }
                    baseV += sc;
                }
            }
        };

        if (hasTess && hsFn && hsPcFn && dsFn) {
            uint32_t inCPCount = vertsPerPrim;
            uint32_t outCPCount = std::max<uint32_t>(1, compiled->hsOutputCPCount);
            uint32_t inAttrScalars = std::max<uint32_t>(1, compiled->numHSInputScalars);
            uint32_t outAttrScalars = std::max<uint32_t>(1, compiled->numHSOutputScalars);
            uint32_t pcScalars = std::max<uint32_t>(1, compiled->numPatchConstScalars);
            uint32_t dsOutScalars =
                hasGS ? std::min<uint32_t>(compiled->numGSInputScalars, kMaxPSInputScalars) : numScalars;

            std::vector<float> inCPAttrs(inCPCount * inAttrScalars, 0.0f);
            std::vector<float> outCPAttrs(outCPCount * outAttrScalars, 0.0f);
            std::vector<float> patchConsts(pcScalars, 0.0f);
            float tessFactors[6]{};
            std::vector<float> dsPos(4, 0.0f);
            std::vector<float> dsAttrs(std::max<uint32_t>(1, dsOutScalars), 0.0f);

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
                default: // integer / undefined
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
                for (uint32_t outCp = 0; outCp < outCPCount; ++outCp) {
                    hsFn(&state.rootCtx, inCPAttrs.data(), p, outCp, &outCPAttrs[outCp * outAttrScalars]);
                }
                std::fill(patchConsts.begin(), patchConsts.end(), 0.0f);
                std::fill(std::begin(tessFactors), std::end(tessFactors), 0.0f);
                hsPcFn(&state.rootCtx, inCPAttrs.data(), outCPAttrs.data(), p, patchConsts.data(), tessFactors);

                auto evalDS = [&](float u, float v, float w) -> ShadedVertex {
                    std::fill(dsPos.begin(), dsPos.end(), 0.0f);
                    std::fill(dsAttrs.begin(), dsAttrs.end(), 0.0f);
                    dsFn(&state.rootCtx, outCPAttrs.data(), patchConsts.data(), tessFactors,
                         p, u, v, w, dsPos.data(), dsAttrs.data());
                    ShadedVertex sv;
                    sv.clipX = dsPos[0];
                    sv.clipY = dsPos[1];
                    sv.clipZ = dsPos[2];
                    sv.clipW = dsPos[3];
                    for (uint32_t s = 0; s < dsOutScalars; ++s) {
                        sv.attrs[s] = dsAttrs[s];
                    }
                    return sv;
                };

                auto emitTri = [&](const ShadedVertex& a, const ShadedVertex& b, const ShadedVertex& c) {
                    if (hasGS && gsFn) {
                        ShadedVertex tri[3] = {a, b, c};
                        runGSOnPrim(tri, 3, p, dsOutScalars);
                    } else {
                        processTriangle(a, b, c, p, vp, rast, outScene, polyA, polyB);
                    }
                };

                if (compiled->tessDomain == 1) { // Isoline
                    if (!(tessFactors[0] > 0.0f) || !(tessFactors[1] > 0.0f)) continue;
                    uint32_t numLines = partitionTF(tessFactors[0]);
                    uint32_t numSegs = partitionTF(tessFactors[1]);
                    for (uint32_t li = 0; li < numLines; ++li) {
                        float v = static_cast<float>(li) / static_cast<float>(numLines);
                        for (uint32_t si = 0; si <= numSegs; ++si) {
                            float u = static_cast<float>(si) / static_cast<float>(numSegs);
                            (void)evalDS(u, v, 0.0f);
                        }
                    }
                } else if (compiled->tessDomain == 2) { // Tri
                    if (!(tessFactors[0] > 0.0f) || !(tessFactors[1] > 0.0f) || !(tessFactors[2] > 0.0f)) continue;
                    uint32_t n = std::max({partitionTF(tessFactors[0]), partitionTF(tessFactors[1]),
                                           partitionTF(tessFactors[2]), partitionTF(tessFactors[4])});
                    std::vector<std::vector<ShadedVertex>> rows(n + 1);
                    for (uint32_t i = 0; i <= n; ++i) {
                        rows[i].resize(n - i + 1);
                        for (uint32_t j = 0; j <= n - i; ++j) {
                            float u = static_cast<float>(i) / static_cast<float>(n);
                            float v = static_cast<float>(j) / static_cast<float>(n);
                            float w = std::max(0.0f, 1.0f - u - v);
                            rows[i][j] = evalDS(u, v, w);
                        }
                    }
                    bool cw = (compiled->tessOutputPrimitive == 3);
                    for (uint32_t i = 0; i < n; ++i) {
                        for (uint32_t j = 0; j < n - i; ++j) {
                            if (cw) {
                                emitTri(rows[i][j], rows[i][j + 1], rows[i + 1][j]);
                            } else {
                                emitTri(rows[i][j], rows[i + 1][j], rows[i][j + 1]);
                            }
                            if (i + j + 1 < n) {
                                if (cw) {
                                    emitTri(rows[i + 1][j], rows[i][j + 1], rows[i + 1][j + 1]);
                                } else {
                                    emitTri(rows[i + 1][j], rows[i + 1][j + 1], rows[i][j + 1]);
                                }
                            }
                        }
                    }
                } else if (compiled->tessDomain == 3) { // Quad
                    if (!(tessFactors[0] > 0.0f) || !(tessFactors[1] > 0.0f) ||
                        !(tessFactors[2] > 0.0f) || !(tessFactors[3] > 0.0f)) {
                        continue;
                    }
                    uint32_t nu = std::max({partitionTF(tessFactors[1]), partitionTF(tessFactors[3]), partitionTF(tessFactors[4])});
                    uint32_t nv = std::max({partitionTF(tessFactors[0]), partitionTF(tessFactors[2]), partitionTF(tessFactors[5])});
                    std::vector<ShadedVertex> grid((nv + 1) * (nu + 1));
                    for (uint32_t j = 0; j <= nv; ++j) {
                        float v = static_cast<float>(j) / static_cast<float>(nv);
                        for (uint32_t i = 0; i <= nu; ++i) {
                            float u = static_cast<float>(i) / static_cast<float>(nu);
                            grid[j * (nu + 1) + i] = evalDS(u, v, 0.0f);
                        }
                    }
                    bool cw = (compiled->tessOutputPrimitive == 3);
                    for (uint32_t j = 0; j < nv; ++j) {
                        for (uint32_t i = 0; i < nu; ++i) {
                            const ShadedVertex& v00 = grid[j * (nu + 1) + i];
                            const ShadedVertex& v10 = grid[j * (nu + 1) + (i + 1)];
                            const ShadedVertex& v01 = grid[(j + 1) * (nu + 1) + i];
                            const ShadedVertex& v11 = grid[(j + 1) * (nu + 1) + (i + 1)];
                            if (cw) {
                                emitTri(v00, v11, v10);
                                emitTri(v00, v01, v11);
                            } else {
                                emitTri(v00, v10, v11);
                                emitTri(v00, v11, v01);
                            }
                        }
                    }
                }
            }
        } else if (hasGS && gsFn) {
            uint32_t numPrims = static_cast<uint32_t>(triVerts.size() / vertsPerPrim);
            std::vector<ShadedVertex> primVerts(vertsPerPrim);
            for (uint32_t p = 0; p < numPrims; ++p) {
                for (uint32_t k = 0; k < vertsPerPrim; ++k) {
                    primVerts[k] = shadedVerts[triVerts[p * vertsPerPrim + k]];
                }
                runGSOnPrim(primVerts.data(), vertsPerPrim, p, vsOutScalars);
            }
        } else {
            for (size_t t = 0; t + 2 < triVerts.size(); t += 3) {
                processTriangle(
                    shadedVerts[triVerts[t + 0]],
                    shadedVerts[triVerts[t + 1]],
                    shadedVerts[triVerts[t + 2]],
                    static_cast<uint32_t>(t / 3),
                    vp,
                    rast,
                    outScene,
                    polyA,
                    polyB);
            }
        }
    }

    return !outScene.triangles.empty();
}

} // namespace nacelle
