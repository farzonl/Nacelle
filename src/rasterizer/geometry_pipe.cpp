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

void setupClippedTriangle(
    const ShadedVertex& v0,
    const ShadedVertex& v1,
    const ShadedVertex& v2,
    const float* flatAttrsV0,
    const D3D12_VIEWPORT& vp,
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

    int32_t minX = std::max<int32_t>(scene.scissorRect.left, static_cast<int32_t>((minFx - 128 + 255) >> 8));
    int32_t maxX = std::min<int32_t>(scene.scissorRect.right, static_cast<int32_t>((maxFx - 128) >> 8) + 1);
    int32_t minY = std::max<int32_t>(scene.scissorRect.top, static_cast<int32_t>((minFy - 128 + 255) >> 8));
    int32_t maxY = std::min<int32_t>(scene.scissorRect.bottom, static_cast<int32_t>((maxFy - 128) >> 8) + 1);

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
    const D3D12_VIEWPORT& vp,
    const D3D12_RASTERIZER_DESC& rast,
    RasterScene& scene,
    std::vector<ShadedVertex>& polyA,
    std::vector<ShadedVertex>& polyB) {
    uint32_t numScalars = scene.numPSInputScalars;

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

    // Fast trivial accept
    auto inside = [](const ShadedVertex& v) {
        return v.clipW > 1e-6f &&
               v.clipX >= -v.clipW && v.clipX <= v.clipW &&
               v.clipY >= -v.clipW && v.clipY <= v.clipW &&
               v.clipZ >= 0.0f && v.clipZ <= v.clipW;
    };
    if (inside(v0) && inside(v1) && inside(v2)) {
        setupClippedTriangle(v0, v1, v2, v0.attrs, vp, rast, scene);
        return;
    }

    // Homogeneous 6-plane Sutherland-Hodgman clipping
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

    for (size_t k = 1; k + 1 < polyB.size(); ++k) {
        setupClippedTriangle(polyB[0], polyB[k], polyB[k + 1], v0.attrs, vp, rast, scene);
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

    D3D12_RECT scissor{0, 0, static_cast<LONG>(fbW), static_cast<LONG>(fbH)};
    if (state.numScissors > 0) {
        scissor.left = std::max<LONG>(0, state.scissors[0].left);
        scissor.top = std::max<LONG>(0, state.scissors[0].top);
        scissor.right = std::min<LONG>(static_cast<LONG>(fbW), state.scissors[0].right);
        scissor.bottom = std::min<LONG>(static_cast<LONG>(fbH), state.scissors[0].bottom);
    }
    if (scissor.left >= scissor.right || scissor.top >= scissor.bottom) {
        return false;
    }

    outScene.fbWidth = fbW;
    outScene.fbHeight = fbH;
    outScene.numTilesX = (fbW + kTileSize - 1) / kTileSize;
    outScene.numTilesY = (fbH + kTileSize - 1) / kTileSize;
    outScene.numPSInputScalars = std::min<uint32_t>(compiled->numPSInputRegs * 4, kMaxPSInputScalars);
    outScene.scissorRect = scissor;
    outScene.triangles.clear();
    outScene.attrBaseArena.clear();
    outScene.attrFlatArena.clear();

    const uint32_t W = std::max<uint32_t>(1, compiled->simdWidth);
    const uint32_t numScalars = outScene.numPSInputScalars;
    const bool isStrip = (state.topology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP ||
                          state.topology == D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ);

    // Build triangle index triplets (referencing unique vertex indices)
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
            triVerts.reserve((n / 3) * 3);
            for (uint32_t i = 0; i + 2 < n; i += 3) {
                triVerts.push_back(internVertex(draw.startIndexOrVertex + i + 0));
                triVerts.push_back(internVertex(draw.startIndexOrVertex + i + 1));
                triVerts.push_back(internVertex(draw.startIndexOrVertex + i + 2));
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
            triVerts.reserve((n / 3) * 3);
            for (uint32_t i = 0; i + 2 < n; i += 3) {
                bool cut = false;
                uint32_t r0 = readIndex(state.indexBuffer, draw.startIndexOrVertex + i + 0, cut);
                uint32_t r1 = readIndex(state.indexBuffer, draw.startIndexOrVertex + i + 1, cut);
                uint32_t r2 = readIndex(state.indexBuffer, draw.startIndexOrVertex + i + 2, cut);
                triVerts.push_back(internVertex(static_cast<uint32_t>(static_cast<int32_t>(r0) + draw.baseVertexLocation)));
                triVerts.push_back(internVertex(static_cast<uint32_t>(static_cast<int32_t>(r1) + draw.baseVertexLocation)));
                triVerts.push_back(internVertex(static_cast<uint32_t>(static_cast<int32_t>(r2) + draw.baseVertexLocation)));
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
                if (stripRun.size() >= 3) {
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
    std::vector<float> outAttrs(std::max<uint32_t>(1, numScalars) * W, 0.0f);
    std::vector<ShadedVertex> polyA, polyB;

    outScene.triangles.reserve((triVerts.size() / 3) * draw.instanceCount);

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
                for (uint32_t s = 0; s < numScalars; ++s) {
                    sv.attrs[s] = outAttrs[s * W + l];
                }
            }
        }

        for (size_t t = 0; t + 2 < triVerts.size(); t += 3) {
            processTriangle(
                shadedVerts[triVerts[t + 0]],
                shadedVerts[triVerts[t + 1]],
                shadedVerts[triVerts[t + 2]],
                vp,
                rast,
                outScene,
                polyA,
                polyB);
        }
    }

    return !outScene.triangles.empty();
}

} // namespace nacelle
