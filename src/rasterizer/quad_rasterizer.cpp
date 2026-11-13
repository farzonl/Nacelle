#include "raster_types.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../api/pipeline.h"
#include "../api/resource.h"
#include "../common/format_utils.h"
#include "../common/fp_env.h"

namespace nacelle {
namespace {

constexpr uint32_t kMaxSimdWidth = 16;

struct alignas(64) TileScratch {
    float colorSOA[8][4 * kTilePixels]{};
    float depthSOA[kTilePixels]{};
    uint8_t stencilSOA[kTilePixels]{};
};

inline uint32_t quadSwizzlePixel(uint32_t tx, uint32_t ty) {
    uint32_t quadIdx = (ty >> 1) * 32u + (tx >> 1);
    return (quadIdx << 2) | ((ty & 1u) << 1) | (tx & 1u);
}

void loadTileRTV(
    const NacelleRenderTargetDescriptor& rtv,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    float* dstSOA) {
    if (!rtv.resource || !rtv.resource->basePtr()) return;
    const auto& sub = rtv.resource->subresourceLayout(rtv.mip_slice);
    const uint8_t* base = rtv.resource->basePtr() + sub.offset;
    const uint32_t rowPitch = sub.row_pitch;

    if (rtv.format == DXGI_FORMAT_R8G8B8A8_UNORM) {
        constexpr float kInv255 = 1.0f / 255.0f;
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            const uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * 4;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                const uint8_t* p = row + tx * 4;
                dstSOA[0 * kTilePixels + pixIdx] = p[0] * kInv255;
                dstSOA[1 * kTilePixels + pixIdx] = p[1] * kInv255;
                dstSOA[2 * kTilePixels + pixIdx] = p[2] * kInv255;
                dstSOA[3 * kTilePixels + pixIdx] = p[3] * kInv255;
            }
        }
        return;
    }

    if (rtv.format == DXGI_FORMAT_B8G8R8A8_UNORM) {
        constexpr float kInv255 = 1.0f / 255.0f;
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            const uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * 4;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                const uint8_t* p = row + tx * 4;
                dstSOA[0 * kTilePixels + pixIdx] = p[2] * kInv255;
                dstSOA[1 * kTilePixels + pixIdx] = p[1] * kInv255;
                dstSOA[2 * kTilePixels + pixIdx] = p[0] * kInv255;
                dstSOA[3 * kTilePixels + pixIdx] = p[3] * kInv255;
            }
        }
        return;
    }

    const FormatInfo fi = getFormatInfo(rtv.format);
    for (uint32_t ty = 0; ty < tileH; ++ty) {
        const uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * fi.bytesPerBlock;
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            uint32_t pixIdx = quadSwizzlePixel(tx, ty);
            float rgba[4]{};
            unpackPixelRGBA32F(rtv.format, row + tx * fi.bytesPerBlock, rgba);
            dstSOA[0 * kTilePixels + pixIdx] = rgba[0];
            dstSOA[1 * kTilePixels + pixIdx] = rgba[1];
            dstSOA[2 * kTilePixels + pixIdx] = rgba[2];
            dstSOA[3 * kTilePixels + pixIdx] = rgba[3];
        }
    }
}

void storeTileRTV(
    const NacelleRenderTargetDescriptor& rtv,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    const float* srcSOA) {
    if (!rtv.resource || !rtv.resource->basePtr()) return;
    const auto& sub = rtv.resource->subresourceLayout(rtv.mip_slice);
    uint8_t* base = rtv.resource->basePtr() + sub.offset;
    const uint32_t rowPitch = sub.row_pitch;

    if (rtv.format == DXGI_FORMAT_R8G8B8A8_UNORM) {
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * 4;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                uint8_t* p = row + tx * 4;
                p[0] = static_cast<uint8_t>(std::clamp(srcSOA[0 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
                p[1] = static_cast<uint8_t>(std::clamp(srcSOA[1 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
                p[2] = static_cast<uint8_t>(std::clamp(srcSOA[2 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
                p[3] = static_cast<uint8_t>(std::clamp(srcSOA[3 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        return;
    }

    if (rtv.format == DXGI_FORMAT_B8G8R8A8_UNORM) {
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * 4;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                uint8_t* p = row + tx * 4;
                p[2] = static_cast<uint8_t>(std::clamp(srcSOA[0 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
                p[1] = static_cast<uint8_t>(std::clamp(srcSOA[1 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
                p[0] = static_cast<uint8_t>(std::clamp(srcSOA[2 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
                p[3] = static_cast<uint8_t>(std::clamp(srcSOA[3 * kTilePixels + pixIdx], 0.0f, 1.0f) * 255.0f + 0.5f);
            }
        }
        return;
    }

    const FormatInfo fi = getFormatInfo(rtv.format);
    for (uint32_t ty = 0; ty < tileH; ++ty) {
        uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * fi.bytesPerBlock;
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            uint32_t pixIdx = quadSwizzlePixel(tx, ty);
            float rgba[4] = {
                srcSOA[0 * kTilePixels + pixIdx],
                srcSOA[1 * kTilePixels + pixIdx],
                srcSOA[2 * kTilePixels + pixIdx],
                srcSOA[3 * kTilePixels + pixIdx],
            };
            packPixelRGBA32F(rtv.format, rgba, row + tx * fi.bytesPerBlock);
        }
    }
}

void loadTileDSV(
    const NacelleRenderTargetDescriptor& dsv,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    float* depthSOA,
    uint8_t* stencilSOA) {
    if (!dsv.resource || !dsv.resource->basePtr()) return;
    const auto& sub = dsv.resource->subresourceLayout(dsv.mip_slice);
    const uint8_t* base = dsv.resource->basePtr() + sub.offset;
    const uint32_t rowPitch = sub.row_pitch;

    if (dsv.format == DXGI_FORMAT_D32_FLOAT) {
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            const float* row = reinterpret_cast<const float*>(base + static_cast<size_t>(tileOriginY + ty) * rowPitch) + tileOriginX;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                depthSOA[quadSwizzlePixel(tx, ty)] = row[tx];
            }
        }
        return;
    }

    const FormatInfo fi = getFormatInfo(dsv.format);
    for (uint32_t ty = 0; ty < tileH; ++ty) {
        const uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * fi.bytesPerBlock;
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            uint32_t pixIdx = quadSwizzlePixel(tx, ty);
            const uint8_t* p = row + tx * fi.bytesPerBlock;
            if (dsv.format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
                uint32_t v = 0;
                std::memcpy(&v, p, 4);
                depthSOA[pixIdx] = static_cast<float>(v & 0x00FFFFFFu) / 16777215.0f;
                stencilSOA[pixIdx] = static_cast<uint8_t>(v >> 24);
            } else if (dsv.format == DXGI_FORMAT_D16_UNORM) {
                uint16_t v = 0;
                std::memcpy(&v, p, 2);
                depthSOA[pixIdx] = static_cast<float>(v) / 65535.0f;
            } else {
                float rgba[4]{};
                unpackPixelRGBA32F(dsv.format, p, rgba);
                depthSOA[pixIdx] = rgba[0];
            }
        }
    }
}

void storeTileDSV(
    const NacelleRenderTargetDescriptor& dsv,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    const float* depthSOA,
    const uint8_t* stencilSOA) {
    if (!dsv.resource || !dsv.resource->basePtr()) return;
    const auto& sub = dsv.resource->subresourceLayout(dsv.mip_slice);
    uint8_t* base = dsv.resource->basePtr() + sub.offset;
    const uint32_t rowPitch = sub.row_pitch;

    if (dsv.format == DXGI_FORMAT_D32_FLOAT) {
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            float* row = reinterpret_cast<float*>(base + static_cast<size_t>(tileOriginY + ty) * rowPitch) + tileOriginX;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                row[tx] = depthSOA[quadSwizzlePixel(tx, ty)];
            }
        }
        return;
    }

    const FormatInfo fi = getFormatInfo(dsv.format);
    for (uint32_t ty = 0; ty < tileH; ++ty) {
        uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * fi.bytesPerBlock;
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            uint32_t pixIdx = quadSwizzlePixel(tx, ty);
            uint8_t* p = row + tx * fi.bytesPerBlock;
            if (dsv.format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
                uint32_t d24 = static_cast<uint32_t>(std::clamp(depthSOA[pixIdx], 0.0f, 1.0f) * 16777215.0f + 0.5f);
                uint32_t v = (d24 & 0x00FFFFFFu) | (static_cast<uint32_t>(stencilSOA[pixIdx]) << 24);
                std::memcpy(p, &v, 4);
            } else if (dsv.format == DXGI_FORMAT_D16_UNORM) {
                uint16_t d16 = static_cast<uint16_t>(std::clamp(depthSOA[pixIdx], 0.0f, 1.0f) * 65535.0f + 0.5f);
                std::memcpy(p, &d16, 2);
            } else {
                float rgba[4] = {depthSOA[pixIdx], 0.0f, 0.0f, 0.0f};
                packPixelRGBA32F(dsv.format, rgba, p);
            }
        }
    }
}

inline bool blockOverlapsTriangle(const SetupTriangle& tri, int32_t bx, int32_t by, int32_t size) {
    int32_t xMax = bx + size - 1;
    int32_t yMax = by + size - 1;
    for (int k = 0; k < 3; ++k) {
        int32_t rx = (tri.stepX[k] >= 0) ? xMax : bx;
        int32_t ry = (tri.stepY[k] >= 0) ? yMax : by;
        int64_t val = tri.stepX[k] * rx + tri.stepY[k] * ry + tri.base[k] + tri.bias[k];
        if (val < 0) return false;
    }
    return true;
}

struct QuadBatchEmitter {
    CompiledPixelQuadFn psFn = nullptr;
    PixelBatchArgs args{};
    uint32_t quadsPerBatch = 1;
    uint32_t qCount = 0;
    alignas(64) uint32_t quadOffsets[kMaxSimdWidth / 4]{};
    alignas(64) float pixelX[kMaxSimdWidth]{};
    alignas(64) float pixelY[kMaxSimdWidth]{};
    alignas(64) float baryI[kMaxSimdWidth]{};
    alignas(64) float baryJ[kMaxSimdWidth]{};

    void flush() {
        if (qCount == 0) return;
        for (uint32_t q = qCount; q < quadsPerBatch; ++q) {
            quadOffsets[q] = 0;
            for (uint32_t s = 0; s < 4; ++s) {
                uint32_t lane = q * 4 + s;
                pixelX[lane] = 0.0f;
                pixelY[lane] = 0.0f;
                baryI[lane] = 0.0f;
                baryJ[lane] = 0.0f;
            }
        }
        psFn(&args);
        qCount = 0;
        args.coverageMask = 0;
    }

    void testAndEmitQuad(
        const SetupTriangle& tri,
        const D3D12_RECT& scissor,
        uint32_t tileOriginX,
        uint32_t tileOriginY,
        int32_t gx,
        int32_t gy) {
        if (gx + 1 < scissor.left || gx >= scissor.right || gy + 1 < scissor.top || gy >= scissor.bottom) {
            return;
        }

        uint32_t quadMask = 0;
        float qPixX[4], qPixY[4], qBaryI[4], qBaryJ[4];

        for (uint32_t sub = 0; sub < 4; ++sub) {
            int32_t px = gx + static_cast<int32_t>(sub & 1u);
            int32_t py = gy + static_cast<int32_t>(sub >> 1);

            int64_t e0 = tri.stepX[0] * px + tri.stepY[0] * py + tri.base[0];
            int64_t e1 = tri.stepX[1] * px + tri.stepY[1] * py + tri.base[1];
            int64_t e2 = tri.stepX[2] * px + tri.stepY[2] * py + tri.base[2];

            qPixX[sub] = static_cast<float>(px) + 0.5f;
            qPixY[sub] = static_cast<float>(py) + 0.5f;
            qBaryI[sub] = static_cast<float>(e1) * tri.invArea;
            qBaryJ[sub] = static_cast<float>(e2) * tri.invArea;

            if (px >= scissor.left && px < scissor.right &&
                py >= scissor.top && py < scissor.bottom &&
                (e0 + tri.bias[0]) >= 0 &&
                (e1 + tri.bias[1]) >= 0 &&
                (e2 + tri.bias[2]) >= 0) {
                quadMask |= (1u << sub);
            }
        }

        if (quadMask == 0) return;

        uint32_t tx = static_cast<uint32_t>(gx - static_cast<int32_t>(tileOriginX));
        uint32_t ty = static_cast<uint32_t>(gy - static_cast<int32_t>(tileOriginY));
        uint32_t quadIdx = (ty >> 1) * 32u + (tx >> 1);

        quadOffsets[qCount] = quadIdx * 4u;
        for (uint32_t sub = 0; sub < 4; ++sub) {
            uint32_t lane = qCount * 4u + sub;
            pixelX[lane] = qPixX[sub];
            pixelY[lane] = qPixY[sub];
            baryI[lane] = qBaryI[sub];
            baryJ[lane] = qBaryJ[sub];
        }
        args.coverageMask |= (quadMask << (qCount * 4u));
        ++qCount;
        if (qCount == quadsPerBatch) {
            flush();
        }
    }
};

} // namespace

void rasterizeTiles(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const RasterScene& scene,
    const std::vector<TileBin>& bins) {
    if (!state.pso || state.pso->isCompute()) return;
    const auto& compiled = state.pso->graphicsPSO();
    if (!compiled) return;

    CompiledPixelQuadFn psFn = compiled->psFn.load(std::memory_order_acquire);
    if (!psFn) {
        waitForAllAsyncJITTasks();
        psFn = compiled->psFn.load(std::memory_order_acquire);
    }
    if (!psFn) return;

    std::vector<uint32_t> activeTiles;
    activeTiles.reserve(bins.size());
    for (uint32_t i = 0; i < static_cast<uint32_t>(bins.size()); ++i) {
        if (!bins[i].triIndices.empty()) activeTiles.push_back(i);
    }
    if (activeTiles.empty()) return;

    const uint32_t W = std::clamp<uint32_t>(compiled->simdWidth, 4, kMaxSimdWidth);
    const uint32_t quadsPerBatch = std::max<uint32_t>(1, W / 4);

    auto processTile = [&](size_t idx) {
        ScopedFlushDenormals flushDenorms;
        thread_local TileScratch scratch;
        uint32_t tileIdx = activeTiles[idx];
        uint32_t txTile = tileIdx % scene.numTilesX;
        uint32_t tyTile = tileIdx / scene.numTilesX;
        uint32_t tileOriginX = txTile * kTileSize;
        uint32_t tileOriginY = tyTile * kTileSize;
        uint32_t tileW = std::min<uint32_t>(kTileSize, scene.fbWidth - tileOriginX);
        uint32_t tileH = std::min<uint32_t>(kTileSize, scene.fbHeight - tileOriginY);

        for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
            loadTileRTV(state.rtvs[rt], tileOriginX, tileOriginY, tileW, tileH, scratch.colorSOA[rt]);
        }
        if (state.hasDSV) {
            loadTileDSV(state.dsv, tileOriginX, tileOriginY, tileW, tileH, scratch.depthSOA, scratch.stencilSOA);
        }

        QuadBatchEmitter emitter;
        emitter.psFn = psFn;
        emitter.quadsPerBatch = quadsPerBatch;
        emitter.args.rootCtx = &state.rootCtx;
        emitter.args.quadOffsets = emitter.quadOffsets;
        emitter.args.pixelX = emitter.pixelX;
        emitter.args.pixelY = emitter.pixelY;
        emitter.args.baryI = emitter.baryI;
        emitter.args.baryJ = emitter.baryJ;
        for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
            emitter.args.tileColorSOA[rt] = scratch.colorSOA[rt];
        }
        emitter.args.tileDepthSOA = state.hasDSV ? scratch.depthSOA : nullptr;
        emitter.args.tileStencilSOA = state.hasDSV ? scratch.stencilSOA : nullptr;

        const D3D12_RECT tileScissor{
            std::max<LONG>(scene.scissorRect.left, static_cast<LONG>(tileOriginX)),
            std::max<LONG>(scene.scissorRect.top, static_cast<LONG>(tileOriginY)),
            std::min<LONG>(scene.scissorRect.right, static_cast<LONG>(tileOriginX + tileW)),
            std::min<LONG>(scene.scissorRect.bottom, static_cast<LONG>(tileOriginY + tileH)),
        };

        for (uint32_t triIdx : bins[tileIdx].triIndices) {
            const SetupTriangle& tri = scene.triangles[triIdx];
            int32_t minX = std::max<int32_t>(tri.minX, tileScissor.left);
            int32_t maxX = std::min<int32_t>(tri.maxX, tileScissor.right);
            int32_t minY = std::max<int32_t>(tri.minY, tileScissor.top);
            int32_t maxY = std::min<int32_t>(tri.maxY, tileScissor.bottom);
            if (minX >= maxX || minY >= maxY) continue;

            emitter.args.triAttrBase = scene.numPSInputScalars > 0 ? (scene.attrBaseArena.data() + tri.attrBaseOffset) : nullptr;
            emitter.args.triAttrFlat = scene.numPSInputScalars > 0 ? (scene.attrFlatArena.data() + tri.attrFlatOffset) : nullptr;
            emitter.args.triInvW = tri.invW;
            emitter.args.triDepth = tri.depth;
            emitter.args.isFrontFace = tri.isFrontFace;

            int32_t qMinX = minX & ~1;
            int32_t qMaxX = (maxX + 1) & ~1;
            int32_t qMinY = minY & ~1;
            int32_t qMaxY = (maxY + 1) & ~1;

            if ((qMaxX - qMinX) <= 8 && (qMaxY - qMinY) <= 8) {
                for (int32_t gy = qMinY; gy < qMaxY; gy += 2) {
                    for (int32_t gx = qMinX; gx < qMaxX; gx += 2) {
                        emitter.testAndEmitQuad(tri, tileScissor, tileOriginX, tileOriginY, gx, gy);
                    }
                }
            } else {
                int32_t b16MinX = qMinX & ~15;
                int32_t b16MaxX = (qMaxX + 15) & ~15;
                int32_t b16MinY = qMinY & ~15;
                int32_t b16MaxY = (qMaxY + 15) & ~15;

                for (int32_t by16 = b16MinY; by16 < b16MaxY; by16 += 16) {
                    for (int32_t bx16 = b16MinX; bx16 < b16MaxX; bx16 += 16) {
                        if (!blockOverlapsTriangle(tri, bx16, by16, 16)) continue;

                        for (int32_t by4 = by16; by4 < by16 + 16; by4 += 4) {
                            if (by4 + 4 <= qMinY || by4 >= qMaxY) continue;
                            for (int32_t bx4 = bx16; bx4 < bx16 + 16; bx4 += 4) {
                                if (bx4 + 4 <= qMinX || bx4 >= qMaxX) continue;
                                if (!blockOverlapsTriangle(tri, bx4, by4, 4)) continue;

                                emitter.testAndEmitQuad(tri, tileScissor, tileOriginX, tileOriginY, bx4 + 0, by4 + 0);
                                emitter.testAndEmitQuad(tri, tileScissor, tileOriginX, tileOriginY, bx4 + 2, by4 + 0);
                                emitter.testAndEmitQuad(tri, tileScissor, tileOriginX, tileOriginY, bx4 + 0, by4 + 2);
                                emitter.testAndEmitQuad(tri, tileScissor, tileOriginX, tileOriginY, bx4 + 2, by4 + 2);
                            }
                        }
                    }
                }
            }
            emitter.flush();
        }

        for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
            storeTileRTV(state.rtvs[rt], tileOriginX, tileOriginY, tileW, tileH, scratch.colorSOA[rt]);
        }
        if (state.hasDSV) {
            storeTileDSV(state.dsv, tileOriginX, tileOriginY, tileW, tileH, scratch.depthSOA, scratch.stencilSOA);
        }
    };

    if (activeTiles.size() == 1) {
        processTile(0);
    } else {
        pool.detach_loop<size_t>(0, activeTiles.size(), processTile);
        pool.wait();
    }
}

void executeRasterDrawCall(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw) {
    RasterScene scene;
    if (!buildRasterScene(pool, state, draw, scene)) return;

    std::vector<TileBin> bins;
    binTrianglesToTiles(scene, bins);
    rasterizeTiles(pool, state, scene, bins);
}

} // namespace nacelle
