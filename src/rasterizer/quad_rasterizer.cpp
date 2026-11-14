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
    uint32_t sampleIdx,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    float* dstSOA) {
    if (!rtv.resource) return;
    if (rtv.resource->isReserved()) {
        const uint32_t mipLevels = std::max<uint32_t>(1, rtv.resource->resourceDesc().MipLevels);
        const uint32_t subIdx = rtv.mip_slice + rtv.first_array_slice * mipLevels;
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                float rgba[4]{};
                if (const uint8_t* p = rtv.resource->texBlockPtr(subIdx, tileOriginX + tx, tileOriginY + ty, 0)) {
                    unpackPixelRGBA32F(rtv.format, p, rgba);
                }
                dstSOA[0 * kTilePixels + pixIdx] = rgba[0];
                dstSOA[1 * kTilePixels + pixIdx] = rgba[1];
                dstSOA[2 * kTilePixels + pixIdx] = rgba[2];
                dstSOA[3 * kTilePixels + pixIdx] = rgba[3];
            }
        }
        return;
    }
    if (!rtv.resource->basePtr()) return;
    const auto& sub = rtv.resource->subresourceLayout(rtv.mip_slice);
    const uint8_t* base = rtv.resource->basePtr() + sub.offset + static_cast<size_t>(sampleIdx) * sub.slice_pitch;
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
    uint32_t sampleIdx,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    const float* srcSOA) {
    if (!rtv.resource) return;
    if (rtv.resource->isReserved()) {
        const uint32_t mipLevels = std::max<uint32_t>(1, rtv.resource->resourceDesc().MipLevels);
        const uint32_t subIdx = rtv.mip_slice + rtv.first_array_slice * mipLevels;
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint8_t* p = rtv.resource->texBlockPtr(subIdx, tileOriginX + tx, tileOriginY + ty, 0);
                if (!p) continue;
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                float rgba[4] = {
                    srcSOA[0 * kTilePixels + pixIdx],
                    srcSOA[1 * kTilePixels + pixIdx],
                    srcSOA[2 * kTilePixels + pixIdx],
                    srcSOA[3 * kTilePixels + pixIdx],
                };
                packPixelRGBA32F(rtv.format, rgba, p);
            }
        }
        return;
    }
    if (!rtv.resource->basePtr()) return;
    const auto& sub = rtv.resource->subresourceLayout(rtv.mip_slice);
    uint8_t* base = rtv.resource->basePtr() + sub.offset + static_cast<size_t>(sampleIdx) * sub.slice_pitch;
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
    uint32_t sampleIdx,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    float* depthSOA,
    uint8_t* stencilSOA) {
    if (!dsv.resource) return;
    if (dsv.resource->isReserved()) {
        const uint32_t mipLevels = std::max<uint32_t>(1, dsv.resource->resourceDesc().MipLevels);
        const uint32_t subIdx = dsv.mip_slice + dsv.first_array_slice * mipLevels;
        static constexpr uint8_t kZeroPx[16]{};
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                const uint8_t* p = dsv.resource->texBlockPtr(subIdx, tileOriginX + tx, tileOriginY + ty, 0);
                if (!p) p = kZeroPx;
                if (dsv.format == DXGI_FORMAT_D32_FLOAT) {
                    std::memcpy(&depthSOA[pixIdx], p, 4);
                } else if (dsv.format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
                    uint32_t v = 0;
                    std::memcpy(&v, p, 4);
                    depthSOA[pixIdx] = static_cast<float>(v & 0x00FFFFFFu) / 16777215.0f;
                    stencilSOA[pixIdx] = static_cast<uint8_t>(v >> 24);
                } else if (dsv.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
                    std::memcpy(&depthSOA[pixIdx], p, 4);
                    stencilSOA[pixIdx] = p[4];
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
        return;
    }
    if (!dsv.resource->basePtr()) return;
    const auto& sub = dsv.resource->subresourceLayout(dsv.mip_slice);
    const uint8_t* base = dsv.resource->basePtr() + sub.offset + static_cast<size_t>(sampleIdx) * sub.slice_pitch;
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
            } else if (dsv.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
                std::memcpy(&depthSOA[pixIdx], p, 4);
                stencilSOA[pixIdx] = p[4];
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
    uint32_t sampleIdx,
    uint32_t tileOriginX,
    uint32_t tileOriginY,
    uint32_t tileW,
    uint32_t tileH,
    const float* depthSOA,
    const uint8_t* stencilSOA) {
    if (!dsv.resource) return;
    if (dsv.resource->isReserved()) {
        const uint32_t mipLevels = std::max<uint32_t>(1, dsv.resource->resourceDesc().MipLevels);
        const uint32_t subIdx = dsv.mip_slice + dsv.first_array_slice * mipLevels;
        for (uint32_t ty = 0; ty < tileH; ++ty) {
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint8_t* p = dsv.resource->texBlockPtr(subIdx, tileOriginX + tx, tileOriginY + ty, 0);
                if (!p) continue;
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                if (dsv.format == DXGI_FORMAT_D32_FLOAT) {
                    std::memcpy(p, &depthSOA[pixIdx], 4);
                } else if (dsv.format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
                    uint32_t d24 = static_cast<uint32_t>(std::clamp(depthSOA[pixIdx], 0.0f, 1.0f) * 16777215.0f + 0.5f);
                    uint32_t v = (d24 & 0x00FFFFFFu) | (static_cast<uint32_t>(stencilSOA[pixIdx]) << 24);
                    std::memcpy(p, &v, 4);
                } else if (dsv.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
                    std::memcpy(p, &depthSOA[pixIdx], 4);
                    p[4] = stencilSOA[pixIdx];
                } else if (dsv.format == DXGI_FORMAT_D16_UNORM) {
                    uint16_t d16 = static_cast<uint16_t>(std::clamp(depthSOA[pixIdx], 0.0f, 1.0f) * 65535.0f + 0.5f);
                    std::memcpy(p, &d16, 2);
                } else {
                    float rgba[4] = {depthSOA[pixIdx], 0.0f, 0.0f, 0.0f};
                    packPixelRGBA32F(dsv.format, rgba, p);
                }
            }
        }
        return;
    }
    if (!dsv.resource->basePtr()) return;
    const auto& sub = dsv.resource->subresourceLayout(dsv.mip_slice);
    uint8_t* base = dsv.resource->basePtr() + sub.offset + static_cast<size_t>(sampleIdx) * sub.slice_pitch;
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
            } else if (dsv.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
                std::memcpy(p, &depthSOA[pixIdx], 4);
                p[4] = stencilSOA[pixIdx];
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
    float sampleOffsetX = 0.0f;
    float sampleOffsetY = 0.0f;
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

    static bool pointInDiamond(float x, float y, int32_t px, int32_t py, bool isYMajor) {
        float qx = x - (static_cast<float>(px) + 0.5f);
        float qy = y - (static_cast<float>(py) + 0.5f);
        float d = std::fabs(qx) + std::fabs(qy);
        if (d < 0.5f - 1e-5f) return true;
        if (std::fabs(d - 0.5f) <= 1e-5f) {
            if (qy > 0.0f) return true;
            if (isYMajor && qx > 0.0f && std::fabs(qy) <= 1e-5f) return true;
        }
        return false;
    }

    void emitLinePixel(
        const SetupTriangle& tri,
        uint32_t tileOriginX,
        uint32_t tileOriginY,
        int32_t px,
        int32_t py,
        float t) {
        int32_t gx = px & ~1;
        int32_t gy = py & ~1;
        uint32_t subHit = static_cast<uint32_t>(((py & 1) << 1) | (px & 1));
        uint32_t tx = static_cast<uint32_t>(gx - static_cast<int32_t>(tileOriginX));
        uint32_t ty = static_cast<uint32_t>(gy - static_cast<int32_t>(tileOriginY));
        uint32_t quadOffset = ((ty >> 1) * 32u + (tx >> 1)) * 4u;

        for (uint32_t q = 0; q < qCount; ++q) {
            if (quadOffsets[q] == quadOffset) {
                uint32_t bit = 1u << (q * 4u + subHit);
                if ((args.coverageMask & bit) == 0) {
                    uint32_t lane = q * 4u + subHit;
                    pixelX[lane] = static_cast<float>(px) + 0.5f + sampleOffsetX;
                    pixelY[lane] = static_cast<float>(py) + 0.5f + sampleOffsetY;
                    baryI[lane] = t;
                    baryJ[lane] = 0.0f;
                    args.coverageMask |= bit;
                    return;
                }
                flush();
                break;
            }
        }

        float dx = tri.lineX1 - tri.lineX0;
        float dy = tri.lineY1 - tri.lineY0;
        float lenSq = dx * dx + dy * dy;

        quadOffsets[qCount] = quadOffset;
        for (uint32_t sub = 0; sub < 4; ++sub) {
            int32_t spx = gx + static_cast<int32_t>(sub & 1u);
            int32_t spy = gy + static_cast<int32_t>(sub >> 1);
            float fx = static_cast<float>(spx) + 0.5f + sampleOffsetX;
            float fy = static_cast<float>(spy) + 0.5f + sampleOffsetY;
            float st = (sub == subHit)
                           ? t
                           : ((lenSq > 0.0f)
                                  ? std::clamp(((fx - tri.lineX0) * dx + (fy - tri.lineY0) * dy) / lenSq, 0.0f, 1.0f)
                                  : t);
            uint32_t lane = qCount * 4u + sub;
            pixelX[lane] = fx;
            pixelY[lane] = fy;
            baryI[lane] = st;
            baryJ[lane] = 0.0f;
        }
        args.coverageMask |= ((1u << subHit) << (qCount * 4u));
        ++qCount;
        if (qCount == quadsPerBatch) {
            flush();
        }
    }

    void rasterizeLineInTile(
        const SetupTriangle& tri,
        const D3D12_RECT& tileScissor,
        uint32_t tileOriginX,
        uint32_t tileOriginY) {
        const double x0 = tri.lineX0;
        const double y0 = tri.lineY0;
        const double x1 = tri.lineX1;
        const double y1 = tri.lineY1;
        const double dx = x1 - x0;
        const double dy = y1 - y0;
        const double adx = std::fabs(dx);
        const double ady = std::fabs(dy);
        if (adx == 0.0 && ady == 0.0) return;

        const int32_t clipMinX = std::max<int32_t>(tri.minX, tileScissor.left);
        const int32_t clipMaxX = std::min<int32_t>(tri.maxX, tileScissor.right);
        const int32_t clipMinY = std::max<int32_t>(tri.minY, tileScissor.top);
        const int32_t clipMaxY = std::min<int32_t>(tri.maxY, tileScissor.bottom);
        if (clipMinX >= clipMaxX || clipMinY >= clipMaxY) return;

        if (adx >= ady) {
            int32_t pxStart = std::max<int32_t>(clipMinX, static_cast<int32_t>(std::floor(std::min(x0, x1))) - 1);
            int32_t pxEnd = std::min<int32_t>(clipMaxX, static_cast<int32_t>(std::ceil(std::max(x0, x1))) + 1);
            double slope = dy / dx;
            for (int32_t px = pxStart; px < pxEnd; ++px) {
                double cx = static_cast<double>(px) + 0.5;
                double yc = y0 + slope * (cx - x0);
                int32_t py = static_cast<int32_t>(std::ceil(yc - 1.0));
                if (py < clipMinY || py >= clipMaxY) continue;
                bool in0 = pointInDiamond(tri.lineX0, tri.lineY0, px, py, false);
                bool in1 = pointInDiamond(tri.lineX1, tri.lineY1, px, py, false);
                bool lit = (dx > 0.0)
                               ? (((x0 < cx) || in0) && (x1 > cx) && !in1)
                               : (((x0 > cx) || in0) && (x1 < cx) && !in1);
                if (!lit) continue;
                float t = static_cast<float>(std::clamp((cx - x0) / dx, 0.0, 1.0));
                emitLinePixel(tri, tileOriginX, tileOriginY, px, py, t);
            }
        } else {
            int32_t pyStart = std::max<int32_t>(clipMinY, static_cast<int32_t>(std::floor(std::min(y0, y1))) - 1);
            int32_t pyEnd = std::min<int32_t>(clipMaxY, static_cast<int32_t>(std::ceil(std::max(y0, y1))) + 1);
            double invSlope = dx / dy;
            for (int32_t py = pyStart; py < pyEnd; ++py) {
                double cy = static_cast<double>(py) + 0.5;
                double xc = x0 + invSlope * (cy - y0);
                int32_t px = static_cast<int32_t>(std::ceil(xc - 1.0));
                if (px < clipMinX || px >= clipMaxX) continue;
                bool in0 = pointInDiamond(tri.lineX0, tri.lineY0, px, py, true);
                bool in1 = pointInDiamond(tri.lineX1, tri.lineY1, px, py, true);
                bool lit = (dy > 0.0)
                               ? (((y0 < cy) || in0) && (y1 > cy) && !in1)
                               : (((y0 > cy) || in0) && (y1 < cy) && !in1);
                if (!lit) continue;
                float t = static_cast<float>(std::clamp((cy - y0) / dy, 0.0, 1.0));
                emitLinePixel(tri, tileOriginX, tileOriginY, px, py, t);
            }
        }
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

            qPixX[sub] = static_cast<float>(px) + 0.5f + sampleOffsetX;
            qPixY[sub] = static_cast<float>(py) + 0.5f + sampleOffsetY;
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

void msaaSampleOffset16ths(uint32_t sampleCount, uint32_t sampleIdx, int32_t& dx16, int32_t& dy16) {
    static constexpr int32_t kOffsets2x[2][2] = {{4, 4}, {-4, -4}};
    static constexpr int32_t kOffsets4x[4][2] = {{-2, -6}, {6, -2}, {-6, 2}, {2, 6}};
    static constexpr int32_t kOffsets8x[8][2] = {
        {1, -3}, {-1, 3}, {5, 1}, {-3, -5}, {-5, 5}, {-7, -1}, {3, 7}, {7, -7}};
    if (sampleCount == 2 && sampleIdx < 2) {
        dx16 = kOffsets2x[sampleIdx][0];
        dy16 = kOffsets2x[sampleIdx][1];
    } else if (sampleCount == 4 && sampleIdx < 4) {
        dx16 = kOffsets4x[sampleIdx][0];
        dy16 = kOffsets4x[sampleIdx][1];
    } else if (sampleCount == 8 && sampleIdx < 8) {
        dx16 = kOffsets8x[sampleIdx][0];
        dy16 = kOffsets8x[sampleIdx][1];
    } else {
        dx16 = 0;
        dy16 = 0;
    }
}

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

    uint32_t sampleCount = 1;
    if (state.numRTVs > 0 && state.rtvs[0].resource) {
        sampleCount = std::max<uint32_t>(1, state.rtvs[0].resource->resourceDesc().SampleDesc.Count);
    } else if (state.hasDSV && state.dsv.resource) {
        sampleCount = std::max<uint32_t>(1, state.dsv.resource->resourceDesc().SampleDesc.Count);
    }

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

        for (uint32_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
            int32_t dx16 = 0;
            int32_t dy16 = 0;
            msaaSampleOffset16ths(sampleCount, sampleIdx, dx16, dy16);

            for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                uint32_t rtSample = (state.rtvs[rt].resource &&
                                     state.rtvs[rt].resource->resourceDesc().SampleDesc.Count > 1)
                                        ? sampleIdx
                                        : 0u;
                loadTileRTV(state.rtvs[rt], rtSample, tileOriginX, tileOriginY, tileW, tileH, scratch.colorSOA[rt]);
            }
            if (state.hasDSV) {
                uint32_t dsSample = (state.dsv.resource &&
                                     state.dsv.resource->resourceDesc().SampleDesc.Count > 1)
                                        ? sampleIdx
                                        : 0u;
                loadTileDSV(state.dsv, dsSample, tileOriginX, tileOriginY, tileW, tileH, scratch.depthSOA, scratch.stencilSOA);
            }

            QuadBatchEmitter emitter;
            emitter.psFn = psFn;
            emitter.quadsPerBatch = quadsPerBatch;
            emitter.sampleOffsetX = static_cast<float>(dx16) / 16.0f;
            emitter.sampleOffsetY = static_cast<float>(dy16) / 16.0f;
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

            for (uint32_t triIdx : bins[tileIdx].triIndices) {
                const SetupTriangle& baseTri = scene.triangles[triIdx];
                SetupTriangle sampleTri = baseTri;
                if (dx16 != 0 || dy16 != 0) {
                    for (int k = 0; k < 3; ++k) {
                        sampleTri.base[k] += (baseTri.stepX[k] * dx16 + baseTri.stepY[k] * dy16) / 16;
                    }
                }
                const uint32_t scIdx = (sampleTri.viewportIndex < scene.numScissors) ? sampleTri.viewportIndex : 0u;
                const D3D12_RECT& vpScissor = scene.scissors[scIdx];
                const D3D12_RECT tileScissor{
                    std::max<LONG>(vpScissor.left, static_cast<LONG>(tileOriginX)),
                    std::max<LONG>(vpScissor.top, static_cast<LONG>(tileOriginY)),
                    std::min<LONG>(vpScissor.right, static_cast<LONG>(tileOriginX + tileW)),
                    std::min<LONG>(vpScissor.bottom, static_cast<LONG>(tileOriginY + tileH)),
                };
                int32_t minX = std::max<int32_t>(sampleTri.minX, tileScissor.left);
                int32_t maxX = std::min<int32_t>(sampleTri.maxX, tileScissor.right);
                int32_t minY = std::max<int32_t>(sampleTri.minY, tileScissor.top);
                int32_t maxY = std::min<int32_t>(sampleTri.maxY, tileScissor.bottom);
                if (minX >= maxX || minY >= maxY) continue;

                emitter.args.triAttrBase = scene.numPSInputScalars > 0 ? (scene.attrBaseArena.data() + sampleTri.attrBaseOffset) : nullptr;
                emitter.args.triAttrFlat = scene.numPSInputScalars > 0 ? (scene.attrFlatArena.data() + sampleTri.attrFlatOffset) : nullptr;
                emitter.args.triInvW = sampleTri.invW;
                emitter.args.triOrigBary = sampleTri.origBary;
                emitter.args.triDepth = sampleTri.depth;
                emitter.args.isFrontFace = sampleTri.isFrontFace;
                emitter.args.primitiveId = sampleTri.primitiveId;
                emitter.args.viewportArrayIndex = sampleTri.viewportIndex;
                emitter.args.shadingRate = sampleTri.shadingRate;

                if (sampleTri.isLine) {
                    emitter.rasterizeLineInTile(sampleTri, tileScissor, tileOriginX, tileOriginY);
                    emitter.flush();
                    continue;
                }

                int32_t qMinX = minX & ~1;
                int32_t qMaxX = (maxX + 1) & ~1;
                int32_t qMinY = minY & ~1;
                int32_t qMaxY = (maxY + 1) & ~1;

                if ((qMaxX - qMinX) <= 8 && (qMaxY - qMinY) <= 8) {
                    for (int32_t gy = qMinY; gy < qMaxY; gy += 2) {
                        for (int32_t gx = qMinX; gx < qMaxX; gx += 2) {
                            emitter.testAndEmitQuad(sampleTri, tileScissor, tileOriginX, tileOriginY, gx, gy);
                        }
                    }
                } else {
                    int32_t b16MinX = qMinX & ~15;
                    int32_t b16MaxX = (qMaxX + 15) & ~15;
                    int32_t b16MinY = qMinY & ~15;
                    int32_t b16MaxY = (qMaxY + 15) & ~15;

                    for (int32_t by16 = b16MinY; by16 < b16MaxY; by16 += 16) {
                        for (int32_t bx16 = b16MinX; bx16 < b16MaxX; bx16 += 16) {
                            if (!blockOverlapsTriangle(sampleTri, bx16, by16, 16)) continue;

                            for (int32_t by4 = by16; by4 < by16 + 16; by4 += 4) {
                                if (by4 + 4 <= qMinY || by4 >= qMaxY) continue;
                                for (int32_t bx4 = bx16; bx4 < bx16 + 16; bx4 += 4) {
                                    if (bx4 + 4 <= qMinX || bx4 >= qMaxX) continue;
                                    if (!blockOverlapsTriangle(sampleTri, bx4, by4, 4)) continue;

                                    emitter.testAndEmitQuad(sampleTri, tileScissor, tileOriginX, tileOriginY, bx4 + 0, by4 + 0);
                                    emitter.testAndEmitQuad(sampleTri, tileScissor, tileOriginX, tileOriginY, bx4 + 2, by4 + 0);
                                    emitter.testAndEmitQuad(sampleTri, tileScissor, tileOriginX, tileOriginY, bx4 + 0, by4 + 2);
                                    emitter.testAndEmitQuad(sampleTri, tileScissor, tileOriginX, tileOriginY, bx4 + 2, by4 + 2);
                                }
                            }
                        }
                    }
                }
                emitter.flush();
            }

            for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                uint32_t rtSample = (state.rtvs[rt].resource &&
                                     state.rtvs[rt].resource->resourceDesc().SampleDesc.Count > 1)
                                        ? sampleIdx
                                        : 0u;
                storeTileRTV(state.rtvs[rt], rtSample, tileOriginX, tileOriginY, tileW, tileH, scratch.colorSOA[rt]);
            }
            if (state.hasDSV) {
                uint32_t dsSample = (state.dsv.resource &&
                                     state.dsv.resource->resourceDesc().SampleDesc.Count > 1)
                                        ? sampleIdx
                                        : 0u;
                storeTileDSV(state.dsv, dsSample, tileOriginX, tileOriginY, tileW, tileH, scratch.depthSOA, scratch.stencilSOA);
            }
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

void executeRasterMeshDispatch(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const MeshDispatchParams& dispatch) {
    RasterScene scene;
    if (!buildMeshRasterScene(pool, state, dispatch, scene)) return;

    std::vector<TileBin> bins;
    binTrianglesToTiles(scene, bins);
    rasterizeTiles(pool, state, scene, bins);
}

void msaaSampleOffset16ths(uint32_t sampleCount, uint32_t sampleIdx, int32_t& dx16, int32_t& dy16) {
    static constexpr int32_t kOffsets2x[2][2] = {{4, 4}, {-4, -4}};
    static constexpr int32_t kOffsets4x[4][2] = {{-2, -6}, {6, -2}, {-6, 2}, {2, 6}};
    static constexpr int32_t kOffsets8x[8][2] = {
        {1, -3}, {-1, 3}, {5, 1}, {-3, -5}, {-5, 5}, {-7, -1}, {3, 7}, {7, -7}};
    if (sampleCount == 2 && sampleIdx < 2) {
        dx16 = kOffsets2x[sampleIdx][0];
        dy16 = kOffsets2x[sampleIdx][1];
    } else if (sampleCount == 4 && sampleIdx < 4) {
        dx16 = kOffsets4x[sampleIdx][0];
        dy16 = kOffsets4x[sampleIdx][1];
    } else if (sampleCount == 8 && sampleIdx < 8) {
        dx16 = kOffsets8x[sampleIdx][0];
        dy16 = kOffsets8x[sampleIdx][1];
    } else {
        dx16 = 0;
        dy16 = 0;
    }
}

} // namespace nacelle
