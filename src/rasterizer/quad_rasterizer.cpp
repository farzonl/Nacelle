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
    float color0SOA[4 * kTilePixels]{};
    alignas(64) float depthSOA[kTilePixels]{};
    alignas(64) uint8_t stencilSOA[kTilePixels]{};
    struct AlignedDeleter {
        void operator()(float* p) const { alignedFree(p); }
    };
    std::unique_ptr<float[], AlignedDeleter> extraColorSOA;
    uint32_t extraRTVsCap = 0;

    std::unique_ptr<float[], AlignedDeleter> msaaColorBuf;
    uint32_t msaaColorFloatsCap = 0;
    std::unique_ptr<float[], AlignedDeleter> msaaDepthBuf;
    uint32_t msaaDepthFloatsCap = 0;
    struct ByteAlignedDeleter {
        void operator()(uint8_t* p) const { alignedFree(p); }
    };
    std::unique_ptr<uint8_t[], ByteAlignedDeleter> msaaStencilBuf;
    uint32_t msaaStencilBytesCap = 0;
    uint32_t msaaSampleStride = 1;

    float* colorPlane(uint32_t rt) {
        if (rt == 0) return color0SOA;
        return extraColorSOA.get() + static_cast<size_t>(rt - 1) * (4 * kTilePixels);
    }

    float* msaaColorPlane(uint32_t rt) {
        return msaaColorBuf.get() + static_cast<size_t>(rt) * msaaSampleStride * (4 * kTilePixels);
    }

    float* msaaDepthPlane() { return msaaDepthBuf.get(); }
    uint8_t* msaaStencilPlane() { return msaaStencilBuf.get(); }

    void ensureRTVs(uint32_t numRTVs) {
        if (numRTVs > 1) {
            const uint32_t neededRTVs = std::min<uint32_t>(numRTVs, 8u) - 1u;
            if (extraRTVsCap < neededRTVs) {
                const size_t bytes = static_cast<size_t>(neededRTVs) * (4 * kTilePixels) * sizeof(float);
                extraColorSOA.reset(static_cast<float*>(alignedAlloc64(bytes)));
                extraRTVsCap = neededRTVs;
            }
        }
    }

    void ensureMSAA(uint32_t numRTVs, uint32_t sampleCount, bool hasDSV) {
        msaaSampleStride = sampleCount;
        const uint32_t rtvCount = std::min<uint32_t>(numRTVs, 8u);
        if (rtvCount > 0) {
            const uint32_t neededFloats = rtvCount * sampleCount * (4 * kTilePixels);
            if (msaaColorFloatsCap < neededFloats) {
                msaaColorBuf.reset(static_cast<float*>(alignedAlloc64(static_cast<size_t>(neededFloats) * sizeof(float))));
                msaaColorFloatsCap = neededFloats;
            }
        }
        if (hasDSV) {
            const uint32_t neededDepth = sampleCount * kTilePixels;
            if (msaaDepthFloatsCap < neededDepth) {
                msaaDepthBuf.reset(static_cast<float*>(alignedAlloc64(static_cast<size_t>(neededDepth) * sizeof(float))));
                msaaDepthFloatsCap = neededDepth;
            }
            if (msaaStencilBytesCap < neededDepth) {
                msaaStencilBuf.reset(static_cast<uint8_t*>(alignedAlloc64(neededDepth)));
                msaaStencilBytesCap = neededDepth;
            }
        }
    }
};

inline uint32_t quadSwizzlePixel(uint32_t tx, uint32_t ty) {
    uint32_t quadIdx = (ty >> 1) * (kTileSize / 2u) + (tx >> 1);
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

    auto packUnorm8 = [](float v) -> uint32_t {
        float c = (v < 0.0f) ? 0.0f : ((v > 1.0f) ? 1.0f : v);
        return static_cast<uint32_t>(c * 255.0f + 0.5f);
    };

    if (rtv.format == DXGI_FORMAT_R8G8B8A8_UNORM || rtv.format == DXGI_FORMAT_B8G8R8A8_UNORM) {
        const bool isBGRA = (rtv.format == DXGI_FORMAT_B8G8R8A8_UNORM);
        const uint32_t c0 = isBGRA ? 2u : 0u;
        const uint32_t c2 = isBGRA ? 0u : 2u;
        const float* src0 = srcSOA + c0 * kTilePixels;
        const float* src1 = srcSOA + 1 * kTilePixels;
        const float* src2 = srcSOA + c2 * kTilePixels;
        const float* src3 = srcSOA + 3 * kTilePixels;

        uint32_t ty = 0;
        for (; ty + 1 < tileH; ty += 2) {
            uint8_t* row0 = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * 4;
            uint8_t* row1 = row0 + rowPitch;
            uint32_t quadRowBase = (ty >> 1) * (kTileSize / 2u);
            uint32_t tx = 0;
            for (; tx + 1 < tileW; tx += 2) {
                uint32_t pixBase = (quadRowBase + (tx >> 1)) << 2;
                uint32_t p00 = packUnorm8(src0[pixBase + 0]) | (packUnorm8(src1[pixBase + 0]) << 8) |
                               (packUnorm8(src2[pixBase + 0]) << 16) | (packUnorm8(src3[pixBase + 0]) << 24);
                uint32_t p10 = packUnorm8(src0[pixBase + 1]) | (packUnorm8(src1[pixBase + 1]) << 8) |
                               (packUnorm8(src2[pixBase + 1]) << 16) | (packUnorm8(src3[pixBase + 1]) << 24);
                uint32_t p01 = packUnorm8(src0[pixBase + 2]) | (packUnorm8(src1[pixBase + 2]) << 8) |
                               (packUnorm8(src2[pixBase + 2]) << 16) | (packUnorm8(src3[pixBase + 2]) << 24);
                uint32_t p11 = packUnorm8(src0[pixBase + 3]) | (packUnorm8(src1[pixBase + 3]) << 8) |
                               (packUnorm8(src2[pixBase + 3]) << 16) | (packUnorm8(src3[pixBase + 3]) << 24);
                std::memcpy(row0 + tx * 4, &p00, 4);
                std::memcpy(row0 + (tx + 1) * 4, &p10, 4);
                std::memcpy(row1 + tx * 4, &p01, 4);
                std::memcpy(row1 + (tx + 1) * 4, &p11, 4);
            }
            if (tx < tileW) {
                uint32_t pix0 = quadSwizzlePixel(tx, ty);
                uint32_t pix1 = quadSwizzlePixel(tx, ty + 1);
                uint32_t p0 = packUnorm8(src0[pix0]) | (packUnorm8(src1[pix0]) << 8) |
                              (packUnorm8(src2[pix0]) << 16) | (packUnorm8(src3[pix0]) << 24);
                uint32_t p1 = packUnorm8(src0[pix1]) | (packUnorm8(src1[pix1]) << 8) |
                              (packUnorm8(src2[pix1]) << 16) | (packUnorm8(src3[pix1]) << 24);
                std::memcpy(row0 + tx * 4, &p0, 4);
                std::memcpy(row1 + tx * 4, &p1, 4);
            }
        }
        for (; ty < tileH; ++ty) {
            uint8_t* row = base + static_cast<size_t>(tileOriginY + ty) * rowPitch + static_cast<size_t>(tileOriginX) * 4;
            for (uint32_t tx = 0; tx < tileW; ++tx) {
                uint32_t pixIdx = quadSwizzlePixel(tx, ty);
                uint32_t p = packUnorm8(src0[pixIdx]) | (packUnorm8(src1[pixIdx]) << 8) |
                             (packUnorm8(src2[pixIdx]) << 16) | (packUnorm8(src3[pixIdx]) << 24);
                std::memcpy(row + tx * 4, &p, 4);
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

inline bool triangleCoversRect(const SetupTriangle& tri, int32_t bx, int32_t by, int32_t w, int32_t h) {
    if (tri.isLine) return false;
    int32_t xMax = bx + w - 1;
    int32_t yMax = by + h - 1;
    for (int k = 0; k < 3; ++k) {
        int32_t rx = (tri.stepX[k] >= 0) ? bx : xMax;
        int32_t ry = (tri.stepY[k] >= 0) ? by : yMax;
        int64_t val = tri.stepX[k] * rx + tri.stepY[k] * ry + tri.base[k] + tri.bias[k];
        if (val < 0) return false;
    }
    return true;
}

struct QuadBatchEmitter {
    static constexpr uint32_t kMaxBatchGroups = 64;
    static constexpr uint32_t kMaxQuads = kMaxBatchGroups * (kMaxSimdWidth / 4);
    CompiledPixelQuadFn psFn = nullptr;
    PixelBatchArgs args{};
    uint32_t quadsPerBatch = 1;
    uint32_t qCount = 0;
    uint32_t batchCount = 0;
    uint32_t triDataCount = 0;
    uint32_t currentTriDataIdx = 0;
    uint32_t lastSceneTriIdx = UINT32_MAX;
    uint32_t currentSceneTriIdx = UINT32_MAX;
    PixelTriData currentTriData{};
    bool currentGroupHasPrevTri = false;
    float sampleOffsetX = 0.0f;
    float sampleOffsetY = 0.0f;
    uint32_t sampleCount = 1;
    int32_t sampleDx16[8]{};
    int32_t sampleDy16[8]{};
    alignas(64) uint32_t coverageMasks[kMaxBatchGroups];
    alignas(64) uint32_t sampleCoverageMasks[kMaxBatchGroups * 8];
    alignas(64) uint32_t quadOffsets[kMaxQuads];
    alignas(64) uint32_t quadTriIndices[kMaxQuads];
    alignas(64) PixelTriData triData[kMaxQuads];
    alignas(64) float pixelX[kMaxBatchGroups * kMaxSimdWidth];
    alignas(64) float pixelY[kMaxBatchGroups * kMaxSimdWidth];
    alignas(64) float baryI[kMaxBatchGroups * kMaxSimdWidth];
    alignas(64) float baryJ[kMaxBatchGroups * kMaxSimdWidth];

    QuadBatchEmitter() {
        coverageMasks[0] = 0;
    }

    void clearBatchSampleMasks(uint32_t bIdx) {
        if (sampleCount > 1) {
            for (uint32_t s = 0; s < sampleCount; ++s) {
                sampleCoverageMasks[bIdx * sampleCount + s] = 0;
            }
        }
    }

    void padRemainingQuadsInGroup() {
        uint32_t baseQ = batchCount * quadsPerBatch;
        for (uint32_t q = qCount; q < quadsPerBatch; ++q) {
            quadOffsets[baseQ + q] = 0;
            quadTriIndices[baseQ + q] = 0;
            uint32_t baseLane = (baseQ + q) * 4u;
            for (uint32_t s = 0; s < 4; ++s) {
                uint32_t lane = baseLane + s;
                pixelX[lane] = 0.0f;
                pixelY[lane] = 0.0f;
                baryI[lane] = 0.0f;
                baryJ[lane] = 0.0f;
            }
        }
    }

    void finishGroup() {
        ++batchCount;
        qCount = 0;
        currentGroupHasPrevTri = false;
        if (batchCount == kMaxBatchGroups) {
            flush();
        } else {
            coverageMasks[batchCount] = 0;
            clearBatchSampleMasks(batchCount);
        }
    }

    void padAndFinishGroup() {
        if (qCount > 0 && qCount < quadsPerBatch) {
            padRemainingQuadsInGroup();
        }
        finishGroup();
    }

    void beginTriangle(uint32_t sceneTriIdx, const PixelTriData& td) {
        currentTriData = td;
        currentSceneTriIdx = sceneTriIdx;
        if (triDataCount > 0 && sceneTriIdx == lastSceneTriIdx) {
            return;
        }
        // Keep large single-triangle runs on the uniform-triangle fast path.
        if (triDataCount == 1 && qCount == 0 && batchCount >= 4) {
            flush();
        }
        if (triDataCount >= kMaxQuads) {
            flush();
        }
        if (triDataCount > 0 && qCount > 0) {
            currentGroupHasPrevTri = true;
        }
        currentTriDataIdx = triDataCount++;
        triData[currentTriDataIdx] = td;
        lastSceneTriIdx = sceneTriIdx;
    }

    void ensureCurrentTriangle() {
        if (triDataCount == 0 && currentSceneTriIdx != UINT32_MAX) {
            currentTriDataIdx = 0;
            triDataCount = 1;
            triData[0] = currentTriData;
            lastSceneTriIdx = currentSceneTriIdx;
        }
    }

    void flush() {
        if (qCount > 0) {
            padRemainingQuadsInGroup();
            ++batchCount;
            qCount = 0;
        }
        if (batchCount == 0) {
            triDataCount = 0;
            lastSceneTriIdx = UINT32_MAX;
            currentGroupHasPrevTri = false;
            return;
        }
        args.batchCount = batchCount;
        if (triDataCount <= 1) {
            const PixelTriData& td = triData[0];
            args.triAttrBase = td.triAttrBase;
            args.triAttrFlat = td.triAttrFlat;
            args.triInvW = td.triInvW;
            args.triOrigBary = td.triOrigBary;
            args.triDepth = td.triDepth;
            args.isFrontFace = td.isFrontFace;
            args.primitiveId = td.primitiveId;
            args.viewportArrayIndex = td.viewportArrayIndex;
            args.shadingRate = td.shadingRate;
            args.triData = nullptr;
            args.quadTriIndices = nullptr;
        } else {
            args.triData = triData;
            args.quadTriIndices = quadTriIndices;
        }
        args.sampleCoverageMasks = (sampleCount > 1) ? sampleCoverageMasks : nullptr;
        psFn(&args);
        batchCount = 0;
        qCount = 0;
        triDataCount = 0;
        lastSceneTriIdx = UINT32_MAX;
        currentGroupHasPrevTri = false;
        coverageMasks[0] = 0;
        clearBatchSampleMasks(0);
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
        uint32_t quadOffset = ((ty >> 1) * (kTileSize / 2u) + (tx >> 1)) * 4u;

        uint32_t baseQ = batchCount * quadsPerBatch;
        for (uint32_t q = 0; q < qCount; ++q) {
            if (quadOffsets[baseQ + q] == quadOffset) {
                uint32_t bit = 1u << (q * 4u + subHit);
                if ((coverageMasks[batchCount] & bit) == 0) {
                    uint32_t lane = (baseQ + q) * 4u + subHit;
                    pixelX[lane] = static_cast<float>(px) + 0.5f + sampleOffsetX;
                    pixelY[lane] = static_cast<float>(py) + 0.5f + sampleOffsetY;
                    baryI[lane] = t;
                    baryJ[lane] = 0.0f;
                    coverageMasks[batchCount] |= bit;
                    if (sampleCount > 1) {
                        for (uint32_t s = 0; s < sampleCount; ++s) {
                            sampleCoverageMasks[batchCount * sampleCount + s] |= bit;
                        }
                    }
                    return;
                }
                flush();
                baseQ = 0;
                break;
            }
        }

        float dx = tri.lineX1 - tri.lineX0;
        float dy = tri.lineY1 - tri.lineY0;
        float lenSq = dx * dx + dy * dy;

        ensureCurrentTriangle();
        quadOffsets[baseQ + qCount] = quadOffset;
        quadTriIndices[baseQ + qCount] = currentTriDataIdx;
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
            uint32_t lane = (baseQ + qCount) * 4u + sub;
            pixelX[lane] = fx;
            pixelY[lane] = fy;
            baryI[lane] = st;
            baryJ[lane] = 0.0f;
        }
        coverageMasks[batchCount] |= ((1u << subHit) << (qCount * 4u));
        if (sampleCount > 1) {
            uint32_t sbit = ((1u << subHit) << (qCount * 4u));
            for (uint32_t s = 0; s < sampleCount; ++s) {
                sampleCoverageMasks[batchCount * sampleCount + s] |= sbit;
            }
        }
        ++qCount;
        if (qCount == quadsPerBatch) {
            finishGroup();
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
        uint32_t quadSampleMask[8]{};
        float qPixX[4], qPixY[4], qBaryI[4], qBaryJ[4];

        int64_t sampleOff[8][3]{};
        if (sampleCount > 1) {
            for (uint32_t s = 0; s < sampleCount; ++s) {
                for (int k = 0; k < 3; ++k) {
                    sampleOff[s][k] = (tri.stepX[k] * sampleDx16[s] + tri.stepY[k] * sampleDy16[s]) / 16;
                }
            }
        }

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
                py >= scissor.top && py < scissor.bottom) {
                if (sampleCount == 1) {
                    if ((e0 + tri.bias[0]) >= 0 &&
                        (e1 + tri.bias[1]) >= 0 &&
                        (e2 + tri.bias[2]) >= 0) {
                        quadMask |= (1u << sub);
                    }
                } else {
                    for (uint32_t s = 0; s < sampleCount; ++s) {
                        if ((e0 + sampleOff[s][0] + tri.bias[0]) >= 0 &&
                            (e1 + sampleOff[s][1] + tri.bias[1]) >= 0 &&
                            (e2 + sampleOff[s][2] + tri.bias[2]) >= 0) {
                            quadSampleMask[s] |= (1u << sub);
                            quadMask |= (1u << sub);
                        }
                    }
                }
            }
        }

        if (quadMask == 0) return;

        uint32_t tx = static_cast<uint32_t>(gx - static_cast<int32_t>(tileOriginX));
        uint32_t ty = static_cast<uint32_t>(gy - static_cast<int32_t>(tileOriginY));
        uint32_t quadOffset = ((ty >> 1) * (kTileSize / 2u) + (tx >> 1)) * 4u;

        uint32_t baseQ = batchCount * quadsPerBatch;
        if (currentGroupHasPrevTri) {
            for (uint32_t q = 0; q < qCount; ++q) {
                if (quadOffsets[baseQ + q] == quadOffset) {
                    padAndFinishGroup();
                    baseQ = batchCount * quadsPerBatch;
                    break;
                }
            }
        }

        ensureCurrentTriangle();
        quadOffsets[baseQ + qCount] = quadOffset;
        quadTriIndices[baseQ + qCount] = currentTriDataIdx;
        uint32_t baseLane = (baseQ + qCount) * 4u;
        for (uint32_t sub = 0; sub < 4; ++sub) {
            uint32_t lane = baseLane + sub;
            pixelX[lane] = qPixX[sub];
            pixelY[lane] = qPixY[sub];
            baryI[lane] = qBaryI[sub];
            baryJ[lane] = qBaryJ[sub];
        }
        coverageMasks[batchCount] |= (quadMask << (qCount * 4u));
        if (sampleCount > 1) {
            for (uint32_t s = 0; s < sampleCount; ++s) {
                sampleCoverageMasks[batchCount * sampleCount + s] |= (quadSampleMask[s] << (qCount * 4u));
            }
        }
        ++qCount;
        if (qCount == quadsPerBatch) {
            finishGroup();
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

    thread_local std::vector<uint32_t> tlActiveTiles;
    auto& activeTiles = tlActiveTiles;
    activeTiles.clear();
    if (activeTiles.capacity() < bins.size()) {
        activeTiles.reserve(bins.size());
    }
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
        scratch.ensureRTVs(state.numRTVs);
        uint32_t tileIdx = activeTiles[idx];
        uint32_t txTile = tileIdx % scene.numTilesX;
        uint32_t tyTile = tileIdx / scene.numTilesX;
        uint32_t tileOriginX = txTile * kTileSize;
        uint32_t tileOriginY = tyTile * kTileSize;
        uint32_t tileW = std::min<uint32_t>(kTileSize, scene.fbWidth - tileOriginX);
        uint32_t tileH = std::min<uint32_t>(kTileSize, scene.fbHeight - tileOriginY);

        const auto& psoDesc = state.pso->graphicsDesc();
        bool tileFullyCoveredNoDiscard = false;
        if (!compiled->psUsesDiscard &&
            !psoDesc.BlendState.AlphaToCoverageEnable &&
            !psoDesc.DepthStencilState.DepthEnable &&
            !psoDesc.DepthStencilState.StencilEnable &&
            sampleCount == 1) {
            for (uint32_t triIdx : bins[tileIdx].triIndices) {
                const SetupTriangle& tri = scene.triangles[triIdx];
                const uint32_t scIdx = (tri.viewportIndex < scene.numScissors) ? tri.viewportIndex : 0u;
                const D3D12_RECT& vpScissor = scene.scissors[scIdx];
                if (vpScissor.left <= static_cast<LONG>(tileOriginX) &&
                    vpScissor.top <= static_cast<LONG>(tileOriginY) &&
                    vpScissor.right >= static_cast<LONG>(tileOriginX + tileW) &&
                    vpScissor.bottom >= static_cast<LONG>(tileOriginY + tileH) &&
                    triangleCoversRect(tri, static_cast<int32_t>(tileOriginX), static_cast<int32_t>(tileOriginY),
                                       static_cast<int32_t>(tileW), static_cast<int32_t>(tileH))) {
                    tileFullyCoveredNoDiscard = true;
                    break;
                }
            }
        }

        const bool usePixelFreqMSAA = compiled->psPixelFreqMSAA && sampleCount > 1 && sampleCount <= 8;
        if (usePixelFreqMSAA) {
            scratch.ensureMSAA(state.numRTVs, sampleCount, state.hasDSV);
            for (uint32_t s = 0; s < sampleCount; ++s) {
                for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                    uint32_t rtSample = (state.rtvs[rt].resource &&
                                         state.rtvs[rt].resource->resourceDesc().SampleDesc.Count > 1)
                                            ? s
                                            : 0u;
                    loadTileRTV(state.rtvs[rt], rtSample, tileOriginX, tileOriginY, tileW, tileH,
                                scratch.msaaColorPlane(rt) + static_cast<size_t>(s) * (4 * kTilePixels));
                }
                if (state.hasDSV) {
                    uint32_t dsSample = (state.dsv.resource &&
                                         state.dsv.resource->resourceDesc().SampleDesc.Count > 1)
                                            ? s
                                            : 0u;
                    loadTileDSV(state.dsv, dsSample, tileOriginX, tileOriginY, tileW, tileH,
                                scratch.msaaDepthPlane() + static_cast<size_t>(s) * kTilePixels,
                                scratch.msaaStencilPlane() + static_cast<size_t>(s) * kTilePixels);
                }
            }

            QuadBatchEmitter emitter;
            emitter.psFn = psFn;
            emitter.quadsPerBatch = quadsPerBatch;
            emitter.sampleCount = sampleCount;
            emitter.clearBatchSampleMasks(0);
            for (uint32_t s = 0; s < sampleCount; ++s) {
                msaaSampleOffset16ths(sampleCount, s, emitter.sampleDx16[s], emitter.sampleDy16[s]);
            }
            emitter.args.rootCtx = &state.rootCtx;
            emitter.args.coverageMasks = emitter.coverageMasks;
            emitter.args.sampleCoverageMasks = emitter.sampleCoverageMasks;
            emitter.args.quadOffsets = emitter.quadOffsets;
            emitter.args.pixelX = emitter.pixelX;
            emitter.args.pixelY = emitter.pixelY;
            emitter.args.baryI = emitter.baryI;
            emitter.args.baryJ = emitter.baryJ;
            for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                emitter.args.tileColorSOA[rt] = scratch.msaaColorPlane(rt);
            }
            emitter.args.tileDepthSOA = state.hasDSV ? scratch.msaaDepthPlane() : nullptr;
            emitter.args.tileStencilSOA = state.hasDSV ? scratch.msaaStencilPlane() : nullptr;

            for (uint32_t triIdx : bins[tileIdx].triIndices) {
                const SetupTriangle& sampleTri = scene.triangles[triIdx];
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

                const size_t triIdxSz = static_cast<size_t>(triIdx);
                const size_t psScalars = static_cast<size_t>(scene.numPSInputScalars);
                PixelTriData td{};
                td.triAttrBase = !scene.attrBaseArena.empty()
                                     ? (scene.attrBaseArena.data() + triIdxSz * psScalars * 3)
                                     : nullptr;
                td.triAttrFlat = !scene.attrFlatArena.empty()
                                     ? (scene.attrFlatArena.data() + triIdxSz * psScalars)
                                     : nullptr;
                td.triInvW = sampleTri.invW;
                td.triOrigBary = !scene.origBaryArena.empty()
                                     ? (scene.origBaryArena.data() + triIdxSz * 9)
                                     : nullptr;
                td.triDepth = sampleTri.depth;
                td.isFrontFace = sampleTri.isFrontFace;
                td.primitiveId = sampleTri.primitiveId;
                td.viewportArrayIndex = sampleTri.viewportIndex;
                td.shadingRate = sampleTri.shadingRate;

                if (sampleTri.isLine) {
                    emitter.flush();
                    emitter.beginTriangle(triIdx, td);
                    emitter.rasterizeLineInTile(sampleTri, tileScissor, tileOriginX, tileOriginY);
                    emitter.flush();
                    continue;
                }

                emitter.beginTriangle(triIdx, td);

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
            }
            emitter.flush();

            for (uint32_t s = 0; s < sampleCount; ++s) {
                for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                    uint32_t rtSample = (state.rtvs[rt].resource &&
                                         state.rtvs[rt].resource->resourceDesc().SampleDesc.Count > 1)
                                            ? s
                                            : 0u;
                    storeTileRTV(state.rtvs[rt], rtSample, tileOriginX, tileOriginY, tileW, tileH,
                                 scratch.msaaColorPlane(rt) + static_cast<size_t>(s) * (4 * kTilePixels));
                }
                if (state.hasDSV) {
                    uint32_t dsSample = (state.dsv.resource &&
                                         state.dsv.resource->resourceDesc().SampleDesc.Count > 1)
                                            ? s
                                            : 0u;
                    storeTileDSV(state.dsv, dsSample, tileOriginX, tileOriginY, tileW, tileH,
                                 scratch.msaaDepthPlane() + static_cast<size_t>(s) * kTilePixels,
                                 scratch.msaaStencilPlane() + static_cast<size_t>(s) * kTilePixels);
                }
            }
            return;
        }

        for (uint32_t sampleIdx = 0; sampleIdx < sampleCount; ++sampleIdx) {
            int32_t dx16 = 0;
            int32_t dy16 = 0;
            msaaSampleOffset16ths(sampleCount, sampleIdx, dx16, dy16);

            for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                const auto& rtBlend = psoDesc.BlendState.RenderTarget[psoDesc.BlendState.IndependentBlendEnable ? rt : 0];
                const bool rtOpaqueOverwrite = !rtBlend.BlendEnable && !rtBlend.LogicOpEnable &&
                                               ((rtBlend.RenderTargetWriteMask & 0xFu) == 0xFu);
                if (tileFullyCoveredNoDiscard && rtOpaqueOverwrite) {
                    continue;
                }
                uint32_t rtSample = (state.rtvs[rt].resource &&
                                     state.rtvs[rt].resource->resourceDesc().SampleDesc.Count > 1)
                                        ? sampleIdx
                                        : 0u;
                loadTileRTV(state.rtvs[rt], rtSample, tileOriginX, tileOriginY, tileW, tileH, scratch.colorPlane(rt));
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
            emitter.args.coverageMasks = emitter.coverageMasks;
            emitter.args.quadOffsets = emitter.quadOffsets;
            emitter.args.pixelX = emitter.pixelX;
            emitter.args.pixelY = emitter.pixelY;
            emitter.args.baryI = emitter.baryI;
            emitter.args.baryJ = emitter.baryJ;
            for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                emitter.args.tileColorSOA[rt] = scratch.colorPlane(rt);
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

                const size_t triIdxSz = static_cast<size_t>(triIdx);
                const size_t psScalars = static_cast<size_t>(scene.numPSInputScalars);
                PixelTriData td{};
                td.triAttrBase = !scene.attrBaseArena.empty()
                                     ? (scene.attrBaseArena.data() + triIdxSz * psScalars * 3)
                                     : nullptr;
                td.triAttrFlat = !scene.attrFlatArena.empty()
                                     ? (scene.attrFlatArena.data() + triIdxSz * psScalars)
                                     : nullptr;
                td.triInvW = baseTri.invW;
                td.triOrigBary = !scene.origBaryArena.empty()
                                     ? (scene.origBaryArena.data() + triIdxSz * 9)
                                     : nullptr;
                td.triDepth = baseTri.depth;
                td.isFrontFace = sampleTri.isFrontFace;
                td.primitiveId = sampleTri.primitiveId;
                td.viewportArrayIndex = sampleTri.viewportIndex;
                td.shadingRate = sampleTri.shadingRate;

                if (sampleTri.isLine) {
                    emitter.flush();
                    emitter.beginTriangle(triIdx, td);
                    emitter.rasterizeLineInTile(sampleTri, tileScissor, tileOriginX, tileOriginY);
                    emitter.flush();
                    continue;
                }

                emitter.beginTriangle(triIdx, td);

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
            }
            emitter.flush();

            for (uint32_t rt = 0; rt < state.numRTVs && rt < 8; ++rt) {
                uint32_t rtSample = (state.rtvs[rt].resource &&
                                     state.rtvs[rt].resource->resourceDesc().SampleDesc.Count > 1)
                                        ? sampleIdx
                                        : 0u;
                storeTileRTV(state.rtvs[rt], rtSample, tileOriginX, tileOriginY, tileW, tileH, scratch.colorPlane(rt));
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

    if (activeTiles.size() <= 2) {
        for (size_t i = 0; i < activeTiles.size(); ++i) {
            processTile(i);
        }
    } else {
        parallelForDynamic(pool, activeTiles.size(), processTile);
    }
}

void executeRasterDrawCall(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw) {
    thread_local RasterScene scene;
    thread_local std::vector<TileBin> bins;
    if (!buildRasterScene(pool, state, draw, scene, &bins)) return;

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
