#pragma once

#include <cstdint>
#include <vector>

#include "rasterizer.h"

namespace nacelle {

constexpr uint32_t kTileSize = 64;
constexpr uint32_t kTilePixels = kTileSize * kTileSize;
constexpr uint32_t kMaxPSInputScalars = 128;

struct ShadedVertex {
    float clipX = 0.0f;
    float clipY = 0.0f;
    float clipZ = 0.0f;
    float clipW = 1.0f;
    float attrs[kMaxPSInputScalars]{};
};

struct SetupTriangle {
    int32_t minX = 0; // inclusive pixel x
    int32_t maxX = 0; // exclusive pixel x
    int32_t minY = 0; // inclusive pixel y
    int32_t maxY = 0; // exclusive pixel y

    // Edge equations in 24.8 fixed point for pixel center (px*256 + 128, py*256 + 128):
    // E_k(px, py) = stepX[k] * px + stepY[k] * py + base[k]
    // Inside test uses (E_k + bias[k]) >= 0.
    // Barycentrics at pixel center: baryI = float(E_1) * invArea, baryJ = float(E_2) * invArea.
    int64_t stepX[3]{};
    int64_t stepY[3]{};
    int64_t base[3]{};
    int64_t bias[3]{};
    float invArea = 0.0f;

    uint32_t isFrontFace = 1;
    uint32_t primitiveId = 0;
    uint32_t viewportIndex = 0;
    uint32_t shadingRate = 0;
    float invW[3]{};   // (1/w0, 1/w1 - 1/w0, 1/w2 - 1/w0)
    float depth[3]{};  // (z0, z1 - z0, z2 - z0)

    uint32_t attrBaseOffset = 0; // offset into RasterScene::attrBaseArena (numScalars * 3)
    uint32_t attrFlatOffset = 0; // offset into RasterScene::attrFlatArena (numScalars)
};

struct RasterScene {
    uint32_t fbWidth = 0;
    uint32_t fbHeight = 0;
    uint32_t numTilesX = 0;
    uint32_t numTilesY = 0;
    uint32_t numPSInputScalars = 0;
    uint32_t clipDistMask = 0;
    uint32_t cullDistMask = 0;
    uint32_t clipAttrOffset = 0;
    uint32_t cullAttrOffset = 0;
    bool hasViewportArrayIndex = false;
    uint32_t vpIndexAttrOffset = 0;
    bool hasPerPrimShadingRate = false;
    uint32_t shadingRateAttrOffset = 0;
    D3D12_SHADING_RATE drawShadingRate = D3D12_SHADING_RATE_1X1;
    D3D12_SHADING_RATE_COMBINER shadingRateCombiner = D3D12_SHADING_RATE_COMBINER_PASSTHROUGH;
    D3D12_VIEWPORT viewports[16]{};
    uint32_t numViewports = 1;
    D3D12_RECT scissors[16]{};
    uint32_t numScissors = 1;

    std::vector<SetupTriangle> triangles;
    std::vector<float> attrBaseArena;
    std::vector<float> attrFlatArena;
};

struct TileBin {
    std::vector<uint32_t> triIndices;
};

bool buildRasterScene(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw,
    RasterScene& outScene);

void binTrianglesToTiles(
    const RasterScene& scene,
    std::vector<TileBin>& outBins);

void rasterizeTiles(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const RasterScene& scene,
    const std::vector<TileBin>& bins);

} // namespace nacelle
