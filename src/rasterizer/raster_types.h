#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "../compiler/shader_compiler.h"
#include "rasterizer.h"

namespace nacelle {

/// Allocator that default-initializes instead of value-initializing elements on resize(),
/// avoiding redundant zero-fill / memset before parallel memcpy overwrites the buffer.
template <typename T>
struct NoInitAllocator : std::allocator<T> {
    using std::allocator<T>::allocator;
    template <typename U>
    struct rebind {
        using other = NoInitAllocator<U>;
    };
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

template <typename T>
using NoInitVec = std::vector<T, NoInitAllocator<T>>;

constexpr uint32_t kTileSize = kRasterTileSize;
constexpr uint32_t kTilePixels = kRasterTilePixels;
constexpr uint32_t kMaxPSInputScalars = 128;

struct ShadedVertex {
    float clipX = 0.0f;
    float clipY = 0.0f;
    float clipZ = 0.0f;
    float clipW = 1.0f;
    float origBary[3]{1.0f, 0.0f, 0.0f}; // Clip-space barycentric weights in original unclipped triangle
    float attrs[kMaxPSInputScalars]{};
};

struct SetupTriangle {
    int32_t minX; // inclusive pixel x
    int32_t maxX; // exclusive pixel x
    int32_t minY; // inclusive pixel y
    int32_t maxY; // exclusive pixel y

    // Edge equations in 24.8 fixed point for pixel center (px*256 + 128, py*256 + 128):
    // E_k(px, py) = stepX[k] * px + stepY[k] * py + base[k]
    // Inside test uses (E_k + bias[k]) >= 0.
    // Barycentrics at pixel center: baryI = float(E_1) * invArea, baryJ = float(E_2) * invArea.
    int64_t stepX[3];
    int64_t stepY[3];
    int64_t base[3];
    int64_t bias[3];
    float invArea;

    uint32_t isFrontFace;
    uint32_t primitiveId;
    uint32_t viewportIndex;
    uint32_t shadingRate;
    float invW[3];   // (1/w0, 1/w1 - 1/w0, 1/w2 - 1/w0)
    float depth[3];  // (z0, z1 - z0, z2 - z0)

    bool isLine;
    float lineX0;
    float lineY0;
    float lineX1;
    float lineY1;
};

struct TileBin {
    NoInitVec<uint32_t> triIndices;
};

struct RasterScene {
    uint32_t fbWidth = 0;
    uint32_t fbHeight = 0;
    uint32_t numTilesX = 0;
    uint32_t numTilesY = 0;
    uint32_t numPSInputScalars = 0;
    uint32_t preRastScalarAttrs = 0;
    uint32_t clipDistMask = 0;
    uint32_t cullDistMask = 0;
    uint32_t clipAttrOffset = 0;
    uint32_t cullAttrOffset = 0;
    bool hasViewportArrayIndex = false;
    uint32_t vpIndexAttrOffset = 0;
    bool hasPerPrimShadingRate = false;
    uint32_t shadingRateAttrOffset = 0;
    bool needsFlatAttrs = false;
    bool needsOrigBary = false;
    bool binsAlreadyPopulated = false;
    D3D12_SHADING_RATE drawShadingRate = D3D12_SHADING_RATE_1X1;
    D3D12_SHADING_RATE_COMBINER shadingRateCombiner = D3D12_SHADING_RATE_COMBINER_PASSTHROUGH;
    D3D12_VIEWPORT viewports[16]{};
    uint32_t numViewports = 1;
    D3D12_RECT scissors[16]{};
    uint32_t numScissors = 1;
    std::array<int16_t, kMaxPSInputScalars> psInputFromPreRastSlot{};

    NoInitVec<SetupTriangle> triangles;
    NoInitVec<float> attrBaseArena;
    NoInitVec<float> attrFlatArena;
    NoInitVec<float> origBaryArena;
};

bool buildRasterScene(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const DrawCallParams& draw,
    RasterScene& outScene,
    std::vector<TileBin>* optOutBins = nullptr);

bool buildMeshRasterScene(
    BS::thread_pool<>& pool,
    const GraphicsExecutionState& state,
    const MeshDispatchParams& dispatch,
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
