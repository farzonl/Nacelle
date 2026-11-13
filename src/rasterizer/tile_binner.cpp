#include "raster_types.h"

#include <algorithm>

namespace nacelle {

void binTrianglesToTiles(
    const RasterScene& scene,
    std::vector<TileBin>& outBins) {
    const uint32_t numTilesX = scene.numTilesX;
    const uint32_t numTilesY = scene.numTilesY;
    outBins.assign(static_cast<size_t>(numTilesX) * numTilesY, TileBin{});
    if (outBins.empty() || scene.triangles.empty()) return;

    for (uint32_t triIdx = 0; triIdx < static_cast<uint32_t>(scene.triangles.size()); ++triIdx) {
        const SetupTriangle& tri = scene.triangles[triIdx];
        if (tri.minX >= tri.maxX || tri.minY >= tri.maxY) continue;

        uint32_t tx0 = static_cast<uint32_t>(std::max<int32_t>(0, tri.minX)) / kTileSize;
        uint32_t ty0 = static_cast<uint32_t>(std::max<int32_t>(0, tri.minY)) / kTileSize;
        uint32_t tx1 = std::min<uint32_t>(numTilesX - 1, static_cast<uint32_t>(std::max<int32_t>(0, tri.maxX - 1)) / kTileSize);
        uint32_t ty1 = std::min<uint32_t>(numTilesY - 1, static_cast<uint32_t>(std::max<int32_t>(0, tri.maxY - 1)) / kTileSize);

        for (uint32_t ty = ty0; ty <= ty1; ++ty) {
            for (uint32_t tx = tx0; tx <= tx1; ++tx) {
                outBins[ty * numTilesX + tx].triIndices.push_back(triIdx);
            }
        }
    }
}

} // namespace nacelle
