#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace nacelle_bench {

// Builds a 64-triangle subdivided octahedron/sphere BLAS (192 float3 vertices)
// and 64 TLAS instance transforms in a 4x4x4 3D grid.
inline std::vector<float> buildHeavyRTBlasVertices() {
    // 8 octahedron faces subdivided 2x2x2 -> 8 * 8 = 64 triangles (or 8 * 4 = 32, let's do 4x4 grid of 2-tri quads = 32 tris, 2 layers = 64 tris).
    std::vector<float> verts;
    verts.reserve(64u * 3u * 3u);
    auto pushTri = [&](float x0, float y0, float z0, float x1, float y1, float z1, float x2, float y2, float z2) {
        verts.push_back(x0); verts.push_back(y0); verts.push_back(z0);
        verts.push_back(x1); verts.push_back(y1); verts.push_back(z1);
        verts.push_back(x2); verts.push_back(y2); verts.push_back(z2);
    };
    constexpr float kPi = 3.14159265f;
    constexpr uint32_t kRings = 4u;
    constexpr uint32_t kSegs = 8u; // 4 * 8 * 2 = 64 triangles
    for (uint32_t r = 0; r < kRings; ++r) {
        float phi0 = kPi * static_cast<float>(r) / static_cast<float>(kRings);
        float phi1 = kPi * static_cast<float>(r + 1u) / static_cast<float>(kRings);
        for (uint32_t s = 0; s < kSegs; ++s) {
            float th0 = 2.0f * kPi * static_cast<float>(s) / static_cast<float>(kSegs);
            float th1 = 2.0f * kPi * static_cast<float>(s + 1u) / static_cast<float>(kSegs);
            auto pt = [](float phi, float th, float& x, float& y, float& z) {
                float sp = std::sin(phi);
                x = 0.42f * sp * std::cos(th);
                y = 0.42f * std::cos(phi);
                z = 0.42f * sp * std::sin(th);
            };
            float x00, y00, z00, x10, y10, z10, x01, y01, z01, x11, y11, z11;
            pt(phi0, th0, x00, y00, z00);
            pt(phi1, th0, x10, y10, z10);
            pt(phi0, th1, x01, y01, z01);
            pt(phi1, th1, x11, y11, z11);
            pushTri(x00, y00, z00, x10, y10, z10, x11, y11, z11);
            pushTri(x00, y00, z00, x11, y11, z11, x01, y01, z01);
        }
    }
    return verts;
}

template <typename InstanceDescT>
inline std::vector<InstanceDescT> buildHeavyRTTlasInstances(uint64_t blasAddress) {
    std::vector<InstanceDescT> out(64);
    for (uint32_t iz = 0; iz < 4u; ++iz) {
        for (uint32_t iy = 0; iy < 4u; ++iy) {
            for (uint32_t ix = 0; ix < 4u; ++ix) {
                uint32_t idx = (iz * 16u) + (iy * 4u) + ix;
                InstanceDescT inst{};
                inst.Transform[0][0] = 1.0f;
                inst.Transform[1][1] = 1.0f;
                inst.Transform[2][2] = 1.0f;
                inst.Transform[0][3] = (static_cast<float>(ix) - 1.5f) * 1.4f;
                inst.Transform[1][3] = (static_cast<float>(iy) - 1.5f) * 1.4f;
                inst.Transform[2][3] = (static_cast<float>(iz) - 1.5f) * 1.4f;
                inst.InstanceID = idx;
                inst.InstanceMask = 0xFFu;
                inst.AccelerationStructure = blasAddress;
                out[idx] = inst;
            }
        }
    }
    return out;
}

} // namespace nacelle_bench
