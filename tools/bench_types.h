#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace nacelle_bench {

enum class BenchCategory {
    Compute,
    Vertex,
    Pixel,
    Geometry,
    Tessellation,
    RayTracing,
    Mesh,
};

inline const char* categoryName(BenchCategory c) {
    switch (c) {
        case BenchCategory::Compute: return "Compute";
        case BenchCategory::Vertex: return "Vertex";
        case BenchCategory::Pixel: return "Pixel";
        case BenchCategory::Geometry: return "Geometry";
        case BenchCategory::Tessellation: return "Tessellation";
        case BenchCategory::RayTracing: return "Ray Tracing";
        case BenchCategory::Mesh: return "Mesh";
    }
    return "Unknown";
}

inline const char* categoryKey(BenchCategory c) {
    switch (c) {
        case BenchCategory::Compute: return "compute";
        case BenchCategory::Vertex: return "vertex";
        case BenchCategory::Pixel: return "pixel";
        case BenchCategory::Geometry: return "geometry";
        case BenchCategory::Tessellation: return "tessellation";
        case BenchCategory::RayTracing: return "raytracing";
        case BenchCategory::Mesh: return "mesh";
    }
    return "unknown";
}

enum class PipelineKind {
    Compute,
    GraphicsVSPS,
    GraphicsVSGSPS,
    GraphicsTessGS,
    RayTracingInline,
    RayTracingPipeline,
};

struct BenchmarkSpec {
    std::string id;
    std::string name;
    BenchCategory category;
    PipelineKind kind;
    std::string entryOrProfile;
    // Shader filenames relative to shaderDir
    std::string csDxil;
    std::string csSpv;
    std::string vsDxil;
    std::string vsSpv;
    std::string hsDxil;
    std::string hsSpv;
    std::string dsDxil;
    std::string dsSpv;
    std::string gsDxil;
    std::string gsSpv;
    std::string psDxil;
    std::string psSpv;
    std::string rtDxil;
};

struct BenchRunResult {
    std::string benchId;
    std::string benchName;
    std::string category;
    std::string backend;
    std::string shaderFormat;
    uint32_t simdWidth = 0;

    // Timing (ms)
    double compileMs = 0.0;
    double firstDispatchMs = 0.0;
    double coldMs = 0.0;       // compileMs + firstDispatchMs
    double meanMs = 0.0;
    double medianMs = 0.0;     // Hot JIT steady-state median
    double minMs = 0.0;
    double shortJobMs = 0.0;   // Small workload dispatch time
    double longJobMs = 0.0;    // Large workload dispatch time

    // Memory profile (MB)
    double rssBeforeMb = 0.0;
    double rssAfterCompileMb = 0.0;
    double rssAfterRunMb = 0.0;
    double rssDeltaMb = 0.0;
    double peakRssMb = 0.0;

    double checksum = 0.0;
    bool ran = false;
    std::string status = "OK";
};

inline double computeMedian(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t mid = vals.size() / 2;
    if (vals.size() % 2 == 0) {
        return 0.5 * (vals[mid - 1] + vals[mid]);
    }
    return vals[mid];
}

inline double computeGeoMean(const std::vector<double>& vals) {
    if (vals.empty()) return 0.0;
    double logSum = 0.0;
    size_t count = 0;
    for (double v : vals) {
        if (v > 0.0) {
            logSum += std::log(v);
            ++count;
        }
    }
    if (count == 0) return 0.0;
    return std::exp(logSum / static_cast<double>(count));
}

inline std::string jsonEscape(const std::string& s) {
    std::ostringstream os;
    for (char c : s) {
        if (c == '\\') os << "\\\\";
        else if (c == '"') os << "\\\"";
        else if (c == '\n') os << "\\n";
        else if (c == '\r') os << "\\r";
        else if (c == '\t') os << "\\t";
        else os << c;
    }
    return os.str();
}

struct MeshBenchVertex {
    float px, py, pz;          // offset 0:  R32G32B32_FLOAT    (POSITION0)
    int16_t nx, ny, nz, nw;    // offset 12: R16G16B16A16_SNORM (NORMAL0)
    uint16_t u, v;             // offset 20: R16G16_UNORM       (TEXCOORD0)
    uint8_t r, g, b, a;        // offset 24: R8G8B8A8_UNORM     (COLOR0)
    uint8_t j0, j1, j2, j3;    // offset 28: R8G8B8A8_UINT      (BLENDINDICES0)
};
static_assert(sizeof(MeshBenchVertex) == 32, "MeshBenchVertex must be 32 bytes");

struct alignas(16) MeshBenchConstants {
    float viewProj[16];
    float boneRows[12][4];
    float lightDirAndTime[4];
    uint8_t pad[512 - (16 + 48 + 4) * sizeof(float)];
};
static_assert(sizeof(MeshBenchConstants) == 512, "MeshBenchConstants must be 256-byte aligned");

struct MeshBenchSceneData {
    MeshBenchConstants constants{};
    std::vector<MeshBenchVertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t numTriangles = 0;
};

inline int16_t packMeshSnorm16(float v) {
    float c = std::clamp(v, -1.0f, 1.0f);
    return static_cast<int16_t>(std::lrintf(c * 32767.0f));
}

inline uint16_t packMeshUnorm16(float v) {
    float c = std::clamp(v, 0.0f, 1.0f);
    return static_cast<uint16_t>(std::lrintf(c * 65535.0f));
}

inline uint8_t packMeshUnorm8(float v) {
    float c = std::clamp(v, 0.0f, 1.0f);
    return static_cast<uint8_t>(std::lrintf(c * 255.0f));
}

inline void getMeshBenchGridAndDims(
    int mode,
    uint32_t baseW,
    uint32_t baseH,
    uint32_t& outGrid,
    uint32_t& outRW,
    uint32_t& outRH) {
    const bool isQuick = (baseW <= 128);
    if (mode == 0) {
        outGrid = 40; // 3,200 triangles
        outRW = 128;
        outRH = 128;
    } else if (mode == 1) {
        outGrid = isQuick ? 80 : 316; // 12,800 (--quick) or 199,712 (~200k) triangles
        outRW = baseW;
        outRH = baseH;
    } else {
        outGrid = isQuick ? 160 : 448; // 51,200 (--quick) or 401,408 (~400k) triangles
        outRW = baseW * 2;
        outRH = baseH * 2;
    }
}

inline MeshBenchSceneData buildMeshBenchScene(uint32_t gridX, uint32_t gridY) {
    MeshBenchSceneData scene;
    scene.numTriangles = gridX * gridY * 2u;

    float vp[16] = {
        0.92f, 0.00f, 0.00f, 0.0f,
        0.00f, 0.92f, 0.00f, 0.0f,
        0.00f, 0.00f, 0.50f, 0.0f,
        0.00f, 0.00f, 0.25f, 1.0f,
    };
    std::memcpy(scene.constants.viewProj, vp, sizeof(vp));

    const float angles[4] = {0.06f, -0.08f, 0.05f, -0.04f};
    const float tx[4] = {0.02f, -0.02f, 0.01f, -0.01f};
    const float ty[4] = {-0.01f, 0.02f, -0.02f, 0.01f};
    const float tz[4] = {0.00f, 0.03f, -0.02f, 0.02f};
    for (uint32_t b = 0; b < 4; ++b) {
        float cs = std::cos(angles[b]);
        float sn = std::sin(angles[b]);
        scene.constants.boneRows[b * 3 + 0][0] = cs;
        scene.constants.boneRows[b * 3 + 0][1] = -sn;
        scene.constants.boneRows[b * 3 + 0][2] = 0.0f;
        scene.constants.boneRows[b * 3 + 0][3] = tx[b];

        scene.constants.boneRows[b * 3 + 1][0] = sn;
        scene.constants.boneRows[b * 3 + 1][1] = cs;
        scene.constants.boneRows[b * 3 + 1][2] = 0.0f;
        scene.constants.boneRows[b * 3 + 1][3] = ty[b];

        scene.constants.boneRows[b * 3 + 2][0] = 0.0f;
        scene.constants.boneRows[b * 3 + 2][1] = 0.0f;
        scene.constants.boneRows[b * 3 + 2][2] = 1.0f;
        scene.constants.boneRows[b * 3 + 2][3] = tz[b];
    }
    scene.constants.lightDirAndTime[0] = 0.45f;
    scene.constants.lightDirAndTime[1] = 0.65f;
    scene.constants.lightDirAndTime[2] = 0.61f;
    scene.constants.lightDirAndTime[3] = 0.35f;

    scene.vertices.resize(static_cast<size_t>(gridX + 1) * (gridY + 1));
    for (uint32_t y = 0; y <= gridY; ++y) {
        for (uint32_t x = 0; x <= gridX; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(gridX);
            float v = static_cast<float>(y) / static_cast<float>(gridY);
            float px = u * 1.84f - 0.92f;
            float py = v * 1.84f - 0.92f;
            float pz = 0.35f + 0.12f * std::sin(u * 6.28318f) * std::cos(v * 6.28318f);

            float nx = -0.25f * std::cos(u * 6.28318f);
            float ny = -0.25f * std::sin(v * 6.28318f);
            float nz = 1.0f;
            float nLen = std::sqrt(nx * nx + ny * ny + nz * nz);
            nx /= nLen;
            ny /= nLen;
            nz /= nLen;

            MeshBenchVertex vert{};
            vert.px = px;
            vert.py = py;
            vert.pz = pz;
            vert.nx = packMeshSnorm16(nx);
            vert.ny = packMeshSnorm16(ny);
            vert.nz = packMeshSnorm16(nz);
            vert.nw = packMeshSnorm16(u - v);
            vert.u = packMeshUnorm16(u);
            vert.v = packMeshUnorm16(v);
            vert.r = packMeshUnorm8(0.25f + 0.65f * u);
            vert.g = packMeshUnorm8(0.30f + 0.55f * v);
            vert.b = packMeshUnorm8(0.85f - 0.45f * u);
            vert.a = packMeshUnorm8(0.20f + 0.70f * (0.5f * (u + v)));
            uint8_t baseJoint = static_cast<uint8_t>((x + y) & 3u);
            vert.j0 = baseJoint;
            vert.j1 = static_cast<uint8_t>((baseJoint + 1u) & 3u);
            vert.j2 = static_cast<uint8_t>((baseJoint + 2u) & 3u);
            vert.j3 = static_cast<uint8_t>((baseJoint + 3u) & 3u);
            scene.vertices[static_cast<size_t>(y) * (gridX + 1) + x] = vert;
        }
    }

    scene.indices.reserve(static_cast<size_t>(gridX) * gridY * 6u);
    for (uint32_t y = 0; y < gridY; ++y) {
        for (uint32_t x = 0; x < gridX; ++x) {
            uint32_t i0 = y * (gridX + 1) + x;
            uint32_t i1 = i0 + 1;
            uint32_t i2 = i0 + (gridX + 1);
            uint32_t i3 = i2 + 1;
            if (((x + y) & 1u) == 0u) {
                scene.indices.push_back(i0);
                scene.indices.push_back(i2);
                scene.indices.push_back(i1);
                scene.indices.push_back(i1);
                scene.indices.push_back(i2);
                scene.indices.push_back(i3);
            } else {
                scene.indices.push_back(i0);
                scene.indices.push_back(i1);
                scene.indices.push_back(i2);
                scene.indices.push_back(i1);
                scene.indices.push_back(i3);
                scene.indices.push_back(i2);
            }
        }
    }

    return scene;
}

} // namespace nacelle_bench
