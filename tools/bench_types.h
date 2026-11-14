#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
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
    std::string dsDxil;
    std::string gsDxil;
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

} // namespace nacelle_bench
