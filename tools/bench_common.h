#pragma once

#include "bench_memory.h"
#include "bench_types.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace nacelle_bench {

inline void finalizeTimingAndMemory(
    BenchRunResult& res,
    const MemorySnapshot& memBefore,
    const MemorySnapshot& memAfterCompile,
    const std::vector<double>& samples) {
    double total = 0.0;
    for (double v : samples) total += v;
    res.meanMs = samples.empty() ? 0.0 : (total / static_cast<double>(samples.size()));
    res.medianMs = computeMedian(samples);
    res.minMs = samples.empty() ? 0.0 : *std::min_element(samples.begin(), samples.end());
    res.coldMs = res.compileMs + res.firstDispatchMs;

    MemorySnapshot memAfterRun = queryProcessMemory();
    res.rssBeforeMb = memBefore.currentRssMb;
    res.rssAfterCompileMb = memAfterCompile.currentRssMb;
    res.rssAfterRunMb = memAfterRun.currentRssMb;
    res.rssDeltaMb = std::max(0.0, memAfterRun.currentRssMb - memBefore.currentRssMb);
    res.peakRssMb = memAfterRun.peakRssMb;
    res.ran = true;
    res.status = "OK";
}

inline bool readBinary(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    if (sz <= 0) return false;
    out.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    return static_cast<bool>(ifs.read(reinterpret_cast<char*>(out.data()), sz));
}

inline bool validateSpirvBinary(const std::vector<uint8_t>& spv, std::string& outErr) {
    if (spv.size() < 20 || (spv.size() % 4) != 0) {
        outErr = "invalid SPIR-V byte size";
        return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, spv.data(), 4);
    if (magic != 0x07230203u) {
        outErr = "invalid SPIR-V magic";
        return false;
    }
    return true;
}

inline std::vector<BenchmarkSpec> buildBenchmarkCatalog() {
    std::vector<BenchmarkSpec> cat;

    auto addCompute = [&](const std::string& id, const std::string& name, const std::string& entry,
                          const std::string& dxil, const std::string& spv) {
        BenchmarkSpec s;
        s.id = id;
        s.name = name;
        s.category = BenchCategory::Compute;
        s.kind = PipelineKind::Compute;
        s.entryOrProfile = entry;
        s.csDxil = dxil;
        s.csSpv = spv;
        cat.push_back(std::move(s));
    };

    addCompute("bench_raymarch", "Raymarcher (ALU+CF)", "main", "bench_raymarch.dxil", "bench_raymarch.spv");
    addCompute("matmul", "MatMul 128x128 (FMA+Mem)", "CSMain", "matmul.dxil", "matmul.spv");
    addCompute("linalg_matmul", "LinAlg MatMul 128x128 (CoopMat)", "CSMain", "linalg_matmul.dxil", "linalg_matmul.spv");
    addCompute("prefix_sum_groupshared", "Prefix Sum (Groupshared)", "CSMain", "prefix_sum_groupshared.dxil", "prefix_sum_groupshared.spv");
    addCompute("divergent_cf_atomics", "Divergent CF + Atomics", "CSMain", "divergent_cf_atomics.dxil", "divergent_cf_atomics.spv");
    addCompute("wave_intrinsics", "Wave Intrinsics (Subgroup)", "CSMain", "wave_intrinsics.dxil", "wave_intrinsics.spv");

    {
        BenchmarkSpec s;
        s.id = "bench_vertex_stress";
        s.name = "Vertex Transform Stress";
        s.category = BenchCategory::Vertex;
        s.kind = PipelineKind::GraphicsVSPS;
        s.vsDxil = "bench_vertex_stress_vs.dxil";
        s.psDxil = "bench_vertex_stress_ps.dxil";
        s.vsSpv = "bench_vertex_stress_vs.spv";
        s.psSpv = "bench_vertex_stress_ps.spv";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "bench_vertex_mesh";
        s.name = "Skinned Mesh (VB+IB+Cull)";
        s.category = BenchCategory::Vertex;
        s.kind = PipelineKind::GraphicsVSPS;
        s.vsDxil = "bench_vertex_mesh_vs.dxil";
        s.psDxil = "bench_vertex_mesh_ps.dxil";
        s.vsSpv = "bench_vertex_mesh_vs.spv";
        s.psSpv = "bench_vertex_mesh_ps.spv";
        cat.push_back(std::move(s));
    }

    auto addPixel = [&](const std::string& id, const std::string& name,
                        const std::string& psDxil, const std::string& psSpv) {
        BenchmarkSpec s;
        s.id = id;
        s.name = name;
        s.category = BenchCategory::Pixel;
        s.kind = PipelineKind::GraphicsVSPS;
        s.vsDxil = "fullscreen_quad_vs.dxil";
        s.psDxil = psDxil;
        s.vsSpv = "fullscreen_quad_vs.spv";
        s.psSpv = psSpv;
        cat.push_back(std::move(s));
    };

    addPixel("light_cycle", "Light Cycle", "light_cycle.dxil", "light_cycle.spv");
    addPixel("aurora_paint", "Aurora Paint", "aurora_paint.dxil", "aurora_paint.spv");
    addPixel("sunflower_fields", "Sunflower Fields", "sunflower_fields.dxil", "sunflower_fields.spv");
    addPixel("mt_fuji_watercolor", "Mt Fuji Watercolor", "mt_fuji_watercolor.dxil", "mt_fuji_watercolor.spv");
    addPixel("disco_maze", "Disco Maze", "disco_maze.dxil", "disco_maze.spv");

    {
        BenchmarkSpec s;
        s.id = "bench_geometry_stress";
        s.name = "Geometry Star Expansion";
        s.category = BenchCategory::Geometry;
        s.kind = PipelineKind::GraphicsVSGSPS;
        s.vsDxil = "bench_geometry_stress_vs.dxil";
        s.vsSpv = "bench_geometry_stress_vs.spv";
        s.gsDxil = "bench_geometry_stress_gs.dxil";
        s.gsSpv = "bench_geometry_stress_gs.spv";
        s.psDxil = "bench_geometry_stress_ps.dxil";
        s.psSpv = "bench_geometry_stress_ps.spv";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "bench_tess_tri";
        s.name = "Tessellation (VS+HS+DS+PS)";
        s.category = BenchCategory::Tessellation;
        s.kind = PipelineKind::GraphicsTessGS;
        s.vsDxil = "bench_tess_tri_vs.dxil";
        s.vsSpv = "bench_tess_tri_vs.spv";
        s.hsDxil = "bench_tess_tri_hs.dxil";
        s.hsSpv = "bench_tess_tri_hs.spv";
        s.dsDxil = "bench_tess_tri_ds.dxil";
        s.dsSpv = "bench_tess_tri_ds.spv";
        s.psDxil = "bench_tess_tri_ps.dxil";
        s.psSpv = "bench_tess_tri_ps.spv";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "tess_point_gs";
        s.name = "Tessellation (VS+HS+DS+GS+PS)";
        s.category = BenchCategory::Tessellation;
        s.kind = PipelineKind::GraphicsTessGS;
        s.vsDxil = "gfx_tess_point_vs.dxil";
        s.vsSpv = "gfx_tess_point_vs.spv";
        s.hsDxil = "gfx_tess_point_hs.dxil";
        s.hsSpv = "gfx_tess_point_hs.spv";
        s.dsDxil = "gfx_tess_point_ds.dxil";
        s.dsSpv = "gfx_tess_point_ds.spv";
        s.gsDxil = "gfx_tess_point_gs.dxil";
        s.gsSpv = "gfx_tess_point_gs.spv";
        s.psDxil = "gfx_tess_point_ps.dxil";
        s.psSpv = "gfx_tess_point_ps.spv";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "raytracing_inline";
        s.name = "RayTracing Inline (RayQuery)";
        s.category = BenchCategory::RayTracing;
        s.kind = PipelineKind::RayTracingInline;
        s.rtDxil = "raytracing_inline_cs.dxil";
        s.csSpv = "raytracing_inline_cs.spv";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "raytracing_pipeline";
        s.name = "RayTracing Pipeline (DispatchRays)";
        s.category = BenchCategory::RayTracing;
        s.kind = PipelineKind::RayTracingPipeline;
        s.rtDxil = "raytracing_pipeline.dxil";
        s.csSpv = "raytracing_pipeline.spv";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "bench_mesh_stress";
        s.name = "Meshlet Grid Expansion";
        s.category = BenchCategory::Mesh;
        s.kind = PipelineKind::GraphicsMSPS;
        s.msDxil = "bench_mesh_stress_ms.dxil";
        s.msSpv = "bench_mesh_stress_ms.spv";
        s.psDxil = "bench_mesh_stress_ps.dxil";
        s.psSpv = "bench_mesh_stress_ps.spv";
        cat.push_back(std::move(s));
    }

    return cat;
}

inline std::string serializeResultJson(const BenchRunResult& r) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(4)
       << "{"
       << "\"bench_id\":\"" << jsonEscape(r.benchId) << "\","
       << "\"bench_name\":\"" << jsonEscape(r.benchName) << "\","
       << "\"category\":\"" << jsonEscape(r.category) << "\","
       << "\"backend\":\"" << jsonEscape(r.backend) << "\","
       << "\"shader_format\":\"" << jsonEscape(r.shaderFormat) << "\","
       << "\"simd_width\":" << r.simdWidth << ","
       << "\"ran\":" << (r.ran ? "true" : "false") << ","
       << "\"status\":\"" << jsonEscape(r.status) << "\","
       << "\"compile_ms\":" << r.compileMs << ","
       << "\"first_dispatch_ms\":" << r.firstDispatchMs << ","
       << "\"cold_ms\":" << r.coldMs << ","
       << "\"mean_ms\":" << r.meanMs << ","
       << "\"median_ms\":" << r.medianMs << ","
       << "\"min_ms\":" << r.minMs << ","
       << "\"short_job_ms\":" << r.shortJobMs << ","
       << "\"long_job_ms\":" << r.longJobMs << ","
       << "\"rss_before_mb\":" << r.rssBeforeMb << ","
       << "\"rss_after_compile_mb\":" << r.rssAfterCompileMb << ","
       << "\"rss_after_run_mb\":" << r.rssAfterRunMb << ","
       << "\"rss_delta_mb\":" << r.rssDeltaMb << ","
       << "\"peak_rss_mb\":" << r.peakRssMb << ","
       << "\"checksum\":" << std::setprecision(6) << r.checksum
       << "}";
    return os.str();
}

struct WorkerCliArgs {
    uint32_t width = 256;
    uint32_t height = 256;
    uint32_t maxSteps = 48;
    uint32_t iters = 5;
    std::string shaderDir = NACELLE_SHADER_BIN_DIR;
    std::string singleBench;
    std::string singleBackend;
    bool listDevices = false;
    bool forceWarp = false;
};

inline WorkerCliArgs parseWorkerCliArgs(int argc, char** argv) {
    WorkerCliArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) {
            args.iters = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            args.width = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            args.height = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--steps" && i + 1 < argc) {
            args.maxSteps = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--shader-dir" && i + 1 < argc) {
            args.shaderDir = argv[++i];
        } else if (arg == "--single-bench" && i + 1 < argc) {
            args.singleBench = argv[++i];
        } else if (arg == "--single-backend" && i + 1 < argc) {
            args.singleBackend = argv[++i];
        } else if (arg == "--list-devices") {
            args.listDevices = true;
        } else if (arg == "--warp") {
            args.forceWarp = true;
        }
    }
    return args;
}

template <typename Fn>
inline int runWorkerMain(int argc, char** argv, Fn&& runSpecFn) {
    WorkerCliArgs args = parseWorkerCliArgs(argc, argv);
    auto catalog = buildBenchmarkCatalog();
    for (const auto& spec : catalog) {
        if (!args.singleBench.empty() && spec.id != args.singleBench) continue;
        std::vector<BenchRunResult> res = runSpecFn(spec, args);
        std::cout << "<<<BENCH_JSON_BEGIN>>>\n";
        for (const auto& r : res) {
            std::cout << serializeResultJson(r) << "\n";
        }
        std::cout << "<<<BENCH_JSON_END>>>\n";
    }
    return 0;
}

} // namespace nacelle_bench
