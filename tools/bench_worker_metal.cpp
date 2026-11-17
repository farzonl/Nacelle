#include "bench_common.h"
#include "bench_memory.h"
#include "metal_bench.h"

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace nacelle_bench;

int main(int argc, char** argv) {
    return runWorkerMain(argc, argv, [](const BenchmarkSpec& spec, const WorkerCliArgs& args) {
        std::vector<BenchRunResult> out;
        auto recordMetal = [&](const MetalBenchResult& mr, const MemorySnapshot& memBefore, const MemorySnapshot& memAfter) {
            BenchRunResult r;
            r.benchId = spec.id;
            r.benchName = spec.name;
            r.category = categoryName(spec.category);
            r.backend = mr.backend;
            r.shaderFormat = mr.shaderFormat;
            r.compileMs = mr.compileMs;
            r.firstDispatchMs = mr.firstDispatchMs;
            r.coldMs = mr.compileMs + mr.firstDispatchMs;
            r.meanMs = mr.meanMs;
            r.medianMs = mr.medianMs;
            r.minMs = mr.minMs;
            r.shortJobMs = mr.shortJobMs;
            r.longJobMs = mr.longJobMs;
            r.rssBeforeMb = memBefore.currentRssMb;
            r.rssAfterCompileMb = memAfter.currentRssMb;
            r.rssAfterRunMb = memAfter.currentRssMb;
            r.rssDeltaMb = std::max(0.0, memAfter.currentRssMb - memBefore.currentRssMb);
            r.peakRssMb = memAfter.peakRssMb;
            r.checksum = mr.checksum;
            r.ran = mr.ran;
            r.status = mr.status;
            out.push_back(std::move(r));
        };

        if (spec.kind == PipelineKind::Compute && !spec.csDxil.empty()) {
            std::vector<uint8_t> dxil;
            if (readBinary(args.shaderDir + "/" + spec.csDxil, dxil)) {
                MemorySnapshot memBefore = queryProcessMemory();
                auto mr = runMetalComputeBench(spec.id, dxil, args.width, args.height, args.maxSteps, args.iters);
                MemorySnapshot memAfter = queryProcessMemory();
                recordMetal(mr, memBefore, memAfter);
            }
        } else if ((spec.kind == PipelineKind::GraphicsVSPS ||
                    spec.kind == PipelineKind::GraphicsVSGSPS ||
                    spec.kind == PipelineKind::GraphicsTessGS) &&
                   !spec.vsDxil.empty() && !spec.psDxil.empty()) {
            std::vector<uint8_t> vsDxil, psDxil, hsDxil, dsDxil, gsDxil;
            bool ok = readBinary(args.shaderDir + "/" + spec.vsDxil, vsDxil) &&
                      readBinary(args.shaderDir + "/" + spec.psDxil, psDxil);
            if (ok && !spec.hsDxil.empty()) ok = readBinary(args.shaderDir + "/" + spec.hsDxil, hsDxil);
            if (ok && !spec.dsDxil.empty()) ok = readBinary(args.shaderDir + "/" + spec.dsDxil, dsDxil);
            if (ok && !spec.gsDxil.empty()) ok = readBinary(args.shaderDir + "/" + spec.gsDxil, gsDxil);
            if (ok) {
                MemorySnapshot memBefore = queryProcessMemory();
                auto mr = runMetalGraphicsBench(
                    spec.id, categoryName(spec.category), vsDxil, psDxil, args.width, args.height, args.iters,
                    hsDxil, dsDxil, gsDxil, spec.hasTexture);
                MemorySnapshot memAfter = queryProcessMemory();
                recordMetal(mr, memBefore, memAfter);
            }
        } else if (spec.kind == PipelineKind::GraphicsMSPS &&
                   !spec.msDxil.empty() && !spec.psDxil.empty()) {
            std::vector<uint8_t> msDxil, psDxil;
            if (readBinary(args.shaderDir + "/" + spec.msDxil, msDxil) &&
                readBinary(args.shaderDir + "/" + spec.psDxil, psDxil)) {
                MemorySnapshot memBefore = queryProcessMemory();
                auto mr = runMetalMeshBench(spec.id, msDxil, psDxil, args.iters);
                MemorySnapshot memAfter = queryProcessMemory();
                recordMetal(mr, memBefore, memAfter);
            }
        } else if ((spec.kind == PipelineKind::RayTracingInline ||
                    spec.kind == PipelineKind::RayTracingPipeline) &&
                   !spec.rtDxil.empty()) {
            std::vector<uint8_t> rtDxil;
            if (readBinary(args.shaderDir + "/" + spec.rtDxil, rtDxil)) {
                MemorySnapshot memBefore = queryProcessMemory();
                auto mr = runMetalRaytracingBench(
                    spec.id, spec.kind == PipelineKind::RayTracingInline, rtDxil, args.width, args.height, args.iters);
                MemorySnapshot memAfter = queryProcessMemory();
                recordMetal(mr, memBefore, memAfter);
            }
        }
        return out;
    });
}
