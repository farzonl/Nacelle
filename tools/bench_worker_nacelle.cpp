#include "bench_common.h"
#include "bench_nacelle_runners.h"
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
#include <nacelle/instrumentation.h>
#include <cstdlib>
#endif

#include <cstdint>
#include <string>
#include <vector>

using namespace nacelle_bench;

int main(int argc, char** argv) {
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
    const char* profilePath = std::getenv("NACELLE_PROFILE_OUTPUT");
    NacelleProfileCapture* capture = nullptr;
    if (profilePath && *profilePath) {
        NacelleProfileOptions options;
        if (FAILED(NacelleBeginProfileCapture(&options, &capture))) return 1;
    }
#endif
    const int result = runWorkerMain(argc, argv, [](const BenchmarkSpec& spec, const WorkerCliArgs& args) {
        std::vector<BenchRunResult> out;
        uint32_t w = 4;
        if (args.singleBackend.rfind("nacelle-w", 0) == 0) {
            w = static_cast<uint32_t>(std::stoul(args.singleBackend.substr(9)));
        }
        if (spec.kind == PipelineKind::Compute) {
            std::vector<uint8_t> dxil;
            if (!readBinary(args.shaderDir + "/" + spec.csDxil, dxil)) return out;
            out.push_back(runNacelleComputeBench(spec, dxil, w, args.width, args.height, args.maxSteps, args.iters));
        } else if (spec.kind == PipelineKind::GraphicsVSPS ||
                   spec.kind == PipelineKind::GraphicsVSGSPS ||
                   spec.kind == PipelineKind::GraphicsTessGS) {
            std::vector<uint8_t> vs, ps, hs, ds, gs;
            if (!readBinary(args.shaderDir + "/" + spec.vsDxil, vs) ||
                !readBinary(args.shaderDir + "/" + spec.psDxil, ps)) return out;
            if (!spec.hsDxil.empty() && !readBinary(args.shaderDir + "/" + spec.hsDxil, hs)) return out;
            if (!spec.dsDxil.empty() && !readBinary(args.shaderDir + "/" + spec.dsDxil, ds)) return out;
            if (!spec.gsDxil.empty() && !readBinary(args.shaderDir + "/" + spec.gsDxil, gs)) return out;
            out.push_back(runNacelleGraphicsBench(spec, vs, ps, hs, ds, gs, w, args.width, args.height, args.iters));
        } else if (spec.kind == PipelineKind::GraphicsMSPS) {
            std::vector<uint8_t> as, ms, ps;
            if (!spec.asDxil.empty() && !readBinary(args.shaderDir + "/" + spec.asDxil, as)) return out;
            if (!readBinary(args.shaderDir + "/" + spec.msDxil, ms) ||
                !readBinary(args.shaderDir + "/" + spec.psDxil, ps)) return out;
            out.push_back(runNacelleMeshBench(spec, ms, ps, w, args.iters, as));
        } else if (spec.kind == PipelineKind::RayTracingInline ||
                   spec.kind == PipelineKind::RayTracingPipeline) {
            std::vector<uint8_t> rt;
            if (!readBinary(args.shaderDir + "/" + spec.rtDxil, rt)) return out;
            out.push_back(runNacelleRaytracingBench(spec, rt, w, args.width, args.height, args.iters));
        }
        return out;
    });
#if defined(NACELLE_ENABLE_INSTRUMENTATION)
    if (capture) {
        NacelleProfileSummary summary;
        const HRESULT status = NacelleEndProfileCapture(capture, profilePath, &summary);
        NacelleDestroyProfileCapture(capture);
        if (status != S_OK) {
            std::fprintf(stderr, "[NacelleProfile] Capture failed or incomplete (%llu dropped)\n",
                         static_cast<unsigned long long>(summary.dropped_events));
            return 1;
        }
    }
#endif
    return result;
}
