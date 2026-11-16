#include "bench_common.h"
#include "bench_vulkan_runners.h"

#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace nacelle_bench;

int main(int argc, char** argv) {
    WorkerCliArgs parsed = parseWorkerCliArgs(argc, argv);
    if (parsed.listDevices) {
        auto devs = listVulkanDevices();
        for (const auto& d : devs) {
            std::cout << d.index << "\t" << d.backendName << "\n";
        }
        return 0;
    }

    return runWorkerMain(argc, argv, [](const BenchmarkSpec& spec, const WorkerCliArgs& args) {
        std::vector<BenchRunResult> out;
        int32_t devIdx = -1;
        if (args.singleBackend.size() > 7 && args.singleBackend.rfind("vulkan:", 0) == 0) {
            devIdx = std::stoi(args.singleBackend.substr(7));
        }
        if (spec.kind == PipelineKind::Compute && !spec.csSpv.empty()) {
            std::vector<uint8_t> spv;
            std::string err;
            if (readBinary(args.shaderDir + "/" + spec.csSpv, spv) && validateSpirvBinary(spv, err)) {
                auto vkRes = runVulkanComputeBench(spec, spv, args.width, args.height, args.maxSteps, args.iters, devIdx);
                out.insert(out.end(), vkRes.begin(), vkRes.end());
            }
        } else if (spec.kind == PipelineKind::GraphicsVSPS && !spec.vsSpv.empty() && !spec.psSpv.empty()) {
            std::vector<uint8_t> vsSpv, psSpv;
            std::string err;
            if (readBinary(args.shaderDir + "/" + spec.vsSpv, vsSpv) && validateSpirvBinary(vsSpv, err) &&
                readBinary(args.shaderDir + "/" + spec.psSpv, psSpv) && validateSpirvBinary(psSpv, err)) {
                auto vkRes = runVulkanGraphicsBench(spec, vsSpv, psSpv, args.width, args.height, args.iters, devIdx);
                out.insert(out.end(), vkRes.begin(), vkRes.end());
            }
        } else if ((spec.kind == PipelineKind::GraphicsVSGSPS || spec.kind == PipelineKind::GraphicsTessGS) &&
                   !spec.vsSpv.empty() && !spec.psSpv.empty()) {
            std::vector<uint8_t> vsSpv, psSpv, gsSpv, hsSpv, dsSpv;
            std::string err;
            bool ok = readBinary(args.shaderDir + "/" + spec.vsSpv, vsSpv) && validateSpirvBinary(vsSpv, err) &&
                      readBinary(args.shaderDir + "/" + spec.psSpv, psSpv) && validateSpirvBinary(psSpv, err);
            if (ok && !spec.gsSpv.empty()) {
                ok = readBinary(args.shaderDir + "/" + spec.gsSpv, gsSpv) && validateSpirvBinary(gsSpv, err);
            }
            if (ok && spec.kind == PipelineKind::GraphicsTessGS) {
                ok = !spec.hsSpv.empty() && !spec.dsSpv.empty() &&
                     readBinary(args.shaderDir + "/" + spec.hsSpv, hsSpv) && validateSpirvBinary(hsSpv, err) &&
                     readBinary(args.shaderDir + "/" + spec.dsSpv, dsSpv) && validateSpirvBinary(dsSpv, err);
            }
            if (ok) {
                auto vkRes = runVulkanGraphicsBench(spec, vsSpv, psSpv, args.width, args.height, args.iters, devIdx, gsSpv, hsSpv, dsSpv);
                out.insert(out.end(), vkRes.begin(), vkRes.end());
            }
        } else if ((spec.kind == PipelineKind::RayTracingInline || spec.kind == PipelineKind::RayTracingPipeline) &&
                   !spec.csSpv.empty()) {
            std::vector<uint8_t> rtSpv;
            std::string err;
            if (readBinary(args.shaderDir + "/" + spec.csSpv, rtSpv) && validateSpirvBinary(rtSpv, err)) {
                auto vkRes = runVulkanRaytracingBench(spec, rtSpv, args.width, args.height, args.iters, devIdx);
                out.insert(out.end(), vkRes.begin(), vkRes.end());
            }
        }
        return out;
    });
}
