#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nacelle_bench {

struct MetalBenchResult {
    std::string backend;
    std::string shaderFormat;
    double compileMs = 0.0;
    double meanMs = 0.0;
    double medianMs = 0.0;
    double minMs = 0.0;
    double checksum = 0.0;
    bool ran = false;
    std::string status = "OK";
};

MetalBenchResult runMetalBench(
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters);

} // namespace nacelle_bench
