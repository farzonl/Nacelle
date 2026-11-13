#include "metal_bench.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstring>

#if defined(NACELLE_HAS_METAL_IRCONVERTER)
#include <metal_irconverter/metal_irconverter.h>
#define IR_PRIVATE_IMPLEMENTATION
#include <metal_irconverter_runtime/metal_irconverter_runtime.h>
#endif

namespace nacelle_bench {

namespace {

struct MetalBenchParams {
    uint32_t width;
    uint32_t height;
    uint32_t maxSteps;
    float time;
};

double computeMedian(std::vector<double> vals) {
    if (vals.empty()) return 0.0;
    std::sort(vals.begin(), vals.end());
    size_t mid = vals.size() / 2;
    if (vals.size() % 2 == 0) {
        return 0.5 * (vals[mid - 1] + vals[mid]);
    }
    return vals[mid];
}

} // namespace

MetalBenchResult runMetalBench(
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    MetalBenchResult res;
    res.backend = "Metal (IRConverter)";
    res.shaderFormat = "DXIL->MTL";

#if !defined(NACELLE_HAS_METAL_IRCONVERTER)
    (void)dxil;
    (void)width;
    (void)height;
    (void)maxSteps;
    (void)iters;
    res.status = "SKIPPED (metal-shaderconverter not found)";
    return res;
#else
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            res.status = "SKIPPED (no Metal device)";
            return res;
        }
        res.backend = std::string("Metal (IRConverter): ") + [[device name] UTF8String];

        auto t0 = std::chrono::steady_clock::now();

        IRRootParameter1 params[3]{};
        params[0].ParameterType = IRRootParameterTypeCBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].Descriptor.RegisterSpace = 0;
        params[0].Descriptor.Flags = IRRootDescriptorFlagNone;
        params[0].ShaderVisibility = IRShaderVisibilityAll;

        params[1].ParameterType = IRRootParameterTypeSRV;
        params[1].Descriptor.ShaderRegister = 0;
        params[1].Descriptor.RegisterSpace = 0;
        params[1].Descriptor.Flags = IRRootDescriptorFlagNone;
        params[1].ShaderVisibility = IRShaderVisibilityAll;

        params[2].ParameterType = IRRootParameterTypeUAV;
        params[2].Descriptor.ShaderRegister = 0;
        params[2].Descriptor.RegisterSpace = 0;
        params[2].Descriptor.Flags = IRRootDescriptorFlagNone;
        params[2].ShaderVisibility = IRShaderVisibilityAll;

        IRVersionedRootSignatureDescriptor rsDesc{};
        rsDesc.version = IRRootSignatureVersion_1_1;
        rsDesc.desc_1_1.NumParameters = 3;
        rsDesc.desc_1_1.pParameters = params;
        rsDesc.desc_1_1.Flags = IRRootSignatureFlagNone;

        IRError* irErr = nullptr;
        IRRootSignature* rootSig = IRRootSignatureCreateFromDescriptor(&rsDesc, &irErr);
        if (!rootSig) {
            if (irErr) IRErrorDestroy(irErr);
            res.status = "FAILED (IRRootSignatureCreateFromDescriptor)";
            return res;
        }

        IRCompiler* compiler = IRCompilerCreate();
        IRCompilerSetGlobalRootSignature(compiler, rootSig);
        IRCompilerSetEntryPointName(compiler, "main");

        IRObject* dxilObj = IRObjectCreateFromDXIL(dxil.data(), dxil.size(), IRBytecodeOwnershipNone);
        IRObject* mtlObj = IRCompilerAllocCompileAndLink(compiler, "main", dxilObj, &irErr);
        if (!mtlObj) {
            if (irErr) IRErrorDestroy(irErr);
            IRObjectDestroy(dxilObj);
            IRCompilerDestroy(compiler);
            IRRootSignatureDestroy(rootSig);
            res.status = "FAILED (IRCompilerAllocCompileAndLink)";
            return res;
        }

        IRMetalLibBinary* libBin = IRMetalLibBinaryCreate();
        if (!IRObjectGetMetalLibBinary(mtlObj, IRShaderStageCompute, libBin)) {
            IRMetalLibBinaryDestroy(libBin);
            IRObjectDestroy(mtlObj);
            IRObjectDestroy(dxilObj);
            IRCompilerDestroy(compiler);
            IRRootSignatureDestroy(rootSig);
            res.status = "FAILED (IRObjectGetMetalLibBinary)";
            return res;
        }

        IRShaderReflection* refl = IRShaderReflectionCreate();
        IRObjectGetReflection(mtlObj, IRShaderStageCompute, refl);
        const char* entryNameCStr = IRShaderReflectionGetEntryPointFunctionName(refl);
        NSString* entryName = entryNameCStr ? [NSString stringWithUTF8String:entryNameCStr] : @"main";

        size_t resCount = IRShaderReflectionGetResourceCount(refl);
        std::vector<IRResourceLocation> resLocs(resCount);
        if (resCount > 0) {
            IRShaderReflectionGetResourceLocations(refl, resLocs.data());
        }

        dispatch_data_t libData = IRMetalLibGetBytecodeData(libBin);
        NSError* mtlErr = nil;
        id<MTLLibrary> mtlLib = [device newLibraryWithData:libData error:&mtlErr];
        id<MTLFunction> fn = mtlLib ? [mtlLib newFunctionWithName:entryName] : nil;
        id<MTLComputePipelineState> pso = fn ? [device newComputePipelineStateWithFunction:fn error:&mtlErr] : nil;

        auto t1 = std::chrono::steady_clock::now();
        res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        IRShaderReflectionDestroy(refl);
        IRMetalLibBinaryDestroy(libBin);
        IRObjectDestroy(mtlObj);
        IRObjectDestroy(dxilObj);
        IRCompilerDestroy(compiler);
        IRRootSignatureDestroy(rootSig);

        if (!pso) {
            res.status = "FAILED (Metal PSO creation)";
            return res;
        }

        id<MTLBuffer> cbBuf = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        auto* cbPtr = static_cast<MetalBenchParams*>([cbBuf contents]);
        cbPtr->width = width;
        cbPtr->height = height;
        cbPtr->maxSteps = maxSteps;
        cbPtr->time = 0.0f;

        id<MTLBuffer> inBuf = [device newBufferWithLength:4 * 16 options:MTLResourceStorageModeShared];
        float spheres[16] = {
            0.0f, 0.2f, 0.5f, 0.8f,
            -1.3f, 0.0f, 1.0f, 0.6f,
            1.3f, 0.1f, 0.9f, 0.65f,
            0.0f, 1.1f, 1.5f, 0.5f,
        };
        std::memcpy([inBuf contents], spheres, sizeof(spheres));

        size_t outBytes = static_cast<size_t>(width) * height * 16;
        id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
        std::memset([outBuf contents], 0, outBytes);

        uint32_t abSize = 256;
        for (const auto& loc : resLocs) {
            if (loc.topLevelOffset + sizeof(IRDescriptorTableEntry) > abSize) {
                abSize = loc.topLevelOffset + static_cast<uint32_t>(sizeof(IRDescriptorTableEntry));
            }
        }
        id<MTLBuffer> abBuf = [device newBufferWithLength:abSize options:MTLResourceStorageModeShared];
        auto* abBytes = static_cast<uint8_t*>([abBuf contents]);
        std::memset(abBytes, 0, abSize);
        for (const auto& loc : resLocs) {
            auto* entry = reinterpret_cast<IRDescriptorTableEntry*>(abBytes + loc.topLevelOffset);
            if (loc.resourceType == IRResourceTypeCBV && loc.slot == 0) {
                IRBufferView view{cbBuf, 0, 256, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            } else if (loc.resourceType == IRResourceTypeSRV && loc.slot == 0) {
                IRBufferView view{inBuf, 0, sizeof(spheres), nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            } else if (loc.resourceType == IRResourceTypeUAV && loc.slot == 0) {
                IRBufferView view{outBuf, 0, outBytes, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            }
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        MTLSize tgSize = MTLSizeMake(8, 8, 1);
        MTLSize gridSize = MTLSizeMake((width + 7) / 8, (height + 7) / 8, 1);

        auto dispatchOnce = [&]() {
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc useResource:cbBuf usage:MTLResourceUsageRead];
            [enc useResource:inBuf usage:MTLResourceUsageRead];
            [enc useResource:outBuf usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
            [enc setBuffer:abBuf offset:0 atIndex:kIRArgumentBufferBindPoint];
            [enc dispatchThreadgroups:gridSize threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
        };

        // Warmup
        dispatchOnce();

        std::vector<double> samples;
        samples.reserve(iters);
        double totalMs = 0.0;
        for (uint32_t i = 0; i < iters; ++i) {
            auto s0 = std::chrono::steady_clock::now();
            dispatchOnce();
            auto s1 = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double, std::milli>(s1 - s0).count();
            samples.push_back(dt);
            totalMs += dt;
        }

        const auto* outPtr = static_cast<const float*>([outBuf contents]);
        double sum = 0.0;
        size_t floatCount = static_cast<size_t>(width) * height * 4;
        for (size_t i = 0; i < floatCount; ++i) {
            sum += outPtr[i];
        }

        res.meanMs = totalMs / static_cast<double>(iters);
        res.medianMs = computeMedian(samples);
        res.minMs = *std::min_element(samples.begin(), samples.end());
        res.checksum = sum;
        res.ran = true;
        res.status = "OK";
    }
    return res;
#endif
}

} // namespace nacelle_bench
