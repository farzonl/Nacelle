#include "metal_bench.h"
#include "bench_types.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <simd/simd.h>

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

void finalizeSamples(MetalBenchResult& res, const std::vector<double>& samples) {
    if (samples.empty()) return;
    double sum = 0.0;
    for (double s : samples) sum += s;
    res.meanMs = sum / static_cast<double>(samples.size());
    res.medianMs = nacelle_bench::computeMedian(samples);
    res.minMs = *std::min_element(samples.begin(), samples.end());
}

#if defined(NACELLE_HAS_METAL_IRCONVERTER)
struct CompiledIRStage {
    id<MTLLibrary> lib = nil;
    id<MTLLibrary> stageInLib = nil;
    id<MTLFunction> fn = nil;
    std::string entryName = "main";
    std::vector<IRResourceLocation> resLocs;
    IRVersionedVSInfo vsInfo{};
    IRVersionedGSInfo gsInfo{};
    IRVersionedHSInfo hsInfo{};
    IRVersionedDSInfo dsInfo{};
    bool hasVSInfo = false;
    bool hasGSInfo = false;
    bool hasHSInfo = false;
    bool hasDSInfo = false;
};

bool compileDxilToMetalStage(
    id<MTLDevice> device,
    const std::vector<uint8_t>& dxil,
    IRRootSignature* rootSig,
    IRShaderStage stage,
    CompiledIRStage& out,
    std::string& errOut,
    bool enableGeomTessEmulation = false,
    const char* entryPoint = nullptr,
    void* rtCfg = nullptr) {
    IRCompiler* compiler = IRCompilerCreate();
    IRCompilerSetGlobalRootSignature(compiler, rootSig);
    if (enableGeomTessEmulation) {
        IRCompilerEnableGeometryAndTessellationEmulation(compiler, true);
    }
#if defined(IR_VERSION_MAJOR) && IR_VERSION_MAJOR >= 3
    if (rtCfg) {
        IRCompilerSetRayTracingPipelineConfiguration(compiler, static_cast<IRRayTracingPipelineConfiguration*>(rtCfg));
    }
#else
    (void)rtCfg;
#endif

    IRObject* dxilObj = IRObjectCreateFromDXIL(dxil.data(), dxil.size(), IRBytecodeOwnershipNone);
    IRError* irErr = nullptr;
    IRObject* mtlObj = IRCompilerAllocCompileAndLink(compiler, entryPoint, dxilObj, &irErr);
    if (!mtlObj) {
        if (irErr) {
            uint32_t code = IRErrorGetCode(irErr);
            errOut = "IRCompilerAllocCompileAndLink code=" + std::to_string(code);
            IRErrorDestroy(irErr);
        } else {
            errOut = "IRCompilerAllocCompileAndLink";
        }
        IRObjectDestroy(dxilObj);
        IRCompilerDestroy(compiler);
        return false;
    }

    IRMetalLibBinary* libBin = IRMetalLibBinaryCreate();
    if (!IRObjectGetMetalLibBinary(mtlObj, stage, libBin)) {
        errOut = "IRObjectGetMetalLibBinary";
        IRMetalLibBinaryDestroy(libBin);
        IRObjectDestroy(mtlObj);
        IRObjectDestroy(dxilObj);
        IRCompilerDestroy(compiler);
        return false;
    }

    IRShaderReflection* refl = IRShaderReflectionCreate();
    IRObjectGetReflection(mtlObj, stage, refl);
    const char* entryNameCStr = IRShaderReflectionGetEntryPointFunctionName(refl);
    out.entryName = entryNameCStr ? entryNameCStr : "main";
    NSString* entryName = [NSString stringWithUTF8String:out.entryName.c_str()];

    size_t resCount = IRShaderReflectionGetResourceCount(refl);
    out.resLocs.resize(resCount);
    if (resCount > 0) {
        IRShaderReflectionGetResourceLocations(refl, out.resLocs.data());
    }
    if (stage == IRShaderStageVertex) {
        out.hasVSInfo = IRShaderReflectionCopyVertexInfo(refl, IRReflectionVersion_1_0, &out.vsInfo);
    } else if (stage == IRShaderStageGeometry) {
        out.hasGSInfo = IRShaderReflectionCopyGeometryInfo(refl, IRReflectionVersion_1_0, &out.gsInfo);
    } else if (stage == IRShaderStageHull) {
        out.hasHSInfo = IRShaderReflectionCopyHullInfo(refl, IRReflectionVersion_1_0, &out.hsInfo);
    } else if (stage == IRShaderStageDomain) {
        out.hasDSInfo = IRShaderReflectionCopyDomainInfo(refl, IRReflectionVersion_1_0, &out.dsInfo);
    }

    dispatch_data_t libData = IRMetalLibGetBytecodeData(libBin);
    NSError* mtlErr = nil;
    out.lib = [device newLibraryWithData:libData error:&mtlErr];
    out.fn = out.lib ? [out.lib newFunctionWithName:entryName] : nil;

    if (enableGeomTessEmulation && stage == IRShaderStageVertex) {
        IRVersionedInputLayoutDescriptor emptyLayout{};
        emptyLayout.version = IRInputLayoutDescriptorVersion_1;
        emptyLayout.desc_1_0.numElements = 0;
        IRMetalLibBinary* stageInBin = IRMetalLibBinaryCreate();
        if (IRMetalLibSynthesizeStageInFunction(compiler, refl, &emptyLayout, stageInBin)) {
            dispatch_data_t siData = IRMetalLibGetBytecodeData(stageInBin);
            NSError* siErr = nil;
            out.stageInLib = [device newLibraryWithData:siData error:&siErr];
        }
        IRMetalLibBinaryDestroy(stageInBin);
    }

    IRShaderReflectionDestroy(refl);
    IRMetalLibBinaryDestroy(libBin);
    IRObjectDestroy(mtlObj);
    IRObjectDestroy(dxilObj);
    IRCompilerDestroy(compiler);
    return out.fn != nil;
}
#endif

} // namespace

MetalBenchResult runMetalComputeBench(
    const std::string& benchId,
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    MetalBenchResult res;
    res.backend = "Metal (IRConverter)";
    res.shaderFormat = "DXIL->MTL";

#if !defined(NACELLE_HAS_METAL_IRCONVERTER)
    (void)benchId;
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

        IRRootParameter1 params[5]{};
        params[0].ParameterType = IRRootParameterTypeCBV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].Descriptor.RegisterSpace = 0;
        params[0].ShaderVisibility = IRShaderVisibilityAll;

        params[1].ParameterType = IRRootParameterTypeSRV;
        params[1].Descriptor.ShaderRegister = 0;
        params[1].Descriptor.RegisterSpace = 0;
        params[1].ShaderVisibility = IRShaderVisibilityAll;

        params[2].ParameterType = IRRootParameterTypeSRV;
        params[2].Descriptor.ShaderRegister = 1;
        params[2].Descriptor.RegisterSpace = 0;
        params[2].ShaderVisibility = IRShaderVisibilityAll;

        params[3].ParameterType = IRRootParameterTypeUAV;
        params[3].Descriptor.ShaderRegister = 0;
        params[3].Descriptor.RegisterSpace = 0;
        params[3].ShaderVisibility = IRShaderVisibilityAll;

        params[4].ParameterType = IRRootParameterTypeUAV;
        params[4].Descriptor.ShaderRegister = 1;
        params[4].Descriptor.RegisterSpace = 0;
        params[4].ShaderVisibility = IRShaderVisibilityAll;

        IRVersionedRootSignatureDescriptor rsDesc{};
        rsDesc.version = IRRootSignatureVersion_1_1;
        rsDesc.desc_1_1.NumParameters = 5;
        rsDesc.desc_1_1.pParameters = params;
        rsDesc.desc_1_1.Flags = IRRootSignatureFlagNone;

        IRError* irErr = nullptr;
        IRRootSignature* rootSig = IRRootSignatureCreateFromDescriptor(&rsDesc, &irErr);
        if (!rootSig) {
            if (irErr) IRErrorDestroy(irErr);
            res.status = "FAILED (IRRootSignatureCreateFromDescriptor)";
            return res;
        }

        CompiledIRStage csStage;
        std::string irErrStr;
        if (!compileDxilToMetalStage(device, dxil, rootSig, IRShaderStageCompute, csStage, irErrStr)) {
            IRRootSignatureDestroy(rootSig);
            res.status = "FAILED (" + irErrStr + ")";
            return res;
        }

        NSError* mtlErr = nil;
        id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:csStage.fn error:&mtlErr];

        auto t1 = std::chrono::steady_clock::now();
        res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        IRRootSignatureDestroy(rootSig);

        if (!pso) {
            res.status = "FAILED (Metal Compute PSO creation)";
            return res;
        }

        uint32_t longW = width * 2;
        uint32_t longH = height * 2;
        size_t maxOutBytes = 16 * 1024 * 1024;
        if (benchId == "bench_raymarch") {
            maxOutBytes = std::max<size_t>(maxOutBytes, static_cast<size_t>(longW) * longH * 16);
        }
        size_t auxBytes = 4 * 1024 * 1024;

        id<MTLBuffer> cbBuf = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        id<MTLBuffer> srv0 = [device newBufferWithLength:auxBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> srv1 = [device newBufferWithLength:auxBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> uav0 = [device newBufferWithLength:maxOutBytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> uav1 = [device newBufferWithLength:auxBytes options:MTLResourceStorageModeShared];

        std::memset([srv0 contents], 0, auxBytes);
        std::memset([srv1 contents], 0, auxBytes);
        std::memset([uav0 contents], 0, maxOutBytes);
        std::memset([uav1 contents], 0, auxBytes);

        if (benchId == "bench_raymarch") {
            float spheres[16] = {
                0.0f, 0.2f, 0.5f, 0.8f,
                -1.3f, 0.0f, 1.0f, 0.6f,
                1.3f, 0.1f, 0.9f, 0.65f,
                0.0f, 1.1f, 1.5f, 0.5f,
            };
            std::memcpy([srv0 contents], spheres, sizeof(spheres));
        } else if (benchId == "matmul" || benchId == "tiled_matmul" || benchId == "linalg_matmul") {
            auto* a = static_cast<float*>([srv0 contents]);
            auto* b = static_cast<float*>([srv1 contents]);
            for (size_t i = 0; i < 256 * 256; ++i) {
                a[i] = static_cast<float>((i % 17) + 1) * 0.01f;
                b[i] = static_cast<float>((i % 13) + 1) * 0.01f;
            }
        } else if (benchId == "prefix_sum_groupshared") {
            auto* p = static_cast<uint32_t*>([srv0 contents]);
            for (size_t i = 0; i < 65536 * 4; ++i) {
                p[i] = static_cast<uint32_t>((i & 63u) + 1u);
            }
        }

        uint32_t abSize = 256;
        for (const auto& loc : csStage.resLocs) {
            if (loc.topLevelOffset + sizeof(IRDescriptorTableEntry) > abSize) {
                abSize = loc.topLevelOffset + static_cast<uint32_t>(sizeof(IRDescriptorTableEntry));
            }
        }
        id<MTLBuffer> abBuf = [device newBufferWithLength:abSize options:MTLResourceStorageModeShared];
        auto* abBytes = static_cast<uint8_t*>([abBuf contents]);
        std::memset(abBytes, 0, abSize);
        for (const auto& loc : csStage.resLocs) {
            auto* entry = reinterpret_cast<IRDescriptorTableEntry*>(abBytes + loc.topLevelOffset);
            if (loc.resourceType == IRResourceTypeCBV && loc.slot == 0) {
                IRBufferView view{cbBuf, 0, 256, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            } else if (loc.resourceType == IRResourceTypeSRV && loc.slot == 0) {
                IRBufferView view{srv0, 0, auxBytes, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            } else if (loc.resourceType == IRResourceTypeSRV && loc.slot == 1) {
                IRBufferView view{srv1, 0, auxBytes, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            } else if (loc.resourceType == IRResourceTypeUAV && loc.slot == 0) {
                IRBufferView view{uav0, 0, maxOutBytes, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            } else if (loc.resourceType == IRResourceTypeUAV && loc.slot == 1) {
                IRBufferView view{uav1, 0, auxBytes, nil, 0, false};
                IRDescriptorTableSetBufferView(entry, &view);
            }
        }

        id<MTLCommandQueue> queue = [device newCommandQueue];
        bool is2D = (benchId == "bench_raymarch" || benchId == "matmul" || benchId == "tiled_matmul");
        MTLSize tgSize = (benchId == "linalg_matmul") ? MTLSizeMake(32, 1, 1)
                       : (benchId == "tiled_matmul")  ? MTLSizeMake(16, 16, 1)
                                                      : (is2D ? MTLSizeMake(8, 8, 1) : MTLSizeMake(64, 1, 1));

        auto getDims = [&](int mode, uint32_t& gx, uint32_t& gy, uint32_t& cbW, uint32_t& cbH, uint32_t& cbSteps) {
            if (benchId == "bench_raymarch") {
                cbW = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
                cbH = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
                cbSteps = (mode == 0) ? 16 : (mode == 1 ? maxSteps : maxSteps * 2);
                gx = (cbW + 7) / 8;
                gy = (cbH + 7) / 8;
            } else if (benchId == "matmul") {
                uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
                cbW = n;
                cbH = n;
                cbSteps = n;
                gx = (n + 7) / 8;
                gy = (n + 7) / 8;
            } else if (benchId == "tiled_matmul") {
                uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
                cbW = n;
                cbH = n;
                cbSteps = n;
                gx = (n + 15) / 16;
                gy = (n + 15) / 16;
            } else if (benchId == "linalg_matmul") {
                uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
                cbW = n;
                cbH = n;
                cbSteps = n;
                gx = (n + 31) / 32;
                gy = (n + 31) / 32;
            } else {
                gx = (mode == 0) ? 64 : (mode == 1 ? 512 : 4096);
                gy = 1;
                cbW = gx * 64;
                cbH = 1;
                cbSteps = 1;
            }
        };

        auto dispatchOnce = [&](uint32_t gx, uint32_t gy, uint32_t cbW, uint32_t cbH, uint32_t cbSteps) -> double {
            if (benchId == "bench_raymarch") {
                RaymarchCB cb{cbW, cbH, cbSteps, 0.0f};
                std::memcpy([cbBuf contents], &cb, sizeof(cb));
            } else if (benchId == "matmul" || benchId == "tiled_matmul" || benchId == "linalg_matmul") {
                MatmulCB cb{cbW, cbH, cbSteps, 0};
                std::memcpy([cbBuf contents], &cb, sizeof(cb));
            }
            auto s0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc useResource:cbBuf usage:MTLResourceUsageRead];
            [enc useResource:srv0 usage:MTLResourceUsageRead];
            [enc useResource:srv1 usage:MTLResourceUsageRead];
            [enc useResource:uav0 usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
            [enc useResource:uav1 usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
            [enc setBuffer:abBuf offset:0 atIndex:kIRArgumentBufferBindPoint];
            [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:tgSize];
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count();
        };

        uint32_t gx = 1, gy = 1, cbW = 1, cbH = 1, cbSteps = 1;
        getDims(1, gx, gy, cbW, cbH, cbSteps);

        res.firstDispatchMs = dispatchOnce(gx, gy, cbW, cbH, cbSteps);

        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            samples.push_back(dispatchOnce(gx, gy, cbW, cbH, cbSteps));
        }

        getDims(0, gx, gy, cbW, cbH, cbSteps);
        res.shortJobMs = dispatchOnce(gx, gy, cbW, cbH, cbSteps);
        getDims(2, gx, gy, cbW, cbH, cbSteps);
        res.longJobMs = dispatchOnce(gx, gy, cbW, cbH, cbSteps);

        const auto* outPtr = static_cast<const float*>([uav0 contents]);
        double sum = 0.0;
        size_t floatCount = (benchId == "bench_raymarch")
            ? static_cast<size_t>(width) * height * 4
            : std::min<size_t>(4096, maxOutBytes / sizeof(float));
        for (size_t i = 0; i < floatCount; ++i) {
            sum += outPtr[i];
        }

        finalizeSamples(res, samples);
        res.checksum = sum;
        res.ran = true;
        res.status = "OK";
    }
    return res;
#endif
}

MetalBenchResult runMetalGraphicsBench(
    const std::string& benchId,
    const std::string& category,
    const std::vector<uint8_t>& vsDxil,
    const std::vector<uint8_t>& psDxil,
    uint32_t width,
    uint32_t height,
    uint32_t iters,
    const std::vector<uint8_t>& hsDxil,
    const std::vector<uint8_t>& dsDxil,
    const std::vector<uint8_t>& gsDxil) {
    MetalBenchResult res;
    res.backend = "Metal (IRConverter)";
    res.shaderFormat = "DXIL->MTL";

#if !defined(NACELLE_HAS_METAL_IRCONVERTER)
    (void)benchId;
    (void)category;
    (void)vsDxil;
    (void)psDxil;
    (void)hsDxil;
    (void)dsDxil;
    (void)gsDxil;
    (void)width;
    (void)height;
    (void)iters;
    res.status = "SKIPPED (metal-shaderconverter not found)";
    return res;
#else
    (void)benchId;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            res.status = "SKIPPED (no Metal device)";
            return res;
        }
        res.backend = std::string("Metal (IRConverter): ") + [[device name] UTF8String];

        const bool hasTess = (!hsDxil.empty() && !dsDxil.empty());
        const bool hasGS = !gsDxil.empty();
        const bool useEmulation = hasTess || hasGS;
#if !defined(IR_VERSION_MAJOR) || (IR_VERSION_MAJOR < 3)
        if (useEmulation) {
            res.status = "SKIPPED (Metal IRConverter < 3.0)";
            return res;
        }
#endif

        auto t0 = std::chrono::steady_clock::now();

        IRRootParameter1 param{};
        param.ParameterType = IRRootParameterTypeCBV;
        param.Descriptor.ShaderRegister = 0;
        param.Descriptor.RegisterSpace = 0;
        param.ShaderVisibility = IRShaderVisibilityAll;

        IRVersionedRootSignatureDescriptor rsDesc{};
        rsDesc.version = IRRootSignatureVersion_1_1;
        rsDesc.desc_1_1.NumParameters = 1;
        rsDesc.desc_1_1.pParameters = &param;
        rsDesc.desc_1_1.Flags = IRRootSignatureFlagAllowInputAssemblerInputLayout;

        IRError* irErr = nullptr;
        IRRootSignature* rootSig = IRRootSignatureCreateFromDescriptor(&rsDesc, &irErr);
        if (!rootSig) {
            if (irErr) IRErrorDestroy(irErr);
            res.status = "FAILED (IRRootSignatureCreateFromDescriptor)";
            return res;
        }

        bool isVertexMesh = (benchId == "bench_vertex_mesh");
        bool isPixel = (category == "Pixel");
        bool isGeometry = (category == "Geometry");
        bool isTess = (category == "Tessellation");

        CompiledIRStage vsStage, hsStage, dsStage, gsStage, psStage;
        std::string irErrStr;
        if (!compileDxilToMetalStage(device, vsDxil, rootSig, IRShaderStageVertex, vsStage, irErrStr, useEmulation) ||
            !compileDxilToMetalStage(device, psDxil, rootSig, IRShaderStageFragment, psStage, irErrStr, useEmulation) ||
            (hasTess && (!compileDxilToMetalStage(device, hsDxil, rootSig, IRShaderStageHull, hsStage, irErrStr, true) ||
                         !compileDxilToMetalStage(device, dsDxil, rootSig, IRShaderStageDomain, dsStage, irErrStr, true))) ||
            (hasGS && !compileDxilToMetalStage(device, gsDxil, rootSig, IRShaderStageGeometry, gsStage, irErrStr, true))) {
            IRRootSignatureDestroy(rootSig);
            res.status = "FAILED (" + irErrStr + " stage compile)";
            return res;
        }

        NSError* mtlErr = nil;
        id<MTLRenderPipelineState> pso = nil;
        id<MTLDepthStencilState> dsState = nil;
#if defined(IR_VERSION_MAJOR) && IR_VERSION_MAJOR >= 3
        IRRuntimeGeometryPipelineConfig geomCfg{};
        IRRuntimeTessellationPipelineConfig tessCfg{};
#endif

        if (!useEmulation) {
            MTLRenderPipelineDescriptor* psoDesc = [[MTLRenderPipelineDescriptor alloc] init];
            psoDesc.vertexFunction = vsStage.fn;
            psoDesc.fragmentFunction = psStage.fn;
            psoDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            if (isVertexMesh) {
                psoDesc.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
                MTLVertexDescriptor* vDesc = [MTLVertexDescriptor vertexDescriptor];
                vDesc.attributes[kIRStageInAttributeStartIndex + 0].format = MTLVertexFormatFloat3;
                vDesc.attributes[kIRStageInAttributeStartIndex + 0].offset = 0;
                vDesc.attributes[kIRStageInAttributeStartIndex + 0].bufferIndex = kIRVertexBufferBindPoint + 0;
                vDesc.attributes[kIRStageInAttributeStartIndex + 1].format = MTLVertexFormatShort4Normalized;
                vDesc.attributes[kIRStageInAttributeStartIndex + 1].offset = 12;
                vDesc.attributes[kIRStageInAttributeStartIndex + 1].bufferIndex = kIRVertexBufferBindPoint + 0;
                vDesc.attributes[kIRStageInAttributeStartIndex + 2].format = MTLVertexFormatUShort2Normalized;
                vDesc.attributes[kIRStageInAttributeStartIndex + 2].offset = 20;
                vDesc.attributes[kIRStageInAttributeStartIndex + 2].bufferIndex = kIRVertexBufferBindPoint + 0;
                vDesc.attributes[kIRStageInAttributeStartIndex + 3].format = MTLVertexFormatUChar4Normalized;
                vDesc.attributes[kIRStageInAttributeStartIndex + 3].offset = 24;
                vDesc.attributes[kIRStageInAttributeStartIndex + 3].bufferIndex = kIRVertexBufferBindPoint + 0;
                vDesc.attributes[kIRStageInAttributeStartIndex + 4].format = MTLVertexFormatUChar4;
                vDesc.attributes[kIRStageInAttributeStartIndex + 4].offset = 28;
                vDesc.attributes[kIRStageInAttributeStartIndex + 4].bufferIndex = kIRVertexBufferBindPoint + 0;
                vDesc.layouts[kIRVertexBufferBindPoint + 0].stride = sizeof(MeshBenchVertex);
                vDesc.layouts[kIRVertexBufferBindPoint + 0].stepFunction = MTLVertexStepFunctionPerVertex;
                psoDesc.vertexDescriptor = vDesc;

                MTLDepthStencilDescriptor* dsd = [[MTLDepthStencilDescriptor alloc] init];
                dsd.depthCompareFunction = MTLCompareFunctionLess;
                dsd.depthWriteEnabled = YES;
                dsState = [device newDepthStencilStateWithDescriptor:dsd];
            } else if (isPixel) {
                MTLVertexDescriptor* vDesc = [MTLVertexDescriptor vertexDescriptor];
                vDesc.attributes[kIRStageInAttributeStartIndex + 0].format = MTLVertexFormatFloat3;
                vDesc.attributes[kIRStageInAttributeStartIndex + 0].offset = 0;
                vDesc.attributes[kIRStageInAttributeStartIndex + 0].bufferIndex = kIRVertexBufferBindPoint + 0;
                vDesc.layouts[kIRVertexBufferBindPoint + 0].stride = 3 * sizeof(float);
                vDesc.layouts[kIRVertexBufferBindPoint + 0].stepFunction = MTLVertexStepFunctionPerVertex;
                psoDesc.vertexDescriptor = vDesc;
            }
            pso = [device newRenderPipelineStateWithDescriptor:psoDesc error:&mtlErr];
#if defined(IR_VERSION_MAJOR) && IR_VERSION_MAJOR >= 3
        } else if (!hasTess && hasGS) {
            MTLMeshRenderPipelineDescriptor* meshDesc = [[MTLMeshRenderPipelineDescriptor alloc] init];
            meshDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
            geomCfg.gsVertexSizeInBytes = vsStage.hasVSInfo ? vsStage.vsInfo.info_1_0.vertex_output_size_in_bytes : 32;
            geomCfg.gsMaxInputPrimitivesPerMeshThreadgroup = gsStage.hasGSInfo ? gsStage.gsInfo.info_1_0.max_input_primitives_per_mesh_threadgroup : 32;

            IRGeometryEmulationPipelineDescriptor geoDesc{};
            geoDesc.stageInLibrary = vsStage.stageInLib;
            geoDesc.vertexLibrary = vsStage.lib;
            geoDesc.vertexFunctionName = vsStage.entryName.c_str();
            geoDesc.geometryLibrary = gsStage.lib;
            geoDesc.geometryFunctionName = gsStage.entryName.c_str();
            geoDesc.fragmentLibrary = psStage.lib;
            geoDesc.fragmentFunctionName = psStage.entryName.c_str();
            geoDesc.basePipelineDescriptor = meshDesc;
            geoDesc.pipelineConfig = geomCfg;
            pso = IRRuntimeNewGeometryEmulationPipeline(device, &geoDesc, &mtlErr);
        } else {
            MTLMeshRenderPipelineDescriptor* meshDesc = [[MTLMeshRenderPipelineDescriptor alloc] init];
            meshDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;

            auto toRtOutPrim = [](IRTessellatorOutputPrimitive p) -> IRRuntimeTessellatorOutputPrimitive {
                switch (p) {
                    case IRTessellatorOutputPoint: return IRRuntimeTessellatorOutputPoint;
                    case IRTessellatorOutputLine: return IRRuntimeTessellatorOutputLine;
                    case IRTessellatorOutputTriangleCW: return IRRuntimeTessellatorOutputTriangleCW;
                    case IRTessellatorOutputTriangleCCW: return IRRuntimeTessellatorOutputTriangleCCW;
                    default: return IRRuntimeTessellatorOutputTriangleCW;
                }
            };

            tessCfg.outputPrimitiveType = hsStage.hasHSInfo ? toRtOutPrim(hsStage.hsInfo.info_1_0.tessellator_output_primitive) : IRRuntimeTessellatorOutputTriangleCW;
            tessCfg.vsOutputSizeInBytes = vsStage.hasVSInfo ? vsStage.vsInfo.info_1_0.vertex_output_size_in_bytes : 32;
            tessCfg.gsMaxInputPrimitivesPerMeshThreadgroup = hasGS && gsStage.hasGSInfo
                ? gsStage.gsInfo.info_1_0.max_input_primitives_per_mesh_threadgroup
                : (dsStage.hasDSInfo ? dsStage.dsInfo.info_1_0.max_input_prims_per_mesh_threadgroup : 32);
            tessCfg.hsMaxPatchesPerObjectThreadgroup = hsStage.hasHSInfo ? hsStage.hsInfo.info_1_0.max_patches_per_object_threadgroup : 8;
            tessCfg.hsInputControlPointCount = hsStage.hasHSInfo ? hsStage.hsInfo.info_1_0.input_control_point_count : 3;
            tessCfg.hsMaxObjectThreadsPerThreadgroup = hsStage.hasHSInfo ? hsStage.hsInfo.info_1_0.max_object_threads_per_patch : 3;
            tessCfg.hsMaxTessellationFactor = hsStage.hasHSInfo ? hsStage.hsInfo.info_1_0.max_tessellation_factor : 16.0f;
            tessCfg.gsInstanceCount = hasGS && gsStage.hasGSInfo ? gsStage.gsInfo.info_1_0.instance_count : 1;

            const char* passthroughGS = (tessCfg.outputPrimitiveType == IRRuntimeTessellatorOutputPoint)
                ? kIRPointPassthroughGeometryShader
                : (tessCfg.outputPrimitiveType == IRRuntimeTessellatorOutputLine
                    ? kIRLinePassthroughGeometryShader
                    : kIRTrianglePassthroughGeometryShader);

            IRGeometryTessellationEmulationPipelineDescriptor tessDesc{};
            tessDesc.stageInLibrary = vsStage.stageInLib;
            tessDesc.vertexLibrary = vsStage.lib;
            tessDesc.vertexFunctionName = vsStage.entryName.c_str();
            tessDesc.hullLibrary = hsStage.lib;
            tessDesc.hullFunctionName = hsStage.entryName.c_str();
            tessDesc.domainLibrary = dsStage.lib;
            tessDesc.domainFunctionName = dsStage.entryName.c_str();
            tessDesc.geometryLibrary = hasGS ? gsStage.lib : nil;
            tessDesc.geometryFunctionName = hasGS ? gsStage.entryName.c_str() : passthroughGS;
            tessDesc.fragmentLibrary = psStage.lib;
            tessDesc.fragmentFunctionName = psStage.entryName.c_str();
            tessDesc.basePipelineDescriptor = meshDesc;
            tessDesc.pipelineConfig = tessCfg;
            pso = IRRuntimeNewGeometryTessellationEmulationPipeline(device, &tessDesc, &mtlErr);
#endif
        }

        auto t1 = std::chrono::steady_clock::now();
        res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        IRRootSignatureDestroy(rootSig);

        if (!pso) {
            res.status = mtlErr
                ? std::string("FAILED (Metal Render PSO: ") + [[mtlErr localizedDescription] UTF8String] + ")"
                : "FAILED (Metal Render PSO creation)";
            return res;
        }

        uint32_t maxW = (isPixel || isVertexMesh) ? std::max(128u, width * 2) : 128;
        uint32_t maxH = (isPixel || isVertexMesh) ? std::max(128u, height * 2) : 128;
        MTLTextureDescriptor* texDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                           width:maxW
                                                                                          height:maxH
                                                                                       mipmapped:NO];
        texDesc.usage = MTLTextureUsageRenderTarget;
        texDesc.storageMode = MTLStorageModeShared;
        id<MTLTexture> rtTex = [device newTextureWithDescriptor:texDesc];

        id<MTLTexture> depthTex = nil;
        MeshBenchSceneData meshScenes[3];
        id<MTLBuffer> meshVBs[3] = {nil, nil, nil};
        id<MTLBuffer> meshIBs[3] = {nil, nil, nil};
        if (isVertexMesh) {
            MTLTextureDescriptor* dDesc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                             width:maxW
                                                                                            height:maxH
                                                                                         mipmapped:NO];
            dDesc.usage = MTLTextureUsageRenderTarget;
            dDesc.storageMode = MTLStorageModePrivate;
            depthTex = [device newTextureWithDescriptor:dDesc];
            for (int m = 0; m < 3; ++m) {
                uint32_t g = 0, mrw = 0, mrh = 0;
                getMeshBenchGridAndDims(m, width, height, g, mrw, mrh);
                meshScenes[m] = buildMeshBenchScene(g, g);
                meshVBs[m] = [device newBufferWithBytes:meshScenes[m].vertices.data()
                                                 length:meshScenes[m].vertices.size() * sizeof(MeshBenchVertex)
                                                options:MTLResourceStorageModeShared];
                meshIBs[m] = [device newBufferWithBytes:meshScenes[m].indices.data()
                                                 length:meshScenes[m].indices.size() * sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
            }
        }

        size_t cbBytes = isVertexMesh ? sizeof(MeshBenchConstants) : 256;
        id<MTLBuffer> cbBuf = [device newBufferWithLength:cbBytes options:MTLResourceStorageModeShared];
        const float quadVerts[18] = {
            -1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f, -1.0f,  1.0f, 0.0f,
        };
        id<MTLBuffer> vbBuf = [device newBufferWithBytes:quadVerts length:sizeof(quadVerts) options:MTLResourceStorageModeShared];
        id<MTLBuffer> emptyVBs = [device newBufferWithLength:sizeof(IRRuntimeVertexBuffers) options:MTLResourceStorageModeShared];
        std::memset([emptyVBs contents], 0, sizeof(IRRuntimeVertexBuffers));

        auto makeAB = [&](const std::vector<IRResourceLocation>& locs) {
            uint32_t sz = 256;
            for (const auto& loc : locs) {
                if (loc.topLevelOffset + sizeof(IRDescriptorTableEntry) > sz) {
                    sz = loc.topLevelOffset + static_cast<uint32_t>(sizeof(IRDescriptorTableEntry));
                }
            }
            id<MTLBuffer> ab = [device newBufferWithLength:sz options:MTLResourceStorageModeShared];
            auto* bytes = static_cast<uint8_t*>([ab contents]);
            std::memset(bytes, 0, sz);
            for (const auto& loc : locs) {
                if (loc.resourceType == IRResourceTypeCBV && loc.slot == 0) {
                    auto* entry = reinterpret_cast<IRDescriptorTableEntry*>(bytes + loc.topLevelOffset);
                    IRBufferView view{cbBuf, 0, cbBytes, nil, 0, false};
                    IRDescriptorTableSetBufferView(entry, &view);
                }
            }
            return ab;
        };
        id<MTLBuffer> vsAB = makeAB(vsStage.resLocs);
        id<MTLBuffer> hsAB = hasTess ? makeAB(hsStage.resLocs) : nil;
        id<MTLBuffer> dsAB = hasTess ? makeAB(dsStage.resLocs) : nil;
        id<MTLBuffer> gsAB = hasGS ? makeAB(gsStage.resLocs) : nil;
        id<MTLBuffer> psAB = makeAB(psStage.resLocs);

        id<MTLCommandQueue> queue = [device newCommandQueue];

        auto getGfxParams = [&](int mode, uint32_t& rw, uint32_t& rh, uint32_t& vc) {
            if (isVertexMesh) {
                uint32_t g = 0;
                getMeshBenchGridAndDims(mode, width, height, g, rw, rh);
                vc = static_cast<uint32_t>(meshScenes[mode].indices.size());
            } else if (isPixel) {
                rw = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
                rh = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
                vc = 6;
            } else if (isGeometry) {
                rw = 128; rh = 128;
                vc = (mode == 0) ? 512 : (mode == 1 ? 4096 : 16384);
            } else if (isTess) {
                rw = 128; rh = 128;
                vc = (mode == 0) ? 64 * 3 : (mode == 1 ? 512 * 3 : 2048 * 3);
            } else {
                rw = 128; rh = 128;
                vc = (mode == 0) ? 3000 : (mode == 1 ? 24000 : 96000);
            }
        };

        auto drawOnce = [&](int mode, uint32_t rw, uint32_t rh, uint32_t vc) -> double {
            if (isVertexMesh) {
                std::memcpy([cbBuf contents], &meshScenes[mode].constants, sizeof(MeshBenchConstants));
            } else if (isPixel) {
                ShadertoyCB cb{static_cast<float>(rw), static_cast<float>(rh), 1.0f, 1.25f, 0.0f, 0.0f, 0.0f, 0.0f};
                std::memcpy([cbBuf contents], &cb, sizeof(cb));
            } else if (isGeometry) {
                GeomStressCB cb{0.05f, 0.5f, vc, 0.0f};
                std::memcpy([cbBuf contents], &cb, sizeof(cb));
            } else {
                VertexStressCB cb{};
                cb.mvp[0] = 1.0f; cb.mvp[5] = 1.0f; cb.mvp[10] = 1.0f; cb.mvp[15] = 1.0f;
                cb.time = 0.5f;
                cb.vertexCount = vc;
                std::memcpy([cbBuf contents], &cb, sizeof(cb));
            }

            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = rtTex;
            rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
            if (isVertexMesh) {
                rpd.depthAttachment.texture = depthTex;
                rpd.depthAttachment.loadAction = MTLLoadActionClear;
                rpd.depthAttachment.storeAction = MTLStoreActionDontCare;
                rpd.depthAttachment.clearDepth = 1.0;
            }

            auto s0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLRenderCommandEncoder> enc = [cmdBuf renderCommandEncoderWithDescriptor:rpd];
            [enc setRenderPipelineState:pso];
            [enc setViewport:(MTLViewport){0.0, 0.0, static_cast<double>(rw), static_cast<double>(rh), 0.0, 1.0}];
            [enc setScissorRect:(MTLScissorRect){0, 0, rw, rh}];
            if (!useEmulation) {
                [enc useResource:cbBuf usage:MTLResourceUsageRead stages:MTLRenderStageVertex | MTLRenderStageFragment];
                [enc setVertexBuffer:vsAB offset:0 atIndex:kIRArgumentBufferBindPoint];
                [enc setFragmentBuffer:psAB offset:0 atIndex:kIRArgumentBufferBindPoint];
                if (isVertexMesh) {
                    [enc setDepthStencilState:dsState];
                    [enc setFrontFacingWinding:MTLWindingClockwise];
                    [enc setCullMode:MTLCullModeBack];
                    [enc setVertexBuffer:meshVBs[mode] offset:0 atIndex:kIRVertexBufferBindPoint + 0];
                    [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle
                                    indexCount:vc
                                     indexType:MTLIndexTypeUInt32
                                   indexBuffer:meshIBs[mode]
                             indexBufferOffset:0];
                } else {
                    if (isPixel) {
                        [enc setVertexBuffer:vbBuf offset:0 atIndex:kIRVertexBufferBindPoint + 0];
                    }
                    [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:vc];
                }
            } else {
#if defined(IR_VERSION_MAJOR) && IR_VERSION_MAJOR >= 3
                [enc useResource:cbBuf usage:MTLResourceUsageRead stages:MTLRenderStageObject | MTLRenderStageMesh | MTLRenderStageFragment];
                [enc setObjectBuffer:emptyVBs offset:0 atIndex:0];
                [enc setObjectBuffer:vsAB offset:0 atIndex:kIRArgumentBufferBindPoint];
                if (hasTess) {
                    [enc setObjectBuffer:hsAB offset:0 atIndex:kIRArgumentBufferHullDomainBindPoint];
                    [enc setMeshBuffer:dsAB offset:0 atIndex:kIRArgumentBufferHullDomainBindPoint];
                }
                if (hasGS) {
                    [enc setMeshBuffer:gsAB offset:0 atIndex:kIRArgumentBufferBindPoint];
                }
                [enc setFragmentBuffer:psAB offset:0 atIndex:kIRArgumentBufferBindPoint];
                if (hasTess) {
                    IRRuntimeDrawPatchesTessellationEmulation(enc, IRRuntimePrimitiveType3ControlPointPatchlist, tessCfg, 1, vc, 0, 0);
                } else {
                    IRRuntimeDrawPrimitivesGeometryEmulation(enc, IRRuntimePrimitiveTypePoint, geomCfg, 1, vc, 0, 0);
                }
#endif
            }
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count();
        };

        uint32_t rw = 128, rh = 128, vc = 6;
        getGfxParams(1, rw, rh, vc);

        res.firstDispatchMs = drawOnce(1, rw, rh, vc);

        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            samples.push_back(drawOnce(1, rw, rh, vc));
        }

        std::vector<uint8_t> px(static_cast<size_t>(rw) * rh * 4, 0);
        [rtTex getBytes:px.data() bytesPerRow:rw * 4 fromRegion:MTLRegionMake2D(0, 0, rw, rh) mipmapLevel:0];
        double sum = 0.0;
        for (uint8_t v : px) sum += v;
        res.checksum = sum;

        getGfxParams(0, rw, rh, vc);
        res.shortJobMs = drawOnce(0, rw, rh, vc);
        getGfxParams(2, rw, rh, vc);
        res.longJobMs = drawOnce(2, rw, rh, vc);

        finalizeSamples(res, samples);
        res.ran = true;
        res.status = "OK";
    }
    return res;
#endif
}

MetalBenchResult runMetalMeshBench(
    const std::string& benchId,
    const std::vector<uint8_t>& msDxil,
    const std::vector<uint8_t>& psDxil,
    uint32_t iters) {
    MetalBenchResult res;
    res.backend = "Metal (IRConverter)";
    res.shaderFormat = "DXIL->MTL";
#if !defined(NACELLE_HAS_METAL_IRCONVERTER) || !defined(IR_VERSION_MAJOR) || (IR_VERSION_MAJOR < 3)
    (void)benchId;
    (void)msDxil;
    (void)psDxil;
    (void)iters;
    res.status = "SKIPPED (Metal IRConverter < 3.0)";
    return res;
#else
    (void)benchId;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            res.status = "UNSUPPORTED (no Metal device)";
            return res;
        }
        if (![device respondsToSelector:@selector(newRenderPipelineStateWithMeshDescriptor:options:reflection:error:)]) {
            res.status = "UNSUPPORTED (no Metal Mesh Shader)";
            return res;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            res.status = "FAILED (no command queue)";
            return res;
        }

        IRRootParameter1 cbvParam{};
        cbvParam.ParameterType = IRRootParameterTypeCBV;
        cbvParam.ShaderVisibility = IRShaderVisibilityAll;
        cbvParam.Descriptor.ShaderRegister = 0;
        cbvParam.Descriptor.RegisterSpace = 0;
        IRVersionedRootSignatureDescriptor rsDesc{};
        rsDesc.version = IRRootSignatureVersion_1_1;
        rsDesc.desc_1_1.NumParameters = 1;
        rsDesc.desc_1_1.pParameters = &cbvParam;
        rsDesc.desc_1_1.Flags = IRRootSignatureFlagNone;
        IRError* rsErr = nullptr;
        IRRootSignature* rootSig = IRRootSignatureCreateFromDescriptor(&rsDesc, &rsErr);
        if (rsErr) IRErrorDestroy(rsErr);

        auto t0 = std::chrono::steady_clock::now();
        std::string errMsg;
        CompiledIRStage msStage, psStage;
        bool ok = compileDxilToMetalStage(device, msDxil, rootSig, IRShaderStageMesh, msStage, errMsg) &&
                  compileDxilToMetalStage(device, psDxil, rootSig, IRShaderStageFragment, psStage, errMsg);

        if (!ok) {
            res.status = "FAILED (IRConverter: " + errMsg + ")";
            if (rootSig) IRRootSignatureDestroy(rootSig);
            return res;
        }

        id<MTLFunction> msFn = [msStage.lib newFunctionWithName:[NSString stringWithUTF8String:msStage.entryName.c_str()]];
        id<MTLFunction> psFn = [psStage.lib newFunctionWithName:[NSString stringWithUTF8String:psStage.entryName.c_str()]];
        MTLMeshRenderPipelineDescriptor* meshDesc = [[MTLMeshRenderPipelineDescriptor alloc] init];
        meshDesc.meshFunction = msFn;
        meshDesc.fragmentFunction = psFn;
        meshDesc.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm;
        meshDesc.maxTotalThreadsPerMeshThreadgroup = 32;

        NSError* mtlErr = nil;
        id<MTLRenderPipelineState> pso = [device newRenderPipelineStateWithMeshDescriptor:meshDesc
                                                                                   options:MTLPipelineOptionNone
                                                                                reflection:nil
                                                                                     error:&mtlErr];
        auto t1 = std::chrono::steady_clock::now();
        res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!pso) {
            std::string desc = mtlErr ? [[mtlErr localizedDescription] UTF8String] : "unknown";
            res.status = "FAILED (MTLMeshPSO: " + desc + ")";
            if (rootSig) IRRootSignatureDestroy(rootSig);
            return res;
        }

        const uint32_t rw = 128, rh = 128;
        MTLTextureDescriptor* td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                      width:rw
                                                                                     height:rh
                                                                                  mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeManaged;
        id<MTLTexture> rtTex = [device newTextureWithDescriptor:td];
        id<MTLBuffer> cbBuf = [device newBufferWithLength:256 options:MTLResourceStorageModeShared];
        id<MTLBuffer> msAB = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        id<MTLBuffer> psAB = [device newBufferWithLength:64 options:MTLResourceStorageModeShared];
        uint64_t cbAddr = [cbBuf gpuAddress];
        std::memcpy([msAB contents], &cbAddr, sizeof(uint64_t));
        std::memcpy([psAB contents], &cbAddr, sizeof(uint64_t));

        auto drawOnce = [&](uint32_t groups) -> double {
            struct { float scale; float time; uint32_t groupCount; float pad; } cb{0.05f, 0.25f, groups, 0.0f};
            std::memcpy([cbBuf contents], &cb, sizeof(cb));

            auto s0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            MTLRenderPassDescriptor* rpd = [MTLRenderPassDescriptor renderPassDescriptor];
            rpd.colorAttachments[0].texture = rtTex;
            rpd.colorAttachments[0].loadAction = MTLLoadActionClear;
            rpd.colorAttachments[0].storeAction = MTLStoreActionStore;
            rpd.colorAttachments[0].clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);

            id<MTLRenderCommandEncoder> enc = [cmdBuf renderCommandEncoderWithDescriptor:rpd];
            [enc setRenderPipelineState:pso];
            MTLViewport vp{0.0, 0.0, static_cast<double>(rw), static_cast<double>(rh), 0.0, 1.0};
            [enc setViewport:vp];
            [enc useResource:cbBuf usage:MTLResourceUsageRead stages:MTLRenderStageMesh | MTLRenderStageFragment];
            [enc setMeshBuffer:msAB offset:0 atIndex:kIRArgumentBufferBindPoint];
            [enc setFragmentBuffer:psAB offset:0 atIndex:kIRArgumentBufferBindPoint];
            [enc drawMeshThreadgroups:MTLSizeMake(groups, 1, 1)
          threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
            threadsPerMeshThreadgroup:MTLSizeMake(32, 1, 1)];
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count();
        };

        res.firstDispatchMs = drawOnce(128);
        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            samples.push_back(drawOnce(128));
        }

        std::vector<uint8_t> px(static_cast<size_t>(rw) * rh * 4, 0);
        [rtTex getBytes:px.data() bytesPerRow:rw * 4 fromRegion:MTLRegionMake2D(0, 0, rw, rh) mipmapLevel:0];
        double sum = 0.0;
        for (uint8_t b : px) sum += b;
        res.checksum = sum;

        res.shortJobMs = drawOnce(16);
        res.longJobMs = drawOnce(512);

        double total = 0.0;
        for (double v : samples) total += v;
        res.meanMs = samples.empty() ? 0.0 : (total / static_cast<double>(samples.size()));
        std::sort(samples.begin(), samples.end());
        res.medianMs = samples.empty() ? 0.0 : samples[samples.size() / 2];
        res.minMs = samples.empty() ? 0.0 : samples.front();
        res.ran = true;
        res.status = "OK";

        IRRootSignatureDestroy(rootSig);
    }
    return res;
#endif
}

MetalBenchResult runMetalRaytracingBench(
    const std::string& benchId,
    bool isInline,
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t iters) {
    MetalBenchResult res;
    res.backend = "Metal (IRConverter)";
    res.shaderFormat = "DXIL->MTL";

#if !defined(NACELLE_HAS_METAL_IRCONVERTER) || !defined(IR_VERSION_MAJOR) || (IR_VERSION_MAJOR < 3)
    (void)benchId;
    (void)isInline;
    (void)dxil;
    (void)width;
    (void)height;
    (void)iters;
    res.status = "SKIPPED (Metal IRConverter < 3.0)";
    return res;
#else
    (void)benchId;
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            res.status = "SKIPPED (no Metal device)";
            return res;
        }
        if (![device supportsRaytracing]) {
            res.status = "SKIPPED (Metal raytracing unsupported)";
            return res;
        }
        res.backend = std::string("Metal (IRConverter): ") + [[device name] UTF8String];

        id<MTLCommandQueue> queue = [device newCommandQueue];

        // Build triangle BLAS + TLAS
        const float triVerts[9] = {0.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f};
        id<MTLBuffer> vb = [device newBufferWithBytes:triVerts length:sizeof(triVerts) options:MTLResourceStorageModeShared];

        MTLAccelerationStructureTriangleGeometryDescriptor* triGeo = [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        triGeo.vertexBuffer = vb;
        triGeo.vertexStride = 12;
        triGeo.triangleCount = 1;
        triGeo.opaque = YES;

        MTLPrimitiveAccelerationStructureDescriptor* blasDesc = [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        blasDesc.geometryDescriptors = @[triGeo];
        MTLAccelerationStructureSizes blasSizes = [device accelerationStructureSizesWithDescriptor:blasDesc];
        id<MTLAccelerationStructure> blas = [device newAccelerationStructureWithSize:blasSizes.accelerationStructureSize];

        id<MTLBuffer> scratch = [device newBufferWithLength:std::max<NSUInteger>(4096, blasSizes.buildScratchBufferSize) options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> asCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> asEnc = [asCmd accelerationStructureCommandEncoder];
        [asEnc buildAccelerationStructure:blas descriptor:blasDesc scratchBuffer:scratch scratchBufferOffset:0];
        [asEnc endEncoding];
        [asCmd commit];
        [asCmd waitUntilCompleted];

        id<MTLBuffer> instBuf = [device newBufferWithLength:sizeof(MTLAccelerationStructureInstanceDescriptor) options:MTLResourceStorageModeShared];
        auto* instDesc = static_cast<MTLAccelerationStructureInstanceDescriptor*>([instBuf contents]);
        std::memset(instDesc, 0, sizeof(MTLAccelerationStructureInstanceDescriptor));
        instDesc->transformationMatrix.columns[0] = MTLPackedFloat3Make(1.0f, 0.0f, 0.0f);
        instDesc->transformationMatrix.columns[1] = MTLPackedFloat3Make(0.0f, 1.0f, 0.0f);
        instDesc->transformationMatrix.columns[2] = MTLPackedFloat3Make(0.0f, 0.0f, 1.0f);
        instDesc->mask = 0xFF;
        instDesc->options = MTLAccelerationStructureInstanceOptionOpaque;
        instDesc->accelerationStructureIndex = 0;

        MTLInstanceAccelerationStructureDescriptor* tlasDesc = [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlasDesc.instancedAccelerationStructures = @[blas];
        tlasDesc.instanceCount = 1;
        tlasDesc.instanceDescriptorBuffer = instBuf;
        MTLAccelerationStructureSizes tlasSizes = [device accelerationStructureSizesWithDescriptor:tlasDesc];
        id<MTLAccelerationStructure> tlas = [device newAccelerationStructureWithSize:tlasSizes.accelerationStructureSize];

        id<MTLBuffer> tlasScratch = [device newBufferWithLength:std::max<NSUInteger>(4096, tlasSizes.buildScratchBufferSize) options:MTLResourceStorageModePrivate];
        id<MTLCommandBuffer> tlasCmd = [queue commandBuffer];
        id<MTLAccelerationStructureCommandEncoder> tlasEnc = [tlasCmd accelerationStructureCommandEncoder];
        [tlasEnc buildAccelerationStructure:tlas descriptor:tlasDesc scratchBuffer:tlasScratch scratchBufferOffset:0];
        [tlasEnc endEncoding];
        [tlasCmd commit];
        [tlasCmd waitUntilCompleted];

        id<MTLBuffer> asHeaderBuf = [device newBufferWithLength:sizeof(IRRaytracingAccelerationStructureGPUHeader) options:MTLResourceStorageModeShared];
        id<MTLBuffer> instContribBuf = [device newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];
        uint32_t zeroContrib = 0;
        IRRaytracingSetAccelerationStructure(
            static_cast<uint8_t*>([asHeaderBuf contents]),
            [tlas gpuResourceID],
            static_cast<uint8_t*>([instContribBuf contents]),
            [instContribBuf gpuAddress],
            &zeroContrib,
            1);

        auto t0 = std::chrono::steady_clock::now();

        IRRootParameter1 params[2]{};
        params[0].ParameterType = IRRootParameterTypeSRV;
        params[0].Descriptor.ShaderRegister = 0;
        params[0].Descriptor.RegisterSpace = 0;
        params[0].ShaderVisibility = IRShaderVisibilityAll;
        params[1].ParameterType = IRRootParameterTypeUAV;
        params[1].Descriptor.ShaderRegister = 0;
        params[1].Descriptor.RegisterSpace = 0;
        params[1].ShaderVisibility = IRShaderVisibilityAll;

        IRVersionedRootSignatureDescriptor rsDesc{};
        rsDesc.version = IRRootSignatureVersion_1_1;
        rsDesc.desc_1_1.NumParameters = 2;
        rsDesc.desc_1_1.pParameters = params;

        IRError* irErr = nullptr;
        IRRootSignature* rootSig = IRRootSignatureCreateFromDescriptor(&rsDesc, &irErr);
        if (!rootSig) {
            if (irErr) IRErrorDestroy(irErr);
            res.status = "FAILED (IRRootSignatureCreateFromDescriptor RT)";
            return res;
        }

        id<MTLComputePipelineState> pso = nil;
        id<MTLVisibleFunctionTable> vft = nil;
        id<MTLIntersectionFunctionTable> ift = nil;
        id<MTLBuffer> sbtBuf = nil;
        std::vector<IRResourceLocation> resLocs;
        std::string irErrStr;

        if (isInline) {
            CompiledIRStage csStage;
            if (!compileDxilToMetalStage(device, dxil, rootSig, IRShaderStageCompute, csStage, irErrStr)) {
                IRRootSignatureDestroy(rootSig);
                res.status = "FAILED (" + irErrStr + " RayQuery)";
                return res;
            }
            resLocs = csStage.resLocs;
            NSError* mtlErr = nil;
            pso = [device newComputePipelineStateWithFunction:csStage.fn error:&mtlErr];
        } else {
            IRRayTracingPipelineConfiguration* rtCfg = IRRayTracingPipelineConfigurationCreate();
            IRRayTracingPipelineConfigurationSetMaxAttributeSizeInBytes(rtCfg, 8);
            IRRayTracingPipelineConfigurationSetMaxRecursiveDepth(rtCfg, 1);
            IRRayTracingPipelineConfigurationSetRayGenerationCompilationMode(rtCfg, IRRayGenerationCompilationVisibleFunction);

            CompiledIRStage rgenStage, missStage, chsStage;
            bool ok = compileDxilToMetalStage(device, dxil, rootSig, IRShaderStageRayGeneration, rgenStage, irErrStr, false, "RayGen", rtCfg) &&
                      compileDxilToMetalStage(device, dxil, rootSig, IRShaderStageMiss, missStage, irErrStr, false, "MissMain", rtCfg) &&
                      compileDxilToMetalStage(device, dxil, rootSig, IRShaderStageClosestHit, chsStage, irErrStr, false, "ClosestHitMain", rtCfg);

            id<MTLLibrary> dispatchLib = nil;
            id<MTLFunction> dispatchFn = nil;
            if (ok) {
                IRCompiler* comp = IRCompilerCreate();
                IRCompilerSetGlobalRootSignature(comp, rootSig);
                IRCompilerSetRayTracingPipelineConfiguration(comp, rtCfg);
                IRMetalLibBinary* dispBin = IRMetalLibBinaryCreate();
                if (IRMetalLibSynthesizeIndirectRayDispatchFunction(comp, dispBin)) {
                    dispatch_data_t dispData = IRMetalLibGetBytecodeData(dispBin);
                    NSError* libErr = nil;
                    dispatchLib = [device newLibraryWithData:dispData error:&libErr];
                    dispatchFn = dispatchLib ? [dispatchLib newFunctionWithName:[NSString stringWithUTF8String:kIRRayDispatchIndirectionKernelName]] : nil;
                }
                IRMetalLibBinaryDestroy(dispBin);
                IRCompilerDestroy(comp);
            }
            IRRayTracingPipelineConfigurationDestroy(rtCfg);

            if (!ok || !dispatchFn) {
                IRRootSignatureDestroy(rootSig);
                res.status = "FAILED (" + (irErrStr.empty() ? std::string("RT stage compile") : irErrStr) + ")";
                return res;
            }

            MTLComputePipelineDescriptor* cpDesc = [[MTLComputePipelineDescriptor alloc] init];
            cpDesc.computeFunction = dispatchFn;
            MTLLinkedFunctions* lf = [MTLLinkedFunctions linkedFunctions];
            lf.functions = @[rgenStage.fn, missStage.fn, chsStage.fn];
            cpDesc.linkedFunctions = lf;

            NSError* mtlErr = nil;
            pso = [device newComputePipelineStateWithDescriptor:cpDesc options:MTLPipelineOptionNone reflection:nil error:&mtlErr];
            if (pso) {
                MTLVisibleFunctionTableDescriptor* vftDesc = [MTLVisibleFunctionTableDescriptor visibleFunctionTableDescriptor];
                vftDesc.functionCount = 3;
                vft = [pso newVisibleFunctionTableWithDescriptor:vftDesc];
                [vft setFunction:[pso functionHandleWithFunction:rgenStage.fn] atIndex:0];
                [vft setFunction:[pso functionHandleWithFunction:missStage.fn] atIndex:1];
                [vft setFunction:[pso functionHandleWithFunction:chsStage.fn] atIndex:2];

                MTLIntersectionFunctionTableDescriptor* iftDesc = [MTLIntersectionFunctionTableDescriptor intersectionFunctionTableDescriptor];
                iftDesc.functionCount = 1;
                ift = [pso newIntersectionFunctionTableWithDescriptor:iftDesc];

                sbtBuf = [device newBufferWithLength:3 * sizeof(IRShaderIdentifier) options:MTLResourceStorageModeShared];
                auto* ids = static_cast<IRShaderIdentifier*>([sbtBuf contents]);
                IRShaderIdentifierInit(&ids[0], 0);
                IRShaderIdentifierInit(&ids[1], 1);
                IRShaderIdentifierInit(&ids[2], 2);
            }
            resLocs = rgenStage.resLocs;
        }

        auto t1 = std::chrono::steady_clock::now();
        res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        IRRootSignatureDestroy(rootSig);

        if (!pso) {
            res.status = "FAILED (Metal RT PSO creation)";
            return res;
        }

        id<MTLBuffer> outBuf = [device newBufferWithLength:4 * 1024 * 1024 options:MTLResourceStorageModeShared];
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
            if (loc.resourceType == IRResourceTypeSRV && loc.slot == 0) {
                IRDescriptorTableSetAccelerationStructure(entry, [asHeaderBuf gpuAddress]);
            } else if (loc.resourceType == IRResourceTypeUAV && loc.slot == 0) {
                IRDescriptorTableSetBuffer(entry, [outBuf gpuAddress], 0);
            }
        }

        auto dispatchOnce = [&](uint32_t w, uint32_t h) -> double {
            auto s0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc useResource:tlas usage:MTLResourceUsageRead];
            [enc useResource:blas usage:MTLResourceUsageRead];
            [enc useResource:asHeaderBuf usage:MTLResourceUsageRead];
            [enc useResource:instContribBuf usage:MTLResourceUsageRead];
            [enc useResource:outBuf usage:MTLResourceUsageWrite];
            [enc setBuffer:abBuf offset:0 atIndex:kIRArgumentBufferBindPoint];

            if (isInline) {
                [enc dispatchThreadgroups:MTLSizeMake((w + 3) / 4, h, 1) threadsPerThreadgroup:MTLSizeMake(4, 1, 1)];
            } else {
                [enc useResource:sbtBuf usage:MTLResourceUsageRead];
                [enc useResource:vft usage:MTLResourceUsageRead];
                [enc useResource:ift usage:MTLResourceUsageRead];
                IRDispatchRaysArgument args{};
                uint64_t sbtAddr = [sbtBuf gpuAddress];
                args.DispatchRaysDesc.RayGenerationShaderRecord = {sbtAddr + 0 * sizeof(IRShaderIdentifier), sizeof(IRShaderIdentifier)};
                args.DispatchRaysDesc.MissShaderTable = {sbtAddr + 1 * sizeof(IRShaderIdentifier), sizeof(IRShaderIdentifier), sizeof(IRShaderIdentifier)};
                args.DispatchRaysDesc.HitGroupTable = {sbtAddr + 2 * sizeof(IRShaderIdentifier), sizeof(IRShaderIdentifier), sizeof(IRShaderIdentifier)};
                args.DispatchRaysDesc.Width = w;
                args.DispatchRaysDesc.Height = h;
                args.DispatchRaysDesc.Depth = 1;
                args.GRS = [abBuf gpuAddress];
                args.ResDescHeap = 0;
                args.SmpDescHeap = 0;
                args.VisibleFunctionTable = [vft gpuResourceID];
                args.IntersectionFunctionTable = [ift gpuResourceID];
                [enc setBytes:&args length:sizeof(args) atIndex:kIRRayDispatchArgumentsBindPoint];
                [enc dispatchThreadgroups:MTLSizeMake((w + 7) / 8, (h + 7) / 8, 1) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
            }
            [enc endEncoding];
            [cmdBuf commit];
            [cmdBuf waitUntilCompleted];
            auto s1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(s1 - s0).count();
        };

        res.firstDispatchMs = dispatchOnce(width, height);
        std::vector<double> samples;
        samples.reserve(iters);
        for (uint32_t i = 0; i < iters; ++i) {
            samples.push_back(dispatchOnce(width, height));
        }

        const auto* up = static_cast<const uint32_t*>([outBuf contents]);
        double sum = 0.0;
        for (size_t i = 0; i < 1024; ++i) sum += static_cast<double>(up[i] & 0xFFFFu);
        res.checksum = sum;

        res.shortJobMs = dispatchOnce(64, 64);
        res.longJobMs = dispatchOnce(width * 2, height * 2);

        finalizeSamples(res, samples);
        res.ran = true;
        res.status = "OK";
    }
    return res;
#endif
}

} // namespace nacelle_bench
