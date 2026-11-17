#pragma once

#include "bench_memory.h"
#include "bench_types.h"
#include "bench_gdeflate.h"
#include "bench_bcn_textures.h"
#include "bench_rt_scene.h"

#include <chrono>
#include <cstring>
#include <vector>

#include <nacelle/nacelle.h>
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"

namespace nacelle_bench {

inline ID3D12Resource* createNacelleBuffer(ID3D12Device* dev, size_t bytes, bool allowUAV = true) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = allowUAV ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&r));
    return r;
}

inline ID3D12Resource* createNacelleRTTexture(ID3D12Device* dev, uint32_t w, uint32_t h) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_CUSTOM;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rd.Width = w;
    rd.Height = h;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ID3D12Resource* r = nullptr;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&r));
    return r;
}

inline BenchRunResult runNacelleComputeBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& dxil,
    uint32_t simdWidth,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Nacelle (W=" + std::to_string(simdWidth) + ")";
    res.shaderFormat = "DXIL";
    res.simdWidth = simdWidth;

    MemorySnapshot memBefore = queryProcessMemory();
    nacelle::setSimdWidthOverride(simdWidth);
    nacelle::clearInMemoryPSOCacheForTesting();

    ID3D12Device* dev = nullptr;
    if (FAILED(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))) || !dev) {
        res.status = "FAILED (CreateDevice)";
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&queue));
    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    ID3D12GraphicsCommandList* cmd = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc, nullptr, IID_PPV_ARGS(&cmd));
    ID3D12Fence* fence = nullptr;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Root signature: b0, t0, t1, u0, u1, u2, u3, u4 (covers all compute benchmarks)
    D3D12_ROOT_PARAMETER params[8]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[1].Descriptor.ShaderRegister = 0;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].Descriptor.ShaderRegister = 1;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 0;
    params[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[4].Descriptor.ShaderRegister = 1;
    params[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[5].Descriptor.ShaderRegister = 2;
    params[6].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[6].Descriptor.ShaderRegister = 3;
    params[7].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[7].Descriptor.ShaderRegister = 4;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 8;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    rsBlob->Release();

    auto t0 = std::chrono::steady_clock::now();
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig;
    psoDesc.CS = {dxil.data(), dxil.size()};
    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = dev->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (CreateComputePipelineState)";
        rootSig->Release();
        fence->Release();
        cmd->Release();
        alloc->Release();
        queue->Release();
        dev->Release();
        return res;
    }

    // Allocate buffers large enough for Long job
    uint32_t longW = width * 2;
    uint32_t longH = height * 2;
    size_t maxOutBytes = 16 * 1024 * 1024; // 16 MB covers 512x512 float4 or 512x512 matmul
    if (spec.id == "bench_raymarch") {
        maxOutBytes = std::max<size_t>(maxOutBytes, static_cast<size_t>(longW) * longH * 16);
    }

    ID3D12Resource* cbBuf = createNacelleBuffer(dev, 256);
    ID3D12Resource* srv0 = createNacelleBuffer(dev, 8 * 1024 * 1024);
    ID3D12Resource* srv1 = createNacelleBuffer(dev, 4 * 1024 * 1024);
    ID3D12Resource* uav0 = createNacelleBuffer(dev, maxOutBytes);
    ID3D12Resource* uav1 = createNacelleBuffer(dev, maxOutBytes);
    ID3D12Resource* uav2 = createNacelleBuffer(dev, 4 * 1024 * 1024);
    ID3D12Resource* uav3 = createNacelleBuffer(dev, 4 * 1024 * 1024);
    ID3D12Resource* uav4 = createNacelleBuffer(dev, 64 * 1024);

    GDeflateStreamData gdShort, gdStd, gdLong;
    uint32_t gdActiveMode = UINT32_MAX;
    if (spec.id == "gdeflate") {
        gdShort = buildGDeflateStream(4u);
        gdStd = buildGDeflateStream(16u);
        gdLong = buildGDeflateStream(64u);
    }

    // Initialize inputs
    if (spec.id == "bench_raymarch") {
        float* sphPtr = nullptr;
        srv0->Map(0, nullptr, reinterpret_cast<void**>(&sphPtr));
        float spheres[16] = {
            0.0f, 0.2f, 0.5f, 0.8f,
            -1.3f, 0.0f, 1.0f, 0.6f,
            1.3f, 0.1f, 0.9f, 0.65f,
            0.0f, 1.1f, 1.5f, 0.5f,
        };
        std::memcpy(sphPtr, spheres, sizeof(spheres));
        srv0->Unmap(0, nullptr);
    } else if (spec.id == "matmul" || spec.id == "tiled_matmul" || spec.id == "tiled_matmul_large" || spec.id == "linalg_matmul" || spec.id == "coalesced_uniform_cs") {
        float* a = nullptr;
        float* b = nullptr;
        srv0->Map(0, nullptr, reinterpret_cast<void**>(&a));
        srv1->Map(0, nullptr, reinterpret_cast<void**>(&b));
        const size_t initCount = (spec.id == "tiled_matmul_large" || spec.id == "coalesced_uniform_cs") ? (1024u * 1024u) : (256u * 256u);
        for (size_t i = 0; i < initCount; ++i) {
            a[i] = static_cast<float>((i % 17) + 1) * 0.01f;
            b[i] = static_cast<float>((i % 13) + 1) * 0.01f;
        }
        srv0->Unmap(0, nullptr);
        srv1->Unmap(0, nullptr);
    } else if (spec.id == "prefix_sum_groupshared") {
        uint32_t* p = nullptr;
        srv0->Map(0, nullptr, reinterpret_cast<void**>(&p));
        for (size_t i = 0; i < 65536 * 4; ++i) p[i] = static_cast<uint32_t>((i & 63u) + 1u);
        srv0->Unmap(0, nullptr);
    }

    uint64_t fenceVal = 1;
    auto runDispatch = [&](uint32_t gx, uint32_t gy, uint32_t cbW, uint32_t cbH, uint32_t cbSteps) -> double {
        void* cbPtr = nullptr;
        cbBuf->Map(0, nullptr, &cbPtr);
        if (spec.id == "bench_raymarch") {
            RaymarchCB cb{cbW, cbH, cbSteps, 0.0f};
            std::memcpy(cbPtr, &cb, sizeof(cb));
        } else if (spec.id == "matmul" || spec.id == "tiled_matmul" || spec.id == "tiled_matmul_large" || spec.id == "linalg_matmul") {
            MatmulCB cb{cbW, cbH, cbSteps, 0};
            std::memcpy(cbPtr, &cb, sizeof(cb));
        } else if (spec.id == "ethash_mine") {
            EthashCB ecb{};
            for (uint32_t k = 0; k < 8; ++k) ecb.header[k] = 0x12345678u + k;
            ecb.startNonce[0] = 1u;
            ecb.startNonce[1] = 0u;
            ecb.numDatasetElements = 16384u;
            std::memcpy(cbPtr, &ecb, sizeof(ecb));
        } else if (spec.id == "coalesced_uniform_cs") {
            CoalescedUniformCB ccb{cbW, cbSteps, 0.75f, 0.25f, {}};
            for (uint32_t k = 0; k < 16; ++k) ccb.weights[k] = 0.25f + 0.03f * static_cast<float>(k);
            std::memcpy(cbPtr, &ccb, sizeof(ccb));
        }
        cbBuf->Unmap(0, nullptr);

        if (spec.id == "gdeflate") {
            const GDeflateStreamData& gd = (cbW <= 4u) ? gdShort : (cbW <= 16u ? gdStd : gdLong);
            if (gdActiveMode != cbW) {
                void* inPtr = nullptr;
                srv0->Map(0, nullptr, &inPtr);
                std::memcpy(inPtr, gd.inputBytes.data(), gd.inputBytes.size());
                srv0->Unmap(0, nullptr);
                gdActiveMode = cbW;
            }
            void* ctrlPtr = nullptr;
            uav0->Map(0, nullptr, &ctrlPtr);
            std::memcpy(ctrlPtr, gd.controlWords.data(), gd.controlWords.size() * sizeof(uint32_t));
            uav0->Unmap(0, nullptr);
            void* scrPtr = nullptr;
            uav2->Map(0, nullptr, &scrPtr);
            std::memset(scrPtr, 0, 16);
            uav2->Unmap(0, nullptr);
        }

        auto s0 = std::chrono::steady_clock::now();
        cmd->SetComputeRootSignature(rootSig);
        cmd->SetPipelineState(pso);
        cmd->SetComputeRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(1, srv0->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(2, srv1->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(3, uav0->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(4, uav1->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(5, uav2->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(6, uav3->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(7, uav4->GetGPUVirtualAddress());
        cmd->Dispatch(gx, gy, 1);
        cmd->Close();
        ID3D12CommandList* lists[] = {cmd};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        ++fenceVal;
        alloc->Reset();
        cmd->Reset(alloc, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    auto getDims = [&](int mode, uint32_t& gx, uint32_t& gy, uint32_t& cbW, uint32_t& cbH, uint32_t& cbSteps) {
        // mode: 0 = short, 1 = standard, 2 = long
        if (spec.id == "bench_raymarch") {
            cbW = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
            cbH = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
            cbSteps = (mode == 0) ? 16 : (mode == 1 ? maxSteps : maxSteps * 2);
            gx = (cbW + 7) / 8;
            gy = (cbH + 7) / 8;
        } else if (spec.id == "matmul") {
            uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
            cbW = n; cbH = n; cbSteps = n;
            gx = (n + 7) / 8;
            gy = (n + 7) / 8;
        } else if (spec.id == "tiled_matmul") {
            uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
            cbW = n; cbH = n; cbSteps = n;
            gx = (n + 15) / 16;
            gy = (n + 15) / 16;
        } else if (spec.id == "tiled_matmul_large") {
            uint32_t n = (mode == 0) ? 256 : (mode == 1 ? 512 : 1024);
            cbW = n; cbH = n; cbSteps = n;
            gx = (n + 15) / 16;
            gy = (n + 15) / 16;
        } else if (spec.id == "linalg_matmul") {
            uint32_t n = (mode == 0) ? 64 : (mode == 1 ? 128 : 256);
            cbW = n; cbH = n; cbSteps = n;
            gx = (n + 31) / 32;
            gy = (n + 31) / 32;
        } else if (spec.id == "gdeflate") {
            uint32_t tiles = (mode == 0) ? 4u : (mode == 1 ? 16u : 64u);
            gx = tiles;
            gy = 1;
            cbW = tiles; cbH = 1; cbSteps = 1;
        } else if (spec.id == "ethash_mine") {
            gx = (mode == 0) ? 16u : (mode == 1 ? 64u : 256u);
            gy = 1;
            cbW = gx * 16u; cbH = 1; cbSteps = 1;
        } else if (spec.id == "coalesced_uniform_cs") {
            gx = (mode == 0) ? 256u : (mode == 1 ? 4096u : 16384u);
            gy = 1;
            cbW = gx * 64u;
            cbH = 1;
            cbSteps = (mode == 0) ? 16u : (mode == 1 ? 32u : 64u);
        } else {
            // 64 threads per group
            gx = (mode == 0) ? 64 : (mode == 1 ? 512 : 4096);
            gy = 1;
            cbW = gx * 64; cbH = 1; cbSteps = 1;
        }
    };

    uint32_t gx = 1, gy = 1, cbW = 1, cbH = 1, cbSteps = 1;
    getDims(1, gx, gy, cbW, cbH, cbSteps);

    // 1. Cold first dispatch (before Tier-2 wait)
    res.firstDispatchMs = runDispatch(gx, gy, cbW, cbH, cbSteps);

    // 2. Wait for Tier-2 async JIT and run warmup
    nacelle::waitForAllAsyncJITTasks();
    runDispatch(gx, gy, cbW, cbH, cbSteps);

    // 3. Hot steady-state iterations
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runDispatch(gx, gy, cbW, cbH, cbSteps));
    }

    // Checksum on standard run
    double sum = 0.0;
    if (spec.id == "bench_raymarch") {
        float* outPtr = nullptr;
        uav0->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        size_t count = static_cast<size_t>(width) * height * 4;
        for (size_t i = 0; i < count; ++i) sum += outPtr[i];
        uav0->Unmap(0, nullptr);
    } else if (spec.id == "matmul" || spec.id == "tiled_matmul" || spec.id == "tiled_matmul_large" || spec.id == "linalg_matmul") {
        float* outPtr = nullptr;
        uav0->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        const size_t count = (spec.id == "tiled_matmul_large") ? (512u * 512u) : (128u * 128u);
        for (size_t i = 0; i < count; ++i) sum += outPtr[i];
        uav0->Unmap(0, nullptr);
    } else if (spec.id == "gdeflate") {
        uint32_t* outPtr = nullptr;
        uav1->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        for (size_t i = 0; i < 4096; ++i) sum += static_cast<double>(outPtr[i] & 0xFFFFu);
        uav1->Unmap(0, nullptr);
    } else if (spec.id == "ethash_mine") {
        uint32_t* outPtr = nullptr;
        uav4->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        for (size_t i = 0; i < 6; ++i) sum += static_cast<double>(outPtr[i] & 0xFFFFu);
        uav4->Unmap(0, nullptr);
    } else if (spec.id == "coalesced_uniform_cs") {
        float* outPtr = nullptr;
        uav0->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        for (size_t i = 0; i < 4096; ++i) sum += outPtr[i];
        uav0->Unmap(0, nullptr);
    } else {
        uint32_t* outPtr = nullptr;
        uav0->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        for (size_t i = 0; i < 1024; ++i) sum += static_cast<double>(outPtr[i] & 0xFFFFu);
        uav0->Unmap(0, nullptr);
    }
    res.checksum = sum;

    // 4. Short & Long job timings
    getDims(0, gx, gy, cbW, cbH, cbSteps);
    res.shortJobMs = runDispatch(gx, gy, cbW, cbH, cbSteps);
    getDims(2, gx, gy, cbW, cbH, cbSteps);
    res.longJobMs = runDispatch(gx, gy, cbW, cbH, cbSteps);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    uav4->Release();
    uav3->Release();
    uav2->Release();
    uav1->Release();
    uav0->Release();
    srv1->Release();
    srv0->Release();
    cbBuf->Release();
    pso->Release();
    rootSig->Release();
    fence->Release();
    cmd->Release();
    alloc->Release();
    queue->Release();
    dev->Release();
    return res;
}

inline BenchRunResult runNacelleGraphicsBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& vsDxil,
    const std::vector<uint8_t>& psDxil,
    const std::vector<uint8_t>& hsDxil,
    const std::vector<uint8_t>& dsDxil,
    const std::vector<uint8_t>& gsDxil,
    uint32_t simdWidth,
    uint32_t width,
    uint32_t height,
    uint32_t iters) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Nacelle (W=" + std::to_string(simdWidth) + ")";
    res.shaderFormat = "DXIL";
    res.simdWidth = simdWidth;

    MemorySnapshot memBefore = queryProcessMemory();
    nacelle::setSimdWidthOverride(simdWidth);
    nacelle::clearInMemoryPSOCacheForTesting();

    ID3D12Device* dev = nullptr;
    if (FAILED(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))) || !dev) {
        res.status = "FAILED (CreateDevice)";
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&queue));
    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ID3D12GraphicsCommandList* cmd = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cmd));
    ID3D12Fence* fence = nullptr;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    const bool isStateHeavyPS = (spec.id == "state_heavy_ps");
    const bool useSrvAndSampler = isStateHeavyPS || spec.hasTexture;

    D3D12_DESCRIPTOR_RANGE srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = isStateHeavyPS ? 2u : 1u;
    srvRange.BaseShaderRegister = 0;
    srvRange.RegisterSpace = 0;
    srvRange.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor.ShaderRegister = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &srvRange;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC staticSampler{};
    staticSampler.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSampler.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSampler.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSampler.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSampler.MaxAnisotropy = 1;
    staticSampler.ComparisonFunc = D3D12_COMPARISON_FUNC_NEVER;
    staticSampler.MaxLOD = D3D12_FLOAT32_MAX;
    staticSampler.ShaderRegister = 0;
    staticSampler.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = useSrvAndSampler ? 2u : 1u;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = useSrvAndSampler ? 1u : 0u;
    rsDesc.pStaticSamplers = useSrvAndSampler ? &staticSampler : nullptr;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob* rsBlob = nullptr;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    rsBlob->Release();

    D3D12_INPUT_ELEMENT_DESC posElem{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
    D3D12_INPUT_ELEMENT_DESC stateHeavyElems[2] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_INPUT_ELEMENT_DESC meshElems[5] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R16G16B16A16_SNORM, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R16G16_UNORM, 0, 20, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R8G8B8A8_UNORM, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"BLENDINDICES", 0, DXGI_FORMAT_R8G8B8A8_UINT, 0, 28, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig;
    psoDesc.VS = {vsDxil.data(), vsDxil.size()};
    psoDesc.PS = {psDxil.data(), psDxil.size()};
    if (!hsDxil.empty()) psoDesc.HS = {hsDxil.data(), hsDxil.size()};
    if (!dsDxil.empty()) psoDesc.DS = {dsDxil.data(), dsDxil.size()};
    if (!gsDxil.empty()) psoDesc.GS = {gsDxil.data(), gsDxil.size()};

    bool isVertexMesh = (spec.id == "bench_vertex_mesh");
    bool usesQuadVB = (spec.kind == PipelineKind::GraphicsVSPS && spec.category == BenchCategory::Pixel);
    if (isVertexMesh) {
        psoDesc.InputLayout = {meshElems, 5};
    } else if (isStateHeavyPS) {
        psoDesc.InputLayout = {stateHeavyElems, 2};
    } else if (usesQuadVB) {
        psoDesc.InputLayout = {&posElem, 1};
    }
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = isVertexMesh ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = FALSE;
    if (isVertexMesh) {
        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    } else if (isStateHeavyPS) {
        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS_EQUAL;
        psoDesc.DepthStencilState.StencilEnable = TRUE;
        psoDesc.DepthStencilState.StencilReadMask = 0xFF;
        psoDesc.DepthStencilState.StencilWriteMask = 0xFF;
        D3D12_DEPTH_STENCILOP_DESC stOp{
            D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_KEEP, D3D12_STENCIL_OP_INCR_SAT, D3D12_COMPARISON_FUNC_ALWAYS};
        psoDesc.DepthStencilState.FrontFace = stOp;
        psoDesc.DepthStencilState.BackFace = stOp;
        psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
        psoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
        psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_INV_SRC_ALPHA;
        psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;
    }
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
    if (spec.kind == PipelineKind::GraphicsTessGS) {
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    } else if (spec.kind == PipelineKind::GraphicsVSGSPS) {
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT;
    } else {
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    }
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = isStateHeavyPS ? 4u : 1u;

    auto t0 = std::chrono::steady_clock::now();
    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = dev->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (CreateGraphicsPipelineState)";
        rootSig->Release();
        fence->Release();
        cmd->Release();
        alloc->Release();
        queue->Release();
        dev->Release();
        return res;
    }

    uint32_t maxW = std::max(128u, width * 2);
    uint32_t maxH = std::max(128u, height * 2);
    ID3D12Resource* rtvTex = nullptr;
    ID3D12Resource* resolveTex = nullptr;
    if (isStateHeavyPS) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        D3D12_RESOURCE_DESC rtDesc{};
        rtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        rtDesc.Width = maxW;
        rtDesc.Height = maxH;
        rtDesc.DepthOrArraySize = 1;
        rtDesc.MipLevels = 1;
        rtDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtDesc.SampleDesc.Count = 4;
        rtDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rtDesc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&rtvTex));
        resolveTex = createNacelleRTTexture(dev, maxW, maxH);
    } else {
        rtvTex = createNacelleRTTexture(dev, maxW, maxH);
    }
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    dev->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap));
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(rtvTex, nullptr, rtvHandle);

    ID3D12Resource* dsvTex = nullptr;
    ID3D12DescriptorHeap* dsvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle{};
    ID3D12Resource* texBC1 = nullptr;
    ID3D12Resource* texBC7 = nullptr;
    ID3D12Resource* texUpload = nullptr;
    ID3D12DescriptorHeap* srvHeap = nullptr;
    MeshBenchSceneData meshScenes[3];
    ID3D12Resource* meshVBs[3] = {nullptr, nullptr, nullptr};
    ID3D12Resource* meshIBs[3] = {nullptr, nullptr, nullptr};

    if (isVertexMesh || isStateHeavyPS) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        D3D12_RESOURCE_DESC dsDesc{};
        dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dsDesc.Width = maxW;
        dsDesc.Height = maxH;
        dsDesc.DepthOrArraySize = 1;
        dsDesc.MipLevels = 1;
        dsDesc.Format = isStateHeavyPS ? DXGI_FORMAT_D24_UNORM_S8_UINT : DXGI_FORMAT_D32_FLOAT;
        dsDesc.SampleDesc.Count = isStateHeavyPS ? 4u : 1u;
        dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &dsDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&dsvTex));
        D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        dev->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap));
        dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        dev->CreateDepthStencilView(dsvTex, nullptr, dsvHandle);
    }

    if (isStateHeavyPS) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        auto mkBC = [&](DXGI_FORMAT fmt) -> ID3D12Resource* {
            D3D12_RESOURCE_DESC td{};
            td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            td.Width = 256;
            td.Height = 256;
            td.DepthOrArraySize = 1;
            td.MipLevels = 1;
            td.Format = fmt;
            td.SampleDesc.Count = 1;
            ID3D12Resource* tex = nullptr;
            dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &td, D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&tex));
            return tex;
        };
        texBC1 = mkBC(DXGI_FORMAT_BC1_UNORM);
        texBC7 = mkBC(DXGI_FORMAT_BC7_UNORM);
        auto bc1Data = buildTestBC1Texture256(); // 64*8 = 512 bytes/row, 64 rows = 32 KiB
        auto bc7Data = buildTestBC7Texture256(); // 64*16 = 1024 bytes/row, 64 rows = 64 KiB
        texUpload = createNacelleBuffer(dev, 128 * 1024, false);
        void* up = nullptr;
        texUpload->Map(0, nullptr, &up);
        std::memcpy(up, bc1Data.data(), bc1Data.size());
        std::memcpy(static_cast<uint8_t*>(up) + 64 * 1024, bc7Data.data(), bc7Data.size());
        texUpload->Unmap(0, nullptr);

        D3D12_TEXTURE_COPY_LOCATION d1{texBC1, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION s1{texUpload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        s1.PlacedFootprint.Offset = 0;
        s1.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 256, 256, 1, 512};
        cmd->CopyTextureRegion(&d1, 0, 0, 0, &s1, nullptr);

        D3D12_TEXTURE_COPY_LOCATION d2{texBC7, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION s2{texUpload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        s2.PlacedFootprint.Offset = 64 * 1024;
        s2.PlacedFootprint.Footprint = {DXGI_FORMAT_BC7_UNORM, 256, 256, 1, 1024};
        cmd->CopyTextureRegion(&d2, 0, 0, 0, &s2, nullptr);

        D3D12_RESOURCE_BARRIER b[2]{};
        b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition = {texBC1, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
        b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[1].Transition = {texBC7, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
        cmd->ResourceBarrier(2, b);
        cmd->Close();
        ID3D12CommandList* initLists[] = {cmd};
        queue->ExecuteCommandLists(1, initLists);
        queue->Signal(fence, 1);
        while (fence->GetCompletedValue() < 1) {}
        alloc->Reset();
        cmd->Reset(alloc, nullptr);

        D3D12_DESCRIPTOR_HEAP_DESC srvHD{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        dev->CreateDescriptorHeap(&srvHD, IID_PPV_ARGS(&srvHeap));
        UINT inc = dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE h0 = srvHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Format = DXGI_FORMAT_BC1_UNORM;
        dev->CreateShaderResourceView(texBC1, &srvDesc, h0);
        h0.ptr += inc;
        srvDesc.Format = DXGI_FORMAT_BC7_UNORM;
        dev->CreateShaderResourceView(texBC7, &srvDesc, h0);
    }

    if (isVertexMesh) {

        for (int m = 0; m < 3; ++m) {
            uint32_t g = 0, mrw = 0, mrh = 0;
            getMeshBenchGridAndDims(m, width, height, g, mrw, mrh);
            meshScenes[m] = buildMeshBenchScene(g, g);
            size_t vbBytes = meshScenes[m].vertices.size() * sizeof(MeshBenchVertex);
            size_t ibBytes = meshScenes[m].indices.size() * sizeof(uint32_t);
            meshVBs[m] = createNacelleBuffer(dev, vbBytes, false);
            meshIBs[m] = createNacelleBuffer(dev, ibBytes, false);
            void* p = nullptr;
            meshVBs[m]->Map(0, nullptr, &p);
            std::memcpy(p, meshScenes[m].vertices.data(), vbBytes);
            meshVBs[m]->Unmap(0, nullptr);
            meshIBs[m]->Map(0, nullptr, &p);
            std::memcpy(p, meshScenes[m].indices.data(), ibBytes);
            meshIBs[m]->Unmap(0, nullptr);
        }
    }

    ID3D12Resource* cbBuf = createNacelleBuffer(dev, isVertexMesh ? sizeof(MeshBenchConstants) : 256, false);
    ID3D12Resource* vbBuf = createNacelleBuffer(dev, isStateHeavyPS ? (6 * 5 * sizeof(float)) : (6 * 3 * sizeof(float)), false);
    {
        void* p = nullptr;
        vbBuf->Map(0, nullptr, &p);
        if (isStateHeavyPS) {
            const float quadVerts[30] = {
                -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
                 1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
                 1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
                 1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
                -1.0f,  1.0f, 0.0f, 0.0f, 0.0f,
            };
            std::memcpy(p, quadVerts, sizeof(quadVerts));
        } else {
            const float quadVerts[18] = {
                -1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f, -1.0f,  1.0f, 0.0f,
            };
            std::memcpy(p, quadVerts, sizeof(quadVerts));
        }
        vbBuf->Unmap(0, nullptr);
    }

    ID3D12Resource* texRes = nullptr;
    if (spec.hasTexture && !isStateHeavyPS) {
        D3D12_HEAP_PROPERTIES texHp{};
        texHp.Type = D3D12_HEAP_TYPE_CUSTOM;
        texHp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        texHp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        D3D12_RESOURCE_DESC texDesc{};
        texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texDesc.Width = 256;
        texDesc.Height = 256;
        texDesc.DepthOrArraySize = 1;
        texDesc.MipLevels = 9;
        texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texDesc.SampleDesc.Count = 1;
        dev->CreateCommittedResource(&texHp, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE, nullptr, IID_PPV_ARGS(&texRes));
        for (uint32_t mip = 0; mip < 9; ++mip) {
            uint32_t mw = std::max(1u, 256u >> mip);
            uint32_t mh = std::max(1u, 256u >> mip);
            auto pixels = buildProceduralTexture256Mip(mip);
            texRes->WriteToSubresource(mip, nullptr, pixels.data(), mw * 4u, mw * mh * 4u);
        }
        D3D12_DESCRIPTOR_HEAP_DESC srvHD{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        dev->CreateDescriptorHeap(&srvHD, IID_PPV_ARGS(&srvHeap));
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.Texture2D.MipLevels = 9;
        dev->CreateShaderResourceView(texRes, &srvDesc, srvHeap->GetCPUDescriptorHandleForHeapStart());
    }

    uint64_t fenceVal = isStateHeavyPS ? 2 : 1;
    auto runDraw = [&](int mode, uint32_t rw, uint32_t rh, uint32_t vertCount) -> double {
        uint32_t layerCount = (mode == 0) ? 8u : (mode == 1 ? 16u : 32u);
        void* p = nullptr;
        cbBuf->Map(0, nullptr, &p);
        if (isVertexMesh) {
            std::memcpy(p, &meshScenes[mode].constants, sizeof(MeshBenchConstants));
        } else if (isStateHeavyPS) {
            struct { uint32_t layerCount; uint32_t width; uint32_t height; float time; } scb{layerCount, rw, rh, 1.25f};
            std::memcpy(p, &scb, sizeof(scb));
        } else if (spec.category == BenchCategory::Pixel) {
            ShadertoyCB cb{static_cast<float>(rw), static_cast<float>(rh), 1.0f, 1.25f, 0.0f, 0.0f, 0.0f, 0.0f};
            std::memcpy(p, &cb, sizeof(cb));
        } else if (spec.category == BenchCategory::Vertex) {
            VertexStressCB cb{};
            cb.mvp[0] = 1.0f; cb.mvp[5] = 1.0f; cb.mvp[10] = 1.0f; cb.mvp[15] = 1.0f;
            cb.time = 0.5f;
            cb.vertexCount = vertCount;
            std::memcpy(p, &cb, sizeof(cb));
        } else if (spec.category == BenchCategory::Geometry) {
            GeomStressCB cb{0.05f, 0.5f, vertCount, 0.0f};
            std::memcpy(p, &cb, sizeof(cb));
        }
        cbBuf->Unmap(0, nullptr);

        auto s0 = std::chrono::steady_clock::now();
        cmd->SetGraphicsRootSignature(rootSig);
        cmd->SetPipelineState(pso);
        cmd->SetGraphicsRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        if (useSrvAndSampler && srvHeap) {
            ID3D12DescriptorHeap* heaps[] = {srvHeap};
            cmd->SetDescriptorHeaps(1, heaps);
            cmd->SetGraphicsRootDescriptorTable(1, srvHeap->GetGPUDescriptorHandleForHeapStart());
        }
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(rw), static_cast<float>(rh), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(rw), static_cast<LONG>(rh)};
        cmd->RSSetViewports(1, &vp);
        cmd->RSSetScissorRects(1, &sc);
        cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, (isVertexMesh || isStateHeavyPS) ? &dsvHandle : nullptr);
        const float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        cmd->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        if (isVertexMesh) {
            cmd->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            D3D12_VERTEX_BUFFER_VIEW vbv{
                meshVBs[mode]->GetGPUVirtualAddress(),
                static_cast<UINT>(meshScenes[mode].vertices.size() * sizeof(MeshBenchVertex)),
                sizeof(MeshBenchVertex)};
            D3D12_INDEX_BUFFER_VIEW ibv{
                meshIBs[mode]->GetGPUVirtualAddress(),
                static_cast<UINT>(meshScenes[mode].indices.size() * sizeof(uint32_t)),
                DXGI_FORMAT_R32_UINT};
            cmd->IASetVertexBuffers(0, 1, &vbv);
            cmd->IASetIndexBuffer(&ibv);
            cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            cmd->DrawIndexedInstanced(static_cast<UINT>(meshScenes[mode].indices.size()), 1, 0, 0, 0);
        } else {
            if (isStateHeavyPS) {
                cmd->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
                D3D12_VERTEX_BUFFER_VIEW vbv{vbBuf->GetGPUVirtualAddress(), 30 * sizeof(float), 5 * sizeof(float)};
                cmd->IASetVertexBuffers(0, 1, &vbv);
                cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                cmd->DrawInstanced(vertCount, layerCount, 0, 0);
                D3D12_RESOURCE_BARRIER rb[2]{};
                rb[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                rb[0].Transition = {rtvTex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE};
                rb[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                rb[1].Transition = {resolveTex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST};
                cmd->ResourceBarrier(2, rb);
                cmd->ResolveSubresource(resolveTex, 0, rtvTex, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
                rb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
                rb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
                rb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
                rb[1].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
                cmd->ResourceBarrier(2, rb);
            } else {
                if (usesQuadVB) {
                    D3D12_VERTEX_BUFFER_VIEW vbv{vbBuf->GetGPUVirtualAddress(), 18 * sizeof(float), 3 * sizeof(float)};
                    cmd->IASetVertexBuffers(0, 1, &vbv);
                    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                } else if (spec.kind == PipelineKind::GraphicsTessGS) {
                    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);
                } else if (spec.kind == PipelineKind::GraphicsVSGSPS) {
                    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
                } else {
                    cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                }
                cmd->DrawInstanced(vertCount, 1, 0, 0);
            }
        }
        cmd->Close();
        ID3D12CommandList* lists[] = {cmd};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        ++fenceVal;
        alloc->Reset();
        cmd->Reset(alloc, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    auto getGfxParams = [&](int mode, uint32_t& rw, uint32_t& rh, uint32_t& vc) {
        if (isVertexMesh) {
            uint32_t g = 0;
            getMeshBenchGridAndDims(mode, width, height, g, rw, rh);
            vc = static_cast<uint32_t>(meshScenes[mode].indices.size());
        } else if (spec.category == BenchCategory::Pixel) {
            rw = (mode == 0) ? 64 : (mode == 1 ? width : width * 2);
            rh = (mode == 0) ? 64 : (mode == 1 ? height : height * 2);
            vc = 6;
        } else if (spec.category == BenchCategory::Vertex) {
            rw = 128; rh = 128;
            vc = (mode == 0) ? 3000 : (mode == 1 ? 24000 : 96000);
        } else if (spec.category == BenchCategory::Geometry) {
            rw = 128; rh = 128;
            vc = (mode == 0) ? 512 : (mode == 1 ? 4096 : 16384);
        } else {
            // Tessellation: 3 control points per patch
            rw = 128; rh = 128;
            vc = (mode == 0) ? 64 * 3 : (mode == 1 ? 512 * 3 : 2048 * 3);
        }
    };

    uint32_t rw = 128, rh = 128, vc = 6;
    getGfxParams(1, rw, rh, vc);

    res.firstDispatchMs = runDraw(1, rw, rh, vc);
    nacelle::waitForAllAsyncJITTasks();
    runDraw(1, rw, rh, vc);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runDraw(1, rw, rh, vc));
    }

    // Checksum
    ID3D12Resource* readTex = isStateHeavyPS ? resolveTex : rtvTex;
    const uint8_t* px = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(rw) * rh * 4};
    if (SUCCEEDED(readTex->Map(0, &range, reinterpret_cast<void**>(const_cast<uint8_t**>(&px)))) && px) {
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(rw) * rh * 4; ++i) sum += px[i];
        readTex->Unmap(0, nullptr);
        res.checksum = sum;
    }

    getGfxParams(0, rw, rh, vc);
    res.shortJobMs = runDraw(0, rw, rh, vc);
    getGfxParams(2, rw, rh, vc);
    res.longJobMs = runDraw(2, rw, rh, vc);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    for (int m = 0; m < 3; ++m) {
        if (meshIBs[m]) meshIBs[m]->Release();
        if (meshVBs[m]) meshVBs[m]->Release();
    }
    if (srvHeap) srvHeap->Release();
    if (texUpload) texUpload->Release();
    if (texBC7) texBC7->Release();
    if (texBC1) texBC1->Release();
    if (resolveTex) resolveTex->Release();
    if (texRes) texRes->Release();
    if (dsvHeap) dsvHeap->Release();
    if (dsvTex) dsvTex->Release();
    vbBuf->Release();
    cbBuf->Release();
    rtvHeap->Release();
    rtvTex->Release();
    pso->Release();
    rootSig->Release();
    fence->Release();
    cmd->Release();
    alloc->Release();
    queue->Release();
    dev->Release();
    return res;
}

inline BenchRunResult runNacelleRaytracingBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& dxil,
    uint32_t simdWidth,
    uint32_t width,
    uint32_t height,
    uint32_t iters) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Nacelle (W=" + std::to_string(simdWidth) + ")";
    res.shaderFormat = "DXIL";
    res.simdWidth = simdWidth;

    MemorySnapshot memBefore = queryProcessMemory();
    nacelle::setSimdWidthOverride(simdWidth);
    nacelle::clearInMemoryPSOCacheForTesting();

    ID3D12Device* baseDev = nullptr;
    if (FAILED(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&baseDev))) || !baseDev) {
        res.status = "FAILED (CreateDevice)";
        return res;
    }
    ID3D12Device5* dev = nullptr;
    if (FAILED(baseDev->QueryInterface(IID_PPV_ARGS(&dev))) || !dev) {
        baseDev->Release();
        res.status = "FAILED (ID3D12Device5)";
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&queue));
    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
    ID3D12GraphicsCommandList* baseCmd = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc, nullptr, IID_PPV_ARGS(&baseCmd));
    ID3D12GraphicsCommandList4* cmd4 = nullptr;
    baseCmd->QueryInterface(IID_PPV_ARGS(&cmd4));
    ID3D12Fence* fence = nullptr;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    // Root signature: t0 (SRV), u0 (UAV)
    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].Descriptor.ShaderRegister = 0;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    rsBlob->Release();

    const bool isHeavyRT = (spec.id == "rt_inline_heavy" || spec.id == "rt_pipeline_heavy");

    // Build triangle BLAS + TLAS
    const float triVerts[9] = {0.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    std::vector<float> heavyVerts;
    const float* vbSrc = triVerts;
    size_t vbBytes = sizeof(triVerts);
    uint32_t vtxCount = 3u;
    if (isHeavyRT) {
        heavyVerts = buildHeavyRTBlasVertices();
        vbSrc = heavyVerts.data();
        vbBytes = heavyVerts.size() * sizeof(float);
        vtxCount = static_cast<uint32_t>(heavyVerts.size() / 3u);
    }
    ID3D12Resource* vb = createNacelleBuffer(dev, vbBytes, false);
    void* mapped = nullptr;
    vb->Map(0, nullptr, &mapped);
    std::memcpy(mapped, vbSrc, vbBytes);
    vb->Unmap(0, nullptr);

    D3D12_RAYTRACING_GEOMETRY_DESC geom{};
    geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geom.Triangles.VertexCount = vtxCount;
    geom.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress();
    geom.Triangles.VertexBuffer.StrideInBytes = 12;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasIn{};
    blasIn.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasIn.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasIn.NumDescs = 1;
    blasIn.pGeometryDescs = &geom;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPre{};
    dev->GetRaytracingAccelerationStructurePrebuildInfo(&blasIn, &blasPre);
    ID3D12Resource* blasBuf = createNacelleBuffer(dev, std::max<UINT64>(65536, blasPre.ResultDataMaxSizeInBytes));
    ID3D12Resource* scratch = createNacelleBuffer(dev, std::max<UINT64>(65536, blasPre.ScratchDataSizeInBytes));

    ID3D12Resource* instBuf = nullptr;
    uint32_t numInstances = 1u;
    if (isHeavyRT) {
        auto insts = buildHeavyRTTlasInstances<D3D12_RAYTRACING_INSTANCE_DESC>(blasBuf->GetGPUVirtualAddress());
        numInstances = static_cast<uint32_t>(insts.size());
        instBuf = createNacelleBuffer(dev, insts.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC), false);
        instBuf->Map(0, nullptr, &mapped);
        std::memcpy(mapped, insts.data(), insts.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
        instBuf->Unmap(0, nullptr);
    } else {
        D3D12_RAYTRACING_INSTANCE_DESC inst{};
        inst.Transform[0][0] = 1.0f; inst.Transform[1][1] = 1.0f; inst.Transform[2][2] = 1.0f;
        inst.InstanceID = 42;
        inst.InstanceMask = 0xFF;
        inst.AccelerationStructure = blasBuf->GetGPUVirtualAddress();
        instBuf = createNacelleBuffer(dev, sizeof(inst), false);
        instBuf->Map(0, nullptr, &mapped);
        std::memcpy(mapped, &inst, sizeof(inst));
        instBuf->Unmap(0, nullptr);
    }

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasIn{};
    tlasIn.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasIn.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasIn.NumDescs = numInstances;
    tlasIn.InstanceDescs = instBuf->GetGPUVirtualAddress();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPre{};
    dev->GetRaytracingAccelerationStructurePrebuildInfo(&tlasIn, &tlasPre);
    ID3D12Resource* tlasBuf = createNacelleBuffer(dev, std::max<UINT64>(65536, tlasPre.ResultDataMaxSizeInBytes));

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildBlas{};
    buildBlas.DestAccelerationStructureData = blasBuf->GetGPUVirtualAddress();
    buildBlas.Inputs = blasIn;
    buildBlas.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    cmd4->BuildRaytracingAccelerationStructure(&buildBlas, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBar{};
    uavBar.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBar.UAV.pResource = blasBuf;
    cmd4->ResourceBarrier(1, &uavBar);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildTlas{};
    buildTlas.DestAccelerationStructureData = tlasBuf->GetGPUVirtualAddress();
    buildTlas.Inputs = tlasIn;
    buildTlas.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();
    cmd4->BuildRaytracingAccelerationStructure(&buildTlas, 0, nullptr);
    cmd4->Close();
    ID3D12CommandList* asLists[] = {baseCmd};
    queue->ExecuteCommandLists(1, asLists);
    uint64_t fenceVal = 1;
    queue->Signal(fence, fenceVal);
    while (fence->GetCompletedValue() < fenceVal) {}
    ++fenceVal;
    alloc->Reset();
    baseCmd->Reset(alloc, nullptr);

    ID3D12PipelineState* pso = nullptr;
    ID3D12StateObject* stateObj = nullptr;
    ID3D12Resource* shaderTable = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS rgenAddr = 0, missAddr = 0, hitAddr = 0;
    uint64_t missStride = 0, hitStride = 0;

    auto t0 = std::chrono::steady_clock::now();
    HRESULT hr = S_OK;
    if (spec.kind == PipelineKind::RayTracingInline) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rootSig;
        psoDesc.CS = {dxil.data(), dxil.size()};
        hr = dev->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso));
    } else {
        D3D12_DXIL_LIBRARY_DESC libDesc{};
        libDesc.DXILLibrary = {dxil.data(), dxil.size()};
        D3D12_HIT_GROUP_DESC hg{};
        hg.HitGroupExport = L"TriangleHitGroup";
        hg.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
        hg.ClosestHitShaderImport = L"ClosestHitMain";
        D3D12_RAYTRACING_SHADER_CONFIG scfg{isHeavyRT ? 32u : static_cast<UINT>(sizeof(uint32_t)), sizeof(float) * 2};
        D3D12_RAYTRACING_PIPELINE_CONFIG pcfg{isHeavyRT ? 4u : 1u};
        D3D12_GLOBAL_ROOT_SIGNATURE grs{rootSig};
        D3D12_STATE_SUBOBJECT subs[5]{};
        subs[0] = {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libDesc};
        subs[1] = {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hg};
        subs[2] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &scfg};
        subs[3] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pcfg};
        subs[4] = {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &grs};
        D3D12_STATE_OBJECT_DESC soDesc{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE, 5, subs};
        hr = dev->CreateStateObject(&soDesc, IID_PPV_ARGS(&stateObj));
        if (SUCCEEDED(hr) && stateObj) {
            ID3D12StateObjectProperties* sop = nullptr;
            stateObj->QueryInterface(IID_PPV_ARGS(&sop));
            shaderTable = createNacelleBuffer(dev, 3 * 64, false);
            uint8_t* st = nullptr;
            shaderTable->Map(0, nullptr, reinterpret_cast<void**>(&st));
            std::memset(st, 0, 3 * 64);
            if (sop) {
                void* rg = sop->GetShaderIdentifier(L"RayGen");
                void* ms = sop->GetShaderIdentifier(L"MissMain");
                void* hgId = sop->GetShaderIdentifier(L"TriangleHitGroup");
                if (rg) std::memcpy(st + 0, rg, 32);
                if (ms) std::memcpy(st + 64, ms, 32);
                if (hgId) std::memcpy(st + 128, hgId, 32);
                sop->Release();
            }
            shaderTable->Unmap(0, nullptr);
            rgenAddr = shaderTable->GetGPUVirtualAddress();
            missAddr = rgenAddr + 64;
            hitAddr = rgenAddr + 128;
            missStride = 64;
            hitStride = 64;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || (spec.kind == PipelineKind::RayTracingInline && !pso) ||
        (spec.kind == PipelineKind::RayTracingPipeline && !stateObj)) {
        res.status = "FAILED (RT Pipeline)";
        if (shaderTable) shaderTable->Release();
        if (stateObj) stateObj->Release();
        if (pso) pso->Release();
        tlasBuf->Release();
        instBuf->Release();
        scratch->Release();
        blasBuf->Release();
        vb->Release();
        rootSig->Release();
        fence->Release();
        cmd4->Release();
        baseCmd->Release();
        alloc->Release();
        queue->Release();
        dev->Release();
        baseDev->Release();
        return res;
    }

    ID3D12Resource* outBuf = createNacelleBuffer(dev, 4 * 1024 * 1024);

    auto runRT = [&](uint32_t w, uint32_t h) -> double {
        auto s0 = std::chrono::steady_clock::now();
        baseCmd->SetComputeRootSignature(rootSig);
        if (spec.kind == PipelineKind::RayTracingInline) {
            baseCmd->SetPipelineState(pso);
            baseCmd->SetComputeRootShaderResourceView(0, tlasBuf->GetGPUVirtualAddress());
            baseCmd->SetComputeRootUnorderedAccessView(1, outBuf->GetGPUVirtualAddress());
            if (isHeavyRT) {
                baseCmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            } else {
                baseCmd->Dispatch((w + 3) / 4, h, 1);
            }
        } else {
            cmd4->SetPipelineState1(stateObj);
            baseCmd->SetComputeRootShaderResourceView(0, tlasBuf->GetGPUVirtualAddress());
            baseCmd->SetComputeRootUnorderedAccessView(1, outBuf->GetGPUVirtualAddress());
            D3D12_DISPATCH_RAYS_DESC dr{};
            dr.RayGenerationShaderRecord = {rgenAddr, 64};
            dr.MissShaderTable = {missAddr, 64, missStride};
            dr.HitGroupTable = {hitAddr, 64, hitStride};
            dr.Width = w;
            dr.Height = h;
            dr.Depth = 1;
            cmd4->DispatchRays(&dr);
        }
        baseCmd->Close();
        ID3D12CommandList* lists[] = {baseCmd};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        ++fenceVal;
        alloc->Reset();
        baseCmd->Reset(alloc, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    res.firstDispatchMs = runRT(width, height);
    nacelle::waitForAllAsyncJITTasks();
    runRT(width, height);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runRT(width, height));
    }

    const size_t checksumCount = std::min<size_t>(
        1024u * 1024u,
        (spec.kind == PipelineKind::RayTracingInline) ? static_cast<size_t>(width) * 4u : static_cast<size_t>(width));
    uint32_t* outPtr = nullptr;
    outBuf->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
    double sum = 0.0;
    for (size_t i = 0; i < checksumCount; ++i) sum += static_cast<double>(outPtr[i] & 0xFFFFu);
    outBuf->Unmap(0, nullptr);
    res.checksum = sum;

    res.shortJobMs = runRT(64, 64);
    res.longJobMs = runRT(width * 2, height * 2);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    outBuf->Release();
    if (shaderTable) shaderTable->Release();
    if (stateObj) stateObj->Release();
    if (pso) pso->Release();
    tlasBuf->Release();
    instBuf->Release();
    scratch->Release();
    blasBuf->Release();
    vb->Release();
    rootSig->Release();
    fence->Release();
    cmd4->Release();
    baseCmd->Release();
    alloc->Release();
    queue->Release();
    dev->Release();
    baseDev->Release();
    return res;
}

inline BenchRunResult runNacelleMeshBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& msDxil,
    const std::vector<uint8_t>& psDxil,
    uint32_t simdWidth,
    uint32_t iters,
    const std::vector<uint8_t>& asDxil = {}) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Nacelle (W=" + std::to_string(simdWidth) + ")";
    res.shaderFormat = "DXIL";
    res.simdWidth = simdWidth;

    MemorySnapshot memBefore = queryProcessMemory();
    nacelle::setSimdWidthOverride(simdWidth);
    nacelle::clearInMemoryPSOCacheForTesting();

    ID3D12Device* dev = nullptr;
    if (FAILED(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&dev))) || !dev) {
        res.status = "FAILED (CreateDevice)";
        return res;
    }
    ID3D12Device2* dev2 = nullptr;
    if (FAILED(dev->QueryInterface(IID_PPV_ARGS(&dev2))) || !dev2) {
        res.status = "FAILED (QI ID3D12Device2)";
        dev->Release();
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, IID_PPV_ARGS(&queue));
    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    ID3D12GraphicsCommandList* cmd = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, IID_PPV_ARGS(&cmd));
    ID3D12GraphicsCommandList6* cmd6 = nullptr;
    if (FAILED(cmd->QueryInterface(IID_PPV_ARGS(&cmd6))) || !cmd6) {
        res.status = "FAILED (QI ID3D12GraphicsCommandList6)";
        cmd->Release();
        alloc->Release();
        queue->Release();
        dev2->Release();
        dev->Release();
        return res;
    }
    ID3D12Fence* fence = nullptr;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    param.Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &param;
    ID3DBlob* rsBlob = nullptr;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    rsBlob->Release();

    const bool hasAS = !asDxil.empty();
    const bool isCullMesh = (spec.id == "bench_mesh_cull");

    struct alignas(void*) MeshPsoStream {
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE; ID3D12RootSignature* v; } rs;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS; D3D12_SHADER_BYTECODE v; } ms;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS; D3D12_SHADER_BYTECODE v; } ps;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER; D3D12_RASTERIZER_DESC v{}; } rast;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND; D3D12_BLEND_DESC v{}; } blend;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL; D3D12_DEPTH_STENCIL_DESC v{}; } dss;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK; UINT v = UINT_MAX; } sampleMask;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY; D3D12_PRIMITIVE_TOPOLOGY_TYPE v = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; } topo;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS; D3D12_RT_FORMAT_ARRAY v{}; } rtfs;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT; DXGI_FORMAT v = DXGI_FORMAT_UNKNOWN; } dsvFmt;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC; DXGI_SAMPLE_DESC v{1, 0}; } sampleDesc;
    } stream;

    struct alignas(void*) MeshASPsoStream {
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE; ID3D12RootSignature* v; } rs;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS; D3D12_SHADER_BYTECODE v; } as;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS; D3D12_SHADER_BYTECODE v; } ms;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS; D3D12_SHADER_BYTECODE v; } ps;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER; D3D12_RASTERIZER_DESC v{}; } rast;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND; D3D12_BLEND_DESC v{}; } blend;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL; D3D12_DEPTH_STENCIL_DESC v{}; } dss;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK; UINT v = UINT_MAX; } sampleMask;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY; D3D12_PRIMITIVE_TOPOLOGY_TYPE v = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE; } topo;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS; D3D12_RT_FORMAT_ARRAY v{}; } rtfs;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT; DXGI_FORMAT v = DXGI_FORMAT_UNKNOWN; } dsvFmt;
        struct alignas(void*) { D3D12_PIPELINE_STATE_SUBOBJECT_TYPE t = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC; DXGI_SAMPLE_DESC v{1, 0}; } sampleDesc;
    } streamAS;

    D3D12_PIPELINE_STATE_STREAM_DESC streamDesc{};
    if (hasAS) {
        streamAS.rs.v = rootSig;
        streamAS.as.v = {asDxil.data(), asDxil.size()};
        streamAS.ms.v = {msDxil.data(), msDxil.size()};
        streamAS.ps.v = {psDxil.data(), psDxil.size()};
        streamAS.rast.v.FillMode = D3D12_FILL_MODE_SOLID;
        streamAS.rast.v.CullMode = isCullMesh ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
        streamAS.rast.v.FrontCounterClockwise = isCullMesh ? TRUE : FALSE;
        streamAS.blend.v.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        if (isCullMesh) {
            streamAS.dss.v.DepthEnable = TRUE;
            streamAS.dss.v.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            streamAS.dss.v.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
            streamAS.dsvFmt.v = DXGI_FORMAT_D32_FLOAT;
        }
        streamAS.rtfs.v.NumRenderTargets = 1;
        streamAS.rtfs.v.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        streamDesc = {sizeof(streamAS), &streamAS};
    } else {
        stream.rs.v = rootSig;
        stream.ms.v = {msDxil.data(), msDxil.size()};
        stream.ps.v = {psDxil.data(), psDxil.size()};
        stream.rast.v.FillMode = D3D12_FILL_MODE_SOLID;
        stream.rast.v.CullMode = isCullMesh ? D3D12_CULL_MODE_BACK : D3D12_CULL_MODE_NONE;
        stream.blend.v.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
        if (isCullMesh) {
            stream.dss.v.DepthEnable = TRUE;
            stream.dss.v.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
            stream.dss.v.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
            stream.dsvFmt.v = DXGI_FORMAT_D32_FLOAT;
        }
        stream.rtfs.v.NumRenderTargets = 1;
        stream.rtfs.v.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        streamDesc = {sizeof(stream), &stream};
    }

    auto t0 = std::chrono::steady_clock::now();
    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = dev2->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&pso));
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (CreatePipelineState Mesh)";
        rootSig->Release();
        fence->Release();
        cmd6->Release();
        cmd->Release();
        alloc->Release();
        queue->Release();
        dev2->Release();
        dev->Release();
        return res;
    }

    const uint32_t rw = isCullMesh ? 256u : 128u;
    const uint32_t rh = isCullMesh ? 256u : 128u;
    ID3D12Resource* rtvTex = createNacelleRTTexture(dev, rw, rh);
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    dev->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap));
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(rtvTex, nullptr, rtvHandle);

    ID3D12Resource* dsvTex = nullptr;
    ID3D12DescriptorHeap* dsvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle{};
    if (isCullMesh) {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = D3D12_HEAP_TYPE_CUSTOM;
        hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
        hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
        D3D12_RESOURCE_DESC dsDesc{};
        dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dsDesc.Width = rw;
        dsDesc.Height = rh;
        dsDesc.DepthOrArraySize = 1;
        dsDesc.MipLevels = 1;
        dsDesc.Format = DXGI_FORMAT_D32_FLOAT;
        dsDesc.SampleDesc.Count = 1;
        dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &dsDesc, D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr, IID_PPV_ARGS(&dsvTex));
        D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        dev->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap));
        dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        dev->CreateDepthStencilView(dsvTex, nullptr, dsvHandle);
    }
    ID3D12Resource* cbBuf = createNacelleBuffer(dev, 256, false);

    uint64_t fenceVal = 1;
    auto runMesh = [&](uint32_t groups, uint32_t gridX = 32, uint32_t layers = 4) -> double {
        void* p = nullptr;
        cbBuf->Map(0, nullptr, &p);
        if (isCullMesh) {
            struct { float scale; float time; uint32_t gridX; uint32_t layerCount; } cb{1.0f, 0.25f, gridX, layers};
            std::memcpy(p, &cb, sizeof(cb));
        } else {
            struct { float scale; float time; uint32_t groupCount; float pad; } cb{0.05f, 0.25f, groups, 0.0f};
            std::memcpy(p, &cb, sizeof(cb));
        }
        cbBuf->Unmap(0, nullptr);

        auto s0 = std::chrono::steady_clock::now();
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(rw), static_cast<float>(rh), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(rw), static_cast<LONG>(rh)};
        const float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};

        cmd->SetGraphicsRootSignature(rootSig);
        cmd->SetPipelineState(pso);
        cmd->SetGraphicsRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        cmd->RSSetViewports(1, &vp);
        cmd->RSSetScissorRects(1, &sc);
        cmd->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        if (isCullMesh) {
            cmd->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
        } else {
            cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        }
        cmd6->DispatchMesh(groups, 1, 1);
        cmd->Close();

        ID3D12CommandList* lists[] = {cmd};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        ++fenceVal;
        alloc->Reset();
        cmd->Reset(alloc, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    const uint32_t steadyGroups = isCullMesh ? 64u : 128u;
    const uint32_t shortGroups = isCullMesh ? 8u : 16u;
    const uint32_t longGroups = isCullMesh ? 256u : 512u;

    res.firstDispatchMs = runMesh(steadyGroups, 32, 4);
    nacelle::waitForAllAsyncJITTasks();
    runMesh(steadyGroups, 32, 4);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runMesh(steadyGroups, 32, 4));
    }

    const uint8_t* px = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(rw) * rh * 4};
    if (SUCCEEDED(rtvTex->Map(0, &range, reinterpret_cast<void**>(const_cast<uint8_t**>(&px)))) && px) {
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(rw) * rh * 4; ++i) sum += px[i];
        rtvTex->Unmap(0, nullptr);
        res.checksum = sum;
    }

    res.shortJobMs = runMesh(shortGroups, 16, 2);
    res.longJobMs = runMesh(longGroups, 64, 4);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    cbBuf->Release();
    if (dsvHeap) dsvHeap->Release();
    if (dsvTex) dsvTex->Release();
    rtvHeap->Release();
    rtvTex->Release();
    pso->Release();
    rootSig->Release();
    fence->Release();
    cmd6->Release();
    cmd->Release();
    alloc->Release();
    queue->Release();
    dev2->Release();
    dev->Release();
    return res;
}

} // namespace nacelle_bench
