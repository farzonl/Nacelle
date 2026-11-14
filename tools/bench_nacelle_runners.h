#pragma once

#include "bench_memory.h"
#include "bench_types.h"

#include <chrono>
#include <cstring>
#include <vector>

#include <nacelle/nacelle.h>
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"

namespace nacelle_bench {

struct RaymarchCB {
    uint32_t width;
    uint32_t height;
    uint32_t maxSteps;
    float time;
};

struct MatmulCB {
    uint32_t M;
    uint32_t N;
    uint32_t K;
    uint32_t pad;
};

struct ShadertoyCB {
    float resX, resY, resZ;
    float iTime;
    float mouseX, mouseY, mouseZ, mouseW;
};

struct VertexStressCB {
    float mvp[16];
    float time;
    uint32_t vertexCount;
    float pad[2];
};

struct GeomStressCB {
    float scale;
    float time;
    uint32_t pointCount;
    float pad;
};

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

    // Root signature: b0, t0, t1, u0, u1 (covers all 5 compute benchmarks)
    D3D12_ROOT_PARAMETER params[5]{};
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

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 5;
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
    ID3D12Resource* srv0 = createNacelleBuffer(dev, 4 * 1024 * 1024);
    ID3D12Resource* srv1 = createNacelleBuffer(dev, 4 * 1024 * 1024);
    ID3D12Resource* uav0 = createNacelleBuffer(dev, maxOutBytes);
    ID3D12Resource* uav1 = createNacelleBuffer(dev, 4 * 1024 * 1024);

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
    } else if (spec.id == "matmul") {
        float* a = nullptr;
        float* b = nullptr;
        srv0->Map(0, nullptr, reinterpret_cast<void**>(&a));
        srv1->Map(0, nullptr, reinterpret_cast<void**>(&b));
        for (size_t i = 0; i < 256 * 256; ++i) {
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
        } else if (spec.id == "matmul") {
            MatmulCB cb{cbW, cbH, cbSteps, 0};
            std::memcpy(cbPtr, &cb, sizeof(cb));
        }
        cbBuf->Unmap(0, nullptr);

        auto s0 = std::chrono::steady_clock::now();
        cmd->SetComputeRootSignature(rootSig);
        cmd->SetPipelineState(pso);
        cmd->SetComputeRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(1, srv0->GetGPUVirtualAddress());
        cmd->SetComputeRootShaderResourceView(2, srv1->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(3, uav0->GetGPUVirtualAddress());
        cmd->SetComputeRootUnorderedAccessView(4, uav1->GetGPUVirtualAddress());
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
    } else if (spec.id == "matmul") {
        float* outPtr = nullptr;
        uav0->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
        for (size_t i = 0; i < 128 * 128; ++i) sum += outPtr[i];
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

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    param.Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &param;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob* rsBlob = nullptr;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rootSig));
    rsBlob->Release();

    D3D12_INPUT_ELEMENT_DESC posElem{"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig;
    psoDesc.VS = {vsDxil.data(), vsDxil.size()};
    psoDesc.PS = {psDxil.data(), psDxil.size()};
    if (!hsDxil.empty()) psoDesc.HS = {hsDxil.data(), hsDxil.size()};
    if (!dsDxil.empty()) psoDesc.DS = {dsDxil.data(), dsDxil.size()};
    if (!gsDxil.empty()) psoDesc.GS = {gsDxil.data(), gsDxil.size()};

    bool usesQuadVB = (spec.kind == PipelineKind::GraphicsVSPS && spec.category == BenchCategory::Pixel);
    if (usesQuadVB) {
        psoDesc.InputLayout = {&posElem, 1};
    }
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
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
    psoDesc.SampleDesc.Count = 1;

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

    uint32_t maxW = width * 2;
    uint32_t maxH = height * 2;
    ID3D12Resource* rtvTex = createNacelleRTTexture(dev, maxW, maxH);
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    dev->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap));
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    dev->CreateRenderTargetView(rtvTex, nullptr, rtvHandle);

    ID3D12Resource* cbBuf = createNacelleBuffer(dev, 256, false);
    ID3D12Resource* vbBuf = createNacelleBuffer(dev, 6 * 3 * sizeof(float), false);
    {
        const float quadVerts[18] = {
            -1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f,
            -1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f, -1.0f,  1.0f, 0.0f,
        };
        void* p = nullptr;
        vbBuf->Map(0, nullptr, &p);
        std::memcpy(p, quadVerts, sizeof(quadVerts));
        vbBuf->Unmap(0, nullptr);
    }

    uint64_t fenceVal = 1;
    auto runDraw = [&](uint32_t rw, uint32_t rh, uint32_t vertCount) -> double {
        void* p = nullptr;
        cbBuf->Map(0, nullptr, &p);
        if (spec.category == BenchCategory::Pixel) {
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
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(rw), static_cast<float>(rh), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(rw), static_cast<LONG>(rh)};
        cmd->RSSetViewports(1, &vp);
        cmd->RSSetScissorRects(1, &sc);
        cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        const float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        cmd->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

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
        if (spec.category == BenchCategory::Pixel) {
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

    res.firstDispatchMs = runDraw(rw, rh, vc);
    nacelle::waitForAllAsyncJITTasks();
    runDraw(rw, rh, vc);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runDraw(rw, rh, vc));
    }

    // Checksum
    const uint8_t* px = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(rw) * rh * 4};
    if (SUCCEEDED(rtvTex->Map(0, &range, reinterpret_cast<void**>(const_cast<uint8_t**>(&px)))) && px) {
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(rw) * rh * 4; ++i) sum += px[i];
        rtvTex->Unmap(0, nullptr);
        res.checksum = sum;
    }

    getGfxParams(0, rw, rh, vc);
    res.shortJobMs = runDraw(rw, rh, vc);
    getGfxParams(2, rw, rh, vc);
    res.longJobMs = runDraw(rw, rh, vc);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

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

    // Build triangle BLAS + TLAS
    const float triVerts[9] = {0.0f, 1.0f, 0.0f, -1.0f, -1.0f, 0.0f, 1.0f, -1.0f, 0.0f};
    ID3D12Resource* vb = createNacelleBuffer(dev, sizeof(triVerts), false);
    void* mapped = nullptr;
    vb->Map(0, nullptr, &mapped);
    std::memcpy(mapped, triVerts, sizeof(triVerts));
    vb->Unmap(0, nullptr);

    D3D12_RAYTRACING_GEOMETRY_DESC geom{};
    geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    geom.Triangles.VertexCount = 3;
    geom.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress();
    geom.Triangles.VertexBuffer.StrideInBytes = 12;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasIn{};
    blasIn.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasIn.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasIn.NumDescs = 1;
    blasIn.pGeometryDescs = &geom;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasPre{};
    dev->GetRaytracingAccelerationStructurePrebuildInfo(&blasIn, &blasPre);
    ID3D12Resource* blasBuf = createNacelleBuffer(dev, std::max<UINT64>(4096, blasPre.ResultDataMaxSizeInBytes));
    ID3D12Resource* scratch = createNacelleBuffer(dev, std::max<UINT64>(4096, blasPre.ScratchDataSizeInBytes));

    D3D12_RAYTRACING_INSTANCE_DESC inst{};
    inst.Transform[0][0] = 1.0f; inst.Transform[1][1] = 1.0f; inst.Transform[2][2] = 1.0f;
    inst.InstanceID = 42;
    inst.InstanceMask = 0xFF;
    inst.AccelerationStructure = blasBuf->GetGPUVirtualAddress();
    ID3D12Resource* instBuf = createNacelleBuffer(dev, sizeof(inst), false);
    instBuf->Map(0, nullptr, &mapped);
    std::memcpy(mapped, &inst, sizeof(inst));
    instBuf->Unmap(0, nullptr);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasIn{};
    tlasIn.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasIn.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasIn.NumDescs = 1;
    tlasIn.InstanceDescs = instBuf->GetGPUVirtualAddress();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPre{};
    dev->GetRaytracingAccelerationStructurePrebuildInfo(&tlasIn, &tlasPre);
    ID3D12Resource* tlasBuf = createNacelleBuffer(dev, std::max<UINT64>(4096, tlasPre.ResultDataMaxSizeInBytes));

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
        D3D12_RAYTRACING_SHADER_CONFIG scfg{sizeof(uint32_t), sizeof(float) * 2};
        D3D12_RAYTRACING_PIPELINE_CONFIG pcfg{1};
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
            baseCmd->Dispatch((w + 3) / 4, h, 1);
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

    res.firstDispatchMs = runRT(64, 1);
    nacelle::waitForAllAsyncJITTasks();
    runRT(64, 1);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runRT(64, 1));
    }

    uint32_t* outPtr = nullptr;
    outBuf->Map(0, nullptr, reinterpret_cast<void**>(&outPtr));
    double sum = 0.0;
    for (size_t i = 0; i < 64; ++i) sum += static_cast<double>(outPtr[i] & 0xFFFFu);
    outBuf->Unmap(0, nullptr);
    res.checksum = sum;

    res.shortJobMs = runRT(16, 1);
    res.longJobMs = runRT(256, 1);

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

} // namespace nacelle_bench
