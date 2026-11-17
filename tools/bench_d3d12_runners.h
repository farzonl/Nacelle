#pragma once

#include "bench_types.h"
#include "bench_common.h"
#include "bench_bcn_textures.h"
#include "bench_rt_scene.h"

#if defined(_WIN32)
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace nacelle_bench {

struct D3D12DeviceInfo {
    uint32_t index = 0;
    std::string backendName;
    bool isWarp = false;
};

using PFN_D3D12CreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);

inline std::string wideToUtf8(const WCHAR* wstr) {
    if (!wstr || !wstr[0]) return "";
    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 1) return "";
    std::string out(static_cast<size_t>(len - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, out.data(), len, nullptr, nullptr);
    return out;
}

inline bool isSoftwareOrBasicRenderAdapter(const DXGI_ADAPTER_DESC1& desc) {
    if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) return true;
    // Microsoft Basic Render Driver (WARP) may be enumerated without FLAG_SOFTWARE on VMs.
    return desc.VendorId == 0x1414 && desc.DeviceId == 0x008c;
}

inline bool selectD3D12Adapter(
    IDXGIFactory4* factory,
    PFN_D3D12CreateDeviceFn pD3D12CreateDevice,
    int32_t targetIdx,
    bool forceWarp,
    IDXGIAdapter1** outAdapter,
    std::string& outBackendName,
    std::string& outErr) {
    *outAdapter = nullptr;
    if (forceWarp) {
        IDXGIAdapter1* warp = nullptr;
        if (FAILED(factory->EnumWarpAdapter(__uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&warp))) || !warp) {
            outErr = "SKIPPED (WARP adapter unavailable)";
            return false;
        }
        *outAdapter = warp;
        outBackendName = "Microsoft WARP";
        return true;
    }

    struct Cand {
        IDXGIAdapter1* adapter = nullptr;
        std::string name;
    };
    std::vector<Cand> cands;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* a = nullptr;
        if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        a->GetDesc1(&desc);
        if (isSoftwareOrBasicRenderAdapter(desc)) {
            a->Release();
            continue;
        }
        if (SUCCEEDED(pD3D12CreateDevice(a, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            std::string name = wideToUtf8(desc.Description);
            cands.push_back({a, name.empty() ? "DirectX 12" : ("DirectX 12: " + name)});
        } else {
            a->Release();
        }
    }

    IDXGIAdapter1* warp = nullptr;
    if (SUCCEEDED(factory->EnumWarpAdapter(__uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&warp))) && warp) {
        if (SUCCEEDED(pD3D12CreateDevice(warp, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
            cands.push_back({warp, "Microsoft WARP"});
        } else {
            warp->Release();
        }
    }

    if (cands.empty()) {
        outErr = "SKIPPED (no D3D12 adapters available)";
        return false;
    }

    size_t idx = (targetIdx >= 0) ? static_cast<size_t>(targetIdx) : 0;
    if (idx >= cands.size()) {
        for (auto& c : cands) c.adapter->Release();
        outErr = "SKIPPED (D3D12 device index out of range)";
        return false;
    }

    for (size_t i = 0; i < cands.size(); ++i) {
        if (i == idx) {
            *outAdapter = cands[i].adapter;
            outBackendName = cands[i].name;
        } else {
            cands[i].adapter->Release();
        }
    }
    return true;
}

inline std::vector<D3D12DeviceInfo> listD3D12Devices() {
    std::vector<D3D12DeviceInfo> out;
    HMODULE dxgiLib = LoadLibraryA("dxgi.dll");
    HMODULE d3d12Lib = LoadLibraryA("d3d12.dll");
    if (!dxgiLib || !d3d12Lib) {
        if (dxgiLib) FreeLibrary(dxgiLib);
        if (d3d12Lib) FreeLibrary(d3d12Lib);
        return out;
    }
    using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
    auto pCreateDXGIFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(dxgiLib, "CreateDXGIFactory1"));
    auto pD3D12CreateDevice = reinterpret_cast<PFN_D3D12CreateDeviceFn>(GetProcAddress(d3d12Lib, "D3D12CreateDevice"));
    IDXGIFactory4* factory = nullptr;
    if (pCreateDXGIFactory1 && pD3D12CreateDevice &&
        SUCCEEDED(pCreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))) && factory) {
        uint32_t idx = 0;
        for (UINT i = 0;; ++i) {
            IDXGIAdapter1* a = nullptr;
            if (factory->EnumAdapters1(i, &a) == DXGI_ERROR_NOT_FOUND) break;
            DXGI_ADAPTER_DESC1 desc{};
            a->GetDesc1(&desc);
            if (!isSoftwareOrBasicRenderAdapter(desc) &&
                SUCCEEDED(pD3D12CreateDevice(a, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                std::string name = wideToUtf8(desc.Description);
                out.push_back({idx++, name.empty() ? "DirectX 12" : ("DirectX 12: " + name), false});
            }
            a->Release();
        }
        IDXGIAdapter1* warp = nullptr;
        if (SUCCEEDED(factory->EnumWarpAdapter(__uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&warp))) && warp) {
            if (SUCCEEDED(pD3D12CreateDevice(warp, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr))) {
                out.push_back({idx++, "Microsoft WARP", true});
            }
            warp->Release();
        }
        factory->Release();
    }
    FreeLibrary(d3d12Lib);
    FreeLibrary(dxgiLib);
    return out;
}

struct WarpDeviceContext {
    HMODULE dxgiLib = nullptr;
    HMODULE d3d12Lib = nullptr;
    IDXGIFactory4* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    ID3D12Device* dev = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* alloc = nullptr;
    ID3D12GraphicsCommandList* cmd = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE fenceEvent = nullptr;
    uint64_t fenceVal = 1;
    std::string backendName = "Microsoft WARP";

    using PFN_D3D12SerializeRootSignature = HRESULT(WINAPI*)(
        const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);
    PFN_D3D12SerializeRootSignature pSerializeRS = nullptr;

    bool init(std::string& outErr, int32_t targetDeviceIdx = -1, bool forceWarp = false) {
        dxgiLib = LoadLibraryA("dxgi.dll");
        d3d12Lib = LoadLibraryA("d3d12.dll");
        if (!dxgiLib || !d3d12Lib) {
            outErr = "SKIPPED (dxgi/d3d12 DLL unavailable)";
            destroy();
            return false;
        }
        using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
        auto pCreateDXGIFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(dxgiLib, "CreateDXGIFactory1"));
        auto pD3D12CreateDevice = reinterpret_cast<PFN_D3D12CreateDeviceFn>(GetProcAddress(d3d12Lib, "D3D12CreateDevice"));
        pSerializeRS = reinterpret_cast<PFN_D3D12SerializeRootSignature>(GetProcAddress(d3d12Lib, "D3D12SerializeRootSignature"));
        if (!pCreateDXGIFactory1 || !pD3D12CreateDevice || !pSerializeRS) {
            outErr = "SKIPPED (D3D12 entry points missing)";
            destroy();
            return false;
        }
        if (FAILED(pCreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))) || !factory) {
            outErr = "SKIPPED (CreateDXGIFactory1 failed)";
            destroy();
            return false;
        }
        if (!selectD3D12Adapter(factory, pD3D12CreateDevice, targetDeviceIdx, forceWarp, &adapter, backendName, outErr)) {
            destroy();
            return false;
        }
        if (FAILED(pD3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&dev))) || !dev) {
            outErr = "FAILED (D3D12CreateDevice)";
            destroy();
            return false;
        }
        D3D12_COMMAND_QUEUE_DESC qDesc{};
        qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(dev->CreateCommandQueue(&qDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue))) ||
            FAILED(dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&alloc))) ||
            FAILED(dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, nullptr, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&cmd))) ||
            FAILED(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&fence)))) {
            outErr = "FAILED (WARP command objects)";
            destroy();
            return false;
        }
        fenceEvent = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        return true;
    }

    void waitFence() {
        const uint64_t val = fenceVal++;
        queue->Signal(fence, val);
        if (fence->GetCompletedValue() < val) {
            fence->SetEventOnCompletion(val, fenceEvent);
            WaitForSingleObject(fenceEvent, INFINITE);
        }
    }

    void destroy() {
        if (fenceEvent) { CloseHandle(fenceEvent); fenceEvent = nullptr; }
        if (fence) { fence->Release(); fence = nullptr; }
        if (cmd) { cmd->Release(); cmd = nullptr; }
        if (alloc) { alloc->Release(); alloc = nullptr; }
        if (queue) { queue->Release(); queue = nullptr; }
        if (dev) { dev->Release(); dev = nullptr; }
        if (adapter) { adapter->Release(); adapter = nullptr; }
        if (factory) { factory->Release(); factory = nullptr; }
        if (d3d12Lib) { FreeLibrary(d3d12Lib); d3d12Lib = nullptr; }
        if (dxgiLib) { FreeLibrary(dxgiLib); dxgiLib = nullptr; }
    }
};

inline ID3D12Resource* createWarpBuffer(
    ID3D12Device* dev,
    UINT64 size,
    D3D12_HEAP_TYPE heapType,
    D3D12_RESOURCE_FLAGS flags,
    D3D12_RESOURCE_STATES initState) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heapType;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ID3D12Resource* res = nullptr;
    if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, initState, nullptr,
                                            __uuidof(ID3D12Resource), reinterpret_cast<void**>(&res)))) {
        return nullptr;
    }
    return res;
}

inline BenchRunResult runWarpGraphicsBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& vs,
    const std::vector<uint8_t>& hs,
    const std::vector<uint8_t>& ds,
    const std::vector<uint8_t>& gs,
    const std::vector<uint8_t>& ps,
    uint32_t width,
    uint32_t height,
    uint32_t iters,
    int32_t targetDeviceIdx = -1,
    bool forceWarp = false) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Microsoft WARP";
    res.shaderFormat = "DXIL";
    if (spec.hasTexture) {
        res.status = "SKIPPED (textured PS)";
        return res;
    }

    MemorySnapshot memBefore = queryProcessMemory();
    WarpDeviceContext ctx;
    std::string initErr;
    if (!ctx.init(initErr, targetDeviceIdx, forceWarp)) {
        res.status = initErr;
        return res;
    }
    res.backend = ctx.backendName;

    const bool isStateHeavyPS = (spec.id == "state_heavy_ps");

    D3D12_DESCRIPTOR_RANGE srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 2;
    srvRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    params[0].Descriptor = {0, 0};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &srvRange;
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC staticSamp{};
    staticSamp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSamp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSamp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSamp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    staticSamp.MaxLOD = D3D12_FLOAT32_MAX;
    staticSamp.ShaderRegister = 0;
    staticSamp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = isStateHeavyPS ? 2u : 1u;
    rsDesc.pParameters = params;
    rsDesc.NumStaticSamplers = isStateHeavyPS ? 1u : 0u;
    rsDesc.pStaticSamplers = isStateHeavyPS ? &staticSamp : nullptr;
    rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    ID3DBlob* rsBlob = nullptr;
    ctx.pSerializeRS(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    ctx.dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                                 __uuidof(ID3D12RootSignature), reinterpret_cast<void**>(&rootSig));
    rsBlob->Release();

    auto t0 = std::chrono::steady_clock::now();
    D3D12_INPUT_ELEMENT_DESC posLayout = {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0};
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
    bool isVertexMesh = (spec.id == "bench_vertex_mesh");
    bool usesQuadVB = (spec.category == BenchCategory::Pixel);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig;
    psoDesc.VS = {vs.data(), vs.size()};
    if (!hs.empty()) psoDesc.HS = {hs.data(), hs.size()};
    if (!ds.empty()) psoDesc.DS = {ds.data(), ds.size()};
    if (!gs.empty()) psoDesc.GS = {gs.data(), gs.size()};
    psoDesc.PS = {ps.data(), ps.size()};
    if (isVertexMesh) psoDesc.InputLayout = {meshElems, 5};
    else if (isStateHeavyPS) psoDesc.InputLayout = {stateHeavyElems, 2};
    else if (usesQuadVB) psoDesc.InputLayout = {&posLayout, 1};
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.SampleMask = UINT_MAX;
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
    psoDesc.PrimitiveTopologyType = (spec.kind == PipelineKind::GraphicsTessGS)
        ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH
        : (spec.kind == PipelineKind::GraphicsVSGSPS ? D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT : D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE);
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc.Count = isStateHeavyPS ? 4u : 1u;

    ID3D12PipelineState* pso = nullptr;
    HRESULT hr = ctx.dev->CreateGraphicsPipelineState(&psoDesc, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(&pso));
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (WARP Graphics PSO)";
        rootSig->Release();
        ctx.destroy();
        return res;
    }

    uint32_t maxW = (spec.category == BenchCategory::Pixel || isVertexMesh) ? std::max(128u, width * 2) : 128;
    uint32_t maxH = (spec.category == BenchCategory::Pixel || isVertexMesh) ? std::max(128u, height * 2) : 128;

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{};
    rtvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    rtvHeapDesc.NumDescriptors = 1;
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    ctx.dev->CreateDescriptorHeap(&rtvHeapDesc, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&rtvHeap));
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();

    D3D12_HEAP_PROPERTIES defHeap{};
    defHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rtDesc{};
    rtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rtDesc.Width = maxW;
    rtDesc.Height = maxH;
    rtDesc.DepthOrArraySize = 1;
    rtDesc.MipLevels = 1;
    rtDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtDesc.SampleDesc.Count = isStateHeavyPS ? 4u : 1u;
    rtDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    D3D12_CLEAR_VALUE clearVal{};
    clearVal.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    clearVal.Color[3] = 1.0f;
    ID3D12Resource* rtvTex = nullptr;
    ctx.dev->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &rtDesc,
                                     D3D12_RESOURCE_STATE_RENDER_TARGET, &clearVal,
                                     __uuidof(ID3D12Resource), reinterpret_cast<void**>(&rtvTex));
    ctx.dev->CreateRenderTargetView(rtvTex, nullptr, rtvHandle);

    ID3D12Resource* resolveTex = nullptr;
    if (isStateHeavyPS) {
        rtDesc.SampleDesc.Count = 1u;
        ctx.dev->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &rtDesc,
                                         D3D12_RESOURCE_STATE_COPY_SOURCE, &clearVal,
                                         __uuidof(ID3D12Resource), reinterpret_cast<void**>(&resolveTex));
    }

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
        D3D12_RESOURCE_DESC dsDesc{};
        dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dsDesc.Width = maxW;
        dsDesc.Height = maxH;
        dsDesc.DepthOrArraySize = 1;
        dsDesc.MipLevels = 1;
        dsDesc.Format = isStateHeavyPS ? DXGI_FORMAT_D24_UNORM_S8_UINT : DXGI_FORMAT_D32_FLOAT;
        dsDesc.SampleDesc.Count = isStateHeavyPS ? 4u : 1u;
        dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        D3D12_CLEAR_VALUE dsClear{};
        dsClear.Format = dsDesc.Format;
        dsClear.DepthStencil.Depth = 1.0f;
        ctx.dev->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &dsDesc,
                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, &dsClear,
                                         __uuidof(ID3D12Resource), reinterpret_cast<void**>(&dsvTex));
        D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{};
        dsvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_DSV;
        dsvHeapDesc.NumDescriptors = 1;
        ctx.dev->CreateDescriptorHeap(&dsvHeapDesc, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&dsvHeap));
        dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        ctx.dev->CreateDepthStencilView(dsvTex, nullptr, dsvHandle);
    }

    if (isStateHeavyPS) {
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
            ctx.dev->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &td,
                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                             __uuidof(ID3D12Resource), reinterpret_cast<void**>(&tex));
            return tex;
        };
        texBC1 = mkBC(DXGI_FORMAT_BC1_UNORM);
        texBC7 = mkBC(DXGI_FORMAT_BC7_UNORM);
        auto bc1Data = buildTestBC1Texture256();
        auto bc7Data = buildTestBC7Texture256();
        texUpload = createWarpBuffer(ctx.dev, 128 * 1024, D3D12_HEAP_TYPE_UPLOAD,
                                     D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        void* up = nullptr;
        texUpload->Map(0, nullptr, &up);
        std::memcpy(up, bc1Data.data(), bc1Data.size());
        std::memcpy(static_cast<uint8_t*>(up) + 64 * 1024, bc7Data.data(), bc7Data.size());
        texUpload->Unmap(0, nullptr);

        D3D12_TEXTURE_COPY_LOCATION d1{texBC1, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION s1{texUpload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        s1.PlacedFootprint.Offset = 0;
        s1.PlacedFootprint.Footprint = {DXGI_FORMAT_BC1_UNORM, 256, 256, 1, 512};
        ctx.cmd->CopyTextureRegion(&d1, 0, 0, 0, &s1, nullptr);

        D3D12_TEXTURE_COPY_LOCATION d2{texBC7, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
        D3D12_TEXTURE_COPY_LOCATION s2{texUpload, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        s2.PlacedFootprint.Offset = 64 * 1024;
        s2.PlacedFootprint.Footprint = {DXGI_FORMAT_BC7_UNORM, 256, 256, 1, 1024};
        ctx.cmd->CopyTextureRegion(&d2, 0, 0, 0, &s2, nullptr);

        D3D12_RESOURCE_BARRIER b[2]{};
        b[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[0].Transition = {texBC1, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
        b[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b[1].Transition = {texBC7, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};
        ctx.cmd->ResourceBarrier(2, b);
        ctx.cmd->Close();
        ID3D12CommandList* initLists[] = {ctx.cmd};
        ctx.queue->ExecuteCommandLists(1, initLists);
        ctx.waitFence();
        ctx.alloc->Reset();
        ctx.cmd->Reset(ctx.alloc, nullptr);

        D3D12_DESCRIPTOR_HEAP_DESC srvHD{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        ctx.dev->CreateDescriptorHeap(&srvHD, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&srvHeap));
        UINT inc = ctx.dev->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        D3D12_CPU_DESCRIPTOR_HANDLE h0 = srvHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
        srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
        srvDesc.Texture2D.MipLevels = 1;
        srvDesc.Format = DXGI_FORMAT_BC1_UNORM;
        ctx.dev->CreateShaderResourceView(texBC1, &srvDesc, h0);
        h0.ptr += inc;
        srvDesc.Format = DXGI_FORMAT_BC7_UNORM;
        ctx.dev->CreateShaderResourceView(texBC7, &srvDesc, h0);
    }

    if (isVertexMesh) {

        for (int m = 0; m < 3; ++m) {
            uint32_t g = 0, mrw = 0, mrh = 0;
            getMeshBenchGridAndDims(m, width, height, g, mrw, mrh);
            meshScenes[m] = buildMeshBenchScene(g, g);
            size_t vbBytes = meshScenes[m].vertices.size() * sizeof(MeshBenchVertex);
            size_t ibBytes = meshScenes[m].indices.size() * sizeof(uint32_t);
            meshVBs[m] = createWarpBuffer(ctx.dev, vbBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            meshIBs[m] = createWarpBuffer(ctx.dev, ibBytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
            void* p = nullptr;
            meshVBs[m]->Map(0, nullptr, &p);
            std::memcpy(p, meshScenes[m].vertices.data(), vbBytes);
            meshVBs[m]->Unmap(0, nullptr);
            meshIBs[m]->Map(0, nullptr, &p);
            std::memcpy(p, meshScenes[m].indices.data(), ibBytes);
            meshIBs[m]->Unmap(0, nullptr);
        }
    }

    UINT64 readbackPitch = (static_cast<UINT64>(maxW) * 4 + 255) & ~255ull;
    ID3D12Resource* rbBuf = createWarpBuffer(ctx.dev, readbackPitch * maxH,
                                             D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_COPY_DEST);
    ID3D12Resource* cbBuf = createWarpBuffer(ctx.dev, isVertexMesh ? sizeof(MeshBenchConstants) : 256,
                                             D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* vbBuf = nullptr;
    if (usesQuadVB) {
        if (isStateHeavyPS) {
            const float quadVerts[30] = {
                -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
                 1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
                 1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
                 1.0f,  1.0f, 0.0f, 1.0f, 0.0f,
                -1.0f,  1.0f, 0.0f, 0.0f, 0.0f,
            };
            vbBuf = createWarpBuffer(ctx.dev, sizeof(quadVerts),
                                     D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ);
            void* p = nullptr;
            vbBuf->Map(0, nullptr, &p);
            std::memcpy(p, quadVerts, sizeof(quadVerts));
            vbBuf->Unmap(0, nullptr);
        } else {
            const float quadVerts[18] = {
                -1.0f, -1.0f, 0.0f,  1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f,
                -1.0f, -1.0f, 0.0f,  1.0f,  1.0f, 0.0f, -1.0f,  1.0f, 0.0f,
            };
            vbBuf = createWarpBuffer(ctx.dev, sizeof(quadVerts),
                                     D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                                     D3D12_RESOURCE_STATE_GENERIC_READ);
            void* p = nullptr;
            vbBuf->Map(0, nullptr, &p);
            std::memcpy(p, quadVerts, sizeof(quadVerts));
            vbBuf->Unmap(0, nullptr);
        }
    }

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
        ctx.cmd->SetGraphicsRootSignature(rootSig);
        ctx.cmd->SetPipelineState(pso);
        ctx.cmd->SetGraphicsRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        if (isStateHeavyPS) {
            ID3D12DescriptorHeap* heaps[] = {srvHeap};
            ctx.cmd->SetDescriptorHeaps(1, heaps);
            ctx.cmd->SetGraphicsRootDescriptorTable(1, srvHeap->GetGPUDescriptorHandleForHeapStart());
        }
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(rw), static_cast<float>(rh), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(rw), static_cast<LONG>(rh)};
        ctx.cmd->RSSetViewports(1, &vp);
        ctx.cmd->RSSetScissorRects(1, &sc);
        ctx.cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, (isVertexMesh || isStateHeavyPS) ? &dsvHandle : nullptr);
        const float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        ctx.cmd->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);

        if (isVertexMesh) {
            ctx.cmd->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            D3D12_VERTEX_BUFFER_VIEW vbv{
                meshVBs[mode]->GetGPUVirtualAddress(),
                static_cast<UINT>(meshScenes[mode].vertices.size() * sizeof(MeshBenchVertex)),
                sizeof(MeshBenchVertex)};
            D3D12_INDEX_BUFFER_VIEW ibv{
                meshIBs[mode]->GetGPUVirtualAddress(),
                static_cast<UINT>(meshScenes[mode].indices.size() * sizeof(uint32_t)),
                DXGI_FORMAT_R32_UINT};
            ctx.cmd->IASetVertexBuffers(0, 1, &vbv);
            ctx.cmd->IASetIndexBuffer(&ibv);
            ctx.cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.cmd->DrawIndexedInstanced(static_cast<UINT>(meshScenes[mode].indices.size()), 1, 0, 0, 0);
        } else if (isStateHeavyPS) {
            ctx.cmd->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
            D3D12_VERTEX_BUFFER_VIEW vbv{vbBuf->GetGPUVirtualAddress(), 30 * sizeof(float), 5 * sizeof(float)};
            ctx.cmd->IASetVertexBuffers(0, 1, &vbv);
            ctx.cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            ctx.cmd->DrawInstanced(vertCount, layerCount, 0, 0);
            D3D12_RESOURCE_BARRIER rb[2]{};
            rb[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            rb[0].Transition = {rtvTex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE};
            rb[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            rb[1].Transition = {resolveTex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RESOLVE_DEST};
            ctx.cmd->ResourceBarrier(2, rb);
            ctx.cmd->ResolveSubresource(resolveTex, 0, rtvTex, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            rb[0].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_SOURCE;
            rb[0].Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
            rb[1].Transition.StateBefore = D3D12_RESOURCE_STATE_RESOLVE_DEST;
            rb[1].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            ctx.cmd->ResourceBarrier(2, rb);
        } else {
            if (usesQuadVB) {
                D3D12_VERTEX_BUFFER_VIEW vbv{vbBuf->GetGPUVirtualAddress(), 18 * sizeof(float), 3 * sizeof(float)};
                ctx.cmd->IASetVertexBuffers(0, 1, &vbv);
                ctx.cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            } else if (spec.kind == PipelineKind::GraphicsTessGS) {
                ctx.cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);
            } else if (spec.kind == PipelineKind::GraphicsVSGSPS) {
                ctx.cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_POINTLIST);
            } else {
                ctx.cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            }
            ctx.cmd->DrawInstanced(vertCount, 1, 0, 0);
        }
        ctx.cmd->Close();
        ID3D12CommandList* lists[] = {ctx.cmd};
        ctx.queue->ExecuteCommandLists(1, lists);
        ctx.waitFence();
        ctx.alloc->Reset();
        ctx.cmd->Reset(ctx.alloc, nullptr);
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
            rw = 128; rh = 128;
            vc = (mode == 0) ? 64 * 3 : (mode == 1 ? 512 * 3 : 2048 * 3);
        }
    };

    uint32_t rw = 128, rh = 128, vc = 6;
    getGfxParams(1, rw, rh, vc);

    res.firstDispatchMs = runDraw(1, rw, rh, vc);
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runDraw(1, rw, rh, vc));
    }

    // Copy RTV to readback buffer for checksum
    ID3D12Resource* srcReadTex = isStateHeavyPS ? resolveTex : rtvTex;
    if (!isStateHeavyPS) {
        D3D12_RESOURCE_BARRIER toSrc{};
        toSrc.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toSrc.Transition = {rtvTex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE};
        ctx.cmd->ResourceBarrier(1, &toSrc);
    }
    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = rbBuf;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, rw, rh, 1, static_cast<UINT>((rw * 4 + 255) & ~255u)};
    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = srcReadTex;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_BOX box{0, 0, 0, rw, rh, 1};
    ctx.cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, &box);
    if (!isStateHeavyPS) {
        D3D12_RESOURCE_BARRIER toRt{};
        toRt.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        toRt.Transition = {rtvTex, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET};
        ctx.cmd->ResourceBarrier(1, &toRt);
    }
    ctx.cmd->Close();
    ID3D12CommandList* copyLists[] = {ctx.cmd};
    ctx.queue->ExecuteCommandLists(1, copyLists);
    ctx.waitFence();
    ctx.alloc->Reset();
    ctx.cmd->Reset(ctx.alloc, nullptr);

    const uint8_t* px = nullptr;
    UINT pitch = dstLoc.PlacedFootprint.Footprint.RowPitch;
    D3D12_RANGE rbRange{0, static_cast<SIZE_T>(pitch) * rh};
    if (SUCCEEDED(rbBuf->Map(0, &rbRange, reinterpret_cast<void**>(const_cast<uint8_t**>(&px)))) && px) {
        double sum = 0.0;
        for (uint32_t y = 0; y < rh; ++y) {
            const uint8_t* row = px + static_cast<size_t>(y) * pitch;
            for (uint32_t x = 0; x < rw * 4; ++x) sum += row[x];
        }
        rbBuf->Unmap(0, nullptr);
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
    if (dsvHeap) dsvHeap->Release();
    if (dsvTex) dsvTex->Release();
    if (vbBuf) vbBuf->Release();
    cbBuf->Release();
    rbBuf->Release();
    rtvTex->Release();
    rtvHeap->Release();
    pso->Release();
    rootSig->Release();
    ctx.destroy();
    return res;
}

inline BenchRunResult runWarpRaytracingBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t iters,
    int32_t targetDeviceIdx = -1,
    bool forceWarp = false) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Microsoft WARP";
    res.shaderFormat = "DXIL";
    if (spec.hasTexture) {
        res.status = "SKIPPED (textured PS)";
        return res;
    }

    MemorySnapshot memBefore = queryProcessMemory();
    WarpDeviceContext ctx;
    std::string initErr;
    if (!ctx.init(initErr, targetDeviceIdx, forceWarp)) {
        res.status = initErr;
        return res;
    }
    res.backend = ctx.backendName;

    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5{};
    if (FAILED(ctx.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5))) ||
        opts5.RaytracingTier == D3D12_RAYTRACING_TIER_NOT_SUPPORTED) {
        res.status = "SKIPPED (WARP raytracing unsupported)";
        ctx.destroy();
        return res;
    }

    ID3D12Device5* dev5 = nullptr;
    ID3D12GraphicsCommandList4* cmd4 = nullptr;
    if (FAILED(ctx.dev->QueryInterface(__uuidof(ID3D12Device5), reinterpret_cast<void**>(&dev5))) ||
        FAILED(ctx.cmd->QueryInterface(__uuidof(ID3D12GraphicsCommandList4), reinterpret_cast<void**>(&cmd4))) ||
        !dev5 || !cmd4) {
        if (cmd4) cmd4->Release();
        if (dev5) dev5->Release();
        res.status = "SKIPPED (WARP ID3D12Device5 unavailable)";
        ctx.destroy();
        return res;
    }

    D3D12_ROOT_PARAMETER params[2]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[0].Descriptor = {0, 0};
    params[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[1].Descriptor = {0, 0};
    params[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = params;
    ID3DBlob* rsBlob = nullptr;
    ctx.pSerializeRS(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev5->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(),
                              __uuidof(ID3D12RootSignature), reinterpret_cast<void**>(&rootSig));
    rsBlob->Release();

    const bool isHeavyRT = (spec.id == "rt_inline_heavy" || spec.id == "rt_pipeline_heavy");

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
    ID3D12Resource* vb = createWarpBuffer(dev5, vbBytes, D3D12_HEAP_TYPE_UPLOAD,
                                          D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
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
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&blasIn, &blasPre);
    ID3D12Resource* blasBuf = createWarpBuffer(dev5, std::max<UINT64>(65536, blasPre.ResultDataMaxSizeInBytes),
                                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    ID3D12Resource* scratch = createWarpBuffer(dev5, std::max<UINT64>(65536, blasPre.ScratchDataSizeInBytes),
                                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    ID3D12Resource* instBuf = nullptr;
    uint32_t numInstances = 1u;
    if (isHeavyRT) {
        auto insts = buildHeavyRTTlasInstances<D3D12_RAYTRACING_INSTANCE_DESC>(blasBuf->GetGPUVirtualAddress());
        numInstances = static_cast<uint32_t>(insts.size());
        instBuf = createWarpBuffer(dev5, insts.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC),
                                   D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
        instBuf->Map(0, nullptr, &mapped);
        std::memcpy(mapped, insts.data(), insts.size() * sizeof(D3D12_RAYTRACING_INSTANCE_DESC));
        instBuf->Unmap(0, nullptr);
    } else {
        D3D12_RAYTRACING_INSTANCE_DESC inst{};
        inst.Transform[0][0] = 1.0f; inst.Transform[1][1] = 1.0f; inst.Transform[2][2] = 1.0f;
        inst.InstanceID = 42;
        inst.InstanceMask = 0xFF;
        inst.AccelerationStructure = blasBuf->GetGPUVirtualAddress();
        instBuf = createWarpBuffer(dev5, sizeof(inst), D3D12_HEAP_TYPE_UPLOAD,
                                   D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
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
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&tlasIn, &tlasPre);
    ID3D12Resource* tlasBuf = createWarpBuffer(dev5, std::max<UINT64>(65536, tlasPre.ResultDataMaxSizeInBytes),
                                               D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);

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
    ID3D12CommandList* asLists[] = {ctx.cmd};
    ctx.queue->ExecuteCommandLists(1, asLists);
    ctx.waitFence();
    ctx.alloc->Reset();
    ctx.cmd->Reset(ctx.alloc, nullptr);

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
        hr = dev5->CreateComputePipelineState(&psoDesc, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(&pso));
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
        hr = dev5->CreateStateObject(&soDesc, __uuidof(ID3D12StateObject), reinterpret_cast<void**>(&stateObj));
        if (SUCCEEDED(hr) && stateObj) {
            ID3D12StateObjectProperties* sop = nullptr;
            stateObj->QueryInterface(__uuidof(ID3D12StateObjectProperties), reinterpret_cast<void**>(&sop));
            shaderTable = createWarpBuffer(dev5, 3 * 64, D3D12_HEAP_TYPE_UPLOAD,
                                           D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
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
        res.status = "FAILED (WARP RT Pipeline)";
        if (shaderTable) shaderTable->Release();
        if (stateObj) stateObj->Release();
        if (pso) pso->Release();
        tlasBuf->Release();
        instBuf->Release();
        scratch->Release();
        blasBuf->Release();
        vb->Release();
        rootSig->Release();
        cmd4->Release();
        dev5->Release();
        ctx.destroy();
        return res;
    }

    ID3D12Resource* outBuf = createWarpBuffer(dev5, 4 * 1024 * 1024, D3D12_HEAP_TYPE_DEFAULT,
                                              D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* rbBuf = createWarpBuffer(dev5, 4 * 1024 * 1024, D3D12_HEAP_TYPE_READBACK,
                                             D3D12_RESOURCE_FLAG_NONE,
                                             D3D12_RESOURCE_STATE_COPY_DEST);

    auto runRT = [&](uint32_t w, uint32_t h) -> double {
        auto s0 = std::chrono::steady_clock::now();
        ctx.cmd->SetComputeRootSignature(rootSig);
        if (spec.kind == PipelineKind::RayTracingInline) {
            ctx.cmd->SetPipelineState(pso);
            ctx.cmd->SetComputeRootShaderResourceView(0, tlasBuf->GetGPUVirtualAddress());
            ctx.cmd->SetComputeRootUnorderedAccessView(1, outBuf->GetGPUVirtualAddress());
            if (isHeavyRT) {
                ctx.cmd->Dispatch((w + 7) / 8, (h + 7) / 8, 1);
            } else {
                ctx.cmd->Dispatch((w + 3) / 4, h, 1);
            }
        } else {
            cmd4->SetPipelineState1(stateObj);
            ctx.cmd->SetComputeRootShaderResourceView(0, tlasBuf->GetGPUVirtualAddress());
            ctx.cmd->SetComputeRootUnorderedAccessView(1, outBuf->GetGPUVirtualAddress());
            D3D12_DISPATCH_RAYS_DESC dr{};
            dr.RayGenerationShaderRecord = {rgenAddr, 64};
            dr.MissShaderTable = {missAddr, 64, missStride};
            dr.HitGroupTable = {hitAddr, 64, hitStride};
            dr.Width = w;
            dr.Height = h;
            dr.Depth = 1;
            cmd4->DispatchRays(&dr);
        }
        ctx.cmd->Close();
        ID3D12CommandList* lists[] = {ctx.cmd};
        ctx.queue->ExecuteCommandLists(1, lists);
        ctx.waitFence();
        ctx.alloc->Reset();
        ctx.cmd->Reset(ctx.alloc, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    res.firstDispatchMs = runRT(width, height);
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runRT(width, height));
    }

    D3D12_RESOURCE_BARRIER toCopy{};
    toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    toCopy.Transition = {outBuf, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
    ctx.cmd->ResourceBarrier(1, &toCopy);
    ctx.cmd->CopyResource(rbBuf, outBuf);
    D3D12_RESOURCE_BARRIER toUav = toCopy;
    toUav.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    toUav.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ctx.cmd->ResourceBarrier(1, &toUav);
    ctx.cmd->Close();
    ID3D12CommandList* copyLists[] = {ctx.cmd};
    ctx.queue->ExecuteCommandLists(1, copyLists);
    ctx.waitFence();
    ctx.alloc->Reset();
    ctx.cmd->Reset(ctx.alloc, nullptr);

    const size_t checksumCount = std::min<size_t>(
        1024u * 1024u,
        (spec.kind == PipelineKind::RayTracingInline) ? static_cast<size_t>(width) * 4u : static_cast<size_t>(width));
    const uint32_t* up = nullptr;
    D3D12_RANGE rbRange{0, checksumCount * sizeof(uint32_t)};
    if (SUCCEEDED(rbBuf->Map(0, &rbRange, reinterpret_cast<void**>(const_cast<uint32_t**>(&up)))) && up) {
        double sum = 0.0;
        for (size_t i = 0; i < checksumCount; ++i) sum += static_cast<double>(up[i] & 0xFFFFu);
        rbBuf->Unmap(0, nullptr);
        res.checksum = sum;
    }

    res.shortJobMs = runRT(64, 64);
    res.longJobMs = runRT(width * 2, height * 2);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    rbBuf->Release();
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
    cmd4->Release();
    dev5->Release();
    ctx.destroy();
    return res;
}

inline BenchRunResult runWarpMeshBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& msDxil,
    const std::vector<uint8_t>& psDxil,
    uint32_t iters,
    int32_t devIdx = -1,
    bool forceWarp = true,
    const std::vector<uint8_t>& asDxil = {}) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = forceWarp ? "Microsoft WARP" : "D3D12 GPU";
    res.shaderFormat = "DXIL";

    MemorySnapshot memBefore = queryProcessMemory();
    WarpDeviceContext ctx;
    std::string err;
    if (!ctx.init(err, devIdx, forceWarp)) {
        res.status = err;
        return res;
    }
    res.backend = ctx.backendName;

    D3D12_FEATURE_DATA_D3D12_OPTIONS7 opts7{};
    if (FAILED(ctx.dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &opts7, sizeof(opts7))) ||
        opts7.MeshShaderTier == D3D12_MESH_SHADER_TIER_NOT_SUPPORTED) {
        res.status = "UNSUPPORTED (no Mesh Shader Tier)";
        ctx.destroy();
        return res;
    }

    ID3D12Device2* dev2 = nullptr;
    ID3D12GraphicsCommandList6* cmd6 = nullptr;
    if (FAILED(ctx.dev->QueryInterface(__uuidof(ID3D12Device2), reinterpret_cast<void**>(&dev2))) || !dev2 ||
        FAILED(ctx.cmd->QueryInterface(__uuidof(ID3D12GraphicsCommandList6), reinterpret_cast<void**>(&cmd6))) || !cmd6) {
        res.status = "UNSUPPORTED (no ID3D12Device2/CmdList6)";
        if (cmd6) cmd6->Release();
        if (dev2) dev2->Release();
        ctx.destroy();
        return res;
    }

    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    param.Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &param;
    ID3DBlob* rsBlob = nullptr;
    ID3D12RootSignature* rootSig = nullptr;
    if (FAILED(ctx.pSerializeRS(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr)) ||
        !rsBlob ||
        FAILED(ctx.dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), __uuidof(ID3D12RootSignature), reinterpret_cast<void**>(&rootSig)))) {
        if (rsBlob) rsBlob->Release();
        res.status = "FAILED (RootSignature)";
        cmd6->Release();
        dev2->Release();
        ctx.destroy();
        return res;
    }
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
    HRESULT hr = dev2->CreatePipelineState(&streamDesc, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(&pso));
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (CreatePipelineState Mesh)";
        rootSig->Release();
        cmd6->Release();
        dev2->Release();
        ctx.destroy();
        return res;
    }

    const uint32_t rw = isCullMesh ? 256u : 128u;
    const uint32_t rh = isCullMesh ? 256u : 128u;
    D3D12_HEAP_PROPERTIES defHeap{};
    defHeap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC rtDesc{};
    rtDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    rtDesc.Width = rw;
    rtDesc.Height = rh;
    rtDesc.DepthOrArraySize = 1;
    rtDesc.MipLevels = 1;
    rtDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    rtDesc.SampleDesc.Count = 1;
    rtDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    ID3D12Resource* rtvTex = nullptr;
    ctx.dev->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &rtDesc,
                                     D3D12_RESOURCE_STATE_COMMON, nullptr,
                                     __uuidof(ID3D12Resource), reinterpret_cast<void**>(&rtvTex));
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ID3D12DescriptorHeap* rtvHeap = nullptr;
    ctx.dev->CreateDescriptorHeap(&rtvHD, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&rtvHeap));
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    ctx.dev->CreateRenderTargetView(rtvTex, nullptr, rtvHandle);

    ID3D12Resource* dsvTex = nullptr;
    ID3D12DescriptorHeap* dsvHeap = nullptr;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle{};
    if (isCullMesh) {
        D3D12_RESOURCE_DESC dsDesc{};
        dsDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        dsDesc.Width = rw;
        dsDesc.Height = rh;
        dsDesc.DepthOrArraySize = 1;
        dsDesc.MipLevels = 1;
        dsDesc.Format = DXGI_FORMAT_D32_FLOAT;
        dsDesc.SampleDesc.Count = 1;
        dsDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        ctx.dev->CreateCommittedResource(&defHeap, D3D12_HEAP_FLAG_NONE, &dsDesc,
                                         D3D12_RESOURCE_STATE_DEPTH_WRITE, nullptr,
                                         __uuidof(ID3D12Resource), reinterpret_cast<void**>(&dsvTex));
        D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        ctx.dev->CreateDescriptorHeap(&dsvHD, __uuidof(ID3D12DescriptorHeap), reinterpret_cast<void**>(&dsvHeap));
        dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        ctx.dev->CreateDepthStencilView(dsvTex, nullptr, dsvHandle);
    }

    ID3D12Resource* cbBuf = createWarpBuffer(ctx.dev, 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* rbBuf = createWarpBuffer(ctx.dev, static_cast<size_t>(rw) * rh * 4, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    auto runMesh = [&](uint32_t groups, uint32_t gridX = 32, uint32_t layers = 4) -> double {
        void* p = nullptr;
        if (SUCCEEDED(cbBuf->Map(0, nullptr, &p)) && p) {
            if (isCullMesh) {
                struct { float scale; float time; uint32_t gridX; uint32_t layerCount; } cb{1.0f, 0.25f, gridX, layers};
                std::memcpy(p, &cb, sizeof(cb));
            } else {
                struct { float scale; float time; uint32_t groupCount; float pad; } cb{0.05f, 0.25f, groups, 0.0f};
                std::memcpy(p, &cb, sizeof(cb));
            }
            cbBuf->Unmap(0, nullptr);
        }

        auto s0 = std::chrono::steady_clock::now();
        D3D12_RESOURCE_BARRIER b0{};
        b0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b0.Transition.pResource = rtvTex;
        b0.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
        b0.Transition.StateAfter = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b0.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        ctx.cmd->ResourceBarrier(1, &b0);

        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(rw), static_cast<float>(rh), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(rw), static_cast<LONG>(rh)};
        const float clearColor[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        ctx.cmd->SetGraphicsRootSignature(rootSig);
        ctx.cmd->SetPipelineState(pso);
        ctx.cmd->SetGraphicsRootConstantBufferView(0, cbBuf->GetGPUVirtualAddress());
        ctx.cmd->RSSetViewports(1, &vp);
        ctx.cmd->RSSetScissorRects(1, &sc);
        ctx.cmd->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
        if (isCullMesh) {
            ctx.cmd->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
            ctx.cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
        } else {
            ctx.cmd->OMSetRenderTargets(1, &rtvHandle, FALSE, nullptr);
        }
        cmd6->DispatchMesh(groups, 1, 1);

        D3D12_RESOURCE_BARRIER b1 = b0;
        b1.Transition.StateBefore = D3D12_RESOURCE_STATE_RENDER_TARGET;
        b1.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
        ctx.cmd->ResourceBarrier(1, &b1);
        ctx.cmd->Close();

        ID3D12CommandList* lists[] = {ctx.cmd};
        ctx.queue->ExecuteCommandLists(1, lists);
        ctx.waitFence();
        ctx.alloc->Reset();
        ctx.cmd->Reset(ctx.alloc, nullptr);
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    const uint32_t steadyGroups = isCullMesh ? 64u : 128u;
    const uint32_t shortGroups = isCullMesh ? 8u : 16u;
    const uint32_t longGroups = isCullMesh ? 256u : 512u;

    res.firstDispatchMs = runMesh(steadyGroups, 32, 4);
    runMesh(steadyGroups, 32, 4);

    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runMesh(steadyGroups, 32, 4));
    }

    D3D12_RESOURCE_BARRIER rb0{};
    rb0.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    rb0.Transition.pResource = rtvTex;
    rb0.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    rb0.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    rb0.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    ctx.cmd->ResourceBarrier(1, &rb0);
    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = rbBuf;
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dstLoc.PlacedFootprint.Footprint.Width = rw;
    dstLoc.PlacedFootprint.Footprint.Height = rh;
    dstLoc.PlacedFootprint.Footprint.Depth = 1;
    dstLoc.PlacedFootprint.Footprint.RowPitch = rw * 4;
    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = rtvTex;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    ctx.cmd->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    D3D12_RESOURCE_BARRIER rb1 = rb0;
    rb1.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
    rb1.Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    ctx.cmd->ResourceBarrier(1, &rb1);
    ctx.cmd->Close();
    ID3D12CommandList* copyLists[] = {ctx.cmd};
    ctx.queue->ExecuteCommandLists(1, copyLists);
    ctx.waitFence();
    ctx.alloc->Reset();
    ctx.cmd->Reset(ctx.alloc, nullptr);

    const uint8_t* px = nullptr;
    D3D12_RANGE range{0, static_cast<SIZE_T>(rw) * rh * 4};
    if (SUCCEEDED(rbBuf->Map(0, &range, reinterpret_cast<void**>(const_cast<uint8_t**>(&px)))) && px) {
        double sum = 0.0;
        for (size_t i = 0; i < static_cast<size_t>(rw) * rh * 4; ++i) sum += px[i];
        rbBuf->Unmap(0, nullptr);
        res.checksum = sum;
    }

    res.shortJobMs = runMesh(shortGroups, 16, 2);
    res.longJobMs = runMesh(longGroups, 64, 4);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    rbBuf->Release();
    cbBuf->Release();
    if (dsvHeap) dsvHeap->Release();
    if (dsvTex) dsvTex->Release();
    rtvHeap->Release();
    rtvTex->Release();
    pso->Release();
    rootSig->Release();
    cmd6->Release();
    dev2->Release();
    ctx.destroy();
    return res;
}

} // namespace nacelle_bench
#endif // _WIN32
