#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>

#include "common/bench_common.h"
#include "workloads/gdeflate.h"
#include "common/bench_memory.h"
#include "runners.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(NACELLE_D3D12_SDK_VERSION) && NACELLE_D3D12_SDK_VERSION > 0
extern "C" {
__declspec(dllexport) extern const unsigned int D3D12SDKVersion = NACELLE_D3D12_SDK_VERSION;
__declspec(dllexport) extern const char* D3D12SDKPath = ".\\D3D12\\";
}
#endif

using namespace nacelle_bench;

namespace {

BenchRunResult runWarpComputeBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters,
    int32_t targetDeviceIdx = -1,
    bool forceWarp = false) {
    BenchRunResult res;
    res.benchId = spec.id;
    res.benchName = spec.name;
    res.category = categoryName(spec.category);
    res.backend = "Microsoft WARP";
    res.shaderFormat = "DXIL";

    MemorySnapshot memBefore = queryProcessMemory();

    HMODULE dxgiLib = LoadLibraryA("dxgi.dll");
    HMODULE d3d12Lib = LoadLibraryA("d3d12.dll");
    if (!dxgiLib || !d3d12Lib) {
        if (dxgiLib) FreeLibrary(dxgiLib);
        if (d3d12Lib) FreeLibrary(d3d12Lib);
        res.status = "SKIPPED (dxgi/d3d12 DLL unavailable)";
        return res;
    }

    using PFN_CreateDXGIFactory1 = HRESULT(WINAPI*)(REFIID, void**);
    using PFN_D3D12SerializeRootSignature = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

    auto pCreateDXGIFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(dxgiLib, "CreateDXGIFactory1"));
    auto pD3D12CreateDevice = reinterpret_cast<PFN_D3D12CreateDeviceFn>(GetProcAddress(d3d12Lib, "D3D12CreateDevice"));
    auto pSerializeRS = reinterpret_cast<PFN_D3D12SerializeRootSignature>(GetProcAddress(d3d12Lib, "D3D12SerializeRootSignature"));
    if (!pCreateDXGIFactory1 || !pD3D12CreateDevice || !pSerializeRS) {
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        res.status = "SKIPPED (D3D12 entry points missing)";
        return res;
    }

    IDXGIFactory4* factory = nullptr;
    if (FAILED(pCreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void**>(&factory))) || !factory) {
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        res.status = "SKIPPED (CreateDXGIFactory1 failed)";
        return res;
    }

    IDXGIAdapter1* adapter = nullptr;
    std::string selErr;
    if (!selectD3D12Adapter(factory, pD3D12CreateDevice, targetDeviceIdx, forceWarp, &adapter, res.backend, selErr)) {
        factory->Release();
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        res.status = selErr;
        return res;
    }

    ID3D12Device* dev = nullptr;
    HRESULT hr = pD3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&dev));
    adapter->Release();
    factory->Release();
    if (FAILED(hr) || !dev) {
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        res.status = "SKIPPED (D3D12CreateDevice failed)";
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue));

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
    pSerializeRS(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ID3D12RootSignature* rootSig = nullptr;
    dev->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), __uuidof(ID3D12RootSignature), reinterpret_cast<void**>(&rootSig));
    rsBlob->Release();

    auto t0 = std::chrono::steady_clock::now();
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rootSig;
    psoDesc.CS = {dxil.data(), dxil.size()};
    ID3D12PipelineState* pso = nullptr;
    hr = dev->CreateComputePipelineState(&psoDesc, __uuidof(ID3D12PipelineState), reinterpret_cast<void**>(&pso));
    auto t1 = std::chrono::steady_clock::now();
    res.compileMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    MemorySnapshot memAfterCompile = queryProcessMemory();

    if (FAILED(hr) || !pso) {
        res.status = "FAILED (WARP CreateComputePipelineState)";
        rootSig->Release();
        queue->Release();
        dev->Release();
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        return res;
    }

    auto mkBuf = [&](D3D12_HEAP_TYPE heapType, size_t bytes, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES initState) -> ID3D12Resource* {
        D3D12_HEAP_PROPERTIES hp{};
        hp.Type = heapType;
        D3D12_RESOURCE_DESC rd{};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = bytes;
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = flags;
        ID3D12Resource* r = nullptr;
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, initState, nullptr, __uuidof(ID3D12Resource), reinterpret_cast<void**>(&r));
        return r;
    };

    size_t maxOutBytes = 16 * 1024 * 1024;
    if (spec.id == "bench_raymarch") {
        maxOutBytes = std::max<size_t>(maxOutBytes, static_cast<size_t>(width * 2) * (height * 2) * 16);
    }

    ID3D12Resource* cb = mkBuf(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* srv0 = mkBuf(D3D12_HEAP_TYPE_UPLOAD, 8 * 1024 * 1024, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* srv1 = mkBuf(D3D12_HEAP_TYPE_UPLOAD, 4 * 1024 * 1024, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* uav0 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, maxOutBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* uav1 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, maxOutBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* uav2 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, 4 * 1024 * 1024, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* uav3 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, 4 * 1024 * 1024, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* uav4 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, 64 * 1024, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* gdResetUpload = mkBuf(D3D12_HEAP_TYPE_UPLOAD, 256, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* rb = mkBuf(D3D12_HEAP_TYPE_READBACK, maxOutBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    void* ptr = nullptr;
    GDeflateStreamData gdShort, gdStd, gdLong;
    uint32_t gdActiveMode = UINT32_MAX;
    if (spec.id == "gdeflate") {
        gdShort = buildGDeflateStream(4u);
        gdStd = buildGDeflateStream(16u);
        gdLong = buildGDeflateStream(64u);
        gdResetUpload->Map(0, nullptr, &ptr);
        std::memset(ptr, 0, 256);
        std::memcpy(ptr, gdStd.controlWords.data(), gdStd.controlWords.size() * sizeof(uint32_t));
        gdResetUpload->Unmap(0, nullptr);
    }

    if (spec.id == "bench_raymarch") {
        srv0->Map(0, nullptr, &ptr);
        float spheres[16] = {
            0.0f, 0.2f, 0.5f, 0.8f,
            -1.3f, 0.0f, 1.0f, 0.6f,
            1.3f, 0.1f, 0.9f, 0.65f,
            0.0f, 1.1f, 1.5f, 0.5f,
        };
        std::memcpy(ptr, spheres, sizeof(spheres));
        srv0->Unmap(0, nullptr);
    } else if (spec.id == "matmul" || spec.id == "tiled_matmul" || spec.id == "tiled_matmul_large" || spec.id == "linalg_matmul" || spec.id == "coalesced_uniform_cs") {
        const size_t initCount = (spec.id == "tiled_matmul_large" || spec.id == "coalesced_uniform_cs") ? (1024u * 1024u) : (256u * 256u);
        srv0->Map(0, nullptr, &ptr);
        float* a = static_cast<float*>(ptr);
        for (size_t i = 0; i < initCount; ++i) a[i] = static_cast<float>((i % 17) + 1) * 0.01f;
        srv0->Unmap(0, nullptr);
        srv1->Map(0, nullptr, &ptr);
        float* b = static_cast<float*>(ptr);
        for (size_t i = 0; i < initCount; ++i) b[i] = static_cast<float>((i % 13) + 1) * 0.01f;
        srv1->Unmap(0, nullptr);
    } else if (spec.id == "prefix_sum_groupshared") {
        srv0->Map(0, nullptr, &ptr);
        uint32_t* p = static_cast<uint32_t*>(ptr);
        for (size_t i = 0; i < 65536 * 4; ++i) p[i] = static_cast<uint32_t>((i & 63u) + 1u);
        srv0->Unmap(0, nullptr);
    }

    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&alloc));
    ID3D12GraphicsCommandList* cl = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc, pso, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&cl));
    cl->Close();
    ID3D12Fence* fence = nullptr;
    dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), reinterpret_cast<void**>(&fence));
    HANDLE ev = CreateEventA(nullptr, FALSE, FALSE, nullptr);
    uint64_t fenceVal = 1;

    auto runOnce = [&](uint32_t gx, uint32_t gy, uint32_t cbW, uint32_t cbH, uint32_t cbSteps) -> double {
        cb->Map(0, nullptr, &ptr);
        if (spec.id == "bench_raymarch") {
            RaymarchCB rcb{cbW, cbH, cbSteps, 0.0f};
            std::memcpy(ptr, &rcb, sizeof(rcb));
        } else if (spec.id == "matmul" || spec.id == "tiled_matmul" || spec.id == "tiled_matmul_large" || spec.id == "linalg_matmul") {
            MatmulCB mcb{cbW, cbH, cbSteps, 0};
            std::memcpy(ptr, &mcb, sizeof(mcb));
        } else if (spec.id == "ethash_mine") {
            EthashCB ecb{};
            for (uint32_t k = 0; k < 8; ++k) ecb.header[k] = 0x12345678u + k;
            ecb.startNonce[0] = 1u;
            ecb.startNonce[1] = 0u;
            ecb.numDatasetElements = 16384u;
            std::memcpy(ptr, &ecb, sizeof(ecb));
        } else if (spec.id == "coalesced_uniform_cs") {
            CoalescedUniformCB ccb{cbW, cbSteps, 0.75f, 0.25f, {}};
            for (uint32_t k = 0; k < 16; ++k) ccb.weights[k] = 0.25f + 0.03f * static_cast<float>(k);
            std::memcpy(ptr, &ccb, sizeof(ccb));
        }
        cb->Unmap(0, nullptr);

        if (spec.id == "gdeflate") {
            const GDeflateStreamData& gd = (cbW <= 4u) ? gdShort : (cbW <= 16u ? gdStd : gdLong);
            if (gdActiveMode != cbW) {
                srv0->Map(0, nullptr, &ptr);
                std::memcpy(ptr, gd.inputBytes.data(), gd.inputBytes.size());
                srv0->Unmap(0, nullptr);
                gdActiveMode = cbW;
            }
            alloc->Reset();
            cl->Reset(alloc, nullptr);
            D3D12_RESOURCE_BARRIER preBarriers[2]{};
            preBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            preBarriers[0].Transition = {uav0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST};
            preBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            preBarriers[1].Transition = {uav2, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST};
            cl->ResourceBarrier(2, preBarriers);
            cl->CopyBufferRegion(uav0, 0, gdResetUpload, 0, 16);
            cl->CopyBufferRegion(uav2, 0, gdResetUpload, 64, 16);
            D3D12_RESOURCE_BARRIER postBarriers[2]{};
            postBarriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            postBarriers[0].Transition = {uav0, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
            postBarriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            postBarriers[1].Transition = {uav2, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS};
            cl->ResourceBarrier(2, postBarriers);
            cl->Close();
            ID3D12CommandList* prepLists[] = {cl};
            queue->ExecuteCommandLists(1, prepLists);
            queue->Signal(fence, fenceVal);
            if (fence->GetCompletedValue() < fenceVal) {
                fence->SetEventOnCompletion(fenceVal, ev);
                WaitForSingleObject(ev, INFINITE);
            }
            ++fenceVal;
        }

        auto s0 = std::chrono::steady_clock::now();
        alloc->Reset();
        cl->Reset(alloc, pso);
        cl->SetComputeRootSignature(rootSig);
        cl->SetComputeRootConstantBufferView(0, cb->GetGPUVirtualAddress());
        cl->SetComputeRootShaderResourceView(1, srv0->GetGPUVirtualAddress());
        cl->SetComputeRootShaderResourceView(2, srv1->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(3, uav0->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(4, uav1->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(5, uav2->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(6, uav3->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(7, uav4->GetGPUVirtualAddress());
        cl->Dispatch(gx, gy, 1);
        cl->Close();
        ID3D12CommandList* lists[] = {cl};
        queue->ExecuteCommandLists(1, lists);
        queue->Signal(fence, fenceVal);
        if (fence->GetCompletedValue() < fenceVal) {
            fence->SetEventOnCompletion(fenceVal, ev);
            WaitForSingleObject(ev, INFINITE);
        }
        ++fenceVal;
        auto s1 = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(s1 - s0).count();
    };

    auto getDims = [&](int mode, uint32_t& gx, uint32_t& gy, uint32_t& cbW, uint32_t& cbH, uint32_t& cbSteps) {
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
            gx = (mode == 0) ? 64 : (mode == 1 ? 512 : 4096);
            gy = 1;
            cbW = gx * 64; cbH = 1; cbSteps = 1;
        }
    };

    uint32_t gx = 1, gy = 1, cbW = 1, cbH = 1, cbSteps = 1;
    getDims(1, gx, gy, cbW, cbH, cbSteps);

    res.firstDispatchMs = runOnce(gx, gy, cbW, cbH, cbSteps);
    std::vector<double> samples;
    samples.reserve(iters);
    for (uint32_t i = 0; i < iters; ++i) {
        samples.push_back(runOnce(gx, gy, cbW, cbH, cbSteps));
    }

    if (spec.id == "gdeflate" || spec.id == "coalesced_uniform_cs") {
        ID3D12Resource* srcUav = (spec.id == "gdeflate") ? uav1 : uav0;
        alloc->Reset();
        cl->Reset(alloc, nullptr);
        D3D12_RESOURCE_BARRIER b{};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition = {srcUav, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE};
        cl->ResourceBarrier(1, &b);
        cl->CopyBufferRegion(rb, 0, srcUav, 0, 4096 * sizeof(uint32_t));
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        cl->ResourceBarrier(1, &b);
        cl->Close();
        ID3D12CommandList* rbLists[] = {cl};
        queue->ExecuteCommandLists(1, rbLists);
        queue->Signal(fence, fenceVal);
        if (fence->GetCompletedValue() < fenceVal) {
            fence->SetEventOnCompletion(fenceVal, ev);
            WaitForSingleObject(ev, INFINITE);
        }
        ++fenceVal;
        void* rbPtr = nullptr;
        rb->Map(0, nullptr, &rbPtr);
        double sum = 0.0;
        if (spec.id == "gdeflate") {
            const uint32_t* outPtr = static_cast<const uint32_t*>(rbPtr);
            for (size_t i = 0; i < 4096; ++i) sum += static_cast<double>(outPtr[i] & 0xFFFFu);
        } else {
            const float* outPtr = static_cast<const float*>(rbPtr);
            for (size_t i = 0; i < 4096; ++i) sum += outPtr[i];
        }
        rb->Unmap(0, nullptr);
        res.checksum = sum;
    }

    getDims(0, gx, gy, cbW, cbH, cbSteps);
    res.shortJobMs = runOnce(gx, gy, cbW, cbH, cbSteps);
    getDims(2, gx, gy, cbW, cbH, cbSteps);
    res.longJobMs = runOnce(gx, gy, cbW, cbH, cbSteps);

    finalizeTimingAndMemory(res, memBefore, memAfterCompile, samples);

    CloseHandle(ev);
    fence->Release();
    cl->Release();
    alloc->Release();
    rb->Release();
    gdResetUpload->Release();
    uav4->Release();
    uav3->Release();
    uav2->Release();
    uav1->Release();
    uav0->Release();
    srv1->Release();
    srv0->Release();
    cb->Release();
    pso->Release();
    rootSig->Release();
    queue->Release();
    dev->Release();
    FreeLibrary(d3d12Lib);
    FreeLibrary(dxgiLib);
    return res;
}

} // namespace

int main(int argc, char** argv) {
    WorkerCliArgs parsed = parseWorkerCliArgs(argc, argv);
    if (parsed.listDevices) {
        auto devs = listD3D12Devices();
        for (const auto& d : devs) {
            std::cout << d.index << "\t" << d.backendName << "\n";
        }
        return 0;
    }

    return runWorkerMain(argc, argv, [](const BenchmarkSpec& spec, const WorkerCliArgs& args) {
        std::vector<BenchRunResult> out;
        int32_t devIdx = -1;
        bool forceWarp = args.forceWarp;
        if (args.singleBackend == "warp" || args.singleBackend == "d3d12:warp") {
            forceWarp = true;
        } else if (args.singleBackend.size() > 6 && args.singleBackend.rfind("d3d12:", 0) == 0) {
            devIdx = std::stoi(args.singleBackend.substr(6));
        }

        if (spec.kind == PipelineKind::Compute) {
            std::vector<uint8_t> dxil;
            if (readBinary(args.shaderDir + "/" + spec.csDxil, dxil)) {
                out.push_back(runWarpComputeBench(spec, dxil, args.width, args.height, args.maxSteps, args.iters, devIdx, forceWarp));
            }
        } else if (spec.kind == PipelineKind::GraphicsVSPS ||
                   spec.kind == PipelineKind::GraphicsVSGSPS ||
                   spec.kind == PipelineKind::GraphicsTessGS) {
            std::vector<uint8_t> vs, hs, ds, gs, ps;
            if (!readBinary(args.shaderDir + "/" + spec.vsDxil, vs)) return out;
            if (!readBinary(args.shaderDir + "/" + spec.psDxil, ps)) return out;
            if (!spec.hsDxil.empty() && !readBinary(args.shaderDir + "/" + spec.hsDxil, hs)) return out;
            if (!spec.dsDxil.empty() && !readBinary(args.shaderDir + "/" + spec.dsDxil, ds)) return out;
            if (!spec.gsDxil.empty() && !readBinary(args.shaderDir + "/" + spec.gsDxil, gs)) return out;
            out.push_back(runWarpGraphicsBench(spec, vs, hs, ds, gs, ps, args.width, args.height, args.iters, devIdx, forceWarp));
        } else if (spec.kind == PipelineKind::GraphicsMSPS) {
            std::vector<uint8_t> as, ms, ps;
            if (!spec.asDxil.empty() && !readBinary(args.shaderDir + "/" + spec.asDxil, as)) return out;
            if (!readBinary(args.shaderDir + "/" + spec.msDxil, ms) ||
                !readBinary(args.shaderDir + "/" + spec.psDxil, ps)) return out;
            out.push_back(runWarpMeshBench(spec, ms, ps, args.iters, devIdx, forceWarp, as));
        } else if (spec.kind == PipelineKind::RayTracingInline ||
                   spec.kind == PipelineKind::RayTracingPipeline) {
            std::vector<uint8_t> rt;
            if (readBinary(args.shaderDir + "/" + spec.rtDxil, rt)) {
                out.push_back(runWarpRaytracingBench(spec, rt, args.width, args.height, args.iters, devIdx, forceWarp));
            }
        }
        return out;
    });
}
