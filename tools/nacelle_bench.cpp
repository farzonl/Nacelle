#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#define NACELLE_POPEN _popen
#define NACELLE_PCLOSE _pclose
#else
#define NACELLE_POPEN popen
#define NACELLE_PCLOSE pclose
#endif

#include "bench_memory.h"
#include "bench_nacelle_runners.h"
#include "bench_types.h"
#include "bench_vulkan_runners.h"

#if defined(__APPLE__)
#include "metal_bench.h"
#endif

namespace {

using namespace nacelle_bench;

bool readBinary(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto sz = ifs.tellg();
    if (sz <= 0) return false;
    out.resize(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    return static_cast<bool>(ifs.read(reinterpret_cast<char*>(out.data()), sz));
}

bool validateSpirvBinary(const std::vector<uint8_t>& spv, std::string& outErr) {
    if (spv.size() < 20 || (spv.size() % 4) != 0) {
        outErr = "invalid SPIR-V byte size";
        return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, spv.data(), 4);
    if (magic != 0x07230203u) {
        outErr = "invalid SPIR-V magic";
        return false;
    }
    return true;
}

std::vector<BenchmarkSpec> buildBenchmarkCatalog() {
    std::vector<BenchmarkSpec> cat;

    auto addCompute = [&](const std::string& id, const std::string& name, const std::string& entry,
                          const std::string& dxil, const std::string& spv) {
        BenchmarkSpec s;
        s.id = id;
        s.name = name;
        s.category = BenchCategory::Compute;
        s.kind = PipelineKind::Compute;
        s.entryOrProfile = entry;
        s.csDxil = dxil;
        s.csSpv = spv;
        cat.push_back(std::move(s));
    };

    addCompute("bench_raymarch", "Raymarcher (ALU+CF)", "main", "bench_raymarch.dxil", "bench_raymarch.spv");
    addCompute("matmul", "MatMul 128x128 (FMA+Mem)", "CSMain", "matmul.dxil", "matmul.spv");
    addCompute("prefix_sum_groupshared", "Prefix Sum (Groupshared)", "CSMain", "prefix_sum_groupshared.dxil", "prefix_sum_groupshared.spv");
    addCompute("divergent_cf_atomics", "Divergent CF + Atomics", "CSMain", "divergent_cf_atomics.dxil", "divergent_cf_atomics.spv");
    addCompute("wave_intrinsics", "Wave Intrinsics (Subgroup)", "CSMain", "wave_intrinsics.dxil", "wave_intrinsics.spv");

    {
        BenchmarkSpec s;
        s.id = "bench_vertex_stress";
        s.name = "Vertex Transform Stress";
        s.category = BenchCategory::Vertex;
        s.kind = PipelineKind::GraphicsVSPS;
        s.vsDxil = "bench_vertex_stress_vs.dxil";
        s.psDxil = "bench_vertex_stress_ps.dxil";
        s.vsSpv = "bench_vertex_stress_vs.spv";
        s.psSpv = "bench_vertex_stress_ps.spv";
        cat.push_back(std::move(s));
    }

    auto addPixel = [&](const std::string& id, const std::string& name,
                        const std::string& psDxil, const std::string& psSpv) {
        BenchmarkSpec s;
        s.id = id;
        s.name = name;
        s.category = BenchCategory::Pixel;
        s.kind = PipelineKind::GraphicsVSPS;
        s.vsDxil = "fullscreen_quad_vs.dxil";
        s.psDxil = psDxil;
        s.vsSpv = "fullscreen_quad_vs.spv";
        s.psSpv = psSpv;
        cat.push_back(std::move(s));
    };

    addPixel("light_cycle", "Light Cycle", "light_cycle.dxil", "light_cycle.spv");
    addPixel("aurora_paint", "Aurora Paint", "aurora_paint.dxil", "aurora_paint.spv");
    addPixel("sunflower_fields", "Sunflower Fields", "sunflower_fields.dxil", "sunflower_fields.spv");
    addPixel("mt_fuji_watercolor", "Mt Fuji Watercolor", "mt_fuji_watercolor.dxil", "mt_fuji_watercolor.spv");
    addPixel("disco_maze", "Disco Maze", "disco_maze.dxil", "disco_maze.spv");

    {
        BenchmarkSpec s;
        s.id = "bench_geometry_stress";
        s.name = "Geometry Star Expansion";
        s.category = BenchCategory::Geometry;
        s.kind = PipelineKind::GraphicsVSGSPS;
        s.vsDxil = "bench_geometry_stress_vs.dxil";
        s.gsDxil = "bench_geometry_stress_gs.dxil";
        s.psDxil = "bench_geometry_stress_ps.dxil";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "tess_point_gs";
        s.name = "Tessellation (VS+HS+DS+GS+PS)";
        s.category = BenchCategory::Tessellation;
        s.kind = PipelineKind::GraphicsTessGS;
        s.vsDxil = "gfx_tess_point_vs.dxil";
        s.hsDxil = "gfx_tess_point_hs.dxil";
        s.dsDxil = "gfx_tess_point_ds.dxil";
        s.gsDxil = "gfx_tess_point_gs.dxil";
        s.psDxil = "gfx_tess_point_ps.dxil";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "raytracing_inline";
        s.name = "RayTracing Inline (RayQuery)";
        s.category = BenchCategory::RayTracing;
        s.kind = PipelineKind::RayTracingInline;
        s.rtDxil = "raytracing_inline_cs.dxil";
        cat.push_back(std::move(s));
    }

    {
        BenchmarkSpec s;
        s.id = "raytracing_pipeline";
        s.name = "RayTracing Pipeline (DispatchRays)";
        s.category = BenchCategory::RayTracing;
        s.kind = PipelineKind::RayTracingPipeline;
        s.rtDxil = "raytracing_pipeline.dxil";
        cat.push_back(std::move(s));
    }

    return cat;
}

#if defined(_WIN32)
BenchRunResult runWarpComputeBench(
    const BenchmarkSpec& spec,
    const std::vector<uint8_t>& dxil,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
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
    using PFN_D3D12CreateDevice = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
    using PFN_D3D12SerializeRootSignature = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**, ID3DBlob**);

    auto pCreateDXGIFactory1 = reinterpret_cast<PFN_CreateDXGIFactory1>(GetProcAddress(dxgiLib, "CreateDXGIFactory1"));
    auto pD3D12CreateDevice = reinterpret_cast<PFN_D3D12CreateDevice>(GetProcAddress(d3d12Lib, "D3D12CreateDevice"));
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

    IDXGIAdapter1* warpAdapter = nullptr;
    if (FAILED(factory->EnumWarpAdapter(__uuidof(IDXGIAdapter1), reinterpret_cast<void**>(&warpAdapter))) || !warpAdapter) {
        factory->Release();
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        res.status = "SKIPPED (WARP adapter unavailable)";
        return res;
    }

    ID3D12Device* dev = nullptr;
    HRESULT hr = pD3D12CreateDevice(warpAdapter, D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&dev));
    warpAdapter->Release();
    factory->Release();
    if (FAILED(hr) || !dev) {
        FreeLibrary(d3d12Lib);
        FreeLibrary(dxgiLib);
        res.status = "SKIPPED (WARP D3D12CreateDevice failed)";
        return res;
    }

    D3D12_COMMAND_QUEUE_DESC qDesc{};
    qDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ID3D12CommandQueue* queue = nullptr;
    dev->CreateCommandQueue(&qDesc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&queue));

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
    ID3D12Resource* srv0 = mkBuf(D3D12_HEAP_TYPE_UPLOAD, 4 * 1024 * 1024, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* srv1 = mkBuf(D3D12_HEAP_TYPE_UPLOAD, 4 * 1024 * 1024, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
    ID3D12Resource* uav0 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, maxOutBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* uav1 = mkBuf(D3D12_HEAP_TYPE_DEFAULT, 4 * 1024 * 1024, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12Resource* rb = mkBuf(D3D12_HEAP_TYPE_READBACK, maxOutBytes, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);

    void* ptr = nullptr;
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
    } else if (spec.id == "matmul") {
        srv0->Map(0, nullptr, &ptr);
        float* a = static_cast<float*>(ptr);
        for (size_t i = 0; i < 256 * 256; ++i) a[i] = static_cast<float>((i % 17) + 1) * 0.01f;
        srv0->Unmap(0, nullptr);
        srv1->Map(0, nullptr, &ptr);
        float* b = static_cast<float*>(ptr);
        for (size_t i = 0; i < 256 * 256; ++i) b[i] = static_cast<float>((i % 13) + 1) * 0.01f;
        srv1->Unmap(0, nullptr);
    } else if (spec.id == "prefix_sum_groupshared") {
        srv0->Map(0, nullptr, &ptr);
        uint32_t* p = static_cast<uint32_t*>(ptr);
        for (size_t i = 0; i < 65536 * 4; ++i) p[i] = static_cast<uint32_t>((i & 63u) + 1u);
        srv0->Unmap(0, nullptr);
    }

    ID3D12CommandAllocator* alloc = nullptr;
    dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, __uuidof(ID3D12CommandAllocator), reinterpret_cast<void**>(&alloc));
    ID3D12GraphicsCommandList* cl = nullptr;
    dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc, pso, __uuidof(ID3D12GraphicsCommandList), reinterpret_cast<void**>(&cl));
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
        } else if (spec.id == "matmul") {
            MatmulCB mcb{cbW, cbH, cbSteps, 0};
            std::memcpy(ptr, &mcb, sizeof(mcb));
        }
        cb->Unmap(0, nullptr);

        auto s0 = std::chrono::steady_clock::now();
        alloc->Reset();
        cl->Reset(alloc, pso);
        cl->SetComputeRootSignature(rootSig);
        cl->SetComputeRootConstantBufferView(0, cb->GetGPUVirtualAddress());
        cl->SetComputeRootShaderResourceView(1, srv0->GetGPUVirtualAddress());
        cl->SetComputeRootShaderResourceView(2, srv1->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(3, uav0->GetGPUVirtualAddress());
        cl->SetComputeRootUnorderedAccessView(4, uav1->GetGPUVirtualAddress());
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
#endif

std::string serializeResultJson(const BenchRunResult& r) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(4)
       << "{"
       << "\"bench_id\":\"" << jsonEscape(r.benchId) << "\","
       << "\"bench_name\":\"" << jsonEscape(r.benchName) << "\","
       << "\"category\":\"" << jsonEscape(r.category) << "\","
       << "\"backend\":\"" << jsonEscape(r.backend) << "\","
       << "\"shader_format\":\"" << jsonEscape(r.shaderFormat) << "\","
       << "\"simd_width\":" << r.simdWidth << ","
       << "\"ran\":" << (r.ran ? "true" : "false") << ","
       << "\"status\":\"" << jsonEscape(r.status) << "\","
       << "\"compile_ms\":" << r.compileMs << ","
       << "\"first_dispatch_ms\":" << r.firstDispatchMs << ","
       << "\"cold_ms\":" << r.coldMs << ","
       << "\"mean_ms\":" << r.meanMs << ","
       << "\"median_ms\":" << r.medianMs << ","
       << "\"min_ms\":" << r.minMs << ","
       << "\"short_job_ms\":" << r.shortJobMs << ","
       << "\"long_job_ms\":" << r.longJobMs << ","
       << "\"rss_before_mb\":" << r.rssBeforeMb << ","
       << "\"rss_after_compile_mb\":" << r.rssAfterCompileMb << ","
       << "\"rss_after_run_mb\":" << r.rssAfterRunMb << ","
       << "\"rss_delta_mb\":" << r.rssDeltaMb << ","
       << "\"peak_rss_mb\":" << r.peakRssMb << ","
       << "\"checksum\":" << std::setprecision(6) << r.checksum
       << "}";
    return os.str();
}

// Lightweight parser for worker JSON lines (we control the exact output format)
bool parseWorkerJsonLine(const std::string& line, BenchRunResult& out) {
    auto getStr = [&](const char* key) -> std::string {
        std::string pat = std::string("\"") + key + "\":\"";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return "";
        pos += pat.size();
        std::string val;
        while (pos < line.size() && line[pos] != '"') {
            if (line[pos] == '\\' && pos + 1 < line.size()) {
                ++pos;
            }
            val.push_back(line[pos++]);
        }
        return val;
    };
    auto getNum = [&](const char* key) -> double {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return 0.0;
        pos += pat.size();
        return std::strtod(line.c_str() + pos, nullptr);
    };
    auto getBool = [&](const char* key) -> bool {
        std::string pat = std::string("\"") + key + "\":";
        size_t pos = line.find(pat);
        if (pos == std::string::npos) return false;
        pos += pat.size();
        return line.compare(pos, 4, "true") == 0;
    };

    out.benchId = getStr("bench_id");
    if (out.benchId.empty()) return false;
    out.benchName = getStr("bench_name");
    out.category = getStr("category");
    out.backend = getStr("backend");
    out.shaderFormat = getStr("shader_format");
    out.simdWidth = static_cast<uint32_t>(getNum("simd_width"));
    out.ran = getBool("ran");
    out.status = getStr("status");
    out.compileMs = getNum("compile_ms");
    out.firstDispatchMs = getNum("first_dispatch_ms");
    out.coldMs = getNum("cold_ms");
    out.meanMs = getNum("mean_ms");
    out.medianMs = getNum("median_ms");
    out.minMs = getNum("min_ms");
    out.shortJobMs = getNum("short_job_ms");
    out.longJobMs = getNum("long_job_ms");
    out.rssBeforeMb = getNum("rss_before_mb");
    out.rssAfterCompileMb = getNum("rss_after_compile_mb");
    out.rssAfterRunMb = getNum("rss_after_run_mb");
    out.rssDeltaMb = getNum("rss_delta_mb");
    out.peakRssMb = getNum("peak_rss_mb");
    out.checksum = getNum("checksum");
    return true;
}

std::vector<BenchRunResult> runSingleBenchmarkBackend(
    const BenchmarkSpec& spec,
    const std::string& backendToken,
    const std::string& shaderDir,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    std::vector<BenchRunResult> out;

    if (backendToken.rfind("nacelle-w", 0) == 0) {
        uint32_t w = static_cast<uint32_t>(std::stoul(backendToken.substr(9)));
        if (spec.kind == PipelineKind::Compute) {
            std::vector<uint8_t> dxil;
            if (!readBinary(shaderDir + "/" + spec.csDxil, dxil)) return out;
            out.push_back(runNacelleComputeBench(spec, dxil, w, width, height, maxSteps, iters));
        } else if (spec.kind == PipelineKind::GraphicsVSPS ||
                   spec.kind == PipelineKind::GraphicsVSGSPS ||
                   spec.kind == PipelineKind::GraphicsTessGS) {
            std::vector<uint8_t> vs, ps, hs, ds, gs;
            if (!readBinary(shaderDir + "/" + spec.vsDxil, vs) ||
                !readBinary(shaderDir + "/" + spec.psDxil, ps)) return out;
            if (!spec.hsDxil.empty() && !readBinary(shaderDir + "/" + spec.hsDxil, hs)) return out;
            if (!spec.dsDxil.empty() && !readBinary(shaderDir + "/" + spec.dsDxil, ds)) return out;
            if (!spec.gsDxil.empty() && !readBinary(shaderDir + "/" + spec.gsDxil, gs)) return out;
            out.push_back(runNacelleGraphicsBench(spec, vs, ps, hs, ds, gs, w, width, height, iters));
        } else if (spec.kind == PipelineKind::RayTracingInline ||
                   spec.kind == PipelineKind::RayTracingPipeline) {
            std::vector<uint8_t> rt;
            if (!readBinary(shaderDir + "/" + spec.rtDxil, rt)) return out;
            out.push_back(runNacelleRaytracingBench(spec, rt, w, iters));
        }
        return out;
    }

    if (backendToken == "warp") {
#if defined(_WIN32)
        if (spec.kind == PipelineKind::Compute) {
            std::vector<uint8_t> dxil;
            if (readBinary(shaderDir + "/" + spec.csDxil, dxil)) {
                out.push_back(runWarpComputeBench(spec, dxil, width, height, maxSteps, iters));
            }
        }
#else
        if (spec.id == "bench_raymarch") {
            BenchRunResult r;
            r.benchId = spec.id;
            r.benchName = spec.name;
            r.category = categoryName(spec.category);
            r.backend = "Microsoft WARP";
            r.shaderFormat = "DXIL";
            r.status = "SKIPPED (Windows-only)";
            out.push_back(r);
        }
#endif
        return out;
    }

    if (backendToken.rfind("vulkan", 0) == 0) {
        int32_t devIdx = -1;
        if (backendToken.size() > 7 && backendToken[6] == ':') {
            devIdx = std::stoi(backendToken.substr(7));
        }
        if (spec.kind == PipelineKind::Compute && !spec.csSpv.empty()) {
            std::vector<uint8_t> spv;
            std::string err;
            if (readBinary(shaderDir + "/" + spec.csSpv, spv) && validateSpirvBinary(spv, err)) {
                auto vkRes = runVulkanComputeBench(spec, spv, width, height, maxSteps, iters, devIdx);
                out.insert(out.end(), vkRes.begin(), vkRes.end());
            }
        } else if (spec.kind == PipelineKind::GraphicsVSPS && !spec.vsSpv.empty() && !spec.psSpv.empty()) {
            std::vector<uint8_t> vsSpv, psSpv;
            std::string err;
            if (readBinary(shaderDir + "/" + spec.vsSpv, vsSpv) && validateSpirvBinary(vsSpv, err) &&
                readBinary(shaderDir + "/" + spec.psSpv, psSpv) && validateSpirvBinary(psSpv, err)) {
                auto vkRes = runVulkanGraphicsBench(spec, vsSpv, psSpv, width, height, iters, devIdx);
                out.insert(out.end(), vkRes.begin(), vkRes.end());
            }
        }
        return out;
    }

    if (backendToken == "metal") {
#if defined(__APPLE__)
        auto recordMetal = [&](const nacelle_bench::MetalBenchResult& mr, const MemorySnapshot& memBefore, const MemorySnapshot& memAfter) {
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
            if (readBinary(shaderDir + "/" + spec.csDxil, dxil)) {
                MemorySnapshot memBefore = queryProcessMemory();
                auto mr = nacelle_bench::runMetalComputeBench(spec.id, dxil, width, height, maxSteps, iters);
                MemorySnapshot memAfter = queryProcessMemory();
                recordMetal(mr, memBefore, memAfter);
            }
        } else if (spec.kind == PipelineKind::GraphicsVSPS && !spec.vsDxil.empty() && !spec.psDxil.empty()) {
            std::vector<uint8_t> vsDxil, psDxil;
            if (readBinary(shaderDir + "/" + spec.vsDxil, vsDxil) &&
                readBinary(shaderDir + "/" + spec.psDxil, psDxil)) {
                MemorySnapshot memBefore = queryProcessMemory();
                auto mr = nacelle_bench::runMetalGraphicsBench(
                    spec.id, categoryName(spec.category), vsDxil, psDxil, width, height, iters);
                MemorySnapshot memAfter = queryProcessMemory();
                recordMetal(mr, memBefore, memAfter);
            }
        }
#endif
        return out;
    }

    return out;
}

std::vector<BenchRunResult> runIsolatedWorker(
    const std::string& exePath,
    const std::string& benchId,
    const std::string& backendToken,
    const std::string& shaderDir,
    uint32_t width,
    uint32_t height,
    uint32_t maxSteps,
    uint32_t iters) {
    std::ostringstream cmd;
    cmd << "\"" << exePath << "\""
        << " --worker"
        << " --single-bench " << benchId
        << " --single-backend " << backendToken
        << " --shader-dir \"" << shaderDir << "\""
        << " --width " << width
        << " --height " << height
        << " --steps " << maxSteps
        << " --iters " << iters
        << " 2>&1";

    std::vector<BenchRunResult> out;
    FILE* pipe = NACELLE_POPEN(cmd.str().c_str(), "r");
    if (!pipe) return out;

    char buf[4096];
    std::string line;
    bool inJson = false;
    while (fgets(buf, sizeof(buf), pipe) != nullptr) {
        line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        if (line == "<<<BENCH_JSON_BEGIN>>>") {
            inJson = true;
            continue;
        }
        if (line == "<<<BENCH_JSON_END>>>") {
            inJson = false;
            continue;
        }
        if (inJson && !line.empty()) {
            BenchRunResult r;
            if (parseWorkerJsonLine(line, r)) {
                out.push_back(std::move(r));
            }
        }
    }
    NACELLE_PCLOSE(pipe);
    return out;
}

struct WinnerInfo {
    std::string category;
    std::string winnerBackend;
    double metricValue = 0.0;
    std::string unit;
};

void writeFullJsonReport(
    const std::string& path,
    const std::vector<BenchRunResult>& allResults,
    const std::vector<WinnerInfo>& categoryWinners,
    const std::vector<WinnerInfo>& regimeWinners,
    const std::map<std::string, double>& overallScores,
    const std::string& overallWinner) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Failed to write JSON report to " << path << "\n";
        return;
    }

    ofs << "{\n"
        << "  \"version\": 2,\n"
        << "  \"overall_winner\": \"" << jsonEscape(overallWinner) << "\",\n"
        << "  \"overall_geomean_ms\": {\n";
    {
        size_t idx = 0;
        for (const auto& [b, ms] : overallScores) {
            ofs << "    \"" << jsonEscape(b) << "\": " << std::fixed << std::setprecision(4) << ms;
            if (++idx < overallScores.size()) ofs << ",";
            ofs << "\n";
        }
    }
    ofs << "  },\n"
        << "  \"category_winners\": [\n";
    for (size_t i = 0; i < categoryWinners.size(); ++i) {
        const auto& w = categoryWinners[i];
        ofs << "    {\"category\":\"" << jsonEscape(w.category) << "\","
            << "\"winner\":\"" << jsonEscape(w.winnerBackend) << "\","
            << "\"geomean_ms\":" << std::fixed << std::setprecision(4) << w.metricValue << "}";
        if (i + 1 < categoryWinners.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n"
        << "  \"regime_winners\": [\n";
    for (size_t i = 0; i < regimeWinners.size(); ++i) {
        const auto& w = regimeWinners[i];
        ofs << "    {\"regime\":\"" << jsonEscape(w.category) << "\","
            << "\"winner\":\"" << jsonEscape(w.winnerBackend) << "\","
            << "\"value\":" << std::fixed << std::setprecision(4) << w.metricValue << ","
            << "\"unit\":\"" << jsonEscape(w.unit) << "\"}";
        if (i + 1 < regimeWinners.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n"
        << "  \"suites\": [\n";
    for (size_t i = 0; i < allResults.size(); ++i) {
        ofs << "    " << serializeResultJson(allResults[i]);
        if (i + 1 < allResults.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ],\n"
        << "  \"benchmarks\": [\n";
    // Backward-compatible top-level benchmarks array for bench_raymarch
    std::vector<const BenchRunResult*> compat;
    for (const auto& r : allResults) {
        if (r.benchId == "bench_raymarch") compat.push_back(&r);
    }
    for (size_t i = 0; i < compat.size(); ++i) {
        ofs << "    " << serializeResultJson(*compat[i]);
        if (i + 1 < compat.size()) ofs << ",";
        ofs << "\n";
    }
    ofs << "  ]\n}\n";
}

} // namespace

int main(int argc, char** argv) {
    std::vector<uint32_t> widths = {4, 8, 16};
    uint32_t width = 256;
    uint32_t height = 256;
    uint32_t maxSteps = 48;
    uint32_t iters = 5;
    std::string jsonOut;
    std::string shaderDir = NACELLE_SHADER_BIN_DIR;
    std::string suiteFilter = "all";
    std::string nameFilter;
    bool isolate = true;
    bool workerMode = false;
    std::string singleBench;
    std::string singleBackend;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--quick") {
            width = 128;
            height = 128;
            maxSteps = 24;
            iters = 2;
        } else if (arg.rfind("--simd-width=", 0) == 0) {
            widths.clear();
            std::stringstream ss(arg.substr(13));
            std::string tok;
            while (std::getline(ss, tok, ',')) {
                if (!tok.empty()) widths.push_back(static_cast<uint32_t>(std::stoul(tok)));
            }
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--width" && i + 1 < argc) {
            width = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--height" && i + 1 < argc) {
            height = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--steps" && i + 1 < argc) {
            maxSteps = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--json" && i + 1 < argc) {
            jsonOut = argv[++i];
        } else if (arg == "--shader-dir" && i + 1 < argc) {
            shaderDir = argv[++i];
        } else if (arg == "--suite" && i + 1 < argc) {
            suiteFilter = argv[++i];
        } else if (arg == "--filter" && i + 1 < argc) {
            nameFilter = argv[++i];
        } else if (arg == "--isolate") {
            isolate = true;
        } else if (arg == "--no-isolate") {
            isolate = false;
        } else if (arg == "--worker") {
            workerMode = true;
        } else if (arg == "--single-bench" && i + 1 < argc) {
            singleBench = argv[++i];
        } else if (arg == "--single-backend" && i + 1 < argc) {
            singleBackend = argv[++i];
        }
    }

    auto catalog = buildBenchmarkCatalog();

    if (workerMode) {
        for (const auto& spec : catalog) {
            if (!singleBench.empty() && spec.id != singleBench) continue;
            auto res = runSingleBenchmarkBackend(spec, singleBackend, shaderDir, width, height, maxSteps, iters);
            std::cout << "<<<BENCH_JSON_BEGIN>>>\n";
            for (const auto& r : res) {
                std::cout << serializeResultJson(r) << "\n";
            }
            std::cout << "<<<BENCH_JSON_END>>>\n";
        }
        return 0;
    }

    // Build backend token list
    std::vector<std::string> backendTokens;
    for (uint32_t w : widths) {
        backendTokens.push_back("nacelle-w" + std::to_string(w));
    }
    backendTokens.push_back("warp");

    auto vkDevs = listVulkanDevices();
    if (vkDevs.empty()) {
        backendTokens.push_back("vulkan");
    } else {
        for (const auto& d : vkDevs) {
            backendTokens.push_back("vulkan:" + std::to_string(d.index));
        }
    }
#if defined(__APPLE__)
    backendTokens.push_back("metal");
#endif

    std::string exePath = argv[0];
    std::vector<BenchRunResult> allResults;
    bool anyNacelleFailed = false;

    for (const auto& spec : catalog) {
        if (suiteFilter != "all" && suiteFilter != categoryKey(spec.category)) continue;
        if (!nameFilter.empty() && spec.id.find(nameFilter) == std::string::npos) continue;

        for (const auto& bt : backendTokens) {
            std::vector<BenchRunResult> res;
            if (isolate) {
                res = runIsolatedWorker(exePath, spec.id, bt, shaderDir, width, height, maxSteps, iters);
            } else {
                res = runSingleBenchmarkBackend(spec, bt, shaderDir, width, height, maxSteps, iters);
            }
            for (auto& r : res) {
                if (bt.rfind("nacelle-", 0) == 0 && !r.ran) anyNacelleFailed = true;
                allResults.push_back(std::move(r));
            }
        }
    }

    // Compute Category Winners (lowest geomean dispatch time in ms)
    std::vector<BenchCategory> cats = {
        BenchCategory::Compute, BenchCategory::Vertex, BenchCategory::Pixel,
        BenchCategory::Geometry, BenchCategory::Tessellation, BenchCategory::RayTracing
    };

    std::map<std::string, std::vector<double>> backendCategoryMs;
    std::vector<WinnerInfo> categoryWinners;

    for (auto c : cats) {
        std::string cname = categoryName(c);
        std::map<std::string, std::vector<double>> msByBackend;
        for (const auto& r : allResults) {
            if (r.ran && r.category == cname && r.medianMs > 0.0) {
                msByBackend[r.backend].push_back(r.medianMs);
            }
        }
        std::string bestBackend = "N/A";
        double bestMs = 1e18;
        for (const auto& [b, vals] : msByBackend) {
            double gm = computeGeoMean(vals);
            if (gm > 0.0) {
                backendCategoryMs[b].push_back(gm);
                if (gm < bestMs) {
                    bestMs = gm;
                    bestBackend = b;
                }
            }
        }
        categoryWinners.push_back({cname, bestBackend, (bestBackend == "N/A" ? 0.0 : bestMs), "ms"});
    }
    categoryWinners.push_back({"Mesh", "NOT IMPLEMENTED (DispatchMesh stub)", 0.0, "ms"});

    // Overall Geomean Time (geomean of category geomean ms, lower is faster)
    std::map<std::string, double> overallMs;
    std::string overallWinner = "N/A";
    double bestOverallMs = 1e18;
    for (const auto& [b, catMs] : backendCategoryMs) {
        double ov = computeGeoMean(catMs);
        overallMs[b] = ov;
        if (ov > 0.0 && ov < bestOverallMs) {
            bestOverallMs = ov;
            overallWinner = b;
        }
    }

    // Regime Winners: Cold JIT, Hot JIT, Short Job, Long Job, Lowest Peak Memory
    std::map<std::string, std::vector<double>> coldByB, hotByB, shortByB, longByB, memByB;
    for (const auto& r : allResults) {
        if (!r.ran) continue;
        if (r.coldMs > 0.0) coldByB[r.backend].push_back(r.coldMs);
        if (r.medianMs > 0.0) hotByB[r.backend].push_back(r.medianMs);
        if (r.shortJobMs > 0.0) shortByB[r.backend].push_back(r.shortJobMs);
        if (r.longJobMs > 0.0) longByB[r.backend].push_back(r.longJobMs);
        if (r.peakRssMb > 0.0) memByB[r.backend].push_back(r.peakRssMb);
    }

    auto pickLowestGeomean = [](const std::map<std::string, std::vector<double>>& m, const std::string& label, const std::string& unit) -> WinnerInfo {
        std::string best = "N/A";
        double bestVal = 1e18;
        for (const auto& [b, vals] : m) {
            double gm = computeGeoMean(vals);
            if (gm > 0.0 && gm < bestVal) {
                bestVal = gm;
                best = b;
            }
        }
        return {label, best, (best == "N/A" ? 0.0 : bestVal), unit};
    };

    std::vector<WinnerInfo> regimeWinners = {
        pickLowestGeomean(coldByB, "Cold JIT", "ms"),
        pickLowestGeomean(hotByB, "Hot JIT", "ms"),
        pickLowestGeomean(shortByB, "Short-Running Job", "ms"),
        pickLowestGeomean(longByB, "Long-Running Job", "ms"),
        pickLowestGeomean(memByB, "Lowest Peak Memory", "MB"),
    };

    // Print Console Report
    std::cout << "\n========================================================================================================================\n";
    std::cout << "  Nacelle Cross-Backend Shader & Rasterizer Benchmark Suite (" << width << "x" << height << ", steps=" << maxSteps << ", iters=" << iters << ")\n";
    std::cout << "========================================================================================================================\n\n";

    size_t benchColW = 20;
    size_t backendColW = 22;
    for (const auto& r : allResults) {
        benchColW = std::max(benchColW, r.benchName.size() + 2);
        backendColW = std::max(backendColW, r.backend.size() + 2);
    }
    const size_t totalLineW = benchColW + 13 + backendColW + 10 * 5 + 10;

    std::cout << std::left
              << std::setw(static_cast<int>(benchColW)) << "Benchmark"
              << std::setw(13) << "Category"
              << std::setw(static_cast<int>(backendColW)) << "Backend"
              << std::setw(10) << "Cold(ms)"
              << std::setw(10) << "Hot(ms)"
              << std::setw(10) << "Short(ms)"
              << std::setw(10) << "Long(ms)"
              << std::setw(10) << "Peak(MB)"
              << "Status\n";
    std::cout << std::string(totalLineW, '-') << "\n";

    for (const auto& r : allResults) {
        std::cout << std::left
                  << std::setw(static_cast<int>(benchColW)) << r.benchName
                  << std::setw(13) << r.category
                  << std::setw(static_cast<int>(backendColW)) << r.backend;
        if (r.ran) {
            std::cout << std::fixed << std::setprecision(2)
                      << std::setw(10) << r.coldMs
                      << std::setw(10) << r.medianMs
                      << std::setw(10) << r.shortJobMs
                      << std::setw(10) << r.longJobMs
                      << std::setprecision(1) << std::setw(10) << r.peakRssMb
                      << r.status << "\n";
        } else {
            std::cout << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << std::setw(10) << "-"
                      << r.status << "\n";
        }
    }

    std::cout << "\n--- Category Winners (Geomean Dispatch Time, lower is faster) ---\n";
    for (const auto& w : categoryWinners) {
        std::cout << "  " << std::left << std::setw(16) << w.category
                  << "-> " << std::setw(36) << w.winnerBackend;
        if (w.metricValue > 0.0) {
            std::cout << std::fixed << std::setprecision(2) << w.metricValue << " ms";
        }
        std::cout << "\n";
    }

    std::cout << "\n--- Regime Winners (Geomean, lower is better) ---\n";
    for (const auto& w : regimeWinners) {
        std::cout << "  " << std::left << std::setw(22) << w.category
                  << "-> " << std::setw(30) << w.winnerBackend
                  << std::fixed << std::setprecision(2) << w.metricValue << " " << w.unit << "\n";
    }

    std::cout << "\n--- Overall Standings (Geomean Dispatch Time, lower is faster) ---\n";
    std::vector<std::pair<std::string, double>> sortedOverall(overallMs.begin(), overallMs.end());
    std::sort(sortedOverall.begin(), sortedOverall.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    for (size_t i = 0; i < sortedOverall.size(); ++i) {
        std::cout << "  #" << (i + 1) << " " << std::left << std::setw(32) << sortedOverall[i].first
                  << std::fixed << std::setprecision(2) << sortedOverall[i].second << " ms\n";
    }
    std::cout << "\nOVERALL WINNER: " << overallWinner << "\n";

    if (!jsonOut.empty()) {
        writeFullJsonReport(jsonOut, allResults, categoryWinners, regimeWinners, overallMs, overallWinner);
        std::cout << "JSON report written to: " << jsonOut << "\n";
    }

    return anyNacelleFailed ? 1 : 0;
}
