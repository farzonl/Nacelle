#include "test_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

ComPtr<ID3D12Resource> createTex2D(
    ID3D12Device* dev,
    UINT w,
    UINT h,
    DXGI_FORMAT fmt,
    UINT sampleCount,
    D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Tex2D(fmt, w, h, 1, 1, sampleCount, 0, flags);
    ComPtr<ID3D12Resource> res;
    testutil::checkHR(
        dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)),
        "CreateTex2D");
    return res;
}

std::vector<float> readbackFloat4Tex2D(testutil::TestRuntime& rt, ID3D12Resource* tex, UINT w, UINT h) {
    constexpr UINT kBpp = 16;
    const UINT rowPitch = (w * kBpp + 255u) & ~255u;
    auto rb = rt.createBuffer(static_cast<UINT64>(rowPitch) * h, D3D12_HEAP_TYPE_READBACK);
    rt.resetCmd();

    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = rb.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    dst.PlacedFootprint.Footprint.Width = w;
    dst.PlacedFootprint.Footprint.Height = h;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = rowPitch;

    D3D12_TEXTURE_COPY_LOCATION src{};
    src.pResource = tex;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;

    rt.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    rt.executeAndWait();

    void* mapped = nullptr;
    testutil::checkHR(rb->Map(0, nullptr, &mapped), "Map readback");
    std::vector<float> out(static_cast<size_t>(w) * h * 4);
    const auto* p = static_cast<const uint8_t*>(mapped);
    for (UINT y = 0; y < h; ++y) {
        std::memcpy(out.data() + static_cast<size_t>(y) * w * 4, p + static_cast<size_t>(y) * rowPitch, w * kBpp);
    }
    rb->Unmap(0, nullptr);
    return out;
}

ComPtr<ID3D12PipelineState> createMeshStreamPSO(
    ID3D12Device* dev,
    ID3D12RootSignature* rs,
    const std::vector<uint8_t>* asBytes,
    const std::vector<uint8_t>& msBytes,
    const std::vector<uint8_t>& psBytes,
    DXGI_FORMAT rtvFormat,
    DXGI_FORMAT dsvFormat,
    const CD3DX12_DEPTH_STENCIL_DESC1& dss,
    D3D12_PRIMITIVE_TOPOLOGY_TYPE topoType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
    UINT sampleCount = 1) {
    ComPtr<ID3D12Device2> dev2;
    testutil::checkHR(dev->QueryInterface(IID_PPV_ARGS(&dev2)), "QueryInterface ID3D12Device2");

    D3D12_RT_FORMAT_ARRAY rtArray{};
    rtArray.NumRenderTargets = 1;
    rtArray.RTFormats[0] = rtvFormat;

    DXGI_SAMPLE_DESC sampleDesc{};
    sampleDesc.Count = sampleCount;

    CD3DX12_PIPELINE_MESH_STATE_STREAM stream;
    stream.pRootSignature = rs;
    if (asBytes && !asBytes->empty()) {
        stream.AS = D3D12_SHADER_BYTECODE{asBytes->data(), asBytes->size()};
    }
    stream.MS = D3D12_SHADER_BYTECODE{msBytes.data(), msBytes.size()};
    stream.PS = D3D12_SHADER_BYTECODE{psBytes.data(), psBytes.size()};
    stream.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    stream.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    stream.DepthStencilState = dss;
    stream.SampleMask = UINT_MAX;
    stream.PrimitiveTopologyType = topoType;
    stream.RTVFormats = rtArray;
    stream.DSVFormat = dsvFormat;
    stream.SampleDesc = sampleDesc;

    D3D12_PIPELINE_STATE_STREAM_DESC streamDesc{sizeof(stream), &stream};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(dev2->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&pso)), "CreatePipelineState (Mesh Stream)");
    return pso;
}

void testMeshShaderFeatureSupport(testutil::TestRuntime& rt) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 opts7{};
    testutil::checkHR(
        rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS7, &opts7, sizeof(opts7)),
        "CheckFeatureSupport OPTIONS7");
    assert(opts7.MeshShaderTier == D3D12_MESH_SHADER_TIER_1);

    ComPtr<ID3D12Device2> dev2;
    testutil::checkHR(rt.device->QueryInterface(IID_PPV_ARGS(&dev2)), "QI ID3D12Device2");
    ComPtr<ID3D12GraphicsCommandList6> cmd6;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmd6)), "QI ID3D12GraphicsCommandList6");
    std::printf("  MeshShaderFeatureSupport (Tier=%d) PASSED\n", static_cast<int>(opts7.MeshShaderTier));
}

// Covers OffloadTest SimpleTriangle.test and SimpleAmplification.test (and verifies exact pixel equivalence)
void testSimpleTriangleAndAmplification(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 256, H = 256;
    auto msSimple = testutil::loadShaderBytes("mesh_simple_triangle_ms.dxil");
    auto asPayload = testutil::loadShaderBytes("mesh_payload_as.dxil");
    auto msPayload = testutil::loadShaderBytes("mesh_payload_ms.dxil");
    auto psColor = testutil::loadShaderBytes("mesh_vertex_color_ps.dxil");

    CD3DX12_DEPTH_STENCIL_DESC1 dss(D3D12_DEFAULT);
    dss.DepthEnable = FALSE;
    dss.StencilEnable = FALSE;

    auto psoTri = createMeshStreamPSO(
        rt.device.Get(), rs, nullptr, msSimple, psColor,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, dss);
    auto psoAmp = createMeshStreamPSO(
        rt.device.Get(), rs, &asPayload, msPayload, psColor,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, dss);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap, dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "CreateDSVHeap");
    const UINT rtvStep = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    auto rtTri = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    auto rtAmp = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    auto dsTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);

    D3D12_CPU_DESCRIPTOR_HANDLE rtv0 = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE rtv1{rtv0.ptr + rtvStep};
    D3D12_CPU_DESCRIPTOR_HANDLE dsv0 = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtTri.Get(), nullptr, rtv0);
    rt.device->CreateRenderTargetView(rtAmp.Get(), nullptr, rtv1);
    rt.device->CreateDepthStencilView(dsTex.Get(), nullptr, dsv0);

    ComPtr<ID3D12GraphicsCommandList6> cmd6;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmd6)), "QI CmdList6");

    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    rt.resetCmd();
    cmd6->SetGraphicsRootSignature(rs);
    cmd6->RSSetViewports(1, &vp);
    cmd6->RSSetScissorRects(1, &sc);

    cmd6->ClearRenderTargetView(rtv0, clearCol, 0, nullptr);
    cmd6->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtv0, FALSE, &dsv0);
    cmd6->SetPipelineState(psoTri.Get());
    cmd6->DispatchMesh(1, 1, 1);

    cmd6->ClearRenderTargetView(rtv1, clearCol, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtv1, FALSE, &dsv0);
    cmd6->SetPipelineState(psoAmp.Get());
    cmd6->DispatchMesh(1, 1, 1);
    rt.executeAndWait();

    auto pxTri = readbackFloat4Tex2D(rt, rtTri.Get(), W, H);
    auto pxAmp = readbackFloat4Tex2D(rt, rtAmp.Get(), W, H);

    auto getPx = [&](const std::vector<float>& img, UINT x, UINT y) -> const float* {
        return &img[(static_cast<size_t>(y) * W + x) * 4];
    };

    // Screen center (128, 128) is NDC (~0, ~0): barycentrics ~(0.5, 0.25, 0.25)
    const float* mid = getPx(pxTri, 128, 128);
    assert(std::fabs(mid[0] - 0.5f) < 0.03f);
    assert(std::fabs(mid[1] - 0.25f) < 0.03f);
    assert(std::fabs(mid[2] - 0.25f) < 0.03f);
    assert(std::fabs(mid[3] - 1.0f) < 1e-4f);

    // Near top vertex (128, 100) -> Red; bottom-right (154, 156) -> Green; bottom-left (102, 156) -> Blue
    const float* top = getPx(pxTri, 128, 100);
    const float* br = getPx(pxTri, 154, 156);
    const float* bl = getPx(pxTri, 102, 156);
    assert(top[0] > 0.85f && top[1] < 0.1f && top[2] < 0.1f);
    assert(br[1] > 0.75f && br[0] < 0.15f && br[2] < 0.15f);
    assert(bl[2] > 0.75f && bl[0] < 0.15f && bl[1] < 0.15f);

    // Background outside triangle must remain clear
    const float* bg = getPx(pxTri, 32, 32);
    assert(bg[0] == 0.0f && bg[1] == 0.0f && bg[2] == 0.0f && bg[3] == 0.0f);

    // SimpleAmplification must match SimpleTriangle pixel-for-pixel
    float maxDiff = 0.0f;
    for (size_t i = 0; i < pxTri.size(); ++i) {
        maxDiff = std::max(maxDiff, std::fabs(pxTri[i] - pxAmp[i]));
    }
    assert(maxDiff < 1e-5f);
    std::printf("  SimpleTriangle & SimpleAmplification (center=(%.3f,%.3f,%.3f), maxDiff=%.2e) PASSED\n",
                mid[0], mid[1], mid[2], maxDiff);
}

// Covers OffloadTest SimpleLines.test ([outputtopology("line")], 32-segment hue circle)
void testSimpleLines(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 256, H = 256;
    auto msLines = testutil::loadShaderBytes("mesh_circle_lines_ms.dxil");
    auto psSat = testutil::loadShaderBytes("mesh_saturate_color_ps.dxil");

    CD3DX12_DEPTH_STENCIL_DESC1 dss(D3D12_DEFAULT);
    dss.DepthEnable = FALSE;
    dss.StencilEnable = FALSE;

    auto pso = createMeshStreamPSO(
        rt.device.Get(), rs, nullptr, msLines, psSat,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, dss,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap, dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "CreateDSVHeap");

    auto rtTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    auto dsTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtTex.Get(), nullptr, rtvH);
    rt.device->CreateDepthStencilView(dsTex.Get(), nullptr, dsvH);

    ComPtr<ID3D12GraphicsCommandList6> cmd6;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmd6)), "QI CmdList6");

    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    rt.resetCmd();
    cmd6->SetGraphicsRootSignature(rs);
    cmd6->RSSetViewports(1, &vp);
    cmd6->RSSetScissorRects(1, &sc);
    cmd6->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
    cmd6->SetPipelineState(pso.Get());
    cmd6->DispatchMesh(1, 1, 1);
    rt.executeAndWait();

    auto px = readbackFloat4Tex2D(rt, rtTex.Get(), W, H);
    UINT litCount = 0;
    UINT octantHits[8]{};
    float rightRed = 0.0f, leftCyanG = 0.0f, leftCyanB = 0.0f;

    for (UINT y = 0; y < H; ++y) {
        for (UINT x = 0; x < W; ++x) {
            const float* c = &px[(static_cast<size_t>(y) * W + x) * 4];
            if (c[3] > 0.5f) {
                ++litCount;
                const float dx = (float(x) + 0.5f) - 128.0f;
                const float dy = (float(y) + 0.5f) - 128.0f;
                const float dist = std::sqrt(dx * dx + dy * dy);
                // 32-gon inscribed in radius 64px has inradius ~63.69px and circumradius 64.0px
                assert(dist >= 61.5f && dist <= 66.5f);
                double ang = std::atan2(dy, dx); // [-pi, pi]
                int oct = static_cast<int>(std::floor((ang + 3.141592653589793) / (2.0 * 3.141592653589793) * 8.0)) & 7;
                ++octantHits[oct];
                if (x >= 190 && std::abs(int(y) - 128) <= 2) rightRed = std::max(rightRed, c[0]);
                if (x <= 66 && std::abs(int(y) - 128) <= 2) {
                    leftCyanG = std::max(leftCyanG, c[1]);
                    leftCyanB = std::max(leftCyanB, c[2]);
                }
            }
        }
    }

    // Center and corners must be unlit
    assert(px[(128 * W + 128) * 4 + 3] == 0.0f);
    assert(px[(8 * W + 8) * 4 + 3] == 0.0f);
    assert(litCount >= 320 && litCount <= 520);
    for (int o = 0; o < 8; ++o) {
        assert(octantHits[o] >= 30);
    }
    assert(rightRed > 0.9f);
    assert(leftCyanG > 0.9f && leftCyanB > 0.9f);
    std::printf("  SimpleLines (litPixels=%u, all 8 octants covered) PASSED\n", litCount);
}

// Covers OffloadTest StencilState.test (1x1 RTV + D32_FLOAT_S8X24_UINT DSV cleared to stencil=1, default StencilRef=0)
void testStencilState(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 1, H = 1;
    auto msFull = testutil::loadShaderBytes("mesh_stencil_fullscreen_ms.dxil");
    auto psRed = testutil::loadShaderBytes("mesh_solid_red_ps.dxil");

    auto makeStencilDSS = [](D3D12_COMPARISON_FUNC cmp) {
        CD3DX12_DEPTH_STENCIL_DESC1 dss(D3D12_DEFAULT);
        dss.DepthEnable = FALSE;
        dss.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ZERO;
        dss.StencilEnable = TRUE;
        dss.StencilReadMask = 0xFF;
        dss.StencilWriteMask = 0xFF;
        dss.FrontFace.StencilFunc = cmp;
        dss.BackFace.StencilFunc = cmp;
        return dss;
    };

    auto psoLess = createMeshStreamPSO(
        rt.device.Get(), rs, nullptr, msFull, psRed,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        makeStencilDSS(D3D12_COMPARISON_FUNC_LESS));
    auto psoEqual = createMeshStreamPSO(
        rt.device.Get(), rs, nullptr, msFull, psRed,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        makeStencilDSS(D3D12_COMPARISON_FUNC_EQUAL));

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap, dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "CreateDSVHeap");

    auto rtTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    auto dsTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtTex.Get(), nullptr, rtvH);
    rt.device->CreateDepthStencilView(dsTex.Get(), nullptr, dsvH);

    ComPtr<ID3D12GraphicsCommandList6> cmd6;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmd6)), "QI CmdList6");
    D3D12_VIEWPORT vp{0, 0, 1.0f, 1.0f, 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, 1, 1};
    const float clearCol[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Case 1: StencilFunc = LESS with default StencilRef (0) against cleared stencil (1) -> PASSES -> [1, 0, 0, 1]
    rt.resetCmd();
    cmd6->SetGraphicsRootSignature(rs);
    cmd6->RSSetViewports(1, &vp);
    cmd6->RSSetScissorRects(1, &sc);
    cmd6->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    cmd6->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 1, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
    cmd6->SetPipelineState(psoLess.Get());
    cmd6->DispatchMesh(1, 1, 1);
    rt.executeAndWait();

    auto pxLess = readbackFloat4Tex2D(rt, rtTex.Get(), W, H);
    assert(pxLess[0] == 1.0f && pxLess[1] == 0.0f && pxLess[2] == 0.0f && pxLess[3] == 1.0f);

    // Case 2: StencilFunc = EQUAL with default StencilRef (0) against cleared stencil (1) -> FAILS -> [0, 0, 0, 0]
    rt.resetCmd();
    cmd6->SetGraphicsRootSignature(rs);
    cmd6->RSSetViewports(1, &vp);
    cmd6->RSSetScissorRects(1, &sc);
    cmd6->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    cmd6->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 1, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
    cmd6->SetPipelineState(psoEqual.Get());
    cmd6->DispatchMesh(1, 1, 1);
    rt.executeAndWait();

    auto pxEqualReject = readbackFloat4Tex2D(rt, rtTex.Get(), W, H);
    assert(pxEqualReject[0] == 0.0f && pxEqualReject[1] == 0.0f && pxEqualReject[2] == 0.0f && pxEqualReject[3] == 0.0f);

    // Case 3: StencilFunc = EQUAL with OMSetStencilRef(1) -> PASSES -> [1, 0, 0, 1]
    rt.resetCmd();
    cmd6->SetGraphicsRootSignature(rs);
    cmd6->RSSetViewports(1, &vp);
    cmd6->RSSetScissorRects(1, &sc);
    cmd6->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    cmd6->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 1, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
    cmd6->OMSetStencilRef(1);
    cmd6->SetPipelineState(psoEqual.Get());
    cmd6->DispatchMesh(1, 1, 1);
    rt.executeAndWait();

    auto pxEqualPass = readbackFloat4Tex2D(rt, rtTex.Get(), W, H);
    assert(pxEqualPass[0] == 1.0f && pxEqualPass[1] == 0.0f && pxEqualPass[2] == 0.0f && pxEqualPass[3] == 1.0f);
    std::printf("  StencilState (Less=PASS, Equal(ref=0)=REJECT, Equal(ref=1)=PASS) PASSED\n");
}

// Covers OffloadTest MSAA/MeshShaderTriangle.test (4x MSAA mesh PSO + ResolveSubresource)
void testMsaaMeshShaderTriangle(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 256, H = 256;
    auto msSimple = testutil::loadShaderBytes("mesh_simple_triangle_ms.dxil");
    auto psSat = testutil::loadShaderBytes("mesh_saturate_color_ps.dxil");

    CD3DX12_DEPTH_STENCIL_DESC1 dss(D3D12_DEFAULT);
    dss.DepthEnable = FALSE;
    dss.StencilEnable = FALSE;

    auto pso = createMeshStreamPSO(
        rt.device.Get(), rs, nullptr, msSimple, psSat,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, dss,
        D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE, /*sampleCount=*/4);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap, dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "CreateDSVHeap");

    auto msaaRT = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    auto msaaDS = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 4, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    auto resolvedRT = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D12_RESOURCE_FLAG_NONE);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(msaaRT.Get(), nullptr, rtvH);
    rt.device->CreateDepthStencilView(msaaDS.Get(), nullptr, dsvH);

    ComPtr<ID3D12GraphicsCommandList6> cmd6;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmd6)), "QI CmdList6");
    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    rt.resetCmd();
    cmd6->SetGraphicsRootSignature(rs);
    cmd6->RSSetViewports(1, &vp);
    cmd6->RSSetScissorRects(1, &sc);
    cmd6->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    cmd6->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
    cmd6->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
    cmd6->SetPipelineState(pso.Get());
    cmd6->DispatchMesh(1, 1, 1);
    cmd6->ResolveSubresource(resolvedRT.Get(), 0, msaaRT.Get(), 0, DXGI_FORMAT_R32G32B32A32_FLOAT);
    rt.executeAndWait();

    auto px = readbackFloat4Tex2D(rt, resolvedRT.Get(), W, H);
    const float* mid = &px[(128 * W + 128) * 4];
    assert(std::fabs(mid[3] - 1.0f) < 1e-4f);
    assert(std::fabs(mid[0] - 0.5f) < 0.03f);

    UINT partialCoveragePixels = 0;
    for (UINT i = 0; i < W * H; ++i) {
        const float a = px[static_cast<size_t>(i) * 4 + 3];
        if (a > 0.1f && a < 0.9f) {
            ++partialCoveragePixels;
        }
    }
    assert(partialCoveragePixels >= 20);
    std::printf("  MsaaMeshShaderTriangle (4x MSAA partialEdgePixels=%u) PASSED\n", partialCoveragePixels);
}

// Covers multi-wave MS barrier, per-primitive attributes, SV_CullPrimitive, and ExecuteIndirect(DISPATCH_MESH)
void testBarrierPerPrimCullAndExecuteIndirect(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 64, H = 64;
    auto msBarrier = testutil::loadShaderBytes("mesh_barrier_prim_ms.dxil");
    auto psPrim = testutil::loadShaderBytes("mesh_prim_attr_ps.dxil");

    CD3DX12_DEPTH_STENCIL_DESC1 dss(D3D12_DEFAULT);
    dss.DepthEnable = FALSE;
    dss.StencilEnable = FALSE;

    auto pso = createMeshStreamPSO(
        rt.device.Get(), rs, nullptr, msBarrier, psPrim,
        DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_UNKNOWN, dss);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");

    auto rtTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R32G32B32A32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtTex.Get(), nullptr, rtvH);

    D3D12_INDIRECT_ARGUMENT_DESC indirectArg{};
    indirectArg.Type = D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH;
    D3D12_COMMAND_SIGNATURE_DESC sigDesc{};
    sigDesc.ByteStride = sizeof(D3D12_DISPATCH_MESH_ARGUMENTS);
    sigDesc.NumArgumentDescs = 1;
    sigDesc.pArgumentDescs = &indirectArg;
    ComPtr<ID3D12CommandSignature> cmdSig;
    testutil::checkHR(rt.device->CreateCommandSignature(&sigDesc, nullptr, IID_PPV_ARGS(&cmdSig)), "CreateCommandSignature DispatchMesh");

    const D3D12_DISPATCH_MESH_ARGUMENTS args{2, 1, 1};
    auto argBuf = rt.createBuffer(sizeof(args), D3D12_HEAP_TYPE_DEFAULT);
    rt.uploadBuffer(argBuf.Get(), &args, sizeof(args));

    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->ExecuteIndirect(cmdSig.Get(), 1, argBuf.Get(), 0, nullptr, 0);
    rt.executeAndWait();

    auto px = readbackFloat4Tex2D(rt, rtTex.Get(), W, H);
    auto getPx = [&](UINT x, UINT y) -> const float* {
        return &px[(static_cast<size_t>(y) * W + x) * 4];
    };

    // Group 0 primitive 0 (x in [0..15]) was culled via SV_CullPrimitive = true
    const float* g0Culled = getPx(4, 8);
    assert(g0Culled[0] == 0.0f && g0Culled[1] == 0.0f && g0Culled[2] == 0.0f && g0Culled[3] == 0.0f);

    // Group 0 primitive 1 (x in [16..31], upper-right corner (28, 8)) = (0.25, 0.5, 0.75, 0) + (0, 1, 0.25, 1)
    const float* g0Kept = getPx(28, 8);
    assert(std::fabs(g0Kept[0] - 0.25f) < 1e-4f);
    assert(std::fabs(g0Kept[1] - 1.50f) < 1e-4f);
    assert(std::fabs(g0Kept[2] - 1.00f) < 1e-4f);
    assert(std::fabs(g0Kept[3] - 1.00f) < 1e-4f);

    // Group 1 primitive 0 (x in [32..47]) was culled via SV_CullPrimitive = true
    const float* g1Culled = getPx(36, 8);
    assert(g1Culled[0] == 0.0f && g1Culled[1] == 0.0f && g1Culled[2] == 0.0f && g1Culled[3] == 0.0f);

    // Group 1 primitive 1 (x in [48..63], upper-right corner (60, 8)) = (0.25, 0.5, 0.75, 0) + (0.25, 0, 1.0, 1)
    const float* g1Kept = getPx(60, 8);
    assert(std::fabs(g1Kept[0] - 0.50f) < 1e-4f);
    assert(std::fabs(g1Kept[1] - 0.50f) < 1e-4f);
    assert(std::fabs(g1Kept[2] - 1.75f) < 1e-4f);
    assert(std::fabs(g1Kept[3] - 1.00f) < 1e-4f);

    std::printf("  BarrierPerPrimCullAndExecuteIndirect PASSED\n");
}

} // namespace

int main() {
    std::printf("=== mesh_shader_test ===\n");
    testutil::TestRuntime rt;
    rt.init();

    D3D12_ROOT_SIGNATURE_DESC emptyRS{};
    auto rs = rt.createRootSig(emptyRS);

    testMeshShaderFeatureSupport(rt);
    testSimpleTriangleAndAmplification(rt, rs.Get());
    testSimpleLines(rt, rs.Get());
    testStencilState(rt, rs.Get());
    testMsaaMeshShaderTriangle(rt, rs.Get());
    testBarrierPerPrimCullAndExecuteIndirect(rt, rs.Get());

    std::printf("=== ALL MESH SHADER TESTS PASSED ===\n");
    return 0;
}
