#include "test_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

ComPtr<ID3D12Resource> createTex2D(
    ID3D12Device* dev,
    UINT w,
    UINT h,
    UINT16 mipLevels,
    DXGI_FORMAT fmt,
    D3D12_RESOURCE_FLAGS flags,
    const D3D12_CLEAR_VALUE* clearVal) {
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Tex2D(fmt, w, h, 1, mipLevels, 1, 0, flags);
    ComPtr<ID3D12Resource> res;
    testutil::checkHR(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, clearVal, IID_PPV_ARGS(&res)),
                      "CreateTex2D");
    return res;
}

std::vector<uint8_t> readbackTex2D(testutil::TestRuntime& rt, ID3D12Resource* tex, UINT w, UINT h, UINT bpp) {
    UINT rowPitch = (w * bpp + 255u) & ~255u;
    auto rb = rt.createBuffer(static_cast<UINT64>(rowPitch) * h, D3D12_HEAP_TYPE_READBACK);
    rt.resetCmd();
    D3D12_TEXTURE_COPY_LOCATION dst{};
    dst.pResource = rb.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format = tex->GetDesc().Format;
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
    std::vector<uint8_t> out(static_cast<size_t>(w) * h * bpp);
    const auto* p = static_cast<const uint8_t*>(mapped);
    for (UINT y = 0; y < h; ++y) {
        std::memcpy(out.data() + static_cast<size_t>(y) * w * bpp, p + static_cast<size_t>(y) * rowPitch, w * bpp);
    }
    rb->Unmap(0, nullptr);
    return out;
}

D3D12_GRAPHICS_PIPELINE_STATE_DESC makeBasePSO(
    ID3D12RootSignature* rs,
    const std::vector<uint8_t>& vs,
    const std::vector<uint8_t>& ps) {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = rs;
    pso.VS = {vs.data(), vs.size()};
    pso.PS = {ps.data(), ps.size()};
    pso.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    pso.SampleMask = UINT_MAX;
    pso.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    pso.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    pso.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    pso.DepthStencilState.DepthEnable = FALSE;
    pso.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pso.NumRenderTargets = 1;
    pso.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    pso.SampleDesc = {1, 0};
    return pso;
}

void testBarycentrics(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 32, H = 32;
    auto vs = testutil::loadShaderBytes("gfx_bary_vs.dxil");
    auto psPersp = testutil::loadShaderBytes("gfx_bary_persp_ps.dxil");
    auto psNoPersp = testutil::loadShaderBytes("gfx_bary_nopersp_ps.dxil");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = makeBasePSO(rs, vs, psPersp);
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    ComPtr<ID3D12PipelineState> psoPersp, psoNoPersp;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&psoPersp)), "CreatePSO BaryPersp");
    psoDesc.PS = {psNoPersp.data(), psNoPersp.size()};
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&psoNoPersp)), "CreatePSO BaryNoPersp");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    UINT rtvStep = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

    auto rtvPerspTex = createTex2D(rt.device.Get(), W, H, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    auto rtvNoPerspTex = createTex2D(rt.device.Get(), W, H, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH0 = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH1{rtvH0.ptr + rtvStep};
    rt.device->CreateRenderTargetView(rtvPerspTex.Get(), nullptr, rtvH0);
    rt.device->CreateRenderTargetView(rtvNoPerspTex.Get(), nullptr, rtvH1);

    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0, 0, 0, 0};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    rt.cmdList->ClearRenderTargetView(rtvH0, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH0, FALSE, nullptr);
    rt.cmdList->SetPipelineState(psoPersp.Get());
    rt.cmdList->DrawInstanced(3, 1, 0, 0);

    rt.cmdList->ClearRenderTargetView(rtvH1, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH1, FALSE, nullptr);
    rt.cmdList->SetPipelineState(psoNoPersp.Get());
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.executeAndWait();

    auto bytesP = readbackTex2D(rt, rtvPerspTex.Get(), W, H, 16);
    auto bytesNP = readbackTex2D(rt, rtvNoPerspTex.Get(), W, H, 16);
    float pxP[4], pxNP[4];
    size_t midIdx = (16 * W + 16) * 16;
    std::memcpy(pxP, bytesP.data() + midIdx, 16);
    std::memcpy(pxNP, bytesNP.data() + midIdx, 16);

    float sumP = pxP[0] + pxP[1] + pxP[2];
    float sumNP = pxNP[0] + pxNP[1] + pxNP[2];
    assert(std::fabs(sumP - 1.0f) < 1e-3f);
    assert(std::fabs(sumNP - 1.0f) < 1e-3f);
    // Non-uniform W=(1,2,4) must produce distinct perspective vs noperspective barycentrics at screen center.
    assert(std::fabs(pxP[0] - pxNP[0]) > 0.05f);
    std::printf("  Barycentrics (persp=(%.3f,%.3f,%.3f) vs nopersp=(%.3f,%.3f,%.3f)) PASSED\n",
                pxP[0], pxP[1], pxP[2], pxNP[0], pxNP[1], pxNP[2]);
}

void testShaderStencilRef(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 32, H = 32;
    auto vs = testutil::loadShaderBytes("gfx_fullscreen_vs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_stencil_ref_ps.dxil");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = makeBasePSO(rs, vs, ps);
    psoDesc.DSVFormat = DXGI_FORMAT_D24_UNORM_S8_UINT;
    psoDesc.DepthStencilState.StencilEnable = TRUE;
    psoDesc.DepthStencilState.StencilReadMask = 0xFF;
    psoDesc.DepthStencilState.StencilWriteMask = 0xFF;
    psoDesc.DepthStencilState.FrontFace.StencilFunc = D3D12_COMPARISON_FUNC_EQUAL;
    psoDesc.DepthStencilState.FrontFace.StencilPassOp = D3D12_STENCIL_OP_REPLACE;
    psoDesc.DepthStencilState.FrontFace.StencilFailOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.FrontFace.StencilDepthFailOp = D3D12_STENCIL_OP_KEEP;
    psoDesc.DepthStencilState.BackFace = psoDesc.DepthStencilState.FrontFace;

    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreatePSO StencilRef");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap, dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "CreateDSVHeap");

    D3D12_CLEAR_VALUE clearRTV{};
    clearRTV.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    auto rtvTex = createTex2D(rt.device.Get(), W, H, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearRTV);
    D3D12_CLEAR_VALUE clearDSV{};
    clearDSV.Format = DXGI_FORMAT_D24_UNORM_S8_UINT;
    clearDSV.DepthStencil.Depth = 1.0f;
    clearDSV.DepthStencil.Stencil = 7;
    auto dsvTex = createTex2D(rt.device.Get(), W, H, 1, DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, &clearDSV);

    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);
    rt.device->CreateDepthStencilView(dsvTex.Get(), nullptr, dsvH);

    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0, 0, 0, 1};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 7, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
    // OMSetStencilRef is intentionally different (1); PS SV_StencilRef (7 on left, 3 on right) must drive comparison.
    rt.cmdList->OMSetStencilRef(1);
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
    size_t leftIdx = (16 * W + 4) * 4;
    size_t rightIdx = (16 * W + 24) * 4;
    // Left half: SV_StencilRef == 7 == DSV stencil -> passes and writes color (64, 191, 128, 255)
    assert(rgba[leftIdx + 1] > 180);
    // Right half: SV_StencilRef == 3 != DSV stencil (7) -> fails and keeps clear color (0, 0, 0, 255)
    assert(rgba[rightIdx + 1] == 0);
    std::printf("  PS-Specified StencilRef PASSED\n");
}

void testShadingRate(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 16, H = 16;
    auto vs = testutil::loadShaderBytes("gfx_fullscreen_vs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_shading_rate_ps.dxil");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = makeBasePSO(rs, vs, ps);
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreatePSO ShadingRate");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    auto rtvTex = createTex2D(rt.device.Get(), W, H, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

    D3D12_VIEWPORT vp{0, 0, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    const float clearCol[4] = {0, 0, 0, 1};

    ComPtr<ID3D12GraphicsCommandList5> cmd5;
    testutil::checkHR(rt.cmdList.As(&cmd5), "QI ID3D12GraphicsCommandList5");

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    cmd5->RSSetShadingRate(D3D12_SHADING_RATE_2X2, nullptr);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
    // D3D12_SHADING_RATE_2X2 == 5 -> 5/15 * 255 == 85
    assert(rgba[0] >= 84 && rgba[0] <= 86);
    std::printf("  Variable Rate Shading (SV_ShadingRate=%u/15) PASSED\n", rgba[0]);
}

void testMultiViewport(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    constexpr UINT W = 32, H = 16;
    auto vs = testutil::loadShaderBytes("gfx_multivp_vs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_multivp_ps.dxil");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = makeBasePSO(rs, vs, ps);
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreatePSO MultiVP");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    auto rtvTex = createTex2D(rt.device.Get(), W, H, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

    D3D12_VIEWPORT vps[2] = {
        {0.0f, 0.0f, 16.0f, 16.0f, 0.0f, 1.0f},
        {16.0f, 0.0f, 16.0f, 16.0f, 0.0f, 1.0f},
    };
    D3D12_RECT scs[2] = {
        {0, 0, 16, 16},
        {16, 0, 32, 16},
    };
    const float clearCol[4] = {0, 0, 0, 1};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->RSSetViewports(2, vps);
    rt.cmdList->RSSetScissorRects(2, scs);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->DrawInstanced(6, 1, 0, 0);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
    size_t leftIdx = (8 * W + 8) * 4;
    size_t rightIdx = (8 * W + 24) * 4;
    // Viewport 0 (left): red
    assert(rgba[leftIdx + 0] == 255 && rgba[leftIdx + 1] == 0);
    // Viewport 1 (right): green
    assert(rgba[rightIdx + 0] == 0 && rgba[rightIdx + 1] == 255);
    std::printf("  Multi-Viewport (SV_ViewportArrayIndex) PASSED\n");
}

void testTextureLodClamp(testutil::TestRuntime& rt) {
    D3D12_DESCRIPTOR_RANGE srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 1;
    rp.DescriptorTable.pDescriptorRanges = &srvRange;
    rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_STATIC_SAMPLER_DESC samp{};
    samp.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.MaxLOD = D3D12_FLOAT32_MAX;
    samp.ShaderRegister = 0;
    samp.ShaderVisibility = D3D12_SHADER_VISIBILITY_PIXEL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &rp;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &samp;
    auto rs = rt.createRootSig(rsDesc);

    auto vs = testutil::loadShaderBytes("gfx_fullscreen_vs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_lod_clamp_ps.dxil");
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc = makeBasePSO(rs.Get(), vs, ps);
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreatePSO LodClamp");

    // 32x32 texture with 3 mip levels: mip 0 = red, mip 1 = green, mip 2 = blue
    auto tex = createTex2D(rt.device.Get(), 32, 32, 3, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, nullptr);
    auto uploadMip = [&](UINT subIdx, UINT sz, uint8_t r, uint8_t g, uint8_t b) {
        UINT rowPitch = (sz * 4 + 255u) & ~255u;
        auto up = rt.createBuffer(static_cast<UINT64>(rowPitch) * sz, D3D12_HEAP_TYPE_UPLOAD);
        void* p = nullptr;
        testutil::checkHR(up->Map(0, nullptr, &p), "Map upload");
        auto* dst = static_cast<uint8_t*>(p);
        for (UINT y = 0; y < sz; ++y) {
            for (UINT x = 0; x < sz; ++x) {
                dst[y * rowPitch + x * 4 + 0] = r;
                dst[y * rowPitch + x * 4 + 1] = g;
                dst[y * rowPitch + x * 4 + 2] = b;
                dst[y * rowPitch + x * 4 + 3] = 255;
            }
        }
        up->Unmap(0, nullptr);

        rt.resetCmd();
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = tex.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = subIdx;

        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = up.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srcLoc.PlacedFootprint.Footprint.Width = sz;
        srcLoc.PlacedFootprint.Footprint.Height = sz;
        srcLoc.PlacedFootprint.Footprint.Depth = 1;
        srcLoc.PlacedFootprint.Footprint.RowPitch = rowPitch;
        rt.cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        rt.executeAndWait();
    };
    uploadMip(0, 32, 255, 0, 0);
    uploadMip(1, 16, 0, 255, 0);
    uploadMip(2, 8, 0, 0, 255);

    D3D12_DESCRIPTOR_HEAP_DESC srvHD{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> srvHeap, rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&srvHD, IID_PPV_ARGS(&srvHeap)), "CreateSRVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Texture2D.MipLevels = 3;
    rt.device->CreateShaderResourceView(tex.Get(), &srvDesc, srvHeap->GetCPUDescriptorHandleForHeapStart());

    auto rtvTex = createTex2D(rt.device.Get(), 32, 32, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

    D3D12_VIEWPORT vp{0, 0, 32.0f, 32.0f, 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, 32, 32};
    const float clearCol[4] = {0, 0, 0, 1};
    ID3D12DescriptorHeap* heaps[] = {srvHeap.Get()};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetGraphicsRootDescriptorTable(0, srvHeap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, rtvTex.Get(), 32, 32, 4);
    size_t midIdx = (16 * 32 + 16) * 4;
    // MinLodClamp = 2.0 forces mip 2 (blue) instead of mip 0 (red)
    assert(rgba[midIdx + 0] == 0 && rgba[midIdx + 1] == 0 && rgba[midIdx + 2] == 255);
    std::printf("  Texture LOD Clamp (SampleBias MinLodClamp=2.0) PASSED\n");
}

void testTessellationPointTopologyWithGS(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    auto vs = testutil::loadShaderBytes("gfx_tess_point_vs.dxil");
    auto hs = testutil::loadShaderBytes("gfx_tess_point_hs.dxil");
    auto ds = testutil::loadShaderBytes("gfx_tess_point_ds.dxil");
    auto gs = testutil::loadShaderBytes("gfx_tess_point_gs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_tess_point_ps.dxil");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs;
    psoDesc.VS = {vs.data(), vs.size()};
    psoDesc.HS = {hs.data(), hs.size()};
    psoDesc.DS = {ds.data(), ds.size()};
    psoDesc.GS = {gs.data(), gs.size()};
    psoDesc.PS = {ps.data(), ps.size()};
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc = {1, 0};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateTessPointGSPSO");

    auto rtvTex = createTex2D(rt.device.Get(), 32, 32, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

    D3D12_VIEWPORT vp{0, 0, 32.0f, 32.0f, 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, 32, 32};
    const float clearCol[4] = {0, 0, 0, 1};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, rtvTex.Get(), 32, 32, 4);
    bool anyGreen = false;
    for (size_t i = 0; i < rgba.size(); i += 4) {
        if (rgba[i + 1] == 255 && rgba[i + 0] == 0) {
            anyGreen = true;
            break;
        }
    }
    assert(anyGreen);
    std::printf("  Tessellation point topology with GS expansion PASSED\n");
}

void testByteAddressBufferLoad4InVS(testutil::TestRuntime& rt) {
    auto vs = testutil::loadShaderBytes("gfx_bab_load4_vs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_bab_load4_ps.dxil");

    D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0};
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {1, &range};
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_VERTEX;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    auto rs = rt.createRootSig(rsDesc);

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.VS = {vs.data(), vs.size()};
    psoDesc.PS = {ps.data(), ps.size()};
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc = {1, 0};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateBABLoad4PSO");

    const float colorData[4] = {0.25f, 0.50f, 0.75f, 1.0f};
    auto buf = rt.createBuffer(sizeof(colorData), D3D12_HEAP_TYPE_UPLOAD);
    void* mapped = nullptr;
    testutil::checkHR(buf->Map(0, nullptr, &mapped), "MapBAB");
    std::memcpy(mapped, colorData, sizeof(colorData));
    buf->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC srvHD{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> srvHeap, rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&srvHD, IID_PPV_ARGS(&srvHeap)), "CreateSRVHeap");
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.Format = DXGI_FORMAT_R32_TYPELESS;
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.Buffer.FirstElement = 0;
    srvDesc.Buffer.NumElements = 4;
    srvDesc.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    rt.device->CreateShaderResourceView(buf.Get(), &srvDesc, srvHeap->GetCPUDescriptorHandleForHeapStart());

    auto rtvTex = createTex2D(rt.device.Get(), 16, 16, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, nullptr);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

    D3D12_VIEWPORT vp{0, 0, 16.0f, 16.0f, 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, 16, 16};
    const float clearCol[4] = {0, 0, 0, 0};
    ID3D12DescriptorHeap* heaps[] = {srvHeap.Get()};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetGraphicsRootDescriptorTable(0, srvHeap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, rtvTex.Get(), 16, 16, 4);
    size_t midIdx = (8 * 16 + 8) * 4;
    assert(std::abs(static_cast<int>(rgba[midIdx + 0]) - 64) <= 1);
    assert(std::abs(static_cast<int>(rgba[midIdx + 1]) - 128) <= 1);
    assert(std::abs(static_cast<int>(rgba[midIdx + 2]) - 191) <= 1);
    assert(rgba[midIdx + 3] == 255);
    std::printf("  ByteAddressBuffer Load4 in VS PASSED\n");
}

void testMultisample4xRenderAndResolve(testutil::TestRuntime& rt, ID3D12RootSignature* rs) {
    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS msaaCap{};
    msaaCap.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    msaaCap.SampleCount = 4;
    testutil::checkHR(rt.device->CheckFeatureSupport(
        D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &msaaCap, sizeof(msaaCap)), "CheckMSAAQuality");
    assert(msaaCap.NumQualityLevels >= 1);

    D3D12_FEATURE_DATA_FORMAT_SUPPORT fmtCap{};
    fmtCap.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    testutil::checkHR(rt.device->CheckFeatureSupport(
        D3D12_FEATURE_FORMAT_SUPPORT, &fmtCap, sizeof(fmtCap)), "CheckFormatSupport");
    assert((fmtCap.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) != 0);
    assert((fmtCap.Support1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE) != 0);

    auto vs = testutil::loadShaderBytes("gfx_msaa_diag_vs.dxil");
    auto ps = testutil::loadShaderBytes("gfx_msaa_diag_ps.dxil");

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs;
    psoDesc.VS = {vs.data(), vs.size()};
    psoDesc.PS = {ps.data(), ps.size()};
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.MultisampleEnable = TRUE;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleDesc = {4, 0};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateMSAA4xPSO");

    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC msaaDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R8G8B8A8_UNORM, 32, 32, 1, 1, 4, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    ComPtr<ID3D12Resource> msaaTex;
    testutil::checkHR(rt.device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &msaaDesc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&msaaTex)),
        "CreateMSAA4xRT");

    auto resolveTex = createTex2D(rt.device.Get(), 32, 32, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "CreateRTVHeap");
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(msaaTex.Get(), nullptr, rtvH);

    D3D12_VIEWPORT vp{0, 0, 32.0f, 32.0f, 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, 32, 32};
    const float clearCol[4] = {0, 0, 0, 0};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs);
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    rt.cmdList->DrawInstanced(3, 1, 0, 0);
    rt.cmdList->ResolveSubresource(resolveTex.Get(), 0, msaaTex.Get(), 0, DXGI_FORMAT_R8G8B8A8_UNORM);
    rt.executeAndWait();

    auto rgba = readbackTex2D(rt, resolveTex.Get(), 32, 32, 4);
    // Top-right interior (28, 4) is fully covered -> (255, 0, 0, 255)
    size_t insideIdx = (4 * 32 + 28) * 4;
    assert(rgba[insideIdx + 0] == 255 && rgba[insideIdx + 3] == 255);
    // Bottom-left (4, 28) is uncovered -> (0, 0, 0, 0)
    size_t outsideIdx = (28 * 32 + 4) * 4;
    assert(rgba[outsideIdx + 0] == 0 && rgba[outsideIdx + 3] == 0);
    // Diagonal edge pixel (16, 16) has partial 4x MSAA coverage -> 0 < A < 255
    size_t edgeIdx = (16 * 32 + 16) * 4;
    assert(rgba[edgeIdx + 3] > 0 && rgba[edgeIdx + 3] < 255);
    std::printf("  4x MSAA Render and ResolveSubresource PASSED\n");
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    D3D12_ROOT_SIGNATURE_DESC emptyRsDesc{};
    auto rs = rt.createRootSig(emptyRsDesc);
    testBarycentrics(rt, rs.Get());
    testShaderStencilRef(rt, rs.Get());
    testShadingRate(rt, rs.Get());
    testMultiViewport(rt, rs.Get());
    testTextureLodClamp(rt);
    testTessellationPointTopologyWithGS(rt, rs.Get());
    testByteAddressBufferLoad4InVS(rt);
    testMultisample4xRenderAndResolve(rt, rs.Get());

    std::printf("graphics_features_test PASSED\n");
    return 0;
}
