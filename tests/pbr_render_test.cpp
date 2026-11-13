#include "test_common.h"

#include <cstring>
#include <fstream>
#include <vector>

namespace {

struct PbrVertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
};

struct LightCB {
    float lightDir[4];
    float baseTint[4];
};

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    auto vsBytes = testutil::loadShaderBytes("pbr_mrt_vs.dxil");
    auto psBytes = testutil::loadShaderBytes("pbr_mrt_ps.dxil");

    // Root signature: b0 (CBV), descriptor table [t0 (SRV)], static sampler s0
    D3D12_DESCRIPTOR_RANGE srvRange{};
    srvRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srvRange.NumDescriptors = 1;
    srvRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER rp[2]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp[1].DescriptorTable.NumDescriptorRanges = 1;
    rp[1].DescriptorTable.pDescriptorRanges = &srvRange;

    D3D12_STATIC_SAMPLER_DESC samp{};
    samp.Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_WRAP;
    samp.ShaderRegister = 0;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = rp;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &samp;

    ComPtr<ID3DBlob> rsBlob;
    testutil::checkHR(NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(rt.device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");

    D3D12_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"NORMAL", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT, 0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.VS = {vsBytes.data(), vsBytes.size()};
    psoDesc.PS = {psBytes.data(), psBytes.size()};
    psoDesc.InputLayout = {layout, 3};
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 2;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.RTVFormats[1] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
    psoDesc.SampleMask = UINT32_MAX;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreatePbrPSO");

    constexpr UINT W = 256, H = 256;
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rtvDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    ComPtr<ID3D12Resource> rtv0, rtv1, dsv;
    testutil::checkHR(rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rtvDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rtv0)), "RTV0");
    testutil::checkHR(rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rtvDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rtv1)), "RTV1");
    D3D12_RESOURCE_DESC dsvDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT, W, H, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
    testutil::checkHR(rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &dsvDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&dsv)), "DSV");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "RTVHeap");
    UINT rtvInc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH0 = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH1{rtvH0.ptr + rtvInc};
    rt.device->CreateRenderTargetView(rtv0.Get(), nullptr, rtvH0);
    rt.device->CreateRenderTargetView(rtv1.Get(), nullptr, rtvH1);

    D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "DSVHeap");
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateDepthStencilView(dsv.Get(), nullptr, dsvH);

    // Checkerboard 16x16 albedo texture
    constexpr UINT TW = 16, TH = 16;
    D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, TW, TH, 1, 1);
    ComPtr<ID3D12Resource> albedoTex;
    testutil::checkHR(rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &texDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&albedoTex)), "AlbedoTex");

    UINT texRowPitch = 256;
    std::vector<uint8_t> texUpload(texRowPitch * TH, 0);
    for (UINT y = 0; y < TH; ++y) {
        for (UINT x = 0; x < TW; ++x) {
            bool checker = ((x / 4 + y / 4) & 1u) != 0u;
            uint8_t* p = texUpload.data() + y * texRowPitch + x * 4;
            p[0] = checker ? 235 : 50;
            p[1] = checker ? 180 : 120;
            p[2] = checker ? 90 : 210;
            p[3] = 255;
        }
    }
    auto texUpBuf = rt.createBuffer(texUpload.size(), D3D12_HEAP_TYPE_UPLOAD);
    void* upPtr = nullptr;
    texUpBuf->Map(0, nullptr, &upPtr);
    std::memcpy(upPtr, texUpload.data(), texUpload.size());
    texUpBuf->Unmap(0, nullptr);

    D3D12_DESCRIPTOR_HEAP_DESC srvHD{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> srvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&srvHD, IID_PPV_ARGS(&srvHeap)), "SRVHeap");
    rt.device->CreateShaderResourceView(albedoTex.Get(), nullptr, srvHeap->GetCPUDescriptorHandleForHeapStart());

    auto cb = rt.createBuffer(256, D3D12_HEAP_TYPE_DEFAULT);
    LightCB lcb{{0.5f, 0.7f, -0.5f, 0.0f}, {1.0f, 0.95f, 0.9f, 1.0f}};
    rt.uploadBuffer(cb.Get(), &lcb, sizeof(lcb));

    PbrVertex verts[4] = {
        {-0.75f, 0.75f, 0.3f, 0.0f, 0.3f, -0.95f, 0.0f, 0.0f},
        {0.75f, 0.75f, 0.3f, 0.3f, 0.3f, -0.90f, 1.0f, 0.0f},
        {0.75f, -0.75f, 0.3f, 0.0f, -0.3f, -0.95f, 1.0f, 1.0f},
        {-0.75f, -0.75f, 0.3f, -0.3f, 0.0f, -0.95f, 0.0f, 1.0f},
    };
    uint32_t indices[6] = {0, 1, 2, 0, 2, 3};
    auto vb = rt.createBuffer(sizeof(verts), D3D12_HEAP_TYPE_DEFAULT);
    auto ib = rt.createBuffer(sizeof(indices), D3D12_HEAP_TYPE_DEFAULT);
    rt.uploadBuffer(vb.Get(), verts, sizeof(verts));
    rt.uploadBuffer(ib.Get(), indices, sizeof(indices));

    rt.resetCmd();
    D3D12_TEXTURE_COPY_LOCATION copyDst{albedoTex.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION copySrc{};
    copySrc.pResource = texUpBuf.Get();
    copySrc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    copySrc.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, TW, TH, 1, texRowPitch};
    rt.cmdList->CopyTextureRegion(&copyDst, 0, 0, 0, &copySrc, nullptr);

    ID3D12DescriptorHeap* heaps[] = {srvHeap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetGraphicsRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetGraphicsRootConstantBufferView(0, cb->GetGPUVirtualAddress());
    rt.cmdList->SetGraphicsRootDescriptorTable(1, srvHeap->GetGPUDescriptorHandleForHeapStart());

    D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvs[2] = {rtvH0, rtvH1};
    rt.cmdList->OMSetRenderTargets(2, rtvs, FALSE, &dsvH);
    const float bg0[4] = {0.05f, 0.08f, 0.14f, 1.0f};
    const float bg1[4] = {0.0f, 0.0f, 0.0f, 1.0f};
    rt.cmdList->ClearRenderTargetView(rtvH0, bg0, 0, nullptr);
    rt.cmdList->ClearRenderTargetView(rtvH1, bg1, 0, nullptr);
    rt.cmdList->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);

    D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(PbrVertex)};
    D3D12_INDEX_BUFFER_VIEW ibv{ib->GetGPUVirtualAddress(), sizeof(indices), DXGI_FORMAT_R32_UINT};
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
    rt.cmdList->IASetIndexBuffer(&ibv);
    rt.cmdList->DrawIndexedInstanced(6, 1, 0, 0, 0);
    rt.executeAndWait();

    // Readback RTV0 and write pbr_output.ppm
    UINT rbPitch = (W * 4 + 255u) & ~255u;
    auto rb = rt.createBuffer(static_cast<UINT64>(rbPitch) * H, D3D12_HEAP_TYPE_READBACK);
    rt.resetCmd();
    D3D12_TEXTURE_COPY_LOCATION rbDst{};
    rbDst.pResource = rb.Get();
    rbDst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    rbDst.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 1, rbPitch};
    D3D12_TEXTURE_COPY_LOCATION rbSrc{rtv0.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    rt.cmdList->CopyTextureRegion(&rbDst, 0, 0, 0, &rbSrc, nullptr);
    rt.executeAndWait();

    void* ptr = nullptr;
    rb->Map(0, nullptr, &ptr);
    const uint8_t* raw = static_cast<const uint8_t*>(ptr);
    std::ofstream ppm("pbr_output.ppm", std::ios::binary);
    ppm << "P6\n" << W << " " << H << "\n255\n";
    uint64_t nonBgPixels = 0;
    for (UINT y = 0; y < H; ++y) {
        const uint8_t* row = raw + static_cast<size_t>(y) * rbPitch;
        for (UINT x = 0; x < W; ++x) {
            uint8_t rgb[3] = {row[x * 4 + 0], row[x * 4 + 1], row[x * 4 + 2]};
            ppm.write(reinterpret_cast<const char*>(rgb), 3);
            if (rgb[0] > 30 || rgb[1] > 30 || rgb[2] > 50) ++nonBgPixels;
        }
    }
    rb->Unmap(0, nullptr);

    assert(nonBgPixels > 10000);
    std::printf("pbr_render_test PASSED (shaded_pixels=%llu, wrote pbr_output.ppm)\n",
                static_cast<unsigned long long>(nonBgPixels));
    return 0;
}
