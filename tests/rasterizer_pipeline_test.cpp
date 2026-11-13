#include "test_common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

struct VertexPosCol {
    float x, y, z;
    float r, g, b, a;
};

struct VertexPosOnly {
    float x, y, z;
};

ComPtr<ID3D12Resource> createTex2D(
    ID3D12Device* dev,
    UINT w,
    UINT h,
    DXGI_FORMAT fmt,
    D3D12_RESOURCE_FLAGS flags,
    const D3D12_CLEAR_VALUE* clearVal) {
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Tex2D(fmt, w, h, 1, 1, 1, 0, flags);
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

    void* ptr = nullptr;
    testutil::checkHR(rb->Map(0, nullptr, &ptr), "MapTexReadback");
    std::vector<uint8_t> packed(static_cast<size_t>(w) * h * bpp);
    for (UINT y = 0; y < h; ++y) {
        std::memcpy(packed.data() + static_cast<size_t>(y) * w * bpp,
                    static_cast<const uint8_t*>(ptr) + static_cast<size_t>(y) * rowPitch,
                    static_cast<size_t>(w) * bpp);
    }
    rb->Unmap(0, nullptr);
    return packed;
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    auto vsBytes = testutil::loadShaderBytes("raster_basic_vs.dxil");
    auto psBytes = testutil::loadShaderBytes("raster_basic_ps.dxil");
    auto shadowVsBytes = testutil::loadShaderBytes("shadow_depth_vs.dxil");

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    ComPtr<ID3DBlob> rsBlob;
    testutil::checkHR(NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(rt.device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHeapDesc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHeapDesc, IID_PPV_ARGS(&rtvHeap)), "RTVHeap");

    D3D12_DESCRIPTOR_HEAP_DESC dsvHeapDesc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHeapDesc, IID_PPV_ARGS(&dsvHeap)), "DSVHeap");

    // 1. Basic triangle + alpha blend test
    {
        constexpr UINT W = 128, H = 128;
        D3D12_CLEAR_VALUE clearC{};
        clearC.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        auto rtvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rs.Get();
        psoDesc.VS = {vsBytes.data(), vsBytes.size()};
        psoDesc.PS = {psBytes.data(), psBytes.size()};
        psoDesc.InputLayout = {layout, 2};
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleMask = UINT32_MAX;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
        psoDesc.BlendState.RenderTarget[0].BlendEnable = TRUE;
        psoDesc.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_SRC_ALPHA;
        psoDesc.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_INV_SRC_ALPHA;
        psoDesc.BlendState.RenderTarget[0].BlendOp = D3D12_BLEND_OP_ADD;
        psoDesc.BlendState.RenderTarget[0].SrcBlendAlpha = D3D12_BLEND_ONE;
        psoDesc.BlendState.RenderTarget[0].DestBlendAlpha = D3D12_BLEND_ZERO;
        psoDesc.BlendState.RenderTarget[0].BlendOpAlpha = D3D12_BLEND_OP_ADD;

        ComPtr<ID3D12PipelineState> pso;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateBasicGfxPSO");

        VertexPosCol verts[3] = {
            {0.0f, 0.7f, 0.5f, 1.0f, 0.0f, 0.0f, 0.5f},
            {0.7f, -0.7f, 0.5f, 0.0f, 1.0f, 0.0f, 0.5f},
            {-0.7f, -0.7f, 0.5f, 0.0f, 0.0f, 1.0f, 0.5f},
        };
        auto vb = rt.createBuffer(sizeof(verts), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb.Get(), verts, sizeof(verts));

        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(VertexPosCol)};
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        rt.resetCmd();
        rt.cmdList->SetGraphicsRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->RSSetViewports(1, &vp);
        rt.cmdList->RSSetScissorRects(1, &sc);
        rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
        const float bg[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        rt.cmdList->ClearRenderTargetView(rtvH, bg, 0, nullptr);
        rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
        rt.cmdList->DrawInstanced(3, 1, 0, 0);
        rt.executeAndWait();

        auto px = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
        size_t centerIdx = (static_cast<size_t>(H / 2) * W + (W / 2)) * 4;
        assert(px[centerIdx + 0] > 10 || px[centerIdx + 1] > 10 || px[centerIdx + 2] > 10);
        assert(px[0] == 0 && px[1] == 0 && px[2] == 0);
    }

    // 2. Depth-only shadow pass (empty PS)
    {
        constexpr UINT W = 64, H = 64;
        D3D12_CLEAR_VALUE clearD{};
        clearD.Format = DXGI_FORMAT_D32_FLOAT;
        clearD.DepthStencil.Depth = 1.0f;
        auto dsvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, &clearD);
        D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateDepthStencilView(dsvTex.Get(), nullptr, dsvH);

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rs.Get();
        psoDesc.VS = {shadowVsBytes.data(), shadowVsBytes.size()};
        psoDesc.InputLayout = {layout, 1};
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        psoDesc.SampleMask = UINT32_MAX;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

        ComPtr<ID3D12PipelineState> pso;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateShadowPSO");

        VertexPosOnly verts[3] = {{0.0f, 0.8f, 0.25f}, {0.8f, -0.8f, 0.25f}, {-0.8f, -0.8f, 0.25f}};
        auto vb = rt.createBuffer(sizeof(verts), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb.Get(), verts, sizeof(verts));

        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(VertexPosOnly)};
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        rt.resetCmd();
        rt.cmdList->SetGraphicsRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->RSSetViewports(1, &vp);
        rt.cmdList->RSSetScissorRects(1, &sc);
        rt.cmdList->OMSetRenderTargets(0, nullptr, FALSE, &dsvH);
        rt.cmdList->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
        rt.cmdList->DrawInstanced(3, 1, 0, 0);
        rt.executeAndWait();

        auto dRaw = readbackTex2D(rt, dsvTex.Get(), W, H, 4);
        float centerZ = 1.0f;
        std::memcpy(&centerZ, dRaw.data() + (static_cast<size_t>(H / 2) * W + (W / 2)) * 4, 4);
        assert(std::fabs(centerZ - 0.25f) < 1e-3f);
    }

    // 3. 108,000-triangle rasterizer throughput benchmark
    {
        constexpr UINT W = 512, H = 512;
        D3D12_CLEAR_VALUE clearC{};
        clearC.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        auto rtvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };
        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rs.Get();
        psoDesc.VS = {vsBytes.data(), vsBytes.size()};
        psoDesc.PS = {psBytes.data(), psBytes.size()};
        psoDesc.InputLayout = {layout, 2};
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleMask = UINT32_MAX;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

        ComPtr<ID3D12PipelineState> pso;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateBenchGfxPSO");

        constexpr UINT kGridX = 225, kGridY = 240; // 225 * 240 * 2 = 108,000 triangles
        std::vector<VertexPosCol> verts((kGridX + 1) * (kGridY + 1));
        for (UINT y = 0; y <= kGridY; ++y) {
            for (UINT x = 0; x <= kGridX; ++x) {
                float u = static_cast<float>(x) / kGridX;
                float v = static_cast<float>(y) / kGridY;
                verts[y * (kGridX + 1) + x] = {u * 1.8f - 0.9f, v * 1.8f - 0.9f, 0.5f, u, v, 0.5f, 1.0f};
            }
        }
        std::vector<uint32_t> indices;
        indices.reserve(kGridX * kGridY * 6);
        for (UINT y = 0; y < kGridY; ++y) {
            for (UINT x = 0; x < kGridX; ++x) {
                uint32_t i0 = y * (kGridX + 1) + x;
                uint32_t i1 = i0 + 1;
                uint32_t i2 = i0 + (kGridX + 1);
                uint32_t i3 = i2 + 1;
                indices.push_back(i0);
                indices.push_back(i1);
                indices.push_back(i2);
                indices.push_back(i1);
                indices.push_back(i3);
                indices.push_back(i2);
            }
        }

        auto vb = rt.createBuffer(verts.size() * sizeof(VertexPosCol), D3D12_HEAP_TYPE_DEFAULT);
        auto ib = rt.createBuffer(indices.size() * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb.Get(), verts.data(), verts.size() * sizeof(VertexPosCol));
        rt.uploadBuffer(ib.Get(), indices.data(), indices.size() * sizeof(uint32_t));

        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), static_cast<UINT>(verts.size() * sizeof(VertexPosCol)), sizeof(VertexPosCol)};
        D3D12_INDEX_BUFFER_VIEW ibv{ib->GetGPUVirtualAddress(), static_cast<UINT>(indices.size() * sizeof(uint32_t)), DXGI_FORMAT_R32_UINT};
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        auto runFrame = [&]() {
            rt.resetCmd();
            rt.cmdList->SetGraphicsRootSignature(rs.Get());
            rt.cmdList->SetPipelineState(pso.Get());
            rt.cmdList->RSSetViewports(1, &vp);
            rt.cmdList->RSSetScissorRects(1, &sc);
            rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
            const float bg[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            rt.cmdList->ClearRenderTargetView(rtvH, bg, 0, nullptr);
            rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
            rt.cmdList->IASetIndexBuffer(&ibv);
            rt.cmdList->DrawIndexedInstanced(static_cast<UINT>(indices.size()), 1, 0, 0, 0);
            rt.executeAndWait();
        };

        // Warmup
        runFrame();

        constexpr int kFrames = 5;
        auto t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < kFrames; ++f) runFrame();
        auto t1 = std::chrono::steady_clock::now();
        double msPerFrame = std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
        double fps = 1000.0 / msPerFrame;
        std::printf("rasterizer_pipeline_test PASSED (108,000 tris @ %ux%u: %.2f ms/frame, %.1f FPS)\n",
                    W, H, msPerFrame, fps);
    }

    return 0;
}
