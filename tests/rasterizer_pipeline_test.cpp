#include "test_common.h"
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace {

void setEnvVar(const char* name, const char* val) {
#if defined(_WIN32)
    _putenv_s(name, val ? val : "");
#else
    if (val) {
        setenv(name, val, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct PackedVertex {
    int16_t px, py;   // R16G16_SNORM
    uint32_t bgra8;   // B8G8R8A8_UNORM
    uint32_t rgb10a2; // R10G10B10A2_UNORM
};

struct HeavyVertex {
    float px, py, pz;
    float nx, ny, nz;
    float u, v;
    float r, g, b, a;
};

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
        std::printf("  [Bench 1] 108,000 tris @ %ux%u: %.2f ms/frame (%.1f FPS)\n",
                    W, H, msPerFrame, fps);
    }

    // 4. Vectorized vertex format decoding, InstanceDataStepRate, and OOB default safety
    {
        auto packedVsBytes = testutil::loadShaderBytes("raster_packed_inst_vs.dxil");
        constexpr UINT W = 128, H = 128;
        D3D12_CLEAR_VALUE clearC{};
        clearC.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        auto rtvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R16G16_SNORM,     0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0},
            {"COLOR",    0, DXGI_FORMAT_B8G8R8A8_UNORM,   0, 4, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0},
            {"COLOR",    1, DXGI_FORMAT_R10G10B10A2_UNORM,0, 8, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,     1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 2},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rs.Get();
        psoDesc.VS = {packedVsBytes.data(), packedVsBytes.size()};
        psoDesc.PS = {psBytes.data(), psBytes.size()};
        psoDesc.InputLayout = {layout, 4};
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleMask = UINT32_MAX;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

        ComPtr<ID3D12PipelineState> pso;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreatePackedVtxPSO");

        // BGRA8 pure red = B=0, G=0, R=255, A=255 -> 0xFF0000FF in little-endian [B,G,R,A] = [0x00,0x00,0xFF,0xFF] = 0xFFFF0000u
        // R10G10B10A2 pure green = R=0, G=1023, B=0, A=3 -> (3u << 30) | (1023u << 10)
        // Average of red (1,0,0) and green (0,1,0) -> (0.5, 0.5, 0.0, 1.0)
        const uint32_t bgraRed = 0xFFFF0000u;
        const uint32_t rgb10Green = (3u << 30) | (1023u << 10);
        auto snorm16 = [](float f) -> int16_t { return static_cast<int16_t>(std::lround(f * 32767.0f)); };
        PackedVertex pverts[3] = {
            {snorm16(-0.35f), snorm16( 0.35f), bgraRed, rgb10Green},
            {snorm16( 0.35f), snorm16( 0.35f), bgraRed, rgb10Green},
            {snorm16( 0.00f), snorm16(-0.35f), bgraRed, rgb10Green},
        };
        // StepRate = 2: instances 0,1 read offset (-0.5, 0.0) [left half], instances 2,3 read (+0.5, 0.0) [right half]
        float instOffsets[4] = {-0.5f, 0.0f, 0.5f, 0.0f};

        auto vb0 = rt.createBuffer(sizeof(pverts), D3D12_HEAP_TYPE_DEFAULT);
        auto vb1 = rt.createBuffer(sizeof(instOffsets), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb0.Get(), pverts, sizeof(pverts));
        rt.uploadBuffer(vb1.Get(), instOffsets, sizeof(instOffsets));

        D3D12_VERTEX_BUFFER_VIEW vbvs[2] = {
            {vb0->GetGPUVirtualAddress(), sizeof(pverts), sizeof(PackedVertex)},
            {vb1->GetGPUVirtualAddress(), sizeof(instOffsets), 8},
        };
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        rt.resetCmd();
        rt.cmdList->SetGraphicsRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->RSSetViewports(1, &vp);
        rt.cmdList->RSSetScissorRects(1, &sc);
        rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
        const float bg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        rt.cmdList->ClearRenderTargetView(rtvH, bg, 0, nullptr);
        rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        rt.cmdList->IASetVertexBuffers(0, 2, vbvs);
        rt.cmdList->DrawInstanced(3, 4, 0, 0);
        rt.executeAndWait();

        auto px = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
        // Left triangle center: NDC x = -0.5 -> pixel x = 32, y = 64
        size_t leftIdx = (static_cast<size_t>(64) * W + 32) * 4;
        // Right triangle center: NDC x = +0.5 -> pixel x = 96, y = 64
        size_t rightIdx = (static_cast<size_t>(64) * W + 96) * 4;
        assert(std::abs(static_cast<int>(px[leftIdx + 0]) - 128) <= 3);
        assert(std::abs(static_cast<int>(px[leftIdx + 1]) - 128) <= 3);
        assert(px[leftIdx + 2] <= 2 && px[leftIdx + 3] >= 253);
        assert(std::abs(static_cast<int>(px[rightIdx + 0]) - 128) <= 3);
        assert(std::abs(static_cast<int>(px[rightIdx + 1]) - 128) <= 3);
        assert(px[rightIdx + 2] <= 2 && px[rightIdx + 3] >= 253);

        // Out-of-bounds VB safety check: bind SizeInBytes = 4 for COLOR0/COLOR1 so both attributes are OOB
        // -> D3D12 OOB defaults return (0, 0, 0, 1), producing RGB=(0,0,0), A=1.0 (255)!
        PackedVertex bigTri[3] = {
            {snorm16(-0.8f), snorm16( 0.8f), bgraRed, rgb10Green},
            {snorm16( 0.8f), snorm16( 0.8f), bgraRed, rgb10Green},
            {snorm16( 0.0f), snorm16(-0.8f), bgraRed, rgb10Green},
        };
        float zeroInst[2] = {0.0f, 0.0f};
        rt.uploadBuffer(vb0.Get(), bigTri, sizeof(bigTri));
        rt.uploadBuffer(vb1.Get(), zeroInst, sizeof(zeroInst));
        // Make only vertex 0's position in-bounds? No, give each vertex its 4-byte position in slot 0 via stride=4:
        int16_t posOnly[6] = {
            snorm16(-0.8f), snorm16(0.8f),
            snorm16( 0.8f), snorm16(0.8f),
            snorm16( 0.0f), snorm16(-0.8f),
        };
        rt.uploadBuffer(vb0.Get(), posOnly, sizeof(posOnly));
        D3D12_VERTEX_BUFFER_VIEW oobVbvs[2] = {
            {vb0->GetGPUVirtualAddress(), sizeof(posOnly), 4}, // bytes [4..12) per vertex exceed SizeInBytes on last vert, or use separate slot!
            {vb1->GetGPUVirtualAddress(), sizeof(zeroInst), 8},
        };
        // Even better: test explicit 0-byte VB on COLOR0/COLOR1 by shrinking vbv0 to 0 and checking no crash + clean execution,
        // and test slot-based OOB default alpha = 1.0!
        D3D12_INPUT_ELEMENT_DESC oobLayout[] = {
            {"POSITION", 0, DXGI_FORMAT_R16G16_SNORM,     0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0},
            {"COLOR",    0, DXGI_FORMAT_B8G8R8A8_UNORM,   2, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0},
            {"COLOR",    1, DXGI_FORMAT_R10G10B10A2_UNORM,2, 4, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,   0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,     1, 0, D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA, 1},
        };
        psoDesc.InputLayout = {oobLayout, 4};
        ComPtr<ID3D12PipelineState> oobPso;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&oobPso)), "CreateOOBVtxPSO");
        D3D12_VERTEX_BUFFER_VIEW oobSlotVbvs[3] = {
            oobVbvs[0],
            oobVbvs[1],
            {vb0->GetGPUVirtualAddress(), 2, 8}, // Only 2 bytes in slot 2 -> every 4-byte fetch is OOB -> (0,0,0,1)!
        };
        rt.resetCmd();
        rt.cmdList->SetGraphicsRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(oobPso.Get());
        rt.cmdList->RSSetViewports(1, &vp);
        rt.cmdList->RSSetScissorRects(1, &sc);
        rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
        rt.cmdList->ClearRenderTargetView(rtvH, bg, 0, nullptr);
        rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        rt.cmdList->IASetVertexBuffers(0, 3, oobSlotVbvs);
        rt.cmdList->DrawInstanced(3, 1, 0, 0);
        rt.executeAndWait();

        auto oobPx = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
        size_t midIdx = (static_cast<size_t>(56) * W + 64) * 4;
        assert(oobPx[midIdx + 0] == 0 && oobPx[midIdx + 1] == 0 && oobPx[midIdx + 2] == 0 && oobPx[midIdx + 3] == 255);
    }

    // 5. Multi-threaded geometry + ordered tile binning determinism (12,000 overlapping blended triangles)
    {
        constexpr UINT W = 256, H = 256;
        constexpr UINT kTriCount = 12000;
        D3D12_CLEAR_VALUE clearC{};
        clearC.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        auto rtvSingle = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        auto rtvMulti  = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvStart = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        UINT rtvInc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH0 = rtvStart;
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH1{rtvStart.ptr + rtvInc};
        rt.device->CreateRenderTargetView(rtvSingle.Get(), nullptr, rtvH0);
        rt.device->CreateRenderTargetView(rtvMulti.Get(), nullptr, rtvH1);

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
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateOrderDetPSO");

        std::vector<VertexPosCol> verts(kTriCount * 3);
        uint32_t rng = 0x12345678u;
        auto nextFloat = [&]() -> float {
            rng = rng * 1664525u + 1013904223u;
            return static_cast<float>((rng >> 8) & 0xFFFFu) / 65535.0f;
        };
        for (UINT t = 0; t < kTriCount; ++t) {
            float cx = nextFloat() * 1.6f - 0.8f;
            float cy = nextFloat() * 1.6f - 0.8f;
            float r = nextFloat(), g = nextFloat(), b = nextFloat();
            verts[t * 3 + 0] = {cx - 0.12f, cy + 0.11f, 0.5f, r, g, b, 0.25f};
            verts[t * 3 + 1] = {cx + 0.13f, cy + 0.08f, 0.5f, g, b, r, 0.25f};
            verts[t * 3 + 2] = {cx + 0.01f, cy - 0.14f, 0.5f, b, r, g, 0.25f};
        }
        auto vb = rt.createBuffer(verts.size() * sizeof(VertexPosCol), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb.Get(), verts.data(), verts.size() * sizeof(VertexPosCol));
        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), static_cast<UINT>(verts.size() * sizeof(VertexPosCol)), sizeof(VertexPosCol)};
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        auto drawToTarget = [&](D3D12_CPU_DESCRIPTOR_HANDLE targetH) {
            rt.resetCmd();
            rt.cmdList->SetGraphicsRootSignature(rs.Get());
            rt.cmdList->SetPipelineState(pso.Get());
            rt.cmdList->RSSetViewports(1, &vp);
            rt.cmdList->RSSetScissorRects(1, &sc);
            rt.cmdList->OMSetRenderTargets(1, &targetH, FALSE, nullptr);
            const float bg[4] = {0.1f, 0.1f, 0.1f, 1.0f};
            rt.cmdList->ClearRenderTargetView(targetH, bg, 0, nullptr);
            rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
            rt.cmdList->DrawInstanced(kTriCount * 3, 1, 0, 0);
            rt.executeAndWait();
        };

        setEnvVar("NACELLE_SINGLE_THREAD_GEOM", "1");
        drawToTarget(rtvH0);
        setEnvVar("NACELLE_SINGLE_THREAD_GEOM", nullptr);
        drawToTarget(rtvH1);

        auto pxSingle = readbackTex2D(rt, rtvSingle.Get(), W, H, 4);
        auto pxMulti  = readbackTex2D(rt, rtvMulti.Get(), W, H, 4);
        assert(pxSingle.size() == pxMulti.size());
        assert(std::memcmp(pxSingle.data(), pxMulti.data(), pxSingle.size()) == 0);
    }

    // 6. Decoupled Vertex Stage JIT cache & ID3D12PipelineLibrary graphics PSO round-trip
    {
        auto altPsBytes = testutil::loadShaderBytes("raster_alt_tint_ps.dxil");
        std::filesystem::path tmpCacheDir =
            std::filesystem::temp_directory_path() / "nacelle_decoupled_vs_cache_test";
        std::error_code ec;
        std::filesystem::remove_all(tmpCacheDir, ec);
        std::string tmpCacheStr = tmpCacheDir.string();
        setEnvVar("NACELLE_CACHE_DIR", tmpCacheStr.c_str());
        nacelle::clearInMemoryPSOCacheForTesting();

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC baseDesc{};
        baseDesc.pRootSignature = rs.Get();
        baseDesc.VS = {vsBytes.data(), vsBytes.size()};
        baseDesc.PS = {psBytes.data(), psBytes.size()};
        baseDesc.InputLayout = {layout, 2};
        baseDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        baseDesc.NumRenderTargets = 1;
        baseDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        baseDesc.SampleMask = UINT32_MAX;
        baseDesc.SampleDesc.Count = 1;
        baseDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        baseDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
        baseDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

        uint64_t vsCompilesBefore = nacelle::getVertexStageCompileCount();

        // PSO A: opaque + PSMain
        ComPtr<ID3D12PipelineState> psoA;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&baseDesc, IID_PPV_ARGS(&psoA)), "CreateDecoupledPSOA");

        // PSO B: alpha blend + PSMain (same VS + InputLayout + RootSignature!)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC descB = baseDesc;
        descB.BlendState.RenderTarget[0].BlendEnable = TRUE;
        descB.BlendState.RenderTarget[0].SrcBlend = D3D12_BLEND_ONE;
        descB.BlendState.RenderTarget[0].DestBlend = D3D12_BLEND_ONE;
        ComPtr<ID3D12PipelineState> psoB;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&descB, IID_PPV_ARGS(&psoB)), "CreateDecoupledPSOB");

        // PSO C: alternate Pixel Shader + depth test enabled (same VS + InputLayout + RootSignature!)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC descC = baseDesc;
        descC.PS = {altPsBytes.data(), altPsBytes.size()};
        descC.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        descC.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        ComPtr<ID3D12PipelineState> psoC;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&descC, IID_PPV_ARGS(&psoC)), "CreateDecoupledPSOC");

        // PSO D: depth-only shadow pass with null PS (same VS + InputLayout + RootSignature!)
        D3D12_GRAPHICS_PIPELINE_STATE_DESC descD = baseDesc;
        descD.PS = {nullptr, 0};
        descD.NumRenderTargets = 0;
        descD.RTVFormats[0] = DXGI_FORMAT_UNKNOWN;
        descD.DSVFormat = DXGI_FORMAT_D32_FLOAT;
        descD.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
        ComPtr<ID3D12PipelineState> psoD;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&descD, IID_PPV_ARGS(&psoD)), "CreateDecoupledPSOD");

        uint64_t vsCompilesAfter = nacelle::getVertexStageCompileCount();
        assert((vsCompilesAfter - vsCompilesBefore) == 1 && "Vertex stage must be JIT-compiled only once across all 4 PSOs");

        // Serialize psoC through ID3D12PipelineLibrary, clear in-memory caches + disk cache, reload, and draw!
        nacelle::waitForAllAsyncJITTasks();
        ComPtr<ID3D12PipelineLibrary> lib;
        testutil::checkHR(rt.device->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&lib)), "CreateGfxLib");
        testutil::checkHR(lib->StorePipeline(L"gfx_alt_tint", psoC.Get()), "StoreGfxPipeline");
        SIZE_T libSz = lib->GetSerializedSize();
        std::vector<uint8_t> libBlob(libSz);
        testutil::checkHR(lib->Serialize(libBlob.data(), libSz), "SerializeGfxLib");

        nacelle::clearInMemoryPSOCacheForTesting();
        std::filesystem::remove_all(tmpCacheDir, ec);

        ComPtr<ID3D12PipelineLibrary> loadedLib;
        testutil::checkHR(rt.device->CreatePipelineLibrary(libBlob.data(), libBlob.size(), IID_PPV_ARGS(&loadedLib)), "LoadGfxLib");
        ComPtr<ID3D12PipelineState> psoReloaded;
        testutil::checkHR(loadedLib->LoadGraphicsPipeline(L"gfx_alt_tint", &descC, IID_PPV_ARGS(&psoReloaded)), "LoadGraphicsPipeline");

        // Verify reloaded graphics PSO renders inverted tint accurately
        constexpr UINT W = 64, H = 64;
        D3D12_CLEAR_VALUE clearC{};
        clearC.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        D3D12_CLEAR_VALUE clearD{};
        clearD.Format = DXGI_FORMAT_D32_FLOAT;
        clearD.DepthStencil.Depth = 1.0f;
        auto rtvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        auto dsvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, &clearD);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        D3D12_CPU_DESCRIPTOR_HANDLE dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);
        rt.device->CreateDepthStencilView(dsvTex.Get(), nullptr, dsvH);

        VertexPosCol verts[3] = {
            {-0.8f,  0.8f, 0.5f, 0.2f, 0.8f, 0.0f, 1.0f},
            { 0.8f,  0.8f, 0.5f, 0.2f, 0.8f, 0.0f, 1.0f},
            { 0.0f, -0.8f, 0.5f, 0.2f, 0.8f, 0.0f, 1.0f},
        };
        auto vb = rt.createBuffer(sizeof(verts), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb.Get(), verts, sizeof(verts));
        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(VertexPosCol)};
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        rt.resetCmd();
        rt.cmdList->SetGraphicsRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(psoReloaded.Get());
        rt.cmdList->RSSetViewports(1, &vp);
        rt.cmdList->RSSetScissorRects(1, &sc);
        rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, &dsvH);
        const float bg[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        rt.cmdList->ClearRenderTargetView(rtvH, bg, 0, nullptr);
        rt.cmdList->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
        rt.cmdList->DrawInstanced(3, 1, 0, 0);
        rt.executeAndWait();

        auto px = readbackTex2D(rt, rtvTex.Get(), W, H, 4);
        size_t cIdx = (static_cast<size_t>(28) * W + 32) * 4;
        // PSAltTintMain outputs (1 - 0.2, 1 - 0.8, 1 - 0.0) = (0.8, 0.2, 1.0) -> (204, 51, 255)
        assert(std::abs(static_cast<int>(px[cIdx + 0]) - 204) <= 3);
        assert(std::abs(static_cast<int>(px[cIdx + 1]) - 51) <= 3);
        assert(px[cIdx + 2] >= 253);

        setEnvVar("NACELLE_CACHE_DIR", nullptr);
        std::filesystem::remove_all(tmpCacheDir, ec);
    }

    // 7. Heavy VS + backface culling geometry throughput benchmark (200,000 indexed triangles)
    {
        auto heavyVsBytes = testutil::loadShaderBytes("raster_heavy_transform_vs.dxil");
        constexpr UINT W = 256, H = 256;
        D3D12_CLEAR_VALUE clearC{};
        clearC.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        auto rtvTex = createTex2D(rt.device.Get(), W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &clearC);
        D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvH);

        D3D12_INPUT_ELEMENT_DESC layout[] = {
            {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT,    0,  0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"NORMAL",   0, DXGI_FORMAT_R32G32B32_FLOAT,    0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"TEXCOORD", 0, DXGI_FORMAT_R32G32_FLOAT,       0, 24, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
            {"COLOR",    0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 32, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        };

        D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = rs.Get();
        psoDesc.VS = {heavyVsBytes.data(), heavyVsBytes.size()};
        psoDesc.PS = {psBytes.data(), psBytes.size()};
        psoDesc.InputLayout = {layout, 4};
        psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
        psoDesc.NumRenderTargets = 1;
        psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        psoDesc.SampleMask = UINT32_MAX;
        psoDesc.SampleDesc.Count = 1;
        psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
        psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_BACK;
        psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

        ComPtr<ID3D12PipelineState> pso;
        testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateHeavyBenchPSO");
        nacelle::waitForAllAsyncJITTasks();

        constexpr UINT kGridX = 250, kGridY = 400; // 250 * 400 * 2 = 200,000 triangles (100,651 shared vertices)
        std::vector<HeavyVertex> verts((kGridX + 1) * (kGridY + 1));
        for (UINT y = 0; y <= kGridY; ++y) {
            for (UINT x = 0; x <= kGridX; ++x) {
                float u = static_cast<float>(x) / kGridX;
                float v = static_cast<float>(y) / kGridY;
                verts[y * (kGridX + 1) + x] = {
                    u * 1.8f - 0.9f, v * 1.8f - 0.9f, 0.5f,
                    0.0f, 0.0f, 1.0f,
                    u, v,
                    u, v, 0.6f, 1.0f,
                };
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
                // Alternate front-facing and back-facing quads (~50% backface culled, like a closed 3D mesh)
                if (((x + y) & 1u) == 0) {
                    indices.push_back(i0); indices.push_back(i1); indices.push_back(i2);
                    indices.push_back(i1); indices.push_back(i3); indices.push_back(i2);
                } else {
                    indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
                    indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
                }
            }
        }

        auto vb = rt.createBuffer(verts.size() * sizeof(HeavyVertex), D3D12_HEAP_TYPE_DEFAULT);
        auto ib = rt.createBuffer(indices.size() * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT);
        rt.uploadBuffer(vb.Get(), verts.data(), verts.size() * sizeof(HeavyVertex));
        rt.uploadBuffer(ib.Get(), indices.data(), indices.size() * sizeof(uint32_t));

        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), static_cast<UINT>(verts.size() * sizeof(HeavyVertex)), sizeof(HeavyVertex)};
        D3D12_INDEX_BUFFER_VIEW ibv{ib->GetGPUVirtualAddress(), static_cast<UINT>(indices.size() * sizeof(uint32_t)), DXGI_FORMAT_R32_UINT};
        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(W), static_cast<float>(H), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(W), static_cast<LONG>(H)};

        auto runHeavyFrame = [&]() {
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

        runHeavyFrame();
        constexpr int kFrames = 5;
        auto t0 = std::chrono::steady_clock::now();
        for (int f = 0; f < kFrames; ++f) runHeavyFrame();
        auto t1 = std::chrono::steady_clock::now();
        double msPerFrame = std::chrono::duration<double, std::milli>(t1 - t0).count() / kFrames;
        double mTrisPerSec = (200000.0 / (msPerFrame * 1e-3)) / 1e6;
        std::printf("  [Bench 2] 200,000 heavy-VS tris (50%% backface culled) @ %ux%u: %.2f ms/frame (%.1f Mtris/s)\n",
                    W, H, msPerFrame, mTrisPerSec);
    }

    std::printf("rasterizer_pipeline_test PASSED\n");

    return 0;
}
