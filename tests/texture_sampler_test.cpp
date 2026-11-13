#include "test_common.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

bool approxEq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

ComPtr<ID3D12Resource> createTextureResource(
    testutil::TestRuntime& rt,
    D3D12_RESOURCE_DIMENSION dim,
    UINT64 width,
    UINT height,
    UINT16 depthOrArray,
    UINT16 mipLevels,
    DXGI_FORMAT format,
    D3D12_RESOURCE_FLAGS flags,
    const D3D12_CLEAR_VALUE* clearVal = nullptr) {
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = dim;
    rd.Width = width;
    rd.Height = height;
    rd.DepthOrArraySize = depthOrArray;
    rd.MipLevels = mipLevels;
    rd.Format = format;
    rd.SampleDesc = {1, 0};
    rd.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    testutil::checkHR(
        rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, clearVal, IID_PPV_ARGS(&res)),
        "CreateTexture");
    return res;
}

void uploadTextureSubresources(
    testutil::TestRuntime& rt,
    ID3D12Resource* dst,
    UINT numSubresources,
    const std::vector<const float*>& subData) {
    D3D12_RESOURCE_DESC desc = dst->GetDesc();
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(numSubresources);
    std::vector<UINT> numRows(numSubresources);
    std::vector<UINT64> rowSizes(numSubresources);
    UINT64 totalBytes = 0;
    rt.device->GetCopyableFootprints(&desc, 0, numSubresources, 0, layouts.data(), numRows.data(), rowSizes.data(), &totalBytes);

    auto upload = rt.createBuffer(totalBytes, D3D12_HEAP_TYPE_UPLOAD);
    uint8_t* mapped = nullptr;
    testutil::checkHR(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "MapTexUpload");
    std::memset(mapped, 0, static_cast<size_t>(totalBytes));

    for (UINT sub = 0; sub < numSubresources; ++sub) {
        const auto& fp = layouts[sub].Footprint;
        const uint8_t* src = reinterpret_cast<const uint8_t*>(subData[sub]);
        for (UINT z = 0; z < fp.Depth; ++z) {
            for (UINT y = 0; y < numRows[sub]; ++y) {
                uint8_t* dstRow = mapped + layouts[sub].Offset + (UINT64(z) * numRows[sub] + y) * fp.RowPitch;
                const uint8_t* srcRow = src + (UINT64(z) * numRows[sub] + y) * rowSizes[sub];
                std::memcpy(dstRow, srcRow, static_cast<size_t>(rowSizes[sub]));
            }
        }
    }
    upload->Unmap(0, nullptr);

    rt.resetCmd();
    for (UINT sub = 0; sub < numSubresources; ++sub) {
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = dst;
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = sub;

        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = upload.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint = layouts[sub];

        rt.cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    }
    rt.executeAndWait();
}

std::vector<float> readbackTextureSubresource(
    testutil::TestRuntime& rt,
    ID3D12Resource* src,
    UINT subresource,
    UINT width,
    UINT height) {
    D3D12_RESOURCE_DESC desc = src->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT numRows = 0;
    UINT64 rowSize = 0;
    UINT64 totalBytes = 0;
    rt.device->GetCopyableFootprints(&desc, subresource, 1, 0, &layout, &numRows, &rowSize, &totalBytes);

    auto rb = rt.createBuffer(totalBytes, D3D12_HEAP_TYPE_READBACK);
    rt.resetCmd();
    D3D12_TEXTURE_COPY_LOCATION dstLoc{};
    dstLoc.pResource = rb.Get();
    dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dstLoc.PlacedFootprint = layout;

    D3D12_TEXTURE_COPY_LOCATION srcLoc{};
    srcLoc.pResource = src;
    srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    srcLoc.SubresourceIndex = subresource;

    rt.cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
    rt.executeAndWait();

    uint8_t* mapped = nullptr;
    testutil::checkHR(rb->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "MapTexReadback");
    std::vector<float> out(size_t(width) * height * 4, 0.0f);
    for (UINT y = 0; y < height; ++y) {
        std::memcpy(
            out.data() + size_t(y) * width * 4,
            mapped + layout.Offset + UINT64(y) * layout.Footprint.RowPitch,
            size_t(width) * 16);
    }
    rb->Unmap(0, nullptr);
    return out;
}

bool testTextureArraysAndSamplers(testutil::TestRuntime& rt) {
    std::cout << "=== Test: texture_arrays_and_samplers ===\n";

    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 4;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 3;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 2;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    D3D12_STATIC_SAMPLER_DESC staticSamps[2]{};
    staticSamps[0].Filter = D3D12_FILTER_MIN_MAG_MIP_LINEAR;
    staticSamps[0].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamps[0].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamps[0].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamps[0].MaxLOD = D3D12_FLOAT32_MAX;
    staticSamps[0].ShaderRegister = 0;

    staticSamps[1].Filter = D3D12_FILTER_MINIMUM_MIN_MAG_MIP_LINEAR;
    staticSamps[1].AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamps[1].AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamps[1].AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    staticSamps[1].MaxLOD = D3D12_FLOAT32_MAX;
    staticSamps[1].ShaderRegister = 1;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &rp;
    rsDesc.NumStaticSamplers = 2;
    rsDesc.pStaticSamplers = staticSamps;
    auto rs = rt.createRootSig(rsDesc);

    auto csBytes = testutil::loadShaderBytes("texture_array_ops.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.CS = {csBytes.data(), csBytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateTexCS");

    // 1. Texture2DArray (2x2, 3 slices)
    auto tex2DArr = createTextureResource(rt, D3D12_RESOURCE_DIMENSION_TEXTURE2D, 2, 2, 3, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    const float s0[16] = {1, 0.5f, 0.25f, 1,  0.2f, 1, 0.5f, 1,  0.4f, 0.6f, 1, 1,  0.8f, 0.3f, 0.9f, 1};
    const float s1[16] = {0, 0, 0, 1,  0, 0, 0, 1,  0, 0, 0, 1,  0.75f, 0.25f, 0.5f, 1};
    const float s2[16] = {0.1f, 0.2f, 0.3f, 1,  0.9f, 0.8f, 0.7f, 1,  0, 0, 0, 1,  1, 1, 1, 1};
    uploadTextureSubresources(rt, tex2DArr.Get(), 3, {s0, s1, s2});

    // 2. Texture1DArray (2x1, 3 slices)
    auto tex1DArr = createTextureResource(rt, D3D12_RESOURCE_DIMENSION_TEXTURE1D, 2, 1, 3, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    const float a1_0[8] = {1, 0, 0, 1,  0, 1, 0, 1};
    const float a1_1[8] = {0, 0, 1, 1,  0.5f, 0.5f, 0.5f, 1};
    const float a1_2[8] = {0.2f, 0.4f, 0.6f, 1,  0.8f, 0.6f, 0.4f, 1};
    uploadTextureSubresources(rt, tex1DArr.Get(), 3, {a1_0, a1_1, a1_2});

    // 3. TextureCubeArray (1x1, 12 slices = 2 cubes)
    auto texCubeArr = createTextureResource(rt, D3D12_RESOURCE_DIMENSION_TEXTURE2D, 1, 1, 12, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    std::vector<std::array<float, 4>> cubeFaces(12, {0, 0, 0, 1});
    cubeFaces[0] = {1, 0, 0, 1};       // cube 0 +X
    cubeFaces[6] = {0.5f, 0.25f, 0, 1}; // cube 1 +X
    std::vector<const float*> cubePtrs(12);
    for (int i = 0; i < 12; ++i) cubePtrs[i] = cubeFaces[i].data();
    uploadTextureSubresources(rt, texCubeArr.Get(), 12, cubePtrs);

    // 4. Texture3D (2x2x2, 2 mips)
    auto tex3D = createTextureResource(rt, D3D12_RESOURCE_DIMENSION_TEXTURE3D, 2, 2, 2, 2, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_NONE);
    std::vector<float> tex3DMip0(8 * 4, 0.0f);
    tex3DMip0[0] = 1; tex3DMip0[3] = 1; // (0,0,0) = Red
    const float tex3DMip1[4] = {0.25f, 0.75f, 0.5f, 1.0f};
    uploadTextureSubresources(rt, tex3D.Get(), 2, {tex3DMip0.data(), tex3DMip1});

    // 5. RWTexture2DArray & RWTexture1DArray & Out buffer
    auto rw2DArr = createTextureResource(rt, D3D12_RESOURCE_DIMENSION_TEXTURE2D, 2, 2, 3, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    uploadTextureSubresources(rt, rw2DArr.Get(), 3, {s0, s1, s2});

    auto rw1DArr = createTextureResource(rt, D3D12_RESOURCE_DIMENSION_TEXTURE1D, 2, 1, 3, 1, DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    uploadTextureSubresources(rt, rw1DArr.Get(), 3, {a1_0, a1_1, a1_2});

    const UINT outCount = 11;
    auto outBuf = rt.createBuffer(outCount * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 7, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateHeap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpuAt = [&](UINT i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = heap->GetCPUDescriptorHandleForHeapStart();
        h.ptr += SIZE_T(i) * inc;
        return h;
    };

    rt.device->CreateShaderResourceView(tex2DArr.Get(), nullptr, cpuAt(0));
    rt.device->CreateShaderResourceView(tex1DArr.Get(), nullptr, cpuAt(1));

    D3D12_SHADER_RESOURCE_VIEW_DESC cubeSrv{};
    cubeSrv.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    cubeSrv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURECUBEARRAY;
    cubeSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    cubeSrv.TextureCubeArray.MipLevels = 1;
    cubeSrv.TextureCubeArray.NumCubes = 2;
    rt.device->CreateShaderResourceView(texCubeArr.Get(), &cubeSrv, cpuAt(2));

    rt.device->CreateShaderResourceView(tex3D.Get(), nullptr, cpuAt(3));
    rt.device->CreateUnorderedAccessView(rw2DArr.Get(), nullptr, nullptr, cpuAt(4));
    rt.device->CreateUnorderedAccessView(rw1DArr.Get(), nullptr, nullptr, cpuAt(5));

    D3D12_UNORDERED_ACCESS_VIEW_DESC bufUav{};
    bufUav.Format = DXGI_FORMAT_UNKNOWN;
    bufUav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    bufUav.Buffer.NumElements = outCount;
    bufUav.Buffer.StructureByteStride = 16;
    rt.device->CreateUnorderedAccessView(outBuf.Get(), nullptr, &bufUav, cpuAt(6));

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto rawOut = rt.readbackBuffer(outBuf.Get(), outCount * 16);
    std::vector<float> out(outCount * 4);
    std::memcpy(out.data(), rawOut.data(), rawOut.size());

    auto check4 = [&](UINT idx, float e0, float e1, float e2, float e3, const char* label) -> bool {
        const float* g = &out[idx * 4];
        if (!approxEq(g[0], e0) || !approxEq(g[1], e1) || !approxEq(g[2], e2) || !approxEq(g[3], e3)) {
            std::cerr << "FAIL " << label << ": got (" << g[0] << "," << g[1] << "," << g[2] << "," << g[3]
                      << ") expected (" << e0 << "," << e1 << "," << e2 << "," << e3 << ")\n";
            return false;
        }
        return true;
    };

    if (!check4(0, 0.75f, 0.25f, 0.5f, 1.0f, "Tex2DArr.Load+offset")) return false;
    if (!check4(1, 0.8f, 0.6f, 0.4f, 1.0f, "Tex1DArr.Load+offset")) return false;
    if (!check4(2, 0.5f, 0.5f, 0.5f, 1.0f, "RWTex1DArr[]")) return false;
    if (!check4(3, 0.9f, 0.8f, 0.7f, 1.0f, "RWTex2DArr[]")) return false;
    if (!check4(4, 0.5f, 0.25f, 0.0f, 1.0f, "TexCubeArr.SampleLevel")) return false;
    if (!check4(5, 0.25f, 0.75f, 0.5f, 1.0f, "Tex3D.SampleGrad(Z)")) return false;
    if (!check4(6, 0.2f, 0.3f, 0.25f, 1.0f, "MinStaticSampler")) return false;
    if (!check4(7, 2, 3, 1, 0, "Tex1DArr.GetDimensions")) return false;
    if (!check4(8, 2, 2, 3, 1, "Tex2DArr.GetDimensions")) return false;
    if (!check4(9, 1, 1, 1, 2, "Tex3D.GetDimensions(mip1)")) return false;
    if (!check4(10, 1, 1, 2, 1, "TexCubeArr.GetDimensions")) return false;

    auto rw1Slice2 = readbackTextureSubresource(rt, rw1DArr.Get(), 2, 2, 1);
    if (!approxEq(rw1Slice2[0], 0.25f) || !approxEq(rw1Slice2[1], 0.5f) || !approxEq(rw1Slice2[2], 0.75f)) {
        std::cerr << "FAIL RWTex1DArr write\n";
        return false;
    }
    auto rw2Slice1 = readbackTextureSubresource(rt, rw2DArr.Get(), 1, 2, 2);
    if (!approxEq(rw2Slice1[2 * 4 + 0], 0.125f) || !approxEq(rw2Slice1[2 * 4 + 1], 0.25f) || !approxEq(rw2Slice1[2 * 4 + 2], 0.375f)) {
        std::cerr << "FAIL RWTex2DArr write\n";
        return false;
    }

    std::cout << "PASS: texture_arrays_and_samplers\n";
    return true;
}

bool testDepthStencilReadback(testutil::TestRuntime& rt) {
    std::cout << "=== Test: depth_stencil_readback ===\n";

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    auto rs = rt.createRootSig(rsDesc);

    auto vsBytes = testutil::loadShaderBytes("depth_stencil_readback_vs.dxil");
    auto psBytes = testutil::loadShaderBytes("depth_stencil_readback_ps.dxil");

    D3D12_INPUT_ELEMENT_DESC ie{};
    ie.SemanticName = "POSITION";
    ie.Format = DXGI_FORMAT_R32G32B32_FLOAT;
    ie.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.VS = {vsBytes.data(), vsBytes.size()};
    psoDesc.PS = {psBytes.data(), psBytes.size()};
    psoDesc.InputLayout = {&ie, 1};
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.RasterizerState.FrontCounterClockwise = TRUE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    psoDesc.DepthStencilState.DepthEnable = TRUE;
    psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
    psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    psoDesc.SampleDesc = {1, 0};

    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateDepthReadbackPSO");

    const UINT W = 3, H = 3;
    D3D12_CLEAR_VALUE rtClear{};
    rtClear.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    auto renderTarget = createTextureResource(
        rt, D3D12_RESOURCE_DIMENSION_TEXTURE2D, W, H, 1, 1,
        DXGI_FORMAT_R32G32B32A32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, &rtClear);

    D3D12_CLEAR_VALUE dsClear{};
    dsClear.Format = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
    dsClear.DepthStencil = {1.0f, 0};
    auto depthStencil = createTextureResource(
        rt, D3D12_RESOURCE_DIMENSION_TEXTURE2D, W, H, 1, 1,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, &dsClear);

    D3D12_DESCRIPTOR_HEAP_DESC rtvHd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHd, IID_PPV_ARGS(&rtvHeap)), "RTVHeap");
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandle = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateRenderTargetView(renderTarget.Get(), nullptr, rtvHandle);

    D3D12_DESCRIPTOR_HEAP_DESC dsvHd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHd, IID_PPV_ARGS(&dsvHeap)), "DSVHeap");
    D3D12_CPU_DESCRIPTOR_HANDLE dsvHandle = dsvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.device->CreateDepthStencilView(depthStencil.Get(), nullptr, dsvHandle);

    const float verts[18] = {
        -1.0f, -1.0f, 0.0f, -1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 0.0f,  1.0f,-1.0f, 0.0f,-1.0f,-1.0f, 0.0f,
    };
    auto vb = rt.createBuffer(sizeof(verts), D3D12_HEAP_TYPE_DEFAULT);
    rt.uploadBuffer(vb.Get(), verts, sizeof(verts));
    D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), 12};

    rt.resetCmd();
    const float clearColor[4] = {0, 0, 0, 0};
    rt.cmdList->ClearRenderTargetView(rtvHandle, clearColor, 0, nullptr);
    rt.cmdList->ClearDepthStencilView(dsvHandle, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
    rt.cmdList->OMSetRenderTargets(1, &rtvHandle, FALSE, &dsvHandle);
    D3D12_VIEWPORT vp{0.0f, 0.0f, float(W), float(H), 0.0f, 1.0f};
    D3D12_RECT sc{0, 0, LONG(W), LONG(H)};
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    rt.cmdList->SetGraphicsRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
    rt.cmdList->DrawInstanced(6, 1, 0, 0);
    rt.executeAndWait();

    auto pixels = readbackTextureSubresource(rt, renderTarget.Get(), 0, W, H);
    for (UINT y = 0; y < H; ++y) {
        for (UINT x = 0; x < W; ++x) {
            const float* p = &pixels[(y * W + x) * 4];
            if (!approxEq(p[0], float(x) + 0.5f) || !approxEq(p[1], float(y) + 0.5f) || !approxEq(p[2], 0.0f) || !approxEq(p[3], 1.0f)) {
                std::cerr << "FAIL depth_stencil_readback at (" << x << "," << y << "): got ("
                          << p[0] << "," << p[1] << "," << p[2] << "," << p[3] << ")\n";
                return false;
            }
        }
    }

    std::cout << "PASS: depth_stencil_readback\n";
    return true;
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();
    if (!testTextureArraysAndSamplers(rt)) return 1;
    if (!testDepthStencilReadback(rt)) return 1;
    return 0;
}
