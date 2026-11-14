#include "test_common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace {

constexpr UINT64 kTileBytes = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; // 65536

bool approxEq(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) <= eps;
}

ComPtr<ID3D12Heap> createTileHeap(ID3D12Device* dev, UINT numTiles) {
    D3D12_HEAP_DESC hd{};
    hd.SizeInBytes = UINT64(numTiles) * kTileBytes;
    hd.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hd.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hd.Flags = D3D12_HEAP_FLAG_NONE;
    ComPtr<ID3D12Heap> heap;
    testutil::checkHR(dev->CreateHeap(&hd, IID_PPV_ARGS(&heap)), "CreateTileHeap");
    return heap;
}

ComPtr<ID3D12Resource> createReservedBuffer(
    ID3D12Device* dev,
    UINT64 byteSize,
    D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(byteSize, flags);
    ComPtr<ID3D12Resource> res;
    testutil::checkHR(
        dev->CreateReservedResource(&rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)),
        "CreateReservedBuffer");
    return res;
}

ComPtr<ID3D12PipelineState> createComputePSO(
    ID3D12Device* dev,
    ID3D12RootSignature* rs,
    const char* shaderFile) {
    auto bytes = testutil::loadShaderBytes(shaderFile);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = rs;
    desc.CS = {bytes.data(), bytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), shaderFile);
    return pso;
}

void testApiAndTilingContracts(testutil::TestRuntime& rt) {
    std::cout << "=== Test: tiled_api_and_tiling_contracts ===\n";

    D3D12_FEATURE_DATA_D3D12_OPTIONS opts{};
    testutil::checkHR(
        rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts)),
        "CheckFeatureSupport(OPTIONS)");
    assert(opts.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_3);

    // Reserved buffer tiling query + Map/GetHeapProperties rejection
    auto rBuf = createReservedBuffer(rt.device.Get(), 3 * kTileBytes);
    assert(rBuf->GetGPUVirtualAddress() != 0);

    void* mapped = nullptr;
    assert(rBuf->Map(0, nullptr, &mapped) == E_INVALIDARG);
    D3D12_HEAP_PROPERTIES hp{};
    D3D12_HEAP_FLAGS hf{};
    assert(rBuf->GetHeapProperties(&hp, &hf) == E_INVALIDARG);

    UINT totalTiles = 0;
    D3D12_PACKED_MIP_INFO packedInfo{};
    D3D12_TILE_SHAPE tileShape{};
    UINT numSubTilings = 1;
    D3D12_SUBRESOURCE_TILING subTiling{};
    rt.device->GetResourceTiling(
        rBuf.Get(), &totalTiles, &packedInfo, &tileShape, &numSubTilings, 0, &subTiling);
    assert(totalTiles == 3);
    assert(tileShape.WidthInTexels == kTileBytes);
    assert(tileShape.HeightInTexels == 1);
    assert(tileShape.DepthInTexels == 1);
    assert(subTiling.WidthInTiles == 3);
    assert(subTiling.HeightInTiles == 1);
    assert(subTiling.DepthInTiles == 1);
    assert(subTiling.StartTileIndexInOverallResource == 0);

    // Reject row-major layout on reserved Texture2D
    {
        D3D12_RESOURCE_DESC badTex{};
        badTex.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        badTex.Width = 128;
        badTex.Height = 128;
        badTex.DepthOrArraySize = 1;
        badTex.MipLevels = 1;
        badTex.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        badTex.SampleDesc = {1, 0};
        badTex.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> badRes;
        assert(rt.device->CreateReservedResource(
                   &badTex, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&badRes)) == E_INVALIDARG);
    }

    // 2D texture with 2 mip levels (128x128 RGBA32F -> tile shape 64x64x1: mip0 = 2x2=4 tiles, mip1 = 64x64 = 1x1=1 tile -> 5 total)
    {
        D3D12_RESOURCE_DESC td{};
        td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width = 128;
        td.Height = 128;
        td.DepthOrArraySize = 1;
        td.MipLevels = 2;
        td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td.SampleDesc = {1, 0};
        td.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        ComPtr<ID3D12Resource> rTex2D;
        testutil::checkHR(
            rt.device->CreateReservedResource(&td, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rTex2D)),
            "CreateReservedTex2D");

        UINT texTiles = 0;
        D3D12_TILE_SHAPE shape2D{};
        UINT numSubs = 2;
        D3D12_SUBRESOURCE_TILING subTilings[2]{};
        rt.device->GetResourceTiling(rTex2D.Get(), &texTiles, nullptr, &shape2D, &numSubs, 0, subTilings);
        assert(texTiles == 5);
        assert(shape2D.WidthInTexels == 64 && shape2D.HeightInTexels == 64 && shape2D.DepthInTexels == 1);
        assert(numSubs == 2);
        assert(subTilings[0].WidthInTiles == 2 && subTilings[0].HeightInTiles == 2 && subTilings[0].StartTileIndexInOverallResource == 0);
        assert(subTilings[1].WidthInTiles == 1 && subTilings[1].HeightInTiles == 1 && subTilings[1].StartTileIndexInOverallResource == 4);
    }

    // 3D texture (32x32x32 RGBA32F -> tile shape 16x16x16 -> 2x2x2 = 8 tiles)
    {
        D3D12_RESOURCE_DESC td3{};
        td3.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        td3.Width = 32;
        td3.Height = 32;
        td3.DepthOrArraySize = 32;
        td3.MipLevels = 1;
        td3.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        td3.SampleDesc = {1, 0};
        td3.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        ComPtr<ID3D12Resource> rTex3D;
        testutil::checkHR(
            rt.device->CreateReservedResource(&td3, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rTex3D)),
            "CreateReservedTex3D");

        UINT tex3DTiles = 0;
        D3D12_TILE_SHAPE shape3D{};
        UINT numSubs3 = 1;
        D3D12_SUBRESOURCE_TILING sub3{};
        rt.device->GetResourceTiling(rTex3D.Get(), &tex3DTiles, nullptr, &shape3D, &numSubs3, 0, &sub3);
        assert(tex3DTiles == 8);
        assert(shape3D.WidthInTexels == 16 && shape3D.HeightInTexels == 16 && shape3D.DepthInTexels == 16);
        assert(sub3.WidthInTiles == 2 && sub3.HeightInTiles == 2 && sub3.DepthInTiles == 2);
    }
}

void testPartiallyMappedResourcesRepro(testutil::TestRuntime& rt) {
    std::cout << "=== Test: partially_mapped_resources_repro ===\n";

    // Descriptor table: t0, t1 (SRVs), u0, u1 (UAVs)
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 2;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 2;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &rp;
    auto rs = rt.createRootSig(rsDesc);
    auto pso = createComputePSO(rt.device.Get(), rs.Get(), "tiled_partially_mapped_cs.dxil");

    constexpr UINT64 kBufBytes = 2 * kTileBytes; // 131072 bytes = 8192 int4s
    constexpr UINT kNumInt4 = static_cast<UINT>(kBufBytes / 16);

    auto heap = createTileHeap(rt.device.Get(), 4);
    auto bufX = createReservedBuffer(rt.device.Get(), kBufBytes);
    auto bufY = createReservedBuffer(rt.device.Get(), kBufBytes);
    auto bufOut = rt.createBuffer(6 * sizeof(int32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufCAFM = rt.createBuffer(4 * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // Map ONLY tile 0 of X -> heap tile 0; leave tile 1 of X and all of Y unmapped (NULL).
    {
        D3D12_TILED_RESOURCE_COORDINATE startCoord{0, 0, 0, 0};
        D3D12_TILE_REGION_SIZE regionSize{1, FALSE, 0, 0, 0};
        D3D12_TILE_RANGE_FLAGS rangeFlags = D3D12_TILE_RANGE_FLAG_NONE;
        UINT heapOffset = 0;
        UINT rangeTileCount = 1;
        rt.queue->UpdateTileMappings(
            bufX.Get(), 1, &startCoord, &regionSize,
            heap.Get(), 1, &rangeFlags, &heapOffset, &rangeTileCount,
            D3D12_TILE_MAPPING_FLAG_NONE);
    }

    // Upload 9001 across all of X and 9002 across all of Y (writes to NULL tiles are discarded).
    std::vector<int32_t> initX(kNumInt4 * 4, 9001);
    std::vector<int32_t> initY(kNumInt4 * 4, 9002);
    rt.uploadBuffer(bufX.Get(), initX.data(), kBufBytes);
    rt.uploadBuffer(bufY.Get(), initY.data(), kBufBytes);

    D3D12_DESCRIPTOR_HEAP_DESC dhd{};
    dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dhd.NumDescriptors = 4;
    dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> descHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(&descHeap)), "CreateDescHeap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpuBase = descHeap->GetCPUDescriptorHandleForHeapStart();

    auto makeStructuredSRV = [&](ID3D12Resource* res, UINT slot) {
        D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
        srv.Format = DXGI_FORMAT_UNKNOWN;
        srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        srv.Buffer.FirstElement = 0;
        srv.Buffer.NumElements = kNumInt4;
        srv.Buffer.StructureByteStride = 16;
        D3D12_CPU_DESCRIPTOR_HANDLE h{cpuBase.ptr + slot * inc};
        rt.device->CreateShaderResourceView(res, &srv, h);
    };
    makeStructuredSRV(bufX.Get(), 0);
    makeStructuredSRV(bufY.Get(), 1);

    auto makeStructuredUAV = [&](ID3D12Resource* res, UINT numElems, UINT slot) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = numElems;
        uav.Buffer.StructureByteStride = 4;
        D3D12_CPU_DESCRIPTOR_HANDLE h{cpuBase.ptr + slot * inc};
        rt.device->CreateUnorderedAccessView(res, nullptr, &uav, h);
    };
    makeStructuredUAV(bufOut.Get(), 6, 2);
    makeStructuredUAV(bufCAFM.Get(), 4, 3);

    auto runDispatch = [&]() {
        rt.resetCmd();
        ID3D12DescriptorHeap* heaps[] = {descHeap.Get()};
        rt.cmdList->SetDescriptorHeaps(1, heaps);
        rt.cmdList->SetComputeRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->SetComputeRootDescriptorTable(0, descHeap->GetGPUDescriptorHandleForHeapStart());
        rt.cmdList->Dispatch(1, 1, 1);
        rt.executeAndWait();
    };

    runDispatch();

    auto cafmBytes = rt.readbackBuffer(bufCAFM.Get(), 4 * sizeof(uint32_t));
    auto outBytes = rt.readbackBuffer(bufOut.Get(), 6 * sizeof(int32_t));
    uint32_t cafm[4]{};
    int32_t out[6]{};
    std::memcpy(cafm, cafmBytes.data(), sizeof(cafm));
    std::memcpy(out, outBytes.data(), sizeof(out));

    assert(cafm[0] == 1 && cafm[1] == 0 && cafm[2] == 0 && cafm[3] == 0);
    assert(out[0] == 9001 && out[1] == 9003 && out[2] == 9003 && out[3] == 9003);
    assert(out[4] == 0 && out[5] == 0); // NULL-mapped tile reads return 0

    // Dynamically remap X tile 1 -> heap tile 1 and Y tile 0 -> heap tile 2 WITHOUT recreating descriptors!
    {
        D3D12_TILED_RESOURCE_COORDINATE cX1{1, 0, 0, 0};
        D3D12_TILE_REGION_SIZE sz1{1, FALSE, 0, 0, 0};
        D3D12_TILE_RANGE_FLAGS rf = D3D12_TILE_RANGE_FLAG_NONE;
        UINT offX1 = 1;
        UINT cnt1 = 1;
        rt.queue->UpdateTileMappings(bufX.Get(), 1, &cX1, &sz1, heap.Get(), 1, &rf, &offX1, &cnt1, D3D12_TILE_MAPPING_FLAG_NONE);

        D3D12_TILED_RESOURCE_COORDINATE cY0{0, 0, 0, 0};
        UINT offY0 = 2;
        rt.queue->UpdateTileMappings(bufY.Get(), 1, &cY0, &sz1, heap.Get(), 1, &rf, &offY0, &cnt1, D3D12_TILE_MAPPING_FLAG_NONE);
    }

    // Upload 7777 to X tile 1 and 8888 to Y tile 0
    {
        std::vector<int32_t> tileDataX1((kTileBytes / 4), 7777);
        std::vector<int32_t> tileDataY0((kTileBytes / 4), 8888);
        auto upX1 = rt.createBuffer(kTileBytes, D3D12_HEAP_TYPE_UPLOAD);
        auto upY0 = rt.createBuffer(kTileBytes, D3D12_HEAP_TYPE_UPLOAD);
        void* p = nullptr;
        testutil::checkHR(upX1->Map(0, nullptr, &p), "MapUpX1");
        std::memcpy(p, tileDataX1.data(), kTileBytes);
        upX1->Unmap(0, nullptr);
        testutil::checkHR(upY0->Map(0, nullptr, &p), "MapUpY0");
        std::memcpy(p, tileDataY0.data(), kTileBytes);
        upY0->Unmap(0, nullptr);

        rt.resetCmd();
        rt.cmdList->CopyBufferRegion(bufX.Get(), kTileBytes, upX1.Get(), 0, kTileBytes);
        rt.cmdList->CopyBufferRegion(bufY.Get(), 0, upY0.Get(), 0, kTileBytes);
        rt.executeAndWait();
    }

    runDispatch();

    cafmBytes = rt.readbackBuffer(bufCAFM.Get(), 4 * sizeof(uint32_t));
    outBytes = rt.readbackBuffer(bufOut.Get(), 6 * sizeof(int32_t));
    std::memcpy(cafm, cafmBytes.data(), sizeof(cafm));
    std::memcpy(out, outBytes.data(), sizeof(out));

    assert(cafm[0] == 1 && cafm[1] == 1 && cafm[2] == 1 && cafm[3] == 0);
    assert(out[0] == 9001 && out[1] == 7777 && out[2] == 8888 && out[3] == 9003);

    // Verify normal committed buffers always report CheckAccessFullyMapped == true
    auto commX = rt.createBuffer(kBufBytes, D3D12_HEAP_TYPE_DEFAULT);
    auto commY = rt.createBuffer(kBufBytes, D3D12_HEAP_TYPE_DEFAULT);
    rt.uploadBuffer(commX.Get(), initX.data(), kBufBytes);
    rt.uploadBuffer(commY.Get(), initY.data(), kBufBytes);
    makeStructuredSRV(commX.Get(), 0);
    makeStructuredSRV(commY.Get(), 1);
    runDispatch();

    cafmBytes = rt.readbackBuffer(bufCAFM.Get(), 4 * sizeof(uint32_t));
    outBytes = rt.readbackBuffer(bufOut.Get(), 6 * sizeof(int32_t));
    std::memcpy(cafm, cafmBytes.data(), sizeof(cafm));
    std::memcpy(out, outBytes.data(), sizeof(out));
    assert(cafm[0] == 1 && cafm[1] == 1 && cafm[2] == 1 && cafm[3] == 1);
    assert(out[0] == 9001 && out[1] == 9001 && out[2] == 9002 && out[3] == 9002);
}

void testDivergentTiledBuffersAndCopyOps(testutil::TestRuntime& rt) {
    std::cout << "=== Test: divergent_tiled_buffers_and_copy_ops ===\n";

    // Root signature: Root CBV b0, Root SRV t0, Root UAV u0, Root UAV u1, Root UAV u2
    D3D12_ROOT_PARAMETER rp[5]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[1].Descriptor.ShaderRegister = 0;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[2].Descriptor.ShaderRegister = 0;
    rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[3].Descriptor.ShaderRegister = 1;
    rp[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[4].Descriptor.ShaderRegister = 2;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 5;
    rsDesc.pParameters = rp;
    auto rs = rt.createRootSig(rsDesc);
    auto pso = createComputePSO(rt.device.Get(), rs.Get(), "tiled_divergent_buf_cs.dxil");

    auto heap = createTileHeap(rt.device.Get(), 6);

    // Reserved CBV buffer (1 tile mapped to heap tile 4)
    auto cbvRes = createReservedBuffer(rt.device.Get(), kTileBytes);
    {
        D3D12_TILED_RESOURCE_COORDINATE c0{0, 0, 0, 0};
        D3D12_TILE_REGION_SIZE sz1{1, FALSE, 0, 0, 0};
        D3D12_TILE_RANGE_FLAGS rf = D3D12_TILE_RANGE_FLAG_NONE;
        UINT off = 4;
        UINT cnt = 1;
        rt.queue->UpdateTileMappings(cbvRes.Get(), 1, &c0, &sz1, heap.Get(), 1, &rf, &off, &cnt, D3D12_TILE_MAPPING_FLAG_NONE);
        uint32_t cbData[4] = {100u, 0u, 0u, 0u};
        rt.uploadBuffer(cbvRes.Get(), cbData, sizeof(cbData));
    }

    // SrcRaw: 4 tiles (256 KiB):
    //   Tile 0 -> heap tile 0
    //   Tile 1 -> NULL
    //   Tile 2 & Tile 3 -> REUSE_SINGLE_TILE on heap tile 1
    auto srcRaw = createReservedBuffer(rt.device.Get(), 4 * kTileBytes);
    {
        D3D12_TILED_RESOURCE_COORDINATE start{0, 0, 0, 0};
        D3D12_TILE_REGION_SIZE region{4, FALSE, 0, 0, 0};
        D3D12_TILE_RANGE_FLAGS flags[3] = {
            D3D12_TILE_RANGE_FLAG_NONE,
            D3D12_TILE_RANGE_FLAG_NULL,
            D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE,
        };
        UINT offsets[3] = {0, 0, 1};
        UINT counts[3] = {1, 1, 2};
        rt.queue->UpdateTileMappings(
            srcRaw.Get(), 1, &start, &region,
            heap.Get(), 3, flags, offsets, counts, D3D12_TILE_MAPPING_FLAG_NONE);
    }

    // Populate SrcRaw tile 0 and tile 2 (shared with tile 3) via CopyTiles!
    {
        auto linearSrc = rt.createBuffer(2 * kTileBytes, D3D12_HEAP_TYPE_UPLOAD);
        uint32_t* p = nullptr;
        testutil::checkHR(linearSrc->Map(0, nullptr, reinterpret_cast<void**>(&p)), "MapLinearSrc");
        std::memset(p, 0, 2 * kTileBytes);
        for (uint32_t i = 0; i < 16; ++i) {
            p[i * 4] = 500u + i;                       // Tile 0 values
            p[(kTileBytes / 4) + i * 4] = 2000u + i;   // Shared tile (2 & 3) values
        }
        linearSrc->Unmap(0, nullptr);

        rt.resetCmd();
        D3D12_TILED_RESOURCE_COORDINATE cTile0{0, 0, 0, 0};
        D3D12_TILED_RESOURCE_COORDINATE cTile2{2, 0, 0, 0};
        D3D12_TILE_REGION_SIZE oneTile{1, FALSE, 0, 0, 0};
        rt.cmdList->CopyTiles(srcRaw.Get(), &cTile0, &oneTile, linearSrc.Get(), 0,
                              D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
        rt.cmdList->CopyTiles(srcRaw.Get(), &cTile2, &oneTile, linearSrc.Get(), kTileBytes,
                              D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
        rt.executeAndWait();
    }

    // Template buffer -> CopyTileMappings into DstRaw (tiles 0 and 2 mapped to heap tiles 2 and 3; 1 and 3 NULL)
    auto mapTemplate = createReservedBuffer(rt.device.Get(), 4 * kTileBytes);
    auto dstRaw = createReservedBuffer(rt.device.Get(), 4 * kTileBytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    {
        D3D12_TILED_RESOURCE_COORDINATE start{0, 0, 0, 0};
        D3D12_TILE_REGION_SIZE region{4, FALSE, 0, 0, 0};
        D3D12_TILE_RANGE_FLAGS flags[4] = {
            D3D12_TILE_RANGE_FLAG_NONE,
            D3D12_TILE_RANGE_FLAG_NULL,
            D3D12_TILE_RANGE_FLAG_NONE,
            D3D12_TILE_RANGE_FLAG_NULL,
        };
        UINT offsets[4] = {2, 0, 3, 0};
        UINT counts[4] = {1, 1, 1, 1};
        rt.queue->UpdateTileMappings(
            mapTemplate.Get(), 1, &start, &region,
            heap.Get(), 4, flags, offsets, counts, D3D12_TILE_MAPPING_FLAG_NONE);
        rt.queue->CopyTileMappings(
            dstRaw.Get(), &start, mapTemplate.Get(), &start, &region, D3D12_TILE_MAPPING_FLAG_NONE);
    }

    auto outValsBuf = rt.createBuffer(64 * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto outStatBuf = rt.createBuffer(64 * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    rt.resetCmd();
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootConstantBufferView(0, cbvRes->GetGPUVirtualAddress());
    rt.cmdList->SetComputeRootShaderResourceView(1, srcRaw->GetGPUVirtualAddress());
    rt.cmdList->SetComputeRootUnorderedAccessView(2, dstRaw->GetGPUVirtualAddress());
    rt.cmdList->SetComputeRootUnorderedAccessView(3, outValsBuf->GetGPUVirtualAddress());
    rt.cmdList->SetComputeRootUnorderedAccessView(4, outStatBuf->GetGPUVirtualAddress());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto valsBytes = rt.readbackBuffer(outValsBuf.Get(), 64 * sizeof(uint32_t));
    auto statBytes = rt.readbackBuffer(outStatBuf.Get(), 64 * sizeof(uint32_t));
    uint32_t vals[64]{}, stats[64]{};
    std::memcpy(vals, valsBytes.data(), sizeof(vals));
    std::memcpy(stats, statBytes.data(), sizeof(stats));

    for (uint32_t lane = 0; lane < 64; ++lane) {
        uint32_t tileIdx = lane / 16u;
        uint32_t inTile = lane & 15u;
        if (tileIdx == 0) {
            assert(stats[lane] == 1u);
            assert(vals[lane] == 500u + inTile + 100u);
        } else if (tileIdx == 1) {
            assert(stats[lane] == 0u);
            assert(vals[lane] == 0u + 100u); // NULL tile read 0 + addend 100
        } else {
            // Tiles 2 and 3 both share heap tile 1!
            assert(stats[lane] == 1u);
            assert(vals[lane] == 2000u + inTile + 100u);
        }
    }

    // Read back DstRaw tile 0 via CopyTiles and verify stores + atomic counter
    {
        auto rbTile0 = rt.createBuffer(kTileBytes, D3D12_HEAP_TYPE_READBACK);
        rt.resetCmd();
        D3D12_TILED_RESOURCE_COORDINATE cTile0{0, 0, 0, 0};
        D3D12_TILE_REGION_SIZE oneTile{1, FALSE, 0, 0, 0};
        rt.cmdList->CopyTiles(dstRaw.Get(), &cTile0, &oneTile, rbTile0.Get(), 0,
                              D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER);
        rt.executeAndWait();

        const uint32_t* p = nullptr;
        testutil::checkHR(rbTile0->Map(0, nullptr, reinterpret_cast<void**>(const_cast<uint32_t**>(&p))), "MapRbTile0");
        for (uint32_t i = 0; i < 16; ++i) {
            assert(p[i * 4] == i + 1000u);
        }
        assert(p[512 / 4] == 16u); // 16 lanes atomically incremented DstRaw tile 0
        rbTile0->Unmap(0, nullptr);
    }

    // Also verify full readback of DstRaw: mapped tiles 0 & 2 have data, NULL tiles 1 & 3 read back 0
    auto dstAll = rt.readbackBuffer(dstRaw.Get(), 4 * kTileBytes);
    const auto* dstWords = reinterpret_cast<const uint32_t*>(dstAll.data());
    assert(dstWords[(1 * kTileBytes + 0) / 4] == 0u);
    assert(dstWords[(1 * kTileBytes + 512) / 4] == 0u);
    assert(dstWords[(2 * kTileBytes + 0) / 4] == 32u + 1000u);
    assert(dstWords[(2 * kTileBytes + 512) / 4] == 16u);
    assert(dstWords[(3 * kTileBytes + 0) / 4] == 0u);
}

void testReservedTexture2DResidency(testutil::TestRuntime& rt) {
    std::cout << "=== Test: reserved_texture2d_residency ===\n";

    // Descriptor table: t0 (Texture2D), u0 (RWTexture2D), u1 (OutTexVals), u2 (OutTexStatus), static sampler s0
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 3;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 2;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    D3D12_STATIC_SAMPLER_DESC samp{};
    samp.Filter = D3D12_FILTER_MIN_MAG_MIP_POINT;
    samp.AddressU = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.AddressV = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.AddressW = D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    samp.MaxLOD = D3D12_FLOAT32_MAX;
    samp.ShaderRegister = 0;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &rp;
    rsDesc.NumStaticSamplers = 1;
    rsDesc.pStaticSamplers = &samp;
    auto rs = rt.createRootSig(rsDesc);
    auto pso = createComputePSO(rt.device.Get(), rs.Get(), "tiled_texture_cs.dxil");

    auto heap = createTileHeap(rt.device.Get(), 4);

    // 128x128 RGBA32F -> 2x2 tiles of 64x64 texels
    D3D12_RESOURCE_DESC td{};
    td.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width = 128;
    td.Height = 128;
    td.DepthOrArraySize = 1;
    td.MipLevels = 1;
    td.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    td.SampleDesc = {1, 0};
    td.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;

    ComPtr<ID3D12Resource> srvTex, uavTex;
    testutil::checkHR(
        rt.device->CreateReservedResource(&td, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&srvTex)),
        "CreateReservedSRVTex");
    td.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    testutil::checkHR(
        rt.device->CreateReservedResource(&td, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&uavTex)),
        "CreateReservedUAVTex");

    // Map srvTex quadrants (0,0) -> heap tile 0 and (1,1) -> heap tile 1 using UseBox=TRUE
    {
        D3D12_TILED_RESOURCE_COORDINATE coords[2] = {{0, 0, 0, 0}, {1, 1, 0, 0}};
        D3D12_TILE_REGION_SIZE boxSizes[2] = {{1, TRUE, 1, 1, 1}, {1, TRUE, 1, 1, 1}};
        D3D12_TILE_RANGE_FLAGS rf[2] = {D3D12_TILE_RANGE_FLAG_NONE, D3D12_TILE_RANGE_FLAG_NONE};
        UINT offs[2] = {0, 1};
        UINT cnts[2] = {1, 1};
        rt.queue->UpdateTileMappings(
            srvTex.Get(), 2, coords, boxSizes, heap.Get(), 2, rf, offs, cnts, D3D12_TILE_MAPPING_FLAG_NONE);
    }

    // Map uavTex quadrants (0,0) -> heap tile 2 and (0,1) -> heap tile 3
    {
        D3D12_TILED_RESOURCE_COORDINATE coords[2] = {{0, 0, 0, 0}, {0, 1, 0, 0}};
        D3D12_TILE_REGION_SIZE boxSizes[2] = {{1, TRUE, 1, 1, 1}, {1, TRUE, 1, 1, 1}};
        D3D12_TILE_RANGE_FLAGS rf[2] = {D3D12_TILE_RANGE_FLAG_NONE, D3D12_TILE_RANGE_FLAG_NONE};
        UINT offs[2] = {2, 3};
        UINT cnts[2] = {1, 1};
        rt.queue->UpdateTileMappings(
            uavTex.Get(), 2, coords, boxSizes, heap.Get(), 2, rf, offs, cnts, D3D12_TILE_MAPPING_FLAG_NONE);
    }

    // Upload 128x128 image to srvTex via CopyTextureRegion
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT numRows = 0;
        UINT64 rowSize = 0, totalBytes = 0;
        D3D12_RESOURCE_DESC srcDesc = srvTex->GetDesc();
        rt.device->GetCopyableFootprints(&srcDesc, 0, 1, 0, &fp, &numRows, &rowSize, &totalBytes);

        auto upload = rt.createBuffer(totalBytes, D3D12_HEAP_TYPE_UPLOAD);
        uint8_t* mapped = nullptr;
        testutil::checkHR(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped)), "MapSrvTexUpload");
        for (UINT y = 0; y < 128; ++y) {
            auto* row = reinterpret_cast<float*>(mapped + fp.Offset + UINT64(y) * fp.Footprint.RowPitch);
            for (UINT x = 0; x < 128; ++x) {
                UINT q = ((y >= 64) ? 2u : 0u) + ((x >= 64) ? 1u : 0u);
                row[x * 4 + 0] = float(q + 1) * 100.0f;
                row[x * 4 + 1] = float(x);
                row[x * 4 + 2] = float(y);
                row[x * 4 + 3] = 1.0f;
            }
        }
        upload->Unmap(0, nullptr);

        rt.resetCmd();
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = srvTex.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        dstLoc.SubresourceIndex = 0;

        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = upload.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        srcLoc.PlacedFootprint = fp;
        rt.cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        rt.executeAndWait();
    }

    auto outVals = rt.createBuffer(8 * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto outStatus = rt.createBuffer(8 * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_DESCRIPTOR_HEAP_DESC dhd{};
    dhd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    dhd.NumDescriptors = 4;
    dhd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> dh;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&dhd, IID_PPV_ARGS(&dh)), "CreateTexDescHeap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = dh->GetCPUDescriptorHandleForHeapStart();

    rt.device->CreateShaderResourceView(srvTex.Get(), nullptr, cpu);
    rt.device->CreateUnorderedAccessView(uavTex.Get(), nullptr, nullptr, D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + 1 * inc});

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavVals{};
    uavVals.Format = DXGI_FORMAT_UNKNOWN;
    uavVals.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavVals.Buffer.NumElements = 8;
    uavVals.Buffer.StructureByteStride = 16;
    rt.device->CreateUnorderedAccessView(outVals.Get(), nullptr, &uavVals, D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + 2 * inc});

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavStat{};
    uavStat.Format = DXGI_FORMAT_UNKNOWN;
    uavStat.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavStat.Buffer.NumElements = 8;
    uavStat.Buffer.StructureByteStride = 4;
    rt.device->CreateUnorderedAccessView(outStatus.Get(), nullptr, &uavStat, D3D12_CPU_DESCRIPTOR_HANDLE{cpu.ptr + 3 * inc});

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {dh.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, dh->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto vRaw = rt.readbackBuffer(outVals.Get(), 8 * 16);
    auto sRaw = rt.readbackBuffer(outStatus.Get(), 8 * 4);
    float v[8][4]{};
    uint32_t s[8]{};
    std::memcpy(v, vRaw.data(), sizeof(v));
    std::memcpy(s, sRaw.data(), sizeof(s));

    // Quadrant 0 (0,0) -> mapped: (100, 8, 8, 1)
    assert(s[0] == 1 && s[1] == 1);
    assert(approxEq(v[0][0], 100.0f) && approxEq(v[0][1], 8.0f) && approxEq(v[0][2], 8.0f) && approxEq(v[0][3], 1.0f));
    assert(approxEq(v[1][0], 100.0f) && approxEq(v[1][1], 8.0f) && approxEq(v[1][2], 8.0f) && approxEq(v[1][3], 1.0f));

    // Quadrant 1 (1,0) -> unmapped: (0,0,0,0), status = 0
    assert(s[2] == 0 && s[3] == 0);
    for (int c = 0; c < 4; ++c) {
        assert(approxEq(v[2][c], 0.0f) && approxEq(v[3][c], 0.0f));
    }

    // Quadrant 2 (0,1) -> unmapped: (0,0,0,0), status = 0
    assert(s[4] == 0 && s[5] == 0);
    for (int c = 0; c < 4; ++c) {
        assert(approxEq(v[4][c], 0.0f) && approxEq(v[5][c], 0.0f));
    }

    // Quadrant 3 (1,1) -> mapped: coord (72, 72) -> (400, 72, 72, 1)
    assert(s[6] == 1 && s[7] == 1);
    assert(approxEq(v[6][0], 400.0f) && approxEq(v[6][1], 72.0f) && approxEq(v[6][2], 72.0f) && approxEq(v[6][3], 1.0f));
    assert(approxEq(v[7][0], 400.0f) && approxEq(v[7][1], 72.0f) && approxEq(v[7][2], 72.0f) && approxEq(v[7][3], 1.0f));

    // Read back uavTex via CopyTextureRegion: quadrants 0 and 2 mapped, 1 and 3 NULL
    {
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        UINT numRows = 0;
        UINT64 rowSize = 0, totalBytes = 0;
        D3D12_RESOURCE_DESC uDesc = uavTex->GetDesc();
        rt.device->GetCopyableFootprints(&uDesc, 0, 1, 0, &fp, &numRows, &rowSize, &totalBytes);

        auto rb = rt.createBuffer(totalBytes, D3D12_HEAP_TYPE_READBACK);
        rt.resetCmd();
        D3D12_TEXTURE_COPY_LOCATION dstLoc{};
        dstLoc.pResource = rb.Get();
        dstLoc.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dstLoc.PlacedFootprint = fp;

        D3D12_TEXTURE_COPY_LOCATION srcLoc{};
        srcLoc.pResource = uavTex.Get();
        srcLoc.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        srcLoc.SubresourceIndex = 0;
        rt.cmdList->CopyTextureRegion(&dstLoc, 0, 0, 0, &srcLoc, nullptr);
        rt.executeAndWait();

        const uint8_t* mapped = nullptr;
        testutil::checkHR(rb->Map(0, nullptr, reinterpret_cast<void**>(const_cast<uint8_t**>(&mapped))), "MapUavTexRb");
        auto pixelAt = [&](UINT x, UINT y) -> const float* {
            return reinterpret_cast<const float*>(mapped + fp.Offset + UINT64(y) * fp.Footprint.RowPitch + UINT64(x) * 16);
        };
        const float* pQ0 = pixelAt(8, 8);
        const float* pQ1 = pixelAt(72, 8);
        const float* pQ2 = pixelAt(8, 72);
        const float* pQ3 = pixelAt(72, 72);

        assert(approxEq(pQ0[0], 10.0f) && approxEq(pQ0[1], 1.0f) && approxEq(pQ0[2], 2.0f) && approxEq(pQ0[3], 3.0f));
        assert(approxEq(pQ1[0], 0.0f) && approxEq(pQ1[1], 0.0f));
        assert(approxEq(pQ2[0], 30.0f) && approxEq(pQ2[1], 1.0f) && approxEq(pQ2[2], 2.0f) && approxEq(pQ2[3], 3.0f));
        assert(approxEq(pQ3[0], 0.0f) && approxEq(pQ3[1], 0.0f));
        rb->Unmap(0, nullptr);
    }
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    testApiAndTilingContracts(rt);
    testPartiallyMappedResourcesRepro(rt);
    testDivergentTiledBuffersAndCopyOps(rt);
    testReservedTexture2DResidency(rt);

    std::cout << "All tiled/reserved resource tests passed!\n";
    return 0;
}
