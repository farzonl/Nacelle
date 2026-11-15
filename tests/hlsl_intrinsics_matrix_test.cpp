#include "test_common.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>

namespace {

ComPtr<ID3D12RootSignature> createRS(ID3D12Device* dev, UINT numParams, const D3D12_ROOT_PARAMETER* params) {
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = numParams;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> err;
    testutil::checkHR(NacelleSerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &err), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");
    return rs;
}

ComPtr<ID3D12PipelineState> createCS(ID3D12Device* dev, ID3D12RootSignature* rs, const char* file) {
    auto bytes = testutil::loadShaderBytes(file);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = rs;
    desc.CS = {bytes.data(), bytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), file);
    return pso;
}

bool testMatrixSwizzle(testutil::TestRuntime& rt) {
    std::cout << "=== Test: matrix_swizzle ===\n";
    D3D12_DESCRIPTOR_RANGE uavRange{};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = 3;
    uavRange.BaseShaderRegister = 0;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 1;
    rp.DescriptorTable.pDescriptorRanges = &uavRange;

    auto rs = createRS(rt.device.Get(), 1, &rp);
    auto pso = createCS(rt.device.Get(), rs.Get(), "matrix_swizzle.dxil");

    auto bufIn = rt.createBuffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufRow = rt.createBuffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufCol = rt.createBuffer(64, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    int32_t inData[4] = {1, 2, 3, 4};
    rt.uploadBuffer(bufIn.Get(), inData, sizeof(inData));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();

    auto makeUav = [&](ID3D12Resource* res, UINT numElems, UINT idx) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_SINT;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = numElems;
        rt.device->CreateUnorderedAccessView(res, nullptr, &uav, {cpu.ptr + idx * inc});
    };
    makeUav(bufIn.Get(), 4, 0);
    makeUav(bufRow.Get(), 16, 1);
    makeUav(bufCol.Get(), 16, 2);

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto rowRaw = rt.readbackBuffer(bufRow.Get(), 64);
    auto colRaw = rt.readbackBuffer(bufCol.Get(), 64);
    int32_t row[16], col[16];
    std::memcpy(row, rowRaw.data(), 64);
    std::memcpy(col, colRaw.data(), 64);

    const int32_t expRow[16] = {1,1,1,1, 2,2,2,2, 3,3,3,3, 4,4,4,4};
    const int32_t expCol[16] = {1,2,3,4, 1,2,3,4, 1,2,3,4, 1,2,3,4};
    bool ok = true;
    for (int i = 0; i < 16; ++i) {
        if (row[i] != expRow[i] || col[i] != expCol[i]) ok = false;
    }
    if (!ok) {
        std::cout << "RowOut: ";
        for (int i = 0; i < 16; ++i) std::cout << row[i] << " ";
        std::cout << "\nColOut: ";
        for (int i = 0; i < 16; ++i) std::cout << col[i] << " ";
        std::cout << "\n";
    } else {
        std::cout << "PASS\n";
    }
    return ok;
}

bool testQuadDerivatives(testutil::TestRuntime& rt, bool useD32S8DepthStencil) {
    std::cout << "=== Test: quad_derivatives"
              << (useD32S8DepthStencil ? " (D32_FLOAT_S8X24_UINT DSV)" : "")
              << " ===\n";
    auto vsBytes = testutil::loadShaderBytes("quad_derivatives_vs.dxil");
    auto psBytes = testutil::loadShaderBytes("quad_derivatives_ps.dxil");

    auto rs = createRS(rt.device.Get(), 0, nullptr);

    D3D12_INPUT_ELEMENT_DESC ie{};
    ie.SemanticName = "POSITION";
    ie.Format = DXGI_FORMAT_R32G32_FLOAT;
    ie.InputSlotClass = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA;

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.VS = {vsBytes.data(), vsBytes.size()};
    psoDesc.PS = {psBytes.data(), psBytes.size()};
    psoDesc.InputLayout = {&ie, 1};
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R32G32B32A32_FLOAT;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_SOLID;
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.BlendState.RenderTarget[0].RenderTargetWriteMask = D3D12_COLOR_WRITE_ENABLE_ALL;
    if (useD32S8DepthStencil) {
        psoDesc.DSVFormat = DXGI_FORMAT_D32_FLOAT_S8X24_UINT;
        psoDesc.DepthStencilState.DepthEnable = TRUE;
        psoDesc.DepthStencilState.DepthWriteMask = D3D12_DEPTH_WRITE_MASK_ALL;
        psoDesc.DepthStencilState.DepthFunc = D3D12_COMPARISON_FUNC_LESS;
    }

    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateGraphicsPSO");

    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R32G32B32A32_FLOAT, 4, 4, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    ComPtr<ID3D12Resource> rtvTex;
    testutil::checkHR(rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&rtvTex)), "RTVTex");

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap)), "RTVHeap");
    rt.device->CreateRenderTargetView(rtvTex.Get(), nullptr, rtvHeap->GetCPUDescriptorHandleForHeapStart());

    ComPtr<ID3D12Resource> dsvTex;
    ComPtr<ID3D12DescriptorHeap> dsvHeap;
    D3D12_CPU_DESCRIPTOR_HANDLE dsvH{};
    if (useD32S8DepthStencil) {
        D3D12_RESOURCE_DESC dsd = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 4, 4, 1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        testutil::checkHR(rt.device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &dsd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&dsvTex)), "DSVTex");
        D3D12_DESCRIPTOR_HEAP_DESC dsvHD{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        testutil::checkHR(rt.device->CreateDescriptorHeap(&dsvHD, IID_PPV_ARGS(&dsvHeap)), "DSVHeap");
        dsvH = dsvHeap->GetCPUDescriptorHandleForHeapStart();
        rt.device->CreateDepthStencilView(dsvTex.Get(), nullptr, dsvH);
    }

    const float verts[12] = {
        -1.f, -1.f, -1.f, 1.f, 1.f, 1.f,
         1.f,  1.f,  1.f,-1.f,-1.f,-1.f
    };
    auto vb = rt.createBuffer(sizeof(verts), D3D12_HEAP_TYPE_DEFAULT);
    rt.uploadBuffer(vb.Get(), verts, sizeof(verts));
    D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), 8};

    D3D12_VIEWPORT vp{0.f, 0.f, 4.f, 4.f, 0.f, 1.f};
    D3D12_RECT sc{0, 0, 4, 4};

    rt.resetCmd();
    rt.cmdList->SetGraphicsRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->RSSetViewports(1, &vp);
    rt.cmdList->RSSetScissorRects(1, &sc);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvH = rtvHeap->GetCPUDescriptorHandleForHeapStart();
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, useD32S8DepthStencil ? &dsvH : nullptr);
    const float clearCol[4] = {0, 0, 0, 0};
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
    if (useD32S8DepthStencil) {
        rt.cmdList->ClearDepthStencilView(dsvH, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.0f, 0, 0, nullptr);
    }
    rt.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    rt.cmdList->IASetVertexBuffers(0, 1, &vbv);
    rt.cmdList->DrawInstanced(6, 1, 0, 0);
    rt.executeAndWait();

    UINT rowPitch = 256;
    auto rb = rt.createBuffer(rowPitch * 4, D3D12_HEAP_TYPE_READBACK);
    rt.resetCmd();
    D3D12_TEXTURE_COPY_LOCATION dst{rb.Get(), D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
    dst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32G32B32A32_FLOAT, 4, 4, 1, rowPitch};
    D3D12_TEXTURE_COPY_LOCATION src{rtvTex.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
    rt.cmdList->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    rt.executeAndWait();

    void* ptr = nullptr;
    testutil::checkHR(rb->Map(0, nullptr, &ptr), "Map");
    bool ok = true;
    for (UINT y = 0; y < 4; ++y) {
        const float* row = reinterpret_cast<const float*>(static_cast<const uint8_t*>(ptr) + y * rowPitch);
        for (UINT x = 0; x < 4; ++x) {
            const float* px = row + x * 4;
            if (px[0] != 1.f || px[1] != 1.f || px[2] != 1.f || px[3] != 1.f) {
                ok = false;
                std::cout << "Pixel(" << x << "," << y << ") = ("
                          << px[0] << ", " << px[1] << ", " << px[2] << ", " << px[3] << ")\n";
            }
        }
    }
    rb->Unmap(0, nullptr);
    if (ok) std::cout << "PASS\n";
    return ok;
}

bool testMathDenormsBits(testutil::TestRuntime& rt) {
    std::cout << "=== Test: math_denorms_bits ===\n";
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 2;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 2;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    auto rs = createRS(rt.device.Get(), 1, &rp);
    auto pso = createCS(rt.device.Get(), rs.Get(), "math_denorms_bits.dxil");

    auto bufInFlt = rt.createBuffer(16, D3D12_HEAP_TYPE_DEFAULT);
    auto bufInU32 = rt.createBuffer(16, D3D12_HEAP_TYPE_DEFAULT);
    auto bufInI32 = rt.createBuffer(16, D3D12_HEAP_TYPE_DEFAULT);
    auto bufOutFlt = rt.createBuffer(6 * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufOutU32 = rt.createBuffer(2 * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    // -denorm, +denorm, -0.0, 2.0
    float inF[4] = {-0x1.e7d42cp-127f, 0x1.e7d42cp-127f, -0.0f, 2.0f};
    uint32_t inU[4] = {0u, 9u, 1u, 0xFFFFFFFFu};
    int32_t inI[4] = {-1, -8, 9, 0};
    rt.uploadBuffer(bufInFlt.Get(), inF, 16);
    rt.uploadBuffer(bufInU32.Get(), inU, 16);
    rt.uploadBuffer(bufInI32.Get(), inI, 16);

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 5, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();

    auto makeSrv = [&](ID3D12Resource* r, UINT num, UINT stride, UINT idx) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = num;
        d.Buffer.StructureByteStride = stride;
        rt.device->CreateShaderResourceView(r, &d, {cpu.ptr + idx * inc});
    };
    auto makeUav = [&](ID3D12Resource* r, UINT num, UINT stride, UINT idx) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements = num;
        d.Buffer.StructureByteStride = stride;
        rt.device->CreateUnorderedAccessView(r, nullptr, &d, {cpu.ptr + idx * inc});
    };
    makeSrv(bufInFlt.Get(), 1, 16, 0);
    makeSrv(bufInU32.Get(), 1, 16, 1);
    makeSrv(bufInI32.Get(), 1, 16, 2);
    makeUav(bufOutFlt.Get(), 6, 16, 3);
    makeUav(bufOutU32.Get(), 2, 16, 4);

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto outFltRaw = rt.readbackBuffer(bufOutFlt.Get(), 6 * 16);
    auto outU32Raw = rt.readbackBuffer(bufOutU32.Get(), 2 * 16);
    float outF[6][4];
    uint32_t outU[2][4];
    std::memcpy(outF, outFltRaw.data(), sizeof(outF));
    std::memcpy(outU, outU32Raw.data(), sizeof(outU));

    bool ok = true;
    auto isNegZero = [](float v) {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return b == 0x80000000u;
    };
    auto isPosZero = [](float v) {
        uint32_t b;
        std::memcpy(&b, &v, 4);
        return b == 0x00000000u;
    };

    // degrees, radians, sqrt: -denorm -> -0, +denorm -> +0
    const char* names[6] = {"degrees", "radians", "sqrt", "rsqrt", "log", "log10"};
    for (int i = 0; i < 3; ++i) {
        if (!isNegZero(outF[i][0]) || !isPosZero(outF[i][1])) {
            ok = false;
            std::cout << names[i] << " denorm mismatch: " << outF[i][0] << ", " << outF[i][1] << "\n";
        }
    }
    // rsqrt(-denorm) == -inf, rsqrt(+denorm) == +inf, rsqrt(-0.0) == -inf
    if (outF[3][0] != -std::numeric_limits<float>::infinity() ||
        outF[3][1] != std::numeric_limits<float>::infinity() ||
        outF[3][2] != -std::numeric_limits<float>::infinity()) {
        ok = false;
        std::cout << "rsqrt denorm mismatch: " << outF[3][0] << ", " << outF[3][1]
                  << ", " << outF[3][2] << "\n";
    }
    // log/log10(±denorm) == -inf
    for (int i = 4; i < 6; ++i) {
        if (outF[i][0] != -std::numeric_limits<float>::infinity() ||
            outF[i][1] != -std::numeric_limits<float>::infinity()) {
            ok = false;
            std::cout << names[i] << " denorm mismatch: " << outF[i][0] << ", " << outF[i][1] << "\n";
        }
    }

    const uint32_t expU[2][4] = {
        {4294967295u, 3u, 0u, 31u},
        {4294967295u, 2u, 3u, 4294967295u}
    };
    for (int r = 0; r < 2; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (outU[r][c] != expU[r][c]) {
                ok = false;
                std::cout << "firstbithigh[" << r << "][" << c << "] = " << outU[r][c]
                          << " (expected " << expU[r][c] << ")\n";
            }
        }
    }
    if (ok) std::cout << "PASS\n";
    return ok;
}

bool testBitAndFPClassIntrinsics(testutil::TestRuntime& rt) {
    std::cout << "=== Test: bit_fpclass_intrinsics ===\n";
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 3;
    ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 3;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 3;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 2;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    auto rs = createRS(rt.device.Get(), 1, &rp);
    auto pso = createCS(rt.device.Get(), rs.Get(), "bit_fpclass_intrinsics.dxil");

    auto bufInI64 = rt.createBuffer(2 * 32, D3D12_HEAP_TYPE_DEFAULT);
    auto bufInU64 = rt.createBuffer(2 * 32, D3D12_HEAP_TYPE_DEFAULT);
    auto bufInFlt = rt.createBuffer(3 * 16, D3D12_HEAP_TYPE_DEFAULT);
    auto bufOutBits = rt.createBuffer(5 * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufOutAdj = rt.createBuffer(3 * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufOutFPC = rt.createBuffer(9 * 16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    const int64_t inI64[2][4] = {
        {0x2All, 0ll, 0x0f0f0f0f0f0f0f0fll, -1ll},
        {0ll, -1ll, -9ll, 0x0000010000000000ll},
    };
    const uint64_t inU64[2][4] = {
        {0x2Aull, 0ull, 0x0f0f0f0f0f0f0f0full, 0xffffffffffffffffull},
        {0ull, 1ull, 0x8000000000000000ull, 0x0000010000000000ull},
    };
    const float posInf = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    const float qNan = std::numeric_limits<float>::quiet_NaN();
    const float inFlt[3][4] = {
        {posInf, negInf, 1.0f, qNan},
        {negInf, 0.0f, qNan, posInf},
        {0.0f, qNan, posInf, negInf},
    };
    const uint32_t initAdj[3][4] = {
        {101u, 102u, 103u, 104u},
        {105u, 106u, 107u, 108u},
        {109u, 110u, 111u, 112u},
    };
    const uint32_t zeroFPC[9][4]{};

    rt.uploadBuffer(bufInI64.Get(), inI64, sizeof(inI64));
    rt.uploadBuffer(bufInU64.Get(), inU64, sizeof(inU64));
    rt.uploadBuffer(bufInFlt.Get(), inFlt, sizeof(inFlt));
    rt.uploadBuffer(bufOutAdj.Get(), initAdj, sizeof(initAdj));
    rt.uploadBuffer(bufOutFPC.Get(), zeroFPC, sizeof(zeroFPC));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 6, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();

    auto makeSrv = [&](ID3D12Resource* r, UINT num, UINT stride, UINT idx) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = num;
        d.Buffer.StructureByteStride = stride;
        rt.device->CreateShaderResourceView(r, &d, {cpu.ptr + idx * inc});
    };
    auto makeUav = [&](ID3D12Resource* r, UINT num, UINT stride, UINT idx) {
        D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        d.Buffer.NumElements = num;
        d.Buffer.StructureByteStride = stride;
        rt.device->CreateUnorderedAccessView(r, nullptr, &d, {cpu.ptr + idx * inc});
    };

    makeSrv(bufInI64.Get(), 2, 32, 0);
    makeSrv(bufInU64.Get(), 2, 32, 1);
    makeSrv(bufInFlt.Get(), 3, 16, 2);
    makeUav(bufOutBits.Get(), 5, 16, 3);
    makeUav(bufOutAdj.Get(), 3, 16, 4);
    makeUav(bufOutFPC.Get(), 9, 16, 5);

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto bitsRaw = rt.readbackBuffer(bufOutBits.Get(), 5 * 16);
    auto adjRaw = rt.readbackBuffer(bufOutAdj.Get(), 3 * 16);
    auto fpcRaw = rt.readbackBuffer(bufOutFPC.Get(), 9 * 16);

    uint32_t outBits[5][4];
    uint32_t outAdj[3][4];
    uint32_t outFPC[9][4];
    std::memcpy(outBits, bitsRaw.data(), sizeof(outBits));
    std::memcpy(outAdj, adjRaw.data(), sizeof(outAdj));
    std::memcpy(outFPC, fpcRaw.data(), sizeof(outFPC));

    const uint32_t expBits[5][4] = {
        {3u, 0u, 32u, 64u},
        {3u, 0u, 32u, 64u},
        {0xFFFFFFFFu, 0u, 63u, 40u},
        {0xFFFFFFFFu, 0u, 63u, 40u},
        {0xFFFFFFFFu, 0xFFFFFFFFu, 3u, 40u},
    };
    const uint32_t expAdj[3][4] = {
        {3u, 0u, 32u, 64u},
        {3u, 0u, 32u, 64u},
        {3u, 0u, 111u, 112u},
    };
    const uint32_t expFPC[9][4] = {
        {1u, 1u, 0u, 0u},
        {1u, 1u, 0u, 0u},
        {1u, 1u, 0u, 0u},
        {1u, 0u, 0u, 1u},
        {0u, 0u, 1u, 1u},
        {1u, 0u, 0u, 0u},
        {0u, 1u, 0u, 0u},
        {0u, 0u, 0u, 1u},
        {0u, 0u, 1u, 0u},
    };

    bool ok = true;
    for (int r = 0; r < 5; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (outBits[r][c] != expBits[r][c]) {
                ok = false;
                std::cout << "outBits[" << r << "][" << c << "] = " << outBits[r][c]
                          << " (expected " << expBits[r][c] << ")\n";
            }
        }
    }
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (outAdj[r][c] != expAdj[r][c]) {
                ok = false;
                std::cout << "outAdj[" << r << "][" << c << "] = " << outAdj[r][c]
                          << " (expected " << expAdj[r][c] << ")\n";
            }
        }
    }
    for (int r = 0; r < 9; ++r) {
        for (int c = 0; c < 4; ++c) {
            if (outFPC[r][c] != expFPC[r][c]) {
                ok = false;
                std::cout << "outFPC[" << r << "][" << c << "] = " << outFPC[r][c]
                          << " (expected " << expFPC[r][c] << ")\n";
            }
        }
    }
    if (ok) std::cout << "PASS\n";
    return ok;
}

uint32_t floatBits(float v) {
    uint32_t b;
    std::memcpy(&b, &v, sizeof(b));
    return b;
}

float flushDenormInput(float v) {
    if (std::isnan(v) || std::isinf(v) || v == 0.0f) return v;
    if (std::fabs(v) < std::numeric_limits<float>::min()) {
        return std::signbit(v) ? -0.0f : 0.0f;
    }
    return v;
}

uint64_t ulpDiffFloat(float got, double refD) {
    const double fltMax = static_cast<double>(std::numeric_limits<float>::max());
    const double fltMin = static_cast<double>(std::numeric_limits<float>::min());
    if (std::isnan(refD)) return std::isnan(got) ? 0 : UINT64_MAX;
    if (std::isnan(got)) return UINT64_MAX;
    if (std::isinf(refD) || std::fabs(refD) > fltMax) {
        bool wantNeg = std::signbit(refD);
        return (std::isinf(got) && (std::signbit(got) == wantNeg)) ? 0 : UINT64_MAX;
    }
    if (std::isinf(got)) return UINT64_MAX;
    if (std::fabs(refD) < fltMin) {
        return (std::fabs(got) <= std::numeric_limits<float>::min()) ? 0 : UINT64_MAX;
    }
    float refF = static_cast<float>(refD);
    auto toOrdered = [](float f) -> int64_t {
        uint32_t u = floatBits(f);
        return (u & 0x80000000u) ? -static_cast<int64_t>(u & 0x7FFFFFFFu) : static_cast<int64_t>(u);
    };
    int64_t d = toOrdered(got) - toOrdered(refF);
    return static_cast<uint64_t>(d < 0 ? -d : d);
}

bool testTranscendentalMath(testutil::TestRuntime& rt) {
    std::cout << "=== Test: transcendental_math ===\n";
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[0].BaseShaderRegister = 0;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 2;
    rp.DescriptorTable.pDescriptorRanges = ranges;

    auto rs = createRS(rt.device.Get(), 1, &rp);
    auto pso = createCS(rt.device.Get(), rs.Get(), "transcendental_math.dxil");

    constexpr UINT kLanes = 64;
    constexpr UINT kFuncs = 16;
    const float posInf = std::numeric_limits<float>::infinity();
    const float negInf = -std::numeric_limits<float>::infinity();
    const float qNan = std::numeric_limits<float>::quiet_NaN();

    float inX[kLanes]{};
    float inY[kLanes]{};

    const float fixedX[25] = {
        0.0f, -0.0f, 0x1.5p-130f, -0x1.5p-130f, posInf, negInf, qNan,
        1.5f, -1.5f, 1.0f, -1.0f, 1e-6f, -1e-6f, 0.9999f, -0.9999f,
        89.0f, -89.0f, 90.0f, -90.0f, 127.5f, 128.0f, -126.0f, -150.0f,
        1.25f, 1.25f
    };
    const float fixedY[25] = {
        2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        -3.0f, 2.5f, 0.0f, 3.0f, 1.0f, 1.0f, 4.0f, 4.0f,
        0.25f, 0.25f, 0.1f, 0.1f, 0.1f, 0.1f, 0.0f, 0.0f,
        -150.0f, 130.0f
    };
    for (UINT i = 0; i < 25; ++i) {
        inX[i] = fixedX[i];
        inY[i] = fixedY[i];
    }
    for (UINT i = 25; i < 45; ++i) {
        float t = static_cast<float>(i - 25) / 19.0f; // [0, 1]
        inX[i] = -0.95f + 1.90f * t;
        if (std::fabs(inX[i]) < 0.02f) inX[i] = 0.035f;
        inY[i] = -2.25f + 4.5f * t;
    }
    for (UINT i = 45; i < kLanes; ++i) {
        float k = static_cast<float>(i - 44);
        float mag = 0.17f + 0.61f * k; // avoids odd multiples of pi/2
        inX[i] = ((i & 1u) == 0u) ? mag : -mag;
        inY[i] = ((i % 3u) == 0u) ? -1.5f : 1.75f;
    }

    auto bufX = rt.createBuffer(sizeof(inX), D3D12_HEAP_TYPE_DEFAULT);
    auto bufY = rt.createBuffer(sizeof(inY), D3D12_HEAP_TYPE_DEFAULT);
    auto bufOut = rt.createBuffer(kFuncs * kLanes * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    rt.uploadBuffer(bufX.Get(), inX, sizeof(inX));
    rt.uploadBuffer(bufY.Get(), inY, sizeof(inY));

    D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "Heap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();

    auto makeSrv = [&](ID3D12Resource* r, UINT num, UINT idx) {
        D3D12_SHADER_RESOURCE_VIEW_DESC d{};
        d.Format = DXGI_FORMAT_UNKNOWN;
        d.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        d.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        d.Buffer.NumElements = num;
        d.Buffer.StructureByteStride = sizeof(float);
        rt.device->CreateShaderResourceView(r, &d, {cpu.ptr + idx * inc});
    };
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.NumElements = kFuncs * kLanes;
    uav.Buffer.StructureByteStride = sizeof(float);

    makeSrv(bufX.Get(), kLanes, 0);
    makeSrv(bufY.Get(), kLanes, 1);
    rt.device->CreateUnorderedAccessView(bufOut.Get(), nullptr, &uav, {cpu.ptr + 2 * inc});

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto outRaw = rt.readbackBuffer(bufOut.Get(), kFuncs * kLanes * sizeof(float));
    float out[kFuncs][kLanes];
    std::memcpy(out, outRaw.data(), sizeof(out));

    const char* fnNames[kFuncs] = {
        "exp2", "exp", "ldexp", "log2", "log", "log10", "pow",
        "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh"
    };

    bool ok = true;
    for (UINT i = 0; i < kLanes; ++i) {
        float xf = flushDenormInput(inX[i]);
        float yf = flushDenormInput(inY[i]);
        double xd = static_cast<double>(xf);
        double yd = static_cast<double>(yf);

        auto check = [&](UINT fn, double ref, uint64_t maxUlp) {
            float got = out[fn][i];
            uint64_t ulp = ulpDiffFloat(got, ref);
            if (ulp > maxUlp) {
                ok = false;
                std::cout << fnNames[fn] << "[lane " << i << "] (x=" << inX[i] << ", y=" << inY[i]
                          << ") got=" << got << " (0x" << std::hex << floatBits(got) << std::dec
                          << ") expected=" << static_cast<float>(ref) << " ulp=" << ulp << "\n";
            }
        };

        check(0, std::exp2(xd), 3);
        // DXC lowers HLSL exp(x) to exp2(float(x * log2(e))) in DXIL
        double expDxilArg = static_cast<double>(xf * 1.4426950408889634f);
        check(1, std::exp2(expDxilArg), 3);

        // HLSL ldexp(x, y) := x * exp2(y)
        double ldexpRef = xd * std::exp2(yd);
        check(2, ldexpRef, 4);

        // log2, log, log10: 0 -> -inf, < 0 -> NaN, +inf -> +inf
        double log2Ref = (xf == 0.0f) ? -std::numeric_limits<double>::infinity()
                       : (xf < 0.0f)  ? std::numeric_limits<double>::quiet_NaN()
                                      : std::log2(xd);
        double logRef  = (xf == 0.0f) ? -std::numeric_limits<double>::infinity()
                       : (xf < 0.0f)  ? std::numeric_limits<double>::quiet_NaN()
                                      : std::log(xd);
        double log10Ref = (xf == 0.0f) ? -std::numeric_limits<double>::infinity()
                        : (xf < 0.0f)  ? std::numeric_limits<double>::quiet_NaN()
                                       : std::log10(xd);
        check(3, log2Ref, 3);
        check(4, logRef, 3);
        check(5, log10Ref, 3);

        // DXC lowers HLSL pow(|x|, y) to exp2(float(y * float(log2(|x|))))
        float absXf = std::fabs(xf);
        double powRef;
        if (std::isnan(xd) || std::isnan(yd)) {
            powRef = std::numeric_limits<double>::quiet_NaN();
        } else if (absXf == 0.0f) {
            powRef = std::exp2(yd * (-std::numeric_limits<double>::infinity()));
        } else {
            float lg2 = static_cast<float>(std::log2(static_cast<double>(absXf)));
            float prod = yf * lg2;
            powRef = std::exp2(static_cast<double>(prod));
        }
        uint64_t powTol = 4 + static_cast<uint64_t>(std::ceil(std::fabs(yd) * 1.5));
        check(6, powRef, powTol);

        // sin, cos, tan: ±inf / NaN -> NaN
        bool trigDomInvalid = std::isnan(xd) || std::isinf(xd);
        check(7, trigDomInvalid ? std::numeric_limits<double>::quiet_NaN() : std::sin(xd), 3);
        check(8, trigDomInvalid ? std::numeric_limits<double>::quiet_NaN() : std::cos(xd), 3);
        check(9, trigDomInvalid ? std::numeric_limits<double>::quiet_NaN() : std::tan(xd), 4);

        // asin, acos: |x| > 1 -> NaN
        bool invTrigInvalid = std::isnan(xd) || std::fabs(xd) > 1.0;
        check(10, invTrigInvalid ? std::numeric_limits<double>::quiet_NaN() : std::asin(xd), 3);
        check(11, invTrigInvalid ? std::numeric_limits<double>::quiet_NaN() : std::acos(xd), 3);
        check(12, std::isnan(xd) ? std::numeric_limits<double>::quiet_NaN() : std::atan(xd), 3);

        // sinh, cosh, tanh: full domain including tiny |x|, near-overflow |x|=89, overflow |x|=90, ±Inf, NaN
        check(13, std::sinh(xd), 3);
        check(14, std::cosh(xd), 3);
        check(15, std::tanh(xd), 3);
    }

    // Explicitly verify small-input non-zero preservation (no catastrophic cancellation to 0)
    if (out[13][11] == 0.0f || out[15][11] == 0.0f) {
        ok = false;
        std::cout << "Catastrophic cancellation for tiny input x=1e-6: sinh="
                  << out[13][11] << " tanh=" << out[15][11] << "\n";
    }

    if (ok) std::cout << "PASS\n";
    return ok;
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    bool ok1 = testMatrixSwizzle(rt);
    bool ok2 = testQuadDerivatives(rt, false);
    bool ok2d = testQuadDerivatives(rt, true);
    bool ok3 = testMathDenormsBits(rt);
    bool ok4 = testBitAndFPClassIntrinsics(rt);
    bool ok5 = testTranscendentalMath(rt);
    return (ok1 && ok2 && ok2d && ok3 && ok4 && ok5) ? 0 : 1;
}
