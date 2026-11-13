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

bool testQuadDerivatives(testutil::TestRuntime& rt) {
    std::cout << "=== Test: quad_derivatives ===\n";
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
    rt.cmdList->OMSetRenderTargets(1, &rtvH, FALSE, nullptr);
    const float clearCol[4] = {0, 0, 0, 0};
    rt.cmdList->ClearRenderTargetView(rtvH, clearCol, 0, nullptr);
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
    // rsqrt(-denorm) == -inf, rsqrt(+denorm) == +inf
    if (outF[3][0] != -std::numeric_limits<float>::infinity() ||
        outF[3][1] != std::numeric_limits<float>::infinity()) {
        ok = false;
        std::cout << "rsqrt denorm mismatch: " << outF[3][0] << ", " << outF[3][1] << "\n";
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

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    bool ok1 = testMatrixSwizzle(rt);
    bool ok2 = testQuadDerivatives(rt);
    bool ok3 = testMathDenormsBits(rt);
    return (ok1 && ok2 && ok3) ? 0 : 1;
}
