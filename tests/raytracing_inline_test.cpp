#include "test_common.h"

#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

float bitsToFloat(uint32_t u) {
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

ComPtr<ID3D12Resource> createUploadAndFill(testutil::TestRuntime& rt, const void* data, size_t bytes) {
    auto buf = rt.createBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD);
    void* mapped = nullptr;
    testutil::checkHR(buf->Map(0, nullptr, &mapped), "MapUpload");
    std::memcpy(mapped, data, bytes);
    buf->Unmap(0, nullptr);
    return buf;
}

ComPtr<ID3D12Resource> createASBuffer(testutil::TestRuntime& rt, UINT64 bytes) {
    D3D12_HEAP_PROPERTIES hp{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(bytes, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComPtr<ID3D12Resource> res;
    testutil::checkHR(rt.device->CreateCommittedResource(
        &hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE,
        nullptr, IID_PPV_ARGS(&res)),
        "CreateASBuffer");
    return res;
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    ComPtr<ID3D12Device5> dev5;
    testutil::checkHR(rt.device->QueryInterface(IID_PPV_ARGS(&dev5)), "QueryDevice5");

    // 1. Verify RaytracingTier 1.1 capability reporting
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 opts5{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &opts5, sizeof(opts5)), "CheckOptions5");
    assert(opts5.RaytracingTier >= D3D12_RAYTRACING_TIER_1_1);

    // 2. Build Triangle BLAS and Procedural AABB BLAS
    const float triVerts[] = {
        0.0f,  1.0f, 0.0f,
       -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
    };
    auto vb = createUploadAndFill(rt, triVerts, sizeof(triVerts));

    const D3D12_RAYTRACING_AABB aabbData{-1.0f, -1.0f, -0.5f, 1.0f, 1.0f, 0.5f};
    auto aabbBuf = createUploadAndFill(rt, &aabbData, sizeof(aabbData));

    D3D12_RAYTRACING_GEOMETRY_DESC triGeom{};
    triGeom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    triGeom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    triGeom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;
    triGeom.Triangles.VertexCount = 3;
    triGeom.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress();
    triGeom.Triangles.VertexBuffer.StrideInBytes = 12;

    D3D12_RAYTRACING_GEOMETRY_DESC aabbGeom{};
    aabbGeom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_PROCEDURAL_PRIMITIVE_AABBS;
    aabbGeom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_NONE;
    aabbGeom.AABBs.AABBCount = 1;
    aabbGeom.AABBs.AABBs.StartAddress = aabbBuf->GetGPUVirtualAddress();
    aabbGeom.AABBs.AABBs.StrideInBytes = sizeof(D3D12_RAYTRACING_AABB);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS triInputs{};
    triInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    triInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    triInputs.NumDescs = 1;
    triInputs.pGeometryDescs = &triGeom;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO triPrebuild{};
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&triInputs, &triPrebuild);
    assert(triPrebuild.ResultDataMaxSizeInBytes > 0);
    auto triBlas = createASBuffer(rt, triPrebuild.ResultDataMaxSizeInBytes);
    auto scratch = rt.createBuffer(std::max<UINT64>(4096, triPrebuild.ScratchDataSizeInBytes), D3D12_HEAP_TYPE_DEFAULT);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS aabbInputs{};
    aabbInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    aabbInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    aabbInputs.NumDescs = 1;
    aabbInputs.pGeometryDescs = &aabbGeom;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO aabbPrebuild{};
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&aabbInputs, &aabbPrebuild);
    auto aabbBlas = createASBuffer(rt, aabbPrebuild.ResultDataMaxSizeInBytes);

    // 3. Build TLAS with two instances (triangle at origin mask 0x01, AABB at x=5 mask 0xFF)
    D3D12_RAYTRACING_INSTANCE_DESC instances[2]{};
    instances[0].Transform[0][0] = 1.0f;
    instances[0].Transform[1][1] = 1.0f;
    instances[0].Transform[2][2] = 1.0f;
    instances[0].InstanceID = 42;
    instances[0].InstanceMask = 0x01;
    instances[0].AccelerationStructure = triBlas->GetGPUVirtualAddress();

    instances[1].Transform[0][0] = 1.0f;
    instances[1].Transform[0][3] = 5.0f;
    instances[1].Transform[1][1] = 1.0f;
    instances[1].Transform[2][2] = 1.0f;
    instances[1].InstanceID = 99;
    instances[1].InstanceMask = 0xFF;
    instances[1].AccelerationStructure = aabbBlas->GetGPUVirtualAddress();

    auto instBuf = createUploadAndFill(rt, instances, sizeof(instances));

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs{};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 2;
    tlasInputs.InstanceDescs = instBuf->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasPrebuild{};
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasPrebuild);
    auto tlas = createASBuffer(rt, tlasPrebuild.ResultDataMaxSizeInBytes);

    // 4. Create Compute PSO & Descriptors
    D3D12_DESCRIPTOR_RANGE ranges[2]{
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 1},
    };
    D3D12_ROOT_PARAMETER param{};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable = {2, ranges};
    D3D12_ROOT_SIGNATURE_DESC rsDesc{1, &param, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    auto rs = rt.createRootSig(rsDesc);

    auto csBytes = testutil::loadShaderBytes("raytracing_inline_cs.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.CS = {csBytes.data(), csBytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateRayQueryPSO");

    auto outBuf = rt.createBuffer(16 * sizeof(uint32_t), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto rbBuf = rt.createBuffer(16 * sizeof(uint32_t), D3D12_HEAP_TYPE_READBACK);

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)), "CreateHeap");
    UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpu{srvCpu.ptr + inc};

    D3D12_SHADER_RESOURCE_VIEW_DESC srvDesc{};
    srvDesc.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    srvDesc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srvDesc.RaytracingAccelerationStructure.Location = tlas->GetGPUVirtualAddress();
    rt.device->CreateShaderResourceView(nullptr, &srvDesc, srvCpu);

    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.NumElements = 16;
    uavDesc.Buffer.StructureByteStride = 4;
    rt.device->CreateUnorderedAccessView(outBuf.Get(), nullptr, &uavDesc, uavCpu);

    // 5. Record AS builds + Dispatch + Readback
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildTri{};
    buildTri.DestAccelerationStructureData = triBlas->GetGPUVirtualAddress();
    buildTri.Inputs = triInputs;
    buildTri.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildAabb{};
    buildAabb.DestAccelerationStructureData = aabbBlas->GetGPUVirtualAddress();
    buildAabb.Inputs = aabbInputs;
    buildAabb.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildTlas{};
    buildTlas.DestAccelerationStructureData = tlas->GetGPUVirtualAddress();
    buildTlas.Inputs = tlasInputs;
    buildTlas.ScratchAccelerationStructureData = scratch->GetGPUVirtualAddress();

    rt.resetCmd();
    ComPtr<ID3D12GraphicsCommandList4> cmd4;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmd4)), "QueryCmdList4");
    cmd4->BuildRaytracingAccelerationStructure(&buildTri, 0, nullptr);
    cmd4->BuildRaytracingAccelerationStructure(&buildAabb, 0, nullptr);
    cmd4->BuildRaytracingAccelerationStructure(&buildTlas, 0, nullptr);

    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.cmdList->CopyResource(rbBuf.Get(), outBuf.Get());
    rt.executeAndWait();

    uint32_t out[16]{};
    void* rbPtr = nullptr;
    testutil::checkHR(rbBuf->Map(0, nullptr, &rbPtr), "MapReadback");
    std::memcpy(out, rbPtr, sizeof(out));
    rbBuf->Unmap(0, nullptr);

    // Lane 0: triangle hit (status=1, instanceID=42, t=2.0, bary.x=0.25)
    assert(out[0] == 1);
    assert(out[1] == 42);
    assert(std::fabs(bitsToFloat(out[2]) - 2.0f) < 1e-4f);
    assert(std::fabs(bitsToFloat(out[3]) - 0.25f) < 1e-4f);

    // Lane 1: procedural hit (status=2, instanceID=99, t=1.5)
    assert(out[4] == 2);
    assert(out[5] == 99);
    assert(std::fabs(bitsToFloat(out[6]) - 1.5f) < 1e-4f);

    // Lane 2: miss (status=0)
    assert(out[8] == 0);
    assert(out[9] == 0xFFFFFFFFu);

    // Lane 3: masked out (status=0)
    assert(out[12] == 0);
    assert(out[13] == 0xFFFFFFFFu);

    std::printf("raytracing_inline_test PASSED\n");
    return 0;
}
