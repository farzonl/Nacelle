#include "test_common.h"

#include <cassert>
#include <cstring>
#include <vector>

namespace {

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

    // Build triangle BLAS + TLAS
    const float vertices[] = {
        0.0f,  1.0f, 0.0f,
       -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
    };
    auto vb = createUploadAndFill(rt, vertices, sizeof(vertices));

    D3D12_RAYTRACING_GEOMETRY_DESC geom{};
    geom.Type = D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
    geom.Flags = D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE;
    geom.Triangles.VertexBuffer.StartAddress = vb->GetGPUVirtualAddress();
    geom.Triangles.VertexBuffer.StrideInBytes = sizeof(float) * 3;
    geom.Triangles.VertexCount = 3;
    geom.Triangles.VertexFormat = DXGI_FORMAT_R32G32B32_FLOAT;

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS blasInputs{};
    blasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
    blasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    blasInputs.NumDescs = 1;
    blasInputs.pGeometryDescs = &geom;

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO blasInfo{};
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&blasInputs, &blasInfo);
    auto blasBuf = createASBuffer(rt, blasInfo.ResultDataMaxSizeInBytes);
    auto blasScratch = createASBuffer(rt, blasInfo.ScratchDataSizeInBytes);

    D3D12_RAYTRACING_INSTANCE_DESC inst{};
    inst.Transform[0][0] = 1.0f;
    inst.Transform[1][1] = 1.0f;
    inst.Transform[2][2] = 1.0f;
    inst.InstanceMask = 0xFF;
    inst.AccelerationStructure = blasBuf->GetGPUVirtualAddress();
    auto instBuf = createUploadAndFill(rt, &inst, sizeof(inst));

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS tlasInputs{};
    tlasInputs.Type = D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;
    tlasInputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
    tlasInputs.NumDescs = 1;
    tlasInputs.InstanceDescs = instBuf->GetGPUVirtualAddress();

    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO tlasInfo{};
    dev5->GetRaytracingAccelerationStructurePrebuildInfo(&tlasInputs, &tlasInfo);
    auto tlasBuf = createASBuffer(rt, tlasInfo.ResultDataMaxSizeInBytes);
    auto tlasScratch = createASBuffer(rt, tlasInfo.ScratchDataSizeInBytes);

    rt.resetCmd();
    ComPtr<ID3D12GraphicsCommandList4> cmdList4;
    testutil::checkHR(rt.cmdList->QueryInterface(IID_PPV_ARGS(&cmdList4)), "QueryCmdList4");

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildBlas{};
    buildBlas.DestAccelerationStructureData = blasBuf->GetGPUVirtualAddress();
    buildBlas.Inputs = blasInputs;
    buildBlas.ScratchAccelerationStructureData = blasScratch->GetGPUVirtualAddress();
    cmdList4->BuildRaytracingAccelerationStructure(&buildBlas, 0, nullptr);

    D3D12_RESOURCE_BARRIER uavBarrier{};
    uavBarrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    uavBarrier.UAV.pResource = blasBuf.Get();
    cmdList4->ResourceBarrier(1, &uavBarrier);

    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC buildTlas{};
    buildTlas.DestAccelerationStructureData = tlasBuf->GetGPUVirtualAddress();
    buildTlas.Inputs = tlasInputs;
    buildTlas.ScratchAccelerationStructureData = tlasScratch->GetGPUVirtualAddress();
    cmdList4->BuildRaytracingAccelerationStructure(&buildTlas, 0, nullptr);

    // Root signature: t0 (TLAS SRV) + u0 (Output UAV)
    D3D12_DESCRIPTOR_RANGE ranges[2]{
        {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 1, 0, 0, 0},
        {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 1},
    };
    D3D12_ROOT_PARAMETER rootParam{};
    rootParam.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rootParam.DescriptorTable = {2, ranges};
    D3D12_ROOT_SIGNATURE_DESC rsDesc{1, &rootParam, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    auto rootSig = rt.createRootSig(rsDesc);

    // State object
    auto dxil = testutil::loadShaderBytes("raytracing_pipeline.dxil");
    D3D12_DXIL_LIBRARY_DESC libDesc{};
    libDesc.DXILLibrary = {dxil.data(), dxil.size()};

    D3D12_HIT_GROUP_DESC hgDesc{};
    hgDesc.HitGroupExport = L"TriangleHitGroup";
    hgDesc.Type = D3D12_HIT_GROUP_TYPE_TRIANGLES;
    hgDesc.ClosestHitShaderImport = L"ClosestHitMain";

    D3D12_GLOBAL_ROOT_SIGNATURE globalRsDesc{};
    globalRsDesc.pGlobalRootSignature = rootSig.Get();

    D3D12_RAYTRACING_SHADER_CONFIG shaderCfg{};
    shaderCfg.MaxPayloadSizeInBytes = sizeof(uint32_t);
    shaderCfg.MaxAttributeSizeInBytes = sizeof(float) * 2;

    D3D12_RAYTRACING_PIPELINE_CONFIG pipeCfg{};
    pipeCfg.MaxTraceRecursionDepth = 1;

    D3D12_STATE_SUBOBJECT subobjects[5]{};
    subobjects[0] = {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY, &libDesc};
    subobjects[1] = {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP, &hgDesc};
    subobjects[2] = {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE, &globalRsDesc};
    subobjects[3] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG, &shaderCfg};
    subobjects[4] = {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG, &pipeCfg};

    D3D12_STATE_OBJECT_DESC stateObjDesc{};
    stateObjDesc.Type = D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE;
    stateObjDesc.NumSubobjects = 5;
    stateObjDesc.pSubobjects = subobjects;

    ComPtr<ID3D12StateObject> stateObj;
    testutil::checkHR(dev5->CreateStateObject(&stateObjDesc, IID_PPV_ARGS(&stateObj)), "CreateStateObject");

    ComPtr<ID3D12StateObjectProperties> props;
    testutil::checkHR(stateObj->QueryInterface(IID_PPV_ARGS(&props)), "QueryStateObjectProperties");

    void* raygenId = props->GetShaderIdentifier(L"RayGen");
    void* missId = props->GetShaderIdentifier(L"MissMain");
    void* hgId = props->GetShaderIdentifier(L"TriangleHitGroup");
    assert(raygenId != nullptr);
    assert(missId != nullptr);
    assert(hgId != nullptr);

    // SBT: 3 records aligned to 64 bytes
    constexpr UINT kRecordSize = D3D12_RAYTRACING_SHADER_TABLE_BYTE_ALIGNMENT;
    std::vector<uint8_t> sbtData(kRecordSize * 3, 0);
    std::memcpy(sbtData.data() + kRecordSize * 0, raygenId, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    std::memcpy(sbtData.data() + kRecordSize * 1, missId, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    std::memcpy(sbtData.data() + kRecordSize * 2, hgId, D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);
    auto sbtBuf = createUploadAndFill(rt, sbtData.data(), sbtData.size());

    // Descriptors & Output buffer
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)), "CreateHeap");
    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE srvCpu = heap->GetCPUDescriptorHandleForHeapStart();
    D3D12_CPU_DESCRIPTOR_HANDLE uavCpu{srvCpu.ptr + inc};

    D3D12_SHADER_RESOURCE_VIEW_DESC tlasSrv{};
    tlasSrv.ViewDimension = D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE;
    tlasSrv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    tlasSrv.RaytracingAccelerationStructure.Location = tlasBuf->GetGPUVirtualAddress();
    rt.device->CreateShaderResourceView(nullptr, &tlasSrv, srvCpu);

    auto outBuf = rt.createBuffer(sizeof(uint32_t) * 2, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto readbackBuf = rt.createBuffer(sizeof(uint32_t) * 2, D3D12_HEAP_TYPE_READBACK);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uavDesc{};
    uavDesc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uavDesc.Buffer.NumElements = 2;
    uavDesc.Buffer.StructureByteStride = sizeof(uint32_t);
    rt.device->CreateUnorderedAccessView(outBuf.Get(), nullptr, &uavDesc, uavCpu);

    ID3D12DescriptorHeap* heaps[] = {heap.Get()};
    cmdList4->SetDescriptorHeaps(1, heaps);
    cmdList4->SetComputeRootSignature(rootSig.Get());
    cmdList4->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
    cmdList4->SetPipelineState1(stateObj.Get());

    D3D12_DISPATCH_RAYS_DESC dispatchDesc{};
    dispatchDesc.RayGenerationShaderRecord.StartAddress = sbtBuf->GetGPUVirtualAddress() + kRecordSize * 0;
    dispatchDesc.RayGenerationShaderRecord.SizeInBytes = kRecordSize;
    dispatchDesc.MissShaderTable.StartAddress = sbtBuf->GetGPUVirtualAddress() + kRecordSize * 1;
    dispatchDesc.MissShaderTable.SizeInBytes = kRecordSize;
    dispatchDesc.MissShaderTable.StrideInBytes = kRecordSize;
    dispatchDesc.HitGroupTable.StartAddress = sbtBuf->GetGPUVirtualAddress() + kRecordSize * 2;
    dispatchDesc.HitGroupTable.SizeInBytes = kRecordSize;
    dispatchDesc.HitGroupTable.StrideInBytes = kRecordSize;
    dispatchDesc.Width = 2;
    dispatchDesc.Height = 1;
    dispatchDesc.Depth = 1;
    cmdList4->DispatchRays(&dispatchDesc);

    cmdList4->CopyResource(readbackBuf.Get(), outBuf.Get());
    rt.executeAndWait();

    void* rbPtr = nullptr;
    testutil::checkHR(readbackBuf->Map(0, nullptr, &rbPtr), "MapReadback");
    uint32_t results[2]{};
    std::memcpy(results, rbPtr, sizeof(results));
    readbackBuf->Unmap(0, nullptr);

    // Ray 0 hits triangle center (0,0,0) -> barycentrics (0.25, 0.25) -> 100 + 0 + 25 + 25 = 150, + dim(2)*1000 = 2150
    // Ray 1 misses -> 0xDEAD (57005) + dim(2)*1000 = 59005
    assert(results[0] == 2150u);
    assert(results[1] == 59005u);
    return 0;
}
