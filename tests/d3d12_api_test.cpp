#include "test_common.h"

#include <cstring>
#include <vector>

int main() {
    // 1. DXCore adapter factory enumeration
    ComPtr<IDXCoreAdapterFactory> factory;
    testutil::checkHR(NacelleCreateAdapterFactory(IID_PPV_ARGS(&factory)), "NacelleCreateAdapterFactory");

    const GUID attrs[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE};
    ComPtr<IDXCoreAdapterList> list;
    testutil::checkHR(factory->CreateAdapterList(1, attrs, IID_PPV_ARGS(&list)), "CreateAdapterList");
    assert(list->GetAdapterCount() >= 1);

    ComPtr<IDXCoreAdapter> adapter;
    testutil::checkHR(list->GetAdapter(0, IID_PPV_ARGS(&adapter)), "GetAdapter(0)");
    char descBuf[128]{};
    testutil::checkHR(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription, sizeof(descBuf), descBuf), "GetDriverDescription");
    assert(std::string(descBuf).find("Nacelle") != std::string::npos);

    bool isHw = true;
    testutil::checkHR(adapter->GetProperty(DXCoreAdapterProperty::IsHardware, sizeof(isHw), &isHw), "IsHardware");
    assert(!isHw);

    // 2. D3D12 Debug interface & Device creation from DXCore adapter
    ComPtr<ID3D12Debug> debug;
    testutil::checkHR(NacelleGetDebugInterface(IID_PPV_ARGS(&debug)), "NacelleGetDebugInterface");
    debug->EnableDebugLayer();

    ComPtr<ID3D12Device> adapterDev;
    testutil::checkHR(NacelleCreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&adapterDev)), "NacelleCreateDevice");

    // 3. TestRuntime copy & fence test
    testutil::TestRuntime rt;
    rt.init();

    constexpr UINT kCount = 256;
    constexpr UINT64 kBytes = kCount * sizeof(uint32_t);
    std::vector<uint32_t> srcData(kCount);
    for (UINT i = 0; i < kCount; ++i) srcData[i] = i * 3u + 7u;

    auto bufA = rt.createBuffer(kBytes, D3D12_HEAP_TYPE_DEFAULT);
    auto bufB = rt.createBuffer(kBytes, D3D12_HEAP_TYPE_DEFAULT);
    rt.uploadBuffer(bufA.Get(), srcData.data(), kBytes);

    rt.resetCmd();
    rt.cmdList->CopyBufferRegion(bufB.Get(), 0, bufA.Get(), 0, kBytes);
    rt.executeAndWait();

    auto outBytes = rt.readbackBuffer(bufB.Get(), kBytes);
    std::vector<uint32_t> dstData(kCount);
    std::memcpy(dstData.data(), outBytes.data(), kBytes);
    assert(dstData == srcData);

    // 4. Descriptor heap & GPU virtual address check
    D3D12_DESCRIPTOR_HEAP_DESC hd{};
    hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    hd.NumDescriptors = 8;
    hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap)), "CreateDescriptorHeap");
    assert(heap->GetCPUDescriptorHandleForHeapStart().ptr != 0);
    assert(heap->GetGPUDescriptorHandleForHeapStart().ptr != 0);

    // 5. Graphics & texture feature capability flags
    D3D12_FEATURE_DATA_D3D12_OPTIONS opts0{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts0, sizeof(opts0)), "OPTIONS");
    assert(opts0.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_2);
    assert(opts0.PSSpecifiedStencilRefSupported == TRUE);
    assert(opts0.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation == TRUE);

    D3D12_FEATURE_DATA_D3D12_OPTIONS3 opts3{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS3, &opts3, sizeof(opts3)), "OPTIONS3");
    assert(opts3.BarycentricsSupported == TRUE);

    D3D12_FEATURE_DATA_D3D12_OPTIONS6 opts6{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS6, &opts6, sizeof(opts6)), "OPTIONS6");
    assert(opts6.VariableShadingRateTier >= D3D12_VARIABLE_SHADING_RATE_TIER_1);

    std::printf("d3d12_api_test PASSED (adapter=\"%s\")\n", descBuf);
    return 0;
}
