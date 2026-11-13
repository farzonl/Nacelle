#pragma once

#include <memory>
#include <vector>

#include "com_base.h"
#include "d3d12_types.h"
#include "resource.h"

namespace nacelle {

class NacelleDescriptorHeap : public D3D12PageableBase<ID3D12DescriptorHeap> {
public:
    NacelleDescriptorHeap(ID3D12Device* device, const D3D12_DESCRIPTOR_HEAP_DESC& desc);
    ~NacelleDescriptorHeap() override = default;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_DESCRIPTOR_HEAP_DESC STDMETHODCALLTYPE GetDesc() override;
    D3D12_CPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetCPUDescriptorHandleForHeapStart() override;
    D3D12_GPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE GetGPUDescriptorHandleForHeapStart() override;
#else
    D3D12_DESCRIPTOR_HEAP_DESC* STDMETHODCALLTYPE GetDesc(D3D12_DESCRIPTOR_HEAP_DESC* RetVal) override;
    D3D12_CPU_DESCRIPTOR_HANDLE* STDMETHODCALLTYPE GetCPUDescriptorHandleForHeapStart(
        D3D12_CPU_DESCRIPTOR_HANDLE* RetVal) override;
    D3D12_GPU_DESCRIPTOR_HANDLE* STDMETHODCALLTYPE GetGPUDescriptorHandleForHeapStart(
        D3D12_GPU_DESCRIPTOR_HANDLE* RetVal) override;
#endif

    NacelleDescriptor* data() { return descriptors_.data(); }
    const NacelleDescriptor* data() const { return descriptors_.data(); }
    uint32_t count() const { return desc_.NumDescriptors; }
    D3D12_DESCRIPTOR_HEAP_TYPE heapType() const { return desc_.Type; }

    static constexpr UINT kDescriptorStride = sizeof(NacelleDescriptor); // 64 bytes

private:
    D3D12_DESCRIPTOR_HEAP_DESC desc_{};
    std::vector<NacelleDescriptor> descriptors_;
    // Extra storage for custom border colors / texture view metadata owned by heap slots
    std::vector<std::unique_ptr<NacelleTextureMetadata>> viewMeta_;
    friend struct DescriptorOps;
};

struct DescriptorOps {
    static void createCBV(const D3D12_CONSTANT_BUFFER_VIEW_DESC* pDesc,
                          D3D12_CPU_DESCRIPTOR_HANDLE dest);
    static void createSRV(ID3D12Resource* pResource,
                          const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc,
                          D3D12_CPU_DESCRIPTOR_HANDLE dest);
    static void createUAV(ID3D12Resource* pResource,
                          ID3D12Resource* pCounterResource,
                          const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc,
                          D3D12_CPU_DESCRIPTOR_HANDLE dest);
    static void createRTV(ID3D12Resource* pResource,
                          const D3D12_RENDER_TARGET_VIEW_DESC* pDesc,
                          D3D12_CPU_DESCRIPTOR_HANDLE dest);
    static void createDSV(ID3D12Resource* pResource,
                          const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc,
                          D3D12_CPU_DESCRIPTOR_HANDLE dest);
    static void createSampler(const D3D12_SAMPLER_DESC* pDesc,
                              D3D12_CPU_DESCRIPTOR_HANDLE dest);
    static void populateSamplerDesc(const D3D12_SAMPLER_DESC& desc, NacelleDescriptor& out);
    static void copyDescriptors(
        UINT NumDestDescriptorRanges,
        const D3D12_CPU_DESCRIPTOR_HANDLE* pDestDescriptorRangeStarts,
        const UINT* pDestDescriptorRangeSizes,
        UINT NumSrcDescriptorRanges,
        const D3D12_CPU_DESCRIPTOR_HANDLE* pSrcDescriptorRangeStarts,
        const UINT* pSrcDescriptorRangeSizes);
};

} // namespace nacelle
