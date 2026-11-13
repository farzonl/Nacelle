#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "com_base.h"
#include "d3d12_types.h"
#include "../common/format_utils.h"

namespace nacelle {

class NacelleHeap : public D3D12PageableBase<ID3D12Heap> {
public:
    NacelleHeap(ID3D12Device* device, const D3D12_HEAP_DESC& desc);
    ~NacelleHeap() override;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_HEAP_DESC STDMETHODCALLTYPE GetDesc() override;
#else
    D3D12_HEAP_DESC* STDMETHODCALLTYPE GetDesc(D3D12_HEAP_DESC* RetVal) override;
#endif

    uint8_t* data() const { return memory_; }
    uint64_t size() const { return desc_.SizeInBytes; }

private:
    D3D12_HEAP_DESC desc_{};
    uint8_t* memory_ = nullptr;
};

class NacelleResource : public D3D12PageableBase<ID3D12Resource> {
public:
    NacelleResource(ID3D12Device* device,
                 const D3D12_HEAP_PROPERTIES& heapProps,
                 const D3D12_RESOURCE_DESC& desc,
                 NacelleHeap* placedHeap,
                 uint64_t heapOffset);
    ~NacelleResource() override;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    HRESULT STDMETHODCALLTYPE Map(
        UINT Subresource,
        const D3D12_RANGE* pReadRange,
        void** ppData) override;

    void STDMETHODCALLTYPE Unmap(
        UINT Subresource,
        const D3D12_RANGE* pWrittenRange) override;

#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_RESOURCE_DESC STDMETHODCALLTYPE GetDesc() override;
#else
    D3D12_RESOURCE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_RESOURCE_DESC* RetVal) override;
#endif

    D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE GetGPUVirtualAddress() override;

    HRESULT STDMETHODCALLTYPE WriteToSubresource(
        UINT DstSubresource,
        const D3D12_BOX* pDstBox,
        const void* pSrcData,
        UINT SrcRowPitch,
        UINT SrcDepthPitch) override;

    HRESULT STDMETHODCALLTYPE ReadFromSubresource(
        void* pDstData,
        UINT DstRowPitch,
        UINT DstDepthPitch,
        UINT SrcSubresource,
        const D3D12_BOX* pSrcBox) override;

    HRESULT STDMETHODCALLTYPE GetHeapProperties(
        D3D12_HEAP_PROPERTIES* pHeapProperties,
        D3D12_HEAP_FLAGS* pHeapFlags) override;

    uint8_t* basePtr() const { return data_; }
    uint64_t totalBytes() const { return totalBytes_; }
    const D3D12_RESOURCE_DESC& resourceDesc() const { return desc_; }
    uint32_t subresourceCount() const { return static_cast<uint32_t>(subresources_.size()); }
    const NacelleMipLayout& subresourceLayout(uint32_t sub) const { return subresources_[sub]; }
    NacelleTextureMetadata* textureMetadata() { return &texMeta_; }

    /// Ensures BCn textures have a decoded RGBA32F mip cache for fast sampling.
    void ensureDecodedBC();
    void invalidateDecodedCache();

    static uint64_t computeResourceFootprints(
        const D3D12_RESOURCE_DESC& desc,
        UINT firstSubresource,
        UINT numSubresources,
        UINT64 baseOffset,
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
        UINT* pNumRows,
        UINT64* pRowSizeInBytes,
        UINT64* pTotalBytes,
        bool alignPlacement = true);

private:
    D3D12_HEAP_PROPERTIES heapProps_{};
    D3D12_RESOURCE_DESC desc_{};
    NacelleHeap* placedHeap_ = nullptr; // AddRef'd if non-null
    uint8_t* data_ = nullptr;
    bool ownsMemory_ = false;
    uint64_t totalBytes_ = 0;
    std::vector<NacelleMipLayout> subresources_;
    NacelleTextureMetadata texMeta_{};

    std::mutex bcMutex_;
    std::vector<float> bcDecodedPixels_;
    std::atomic<bool> bcDirty_{true};
};

} // namespace nacelle
