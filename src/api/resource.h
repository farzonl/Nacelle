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
    static constexpr uint32_t kTileSizeBytes = 65536;
    static constexpr uint64_t kReservedGpuVaFlag = 0x8000000000000000ULL;

    NacelleResource(ID3D12Device* device,
                    const D3D12_HEAP_PROPERTIES& heapProps,
                    const D3D12_RESOURCE_DESC& desc,
                    NacelleHeap* placedHeap,
                    uint64_t heapOffset,
                    bool isReserved = false);
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

    // Reserved / tiled resource accessors & operations:
    bool isReserved() const { return isReserved_; }
    uint32_t totalTileCount() const { return totalTiles_; }
    const D3D12_TILE_SHAPE& tileShape() const { return tileShape_; }
    const std::vector<D3D12_SUBRESOURCE_TILING>& subresourceTilings() const { return subresourceTilings_; }
    uint8_t* const* tileTablePtr() const { return tilePtrs_.data(); }

    uint8_t* tilePtr(uint32_t tileIdx) const {
        return (tileIdx < tilePtrs_.size()) ? tilePtrs_[tileIdx] : nullptr;
    }
    NacelleHeap* tileHeap(uint32_t tileIdx) const {
        return (tileIdx < tileHeaps_.size()) ? tileHeaps_[tileIdx] : nullptr;
    }
    uint32_t tileHeapIndex(uint32_t tileIdx) const {
        return (tileIdx < tileHeapIndices_.size()) ? tileHeapIndices_[tileIdx] : 0;
    }

    void setTileMapping(uint32_t tileIdx, NacelleHeap* heap, uint32_t heapTileIdx);
    void finalizeTileMappingsUpdate();

    void enumerateRegionTiles(const D3D12_TILED_RESOURCE_COORDINATE& coord,
                              const D3D12_TILE_REGION_SIZE* pSize,
                              std::vector<uint32_t>& outTileIndices) const;

    uint8_t* bufferBytePtr(uint64_t byteOffset);
    const uint8_t* bufferBytePtr(uint64_t byteOffset) const;

    uint8_t* texBlockPtr(uint32_t subresource, uint32_t bx, uint32_t by, uint32_t z);
    const uint8_t* texBlockPtr(uint32_t subresource, uint32_t bx, uint32_t by, uint32_t z) const;

    static void copyBufferBytes(NacelleResource* dst,
                                uint64_t dstOffset,
                                const NacelleResource* src,
                                uint64_t srcOffset,
                                uint64_t numBytes);

    static NacelleResource* resolveReservedGpuVA(D3D12_GPU_VIRTUAL_ADDRESS va,
                                                 uint64_t* outByteOffset = nullptr);

    static D3D12_TILE_SHAPE computeStandardTileShape(const D3D12_RESOURCE_DESC& desc);

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
    bool isReserved_ = false;
    uint64_t totalBytes_ = 0;
    std::vector<NacelleMipLayout> subresources_;
    NacelleTextureMetadata texMeta_{};

    // Tiling metadata & page table (allocated for reserved resources):
    D3D12_TILE_SHAPE tileShape_{};
    uint32_t tileBlocksX_ = 1;
    uint32_t tileBlocksY_ = 1;
    uint32_t tileBlocksZ_ = 1;
    uint32_t totalTiles_ = 0;
    std::vector<D3D12_SUBRESOURCE_TILING> subresourceTilings_;
    std::vector<uint8_t*> tilePtrs_;
    std::vector<NacelleHeap*> tileHeaps_;
    std::vector<uint32_t> tileHeapIndices_;
    D3D12_GPU_VIRTUAL_ADDRESS reservedGpuVa_ = 0;

    std::mutex bcMutex_;
    std::vector<float> bcDecodedPixels_;
    std::atomic<bool> bcDirty_{true};
};

} // namespace nacelle
