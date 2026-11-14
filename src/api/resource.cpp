#include "resource.h"

#include <algorithm>
#include <cstring>

namespace nacelle {

static inline uint64_t alignUp(uint64_t v, uint64_t a) {
    return (v + a - 1) & ~(a - 1);
}

NacelleHeap::NacelleHeap(ID3D12Device* device, const D3D12_HEAP_DESC& desc)
    : D3D12PageableBase(device), desc_(desc) {
    memory_ = static_cast<uint8_t*>(alignedAlloc64(static_cast<size_t>(desc.SizeInBytes)));
    if (memory_ && desc.SizeInBytes > 0) {
        std::memset(memory_, 0, static_cast<size_t>(desc.SizeInBytes));
    }
}

NacelleHeap::~NacelleHeap() {
    alignedFree(memory_);
}

HRESULT STDMETHODCALLTYPE NacelleHeap::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12Heap))) {
        *ppvObject = static_cast<ID3D12Heap*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

#if defined(_MSC_VER) || !defined(_WIN32)
D3D12_HEAP_DESC STDMETHODCALLTYPE NacelleHeap::GetDesc() {
    return desc_;
}
#else
D3D12_HEAP_DESC* STDMETHODCALLTYPE NacelleHeap::GetDesc(D3D12_HEAP_DESC* RetVal) {
    *RetVal = desc_;
    return RetVal;
}
#endif

uint64_t NacelleResource::computeResourceFootprints(
    const D3D12_RESOURCE_DESC& desc,
    UINT firstSubresource,
    UINT numSubresources,
    UINT64 baseOffset,
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
    UINT* pNumRows,
    UINT64* pRowSizeInBytes,
    UINT64* pTotalBytes,
    bool alignPlacement) {
    if (desc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        if (firstSubresource == 0 && numSubresources >= 1) {
            if (pLayouts) {
                pLayouts[0].Offset = baseOffset;
                pLayouts[0].Footprint.Format = desc.Format;
                pLayouts[0].Footprint.Width = static_cast<UINT>(desc.Width);
                pLayouts[0].Footprint.Height = 1;
                pLayouts[0].Footprint.Depth = 1;
                pLayouts[0].Footprint.RowPitch = static_cast<UINT>(alignUp(desc.Width, 256));
            }
            if (pNumRows) pNumRows[0] = 1;
            if (pRowSizeInBytes) pRowSizeInBytes[0] = desc.Width;
        }
        if (pTotalBytes) *pTotalBytes = desc.Width;
        return desc.Width;
    }

    const UINT mipLevels = std::max<UINT>(1, desc.MipLevels);
    const UINT arraySize = (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                               ? 1
                               : std::max<UINT>(1, desc.DepthOrArraySize);
    const FormatInfo fi = getFormatInfo(desc.Format);

    UINT64 curOffset = baseOffset;
    UINT64 total = 0;

    for (UINT i = 0; i < numSubresources; ++i) {
        const UINT sub = firstSubresource + i;
        const UINT mip = sub % mipLevels;
        const UINT w = std::max<UINT>(1, static_cast<UINT>(desc.Width >> mip));
        const UINT h = std::max<UINT>(1, desc.Height >> mip);
        const UINT d = (desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                           ? std::max<UINT>(1, desc.DepthOrArraySize >> mip)
                           : 1;

        const UINT blocksX = std::max<UINT>(1, (w + fi.blockWidth - 1) / fi.blockWidth);
        const UINT blocksY = std::max<UINT>(1, (h + fi.blockHeight - 1) / fi.blockHeight);
        const UINT64 rowSize = static_cast<UINT64>(blocksX) * fi.bytesPerBlock;
        const UINT64 pitchAlign = alignPlacement ? D3D12_TEXTURE_DATA_PITCH_ALIGNMENT : 64;
        const UINT64 rowPitch = alignUp(rowSize, pitchAlign);
        const UINT64 slicePitch = rowPitch * blocksY;
        const UINT sampleCount = (!alignPlacement && desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D)
                                     ? std::max<UINT>(1, desc.SampleDesc.Count)
                                     : 1u;
        const UINT64 subSize = slicePitch * d * sampleCount;

        if (i > 0 && alignPlacement) {
            curOffset = alignUp(curOffset, D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);
        } else if (i > 0) {
            curOffset = alignUp(curOffset, 64);
        }

        if (pLayouts) {
            pLayouts[i].Offset = curOffset;
            pLayouts[i].Footprint.Format = desc.Format;
            pLayouts[i].Footprint.Width = w;
            pLayouts[i].Footprint.Height = h;
            pLayouts[i].Footprint.Depth = d;
            pLayouts[i].Footprint.RowPitch = static_cast<UINT>(rowPitch);
        }
        if (pNumRows) pNumRows[i] = blocksY;
        if (pRowSizeInBytes) pRowSizeInBytes[i] = rowSize;

        curOffset += subSize;
        total = curOffset - baseOffset;
    }

    if (pTotalBytes) *pTotalBytes = total;
    return total;
}

NacelleResource::NacelleResource(ID3D12Device* device,
                           const D3D12_HEAP_PROPERTIES& heapProps,
                           const D3D12_RESOURCE_DESC& desc,
                           NacelleHeap* placedHeap,
                           uint64_t heapOffset)
    : D3D12PageableBase(device), heapProps_(heapProps), desc_(desc), placedHeap_(placedHeap) {
    if (desc_.MipLevels == 0) {
        if (desc_.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
            desc_.MipLevels = 1;
        } else {
            uint64_t m = std::max<uint64_t>(desc_.Width, desc_.Height);
            if (desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) {
                m = std::max<uint64_t>(m, desc_.DepthOrArraySize);
            }
            uint16_t levels = 1;
            while (m > 1) {
                m >>= 1;
                ++levels;
            }
            desc_.MipLevels = levels;
        }
    }
    if (desc_.DepthOrArraySize == 0) desc_.DepthOrArraySize = 1;

    const UINT mipLevels = std::max<UINT>(1, desc_.MipLevels);
    const UINT arraySize = (desc_.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                               ? 1
                               : std::max<UINT>(1, desc_.DepthOrArraySize);
    const UINT totalSubs = (desc_.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) ? 1 : (mipLevels * arraySize);

    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(totalSubs);
    std::vector<UINT> numRows(totalSubs);
    totalBytes_ = computeResourceFootprints(
        desc_, 0, totalSubs, 0, layouts.data(), numRows.data(), nullptr, &totalBytes_, false);

    subresources_.resize(totalSubs);
    for (UINT i = 0; i < totalSubs; ++i) {
        subresources_[i].offset = layouts[i].Offset;
        subresources_[i].row_pitch = layouts[i].Footprint.RowPitch;
        subresources_[i].slice_pitch = layouts[i].Footprint.RowPitch * numRows[i];
        subresources_[i].width = layouts[i].Footprint.Width;
        subresources_[i].height = layouts[i].Footprint.Height;
        subresources_[i].depth = layouts[i].Footprint.Depth;
    }

    if (placedHeap_) {
        placedHeap_->AddRef();
        data_ = placedHeap_->data() + heapOffset;
        ownsMemory_ = false;
    } else {
        data_ = static_cast<uint8_t*>(alignedAlloc64(static_cast<size_t>(std::max<uint64_t>(64, totalBytes_))));
        if (data_ && totalBytes_ > 0) {
            std::memset(data_, 0, static_cast<size_t>(totalBytes_));
        }
        ownsMemory_ = true;
    }

    // Populate base texture metadata (slice 0 mips; SRV creation copies & adjusts for array/view)
    const FormatInfo fi = getFormatInfo(desc_.Format);
    texMeta_.array_size = arraySize;
    texMeta_.mip_levels = std::min<uint32_t>(mipLevels, NacelleTextureMetadata::kMaxMips);
    texMeta_.most_detailed_mip = 0;
    texMeta_.dimension = desc_.Dimension;
    texMeta_.is_compressed = fi.isCompressed ? 1u : 0u;
    texMeta_.array_slice_stride = (arraySize > 1 && mipLevels < subresources_.size())
                                      ? subresources_[mipLevels].offset
                                      : 0;
    for (uint32_t m = 0; m < texMeta_.mip_levels; ++m) {
        texMeta_.mips[m] = subresources_[m];
    }
}

NacelleResource::~NacelleResource() {
    if (ownsMemory_) {
        alignedFree(data_);
    }
    if (placedHeap_) {
        placedHeap_->Release();
    }
}

HRESULT STDMETHODCALLTYPE NacelleResource::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12Resource))) {
        *ppvObject = static_cast<ID3D12Resource*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

HRESULT STDMETHODCALLTYPE NacelleResource::Map(UINT Subresource, const D3D12_RANGE*, void** ppData) {
    if (!ppData) return E_INVALIDARG;
    if (Subresource >= subresources_.size()) return E_INVALIDARG;
    *ppData = data_ + subresources_[Subresource].offset;
    return S_OK;
}

void STDMETHODCALLTYPE NacelleResource::Unmap(UINT, const D3D12_RANGE* pWrittenRange) {
    if (!pWrittenRange || pWrittenRange->End > pWrittenRange->Begin) {
        invalidateDecodedCache();
        if (texMeta_.is_compressed && texMeta_.decoded_rgba32f != nullptr) {
            ensureDecodedBC();
        }
    }
}

#if defined(_MSC_VER) || !defined(_WIN32)
D3D12_RESOURCE_DESC STDMETHODCALLTYPE NacelleResource::GetDesc() {
    return desc_;
}
#else
D3D12_RESOURCE_DESC* STDMETHODCALLTYPE NacelleResource::GetDesc(D3D12_RESOURCE_DESC* RetVal) {
    *RetVal = desc_;
    return RetVal;
}
#endif

D3D12_GPU_VIRTUAL_ADDRESS STDMETHODCALLTYPE NacelleResource::GetGPUVirtualAddress() {
    return reinterpret_cast<D3D12_GPU_VIRTUAL_ADDRESS>(data_);
}

HRESULT STDMETHODCALLTYPE NacelleResource::WriteToSubresource(
    UINT DstSubresource,
    const D3D12_BOX* pDstBox,
    const void* pSrcData,
    UINT SrcRowPitch,
    UINT SrcDepthPitch) {
    if (!pSrcData || DstSubresource >= subresources_.size()) return E_INVALIDARG;
    const NacelleMipLayout& dst = subresources_[DstSubresource];
    const FormatInfo fi = getFormatInfo(desc_.Format);

    UINT x0 = pDstBox ? pDstBox->left : 0;
    UINT y0 = pDstBox ? pDstBox->top : 0;
    UINT z0 = pDstBox ? pDstBox->front : 0;
    UINT x1 = pDstBox ? pDstBox->right : dst.width;
    UINT y1 = pDstBox ? pDstBox->bottom : dst.height;
    UINT z1 = pDstBox ? pDstBox->back : dst.depth;

    UINT bx0 = x0 / fi.blockWidth;
    UINT by0 = y0 / fi.blockHeight;
    UINT bw = std::max<UINT>(1, (x1 - x0 + fi.blockWidth - 1) / fi.blockWidth);
    UINT bh = std::max<UINT>(1, (y1 - y0 + fi.blockHeight - 1) / fi.blockHeight);
    UINT copyBytes = bw * fi.bytesPerBlock;

    for (UINT z = z0; z < z1; ++z) {
        const uint8_t* srcSlice = static_cast<const uint8_t*>(pSrcData) + (z - z0) * SrcDepthPitch;
        uint8_t* dstSlice = data_ + dst.offset + z * dst.slice_pitch;
        for (UINT by = 0; by < bh; ++by) {
            std::memcpy(
                dstSlice + (by0 + by) * dst.row_pitch + bx0 * fi.bytesPerBlock,
                srcSlice + by * SrcRowPitch,
                copyBytes);
        }
    }
    invalidateDecodedCache();
    if (texMeta_.is_compressed && texMeta_.decoded_rgba32f != nullptr) {
        ensureDecodedBC();
    }
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleResource::ReadFromSubresource(
    void* pDstData,
    UINT DstRowPitch,
    UINT DstDepthPitch,
    UINT SrcSubresource,
    const D3D12_BOX* pSrcBox) {
    if (!pDstData || SrcSubresource >= subresources_.size()) return E_INVALIDARG;
    const NacelleMipLayout& src = subresources_[SrcSubresource];
    const FormatInfo fi = getFormatInfo(desc_.Format);

    UINT x0 = pSrcBox ? pSrcBox->left : 0;
    UINT y0 = pSrcBox ? pSrcBox->top : 0;
    UINT z0 = pSrcBox ? pSrcBox->front : 0;
    UINT x1 = pSrcBox ? pSrcBox->right : src.width;
    UINT y1 = pSrcBox ? pSrcBox->bottom : src.height;
    UINT z1 = pSrcBox ? pSrcBox->back : src.depth;

    UINT bx0 = x0 / fi.blockWidth;
    UINT by0 = y0 / fi.blockHeight;
    UINT bw = std::max<UINT>(1, (x1 - x0 + fi.blockWidth - 1) / fi.blockWidth);
    UINT bh = std::max<UINT>(1, (y1 - y0 + fi.blockHeight - 1) / fi.blockHeight);
    UINT copyBytes = bw * fi.bytesPerBlock;

    for (UINT z = z0; z < z1; ++z) {
        uint8_t* dstSlice = static_cast<uint8_t*>(pDstData) + (z - z0) * DstDepthPitch;
        const uint8_t* srcSlice = data_ + src.offset + z * src.slice_pitch;
        for (UINT by = 0; by < bh; ++by) {
            std::memcpy(
                dstSlice + by * DstRowPitch,
                srcSlice + (by0 + by) * src.row_pitch + bx0 * fi.bytesPerBlock,
                copyBytes);
        }
    }
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleResource::GetHeapProperties(
    D3D12_HEAP_PROPERTIES* pHeapProperties,
    D3D12_HEAP_FLAGS* pHeapFlags) {
    if (pHeapProperties) *pHeapProperties = heapProps_;
    if (pHeapFlags) *pHeapFlags = D3D12_HEAP_FLAG_NONE;
    return S_OK;
}

void NacelleResource::invalidateDecodedCache() {
    bcDirty_.store(true, std::memory_order_release);
}

void NacelleResource::ensureDecodedBC() {
    if (!texMeta_.is_compressed || !bcDirty_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> lock(bcMutex_);
    if (!bcDirty_.load(std::memory_order_relaxed)) return;

    const FormatInfo fi = getFormatInfo(desc_.Format);
    const uint32_t mips = texMeta_.mip_levels;
    const uint32_t arr = texMeta_.array_size;

    // Compute total RGBA32F float count across all mips & slices
    size_t totalFloats = 0;
    for (uint32_t m = 0; m < mips; ++m) {
        const auto& srcMip = subresources_[m];
        texMeta_.decoded_mips[m].offset = totalFloats * sizeof(float);
        texMeta_.decoded_mips[m].width = srcMip.width;
        texMeta_.decoded_mips[m].height = srcMip.height;
        texMeta_.decoded_mips[m].depth = srcMip.depth;
        texMeta_.decoded_mips[m].row_pitch = srcMip.width * 16;
        texMeta_.decoded_mips[m].slice_pitch = srcMip.width * srcMip.height * 16;
        totalFloats += size_t(srcMip.width) * srcMip.height * srcMip.depth * arr * 4;
    }

    bcDecodedPixels_.resize(totalFloats);
    float blockOut[64];

    for (uint32_t slice = 0; slice < arr; ++slice) {
        for (uint32_t m = 0; m < mips; ++m) {
            const uint32_t sub = slice * mips + m;
            const auto& src = subresources_[sub];
            const auto& dstMip = texMeta_.decoded_mips[m];
            const uint32_t bxCount = (src.width + 3) / 4;
            const uint32_t byCount = (src.height + 3) / 4;

            for (uint32_t z = 0; z < src.depth; ++z) {
                const uint8_t* srcSlice = data_ + src.offset + z * src.slice_pitch;
                float* dstSlice = bcDecodedPixels_.data() +
                                  (dstMip.offset / sizeof(float)) +
                                  size_t(slice * src.depth + z) * src.width * src.height * 4;

                for (uint32_t by = 0; by < byCount; ++by) {
                    for (uint32_t bx = 0; bx < bxCount; ++bx) {
                        const uint8_t* blk = srcSlice + by * src.row_pitch + bx * fi.bytesPerBlock;
                        decompressBCBlockRGBA32F(desc_.Format, blk, blockOut);
                        for (uint32_t py = 0; py < 4; ++py) {
                            const uint32_t y = by * 4 + py;
                            if (y >= src.height) break;
                            for (uint32_t px = 0; px < 4; ++px) {
                                const uint32_t x = bx * 4 + px;
                                if (x >= src.width) break;
                                float* dstPx = dstSlice + (size_t(y) * src.width + x) * 4;
                                const float* srcPx = blockOut + (py * 4 + px) * 4;
                                dstPx[0] = srcPx[0];
                                dstPx[1] = srcPx[1];
                                dstPx[2] = srcPx[2];
                                dstPx[3] = srcPx[3];
                            }
                        }
                    }
                }
            }
        }
    }

    texMeta_.decoded_rgba32f = bcDecodedPixels_.data();
    bcDirty_.store(false, std::memory_order_release);
}

} // namespace nacelle
