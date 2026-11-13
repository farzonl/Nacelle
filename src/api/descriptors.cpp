#include "descriptors.h"

#include <algorithm>
#include <cstring>

namespace nacelle {

NacelleDescriptorHeap::NacelleDescriptorHeap(ID3D12Device* device, const D3D12_DESCRIPTOR_HEAP_DESC& desc)
    : D3D12PageableBase(device), desc_(desc) {
    descriptors_.resize(desc.NumDescriptors);
}

HRESULT STDMETHODCALLTYPE NacelleDescriptorHeap::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12DescriptorHeap))) {
        *ppvObject = static_cast<ID3D12DescriptorHeap*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

#if defined(_MSC_VER) || !defined(_WIN32)
D3D12_DESCRIPTOR_HEAP_DESC STDMETHODCALLTYPE NacelleDescriptorHeap::GetDesc() {
    return desc_;
}
D3D12_CPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE NacelleDescriptorHeap::GetCPUDescriptorHandleForHeapStart() {
    return D3D12_CPU_DESCRIPTOR_HANDLE{reinterpret_cast<SIZE_T>(descriptors_.data())};
}
D3D12_GPU_DESCRIPTOR_HANDLE STDMETHODCALLTYPE NacelleDescriptorHeap::GetGPUDescriptorHandleForHeapStart() {
    return D3D12_GPU_DESCRIPTOR_HANDLE{reinterpret_cast<UINT64>(descriptors_.data())};
}
#else
D3D12_DESCRIPTOR_HEAP_DESC* STDMETHODCALLTYPE NacelleDescriptorHeap::GetDesc(D3D12_DESCRIPTOR_HEAP_DESC* RetVal) {
    *RetVal = desc_;
    return RetVal;
}
D3D12_CPU_DESCRIPTOR_HANDLE* STDMETHODCALLTYPE NacelleDescriptorHeap::GetCPUDescriptorHandleForHeapStart(
    D3D12_CPU_DESCRIPTOR_HANDLE* RetVal) {
    RetVal->ptr = reinterpret_cast<SIZE_T>(descriptors_.data());
    return RetVal;
}
D3D12_GPU_DESCRIPTOR_HANDLE* STDMETHODCALLTYPE NacelleDescriptorHeap::GetGPUDescriptorHandleForHeapStart(
    D3D12_GPU_DESCRIPTOR_HANDLE* RetVal) {
    RetVal->ptr = reinterpret_cast<UINT64>(descriptors_.data());
    return RetVal;
}
#endif

void DescriptorOps::createCBV(const D3D12_CONSTANT_BUFFER_VIEW_DESC* pDesc,
                              D3D12_CPU_DESCRIPTOR_HANDLE dest) {
    if (!dest.ptr || !pDesc) return;
    auto* sd = reinterpret_cast<NacelleDescriptor*>(dest.ptr);
    *sd = NacelleDescriptor{};
    sd->base_ptr = reinterpret_cast<void*>(pDesc->BufferLocation);
    sd->width = pDesc->SizeInBytes;
    sd->byte_size = pDesc->SizeInBytes;
    sd->elem_stride = 16;
}

void DescriptorOps::createSRV(ID3D12Resource* pResource,
                              const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc,
                              D3D12_CPU_DESCRIPTOR_HANDLE dest) {
    if (!dest.ptr) return;
    auto* sd = reinterpret_cast<NacelleDescriptor*>(dest.ptr);
    *sd = NacelleDescriptor{};
    if (!pResource) return;

    auto* res = static_cast<NacelleResource*>(pResource);
    const D3D12_RESOURCE_DESC rdesc = res->resourceDesc();
    const DXGI_FORMAT fmt = (pDesc && pDesc->Format != DXGI_FORMAT_UNKNOWN) ? pDesc->Format : rdesc.Format;

    sd->format = static_cast<uint32_t>(fmt);
    sd->base_ptr = res->basePtr();

    if (rdesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        uint64_t firstElem = pDesc ? pDesc->Buffer.FirstElement : 0;
        uint32_t numElems = pDesc ? pDesc->Buffer.NumElements : static_cast<uint32_t>(rdesc.Width / 4);
        uint32_t stride = (pDesc && pDesc->Buffer.StructureByteStride > 0)
                              ? pDesc->Buffer.StructureByteStride
                              : (fmt != DXGI_FORMAT_UNKNOWN ? getFormatInfo(fmt).bytesPerBlock : 4);
        uint32_t isRaw = 0;
        if (pDesc && (pDesc->Buffer.Flags & D3D12_BUFFER_SRV_FLAG_RAW)) {
            stride = 4;
            isRaw = 1;
        }
        sd->base_ptr = res->basePtr() + firstElem * stride;
        sd->width = numElems;
        sd->elem_stride = stride;
        sd->byte_size = uint64_t(numElems) * stride;
        sd->view_flags = D3D12_SRV_DIMENSION_BUFFER | (isRaw << 8);
    } else {
        if (res->textureMetadata()->is_compressed) {
            res->ensureDecodedBC();
        }
        uint32_t mostDetailedMip = 0;
        uint32_t mipLevels = rdesc.MipLevels;
        uint32_t firstArraySlice = 0;
        D3D12_SRV_DIMENSION dim = D3D12_SRV_DIMENSION_TEXTURE2D;
        if (pDesc) {
            dim = pDesc->ViewDimension;
            switch (dim) {
            case D3D12_SRV_DIMENSION_TEXTURE1D:
                mostDetailedMip = pDesc->Texture1D.MostDetailedMip;
                mipLevels = pDesc->Texture1D.MipLevels;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
                mostDetailedMip = pDesc->Texture1DArray.MostDetailedMip;
                mipLevels = pDesc->Texture1DArray.MipLevels;
                firstArraySlice = pDesc->Texture1DArray.FirstArraySlice;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2D:
                mostDetailedMip = pDesc->Texture2D.MostDetailedMip;
                mipLevels = pDesc->Texture2D.MipLevels;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
                mostDetailedMip = pDesc->Texture2DArray.MostDetailedMip;
                mipLevels = pDesc->Texture2DArray.MipLevels;
                firstArraySlice = pDesc->Texture2DArray.FirstArraySlice;
                break;
            case D3D12_SRV_DIMENSION_TEXTURE3D:
                mostDetailedMip = pDesc->Texture3D.MostDetailedMip;
                mipLevels = pDesc->Texture3D.MipLevels;
                break;
            case D3D12_SRV_DIMENSION_TEXTURECUBE:
                mostDetailedMip = pDesc->TextureCube.MostDetailedMip;
                mipLevels = pDesc->TextureCube.MipLevels;
                break;
            case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
                mostDetailedMip = pDesc->TextureCubeArray.MostDetailedMip;
                mipLevels = pDesc->TextureCubeArray.MipLevels;
                firstArraySlice = pDesc->TextureCubeArray.First2DArrayFace;
                break;
            default:
                break;
            }
        } else {
            if (rdesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D) dim = D3D12_SRV_DIMENSION_TEXTURE1D;
            else if (rdesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D) dim = D3D12_SRV_DIMENSION_TEXTURE3D;
        }
        if (mipLevels == UINT32_MAX || mipLevels == 0) {
            mipLevels = rdesc.MipLevels - mostDetailedMip;
        }

        const auto& mip0 = res->subresourceLayout(mostDetailedMip);
        sd->width = mip0.width;
        sd->height = mip0.height;
        sd->depth_or_array = (rdesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                                 ? mip0.depth
                                 : rdesc.DepthOrArraySize;
        sd->row_pitch = mip0.row_pitch;
        sd->mip_levels = mipLevels;
        sd->byte_size = res->totalBytes();
        sd->elem_stride = getFormatInfo(fmt).bytesPerBlock;
        sd->extra_ptr = res->textureMetadata();
        sd->view_flags = (static_cast<uint32_t>(dim) & 0xFFu) |
                         ((mostDetailedMip & 0xFFu) << 8) |
                         ((firstArraySlice & 0xFFFFu) << 16);
    }
}

void DescriptorOps::createUAV(ID3D12Resource* pResource,
                              ID3D12Resource* pCounterResource,
                              const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc,
                              D3D12_CPU_DESCRIPTOR_HANDLE dest) {
    if (!dest.ptr) return;
    auto* sd = reinterpret_cast<NacelleDescriptor*>(dest.ptr);
    *sd = NacelleDescriptor{};
    if (!pResource) return;

    auto* res = static_cast<NacelleResource*>(pResource);
    const D3D12_RESOURCE_DESC rdesc = res->resourceDesc();
    const DXGI_FORMAT fmt = (pDesc && pDesc->Format != DXGI_FORMAT_UNKNOWN) ? pDesc->Format : rdesc.Format;

    sd->format = static_cast<uint32_t>(fmt);
    sd->base_ptr = res->basePtr();

    if (rdesc.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER) {
        uint64_t firstElem = pDesc ? pDesc->Buffer.FirstElement : 0;
        uint32_t numElems = pDesc ? pDesc->Buffer.NumElements : static_cast<uint32_t>(rdesc.Width / 4);
        uint32_t stride = (pDesc && pDesc->Buffer.StructureByteStride > 0)
                              ? pDesc->Buffer.StructureByteStride
                              : (fmt != DXGI_FORMAT_UNKNOWN ? getFormatInfo(fmt).bytesPerBlock : 4);
        uint32_t isRaw = 0;
        if (pDesc && (pDesc->Buffer.Flags & D3D12_BUFFER_UAV_FLAG_RAW)) {
            stride = 4;
            isRaw = 1;
        }
        sd->base_ptr = res->basePtr() + firstElem * stride;
        sd->width = numElems;
        sd->elem_stride = stride;
        sd->byte_size = uint64_t(numElems) * stride;
        sd->view_flags = D3D12_UAV_DIMENSION_BUFFER | (isRaw << 8);
        if (pCounterResource) {
            auto* counterRes = static_cast<NacelleResource*>(pCounterResource);
            sd->extra_ptr = counterRes->basePtr() + pDesc->Buffer.CounterOffsetInBytes;
        }
    } else {
        uint32_t mipSlice = pDesc ? pDesc->Texture2D.MipSlice : 0;
        const auto& mip = res->subresourceLayout(mipSlice);
        sd->base_ptr = res->basePtr() + mip.offset;
        sd->width = mip.width;
        sd->height = mip.height;
        sd->depth_or_array = (rdesc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D)
                                 ? mip.depth
                                 : rdesc.DepthOrArraySize;
        sd->row_pitch = mip.row_pitch;
        sd->mip_levels = 1;
        sd->byte_size = res->totalBytes() - mip.offset;
        sd->elem_stride = getFormatInfo(fmt).bytesPerBlock;
        sd->extra_ptr = res->textureMetadata();
        sd->view_flags = pDesc ? static_cast<uint32_t>(pDesc->ViewDimension) : D3D12_UAV_DIMENSION_TEXTURE2D;
    }
}

void DescriptorOps::createRTV(ID3D12Resource* pResource,
                              const D3D12_RENDER_TARGET_VIEW_DESC* pDesc,
                              D3D12_CPU_DESCRIPTOR_HANDLE dest) {
    if (!dest.ptr) return;
    auto* rtv = reinterpret_cast<NacelleRenderTargetDescriptor*>(dest.ptr);
    *rtv = NacelleRenderTargetDescriptor{};
    if (!pResource) return;

    auto* res = static_cast<NacelleResource*>(pResource);
    const auto rdesc = res->resourceDesc();
    rtv->resource = res;
    rtv->format = (pDesc && pDesc->Format != DXGI_FORMAT_UNKNOWN) ? pDesc->Format : rdesc.Format;
    if (pDesc) {
        if (pDesc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2D) {
            rtv->mip_slice = pDesc->Texture2D.MipSlice;
        } else if (pDesc->ViewDimension == D3D12_RTV_DIMENSION_TEXTURE2DARRAY) {
            rtv->mip_slice = pDesc->Texture2DArray.MipSlice;
            rtv->first_array_slice = pDesc->Texture2DArray.FirstArraySlice;
            rtv->array_size = pDesc->Texture2DArray.ArraySize;
        }
    }
}

void DescriptorOps::createDSV(ID3D12Resource* pResource,
                              const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc,
                              D3D12_CPU_DESCRIPTOR_HANDLE dest) {
    if (!dest.ptr) return;
    auto* dsv = reinterpret_cast<NacelleRenderTargetDescriptor*>(dest.ptr);
    *dsv = NacelleRenderTargetDescriptor{};
    if (!pResource) return;

    auto* res = static_cast<NacelleResource*>(pResource);
    const auto rdesc = res->resourceDesc();
    dsv->resource = res;
    dsv->format = (pDesc && pDesc->Format != DXGI_FORMAT_UNKNOWN) ? pDesc->Format : rdesc.Format;
    if (pDesc) {
        if (pDesc->ViewDimension == D3D12_DSV_DIMENSION_TEXTURE2D) {
            dsv->mip_slice = pDesc->Texture2D.MipSlice;
        }
        dsv->flags = static_cast<uint32_t>(pDesc->Flags);
    }
}

void DescriptorOps::populateSamplerDesc(const D3D12_SAMPLER_DESC& desc, NacelleDescriptor& out) {
    out = NacelleDescriptor{};
    uint32_t u = (std::max<uint32_t>(1, desc.AddressU) - 1) & 0x7u;
    uint32_t v = (std::max<uint32_t>(1, desc.AddressV) - 1) & 0x7u;
    uint32_t w = (std::max<uint32_t>(1, desc.AddressW) - 1) & 0x7u;
    uint32_t cmp = static_cast<uint32_t>(desc.ComparisonFunc) & 0xFu;
    uint32_t aniso = std::clamp<uint32_t>(desc.MaxAnisotropy, 1, 16) & 0x1Fu;
    uint32_t border = 0;
    if (desc.BorderColor[3] > 0.5f) {
        border = (desc.BorderColor[0] > 0.5f) ? 2u : 1u;
    }

    out.sampler_flags = (static_cast<uint32_t>(desc.Filter) & 0xFFu) |
                        (u << SamplerFlags::kAddrUShift) |
                        (v << SamplerFlags::kAddrVShift) |
                        (w << SamplerFlags::kAddrWShift) |
                        (cmp << SamplerFlags::kCmpFuncShift) |
                        (aniso << SamplerFlags::kMaxAnisoShift) |
                        (border << SamplerFlags::kBorderShift);
    out.mip_lod_bias = desc.MipLODBias;
    // Pack min_lod (clamped 0..15.99 in 8.8) & max_lod (in 8.8) into view_flags
    uint16_t minLodQ8 = static_cast<uint16_t>(std::clamp(desc.MinLOD, 0.0f, 255.0f) * 256.0f);
    uint16_t maxLodQ8 = static_cast<uint16_t>(std::clamp(desc.MaxLOD, 0.0f, 255.0f) * 256.0f);
    out.view_flags = static_cast<uint32_t>(minLodQ8) | (static_cast<uint32_t>(maxLodQ8) << 16);
}

void DescriptorOps::createSampler(const D3D12_SAMPLER_DESC* pDesc,
                                  D3D12_CPU_DESCRIPTOR_HANDLE dest) {
    if (!dest.ptr || !pDesc) return;
    auto* sd = reinterpret_cast<NacelleDescriptor*>(dest.ptr);
    populateSamplerDesc(*pDesc, *sd);
}

void DescriptorOps::copyDescriptors(
    UINT NumDestDescriptorRanges,
    const D3D12_CPU_DESCRIPTOR_HANDLE* pDestDescriptorRangeStarts,
    const UINT* pDestDescriptorRangeSizes,
    UINT NumSrcDescriptorRanges,
    const D3D12_CPU_DESCRIPTOR_HANDLE* pSrcDescriptorRangeStarts,
    const UINT* pSrcDescriptorRangeSizes) {
    UINT dstRange = 0, dstOffset = 0;
    UINT srcRange = 0, srcOffset = 0;

    while (dstRange < NumDestDescriptorRanges && srcRange < NumSrcDescriptorRanges) {
        const UINT dstLen = pDestDescriptorRangeSizes ? pDestDescriptorRangeSizes[dstRange] : 1;
        const UINT srcLen = pSrcDescriptorRangeSizes ? pSrcDescriptorRangeSizes[srcRange] : 1;
        if (dstOffset >= dstLen) { ++dstRange; dstOffset = 0; continue; }
        if (srcOffset >= srcLen) { ++srcRange; srcOffset = 0; continue; }

        const UINT count = std::min(dstLen - dstOffset, srcLen - srcOffset);
        auto* dstPtr = reinterpret_cast<NacelleDescriptor*>(pDestDescriptorRangeStarts[dstRange].ptr) + dstOffset;
        const auto* srcPtr = reinterpret_cast<const NacelleDescriptor*>(pSrcDescriptorRangeStarts[srcRange].ptr) + srcOffset;
        std::memcpy(dstPtr, srcPtr, count * sizeof(NacelleDescriptor));

        dstOffset += count;
        srcOffset += count;
    }
}

} // namespace nacelle
