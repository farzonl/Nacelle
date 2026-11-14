#include "device.h"

#include "../compiler/dxil_loader.h"
#include "../compiler/jit_engine.h"
#include "../rasterizer/raytracing_bvh.h"
#include <llvm/IR/LLVMContext.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>

#include "../../include/nacelle/nacelle.h"

namespace nacelle {

NacelleDevice::~NacelleDevice() {
    waitForAllAsyncJITTasks();
}

HRESULT STDMETHODCALLTYPE NacelleDevice::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12Device)) ||
        InlineIsEqualGUID(riid, __uuidof(ID3D12Device1)) ||
        InlineIsEqualGUID(riid, __uuidof(ID3D12Device2)) ||
        InlineIsEqualGUID(riid, __uuidof(ID3D12Device3)) ||
        InlineIsEqualGUID(riid, __uuidof(ID3D12Device4)) ||
        InlineIsEqualGUID(riid, __uuidof(ID3D12Device5))) {
        *ppvObject = static_cast<ID3D12Device5*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12ObjectBase::QueryInterface(riid, ppvObject);
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateCommandQueue(
    const D3D12_COMMAND_QUEUE_DESC* pDesc, REFIID riid, void** ppCommandQueue) {
    if (!pDesc || !ppCommandQueue) return E_INVALIDARG;
    auto* obj = new NacelleCommandQueue(this, *pDesc);
    HRESULT hr = obj->QueryInterface(riid, ppCommandQueue);
    obj->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateCommandAllocator(
    D3D12_COMMAND_LIST_TYPE type, REFIID riid, void** ppCommandAllocator) {
    if (!ppCommandAllocator) return E_INVALIDARG;
    auto* obj = new NacelleCommandAllocator(this, type);
    HRESULT hr = obj->QueryInterface(riid, ppCommandAllocator);
    obj->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateGraphicsPipelineState(
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC* pDesc, REFIID riid, void** ppPipelineState) {
    if (!pDesc || !ppPipelineState) return E_INVALIDARG;
    auto* rs = static_cast<NacelleRootSignature*>(pDesc->pRootSignature);
    RootSignatureLayout emptyLayout;
    const RootSignatureLayout& layout = rs ? rs->layout() : emptyLayout;

    auto compiled = std::make_shared<CompiledGraphicsPSO>();
    std::string err;
    if (!compileGraphicsPSO(*pDesc, layout, *compiled, err)) {
        std::fprintf(stderr, "[Nacelle] compileGraphicsPSO error: %s\n", err.c_str());
        return E_FAIL;
    }

    auto* pso = new NacellePipelineState(this, *pDesc, rs, std::move(compiled));
    HRESULT hr = pso->QueryInterface(riid, ppPipelineState);
    pso->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateComputePipelineState(
    const D3D12_COMPUTE_PIPELINE_STATE_DESC* pDesc, REFIID riid, void** ppPipelineState) {
    if (!pDesc || !ppPipelineState) return E_INVALIDARG;
    auto* rs = static_cast<NacelleRootSignature*>(pDesc->pRootSignature);
    RootSignatureLayout emptyLayout;
    const RootSignatureLayout& layout = rs ? rs->layout() : emptyLayout;

    auto compiled = std::make_shared<CompiledComputePSO>();
    std::string err;
    if (!compileComputePSO(*pDesc, layout, *compiled, err)) {
        std::fprintf(stderr, "[Nacelle] compileComputePSO error: %s\n", err.c_str());
        return E_FAIL;
    }

    auto* pso = new NacellePipelineState(this, *pDesc, rs, std::move(compiled));
    HRESULT hr = pso->QueryInterface(riid, ppPipelineState);
    pso->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateCommandList(
    UINT, D3D12_COMMAND_LIST_TYPE type, ID3D12CommandAllocator* pAllocator,
    ID3D12PipelineState* pInitialState, REFIID riid, void** ppCommandList) {
    if (!ppCommandList) return E_INVALIDARG;
    auto* obj = new NacelleCommandList(this, type, static_cast<NacelleCommandAllocator*>(pAllocator), pInitialState);
    HRESULT hr = obj->QueryInterface(riid, ppCommandList);
    obj->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CheckFeatureSupport(
    D3D12_FEATURE Feature, void* pData, UINT size) {
    if (!pData) return E_INVALIDARG;
    switch (Feature) {
    case D3D12_FEATURE_D3D12_OPTIONS: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS*>(pData);
        std::memset(o, 0, sizeof(*o));
        o->ResourceBindingTier = D3D12_RESOURCE_BINDING_TIER_3;
        o->ResourceHeapTier = D3D12_RESOURCE_HEAP_TIER_2;
        o->TiledResourcesTier = D3D12_TILED_RESOURCES_TIER_2;
        o->DoublePrecisionFloatShaderOps = TRUE;
        o->OutputMergerLogicOp = TRUE;
        o->PSSpecifiedStencilRefSupported = TRUE;
        o->VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation = TRUE;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS1: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS1)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS1*>(pData);
        uint32_t w = getTargetSimdWidth();
        o->WaveOps = TRUE;
        o->WaveLaneCountMin = 4;
        o->WaveLaneCountMax = 128;
        o->TotalLaneCount = w * std::max<uint32_t>(1, std::thread::hardware_concurrency());
        o->ExpandedComputeResourceStates = TRUE;
        o->Int64ShaderOps = TRUE;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS3: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS3)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS3*>(pData);
        std::memset(o, 0, sizeof(*o));
        o->BarycentricsSupported = TRUE;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS4: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS4)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS4*>(pData);
        std::memset(o, 0, sizeof(*o));
        o->Native16BitShaderOpsSupported = TRUE;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS5: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS5)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS5*>(pData);
        std::memset(o, 0, sizeof(*o));
        o->SRVOnlyTiledResourceTier3 = FALSE;
        o->RenderPassesTier = D3D12_RENDER_PASS_TIER_0;
        o->RaytracingTier = D3D12_RAYTRACING_TIER_1_1;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS6: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS6)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS6*>(pData);
        std::memset(o, 0, sizeof(*o));
        o->AdditionalShadingRatesSupported = TRUE;
        o->PerPrimitiveShadingRateSupportedWithViewportIndexing = TRUE;
        o->VariableShadingRateTier = D3D12_VARIABLE_SHADING_RATE_TIER_1;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS9: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS9)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS9*>(pData);
        std::memset(o, 0, sizeof(*o));
        o->AtomicInt64OnTypedResourceSupported = TRUE;
        o->AtomicInt64OnGroupSharedSupported = TRUE;
        o->DerivativesInMeshAndAmplificationShadersSupported = FALSE;
        o->WaveMMATier = D3D12_WAVE_MMA_TIER_NOT_SUPPORTED;
        return S_OK;
    }
    case D3D12_FEATURE_D3D12_OPTIONS11: {
        if (size < sizeof(D3D12_FEATURE_DATA_D3D12_OPTIONS11)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_D3D12_OPTIONS11*>(pData);
        o->AtomicInt64OnDescriptorHeapResourceSupported = TRUE;
        return S_OK;
    }
    case D3D12_FEATURE_ARCHITECTURE: {
        if (size < sizeof(D3D12_FEATURE_DATA_ARCHITECTURE)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_ARCHITECTURE*>(pData);
        o->TileBasedRenderer = TRUE;
        o->UMA = TRUE;
        o->CacheCoherentUMA = TRUE;
        return S_OK;
    }
    case D3D12_FEATURE_ARCHITECTURE1: {
        if (size < sizeof(D3D12_FEATURE_DATA_ARCHITECTURE1)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_ARCHITECTURE1*>(pData);
        o->TileBasedRenderer = TRUE;
        o->UMA = TRUE;
        o->CacheCoherentUMA = TRUE;
        o->IsolatedMMU = FALSE;
        return S_OK;
    }
    case D3D12_FEATURE_FEATURE_LEVELS: {
        if (size < sizeof(D3D12_FEATURE_DATA_FEATURE_LEVELS)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_FEATURE_LEVELS*>(pData);
        o->MaxSupportedFeatureLevel = D3D_FEATURE_LEVEL_12_1;
        return S_OK;
    }
    case D3D12_FEATURE_SHADER_MODEL: {
        if (size < sizeof(D3D12_FEATURE_DATA_SHADER_MODEL)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_SHADER_MODEL*>(pData);
        o->HighestShaderModel = std::min(o->HighestShaderModel, D3D_SHADER_MODEL_6_9);
        return S_OK;
    }
    case D3D12_FEATURE_ROOT_SIGNATURE: {
        if (size < sizeof(D3D12_FEATURE_DATA_ROOT_SIGNATURE)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_ROOT_SIGNATURE*>(pData);
        o->HighestVersion = std::min(o->HighestVersion, D3D_ROOT_SIGNATURE_VERSION_1_1);
        return S_OK;
    }
    case D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS: {
        if (size < sizeof(D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS*>(pData);
        const UINT sc = o->SampleCount;
        o->NumQualityLevels = (sc == 1 || sc == 2 || sc == 4 || sc == 8) ? 1u : 0u;
        return S_OK;
    }
    case D3D12_FEATURE_FORMAT_SUPPORT: {
        if (size < sizeof(D3D12_FEATURE_DATA_FORMAT_SUPPORT)) return E_INVALIDARG;
        auto* o = static_cast<D3D12_FEATURE_DATA_FORMAT_SUPPORT*>(pData);
        o->Support1 = D3D12_FORMAT_SUPPORT1_BUFFER |
                      D3D12_FORMAT_SUPPORT1_TEXTURE1D |
                      D3D12_FORMAT_SUPPORT1_TEXTURE2D |
                      D3D12_FORMAT_SUPPORT1_TEXTURE3D |
                      D3D12_FORMAT_SUPPORT1_TEXTURECUBE |
                      D3D12_FORMAT_SUPPORT1_SHADER_LOAD |
                      D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE |
                      D3D12_FORMAT_SUPPORT1_MIP |
                      D3D12_FORMAT_SUPPORT1_RENDER_TARGET |
                      D3D12_FORMAT_SUPPORT1_BLENDABLE |
                      D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL |
                      D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE |
                      D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET |
                      D3D12_FORMAT_SUPPORT1_MULTISAMPLE_LOAD |
                      D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
        o->Support2 = D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD |
                      D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
        return S_OK;
    }
    default:
        return S_OK;
    }
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateDescriptorHeap(
    const D3D12_DESCRIPTOR_HEAP_DESC* pDesc, REFIID riid, void** ppvHeap) {
    if (!pDesc || !ppvHeap) return E_INVALIDARG;
    auto* obj = new NacelleDescriptorHeap(this, *pDesc);
    HRESULT hr = obj->QueryInterface(riid, ppvHeap);
    obj->Release();
    return hr;
}

UINT STDMETHODCALLTYPE NacelleDevice::GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE) {
    return NacelleDescriptorHeap::kDescriptorStride;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateRootSignature(
    UINT, const void* pBlob, SIZE_T size, REFIID riid, void** ppvRootSignature) {
    if (!pBlob || !ppvRootSignature) return E_INVALIDARG;
    RootSignatureLayout layout;
    HRESULT hr = NacelleRootSignature::parseBlob(pBlob, size, layout);
    if (FAILED(hr)) return hr;
    auto* obj = new NacelleRootSignature(this, std::move(layout));
    hr = obj->QueryInterface(riid, ppvRootSignature);
    obj->Release();
    return hr;
}

void STDMETHODCALLTYPE NacelleDevice::CreateConstantBufferView(const D3D12_CONSTANT_BUFFER_VIEW_DESC* pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Dest) {
    DescriptorOps::createCBV(pDesc, Dest);
}

void STDMETHODCALLTYPE NacelleDevice::CreateShaderResourceView(ID3D12Resource* pRes, const D3D12_SHADER_RESOURCE_VIEW_DESC* pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Dest) {
    DescriptorOps::createSRV(pRes, pDesc, Dest);
}

void STDMETHODCALLTYPE NacelleDevice::CreateUnorderedAccessView(ID3D12Resource* pRes, ID3D12Resource* pCounter, const D3D12_UNORDERED_ACCESS_VIEW_DESC* pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Dest) {
    DescriptorOps::createUAV(pRes, pCounter, pDesc, Dest);
}

void STDMETHODCALLTYPE NacelleDevice::CreateRenderTargetView(ID3D12Resource* pRes, const D3D12_RENDER_TARGET_VIEW_DESC* pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Dest) {
    DescriptorOps::createRTV(pRes, pDesc, Dest);
}

void STDMETHODCALLTYPE NacelleDevice::CreateDepthStencilView(ID3D12Resource* pRes, const D3D12_DEPTH_STENCIL_VIEW_DESC* pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Dest) {
    DescriptorOps::createDSV(pRes, pDesc, Dest);
}

void STDMETHODCALLTYPE NacelleDevice::CreateSampler(const D3D12_SAMPLER_DESC* pDesc, D3D12_CPU_DESCRIPTOR_HANDLE Dest) {
    DescriptorOps::createSampler(pDesc, Dest);
}

void STDMETHODCALLTYPE NacelleDevice::CopyDescriptors(
    UINT NumDest, const D3D12_CPU_DESCRIPTOR_HANDLE* pDestStarts, const UINT* pDestSizes,
    UINT NumSrc, const D3D12_CPU_DESCRIPTOR_HANDLE* pSrcStarts, const UINT* pSrcSizes,
    D3D12_DESCRIPTOR_HEAP_TYPE) {
    DescriptorOps::copyDescriptors(NumDest, pDestStarts, pDestSizes, NumSrc, pSrcStarts, pSrcSizes);
}

void STDMETHODCALLTYPE NacelleDevice::CopyDescriptorsSimple(
    UINT Num, D3D12_CPU_DESCRIPTOR_HANDLE DestStart, D3D12_CPU_DESCRIPTOR_HANDLE SrcStart, D3D12_DESCRIPTOR_HEAP_TYPE) {
    DescriptorOps::copyDescriptors(1, &DestStart, &Num, 1, &SrcStart, &Num);
}

#if defined(_MSC_VER) || !defined(_WIN32)
D3D12_RESOURCE_ALLOCATION_INFO STDMETHODCALLTYPE NacelleDevice::GetResourceAllocationInfo(
    UINT, UINT numDescs, const D3D12_RESOURCE_DESC* pDescs) {
    D3D12_RESOURCE_ALLOCATION_INFO info{0, 64};
    for (UINT i = 0; i < numDescs && pDescs; ++i) {
        UINT64 total = 0;
        const UINT subs = (pDescs[i].Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                              ? 1
                              : std::max<UINT>(1, pDescs[i].MipLevels) * std::max<UINT>(1, pDescs[i].DepthOrArraySize);
        NacelleResource::computeResourceFootprints(pDescs[i], 0, subs, 0, nullptr, nullptr, nullptr, &total, false);
        info.SizeInBytes += (total + 63) & ~UINT64(63);
    }
    return info;
}

D3D12_HEAP_PROPERTIES STDMETHODCALLTYPE NacelleDevice::GetCustomHeapProperties(UINT, D3D12_HEAP_TYPE heapType) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heapType;
    hp.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK;
    hp.MemoryPoolPreference = D3D12_MEMORY_POOL_L0;
    hp.CreationNodeMask = 1;
    hp.VisibleNodeMask = 1;
    return hp;
}

LUID STDMETHODCALLTYPE NacelleDevice::GetAdapterLuid() {
    return LUID{0x4E434C31, 0x00000001};
}
#else
D3D12_RESOURCE_ALLOCATION_INFO* STDMETHODCALLTYPE NacelleDevice::GetResourceAllocationInfo(
    D3D12_RESOURCE_ALLOCATION_INFO* RetVal, UINT, UINT numDescs, const D3D12_RESOURCE_DESC* pDescs) {
    D3D12_RESOURCE_ALLOCATION_INFO info{0, 64};
    for (UINT i = 0; i < numDescs && pDescs; ++i) {
        UINT64 total = 0;
        const UINT subs = (pDescs[i].Dimension == D3D12_RESOURCE_DIMENSION_BUFFER)
                              ? 1
                              : std::max<UINT>(1, pDescs[i].MipLevels) * std::max<UINT>(1, pDescs[i].DepthOrArraySize);
        NacelleResource::computeResourceFootprints(pDescs[i], 0, subs, 0, nullptr, nullptr, nullptr, &total, false);
        info.SizeInBytes += (total + 63) & ~UINT64(63);
    }
    *RetVal = info;
    return RetVal;
}

D3D12_HEAP_PROPERTIES* STDMETHODCALLTYPE NacelleDevice::GetCustomHeapProperties(
    D3D12_HEAP_PROPERTIES* RetVal, UINT, D3D12_HEAP_TYPE heapType) {
    *RetVal = D3D12_HEAP_PROPERTIES{heapType, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0, 1, 1};
    return RetVal;
}

LUID* STDMETHODCALLTYPE NacelleDevice::GetAdapterLuid(LUID* RetVal) {
    *RetVal = LUID{0x4E434C31, 0x00000001};
    return RetVal;
}
#endif

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateCommittedResource(
    const D3D12_HEAP_PROPERTIES* pHeapProperties, D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC* pDesc, D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*, REFIID riidResource, void** ppvResource) {
    if (!pHeapProperties || !pDesc) return E_INVALIDARG;
    auto* res = new NacelleResource(this, *pHeapProperties, *pDesc, nullptr, 0);
    if (!ppvResource) { res->Release(); return S_OK; }
    HRESULT hr = res->QueryInterface(riidResource, ppvResource);
    res->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateHeap(
    const D3D12_HEAP_DESC* pDesc, REFIID riid, void** ppvHeap) {
    if (!pDesc) return E_INVALIDARG;
    auto* heap = new NacelleHeap(this, *pDesc);
    if (!ppvHeap) { heap->Release(); return S_OK; }
    HRESULT hr = heap->QueryInterface(riid, ppvHeap);
    heap->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreatePlacedResource(
    ID3D12Heap* pHeap, UINT64 HeapOffset, const D3D12_RESOURCE_DESC* pDesc,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID riid, void** ppvResource) {
    if (!pHeap || !pDesc) return E_INVALIDARG;
    auto* softHeap = static_cast<NacelleHeap*>(pHeap);
    auto* res = new NacelleResource(this, softHeap->GetDesc().Properties, *pDesc, softHeap, HeapOffset);
    if (!ppvResource) { res->Release(); return S_OK; }
    HRESULT hr = res->QueryInterface(riid, ppvResource);
    res->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateFence(
    UINT64 InitialValue, D3D12_FENCE_FLAGS Flags, REFIID riid, void** ppFence) {
    if (!ppFence) return E_INVALIDARG;
    auto* fence = new NacelleFence(this, InitialValue, Flags);
    HRESULT hr = fence->QueryInterface(riid, ppFence);
    fence->Release();
    return hr;
}

void STDMETHODCALLTYPE NacelleDevice::GetCopyableFootprints(
    const D3D12_RESOURCE_DESC* pResourceDesc, UINT FirstSubresource, UINT NumSubresources,
    UINT64 BaseOffset, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* pLayouts,
    UINT* pNumRows, UINT64* pRowSizeInBytes, UINT64* pTotalBytes) {
    if (!pResourceDesc) return;
    NacelleResource::computeResourceFootprints(
        *pResourceDesc, FirstSubresource, NumSubresources, BaseOffset,
        pLayouts, pNumRows, pRowSizeInBytes, pTotalBytes, true);
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateQueryHeap(
    const D3D12_QUERY_HEAP_DESC* pDesc, REFIID riid, void** ppvHeap) {
    if (!pDesc || !ppvHeap) return E_INVALIDARG;
    auto* obj = new NacelleQueryHeap(this, *pDesc);
    HRESULT hr = obj->QueryInterface(riid, ppvHeap);
    obj->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateCommandSignature(
    const D3D12_COMMAND_SIGNATURE_DESC* pDesc, ID3D12RootSignature*,
    REFIID riid, void** ppvCommandSignature) {
    if (!pDesc || !ppvCommandSignature) return E_INVALIDARG;
    auto* obj = new NacelleCommandSignature(this, *pDesc);
    HRESULT hr = obj->QueryInterface(riid, ppvCommandSignature);
    obj->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreatePipelineLibrary(
    const void* pLibraryBlob, SIZE_T BlobLength, REFIID riid, void** ppPipelineLibrary) {
    if (!ppPipelineLibrary) return E_INVALIDARG;
    auto* obj = new NacellePipelineLibrary(this, pLibraryBlob, BlobLength);
    HRESULT hr = obj->QueryInterface(riid, ppPipelineLibrary);
    obj->Release();
    return hr;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::SetEventOnMultipleFenceCompletion(
    ID3D12Fence* const* ppFences, const UINT64* pFenceValues, UINT NumFences,
    D3D12_MULTIPLE_FENCE_WAIT_FLAGS Flags, HANDLE hEvent) {
    if (!ppFences || !pFenceValues) return E_INVALIDARG;
    if (Flags == D3D12_MULTIPLE_FENCE_WAIT_FLAG_ANY) {
        // Poll briefly or wait on first; for software rendering wait on all or first satisfied
        for (UINT i = 0; i < NumFences; ++i) {
            if (ppFences[i]->GetCompletedValue() >= pFenceValues[i]) {
                signalOsEvent(hEvent);
                return S_OK;
            }
        }
    }
    for (UINT i = 0; i < NumFences; ++i) {
        static_cast<NacelleFence*>(ppFences[i])->waitForValue(pFenceValues[i]);
    }
    signalOsEvent(hEvent);
    return S_OK;
}

// --- NacelleSwapChain ---

NacelleSwapChain::NacelleSwapChain(ID3D12Device* device, ID3D12CommandQueue* queue,
                                   uint32_t width, uint32_t height, uint32_t bufferCount,
                                   DXGI_FORMAT format, PFN_NacellePresentCallback callback, void* userData)
    : device_(device), queue_(queue), width_(width), height_(height), format_(format),
      bufferCount_(std::max<uint32_t>(2, bufferCount)), callback_(callback), userData_(userData) {
    if (device_) device_->AddRef();
    if (queue_) queue_->AddRef();
    allocateBuffers();
}

NacelleSwapChain::~NacelleSwapChain() {
    releaseBuffers();
    if (queue_) queue_->Release();
    if (device_) device_->Release();
}

HRESULT STDMETHODCALLTYPE NacelleSwapChain::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(IUnknown)) ||
        InlineIsEqualGUID(riid, IID_INacelleSwapChain)) {
        *ppvObject = static_cast<INacelleSwapChain*>(this);
        AddRef();
        return S_OK;
    }
    *ppvObject = nullptr;
    return E_NOINTERFACE;
}

void NacelleSwapChain::allocateBuffers() {
    releaseBuffers();
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width_;
    desc.Height = height_;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format_;
    desc.SampleDesc = {1, 0};
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;

    for (uint32_t i = 0; i < bufferCount_; ++i) {
        buffers_.push_back(new NacelleResource(device_, hp, desc, nullptr, 0));
    }
    currentBackIndex_ = 0;
    frontIndex_ = 0;
}

void NacelleSwapChain::releaseBuffers() {
    for (auto* b : buffers_) {
        if (b) b->Release();
    }
    buffers_.clear();
}

HRESULT STDMETHODCALLTYPE NacelleSwapChain::GetBuffer(uint32_t Buffer, REFIID riid, void** ppSurface) {
    if (!ppSurface || Buffer >= buffers_.size()) return E_INVALIDARG;
    return buffers_[Buffer]->QueryInterface(riid, ppSurface);
}

HRESULT STDMETHODCALLTYPE NacelleSwapChain::Present(uint32_t, uint32_t) {
    frontIndex_ = currentBackIndex_;
    currentBackIndex_ = (currentBackIndex_ + 1) % bufferCount_;
    if (callback_ && frontIndex_ < buffers_.size()) {
        auto* front = buffers_[frontIndex_];
        const auto& sub = front->subresourceLayout(0);
        callback_(front->basePtr() + sub.offset, width_, height_, sub.row_pitch, format_, userData_);
    }
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleSwapChain::ResizeBuffers(
    uint32_t BufferCount, uint32_t Width, uint32_t Height, DXGI_FORMAT NewFormat, uint32_t) {
    if (BufferCount > 0) bufferCount_ = std::max<uint32_t>(2, BufferCount);
    if (Width > 0) width_ = Width;
    if (Height > 0) height_ = Height;
    if (NewFormat != DXGI_FORMAT_UNKNOWN) format_ = NewFormat;
    allocateBuffers();
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleSwapChain::MapFrontBufferPtr(
    const void** ppPixels, uint32_t* pWidth, uint32_t* pHeight, uint32_t* pRowPitch) {
    if (!ppPixels || frontIndex_ >= buffers_.size()) return E_INVALIDARG;
    auto* front = buffers_[frontIndex_];
    const auto& sub = front->subresourceLayout(0);
    *ppPixels = front->basePtr() + sub.offset;
    if (pWidth) *pWidth = width_;
    if (pHeight) *pHeight = height_;
    if (pRowPitch) *pRowPitch = sub.row_pitch;
    return S_OK;
}

void STDMETHODCALLTYPE NacelleDevice::GetRaytracingAccelerationStructurePrebuildInfo(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_INPUTS* pDesc,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO* pInfo) {
    getRaytracingAccelerationStructurePrebuildInfo(pDesc, pInfo);
}

void STDMETHODCALLTYPE NacelleSwapChain::SetPresentCallback(PFN_NacellePresentCallback Callback, void* UserData) {
    callback_ = Callback;
    userData_ = UserData;
}

extern "C" {

HRESULT NacelleCreateDevice(
    IUnknown* pAdapter, D3D_FEATURE_LEVEL, REFIID riid, void** ppDevice) {
    if (!ppDevice) return E_INVALIDARG;
    (void)pAdapter;
    auto* dev = new NacelleDevice();
    HRESULT hr = dev->QueryInterface(riid, ppDevice);
    dev->Release();
    return hr;
}

HRESULT NacelleCreateSwapChain(
    ID3D12CommandQueue* pQueue, uint32_t width, uint32_t height,
    uint32_t bufferCount, DXGI_FORMAT format,
    PFN_NacellePresentCallback callback, void* userData,
    INacelleSwapChain** ppSwapChain) {
    if (!pQueue || !ppSwapChain) return E_INVALIDARG;
    ID3D12Device* dev = nullptr;
    HRESULT hr = pQueue->GetDevice(IID_PPV_ARGS(&dev));
    if (FAILED(hr)) return hr;
    *ppSwapChain = new NacelleSwapChain(dev, pQueue, width, height, bufferCount, format, callback, userData);
    dev->Release();
    return S_OK;
}

} // extern "C"

static std::string wideToNarrowAscii(LPCWSTR ws) {
    std::string s;
    if (!ws) return s;
    for (const wchar_t* p = ws; *p; ++p) {
        s.push_back(static_cast<char>(*p & 0x7F));
    }
    return s;
}

HRESULT STDMETHODCALLTYPE NacelleDevice::CreateStateObject(
    const D3D12_STATE_OBJECT_DESC* pDesc,
    REFIID riid,
    void** ppStateObject) {
    if (!ppStateObject) return E_POINTER;
    *ppStateObject = nullptr;
    if (!pDesc || !pDesc->pSubobjects) return E_INVALIDARG;

    struct LibSubobj {
        D3D12_SHADER_BYTECODE bytecode{};
        std::vector<std::pair<std::string, std::string>> exports; // (exportName, importName)
    };
    struct HitGroupInfo {
        std::string exportName;
        std::string chName;
        std::string ahName;
        std::string isName;
    };

    std::vector<LibSubobj> libs;
    std::vector<HitGroupInfo> hitGroups;
    RootSignatureLayout globalRootLayout;
    uint32_t maxPayloadBytes = 64;
    uint32_t maxAttrBytes = 32;
    uint32_t maxRecursionDepth = 1;

    for (UINT i = 0; i < pDesc->NumSubobjects; ++i) {
        const auto& sub = pDesc->pSubobjects[i];
        if (!sub.pDesc) continue;
        switch (sub.Type) {
            case D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY: {
                const auto* ld = static_cast<const D3D12_DXIL_LIBRARY_DESC*>(sub.pDesc);
                LibSubobj ls;
                ls.bytecode = ld->DXILLibrary;
                for (UINT e = 0; e < ld->NumExports; ++e) {
                    const auto& ex = ld->pExports[e];
                    std::string expName = wideToNarrowAscii(ex.Name);
                    std::string impName = ex.ExportToRename ? wideToNarrowAscii(ex.ExportToRename) : expName;
                    ls.exports.emplace_back(std::move(expName), std::move(impName));
                }
                libs.push_back(std::move(ls));
                break;
            }
            case D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP: {
                const auto* hg = static_cast<const D3D12_HIT_GROUP_DESC*>(sub.pDesc);
                HitGroupInfo info;
                info.exportName = wideToNarrowAscii(hg->HitGroupExport);
                info.chName = wideToNarrowAscii(hg->ClosestHitShaderImport);
                info.ahName = wideToNarrowAscii(hg->AnyHitShaderImport);
                info.isName = wideToNarrowAscii(hg->IntersectionShaderImport);
                hitGroups.push_back(std::move(info));
                break;
            }
            case D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE: {
                const auto* grs = static_cast<const D3D12_GLOBAL_ROOT_SIGNATURE*>(sub.pDesc);
                if (grs->pGlobalRootSignature) {
                    auto* rs = static_cast<NacelleRootSignature*>(grs->pGlobalRootSignature);
                    globalRootLayout = rs->layout();
                }
                break;
            }
            case D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG: {
                const auto* sc = static_cast<const D3D12_RAYTRACING_SHADER_CONFIG*>(sub.pDesc);
                if (sc->MaxPayloadSizeInBytes > 0) maxPayloadBytes = sc->MaxPayloadSizeInBytes;
                if (sc->MaxAttributeSizeInBytes > 0) maxAttrBytes = sc->MaxAttributeSizeInBytes;
                break;
            }
            case D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG: {
                const auto* pc = static_cast<const D3D12_RAYTRACING_PIPELINE_CONFIG*>(sub.pDesc);
                if (pc->MaxTraceRecursionDepth > 0) maxRecursionDepth = pc->MaxTraceRecursionDepth;
                break;
            }
            default:
                break;
        }
    }

    CompiledDxrStateObject compiled;
    compiled.maxPayloadBytes = maxPayloadBytes;
    compiled.maxAttrBytes = maxAttrBytes;
    compiled.maxRecursionDepth = maxRecursionDepth;
    compiled.simdWidth = 4;

    std::unordered_map<std::string, uint32_t> shaderNameToIdx;
    for (const auto& lib : libs) {
        llvm::LLVMContext listCtx;
        std::vector<DxilLibraryExport> libExports;
        std::string err;
        if (!listDxilLibraryExports(lib.bytecode, listCtx, libExports, err)) {
            std::fprintf(stderr, "[Nacelle] CreateStateObject listDxilLibraryExports failed: %s\n", err.c_str());
            return E_FAIL;
        }

        auto compileOne = [&](const std::string& publicName, const std::string& importName, uint32_t kind) -> bool {
            DxrShaderFn fn = nullptr;
            std::string compErr;
            if (!compileDxrShader(lib.bytecode, importName, globalRootLayout, compiled.simdWidth, fn, compErr)) {
                std::fprintf(stderr, "[Nacelle] CreateStateObject compileDxrShader(%s) failed: %s\n",
                             importName.c_str(), compErr.c_str());
                return false;
            }
            const uint32_t idx = static_cast<uint32_t>(compiled.shaders.size());
            CompiledDxrShader cs;
            cs.exportName = publicName;
            cs.shaderKind = kind;
            cs.fn = fn;
            compiled.shaders.push_back(std::move(cs));
            shaderNameToIdx[publicName] = idx;

            NacelleShaderIdentifier id{};
            id.entryIndex = idx;
            id.isHitGroup = 0;
            compiled.identifiers[publicName] = id;
            return true;
        };

        if (lib.exports.empty()) {
            for (const auto& le : libExports) {
                if (!compileOne(le.exportName, le.exportName, le.shaderKind)) {
                    return E_FAIL;
                }
            }
        } else {
            for (const auto& [expName, impName] : lib.exports) {
                uint32_t kind = 0;
                for (const auto& le : libExports) {
                    if (le.exportName == impName) {
                        kind = le.shaderKind;
                        break;
                    }
                }
                if (!compileOne(expName, impName, kind)) {
                    return E_FAIL;
                }
            }
        }
    }

    for (const auto& hg : hitGroups) {
        CompiledDxrHitGroup chg;
        chg.exportName = hg.exportName;
        if (!hg.chName.empty()) {
            auto it = shaderNameToIdx.find(hg.chName);
            if (it != shaderNameToIdx.end()) chg.closestHitIdx = it->second;
        }
        if (!hg.ahName.empty()) {
            auto it = shaderNameToIdx.find(hg.ahName);
            if (it != shaderNameToIdx.end()) chg.anyHitIdx = it->second;
        }
        if (!hg.isName.empty()) {
            auto it = shaderNameToIdx.find(hg.isName);
            if (it != shaderNameToIdx.end()) chg.intersectionIdx = it->second;
        }
        const uint32_t hgIdx = static_cast<uint32_t>(compiled.hitGroups.size());
        compiled.hitGroups.push_back(std::move(chg));

        NacelleShaderIdentifier id{};
        id.entryIndex = hgIdx;
        id.isHitGroup = 1;
        compiled.identifiers[hg.exportName] = id;
    }

    auto* stateObj = new NacelleStateObject(this, std::move(compiled));
    HRESULT hr = stateObj->QueryInterface(riid, ppStateObject);
    stateObj->Release();
    return hr;
}

} // namespace nacelle
