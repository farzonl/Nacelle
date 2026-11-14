#include "command_list.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>

#include "../common/format_utils.h"
#include "../common/fp_env.h"
#include "../rasterizer/raytracing_bvh.h"

namespace nacelle {

// --- NacelleFence ---

NacelleFence::NacelleFence(ID3D12Device* device, UINT64 initialValue, D3D12_FENCE_FLAGS flags)
    : D3D12PageableBase(device), value_(initialValue), flags_(flags) {}

HRESULT STDMETHODCALLTYPE NacelleFence::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12Fence))) {
        *ppvObject = static_cast<ID3D12Fence*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

UINT64 STDMETHODCALLTYPE NacelleFence::GetCompletedValue() {
    return value_.load(std::memory_order_acquire);
}

HRESULT STDMETHODCALLTYPE NacelleFence::SetEventOnCompletion(UINT64 Value, HANDLE hEvent) {
    if (value_.load(std::memory_order_acquire) >= Value) {
        if (hEvent) signalOsEvent(hEvent);
        return S_OK;
    }
    if (!hEvent) {
        waitForValue(Value);
        return S_OK;
    }
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (value_.load(std::memory_order_acquire) >= Value) {
            signalOsEvent(hEvent);
            return S_OK;
        }
        waiters_.push_back({Value, hEvent});
    }
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleFence::Signal(UINT64 Value) {
    std::vector<HANDLE> toSignal;
    {
        std::lock_guard<std::mutex> lock(mu_);
        value_.store(Value, std::memory_order_release);
        for (auto it = waiters_.begin(); it != waiters_.end();) {
            if (it->targetValue <= Value) {
                if (it->hEvent) toSignal.push_back(it->hEvent);
                it = waiters_.erase(it);
            } else {
                ++it;
            }
        }
    }
    cv_.notify_all();
    for (HANDLE h : toSignal) signalOsEvent(h);
    return S_OK;
}

void NacelleFence::waitForValue(UINT64 value) {
    if (value_.load(std::memory_order_acquire) >= value) return;
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return value_.load(std::memory_order_acquire) >= value; });
}

// --- NacelleCommandAllocator ---

NacelleCommandAllocator::NacelleCommandAllocator(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type)
    : D3D12PageableBase(device), type_(type) {}

HRESULT STDMETHODCALLTYPE NacelleCommandAllocator::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12CommandAllocator))) {
        *ppvObject = static_cast<ID3D12CommandAllocator*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

HRESULT STDMETHODCALLTYPE NacelleCommandAllocator::Reset() { return S_OK; }

// --- NacelleCommandSignature ---

NacelleCommandSignature::NacelleCommandSignature(ID3D12Device* device, const D3D12_COMMAND_SIGNATURE_DESC& desc)
    : D3D12PageableBase(device), desc_(desc) {
    if (desc.NumArgumentDescs > 0 && desc.pArgumentDescs) {
        args_.assign(desc.pArgumentDescs, desc.pArgumentDescs + desc.NumArgumentDescs);
        desc_.pArgumentDescs = args_.data();
    }
}

HRESULT STDMETHODCALLTYPE NacelleCommandSignature::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12CommandSignature))) {
        *ppvObject = static_cast<ID3D12CommandSignature*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

// --- NacelleQueryHeap ---

NacelleQueryHeap::NacelleQueryHeap(ID3D12Device* device, const D3D12_QUERY_HEAP_DESC& d)
    : D3D12PageableBase(device), values(d.Count, 0), desc(d) {}

HRESULT STDMETHODCALLTYPE NacelleQueryHeap::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12QueryHeap))) {
        *ppvObject = static_cast<ID3D12QueryHeap*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

// --- NacelleCommandList ---

NacelleCommandList::NacelleCommandList(ID3D12Device* device,
                                 D3D12_COMMAND_LIST_TYPE type,
                                 NacelleCommandAllocator*,
                                 ID3D12PipelineState* initialState)
    : D3D12DeviceChildBase(device), type_(type) {
    if (initialState) {
        SetPipelineState(initialState);
    }
}

NacelleCommandList::~NacelleCommandList() {
    releaseRecordedRefs();
}

void NacelleCommandList::releaseRecordedRefs() {
    for (IUnknown* u : heldRefs_) {
        if (u) u->Release();
    }
    heldRefs_.clear();
}

HRESULT STDMETHODCALLTYPE NacelleCommandList::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList6))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList6*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList5))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList5*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList4))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList4*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList3))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList3*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList2))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList2*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList1))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList1*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12GraphicsCommandList))) {
        *ppvObject = static_cast<ID3D12GraphicsCommandList*>(this);
        AddRef();
        return S_OK;
    }
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12CommandList))) {
        *ppvObject = static_cast<ID3D12CommandList*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12DeviceChildBase::QueryInterface(riid, ppvObject);
}

HRESULT STDMETHODCALLTYPE NacelleCommandList::Close() {
    closed_ = true;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleCommandList::Reset(ID3D12CommandAllocator*, ID3D12PipelineState* pInitialState) {
    closed_ = false;
    commands_.clear();
    releaseRecordedRefs();
    if (pInitialState) {
        SetPipelineState(pInitialState);
    }
    return S_OK;
}

void STDMETHODCALLTYPE NacelleCommandList::ClearState(ID3D12PipelineState* pPipelineState) {
    commands_.push_back(CmdSetPipelineState{static_cast<NacellePipelineState*>(pPipelineState)});
}

void STDMETHODCALLTYPE NacelleCommandList::DrawInstanced(UINT vCount, UINT iCount, UINT startV, UINT startI) {
    commands_.push_back(CmdDrawInstanced{vCount, iCount, startV, startI});
}

void STDMETHODCALLTYPE NacelleCommandList::DrawIndexedInstanced(UINT idxCount, UINT iCount, UINT startIdx, INT baseV, UINT startI) {
    commands_.push_back(CmdDrawIndexedInstanced{idxCount, iCount, startIdx, baseV, startI});
}

void STDMETHODCALLTYPE NacelleCommandList::Dispatch(UINT x, UINT y, UINT z) {
    commands_.push_back(CmdDispatch{x, y, z});
}

void STDMETHODCALLTYPE NacelleCommandList::CopyBufferRegion(ID3D12Resource* pDst, UINT64 dstOff, ID3D12Resource* pSrc, UINT64 srcOff, UINT64 bytes) {
    if (pDst) { pDst->AddRef(); heldRefs_.push_back(pDst); }
    if (pSrc) { pSrc->AddRef(); heldRefs_.push_back(pSrc); }
    commands_.push_back(CmdCopyBufferRegion{static_cast<NacelleResource*>(pDst), dstOff, static_cast<NacelleResource*>(pSrc), srcOff, bytes});
}

void STDMETHODCALLTYPE NacelleCommandList::CopyTextureRegion(
    const D3D12_TEXTURE_COPY_LOCATION* pDst, UINT dstX, UINT dstY, UINT dstZ,
    const D3D12_TEXTURE_COPY_LOCATION* pSrc, const D3D12_BOX* pSrcBox) {
    if (!pDst || !pSrc) return;
    if (pDst->pResource) { pDst->pResource->AddRef(); heldRefs_.push_back(pDst->pResource); }
    if (pSrc->pResource) { pSrc->pResource->AddRef(); heldRefs_.push_back(pSrc->pResource); }
    CmdCopyTextureRegion cmd{};
    cmd.dst = *pDst;
    cmd.src = *pSrc;
    cmd.dstX = dstX;
    cmd.dstY = dstY;
    cmd.dstZ = dstZ;
    if (pSrcBox) {
        cmd.hasBox = true;
        cmd.srcBox = *pSrcBox;
    }
    commands_.push_back(cmd);
}

void STDMETHODCALLTYPE NacelleCommandList::CopyResource(ID3D12Resource* pDst, ID3D12Resource* pSrc) {
    if (pDst) { pDst->AddRef(); heldRefs_.push_back(pDst); }
    if (pSrc) { pSrc->AddRef(); heldRefs_.push_back(pSrc); }
    commands_.push_back(CmdCopyResource{static_cast<NacelleResource*>(pDst), static_cast<NacelleResource*>(pSrc)});
}

void STDMETHODCALLTYPE NacelleCommandList::ResolveSubresource(
    ID3D12Resource* pDst, UINT dstSub, ID3D12Resource* pSrc, UINT srcSub, DXGI_FORMAT fmt) {
    if (pDst) { pDst->AddRef(); heldRefs_.push_back(pDst); }
    if (pSrc) { pSrc->AddRef(); heldRefs_.push_back(pSrc); }
    commands_.push_back(CmdResolveSubresource{static_cast<NacelleResource*>(pDst), dstSub, static_cast<NacelleResource*>(pSrc), srcSub, fmt});
}

void STDMETHODCALLTYPE NacelleCommandList::IASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY topo) {
    commands_.push_back(CmdIASetTopology{topo});
}

void STDMETHODCALLTYPE NacelleCommandList::RSSetViewports(UINT num, const D3D12_VIEWPORT* pVps) {
    CmdRSSetViewports cmd;
    if (num > 0 && pVps) cmd.vps.assign(pVps, pVps + num);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::RSSetScissorRects(UINT num, const D3D12_RECT* pRects) {
    CmdRSSetScissors cmd;
    if (num > 0 && pRects) cmd.rects.assign(pRects, pRects + num);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::OMSetBlendFactor(const FLOAT f[4]) {
    CmdOMSetBlendFactor cmd{};
    if (f) std::memcpy(cmd.factor, f, 16);
    commands_.push_back(cmd);
}

void STDMETHODCALLTYPE NacelleCommandList::OMSetStencilRef(UINT ref) {
    commands_.push_back(CmdOMSetStencilRef{ref});
}

void STDMETHODCALLTYPE NacelleCommandList::RSSetShadingRate(
    D3D12_SHADING_RATE baseShadingRate,
    const D3D12_SHADING_RATE_COMBINER* combiners) {
    CmdRSSetShadingRate cmd;
    cmd.rate = baseShadingRate;
    if (combiners) {
        cmd.combiners[0] = combiners[0];
        cmd.combiners[1] = combiners[1];
    }
    commands_.push_back(cmd);
}

void STDMETHODCALLTYPE NacelleCommandList::SetPipelineState(ID3D12PipelineState* pPSO) {
    if (pPSO) { pPSO->AddRef(); heldRefs_.push_back(pPSO); }
    commands_.push_back(CmdSetPipelineState{static_cast<NacellePipelineState*>(pPSO)});
}

void STDMETHODCALLTYPE NacelleCommandList::ResourceBarrier(UINT, const D3D12_RESOURCE_BARRIER*) {
    // Unified host memory: barriers are implicit between sequentially executed commands on the queue.
    std::atomic_thread_fence(std::memory_order_seq_cst);
}

void STDMETHODCALLTYPE NacelleCommandList::ExecuteBundle(ID3D12GraphicsCommandList* pCmd) {
    if (pCmd) { pCmd->AddRef(); heldRefs_.push_back(pCmd); }
    commands_.push_back(CmdExecuteBundle{static_cast<NacelleCommandList*>(pCmd)});
}

void STDMETHODCALLTYPE NacelleCommandList::SetDescriptorHeaps(UINT num, ID3D12DescriptorHeap* const* ppHeaps) {
    CmdSetDescriptorHeaps cmd{};
    for (UINT i = 0; i < num; ++i) {
        if (!ppHeaps[i]) continue;
        ppHeaps[i]->AddRef();
        heldRefs_.push_back(ppHeaps[i]);
        auto* h = static_cast<NacelleDescriptorHeap*>(ppHeaps[i]);
        if (h->heapType() == D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) cmd.resHeap = h;
        else if (h->heapType() == D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER) cmd.sampHeap = h;
    }
    commands_.push_back(cmd);
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRootSignature(ID3D12RootSignature* pSig) {
    if (pSig) { pSig->AddRef(); heldRefs_.push_back(pSig); }
    commands_.push_back(CmdSetRootSignature{true, static_cast<NacelleRootSignature*>(pSig)});
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRootSignature(ID3D12RootSignature* pSig) {
    if (pSig) { pSig->AddRef(); heldRefs_.push_back(pSig); }
    commands_.push_back(CmdSetRootSignature{false, static_cast<NacelleRootSignature*>(pSig)});
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRootDescriptorTable(UINT idx, D3D12_GPU_DESCRIPTOR_HANDLE h) {
    commands_.push_back(CmdSetRootDescTable{true, idx, h});
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRootDescriptorTable(UINT idx, D3D12_GPU_DESCRIPTOR_HANDLE h) {
    commands_.push_back(CmdSetRootDescTable{false, idx, h});
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRoot32BitConstant(UINT idx, UINT val, UINT off) {
    commands_.push_back(CmdSetRoot32BitConsts{true, idx, off, {val}});
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRoot32BitConstant(UINT idx, UINT val, UINT off) {
    commands_.push_back(CmdSetRoot32BitConsts{false, idx, off, {val}});
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRoot32BitConstants(UINT idx, UINT num, const void* src, UINT off) {
    CmdSetRoot32BitConsts cmd{true, idx, off, {}};
    if (num > 0 && src) {
        cmd.data.resize(num);
        std::memcpy(cmd.data.data(), src, num * sizeof(UINT));
    }
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRoot32BitConstants(UINT idx, UINT num, const void* src, UINT off) {
    CmdSetRoot32BitConsts cmd{false, idx, off, {}};
    if (num > 0 && src) {
        cmd.data.resize(num);
        std::memcpy(cmd.data.data(), src, num * sizeof(UINT));
    }
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRootConstantBufferView(UINT idx, D3D12_GPU_VIRTUAL_ADDRESS addr) {
    commands_.push_back(CmdSetRootView{true, idx, addr});
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRootConstantBufferView(UINT idx, D3D12_GPU_VIRTUAL_ADDRESS addr) {
    commands_.push_back(CmdSetRootView{false, idx, addr});
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRootShaderResourceView(UINT idx, D3D12_GPU_VIRTUAL_ADDRESS addr) {
    commands_.push_back(CmdSetRootView{true, idx, addr});
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRootShaderResourceView(UINT idx, D3D12_GPU_VIRTUAL_ADDRESS addr) {
    commands_.push_back(CmdSetRootView{false, idx, addr});
}

void STDMETHODCALLTYPE NacelleCommandList::SetComputeRootUnorderedAccessView(UINT idx, D3D12_GPU_VIRTUAL_ADDRESS addr) {
    commands_.push_back(CmdSetRootView{true, idx, addr});
}

void STDMETHODCALLTYPE NacelleCommandList::SetGraphicsRootUnorderedAccessView(UINT idx, D3D12_GPU_VIRTUAL_ADDRESS addr) {
    commands_.push_back(CmdSetRootView{false, idx, addr});
}

void STDMETHODCALLTYPE NacelleCommandList::IASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW* pView) {
    CmdIASetIndexBuffer cmd{};
    if (pView) { cmd.view = *pView; cmd.valid = true; }
    commands_.push_back(cmd);
}

void STDMETHODCALLTYPE NacelleCommandList::IASetVertexBuffers(UINT startSlot, UINT numViews, const D3D12_VERTEX_BUFFER_VIEW* pViews) {
    CmdIASetVertexBuffers cmd{startSlot, {}};
    if (numViews > 0 && pViews) cmd.views.assign(pViews, pViews + numViews);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::OMSetRenderTargets(
    UINT numRTs, const D3D12_CPU_DESCRIPTOR_HANDLE* pRTVs, BOOL singleRange,
    const D3D12_CPU_DESCRIPTOR_HANDLE* pDSV) {
    CmdOMSetRenderTargets cmd{};
    for (UINT i = 0; i < numRTs && pRTVs; ++i) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = singleRange
            ? D3D12_CPU_DESCRIPTOR_HANDLE{pRTVs[0].ptr + i * NacelleDescriptorHeap::kDescriptorStride}
            : pRTVs[i];
        if (h.ptr) cmd.rtvs.push_back(*reinterpret_cast<const NacelleRenderTargetDescriptor*>(h.ptr));
    }
    if (pDSV && pDSV->ptr) {
        cmd.hasDSV = true;
        cmd.dsv = *reinterpret_cast<const NacelleRenderTargetDescriptor*>(pDSV->ptr);
    }
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::ClearDepthStencilView(
    D3D12_CPU_DESCRIPTOR_HANDLE dsv, D3D12_CLEAR_FLAGS flags, FLOAT depth, UINT8 stencil, UINT numRects, const D3D12_RECT* pRects) {
    if (!dsv.ptr) return;
    CmdClearDSV cmd{};
    cmd.dsv = *reinterpret_cast<const NacelleRenderTargetDescriptor*>(dsv.ptr);
    cmd.flags = flags;
    cmd.depth = depth;
    cmd.stencil = stencil;
    if (numRects > 0 && pRects) cmd.rects.assign(pRects, pRects + numRects);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::ClearRenderTargetView(
    D3D12_CPU_DESCRIPTOR_HANDLE rtv, const FLOAT rgba[4], UINT numRects, const D3D12_RECT* pRects) {
    if (!rtv.ptr || !rgba) return;
    CmdClearRTV cmd{};
    cmd.rtv = *reinterpret_cast<const NacelleRenderTargetDescriptor*>(rtv.ptr);
    std::memcpy(cmd.rgba, rgba, 16);
    if (numRects > 0 && pRects) cmd.rects.assign(pRects, pRects + numRects);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::ClearUnorderedAccessViewUint(
    D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE cpuH, ID3D12Resource* pRes,
    const UINT vals[4], UINT numRects, const D3D12_RECT* pRects) {
    if (!cpuH.ptr || !vals) return;
    if (pRes) { pRes->AddRef(); heldRefs_.push_back(pRes); }
    CmdClearUAV cmd{};
    cmd.desc = *reinterpret_cast<const NacelleDescriptor*>(cpuH.ptr);
    cmd.res = static_cast<NacelleResource*>(pRes);
    std::memcpy(cmd.uVal, vals, 16);
    cmd.isFloat = false;
    if (numRects > 0 && pRects) cmd.rects.assign(pRects, pRects + numRects);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::ClearUnorderedAccessViewFloat(
    D3D12_GPU_DESCRIPTOR_HANDLE, D3D12_CPU_DESCRIPTOR_HANDLE cpuH, ID3D12Resource* pRes,
    const FLOAT vals[4], UINT numRects, const D3D12_RECT* pRects) {
    if (!cpuH.ptr || !vals) return;
    if (pRes) { pRes->AddRef(); heldRefs_.push_back(pRes); }
    CmdClearUAV cmd{};
    cmd.desc = *reinterpret_cast<const NacelleDescriptor*>(cpuH.ptr);
    cmd.res = static_cast<NacelleResource*>(pRes);
    std::memcpy(cmd.fVal, vals, 16);
    cmd.isFloat = true;
    if (numRects > 0 && pRects) cmd.rects.assign(pRects, pRects + numRects);
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::EndQuery(ID3D12QueryHeap* pHeap, D3D12_QUERY_TYPE Type, UINT Index) {
    if (pHeap) { pHeap->AddRef(); heldRefs_.push_back(pHeap); }
    commands_.push_back(CmdEndQuery{static_cast<NacelleQueryHeap*>(pHeap), Type, Index});
}

void STDMETHODCALLTYPE NacelleCommandList::ResolveQueryData(
    ID3D12QueryHeap* pHeap, D3D12_QUERY_TYPE, UINT StartIndex, UINT NumQueries,
    ID3D12Resource* pDst, UINT64 Offset) {
    if (pHeap) { pHeap->AddRef(); heldRefs_.push_back(pHeap); }
    if (pDst) { pDst->AddRef(); heldRefs_.push_back(pDst); }
    commands_.push_back(CmdResolveQueryData{static_cast<NacelleQueryHeap*>(pHeap), StartIndex, NumQueries, static_cast<NacelleResource*>(pDst), Offset});
}

void STDMETHODCALLTYPE NacelleCommandList::ExecuteIndirect(
    ID3D12CommandSignature* pSig, UINT maxCount, ID3D12Resource* pArgBuf, UINT64 argOff,
    ID3D12Resource* pCntBuf, UINT64 cntOff) {
    if (pSig) { pSig->AddRef(); heldRefs_.push_back(pSig); }
    if (pArgBuf) { pArgBuf->AddRef(); heldRefs_.push_back(pArgBuf); }
    if (pCntBuf) { pCntBuf->AddRef(); heldRefs_.push_back(pCntBuf); }
    commands_.push_back(CmdExecuteIndirect{
        static_cast<NacelleCommandSignature*>(pSig), maxCount,
        static_cast<NacelleResource*>(pArgBuf), argOff,
        static_cast<NacelleResource*>(pCntBuf), cntOff});
}

void STDMETHODCALLTYPE NacelleCommandList::BuildRaytracingAccelerationStructure(
    const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC* pDesc,
    UINT /*NumPostbuildInfoDescs*/,
    const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC* /*pPostbuildInfoDescs*/) {
    if (closed_ || !pDesc) return;
    CmdBuildRaytracingAS cmd{};
    cmd.desc = *pDesc;
    if (pDesc->Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
        cmd.geomDescs.reserve(pDesc->Inputs.NumDescs);
        for (UINT i = 0; i < pDesc->Inputs.NumDescs; ++i) {
            if (pDesc->Inputs.DescsLayout == D3D12_ELEMENTS_LAYOUT_ARRAY_OF_POINTERS) {
                if (pDesc->Inputs.ppGeometryDescs && pDesc->Inputs.ppGeometryDescs[i]) {
                    cmd.geomDescs.push_back(*pDesc->Inputs.ppGeometryDescs[i]);
                }
            } else if (pDesc->Inputs.pGeometryDescs) {
                cmd.geomDescs.push_back(pDesc->Inputs.pGeometryDescs[i]);
            }
        }
        cmd.desc.Inputs.DescsLayout = D3D12_ELEMENTS_LAYOUT_ARRAY;
        cmd.desc.Inputs.pGeometryDescs = nullptr;
    }
    commands_.push_back(std::move(cmd));
}

void STDMETHODCALLTYPE NacelleCommandList::CopyRaytracingAccelerationStructure(
    D3D12_GPU_VIRTUAL_ADDRESS DestAccelerationStructureData,
    D3D12_GPU_VIRTUAL_ADDRESS SourceAccelerationStructureData,
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE Mode) {
    if (closed_) return;
    commands_.push_back(CmdCopyRaytracingAS{
        DestAccelerationStructureData, SourceAccelerationStructureData, Mode});
}

namespace {

void applyRootSignatureStaticSamplers(const NacelleRootSignature* sig, DriverRootContext& ctx) {
    if (!sig) return;
    const auto& samplers = sig->layout().staticSamplers;
    const size_t n = std::min<size_t>(samplers.size(), DriverRootContext::kMaxStaticSamplers);
    for (size_t i = 0; i < n; ++i) {
        DescriptorOps::populateSamplerDesc(samplers[i].desc, ctx.static_samplers[i]);
    }
}

void execCopyTextureRegion(const CmdCopyTextureRegion& c) {
    auto* dstRes = static_cast<NacelleResource*>(c.dst.pResource);
    auto* srcRes = static_cast<NacelleResource*>(c.src.pResource);
    if (!dstRes || !srcRes) return;

    if (c.src.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT &&
        c.dst.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX) {
        const auto& fp = c.src.PlacedFootprint;
        const auto& dstSub = dstRes->subresourceLayout(c.dst.SubresourceIndex);
        const FormatInfo fi = getFormatInfo(dstRes->resourceDesc().Format);
        UINT x0 = c.hasBox ? c.srcBox.left : 0;
        UINT y0 = c.hasBox ? c.srcBox.top : 0;
        UINT z0 = c.hasBox ? c.srcBox.front : 0;
        UINT w = c.hasBox ? (c.srcBox.right - c.srcBox.left) : fp.Footprint.Width;
        UINT h = c.hasBox ? (c.srcBox.bottom - c.srcBox.top) : fp.Footprint.Height;
        UINT d = c.hasBox ? (c.srcBox.back - c.srcBox.front) : fp.Footprint.Depth;

        UINT bxSrc = x0 / fi.blockWidth, bySrc = y0 / fi.blockHeight;
        UINT bxDst = c.dstX / fi.blockWidth, byDst = c.dstY / fi.blockHeight;
        UINT bw = std::max<UINT>(1, (w + fi.blockWidth - 1) / fi.blockWidth);
        UINT bh = std::max<UINT>(1, (h + fi.blockHeight - 1) / fi.blockHeight);
        UINT copyBytes = bw * fi.bytesPerBlock;
        UINT64 srcSlicePitch = UINT64(fp.Footprint.RowPitch) * ((fp.Footprint.Height + fi.blockHeight - 1) / fi.blockHeight);

        for (UINT z = 0; z < d; ++z) {
            const uint8_t* srcSlice = srcRes->basePtr() + fp.Offset + (z0 + z) * srcSlicePitch;
            uint8_t* dstSlice = dstRes->basePtr() + dstSub.offset + (c.dstZ + z) * dstSub.slice_pitch;
            for (UINT by = 0; by < bh; ++by) {
                std::memcpy(
                    dstSlice + (byDst + by) * dstSub.row_pitch + bxDst * fi.bytesPerBlock,
                    srcSlice + (bySrc + by) * fp.Footprint.RowPitch + bxSrc * fi.bytesPerBlock,
                    copyBytes);
            }
        }
        dstRes->invalidateDecodedCache();
        if (dstRes->textureMetadata()->is_compressed && dstRes->textureMetadata()->decoded_rgba32f != nullptr) {
            dstRes->ensureDecodedBC();
        }
    } else if (c.src.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX &&
               c.dst.Type == D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT) {
        const auto& fp = c.dst.PlacedFootprint;
        const auto& srcSub = srcRes->subresourceLayout(c.src.SubresourceIndex);
        const FormatInfo fi = getFormatInfo(srcRes->resourceDesc().Format);
        UINT x0 = c.hasBox ? c.srcBox.left : 0;
        UINT y0 = c.hasBox ? c.srcBox.top : 0;
        UINT z0 = c.hasBox ? c.srcBox.front : 0;
        UINT w = c.hasBox ? (c.srcBox.right - c.srcBox.left) : srcSub.width;
        UINT h = c.hasBox ? (c.srcBox.bottom - c.srcBox.top) : srcSub.height;
        UINT d = c.hasBox ? (c.srcBox.back - c.srcBox.front) : srcSub.depth;

        UINT bxSrc = x0 / fi.blockWidth, bySrc = y0 / fi.blockHeight;
        UINT bxDst = c.dstX / fi.blockWidth, byDst = c.dstY / fi.blockHeight;
        UINT bw = std::max<UINT>(1, (w + fi.blockWidth - 1) / fi.blockWidth);
        UINT bh = std::max<UINT>(1, (h + fi.blockHeight - 1) / fi.blockHeight);
        UINT copyBytes = bw * fi.bytesPerBlock;
        UINT64 dstSlicePitch = UINT64(fp.Footprint.RowPitch) * ((fp.Footprint.Height + fi.blockHeight - 1) / fi.blockHeight);

        for (UINT z = 0; z < d; ++z) {
            const uint8_t* srcSlice = srcRes->basePtr() + srcSub.offset + (z0 + z) * srcSub.slice_pitch;
            uint8_t* dstSlice = dstRes->basePtr() + fp.Offset + (c.dstZ + z) * dstSlicePitch;
            for (UINT by = 0; by < bh; ++by) {
                std::memcpy(
                    dstSlice + (byDst + by) * fp.Footprint.RowPitch + bxDst * fi.bytesPerBlock,
                    srcSlice + (bySrc + by) * srcSub.row_pitch + bxSrc * fi.bytesPerBlock,
                    copyBytes);
            }
        }
        dstRes->invalidateDecodedCache();
        if (dstRes->textureMetadata()->is_compressed && dstRes->textureMetadata()->decoded_rgba32f != nullptr) {
            dstRes->ensureDecodedBC();
        }
    } else if (c.src.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX &&
               c.dst.Type == D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX) {
        const auto& srcSub = srcRes->subresourceLayout(c.src.SubresourceIndex);
        const auto& dstSub = dstRes->subresourceLayout(c.dst.SubresourceIndex);
        const FormatInfo fi = getFormatInfo(srcRes->resourceDesc().Format);
        UINT x0 = c.hasBox ? c.srcBox.left : 0;
        UINT y0 = c.hasBox ? c.srcBox.top : 0;
        UINT z0 = c.hasBox ? c.srcBox.front : 0;
        UINT w = c.hasBox ? (c.srcBox.right - c.srcBox.left) : std::min(srcSub.width, dstSub.width - c.dstX);
        UINT h = c.hasBox ? (c.srcBox.bottom - c.srcBox.top) : std::min(srcSub.height, dstSub.height - c.dstY);
        UINT d = c.hasBox ? (c.srcBox.back - c.srcBox.front) : std::min(srcSub.depth, dstSub.depth - c.dstZ);

        UINT bxSrc = x0 / fi.blockWidth, bySrc = y0 / fi.blockHeight;
        UINT bxDst = c.dstX / fi.blockWidth, byDst = c.dstY / fi.blockHeight;
        UINT bw = std::max<UINT>(1, (w + fi.blockWidth - 1) / fi.blockWidth);
        UINT bh = std::max<UINT>(1, (h + fi.blockHeight - 1) / fi.blockHeight);
        UINT copyBytes = bw * fi.bytesPerBlock;

        for (UINT z = 0; z < d; ++z) {
            const uint8_t* srcSlice = srcRes->basePtr() + srcSub.offset + (z0 + z) * srcSub.slice_pitch;
            uint8_t* dstSlice = dstRes->basePtr() + dstSub.offset + (c.dstZ + z) * dstSub.slice_pitch;
            for (UINT by = 0; by < bh; ++by) {
                std::memcpy(
                    dstSlice + (byDst + by) * dstSub.row_pitch + bxDst * fi.bytesPerBlock,
                    srcSlice + (bySrc + by) * srcSub.row_pitch + bxSrc * fi.bytesPerBlock,
                    copyBytes);
            }
        }
        dstRes->invalidateDecodedCache();
        if (dstRes->textureMetadata()->is_compressed && dstRes->textureMetadata()->decoded_rgba32f != nullptr) {
            dstRes->ensureDecodedBC();
        }
    }
}

void execClearRTV(const CmdClearRTV& c) {
    if (!c.rtv.resource) return;
    const auto& sub = c.rtv.resource->subresourceLayout(c.rtv.mip_slice);
    const FormatInfo fi = getFormatInfo(c.rtv.format);
    uint8_t encoded[16]{};
    packPixelRGBA32F(c.rtv.format, c.rgba, encoded);
    const UINT sampleCount = std::max<UINT>(1, c.rtv.resource->resourceDesc().SampleDesc.Count);

    auto clearRect = [&](uint8_t* base, LONG x0, LONG y0, LONG x1, LONG y1) {
        x0 = std::max<LONG>(0, x0);
        y0 = std::max<LONG>(0, y0);
        x1 = std::min<LONG>(sub.width, x1);
        y1 = std::min<LONG>(sub.height, y1);
        for (LONG y = y0; y < y1; ++y) {
            uint8_t* row = base + y * sub.row_pitch;
            for (LONG x = x0; x < x1; ++x) {
                std::memcpy(row + x * fi.bytesPerBlock, encoded, fi.bytesPerBlock);
            }
        }
    };

    for (UINT s = 0; s < sampleCount; ++s) {
        uint8_t* base = c.rtv.resource->basePtr() + sub.offset + static_cast<size_t>(s) * sub.slice_pitch;
        if (c.rects.empty()) {
            clearRect(base, 0, 0, sub.width, sub.height);
        } else {
            for (const auto& r : c.rects) clearRect(base, r.left, r.top, r.right, r.bottom);
        }
    }
}

void execClearDSV(const CmdClearDSV& c) {
    if (!c.dsv.resource) return;
    const auto& sub = c.dsv.resource->subresourceLayout(c.dsv.mip_slice);
    const FormatInfo fi = getFormatInfo(c.dsv.format);
    const UINT sampleCount = std::max<UINT>(1, c.dsv.resource->resourceDesc().SampleDesc.Count);

    auto clearRect = [&](uint8_t* base, LONG x0, LONG y0, LONG x1, LONG y1) {
        x0 = std::max<LONG>(0, x0);
        y0 = std::max<LONG>(0, y0);
        x1 = std::min<LONG>(sub.width, x1);
        y1 = std::min<LONG>(sub.height, y1);
        for (LONG y = y0; y < y1; ++y) {
            uint8_t* row = base + y * sub.row_pitch;
            for (LONG x = x0; x < x1; ++x) {
                uint8_t* px = row + x * fi.bytesPerBlock;
                if (c.dsv.format == DXGI_FORMAT_D32_FLOAT || c.dsv.format == DXGI_FORMAT_R32_TYPELESS) {
                    if (c.flags & D3D12_CLEAR_FLAG_DEPTH) std::memcpy(px, &c.depth, 4);
                } else if (c.dsv.format == DXGI_FORMAT_D16_UNORM) {
                    if (c.flags & D3D12_CLEAR_FLAG_DEPTH) {
                        uint16_t d16 = static_cast<uint16_t>(std::clamp(c.depth, 0.0f, 1.0f) * 65535.0f + 0.5f);
                        std::memcpy(px, &d16, 2);
                    }
                } else if (c.dsv.format == DXGI_FORMAT_D24_UNORM_S8_UINT) {
                    uint32_t cur = 0;
                    std::memcpy(&cur, px, 4);
                    if (c.flags & D3D12_CLEAR_FLAG_DEPTH) {
                        uint32_t d24 = static_cast<uint32_t>(std::clamp(c.depth, 0.0f, 1.0f) * 16777215.0f + 0.5f);
                        cur = (cur & 0xFF000000u) | (d24 & 0x00FFFFFFu);
                    }
                    if (c.flags & D3D12_CLEAR_FLAG_STENCIL) {
                        cur = (cur & 0x00FFFFFFu) | (uint32_t(c.stencil) << 24);
                    }
                    std::memcpy(px, &cur, 4);
                } else if (c.dsv.format == DXGI_FORMAT_D32_FLOAT_S8X24_UINT) {
                    if (c.flags & D3D12_CLEAR_FLAG_DEPTH) std::memcpy(px, &c.depth, 4);
                    if (c.flags & D3D12_CLEAR_FLAG_STENCIL) px[4] = c.stencil;
                }
            }
        }
    };

    for (UINT s = 0; s < sampleCount; ++s) {
        uint8_t* base = c.dsv.resource->basePtr() + sub.offset + static_cast<size_t>(s) * sub.slice_pitch;
        if (c.rects.empty()) clearRect(base, 0, 0, sub.width, sub.height);
        else for (const auto& r : c.rects) clearRect(base, r.left, r.top, r.right, r.bottom);
    }
}

void execDispatch(BS::thread_pool<>& pool, const ExecutionState& state, UINT gx, UINT gy, UINT gz) {
    if (!state.pso || !state.pso->isCompute() || !state.pso->computePSO()) return;
    const auto& cpso = state.pso->computePSO();
    CompiledComputeFn fn = cpso->fn.load(std::memory_order_acquire);
    if (!fn || gx == 0 || gy == 0 || gz == 0) return;

    const uint64_t totalGroups = uint64_t(gx) * gy * gz;
    const uint32_t scratchSize = std::max<uint32_t>(64, cpso->groupSharedBytes);
    const DriverRootContext ctxSnapshot = state.computeCtx;

    auto runRange = [&](uint64_t begin, uint64_t end) {
        ScopedFlushDenormals flushDenorms;
        thread_local std::vector<uint8_t> tlScratch;
        if (tlScratch.size() < scratchSize) tlScratch.resize(scratchSize);
        for (uint64_t idx = begin; idx < end; ++idx) {
            uint32_t x = static_cast<uint32_t>(idx % gx);
            uint32_t yz = static_cast<uint32_t>(idx / gx);
            uint32_t y = yz % gy;
            uint32_t z = yz / gy;
            if (cpso->groupSharedBytes > 0) {
                std::memset(tlScratch.data(), 0, cpso->groupSharedBytes);
            }
            fn(&ctxSnapshot, tlScratch.data(), x, y, z, gx, gy, gz);
        }
    };

    if (totalGroups <= 1) {
        runRange(0, totalGroups);
    } else {
        pool.submit_blocks<uint64_t>(0, totalGroups, runRange).wait();
    }
}

} // namespace

void NacelleCommandList::execute(BS::thread_pool<>& pool, ExecutionState& state) const {
    for (const auto& cmd : commands_) {
        std::visit([&](const auto& c) {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, CmdSetPipelineState>) {
                state.pso = c.pso;
                state.gfx.pso = c.pso;
            } else if constexpr (std::is_same_v<T, CmdSetRootSignature>) {
                if (c.compute) {
                    state.computeRootSig = c.sig;
                    applyRootSignatureStaticSamplers(c.sig, state.computeCtx);
                } else {
                    state.graphicsRootSig = c.sig;
                    applyRootSignatureStaticSamplers(c.sig, state.gfx.rootCtx);
                }
            } else if constexpr (std::is_same_v<T, CmdSetDescriptorHeaps>) {
                if (c.resHeap) {
                    state.computeCtx.resource_heap_base = c.resHeap->data();
                    state.computeCtx.resource_heap_count = c.resHeap->count();
                    state.gfx.rootCtx.resource_heap_base = c.resHeap->data();
                    state.gfx.rootCtx.resource_heap_count = c.resHeap->count();
                }
                if (c.sampHeap) {
                    state.computeCtx.sampler_heap_base = c.sampHeap->data();
                    state.computeCtx.sampler_heap_count = c.sampHeap->count();
                    state.gfx.rootCtx.sampler_heap_base = c.sampHeap->data();
                    state.gfx.rootCtx.sampler_heap_count = c.sampHeap->count();
                }
            } else if constexpr (std::is_same_v<T, CmdSetRootDescTable>) {
                DriverRootContext& ctx = c.compute ? state.computeCtx : state.gfx.rootCtx;
                if (c.idx < DriverRootContext::kMaxRootSlots) {
                    ctx.root_slots[c.idx] = reinterpret_cast<const void*>(c.handle.ptr);
                }
            } else if constexpr (std::is_same_v<T, CmdSetRoot32BitConsts>) {
                DriverRootContext& ctx = c.compute ? state.computeCtx : state.gfx.rootCtx;
                const NacelleRootSignature* sig = c.compute ? state.computeRootSig : state.graphicsRootSig;
                if (c.idx < DriverRootContext::kMaxRootSlots && sig && c.idx < sig->layout().params.size()) {
                    uint32_t base = sig->layout().params[c.idx].constDwordOffset;
                    uint32_t dstStart = base + c.destOffset;
                    if (dstStart + c.data.size() <= std::size(ctx.inline_constants)) {
                        std::memcpy(&ctx.inline_constants[dstStart], c.data.data(), c.data.size() * sizeof(UINT));
                    }
                    ctx.root_slots[c.idx] = &ctx.inline_constants[base];
                }
            } else if constexpr (std::is_same_v<T, CmdSetRootView>) {
                DriverRootContext& ctx = c.compute ? state.computeCtx : state.gfx.rootCtx;
                if (c.idx < DriverRootContext::kMaxRootSlots) {
                    auto& sd = ctx.inline_descriptors[c.idx];
                    sd = NacelleDescriptor{};
                    sd.base_ptr = reinterpret_cast<void*>(c.addr);
                    sd.byte_size = UINT64_MAX;
                    sd.width = UINT32_MAX;
                    sd.elem_stride = 4;
                    ctx.root_slots[c.idx] = &sd;
                }
            } else if constexpr (std::is_same_v<T, CmdDispatch>) {
                execDispatch(pool, state, c.x, c.y, c.z);
            } else if constexpr (std::is_same_v<T, CmdDrawInstanced>) {
                executeRasterDrawCall(pool, state.gfx, DrawCallParams{false, c.vCount, c.iCount, c.startV, 0, c.startI});
            } else if constexpr (std::is_same_v<T, CmdDrawIndexedInstanced>) {
                executeRasterDrawCall(pool, state.gfx, DrawCallParams{true, c.idxCount, c.iCount, c.startIdx, c.baseV, c.startI});
            } else if constexpr (std::is_same_v<T, CmdCopyBufferRegion>) {
                if (c.dst && c.src && c.bytes > 0) {
                    std::memcpy(c.dst->basePtr() + c.dstOff, c.src->basePtr() + c.srcOff, static_cast<size_t>(c.bytes));
                    c.dst->invalidateDecodedCache();
                }
            } else if constexpr (std::is_same_v<T, CmdCopyTextureRegion>) {
                execCopyTextureRegion(c);
            } else if constexpr (std::is_same_v<T, CmdCopyResource>) {
                if (c.dst && c.src) {
                    size_t n = static_cast<size_t>(std::min(c.dst->totalBytes(), c.src->totalBytes()));
                    std::memcpy(c.dst->basePtr(), c.src->basePtr(), n);
                    c.dst->invalidateDecodedCache();
                }
            } else if constexpr (std::is_same_v<T, CmdResolveSubresource>) {
                if (c.dst && c.src) {
                    const auto& dSub = c.dst->subresourceLayout(c.dstSub);
                    const auto& sSub = c.src->subresourceLayout(c.srcSub);
                    const UINT sampleCount = std::max<UINT>(1, c.src->resourceDesc().SampleDesc.Count);
                    const DXGI_FORMAT fmt = (c.fmt != DXGI_FORMAT_UNKNOWN) ? c.fmt : c.dst->resourceDesc().Format;
                    if (sampleCount <= 1) {
                        size_t n = std::min<size_t>(dSub.slice_pitch, sSub.slice_pitch);
                        std::memcpy(c.dst->basePtr() + dSub.offset, c.src->basePtr() + sSub.offset, n);
                    } else {
                        const FormatInfo fi = getFormatInfo(fmt);
                        const uint32_t w = std::min(dSub.width, sSub.width);
                        const uint32_t h = std::min(dSub.height, sSub.height);
                        const float invSamples = 1.0f / static_cast<float>(sampleCount);
                        for (uint32_t y = 0; y < h; ++y) {
                            uint8_t* dstRow = c.dst->basePtr() + dSub.offset + static_cast<size_t>(y) * dSub.row_pitch;
                            for (uint32_t x = 0; x < w; ++x) {
                                float sum[4]{0.0f, 0.0f, 0.0f, 0.0f};
                                for (UINT s = 0; s < sampleCount; ++s) {
                                    const uint8_t* srcPx = c.src->basePtr() + sSub.offset +
                                                           static_cast<size_t>(s) * sSub.slice_pitch +
                                                           static_cast<size_t>(y) * sSub.row_pitch +
                                                           static_cast<size_t>(x) * fi.bytesPerBlock;
                                    float rgba[4]{};
                                    unpackPixelRGBA32F(fmt, srcPx, rgba);
                                    for (int ch = 0; ch < 4; ++ch) sum[ch] += rgba[ch];
                                }
                                for (int ch = 0; ch < 4; ++ch) sum[ch] *= invSamples;
                                packPixelRGBA32F(fmt, sum, dstRow + static_cast<size_t>(x) * fi.bytesPerBlock);
                            }
                        }
                    }
                    c.dst->invalidateDecodedCache();
                }
            } else if constexpr (std::is_same_v<T, CmdIASetTopology>) {
                state.gfx.topology = c.topo;
            } else if constexpr (std::is_same_v<T, CmdIASetIndexBuffer>) {
                state.gfx.indexBuffer = c.view;
            } else if constexpr (std::is_same_v<T, CmdIASetVertexBuffers>) {
                for (size_t i = 0; i < c.views.size() && (c.startSlot + i) < 16; ++i) {
                    state.gfx.vertexBuffers[c.startSlot + i] = c.views[i];
                    state.gfx.rootCtx.vertex_buffers[c.startSlot + i] = c.views[i].BufferLocation;
                    state.gfx.rootCtx.vertex_strides[c.startSlot + i] = c.views[i].StrideInBytes;
                    state.gfx.rootCtx.vertex_sizes[c.startSlot + i] = c.views[i].SizeInBytes;
                }
            } else if constexpr (std::is_same_v<T, CmdRSSetViewports>) {
                state.gfx.numViewports = static_cast<uint32_t>(std::min<size_t>(c.vps.size(), 16));
                for (uint32_t i = 0; i < state.gfx.numViewports; ++i) state.gfx.viewports[i] = c.vps[i];
            } else if constexpr (std::is_same_v<T, CmdRSSetScissors>) {
                state.gfx.numScissors = static_cast<uint32_t>(std::min<size_t>(c.rects.size(), 16));
                for (uint32_t i = 0; i < state.gfx.numScissors; ++i) state.gfx.scissors[i] = c.rects[i];
            } else if constexpr (std::is_same_v<T, CmdOMSetBlendFactor>) {
                std::memcpy(state.gfx.rootCtx.blend_factor, c.factor, 16);
            } else if constexpr (std::is_same_v<T, CmdOMSetStencilRef>) {
                state.gfx.rootCtx.stencil_ref = c.ref;
            } else if constexpr (std::is_same_v<T, CmdRSSetShadingRate>) {
                state.gfx.shadingRate = c.rate;
                state.gfx.shadingRateCombiners[0] = c.combiners[0];
                state.gfx.shadingRateCombiners[1] = c.combiners[1];
            } else if constexpr (std::is_same_v<T, CmdOMSetRenderTargets>) {
                state.gfx.numRTVs = static_cast<uint32_t>(std::min<size_t>(c.rtvs.size(), 8));
                for (uint32_t i = 0; i < state.gfx.numRTVs; ++i) state.gfx.rtvs[i] = c.rtvs[i];
                state.gfx.hasDSV = c.hasDSV;
                state.gfx.dsv = c.dsv;
            } else if constexpr (std::is_same_v<T, CmdClearRTV>) {
                execClearRTV(c);
            } else if constexpr (std::is_same_v<T, CmdClearDSV>) {
                execClearDSV(c);
            } else if constexpr (std::is_same_v<T, CmdClearUAV>) {
                if (c.desc.base_ptr && c.desc.byte_size > 0) {
                    const uint32_t* u32 = c.isFloat ? reinterpret_cast<const uint32_t*>(c.fVal) : c.uVal;
                    auto* dst = static_cast<uint32_t*>(c.desc.base_ptr);
                    size_t words = c.desc.byte_size / 4;
                    for (size_t i = 0; i < words; ++i) dst[i] = u32[i & 3];
                }
            } else if constexpr (std::is_same_v<T, CmdExecuteBundle>) {
                if (c.bundle) c.bundle->execute(pool, state);
            } else if constexpr (std::is_same_v<T, CmdEndQuery>) {
                if (c.heap && c.index < c.heap->values.size()) {
                    c.heap->values[c.index] = static_cast<uint64_t>(
                        std::chrono::steady_clock::now().time_since_epoch().count());
                }
            } else if constexpr (std::is_same_v<T, CmdResolveQueryData>) {
                if (c.heap && c.dst && c.start + c.count <= c.heap->values.size()) {
                    std::memcpy(c.dst->basePtr() + c.offset,
                                &c.heap->values[c.start],
                                c.count * sizeof(uint64_t));
                }
            } else if constexpr (std::is_same_v<T, CmdBuildRaytracingAS>) {
                D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC localDesc = c.desc;
                if (localDesc.Inputs.Type == D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL) {
                    localDesc.Inputs.NumDescs = static_cast<UINT>(c.geomDescs.size());
                    localDesc.Inputs.pGeometryDescs = c.geomDescs.data();
                }
                buildRaytracingAccelerationStructure(&localDesc);
            } else if constexpr (std::is_same_v<T, CmdCopyRaytracingAS>) {
                copyRaytracingAccelerationStructure(c.dest, c.src, c.mode);
            } else if constexpr (std::is_same_v<T, CmdExecuteIndirect>) {
                if (!c.sig || !c.argBuf) return;
                UINT count = c.maxCount;
                if (c.cntBuf) {
                    UINT actual = 0;
                    std::memcpy(&actual, c.cntBuf->basePtr() + c.cntOff, 4);
                    count = std::min(count, actual);
                }
                const uint8_t* ptr = c.argBuf->basePtr() + c.argOff;
                const UINT stride = c.sig->desc().ByteStride;
                for (UINT i = 0; i < count; ++i) {
                    const uint8_t* rec = ptr + i * stride;
                    size_t off = 0;
                    for (const auto& arg : c.sig->args()) {
                        if (arg.Type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW) {
                            D3D12_DRAW_ARGUMENTS da;
                            std::memcpy(&da, rec + off, sizeof(da));
                            executeRasterDrawCall(pool, state.gfx, DrawCallParams{
                                false, da.VertexCountPerInstance, da.InstanceCount,
                                da.StartVertexLocation, 0, da.StartInstanceLocation});
                            off += sizeof(da);
                        } else if (arg.Type == D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED) {
                            D3D12_DRAW_INDEXED_ARGUMENTS da;
                            std::memcpy(&da, rec + off, sizeof(da));
                            executeRasterDrawCall(pool, state.gfx, DrawCallParams{
                                true, da.IndexCountPerInstance, da.InstanceCount,
                                da.StartIndexLocation, da.BaseVertexLocation, da.StartInstanceLocation});
                            off += sizeof(da);
                        } else if (arg.Type == D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH) {
                            D3D12_DISPATCH_ARGUMENTS da;
                            std::memcpy(&da, rec + off, sizeof(da));
                            execDispatch(pool, state, da.ThreadGroupCountX, da.ThreadGroupCountY, da.ThreadGroupCountZ);
                            off += sizeof(da);
                        }
                    }
                }
            }
        }, cmd);
    }
}

// --- NacelleCommandQueue ---

NacelleCommandQueue::NacelleCommandQueue(ID3D12Device* device, const D3D12_COMMAND_QUEUE_DESC& desc)
    : D3D12PageableBase(device), desc_(desc) {}

NacelleCommandQueue::~NacelleCommandQueue() {
    pool_.wait();
}

HRESULT STDMETHODCALLTYPE NacelleCommandQueue::QueryInterface(REFIID riid, void** ppvObject) {
    if (!ppvObject) return E_POINTER;
    if (InlineIsEqualGUID(riid, __uuidof(ID3D12CommandQueue))) {
        *ppvObject = static_cast<ID3D12CommandQueue*>(this);
        AddRef();
        return S_OK;
    }
    return D3D12PageableBase::QueryInterface(riid, ppvObject);
}

void STDMETHODCALLTYPE NacelleCommandQueue::ExecuteCommandLists(
    UINT NumCommandLists, ID3D12CommandList* const* ppCommandLists) {
    std::lock_guard<std::mutex> lock(execMutex_);
    ExecutionState state{};
    for (UINT i = 0; i < NumCommandLists; ++i) {
        if (!ppCommandLists[i]) continue;
        auto* list = static_cast<NacelleCommandList*>(ppCommandLists[i]);
        list->execute(pool_, state);
    }
}

HRESULT STDMETHODCALLTYPE NacelleCommandQueue::Signal(ID3D12Fence* pFence, UINT64 Value) {
    if (!pFence) return E_INVALIDARG;
    std::lock_guard<std::mutex> lock(execMutex_);
    return pFence->Signal(Value);
}

HRESULT STDMETHODCALLTYPE NacelleCommandQueue::Wait(ID3D12Fence* pFence, UINT64 Value) {
    if (!pFence) return E_INVALIDARG;
    static_cast<NacelleFence*>(pFence)->waitForValue(Value);
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleCommandQueue::GetTimestampFrequency(UINT64* pFrequency) {
    if (!pFrequency) return E_INVALIDARG;
    *pFrequency = 1'000'000'000ULL;
    return S_OK;
}

HRESULT STDMETHODCALLTYPE NacelleCommandQueue::GetClockCalibration(UINT64* pCpuTimestamp, UINT64* pGpuTimestamp) {
    uint64_t t = static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    if (pCpuTimestamp) *pCpuTimestamp = t;
    if (pGpuTimestamp) *pGpuTimestamp = t;
    return S_OK;
}

} // namespace nacelle
