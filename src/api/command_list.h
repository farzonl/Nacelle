#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <variant>
#include <vector>

#include <BS_thread_pool.hpp>

#include "com_base.h"
#include "d3d12_types.h"
#include "descriptors.h"
#include "pipeline.h"
#include "resource.h"
#include "../rasterizer/rasterizer.h"

namespace nacelle {

class NacelleFence : public D3D12PageableBase<ID3D12Fence> {
public:
    NacelleFence(ID3D12Device* device, UINT64 initialValue, D3D12_FENCE_FLAGS flags);

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    UINT64 STDMETHODCALLTYPE GetCompletedValue() override;
    HRESULT STDMETHODCALLTYPE SetEventOnCompletion(UINT64 Value, HANDLE hEvent) override;
    HRESULT STDMETHODCALLTYPE Signal(UINT64 Value) override;

    void waitForValue(UINT64 value);

private:
    struct Waiter {
        UINT64 targetValue = 0;
        HANDLE hEvent = nullptr;
    };

    std::atomic<UINT64> value_{0};
    D3D12_FENCE_FLAGS flags_ = D3D12_FENCE_FLAG_NONE;
    std::mutex mu_;
    std::condition_variable cv_;
    std::vector<Waiter> waiters_;
};

class NacelleCommandAllocator : public D3D12PageableBase<ID3D12CommandAllocator> {
public:
    NacelleCommandAllocator(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type);
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
    HRESULT STDMETHODCALLTYPE Reset() override;

    D3D12_COMMAND_LIST_TYPE listType() const { return type_; }

private:
    D3D12_COMMAND_LIST_TYPE type_;
};

class NacelleCommandSignature : public D3D12PageableBase<ID3D12CommandSignature> {
public:
    NacelleCommandSignature(ID3D12Device* device, const D3D12_COMMAND_SIGNATURE_DESC& desc);
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    const D3D12_COMMAND_SIGNATURE_DESC& desc() const { return desc_; }
    const std::vector<D3D12_INDIRECT_ARGUMENT_DESC>& args() const { return args_; }

private:
    D3D12_COMMAND_SIGNATURE_DESC desc_{};
    std::vector<D3D12_INDIRECT_ARGUMENT_DESC> args_;
};

class NacelleQueryHeap : public D3D12PageableBase<ID3D12QueryHeap> {
public:
    NacelleQueryHeap(ID3D12Device* device, const D3D12_QUERY_HEAP_DESC& desc);
    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    std::vector<uint64_t> values;
    D3D12_QUERY_HEAP_DESC desc{};
};

// Recorded command variants
struct CmdSetPipelineState { NacellePipelineState* pso = nullptr; };
struct CmdSetRootSignature { bool compute = false; NacelleRootSignature* sig = nullptr; };
struct CmdSetDescriptorHeaps { NacelleDescriptorHeap* resHeap = nullptr; NacelleDescriptorHeap* sampHeap = nullptr; };
struct CmdSetRootDescTable { bool compute = false; UINT idx = 0; D3D12_GPU_DESCRIPTOR_HANDLE handle{}; };
struct CmdSetRoot32BitConsts { bool compute = false; UINT idx = 0; UINT destOffset = 0; std::vector<UINT> data; };
struct CmdSetRootView { bool compute = false; UINT idx = 0; D3D12_GPU_VIRTUAL_ADDRESS addr = 0; };
struct CmdDispatch { UINT x = 1, y = 1, z = 1; };
struct CmdDrawInstanced { UINT vCount = 0, iCount = 1, startV = 0, startI = 0; };
struct CmdDrawIndexedInstanced { UINT idxCount = 0, iCount = 1, startIdx = 0; INT baseV = 0; UINT startI = 0; };
struct CmdCopyBufferRegion { NacelleResource* dst = nullptr; UINT64 dstOff = 0; NacelleResource* src = nullptr; UINT64 srcOff = 0; UINT64 bytes = 0; };
struct CmdCopyTextureRegion { D3D12_TEXTURE_COPY_LOCATION dst{}; D3D12_TEXTURE_COPY_LOCATION src{}; UINT dstX = 0, dstY = 0, dstZ = 0; bool hasBox = false; D3D12_BOX srcBox{}; };
struct CmdCopyResource { NacelleResource* dst = nullptr; NacelleResource* src = nullptr; };
struct CmdResolveSubresource { NacelleResource* dst = nullptr; UINT dstSub = 0; NacelleResource* src = nullptr; UINT srcSub = 0; DXGI_FORMAT fmt = DXGI_FORMAT_UNKNOWN; };
struct CmdIASetTopology { D3D12_PRIMITIVE_TOPOLOGY topo = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST; };
struct CmdIASetIndexBuffer { D3D12_INDEX_BUFFER_VIEW view{}; bool valid = false; };
struct CmdIASetVertexBuffers { UINT startSlot = 0; std::vector<D3D12_VERTEX_BUFFER_VIEW> views; };
struct CmdRSSetViewports { std::vector<D3D12_VIEWPORT> vps; };
struct CmdRSSetScissors { std::vector<D3D12_RECT> rects; };
struct CmdOMSetBlendFactor { float factor[4]{1, 1, 1, 1}; };
struct CmdOMSetStencilRef { UINT ref = 0; };
struct CmdRSSetShadingRate {
    D3D12_SHADING_RATE rate = D3D12_SHADING_RATE_1X1;
    D3D12_SHADING_RATE_COMBINER combiners[2]{
        D3D12_SHADING_RATE_COMBINER_PASSTHROUGH,
        D3D12_SHADING_RATE_COMBINER_PASSTHROUGH,
    };
};
struct CmdOMSetRenderTargets { std::vector<NacelleRenderTargetDescriptor> rtvs; bool hasDSV = false; NacelleRenderTargetDescriptor dsv{}; };
struct CmdClearRTV { NacelleRenderTargetDescriptor rtv{}; float rgba[4]{}; std::vector<D3D12_RECT> rects; };
struct CmdClearDSV { NacelleRenderTargetDescriptor dsv{}; D3D12_CLEAR_FLAGS flags{}; float depth = 1.0f; UINT8 stencil = 0; std::vector<D3D12_RECT> rects; };
struct CmdClearUAV { NacelleDescriptor desc{}; NacelleResource* res = nullptr; uint32_t uVal[4]{}; float fVal[4]{}; bool isFloat = false; std::vector<D3D12_RECT> rects; };
struct CmdExecuteBundle { class NacelleCommandList* bundle = nullptr; };
struct CmdEndQuery { NacelleQueryHeap* heap = nullptr; D3D12_QUERY_TYPE type{}; UINT index = 0; };
struct CmdResolveQueryData { NacelleQueryHeap* heap = nullptr; UINT start = 0; UINT count = 0; NacelleResource* dst = nullptr; UINT64 offset = 0; };
struct CmdExecuteIndirect { NacelleCommandSignature* sig = nullptr; UINT maxCount = 0; NacelleResource* argBuf = nullptr; UINT64 argOff = 0; NacelleResource* cntBuf = nullptr; UINT64 cntOff = 0; };

using RecordedCommand = std::variant<
    CmdSetPipelineState, CmdSetRootSignature, CmdSetDescriptorHeaps,
    CmdSetRootDescTable, CmdSetRoot32BitConsts, CmdSetRootView,
    CmdDispatch, CmdDrawInstanced, CmdDrawIndexedInstanced,
    CmdCopyBufferRegion, CmdCopyTextureRegion, CmdCopyResource, CmdResolveSubresource,
    CmdIASetTopology, CmdIASetIndexBuffer, CmdIASetVertexBuffers,
    CmdRSSetViewports, CmdRSSetScissors, CmdOMSetBlendFactor, CmdOMSetStencilRef,
    CmdRSSetShadingRate,
    CmdOMSetRenderTargets, CmdClearRTV, CmdClearDSV, CmdClearUAV,
    CmdExecuteBundle, CmdEndQuery, CmdResolveQueryData, CmdExecuteIndirect>;

struct ExecutionState {
    NacellePipelineState* pso = nullptr;
    NacelleRootSignature* computeRootSig = nullptr;
    NacelleRootSignature* graphicsRootSig = nullptr;
    DriverRootContext computeCtx{};
    GraphicsExecutionState gfx{};
};

class NacelleCommandList : public D3D12DeviceChildBase<ID3D12GraphicsCommandList6> {
public:
    NacelleCommandList(ID3D12Device* device,
                    D3D12_COMMAND_LIST_TYPE type,
                    NacelleCommandAllocator* allocator,
                    ID3D12PipelineState* initialState);
    ~NacelleCommandList() override;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
    D3D12_COMMAND_LIST_TYPE STDMETHODCALLTYPE GetType() override { return type_; }

    HRESULT STDMETHODCALLTYPE Close() override;
    HRESULT STDMETHODCALLTYPE Reset(ID3D12CommandAllocator* pAllocator, ID3D12PipelineState* pInitialState) override;
    void STDMETHODCALLTYPE ClearState(ID3D12PipelineState* pPipelineState) override;
    void STDMETHODCALLTYPE DrawInstanced(UINT VertexCountPerInstance, UINT InstanceCount, UINT StartVertexLocation, UINT StartInstanceLocation) override;
    void STDMETHODCALLTYPE DrawIndexedInstanced(UINT IndexCountPerInstance, UINT InstanceCount, UINT StartIndexLocation, INT BaseVertexLocation, UINT StartInstanceLocation) override;
    void STDMETHODCALLTYPE Dispatch(UINT ThreadGroupCountX, UINT ThreadGroupCountY, UINT ThreadGroupCountZ) override;
    void STDMETHODCALLTYPE CopyBufferRegion(ID3D12Resource* pDstBuffer, UINT64 DstOffset, ID3D12Resource* pSrcBuffer, UINT64 SrcOffset, UINT64 NumBytes) override;
    void STDMETHODCALLTYPE CopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION* pDst, UINT DstX, UINT DstY, UINT DstZ, const D3D12_TEXTURE_COPY_LOCATION* pSrc, const D3D12_BOX* pSrcBox) override;
    void STDMETHODCALLTYPE CopyResource(ID3D12Resource* pDstResource, ID3D12Resource* pSrcResource) override;
    void STDMETHODCALLTYPE CopyTiles(ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Resource*, UINT64, D3D12_TILE_COPY_FLAGS) override {}
    void STDMETHODCALLTYPE ResolveSubresource(ID3D12Resource* pDstResource, UINT DstSubresource, ID3D12Resource* pSrcResource, UINT SrcSubresource, DXGI_FORMAT Format) override;
    void STDMETHODCALLTYPE IASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY PrimitiveTopology) override;
    void STDMETHODCALLTYPE RSSetViewports(UINT NumViewports, const D3D12_VIEWPORT* pViewports) override;
    void STDMETHODCALLTYPE RSSetScissorRects(UINT NumRects, const D3D12_RECT* pRects) override;
    void STDMETHODCALLTYPE OMSetBlendFactor(const FLOAT BlendFactor[4]) override;
    void STDMETHODCALLTYPE OMSetStencilRef(UINT StencilRef) override;
    void STDMETHODCALLTYPE SetPipelineState(ID3D12PipelineState* pPipelineState) override;
    void STDMETHODCALLTYPE ResourceBarrier(UINT NumBarriers, const D3D12_RESOURCE_BARRIER* pBarriers) override;
    void STDMETHODCALLTYPE ExecuteBundle(ID3D12GraphicsCommandList* pCommandList) override;
    void STDMETHODCALLTYPE SetDescriptorHeaps(UINT NumDescriptorHeaps, ID3D12DescriptorHeap* const* ppDescriptorHeaps) override;
    void STDMETHODCALLTYPE SetComputeRootSignature(ID3D12RootSignature* pRootSignature) override;
    void STDMETHODCALLTYPE SetGraphicsRootSignature(ID3D12RootSignature* pRootSignature) override;
    void STDMETHODCALLTYPE SetComputeRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor) override;
    void STDMETHODCALLTYPE SetGraphicsRootDescriptorTable(UINT RootParameterIndex, D3D12_GPU_DESCRIPTOR_HANDLE BaseDescriptor) override;
    void STDMETHODCALLTYPE SetComputeRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues) override;
    void STDMETHODCALLTYPE SetGraphicsRoot32BitConstant(UINT RootParameterIndex, UINT SrcData, UINT DestOffsetIn32BitValues) override;
    void STDMETHODCALLTYPE SetComputeRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, const void* pSrcData, UINT DestOffsetIn32BitValues) override;
    void STDMETHODCALLTYPE SetGraphicsRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, const void* pSrcData, UINT DestOffsetIn32BitValues) override;
    void STDMETHODCALLTYPE SetComputeRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) override;
    void STDMETHODCALLTYPE SetGraphicsRootConstantBufferView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) override;
    void STDMETHODCALLTYPE SetComputeRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) override;
    void STDMETHODCALLTYPE SetGraphicsRootShaderResourceView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) override;
    void STDMETHODCALLTYPE SetComputeRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) override;
    void STDMETHODCALLTYPE SetGraphicsRootUnorderedAccessView(UINT RootParameterIndex, D3D12_GPU_VIRTUAL_ADDRESS BufferLocation) override;
    void STDMETHODCALLTYPE IASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW* pView) override;
    void STDMETHODCALLTYPE IASetVertexBuffers(UINT StartSlot, UINT NumViews, const D3D12_VERTEX_BUFFER_VIEW* pViews) override;
    void STDMETHODCALLTYPE SOSetTargets(UINT, UINT, const D3D12_STREAM_OUTPUT_BUFFER_VIEW*) override {}
    void STDMETHODCALLTYPE OMSetRenderTargets(UINT NumRenderTargetDescriptors, const D3D12_CPU_DESCRIPTOR_HANDLE* pRenderTargetDescriptors, BOOL RTsSingleHandleToDescriptorRange, const D3D12_CPU_DESCRIPTOR_HANDLE* pDepthStencilDescriptor) override;
    void STDMETHODCALLTYPE ClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE DepthStencilView, D3D12_CLEAR_FLAGS ClearFlags, FLOAT Depth, UINT8 Stencil, UINT NumRects, const D3D12_RECT* pRects) override;
    void STDMETHODCALLTYPE ClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE RenderTargetView, const FLOAT ColorRGBA[4], UINT NumRects, const D3D12_RECT* pRects) override;
    void STDMETHODCALLTYPE ClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap, D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle, ID3D12Resource* pResource, const UINT Values[4], UINT NumRects, const D3D12_RECT* pRects) override;
    void STDMETHODCALLTYPE ClearUnorderedAccessViewFloat(D3D12_GPU_DESCRIPTOR_HANDLE ViewGPUHandleInCurrentHeap, D3D12_CPU_DESCRIPTOR_HANDLE ViewCPUHandle, ID3D12Resource* pResource, const FLOAT Values[4], UINT NumRects, const D3D12_RECT* pRects) override;
    void STDMETHODCALLTYPE DiscardResource(ID3D12Resource*, const D3D12_DISCARD_REGION*) override {}
    void STDMETHODCALLTYPE BeginQuery(ID3D12QueryHeap*, D3D12_QUERY_TYPE, UINT) override {}
    void STDMETHODCALLTYPE EndQuery(ID3D12QueryHeap* pQueryHeap, D3D12_QUERY_TYPE Type, UINT Index) override;
    void STDMETHODCALLTYPE ResolveQueryData(ID3D12QueryHeap* pQueryHeap, D3D12_QUERY_TYPE Type, UINT StartIndex, UINT NumQueries, ID3D12Resource* pDestinationBuffer, UINT64 AlignedDestinationBufferOffset) override;
    void STDMETHODCALLTYPE SetPredication(ID3D12Resource*, UINT64, D3D12_PREDICATION_OP) override {}
    void STDMETHODCALLTYPE SetMarker(UINT, const void*, UINT) override {}
    void STDMETHODCALLTYPE BeginEvent(UINT, const void*, UINT) override {}
    void STDMETHODCALLTYPE EndEvent() override {}
    void STDMETHODCALLTYPE ExecuteIndirect(ID3D12CommandSignature* pCommandSignature, UINT MaxCommandCount, ID3D12Resource* pArgumentBuffer, UINT64 ArgumentBufferOffset, ID3D12Resource* pCountBuffer, UINT64 CountBufferOffset) override;

    // ID3D12GraphicsCommandList1
    void STDMETHODCALLTYPE AtomicCopyBufferUINT(ID3D12Resource*, UINT64, ID3D12Resource*, UINT64, UINT, ID3D12Resource* const*, const D3D12_SUBRESOURCE_RANGE_UINT64*) override {}
    void STDMETHODCALLTYPE AtomicCopyBufferUINT64(ID3D12Resource*, UINT64, ID3D12Resource*, UINT64, UINT, ID3D12Resource* const*, const D3D12_SUBRESOURCE_RANGE_UINT64*) override {}
    void STDMETHODCALLTYPE OMSetDepthBounds(FLOAT, FLOAT) override {}
    void STDMETHODCALLTYPE SetSamplePositions(UINT, UINT, D3D12_SAMPLE_POSITION*) override {}
    void STDMETHODCALLTYPE ResolveSubresourceRegion(ID3D12Resource*, UINT, UINT, UINT, ID3D12Resource*, UINT, D3D12_RECT*, DXGI_FORMAT, D3D12_RESOLVE_MODE) override {}
    void STDMETHODCALLTYPE SetViewInstanceMask(UINT) override {}

    // ID3D12GraphicsCommandList2
    void STDMETHODCALLTYPE WriteBufferImmediate(UINT, const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER*, const D3D12_WRITEBUFFERIMMEDIATE_MODE*) override {}

    // ID3D12GraphicsCommandList3
    void STDMETHODCALLTYPE SetProtectedResourceSession(ID3D12ProtectedResourceSession*) override {}

    // ID3D12GraphicsCommandList4
    void STDMETHODCALLTYPE BeginRenderPass(UINT, const D3D12_RENDER_PASS_RENDER_TARGET_DESC*, const D3D12_RENDER_PASS_DEPTH_STENCIL_DESC*, D3D12_RENDER_PASS_FLAGS) override {}
    void STDMETHODCALLTYPE EndRenderPass() override {}
    void STDMETHODCALLTYPE InitializeMetaCommand(ID3D12MetaCommand*, const void*, SIZE_T) override {}
    void STDMETHODCALLTYPE ExecuteMetaCommand(ID3D12MetaCommand*, const void*, SIZE_T) override {}
    void STDMETHODCALLTYPE BuildRaytracingAccelerationStructure(const D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC*, UINT, const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*) override {}
    void STDMETHODCALLTYPE EmitRaytracingAccelerationStructurePostbuildInfo(const D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC*, UINT, const D3D12_GPU_VIRTUAL_ADDRESS*) override {}
    void STDMETHODCALLTYPE CopyRaytracingAccelerationStructure(D3D12_GPU_VIRTUAL_ADDRESS, D3D12_GPU_VIRTUAL_ADDRESS, D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE) override {}
    void STDMETHODCALLTYPE SetPipelineState1(ID3D12StateObject*) override {}
    void STDMETHODCALLTYPE DispatchRays(const D3D12_DISPATCH_RAYS_DESC*) override {}

    // ID3D12GraphicsCommandList5
    void STDMETHODCALLTYPE RSSetShadingRate(D3D12_SHADING_RATE baseShadingRate, const D3D12_SHADING_RATE_COMBINER* combiners) override;
    void STDMETHODCALLTYPE RSSetShadingRateImage(ID3D12Resource*) override {}

    // ID3D12GraphicsCommandList6
    void STDMETHODCALLTYPE DispatchMesh(UINT, UINT, UINT) override {}

    void execute(BS::thread_pool<>& pool, ExecutionState& state) const;

private:
    void releaseRecordedRefs();

    D3D12_COMMAND_LIST_TYPE type_;
    bool closed_ = false;
    std::vector<RecordedCommand> commands_;
    std::vector<IUnknown*> heldRefs_;
};

class NacelleCommandQueue : public D3D12PageableBase<ID3D12CommandQueue> {
public:
    NacelleCommandQueue(ID3D12Device* device, const D3D12_COMMAND_QUEUE_DESC& desc);
    ~NacelleCommandQueue() override;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    void STDMETHODCALLTYPE UpdateTileMappings(ID3D12Resource*, UINT, const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, ID3D12Heap*, UINT, const D3D12_TILE_RANGE_FLAGS*, const UINT*, const UINT*, D3D12_TILE_MAPPING_FLAGS) override {}
    void STDMETHODCALLTYPE CopyTileMappings(ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*, ID3D12Resource*, const D3D12_TILED_RESOURCE_COORDINATE*, const D3D12_TILE_REGION_SIZE*, D3D12_TILE_MAPPING_FLAGS) override {}
    void STDMETHODCALLTYPE ExecuteCommandLists(UINT NumCommandLists, ID3D12CommandList* const* ppCommandLists) override;
    void STDMETHODCALLTYPE SetMarker(UINT, const void*, UINT) override {}
    void STDMETHODCALLTYPE BeginEvent(UINT, const void*, UINT) override {}
    void STDMETHODCALLTYPE EndEvent() override {}
    HRESULT STDMETHODCALLTYPE Signal(ID3D12Fence* pFence, UINT64 Value) override;
    HRESULT STDMETHODCALLTYPE Wait(ID3D12Fence* pFence, UINT64 Value) override;
    HRESULT STDMETHODCALLTYPE GetTimestampFrequency(UINT64* pFrequency) override;
    HRESULT STDMETHODCALLTYPE GetClockCalibration(UINT64* pCpuTimestamp, UINT64* pGpuTimestamp) override;

#if defined(_MSC_VER) || !defined(_WIN32)
    D3D12_COMMAND_QUEUE_DESC STDMETHODCALLTYPE GetDesc() override { return desc_; }
#else
    D3D12_COMMAND_QUEUE_DESC* STDMETHODCALLTYPE GetDesc(D3D12_COMMAND_QUEUE_DESC* RetVal) override {
        *RetVal = desc_;
        return RetVal;
    }
#endif

    BS::thread_pool<>& threadPool() { return pool_; }

private:
    D3D12_COMMAND_QUEUE_DESC desc_{};
    std::mutex execMutex_;
    BS::thread_pool<> pool_;
};

} // namespace nacelle
