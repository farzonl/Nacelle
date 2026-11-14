#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "com_base.h"
#include "../compiler/shader_compiler.h"
#include "../rasterizer/raytracing_pipeline.h"

namespace nacelle {

class NacelleBlob : public ComObject<ID3DBlob> {
public:
    explicit NacelleBlob(size_t size);
    NacelleBlob(const void* data, size_t size);

    LPVOID STDMETHODCALLTYPE GetBufferPointer() override { return data_.data(); }
    SIZE_T STDMETHODCALLTYPE GetBufferSize() override { return data_.size(); }

    uint8_t* mutableData() { return data_.data(); }

private:
    std::vector<uint8_t> data_;
};

class NacelleRootSignature : public D3D12DeviceChildBase<ID3D12RootSignature> {
public:
    NacelleRootSignature(ID3D12Device* device, RootSignatureLayout layout);

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    const RootSignatureLayout& layout() const { return layout_; }

    static HRESULT parseBlob(const void* blob, size_t size, RootSignatureLayout& outLayout);

private:
    RootSignatureLayout layout_;
};

class NacellePipelineState : public D3D12PageableBase<ID3D12PipelineState> {
public:
    NacellePipelineState(ID3D12Device* device,
                      const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc,
                      NacelleRootSignature* rootSig,
                      std::shared_ptr<CompiledComputePSO> compiled);

    NacellePipelineState(ID3D12Device* device,
                      const D3D12_GRAPHICS_PIPELINE_STATE_DESC& desc,
                      NacelleRootSignature* rootSig,
                      std::shared_ptr<CompiledGraphicsPSO> compiled);

    ~NacellePipelineState() override;

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
    HRESULT STDMETHODCALLTYPE GetCachedBlob(ID3DBlob** ppBlob) override;

    bool isCompute() const { return isCompute_; }
    NacelleRootSignature* rootSignature() const { return rootSig_; }
    const D3D12_GRAPHICS_PIPELINE_STATE_DESC& graphicsDesc() const { return gfxDesc_; }
    const D3D12_COMPUTE_PIPELINE_STATE_DESC& computeDesc() const { return csDesc_; }
    const std::vector<D3D12_INPUT_ELEMENT_DESC>& inputElements() const { return inputElements_; }
    const std::shared_ptr<CompiledComputePSO>& computePSO() const { return computePSO_; }
    const std::shared_ptr<CompiledGraphicsPSO>& graphicsPSO() const { return graphicsPSO_; }

private:
    bool isCompute_ = false;
    NacelleRootSignature* rootSig_ = nullptr; // AddRef'd
    D3D12_COMPUTE_PIPELINE_STATE_DESC csDesc_{};
    D3D12_GRAPHICS_PIPELINE_STATE_DESC gfxDesc_{};
    std::vector<D3D12_INPUT_ELEMENT_DESC> inputElements_;
    std::vector<std::string> semanticNames_;
    std::shared_ptr<CompiledComputePSO> computePSO_;
    std::shared_ptr<CompiledGraphicsPSO> graphicsPSO_;
};

class NacellePipelineLibrary : public D3D12DeviceChildBase<ID3D12PipelineLibrary> {
public:
    NacellePipelineLibrary(ID3D12Device* device, const void* blob, size_t size);

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;

    HRESULT STDMETHODCALLTYPE StorePipeline(
        LPCWSTR pName,
        ID3D12PipelineState* pPipeline) override;

    HRESULT STDMETHODCALLTYPE LoadGraphicsPipeline(
        LPCWSTR pName,
        const D3D12_GRAPHICS_PIPELINE_STATE_DESC* pDesc,
        REFIID riid,
        void** ppPipelineState) override;

    HRESULT STDMETHODCALLTYPE LoadComputePipeline(
        LPCWSTR pName,
        const D3D12_COMPUTE_PIPELINE_STATE_DESC* pDesc,
        REFIID riid,
        void** ppPipelineState) override;

    SIZE_T STDMETHODCALLTYPE GetSerializedSize() override;

    HRESULT STDMETHODCALLTYPE Serialize(
        void* pData,
        SIZE_T DataSizeInBytes) override;

private:
    struct StoredEntry {
        bool isCompute = false;
        uint64_t hashLo = 0;
        uint64_t hashHi = 0;
        std::vector<uint8_t> serializedObj;
    };

    std::mutex mu_;
    std::unordered_map<std::string, StoredEntry> entries_;
};

class NacelleStateObject
    : public D3D12PageableBase<ID3D12StateObject>,
      public ID3D12StateObjectProperties {
public:
    NacelleStateObject(ID3D12Device* device, CompiledDxrStateObject compiled);

    HRESULT STDMETHODCALLTYPE QueryInterface(REFIID riid, void** ppvObject) override;
    ULONG STDMETHODCALLTYPE AddRef() override { return D3D12PageableBase::AddRef(); }
    ULONG STDMETHODCALLTYPE Release() override { return D3D12PageableBase::Release(); }

    void* STDMETHODCALLTYPE GetShaderIdentifier(LPCWSTR pExportName) override;
    UINT64 STDMETHODCALLTYPE GetShaderStackSize(LPCWSTR pExportName) override {
        (void)pExportName;
        return 4096;
    }
    UINT64 STDMETHODCALLTYPE GetPipelineStackSize() override { return stackSize_; }
    void STDMETHODCALLTYPE SetPipelineStackSize(UINT64 PipelineStackSizeInBytes) override {
        stackSize_ = PipelineStackSizeInBytes;
    }

    const CompiledDxrStateObject& compiled() const { return compiled_; }

private:
    CompiledDxrStateObject compiled_;
    UINT64 stackSize_ = 65536;
};

} // namespace nacelle
