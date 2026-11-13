#pragma once

#include <nacelle/nacelle.h>
#include <directx/d3dx12.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if !defined(_WIN32)
#include <wsl/wrladapter.h>
using Microsoft::WRL::ComPtr;
#else
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

namespace testutil {

inline void checkHR(HRESULT hr, const char* msg) {
    if (FAILED(hr)) {
        std::fprintf(stderr, "FAILED (0x%08x): %s\n", static_cast<unsigned>(hr), msg);
        std::abort();
    }
}

inline std::vector<uint8_t> loadShaderBytes(const char* filename) {
    std::string path = std::string(NACELLE_SHADER_BIN_DIR) + "/" + filename;
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
        std::fprintf(stderr, "Failed to open compiled shader: %s\n", path.c_str());
        std::abort();
    }
    auto sz = ifs.tellg();
    std::vector<uint8_t> data(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (sz > 0) ifs.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

struct TestRuntime {
    ComPtr<ID3D12Device1> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cmdList;
    ComPtr<ID3D12Fence> fence;
    UINT64 fenceVal = 0;

    void init() {
        checkHR(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "CreateDevice");
        D3D12_COMMAND_QUEUE_DESC qd{};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        checkHR(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
        checkHR(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc)), "CreateCommandAllocator");
        checkHR(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cmdList)), "CreateCommandList");
        checkHR(cmdList->Close(), "Initial Close");
        checkHR(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    }

    void resetCmd() {
        checkHR(alloc->Reset(), "AllocReset");
        checkHR(cmdList->Reset(alloc.Get(), nullptr), "CmdReset");
    }

    void executeAndWait() {
        checkHR(cmdList->Close(), "CmdClose");
        ID3D12CommandList* lists[] = {cmdList.Get()};
        queue->ExecuteCommandLists(1, lists);
        ++fenceVal;
        checkHR(queue->Signal(fence.Get(), fenceVal), "Signal");
        if (fence->GetCompletedValue() < fenceVal) {
            checkHR(fence->SetEventOnCompletion(fenceVal, nullptr), "SetEventOnCompletion");
        }
    }

    ComPtr<ID3D12Resource> createBuffer(UINT64 bytes, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
        D3D12_HEAP_PROPERTIES hp{heapType};
        D3D12_RESOURCE_DESC rd = CD3DX12_RESOURCE_DESC::Buffer(bytes, flags);
        ComPtr<ID3D12Resource> res;
        checkHR(device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&res)),
                "CreateBuffer");
        return res;
    }

    void uploadBuffer(ID3D12Resource* dst, const void* data, UINT64 bytes) {
        auto up = createBuffer(bytes, D3D12_HEAP_TYPE_UPLOAD);
        void* p = nullptr;
        checkHR(up->Map(0, nullptr, &p), "MapUpload");
        std::memcpy(p, data, static_cast<size_t>(bytes));
        up->Unmap(0, nullptr);
        resetCmd();
        cmdList->CopyBufferRegion(dst, 0, up.Get(), 0, bytes);
        executeAndWait();
    }

    std::vector<uint8_t> readbackBuffer(ID3D12Resource* src, UINT64 bytes) {
        auto rb = createBuffer(bytes, D3D12_HEAP_TYPE_READBACK);
        resetCmd();
        cmdList->CopyBufferRegion(rb.Get(), 0, src, 0, bytes);
        executeAndWait();
        void* p = nullptr;
        checkHR(rb->Map(0, nullptr, &p), "MapReadback");
        std::vector<uint8_t> out(static_cast<size_t>(bytes));
        std::memcpy(out.data(), p, static_cast<size_t>(bytes));
        rb->Unmap(0, nullptr);
        return out;
    }

    ComPtr<ID3D12RootSignature> createRootSig(const D3D12_ROOT_SIGNATURE_DESC& desc) {
        ComPtr<ID3DBlob> blob, errBlob;
        checkHR(NacelleSerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &errBlob),
                "SerializeRootSig");
        ComPtr<ID3D12RootSignature> sig;
        checkHR(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&sig)),
                "CreateRootSignature");
        return sig;
    }
};

} // namespace testutil
