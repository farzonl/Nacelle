#include "replay_capture.h"
#include <array>
#include <bit>
#include <cstring>
#include <sstream>
#if defined(_WIN32)
#include <wrl/client.h>
#else
#include <wsl/wrladapter.h>
#endif

namespace nacelle::replay {
using Microsoft::WRL::ComPtr;

bool executeBundle(const Bundle& bundle, uint32_t repetitions, std::string& error) {
    if (!sameEnvironment(bundle.manifest.environment, currentEnvironment())) {
        error = "Replay build/compiler/CPU/SIMD/FP environment mismatch";
        return false;
    }
    if (!validateSemantics(bundle, error)) return false;
    const auto check = [&](HRESULT result, const char* operation) {
        if (SUCCEEDED(result)) return true;
        std::ostringstream message;
        message << "Replay " << operation << " failed: 0x" << std::hex << static_cast<uint32_t>(result);
        error = message.str();
        return false;
    };
    const auto& metadata = bundle.metadata;
    for (uint32_t repetition = 0; repetition < repetitions; ++repetition) {
        ComPtr<ID3D12Device> device;
        if (!check(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)), "device creation"))
            return false;
        std::vector<ComPtr<ID3D12Resource>> resources(metadata.resources.size());
        std::vector<ComPtr<ID3D12RootSignature>> roots(metadata.roots.size());
        std::vector<ComPtr<ID3D12PipelineState>> pipelines(metadata.pipelines.size());
        std::vector<ComPtr<ID3D12DescriptorHeap>> heaps(metadata.heaps.size());
        for (const auto& resource : metadata.resources) {
            D3D12_HEAP_PROPERTIES properties{};
            properties.Type = static_cast<D3D12_HEAP_TYPE>(resource.heapType);
            D3D12_RESOURCE_DESC desc{};
            desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Width = resource.size;
            desc.Height = 1;
            desc.DepthOrArraySize = desc.MipLevels = 1;
            desc.SampleDesc.Count = 1;
            desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags = static_cast<D3D12_RESOURCE_FLAGS>(resource.flags);
            auto& object = resources[resource.id - 1];
            if (!check(device->CreateCommittedResource(&properties, D3D12_HEAP_FLAG_NONE, &desc,
                D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&object)), "buffer creation")) return false;
            const auto initial = blobBytes(bundle, resource.initialBlob);
            std::memcpy(static_cast<NacelleResource*>(object.Get())->basePtr(), initial.data(), initial.size());
        }
        for (const auto& root : metadata.roots) {
            const auto bytes = blobBytes(bundle, root.blob);
            if (!check(device->CreateRootSignature(0, bytes.data(), bytes.size(), IID_PPV_ARGS(&roots[root.id - 1])),
                "root signature creation")) return false;
        }
        for (const auto& pipeline : metadata.pipelines) {
            const auto bytes = blobBytes(bundle, pipeline.shaderBlob);
            D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
            desc.pRootSignature = roots[pipeline.root - 1].Get();
            desc.CS = {bytes.data(), bytes.size()};
            desc.NodeMask = pipeline.nodeMask;
            desc.Flags = static_cast<D3D12_PIPELINE_STATE_FLAGS>(pipeline.flags);
            if (!check(device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pipelines[pipeline.id - 1])),
                "compute pipeline creation")) {
                waitForAllAsyncJITTasks();
                return false;
            }
        }
        waitForAllAsyncJITTasks();
        for (const auto& pipeline : metadata.pipelines) {
            const auto& compiled = static_cast<NacellePipelineState*>(pipelines[pipeline.id - 1].Get())->computePSO();
            const auto* entry = compiled->profileEntries.load();
            if (!entry || !entry->fn || entry->tier != 2 || entry->artifactId != pipeline.nativeId ||
                compiled->simdWidth != bundle.manifest.environment.simdWidth) {
                error = "Replay exact stable Tier-2 native code identity mismatch before execution";
                return false;
            }
        }
        for (const auto& heap : metadata.heaps) {
            D3D12_DESCRIPTOR_HEAP_DESC desc{};
            desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            desc.NumDescriptors = heap.count;
            desc.Flags = static_cast<D3D12_DESCRIPTOR_HEAP_FLAGS>(heap.flags);
            if (!check(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&heaps[heap.id - 1])), "heap creation"))
                return false;
        }
        const uint64_t stride = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const auto cpuHandle = [&](uint32_t heap, uint32_t slot) {
            auto handle = heaps[heap - 1]->GetCPUDescriptorHandleForHeapStart();
            handle.ptr += stride * slot;
            return handle;
        };
        const auto gpuHandle = [&](uint32_t heap, uint32_t slot) {
            auto handle = heaps[heap - 1]->GetGPUDescriptorHandleForHeapStart();
            handle.ptr += stride * slot;
            return handle;
        };
        for (const auto& view : metadata.views) {
            auto* resource = resources[view.resource - 1].Get();
            const auto destination = cpuHandle(view.heap, view.slot);
            if (view.kind == "cbv") {
                D3D12_CONSTANT_BUFFER_VIEW_DESC desc{resource->GetGPUVirtualAddress() + view.first,
                    static_cast<UINT>(view.bytes)};
                device->CreateConstantBufferView(&desc, destination);
            } else if (view.kind == "srv") {
                D3D12_SHADER_RESOURCE_VIEW_DESC desc{};
                desc.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
                desc.Format = static_cast<DXGI_FORMAT>(view.format);
                desc.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
                desc.Buffer.FirstElement = view.first;
                desc.Buffer.NumElements = view.count;
                desc.Buffer.StructureByteStride = !view.format && !view.flags ? view.stride : 0;
                desc.Buffer.Flags = static_cast<D3D12_BUFFER_SRV_FLAGS>(view.flags);
                device->CreateShaderResourceView(resource, &desc, destination);
            } else {
                D3D12_UNORDERED_ACCESS_VIEW_DESC desc{};
                desc.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
                desc.Format = static_cast<DXGI_FORMAT>(view.format);
                desc.Buffer.FirstElement = view.first;
                desc.Buffer.NumElements = view.count;
                desc.Buffer.StructureByteStride = !view.format && !view.flags ? view.stride : 0;
                desc.Buffer.Flags = static_cast<D3D12_BUFFER_UAV_FLAGS>(view.flags);
                device->CreateUnorderedAccessView(resource, nullptr, &desc, destination);
            }
        }
        ComPtr<ID3D12CommandQueue> queue;
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        queueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        if (!check(device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&queue)), "queue creation") ||
            !check(device->CreateCommandAllocator(queueDesc.Type, IID_PPV_ARGS(&allocator)), "allocator creation") ||
            !check(device->CreateCommandList(0, queueDesc.Type, allocator.Get(), nullptr, IID_PPV_ARGS(&list)), "list creation"))
            return false;
        bool recording = false;
        uint32_t root = 0;
        const auto submit = [&]() {
            if (!check(list->Close(), "list close")) return false;
            ID3D12CommandList* lists[]{list.Get()};
            queue->ExecuteCommandLists(1, lists);
            return true;
        };
        for (const auto& command : metadata.commands) {
            const auto& words = command.data;
            if (command.kind == "submission") {
                if (recording) {
                    if (!submit() || !check(allocator->Reset(), "allocator reset") ||
                        !check(list->Reset(allocator.Get(), nullptr), "list reset")) return false;
                }
                recording = true;
                root = 0;
            } else if (command.kind == "pipeline") list->SetPipelineState(pipelines[words[0] - 1].Get());
            else if (command.kind == "root") {
                root = words[0];
                list->SetComputeRootSignature(roots[root - 1].Get());
            } else if (command.kind == "heap") {
                ID3D12DescriptorHeap* selected[]{heaps[words[0] - 1].Get()};
                list->SetDescriptorHeaps(1, selected);
            } else if (command.kind == "table")
                list->SetComputeRootDescriptorTable(words[0], gpuHandle(words[1], words[2]));
            else if (command.kind == "constants") {
                std::vector<uint32_t> values(words.begin() + 2, words.end());
                list->SetComputeRoot32BitConstants(words[0], values.size(), values.data(), words[1]);
            } else if (command.kind == "view") {
                const auto& layout = static_cast<NacelleRootSignature*>(roots[root - 1].Get())->layout();
                const auto address = resources[words[1] - 1]->GetGPUVirtualAddress() + words[2];
                switch (layout.params[words[0]].type) {
                    case D3D12_ROOT_PARAMETER_TYPE_CBV: list->SetComputeRootConstantBufferView(words[0], address); break;
                    case D3D12_ROOT_PARAMETER_TYPE_SRV: list->SetComputeRootShaderResourceView(words[0], address); break;
                    case D3D12_ROOT_PARAMETER_TYPE_UAV: list->SetComputeRootUnorderedAccessView(words[0], address); break;
                    default: error = "Replay root descriptor changed after validation"; return false;
                }
            } else if (command.kind == "dispatch") list->Dispatch(words[0], words[1], words[2]);
            else if (command.kind == "copy") list->CopyBufferRegion(resources[words[0] - 1].Get(), words[1],
                resources[words[2] - 1].Get(), words[3], words[4]);
            else if (command.kind == "clear") {
                if (words[3]) {
                    std::array<float, 4> values;
                    for (size_t index = 0; index < 4; ++index) values[index] = std::bit_cast<float>(static_cast<uint32_t>(words[index + 4]));
                    list->ClearUnorderedAccessViewFloat(gpuHandle(words[0], words[1]), cpuHandle(words[0], words[1]),
                        resources[words[2] - 1].Get(), values.data(), 0, nullptr);
                } else {
                    std::array<UINT, 4> values;
                    for (size_t index = 0; index < 4; ++index) values[index] = words[index + 4];
                    list->ClearUnorderedAccessViewUint(gpuHandle(words[0], words[1]), cpuHandle(words[0], words[1]),
                        resources[words[2] - 1].Get(), values.data(), 0, nullptr);
                }
            } else if (command.kind == "barrier") {
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Flags = static_cast<D3D12_RESOURCE_BARRIER_FLAGS>(words[5]);
                if (!words[0]) {
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barrier.Transition = {resources[words[1] - 1].Get(), static_cast<UINT>(words[2]),
                        static_cast<D3D12_RESOURCE_STATES>(words[3]), static_cast<D3D12_RESOURCE_STATES>(words[4])};
                } else {
                    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                    barrier.UAV.pResource = words[1] ? resources[words[1] - 1].Get() : nullptr;
                }
                list->ResourceBarrier(1, &barrier);
            } else { error = "Replay command changed after validation"; return false; }
        }
        if (!submit()) return false;
        for (const auto& resource : metadata.resources) {
            if (!(resource.access & 2)) continue;
            const auto expected = blobBytes(bundle, resource.expectedBlob);
            const auto* actual = static_cast<NacelleResource*>(resources[resource.id - 1].Get())->basePtr();
            for (size_t offset = 0; offset < expected.size(); ++offset)
                if (actual[offset] != static_cast<uint8_t>(expected[offset])) {
                    error = "Replay output mismatch: repetition " + std::to_string(repetition + 1) +
                        ", resource " + std::to_string(resource.id) + ", byte " + std::to_string(offset);
                    return false;
                }
        }
    }
    return true;
}
}
