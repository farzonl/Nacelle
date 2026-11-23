#include "test_common.h"
#include <nacelle/replay.h>
#include "instrumentation/replay_format.h"
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Expected output path and fixture mode");
        const std::string path = argv[1], mode = argv[2];
        std::filesystem::remove(path);
        NacelleReplayConfiguration configuration;
        configuration.flags = mode == "disabled" ? 0 : NACELLE_REPLAY_SNAPSHOTS | NACELLE_REPLAY_SHADER_BINARIES;
        if (mode == "source") configuration.flags |= NACELLE_REPLAY_EMBEDDED_SOURCES;
        testutil::checkHR(NacelleConfigureReplay(&configuration), "Enable replay provenance");
        testutil::TestRuntime runtime;
#if defined(NACELLE_PROFILE_SHARED_RUNTIME)
        testutil::checkHR(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&runtime.device)), "Shared device");
        D3D12_COMMAND_QUEUE_DESC queueDesc{};
        testutil::checkHR(runtime.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&runtime.queue)), "Queue");
        testutil::checkHR(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&runtime.alloc)), "Allocator");
        testutil::checkHR(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, runtime.alloc.Get(),
            nullptr, IID_PPV_ARGS(&runtime.cmdList)), "List");
        testutil::checkHR(runtime.cmdList->Close(), "Initial close");
#else
        runtime.init();
#endif
        constexpr size_t count = 512;
        const auto seed = runtime.createBuffer(count * 4, D3D12_HEAP_TYPE_UPLOAD);
        const auto a = runtime.createBuffer(count * 4, D3D12_HEAP_TYPE_DEFAULT);
        const auto b = runtime.createBuffer(count * 4, D3D12_HEAP_TYPE_UPLOAD);
        const auto output = runtime.createBuffer(count * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        for (const auto& resource : {seed, a, b, output}) {
            void* data = nullptr;
            testutil::checkHR(resource->Map(0, nullptr, &data), "Initialize buffer");
            for (size_t index = 0; index < count; ++index)
                static_cast<float*>(data)[index] = resource.Get() == b.Get() ? 2 * index + 1 :
                    resource.Get() == seed.Get() ? index : -17;
            resource->Unmap(0, nullptr);
        }
        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0] = {D3D12_DESCRIPTOR_RANGE_TYPE_SRV, 2, 0, 0, 0};
        ranges[1] = {D3D12_DESCRIPTOR_RANGE_TYPE_UAV, 1, 0, 0, 2};
        D3D12_ROOT_PARAMETER parameters[4]{};
        const bool rootViews = mode == "root";
        if (rootViews) {
            for (UINT index = 0; index < 3; ++index) {
                parameters[index].ParameterType = index == 2 ? D3D12_ROOT_PARAMETER_TYPE_UAV : D3D12_ROOT_PARAMETER_TYPE_SRV;
                parameters[index].Descriptor.ShaderRegister = index == 2 ? 0 : index;
            }
        } else {
            parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[0].DescriptorTable = {2, ranges};
        }
        const UINT constants = rootViews ? 3 : 1;
        parameters[constants].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[constants].Constants = {0, 0, 3};
        D3D12_ROOT_SIGNATURE_DESC signatureDesc{};
        signatureDesc.NumParameters = constants + 1;
        signatureDesc.pParameters = parameters;
        ComPtr<ID3DBlob> rootBlob;
#if defined(NACELLE_PROFILE_SHARED_RUNTIME)
        testutil::checkHR(D3D12SerializeRootSignature(&signatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0,
            &rootBlob, nullptr), "Serialize shared root");
#else
        testutil::checkHR(NacelleSerializeRootSignature(&signatureDesc, D3D_ROOT_SIGNATURE_VERSION_1_0,
            &rootBlob, nullptr), "Serialize root");
#endif
        ComPtr<ID3D12RootSignature> signature;
        testutil::checkHR(runtime.device->CreateRootSignature(0, rootBlob->GetBufferPointer(), rootBlob->GetBufferSize(),
            IID_PPV_ARGS(&signature)), "Root signature");
        const bool debug = mode == "source" || mode == "source-denied";
        const auto shader = testutil::loadShaderBytes(debug ? "vec_add_debug.dxil" :
            mode == "raw" ? "vec_add_raw.dxil" : mode == "typed" ? "vec_add_typed.dxil" :
            mode == "group" ? "vec_add_group.dxil" : mode == "atomic" ? "vec_add_atomic.dxil" : "vec_add_volume.dxil");
        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
        psoDesc.pRootSignature = signature.Get();
        psoDesc.CS = {shader.data(), shader.size()};
        ComPtr<ID3D12PipelineState> pipeline;
        testutil::checkHR(runtime.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pipeline)), "Compute pipeline");
        testutil::checkHR(NacelleWaitForReplayCompilation(), "Drain existing Tier-2 compilation");
        D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
        heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        heapDesc.NumDescriptors = 6;
        ComPtr<ID3D12DescriptorHeap> heap;
        testutil::checkHR(runtime.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&heap)), "Descriptor heap");
        const auto stride = runtime.device->GetDescriptorHandleIncrementSize(heapDesc.Type);
        auto destination = heap->GetCPUDescriptorHandleForHeapStart();
        for (const auto& resource : {a, b}) {
            D3D12_SHADER_RESOURCE_VIEW_DESC view{};
            view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
            view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            view.Buffer.NumElements = count;
            view.Buffer.StructureByteStride = 4;
            if (mode == "raw" || mode == "typed") {
                view.Buffer.StructureByteStride = 0;
                view.Format = mode == "raw" ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_R32_FLOAT;
                view.Buffer.Flags = mode == "raw" ? D3D12_BUFFER_SRV_FLAG_RAW : D3D12_BUFFER_SRV_FLAG_NONE;
            }
            runtime.device->CreateShaderResourceView(resource.Get(), &view, destination);
            destination.ptr += stride;
        }
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = count;
        uav.Buffer.StructureByteStride = 4;
        if (mode == "raw" || mode == "typed") {
            uav.Buffer.StructureByteStride = 0;
            uav.Format = mode == "raw" ? DXGI_FORMAT_R32_TYPELESS : DXGI_FORMAT_R32_FLOAT;
            uav.Buffer.Flags = mode == "raw" ? D3D12_BUFFER_UAV_FLAG_RAW : D3D12_BUFFER_UAV_FLAG_NONE;
        }
        runtime.device->CreateUnorderedAccessView(output.Get(), nullptr, &uav, destination);
        destination.ptr += stride;
        runtime.device->CopyDescriptorsSimple(3, destination, heap->GetCPUDescriptorHandleForHeapStart(), heapDesc.Type);
        auto table = heap->GetGPUDescriptorHandleForHeapStart();
        table.ptr += 3 * stride;
        NacelleReplayResource declarations[]{{seed.Get(), 1}, {a.Get(), 3}, {b.Get(), 1}, {output.Get(), 3}};
        NacelleReplayCaptureOptions options;
        options.contracts = NACELLE_REPLAY_RACE_FREE_INPUTS | NACELLE_REPLAY_HOST_QUIESCENT;
        options.resources = declarations;
        options.resource_count = 4;
        NacelleReplayCapture* capture = nullptr;
        uint64_t beginNs = 0, endNs = 0, replayNs = 0;
        const auto elapsed = [](auto start) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count();
        };
        if (mode == "mapped") {
            void* data = nullptr;
            testutil::checkHR(seed->Map(0, nullptr, &data), "Hold input map");
            require(NacelleBeginReplayCapture(&options, &capture) == E_INVALIDARG && !capture, "Reject open map");
            seed->Unmap(0, nullptr);
        } else {
            if (mode != "disabled") {
                const auto start = std::chrono::steady_clock::now();
                testutil::checkHR(NacelleBeginReplayCapture(&options, &capture), "Begin replay capture");
                beginNs = elapsed(start);
            }
            for (UINT iteration = 0; iteration < 2; ++iteration) {
                runtime.resetCmd();
                runtime.cmdList->SetPipelineState(pipeline.Get());
                runtime.cmdList->SetComputeRootSignature(signature.Get());
                ID3D12DescriptorHeap* heaps[]{heap.Get()};
                runtime.cmdList->SetDescriptorHeaps(1, heaps);
                if (rootViews) {
                    runtime.cmdList->SetComputeRootShaderResourceView(0, a->GetGPUVirtualAddress());
                    runtime.cmdList->SetComputeRootShaderResourceView(1, b->GetGPUVirtualAddress());
                    runtime.cmdList->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
                } else runtime.cmdList->SetComputeRootDescriptorTable(0, table);
                struct Parameters { float scale, bias; UINT count; } values{float(iteration + 2), float(iteration + 3), count};
                runtime.cmdList->SetComputeRoot32BitConstants(constants, 3, &values, 0);
                if (!iteration) {
                    runtime.cmdList->CopyBufferRegion(a.Get(), 0, seed.Get(), 0, count * 4);
                    UINT clear[4]{};
                    auto gpu = table; gpu.ptr += 2 * stride;
                    auto cpu = destination; cpu.ptr += 2 * stride;
                    runtime.cmdList->ClearUnorderedAccessViewUint(gpu, cpu, output.Get(), clear, 0, nullptr);
                }
                D3D12_RESOURCE_BARRIER barrier{};
                barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrier.UAV.pResource = output.Get();
                runtime.cmdList->ResourceBarrier(1, &barrier);
                if (mode == "unsupported") runtime.cmdList->BeginQuery(nullptr, D3D12_QUERY_TYPE_OCCLUSION, 0);
                const bool linear = debug || mode == "group" || mode == "atomic" || mode == "raw" || mode == "typed";
                runtime.cmdList->Dispatch(linear ? 8 : 4, linear ? 1 : 2, linear ? 1 : 2);
                testutil::checkHR(runtime.cmdList->Close(), "Close captured list");
                ID3D12CommandList* lists[]{runtime.cmdList.Get()};
                runtime.queue->ExecuteCommandLists(1, lists);
            }
            const bool rejected = mode == "unsupported" || mode == "source-denied" || mode == "atomic";
            if (capture) {
                const auto start = std::chrono::steady_clock::now();
                const auto result = NacelleEndReplayCapture(capture, path.c_str());
                endNs = elapsed(start);
                require(rejected ? FAILED(result) && !std::filesystem::exists(path) : result == S_OK, "Capture result/output contract");
            }
            NacelleDestroyReplayCapture(capture);
            void* data = nullptr;
            testutil::checkHR(output->Map(0, nullptr, &data), "Check unchanged shader execution");
            for (size_t index = 0; index < count; ++index) {
                const float expected = mode == "group" ?
                    3 * (index / 64 * 64 + 63 - index % 64) + 2 * index + 5 : 5 * index + 5;
                require(static_cast<float*>(data)[index] == expected, "Capture must preserve shader output");
            }
            output->Unmap(0, nullptr);
            if (!rejected && mode != "disabled" && mode != "capture-only") {
                testutil::checkHR(NacelleReplayBundle(path.c_str(), 0, 3), "Non-executing validation");
                const auto start = std::chrono::steady_clock::now();
                testutil::checkHR(NacelleReplayBundle(path.c_str(), 1, 3), "Fresh-state byte-exact repeated replay");
                replayNs = elapsed(start);
                nacelle::replay::Bundle bundle;
                std::string error;
                require(nacelle::replay::readBundle(path, bundle, error), "Read produced replay");
                auto bad = bundle;
                bad.bytes[0] ^= 1;
                require(!nacelle::replay::validateBundle(bad, error), "Reject corrupt snapshot");
                bad = bundle;
                bad.metadata.resources[0].initialBlob = 0;
                require(!nacelle::replay::validateBundle(bad, error), "Reject missing snapshot");
                const auto rejectExecution = [&](nacelle::replay::Bundle altered, const char* message) {
                    const auto changed = path + ".changed";
                    std::filesystem::remove(changed);
                    require(nacelle::replay::writeBundle(changed, altered, error), error.c_str());
                    require(FAILED(NacelleReplayBundle(changed.c_str(), 1, 3)), message);
                    std::filesystem::remove(changed);
                };
                bad = bundle;
                bad.manifest.environment.simdWidth = bad.manifest.environment.simdWidth == 4 ? 8 : 4;
                rejectExecution(bad, "Reject incompatible SIMD before execution");
                bad = bundle;
                bad.metadata.pipelines[0].nativeId = std::string(64, '0');
                rejectExecution(bad, "Reject native code identity mismatch before entry invocation");
                bad = bundle;
                const auto expectedBlob = bad.metadata.resources.back().expectedBlob;
                const auto& expectedMetadata = bad.metadata.blobs[expectedBlob - 1];
                bad.bytes[expectedMetadata.offset] ^= 1;
                bad.metadata.blobs[expectedBlob - 1].digest =
                    nacelle::replay::contentDigest(nacelle::replay::blobBytes(bad, expectedBlob));
                rejectExecution(bad, "Reject byte-different expected output");
                bad = bundle;
                for (auto& command : bad.metadata.commands)
                    if (command.kind == "constants") { command.data[0] = 63; break; }
                rejectExecution(bad, "Reject invalid semantic root index before execution");
            } else if (mode == "capture-only")
                testutil::checkHR(NacelleReplayBundle(path.c_str(), 0, 3), "Validate capture without replay cost");
        }
        configuration.flags = 0;
        testutil::checkHR(NacelleConfigureReplay(&configuration), "Release replay provenance");
        std::cout << "replay_metrics begin_ns=" << beginNs << " end_ns=" << endNs << " replay_ns=" << replayNs
            << " file_bytes=" << (std::filesystem::exists(path) ? std::filesystem::file_size(path) : 0) << '\n';
        if (!std::getenv("NACELLE_REPLAY_KEEP_OUTPUT")) std::filesystem::remove(path);
        std::cout << "replay_runtime_test passed: " << mode << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "replay_runtime_test: " << error.what() << '\n';
        return 1;
    }
}
