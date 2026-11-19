#include "test_common.h"
#include <nacelle/instrumentation.h>
#include "instrumentation/profile_format.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ComPtr<ID3D12RootSignature> createSignature(testutil::TestRuntime& runtime,
                                           const D3D12_ROOT_SIGNATURE_DESC& desc) {
    ComPtr<ID3DBlob> blob;
    testutil::checkHR(NacelleSerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                                &blob, nullptr), "Serialize signature");
    ComPtr<ID3D12RootSignature> signature;
    testutil::checkHR(runtime.device->CreateRootSignature(0, blob->GetBufferPointer(),
        blob->GetBufferSize(), IID_PPV_ARGS(&signature)), "Create signature");
    return signature;
}

void initialize(testutil::TestRuntime& runtime) {
    testutil::checkHR(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0,
                                      IID_PPV_ARGS(&runtime.device)), "Create device");
    D3D12_COMMAND_QUEUE_DESC queueDesc{};
    testutil::checkHR(runtime.device->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&runtime.queue)), "Queue");
    testutil::checkHR(runtime.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(&runtime.alloc)), "Allocator");
    testutil::checkHR(runtime.device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
        runtime.alloc.Get(), nullptr, IID_PPV_ARGS(&runtime.cmdList)), "Command list");
    testutil::checkHR(runtime.cmdList->Close(), "Close");
    testutil::checkHR(runtime.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&runtime.fence)), "Fence");
}

void compute(testutil::TestRuntime& runtime) {
    D3D12_ROOT_PARAMETER parameters[4]{};
    for (UINT i = 0; i < 3; ++i) {
        parameters[i].ParameterType = i == 2 ? D3D12_ROOT_PARAMETER_TYPE_UAV : D3D12_ROOT_PARAMETER_TYPE_SRV;
        parameters[i].Descriptor.ShaderRegister = i == 2 ? 0 : i;
    }
    parameters[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[3].Constants.ShaderRegister = 0;
    parameters[3].Constants.Num32BitValues = 3;
    D3D12_ROOT_SIGNATURE_DESC signatureDesc{};
    signatureDesc.NumParameters = 4;
    signatureDesc.pParameters = parameters;
    auto signature = createSignature(runtime, signatureDesc);
    auto shader = testutil::loadShaderBytes("vec_add.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(runtime.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), "Compute PSO");
    // A second creation exercises the same capture's cache metadata.
    ComPtr<ID3D12PipelineState> cached;
    testutil::checkHR(runtime.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&cached)), "Cached compute PSO");
    std::array<float, 1024> a{}, b{};
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(2 * i + 1);
    }
    auto inputA = runtime.createBuffer(sizeof(a), D3D12_HEAP_TYPE_DEFAULT);
    auto inputB = runtime.createBuffer(sizeof(b), D3D12_HEAP_TYPE_DEFAULT);
    auto output = runtime.createBuffer(sizeof(a), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    runtime.uploadBuffer(inputA.Get(), a.data(), sizeof(a));
    runtime.uploadBuffer(inputB.Get(), b.data(), sizeof(b));
    runtime.resetCmd();
    runtime.cmdList->SetComputeRootSignature(signature.Get());
    runtime.cmdList->SetPipelineState(pso.Get());
    runtime.cmdList->SetComputeRootShaderResourceView(0, inputA->GetGPUVirtualAddress());
    runtime.cmdList->SetComputeRootShaderResourceView(1, inputB->GetGPUVirtualAddress());
    runtime.cmdList->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
    const float scale = 1.0f;
    uint32_t constants[3]{0, 0, static_cast<uint32_t>(a.size())};
    std::memcpy(&constants[0], &scale, sizeof(scale));
    runtime.cmdList->SetComputeRoot32BitConstants(3, 3, constants, 0);
    runtime.cmdList->Dispatch(16, 1, 1);
    runtime.executeAndWait();
    const auto bytes = runtime.readbackBuffer(output.Get(), sizeof(a));
    std::array<float, 1024> actual{};
    std::memcpy(actual.data(), bytes.data(), bytes.size());
    for (size_t i = 0; i < actual.size(); ++i)
        require(actual[i] == a[i] + b[i], "Capture changed compute output");
}

void draw(testutil::TestRuntime& runtime) {
    D3D12_ROOT_SIGNATURE_DESC signatureDesc{};
    auto signature = createSignature(runtime, signatureDesc);
    auto vs = testutil::loadShaderBytes("raster_basic_vs.dxil");
    auto ps = testutil::loadShaderBytes("raster_basic_ps.dxil");
    const D3D12_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.VS = {vs.data(), vs.size()};
    desc.PS = {ps.data(), ps.size()};
    desc.InputLayout = {layout, 2};
    desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    desc.NumRenderTargets = 1;
    desc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleMask = UINT32_MAX;
    desc.SampleDesc.Count = 1;
    desc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    desc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    desc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    desc.DepthStencilState.DepthEnable = FALSE;
    desc.DepthStencilState.StencilEnable = FALSE;
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(runtime.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pso)), "Graphics PSO");
    const float vertices[][7] = {
        {-1, -1, 0.5f, 1, 0, 0, 1}, {-1, 3, 0.5f, 1, 0, 0, 1}, {3, -1, 0.5f, 1, 0, 0, 1},
    };
    auto vertexBuffer = runtime.createBuffer(sizeof(vertices), D3D12_HEAP_TYPE_DEFAULT);
    runtime.uploadBuffer(vertexBuffer.Get(), vertices, sizeof(vertices));
    constexpr UINT width = 128;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    auto textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, width, width,
        1, 1, 1, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    ComPtr<ID3D12Resource> target;
    testutil::checkHR(runtime.device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &textureDesc,
        D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&target)), "Render target");
    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> descriptors;
    testutil::checkHR(runtime.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&descriptors)), "RTV heap");
    auto rtv = descriptors->GetCPUDescriptorHandleForHeapStart();
    runtime.device->CreateRenderTargetView(target.Get(), nullptr, rtv);
    D3D12_VERTEX_BUFFER_VIEW vb{vertexBuffer->GetGPUVirtualAddress(), sizeof(vertices), sizeof(vertices[0])};
    D3D12_VIEWPORT viewport{0, 0, width, width, 0, 1};
    D3D12_RECT scissor{0, 0, width, width};
    runtime.resetCmd();
    runtime.cmdList->SetGraphicsRootSignature(signature.Get());
    runtime.cmdList->SetPipelineState(pso.Get());
    runtime.cmdList->RSSetViewports(1, &viewport);
    runtime.cmdList->RSSetScissorRects(1, &scissor);
    runtime.cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    runtime.cmdList->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    runtime.cmdList->IASetVertexBuffers(0, 1, &vb);
    runtime.cmdList->DrawInstanced(3, 1, 0, 0);
    runtime.executeAndWait();
    auto readback = runtime.createBuffer(width * width * 4, D3D12_HEAP_TYPE_READBACK);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, width, width, 1, width * 4};
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = target.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    runtime.resetCmd();
    runtime.cmdList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    runtime.executeAndWait();
    void* data = nullptr;
    testutil::checkHR(readback->Map(0, nullptr, &data), "Map graphics result");
    const auto* pixels = static_cast<const uint8_t*>(data);
    const size_t center = (width / 2 * width + width / 2) * 4;
    const bool correct = pixels[center] == 255 && pixels[center + 1] == 0 &&
                         pixels[center + 2] == 0 && pixels[center + 3] == 255;
    readback->Unmap(0, nullptr);
    require(correct, "Capture changed graphics output");
}
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: profile_runtime_test <new-profile-path>\n";
        return 1;
    }
    NacelleProfileCapture* capture = nullptr;
    try {
        NacelleProfileOptions options;
        testutil::checkHR(NacelleBeginProfileCapture(&options, &capture), "Begin profile");
        testutil::TestRuntime runtime;
        initialize(runtime);
        compute(runtime);
        draw(runtime);
        NacelleProfileSummary summary;
        testutil::checkHR(NacelleEndProfileCapture(capture, argv[1], &summary), "End profile");
        NacelleDestroyProfileCapture(capture);
        capture = nullptr;
        nacelle::profile::Profile profile;
        std::string error;
        require(nacelle::profile::readProfile(argv[1], profile, error), error.c_str());
        for (const char* name : {"Compile compute PSO", "Compile graphics PSO",
                                 "Compute dispatch", "Graphics draw", "Raster worker", "Compute Tier 2"}) {
            require(std::any_of(profile.trace.traceEvents.begin(), profile.trace.traceEvents.end(),
                [&](const auto& event) { return event.name == name; }), name);
        }
        require(profile.manifest.complete && summary.recorded_events > 10, "Complete runtime capture");
        std::cout << "Compute output + graphics output + runtime profile spans: PASS ("
                  << summary.recorded_events << " events)\n";
        return 0;
    } catch (const std::exception& error) {
        NacelleDestroyProfileCapture(capture);
        std::cerr << "profile_runtime_test: " << error.what() << '\n';
        return 1;
    }
}
