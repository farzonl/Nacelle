#include <nacelle/nacelle.h>
#include <directx/d3dx12.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#if defined(NACELLE_HAS_SDL3)
#include <SDL3/SDL.h>
#endif

#if !defined(_WIN32)
#include <wsl/wrladapter.h>
using Microsoft::WRL::ComPtr;
#else
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

namespace {

struct VertexPosCol {
    float x, y, z;
    float r, g, b, a;
};

struct PresentState {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> lastFrame;
    uint32_t framesPresented = 0;
};

void presentCallback(
    const void* pixels,
    uint32_t width,
    uint32_t height,
    uint32_t rowPitch,
    DXGI_FORMAT /*format*/,
    void* userData) {
    auto* st = static_cast<PresentState*>(userData);
    st->width = width;
    st->height = height;
    st->lastFrame.resize(static_cast<size_t>(width) * height * 4);
    const auto* src = static_cast<const uint8_t*>(pixels);
    for (uint32_t y = 0; y < height; ++y) {
        std::memcpy(st->lastFrame.data() + static_cast<size_t>(y) * width * 4,
                    src + static_cast<size_t>(y) * rowPitch,
                    static_cast<size_t>(width) * 4);
    }
    ++st->framesPresented;
}

std::vector<uint8_t> loadShader(const char* name) {
    std::string path = std::string(NACELLE_SHADER_BIN_DIR) + "/" + name;
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return {};
    auto sz = ifs.tellg();
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (sz > 0) ifs.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

void writePPM(const std::string& path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (uint32_t i = 0; i < w * h; ++i) {
        ofs.write(reinterpret_cast<const char*>(rgba + i * 4), 3);
    }
}

} // namespace

int main(int argc, char** argv) {
    bool headless = false;
    uint32_t maxFrames = 0;
    std::string dumpPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless") {
            headless = true;
        } else if (arg == "--frames" && i + 1 < argc) {
            maxFrames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--dump" && i + 1 < argc) {
            dumpPath = argv[++i];
        }
    }

    constexpr uint32_t kW = 640;
    constexpr uint32_t kH = 480;

    ComPtr<ID3D12Device> device;
    if (FAILED(NacelleCreateDevice(nullptr, D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&device)))) {
        std::fprintf(stderr, "Failed to create Nacelle device\n");
        return 1;
    }

    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));

    ComPtr<ID3D12CommandAllocator> alloc;
    device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));

    ComPtr<ID3D12GraphicsCommandList> cmd;
    device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&cmd));
    cmd->Close();

    ComPtr<ID3D12Fence> fence;
    device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));

    PresentState presentState;
    ComPtr<INacelleSwapChain> swapChain;
    if (FAILED(NacelleCreateSwapChain(queue.Get(), kW, kH, 2, DXGI_FORMAT_R8G8B8A8_UNORM,
                                      presentCallback, &presentState, &swapChain))) {
        std::fprintf(stderr, "Failed to create Nacelle swapchain\n");
        return 1;
    }

    D3D12_DESCRIPTOR_HEAP_DESC rtvHD{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    ComPtr<ID3D12DescriptorHeap> rtvHeap;
    device->CreateDescriptorHeap(&rtvHD, IID_PPV_ARGS(&rtvHeap));
    UINT rtvInc = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
    D3D12_CPU_DESCRIPTOR_HANDLE rtvHandles[2]{};
    for (UINT b = 0; b < 2; ++b) {
        ComPtr<ID3D12Resource> bb;
        swapChain->GetBuffer(b, IID_PPV_ARGS(&bb));
        rtvHandles[b] = {rtvHeap->GetCPUDescriptorHandleForHeapStart().ptr + b * rtvInc};
        device->CreateRenderTargetView(bb.Get(), nullptr, rtvHandles[b]);
    }

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    ComPtr<ID3DBlob> rsBlob;
    NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr);
    ComPtr<ID3D12RootSignature> rs;
    device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rs));

    auto vsBytes = loadShader("raster_basic_vs.dxil");
    auto psBytes = loadShader("raster_basic_ps.dxil");
    if (vsBytes.empty() || psBytes.empty()) {
        std::fprintf(stderr, "Failed to load viewer shaders from %s\n", NACELLE_SHADER_BIN_DIR);
        return 1;
    }

    D3D12_INPUT_ELEMENT_DESC layout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.VS = {vsBytes.data(), vsBytes.size()};
    psoDesc.PS = {psBytes.data(), psBytes.size()};
    psoDesc.InputLayout = {layout, 2};
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleMask = UINT32_MAX;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

    ComPtr<ID3D12PipelineState> pso;
    device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso));

    D3D12_HEAP_PROPERTIES upHP{D3D12_HEAP_TYPE_UPLOAD};
    D3D12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(sizeof(VertexPosCol) * 3);
    ComPtr<ID3D12Resource> vb;
    device->CreateCommittedResource(&upHP, D3D12_HEAP_FLAG_NONE, &vbDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&vb));

#if defined(NACELLE_HAS_SDL3)
    SDL_Window* win = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* tex = nullptr;
    if (!headless) {
        if (SDL_Init(SDL_INIT_VIDEO)) {
            win = SDL_CreateWindow("Nacelle D3D12 Viewer", kW, kH, 0);
            renderer = SDL_CreateRenderer(win, nullptr);
            tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, kW, kH);
        }
    }
#endif

    UINT64 fenceVal = 0;
    uint32_t frameIdx = 0;
    bool running = true;

    while (running) {
#if defined(NACELLE_HAS_SDL3)
        if (!headless && win) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_EVENT_QUIT) running = false;
            }
        }
#endif
        float angle = static_cast<float>(frameIdx) * 0.15f;
        float c = std::cos(angle);
        float s = std::sin(angle);
        auto rot = [&](float x, float y) -> std::pair<float, float> {
            return {x * c - y * s, x * s + y * c};
        };
        auto [x0, y0] = rot(0.0f, 0.7f);
        auto [x1, y1] = rot(0.65f, -0.6f);
        auto [x2, y2] = rot(-0.65f, -0.6f);

        VertexPosCol verts[3] = {
            {x0, y0, 0.5f, 1.0f, 0.25f, 0.2f, 1.0f},
            {x1, y1, 0.5f, 0.2f, 0.95f, 0.3f, 1.0f},
            {x2, y2, 0.5f, 0.2f, 0.4f, 1.0f, 1.0f},
        };
        void* vbPtr = nullptr;
        vb->Map(0, nullptr, &vbPtr);
        std::memcpy(vbPtr, verts, sizeof(verts));
        vb->Unmap(0, nullptr);

        uint32_t bbIdx = swapChain->GetCurrentBackBufferIndex();
        alloc->Reset();
        cmd->Reset(alloc.Get(), pso.Get());

        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(kW), static_cast<float>(kH), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(kW), static_cast<LONG>(kH)};
        cmd->SetGraphicsRootSignature(rs.Get());
        cmd->RSSetViewports(1, &vp);
        cmd->RSSetScissorRects(1, &sc);
        cmd->OMSetRenderTargets(1, &rtvHandles[bbIdx], FALSE, nullptr);
        const float clearCol[4] = {0.06f, 0.08f, 0.12f, 1.0f};
        cmd->ClearRenderTargetView(rtvHandles[bbIdx], clearCol, 0, nullptr);

        D3D12_VERTEX_BUFFER_VIEW vbv{vb->GetGPUVirtualAddress(), sizeof(verts), sizeof(VertexPosCol)};
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, 1, &vbv);
        cmd->DrawInstanced(3, 1, 0, 0);
        cmd->Close();

        ID3D12CommandList* lists[] = {cmd.Get()};
        queue->ExecuteCommandLists(1, lists);
        ++fenceVal;
        queue->Signal(fence.Get(), fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        swapChain->Present(0, 0);

#if defined(NACELLE_HAS_SDL3)
        if (!headless && renderer && tex && !presentState.lastFrame.empty()) {
            SDL_UpdateTexture(tex, nullptr, presentState.lastFrame.data(), kW * 4);
            SDL_RenderClear(renderer);
            SDL_RenderTexture(renderer, tex, nullptr, nullptr);
            SDL_RenderPresent(renderer);
        }
#endif

        ++frameIdx;
        if (maxFrames > 0 && frameIdx >= maxFrames) running = false;
        if (headless && maxFrames == 0) running = false;
    }

    if (!dumpPath.empty() && !presentState.lastFrame.empty()) {
        writePPM(dumpPath, presentState.lastFrame.data(), presentState.width, presentState.height);
        std::printf("Wrote viewer frame to %s (%ux%u, frames=%u)\n",
                    dumpPath.c_str(), presentState.width, presentState.height, presentState.framesPresented);
    }

#if defined(NACELLE_HAS_SDL3)
    if (tex) SDL_DestroyTexture(tex);
    if (renderer) SDL_DestroyRenderer(renderer);
    if (win) SDL_DestroyWindow(win);
    if (!headless) SDL_Quit();
#endif

    return 0;
}
