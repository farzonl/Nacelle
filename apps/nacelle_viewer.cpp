#include <nacelle/nacelle.h>
#include "compiler/dxil_loader.h"
#include <directx/d3dx12.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <algorithm>
#include <chrono>
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

struct VertexPos {
    float x, y, z;
};

struct ShadertoyCB {
    float iResolution[3];
    float iTime;
    float iMouse[4];
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

std::vector<uint8_t> loadBinaryFile(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) return {};
    auto sz = ifs.tellg();
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    ifs.seekg(0, std::ios::beg);
    if (sz > 0) ifs.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

std::vector<uint8_t> loadShader(const char* name) {
    return loadBinaryFile(std::string(NACELLE_SHADER_BIN_DIR) + "/" + name);
}

void writePPM(const std::string& path, const uint8_t* rgba, uint32_t w, uint32_t h) {
    std::ofstream ofs(path, std::ios::binary);
    ofs << "P6\n" << w << " " << h << "\n255\n";
    for (uint32_t i = 0; i < w * h; ++i) {
        ofs.write(reinterpret_cast<const char*>(rgba + i * 4), 3);
    }
}

// Extract embedded RTS0 root signature chunk from a DXBC/DXIL container if present.
std::vector<uint8_t> extractRts0Chunk(const std::vector<uint8_t>& container) {
    if (container.size() < 32) return {};
    if (std::memcmp(container.data(), "DXBC", 4) != 0) return {};
    uint32_t partCount = 0;
    std::memcpy(&partCount, container.data() + 28, sizeof(partCount));
    if (32 + static_cast<size_t>(partCount) * 4 > container.size()) return {};
    for (uint32_t i = 0; i < partCount; ++i) {
        uint32_t offset = 0;
        std::memcpy(&offset, container.data() + 32 + i * 4, sizeof(offset));
        if (offset + 8 > container.size()) continue;
        if (std::memcmp(container.data() + offset, "RTS0", 4) == 0) {
            uint32_t size = 0;
            std::memcpy(&size, container.data() + offset + 4, sizeof(size));
            if (offset + 8 + static_cast<size_t>(size) <= container.size()) {
                return std::vector<uint8_t>(
                    container.begin() + offset + 8,
                    container.begin() + offset + 8 + size);
            }
        }
    }
    return {};
}

// Build a root signature defaulting to DXIL container metadata (RTS0 chunk first,
// then DxilLoadedShader::resources reflection).
bool createRootSignatureFromDxilMetadata(
    ID3D12Device* device,
    const std::vector<uint8_t>& dxilBytes,
    bool isGraphics,
    ComPtr<ID3D12RootSignature>& outRs,
    bool& outUsesCbv0) {
    outUsesCbv0 = false;

    // 1. Prefer embedded RTS0 root signature chunk in the DXIL container.
    auto rts0 = extractRts0Chunk(dxilBytes);
    if (!rts0.empty()) {
        if (SUCCEEDED(device->CreateRootSignature(0, rts0.data(), rts0.size(), IID_PPV_ARGS(&outRs)))) {
            // Inspect DXIL metadata to check whether b0 is bound.
            llvm::LLVMContext ctx;
            nacelle::DxilLoadedShader loaded;
            std::string err;
            D3D12_SHADER_BYTECODE bc{dxilBytes.data(), dxilBytes.size()};
            if (nacelle::loadAndUpgradeDXIL(bc, ctx, loaded, err)) {
                for (const auto& r : loaded.resources) {
                    if (r.resClass == 2 && r.lowerBound == 0) {
                        outUsesCbv0 = true;
                        break;
                    }
                }
            }
            return true;
        }
    }

    // 2. Otherwise synthesize a root signature from DXIL resource metadata.
    llvm::LLVMContext ctx;
    nacelle::DxilLoadedShader loaded;
    std::string err;
    std::vector<D3D12_ROOT_PARAMETER> params;
    D3D12_SHADER_BYTECODE bc{dxilBytes.data(), dxilBytes.size()};
    if (nacelle::loadAndUpgradeDXIL(bc, ctx, loaded, err)) {
        for (const auto& r : loaded.resources) {
            D3D12_ROOT_PARAMETER p{};
            p.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
            p.Descriptor.ShaderRegister = r.lowerBound;
            p.Descriptor.RegisterSpace = r.space;
            if (r.resClass == 2) {
                p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
                if (r.lowerBound == 0) outUsesCbv0 = true;
                params.push_back(p);
            } else if (r.resClass == 0) {
                p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
                params.push_back(p);
            } else if (r.resClass == 1) {
                p.ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
                params.push_back(p);
            }
        }
    }

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = static_cast<UINT>(params.size());
    rsDesc.pParameters = params.empty() ? nullptr : params.data();
    if (isGraphics) {
        rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT;
    }

    ComPtr<ID3DBlob> rsBlob;
    if (FAILED(NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr))) {
        return false;
    }
    return SUCCEEDED(device->CreateRootSignature(
        0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&outRs)));
}

} // namespace

int main(int argc, char** argv) {
    bool headless = false;
    uint32_t maxFrames = 0;
    uint32_t kW = 640;
    uint32_t kH = 480;
    std::string dumpPath;
    std::string psPath;
    std::string vsPath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--headless" || arg == "-headless") {
            headless = true;
        } else if ((arg == "--frames" || arg == "-frames") && i + 1 < argc) {
            maxFrames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if ((arg == "--dump" || arg == "-dump") && i + 1 < argc) {
            dumpPath = argv[++i];
        } else if ((arg == "--ps" || arg == "-ps") && i + 1 < argc) {
            psPath = argv[++i];
        } else if ((arg == "--vs" || arg == "-vs") && i + 1 < argc) {
            vsPath = argv[++i];
        } else if ((arg == "--width" || arg == "-width") && i + 1 < argc) {
            kW = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if ((arg == "--height" || arg == "-height") && i + 1 < argc) {
            kH = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else {
            std::fprintf(stderr, "Unknown or incomplete option: %s\n", arg.c_str());
            return 1;
        }
    }

#if !defined(NACELLE_HAS_SDL3)
    if (!headless) {
        std::fprintf(stderr, "nacelle_viewer was built without SDL3; pass --headless (and optionally --dump <file.ppm>)\n");
        return 1;
    }
#endif

    const bool customPsMode = !psPath.empty();

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

    std::vector<uint8_t> vsBytes = vsPath.empty()
        ? loadShader(customPsMode ? "fullscreen_quad_vs.dxil" : "raster_basic_vs.dxil")
        : loadBinaryFile(vsPath);
    std::vector<uint8_t> psBytes = psPath.empty()
        ? loadShader("raster_basic_ps.dxil")
        : loadBinaryFile(psPath);

    if (vsBytes.empty() || psBytes.empty()) {
        std::fprintf(stderr, "Failed to load shaders (vs=%s, ps=%s)\n",
                     vsPath.empty() ? NACELLE_SHADER_BIN_DIR : vsPath.c_str(),
                     psPath.empty() ? NACELLE_SHADER_BIN_DIR : psPath.c_str());
        return 1;
    }

    ComPtr<ID3D12RootSignature> rs;
    bool usesCbv0 = false;
    if (!createRootSignatureFromDxilMetadata(device.Get(), psBytes, true, rs, usesCbv0)) {
        std::fprintf(stderr, "Failed to create root signature from DXIL metadata\n");
        return 1;
    }

    D3D12_INPUT_ELEMENT_DESC triLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
        {"COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };
    D3D12_INPUT_ELEMENT_DESC quadLayout[] = {
        {"POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0},
    };

    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.VS = {vsBytes.data(), vsBytes.size()};
    psoDesc.PS = {psBytes.data(), psBytes.size()};
    psoDesc.InputLayout = customPsMode
        ? D3D12_INPUT_LAYOUT_DESC{quadLayout, 1}
        : D3D12_INPUT_LAYOUT_DESC{triLayout, 2};
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    psoDesc.SampleMask = UINT32_MAX;
    psoDesc.SampleDesc.Count = 1;
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.RasterizerState.CullMode = D3D12_CULL_MODE_NONE;
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);

    ComPtr<ID3D12PipelineState> pso;
    if (FAILED(device->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&pso))) || !pso) {
        std::fprintf(stderr, "Failed to create graphics PSO\n");
        return 1;
    }

    D3D12_HEAP_PROPERTIES upHP{D3D12_HEAP_TYPE_UPLOAD};
    const UINT vbSize = customPsMode ? sizeof(VertexPos) * 6 : sizeof(VertexPosCol) * 3;
    D3D12_RESOURCE_DESC vbDesc = CD3DX12_RESOURCE_DESC::Buffer(vbSize);
    ComPtr<ID3D12Resource> vb;
    device->CreateCommittedResource(&upHP, D3D12_HEAP_FLAG_NONE, &vbDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&vb));

    if (customPsMode) {
        VertexPos quadVerts[6] = {
            {-1.0f, -1.0f, 0.5f},
            { 1.0f, -1.0f, 0.5f},
            { 1.0f,  1.0f, 0.5f},
            {-1.0f, -1.0f, 0.5f},
            { 1.0f,  1.0f, 0.5f},
            {-1.0f,  1.0f, 0.5f},
        };
        void* vbPtr = nullptr;
        vb->Map(0, nullptr, &vbPtr);
        std::memcpy(vbPtr, quadVerts, sizeof(quadVerts));
        vb->Unmap(0, nullptr);
    }

    ComPtr<ID3D12Resource> cb;
    if (usesCbv0) {
        D3D12_RESOURCE_DESC cbDesc = CD3DX12_RESOURCE_DESC::Buffer(256);
        device->CreateCommittedResource(&upHP, D3D12_HEAP_FLAG_NONE, &cbDesc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&cb));
    }

#if defined(NACELLE_HAS_SDL3)
    SDL_Window* win = nullptr;
    SDL_Renderer* renderer = nullptr;
    SDL_Texture* tex = nullptr;
    if (!headless) {
        if (!SDL_Init(SDL_INIT_VIDEO)) {
            std::fprintf(stderr, "SDL_Init(SDL_INIT_VIDEO) failed: %s\n", SDL_GetError());
            return 1;
        }
        win = SDL_CreateWindow("Nacelle D3D12 Viewer", static_cast<int>(kW), static_cast<int>(kH), 0);
        if (!win) {
            std::fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
            SDL_Quit();
            return 1;
        }
        renderer = SDL_CreateRenderer(win, nullptr);
        if (!renderer) {
            std::fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
            SDL_DestroyWindow(win);
            SDL_Quit();
            return 1;
        }
        SDL_SetRenderVSync(renderer, 1);
        tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_STREAMING, static_cast<int>(kW), static_cast<int>(kH));
    }
#endif

    UINT64 fenceVal = 0;
    uint32_t frameIdx = 0;
    bool running = true;
    const auto startTime = std::chrono::steady_clock::now();
    float mouseX = 0.0f, mouseY = 0.0f;
    bool mouseDown = false;

    while (running) {
#if defined(NACELLE_HAS_SDL3)
        if (!headless && win) {
            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_EVENT_QUIT) running = false;
                if (ev.type == SDL_EVENT_MOUSE_MOTION) {
                    mouseX = ev.motion.x;
                    mouseY = static_cast<float>(kH) - ev.motion.y;
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN) {
                    mouseDown = true;
                    mouseX = ev.button.x;
                    mouseY = static_cast<float>(kH) - ev.button.y;
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                    mouseDown = false;
                }
            }
        }
#endif
        float timeSec = headless
            ? static_cast<float>(frameIdx) * (1.0f / 60.0f)
            : std::chrono::duration<float>(std::chrono::steady_clock::now() - startTime).count();

        if (customPsMode) {
            if (usesCbv0 && cb) {
                ShadertoyCB cbData{};
                cbData.iResolution[0] = static_cast<float>(kW);
                cbData.iResolution[1] = static_cast<float>(kH);
                cbData.iResolution[2] = 1.0f;
                cbData.iTime = timeSec;
                cbData.iMouse[0] = mouseX;
                cbData.iMouse[1] = mouseY;
                cbData.iMouse[2] = mouseDown ? mouseX : -mouseX;
                cbData.iMouse[3] = mouseDown ? mouseY : -mouseY;
                void* cbPtr = nullptr;
                cb->Map(0, nullptr, &cbPtr);
                std::memcpy(cbPtr, &cbData, sizeof(cbData));
                cb->Unmap(0, nullptr);
            }
        } else {
            float angle = headless
                ? static_cast<float>(frameIdx) * 0.15f
                : timeSec * 1.2f;
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
        }

        uint32_t bbIdx = swapChain->GetCurrentBackBufferIndex();
        alloc->Reset();
        cmd->Reset(alloc.Get(), pso.Get());

        D3D12_VIEWPORT vp{0.0f, 0.0f, static_cast<float>(kW), static_cast<float>(kH), 0.0f, 1.0f};
        D3D12_RECT sc{0, 0, static_cast<LONG>(kW), static_cast<LONG>(kH)};
        cmd->SetGraphicsRootSignature(rs.Get());
        if (usesCbv0 && cb) {
            cmd->SetGraphicsRootConstantBufferView(0, cb->GetGPUVirtualAddress());
        }
        cmd->RSSetViewports(1, &vp);
        cmd->RSSetScissorRects(1, &sc);
        cmd->OMSetRenderTargets(1, &rtvHandles[bbIdx], FALSE, nullptr);
        const float clearCol[4] = {0.06f, 0.08f, 0.12f, 1.0f};
        cmd->ClearRenderTargetView(rtvHandles[bbIdx], clearCol, 0, nullptr);

        D3D12_VERTEX_BUFFER_VIEW vbv{
            vb->GetGPUVirtualAddress(),
            vbSize,
            customPsMode ? static_cast<UINT>(sizeof(VertexPos)) : static_cast<UINT>(sizeof(VertexPosCol))};
        cmd->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        cmd->IASetVertexBuffers(0, 1, &vbv);
        cmd->DrawInstanced(customPsMode ? 6 : 3, 1, 0, 0);
        cmd->Close();

        ID3D12CommandList* lists[] = {cmd.Get()};
        queue->ExecuteCommandLists(1, lists);
        ++fenceVal;
        queue->Signal(fence.Get(), fenceVal);
        while (fence->GetCompletedValue() < fenceVal) {}
        swapChain->Present(0, 0);

#if defined(NACELLE_HAS_SDL3)
        if (!headless && renderer && tex && !presentState.lastFrame.empty()) {
            SDL_UpdateTexture(tex, nullptr, presentState.lastFrame.data(), static_cast<int>(kW * 4));
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
