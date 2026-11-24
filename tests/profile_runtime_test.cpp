#include "test_common.h"
#include <nacelle/instrumentation.h>
#include <nacelle/native_debug.h>
#include <nacelle/source_debug.h>
#include <nacelle/replay.h>
#include "api/pipeline.h"
#include "instrumentation/profile_format.h"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace {
void require(bool condition, const char* message);
uint64_t sourceSession = 0;
uint32_t optimizedSourceTier = 2;
bool optimizedSourceMode() { return std::getenv("NACELLE_TEST_SOURCE_OPTIMIZED_MODE") != nullptr; }
bool nativeSourceCost() {
    const auto* cost = std::getenv("NACELLE_TEST_SOURCE_COST");
    return cost && std::string_view(cost) == "native";
}
bool ordinarySourceCost() {
    const auto* cost = std::getenv("NACELLE_TEST_SOURCE_COST");
    return cost && std::string_view(cost) == "ordinary";
}
void armSourceDebug(uint32_t stage) {
    if (nativeSourceCost() || ordinarySourceCost()) return;
    if (std::getenv("NACELLE_TEST_SOURCE_HELPER")) {
        bool mappedHelper = false;
        for (uint32_t id = 1; ; ++id) {
            NacelleSourceDebugSite site;
            const auto status = NacelleGetSourceDebugSite(id, &site);
            if (status == S_FALSE) break;
            require(status == S_OK, "Enumerate helper inline frames");
            mappedHelper |= site.inline_depth && std::string_view(site.frames[0].function) == "addOne";
        }
        require(mappedHelper, "Fixture has no real inlined-helper source mappings");
    }
    NacelleSourceDebugSite selected;
    uint32_t line = 0;
    for (uint32_t id = 1; ; ++id) {
        NacelleSourceDebugSite site;
        const auto status = NacelleGetSourceDebugSite(id, &site);
        if (status == S_FALSE) break;
        require(status == S_OK, "Enumerate source sites");
        if (site.stage != stage || (optimizedSourceMode() && !site.available)) continue;
        if (optimizedSourceMode()) {
            NacelleOptimizedSourceDebugSite detail;
            require(NacelleGetOptimizedSourceDebugSite(id, &detail) == S_OK, "Optimized site query");
            if (detail.tier != optimizedSourceTier) continue;
        }
        uint64_t size = 0;
        require(NacelleGetSourceDebugSource(site.source_id, nullptr, &size) == S_OK, "Source size");
        std::string source(size, '\0');
        require(NacelleGetSourceDebugSource(site.source_id, source.data(), &size) == S_OK, "Source bytes");
        const auto start = source.find(stage == NACELLE_SOURCE_COMPUTE ?
            (std::getenv("NACELLE_TEST_SOURCE_LOOP") ? "value += iteration" :
             (std::getenv("NACELLE_TEST_SOURCE_HELPER") ? "value -= 1.0; // SOURCE_HELPER" : "sharedValues[index] = value;")) :
            (optimizedSourceMode() ? "uint pixelY = uint(input.pos.y);" : "float restored = value;"));
        require(start != std::string::npos, "Fixture source breakpoint text");
        line = 1 + std::count(source.begin(), source.begin() + start, '\n');
        if (site.line == line) { selected = site; break; }
    }
    require(selected.id != 0, "Source breakpoint did not resolve");
    NacelleSourceDebugSelection selection;
    std::strcpy(selection.shader_id, selected.shader_id);
    selection.xyz[0] = std::getenv("NACELLE_TEST_SOURCE_MISSING") ? UINT32_MAX : 5;
    selection.xyz[1] = stage == NACELLE_SOURCE_PIXEL ? (optimizedSourceMode() ? 5 : 4) : 0;
    require(NacelleBeginSourceDebug(&selection, &sourceSession) == S_OK, "Begin source session");
    require(NacelleSourceDebugBreakpoint(sourceSession, selected.source_id, line) == S_OK, "Source breakpoint");
    std::cout << "SOURCE DEBUG READY session=" << sourceSession << " line=" << line <<
        " shader=" << selected.shader_id << std::endl;
}

void finishSourceDebug() {
    if (nativeSourceCost() || ordinarySourceCost()) return;
    if (std::getenv("NACELLE_TEST_SOURCE_MISSING") || std::getenv("NACELLE_TEST_SOURCE_ZERO")) {
        NacelleSourceDebugSnapshot missing;
        require(NacelleGetSourceDebugSnapshot(sourceSession, &missing) == S_FALSE, "Missing target fabricated stop");
        NacelleSourceDebugSummary summary;
        require(NacelleEndSourceDebug(sourceSession, &summary) == S_FALSE &&
            !summary.target_seen && !summary.stops, "Missing target looked successful");
        std::cout << "SOURCE DEBUG TARGET UNAVAILABLE PASS" << std::endl;
        return;
    }
    NacelleSourceDebugSnapshot snapshot;
    require(NacelleGetSourceDebugSnapshot(sourceSession, &snapshot) == S_OK, "Last source snapshot");
    require(snapshot.width >= 4 && snapshot.width <= 16 && snapshot.code.code_version && snapshot.code.object_sha256[0],
        "Source snapshot native identity");
    require(snapshot.selected_lane < snapshot.width &&
        snapshot.xyz[snapshot.selected_lane][0] == 5, "Selected lane mapping");
    if (optimizedSourceMode()) {
        NacelleOptimizedSourceDebugSnapshot detail;
        require(NacelleGetOptimizedSourceDebugSnapshot(sourceSession, &detail) == S_OK &&
            detail.tier == optimizedSourceTier && detail.entry_version == optimizedSourceTier && detail.compilation &&
            detail.snapshot.code.code_version == snapshot.code.code_version,
            "Source stop did not execute real promoted Tier-2 code");
        bool available = false;
        for (uint32_t local = 0; local < snapshot.site.local_count; ++local) {
            if (!snapshot.locals[local].kind) {
                require(!detail.valid_mask[local] && snapshot.locals[local].reason[0],
                    "Unavailable optimized value has no explicit validity/reason");
                continue;
            }
            require(detail.valid_mask[local] == snapshot.active_mask,
                "Optimized local validity does not match actual executing lanes");
            const auto name = std::string_view(snapshot.locals[local].name);
            if (snapshot.site.stage == NACELLE_SOURCE_PIXEL) {
                if (name != "pixelX" && name != "pixelY") continue;
                require(snapshot.locals[local].kind == NACELLE_SOURCE_UNSIGNED && snapshot.locals[local].bits == 32,
                    "Optimized pixel coordinate local type does not match HLSL");
                for (uint32_t lane = 0; lane < snapshot.width; ++lane)
                    if (detail.valid_mask[local] & (1u << lane))
                        require(snapshot.values[local][lane] == snapshot.xyz[lane][name == "pixelX" ? 0 : 1],
                            "Optimized pixel scalar does not match the shader's actual coordinate");
                available = true;
                continue;
            }
            if (name != "value") continue;
            require(snapshot.locals[local].kind == NACELLE_SOURCE_FLOAT && snapshot.locals[local].bits == 32,
                "Optimized scalar type does not match HLSL");
            for (uint32_t lane = 0; lane < snapshot.width; ++lane) {
                if (!(detail.valid_mask[local] & (1u << lane))) continue;
                const auto x = snapshot.xyz[lane][0];
                const float expected = float(3 * x + 1) + ((x & 1) ? 1.f : -1.f);
                require(snapshot.values[local][lane] == std::bit_cast<uint32_t>(expected),
                    "Optimized per-lane scalar does not match actual shader value");
            }
            available = true;
        }
        require(available || !std::getenv("NACELLE_TEST_SOURCE_UNATTENDED"),
            "The optimized first observation has no supported scalar value");
        NacelleOptimizedSourceDebugStatus compilation;
        require(NacelleGetOptimizedSourceDebugStatus(&compilation) == S_OK &&
            compilation.tier1_compilations && (optimizedSourceTier == 1 || compilation.tier2_compilations),
            "Asynchronous optimized compilation failed or was disabled");
        NacelleSourceDebugSummary summary;
        require(NacelleEndSourceDebug(sourceSession, &summary) == S_FALSE &&
            summary.target_seen && summary.cancelled &&
            summary.stops == (std::getenv("NACELLE_TEST_SOURCE_UNATTENDED") ? 1 : 2),
            "Optimized source step/cancellation did not preserve original work");
        sourceSession = 0;
        std::cout << "OPTIMIZED SOURCE TIER" << optimizedSourceTier <<
            (std::getenv("NACELLE_TEST_SOURCE_UNATTENDED") ? " OBSERVATION PASS" : " LOCAL STEP PASS") << std::endl;
        return;
    }
    bool foundValue = false;
    for (uint32_t local = 0; local < snapshot.site.local_count; ++local) {
        if (std::string_view(snapshot.locals[local].name) != "value") continue;
        require(snapshot.locals[local].kind == NACELLE_SOURCE_FLOAT &&
            snapshot.locals[local].bits == 32, "Supported float local was not available");
        for (uint32_t lane = 0; lane < snapshot.width; ++lane) {
            if (!(snapshot.active_mask & (1u << lane))) continue;
            const auto x = snapshot.xyz[lane][0];
            const float expected = snapshot.site.stage == NACELLE_SOURCE_COMPUTE ?
                float(3 * x + 1) + ((x & 1) ? 1.f : -1.f) +
                    (std::getenv("NACELLE_TEST_SOURCE_LOOP") ? 1.f : 0.f) : ((x & 1) ? 3.f : 1.f);
            require(snapshot.values[local][lane] == std::bit_cast<uint32_t>(expected),
                "Per-lane local does not match the shader's actual value");
        }
        foundValue = true;
    }
    require(foundValue, "Expected HLSL scalar local is missing");
    std::cout << "SOURCE DEBUG LOCAL PASS" << std::endl;
    NacelleSourceDebugSummary summary;
    const auto status = NacelleEndSourceDebug(sourceSession, &summary);
    require((status == S_OK || status == S_FALSE) && summary.target_seen && !summary.concurrent_work &&
        summary.stops == (std::getenv("NACELLE_TEST_SOURCE_UNATTENDED") ? 1 :
                         std::getenv("NACELLE_TEST_SOURCE_HELPER") ? 4 : 3) &&
        summary.cancelled, "Source stop, step and cancellation acceptance");
    sourceSession = 0;
    std::cout << "SOURCE DEBUG STEP PASS stops=" << summary.stops << std::endl;
}

#if defined(NACELLE_PROFILE_SHARED_RUNTIME)
constexpr auto createDevice = &D3D12CreateDevice;
constexpr auto serializeRootSignature = &D3D12SerializeRootSignature;
#else
constexpr auto createDevice = &NacelleCreateDevice;
constexpr auto serializeRootSignature = &NacelleSerializeRootSignature;
#endif

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void checkNativeCode(uint64_t address) {
    NacelleNativeDebugCode code;
    require(NacelleGetNativeDebugCode(address, &code) == S_OK, "Missing native debug code");
    require(code.begin == address && code.end > address && code.code_version &&
            std::strlen(code.object_sha256) == 64 && code.symbol[0], "Invalid native debug identity");
    std::cout << "NATIVE DEBUG READY address=" << address << " end=" << code.end <<
        " symbol=" << code.symbol << std::endl;
    const auto version = code.code_version;
    require(NacelleGetNativeDebugCode(address + 1, &code) == S_OK && code.code_version == version,
            "Interior address lost native code version");
    require(NacelleGetNativeDebugCode(UINT64_MAX, &code) == S_FALSE && !code.begin && !code.symbol[0],
            "Unknown native address did not clear output");
}

void pauseForNativeAttach() {
    if (!std::getenv("NACELLE_TEST_NATIVE_ATTACH")) return;
#if defined(__linux__)
    std::cout << "NATIVE DEBUG ATTACH pid=" << getpid() << "; press Enter after attaching" << std::endl;
    require(std::getchar() != EOF, "Attach fixture requires terminal input");
#else
    require(false, "Native debug attach fixture is supported on Linux only");
#endif
}

void beginCapture(NacelleProfileCapture** capture) {
    NacelleProfileOptions options;
    testutil::checkHR(NacelleBeginProfileCapture(&options, capture), "Begin profile");
}

ComPtr<ID3D12RootSignature> createSignature(testutil::TestRuntime& runtime,
                                           const D3D12_ROOT_SIGNATURE_DESC& desc) {
    ComPtr<ID3DBlob> blob;
    testutil::checkHR(serializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0,
                                                &blob, nullptr), "Serialize signature");
    ComPtr<ID3D12RootSignature> signature;
    testutil::checkHR(runtime.device->CreateRootSignature(0, blob->GetBufferPointer(),
        blob->GetBufferSize(), IID_PPV_ARGS(&signature)), "Create signature");
    return signature;
}

void initialize(testutil::TestRuntime& runtime) {
    testutil::checkHR(createDevice(nullptr, D3D_FEATURE_LEVEL_12_0,
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

void finishOptimizedPromotion(testutil::TestRuntime& runtime, uint32_t stage, ID3D12PipelineState* pipeline) {
    NacelleOptimizedSourceDebugSnapshot initial;
    require(NacelleGetOptimizedSourceDebugSnapshot(sourceSession, &initial) == S_OK && initial.tier == 1,
        "Earlier work did not execute its selected Tier-1 entry");
    finishSourceDebug();
    testutil::checkHR(NacelleWaitForReplayCompilation(), "Await asynchronous promotion after selected Tier-1 work");
    const auto* state = static_cast<nacelle::NacellePipelineState*>(pipeline);
    const auto address = stage == NACELLE_SOURCE_COMPUTE ?
        reinterpret_cast<uint64_t>(state->computePSO()->fn.load(std::memory_order_acquire)) :
        reinterpret_cast<uint64_t>(state->graphicsPSO()->psFn.load(std::memory_order_acquire));
    NacelleNativeDebugCode promoted, retained;
    require(NacelleGetNativeDebugCode(address, &promoted) == S_OK &&
        NacelleGetNativeDebugCode(initial.snapshot.code.begin, &retained) == S_OK &&
        retained.code_version == initial.snapshot.code.code_version &&
        promoted.code_version != retained.code_version && promoted.begin != retained.begin,
        "Asynchronous promotion overwrote the earlier native entry or failed to install new code");
    optimizedSourceTier = 2;
    armSourceDebug(stage);
    ID3D12CommandList* lists[]{runtime.cmdList.Get()};
    runtime.queue->ExecuteCommandLists(1, lists);
    testutil::checkHR(runtime.queue->Signal(runtime.fence.Get(), ++runtime.fenceVal), "Signal promoted work");
    if (runtime.fence->GetCompletedValue() < runtime.fenceVal)
        testutil::checkHR(runtime.fence->SetEventOnCompletion(runtime.fenceVal, nullptr), "Await promoted work");
    finishSourceDebug();
    std::cout << "OPTIMIZED SOURCE ASYNCHRONOUS ENTRY ISOLATION PASS" << std::endl;
}

void compute(testutil::TestRuntime& runtime, NacelleProfileCapture** executionCapture,
             const std::string& mode, const std::string& outputPath) {
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
    const auto* artifactMode = std::getenv("NACELLE_TEST_ARTIFACTS");
    const auto* spatialMode = std::getenv("NACELLE_TEST_SPATIAL");
    const bool volume = spatialMode && std::string_view(spatialMode) == "volume";
    auto shader = testutil::loadShaderBytes(mode == "source-compute" ?
        (std::getenv("NACELLE_TEST_SOURCE_STRIPPED") ? "vec_add.dxil" :
         (std::getenv("NACELLE_TEST_SOURCE_OPTIMIZED") ||
          (optimizedSourceMode() && !std::getenv("NACELLE_TEST_SOURCE_INPUT_O0"))) ?
             "source_debug_optimized_cs.dxil" : "source_debug_cs.dxil") :
        artifactMode && (std::string_view(artifactMode) == "source" ||
        std::string_view(artifactMode) == "no-source")
        ? "vec_add_debug.dxil" : volume ? "vec_add_volume.dxil" : "vec_add.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = signature.Get();
    desc.CS = {shader.data(), shader.size()};
    ComPtr<ID3D12PipelineState> pso;
    const auto compileStart = std::chrono::steady_clock::now();
    const auto pipelineStatus = runtime.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso));
    const auto compileTime = std::chrono::steady_clock::now() - compileStart;
    if (mode == "source-compute" &&
        (std::getenv("NACELLE_TEST_SOURCE_STRIPPED") || std::getenv("NACELLE_TEST_SOURCE_OPTIMIZED"))) {
        require(FAILED(pipelineStatus) && !pso, "Invalid source debug input produced a pipeline");
        std::cout << "SOURCE DEBUG REJECTION PASS" << std::endl;
        return;
    }
    testutil::checkHR(pipelineStatus, "Compute PSO");
    if (mode == "promoted" || mode == "cached") {
        NacelleProfileSummary summary;
        testutil::checkHR(NacelleEndProfileCapture(*executionCapture,
            (outputPath + ".warmup").c_str(), &summary), "Drain captured Tier-2 compilation");
        NacelleDestroyProfileCapture(*executionCapture);
        *executionCapture = nullptr;
    }
    // A second creation exercises the same capture's cache metadata.
    ComPtr<ID3D12PipelineState> cached;
    testutil::checkHR(runtime.device->CreateComputePipelineState(&desc, IID_PPV_ARGS(&cached)), "Cached compute PSO");
    if (mode == "native-compute") {
        const auto& compiled = static_cast<nacelle::NacellePipelineState*>(pso.Get())->computePSO();
        const auto entry = compiled->fn.load();
        checkNativeCode(reinterpret_cast<uint64_t>(entry));
        testutil::checkHR(NacelleWaitForReplayCompilation(), "Drain native debug compilation");
        require(compiled->fn.load() == entry && !compiled->asyncBox, "Native debug code was promoted");
        pauseForNativeAttach();
    }
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
    if (executionCapture) {
        if (artifactMode && std::string_view(artifactMode) == "missing") {
            NacelleProfileArtifactOptions options;
            options.flags = NACELLE_PROFILE_RETAIN_SHADER_TEXT;
            testutil::checkHR(NacelleConfigureProfileArtifacts(&options), "Enable late retention");
        }
        beginCapture(executionCapture);
    }
    runtime.resetCmd();
    runtime.cmdList->SetComputeRootSignature(signature.Get());
    runtime.cmdList->SetPipelineState(executionCapture && mode != "promoted" ? cached.Get() : pso.Get());
    runtime.cmdList->SetComputeRootShaderResourceView(0, inputA->GetGPUVirtualAddress());
    runtime.cmdList->SetComputeRootShaderResourceView(1, inputB->GetGPUVirtualAddress());
    runtime.cmdList->SetComputeRootUnorderedAccessView(2, output->GetGPUVirtualAddress());
    const float scale = 1.0f;
    uint32_t constants[3]{0, 0, static_cast<uint32_t>(a.size())};
    std::memcpy(&constants[0], &scale, sizeof(scale));
    runtime.cmdList->SetComputeRoot32BitConstants(3, 3, constants, 0);
    const auto promotionStart = std::chrono::steady_clock::now();
    if (optimizedSourceMode() && optimizedSourceTier == 2)
        testutil::checkHR(NacelleWaitForReplayCompilation(), "Await actual asynchronous Tier-2 publication");
    const auto promotionTime = std::chrono::steady_clock::now() - promotionStart;
    if (mode == "source-compute") armSourceDebug(NACELLE_SOURCE_COMPUTE);
    if (mode == "source-compute" && std::getenv("NACELLE_TEST_SOURCE_ZERO"))
        runtime.cmdList->Dispatch(0, 1, 1);
    if (volume) runtime.cmdList->Dispatch(4, 2, 4);
    else runtime.cmdList->Dispatch(16, 1, 1);
    const auto executionStart = std::chrono::steady_clock::now();
    runtime.executeAndWait();
    const auto executionTime = std::chrono::steady_clock::now() - executionStart;
    if (optimizedSourceMode() && optimizedSourceTier == 1)
        finishOptimizedPromotion(runtime, NACELLE_SOURCE_COMPUTE, pso.Get());
    else if (mode == "source-compute") finishSourceDebug();
    if (std::getenv("NACELLE_TEST_SOURCE_COST")) {
        uint64_t sites = 0, metadata = 0, sourceBytes = 0;
        std::set<std::string> sources;
        for (uint32_t id = 1; ; ++id) {
            NacelleSourceDebugSite site;
            const auto status = NacelleGetSourceDebugSite(id, &site);
            if (status == S_FALSE) break;
            require(status == S_OK, "Diagnostic site query");
            ++sites;
            metadata += sizeof(site) + site.local_count * sizeof(NacelleSourceDebugLocal);
            if (sources.insert(site.source_id).second) {
                uint64_t size = 0;
                require(NacelleGetSourceDebugSource(site.source_id, nullptr, &size) == S_OK,
                    "Diagnostic source size");
                sourceBytes += size;
            }
        }
        NacelleOptimizedSourceDebugStatus optimized;
        if (optimizedSourceMode() && !nativeSourceCost() && !ordinarySourceCost())
            require(NacelleGetOptimizedSourceDebugStatus(&optimized) == S_OK, "Diagnostic optimized compiler status");
        std::cout << "SOURCE DEBUG COST native=" << nativeSourceCost() << " ordinary=" << ordinarySourceCost() <<
            " compile_ms=" << std::chrono::duration<double, std::milli>(compileTime).count() <<
            " promotion_wait_ms=" << std::chrono::duration<double, std::milli>(promotionTime).count() <<
            " execution_ms=" << std::chrono::duration<double, std::milli>(executionTime).count() <<
            " sites=" << sites << " metadata_payload_bytes=" << metadata <<
            " source_payload_bytes=" << sourceBytes <<
            " snapshot_bytes=" << sizeof(NacelleSourceDebugSnapshot) <<
            " retained_source_bytes=" << optimized.retained_source_bytes <<
            " retained_metadata_bytes=" << optimized.retained_metadata_bytes <<
            " optimized_snapshot_bytes=" << sizeof(NacelleOptimizedSourceDebugSnapshot) << std::endl;
    }
    const auto bytes = runtime.readbackBuffer(output.Get(), sizeof(a));
    std::array<float, 1024> actual{};
    std::memcpy(actual.data(), bytes.data(), bytes.size());
    for (size_t i = 0; i < actual.size(); ++i)
        require(actual[i] == a[i] + b[i], "Capture changed compute output");
}

void draw(testutil::TestRuntime& runtime, NacelleProfileCapture** executionCapture = nullptr,
          const std::string& mode = "all", const std::string& outputPath = {}) {
    D3D12_ROOT_SIGNATURE_DESC signatureDesc{};
    auto signature = createSignature(runtime, signatureDesc);
    const bool extended = mode.starts_with("graphics-extended");
    const bool mesh = mode.starts_with("graphics-mesh");
    const auto* spatialMode = std::getenv("NACELLE_TEST_SPATIAL");
    const bool edges = spatialMode && (std::string_view(spatialMode) == "edges" ||
                                      std::string_view(spatialMode) == "derivatives");
    const bool derivatives = spatialMode && std::string_view(spatialMode) == "derivatives";
    auto vs = testutil::loadShaderBytes(extended ? "gfx_tess_point_vs.dxil" : "raster_basic_vs.dxil");
    auto ps = testutil::loadShaderBytes(extended ? "gfx_tess_point_ps.dxil" :
                                      mesh ? "mesh_solid_red_ps.dxil" :
                                      mode == "source-graphics" ?
                                          (optimizedSourceMode() ?
                                              (std::getenv("NACELLE_TEST_SOURCE_INPUT_O0") ?
                                                  "source_debug_pixel_o0.dxil" : "source_debug_optimized_ps.dxil") :
                                              "source_debug_ps.dxil") :
                                      derivatives ? "raster_quad_coverage_ps.dxil" : "raster_basic_ps.dxil");
    std::vector<uint8_t> hs, ds, gs, as, ms;
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
    if (extended) {
        hs = testutil::loadShaderBytes("gfx_tess_point_hs.dxil");
        ds = testutil::loadShaderBytes("gfx_tess_point_ds.dxil");
        gs = testutil::loadShaderBytes("gfx_tess_point_gs.dxil");
        desc.HS = {hs.data(), hs.size()};
        desc.DS = {ds.data(), ds.size()};
        desc.GS = {gs.data(), gs.size()};
        desc.InputLayout = {};
        desc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH;
    }
    const auto createPipeline = [&] {
        ComPtr<ID3D12PipelineState> pipeline;
        if (mesh) {
            ComPtr<ID3D12Device2> device;
            testutil::checkHR(runtime.device.As(&device), "Mesh-capable device");
            as = testutil::loadShaderBytes("mesh_payload_as.dxil");
            ms = testutil::loadShaderBytes("mesh_payload_ms.dxil");
            CD3DX12_PIPELINE_MESH_STATE_STREAM stream;
            stream.pRootSignature = signature.Get();
            stream.AS = D3D12_SHADER_BYTECODE{as.data(), as.size()};
            stream.MS = D3D12_SHADER_BYTECODE{ms.data(), ms.size()};
            stream.PS = desc.PS;
            stream.RasterizerState = CD3DX12_RASTERIZER_DESC(desc.RasterizerState);
            stream.BlendState = CD3DX12_BLEND_DESC(desc.BlendState);
            CD3DX12_DEPTH_STENCIL_DESC1 depth(D3D12_DEFAULT);
            depth.DepthEnable = FALSE;
            depth.StencilEnable = FALSE;
            stream.DepthStencilState = depth;
            stream.SampleMask = UINT32_MAX;
            stream.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
            D3D12_RT_FORMAT_ARRAY formats{};
            formats.NumRenderTargets = 1;
            formats.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            stream.RTVFormats = formats;
            stream.SampleDesc = desc.SampleDesc;
            D3D12_PIPELINE_STATE_STREAM_DESC streamDesc{sizeof(stream), &stream};
            testutil::checkHR(device->CreatePipelineState(&streamDesc, IID_PPV_ARGS(&pipeline)), "Mesh PSO");
        } else {
            testutil::checkHR(runtime.device->CreateGraphicsPipelineState(&desc, IID_PPV_ARGS(&pipeline)), "Graphics PSO");
        }
        return pipeline;
    };
    auto pso = createPipeline();
    if (mode == "native-graphics") {
        const auto& compiled = static_cast<nacelle::NacellePipelineState*>(pso.Get())->graphicsPSO();
        const auto vertex = compiled->vsFn.load(std::memory_order_acquire);
        const auto pixel = compiled->psFn.load(std::memory_order_acquire);
        checkNativeCode(reinterpret_cast<uint64_t>(vertex));
        checkNativeCode(reinterpret_cast<uint64_t>(pixel));
        testutil::checkHR(NacelleWaitForReplayCompilation(), "Drain native graphics compilation");
        require(compiled->vsFn.load(std::memory_order_acquire) == vertex &&
                compiled->psFn.load(std::memory_order_acquire) == pixel, "Native graphics code was promoted");
        pauseForNativeAttach();
    }
    if (mode == "graphics-mixed-cache") {
        ps = testutil::loadShaderBytes("mesh_solid_red_ps.dxil");
        desc.PS = {ps.data(), ps.size()};
        pso = createPipeline();
    }
    if (mode.ends_with("promoted") || mode.ends_with("cached")) {
        NacelleProfileSummary summary;
        testutil::checkHR(NacelleEndProfileCapture(*executionCapture,
            (outputPath + ".warmup").c_str(), &summary), "Drain graphics promotion");
        NacelleDestroyProfileCapture(*executionCapture);
        *executionCapture = nullptr;
    }
    if (mode.ends_with("cached")) pso = createPipeline();
    const float vertices[][7] = {
        {-1, -1, 0.5f, 1, 0, 0, 1}, {-1, 3, 0.5f, 1, 0, 0, 1}, {3, -1, 0.5f, 1, 0, 0, 1},
    };
    auto vertexBuffer = runtime.createBuffer(sizeof(vertices), D3D12_HEAP_TYPE_DEFAULT);
    runtime.uploadBuffer(vertexBuffer.Get(), vertices, sizeof(vertices));
    const UINT width = edges ? 129 : 128;
    const UINT height = edges ? 67 : 128;
    const UINT rowPitch = (width * 4 + 255) & ~255u;
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    auto textureDesc = CD3DX12_RESOURCE_DESC::Tex2D(DXGI_FORMAT_R8G8B8A8_UNORM, width, height,
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
    D3D12_VIEWPORT viewport{0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1};
    D3D12_RECT scissor{edges ? 1 : 0, edges ? 1 : 0, static_cast<LONG>(width), static_cast<LONG>(height)};
    if (executionCapture) beginCapture(executionCapture);
    runtime.resetCmd();
    runtime.cmdList->SetGraphicsRootSignature(signature.Get());
    runtime.cmdList->SetPipelineState(pso.Get());
    runtime.cmdList->RSSetViewports(1, &viewport);
    runtime.cmdList->RSSetScissorRects(1, &scissor);
    runtime.cmdList->OMSetRenderTargets(1, &rtv, FALSE, nullptr);
    runtime.cmdList->IASetPrimitiveTopology(extended ? D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST :
                                                     D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    runtime.cmdList->IASetVertexBuffers(0, 1, &vb);
    const float clear[4]{0, 0, 0, 1};
    runtime.cmdList->ClearRenderTargetView(rtv, clear, 0, nullptr);
    if (mesh) {
        ComPtr<ID3D12GraphicsCommandList6> commands;
        testutil::checkHR(runtime.cmdList.As(&commands), "Mesh command list");
        commands->DispatchMesh(4, 1, 1);
    } else {
        runtime.cmdList->DrawInstanced(3, 1, 0, 0);
    }
    if (optimizedSourceMode() && optimizedSourceTier == 2)
        testutil::checkHR(NacelleWaitForReplayCompilation(), "Await actual asynchronous pixel Tier-2 publication");
    if (std::getenv("NACELLE_TEST_SOURCE_MIXED")) {
        const auto& compiled = static_cast<nacelle::NacellePipelineState*>(pso.Get())->graphicsPSO();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (compiled->vsFn.snapshot()->tier != 2) {
            require(std::chrono::steady_clock::now() < deadline, "Independent vertex promotion did not complete");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(compiled->psFn.snapshot()->tier == 1, "Mixed-stage test did not preserve the earlier pixel entry");
        std::cout << "OPTIMIZED SOURCE MIXED VS2 PS1 READY" << std::endl;
    }
    if (mode == "source-graphics") armSourceDebug(NACELLE_SOURCE_PIXEL);
    runtime.executeAndWait();
    if (optimizedSourceMode() && optimizedSourceTier == 1)
        finishOptimizedPromotion(runtime, NACELLE_SOURCE_PIXEL, pso.Get());
    else if (mode == "source-graphics") finishSourceDebug();
    auto readback = runtime.createBuffer(rowPitch * height, D3D12_HEAP_TYPE_READBACK);
    D3D12_TEXTURE_COPY_LOCATION destination{};
    destination.pResource = readback.Get();
    destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination.PlacedFootprint.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, width, height, 1, rowPitch};
    D3D12_TEXTURE_COPY_LOCATION source{};
    source.pResource = target.Get();
    source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    runtime.resetCmd();
    runtime.cmdList->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
    runtime.executeAndWait();
    void* data = nullptr;
    testutil::checkHR(readback->Map(0, nullptr, &data), "Map graphics result");
    const auto* pixels = static_cast<const uint8_t*>(data);
    const size_t center = (height / 2 * rowPitch + width / 2 * 4);
    bool correct = pixels[center] == 255 && pixels[center + 1] == 0 &&
                   pixels[center + 2] == 0 && pixels[center + 3] == 255;
    if (extended) {
        correct = false;
        for (size_t i = 0; i < width * width * 4; i += 4) {
            if (pixels[i] == 0 && pixels[i + 1] == 255 && pixels[i + 2] == 0 && pixels[i + 3] == 255)
                correct = true;
        }
        if (spatialMode && !mesh && !extended) {
            correct = true;
            for (UINT y = 0; y < height; ++y)
                for (UINT x = 0; x < width; ++x) {
                    const size_t offset = y * rowPitch + x * 4;
                    const uint8_t red = (edges && (x == 0 || y == 0)) || (derivatives && (x + y) % 3 == 0) ? 0 : 255;
                    correct &= pixels[offset] == red && pixels[offset + 1] == 0 &&
                               pixels[offset + 2] == 0 && pixels[offset + 3] == 255;
                }
        }
    }
    if (mode == "source-graphics") {
        correct = true;
        for (UINT y = 0; y < height; ++y)
            for (UINT x = 0; x < width; ++x) {
                const size_t offset = y * rowPitch + x * 4;
                const uint8_t red = (x + y) % 3 == 0 ? 0 : 255;
                correct &= pixels[offset] == red && pixels[offset + 1] == 0 &&
                    pixels[offset + 2] == 0 && pixels[offset + 3] == 255;
            }
    }
    readback->Unmap(0, nullptr);
    require(correct, "Capture changed graphics output");
}
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) {
        std::cerr << "Usage: profile_runtime_test <new-profile-path> "
                     "[all|compute|graphics|tier1|promoted|cached|disk|"
                     "graphics-tier1|graphics-promoted|graphics-cached|graphics-disk|"
                     "native-compute|native-graphics]\n";
        return 1;
    }
    const std::string mode = argc == 3 ? argv[2] : "all";
    if (mode != "all" && mode != "compute" && mode != "graphics" && mode != "tier1" &&
        mode != "promoted" && mode != "cached" && mode != "disk" &&
        mode != "graphics-tier1" && mode != "graphics-promoted" &&
        mode != "graphics-cached" && mode != "graphics-disk" && mode != "graphics-mixed-cache" &&
        mode != "graphics-extended" && mode != "graphics-extended-promoted" && mode != "graphics-extended-cached" &&
        mode != "graphics-mesh" && mode != "graphics-mesh-promoted" && mode != "graphics-mesh-cached" &&
        mode != "native-compute" && mode != "native-graphics" &&
        mode != "source-compute" && mode != "source-graphics") {
        std::cerr << "Invalid capture mode: " << mode << '\n';
        return 1;
    }
    NacelleProfileCapture* capture = nullptr;
    const bool counterMode = std::getenv("NACELLE_TEST_CPU_COUNTERS") != nullptr;
    try {
        if (counterMode) {
            NacelleProfileCounterOptions options;
            options.flags = 1;
            testutil::checkHR(NacelleConfigureProfileCounters(&options), "Configure CPU counters");
        }
        if (mode.starts_with("source-")) {
            NacelleSourceDebugOptions options;
            options.enabled = 1;
            options.consent = 3;
            if (ordinarySourceCost()) {
                std::cout << "SOURCE DEBUG COST ordinary runtime policy" << std::endl;
            } else if (nativeSourceCost()) {
                NacelleNativeDebugOptions native;
                native.enabled = 1;
                testutil::checkHR(NacelleConfigureNativeDebug(&native), "Enable matched native-debug cost baseline");
            } else if (optimizedSourceMode()) {
                if (std::getenv("NACELLE_TEST_SOURCE_ASYNC")) optimizedSourceTier = 1;
                NacelleOptimizedSourceDebugOptions optimized;
                optimized.enabled = 1;
                optimized.consent = 3;
                testutil::checkHR(NacelleConfigureOptimizedSourceDebug(&optimized), "Enable optimized source debugging");
            } else testutil::checkHR(NacelleConfigureSourceDebug(&options), "Enable source debug");
            testutil::TestRuntime runtime;
            initialize(runtime);
            if (mode == "source-compute") compute(runtime, nullptr, mode, argv[1]);
            else draw(runtime, nullptr, mode, argv[1]);
            require(FAILED(NacelleConfigureSourceDebug(&options)), "Late source configuration accepted");
            std::cout << "SOURCE DEBUG OUTPUT PASS" << std::endl;
            return 0;
        }
        if (mode.starts_with("native-")) {
            NacelleNativeDebugOptions options;
            options.enabled = 1;
            testutil::checkHR(NacelleConfigureNativeDebug(&options), "Enable native debug");
            testutil::TestRuntime runtime;
            initialize(runtime);
            if (mode == "native-compute") compute(runtime, nullptr, mode, argv[1]);
            else draw(runtime, nullptr, mode, argv[1]);
            require(FAILED(NacelleConfigureNativeDebug(&options)), "Late debug configuration accepted");
            std::cout << "NATIVE DEBUG OUTPUT PASS" << std::endl;
            return 0;
        }
        const auto* spatialMode = std::getenv("NACELLE_TEST_SPATIAL");
        if (spatialMode) {
            NacelleProfileSpatialOptions options;
            options.flags = NACELLE_PROFILE_SPATIAL_COMPUTE | NACELLE_PROFILE_SPATIAL_PIXEL;
            if (std::string_view(spatialMode) == "overflow") options.record_bytes = 424;
            if (std::string_view(spatialMode) == "sampled") options.sample_stride = 2;
            if (std::string_view(spatialMode) == "roi") {
                options.min_xyz[0] = 1;
                options.max_xyz[0] = 2;
            }
            testutil::checkHR(NacelleConfigureProfileSpatial(&options), "Configure spatial observation");
        }
        const auto* artifactMode = std::getenv("NACELLE_TEST_ARTIFACTS");
        if (artifactMode && std::string_view(artifactMode) != "missing") {
            NacelleProfileArtifactOptions options;
            options.flags = NACELLE_PROFILE_RETAIN_SHADER_TEXT;
            if (std::string_view(artifactMode) == "source") options.flags |= NACELLE_PROFILE_EMBED_SHADER_SOURCES;
            testutil::checkHR(NacelleConfigureProfileArtifacts(&options), "Configure retained artifacts");
        }
        const bool graphicsOnly = mode.starts_with("graphics");
        if (mode == "all" || mode == "promoted" || mode == "cached" ||
            (graphicsOnly && (mode.ends_with("promoted") || mode.ends_with("cached")))) beginCapture(&capture);
        testutil::TestRuntime runtime;
        initialize(runtime);
        if (!graphicsOnly)
            compute(runtime, mode == "all" ? nullptr : &capture, mode, argv[1]);
        if (mode == "all" || graphicsOnly)
            draw(runtime, graphicsOnly ? &capture : nullptr, mode, argv[1]);
        NacelleProfileSummary summary;
        testutil::checkHR(NacelleEndProfileCapture(capture, argv[1], &summary), "End profile");
        NacelleDestroyProfileCapture(capture);
        capture = nullptr;
        nacelle::profile::Profile profile;
        std::string error;
        const auto parsed = nacelle::profile::readProfile(argv[1], profile, error);
        require(parsed, error.c_str());
        const auto hasSpan = [&](const char* name) {
            return std::any_of(profile.trace.traceEvents.begin(), profile.trace.traceEvents.end(),
                [&](const auto& event) { return event.name == name; });
        };
        if (mode == "all") {
            for (const char* name : {"Compile compute PSO", "Compile graphics PSO",
                                     "Compute dispatch", "Graphics draw", "Raster worker", "Compute Tier 2"})
                require(hasSpan(name), name);
        } else {
            require(!hasSpan("Compile compute PSO") && !hasSpan("Compile graphics PSO"),
                    "Execution-only capture must not require pipeline creation");
            require(hasSpan(mode.starts_with("graphics-mesh") ? "Mesh dispatch" :
                            graphicsOnly ? "Graphics draw" : "Compute dispatch"),
                    "Pre-existing pipeline execution recorded");
        }
        if (!graphicsOnly) {
            require(!profile.shaders.codeVersions.empty(), "Executed compute version metadata exists");
            for (const auto& event : profile.trace.traceEvents) {
                if (event.name != "Compute dispatch" && event.name != "Compute worker") continue;
                require(event.args.pipelineId && event.args.codeVersionId,
                        "Compute dispatches and workers identify the selected native entry");
                const auto& code = profile.shaders.codeVersions.at(*event.args.codeVersionId - 1);
                require(code.shaderId && code.shaderId->size() == 64, "Compute identifies exact input bytecode content");
                if (mode == "tier1") require(!code.artifactId, "IR-compiled Tier 1 object content is unavailable");
                if (mode == "promoted" || mode == "cached" || mode == "disk")
                    require(code.artifactId && code.artifactId->size() == 64, "Tier 2 identifies actual object content");
                if (mode == "tier1")
                    require(code.version == 1 && code.tier == 1 && code.cache == "miss",
                            "Cold Tier-1 execution");
                if (mode == "promoted")
                    require(code.version == 2 && code.tier == 2 && code.cache == "miss",
                            "Execute promoted entry on the original pipeline");
                if (mode == "cached" || mode == "disk")
                    require(code.version == 1 && code.tier == 2 && code.cache == "hit",
                            "Cached native entry is Tier 2 on a new pipeline");
            }
        }
        if (mode == "all" || graphicsOnly) {
            const bool extended = mode.starts_with("graphics-extended");
            const bool mesh = mode.starts_with("graphics-mesh");
            require(profile.shaders.schemaVersion == 2 &&
                    hasSpan(extended ? "Extended geometry" : mesh ? "Mesh geometry" : "Vertex geometry") &&
                    hasSpan("Pixel raster"),
                    "Graphics stage execution metadata available");
            std::optional<uint64_t> graphicsPipeline;
            for (const auto& event : profile.trace.traceEvents) {
                if (event.name == "Graphics draw" || event.name == "Mesh dispatch") {
                    require(!event.args.codeVersionId, "A draw does not pretend to execute one combined version");
                    require(event.args.codeSelectionId && profile.shaders.codeSelections,
                            "Draw references observed entry selections");
                    const auto& selection = profile.shaders.codeSelections->at(*event.args.codeSelectionId - 1);
                    require(selection.codeVersionIds.size() == (extended ? 6 : mesh ? 3 : 2),
                            "Draw selection includes precisely the participating stage entries");
                }
                if (event.name != "Vertex geometry" && event.name != "Pixel raster" &&
                    event.name != "Mesh geometry" && event.name != "Amplification stage") continue;
                require(event.args.pipelineId && event.args.codeVersionId, "Graphics region selects an immutable entry");
                const auto& code = profile.shaders.codeVersions.at(*event.args.codeVersionId - 1);
                require(code.entryRole == code.stage && code.shaderId && code.shaderId->size() == 64,
                        "Graphics entries identify stage roles and exact bytecode content");
                if (const auto* width = std::getenv("NACELLE_SIMD_WIDTH"))
                    require(code.simdWidth == std::stoul(width), "Graphics metadata matches the selected SIMD width");
                require(!graphicsPipeline || graphicsPipeline == code.pipelineId,
                        "Vertex and pixel versions share the same capture-local pipeline");
                graphicsPipeline = code.pipelineId;
                if (mode == "graphics-tier1")
                    require(code.tier == 1 && code.cache == "miss", "Graphics cold Tier 1");
                if (mode == "graphics-tier1" || mode == "graphics-mixed-cache")
                    require(!code.artifactId, "Graphics Tier 1 does not invent native object content");
                if (mode == "graphics-mixed-cache")
                    require(code.version == 1 && code.tier == 1 &&
                            code.cache == (code.stage == "vertex" ? "hit" : "miss"),
                            "Independent graphics stages preserve mixed Tier-1 cache outcomes");
                if (mode.ends_with("promoted"))
                    require(code.version >= 2 && code.tier == 2 && code.cache == "miss",
                            "Original graphics entry promoted without changing old snapshots");
                if (mode.ends_with("cached") || mode == "graphics-disk")
                    require(code.version == 1 && code.tier == 2 && code.cache == "hit",
                            "Graphics cache-hit entries start at version 1");
                if (mode.ends_with("promoted") || mode.ends_with("cached") || mode == "graphics-disk")
                    require(code.artifactId && code.artifactId->size() == 64,
                            "Graphics cache/promoted entries identify actual native object content");
            }
            if (extended) {
                std::set<std::string> roles;
                for (const auto& code : profile.shaders.codeVersions) {
                    require(code.entryRole.has_value(), "Extended native entries have explicit roles");
                    roles.insert(*code.entryRole);
                }
                require(roles == std::set<std::string>{"vertex", "pixel", "geometry", "domain",
                                                     "hull-control-point", "hull-patch-constant"},
                        "Tessellation control-point and patch-constant entries are distinct");
            }
        }
        const auto& build = profile.manifest.build;
        if (spatialMode) {
            require(profile.manifest.schemaVersion == 3 && profile.spatial,
                    "Explicit spatial capture declares versioned bounded payload");
            const auto& spatial = *profile.spatial;
            require(!spatial.operations.empty(), "Spatial capture identifies eligible operations");
            const bool overflow = std::string_view(spatialMode) == "overflow";
            require(spatial.complete != overflow && (spatial.dropped > 0) == overflow,
                    "Spatial overflow is incomplete with exact explicit drop counters");
            require(spatial.recorded > 0, "Host entry-call observations were captured");
            if (std::string_view(spatialMode) == "sampled" || std::string_view(spatialMode) == "roi")
                require(spatial.skipped > 0, "Deterministic sampling skips observations without changing results");
            for (const auto& operation : spatial.operations) {
                require(operation.codeVersionId && operation.pipelineId &&
                        operation.pso == profile.shaders.codeVersions[operation.codeVersionId - 1].pso,
                        "Spatial cells resolve immutable executed entries");
                if (const auto* width = std::getenv("NACELLE_SIMD_WIDTH"))
                    require(operation.simdWidth == std::stoul(width), "Requested SIMD width is actually exercised");
                if (std::string_view(spatialMode) == "roi")
                    for (const auto& cell : operation.cells)
                        require(cell.xyz[0] == 1, "ROI limits only observed coordinates");
                if (operation.stage == "compute") {
                    const bool volume = std::string_view(spatialMode) == "volume";
                    require(operation.dimensions == (volume ? std::array<uint32_t, 3>{4, 2, 4} :
                                                              std::array<uint32_t, 3>{16, 1, 1}) &&
                            operation.threadsPerGroup == (volume ? std::array<uint32_t, 3>{8, 2, 2} :
                                                                  std::array<uint32_t, 3>{64, 1, 1}),
                            "Compute spatial coordinates preserve logical threadgroups");
                } else {
                    require(operation.orientation == "top-left-y-down" &&
                            (overflow ? operation.cells.empty() && operation.dropped > 0 : !operation.cells.empty()),
                            "Pixel tile orientation and submitted-lane metrics are explicit");
                    for (const auto& cell : operation.cells)
                        require(cell.coveredLanes.size() == operation.simdWidth &&
                                cell.companionLanes.size() == operation.simdWidth,
                                "Spatial pixel coverage/companion ABI lanes remain distinguishable");
                    if (std::string_view(spatialMode) == "edges" || std::string_view(spatialMode) == "derivatives") {
                        uint64_t companions = 0, covered = 0;
                        for (const auto& cell : operation.cells)
                            for (size_t lane = 0; lane < cell.companionLanes.size(); ++lane) {
                                companions += cell.companionLanes[lane];
                                covered += cell.coveredLanes[lane];
                            }
                        require(companions > 0 && covered == 128 * 66 &&
                                operation.extent == std::array<uint32_t, 3>{129, 67, 1},
                                "Odd edges retain real quad companions rather than filtering shader execution");
                    }
                }
            }
        }
        if (artifactMode) {
            require(profile.manifest.schemaVersion >= 2 && profile.artifacts.has_value(),
                    "Opt-in capture contains versioned declared artifacts");
            const auto& artifacts = *profile.artifacts;
            const auto hasKind = [&](const char* kind) {
                return std::any_of(artifacts.payloads.begin(), artifacts.payloads.end(),
                    [&](const auto& payload) { return payload.kind == kind; });
            };
            if (std::string_view(artifactMode) == "missing") {
                require(artifacts.payloads.empty() && artifacts.selections.front().shaderAvailability == "not-retained",
                        "Pre-existing pipelines without retention do not recompile or invent artifacts");
            } else {
                require(hasKind("dxil") && (mode == "all" || hasKind("assembly")),
                        "Cache-hit entries retain exact shader and native text");
                require(hasKind("hlsl") == (std::string_view(artifactMode) == "source"),
                        "Source embedding is explicit and stripped/no-consent input has no source text");
                if (std::string_view(artifactMode) == "source") {
                    require(std::any_of(artifacts.payloads.begin(), artifacts.payloads.end(),
                        [](const auto& payload) { return payload.kind == "dxil" && !payload.mappings.empty(); }),
                        "Real embedded DXC source/debug locations produce preserved HLSL-to-DXIL mappings");
                }
            }
            for (const auto& payload : artifacts.payloads)
                require(payload.label.find('/') == std::string::npos && payload.label.find('\\') == std::string::npos,
                        "Artifact labels never expose host paths");
        }
        require(!build.compilerHash.empty() && !build.targetTriple.empty() && !build.cpu.empty(),
                "Compiler and CPU metadata available without requiring pipeline creation");
        require(std::is_sorted(build.cpuFeatures.begin(), build.cpuFeatures.end()),
                "CPU features have stable ordering");
        if (counterMode) {
            const auto eligibleRegions = std::count_if(profile.trace.traceEvents.begin(), profile.trace.traceEvents.end(),
                [](const auto& event) {
                    return event.ph == "X" && event.args.codeVersionId &&
                        ((event.name == "Compute worker" || event.name == "Compute direct") && event.args.stage == "compute" ||
                         event.name == "Raster worker" && event.args.stage == "pixel");
                });
            require(profile.counters && profile.counters->attempted > 0 &&
                profile.counters->attempted == static_cast<uint64_t>(eligibleRegions) &&
                profile.counters->recorded == profile.counters->attempted && !profile.counters->dropped,
                "CPU counter scopes missing or dropped");
            std::cout << "CPU COUNTERS provider=" << profile.counters->provider
                      << " availability=" << profile.counters->availability
                      << " records=" << profile.counters->recorded;
            for (const auto& record : profile.counters->records)
                require(record.codeVersionId && record.eventId, "Counters lack exact native version attribution");
            if (!profile.counters->records.empty())
                std::cout << " first-status=" << profile.counters->records.front().status;
            std::cout << '\n';
        } else require(!profile.counters && profile.manifest.schemaVersion < 4, "Counters enabled without consent");
        const bool expectedComplete = (!spatialMode || std::string_view(spatialMode) != "overflow") &&
            (!counterMode || profile.counters->complete);
        require(profile.manifest.complete == expectedComplete && bool(summary.complete) == expectedComplete &&
                summary.recorded_events > (mode == "all" ? 10u : 0u),
                "Complete runtime capture");
        std::cout << mode << " output + runtime profile spans + compiler metadata: PASS ("
                  << summary.recorded_events << " events)\n";
        return 0;
    } catch (const std::exception& error) {
        NacelleDestroyProfileCapture(capture);
        std::cerr << "profile_runtime_test: " << error.what() << '\n';
        return 1;
    }
}
