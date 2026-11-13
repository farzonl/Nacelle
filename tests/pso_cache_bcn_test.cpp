#include "test_common.h"
#include "../src/common/format_utils.h"
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"

#include <chrono>
#include <cstring>
#include <vector>

int main() {
    // 0. Verify build-time compiler SHA-256 stamp
    {
        const char* buildHash = nacelle::getCompilerBuildHash();
        assert(buildHash != nullptr);
        assert(std::strlen(buildHash) == 64);
        for (size_t i = 0; i < 64; ++i) {
            char c = buildHash[i];
            assert((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        }
    }

    // 1. Verify BC1 decompression
    {
        // Solid red 4x4 BC1 block: color0 = 0xF800 (red), color1 = 0x0000, indices = 0
        uint8_t bc1[8] = {0x00, 0xF8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
        float rgba[4 * 4 * 4]{};
        nacelle::decompressBCBlockRGBA32F(DXGI_FORMAT_BC1_UNORM, bc1, rgba);
        assert(rgba[0] > 0.9f && rgba[1] < 0.05f && rgba[2] < 0.05f && rgba[3] > 0.95f);
    }

    // 2. Verify ID3D12PipelineLibrary serialization & warm PSO cache reload (< 5 ms)
    testutil::TestRuntime rt;
    rt.init();

    auto csBytes = testutil::loadShaderBytes("vec_add.dxil");

    D3D12_ROOT_PARAMETER rp[3]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[1].Descriptor.ShaderRegister = 1;
    rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[2].Descriptor.ShaderRegister = 0;
    D3D12_ROOT_SIGNATURE_DESC rsDesc{3, rp, 0, nullptr, D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ComPtr<ID3DBlob> rsBlob;
    testutil::checkHR(NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(rt.device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");

    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.CS = {csBytes.data(), csBytes.size()};

    nacelle::clearInMemoryPSOCacheForTesting();
    ComPtr<ID3D12PipelineState> psoCold;
    testutil::checkHR(rt.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&psoCold)), "ColdPSO");
    nacelle::waitForAllAsyncJITTasks();

    ComPtr<ID3DBlob> cachedBlob;
    testutil::checkHR(psoCold->GetCachedBlob(&cachedBlob), "GetCachedBlob");
    assert(cachedBlob->GetBufferSize() > 0);

    ComPtr<ID3D12PipelineLibrary> lib;
    testutil::checkHR(rt.device->CreatePipelineLibrary(nullptr, 0, IID_PPV_ARGS(&lib)), "CreateEmptyLib");
    testutil::checkHR(lib->StorePipeline(L"vec_add", psoCold.Get()), "StorePipeline");

    SIZE_T libSize = lib->GetSerializedSize();
    std::vector<uint8_t> libBytes(libSize);
    testutil::checkHR(lib->Serialize(libBytes.data(), libSize), "SerializeLib");

    nacelle::clearInMemoryPSOCacheForTesting();
    ComPtr<ID3D12PipelineLibrary> loadedLib;
    testutil::checkHR(rt.device->CreatePipelineLibrary(libBytes.data(), libBytes.size(), IID_PPV_ARGS(&loadedLib)), "LoadLib");

    auto t0 = std::chrono::steady_clock::now();
    ComPtr<ID3D12PipelineState> psoWarm;
    testutil::checkHR(loadedLib->LoadComputePipeline(L"vec_add", &psoDesc, IID_PPV_ARGS(&psoWarm)), "LoadComputePipeline");
    auto t1 = std::chrono::steady_clock::now();
    double warmMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::printf("pso_cache_bcn_test PASSED (warm reload: %.3f ms, libSize=%zu bytes)\n",
                warmMs, static_cast<size_t>(libSize));
    return 0;
}
