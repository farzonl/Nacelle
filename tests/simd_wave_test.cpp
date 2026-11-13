#include "test_common.h"
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

void runWaveWidth(uint32_t width) {
    nacelle::waitForAllAsyncJITTasks();
    nacelle::setSimdWidthOverride(width);
    nacelle::clearInMemoryPSOCacheForTesting();

    testutil::TestRuntime rt;
    rt.init();

    D3D12_ROOT_PARAMETER rp[2]{};
    rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    rp[0].Descriptor.ShaderRegister = 0;
    rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    rp[1].Descriptor.ShaderRegister = 0;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 2;
    rsDesc.pParameters = rp;
    ComPtr<ID3DBlob> rsBlob;
    testutil::checkHR(NacelleSerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1_0, &rsBlob, nullptr), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(rt.device->CreateRootSignature(0, rsBlob->GetBufferPointer(), rsBlob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");

    auto csBytes = testutil::loadShaderBytes("wave_intrinsics.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.CS = {csBytes.data(), csBytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)), "CreateComputePSO");

    constexpr UINT N = 32;
    std::vector<uint32_t> in(N);
    for (UINT i = 0; i < N; ++i) in[i] = i + 1;

    auto bIn = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT);
    auto bOut = rt.createBuffer(N * 4 * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    rt.uploadBuffer(bIn.Get(), in.data(), N * 4);

    rt.resetCmd();
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootShaderResourceView(0, bIn->GetGPUVirtualAddress());
    rt.cmdList->SetComputeRootUnorderedAccessView(1, bOut->GetGPUVirtualAddress());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    auto raw = rt.readbackBuffer(bOut.Get(), N * 4 * 4);
    std::vector<uint32_t> out(N * 4);
    std::memcpy(out.data(), raw.data(), raw.size());

    // Verify outputs are non-zero and consistent with wave width
    uint32_t laneCount0 = out[0];
    assert(laneCount0 == width);
    std::printf("  simd_wave_test (W=%u): WaveGetLaneCount=%u, WaveActiveSum[0]=%u\n",
                width, laneCount0, out[1]);
}

} // namespace

int main() {
    if (const char* envW = std::getenv("NACELLE_SIMD_WIDTH")) {
        uint32_t w = static_cast<uint32_t>(std::stoul(envW));
        runWaveWidth(w);
    } else {
        runWaveWidth(4);
        runWaveWidth(8);
        runWaveWidth(16);
    }
    std::printf("simd_wave_test PASSED\n");
    return 0;
}
