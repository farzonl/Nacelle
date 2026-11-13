#include "test_common.h"

#include <cmath>
#include <cstring>
#include <vector>

namespace {

ComPtr<ID3D12RootSignature> createComputeRS(ID3D12Device* dev, UINT numRootParams, const D3D12_ROOT_PARAMETER* params) {
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = numRootParams;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    ComPtr<ID3DBlob> err;
    testutil::checkHR(NacelleSerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1_0, &blob, &err), "SerializeRS");
    ComPtr<ID3D12RootSignature> rs;
    testutil::checkHR(dev->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(), IID_PPV_ARGS(&rs)), "CreateRS");
    return rs;
}

ComPtr<ID3D12PipelineState> createComputePSO(ID3D12Device* dev, ID3D12RootSignature* rs, const char* shaderFile) {
    auto bytes = testutil::loadShaderBytes(shaderFile);
    D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};
    desc.pRootSignature = rs;
    desc.CS = {bytes.data(), bytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(dev->CreateComputePipelineState(&desc, IID_PPV_ARGS(&pso)), shaderFile);
    return pso;
}

} // namespace

int main() {
    testutil::TestRuntime rt;
    rt.init();

    // 1. vec_add.dxil: SRV t0, SRV t1, UAV u0
    {
        D3D12_ROOT_PARAMETER rp[3]{};
        rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[0].Descriptor.ShaderRegister = 0;
        rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[1].Descriptor.ShaderRegister = 1;
        rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[2].Descriptor.ShaderRegister = 0;

        auto rs = createComputeRS(rt.device.Get(), 3, rp);
        auto pso = createComputePSO(rt.device.Get(), rs.Get(), "vec_add.dxil");

        constexpr UINT N = 1024;
        std::vector<float> a(N), b(N);
        for (UINT i = 0; i < N; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(i) * 2.0f + 1.0f;
        }
        auto bufA = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT);
        auto bufB = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT);
        auto bufC = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        rt.uploadBuffer(bufA.Get(), a.data(), N * 4);
        rt.uploadBuffer(bufB.Get(), b.data(), N * 4);

        rt.resetCmd();
        rt.cmdList->SetComputeRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->SetComputeRootShaderResourceView(0, bufA->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootShaderResourceView(1, bufB->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootUnorderedAccessView(2, bufC->GetGPUVirtualAddress());
        rt.cmdList->Dispatch(N / 64, 1, 1);
        rt.executeAndWait();

        auto outRaw = rt.readbackBuffer(bufC.Get(), N * 4);
        std::vector<float> c(N);
        std::memcpy(c.data(), outRaw.data(), N * 4);
        for (UINT i = 0; i < N; ++i) {
            assert(std::fabs(c[i] - (a[i] + b[i])) < 1e-4f);
        }
    }

    // 2. matmul.dxil: CBV b0, SRV t0, SRV t1, UAV u0
    {
        D3D12_ROOT_PARAMETER rp[4]{};
        rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
        rp[0].Descriptor.ShaderRegister = 0;
        rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[1].Descriptor.ShaderRegister = 0;
        rp[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[2].Descriptor.ShaderRegister = 1;
        rp[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[3].Descriptor.ShaderRegister = 0;

        auto rs = createComputeRS(rt.device.Get(), 4, rp);
        auto pso = createComputePSO(rt.device.Get(), rs.Get(), "matmul.dxil");

        constexpr UINT M = 16, N = 16, K = 16;
        uint32_t dims[4] = {M, N, K, 0};
        std::vector<float> matA(M * K), matB(K * N);
        for (UINT r = 0; r < M; ++r)
            for (UINT c = 0; c < K; ++c)
                matA[r * K + c] = static_cast<float>((r + c) % 5 + 1);
        for (UINT r = 0; r < K; ++r)
            for (UINT c = 0; c < N; ++c)
                matB[r * N + c] = (r == c) ? 2.0f : 0.0f;

        auto cb = rt.createBuffer(256, D3D12_HEAP_TYPE_DEFAULT);
        auto bA = rt.createBuffer(M * K * 4, D3D12_HEAP_TYPE_DEFAULT);
        auto bB = rt.createBuffer(K * N * 4, D3D12_HEAP_TYPE_DEFAULT);
        auto bC = rt.createBuffer(M * N * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        rt.uploadBuffer(cb.Get(), dims, sizeof(dims));
        rt.uploadBuffer(bA.Get(), matA.data(), M * K * 4);
        rt.uploadBuffer(bB.Get(), matB.data(), K * N * 4);

        rt.resetCmd();
        rt.cmdList->SetComputeRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->SetComputeRootConstantBufferView(0, cb->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootShaderResourceView(1, bA->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootShaderResourceView(2, bB->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootUnorderedAccessView(3, bC->GetGPUVirtualAddress());
        rt.cmdList->Dispatch(2, 2, 1);
        rt.executeAndWait();

        auto outRaw = rt.readbackBuffer(bC.Get(), M * N * 4);
        std::vector<float> matC(M * N);
        std::memcpy(matC.data(), outRaw.data(), M * N * 4);
        for (UINT i = 0; i < M * N; ++i) {
            assert(std::fabs(matC[i] - matA[i] * 2.0f) < 1e-4f);
        }
    }

    // 3. divergent_cf_atomics.dxil: UAV u0 (outAtomics), UAV u1 (outValues)
    {
        D3D12_ROOT_PARAMETER rp[2]{};
        rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[0].Descriptor.ShaderRegister = 0;
        rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[1].Descriptor.ShaderRegister = 1;

        auto rs = createComputeRS(rt.device.Get(), 2, rp);
        auto pso = createComputePSO(rt.device.Get(), rs.Get(), "divergent_cf_atomics.dxil");

        constexpr UINT N = 128;
        std::vector<uint32_t> expectedOut(N, 0), zeros(4, 0);
        uint32_t expectedMax = 0, expectedSum = 0;
        for (UINT tid = 0; tid < N; ++tid) {
            uint32_t acc = 0;
            if ((tid & 1u) == 0u) {
                uint32_t n = (tid & 7u) + 1u;
                for (uint32_t i = 0; i < n; ++i) acc += (i + 1u) * 2u;
                if ((tid & 2u) != 0u) acc += 100u;
            } else {
                uint32_t n = (tid % 5u) + 2u;
                for (uint32_t i = 0; i < n; ++i) acc += (i * 3u + 1u);
            }
            expectedOut[tid] = acc;
            if (acc > expectedMax) expectedMax = acc;
            expectedSum += acc;
        }
        auto bAtom = rt.createBuffer(16, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        auto bOut = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        rt.uploadBuffer(bAtom.Get(), zeros.data(), 16);

        rt.resetCmd();
        rt.cmdList->SetComputeRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->SetComputeRootUnorderedAccessView(0, bAtom->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootUnorderedAccessView(1, bOut->GetGPUVirtualAddress());
        rt.cmdList->Dispatch(N / 64, 1, 1);
        rt.executeAndWait();

        auto atomRaw = rt.readbackBuffer(bAtom.Get(), 16);
        uint32_t atomVals[4]{};
        std::memcpy(atomVals, atomRaw.data(), 16);
        assert(atomVals[0] == N);
        assert(atomVals[1] == expectedMax);
        assert(atomVals[2] == expectedSum);

        auto outRaw = rt.readbackBuffer(bOut.Get(), N * 4);
        std::vector<uint32_t> outVals(N);
        std::memcpy(outVals.data(), outRaw.data(), N * 4);
        for (UINT tid = 0; tid < N; ++tid) {
            assert(outVals[tid] == expectedOut[tid]);
        }
    }

    // 4. prefix_sum_groupshared.dxil: SRV t0, UAV u0
    {
        D3D12_ROOT_PARAMETER rp[2]{};
        rp[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
        rp[0].Descriptor.ShaderRegister = 0;
        rp[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rp[1].Descriptor.ShaderRegister = 0;

        auto rs = createComputeRS(rt.device.Get(), 2, rp);
        auto pso = createComputePSO(rt.device.Get(), rs.Get(), "prefix_sum_groupshared.dxil");

        constexpr UINT N = 64;
        std::vector<uint32_t> in(N, 1);
        auto bIn = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT);
        auto bOut = rt.createBuffer(N * 4, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        rt.uploadBuffer(bIn.Get(), in.data(), N * 4);

        rt.resetCmd();
        rt.cmdList->SetComputeRootSignature(rs.Get());
        rt.cmdList->SetPipelineState(pso.Get());
        rt.cmdList->SetComputeRootShaderResourceView(0, bIn->GetGPUVirtualAddress());
        rt.cmdList->SetComputeRootUnorderedAccessView(1, bOut->GetGPUVirtualAddress());
        rt.cmdList->Dispatch(1, 1, 1);
        rt.executeAndWait();

        auto outRaw = rt.readbackBuffer(bOut.Get(), N * 4);
        std::vector<uint32_t> out(N);
        std::memcpy(out.data(), outRaw.data(), N * 4);
        for (UINT i = 0; i < N; ++i) {
            assert(out[i] == i + 1);
        }
    }

    std::printf("compute_dispatch_test PASSED (vec_add, matmul, divergent_cf_atomics, prefix_sum_groupshared)\n");
    return 0;
}
