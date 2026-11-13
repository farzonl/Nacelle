#include "test_common.h"
#include "../src/compiler/jit_engine.h"
#include "../src/compiler/shader_compiler.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint32_t floatBits(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits;
}

void runInterlockedBarrierForWidth(uint32_t simdWidth) {
    nacelle::waitForAllAsyncJITTasks();
    nacelle::setSimdWidthOverride(simdWidth);
    nacelle::clearInMemoryPSOCacheForTesting();

    testutil::TestRuntime rt;
    rt.init();

    // Verify 64-bit atomic capability reporting.
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 opt9{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS9, &opt9, sizeof(opt9)),
                      "CheckFeatureSupport(OPTIONS9)");
    assert(opt9.AtomicInt64OnTypedResourceSupported == TRUE);
    assert(opt9.AtomicInt64OnGroupSharedSupported == TRUE);

    D3D12_FEATURE_DATA_D3D12_OPTIONS11 opt11{};
    testutil::checkHR(rt.device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS11, &opt11, sizeof(opt11)),
                      "CheckFeatureSupport(OPTIONS11)");
    assert(opt11.AtomicInt64OnDescriptorHeapResourceSupported == TRUE);

    // Single UAV descriptor table covering u0..u4.
    D3D12_DESCRIPTOR_RANGE uavRange{};
    uavRange.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uavRange.NumDescriptors = 5;
    uavRange.BaseShaderRegister = 0;
    uavRange.RegisterSpace = 0;
    uavRange.OffsetInDescriptorsFromTableStart = 0;

    D3D12_ROOT_PARAMETER rp{};
    rp.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    rp.DescriptorTable.NumDescriptorRanges = 1;
    rp.DescriptorTable.pDescriptorRanges = &uavRange;
    rp.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC rsDesc{};
    rsDesc.NumParameters = 1;
    rsDesc.pParameters = &rp;
    auto rs = rt.createRootSig(rsDesc);

    auto csBytes = testutil::loadShaderBytes("interlocked_barrier.dxil");
    D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc{};
    psoDesc.pRootSignature = rs.Get();
    psoDesc.CS = {csBytes.data(), csBytes.size()};
    ComPtr<ID3D12PipelineState> pso;
    testutil::checkHR(rt.device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&pso)),
                      "CreateComputePSO(interlocked_barrier)");

    constexpr UINT kNumThreads = 16;
    constexpr UINT kOut64Count = 32;
    constexpr UINT kOutS64Count = 8;
    constexpr UINT kRawBytes = 64;
    constexpr UINT kFloatCount = 16;
    constexpr UINT kTexWidth = 16;
    constexpr UINT kTexHeight = 2;

    auto bufOut64 = rt.createBuffer(kOut64Count * sizeof(uint64_t), D3D12_HEAP_TYPE_DEFAULT,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufOutS64 = rt.createBuffer(kOutS64Count * sizeof(int64_t), D3D12_HEAP_TYPE_DEFAULT,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufRaw = rt.createBuffer(kRawBytes, D3D12_HEAP_TYPE_DEFAULT,
                                  D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    auto bufFloat = rt.createBuffer(kFloatCount * sizeof(float), D3D12_HEAP_TYPE_DEFAULT,
                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

    D3D12_HEAP_PROPERTIES defaultHeap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC texDesc = CD3DX12_RESOURCE_DESC::Tex2D(
        DXGI_FORMAT_R32_UINT, kTexWidth, kTexHeight, 1, 1, 1, 0,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    ComPtr<ID3D12Resource> texUav;
    testutil::checkHR(rt.device->CreateCommittedResource(&defaultHeap, D3D12_HEAP_FLAG_NONE, &texDesc,
                                                         D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                         IID_PPV_ARGS(&texUav)),
                      "CreateTex2D(R32_UINT)");

    D3D12_DESCRIPTOR_HEAP_DESC heapDesc{};
    heapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heapDesc.NumDescriptors = 5;
    heapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> uavHeap;
    testutil::checkHR(rt.device->CreateDescriptorHeap(&heapDesc, IID_PPV_ARGS(&uavHeap)), "CreateUAVHeap");

    const UINT inc = rt.device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpuStart = uavHeap->GetCPUDescriptorHandleForHeapStart();

    // u0: RWStructuredBuffer<uint64_t>
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = kOut64Count;
        uav.Buffer.StructureByteStride = sizeof(uint64_t);
        rt.device->CreateUnorderedAccessView(bufOut64.Get(), nullptr, &uav, {cpuStart.ptr + 0 * inc});
    }
    // u1: RWByteAddressBuffer
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_TYPELESS;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = kRawBytes / 4;
        uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
        rt.device->CreateUnorderedAccessView(bufRaw.Get(), nullptr, &uav, {cpuStart.ptr + 1 * inc});
    }
    // u2: RWBuffer<float>
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_FLOAT;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = kFloatCount;
        rt.device->CreateUnorderedAccessView(bufFloat.Get(), nullptr, &uav, {cpuStart.ptr + 2 * inc});
    }
    // u3: RWTexture2D<uint>
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_R32_UINT;
        uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
        uav.Texture2D.MipSlice = 0;
        rt.device->CreateUnorderedAccessView(texUav.Get(), nullptr, &uav, {cpuStart.ptr + 3 * inc});
    }
    // u4: RWStructuredBuffer<int64_t>
    {
        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.FirstElement = 0;
        uav.Buffer.NumElements = kOutS64Count;
        uav.Buffer.StructureByteStride = sizeof(int64_t);
        rt.device->CreateUnorderedAccessView(bufOutS64.Get(), nullptr, &uav, {cpuStart.ptr + 4 * inc});
    }

    rt.resetCmd();
    ID3D12DescriptorHeap* heaps[] = {uavHeap.Get()};
    rt.cmdList->SetDescriptorHeaps(1, heaps);
    rt.cmdList->SetComputeRootSignature(rs.Get());
    rt.cmdList->SetPipelineState(pso.Get());
    rt.cmdList->SetComputeRootDescriptorTable(0, uavHeap->GetGPUDescriptorHandleForHeapStart());
    rt.cmdList->Dispatch(1, 1, 1);
    rt.executeAndWait();

    // 1. Check gOut64 results (64-bit resource atomics, groupshared summaries, and per-lane loop spills).
    auto out64Bytes = rt.readbackBuffer(bufOut64.Get(), kOut64Count * sizeof(uint64_t));
    std::vector<uint64_t> out64(kOut64Count);
    std::memcpy(out64.data(), out64Bytes.data(), out64Bytes.size());

    constexpr uint64_t kThreadSum = 136ull; // sum_{tid=0..15} (tid + 1)
    assert(out64[0] == ((kThreadSum << 32) | kThreadSum));
    assert(out64[1] == 0x99AABBCCDDEEFF00ull);
    assert(out64[2] == 0xCAFEBABE00000002ull);
    assert(out64[3] == kThreadSum * 4ull);
    assert(out64[4] == (kThreadSum * 4ull) * 0x100000001ull);
    assert(out64[5] == static_cast<uint64_t>(floatBits(7.25f)));
    assert(out64[6] == 0x22222222ull);

    // Per-lane accumulator across 4 barrier-split loop iterations:
    // (tid + 10) + (136 + 272 + 408 + 544) = tid + 1370.
    for (UINT tid = 0; tid < kNumThreads; ++tid) {
        assert(out64[8 + tid] == static_cast<uint64_t>(tid + 1370u));
    }
    assert(out64[24] == 0xEEEEEEEEEEEEEEEEull);
    assert(out64[25] == 0x1111111111111111ull);
    assert(out64[26] == 0xBBBBBBBBBBBBBBBBull);
    assert(out64[27] == 5ull * 0x100000001ull);
    assert(out64[28] == 20ull * 0x100000001ull);

    auto outS64Bytes = rt.readbackBuffer(bufOutS64.Get(), kOutS64Count * sizeof(int64_t));
    std::vector<int64_t> outS64(kOutS64Count);
    std::memcpy(outS64.data(), outS64Bytes.data(), outS64Bytes.size());
    assert(static_cast<uint64_t>(outS64[0]) == 0xFFFFFFFFFFFF0000ull);
    assert(static_cast<uint64_t>(outS64[1]) == 0x0000FFFF00000000ull);
    assert(outS64[2] == 0x4444444444444444ll);
    assert(outS64[3] == -50ll);
    assert(outS64[4] == 10ll);

    // 2. Check gRaw results (divergent 32-bit InterlockedAdd + 64-bit Interlocked{Add,And,Or,Xor,Min,Max}64).
    auto rawBytes = rt.readbackBuffer(bufRaw.Get(), kRawBytes);
    uint32_t rawEvenCount = 0;
    uint64_t rawSum64 = 0, rawAnd64 = 0, rawOr64 = 0, rawXor64 = 0, rawMin64 = 0, rawMax64 = 0;
    std::memcpy(&rawEvenCount, rawBytes.data() + 0, sizeof(rawEvenCount));
    std::memcpy(&rawSum64, rawBytes.data() + 8, sizeof(rawSum64));
    std::memcpy(&rawAnd64, rawBytes.data() + 16, sizeof(rawAnd64));
    std::memcpy(&rawOr64, rawBytes.data() + 24, sizeof(rawOr64));
    std::memcpy(&rawXor64, rawBytes.data() + 32, sizeof(rawXor64));
    std::memcpy(&rawMin64, rawBytes.data() + 40, sizeof(rawMin64));
    std::memcpy(&rawMax64, rawBytes.data() + 48, sizeof(rawMax64));
    assert(rawEvenCount == 8u);
    assert(rawSum64 == static_cast<uint64_t>(kNumThreads) * 0x100000002ull);
    assert(rawAnd64 == 0xFFFFFFFFFFFF0000ull);
    assert(rawOr64 == 0x0000FFFF00000000ull);
    assert(rawXor64 == 0xFFFF0000FFFFFFFFull);
    assert(rawMin64 == 7ull * 0x100000001ull);
    assert(rawMax64 == 22ull * 0x100000001ull);

    // 3. Check gFloatBuf results (float InterlockedExchange + InterlockedCompareExchangeFloatBitwise).
    auto floatBytes = rt.readbackBuffer(bufFloat.Get(), kFloatCount * sizeof(float));
    std::vector<float> floats(kFloatCount);
    std::memcpy(floats.data(), floatBytes.data(), floatBytes.size());
    assert(floatBits(floats[0]) == floatBits(77.0f));
    assert(floatBits(floats[1]) == floatBits(9.0f));
    assert(floatBits(floats[2]) == floatBits(4.5f));
    assert(floatBits(floats[3]) == floatBits(1.5f));
    assert(floatBits(floats[4]) == floatBits(1.25f));
    assert(floatBits(floats[5]) == floatBits(9.0f));

    // 4. Check gTex results (RWTexture2D<uint> lossless 0xFFFFFFFFu roundtrip + InterlockedCompareStore).
    constexpr UINT kRowPitch = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT;
    auto texRb = rt.createBuffer(kRowPitch * kTexHeight, D3D12_HEAP_TYPE_READBACK);
    rt.resetCmd();
    D3D12_TEXTURE_COPY_LOCATION copySrc{texUav.Get(), D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {0}};
    D3D12_TEXTURE_COPY_LOCATION copyDst{};
    copyDst.pResource = texRb.Get();
    copyDst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    copyDst.PlacedFootprint.Footprint = {DXGI_FORMAT_R32_UINT, kTexWidth, kTexHeight, 1, kRowPitch};
    rt.cmdList->CopyTextureRegion(&copyDst, 0, 0, 0, &copySrc, nullptr);
    rt.executeAndWait();

    void* rbPtr = nullptr;
    testutil::checkHR(texRb->Map(0, nullptr, &rbPtr), "MapTexReadback");
    const auto* row0 = reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(rbPtr) + 0 * kRowPitch);
    const auto* row1 = reinterpret_cast<const uint32_t*>(static_cast<const uint8_t*>(rbPtr) + 1 * kRowPitch);
    for (UINT tid = 0; tid < kNumThreads; ++tid) {
        assert(row0[tid] == 100u + tid);
        assert(row1[tid] == 0xFFFFFFFFu);
    }
    texRb->Unmap(0, nullptr);

    std::printf("  interlocked_barrier_test (W=%u): OK\n", simdWidth);
}

} // namespace

int main() {
    if (const char* envW = std::getenv("NACELLE_SIMD_WIDTH")) {
        runInterlockedBarrierForWidth(static_cast<uint32_t>(std::stoul(envW)));
    } else {
        runInterlockedBarrierForWidth(4);
        runInterlockedBarrierForWidth(8);
        runInterlockedBarrierForWidth(0);
    }
    nacelle::setSimdWidthOverride(0);
    std::printf("interlocked_barrier_test PASSED\n");
    return 0;
}
