// Tiled / Reserved resource test shaders

// 1. Exact reproduction of offload-test-suite PartiallyMappedResources.test
StructuredBuffer<int4> X : register(t0);
StructuredBuffer<int4> Y : register(t1);
RWStructuredBuffer<int> Out : register(u0);
RWStructuredBuffer<uint> CAFM : register(u1);

[numthreads(1, 1, 1)]
void PartiallyMappedCS() {
    uint status = 0;
    int4 x0 = X.Load(0, status);
    CAFM[0] = CheckAccessFullyMapped(status) ? 1u : 0u;
    Out[0] = CAFM[0] ? x0.x : 9003;

    int4 x1 = X.Load(4100, status);
    CAFM[1] = CheckAccessFullyMapped(status) ? 1u : 0u;
    Out[1] = CAFM[1] ? x1.x : 9003;

    int4 y0 = Y.Load(0, status);
    CAFM[2] = CheckAccessFullyMapped(status) ? 1u : 0u;
    Out[2] = CAFM[2] ? y0.x : 9003;

    int4 y1 = Y.Load(4100, status);
    CAFM[3] = CheckAccessFullyMapped(status) ? 1u : 0u;
    Out[3] = CAFM[3] ? y1.x : 9003;

    // Unmapped tiles must read 0 per D3D12 Tier 1/2/3 spec
    Out[4] = x1.x;
    Out[5] = y0.w;
}

// 2. Divergent SIMD per-lane loads, stores, atomics, and CBV over reserved buffers
cbuffer TiledParams : register(b0) {
    uint addend;
    uint pad0;
    uint pad1;
    uint pad2;
};

ByteAddressBuffer SrcRaw : register(t0);
RWByteAddressBuffer DstRaw : register(u0);
RWStructuredBuffer<uint> OutVals : register(u1);
RWStructuredBuffer<uint> OutStatus : register(u2);

[numthreads(64, 1, 1)]
void DivergentTiledBufferCS(uint3 dtid : SV_DispatchThreadID) {
    uint lane = dtid.x;
    uint tileIdx = lane / 16u;
    uint inTileIdx = lane & 15u;
    uint byteAddr = tileIdx * 65536u + inTileIdx * 16u;

    uint status = 0;
    uint4 loaded = SrcRaw.Load4(byteAddr, status);
    bool mapped = CheckAccessFullyMapped(status);
    OutStatus[lane] = mapped ? 1u : 0u;
    OutVals[lane] = loaded.x + addend;

    // Mapped tiles receive stores & atomics; NULL-mapped tiles discard writes
    DstRaw.Store(byteAddr, lane + 1000u);
    uint prev = 0;
    DstRaw.InterlockedAdd(tileIdx * 65536u + 512u, 1u, prev);
}

// 3. Reserved Texture2D Load + SampleLevel residency feedback & RWTexture2D writes
Texture2D<float4> TiledTex : register(t0);
SamplerState PointClampSamp : register(s0);
RWTexture2D<float4> TiledUAV : register(u0);
RWStructuredBuffer<float4> OutTexVals : register(u1);
RWStructuredBuffer<uint> OutTexStatus : register(u2);

[numthreads(4, 1, 1)]
void TiledTextureCS(uint3 dtid : SV_DispatchThreadID) {
    uint q = dtid.x; // quadrant 0=(0,0), 1=(1,0), 2=(0,1), 3=(1,1)
    int2 coord = int2((q & 1u) * 64u + 8u, (q >> 1u) * 64u + 8u);

    uint loadStatus = 0;
    float4 vLoad = TiledTex.Load(int3(coord, 0), loadStatus);
    bool loadMapped = CheckAccessFullyMapped(loadStatus);

    float2 uv = (float2(coord) + 0.5f) / 128.0f;
    uint sampStatus = 0;
    float4 vSamp = TiledTex.SampleLevel(PointClampSamp, uv, 0.0f, int2(0, 0), sampStatus);
    bool sampMapped = CheckAccessFullyMapped(sampStatus);

    OutTexVals[q * 2u + 0u] = vLoad;
    OutTexVals[q * 2u + 1u] = vSamp;
    OutTexStatus[q * 2u + 0u] = loadMapped ? 1u : 0u;
    OutTexStatus[q * 2u + 1u] = sampMapped ? 1u : 0u;

    TiledUAV[coord] = float4(float(q + 1u) * 10.0f, 1.0f, 2.0f, 3.0f);
}
