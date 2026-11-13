// Compute regression shader covering SM 6.6 resource binding, 32-bit/64-bit/float
// Interlocked* intrinsics, RWTexture2D<uint> bitwise load/store, divergent memory
// fences, and barriers inside structured loops with live per-lane values.

RWStructuredBuffer<uint64_t> gOut64    : register(u0);
RWByteAddressBuffer          gRaw      : register(u1);
RWBuffer<float>              gFloatBuf : register(u2);
RWTexture2D<uint>            gTex      : register(u3);

groupshared uint     gsSum32;
groupshared uint64_t gsSum64;
groupshared float    gsFloatSlot;
groupshared uint     gsCompareSlot;

[numthreads(16, 1, 1)]
void CSMain(uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;

    if (tid == 0u) {
        gsSum32 = 0u;
        gsSum64 = 0ull;
        gsFloatSlot = 1.5f;
        gsCompareSlot = 0x11111111u;

        gOut64[0] = 0ull;
        gOut64[1] = 0x1122334455667788ull;
        gOut64[2] = 0xCAFEBABE00000001ull;

        gRaw.Store(0, 0u);
        gRaw.Store<uint64_t>(8, 0ull);

        gFloatBuf[0] = -5.0f;
        gFloatBuf[1] = 9.0f;
        gFloatBuf[2] = 1.25f;
    }

    gTex[uint2(tid, 0u)] = 0xFFFFFFFFu;
    AllMemoryBarrierWithGroupSync();

    // Structured loop with barriers inside the loop body and a live per-lane
    // accumulator carried across barriers.
    uint localAcc = tid + 10u;
    [loop]
    for (uint iter = 0u; iter < 4u; ++iter) {
        InterlockedAdd(gsSum32, tid + 1u);
        InterlockedAdd(gsSum64, uint64_t(tid + 1u) * 0x100000001ull);
        GroupMemoryBarrierWithGroupSync();

        localAcc += gsSum32;
        DeviceMemoryBarrierWithGroupSync();
    }

    // Non-GroupSync memory fence inside divergent control flow.
    if ((tid & 1u) == 0u) {
        gRaw.InterlockedAdd(0, 1u);
        DeviceMemoryBarrier();
    }

    // Float InterlockedExchange and InterlockedCompareExchangeFloatBitwise.
    if (tid == 3u) {
        float prevGsF;
        InterlockedExchange(gsFloatSlot, 2.5f, prevGsF);
        gFloatBuf[3] = prevGsF;
    }
    GroupMemoryBarrierWithGroupSync();

    float origGsCmp;
    InterlockedCompareExchangeFloatBitwise(gsFloatSlot, 2.5f, 7.25f, origGsCmp);

    float origBufMatch;
    float origBufNoMatch;
    InterlockedCompareExchangeFloatBitwise(gFloatBuf[0], -5.0f, 77.0f, origBufMatch);
    InterlockedCompareExchangeFloatBitwise(gFloatBuf[1], 10.0f, 0.0f, origBufNoMatch);
    if (tid == 0u) {
        float prevBufF;
        InterlockedExchange(gFloatBuf[2], 4.5f, prevBufF);
        gFloatBuf[4] = prevBufF;
        gFloatBuf[5] = origBufNoMatch;
    }

    // 32-bit and 64-bit InterlockedCompareStore, InterlockedCompareExchange, and InterlockedAdd64.
    InterlockedCompareStore(gsCompareSlot, 0x11111111u, 0x22222222u);
    InterlockedAdd(gOut64[0], uint64_t(tid + 1u) * 0x100000000ull + uint64_t(tid + 1u));
    InterlockedCompareStore(gOut64[1], 0x1122334455667788ull, 0x99AABBCCDDEEFF00ull);

    uint64_t prev64;
    InterlockedCompareExchange(gOut64[2], 0xCAFEBABE00000001ull, 0xCAFEBABE00000002ull, prev64);

    uint64_t prevRaw64;
    gRaw.InterlockedAdd64(8, 0x100000002ull, prevRaw64);

    // RWTexture2D<uint> lossless 0xFFFFFFFFu load/store + InterlockedCompareStore.
    uint texBefore = gTex[uint2(tid, 0u)];
    gTex[uint2(tid, 1u)] = texBefore;
    DeviceMemoryBarrierWithGroupSync();
    InterlockedCompareStore(gTex[uint2(tid, 0u)], 0xFFFFFFFFu, 100u + tid);

    AllMemoryBarrierWithGroupSync();

    if (tid == 0u) {
        gOut64[3] = gsSum32;
        gOut64[4] = gsSum64;
        gOut64[5] = asuint(gsFloatSlot);
        gOut64[6] = gsCompareSlot;
    }
    gOut64[8u + tid] = localAcc;
}
