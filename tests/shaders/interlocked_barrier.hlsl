// Compute regression shader covering SM 6.6 resource binding, 32-bit/64-bit/float
// Interlocked* intrinsics, RWTexture2D<uint> bitwise load/store, divergent memory
// fences, and barriers inside structured loops with live per-lane values.

RWStructuredBuffer<uint64_t> gOut64    : register(u0);
RWByteAddressBuffer          gRaw      : register(u1);
RWBuffer<float>              gFloatBuf : register(u2);
RWTexture2D<uint>            gTex      : register(u3);
RWStructuredBuffer<int64_t>  gOutS64   : register(u4);

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
        gOut64[24] = 0xFFFFFFFFFFFFFFFFull;
        gOut64[25] = 0ull;
        gOut64[26] = 0xAAAAAAAAAAAAAAAAull;
        gOut64[27] = 0xFFFFFFFFFFFFFFFFull;
        gOut64[28] = 0ull;

        gOutS64[0] = -1ll;
        gOutS64[1] = 0ll;
        gOutS64[2] = 0x5555555555555555ll;
        gOutS64[3] = 1000ll;
        gOutS64[4] = -1000ll;

        gRaw.Store(0, 0u);
        gRaw.Store<uint64_t>(8, 0ull);
        gRaw.Store<uint64_t>(16, 0xFFFFFFFFFFFFFFFFull);
        gRaw.Store<uint64_t>(24, 0ull);
        gRaw.Store<uint64_t>(32, 0xFFFF0000FFFF0000ull);
        gRaw.Store<uint64_t>(40, 0xFFFFFFFFFFFFFFFFull);
        gRaw.Store<uint64_t>(48, 0ull);

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

    // 64-bit resource InterlockedAnd / Or / Xor / Min / Max (unsigned, signed, and raw buffer).
    uint64_t nibBit = 1ull << (tid * 4u);
    InterlockedAnd(gOut64[24], ~nibBit);
    InterlockedOr(gOut64[25], nibBit);
    InterlockedXor(gOut64[26], nibBit);
    InterlockedMin(gOut64[27], uint64_t(tid + 5u) * 0x100000001ull);
    InterlockedMax(gOut64[28], uint64_t(tid + 5u) * 0x100000001ull);

    int64_t prevS64;
    InterlockedAnd(gOutS64[0], ~int64_t(1ull << tid), prevS64);
    InterlockedOr(gOutS64[1], int64_t(1ull << (tid + 32u)), prevS64);
    InterlockedXor(gOutS64[2], int64_t(nibBit), prevS64);
    InterlockedMin(gOutS64[3], int64_t(tid) - 50ll, prevS64);
    InterlockedMax(gOutS64[4], int64_t(tid) - 5ll, prevS64);

    uint64_t prevRawBit64;
    gRaw.InterlockedAnd64(16, ~(1ull << tid));
    gRaw.InterlockedOr64(24, 1ull << (tid + 32u), prevRawBit64);
    gRaw.InterlockedXor64(32, 1ull << tid, prevRawBit64);
    gRaw.InterlockedMin64(40, uint64_t(tid + 7u) * 0x100000001ull);
    gRaw.InterlockedMax64(48, uint64_t(tid + 7u) * 0x100000001ull, prevRawBit64);

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
