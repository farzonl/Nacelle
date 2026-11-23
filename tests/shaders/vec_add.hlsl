cbuffer Params : register(b0) {
    float scale;
    float bias;
    uint count;
};

StructuredBuffer<float> bufA : register(t0);
StructuredBuffer<float> bufB : register(t1);
RWStructuredBuffer<float> bufOut : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x < count) {
        bufOut[dtid.x] = bufA[dtid.x] * scale + bufB[dtid.x] + bias;
    }
}

[numthreads(8, 2, 2)]
void CSVolumeMain(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex) {
    uint i = ((gid.z * 2 + gid.y) * 4 + gid.x) * 32 + gi;
    if (i < count) {
        bufOut[i] = bufA[i] * scale + bufB[i] + bias;
    }

}

groupshared float groupInput[64];
[numthreads(64, 1, 1)]
void CSGroupMain(uint3 dtid : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    groupInput[gi] = bufA[dtid.x];
    GroupMemoryBarrierWithGroupSync();
    bufOut[dtid.x] = groupInput[63 - gi] * scale + bufB[dtid.x] + bias;
}

[numthreads(64, 1, 1)]
void CSAtomicMain(uint3 dtid : SV_DispatchThreadID) {
    float previous;
    InterlockedCompareExchangeFloatBitwise(bufOut[dtid.x], 0.0f, 1.0f, previous);
    bufOut[dtid.x] = bufA[dtid.x] * scale + bufB[dtid.x] + bias;
}

ByteAddressBuffer rawA : register(t0);
ByteAddressBuffer rawB : register(t1);
RWByteAddressBuffer rawOut : register(u0);
[numthreads(64, 1, 1)]
void CSRawMain(uint3 dtid : SV_DispatchThreadID) {
    rawOut.Store(dtid.x * 4, asuint(asfloat(rawA.Load(dtid.x * 4)) * scale +
        asfloat(rawB.Load(dtid.x * 4)) + bias));
}

Buffer<float> typedA : register(t0);
Buffer<float> typedB : register(t1);
RWBuffer<float> typedOut : register(u0);
[numthreads(64, 1, 1)]
void CSTypedMain(uint3 dtid : SV_DispatchThreadID) {
    typedOut[dtid.x] = typedA[dtid.x] * scale + typedB[dtid.x] + bias;
}
