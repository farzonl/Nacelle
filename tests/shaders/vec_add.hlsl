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
