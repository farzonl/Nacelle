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
