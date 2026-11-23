cbuffer Params : register(b0) {
    float scale;
    float bias;
    uint count;
};
StructuredBuffer<float> bufA : register(t0);
StructuredBuffer<float> bufB : register(t1);
RWStructuredBuffer<float> bufOut : register(u0);
groupshared float sharedValues[64];

float addOne(float value) {
    return value + 1.0;
}

[numthreads(64, 1, 1)]
void CSMain(uint3 tid : SV_DispatchThreadID, uint index : SV_GroupIndex) {
    float value = bufA[tid.x] * scale + bufB[tid.x];
    if ((tid.x & 1) != 0)
        value += 1.0;
    else
        value -= 1.0;
    [loop] for (uint iteration = 0; iteration < 2; ++iteration)
        value += iteration == 0 ? 1.0 : -1.0;
    value -= 1.0; // SOURCE_HELPER
    value = addOne(value);
    sharedValues[index] = value; // SOURCE_BREAK
    GroupMemoryBarrierWithGroupSync();
    float restored = sharedValues[index];
    if ((tid.x & 1) != 0)
        restored -= 1.0;
    else
        restored += 1.0;
    uint waveCount = WaveActiveCountBits(true);
    bufOut[tid.x] = restored + bias + float(waveCount - WaveGetLaneCount());
}

struct PSIn {
    float4 pos : SV_Position;
    float4 col : COLOR;
};

float4 PSMain(PSIn input) : SV_Target {
    float dx = ddx(input.pos.x);
    float dy = ddy(input.pos.y);
    float value = dx + dy;
    if ((uint(input.pos.x) & 1) != 0)
        value += 1.0;
    else
        value -= 1.0;
    float restored = value; // SOURCE_BREAK
    if ((uint(input.pos.x) & 1) != 0)
        restored -= 1.0;
    else
        restored += 1.0;
    if ((uint(input.pos.x) + uint(input.pos.y)) % 3 == 0)
        discard;
    return float4(restored * 0.5, 0, 0, 1);
}
