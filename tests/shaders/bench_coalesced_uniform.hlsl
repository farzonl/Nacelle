cbuffer CoalescedUniformCB : register(b0) {
    uint count;
    uint passes;
    float alpha;
    float beta;
    float4 weights[4];
};

StructuredBuffer<float> InA : register(t0);
StructuredBuffer<float> InB : register(t1);
RWStructuredBuffer<float> Out : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint idx = dtid.x;
    if (idx >= count) return;

    float a = InA[idx];
    float b = InB[idx];
    float acc = a * alpha + b * beta;

    [loop]
    for (uint p = 0; p < passes; ++p) {
        float4 w = weights[p & 3u];
        acc = mad(acc, w.x, a * w.y);
        acc = mad(acc, w.z, b * w.w);
        acc = acc * 0.9995f + 0.0005f;
    }

    Out[idx] = acc;
}
