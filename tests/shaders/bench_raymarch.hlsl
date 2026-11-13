cbuffer BenchParams : register(b0) {
    uint width;
    uint height;
    uint maxSteps;
    float time;
};

StructuredBuffer<float4> InSpheres : register(t0);
RWStructuredBuffer<float4> OutPixels : register(u0);

float sdSphere(float3 p, float4 s) {
    return length(p - s.xyz) - s.w;
}

float mapScene(float3 p) {
    float d = p.y + 1.0;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        d = min(d, sdSphere(p, InSpheres[i]));
    }
    return d;
}

float3 calcNormal(float3 p) {
    const float e = 0.001;
    float d0 = mapScene(p);
    float3 n = float3(
        mapScene(p + float3(e, 0.0, 0.0)) - d0,
        mapScene(p + float3(0.0, e, 0.0)) - d0,
        mapScene(p + float3(0.0, 0.0, e)) - d0);
    return normalize(n);
}

[numthreads(8, 8, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= width || dtid.y >= height) return;

    float2 uv = (float2(dtid.xy) + 0.5) / float2(width, height);
    float2 ndc = uv * 2.0 - 1.0;
    ndc.x *= float(width) / float(height);

    float3 ro = float3(0.0, 0.4, -3.5);
    float3 rd = normalize(float3(ndc.x, ndc.y, 1.8));

    float t = 0.0;
    bool hit = false;
    for (uint s = 0; s < maxSteps; ++s) {
        float3 p = ro + rd * t;
        float d = mapScene(p);
        if (d < 0.001) {
            hit = true;
            break;
        }
        t += d;
        if (t > 20.0) break;
    }

    float3 col = float3(0.08, 0.12, 0.22);
    if (hit) {
        float3 p = ro + rd * t;
        float3 n = calcNormal(p);
        float3 lightDir = normalize(float3(0.6, 0.8, -0.5));
        float diff = max(dot(n, lightDir), 0.05);
        col = float3(0.85, 0.45, 0.25) * diff + float3(0.05, 0.07, 0.10);
    }

    OutPixels[dtid.y * width + dtid.x] = float4(col, 1.0);
}
