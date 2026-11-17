cbuffer VertexStressCB : register(b0) {
    float4x4 mvp;
    float time;
    uint vertexCount;
    float2 pad;
};

struct PSIn {
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

PSIn VSMain(uint vid : SV_VertexID) {
    uint tri = vid / 3u;
    uint corner = vid % 3u;
    uint triCount = max(vertexCount / 3u, 1u);
    uint cellCount = max((triCount + 1u) / 2u, 1u);
    uint gridDim = max(uint(ceil(sqrt(float(cellCount)))), 1u);

    uint cell = tri / 2u;
    uint halfTri = tri % 2u;
    uint gx = cell % gridDim;
    uint gy = cell / gridDim;

    float invGrid = 1.0f / float(gridDim);
    float2 baseUV = float2(float(gx), float(gy)) * invGrid;

    float2 cornerUV;
    if (halfTri == 0u) {
        cornerUV = (corner == 0u) ? float2(0.0f, 0.0f)
                 : (corner == 1u) ? float2(invGrid, 0.0f)
                                  : float2(0.0f, invGrid);
    } else {
        cornerUV = (corner == 0u) ? float2(invGrid, 0.0f)
                 : (corner == 1u) ? float2(invGrid, invGrid)
                                  : float2(0.0f, invGrid);
    }

    float2 uv = baseUV + cornerUV;
    float2 ndc = float2(-1.0f + 2.0f * uv.x, 1.0f - 2.0f * uv.y);

    PSIn o;
    o.pos = mul(mvp, float4(ndc, 0.5f, 1.0f));
    o.uv = uv;
    return o;
}

float4 PSMain(PSIn input) : SV_Target0 {
    float2 p = input.uv * 4.0f - 2.0f + float2(time * 0.1f, -time * 0.1f);
    float3 acc = float3(0.0f, 0.0f, 0.0f);
    float amp = 0.5f;
    float freq = 1.0f;
    [unroll]
    for (int k = 0; k < 12; ++k) {
        float s = sin(p.x * freq + float(k) * 0.7f);
        float c = cos(p.y * freq - float(k) * 1.1f);
        float d = s * c;
        acc += amp * float3(abs(d), s * s, c * c);
        p = float2(p.x * 0.8f - p.y * 0.6f, p.x * 0.6f + p.y * 0.8f);
        freq *= 1.37f;
        amp *= 0.72f;
    }
    float3 n = normalize(acc + float3(0.1f, 0.2f, 0.3f));
    float diff = saturate(dot(n, normalize(float3(0.577f, 0.577f, 0.577f))));
    return float4(acc * diff, 1.0f);
}
