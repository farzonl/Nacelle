cbuffer VertexStressCB : register(b0) {
    float4x4 mvp;
    float time;
    uint vertexCount;
    float2 pad;
};

struct PSIn {
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

PSIn VSMain(uint vid : SV_VertexID) {
    uint tri = vid / 3u;
    uint corner = vid % 3u;

    float angle = float(tri) * 0.137 + time;
    float ring = float(tri % 64u) * 0.015625;
    float2 offset = float2(cos(angle), sin(angle)) * (0.1 + 0.8 * ring);

    float2 triOff = (corner == 0u) ? float2(-0.02, -0.02)
                  : (corner == 1u) ? float2( 0.02, -0.02)
                                   : float2( 0.00,  0.025);

    float3 p = float3(offset + triOff, 0.25 * sin(angle * 3.0 + float(corner)));

    // Simulate multi-wave vertex displacement & normal shading in VS
    float3 n = normalize(float3(sin(angle), cos(angle), 0.5 + 0.5 * sin(ring * 6.28318)));
    float4 worldPos = float4(p, 1.0);
    worldPos.x += 0.03 * sin(worldPos.y * 8.0 + time);
    worldPos.y += 0.03 * cos(worldPos.x * 8.0 - time);

    PSIn o;
    o.pos = mul(mvp, worldPos);
    float ndotl = max(dot(n, normalize(float3(0.4, 0.7, 0.6))), 0.1);
    o.col = float4(ndotl * (0.5 + 0.5 * sin(ring * 6.28318 + float3(0.0, 2.09, 4.18))), 1.0);
    return o;
}

float4 PSMain(PSIn input) : SV_Target0 {
    return input.col;
}
