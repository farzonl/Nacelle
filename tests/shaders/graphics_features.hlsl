// Fullscreen-triangle VS (no vertex buffer)
float4 FullscreenVS(uint vid : SV_VertexID) : SV_Position {
    float2 pos[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
    return float4(pos[vid], 0.5, 1.0);
}

// 1. Barycentrics (perspective & noperspective) with non-uniform W
struct BaryVSOut {
    float4 pos : SV_Position;
};

BaryVSOut BaryVS(uint vid : SV_VertexID) {
    BaryVSOut o;
    // Triangle covering whole screen with distinct W per vertex (1, 2, 4)
    float w[3] = { 1.0, 2.0, 4.0 };
    float2 xy[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
    o.pos = float4(xy[vid] * w[vid], 0.5 * w[vid], w[vid]);
    return o;
}

float4 BaryPerspPS(float4 pos : SV_Position, float3 bary : SV_Barycentrics) : SV_Target0 {
    return float4(bary, 1.0);
}

float4 BaryNoPerspPS(float4 pos : SV_Position, noperspective float3 bary : SV_Barycentrics) : SV_Target0 {
    return float4(bary, 1.0);
}

// 2. PS-Specified Stencil Ref
struct StencilPSOut {
    float4 col : SV_Target0;
    uint stencilRef : SV_StencilRef;
};

StencilPSOut StencilRefPS(float4 pos : SV_Position) {
    StencilPSOut o;
    o.col = float4(0.25, 0.75, 0.5, 1.0);
    // Left half outputs stencil ref 7, right half outputs stencil ref 3
    o.stencilRef = (pos.x < 16.0) ? 7u : 3u;
    return o;
}

// 3. Variable Rate Shading (SV_ShadingRate)
float4 ShadingRatePS(float4 pos : SV_Position, uint rate : SV_ShadingRate) : SV_Target0 {
    return float4(float(rate) / 15.0, 0.0, 0.0, 1.0);
}

// 4. Multi-Viewport (SV_ViewportArrayIndex from VS -> Rasterizer & PS)
struct MultiVPVSOut {
    float4 pos : SV_Position;
    uint vpIdx : SV_ViewportArrayIndex;
};

MultiVPVSOut MultiVPVS(uint vid : SV_VertexID) {
    MultiVPVSOut o;
    uint tri = vid / 3u;
    uint v = vid % 3u;
    float2 xy[3] = { float2(-1.0, -1.0), float2(-1.0, 3.0), float2(3.0, -1.0) };
    o.pos = float4(xy[v], 0.5, 1.0);
    o.vpIdx = tri; // tri 0 -> viewport 0, tri 1 -> viewport 1
    return o;
}

float4 MultiVPPS(float4 pos : SV_Position, uint vpIdx : SV_ViewportArrayIndex) : SV_Target0 {
    return (vpIdx == 0u) ? float4(1.0, 0.0, 0.0, 1.0) : float4(0.0, 1.0, 0.0, 1.0);
}

// 5. Texture LOD Clamp (Sample & SampleBias with MinLodClamp)
Texture2D<float4> gTex : register(t0);
SamplerState gSamp : register(s0);

float4 LodClampPS(float4 pos : SV_Position) : SV_Target0 {
    float2 uv = pos.xy / 32.0;
    return gTex.SampleBias(gSamp, uv, 0.0, 2.0);
}
