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

// 6. Tessellation point topology with GS expansion
struct TessCtrlPt {
    float4 pos : POSITION;
};
struct TessPatchConst {
    float edges[3] : SV_TessFactor;
    float inside   : SV_InsideTessFactor;
};
struct TessDsOut {
    float4 pos : SV_Position;
};
struct TessGsOut {
    float4 pos : SV_Position;
};

TessCtrlPt TessPointVS(uint vid : SV_VertexID) {
    TessCtrlPt o;
    float2 p = (vid == 0u) ? float2(-0.5, -0.5) : (vid == 1u) ? float2(0.5, -0.5) : float2(0.0, 0.5);
    o.pos = float4(p, 0.0, 1.0);
    return o;
}

TessPatchConst TessPointHSConst(InputPatch<TessCtrlPt, 3> patch) {
    TessPatchConst pc;
    pc.edges[0] = 1.0;
    pc.edges[1] = 1.0;
    pc.edges[2] = 1.0;
    pc.inside = 1.0;
    return pc;
}

[domain("tri")]
[partitioning("integer")]
[outputtopology("point")]
[outputcontrolpoints(3)]
[patchconstantfunc("TessPointHSConst")]
TessCtrlPt TessPointHS(InputPatch<TessCtrlPt, 3> patch, uint i : SV_OutputControlPointID) {
    return patch[i];
}

[domain("tri")]
TessDsOut TessPointDS(TessPatchConst pc, float3 bary : SV_DomainLocation, const OutputPatch<TessCtrlPt, 3> patch) {
    TessDsOut o;
    o.pos = patch[0].pos * bary.x + patch[1].pos * bary.y + patch[2].pos * bary.z;
    return o;
}

[maxvertexcount(3)]
void TessPointGS(point TessDsOut input[1], inout TriangleStream<TessGsOut> stream) {
    float2 c = input[0].pos.xy;
    TessGsOut v0, v1, v2;
    v0.pos = float4(c + float2(-0.25, -0.25), 0.0, 1.0);
    v1.pos = float4(c + float2( 0.25, -0.25), 0.0, 1.0);
    v2.pos = float4(c + float2( 0.00,  0.25), 0.0, 1.0);
    stream.Append(v0);
    stream.Append(v1);
    stream.Append(v2);
}

float4 TessPointPS(TessGsOut input) : SV_Target0 {
    return float4(0.0, 1.0, 0.0, 1.0);
}

// 7. ByteAddressBuffer multi-component Load4 in Vertex Shader
ByteAddressBuffer gRawSrc : register(t0);
struct BabLoad4VSOut {
    float4 pos : SV_Position;
    float4 col : COLOR;
};

BabLoad4VSOut BabLoad4VS(uint vid : SV_VertexID) {
    BabLoad4VSOut o;
    float2 p = (vid == 0u) ? float2(-1.0, -1.0) : (vid == 1u) ? float2(3.0, -1.0) : float2(-1.0, 3.0);
    o.pos = float4(p, 0.0, 1.0);
    o.col = asfloat(gRawSrc.Load4(0));
    return o;
}

float4 BabLoad4PS(BabLoad4VSOut input) : SV_Target0 {
    return input.col;
}

// 8. 4x MSAA diagonal triangle
float4 MsaaDiagVS(uint vid : SV_VertexID) : SV_Position {
    float2 p = (vid == 0u) ? float2(-1.0, 1.0) : (vid == 1u) ? float2(1.0, 1.0) : float2(1.0, -1.0);
    return float4(p, 0.0, 1.0);
}

float4 MsaaDiagPS(float4 pos : SV_Position) : SV_Target0 {
    return float4(1.0, 0.0, 0.0, 1.0);
}
