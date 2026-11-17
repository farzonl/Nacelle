struct TessCtrlPt {
    float4 pos : SV_Position;
    float4 col : COLOR;
    float2 uv  : TEXCOORD0;
};

struct TessPatchConst {
    float edges[3] : SV_TessFactor;
    float inside   : SV_InsideTessFactor;
};

struct DSOut {
    float4 pos    : SV_Position;
    float4 col    : COLOR;
    float3 normal : NORMAL;
};

TessCtrlPt VSMain(uint vid : SV_VertexID) {
    uint patchIdx = vid / 3u;
    uint corner = vid % 3u;
    float cx = ((float)(patchIdx % 23u) + 0.5f) / 11.5f - 1.0f;
    float cy = ((float)((patchIdx / 23u) % 23u) + 0.5f) / 11.5f - 1.0f;
    float2 offs = (corner == 0u) ? float2(-0.075f, -0.075f)
                : (corner == 1u) ? float2( 0.075f, -0.075f)
                                 : float2( 0.000f,  0.075f);
    TessCtrlPt o;
    o.pos = float4(cx + offs.x, cy + offs.y, 0.5f, 1.0f);
    o.col = float4(0.25f + 0.75f * (float)(corner == 0u),
                   0.25f + 0.75f * (float)(corner == 1u),
                   0.25f + 0.75f * (float)(corner == 2u), 1.0f);
    o.uv = (corner == 0u) ? float2(0.0f, 0.0f)
         : (corner == 1u) ? float2(1.0f, 0.0f)
                          : float2(0.5f, 1.0f);
    return o;
}

TessPatchConst HSConst(InputPatch<TessCtrlPt, 3> patch) {
    TessPatchConst pc;
    pc.edges[0] = 16.0f;
    pc.edges[1] = 16.0f;
    pc.edges[2] = 16.0f;
    pc.inside   = 16.0f;
    return pc;
}

[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("HSConst")]
TessCtrlPt HSMain(InputPatch<TessCtrlPt, 3> patch, uint i : SV_OutputControlPointID) {
    return patch[i];
}

[domain("tri")]
DSOut DSMain(TessPatchConst pc, float3 bary : SV_DomainLocation, const OutputPatch<TessCtrlPt, 3> patch) {
    float4 p = patch[0].pos * bary.x + patch[1].pos * bary.y + patch[2].pos * bary.z;
    float2 uv = patch[0].uv * bary.x + patch[1].uv * bary.y + patch[2].uv * bary.z;
    float4 col = patch[0].col * bary.x + patch[1].col * bary.y + patch[2].col * bary.z;

    float wave1 = sin(p.x * 18.0f + uv.x * 6.28318f);
    float wave2 = cos(p.y * 18.0f + uv.y * 6.28318f);
    p.x += wave1 * 0.004f;
    p.y += wave2 * 0.004f;
    p.z = clamp(0.5f + 0.1f * wave1 * wave2, 0.05f, 0.95f);

    DSOut o;
    o.pos = p;
    o.col = col;
    o.normal = normalize(float3(-wave1 * 0.4f, -wave2 * 0.4f, 1.0f));
    return o;
}

float4 PSMain(DSOut input) : SV_Target0 {
    float3 lightDir = normalize(float3(0.5f, 0.6f, 0.7f));
    float diff = max(dot(normalize(input.normal), lightDir), 0.15f);
    return float4(input.col.rgb * diff, 1.0f);
}
