struct TessCtrlPt {
    float4 pos : SV_Position;
    float4 col : COLOR;
};

struct TessPatchConst {
    float edges[3] : SV_TessFactor;
    float inside   : SV_InsideTessFactor;
};

TessCtrlPt VSMain(uint vid : SV_VertexID) {
    uint patchIdx = vid / 3u;
    uint corner = vid % 3u;
    float cx = ((float)(patchIdx % 16u) + 0.5) / 8.0 - 1.0;
    float cy = ((float)(patchIdx / 16u) + 0.5) / 8.0 - 1.0;
    float2 offs = (corner == 0u) ? float2(-0.08, -0.08)
                : (corner == 1u) ? float2( 0.08, -0.08)
                                 : float2( 0.00,  0.08);
    TessCtrlPt o;
    o.pos = float4(cx + offs.x, cy + offs.y, 0.0, 1.0);
    o.col = float4(0.2 + 0.8 * (float)(corner == 0u),
                   0.2 + 0.8 * (float)(corner == 1u),
                   0.2 + 0.8 * (float)(corner == 2u), 1.0);
    return o;
}

TessPatchConst HSConst(InputPatch<TessCtrlPt, 3> patch) {
    TessPatchConst pc;
    pc.edges[0] = 4.0;
    pc.edges[1] = 4.0;
    pc.edges[2] = 4.0;
    pc.inside = 4.0;
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
TessCtrlPt DSMain(TessPatchConst pc, float3 bary : SV_DomainLocation, const OutputPatch<TessCtrlPt, 3> patch) {
    TessCtrlPt o;
    o.pos = patch[0].pos * bary.x + patch[1].pos * bary.y + patch[2].pos * bary.z;
    o.col = patch[0].col * bary.x + patch[1].col * bary.y + patch[2].col * bary.z;
    return o;
}

float4 PSMain(TessCtrlPt input) : SV_Target0 {
    return input.col;
}
