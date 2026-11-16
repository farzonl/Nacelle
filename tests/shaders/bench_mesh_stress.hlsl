cbuffer MeshStressCB : register(b0) {
    float scale;
    float time;
    uint groupCount;
    float pad;
};

struct MeshOut {
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSMain(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    out vertices MeshOut verts[96],
    out indices uint3 tris[32]) {
    SetMeshOutputCounts(96, 32);

    float idx = float(gid * 32u + gtid);
    float total = max(float(groupCount * 32u), 1.0);
    float t = idx / total;
    float a = t * 6.2831853 * 12.0 + time;
    float r = 0.1 + 0.75 * t;
    float2 c = float2(cos(a) * r, sin(a) * r);
    float4 col = float4(0.5 + 0.5 * sin(t * 6.28318 + float3(0.0, 2.1, 4.2)), 1.0);

    float s = scale;
    float a0 = float(gtid & 3u) * 1.5707963;
    float a1 = a0 + 0.7853981;

    uint baseV = gtid * 3u;
    verts[baseV + 0u].pos = float4(c, 0.0, 1.0);
    verts[baseV + 0u].col = col;
    verts[baseV + 1u].pos = float4(c + float2(cos(a0), sin(a0)) * s, 0.0, 1.0);
    verts[baseV + 1u].col = col * 0.8;
    verts[baseV + 2u].pos = float4(c + float2(cos(a1), sin(a1)) * s, 0.0, 1.0);
    verts[baseV + 2u].col = col * 0.6;

    tris[gtid] = uint3(baseV + 0u, baseV + 1u, baseV + 2u);
}

float4 PSMain(MeshOut input) : SV_Target0 {
    return input.col;
}
