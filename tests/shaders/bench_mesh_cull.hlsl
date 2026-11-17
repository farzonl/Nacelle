cbuffer MeshCullCB : register(b0) {
    float scale;
    float time;
    uint gridX;
    uint layerCount;
};

struct MeshPayload {
    uint clusterIds[32];
};

struct MeshOut {
    float4 pos : SV_Position;
    float3 worldPos : TEXCOORD0;
    float3 normal : NORMAL0;
};

struct PrimOut {
    bool cull : SV_CullPrimitive;
    nointerpolation float primShade : COLOR1;
};

groupshared uint gsTotalVisible;
groupshared MeshPayload gsPayload;

static float evalHeight(float2 p, float t, float layerIdx) {
    float a = sin(p.x * 5.3 + t + layerIdx * 1.1) * cos(p.y * 4.7 - t * 0.8);
    float b = 0.5 * sin((p.x + p.y) * 9.1 - t * 1.3 + layerIdx * 0.7);
    return (a + b) * 0.14;
}

[numthreads(32, 1, 1)]
void ASMain(uint gtid : SV_GroupThreadID, uint gid : SV_GroupID) {
    uint gxCount = max(gridX, 1u);
    uint perLayer = gxCount * gxCount;
    uint idx = gid * 32u + gtid;
    uint layer = idx / perLayer;
    uint rem = idx - layer * perLayer;
    uint cy = rem / gxCount;
    uint cx = rem - cy * gxCount;

    bool visible = false;
    if (layer < max(layerCount, 1u)) {
        float invG = 1.0 / float(gxCount);
        float2 centerUV = (float2(float(cx) + 0.5, float(cy) + 0.5) * invG) * 2.0 - 1.0;
        float rad2 = dot(centerUV, centerUV);
        float waveMask = sin(centerUV.x * 5.0 + time + float(layer) * 1.7) *
                         cos(centerUV.y * 5.0 - time * 0.5);
        visible = (rad2 < 0.65) && (waveMask > -0.1);
    }

    if (gtid == 0u) {
        gsTotalVisible = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    if (visible) {
        uint outIdx = 0u;
        InterlockedAdd(gsTotalVisible, 1u, outIdx);
        if (outIdx < 32u) {
            gsPayload.clusterIds[outIdx] = idx;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    DispatchMesh(min(gsTotalVisible, 32u), 1u, 1u, gsPayload);
}

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void MSMain(
    uint gtid : SV_GroupThreadID,
    uint gid : SV_GroupID,
    in payload MeshPayload inPayload,
    out vertices MeshOut verts[96],
    out indices uint3 tris[32],
    out primitives PrimOut prims[32]) {
    SetMeshOutputCounts(96u, 32u);

    uint clusterIdx = inPayload.clusterIds[gid];
    uint gxCount = max(gridX, 1u);
    uint perLayer = gxCount * gxCount;
    uint layer = clusterIdx / perLayer;
    uint rem = clusterIdx - layer * perLayer;
    uint cy = rem / gxCount;
    uint cx = rem - cy * gxCount;

    uint qx = gtid & 3u;
    uint qy = (gtid >> 2u) & 3u;
    uint halfTri = gtid >> 4u;

    float invG = 1.0 / float(gxCount);
    float patchSize = invG * 2.0 * max(scale, 0.25);
    float2 patchOrigin = float2(float(cx), float(cy)) * invG * 2.0 - 1.0;
    float2 cellSize = float2(patchSize * 0.25, patchSize * 0.25);
    float2 p00 = patchOrigin + float2(float(qx), float(qy)) * cellSize;

    float2 off0 = (halfTri == 0u) ? float2(0.0, 0.0) : float2(cellSize.x, 0.0);
    float2 off1 = (halfTri == 0u) ? float2(cellSize.x, 0.0) : float2(cellSize.x, cellSize.y);
    float2 off2 = float2(0.0, cellSize.y);

    float2 xy0 = p00 + off0;
    float2 xy1 = p00 + off1;
    float2 xy2 = p00 + off2;

    float layerZ = 0.22 + float(layer) * 0.16;
    float h0 = evalHeight(xy0, time, float(layer));
    float h1 = evalHeight(xy1, time, float(layer));
    float h2 = evalHeight(xy2, time, float(layer));

    float3 nrm = normalize(float3(h0 - h1, h0 - h2, 0.025));

    float z0 = clamp(layerZ + h0 * 0.45, 0.05, 0.95);
    float z1 = clamp(layerZ + h1 * 0.45, 0.05, 0.95);
    float z2 = clamp(layerZ + h2 * 0.45, 0.05, 0.95);

    uint baseV = gtid * 3u;
    verts[baseV + 0u].pos = float4(xy0, z0, 1.0);
    verts[baseV + 0u].worldPos = float3(xy0, z0);
    verts[baseV + 0u].normal = nrm;

    verts[baseV + 1u].pos = float4(xy1, z1, 1.0);
    verts[baseV + 1u].worldPos = float3(xy1, z1);
    verts[baseV + 1u].normal = nrm;

    verts[baseV + 2u].pos = float4(xy2, z2, 1.0);
    verts[baseV + 2u].worldPos = float3(xy2, z2);
    verts[baseV + 2u].normal = nrm;

    float2 e0 = xy1 - xy0;
    float2 e1 = xy2 - xy0;
    float det = e0.x * e1.y - e0.y * e1.x;
    bool culled = (det <= 1e-6) || ((gtid & 1u) != 0u && h0 < 0.0);

    tris[gtid] = uint3(baseV + 0u, baseV + 1u, baseV + 2u);
    prims[gtid].cull = culled;
    prims[gtid].primShade = 0.65 + 0.35 * sin(float(clusterIdx + gtid) * 0.37);
}

float4 PSMain(MeshOut input, PrimOut prim) : SV_Target0 {
    float3 N = normalize(input.normal + 0.12 * float3(
        sin(input.worldPos.x * 48.0),
        cos(input.worldPos.y * 48.0),
        0.0));
    float3 L = normalize(float3(0.45, 0.65, 0.60));
    float3 V = normalize(float3(-input.worldPos.xy, 1.2));
    float3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.05);
    float ndoth = max(dot(N, H), 0.0);
    float spec = pow(ndoth, 24.0);
    float fresnel = pow(1.0 - max(dot(N, V), 0.0), 3.0);

    float3 baseCol = lerp(
        float3(0.16, 0.38, 0.56),
        float3(0.84, 0.52, 0.24),
        saturate(input.worldPos.z * 1.4));
    float3 col = baseCol * ndotl * prim.primShade + float3(0.95, 0.90, 0.80) * (spec * 0.45 + fresnel * 0.2);
    return float4(saturate(col), 1.0);
}
