// HLSL Shader Model 6.5 Amplification + Mesh + Pixel test shaders

struct MeshVertex {
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

// 1. SimpleTriangle (matches OffloadTest Graphics/MeshShaders/SimpleTriangle.test & MSAA/MeshShaderTriangle.test)
[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void SimpleTriangleMS(
    uint groupThreadID : SV_GroupThreadID,
    out vertices MeshVertex verts[3],
    out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);

    float4 position;
    float4 color;
    if (groupThreadID == 0) {
        position = float4(0.0, 0.25, 0.0, 1.0);
        color = float4(1.0, 0.0, 0.0, 1.0);
    } else if (groupThreadID == 1) {
        position = float4(0.25, -0.25, 0.0, 1.0);
        color = float4(0.0, 1.0, 0.0, 1.0);
    } else {
        position = float4(-0.25, -0.25, 0.0, 1.0);
        color = float4(0.0, 0.0, 1.0, 1.0);
    }

    verts[groupThreadID].position = position;
    verts[groupThreadID].color = color;

    if (groupThreadID == 0) {
        tris[0] = uint3(0, 1, 2);
    }
}

float4 VertexColorPS(MeshVertex input) : SV_TARGET {
    return input.color;
}

float4 SaturateColorPS(MeshVertex input) : SV_TARGET {
    return saturate(input.color);
}

// 2. SimpleLines (matches OffloadTest Graphics/MeshShaders/SimpleLines.test)
float3 hueToRgb(in float hue) {
    return saturate(float3(
        abs(hue * 6.0 - 3.0) - 1.0,
        2.0 - abs(hue * 6.0 - 2.0),
        2.0 - abs(hue * 6.0 - 4.0)
    ));
}

[outputtopology("line")]
[numthreads(32, 1, 1)]
void CircleLinesMS(
    uint groupThreadID : SV_GroupThreadID,
    out vertices MeshVertex verts[32],
    out indices uint2 lines[32]) {
    SetMeshOutputCounts(32, 32);

    const float radius = 0.5;
    const float angle = float(groupThreadID) / 32.0f;
    const float tau = 2.0 * 3.14159265359;
    const float x = cos(angle * tau) * radius;
    const float y = sin(angle * tau) * radius;

    const float3 color = hueToRgb(angle);

    verts[groupThreadID].position = float4(x, y, 0.0, 1.0);
    verts[groupThreadID].color = float4(color, 1.0);

    lines[groupThreadID] = uint2(groupThreadID, (groupThreadID + 1) % 32);
}

// 3. SimpleAmplification (matches OffloadTest Graphics/MeshShaders/SimpleAmplification.test)
struct MeshPayload {
    float3 color;
};

groupshared MeshPayload gs_payload;

[numthreads(1, 1, 1)]
void PayloadAS() {
    gs_payload.color = float3(1.0, 0.5, 1.0);
    DispatchMesh(1, 1, 1, gs_payload);
}

[outputtopology("triangle")]
[numthreads(3, 1, 1)]
void PayloadMS(
    uint groupThreadID : SV_GroupThreadID,
    in payload MeshPayload payload,
    out vertices MeshVertex verts[3],
    out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);

    float4 position;
    float4 color = float4(payload.color, 1.0);
    if (groupThreadID == 0) {
        position = float4(0.0, 0.25, 0.0, 1.0);
        color *= float4(1.0, 0.0, 0.0, 1.0);
    } else if (groupThreadID == 1) {
        position = float4(0.25, -0.25, 0.0, 1.0);
        color *= float4(0.0, 2.0, 0.0, 1.0);
    } else {
        position = float4(-0.25, -0.25, 0.0, 1.0);
        color *= float4(0.0, 0.0, 1.0, 1.0);
    }

    verts[groupThreadID].position = position;
    verts[groupThreadID].color = color;

    if (groupThreadID == 0) {
        tris[0] = uint3(0, 1, 2);
    }
}

// 4. StencilState (matches OffloadTest Graphics/MeshShaders/StencilState.test)
struct PosOnlyVertex {
    float4 position : SV_Position;
};

[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void StencilFullscreenMS(
    out vertices PosOnlyVertex vertices[3],
    out indices uint3 triangles[1]) {
    SetMeshOutputCounts(3, 1);
    vertices[0].position = float4(0.0, 3.0, 0.5, 1.0);
    vertices[1].position = float4(3.0, -3.0, 0.5, 1.0);
    vertices[2].position = float4(-3.0, -3.0, 0.5, 1.0);
    triangles[0] = uint3(0, 1, 2);
}

float4 SolidRedPS() : SV_Target {
    return float4(1.0, 0.0, 0.0, 1.0);
}

// 5. Multi-wave groupshared barrier + per-primitive attributes + SV_CullPrimitive + multi-group DispatchMesh
struct PrimVertOut {
    float4 position : SV_Position;
    float4 vertTint : COLOR0;
};

struct PrimAttrOut {
    float4 primColor : PRIMCOLOR;
    bool cull : SV_CullPrimitive;
};

struct PrimPSIn {
    float4 position : SV_Position;
    float4 vertTint : COLOR0;
    float4 primColor : PRIMCOLOR;
};

groupshared float4 gs_verts[6];

[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void BarrierPrimAttrMS(
    uint gtid : SV_GroupThreadID,
    uint3 gid : SV_GroupID,
    out vertices PrimVertOut verts[6],
    out indices uint3 tris[2],
    out primitives PrimAttrOut prims[2]) {
    SetMeshOutputCounts(6, 2);

    const float xBase = (gid.x == 0) ? -1.0f : 0.0f;
    const float xMid = xBase + 0.5f;
    const float xEnd = xBase + 1.0f;

    // Populate in wave 0 (lanes 0..5)
    if (gtid == 0) gs_verts[0] = float4(xBase,  1.0f, 0.5f, 1.0f);
    if (gtid == 1) gs_verts[1] = float4(xMid,   1.0f, 0.5f, 1.0f);
    if (gtid == 2) gs_verts[2] = float4(xBase, -1.0f, 0.5f, 1.0f);
    if (gtid == 3) gs_verts[3] = float4(xMid,   1.0f, 0.5f, 1.0f);
    if (gtid == 4) gs_verts[4] = float4(xEnd,   1.0f, 0.5f, 1.0f);
    if (gtid == 5) gs_verts[5] = float4(xEnd,  -1.0f, 0.5f, 1.0f);

    GroupMemoryBarrierWithGroupSync();

    // Consume in the final wave (lanes 24..29) to verify cross-wave barrier sync
    if (gtid >= 24 && gtid < 30) {
        uint vIdx = gtid - 24;
        verts[vIdx].position = gs_verts[vIdx];
        verts[vIdx].vertTint = float4(0.25f, 0.5f, 0.75f, 0.0f);
    }

    if (gtid == 30) {
        tris[0] = uint3(0, 1, 2);
        prims[0].primColor = float4(1.0f, 0.0f, 0.0f, 1.0f);
        prims[0].cull = true; // Cull primitive 0
    }
    if (gtid == 31) {
        tris[1] = uint3(3, 4, 5);
        prims[1].primColor = (gid.x == 0)
            ? float4(0.0f, 1.0f, 0.25f, 1.0f)
            : float4(0.25f, 0.0f, 1.0f, 1.0f);
        prims[1].cull = false; // Keep primitive 1
    }
}

float4 PrimAttrPS(PrimPSIn input) : SV_Target {
    return input.vertTint + input.primColor;
}
