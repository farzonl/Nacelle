#ifdef __spirv__
#define VK_LOC(n) [[vk::location(n)]]
#else
#define VK_LOC(n)
#endif

cbuffer MeshBenchParams : register(b0) {
    row_major float4x4 viewProj;
    float4 boneRows[12]; // 4 affine 4x3 skinning matrices (3 rows per bone)
    float4 lightDirAndTime;
};

struct VSIn {
    VK_LOC(0) float3 pos    : POSITION0;
    VK_LOC(1) float4 normal : NORMAL0;
    VK_LOC(2) float2 uv     : TEXCOORD0;
    VK_LOC(3) float4 color  : COLOR0;
    VK_LOC(4) uint4  joints : BLENDINDICES0;
};

struct VSOut {
    float4 pos : SV_POSITION;
    VK_LOC(0) float4 color   : COLOR0;
    VK_LOC(1) float4 normUv  : TEXCOORD0;
};

float3 evalBonePos(uint boneIdx, float4 p) {
    uint baseRow = (boneIdx & 3u) * 3u;
    return float3(
        dot(boneRows[baseRow + 0u], p),
        dot(boneRows[baseRow + 1u], p),
        dot(boneRows[baseRow + 2u], p));
}

float3 evalBoneDir(uint boneIdx, float3 n) {
    uint baseRow = (boneIdx & 3u) * 3u;
    return float3(
        dot(boneRows[baseRow + 0u].xyz, n),
        dot(boneRows[baseRow + 1u].xyz, n),
        dot(boneRows[baseRow + 2u].xyz, n));
}

VSOut VSMain(VSIn input) {
    float t = lightDirAndTime.w;

    // Derive normalized 4-bone skinning weights from packed vertex attributes
    float4 rawW = float4(
        input.color.a + 0.15,
        input.uv.x * 0.65 + 0.10,
        input.uv.y * 0.65 + 0.10,
        abs(input.normal.w) * 0.45 + 0.10);
    float4 weights = rawW / dot(rawW, float4(1.0, 1.0, 1.0, 1.0));

    // Procedural multi-octave surface ripple before skinning
    float wave0 = sin(input.pos.x * 9.0 + t) * cos(input.pos.y * 7.0 - t * 0.5);
    float wave1 = cos((input.uv.x + input.uv.y) * 12.56637 + t * 0.75);
    float3 localPos = input.pos + input.normal.xyz * ((wave0 * 0.025) + (wave1 * 0.012));
    float4 p4 = float4(localPos, 1.0);

    // 4-matrix linear blend skinning (position + normal)
    float3 skinnedPos =
        evalBonePos(input.joints.x, p4) * weights.x +
        evalBonePos(input.joints.y, p4) * weights.y +
        evalBonePos(input.joints.z, p4) * weights.z +
        evalBonePos(input.joints.w, p4) * weights.w;

    float3 skinnedNrm = normalize(
        evalBoneDir(input.joints.x, input.normal.xyz) * weights.x +
        evalBoneDir(input.joints.y, input.normal.xyz) * weights.y +
        evalBoneDir(input.joints.z, input.normal.xyz) * weights.z +
        evalBoneDir(input.joints.w, input.normal.xyz) * weights.w);

    // Two-light Blinn-Phong + hemisphere ambient + Fresnel rim lighting in VS
    float3 L0 = normalize(lightDirAndTime.xyz);
    float3 L1 = normalize(float3(-0.55, -0.35, 0.75));
    float3 V  = normalize(float3(0.0, 0.0, 1.2) - skinnedPos);
    float3 H0 = normalize(L0 + V);
    float3 H1 = normalize(L1 + V);

    float ndl0 = saturate(dot(skinnedNrm, L0));
    float ndl1 = saturate(dot(skinnedNrm, L1));
    float spec0 = pow(saturate(dot(skinnedNrm, H0)), 16.0);
    float spec1 = pow(saturate(dot(skinnedNrm, H1)), 8.0);
    float rim = pow(1.0 - saturate(dot(skinnedNrm, V)), 3.0);
    float hemi = skinnedNrm.y * 0.5 + 0.5;

    float3 ambient = lerp(float3(0.06, 0.08, 0.14), float3(0.16, 0.20, 0.28), hemi);
    float3 lit = input.color.rgb * (ambient + ndl0 * 0.78 + ndl1 * 0.28)
               + float3(0.35, 0.32, 0.28) * spec0
               + float3(0.12, 0.18, 0.26) * spec1
               + float3(0.08, 0.14, 0.22) * rim;

    VSOut o;
    o.pos = mul(float4(skinnedPos, 1.0), viewProj);
    o.color = float4(saturate(lit), 1.0);
    o.normUv = float4(skinnedNrm.xy * 0.5 + 0.5, input.uv);
    return o;
}

float4 PSMain(VSOut input) : SV_TARGET {
    float3 modulated = input.color.rgb * (0.88 + 0.12 * (input.normUv.x * input.normUv.z + input.normUv.y * input.normUv.w));
    return float4(modulated, 1.0);
}
