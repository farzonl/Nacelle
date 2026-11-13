cbuffer LightCB : register(b0) {
    float4 lightDir;
    float4 baseTint;
};

Texture2D AlbedoTex : register(t0);
SamplerState LinearSamp : register(s0);

struct VSIn {
    float3 pos : POSITION;
    float3 norm : NORMAL;
    float2 uv : TEXCOORD0;
};

struct PSIn {
    float4 pos : SV_Position;
    float3 norm : NORMAL;
    float2 uv : TEXCOORD0;
};

struct PSOut {
    float4 color : SV_Target0;
    float4 normal : SV_Target1;
};

PSIn VSMain(VSIn input) {
    PSIn o;
    o.pos = float4(input.pos, 1.0);
    o.norm = input.norm;
    o.uv = input.uv;
    return o;
}

PSOut PSMain(PSIn input) {
    PSOut o;
    float3 n = normalize(input.norm);
    float3 l = normalize(lightDir.xyz);
    float ndotl = max(dot(n, l), 0.15);
    float4 tex = AlbedoTex.SampleLevel(LinearSamp, input.uv, 0.0);
    float3 shaded = tex.rgb * baseTint.rgb * ndotl;
    o.color = float4(shaded, 1.0);
    o.normal = float4(n * 0.5 + 0.5, 1.0);
    return o;
}
