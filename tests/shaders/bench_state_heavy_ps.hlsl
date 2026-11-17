cbuffer StateHeavyCB : register(b0) {
    uint layerCount;
    uint width;
    uint height;
    float time;
};

Texture2D<float4> albedoBC1 : register(t0);
Texture2D<float4> detailBC7 : register(t1);
SamplerState linearSamp : register(s0);

struct VSIn {
    float3 pos : POSITION;
    float2 uv  : TEXCOORD0;
    uint inst  : SV_InstanceID;
};

struct PSIn {
    float4 pos   : SV_Position;
    float2 uv    : TEXCOORD0;
    uint   layer : TEXCOORD1;
};

PSIn VSMain(VSIn v) {
    PSIn o;
    // Decreasing depth per layer (0.90 down to ~0.26) so every layer passes LESS_EQUAL
    // and triggers depth write, stencil increment, alpha blending, and 4x MSAA coverage.
    float z = 0.90f - float(v.inst) * 0.02f;
    // Slight per-layer inset/tilt to exercise 4x MSAA subpixel edge coverage on interior layers.
    float shrink = 1.0f - float(v.inst) * 0.008f;
    o.pos = float4(v.pos.x * shrink, v.pos.y * shrink, z, 1.0f);
    float offset = float(v.inst) * 0.03125f + time * 0.05f;
    o.uv = v.uv * (1.0f + float(v.inst) * 0.0625f) + float2(offset, offset * 0.5f);
    o.layer = v.inst;
    return o;
}

float4 PSMain(PSIn p) : SV_Target0 {
    float4 c0 = albedoBC1.Sample(linearSamp, p.uv);
    float4 c1 = detailBC7.Sample(linearSamp, p.uv * 2.0f + float2(0.125f, 0.25f));
    float3 rgb = c0.rgb * 0.65f + c1.rgb * 0.35f;
    float tint = 0.75f + 0.25f * sin(float(p.layer) * 0.7f + p.uv.x * 3.14159f);
    float alpha = 0.22f + 0.01f * float(p.layer & 7u);
    return float4(rgb * tint, alpha);
}
