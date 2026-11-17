cbuffer ShadertoyCB : register(b0) {
    float3 iResolution;
    float iTime;
    float4 iMouse;
};

Texture2D<float4> ColorTex : register(t0);
#ifdef __spirv__
[[vk::binding(24, 0)]]
#endif
SamplerState LinearSamp : register(s0);

float4 PSMain(float4 pos : SV_Position) : SV_Target0 {
    float2 uv = pos.xy / max(iResolution.xy, float2(1.0, 1.0));
    float2 center = float2(0.5, 0.5);
    float2 dir = uv - center;
    float angle = iTime * 0.25;
    float s = sin(angle);
    float c = cos(angle);

    float4 accum = float4(0.0, 0.0, 0.0, 0.0);
    [unroll]
    for (int i = 0; i < 8; ++i) {
        float fi = float(i) * 0.125;
        float2 offset = float2(c * fi - s * (0.04 * fi), s * fi + c * (0.04 * fi));
        float2 tapUV = uv * (1.0 + 0.15 * fi) + dir * (0.06 * fi) + offset * 0.05;
        accum += ColorTex.Sample(LinearSamp, tapUV);
    }
    return accum * 0.125;
}
