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
    float3 albedo = ColorTex.Sample(LinearSamp, uv * 2.0).rgb;
    float3 nTap = ColorTex.Sample(LinearSamp, uv * 4.0 + float2(0.13, 0.27)).rgb * 2.0 - 1.0;
    float4 mrTap = ColorTex.Sample(LinearSamp, uv * 3.0 + float2(0.41, 0.17));
    float3 detail = ColorTex.Sample(LinearSamp, uv * 8.0 - float2(iTime * 0.05, 0.0)).rgb;

    albedo = lerp(albedo, albedo * detail * 1.4, 0.35);
    float roughness = clamp(mrTap.g, 0.08, 0.95);
    float metallic = mrTap.b;

    float3 N = normalize(float3(nTap.xy * 0.6, 1.0));
    float3 V = float3(0.0, 0.0, 1.0);
    float3 L = normalize(float3(cos(iTime) * 0.7, sin(iTime) * 0.7, 0.8));
    float3 H = normalize(V + L);

    float NdotL = max(dot(N, L), 0.0);
    float NdotH = max(dot(N, H), 0.0);
    float a2 = roughness * roughness * roughness * roughness;
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    float D = a2 / max(3.14159265 * denom * denom, 1e-4);
    float3 F0 = lerp(float3(0.04, 0.04, 0.04), albedo, metallic);
    float3 F = F0 + (1.0 - F0) * pow(1.0 - max(dot(H, V), 0.0), 5.0);
    float3 diff = (1.0 - metallic) * albedo / 3.14159265;
    float3 color = (diff + F * D * 0.25) * NdotL + albedo * 0.06;
    return float4(color, 1.0);
}
