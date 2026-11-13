struct PSInput {
    float4 position : SV_POSITION;
};

PSInput VSMain(float3 position : POSITION) {
    PSInput o;
    o.position = float4(position, 1.0f);
    return o;
}

float4 PSMain(PSInput input) : SV_TARGET {
    return float4(input.position.x, input.position.y, 0.0f, 1.0f);
}
