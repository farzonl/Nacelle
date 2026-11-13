struct VSIn {
    float3 pos : POSITION;
};

float4 VSMain(VSIn input) : SV_Position {
    return float4(input.pos, 1.0);
}
