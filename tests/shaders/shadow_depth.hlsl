struct VSIn {
    float3 pos : POSITION;
};

struct VSOut {
    float4 pos : SV_Position;
};

VSOut VSMain(VSIn input) {
    VSOut o;
    o.pos = float4(input.pos, 1.0);
    return o;
}
