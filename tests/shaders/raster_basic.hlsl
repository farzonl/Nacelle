struct VSIn {
    float3 pos : POSITION;
    float4 col : COLOR;
};

struct PSIn {
    float4 pos : SV_Position;
    float4 col : COLOR;
};

PSIn VSMain(VSIn input) {
    PSIn o;
    o.pos = float4(input.pos, 1.0);
    o.col = input.col;
    return o;
}

float4 PSMain(PSIn input) : SV_Target {
    return input.col;
}
