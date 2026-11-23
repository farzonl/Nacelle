struct PSIn {
    float4 pos : SV_Position;
    float4 col : COLOR;
};

float4 PSMain(PSIn input) : SV_Target {
    uint pixelX = uint(input.pos.x);
    uint pixelY = uint(input.pos.y);
    float dx = ddx(input.pos.x);
    float dy = ddy(input.pos.y);
    float value = dx + dy;
    if ((pixelX & 1) != 0)
        value += 1.0;
    else
        value -= 1.0;
    float restored = value;
    if ((pixelX & 1) != 0)
        restored -= 1.0;
    else
        restored += 1.0;
    if ((pixelX + pixelY) % 3 == 0)
        discard;
    return float4(restored * 0.5, 0, 0, 1);
}
