cbuffer GeomStressCB : register(b0) {
    float scale;
    float time;
    uint pointCount;
    float pad;
};

struct VSOut {
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

struct GSOut {
    float4 pos : SV_Position;
    float4 col : COLOR0;
};

VSOut VSMain(uint vid : SV_VertexID) {
    float t = float(vid) / max(float(pointCount), 1.0);
    float a = t * 6.2831853 * 12.0 + time;
    float r = 0.1 + 0.75 * t;
    VSOut o;
    o.pos = float4(cos(a) * r, sin(a) * r, 0.0, 1.0);
    o.col = float4(0.5 + 0.5 * sin(t * 6.28318 + float3(0.0, 2.1, 4.2)), 1.0);
    return o;
}

// Expand each input point into a 4-triangle star (12 vertices)
[maxvertexcount(12)]
void GSMain(point VSOut input[1], inout TriangleStream<GSOut> stream) {
    float2 c = input[0].pos.xy;
    float4 col = input[0].col;
    float s = scale;

    for (int i = 0; i < 4; ++i) {
        float a0 = float(i) * 1.5707963;
        float a1 = a0 + 0.7853981;
        GSOut v0, v1, v2;
        v0.pos = float4(c, 0.0, 1.0);
        v0.col = col;
        v1.pos = float4(c + float2(cos(a0), sin(a0)) * s, 0.0, 1.0);
        v1.col = col * 0.8;
        v2.pos = float4(c + float2(cos(a1), sin(a1)) * s, 0.0, 1.0);
        v2.col = col * 0.6;
        stream.Append(v0);
        stream.Append(v1);
        stream.Append(v2);
        stream.RestartStrip();
    }
}

float4 PSMain(GSOut input) : SV_Target0 {
    return input.col;
}
