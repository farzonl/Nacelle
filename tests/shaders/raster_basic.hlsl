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

float4 PSAltTintMain(PSIn input) : SV_Target {
    return float4(1.0 - input.col.rgb, input.col.a);
}

struct VSPackedInstIn {
    float2 pos : POSITION;
    float4 colA : COLOR0;
    float4 colB : COLOR1;
    float2 instOff : TEXCOORD0;
};

PSIn VSPackedInstMain(VSPackedInstIn input) {
    PSIn o;
    o.pos = float4(input.pos + input.instOff, 0.5, 1.0);
    o.col = float4(input.colA.rgb * 0.5 + input.colB.rgb * 0.5, input.colA.a);
    return o;
}

struct VSHeavyIn {
    float3 pos : POSITION;
    float3 nrm : NORMAL;
    float2 uv  : TEXCOORD0;
    float4 col : COLOR;
    uint   vid : SV_VertexID;
    uint   iid : SV_InstanceID;
};

struct VSHeavyOut {
    float4 pos : SV_Position;
    float4 extraUvNrm : TEXCOORD0;
    float4 col : COLOR;
};

VSHeavyOut VSHeavyTransformMain(VSHeavyIn input) {
    float phase = float(input.iid) * 0.125 + (input.uv.x + input.uv.y) * 3.14159265;
    float s = sin(phase);
    float c = cos(phase);
    float3 p = input.pos;
    float3 pRot = float3(p.x * c - p.y * s * 0.02, p.x * s * 0.02 + p.y * c, p.z + 0.05 * s);
    float3 n = normalize(input.nrm + float3(0.1 * s, 0.1 * c, 1.0));
    float ndl = saturate(dot(n, normalize(float3(0.3, 0.5, 0.8))));

    VSHeavyOut o;
    o.pos = float4(pRot, 1.0);
    o.extraUvNrm = float4(input.uv, n.xy);
    o.col = float4(input.col.rgb * (0.25 + 0.75 * ndl), input.col.a);
    return o;
}
