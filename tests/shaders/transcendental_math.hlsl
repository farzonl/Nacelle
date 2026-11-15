StructuredBuffer<float> InX : register(t0);
StructuredBuffer<float> InY : register(t1);
RWStructuredBuffer<float> Out : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    float x = InX[i];
    float y = InY[i];

    Out[0 * 64 + i]  = exp2(x);
    Out[1 * 64 + i]  = exp(x);
    Out[2 * 64 + i]  = ldexp(x, y);
    Out[3 * 64 + i]  = log2(x);
    Out[4 * 64 + i]  = log(x);
    Out[5 * 64 + i]  = log10(x);
    Out[6 * 64 + i]  = pow(abs(x), y);
    Out[7 * 64 + i]  = sin(x);
    Out[8 * 64 + i]  = cos(x);
    Out[9 * 64 + i]  = tan(x);
    Out[10 * 64 + i] = asin(x);
    Out[11 * 64 + i] = acos(x);
    Out[12 * 64 + i] = atan(x);
    Out[13 * 64 + i] = sinh(x);
    Out[14 * 64 + i] = cosh(x);
    Out[15 * 64 + i] = tanh(x);
}
