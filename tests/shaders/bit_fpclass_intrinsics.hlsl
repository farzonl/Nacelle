StructuredBuffer<int64_t4>  InI64  : register(t0);
StructuredBuffer<uint64_t4> InU64  : register(t1);
StructuredBuffer<float4>    InFlt  : register(t2);

RWStructuredBuffer<uint4>   OutBits : register(u0);
RWStructuredBuffer<uint4>   OutAdj  : register(u1);
RWStructuredBuffer<uint4>   OutFPC  : register(u2);

[numthreads(1, 1, 1)]
void CSMain() {
    // 64-bit countbits (signed and unsigned)
    OutBits[0] = countbits(InI64[0]);
    OutBits[1] = countbits(InU64[0]);

    // 64-bit firstbitlow and firstbithigh (unsigned and signed)
    OutBits[2] = firstbitlow(InU64[1]);
    OutBits[3] = firstbithigh(InU64[1]);
    OutBits[4] = firstbithigh(InI64[1]);

    // Adjacent partial component stores with 64-bit countbits
    OutAdj[0] = countbits(InI64[0]);
    OutAdj[1].xyz = countbits(InI64[0].xyz);
    OutAdj[1].w = countbits(InI64[0].w);
    OutAdj[2].xy = countbits(InI64[0].xy);

    // Float special-value classification: vector, swizzle, and matrix forms
    OutFPC[0] = uint4(isinf(InFlt[0]));
    bool4 tmpInf = bool4(isinf(InFlt[0].xyz), isinf(InFlt[0].w));
    OutFPC[1] = uint4(tmpInf);
    OutFPC[2].xy = uint2(isinf(InFlt[0].xy));

    bool2x4 mat24 = isinf(float2x4(InFlt[1], InFlt[2]));
    OutFPC[3] = uint4(mat24[0]);
    OutFPC[4] = uint4(mat24[1]);

    bool3x2 mat32 = isinf(float3x2(InFlt[1].xyz, InFlt[2].xyz));
    OutFPC[5] = uint4(mat32[0], mat32[1]);
    OutFPC[6] = uint4(mat32[2], false, false);

    OutFPC[7] = uint4(isnan(InFlt[0]));
    OutFPC[8] = uint4(isfinite(InFlt[0]));
}
