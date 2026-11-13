ByteAddressBuffer BufA : register(t0);
RWByteAddressBuffer BufB : register(u1);
RWStructuredBuffer<uint> Out : register(u0);

[numthreads(1, 1, 1)]
void CSMain() {
    uint dimA = 0;
    uint dimB = 0;
    BufA.GetDimensions(dimA);
    BufB.GetDimensions(dimB);
    Out[0] = dimA;
    Out[1] = dimB;
}
