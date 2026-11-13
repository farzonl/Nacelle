RWStructuredBuffer<uint> Out : register(u0);

[numthreads(1, 1, 1)]
void CSMain(uint GI : SV_GroupIndex) {
    Out.DecrementCounter();
    Out.DecrementCounter();
    Out.DecrementCounter();
    Out[GI] = Out.DecrementCounter();
}
