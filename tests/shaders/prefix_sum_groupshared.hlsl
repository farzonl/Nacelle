StructuredBuffer<uint> inData : register(t0);
RWStructuredBuffer<uint> outScan : register(u0);

groupshared uint sdata[64];

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    uint lid = gtid.x;
    sdata[lid] = inData[dtid.x];
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele inclusive prefix sum across 64 threads with barrier sync per step
    [unroll]
    for (uint offset = 1u; offset < 64u; offset <<= 1u) {
        uint addend = (lid >= offset) ? sdata[lid - offset] : 0u;
        GroupMemoryBarrierWithGroupSync();
        sdata[lid] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    outScan[dtid.x] = sdata[lid];
}
