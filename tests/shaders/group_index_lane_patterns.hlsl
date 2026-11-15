cbuffer DynamicTable : register(b0) {
    uint4 TableVec[8];
};

struct PackedRecord {
    uint tag;
    float weight;
    uint payload;
};

StructuredBuffer<PackedRecord> Records : register(t0);
RWStructuredBuffer<uint> OutGroupIndex : register(u0);

groupshared float4 SharedVec[32];

[numthreads(16, 2, 1)]
void GroupIndexCS(uint gi : SV_GroupIndex, uint3 gid : SV_GroupID) {
    uint globalIdx = gid.x * 32u + gi;
    PackedRecord rec = Records[globalIdx];
    uint cbVal = TableVec[gi & 7u].x;

    SharedVec[gi] = float4(
        (float)(gi + 1u),
        rec.weight,
        (float)(rec.tag + cbVal),
        (float)rec.payload);

    GroupMemoryBarrierWithGroupSync();

    uint mateA = (gi + 1u) & 31u;
    uint mateB = (gi + 17u) & 31u;
    uint accum = (uint)SharedVec[mateA].x
               + (uint)SharedVec[gi].y
               + (uint)SharedVec[mateB].z
               + (uint)SharedVec[gi].w;
    OutGroupIndex[globalIdx] = accum;
}

StructuredBuffer<uint> ColWeights : register(t0);
RWStructuredBuffer<uint> OutSubWave : register(u0);

groupshared uint SharedCols[4];

[numthreads(4, 4, 1)]
void SubWaveGroupThreadCS(uint3 dtid : SV_DispatchThreadID, uint3 gtid : SV_GroupThreadID) {
    if (gtid.y == 0u) {
        SharedCols[gtid.x] = gtid.x * 10u + 3u;
    }
    GroupMemoryBarrierWithGroupSync();

    uint idx = dtid.y * 8u + dtid.x;
    OutSubWave[idx] = ColWeights[dtid.x] * 100u + SharedCols[gtid.x] + dtid.y;
}
