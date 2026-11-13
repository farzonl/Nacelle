struct WaveOut {
    uint laneCount;
    uint laneIndex;
    uint isFirst;
    uint sumUniform;
    uint minUniform;
    uint maxUniform;
    uint prefixUniform;
    uint ballotUniform;
    uint firstValUniform;
    uint anyTrueDiv;
    uint allTrueDiv;
    uint sumDiv;
};

RWStructuredBuffer<WaveOut> waveResults : register(u0);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint tid = dtid.x;
    WaveOut o;
    o.laneCount = WaveGetLaneCount();
    o.laneIndex = WaveGetLaneIndex();
    o.isFirst = WaveIsFirstLane() ? 1u : 0u;

    uint v = tid + 1u;
    o.sumUniform = WaveActiveSum(v);
    o.minUniform = WaveActiveMin(v);
    o.maxUniform = WaveActiveMax(v);
    o.prefixUniform = WavePrefixSum(v);
    o.ballotUniform = WaveActiveBallot((tid & 1u) == 0u).x;
    o.firstValUniform = WaveReadLaneFirst(v);

    o.anyTrueDiv = 0u;
    o.allTrueDiv = 0u;
    o.sumDiv = 0u;
    if ((tid % 3u) != 0u) {
        o.anyTrueDiv = WaveActiveAnyTrue((tid & 7u) == 5u) ? 1u : 0u;
        o.allTrueDiv = WaveActiveAllTrue(v > 0u) ? 1u : 0u;
        o.sumDiv = WaveActiveSum(v);
    }

    waveResults[tid] = o;
}
