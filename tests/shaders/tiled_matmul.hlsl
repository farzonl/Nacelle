cbuffer MatDims : register(b0) {
    uint M;
    uint N;
    uint K;
};

StructuredBuffer<float> matA : register(t0);
StructuredBuffer<float> matB : register(t1);
RWStructuredBuffer<float> matC : register(u0);

groupshared float tileA[16][16];
groupshared float tileB[16][16];

[numthreads(16, 16, 1)]
void CSMain(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint lx = gtid.x;
    uint ly = gtid.y;
    uint row = gid.y * 16u + ly;
    uint col = gid.x * 16u + lx;

    float acc = 0.0f;
    uint numTiles = (K + 15u) / 16u;
    for (uint t = 0u; t < numTiles; ++t) {
        uint aCol = t * 16u + lx;
        float aVal = 0.0f;
        if (row < M && aCol < K) {
            aVal = matA[row * K + aCol];
        }
        tileA[ly][lx] = aVal;

        uint bRow = t * 16u + ly;
        float bVal = 0.0f;
        if (bRow < K && col < N) {
            bVal = matB[bRow * N + col];
        }
        tileB[ly][lx] = bVal;

        GroupMemoryBarrierWithGroupSync();

        [unroll]
        for (uint p = 0u; p < 16u; ++p) {
            acc += tileA[ly][p] * tileB[p][lx];
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < M && col < N) {
        matC[row * N + col] = acc;
    }
}
