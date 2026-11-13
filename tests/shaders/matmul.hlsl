cbuffer MatDims : register(b0) {
    uint M;
    uint N;
    uint K;
};

StructuredBuffer<float> matA : register(t0);
StructuredBuffer<float> matB : register(t1);
RWStructuredBuffer<float> matC : register(u0);

[numthreads(8, 8, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint row = dtid.y;
    uint col = dtid.x;
    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (uint k = 0; k < K; ++k) {
        sum += matA[row * K + k] * matB[k * N + col];
    }
    matC[row * N + col] = sum;
}
