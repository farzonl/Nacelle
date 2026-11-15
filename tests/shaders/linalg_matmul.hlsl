// Reference SM 6.10 Linear Algebra (Cooperative Matrix) tiled GEMM shader.
// Equivalent DXIL IR is assembled from linalg_matmul.ll until DXC SM 6.10 is installed.
#include <dx/linalg.h>

cbuffer MatDims : register(b0) {
    uint M;
    uint N;
    uint K;
};

ByteAddressBuffer matA : register(t0);
ByteAddressBuffer matB : register(t1);
RWByteAddressBuffer matC : register(u0);

[numthreads(32, 1, 1)]
void CSMain(uint3 gid : SV_GroupID) {
    uint rowBase = gid.y * 32;
    uint colBase = gid.x * 32;
    if (rowBase >= M || colBase >= N) return;

    dx::linalg::Matrix<dx::linalg::ComponentType::F32, 32, 32,
                       dx::linalg::MatrixUse::Accumulator,
                       dx::linalg::MatrixScope::Wave> acc =
        dx::linalg::Matrix<dx::linalg::ComponentType::F32, 32, 32,
                           dx::linalg::MatrixUse::Accumulator,
                           dx::linalg::MatrixScope::Wave>::Splat(0.0f);

    for (uint k = 0; k < K; k += 32) {
        dx::linalg::MatrixRef<dx::linalg::ComponentType::F32, 32, 32,
                              dx::linalg::MATRIX_LAYOUT_ROW_MAJOR> aRef = {
            matA, (rowBase * K + k) * 4, K * 4};
        dx::linalg::MatrixRef<dx::linalg::ComponentType::F32, 32, 32,
                              dx::linalg::MATRIX_LAYOUT_ROW_MAJOR> bRef = {
            matB, (k * N + colBase) * 4, N * 4};

        auto matATile = dx::linalg::Matrix<dx::linalg::ComponentType::F32, 32, 32,
                                           dx::linalg::MatrixUse::A,
                                           dx::linalg::MatrixScope::Wave>::Load(aRef);
        auto matBTile = dx::linalg::Matrix<dx::linalg::ComponentType::F32, 32, 32,
                                           dx::linalg::MatrixUse::B,
                                           dx::linalg::MatrixScope::Wave>::Load(bRef);
        acc = dx::linalg::MultiplyAccumulate(matATile, matBTile, acc);
    }

    acc.Store(matC, (rowBase * N + colBase) * 4, N * 4,
              dx::linalg::MATRIX_LAYOUT_ROW_MAJOR);
}
