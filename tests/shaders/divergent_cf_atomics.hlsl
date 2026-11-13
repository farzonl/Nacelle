RWByteAddressBuffer outAtomics : register(u0);
RWStructuredBuffer<uint> outValues : register(u1);

[numthreads(64, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    uint tid = dtid.x;
    uint acc = 0;

    // Divergent nested if/else + variable-trip while loop across SIMD lanes
    if ((tid & 1u) == 0u) {
        uint n = (tid & 7u) + 1u;
        uint i = 0;
        while (i < n) {
            acc += (i + 1u) * 2u;
            i += 1u;
        }
        if ((tid & 2u) != 0u) {
            acc += 100u;
        }
    } else {
        uint n = (tid % 5u) + 2u;
        uint i = 0;
        while (i < n) {
            acc += (i * 3u + 1u);
            i += 1u;
        }
    }

    outValues[tid] = acc;

    uint prev;
    outAtomics.InterlockedAdd(0, 1u, prev);
    outAtomics.InterlockedMax(4, acc, prev);
    outAtomics.InterlockedAdd(8, acc, prev);
}
