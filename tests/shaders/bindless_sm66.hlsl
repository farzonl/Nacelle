cbuffer HeapIndices : register(b0) {
    uint srvIdxA;
    uint srvIdxB;
    uint uavIdxOut;
    uint count;
};

[numthreads(32, 1, 1)]
void CSMain(uint3 dtid : SV_DispatchThreadID) {
    if (dtid.x >= count) return;
    ByteAddressBuffer bufA = ResourceDescriptorHeap[srvIdxA];
    ByteAddressBuffer bufB = ResourceDescriptorHeap[srvIdxB];
    RWByteAddressBuffer bufOut = ResourceDescriptorHeap[uavIdxOut];

    uint a = bufA.Load(dtid.x * 4);
    uint b = bufB.Load(dtid.x * 4);
    bufOut.Store(dtid.x * 4, a * 3u + b + 7u);
}
