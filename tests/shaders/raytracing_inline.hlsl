RaytracingAccelerationStructure Scene : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[numthreads(4, 1, 1)]
void RayQueryMain(uint3 tid : SV_DispatchThreadID) {
    RayDesc ray;
    ray.TMin = 0.01;
    ray.TMax = 100.0;

    if (tid.x == 0u) {
        // Hit triangle at z = 0, barycentrics (0.25, 0.25), instanceID 42
        ray.Origin = float3(0.0, 0.0, 2.0);
        ray.Direction = float3(0.0, 0.0, -1.0);
    } else if (tid.x == 1u) {
        // Hit procedural AABB at x = 5, z in [-0.5, 0.5]
        ray.Origin = float3(5.0, 0.0, 2.0);
        ray.Direction = float3(0.0, 0.0, -1.0);
    } else if (tid.x == 2u) {
        // Miss everything
        ray.Origin = float3(20.0, 0.0, 2.0);
        ray.Direction = float3(0.0, 0.0, -1.0);
    } else {
        // Masked out instance (mask 0x02 vs instance mask 0x01)
        ray.Origin = float3(0.0, 0.0, 2.0);
        ray.Direction = float3(0.0, 0.0, -1.0);
    }

    uint mask = (tid.x == 3u) ? 0x02u : 0xFFu;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(Scene, RAY_FLAG_NONE, mask, ray);
    while (q.Proceed()) {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE) {
            q.CommitProceduralPrimitiveHit(1.5);
        }
    }

    uint base = tid.x * 4u;
    Output[base + 0u] = (uint)q.CommittedStatus();
    Output[base + 1u] = q.CommittedStatus() != COMMITTED_NOTHING ? q.CommittedInstanceID() : 0xFFFFFFFFu;
    Output[base + 2u] = asuint(q.CommittedStatus() != COMMITTED_NOTHING ? q.CommittedRayT() : 0.0);
    Output[base + 3u] = asuint(q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedTriangleBarycentrics().x : 0.0);
}
