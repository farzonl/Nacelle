RaytracingAccelerationStructure Scene : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

float3 getNormal(uint primIdx, float2 bary) {
    float a = float(primIdx & 7u) * 0.785398f + bary.x * 0.5f;
    float b = float((primIdx >> 3) & 7u) * 0.392699f + bary.y * 0.5f;
    return normalize(float3(sin(a) * cos(b), cos(a), sin(a) * sin(b)));
}

[numthreads(8, 8, 1)]
void RayQueryMain(uint3 tid : SV_DispatchThreadID) {
    uint dim = 256u;
    if (tid.x >= 512u || tid.y >= 512u) return;

    float2 uv = (float2(tid.xy) + 0.5f) / float(dim);
    float2 ndc = uv * 2.0f - 1.0f;

    RayDesc ray;
    ray.Origin = float3(0.0f, 0.0f, -6.5f);
    ray.Direction = normalize(float3(ndc.x, ndc.y, 1.35f));
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;

    float3 accum = float3(0.0f, 0.0f, 0.0f);
    float3 throughput = float3(1.0f, 1.0f, 1.0f);
    float3 lightDir = normalize(float3(0.6f, 0.8f, -0.4f));

    for (uint bounce = 0; bounce < 4u; ++bounce) {
        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(Scene, RAY_FLAG_NONE, 0xFFu, ray);
        while (q.Proceed()) {}

        if (q.CommittedStatus() == COMMITTED_NOTHING) {
            float sky = 0.5f * (ray.Direction.y + 1.0f);
            accum += throughput * lerp(float3(0.15f, 0.2f, 0.3f), float3(0.55f, 0.75f, 0.95f), sky);
            break;
        }

        float tHit = q.CommittedRayT();
        uint instId = q.CommittedInstanceID();
        uint primId = q.CommittedPrimitiveIndex();
        float2 bary = q.CommittedTriangleBarycentrics();
        float3 hitPos = ray.Origin + ray.Direction * tHit;
        float3 n = getNormal(primId + instId, bary);
        if (dot(n, ray.Direction) > 0.0f) n = -n;

        // Shadow ray
        RayDesc sray;
        sray.Origin = hitPos + n * 0.002f;
        sray.Direction = lightDir;
        sray.TMin = 0.001f;
        sray.TMax = 50.0f;
        RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> sq;
        sq.TraceRayInline(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFFu, sray);
        while (sq.Proceed()) {}

        float vis = (sq.CommittedStatus() == COMMITTED_NOTHING) ? 1.0f : 0.15f;
        float ndotl = max(dot(n, lightDir), 0.05f);
        float3 albedo = float3(
            0.35f + 0.55f * float((instId * 5u) & 7u) / 7.0f,
            0.35f + 0.55f * float((instId * 3u) & 7u) / 7.0f,
            0.35f + 0.55f * float((instId * 7u) & 7u) / 7.0f);
        accum += throughput * albedo * ndotl * vis * 0.65f;
        throughput *= albedo * 0.55f;

        ray.Origin = hitPos + n * 0.002f;
        ray.Direction = reflect(ray.Direction, n);
        ray.TMin = 0.001f;
        ray.TMax = 100.0f;
    }

    uint r8 = min(255u, uint(accum.x * 255.0f));
    uint g8 = min(255u, uint(accum.y * 255.0f));
    uint b8 = min(255u, uint(accum.z * 255.0f));
    Output[tid.y * 512u + tid.x] = (r8) | (g8 << 8) | (b8 << 16) | 0xFF000000u;
}
