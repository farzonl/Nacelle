struct Payload {
    float3 color;
    uint depth;
    float3 throughput;
    uint isShadow;
};

RaytracingAccelerationStructure Scene : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

float3 getNormal(uint primIdx, float2 bary) {
    float a = float(primIdx & 7u) * 0.785398f + bary.x * 0.5f;
    float b = float((primIdx >> 3) & 7u) * 0.392699f + bary.y * 0.5f;
    return normalize(float3(sin(a) * cos(b), cos(a), sin(a) * sin(b)));
}

[shader("raygeneration")]
void RayGen() {
    uint2 idx = DispatchRaysIndex().xy;
    uint2 dim = DispatchRaysDimensions().xy;

    float2 uv = (float2(idx) + 0.5f) / float2(dim);
    float2 ndc = uv * 2.0f - 1.0f;

    RayDesc ray;
    ray.Origin = float3(0.0f, 0.0f, -6.5f);
    ray.Direction = normalize(float3(ndc.x, ndc.y, 1.35f));
    ray.TMin = 0.001f;
    ray.TMax = 100.0f;

    Payload p;
    p.color = float3(0.0f, 0.0f, 0.0f);
    p.depth = 0u;
    p.throughput = float3(1.0f, 1.0f, 1.0f);
    p.isShadow = 0u;

    TraceRay(Scene, RAY_FLAG_NONE, 0xFFu, 0, 1, 0, ray, p);

    uint r8 = min(255u, uint(p.color.x * 255.0f));
    uint g8 = min(255u, uint(p.color.y * 255.0f));
    uint b8 = min(255u, uint(p.color.z * 255.0f));
    Output[idx.y * dim.x + idx.x] = (r8) | (g8 << 8) | (b8 << 16) | 0xFF000000u;
}

[shader("miss")]
void MissMain(inout Payload p) {
    if (p.isShadow != 0u) {
        p.isShadow = 2u; // Shadow ray missed -> lit!
        return;
    }
    float sky = 0.5f * (WorldRayDirection().y + 1.0f);
    p.color += p.throughput * lerp(float3(0.15f, 0.2f, 0.3f), float3(0.55f, 0.75f, 0.95f), sky);
}

[shader("closesthit")]
void ClosestHitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
    if (p.isShadow != 0u) {
        p.isShadow = 1u; // Occluded
        return;
    }

    float3 hitPos = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    uint instId = InstanceID();
    uint primId = PrimitiveIndex();
    float3 n = getNormal(primId + instId, attr.barycentrics);
    if (dot(n, WorldRayDirection()) > 0.0f) n = -n;

    float3 lightDir = normalize(float3(0.6f, 0.8f, -0.4f));

    // Shadow ray
    Payload sp;
    sp.color = float3(0.0f, 0.0f, 0.0f);
    sp.depth = p.depth + 1u;
    sp.throughput = float3(0.0f, 0.0f, 0.0f);
    sp.isShadow = 1u;

    RayDesc sray;
    sray.Origin = hitPos + n * 0.002f;
    sray.Direction = lightDir;
    sray.TMin = 0.001f;
    sray.TMax = 50.0f;
    TraceRay(Scene, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFFu, 0, 1, 0, sray, sp);

    float vis = (sp.isShadow == 2u) ? 1.0f : 0.15f;
    float ndotl = max(dot(n, lightDir), 0.05f);
    float3 albedo = float3(
        0.35f + 0.55f * float((instId * 5u) & 7u) / 7.0f,
        0.35f + 0.55f * float((instId * 3u) & 7u) / 7.0f,
        0.35f + 0.55f * float((instId * 7u) & 7u) / 7.0f);
    p.color += p.throughput * albedo * ndotl * vis * 0.65f;

    if (p.depth < 2u) {
        p.depth += 1u;
        p.throughput *= albedo * 0.55f;
        RayDesc rray;
        rray.Origin = hitPos + n * 0.002f;
        rray.Direction = reflect(WorldRayDirection(), n);
        rray.TMin = 0.001f;
        rray.TMax = 100.0f;
        TraceRay(Scene, RAY_FLAG_NONE, 0xFFu, 0, 1, 0, rray, p);
    }
}
