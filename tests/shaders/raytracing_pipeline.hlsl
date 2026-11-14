struct Payload {
    uint value;
};

RaytracingAccelerationStructure Scene : register(t0);
RWStructuredBuffer<uint> Output : register(u0);

[shader("raygeneration")]
void RayGen() {
    uint idx = DispatchRaysIndex().x;
    uint dim = DispatchRaysDimensions().x;

    Payload p;
    p.value = 0;

    RayDesc ray;
    // idx 0 -> hits triangle (0,1,0), (-1,-1,0), (1,-1,0) at (0,0,0) -> ClosestHitMain
    // idx 1 -> misses triangle at (5,0,0) -> MissMain
    ray.Origin = float3(idx == 0 ? 0.0f : 5.0f, 0.0f, 1.0f);
    ray.Direction = float3(0.0f, 0.0f, -1.0f);
    ray.TMin = 0.0f;
    ray.TMax = 100.0f;

    TraceRay(Scene, RAY_FLAG_NONE, 0xFF, 0, 1, 0, ray, p);
    Output[idx] = p.value + dim * 1000u;
}

[shader("miss")]
void MissMain(inout Payload p) {
    p.value = 0xDEADu;
}

[shader("closesthit")]
void ClosestHitMain(inout Payload p, in BuiltInTriangleIntersectionAttributes attr) {
    uint prim = PrimitiveIndex();
    uint b0 = (uint)(attr.barycentrics.x * 100.0f + 0.5f);
    uint b1 = (uint)(attr.barycentrics.y * 100.0f + 0.5f);
    p.value = 100u + prim + b0 + b1;
}
