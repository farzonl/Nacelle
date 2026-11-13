StructuredBuffer<float4> InFlt : register(t0);
StructuredBuffer<uint4> InU32 : register(t1);
StructuredBuffer<int4> InI32 : register(t2);
RWStructuredBuffer<float4> OutFlt : register(u0);
RWStructuredBuffer<uint4> OutU32 : register(u1);

[numthreads(1,1,1)]
void CSMain() {
  // 0: degrees, 1: radians, 2: sqrt, 3: rsqrt, 4: log, 5: log10
  OutFlt[0] = degrees(InFlt[0]);
  OutFlt[1] = radians(InFlt[0]);
  OutFlt[2] = sqrt(InFlt[0]);
  OutFlt[3] = rsqrt(InFlt[0]);
  OutFlt[4] = log(InFlt[0]);
  OutFlt[5] = log10(InFlt[0]);

  OutU32[0] = firstbithigh(InU32[0]);
  OutU32[1] = firstbithigh(InI32[0]);
}
