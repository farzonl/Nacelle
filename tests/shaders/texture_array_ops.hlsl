Texture2DArray<float4> Tex2DArr : register(t0);
Texture1DArray<float4> Tex1DArr : register(t1);
TextureCubeArray<float4> TexCubeArr : register(t2);
Texture3D<float4> Tex3D : register(t3);

RWTexture2DArray<float4> RWTex2DArr : register(u0);
RWTexture1DArray<float4> RWTex1DArr : register(u1);
RWStructuredBuffer<float4> Out : register(u2);

SamplerState LinearClampSamp : register(s0);
SamplerState MinSamp : register(s1);

[numthreads(1, 1, 1)]
void CSMain() {
    // 0: Texture2DArray.Load with offset
    Out[0] = Tex2DArr.Load(int4(0, 0, 1, 0), int2(1, 1));
    // 1: Texture1DArray.Load (slice in y) and offset
    Out[1] = Tex1DArr.Load(int3(0, 2, 0), 1);
    // 2: RWTexture1DArray operator[] read
    Out[2] = RWTex1DArr[uint2(1, 1)];
    // 3: RWTexture2DArray operator[] read
    Out[3] = RWTex2DArr[uint3(1, 0, 2)];

    // Write via RWTexture1DArray/RWTexture2DArray operator[]
    RWTex1DArr[uint2(0, 2)] = float4(0.25, 0.5, 0.75, 1.0);
    RWTex2DArr[uint3(0, 1, 1)] = float4(0.125, 0.25, 0.375, 1.0);

    // 4: TextureCubeArray.SampleLevel (cube 1, +X face)
    Out[4] = TexCubeArr.SampleLevel(LinearClampSamp, float4(1, 0, 0, 1), 0.0);

    // 5: Texture3D.SampleGrad with Z-only gradient (selects mip 1)
    Out[5] = Tex3D.SampleGrad(LinearClampSamp, float3(0.25, 0.25, 0.375), float3(0, 0, 0), float3(0, 0, 2.0));

    // 6: MIN reduction static sampler across 2x2 texels
    Out[6] = Tex2DArr.SampleLevel(MinSamp, float3(0.5, 0.5, 0), 0.0);

    // 7..10: GetDimensions
    uint w1, s1, l1;
    Tex1DArr.GetDimensions(0, w1, s1, l1);
    Out[7] = float4(w1, s1, l1, 0);

    uint w2, h2, s2, l2;
    Tex2DArr.GetDimensions(0, w2, h2, s2, l2);
    Out[8] = float4(w2, h2, s2, l2);

    uint w3, h3, d3, l3;
    Tex3D.GetDimensions(1, w3, h3, d3, l3);
    Out[9] = float4(w3, h3, d3, l3);

    uint wc, hc, nc, lc;
    TexCubeArr.GetDimensions(0, wc, hc, nc, lc);
    Out[10] = float4(wc, hc, nc, lc);
}
