RWBuffer<int> In : register(u0);
RWBuffer<int> RowOut : register(u1);
RWBuffer<int> ColOut : register(u2);

[numthreads(1,1,1)]
void CSMain() {
  for(int I = 0; I < 4; I++) {
    int4x4 A;
    int4x4 B;
    switch(I) {
      case 0:
        A._m00_m01_m02_m03 = In[I].xxxx;
        B._m00_m10_m20_m30 = In[I].xxxx;
        break;
      case 1:
        A._m10_m11_m12_m13 = In[I].xxxx;
        B._m01_m11_m21_m31 = In[I].xxxx;
        break;
      case 2:
        A._m20_m21_m22_m23 = In[I].xxxx;
        B._m02_m12_m22_m32 = In[I].xxxx;
        break;
      case 3:
        A._m30_m31_m32_m33 = In[I].xxxx;
        B._m03_m13_m23_m33 = In[I].xxxx;
        break;
    }
    int4 vec1;
    int4 vec2;
    switch(I) {
      case 0:
        vec1 = A._m00_m01_m02_m03;
        vec2 = B._m00_m10_m20_m30;
        break;
      case 1:
        vec1 = A._m10_m11_m12_m13;
        vec2 = B._m01_m11_m21_m31;
        break;
      case 2:
        vec1 = A._m20_m21_m22_m23;
        vec2 = B._m02_m12_m22_m32;
        break;
      case 3:
        vec1 = A._m30_m31_m32_m33;
        vec2 = B._m03_m13_m23_m33;
        break;
    }
    for(int VI = 0; VI < 4; VI++) {
      RowOut[I*4+VI] = vec1[VI];
      ColOut[VI*4+I] = vec2[VI];
    }
  }
}
