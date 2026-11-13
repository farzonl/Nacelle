cbuffer ShadertoyCB : register(b0) {
    float3 iResolution;
    float iTime;
    float4 iMouse;
};

float glsl_mod(float x, float y) { return x - y * floor(x / y); }
float2 glsl_mod(float2 x, float y) { return x - y * floor(x / y); }
float2 glsl_mod(float2 x, float2 y) { return x - y * floor(x / y); }
float3 glsl_mod(float3 x, float y) { return x - y * floor(x / y); }
float3 glsl_mod(float3 x, float3 y) { return x - y * floor(x / y); }
float4 glsl_mod(float4 x, float y) { return x - y * floor(x / y); }
float4 glsl_mod(float4 x, float4 y) { return x - y * floor(x / y); }

// Goldeen Shader
// By Noztol

float DistanceToSphere(float3 p, float r);
float DistanceToCylinder(float3 p, float r, float h);
float DistanceToTube(float3 p, float r, float w, float h);
float SmoothUnion(float a, float b, float r);
float SmoothBump(float lo, float hi, float w, float x);
float3 RotateToDirectionLimited(float3 v1, float3 v2, float3 p, float aMax);
float2 Rotate2D(float2 q, float a);
float MinimumComponent(float3 p);
float3 Hash3D(float p);

static float3 bodyDomain, lightDirection, bubbleGrid, eyePosition, viewPosition;
static float maxDistance, currentTime, mouthShape;
static int currentObjectId;

static const int idBody = 1, idVerticalFin = 2, idSideFin = 3, idTail = 4, idMouth = 5, idEye = 6, idBubble = 7, idHorn = 8;
static const float pi = 3.14159, phi = 1.618034;

float BubbleShapeSDF(float3 p, float3 pw, float3 vr)
{
  float3 b;
  float s, t, r, d, a;
  t = currentTime;
  vr -= 0.5;
  s = pow(frac(dot(vr, ((float3)(1.0))) * 100.0), 4.0);
  pw.y += t;
  pw = 7.0 * pw + 2.0 * pi * vr + float3(t, 0.0, t);
  d = sin((t + vr.y) * 3.0 * pi * (1.0 - s)) * 0.3 * vr.x * (1.0 - s);
  a = vr.z * t;
  b = d * float3(cos(a), 0.0, sin(a));
  r = 0.1 + s * (0.02 * dot(sin(pw), ((float3)(1.0))) - 0.05);
  return 0.9 * DistanceToSphere(p - 0.1 * b, 0.1 * r);
}

float DistanceToBubbles(float3 p, float3 cellId)
{
  float3 vr = Hash3D(dot(cellId, float3(31.0, 33.0, 35.0)));
  return (vr.x * smoothstep(2.0, 5.0, length(cellId)) > 0.9) ?
     BubbleShapeSDF(p - bubbleGrid * (cellId + 0.5), p, vr) : maxDistance;
}

float RaycastBubbles(float3 ro, float3 rd)
{
  float3 p, cellId, s;
  float hitDistance, d, eps;
  eps = 0.0005;
  if (rd.x == 0.0) rd.x = 0.001;
  if (rd.y == 0.0) rd.y = 0.001;
  if (rd.z == 0.0) rd.z = 0.001;
  
  hitDistance = eps;
  for (int j = 0; j < 120; j++) {
    p = ro + hitDistance * rd;
    p.y -= 0.4 * currentTime + 40.0;
    cellId = floor(p / bubbleGrid);
    d = DistanceToBubbles(p, cellId);
    s = (bubbleGrid * (cellId + step(0.0, rd)) - p) / rd;
    d = min(d, MinimumComponent(s) + eps);
    if (d < eps || hitDistance > maxDistance) break;
    hitDistance += d;
  }
  if (d >= eps) hitDistance = maxDistance;
  return hitDistance;
}

float BubbleGridSDF(float3 p)
{
  return DistanceToBubbles(p, floor(p / bubbleGrid));
}

float3 CalculateBubbleNormal(float3 p)
{
  float4 v;
  float2 e = float2(0.0001, -0.0001);
  p.y -= 0.4 * currentTime + 40.0;
  v = float4(BubbleGridSDF(p + e.xxx), BubbleGridSDF(p + e.xyy), BubbleGridSDF(p + e.yxy), BubbleGridSDF(p + e.yyx));
  return normalize(((float3)(v.x - v.y - v.z - v.w)) + 2.0 * v.yzw);
}

float GoldeenBodySDF(float3 p)
{
  float3 q;
  float db, dt;
  bodyDomain = p;
  p.x *= 1.0 + 1.5 * (p.y + 0.2) * (p.y + 0.2);
  p.xy *= 1.0 + 0.5 * smoothstep(0.0, 0.5, -p.z);
  
  q = (p - float3(0.0, 0.2, 0.5 * (p.y - 0.2) - 0.14)) * float3(1.4, 1.7, 0.6);
  dt = DistanceToSphere(q, 0.4 - 0.2 * smoothstep(0.0, 1.5, 1.3 - p.z));
  
  q = p * float3(1.2, 1.2, 1.0 + 0.5 * smoothstep(0.0, 0.5, -p.y - p.z));
  db = DistanceToSphere(q, 0.37 + smoothstep(0.0, 1.7, -p.z));
  
  q = bodyDomain;
  q.yz -= float2(0.1, 0.1);
  q.xy *= float2(0.5, 1.0) + float2(0.1, -0.4) * mouthShape;
  db = max(db, -DistanceToCylinder(q, 0.02, 0.3));
  
  return SmoothUnion(db, dt, 0.05);
}

float GoldeenMouthSDF(float3 p)
{
  p.yz -= float2(0.1, 0.34 - 2.0 * p.x * p.x);
  p.xy *= float2(0.5 + 0.1 * mouthShape, 1.0 - 0.5 * mouthShape);
  return DistanceToTube(p, 0.015, 0.0015, 0.03);
}

float GoldeenHornSDF(float3 p)
{
  float3 q = p;
  // Push the base deep into the forehead so it fully touches the head all the way around
  q.yz -= float2(0.24, 0.20); 
  q.yz = Rotate2D(q.yz, -0.7); // Tilt forward
  
  float height = 0.35;
  float y = clamp(q.y, 0.0, height);
  // Base is wide, top is perfectly sharp (0.0)
  float radius = lerp(0.07, 0.0, y / height); 
  
  // Create a sharp cone segment without subtraction rounding
  return max(length(q.xz) - radius, max(-q.y, q.y - height));
}

float GoldeenTailSDF(float3 p)
{
  float d, r, a;
  p.yz -= float2(0.11, -0.45);
  p.xz = Rotate2D(p.xz, 0.5 * cos(2.0 * currentTime - 3.0 * p.y + 5.0 * p.z));
  a = 0.003 * sin(32.0 * atan2(p.y, p.z));
  r = length(p.yz);
  d = min(0.01 - 0.008 * smoothstep(0.15, 0.25, r) - abs(p.x - a * smoothstep(0.04, 0.08, r)), 
          0.6 - 0.05 * p.y - 0.01 * cos(a * 1024.0) - r);
  d = -SmoothUnion(abs(p.y) + 0.3 * p.z, SmoothUnion(-0.3 * abs(p.y) - p.z, d, 0.02), 0.02);
  return d;
}

float GoldeenSideFinSDF(float3 p)
{
  float d, r, a, t, w;
  t = 5.0 * currentTime + 0.2 * sign(p.x);
  p.x = abs(p.x) - 0.26;
  p.xz = Rotate2D(p.xz, 0.4 * pi);
  w = 0.15 * (1.0 + 5.0 * length(p));
  p.yz = Rotate2D(p.yz, 0.2 + 0.5 * w * cos(t + 2.0 * atan2(p.x, -p.y)) - 0.5 * pi);
  p.xz = Rotate2D(p.xz, 1.2 + w * sin(t - w) - 0.5 * pi);
  a = atan2(p.x, -p.y);
  r = length(p.xy);
  d = min(0.01 - 0.008 * smoothstep(0.2, 0.3, r) - abs(p.z + 0.002 * sin(32.0 * a) * smoothstep(0.05, 0.08, r)),
          0.8 - 0.15 * smoothstep(1.0, 3.0, abs(a)) - 0.01 * cos(32.0 * a) - r);
  d = -0.9 * SmoothUnion(-0.2 * p.x + p.y, SmoothUnion(p.x - 0.7 * p.y, d, 0.02), 0.02);
  return d;
}

float GoldeenVerticalFinSDF(float3 p)
{
  float d, r, a, y;
  p.y = abs(p.y) - 0.26 - 0.02 * sign(p.y);
  p.z -= -0.1;
  y = smoothstep(0.0, 0.2, p.y);
  p.z *= 1.0 - 0.3 * y * y;
  a = -cos(4.0 * currentTime + 5.0 * (-p.y + p.z)) * (0.1 - 0.3 * p.z);
  p.xz = Rotate2D(p.xz, a);
  p.xy = Rotate2D(p.xy, a);
  a = 0.0025 * sin(32.0 * atan2(p.y, p.z));
  r = length(p.yz);
  d = min(0.01 - 0.009 * smoothstep(0.1, 0.2, r) - abs(p.x + a * smoothstep(0.04, 0.1, r)), 
          0.22 - 0.015 * cos(512.0 * a) - r);
  d = -SmoothUnion(p.y + 0.4 * p.z, SmoothUnion(-0.4 * p.y - p.z, d, 0.02), 0.02);
  return d;
}

float GoldeenSDF(float3 p)
{
  float dMin, d, a, c;
  c = cos(2.0 * currentTime);
  a = -0.5 * smoothstep(-0.2, 1.1, -p.z) * c;
  p.xz = Rotate2D(p.xz, 0.1 * c + a);
  p.xy = Rotate2D(p.xy, 0.5 * a);
  
  dMin = 0.3 * GoldeenBodySDF(p);
  currentObjectId = idBody;
  
  d = GoldeenMouthSDF(p);
  if (abs(d) < dMin) currentObjectId = idMouth;
  dMin = SmoothUnion(dMin, d, 0.01);
  
  d = GoldeenHornSDF(p);
  if (d < dMin) currentObjectId = idHorn;
  dMin = SmoothUnion(dMin, d, 0.04); // Blends the base of the horn smoothly into the head
  
  // 3-Tails implementation
  float3 pT1 = p;
  float3 pT2 = p; pT2.xy = Rotate2D(pT2.xy, 0.3); pT2.yz = Rotate2D(pT2.yz, -0.05);
  float3 pT3 = p; pT3.xy = Rotate2D(pT3.xy, -0.3); pT3.yz = Rotate2D(pT3.yz, -0.05);
  float dt0 = GoldeenTailSDF(pT1);
  float dt1 = GoldeenTailSDF(pT2);
  float dt2 = GoldeenTailSDF(pT3);
  d = 0.6 * SmoothUnion(dt0, SmoothUnion(dt1, dt2, 0.03), 0.03);
  
  if (d < dMin) currentObjectId = idTail;
  dMin = SmoothUnion(dMin, d, 0.01);
  
  d = 0.7 * GoldeenSideFinSDF(p);
  if (d < dMin) currentObjectId = idSideFin;
  dMin = SmoothUnion(dMin, d, 0.01);

  d = 0.7 * GoldeenVerticalFinSDF(p);
  if (d < dMin) currentObjectId = idVerticalFin;
  dMin = SmoothUnion(dMin, d, 0.01);
  
  p.x = abs(p.x);
  d = DistanceToSphere(p - eyePosition, 0.13);
  if (d < dMin) currentObjectId = idEye;
  dMin = SmoothUnion(dMin, d, 0.02);
  
  return dMin;
}

float RaycastGoldeen(float3 ro, float3 rd)
{
  float d, h;
  d = 0.0;
  for (int j = 0; j < 120; j++) {
    h = GoldeenSDF(ro + d * rd);
    d += h;
    if (h < 0.0005 || d > maxDistance) break;
  }
  return d;
}

float3 CalculateGoldeenNormal(float3 p)
{
  float4 v;
  float2 e = float2(0.0001, -0.0001);
  v = float4(GoldeenSDF(p + e.xxx), GoldeenSDF(p + e.xyy), GoldeenSDF(p + e.yxy), GoldeenSDF(p + e.yyx));
  return normalize(((float3)(v.x - v.y - v.z - v.w)) + 2.0 * v.yzw);
}

float CalculateGoldeenShadow(float3 ro, float3 rd)
{
  float sh, d, h;
  sh = 1.0;
  d = 0.01;
  for (int j = 0; j < 30; j++) {
    h = GoldeenSDF(ro + d * rd);
    sh = min(sh, smoothstep(0.0, 0.05 * d, h));
    d += h;
    if (sh < 0.05) break;
  }
  return 0.7 + 0.3 * sh;
}

float GetTurbulentLight(float3 p, float3 n, float t)
{
  float4 b;
  float2 q, qq;
  float c, tt;
  q = 2.0 * pi * glsl_mod(float2(dot(p.yzx, n), dot(p.zxy, n)), 1.0) - 256.0;
  t += 11.0;
  c = 0.0;
  qq = q;
  for (float j = 1.0; j <= 7.0; j++) {
    tt = t * (1.0 + 1.0 / j);
    b = sin(tt + float4(-qq + float2(0.5 * pi, 0.0), qq + float2(0.0, 0.5 * pi)));
    qq = q + tt + b.xy + b.zw;
    c += 1.0 / length(q / sin(qq));
  }
  return clamp(pow(abs(1.25 - abs(0.167 + 40.0 * c)), 8.0), 0.0, 1.0);
}

float GetWaterShadow(float3 rd)
{
  float2 p;
  float t, h;
  p = 20.0 * rd.xz / rd.y;
  t = currentTime * 2.0;
  h = sin(p.x * 2.0 + t * 0.77 + sin(p.y * 0.73 - t)) +
      sin(p.y * 0.81 - t * 0.89 + sin(p.x * 0.33 + t * 0.34)) +
      (sin(p.x * 1.43 - t) + sin(p.y * 0.63 + t)) * 0.5;
  h *= smoothstep(0.5, 1.0, rd.y) * 0.04;
  return h;
}

float3 GetBackgroundColor(float3 rd)
{
  float t, gd, b;
  t = 4.0 * currentTime;
  b = dot(float2(atan2(rd.x, rd.z), 0.5 * pi - acos(rd.y)), float2(2.0, sin(rd.x)));
  gd = clamp(sin(5.0 * b + t), 0.0, 1.0) * clamp(sin(3.5 * b - t), 0.0, 1.0) +
       clamp(sin(21.0 * b - t), 0.0, 1.0) * clamp(sin(17.0 * b + t), 0.0, 1.0);
  return float3(0.2, 0.5, 1.0) * (0.24 + 0.44 * (rd.y + 1.0) * (rd.y + 1.0)) * (1.0 + gd * 0.05);
}

float4 GetGoldeenColor(float3 p)
{
  float4 colorProperties;
  float3 eyeVec;
  float s;
  colorProperties = float4(1.0, 0.98, 0.95, 0.05); 
  
  if (currentObjectId == idBody) {
    colorProperties.rgb *= 0.85 + 0.15 * smoothstep(-0.1, 0.1, bodyDomain.y);
    float orangePatch = smoothstep(0.4, 0.5, sin(8.0 * bodyDomain.x) * cos(6.0 * bodyDomain.z));
    colorProperties.rgb = lerp(colorProperties.rgb, float3(1.0, 0.45, 0.0), orangePatch * 0.4 * smoothstep(0.0, 0.3, bodyDomain.y));
  } else if (currentObjectId == idMouth) {
    colorProperties = float4(1.0, 0.3, 0.6, 0.1); 
  } else if (currentObjectId == idHorn) {
    colorProperties = float4(1.0, 0.98, 0.95, 0.1); 
  } else if (currentObjectId == idSideFin || currentObjectId == idVerticalFin) {
    float radialDist = length(p.xy);
    colorProperties.rgb = lerp(float3(1.0, 0.4, 0.0), float3(1.0, 0.95, 0.9), smoothstep(0.2, 0.7, radialDist));
  } else if (currentObjectId == idTail) {
    float radialDist = length(p.xyz);
    colorProperties.rgb = lerp(float3(1.0, 0.4, 0.0), float3(1.0, 0.95, 0.9), smoothstep(0.4, 0.9, radialDist));
  } else if (currentObjectId == idEye) {
    float eyeOpen = SmoothBump(0.05, 0.95, 0.05, glsl_mod(0.5 * currentTime, 1.0));
    float openGapY = lerp(-0.01, 0.1, eyeOpen); 
    
    if (abs(p.y - eyePosition.y) > openGapY) {
      colorProperties.rgb = float3(1.0, 0.4, 0.0); 
      colorProperties.a = 0.05; 
    } else {
      eyeVec = eyePosition * float3(sign(p.x), 1.0, 1.0);
      
      // Bounded the rotation to 0.15 * pi (approx 27 degrees) so it cannot track past the eye socket constraints
      eyeVec = RotateToDirectionLimited(normalize(viewPosition - eyeVec), float3(sign(p.x), 0.0, 0.0), p - eyeVec, 0.15 * pi);
      s = length(eyeVec.yz);
      
      if (s < 0.025) {
        colorProperties.rgb = ((float3)(0.05)); // Black Pupil
      } else if (s < 0.06) {
        colorProperties.rgb = float3(0.2, 0.8, 0.3); // Green Iris
      } else {
        colorProperties.rgb = ((float3)(1.0)); // White Sclera
      }
      colorProperties.a = 0.4; 
    }
  }
  return colorProperties;
}

float3 RenderScene(float3 ro, float3 rd)
{
  float4 colorData;
  float3 normal, finalColor, bgColor;
  float distFish, distBubbles, specularity, shadow;
  
  bgColor = GetBackgroundColor(rd);
  distFish = RaycastGoldeen(ro, rd);
  distBubbles = RaycastBubbles(ro, rd);
  
  if (min(distBubbles, distFish) < maxDistance) {
    if (distFish < distBubbles) {
      ro += distFish * rd;
      normal = CalculateGoldeenNormal(ro);
      colorData = GetGoldeenColor(ro);
      finalColor = colorData.rgb;
      specularity = colorData.a;
      shadow = CalculateGoldeenShadow(ro, lightDirection);
      if (specularity >= 0.0) finalColor = lerp(finalColor, GetBackgroundColor(reflect(rd, normal)), 0.2);
    } else if (distBubbles < maxDistance) {
      ro += distBubbles * rd;
      normal = CalculateBubbleNormal(ro);
      finalColor = lerp(((float3)(1.0)), GetBackgroundColor(reflect(rd, normal)), 0.7);
      specularity = 0.5;
      shadow = 1.0;
      currentObjectId = idBubble;
    }
    
    if (specularity >= 0.0) {
      finalColor = finalColor * (0.3 + 0.2 * bgColor + 0.6 * shadow * max(dot(normal, lightDirection), 0.0)) +
                   specularity * shadow * pow(max(dot(normalize(lightDirection - rd), normal), 0.0), 32.0);
      finalColor += 0.3 * GetTurbulentLight(0.5 * ro, abs(normal), 0.5 * currentTime) * smoothstep(-0.3, -0.1, normal.y);
      if (currentObjectId == idBubble) finalColor *= 0.5 + 0.5 * clamp(rd.y + 1.0, 0.0, 1.5);
    } else {
      rd = reflect(rd, normal);
      finalColor = 0.5 * (GetBackgroundColor(rd) + GetWaterShadow(rd));
    }
    finalColor = lerp(finalColor, bgColor, smoothstep(0.3, 0.95, min(distBubbles, distFish) / maxDistance));
  } else {
    finalColor = bgColor + GetWaterShadow(rd);
  }
  
  return finalColor;
}

void mainImage(out float4 fragColor, in float2 fragCoord)
{
  float3x3 viewMatrix;
  float4 mousePointer;
  float3 ro, rd, finalColor;
  float2 canvas, uv, orientation, cosAngles, sinAngles;
  float elevation, azimuth, zoomFactor;
  
  canvas = iResolution.xy;
  uv = 2.0 * fragCoord.xy / canvas - 1.0;
  uv.x *= canvas.x / canvas.y;
  currentTime = iTime;
  mousePointer = iMouse;
  mousePointer.xy = mousePointer.xy / canvas - 0.5;
  
  azimuth = -pi;
  elevation = 0.0;
  if (mousePointer.z > 0.0) {
    azimuth += 2.0 * pi * mousePointer.x;
    elevation += 0.7 * pi * mousePointer.y;
  } else {
    azimuth += 2.0 * pi * sin(0.01 * pi * currentTime);
    elevation -= 0.1 + 0.3 * pi * sin(0.016 * pi * currentTime);
  }
  
  orientation = float2(elevation, azimuth);
  cosAngles = cos(orientation);
  sinAngles = sin(orientation);
  viewMatrix = float3x3(cosAngles.y, 0.0, -sinAngles.y, 0.0, 1.0, 0.0, sinAngles.y, 0.0, cosAngles.y) *
               float3x3(1.0, 0.0, 0.0, 0.0, cosAngles.x, -sinAngles.x, 0.0, sinAngles.x, cosAngles.x);
               
  zoomFactor = 3.0;
  ro = mul(float3(0.0, 0.0, -2.0 - 0.5 * sin(0.35 * currentTime)), viewMatrix);
  rd = mul(normalize(float3(uv, zoomFactor)), viewMatrix);
  lightDirection = normalize(float3(0.5, 2.0, 1.0));
  bubbleGrid = ((float3)(0.5));
  mouthShape = sin(6.0 * currentTime);
  
  eyePosition = ((float3)(0.13));
  viewPosition = ro;
  maxDistance = 10.0;
  
  finalColor = RenderScene(ro, rd);
  fragColor = float4(clamp(finalColor, 0.0, 1.0), 1.0);
}

float DistanceToSphere(float3 p, float r)
{
  return length(p) - r;
}

float DistanceToCylinder(float3 p, float r, float h)
{
  return max(length(p.xy) - r, abs(p.z) - h);
}

float DistanceToTube(float3 p, float r, float w, float h)
{
  return max(abs(length(p.xy) - r) - w, abs(p.z) - h);
}

float SmoothUnion(float a, float b, float r)
{
  float h = clamp(0.5 + 0.5 * (b - a) / r, 0.0, 1.0);
  return lerp(b, a, h) - r * h * (1.0 - h);
}

float SmoothBump(float lo, float hi, float w, float x)
{
  return (1.0 - smoothstep(hi - w, hi + w, x)) * smoothstep(lo - w, lo + w, x);
}

float3 RotateToDirectionLimited(float3 v1, float3 v2, float3 p, float aMax)
{
  float3 n = normalize(cross(v1, v2));
  float c = max(dot(v1, v2), cos(aMax));
  return c * p + sqrt(1.0 - c * c) * cross(n, p) + (1.0 - c) * dot(n, p) * n;
}

float2 Rotate2D(float2 q, float a)
{
  float2 cs = sin(a + float2(0.5 * pi, 0.0));
  return float2(dot(q, float2(cs.x, -cs.y)), dot(q.yx, cs));
}

float MinimumComponent(float3 p)
{
  return min(p.x, min(p.y, p.z));
}

static const float hashMultiplier = 43758.54;

float3 Hash3D(float p)
{
  return frac(sin(p + float3(37.0, 39.0, 41.0)) * hashMultiplier);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
