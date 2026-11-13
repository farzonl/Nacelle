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

// Aurora Paint
// By Noztol

static const float pi = 3.14159;
static const float pi2 = pi * 2.;

float opU( float d1, float d2 ){ return min(d1,d2); }
float opS( float d2, float d1 ){ return max(-d1,d2); }
float opI( float d1, float d2) { return max(d1,d2); }

float magicBox(float3 p) {
    const int MAGIC_BOX_ITERS = 13;
    const float MAGIC_BOX_MAGIC = 0.55;
    p = 1.0 - abs(1.0 - glsl_mod(p, 2.0));
    float lastLength = length(p);
    float tot = 0.0;
    for (int i=0; i < MAGIC_BOX_ITERS; i++) {
      p = abs(p)/(lastLength*lastLength) - MAGIC_BOX_MAGIC;
      float newLength = length(p);
      tot += abs(newLength-lastLength);
      lastLength = newLength;
    }
    return tot;
}

float magicBox(float2 uv){
    const float3x3 M = float3x3(0.28862355854826727, 0.6997227302779844, 0.6535170557707412,
                        0.06997493955670424, 0.6653237235314099, -0.7432683571499161,
                        -0.9548821651308448, 0.26025457467376617, 0.14306504491456504);
    float3 p = mul(float3(uv, 0.0), 0.5*M);
    return magicBox(p);
}

float2x2 rot2D(float r) {
    float c = cos(r), s = sin(r);
    return float2x2(c, s, -s, c);
}
float nsin(float a){return .5+.5*sin(a);}
float ncos(float a){return .5+.5*cos(a);}
float3 saturate(float3 a){return clamp(a,0.,1.);}
float rand(float2 co){ return frac(sin(dot(co.xy ,float2(12.9898,78.233))) * 43758.5453); }
float rand(float n){ return frac(cos(n*89.42)*343.42); }
float dtoa(float d, float amount) { return clamp(1.0 / (clamp(d, 1.0/amount, 1.0)*amount), 0.,1.); }
float sdAxisAlignedRect(float2 uv, float2 tl, float2 br) {
    float2 d = max(tl-uv, uv-br);
    return length(max(((float2)(0.0)), d)) + min(0.0, max(d.x, d.y));
}
float sdCircle(float2 uv, float2 origin, float radius) { return length(uv - origin) - radius; }
float smoothstep4(float e1, float e2, float e3, float e4, float val) {
    return min(smoothstep(e1,e2,val), 1.-smoothstep(e3,e4,val));
}
float noise01(float2 p) { return clamp((frac(sin(dot(p, float2(12.9898, 78.233))) * 43758.5453)+.5)*.5, 0.,1.); }
float smoothf(float x) { return x*x*x*(x*(x*6. - 15.) + 10.); }


float sdTri(float2 p, float bottom, float top, float halfWidth) {
    float2 p0 = float2(0.0, top);
    float2 p1 = float2(halfWidth, bottom);
    float2 p2 = float2(-halfWidth, bottom);
    float2 e0 = p1 - p0; float2 e1 = p2 - p1; float2 e2 = p0 - p2;
    float2 v0 = p - p0;  float2 v1 = p - p1;  float2 v2 = p - p2;
    float2 pq0 = v0 - e0*clamp(dot(v0,e0)/dot(e0,e0), 0.0, 1.0);
    float2 pq1 = v1 - e1*clamp(dot(v1,e1)/dot(e1,e1), 0.0, 1.0);
    float2 pq2 = v2 - e2*clamp(dot(v2,e2)/dot(e2,e2), 0.0, 1.0);
    float s = sign( e0.x*e2.y - e0.y*e2.x );
    float2 d = min( min( float2( dot( pq0, pq0 ), s*(v0.x*e0.y-v0.y*e0.x) ),
                       float2( dot( pq1, pq1 ), s*(v1.x*e1.y-v1.y*e1.x) )),
                       float2( dot( pq2, pq2 ), s*(v2.x*e2.y-v2.y*e2.x) ));
    return -sqrt(d.x)*sign(d.y);
}

float sdTree(float2 p, float2 pos, float scale) {
    float2 q = (p - pos) / scale;
    // Slight wind sway
    q.x -= sin(iTime * 1.5 + pos.x * 10.0) * 0.02 * max(0.0, q.y);
    
    // Trunk
    float d = max(abs(q.x) - 0.03, max(-q.y, q.y - 0.2));
    
    // Jagged canopy layers
    d = min(d, sdTri(q, 0.1, 0.45, 0.25));
    d = min(d, sdTri(q, 0.25, 0.65, 0.2));
    d = min(d, sdTri(q, 0.45, 0.85, 0.15));
    d = min(d, sdTri(q, 0.65, 1.0, 0.1));
    return d * scale;
}

// Solid sloping landmasses framing the center
float getLand(float2 uv, float aspect) {
    // Left shore sloping down
    float leftShore = 0.45 * exp(-0.5 * pow(uv.x, 2.0));
    // Right shore sloping down
    float rightShore = 0.45 * exp(-0.5 * pow(uv.x - aspect, 2.0));
    
    float terrainHeight = max(leftShore, rightShore);
    
    // Add tiny bit of noise to the ground for texture
    terrainHeight += (noise01(uv * float2(20.0, 1.0)) * 0.015);
    
    // Distance field is just the Y coordinate minus the terrain height
    return uv.y - terrainHeight;
}

// Trees planted along the new slopes
float getTrees(float2 uv, float aspect) {
    float d = 1.0;
    // Left shore trees
    d = min(d, sdTree(uv, float2(aspect*0.02, 0.42), 0.18));
    d = min(d, sdTree(uv, float2(aspect*0.08, 0.38), 0.22));
    d = min(d, sdTree(uv, float2(aspect*0.14, 0.42), 0.14));
    d = min(d, sdTree(uv, float2(aspect*0.18, 0.41), 0.09));
    d = min(d, sdTree(uv, float2(aspect*0.22, 0.41), 0.08));
    d = min(d, sdTree(uv, float2(aspect*0.24, 0.41), 0.08));
    d = min(d, sdTree(uv, float2(aspect*0.29, 0.38), 0.08));
    
    // Right shore trees
    d = min(d, sdTree(uv, float2(aspect*0.70, 0.38), 0.08));
    d = min(d, sdTree(uv, float2(aspect*0.74, 0.40), 0.08));
    d = min(d, sdTree(uv, float2(aspect*0.78, 0.42), 0.1));
    d = min(d, sdTree(uv, float2(aspect*0.82, 0.41), 0.09));
    d = min(d, sdTree(uv, float2(aspect*0.86, 0.42), 0.14));
    d = min(d, sdTree(uv, float2(aspect*0.92, 0.38), 0.22));
    d = min(d, sdTree(uv, float2(aspect*0.98, 0.42), 0.18));
    return d;
}

float3 colorBrushStroke(float2 uvLine, float2 uvPaper, float2 lineSize, float sdGeometry, float3 inpColor, float4 brushColor)
{
    float posInLineY = (uvLine.y / lineSize.y);
    
    if(posInLineY > 0.) {
        float smoothWave = 0.5 + 0.5 * sin(iTime * 1.5);
        float autoX = lerp(0.15, 0.7, smoothWave);
        posInLineY = pow(posInLineY, (pow(autoX,2.) * 15.) + 1.5);
    }
    
    float strokeBoundary = dtoa(sdGeometry, 300.);
    float strokeTexture = 0.
        + noise01(uvLine * float2(min(iResolution.y,iResolution.x)*0.2, 1.))
        + noise01(uvLine * float2(79., 1.))
        + noise01(uvLine * float2(14., 1.));
    strokeTexture *= 0.333 * strokeBoundary;
    strokeTexture = max(0.008, strokeTexture);
    
    float strokeAlpha = pow(strokeTexture, max(0.,posInLineY)+0.09);
    const float strokeAlphaBoost = 1.09;
    
    if(posInLineY > 0.) strokeAlpha = strokeAlphaBoost * max(0., strokeAlpha - pow(posInLineY,0.5));
    else strokeAlpha *= strokeAlphaBoost;

    strokeAlpha = smoothf(strokeAlpha);
    float paperBleedAmt = 60. + (rand(uvPaper.y) * 30.) + (rand(uvPaper.x) * 30.);
    float alpha = strokeAlpha * brushColor.a * dtoa(sdGeometry, paperBleedAmt);
    alpha = clamp(alpha, 0.,1.);
    return lerp(inpColor, brushColor.rgb, alpha);
}

float3 colorBrushStrokeLine(float2 uv, float3 inpColor, float4 brushColor, float2 p1_, float2 p2_, float lineWidth)
{
    float lineAngle = pi-atan2(p1_.x - p2_.x, p1_.y - p2_.y);
    float2x2 rotMat = rot2D(lineAngle);
    float lineLength = distance(p2_, p1_);
    float2 tl = mul(rotMat, p1_);
    float2 br = tl + float2(0,lineLength);
    float2 uvLine = mul(rotMat, uv);

    lineWidth *= lerp(1., .9, smoothstep(tl.y,br.y,uvLine.y));
    uvLine.x += (noise01(uvLine * 1.)-0.5) * 0.02;
    uvLine.x += cos(uvLine.y * 3.) * 0.009;
    uvLine.x += (noise01(uvLine * 5.)-0.5) * 0.005;

    float d = sdAxisAlignedRect(uvLine, tl, br) - lineWidth / 2.;
    uvLine = tl - uvLine;
    float2 lineSize = float2(lineWidth, lineLength);
    return colorBrushStroke(float2(uvLine.x, -uvLine.y), uv, lineSize, d, inpColor, brushColor);
}

float2 getuv_centerX(float2 fragCoord, float2 newTL, float2 newSize)
{
    float2 ret = float2(fragCoord.x / iResolution.x, fragCoord.y / iResolution.y);
    ret *= newSize;
    float aspect = iResolution.x / iResolution.y;
    ret.x *= aspect;
    float newWidth = newSize.x * aspect;
    return ret + float2(newTL.x - (newWidth - newSize.x) / 2.0, newTL.y);
}

void mainImage(out float4 fragColor, in float2 fragCoord)
{
    float2 uv = getuv_centerX(fragCoord, float2(-1,-1), float2(2,2)); // Used for Auroras (-1 to 1)
    float2 landscapeUV = fragCoord / iResolution.y;               // Used for Landscape SDFs (0 to Aspect)
    float aspect = iResolution.x / iResolution.y;
    
    float t = iTime * 0.4;
    
    // Define the exact water surface level
    float waterLevelLandscape = 0.34; 
    float waterLevelUV = waterLevelLandscape * 2.0 - 1.0; 
    bool isWater = landscapeUV.y < waterLevelLandscape;

    float3 skyColor = lerp(float3(0.08, 0.15, 0.35), float3(0.2, 0.4, 0.6), smoothstep(-0.2, 1.0, isWater ? (waterLevelUV + (waterLevelUV - uv.y)) : uv.y));
    float3 col = skyColor;
    if (isWater) {
        float3 waterBase = float3(0.06, 0.18, 0.28);
        col = lerp(col, waterBase, 0.85); // Darken for water depth
    }

    if (!isWater) {
        // 1. Paint Blotch Stars
        float blotchAmt = smoothstep(12.0, 25.0, magicBox((uv+12.0)*8.0)); // UV scaled by 8.0 for smaller paint flicks
        blotchAmt = pow(blotchAmt, 1.5);
        col += float3(0.6, 0.85, 0.95) * blotchAmt * 0.6 * (0.5 + 0.5 * sin(iTime*1.5 + uv.x*5.0));

        // 2.Aurora
        float2 aUV1 = uv; aUV1.y += sin(aUV1.x * 2.0 + 1.0 + t) * 0.35 + sin(aUV1.x * 4.0 - t*1.5) * 0.1;
        float2 aUV2 = uv; aUV2.y += cos(aUV2.x * 2.5 - 0.5 - t*0.8) * 0.4 + sin(aUV2.x * 1.5 + t*1.2) * 0.2;
        float2 aUV3 = uv; aUV3.y += sin(aUV3.x * 1.8 + 2.0 + t*0.5) * 0.35;

        col = colorBrushStrokeLine(aUV3, col, float4(0.1, 0.8, 0.5, 0.6), float2(-2.5, 0.6), float2(2.5, 0.6), 0.4);
        col = colorBrushStrokeLine(aUV1, col, float4(0.3, 0.9, 0.7, 0.8), float2(-2.5, 0.4), float2(2.5, 0.4), 0.3);
        col = colorBrushStrokeLine(aUV2, col, float4(0.0, 0.6, 0.5, 0.9), float2(2.5, 0.2), float2(-2.5, 0.2), 0.25);
        col = colorBrushStrokeLine(aUV1, col, float4(0.6, 1.0, 0.8, 0.9), float2(-2.5, 0.3), float2(2.5, 0.3), 0.15);

    } else {
        // 3. Aurora reflections
        float depthUV = waterLevelUV - uv.y;
        float2 refUV = uv;
        refUV.y = waterLevelUV + depthUV;
        refUV.x += sin(uv.y * 30.0 + iTime * 2.0) * depthUV * 0.15;
        refUV.y += cos(uv.x * 15.0 + iTime * 1.5) * depthUV * 0.025;

        float2 rUV1 = refUV; rUV1.y += sin(rUV1.x * 2.0 + 1.0 + t) * 0.35 + sin(rUV1.x * 4.0 - t*1.5) * 0.1;
        float2 rUV2 = refUV; rUV2.y += cos(rUV2.x * 2.5 - 0.5 - t*0.8) * 0.4 + sin(rUV2.x * 1.5 + t*1.2) * 0.2;
        float2 rUV3 = refUV; rUV3.y += sin(rUV3.x * 1.8 + 2.0 + t*0.5) * 0.35;

        col = colorBrushStrokeLine(rUV3, col, float4(0.1, 0.7, 0.5, 0.3), float2(-2.5, 0.6), float2(2.5, 0.6), 0.4);
        col = colorBrushStrokeLine(rUV1, col, float4(0.2, 0.8, 0.6, 0.4), float2(-2.5, 0.4), float2(2.5, 0.4), 0.3);
        col = colorBrushStrokeLine(rUV2, col, float4(0.0, 0.5, 0.4, 0.45), float2(2.5, 0.2), float2(-2.5, 0.2), 0.25);
        col = colorBrushStrokeLine(rUV1, col, float4(0.4, 0.8, 0.7, 0.35), float2(-2.5, 0.3), float2(2.5, 0.3), 0.15);
    }

    // 4. Island Silhouettes
    if (!isWater) {
        float sceneDist = min(getLand(landscapeUV, aspect), getTrees(landscapeUV, aspect));
        col = lerp(col, float3(0.02, 0.04, 0.08), smoothstep(0.005, 0.0, sceneDist));
    } else {
        // Fold landscape UV for exact island reflections
        float depthLandscape = waterLevelLandscape - landscapeUV.y;
        float2 refLandUV = landscapeUV;
        refLandUV.y = waterLevelLandscape + depthLandscape;
        refLandUV.x += sin(landscapeUV.y * 50.0 + iTime * 2.0) * depthLandscape * 0.1;

        float refSceneDist = min(getLand(refLandUV, aspect), getTrees(refLandUV, aspect));
        col = lerp(col, lerp(col, float3(0.02, 0.04, 0.08), 0.8), smoothstep(0.005, 0.0, refSceneDist));
        
        // Add localized horizontal brush strokes for surface water texture
        float2 surfaceUV = uv;
        surfaceUV.x += iTime * 0.05; 
        col = colorBrushStrokeLine(surfaceUV, col, float4(0.3, 0.8, 0.6, 0.15), float2(2.5, waterLevelUV - 0.1), float2(-2.5, waterLevelUV - 0.1), 0.08);
        col = colorBrushStrokeLine(surfaceUV, col, float4(0.1, 0.6, 0.5, 0.2), float2(-2.5, waterLevelUV - 0.3), float2(2.5, waterLevelUV - 0.3), 0.12);
    }

    col.rgb += (rand(uv)-.5)*.06;
    col.rgb = saturate(col.rgb);

    float2 uvScreen = (fragCoord / iResolution.xy * 2.)-1.;
    float vignetteAmt = 1.0 - dot(uvScreen*0.5, uvScreen* 0.62);
    col *= vignetteAmt;
    
    fragColor = float4(col, 1.);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
