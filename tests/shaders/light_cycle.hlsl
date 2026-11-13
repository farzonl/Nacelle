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

// Light Cycle
// By Noztol

#define MAX_STEPS 110
#define MAX_DIST 80.0
#define SURF_DIST 0.001

static const float3 COL_BG     = float3(0.004, 0.007, 0.014);
static const float3 COL_CYAN   = float3(0.0, 0.85, 1.0);
static const float3 COL_ORANGE = float3(1.0, 0.28, 0.0);
static const float3 COL_BLUE   = float3(0.04, 0.18, 0.75);

float2x2 rot(float a) {
    float s = sin(a), c = cos(a);
    return float2x2(c, -s, s, c);
}

float sdBox(float3 p, float3 b) {
    float3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float sdBox2D(float2 p, float2 b) {
    float2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

float2 getPath(float z) {
    return float2(
        sin(z * 0.05) * 2.4 + cos(z * 0.025) * 1.0,
        cos(z * 0.04) * 2.0 + sin(z * 0.02) * 0.8
    );
}

float3 toTunnelSpace(float3 p) {
    p.xy -= getPath(p.z);
    p.xy = mul(rot(p.z * 0.02 + iTime * 0.05), p.xy);
    return p;
}

float2 foldSquare(float2 p) {
    p = abs(p);
    if (p.y > p.x) p.xy = p.yx;
    return p;
}


float3 map(float3 pWorld) {
    float3 p = toTunnelSpace(pWorld);
    float3 q = float3(foldSquare(p.xy), p.z);
    
    float zPeriod = 4.0;
    float3 qCell = q;
    qCell.z = glsl_mod(q.z + zPeriod * 0.5, zPeriod) - zPeriod * 0.5;
    
    // 1. Watertight Closed Square Tunnel
    // Solid square inner frame at q.x = 4.0
    float dWalls = 4.0 - q.x;
    
    // Carve 0.5-unit deep recessed architectural bays (from q.x = 4.0 to 4.5)
    // Slight 0.012 bevel on cutouts prevents single-pixel normal chatter on edges
    float dRecessCenter = sdBox(qCell - float3(4.0, 0.0, 0.0), float3(0.5, 1.488, 1.488)) - 0.012;
    float dRecessSide   = sdBox(qCell - float3(4.0, 2.75, 0.0), float3(0.5, 0.738, 1.488)) - 0.012;
    
    float dCarve = min(dRecessCenter, dRecessSide);
    float dArch = max(dWalls, -dCarve);
    
    // 2. Flush 360-Degree Cyan Light Ribs
    float dCyanRail1 = sdBox2D(q.xy - float2(4.0, 1.5), float2(0.05, 0.05));
    float dCyanRail2 = sdBox2D(q.xy - float2(4.0, 2.0), float2(0.05, 0.05));
    float dCyanRail3 = sdBox2D(q.xy - float2(4.0, 3.5), float2(0.05, 0.05));
    float dCyanRails = min(min(dCyanRail1, dCyanRail2), dCyanRail3);
    
    float dCyanRing1 = sdBox(float3(qCell.x - 4.0, qCell.y - 2.0, abs(qCell.z) - 1.5), float3(0.05, 2.0, 0.05));
    float dCyanRing2 = sdBox(float3(qCell.x - 4.0, qCell.y - 2.0, abs(qCell.z) - 2.0), float3(0.05, 2.0, 0.05));
    float dCyanRings = min(dCyanRing1, dCyanRing2);
    
    float dCyan = min(dCyanRails, dCyanRings);
    
    // 3. Corner Orange Light Streams
    float dOrange = sdBox2D(q.xy - float2(3.92, 3.92), float2(0.11, 0.11));
    
    // Combine geometry into a single smooth surface so rays never experience
    // jagged material-ID silhouette jumps
    float dScene = min(dArch, min(dCyan, dOrange));
    
    return float3(dScene, dCyan, dOrange);
}

float3 getNormal(float3 p) {
    const float2 k = float2(1.0, -1.0);
    const float h = 0.002;
    return normalize(
        k.xyy * map(p + k.xyy * h).x +
        k.yyx * map(p + k.yyx * h).x +
        k.yxy * map(p + k.yxy * h).x +
        k.xxx * map(p + k.xxx * h).x
    );
}

float boxFilteredLine(float coord, float scale, float lineWidth, float footprint) {
    float w = max(footprint * scale, 0.0001);
    float p = coord * scale;
    float a = p + 0.5 * w;
    float b = p - 0.5 * w;
    float coverage = (floor(a) + min(frac(a) / lineWidth, 1.0) -
                      floor(b) - min(frac(b) / lineWidth, 1.0)) / (w / lineWidth);
    return clamp(coverage, 0.0, 1.0);
}

float getAntiAliasedGrid(float3 pWorld, float3 nWorld, float3 rd, float t, float pixelSpread) {
    float3 p = toTunnelSpace(pWorld);
    float3 q = float3(foldSquare(p.xy), p.z);
    
    float3 nLocal = nWorld;
    nLocal.xy = mul(rot(p.z * 0.02 + iTime * 0.05), nLocal.xy);
    float3 nAbs = abs(nLocal);
    if (abs(p.y) > abs(p.x)) nAbs.xy = nAbs.yx;
    
    float3 axisWeight = 1.0 - smoothstep(0.35, 0.65, nAbs);
    
    float cosTheta = max(abs(dot(nWorld, rd)), 0.15);
    float worldFootprint = (t * pixelSpread * 1.35) / cosTheta;
    
    float gxMinor = boxFilteredLine(q.x + 0.25, 2.0, 0.035, worldFootprint) * axisWeight.x;
    float gyMinor = boxFilteredLine(q.y + 0.25, 2.0, 0.035, worldFootprint) * axisWeight.y;
    float gzMinor = boxFilteredLine(q.z + 0.25, 2.0, 0.035, worldFootprint) * axisWeight.z;
    float lineMinor = max(gxMinor, max(gyMinor, gzMinor));
    
    float gyMajor = boxFilteredLine(q.y + 0.5, 1.0, 0.032, worldFootprint) * axisWeight.y;
    float gzMajor = boxFilteredLine(q.z + 0.5, 1.0, 0.032, worldFootprint) * axisWeight.z;
    float lineMajor = max(gyMajor, gzMajor);
    
    float fadeMajor = exp(-0.0007 * t * t);
    float fadeMinor = exp(-0.0016 * t * t);
    
    float c = lineMajor * 0.85 * fadeMajor + lineMinor * 0.45 * fadeMinor;
    
    float pulse = sin(p.z * 0.35 - iTime * 3.0) * 0.5 + 0.5;
    c *= (0.45 + 0.55 * pow(pulse, 3.0));
    
    return clamp(c, 0.0, 1.0);
}

void mainImage(out float4 fragColor, in float2 fragCoord) {
    float minRes = min(iResolution.x, iResolution.y);
    float2 uv = (fragCoord - 0.5 * iResolution.xy) / minRes;
    float pixelSpread = 1.0 / minRes;
    
    uv *= 1.0 + dot(uv, uv) * 0.05;
    
    float speed = 8.0;
    float camZ = iTime * speed;
    float3 ro = float3(getPath(camZ), camZ);
    
    float targetZ = camZ + 5.0;
    float3 ta = float3(getPath(targetZ), targetZ);
    
    float3 fwd = normalize(ta - ro);
    float bankAngle = -fwd.x * 0.45 + sin(iTime * 0.3) * 0.1;
    float3 up = float3(sin(bankAngle), cos(bankAngle), 0.0);
    float3 right = normalize(cross(fwd, up));
    up = cross(right, fwd);
    
    float3 rd = normalize(fwd * 0.9 + right * uv.x + up * uv.y);
    
    float t = 0.0;
    float3 glowAccum = ((float3)(0.0));
    
    float minCyanCone   = 1e5;
    float minOrangeCone = 1e5;
    float hitZ = 0.0;
    
    for (int i = 0; i < MAX_STEPS; i++) {
        float3 p = ro + rd * t;
        float3 res = map(p);
        float d       = res.x;
        float dCyan   = res.y;
        float dOrange = res.z;
        
        float coneRadius = max(t * pixelSpread * 1.25, 0.003);
        
        minCyanCone   = min(minCyanCone,   dCyan   / coneRadius);
        minOrangeCone = min(minOrangeCone, dOrange / coneRadius);
        
        float stepDist = max(d * 0.65, 0.002);
        
        glowAccum += COL_CYAN * (stepDist * 0.022 / (0.015 + dCyan * dCyan * 3.5));
        
        float packet = pow(sin(p.z * 0.25 + iTime * 10.0) * 0.5 + 0.5, 2.5);
        float3 orangeEmit = lerp(COL_ORANGE * 0.45, COL_ORANGE * 2.2, packet);
        glowAccum += orangeEmit * (stepDist * 0.035 / (0.012 + dOrange * dOrange * 3.5));
        
        glowAccum += COL_BLUE * (stepDist * 0.0035 / (0.2 + d * d));
        
        if (d < SURF_DIST || t > MAX_DIST) {
            hitZ = p.z;
            break;
        }
        
        t += stepDist;
    }
    
    float3 col = COL_BG;
    
    if (t < MAX_DIST) {
        float3 p = ro + rd * t;
        float3 n = getNormal(p);
        float3 v = -rd;
        
        float fresnel = pow(clamp(1.0 - max(dot(n, v), 0.0), 0.0, 1.0), 3.0);
        
        float grid = getAntiAliasedGrid(p, n, rd, t, pixelSpread);
        float3 baseColor = float3(0.008, 0.012, 0.022);
        float3 rimColor  = lerp(COL_BLUE, COL_CYAN, 0.55) * fresnel * 0.9;
        float3 gridColor = COL_CYAN * grid * 1.5;
        col = baseColor + rimColor + gridColor;
        
        float cyanCoverage = 1.0 - smoothstep(-0.5, 1.2, minCyanCone);
        float3 cyanSurf = COL_CYAN * 2.4 + ((float3)(0.75));
        col = lerp(col, cyanSurf, cyanCoverage);
        
        float orangeCoverage = 1.0 - smoothstep(-0.5, 1.2, minOrangeCone);
        float packet = pow(sin(hitZ * 0.25 + iTime * 10.0) * 0.5 + 0.5, 2.5);
        float3 orangeSurf = lerp(COL_ORANGE * 1.0, COL_ORANGE * 2.8 + float3(0.95, 0.65, 0.25), packet);
        col = lerp(col, orangeSurf, orangeCoverage);
        
        float fog = 1.0 - exp(-0.00075 * t * t);
        col = lerp(col, COL_BG, fog);
    }
    
    col += glowAccum * 0.8;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);
    
    float2 q = fragCoord / iResolution.xy;
    float vig = pow(16.0 * q.x * q.y * (1.0 - q.x) * (1.0 - q.y), 0.15);
    col *= lerp(0.65, 1.0, vig);
    
    col = pow(max(col, 0.0), ((float3)(1.0 / 2.2)));
    
    float dither = frac(sin(dot(fragCoord, float2(12.9898, 78.233))) * 43758.5453);
    col += (dither - 0.5) / 255.0;
    
    fragColor = float4(col, 1.0);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
