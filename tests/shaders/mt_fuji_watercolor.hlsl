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

// Mt. Fuji Watercolor
// by noztol

float hash(float2 p) {
    p = frac(p * float2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return frac(p.x * p.y);
}

float noise(float2 p) {
    float2 i = floor(p);
    float2 f = frac(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash(i);
    float b = hash(i + float2(1.0, 0.0));
    float c = hash(i + float2(0.0, 1.0));
    float d = hash(i + float2(1.0, 1.0));
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

float fbm(float2 p) {
    float value = 0.0;
    float amp = 0.5;
    for (int i = 0; i < 5; i++) {
        value += amp * noise(p);
        p *= 2.0;
        amp *= 0.5;
    }
    return value;
}

// Organic Calligraphy Brush Stroke (Guarantees A -> B connection)
float brushStroke(float2 p, float2 a, float2 b, float rStart, float rEnd, float droop) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    

    float2 pBended = p;
    pBended.y += sin(h * 3.14159) * droop;
    
    // Recalculate distance from the newly bended space
    pa = pBended - a;
    float d = length(pa - ba * h);
    float taper = smoothstep(1.0, 0.0, h);
    float r = lerp(rEnd, rStart, taper);
    
    return d - r - (fbm(p * 60.0) * 0.005);
}


void mainImage(out float4 fragColor, in float2 fragCoord) {
    float2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    
    // animation times
    float t = iTime * 0.8; 
    float t_sun     = clamp((t - 0.5) / 1.5, 0.0, 1.0); 
    float t_mnt     = clamp((t - 2.0) / 2.0, 0.0, 1.0); 
    float t_branch  = clamp((t - 4.0) / 2.0, 0.0, 1.0); 
    float t_flowers = clamp((t - 6.0) / 2.5, 0.0, 1.0); 
    float swayStrength = smoothstep(9.0, 11.0, t) * 0.02;         
    
    // 1. PAPER BACKGROUND
    float3 col = float3(0.97, 0.95, 0.92);
    col -= fbm(uv * 15.0) * 0.04; 

    float wetEdgeJitter = fbm(uv * 40.0) * 0.08;

    // 2. THE RED SUN 
    float2 sunUV = uv - float2(0.6, 0.3); 
    float sunRadius = 0.15;
    float sunDist = length(sunUV) - sunRadius + fbm(uv * 20.0) * 0.03;
    float sunShape = smoothstep(0.01, -0.01, sunDist);
    float sunReveal = smoothstep(0.02, -0.02, (sunUV.x + sunUV.y) - (t_sun * 0.6 - 0.3) + wetEdgeJitter);
    
    sunShape *= (fbm(uv * 50.0) * 0.5 + 0.5) * sunReveal;
    col = lerp(col, float3(0.8, 0.15, 0.15), clamp(sunShape * 0.9, 0.0, 1.0));

    // 3. MOUNTAIN
    float2 mntUV = uv;
    mntUV.x += fbm(uv * 10.0) * 0.05;
    mntUV.y += fbm(uv * 15.0) * 0.03;
    
    float fujiX = mntUV.x + 0.05; 
    float mountainShape = -0.1 - pow(abs(fujiX), 0.8) * 0.6; 
    float mountainMask = smoothstep(0.0, 0.02, mountainShape - mntUV.y);
    
    float outlineDist = abs(mountainShape - mntUV.y);
    float mntOutline = smoothstep(0.01, 0.002, outlineDist) * smoothstep(0.0, 0.05, mntUV.y + 0.15);
    float ridgeTexture = fbm(float2(mntUV.x * 25.0, mntUV.y * 5.0)); 
    float lightSide = smoothstep(-0.15, 0.25, fujiX + ridgeTexture * 0.1);
    
    float3 deepBlackInk = float3(0.04, 0.04, 0.05);
    float3 grayWaterWash = float3(0.6, 0.6, 0.65);
    float3 baseMntColor = lerp(deepBlackInk, grayWaterWash, lightSide);
    baseMntColor *= (0.6 + 0.4 * fbm(uv * 40.0)); 

    float snowLine = -0.22 + abs(fujiX) * 0.5 + (fbm(uv * 30.0) * 0.04);
    float snowCap = smoothstep(snowLine - 0.01, snowLine + 0.01, mntUV.y);
    float3 mountainColor = lerp(baseMntColor, float3(0.98, 0.99, 1.0), snowCap);
    mountainColor = lerp(mountainColor, ((float3)(0.02)), mntOutline); 

    float mntReveal = smoothstep(0.05, -0.05, mntUV.x - (t_mnt * 2.0 - 1.0) + wetEdgeJitter);
    col = lerp(col, mountainColor, mountainMask * mntReveal);

    // 4. BRANCHES
    float2 branchUV = uv;
    branchUV.x += sin(iTime * 1.5 + uv.y * 3.0) * swayStrength;
    branchUV.y += cos(iTime * 1.2 + uv.x * 2.0) * swayStrength;

    float2 n0 = float2(-1.0, -0.2);  // Root
    float2 n1 = float2(-0.6,  0.0);  // Joint 1
    float2 n2 = float2(-0.2,  0.1);  // Joint 2
    

    float2 t1 = float2( 0.2,  0.15); // Tip 1 (Main sweeping end)
    float2 t2 = float2(-0.2, -0.15); // Tip 2 (Lower drop)
    float2 t3 = float2(-0.0,  0.3);  // Tip 3 (Upper shoot)
    float2 t4 = float2(-0.6, -0.25); // Tip 4 (Inner drop)
    float2 t5 = float2(-0.1, -0.0);  // Tip 5 (Forward drop)

    float bd = 10.0;
    
    // Main Trunk (droops downward slightly using the 0.03 bend parameter)
    bd = min(bd, brushStroke(branchUV, n0, n1, 0.020, 0.015, 0.03));
    bd = min(bd, brushStroke(branchUV, n1, n2, 0.015, 0.010, 0.03));
    bd = min(bd, brushStroke(branchUV, n2, t1, 0.010, 0.003, 0.02)); // Connects perfectly to t1
    
    // Lower Sub Branch 1
    float2 s1_mid = float2(-0.4, -0.05);
    bd = min(bd, brushStroke(branchUV, n1, s1_mid, 0.00, 0.006, -0.02)); 
    bd = min(bd, brushStroke(branchUV, s1_mid, t2, 0.006, 0.002, 0.01)); // Connects perfectly to t2
    
    // Upper Sub Branch 2
    bd = min(bd, brushStroke(branchUV, n2, t3, 0.01, 0.009, -0.02));    // Connects perfectly to t3
    
    // Inner Sub Branch 3
    float2 s3_start = lerp(n0, n1, 0.2);
    bd = min(bd, brushStroke(branchUV, s3_start, t4, 0.008, 0.002, 0.01)); // Connects perfectly to t4
    
    // Forward Sub Branch 4
    float2 s4_start = lerp(n1, n2, 0.9);
    bd = min(bd, brushStroke(branchUV, s4_start, t5, 0.006, 0.002, 0.01)); // Connects perfectly to t5

    float branchMask = smoothstep(0.004, -0.002, bd);
    float branchReveal = smoothstep(0.05, -0.05, branchUV.x - (t_branch * 1.5 - 1.0) + wetEdgeJitter);
    col = lerp(col, float3(0.04, 0.03, 0.03), branchMask * branchReveal);

    // 5. FLOWERS 
    float flowerMask = 0.0;
    float2 targets[5];
    targets[0] = t1; targets[1] = t2; targets[2] = t3; 
    targets[3] = t4; targets[4] = t5; 
    
    for(int i = 0; i < 5; i++) {
        float individualTime = clamp((t_flowers - float(i)*0.2) * 5.0, 0.0, 1.0);
        
        if (individualTime > 0.0) {
            float2 center = targets[i];
            
            // Draw a cluster of 3 textured splotches exactly on the tip
            for(float j = 0.0; j < 3.0; j++) {
                float2 offset = float2(noise(float2(float(i), j*10.0)) - 0.5, noise(float2(float(i), j*20.0)) - 0.5) * 0.03;
                float2 fpos = center + offset;
                
                float currentRadius = lerp(0.0, 0.012, smoothstep(0.0, 1.0, individualTime));
                float fdist = length(branchUV - fpos) - currentRadius - fbm(branchUV * 70.0 + float(i)*j) * 0.02;
                
                flowerMask = max(flowerMask, smoothstep(0.008, -0.005, fdist));
            }
        }
    }
    
    col = lerp(col, float3(0.85, 0.1, 0.2), flowerMask);

    fragColor = float4(col, 1.0);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
