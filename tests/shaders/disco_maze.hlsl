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

// Disco Maze
// By Noztol

#define PI 3.1415926535

//  Disco Walls
float hash1( float n ) { return frac(sin(n)*437518.56453)+0.1; }

float3 hsb2rgb( in float3 c ){
    float3 rgb = clamp(abs(glsl_mod(c.x*6.0+float3(0.0,4.0,2.0),6.0)-3.0)-1.0, 0.0, 1.0 );
    rgb = rgb*rgb*(3.0-2.0*rgb);
    return c.z * lerp(((float3)(1.0)), rgb, c.y);
}

float fbm (in float2 st, float t) {
    float value = (sin(t*5.0*hash1(st.x+hash1(st.y)))+4.0)*0.25;
    return max(0.75,pow(value,3.));
}

float3 squaresColors(float2 p, float t)
{
    float i = p.x;
    float j = p.y;
    return hsb2rgb(float3((hash1(j+hash1(i))), 0.7, fbm(float2(i,j), t)));
}

float3 getDiscoSquares(float2 p, float border, float t)
{
    float2 ip = floor(p);
    float2 fp = frac(p);
    
    // Wrap horizontal colors slightly for a repeating pattern
    ip.x = glsl_mod(ip.x, 16.0); 
    
    float3 tileColor = squaresColors(ip, t);
    float glow = 1.0 - length(fp - 0.5) * 1.2;
    glow = clamp(glow, 0.4, 1.0);
    tileColor *= glow;
    
    float edgeDist = max(abs(fp.x - 0.5), abs(fp.y - 0.5));
    float borderMask = smoothstep(border - 0.03, border, edgeDist);
    return lerp(tileColor, ((float3)(0.01)), borderMask);
}

// Hive Floor & Ceiling
// 
float3 discoColors(float2 p, float t)
{
    p += ((float2)(t * 0.2));
    
    float3 orange = float3(1.0, 0.4, 0.1) * 2.0;
    float3 purple = float3(1.0, 0.2, 0.5) * 0.8;
    
    float l = pow(0.5 + 0.5 * cos(p.x * PI * 2.0 + cos(p.y) * 8.0) * sin(p.y * 2.0), 4.0) * 2.0;
    float3 c = pow(l * (lerp(orange, purple, 0.5 + 0.5 * cos(p.x * PI * 12.0 + sin(p.y * 10.0) * 3.0)) +
                      lerp(orange, purple, 0.5 + 0.5 * cos(p.x * PI * 6.0 + sin(p.y * 3.0) * 3.0))), ((float3)(1.2))) * 0.7;
    
    c += float3(1.0, 0.8, 0.4) * pow(0.5 + 0.5 * cos(p.x * PI * 6.0) * sin(p.y * 12.0), 20.0) * 2.0;
    c += float3(0.1, 0.5 + 0.5 * cos(p * PI * 6.0)) * float3(0.05, 0.1, 0.4).bgr * 0.7;
    return c;
}

float4 hexCoords(float2 uv) 
{
    float2 r = float2(1.0, 1.7320508);
    float2 h = r * 0.5;
    
    float2 a = glsl_mod(uv, r) - h;
    float2 b = glsl_mod(uv - h, r) - h;
    
    float2 gv = dot(a, a) < dot(b, b) ? a : b;
    float2 id = uv - gv;
    return float4(gv.x, gv.y, id.x, id.y);
}

float3 getDiscoHexes(float2 p, float border, float t)
{
    float4 hc = hexCoords(p);
    float2 gv = hc.xy;
    float2 id = hc.zw * 0.125; 
    
    float d = max(abs(gv.x), dot(abs(gv), float2(0.5, 0.8660254)));
    
    // Wireframe grid
    float m = 1.0 - smoothstep(border - 0.03, border, d);
    // Center glowing dot
    m += 1.0 - smoothstep(0.0, 0.22, length(gv));
    
    float3 hexCol = m * discoColors(id, t);
    float3 bgCol = float3(0.03, 0.01, 0.05); // Deep purple gap background
    
    return bgCol + hexCol;
}

float map(float2 p) {
    float2 id = floor(p);
    float L = 6.0; 
    
    float corridorX = abs(id.y - floor(id.x / L) * L);
    float corridorY = abs(id.x - floor(id.y / L + 1.0) * L);
    
    if (corridorX <= 1.0 || corridorY <= 1.0) return 0.0;
    
    float n = frac(sin(dot(id, float2(12.9898, 78.233))) * 43758.5453);
    return n > 0.3 ? 1.0 : 0.0; 
}



void mainImage(out float4 fragColor, in float2 fragCoord)
{
    float2 screenUV = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
    float globTime = iTime + 1.0;
    
    // Camera Path Navigation 
    float L = 6.0; 
    float t = iTime * 2.5; 
    
    float seg = floor(t / L);
    float local_t = frac(t / L) * L;
    
    float2 ro;
    if (glsl_mod(seg, 2.0) < 0.5) {
        ro.x = floor(seg / 2.0) * L + local_t;
        ro.y = floor(seg / 2.0) * L;
    } else {
        ro.x = floor(seg / 2.0 + 0.5) * L;
        ro.y = floor(seg / 2.0) * L + local_t;
    }
    
    float turnProgress = smoothstep(L - 2.0, L, local_t);
    float currentAngle = glsl_mod(seg, 2.0) < 0.5 ? 0.0 : 1.5707963;
    float nextAngle = glsl_mod(seg, 2.0) < 0.5 ? 1.5707963 : 0.0;
    
    float camAngle = lerp(currentAngle, nextAngle, turnProgress);
    
    // Add the banking roll to the camera as it turns
    float bankRoll = sin(globTime * 0.6) * 0.15;
    float2x2 rot = float2x2(cos(bankRoll), -sin(bankRoll), sin(bankRoll), cos(bankRoll));
    screenUV = mul(rot, screenUV);
    
    float2 camDir = float2(cos(camAngle), sin(camAngle));
    float2 camPlane = float2(-camDir.y, camDir.x) * 0.8; 
    float2 rd = camDir + camPlane * screenUV.x;
    
    // DDA Setup
    float2 mapPos = floor(ro);
    float2 deltaDist = abs(1.0 / rd);
    float2 stepDir = sign(rd);
    float2 sideDist = (stepDir * (mapPos - ro) + (stepDir * 0.5) + 0.5) * deltaDist;
    
    float side = 0.0;
    float cellId = 0.0;
    
    for(int i = 0; i < 64; i++) {
        if(sideDist.x < sideDist.y) {
            sideDist.x += deltaDist.x;
            mapPos.x += stepDir.x;
            side = 0.0;
        } else {
            sideDist.y += deltaDist.y;
            mapPos.y += stepDir.y;
            side = 1.0;
        }
        cellId = map(mapPos);
        if(cellId > 0.0) break;
    }
    
    float3 finalCol = ((float3)(0.0));
    float3 voidFog = float3(0.01, 0.002, 0.03); 
    
    float perpWallDist = (side == 0.0) ? (mapPos.x - ro.x + (1.0 - stepDir.x) * 0.5) / rd.x
                                       : (mapPos.y - ro.y + (1.0 - stepDir.y) * 0.5) / rd.y;
    
    float projectedHeight = 1.0 / perpWallDist;
    
    if (abs(screenUV.y) < projectedHeight && cellId > 0.0) {
        
        // disco walls
        float wallX = (side == 0.0) ? ro.y + perpWallDist * rd.y : ro.x + perpWallDist * rd.x;
        float2 wallUV = float2(wallX, screenUV.y / projectedHeight);
        
        // Scale to fit 4x4 squares per block panel
        wallUV *= 4.0; 
        
        finalCol = getDiscoSquares(wallUV, 0.46, globTime);
        
        // Darken side walls for 3D depth perception
        if(side == 1.0) finalCol *= 0.6; 
        finalCol = lerp(finalCol, voidFog, smoothstep(2.0, 16.0, perpWallDist));
        
    } else {
        
        // draw hive
        float floorDist = 1.0 / abs(screenUV.y); 
        float2 floorPos = ro + rd * floorDist;
                floorPos *= 2.5; 
        
        finalCol = getDiscoHexes(floorPos, 0.46, globTime);
        finalCol = lerp(finalCol, voidFog, smoothstep(2.0, 16.0, floorDist));
    }
    
    // Brighten slightly
    finalCol *= 1.2;

    fragColor = float4(finalCol, 1.0);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
