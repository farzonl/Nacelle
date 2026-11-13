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

// Rainbow Travel
// By Noztol
// Based on https://fragcoord.xyz/s/fx196wc1 
// but using XorDev's color scheme 


// Calculates the base winding 3D path of the tunnel
float3 getPathPosition(float z) {
    return float3(12.0 * cos(z * float2(0.1, 0.12)), z);
}

void mainImage(out float4 fragColor, in float2 fragCoord) {
    // Normalize screen coordinates
    float2 uv = (fragCoord - iResolution.xy * 0.5) / iResolution.y;
    
    // Time variables for animation
    float animTime = iTime * 4.0 + 5.0 + 5.0 * sin(iTime * 0.3);
    
    // Setup Camera
    float3 rayOrigin = getPathPosition(animTime);
    float3 lookTarget = getPathPosition(animTime + 4.0);
    
    float3 forward = normalize(lookTarget - rayOrigin);
    float3 right = normalize(float3(-forward.z, 0.0, forward.x)); 
    float3 up = cross(forward, right);
    float3 rayDir = normalize(uv.x * right + uv.y * up + forward);

    // Initializations
    float stepDist = 1.0; 
    float totalDist = 0.0;
    float orbDist = 1.0;  
    float3 accumulatedColor = ((float3)(0.0));
    float3 rayPos = rayOrigin; 

    // Raymarching Loop
    for (float i = 1.0; i <= 28.0; i++) {
        if (totalDist >= 30.0) break;
        
        // 1. March Ray Forward
        rayPos += rayDir * stepDist;
        
        // 2. Get path center and time vars
        float3 pathCenter = getPathPosition(rayPos.z);
        float sineTime = sin(iTime);
        
        // 3. Orb Geometry
        float3 orbCenter = float3(
            pathCenter.x + sineTime,
            pathCenter.y + sineTime * 2.0,
            6.0 + animTime + sineTime * 2.0
        );
        orbDist = length(rayPos - orbCenter) - 0.01;
        
        // 4. Base Tunnel Structure 
        float baseRadius = cos(rayPos.z * 0.6) * 2.0 + 4.0;
        
        float tunnelStructure = min(
            length(rayPos.xy - pathCenter.x - 6.0),
            length((rayPos - pathCenter).xy)
        );
        
        // Crisp architectural scoops
        float largeScoops = abs(dot(sin(0.4 * rayPos), ((float3)(0.25)))) / 0.1;
        float detailTexture = abs(dot(sin(animTime + 16.0 * rayPos), ((float3)(0.22)))) / 2.0;
        
        float carvedDist = baseRadius - tunnelStructure + largeScoops + detailTexture;
        
        // 5. FIREWALL FLUIDITY
        float3 fluidPos = rayPos;
        for (float j = 1.0; j <= 7.0; j++) {
            fluidPos += sin(fluidPos.yzx * j + iTime + 0.5 * i) / j;
        }
        
        float fluidTunnelDist = 0.4 * length(float4(0.3 * cos(fluidPos) - 0.3, carvedDist));
        
        // 6. Update Distances
        stepDist = min(orbDist, fluidTunnelDist);
        totalDist += stepDist;
        
        // 7. Accumulate Color 
        float3 palette = 1.0 + cos(fluidPos.y + i * 0.4 + float3(6.0, 1.0, 2.0));
        
        // BRIGHTNESS FIX: Multiplied the wall lighting (palette / stepDist) by 2.5
        // This makes the geometry glow much brighter against the dark carved-out holes
        accumulatedColor += (2.5 * palette / stepDist + 10.0 * palette / max(orbDist, 0.6)) / i;
    }
    
    // Tonemapping
    // Lowered divisor slightly to 1500.0 to raise the overall exposure
    fragColor = float4(tanh(accumulatedColor * accumulatedColor / 1500.0), 1.0);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
