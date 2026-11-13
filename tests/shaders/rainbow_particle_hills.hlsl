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

// Rainbow Hills
// By Noztol
// Inspired by:
// https://fragcoord.xyz/s/hfy98teh (except no golf style)
// https://fragcoord.xyz/s/siovosn2
// Some Code copied from: https://fragcoord.xyz/s/gcbixaag

static const float BASE_SIZE            = 0.18;
static const float BASE_OFFSET          = 0.2;
static const float BASE_GRADIENT        = -1.8;
static const float ANIMATE_SPEED        = 1.0;
static const float ANIMATE_BASE         = 0.53;
static const float ANIMATE_DETAIL       = -0.11;
static const float ANIMATE_DETAIL2      = 0.1;
static const float3  ANIMATE_BASE_DIR     = float3(0.0, 0.00, 1.00);
static const float3  ANIMATE_DETAIL_DIR   = float3(1.0, 0.31, 0.55);
static const float3  ANIMATE_DETAIL2_DIR  = float3(1.0, 0.00, 0.00);
static const float BASE_SHAPE           = 1.54;
static const float BASE_WEIGHT          = 2.06;
static const float DETAIL_WEIGHT        = 0.93;
static const float DETAIL_COMBINE       = -0.25;
static const float DENSITY_CUTOFF       = 0.47;
static const float4  LOW_DENSITY_COLOR    = float4(0.668, 0.586, 0.801, 0.043);
static const float4  HIGH_DENSITY_COLOR   = float4(0.317, 0.250, 0.368, 0.270);
static const float4  SUN_COLOR            = float4(1.000, 0.572, 0.410, 0.620);
static const float4  BACKGROUND_COLOR     = float4(0.317, 0.250, 0.368, 0.886);
static const float4  BACKGROUND_SUN_COLOR = float4(0.698, 0.321, 0.239, 0.317);
static const float BACKGROUND_SUN_SIZE  = 0.22;
static const float NEAR_CLIP            = 6.0;
static const float FAR_CLIP             = 42.0;
static const float FAR_FADE             = 0.75;
static const float3  SUN_DIR              = float3(-0.61, -0.1, 1.96);
static const float SUN_CURVE            = 0.286;
static const float SUN_OFFSET           = 0.963;
static const float GAMMA_VALUE          = 4.0;
static const float DETAIL_CUTOFF        = -0.4;
static const float ALPHA_MAX            = 0.93;
static const float STEP_SIZE_INSIDE     = 0.27;
static const float STEP_SIZE_OUTSIDE    = 0.66;
static const float JITTER_AMOUNT        = 2.0;
static const int   MAX_RAYMARCH_STEPS   = 159;

// ==============================================================================
// MATH & COLOR HELPER FUNCTIONS
// ==============================================================================

float applyGamma(float colorValue) { return pow(colorValue, GAMMA_VALUE); }
float3 applyGamma(float3 colorVec) { return pow(colorVec, ((float3)(GAMMA_VALUE))); }
float4 applyGamma(float4 colorVec) { colorVec.rgb = pow(colorVec.rgb, ((float3)(GAMMA_VALUE))); return colorVec; }

float3 generateRainbowPalette(float3 position) {
    return 0.5 + 0.5 * cos(iTime * 0.5 + position.x * 2.0 + position.y * 3.0 + position.z * 1.5 + float3(0.0, 2.0, 4.0));
}

float3 smoothInterpolationCurve(float3 fractionalPos) {
    return fractionalPos * fractionalPos * (3.0 - 2.0 * fractionalPos);
}

// ==============================================================================
// RANDOMIZATION & NOISE HELPER FUNCTIONS
// ==============================================================================

float generate2DHash(uint2 query) {   
    query *= uint2(1597334677U, 3812015801U);
    uint result = (query.x ^ query.y) * 1597334677U;
    return float(result) * (1.0 / float(0xffffffffU));
}

float generateJitter(float2 seed) {
    return generate2DHash(((uint2)(seed * 9e5)));
}

float generate3DHash(float3 position) {
    uint3 query = ((uint3)(position * 9e5));
    query *= uint3(1597334677U, 3812015801U, 2798796415U);
    uint result = (query.x ^ query.y ^ query.z) * 1597334677U;
    return float(result) * (1.0 / float(0xffffffffU));
}

float generateValueNoise3D(float3 position) {
    float3 gridIndex = floor(position);
    float3 fractionalPos = frac(position);
    float3 smoothCurve = smoothInterpolationCurve(fractionalPos);
    
    // Sample the 8 corners of the current 3D grid cell
    float frontBottomLeft  = generate3DHash(gridIndex + float3(0.0, 0.0, 0.0));
    float frontBottomRight = generate3DHash(gridIndex + float3(1.0, 0.0, 0.0));
    float frontTopLeft     = generate3DHash(gridIndex + float3(0.0, 1.0, 0.0));
    float frontTopRight    = generate3DHash(gridIndex + float3(1.0, 1.0, 0.0));
    
    float backBottomLeft   = generate3DHash(gridIndex + float3(0.0, 0.0, 1.0));
    float backBottomRight  = generate3DHash(gridIndex + float3(1.0, 0.0, 1.0));
    float backTopLeft      = generate3DHash(gridIndex + float3(0.0, 1.0, 1.0));
    float backTopRight     = generate3DHash(gridIndex + float3(1.0, 1.0, 1.0));
    
    // Interpolate along the X-axis
    float interpFrontBottom = lerp(frontBottomLeft, frontBottomRight, smoothCurve.x);
    float interpFrontTop    = lerp(frontTopLeft, frontTopRight, smoothCurve.x);
    float interpBackBottom  = lerp(backBottomLeft, backBottomRight, smoothCurve.x);
    float interpBackTop     = lerp(backTopLeft, backTopRight, smoothCurve.x);
    
    // Interpolate along the Y-axis
    float interpFrontPlane = lerp(interpFrontBottom, interpFrontTop, smoothCurve.y);
    float interpBackPlane  = lerp(interpBackBottom, interpBackTop, smoothCurve.y);
    
    // Interpolate along the Z-axis for the final value
    float finalNoiseValue = lerp(interpFrontPlane, interpBackPlane, smoothCurve.z);
    
    return finalNoiseValue * 2.0 - 1.0;
}

// ==============================================================================
// CLOUD DENSITY SHAPING FUNCTIONS
// ==============================================================================

// FIXED: Added 'inout' to position so the forward velocity carries over to details!
float calculateBaseCloudShape(inout float3 position, out float normalizedNoiseOut) {
    float timeOffset = iTime * ANIMATE_SPEED;
    position += timeOffset * ANIMATE_BASE * ANIMATE_BASE_DIR;
    
    float3x3 rotationMatrix = float3x3(0.5, -0.5, 0.70711, 0.85355, 0.14644, -0.5, 0.14644, 0.85355, 0.5);
    float baseNoise = generateValueNoise3D(mul(position, rotationMatrix));
    
    normalizedNoiseOut = 0.5 + 0.5 * baseNoise; 
    normalizedNoiseOut = pow(normalizedNoiseOut, BASE_SHAPE);
    
    return BASE_WEIGHT * (normalizedNoiseOut * 2.0 - 1.0);
}

float calculateCloudDetails(float3 position, float distanceToCamera, float normalizedBaseNoise) {
    float timeOffset = iTime * ANIMATE_SPEED;
    
    position += timeOffset * ANIMATE_DETAIL * ANIMATE_DETAIL_DIR;
    float fractalDetail = 0.28 * generateValueNoise3D(position * 5.0);
    fractalDetail += 0.2 * generateValueNoise3D(position * 10.1);
    
    position += timeOffset * ANIMATE_DETAIL2 * ANIMATE_DETAIL2_DIR;
    fractalDetail += 0.1 * generateValueNoise3D(position * 22.52);
    
    float fadeDistance = 6.0;
    if (distanceToCamera < fadeDistance) {
        fractalDetail += 0.057 * generateValueNoise3D(position * 54.28);
        float distanceFadeRatio = (fadeDistance - distanceToCamera) * (1.0 / fadeDistance);
        fractalDetail += 0.1 * distanceFadeRatio * generateValueNoise3D(position * 154.28);
    }
    
    return DETAIL_WEIGHT * fractalDetail * (1.0 - DETAIL_COMBINE * (normalizedBaseNoise - 0.5));
}

float computeTotalCloudDensity(float3 position, float distanceToCamera) {
    float heightGradient = (position.y * BASE_GRADIENT + BASE_OFFSET) * BASE_WEIGHT;
    
    // Early exit if we are too low below the cloud layer
    if (heightGradient < -BASE_WEIGHT) {
        return -1000.0;
    }
    
    float normalizedBaseNoise;
    // Calling this now properly mutates the local 'position' variable
    float totalDensity = calculateBaseCloudShape(position, normalizedBaseNoise) + heightGradient;
    
    if (totalDensity > DETAIL_CUTOFF) {
        // The mutated 'position' is now correctly passed to the details layer
        totalDensity += calculateCloudDetails(position, distanceToCamera, normalizedBaseNoise);
    }    
    
    return totalDensity;
}

// ==============================================================================
// LIGHTING AND RAYMARCHING
// ==============================================================================

float calculateDirectionalLighting(float3 position, float distanceToCamera, float currentDensity) {
    float3 sunDirectionNorm = normalize(SUN_DIR);
    float offsetDensity = computeTotalCloudDensity(position + 0.06 * sunDirectionNorm, distanceToCamera);
    float densityDifference = offsetDensity - currentDensity;
    
    float shadowCurve = pow(max(0.0, SUN_OFFSET - position.y * BASE_GRADIENT * SUN_CURVE), 8.0);
    return max(0.0, -shadowCurve * densityDifference / 0.2);
}

float4 performRaymarching(float3 rayOrigin, float3 rayDirection, float2 screenCoords) {
    float4 accumulatedColor = ((float4)(0.0));
    float depthAlphaSum = 0.0;
    
    // Add jitter to hide rigid raymarching steps
    float rayDistance = JITTER_AMOUNT * generateJitter(screenCoords + frac(iTime));

    float4 lowDensityColor  = applyGamma(LOW_DENSITY_COLOR);  
    float4 highDensityColor = applyGamma(HIGH_DENSITY_COLOR);
    float3 baseSunColor     = applyGamma(SUN_COLOR).rgb;
    float ambientLight    = applyGamma(SUN_COLOR.a);

    for (int stepIndex = 0; stepIndex < MAX_RAYMARCH_STEPS; ++stepIndex) {
        float3 currentPos = rayOrigin + rayDistance * rayDirection;
        currentPos *= BASE_SIZE;
        float currentDensity = computeTotalCloudDensity(currentPos, rayDistance * BASE_SIZE);

        if (currentDensity < -100.0 && sign(BASE_GRADIENT) * rayDirection.y < 0.0) {
            break; // Stop marching if we hit the void below the clouds
        }
            
        if (currentDensity > DENSITY_CUTOFF) {
            float sunLightIntensity = calculateDirectionalLighting(currentPos, rayDistance * BASE_SIZE, currentDensity);
            
            currentDensity = clamp(currentDensity, 0.0, 1.0);
            
            float3 dynamicRainbowPalette = generateRainbowPalette(currentPos);
            float4 sampledColor = float4(applyGamma(dynamicRainbowPalette) * lerp(0.35, 1.0, currentDensity), 
                                     lerp(lowDensityColor.a, highDensityColor.a, currentDensity));

            sampledColor.rgb *= ambientLight + baseSunColor * sunLightIntensity * 5.0;
            sampledColor.a *= min(1.0, rayDistance / NEAR_CLIP);

            float weight = sampledColor.a * (1.0 - accumulatedColor.a);
            depthAlphaSum += weight * smoothstep(1.0 - FAR_FADE, 1.0, rayDistance / FAR_CLIP);                               
            
            accumulatedColor.rgb += weight * sampledColor.rgb;
            accumulatedColor.a += weight;
        }

        if (currentDensity > DETAIL_CUTOFF) {
            rayDistance += STEP_SIZE_INSIDE;
        } else {
            rayDistance += STEP_SIZE_OUTSIDE;
        }

        if (rayDistance > FAR_CLIP || accumulatedColor.a > ALPHA_MAX) {
            break;
        }
    }

    float finalDepthAlpha = 1.0 - depthAlphaSum / (accumulatedColor.a + 0.00001);
    accumulatedColor.a = accumulatedColor.a / ALPHA_MAX;
    accumulatedColor.a *= finalDepthAlpha;
    accumulatedColor.a = min(1.0, accumulatedColor.a);
    
    return accumulatedColor;
}

// ==============================================================================
// MAIN RENDER LOOP & CAMERA SETUP
// ==============================================================================

float4 compositeCloudAndBackground(float3 rayOrigin, float3 rayDirection, float2 screenCoords) {
    float4 cloudColor = performRaymarching(rayOrigin, rayDirection, screenCoords);
    
    float sunGlow = 0.5 + 0.5 * dot(normalize(SUN_DIR), rayDirection);
    sunGlow = pow((1.0 - pow(1.0 - sunGlow, BACKGROUND_SUN_SIZE) ), 1.0 / BACKGROUND_SUN_SIZE) * 5.0;
    sunGlow *= 1.0 + cloudColor.a * 6.0;
    
    float4 backgroundColor = applyGamma(BACKGROUND_COLOR) + BACKGROUND_SUN_COLOR.a * sunGlow * applyGamma(BACKGROUND_SUN_COLOR);
    
    float4 finalOutputColor = lerp(backgroundColor, float4(cloudColor.rgb, 1.0), cloudColor.a);
    finalOutputColor.rgb = pow(finalOutputColor.rgb, ((float3)(1.0 / GAMMA_VALUE)));
    
    finalOutputColor.rgb *= max(1.0, finalOutputColor.a);
    finalOutputColor.a = min(1.0, finalOutputColor.a);
    
    return finalOutputColor;
}

float2 rotate2D(float2 pt, float angle) { 
    float sine = sin(angle);
    float cosine = cos(angle); 
    return float2(pt.x * cosine - pt.y * sine, pt.x * sine + pt.y * cosine); 
} 

float3 calculateCameraRayDirection(float2 uv, float aspectRatio, float3 position, float3 target, float fovDegrees, float rollAngle) { 
    float3 forwardAxis = normalize(target - position);
    float3 rightAxis = normalize(cross(float3(0.0, 1.0, 0.0), forwardAxis));
    float3 upAxis = normalize(cross(forwardAxis, rightAxis));
    
    float halfHeight = tan(radians(fovDegrees) * 0.5);
    uv = (2.0 * uv - 1.0) * float2(halfHeight * aspectRatio, halfHeight);
    
    float2 rotatedUV = mul(uv, float2x2(cos(rollAngle), -sin(rollAngle), sin(rollAngle), cos(rollAngle)));
    return normalize(forwardAxis + rotatedUV.x * rightAxis + rotatedUV.y * upAxis);
}

void mainImage(out float4 fragColor, in float2 fragCoord)
{
    float2 normalizedCoords = fragCoord.xy / iResolution.xy;
    float aspectRatio = iResolution.x / iResolution.y;
    float2 mouseCoords = (iMouse.xy - 0.5 * iResolution.xy) / iResolution.y;
    
    float3 cameraPos = float3(0.0, 1.7, -3.0);
    float3 cameraTarget = float3(0.0, 1.7, 6.0);
    
    float orbitAngle = 0.0;
    float upDownAngle = 0.0;
    if (iMouse.z > 0.0) {
        upDownAngle = clamp(-4.0 * mouseCoords.y, -1.57079, 1.57079);
        orbitAngle = -4.0 * mouseCoords.x;
    }
    
    cameraPos -= cameraTarget;
    cameraPos.yz = rotate2D(cameraPos.yz, upDownAngle);
    cameraPos.xz = rotate2D(cameraPos.xz, orbitAngle);    
    cameraPos += cameraTarget;
 
    float3 rayDirection = calculateCameraRayDirection(normalizedCoords, aspectRatio, cameraPos, cameraTarget, 40.0, 0.0);
    cameraPos += float3(125.0, 0.0, 64.0); 
    
    fragColor = compositeCloudAndBackground(cameraPos, rayDirection, normalizedCoords);
}

[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
