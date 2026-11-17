cbuffer ShadertoyCB : register(b0) {
    float3 iResolution;
    float iTime;
    float4 iMouse;
};

float glsl_mod(float x, float y) { return x - y * floor(x / y); }

float smin2(float a, float b, float k) {
    float invK = 1.0 / max(k, 1e-5);
    float h = max(k - abs(a - b), 0.0) * invK;
    float m = min(a, b) - h * h * k * 0.25;
    return m - k * 0.0625 * exp2(-abs(a - b) * invK * 1.44269504);
}

float gyroidHarmonic(float3 p, float scale, float phase) {
    float3 ps = p * scale + phase;
    float3 s = sin(ps);
    float3 c = cos(ps.yzx * 1.1547005 + s * 0.35);
    return dot(s, c) / scale;
}

float3 divergencePalette(float t) {
    t = clamp(t, 0.0, 1.0);
    const float3 c0 = float3(0.05, 0.03, 0.18);
    const float3 c1 = float3(0.36, 0.09, 0.55);
    const float3 c2 = float3(0.84, 0.24, 0.31);
    const float3 c3 = float3(0.99, 0.72, 0.16);
    const float3 c4 = float3(0.95, 0.99, 0.72);
    float3 col = lerp(c0, c1, smoothstep(0.00, 0.25, t));
    col = lerp(col, c2, smoothstep(0.25, 0.52, t));
    col = lerp(col, c3, smoothstep(0.52, 0.80, t));
    col = lerp(col, c4, smoothstep(0.80, 1.00, t));
    return col;
}

float4 mapCathedral(float3 p, float4 uParams, out float orbitTrap, out float causticEnergy) {
    float morph = uParams.x;
    float harmonics = uParams.y;
    float tFlow = iTime * 0.32;

    float r0 = length(p);

    float3 warpPhase = p * (1.35 + 0.65 * morph) + float3(tFlow, -tFlow * 0.7, tFlow * 0.5);
    float3 warpVec = sin(warpPhase) * cos(warpPhase.zxy * 1.25 + 0.5);
    float3 z = p + warpVec * (0.14 * harmonics);

    float bayPeriod = 4.2;
    z.z = glsl_mod(z.z + bayPeriod * 0.5, bayPeriod) - bayPeriod * 0.5;

    float2 a = abs(z.xy);
    if (a.x < a.y) a = a.yx;
    z.xy = a;

    float scale = 1.0;
    float trap = 1e6;
    float3 trapColor = float3(0.0, 0.0, 0.0);

    float3 foldClamp = lerp(float3(1.12, 1.18, 1.05), float3(1.38, 0.96, 1.26), morph);
    float minRad2 = lerp(0.28, 0.42, sin(tFlow * 0.8) * 0.5 + 0.5);
    float fixedRad2 = 1.35;

    for (int i = 0; i < 5; i++) {
        z = clamp(z, -foldClamp, foldClamp) * 2.0 - z;

        float r2 = dot(z, z);
        float k = clamp(fixedRad2 / max(r2, minRad2), 1.0, fixedRad2 / minRad2);
        z *= k;
        scale *= k;

        float fi = float(i);
        float3 trigPerturb = sin(z * (2.4 + fi * 0.9) + tFlow) * (0.035 * harmonics * exp2(-fi * 0.65));
        z += trigPerturb;

        float dTrap = length(z.xy - float2(0.45, 0.32));
        trap = min(trap, dTrap);
        trapColor += exp2(-float3(1.4, 2.1, 3.0) * dTrap);
    }

    float dArch = (length(z) - 0.85) / scale;

    float g1 = gyroidHarmonic(p * 1.45, 2.2 * harmonics + 0.4, tFlow * 0.9);
    float g2 = gyroidHarmonic(p * 3.10 + warpVec * 0.4, 3.6 * harmonics + 0.6, -tFlow * 1.3);
    float dFiligree = abs(g1 * 0.65 + g2 * 0.35) - (0.022 + 0.012 * morph);

    float dCathedral = smin2(dArch, max(dArch - 0.14, dFiligree), 0.085);

    float3 pCore = p;
    pCore.z = glsl_mod(pCore.z + 2.1, 4.2) - 2.1;
    float ringAngle = atan2(pCore.y, pCore.x) * 4.0 + tFlow * 2.5;
    float ringWave = sin(ringAngle) * cos(pCore.z * 5.0 - tFlow * 3.0);
    float dRing = length(float2(length(pCore.xy) - (0.72 + 0.08 * ringWave), pCore.z)) - 0.038;

    float dFinal = smin2(dCathedral, dRing, 0.06);

    float causticWave = pow(max(0.0, cos(g1 * 7.5 + tFlow * 2.0) * sin(g2 * 6.5 - tFlow * 1.5)), 3.0);
    causticEnergy = causticWave * exp2(-abs(dFinal) * 3.5);
    orbitTrap = clamp(trap * 0.85 + length(trapColor) * 0.08, 0.0, 2.0);

    float matBlend = smoothstep(0.04, 0.0, dRing - dCathedral);

    return float4(dFinal, matBlend, r0, trap);
}

float mapDist(float3 p, float4 uParams) {
    float t1, t2;
    return mapCathedral(p, uParams, t1, t2).x;
}

float3 calcNormal(float3 p, float eps, float4 uParams) {
    float2 e = float2(1.0, -1.0) * 0.57735027 * max(eps * 1.6, 0.0006);
    return normalize(
        e.xyy * mapDist(p + e.xyy, uParams) +
        e.yyx * mapDist(p + e.yyx, uParams) +
        e.yxy * mapDist(p + e.yxy, uParams) +
        e.xxx * mapDist(p + e.xxx, uParams)
    );
}

float3 thinFilmIridescence(float cosTheta, float phase, int preset) {
    float3 freq, offset, amp, bias;
    if (preset == 0) {
        bias = float3(0.56, 0.52, 0.58);
        amp = float3(0.42, 0.38, 0.40);
        freq = float3(1.00, 1.00, 1.00);
        offset = float3(0.00, 0.14, 0.28);
    } else if (preset == 1) {
        bias = float3(0.18, 0.48, 0.56);
        amp = float3(0.35, 0.44, 0.42);
        freq = float3(1.10, 0.95, 1.05);
        offset = float3(0.52, 0.15, 0.05);
    } else if (preset == 2) {
        bias = float3(0.48, 0.22, 0.10);
        amp = float3(0.48, 0.32, 0.18);
        freq = float3(0.85, 1.05, 1.30);
        offset = float3(0.00, 0.12, 0.26);
    } else {
        bias = float3(0.50, 0.46, 0.55);
        amp = float3(0.48, 0.46, 0.48);
        freq = float3(1.35, 1.25, 1.15);
        offset = float3(0.65, 0.33, 0.05);
    }

    float opticalPath = (1.0 - cosTheta) * 2.6 + phase;
    return clamp(bias + amp * cos(6.2831853 * (freq * opticalPath + offset)), 0.0, 1.0);
}

float3 presetGlowTint(int preset) {
    if (preset == 0) return float3(1.00, 0.78, 0.42);
    if (preset == 1) return float3(0.14, 0.92, 0.84);
    if (preset == 2) return float3(1.00, 0.42, 0.12);
    return float3(0.86, 0.36, 1.00);
}

float3 presetVoidTint(int preset) {
    if (preset == 0) return float3(0.03, 0.04, 0.09);
    if (preset == 1) return float3(0.01, 0.04, 0.07);
    if (preset == 2) return float3(0.04, 0.02, 0.02);
    return float3(0.03, 0.02, 0.07);
}

float3 renderCosmicBackdrop(float3 rd, int preset) {
    float3 baseVoid = presetVoidTint(preset);
    float3 glowTint = presetGlowTint(preset);

    float3 p = rd * 3.2 + float3(iTime * 0.04, iTime * 0.04, iTime * 0.04);
    float n1 = sin(p.x * 2.1 + cos(p.y * 2.7 + p.z * 1.9));
    float n2 = cos(p.y * 3.4 - sin(p.z * 2.9 - p.x * 2.3));
    float nebula = pow(clamp(n1 * n2 * 0.5 + 0.5, 0.0, 1.0), 3.0);

    float3 starGrid = frac(rd * 165.0) - 0.5;
    float3 starCell = floor(rd * 165.0);
    float h = frac(sin(dot(starCell, float3(127.1, 311.7, 74.7))) * 43758.5453);
    float star = smoothstep(0.985, 0.999, h) * exp2(-dot(starGrid, starGrid) * 42.0);

    float3 sky = baseVoid + glowTint * nebula * 0.24;
    sky += lerp(float3(0.7, 0.85, 1.0), glowTint, 0.45) * star * 1.6;
    return sky;
}

float4 PSMain(float4 fragCoord : SV_Position) : SV_Target0 {
    float4 uCameraPos = float4(0.0, 0.0, -3.2, 1.5);
    float4 uCameraTarget = float4(0.0, 0.0, 0.0, 0.0);
    float4 uParams = float4(0.35 + 0.25 * sin(iTime * 0.5), 1.0, 0.0015, 96.0);
    int2 uModes = int2(0, 0);

    float2 uv = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);

    float3 ro = uCameraPos.xyz;
    float3 ta = uCameraTarget.xyz;
    float3 forward = normalize(ta - ro);
    float3 right = normalize(cross(forward, float3(0.0, 1.0, 0.0)));
    float3 up = cross(right, forward);

    float r2Lens = dot(uv, uv);
    float3 rd = normalize(forward * uCameraPos.w + (right * uv.x + up * uv.y) * (1.0 + 0.045 * r2Lens));

    float epsilon = uParams.z;
    int maxSteps = int(uParams.w);
    int preset = uModes.x;
    int heatmap = uModes.y;

    const float MAX_DIST = 26.0;
    float t = 0.08;
    float omega = 1.18;
    float prevD = 1e6;

    float volGlow = 0.0;
    float3 volColor = float3(0.0, 0.0, 0.0);
    float3 glowTint = presetGlowTint(preset);

    bool hit = false;
    int stepsTaken = 0;
    float orbitTrap = 0.0;
    float causticEnergy = 0.0;
    float matBlend = 0.0;

    for (int i = 0; i < 180; i++) {
        if (i >= maxSteps) break;
        stepsTaken = i + 1;

        float3 pos = ro + rd * t;
        float4 res = mapCathedral(pos, uParams, orbitTrap, causticEnergy);
        float d = res.x;

        float stepLen = d * omega;
        if (omega > 1.0 && (d + prevD) < stepLen) {
            stepLen = d;
            omega = 1.0;
        }

        float glowSample = exp2(-abs(d) * 7.5) * (0.018 + 0.045 * causticEnergy);
        volGlow += glowSample;
        volColor += thinFilmIridescence(0.5, orbitTrap * 0.4 + t * 0.08, preset) * glowSample;

        float adaptiveEps = epsilon * (1.0 + t * 0.12);
        if (d < adaptiveEps) {
            hit = true;
            matBlend = res.y;
            break;
        }
        if (t > MAX_DIST) {
            break;
        }

        prevD = d;
        t += max(stepLen, adaptiveEps * 0.65);
    }

    if (heatmap == 1) {
        float normSteps = float(stepsTaken) / float(maxSteps);
        float3 heat = divergencePalette(pow(normSteps, 0.78));
        float contour = smoothstep(0.92, 1.0, cos(float(stepsTaken) * 0.392699));
        heat = lerp(heat, float3(1.0, 1.0, 1.0), contour * 0.25);
        return float4(heat, 1.0);
    }

    float3 col = renderCosmicBackdrop(rd, preset);

    if (hit) {
        float3 p = ro + rd * t;
        float3 n = calcNormal(p, epsilon, uParams);
        float3 v = -rd;

        float3 lightDir1 = normalize(float3(0.65, 0.85, -0.55));
        float3 lightDir2 = normalize(float3(-0.75, -0.35, 0.65));

        float NdotV = max(dot(n, v), 0.001);
        float NdotL1 = max(dot(n, lightDir1), 0.0);
        float NdotL2 = max(dot(n, lightDir2), 0.0);

        float3 h1 = normalize(lightDir1 + v);
        float3 h2 = normalize(lightDir2 + v);
        float spec1 = pow(max(dot(n, h1), 0.0), 48.0);
        float spec2 = pow(max(dot(n, h2), 0.0), 24.0);

        float fresnel = exp2(-8.65 * NdotV);

        float3 irid = thinFilmIridescence(NdotV, orbitTrap * 0.65 + causticEnergy * 0.3, preset);
        float3 goldCore = lerp(float3(1.0, 0.82, 0.42), glowTint * 1.35, 0.45);
        float3 baseAlbedo = lerp(irid, goldCore, matBlend);

        float ao = 1.0;
        float aoScale = 1.0;
        for (int k = 1; k <= 3; k++) {
            float hr = 0.055 * float(k);
            float dd = mapDist(p + n * hr, uParams);
            ao -= (hr - dd) * aoScale;
            aoScale *= 0.55;
        }
        ao = clamp(ao, 0.12, 1.0);

        float3 reflDir = reflect(-v, n);
        float3 envRefl = renderCosmicBackdrop(reflDir, preset) * 1.45 +
                         thinFilmIridescence(max(dot(reflDir, n), 0.0), orbitTrap, preset) * 0.42;

        float3 direct = baseAlbedo * (NdotL1 * 1.15 + NdotL2 * 0.45 + 0.14) * ao;
        float3 specular = lerp(float3(1.0, 1.0, 1.0), glowTint, 0.35) * (spec1 * 1.35 + spec2 * 0.55);
        float3 subsurface = glowTint * causticEnergy * (0.65 + 0.85 * matBlend);

        col = direct + specular + subsurface;
        col = lerp(col, envRefl, clamp(fresnel * 0.78 + 0.12, 0.0, 0.92));

        float fogFactor = exp2(-t * 0.095);
        col = lerp(renderCosmicBackdrop(rd, preset), col, fogFactor);
    }

    col += volColor * 1.35 + glowTint * pow(volGlow, 1.35) * 0.48;

    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);

    float vig = 1.0 - 0.22 * r2Lens;
    col = pow(clamp(col * vig, 0.0, 1.0), float3(0.454545, 0.454545, 0.454545));

    return float4(col, 1.0);
}
