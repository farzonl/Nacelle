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

// Van Gogh Sunset 
// By Noztol
// Inspired by  bitless Coastal Landscape https://www.shadertoy.com/view/fstyD4

#define p(t, a, b, c, d) ( a + b*cos( 6.28318*(c*t+d) ) ) 
#define sp(t) p(t,float3(.26,.76,.77),float3(1,.3,1),float3(.8,.4,.7),float3(0,.12,.54)) 
#define hue(v) ( .6 + .76 * cos(6.3*(v) + float4(0,23,21,0) ) ) 

#define smoothing      0.006
#define TWO_PI         6.28318530718

#define MountainLayerThreecol float3(26., 65., 74.)/255.
#define MountainLayerFourCol float3(14., 49., 55.)/255.
#define FieldMid              float3(94., 121., 62.)/255.


float hash(float p) {
    p = frac(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return frac(p);
}

float noise(float x) {
    float i = floor(x);
    float f = frac(x);
    f = f * f * (3.0 - 2.0 * f);
    return lerp(hash(i), hash(i + 1.0), f);
}

float hash12(float2 p) {
    float3 p3  = frac(((float3)(p.xyx)) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return frac((p3.x + p3.y) * p3.z);
}

float2 hash22(float2 p) {
    float3 p3 = frac(((float3)(p.xyx)) * float3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx+33.33);
    return frac((p3.xx+p3.yz)*p3.zy);
}

float2 rotate2D (float2 st, float a){
    return mul(st, float2x2(cos(a),-sin(a),sin(a),cos(a)));
}

float st(float a, float b, float s) { return smoothstep (a-s, a+s, b); }

float noise( in float2 p ) {
    float2 i = floor( p );
    float2 f = frac( p );
    float2 u = f*f*(3.-2.*f);
    return lerp( lerp( dot( hash22( i+float2(0,0) ), f-float2(0,0) ), 
                     dot( hash22( i+float2(1,0) ), f-float2(1,0) ), u.x),
                lerp( dot( hash22( i+float2(0,1) ), f-float2(0,1) ), 
                     dot( hash22( i+float2(1,1) ), f-float2(1,1) ), u.x), u.y);
}

float s_noise(float2 p) { return noise(p)*0.5 + 0.5; }


float3 getPaintedSkyAndMountains(float2 g, float2 r, float time) {
    float2 uv = (g+g-r)/r.y;
    float2 sun_pos = float2(r.x/r.y * 0.35, -0.15); 
    float2 sh = rotate2D(sun_pos, noise(uv+time*.25)*.3); 
    
    float3 f = ((float3)(0));
    float sm = 3./r.y; 
    float2 u, id, lc, t;
    float xd, yd, h;

    u = uv + sh;
    yd = 60.; 
    id =  float2((length(u)+.01)*yd,0); 
    xd = floor(id.x)*.09; 
    h = (hash12(floor(id.xx))*.5+.25)*(time+10.)*.25; 
    t = rotate2D (u,h); 
    id.y = atan2(t.y,t.x)*xd;
    lc = frac(id); 
    id -= lc;

    t = float2(cos((id.y+.5)/xd)*(id.x+.5)/yd,sin((id.y+.5)/xd)*(id.x+.5)/yd); 
    t = rotate2D(t,-h) - sh;

    h = noise(t*float2(.5,1)-float2(time*.2,0)) * step(-.25,t.y); 
    h = smoothstep (.052,.055, h);
    lc += (noise(lc*float2(1,4)+id))*float2(.7,.2); 
    
    f = lerp (sp(sin(length(u)-.1))*.35, 
             lerp(sp(sin(length(u)-.1)+(hash12(id)-.5)*.15),((float3)(1)),h), 
             st(abs(lc.x-.5),.4,sm*yd)*st(abs(lc.y-.5),.48,sm*xd));
             

    float cld = noise(-sh*float2(.5,1)  - float2(time*.2,0)); 
    cld = 1.- smoothstep(.0,.15,cld)*.5;
    u = (uv - float2(0.0, 0.25)) * float2(1, 15); 
    id = floor(u);

    for (float i = 1.; i > -1.; i--) {
        if (id.y+i < 0.0) {
            lc = frac(u)-.5;
            lc.y = (lc.y+(sin(uv.x*8.-time*0.8+id.y+i))*.3-i)*4.; 
            h = hash12(float2(id.y+i,floor(lc.y))); 
            xd = 6.+h*4.;
            yd = 30.;
            lc.x = uv.x*xd+sh.x*9.; 
            lc.x += sin(time * (.2 + h*1.5))*.5; 
            h = .8*smoothstep(5.,.0,abs(floor(lc.x)))*cld+.1; 
            
            f = lerp(f,lerp(MountainLayerFourCol,MountainLayerThreecol,h),st(lc.y,0.,sm*yd)); 
            lc += noise(lc*float2(3,.5))*float2(.1,.6); 
            
            float3 strokeCol = hue(hash12(floor(lc))*.1+.35).rgb*(1.2+floor(lc.y)*.17);
            f = lerp(f, lerp(strokeCol,FieldMid,h), st(lc.y,0.,sm*xd)*st(abs(frac(lc.x)-.5),.48,sm*xd)*st(abs(frac(lc.y)-.5),.3,sm*yd));
        }
    }
    return f;
}


float3 getPaintedWaterAndSand(float2 g, float2 r, float time, float3 bgCol) {
    float2 uv = (g+g-r)/r.y;
    float2 sun_pos = float2(r.x/r.y * 0.35, -0.15); 
    float2 sh = rotate2D(sun_pos, noise(uv+time*.25)*.3); 
    float sm = 3./r.y;
    float3 f = bgCol;
    
    float waterZone = smoothstep(0.16, 0.14, uv.y);
    f = lerp(f, float3(0.05, 0.15, 0.3), waterZone);

    float cld = noise(-sh*float2(.5,1)  - float2(time*.2,0));
    cld = 1.- smoothstep(.0,.15,cld)*.5;

    float2 u = (uv - float2(0.0, 0.15)) * float2(1,15);
    float2 id = floor(u);

    for (float i = 1.; i > -1.; i--) {
        if (id.y+i < 0.0) {
            float2 lc = frac(u)-.5;
            lc.y = (lc.y+(sin(uv.x*12.-time*3.+id.y+i))*.25-i)*4.;
            float h = hash12(float2(id.y+i,floor(lc.y)));
            
            float xd = 6.+h*4.;
            float yd = 30.;
            lc.x = uv.x*xd+sh.x*9.;
            lc.x += sin(time * (.5 + h*2.))*.5;
            h = .8*smoothstep(5.,.0,abs(floor(lc.x)))*cld+.1;
            
            float3 waterBase = lerp(float3(0.0, 0.15, 0.4), float3(0.6, 0.4, 0.1), h);
            f = lerp(f, waterBase, st(lc.y, 0., sm*yd));
            lc += noise(lc*float2(3,.5))*float2(.1,.6);
            
            float3 strokeCol = lerp(hue(hash12(floor(lc))*.1+.56).rgb*(1.2+floor(lc.y)*.17), float3(1.0, 0.8, 0.2), h);
            f = lerp(f, strokeCol, st(lc.y,0.,sm*xd)*st(abs(frac(lc.x)-.5),.48,sm*xd)*st(abs(frac(lc.y)-.5),.3,sm*yd));
        }
    }
    
    float2 u_sand = uv + noise(uv*2.)*.1 + float2(0, sin(uv.x*1.2+3.)*.2 + 0.7); 
    float3 sandCol = float3(0.85, 0.75, 0.50); 
    float3 sandDark = float3(0.60, 0.45, 0.25);
    f = lerp(f, sandDark * 0.7, step(u_sand.y, .0)); 

    float xd_s = 60.; 
    u_sand = u_sand * float2(xd_s, xd_s/3.5); 
    
    if (u_sand.y < 1.2) {
        for (float y = 0.; y > -3.; y--) {
            for (float x = -2.; x < 3.; x++) {
                float2 id_s = floor(u_sand) + float2(x,y);
                float2 lc_s = (frac(u_sand) + float2(1.-x,-y))/float2(5,3);
                float h_s = (hash12(id_s)-.5)*.25+.5; 

                lc_s -= float2(.3, .5-h_s*.4);
                lc_s.x += sin(((time*0.5+h_s*2.-id_s.x*.05-id_s.y*.05)*1.1+id_s.y*.5)*2.)*(lc_s.y+.5)*.2;
                float2 t_s = abs(lc_s)-float2(.03, .5-h_s*.5); 
                float l = length(max(t_s,0.)) + min(max(t_s.x,t_s.y),0.); 

                l -= noise(lc_s*7.+id_s)*.1; 
                
                float4 C = float4(sandDark * 0.5, st(l, .1, sm*xd_s*.09)); 
                float4 fg = float4(sandCol * (1.1+lc_s.y*1.2) * (1.2-h_s*1.2), 1.);
                C = lerp(C, fg, st(l, .04, sm*xd_s*.09));
                
                f = lerp(f, C.rgb, C.a * step(id_s.y, -1.));
            }
        }
    }
    return f;
}


float sdTri(float2 p, float base, float tip, float width) {
    float dY = max(base - p.y, p.y - tip);
    float frac = clamp((p.y - base) / (tip - base), 0.0, 1.0);
    float curWidth = width * (1.0 - frac);
    float dX = abs(p.x) - curWidth;
    return max(dX, dY);
}

float sdTree(float2 p, float2 pos, float scale) {
    float2 q = (p - pos) / scale;
    q.x -= sin(iTime * 1.5 + pos.x * 10.0) * 0.02 * max(0.0, q.y);
    float d = max(abs(q.x) - 0.03, max(-q.y, q.y - 0.2));
    d = min(d, sdTri(q, 0.1, 0.45, 0.25));
    d = min(d, sdTri(q, 0.25, 0.65, 0.2));
    d = min(d, sdTri(q, 0.45, 0.85, 0.15));
    d = min(d, sdTri(q, 0.65, 1.0, 0.1));
    return d * scale;
}

float getLand(float2 uv, float aspect) {
    float d = 1.0;
    
    // Left Landmass
    float w1 = 0.05 * exp(-12.0 * pow(uv.x - aspect*0.1, 2.0)); 
    float c1 = 0.36 - 0.01 * uv.x;
    // Taper Cutoff: smoothstep adds distance penalty when width drops below 0.008
    float land1 = abs(uv.y - c1) - w1 + smoothstep(0.008, 0.0, w1);
    d = min(d, land1);

    // Center Island
    float w2 = 0.03 * exp(-30.0 * pow(uv.x - aspect*0.52, 2.0));
    float c2 = 0.48;
    // Taper Cutoff
    float land2 = abs(uv.y - c2) - w2 + smoothstep(0.008, 0.0, w2);
    d = min(d, land2);
    
    return d;
}

float getTrees(float2 uv, float aspect) {
    float d = 1.0;
    d = min(d, sdTree(uv, float2(aspect*0.05, 0.41), 0.18));
    d = min(d, sdTree(uv, float2(aspect*0.11, 0.40), 0.22));
    d = min(d, sdTree(uv, float2(aspect*0.17, 0.38), 0.14));
    d = min(d, sdTree(uv, float2(aspect*0.23, 0.36), 0.09));
    d = min(d, sdTree(uv, float2(aspect*0.48, 0.508), 0.07));
    d = min(d, sdTree(uv, float2(aspect*0.51, 0.51), 0.10));
    d = min(d, sdTree(uv, float2(aspect*0.54, 0.50), 0.06));
    return d;
}


float4 getPaintedTrees(float2 uv, float2 r, float time, float aspect) {
    float sm = 3.0 / r.y;
    float treeDist = getTrees(uv, aspect);
    
    float treeMask = smoothstep(0.02, -0.01, treeDist); 
    float coreMask = smoothstep(0.003, 0.0, treeDist);
    
    if (treeMask <= 0.0) return ((float4)(0.0));
    
    float3 baseCol = float3(0.05, 0.1, 0.08); 
    float4 C = float4(baseCol, coreMask); 
    
    float xd = 160.0;
    float yd = 45.0;
    
    for (float i = 0.0; i < 3.0; i++) {
        float2 u = uv;
        u.x -= sin(time * 1.5 + uv.x * 10.0) * 0.02 * max(0.0, uv.y - 0.4);
        u += noise(u * 12.0 + float2(time * 0.3, i * 5.0)) * 0.025; 
        
        float2 t = u * float2(xd, yd) + float2(i * 11.3, i * 17.7);
        
        float2 id = floor(t);
        float2 lc = frac(t);
        float h = hash12(id);
        
        if (h < 0.4) continue;
        
        float taper = lerp(1.2, 0.2, lc.y);
        
        float outline = st(abs(lc.x - 0.5), 0.35 * taper, sm * xd) * st(abs(lc.y - 0.5), 0.48, sm * yd);
        float body = st(abs(lc.x - 0.5), 0.22 * taper, sm * xd) * st(abs(lc.y - 0.5), 0.4, sm * yd);
        
        float3 strokeCol = lerp(float3(0.05, 0.25, 0.1), float3(0.15, 0.4, 0.15), h);
        if (h > 0.8) strokeCol = lerp(strokeCol, float3(0.3, 0.5, 0.15), 0.6); 
        if (h < 0.55) strokeCol = lerp(strokeCol, float3(0.0, 0.15, 0.2), 0.7); 
        
        float strokeAlpha = outline * treeMask;
        
        C.rgb = lerp(C.rgb, float3(0.02, 0.04, 0.03), strokeAlpha); 
        C.rgb = lerp(C.rgb, strokeCol, body * treeMask);          
        C.a = max(C.a, strokeAlpha);                             
    }
    return C;
}

float4 getIslandGrass(float2 uv, float2 r, float time, float aspect) {
    float sm = 3.0 / r.y;
    float4 O = ((float4)(0.0));
    
    float2 u = uv + noise(uv*2.0)*0.05; 
    float xd = 160.0; 
    u = u * float2(xd, xd/2.0); 
    
    float3 f_col = lerp(float3(0.3, 0.65, 0.2), float3(0.5, 0.8, 0.2), sin(time*0.5)*0.5 + 0.5);
    float3 cRock = float3(0.08, 0.1, 0.12); 
    
    for (float y = 0.0; y > -4.0; y--) { 
        for (float x = -2.0; x < 3.0; x++) {
            float2 id = floor(u) + float2(x,y);
            float2 root_uv = id / float2(xd, xd/2.0);
            
            // Re-apply the taper cutoff logic here so grass respects the boundary
            float w1 = 0.05 * exp(-12.0 * pow(root_uv.x - aspect*0.1, 2.0)); 
            float c1 = 0.36 - 0.01 * root_uv.x;
            float land1 = abs(root_uv.y - c1) - w1 + smoothstep(0.008, 0.0, w1);

            float w2 = 0.03 * exp(-30.0 * pow(root_uv.x - aspect*0.52, 2.0));
            float c2 = 0.48;
            float land2 = abs(root_uv.y - c2) - w2 + smoothstep(0.008, 0.0, w2);
            
            float landDist = min(land1, land2);
            float rootAlpha = smoothstep(0.01, -0.01, landDist);
            
            if (rootAlpha <= 0.0) continue;

            float2 lc = (frac(u) + float2(1.0-x, -y)) / float2(5.0, 3.0);
            float h = (hash12(id) - 0.5) * 0.25 + 0.5;

            lc -= float2(0.3, 0.5 - h*0.4);
            lc.x += sin(((time*1.5 + h*2.0 - id.x*0.05 - id.y*0.05)*1.1 + id.y*0.5)*2.0) * (lc.y + 0.5) * 0.25;
            
            float2 t = abs(lc) - float2(0.02, 0.5 - h*0.5); 
            float l = length(max(t, 0.0)) + min(max(t.x, t.y), 0.0);

            l -= noise(lc*7.0 + id) * 0.1; 
            
            float4 C = float4(cRock, st(l, 0.1, sm*xd*0.09)); 
            float4 fg = float4(f_col * (1.2 + lc.y*2.0) * (1.8 - h*2.5), 1.0);
            C = lerp(C, fg, st(l, 0.04, sm*xd*0.09));
            
            C.a *= rootAlpha; 
            O = lerp(O, C, C.a);
        }
    }
    return O;
}

void mainImage(out float4 fragColor, in float2 fragCoord) {
    float2 uv = fragCoord/iResolution.xy;
    float aspect = iResolution.x/iResolution.y;
    uv.x *= aspect; 

    // 1. Background (Painted Sky & Mountains)
    float3 col = getPaintedSkyAndMountains(fragCoord, iResolution.xy, iTime);

    // 2. Midground/Foreground (Painted Water & Sand)
    col = getPaintedWaterAndSand(fragCoord, iResolution.xy, iTime, col);

    // 3. Composite Land Base
    float landDist = getLand(uv, aspect);
    float landMask = smoothstep(0.003, 0.0, landDist); 
    float3 cRock = float3(0.08, 0.1, 0.12);
    col = lerp(col, cRock, landMask);

    // 4. Composite Painted Trees First
    float4 trees = getPaintedTrees(uv, iResolution.xy, iTime, aspect);
    col = lerp(col, trees.rgb, trees.a);

    // 5. Composite Full-Coverage Island Grass (Overlaps the trees to plant them)
    float4 grass = getIslandGrass(uv, iResolution.xy, iTime, aspect);
    col = lerp(col, grass.rgb, grass.a);

    fragColor = float4(col, 1.0);
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
