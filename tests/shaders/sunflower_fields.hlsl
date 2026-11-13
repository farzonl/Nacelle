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

// Sunflower Fields
// By Bitless, Ircss, & Noztol

#define p(t, a, b, c, d) ( a + b*cos( 6.28318*(c*t+d) ) ) 
#define sp(t) p(t,float3(.26,.76,.77),float3(1,.3,1),float3(.8,.4,.7),float3(0,.12,.54)) 
#define hue(v) ( .6 + .76 * cos(6.3*(v) + float4(0,23,21,0) ) ) 

#define smoothing      0.006
#define TWO_PI         6.28318530718
#define lineSize       0.01

#define MountainLayerThreecol float3(26., 65., 74.)/255.
#define MountainLayerFourCol float3(14., 49., 55.)/255.
#define SunflowerInsideOne    float3(203., 77., 23.)/255.
#define SunflowerInsideTwo    float3(134., 71., 48.)/255.
#define SunflowerInsideThree  float3(158., 159., 33.)/255.
#define SunflowerLeavesOne    float3(247., 214., 0.)/255.
#define SunflowerHighlight    float3(247., 218., 63.)/255.
#define SunflowerLeavesTwo    float3(236., 168., 3.)/255.
#define SunflowerStem         float3(97., 128., 52.)/255.
#define SunflowerStemBright   float3(176., 186., 53.)/255.
#define FieldDark             float3(44., 62., 40.)/255.
#define FieldMid              float3(94., 121., 62.)/255.

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

float fbm(float2 p) {
    float v = 0.0; float a = 0.5; float2 shift = ((float2)(100.0));
    float2x2 rot = float2x2(cos(0.5), sin(0.5), -sin(0.5), cos(0.50));
    for (int i = 0; i < 5; ++i) { v += a * s_noise(p); p = mul(p, rot) * 2.0 + shift; a *= 0.5; }
    return v;
}


float randOneD(float x) { return frac(sin(x *52.163)*268.2156); }


void DrawHalfVectorWithLength(float2 origin, float2 vector, float len, float2 uv, float size, float3 lineColor, inout float3 sceneColor){
    uv  -= origin;
    float v2   = dot(vector, vector);
    float vUv  = dot(vector, uv);
    float2  p    = vector * vUv/v2;
    float d    = distance(p, uv);
    float m    = 1. - step(0.,vUv/v2);
          m   += step(len, vUv/v2);
    sceneColor = lerp(lineColor, sceneColor, clamp(smoothstep(size, size + 0.01, d)+ m, 0. ,1.)); 
}

void DrawStemLeave(float2 origin, float2 vector, float len, float2 uv, float size, float3 lineColor, inout float3 sceneColor){
    uv  -= origin;
    float v2   = dot(vector, vector);
    float vUv  = dot(vector, uv);
    uv.y += pow(vUv/len, 2.)*4.;
    float2  p    = vector * vUv/v2;
    float d    = distance(p, uv);
    float m    = 1. - step(0.,vUv/v2);
          m   += step(len, vUv/v2);
          size *= smoothstep(0.5, 0.0, abs(vUv - 0.5)/len) *0.5;
    sceneColor = lerp(lineColor, sceneColor, clamp(smoothstep(size, size + 0.01, d)+ m, 0. ,1.)); 
}

void DrawPetals(float2 uv, inout float3 col, float seed, float offset, float3 petalColor) {
     float leavesSpread = 0.35;
     float2  petalSpace = float2(frac((offset+ uv.x)* TWO_PI *leavesSpread), uv.y);
     float petalSpaceID = floor((uv.x+offset)* TWO_PI * leavesSpread);
     float petalLength = 1.;
     float petalThickness = smoothstep(-0.01, 1., pow( (1.-(petalSpace.y / petalLength)), 0.85)) *0.9 - smoothstep(0.7,0., petalSpace.y / petalLength);
     petalSpace.x += sin((petalSpace.y + randOneD(petalSpaceID + seed) * TWO_PI )*4. )*0.3 * smoothstep(0.5, 1., (petalSpace.y / petalLength));                  
     DrawHalfVectorWithLength(float2(0.5, 0.), float2(0.,1.), 1., petalSpace, petalThickness,  petalColor, col);
}

void DrawSunFlower(float2 uv, float seed, inout float3 col, float mask) {
    float2 stroke_warp = float2(s_noise(uv*20.0), s_noise(uv*20.0 + 10.0)) * 0.08;
    float2 w_uv = uv + stroke_warp; 
    float paintStroke = fbm(uv * 25.0) * 0.25; 

    float2 stem_uv = w_uv;
    
    // Calculate the sine wave ripple moving up the stem
    float flowAmount = sin(stem_uv.y * 1.5 - iTime * 2.0 + seed * TWO_PI) * 0.2;
    // Anchor the root (at y = -7.0) so it doesn't move, maximum sway at top (y = 0.0)
    float bendFactor = smoothstep(-7.0, 0.0, stem_uv.y);
    stem_uv.x -= flowAmount * bendFactor;

    // Original Stem & Leaves (now automatically curved by the warped space)
    DrawHalfVectorWithLength(((float2)(0.)), float2(0.,-1.), 7., stem_uv, 0.15, SunflowerStem + paintStroke, col);
    DrawStemLeave(float2(0.,-2.+ randOneD(seed+5.125) *-2.), normalize(float2(max(0.2,randOneD(seed+712.125)),(randOneD(seed+81.215) -0.3) * 0.3)), 5., stem_uv, 0.3 +randOneD(seed+12.125) *0.4 , SunflowerStemBright + paintStroke, col);
    DrawStemLeave(float2(0.,-3.+ randOneD(seed+61.125) *-2.), normalize(float2(-1.0 * max(0.2,randOneD(seed+4.25)),(randOneD(seed+73.25) -0.3) * 0.3)), 5., stem_uv, 0.3 + randOneD(seed+0.125) * 0.4, SunflowerStem + paintStroke, col);
    
    float2 head_uv = w_uv;
    
    // Calculate exact displacement of the stem tip (at y = 0.0) so head perfectly connects
    float headFlow = sin(0.0 * 1.5 - iTime * 2.0 + seed * TWO_PI) * 0.2;
    float headBend = smoothstep(-7.0, 0.0, 0.0); // equals 1.0
    head_uv.x -= headFlow * headBend;

    // Convert to polar coordinates
    head_uv = float2(atan2(head_uv.y, head_uv.x), length(head_uv) * 0.55);
    
    // Slowly rotate the petals counter-clockwise
    float spinSpeed = 0.15 + (randOneD(seed) * 0.1); 
    head_uv.x -= iTime * spinSpeed; 
    
    float3 DrawnFlower= col;
    
    // Original Petals
    DrawPetals(head_uv, DrawnFlower, 53.126 + seed, +0.4 + randOneD(seed), SunflowerLeavesTwo + paintStroke);
    DrawPetals(head_uv, DrawnFlower, 0. + seed, randOneD(seed+7.125)*-0.5, SunflowerLeavesOne + paintStroke);

    // Segmented Spiral Center
    float centerMask = smoothstep(lineSize, lineSize + smoothing, 0.45 - head_uv.y);
    
    if (centerMask > 0.0) {
        float yd = 20.0; 
        float r_id = floor((head_uv.y + 0.01) * yd); 
        float xd = max(4.0, floor(r_id * TWO_PI * 0.8)); 
        float spiralAngle = head_uv.x + head_uv.y * 5.0 + seed * 10.0;
        float seg_x = (spiralAngle / TWO_PI) * xd;
        float seg_y = (head_uv.y + 0.01) * yd;
        
        float2 cell_id = float2(floor(seg_x), floor(seg_y));
        float2 lc = float2(frac(seg_x), frac(seg_y));
        float n = s_noise(cell_id * 5.0 + seed);
        lc += (float2(s_noise(cell_id*3.0), s_noise(cell_id*7.0)) - 0.5) * 0.4;
        
        float dashMask = st(abs(lc.x - 0.5), 0.4, 0.08) * st(abs(lc.y - 0.5), 0.4, 0.08);
        
        float3 midRing = lerp(SunflowerInsideTwo, SunflowerInsideThree, n);
        float3 centerBgCol = lerp(SunflowerInsideOne, midRing, smoothstep(0.45, 0.20, head_uv.y));
        centerBgCol = lerp(centerBgCol, SunflowerInsideThree, smoothstep(0.12, 0.0, head_uv.y));
        
        float3 strokeCol = centerBgCol * (1.1 + n * 0.4); 
        float3 gapCol = centerBgCol * 0.4;                
        
        float3 finalCenterCol = lerp(gapCol, strokeCol, dashMask);
        DrawnFlower = lerp(DrawnFlower, finalCenterCol, centerMask);
    }
    
    col = lerp(col, DrawnFlower, mask);
}

void DrawSunFlowerField(float2 OG_UV, float seed, float2 offset, float fieldMask, float totalMovementSpeed, inout float3 col,float tiling) {
     OG_UV += offset; OG_UV.x += iTime *totalMovementSpeed;
     float2  flowerRepeatedSpace = float2(frac(OG_UV.x*tiling), OG_UV.y*tiling);
     float2  idFlowerCoord = float2(floor(OG_UV.x*tiling), seed*21.215);
     flowerRepeatedSpace -= ((float2)(0.5)) + float2(0.15,0.5) *(randOneD (dot(idFlowerCoord , float2(1.126, 26.6))) - 0.5) ;
     flowerRepeatedSpace *= 4. + 0.2 *(randOneD (dot(idFlowerCoord , float2(8.136, 5.316))) - 0.5);
     DrawSunFlower(flowerRepeatedSpace, randOneD (dot(idFlowerCoord , float2(21.126, 8.3216))), col,fieldMask );
}

void mainImage(out float4 O, in float2 g) {
    float2 r = iResolution.xy;
    float2 uv = (g+g-r)/r.y;
    
    float2 sun_pos = float2(r.x/r.y * 0.45, -.45); 
    float2 tree_pos = float2(-r.x/r.y * 0.2, -.2); 
    
    float2 sh, u, id, lc, t;
    float3 f = ((float3)(0)), c;
    float xd, yd, h, a=0.0, l;
    float4 C;
    float sm = 3./r.y; 

    sh = rotate2D(sun_pos, noise(uv+iTime*.25)*.3); 
    if (uv.y > -.8) 
    {
        u = uv + sh;
        yd = 60.; 
        id =  float2((length(u)+.01)*yd,0); 
        xd = floor(id.x)*.09; 
        h = (hash12(floor(id.xx))*.5+.25)*(iTime+10.)*.25; 
        t = rotate2D (u,h); 
    
        id.y = atan2(t.y,t.x)*xd;
        lc = frac(id); 
        id -= lc;
    
        t = float2(cos((id.y+.5)/xd)*(id.x+.5)/yd,sin((id.y+.5)/xd)*(id.x+.5)/yd); 
        t = rotate2D(t,-h) - sh;
    
        h = noise(t*float2(.5,1)-float2(iTime*.2,0)) * step(-.25,t.y); 
        h = smoothstep (.052,.055, h);
        
        lc += (noise(lc*float2(1,4)+id))*float2(.7,.2); 
        
        f = lerp (sp(sin(length(u)-.1))*.35, 
                 lerp(sp(sin(length(u)-.1)+(hash12(id)-.5)*.15),((float3)(1)),h), 
                 st(abs(lc.x-.5),.4,sm*yd)*st(abs(lc.y-.5),.48,sm*xd));
                 
        float2 sun_uv = uv - sun_pos;
        float2 spun_uv = rotate2D(sun_uv, -iTime * 0.5); 
        
        float dToSun = length(sun_uv);
        float angle = atan2(spun_uv.y, spun_uv.x);
        
        float rays = sin(angle * 12.0 + s_noise(spun_uv*15.0)*2.0) * 0.03;
        float sunCore = smoothstep(0.12, 0.02, dToSun);
        float sunCorona = smoothstep(0.35, 0.1, dToSun + rays) * s_noise(spun_uv * 20.0);
        
        float3 sunColor = lerp(float3(0.9, 0.4, 0.0), float3(1.0, 1.0, 0.2), smoothstep(0.15, 0.0, dToSun));
        
        float swirl = smoothstep(0.0, 0.8, sin(dToSun * 40.0 + angle * 4.0));
        sunColor += swirl * 0.3 * sunCore;
        
        f = lerp(f, sunColor, clamp(sunCore + sunCorona, 0.0, 1.0));
    };

    // mountains
    if (uv.y < 0.25)
    {
        float cld = noise(-sh*float2(.5,1)  - float2(iTime*.2,0)); 
        cld = 1.- smoothstep(.0,.15,cld)*.5;

        u = (uv - float2(0.0, 0.25)) * float2(1, 15); 
        id = floor(u);

        for (float i = 1.; i > -1.; i--)
        {
            if (id.y+i < 0.0)
            {
                lc = frac(u)-.5;
                lc.y = (lc.y+(sin(uv.x*8.-iTime*0.8+id.y+i))*.3-i)*4.; 
                h = hash12(float2(id.y+i,floor(lc.y))); 
                
                xd = 6.+h*4.;
                yd = 30.;
                lc.x = uv.x*xd+sh.x*9.; 
                lc.x += sin(iTime * (.2 + h*1.5))*.5; 
                h = .8*smoothstep(5.,.0,abs(floor(lc.x)))*cld+.1; 
                
                float3 mtnBase = MountainLayerFourCol;
                float3 mtnHigh = MountainLayerThreecol;
                f = lerp(f,lerp(mtnBase,mtnHigh,h),st(lc.y,0.,sm*yd)); 
                lc += noise(lc*float2(3,.5))*float2(.1,.6); 
                
                float3 strokeCol = hue(hash12(floor(lc))*.1+.35).rgb*(1.2+floor(lc.y)*.17);
                f = lerp(f, 
                    lerp(strokeCol,FieldMid,h),
                    st(lc.y,0.,sm*xd)
                    *st(abs(frac(lc.x)-.5),.48,sm*xd)*st(abs(frac(lc.y)-.5),.3,sm*yd)
                );
            }
        }
    }


    float2 uv_sun = g/r; uv_sun -= 0.5; uv_sun.x *= r.x/r.y; uv_sun *= 5.; 
    float totalMovementSpeed = 0.05; float movement = iTime * totalMovementSpeed;

    float fieldMask = smoothstep(0.05, 0.15, 0.1 - uv.y); 
    float3 fieldBaseColor = lerp(FieldMid + pow(s_noise((uv_sun + float2(movement * 4.7,0.))*20.0), 2.)*0.05,
                              SunflowerLeavesOne + pow(s_noise((uv_sun + float2(movement*1.,0.))*20.0), 2.)*0.05, smoothstep(-0.2, 0.0, uv_sun.y));
    fieldBaseColor = lerp(fieldBaseColor, FieldDark + pow(s_noise((uv_sun + float2(movement*6.2,0.))*20.0), 2.)*0.05, smoothstep(-0.6, -1.4, uv_sun.y));
    f = lerp(f, fieldBaseColor, fieldMask);
    
    O = float4(f, 1.0); 

    // Sunflowers (Background & Midground)
    float2 OG_UV = g/r.xy; OG_UV.x *= r.x/r.y;
    float fm = smoothstep(0.01, 0.05, 0.1 - uv_sun.y); 

    if (uv.y < 0.1) {
        float3 tCol = O.rgb;
        DrawSunFlowerField(OG_UV, 0., float2(0.51,-0.48),    fm, totalMovementSpeed, tCol, 90.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 6.621, float2(0.51,-0.46), fm,  totalMovementSpeed, tCol, 50.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 7.23, float2(0.51,-0.43),  fm,  totalMovementSpeed, tCol, 29.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 12.6, float2(0.51,-0.4),   fm,  totalMovementSpeed, tCol, 22.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, -7.21, float2(0.51,-0.35), fm,  totalMovementSpeed, tCol, 15.);
        totalMovementSpeed *= 1.1; DrawSunFlowerField(OG_UV, 2.125, float2(0.51,-0.3),  fm,  totalMovementSpeed, tCol, 12.);
        O.rgb = tCol; 
    }

    // cypress tree
    float T = sin(iTime*.5); 

    if (abs(uv.x+tree_pos.x-.1-T*.1) < .6) {
        u = uv + tree_pos; u.x -= sin(u.y+1.)*.2*(T+.75); u += noise(u*4.5-7.)*.25; 
        xd = 10., yd = 60.; t = u * float2(1,yd); h = hash12(floor(t.yy)); t.x += h*.01; t.x *= xd; lc = frac(t); 
        float m = st(abs(t.x-.5),.5,sm*xd)*step(abs(t.y+20.),45.); 
        C = lerp(((float4)(.07)), float4(.5,.3,0,1)*(.4+h*.4), st(abs(lc.y-.5),.4,sm*yd)*st(abs(lc.x-.5),.45,sm*xd)); C.a = m;
        xd = 30., yd = 15.;
        for (float xs =0.;xs<4.;xs++) {
            u = uv + tree_pos + float2 (xs/xd*.5 -(T +.75)*.15,-.7); 
            u += noise(u*float2(2,1)+float2(-iTime+xs*.05,0))*float2(-.25,.1)*smoothstep (.5,-1.,u.y+.7)*.75; 
            t = u * float2(xd,1.); h = hash12(floor(t.xx)+xs*1.4); yd = 5.+ h*7.; t.y *= yd; sh = t; lc = frac(t); h = hash12(t-lc); 
            t = (t-lc)/float2(xd,yd)+float2(0,.7);
            m = (step(0.,t.y)*step (length(t),.45) + step (t.y,0.)*step (-0.7+sin((floor(u.x)+xs*.5)*15.)*.2,t.y)) *step (abs(t.x),.5) *st(abs(lc.x-.5),.35,sm*xd*.5); 
            lc += noise((sh)*float2(1.,3.))*float2(.3,.3); f = hue((h+(sin(iTime*.2)*.5+.5))*.2).rgb-t.x; 
            C = lerp(C, float4(lerp(f*.15,f*.6*(.7+xs*.2), st(abs(lc.y-.5),.47,sm*yd)*st(abs(lc.x-.5),.2,sm*xd)),m), m);
        }
        O = lerp (O,C,C.a); 
    }
    
    // Sunflowers (Foreground)
    if (uv.y < 0.1) {
        float3 tCol = O.rgb;
        totalMovementSpeed *= 1.2; DrawSunFlowerField(OG_UV, 1., float2(1.126,-0.2),    fm,  totalMovementSpeed, tCol, 8.);
        totalMovementSpeed *= 1.2; DrawSunFlowerField(OG_UV, 5., float2(0.,-0.05),      fm,  totalMovementSpeed, tCol, 4.);
        totalMovementSpeed *= 1.3; DrawSunFlowerField(OG_UV, 71.612, float2(0.,0.1),    fm,  totalMovementSpeed, tCol, 3.);
        O.rgb = tCol; 
    }
    
    float global_canvas = fbm(uv * 100.0) * 0.05;
    O.rgb -= global_canvas;
}
[RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0, visibility=SHADER_VISIBILITY_PIXEL)")]
float4 PSMain(float4 pos : SV_Position) : SV_Target {
    float4 fragColor = float4(0.0, 0.0, 0.0, 1.0);
    float2 fragCoord = float2(pos.x, iResolution.y - pos.y);
    mainImage(fragColor, fragCoord);
    return fragColor;
}
