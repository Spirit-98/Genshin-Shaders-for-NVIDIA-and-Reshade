/*=============================================================================

   Copyright (c) Pascal Gilcher. All rights reserved.

	ReShade effect file
    github.com/martymcmodding

	Support me:
   		patreon.com/mcflypg

    ReLight

    0.1 - initial release
    0.2 - fix black splotches bug
        - allow negative bump strength in case height from luma is wrong
        - add temporal supersampling
        - add ReSTIR based sampling
        - automatically hide overlay when screenshot is taken

    * Unauthorized copying of this file, via any medium is strictly prohibited
 	* Proprietary and confidential
    * See accompanying license document for terms and conditions

=============================================================================*/


/*=============================================================================
	Preprocessor settings
=============================================================================*/

#ifndef AMOUNT_OF_LIGHTS
 #define AMOUNT_OF_LIGHTS 2
#endif

/*=============================================================================
	UI Uniforms
=============================================================================*/

uniform bool RT_USE_SMOOTHNORMALS <
    ui_label = "Use Smooth Normals";
    ui_category = "Global";
> = true;

uniform float NORMAL_BUMP <
	ui_type = "drag";
    ui_label = "Relief Strength";
	ui_min = -1.0; ui_max = 1.0;
    ui_category = "Global";
> = 0.2;

uniform bool USE_SHADOWS <
    ui_label = "Trace Shadows";
    ui_category = "Shadows";
> = true;

uniform bool USE_SHADOW_FILTERING <
    ui_label = "Filter Shadows";
    ui_category = "Shadows";
> = true;

uniform bool USE_TEMPORAL_SS <
    ui_label = "Use Temporal Supersampling";
    ui_category = "Shadows";
> = true;

uniform int SHADOW_MODE <
	ui_type = "combo";
    ui_label = "Shadow Mode";
	ui_items = "Trace All (slow, accurate)\0ReSTIR (faster, noisy)\0";
    ui_tooltip = "ReSTIR (Spatiotemporal Reservoir Resampling) is a sophisticated\nheuristic for efficient ray tracing with many light sources.\n\nWith ReSTIR enabled, only a single light is traced per pixel,\ngreatly improving performance with many lights at the cost of some noise.\n\nBest combined with higher shadow quality and temporal supersampling!";	
    ui_category = "Shadows";
> = 0;

uniform int SHADOWS_QUALITY <
	ui_type = "combo";
    ui_label = "Shadow Quality";
	ui_items = "Low\0Medium\0High\0Ultra\0";	
    ui_category = "Shadows";
> = 2;

uniform float SHADOWS_SHARPNESS <
	ui_type = "drag";
    ui_label = "Shadow Sharpness";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Shadows";
> = 0.2;

uniform float SHADOWS_OBJ_THICKNESS <
	ui_type = "drag";
    ui_label = "Z Thickness";
	ui_min = 0.0; ui_max = 10.0;
    ui_category = "Shadows";
> = 4.0;

uniform bool LIGHT0_ENABLE <
    ui_label = "Active";
    ui_category = "Light 0";
> = true;

uniform float3 LIGHT0_POS <
	ui_type = "drag";
    ui_label = "Position";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 0";
> = float3(0.45, 0.7, 0.2);

uniform float3 LIGHT0_COL <
  	ui_type = "color";
  	ui_label="Tint";
  	ui_category = "Light 0";
> = float3(1.0, 0.93, 0.82);

uniform float LIGHT0_INT <
	ui_type = "drag";
    ui_label = "Intensity";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 0";
> = 1.0;

#if AMOUNT_OF_LIGHTS > 1
uniform bool LIGHT1_ENABLE <
    ui_label = "Active";
    ui_category = "Light 1";
> = false;

uniform float3 LIGHT1_POS <
	ui_type = "drag";
    ui_label = "Position";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 1";
> = float3(0.7, 0.7, 0.2);

uniform float3 LIGHT1_COL <
  	ui_type = "color";
  	ui_label="Tint";
  	ui_category = "Light 1";
> = float3(0.95, 0.988, 1.0);

uniform float LIGHT1_INT <
	ui_type = "drag";
    ui_label = "Intensity";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 1";
> = 0.3;
#endif

#if AMOUNT_OF_LIGHTS > 2
uniform bool LIGHT2_ENABLE <
    ui_label = "Active";
    ui_category = "Light 2";
> = false;

uniform float3 LIGHT2_POS <
	ui_type = "drag";
    ui_label = "Position";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 2";
> = float3(0.7, 0.7, 0.1);

uniform float3 LIGHT2_COL <
  	ui_type = "color";
  	ui_label="Tint";
  	ui_category = "Light 2";
> = float3(1.0, 1.0, 1.0);

uniform float LIGHT2_INT <
	ui_type = "drag";
    ui_label = "Intensity";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 2";
> = 0.2;
#endif

#if AMOUNT_OF_LIGHTS > 3
uniform bool LIGHT3_ENABLE <
    ui_label = "Active";
    ui_category = "Light 3";
> = false;

uniform float3 LIGHT3_POS <
	ui_type = "drag";
    ui_label = "Position";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 3";
> = float3(0.7, 0.2, 0.1);

uniform float3 LIGHT3_COL <
  	ui_type = "color";
  	ui_label="Tint";
  	ui_category = "Light 3";
> = float3(1.0, 1.0, 1.0);

uniform float LIGHT3_INT <
	ui_type = "drag";
    ui_label = "Intensity";
	ui_min = 0.0; ui_max = 1.0;
    ui_category = "Light 3";
> = 0.2;
#endif

uniform bool RT_SHOW_DEBUG <
    ui_label = "Visualize Light Position";
    ui_tooltip = "Draws little spheres at the light positions.\nDon't worry - they're automatically hidden for screenshots ;)";
    ui_category = "Debug";
> = true;
 /*
uniform float4 tempF1 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF2 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF3 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF4 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);

uniform float4 tempF5 <
    ui_type = "drag";
    ui_min = -100.0;
    ui_max = 100.0;
> = float4(1,1,1,1);
 */
/*=============================================================================
	Textures, Samplers, Globals
=============================================================================*/

uniform bool OVERLAY_OPEN < source = "overlay_open"; >;
uniform uint  FRAMECOUNT  < source = "framecount"; >;
uniform bool SCREENSHOT < source = "screenshot"; >;

uniform float TIMER < source = "timer"; >;

texture ColorInputTex : COLOR;
texture DepthInputTex : DEPTH;
sampler ColorInput 	            { Texture = ColorInputTex; };
sampler DepthInput              { Texture = DepthInputTex; }; 

#define MAX_LIGHT_LUMINANCE     512.0
#define POINT_LIGHT_RADIUS      0.3

texture GBTexRaw           { Width = BUFFER_WIDTH;         Height = BUFFER_HEIGHT;     Format = RGBA16F; };
sampler sGBTexRaw           { Texture = GBTexRaw;      };
texture GBTex                  { Width = BUFFER_WIDTH;         Height = BUFFER_HEIGHT;     Format = RGBA16F; MipLevels = 3; };
sampler sGBTex              { Texture = GBTex;      };

texture GBTexPrev            { Width = BUFFER_WIDTH;         Height = BUFFER_HEIGHT;     Format = RGBA16F; };
sampler sGBTexPrev              { Texture = GBTexPrev;      };

texture ZTexLow              { Width = BUFFER_WIDTH/2;         Height = BUFFER_HEIGHT/2;     Format = R16F;   MipLevels = 4;   };
sampler sZTexLow	                { Texture = ZTexLow;    };

texture ReSTIRReservoir     { Width = BUFFER_WIDTH / 4;         Height = BUFFER_HEIGHT / 4;     Format = RGBA16F; };
sampler sReSTIRReservoir	{ Texture = ReSTIRReservoir;   };

texture ReLightCurr          { Width = BUFFER_WIDTH;         Height = BUFFER_HEIGHT;     Format = RGBA16F; };
sampler sReLightCurr	     { Texture = ReLightCurr;   };
texture ReLightTmp           { Width = BUFFER_WIDTH;         Height = BUFFER_HEIGHT;     Format = RGBA16F; };
sampler sReLightTmp	         { Texture = ReLightTmp;   };
texture ReLightPrev          { Width = BUFFER_WIDTH;         Height = BUFFER_HEIGHT;     Format = RGBA16F; };
sampler sReLightPrev	     { Texture = ReLightPrev;   };

texture texMotionVectors          { Width = BUFFER_WIDTH;   Height = BUFFER_HEIGHT;   Format = RG16F; };
sampler sMotionVectorTex         { Texture = texMotionVectors;  };

texture JitterTexHi       < source = "bluenoisehi.png"; > { Width = 1024; Height = 1024; Format = RGBA8; };
sampler	sJitterTexHi      { Texture = JitterTexHi; AddressU = WRAP; AddressV = WRAP; };

struct VSOUT
{
	float4 vpos : SV_Position;
    float2 uv   : TEXCOORD0;
};

#include "qUINT\Global.fxh"
#include "qUINT\Depth.fxh"
#include "qUINT\Projection.fxh"
#include "qUINT\Normal.fxh"

/*=============================================================================
	Functions
=============================================================================*/

float3 srgb_to_acescg(float3 srgb)
{
    float3x3 m = float3x3(  0.613097, 0.339523, 0.047379,
                            0.070194, 0.916354, 0.013452,
                            0.020616, 0.109570, 0.869815);
    return mul(m, srgb);           
}

float3 acescg_to_srgb(float3 acescg)
{     
    float3x3 m = float3x3(  1.704859, -0.621715, -0.083299,
                            -0.130078,  1.140734, -0.010560,
                            -0.023964, -0.128975,  1.153013);                 
    return mul(m, acescg);            
}

float3 unpack_hdr(float3 color)
{
    color  = saturate(color);
    color *= color;    
    color = srgb_to_acescg(color);
    color = color * rcp(1.04 - saturate(color));   
    
    return color;
}

float3 pack_hdr(float3 color)
{
    color =  1.04 * color * rcp(color + 1.0);   
    color = acescg_to_srgb(color);    
    color  = saturate(color);    
    color = sqrt(color);   
    return color;     
}

float2 ray_sphere_intersection(float3 ray_origin, float3 ray_dir, float3 sphere_center, float sphere_radius)
{
    float3 delta_vec = ray_origin - sphere_center;
    float b = dot(delta_vec, ray_dir);
    float c = dot(delta_vec, delta_vec) - sphere_radius * sphere_radius;
    float h = b * b - c;
    float2 ret =  h < 0 ? 100000000 : -b + float2(-1, 1) * sqrt(h);
    ret = ret < 0.0.xx ? 100000000 : ret;
    return ret;
}

struct Light
{
    float3 pos;
    float intensity;
    float3 tint;
    float wrap;
};

void setup_lights(inout Light lights[AMOUNT_OF_LIGHTS])
{
    lights[0].pos = LIGHT0_POS;
    lights[0].intensity = LIGHT0_INT * LIGHT0_ENABLE;
    lights[0].tint = LIGHT0_COL;
#if AMOUNT_OF_LIGHTS > 1
    lights[1].pos = LIGHT1_POS;
    //lights[1].pos.xy = float2(sin(TIMER * 0.001), sin(TIMER * 2.0 * 0.001)) * 0.4 + 0.5;
    lights[1].intensity = LIGHT1_INT * LIGHT1_ENABLE;
    lights[1].tint = LIGHT1_COL;
#endif
#if AMOUNT_OF_LIGHTS > 2
    lights[2].pos = LIGHT2_POS;
    lights[2].intensity = LIGHT2_INT * LIGHT2_ENABLE;
    lights[2].tint = LIGHT2_COL;
#endif
#if AMOUNT_OF_LIGHTS > 3
    lights[3].pos = LIGHT3_POS;
    lights[3].intensity = LIGHT3_INT * LIGHT3_ENABLE;
    lights[3].tint = LIGHT3_COL;
#endif  
}

float3 get_normal_from_color(float2 uv, float scale)
{
    const float3 lumc = float3(0.212656, 0.715158, 0.072186);
    const float3 offs = float3(BUFFER_PIXEL_SIZE, 0);

    float4 grey, depths; float3 t;
    t = tex2Dlod(ColorInput, uv - offs.xz, 0).rgb;
    grey.x = dot(t * t, lumc);
    t = tex2Dlod(ColorInput, uv + offs.xz, 0).rgb; 
    grey.y = dot(t * t, lumc);    
    t = tex2Dlod(ColorInput, uv - offs.zy, 0).rgb;
    grey.z = dot(t * t, lumc);
    t = tex2Dlod(ColorInput, uv + offs.zy, 0).rgb;
    grey.w = dot(t * t, lumc);

    depths.x = Depth::get_linear_depth(uv - offs.xz);
    depths.y = Depth::get_linear_depth(uv + offs.xz);
    depths.z = Depth::get_linear_depth(uv - offs.zy);
    depths.w = Depth::get_linear_depth(uv + offs.zy);

    float2 mask = (depths.xz - depths.yw) * RESHADE_DEPTH_LINEARIZATION_FAR_PLANE * 4;

    float3 n;
    n.xy = (grey.xz - grey.yw) * saturate(1 - mask) * scale * 4;
    n.xy /= dot(grey, 1) + 0.1;
    n.z = 1;
    return n * rsqrt(dot(n, n) + 1e-3);
}


float3 blend_normals(float3 n1, float3 n2)
{
    n1 += float3( 0, 0, 1);
    n2 *= float3(-1, -1, 1);
    n1.z = abs(n1.z) < 0.0001 ? 0.0001 : n1.z;//fix div 0
    return n1*dot(n1, n2)/n1.z - n2;
}

/*=============================================================================
	Shader entry points
=============================================================================*/

VSOUT VSMain(in uint id : SV_VertexID)
{
    VSOUT o;
    o.uv = id.xx == uint2(2, 1) ? 2.0.xx : 0.0.xx;
    o.vpos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

VSOUT VSShadows(in uint id : SV_VertexID)
{
    VSOUT o;
    o.uv = id.xx == uint2(2, 1) ? 2.0.xx : 0.0.xx;
    o.vpos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.vpos = !USE_SHADOWS ? 0 : o.vpos;
    return o;
}

VSOUT VSNoShadows(in uint id : SV_VertexID)
{
    VSOUT o;
    o.uv = id.xx == uint2(2, 1) ? 2.0.xx : 0.0.xx;
    o.vpos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    o.vpos = USE_SHADOWS ? 0 : o.vpos;
    return o;
}

void PS_MakeInput_Gbuf(in VSOUT i, out float4 gbuffer : SV_Target0)
{ 
    float depth = Depth::get_linear_depth(i.uv);
    float3 n = Normal::normal_from_depth(i.uv);    
    gbuffer = float4(n, Projection::depth_to_z(depth));     
}

void PS_Smoothnormals(in VSOUT i, out float4 gbuffer : SV_Target0)
{ 
    const float max_n_n = 0.63;
    const float max_v_s = 0.65;
    const float max_c_p = 0.5;
    const float searchsize = 0.0125;
    const int dirs = 5;

    float4 gbuf_center = tex2D(sGBTexRaw, i.uv);
    float3 n_sum = 0.001 * gbuf_center.xyz;

    [branch]
    if(RT_USE_SMOOTHNORMALS)
    {  
        float3 n_center = gbuf_center.xyz;
        float3 p_center = Projection::uv_to_proj(i.uv, gbuf_center.w);
        float radius = searchsize + searchsize * rcp(p_center.z) * 2.0;
        float worldradius = radius * p_center.z;

        int steps = clamp(ceil(radius * 300.0) + 1, 1, 7);        

        for(float j = 0; j < dirs; j++)
        {
            float2 dir; sincos(radians(360.0 * j / dirs + 0.666), dir.y, dir.x);

            float3 n_candidate = n_center;
            float3 p_prev = p_center;

            for(float stp = 1.0; stp <= steps; stp++)
            {
                float fi = stp / steps;   
                fi *= fi * rsqrt(fi);

                float offs = fi * radius;
                offs += length(BUFFER_PIXEL_SIZE);

                float2 uv = i.uv + dir * offs * BUFFER_ASPECT_RATIO;            
                if(!all(saturate(uv - uv*uv))) break;

                float4 gbuf = tex2Dlod(sGBTexRaw, uv, 0);
                float3 n = gbuf.xyz;
                float3 p = Projection::uv_to_proj(uv, gbuf.w);

                float3 v_increment  = normalize(p - p_prev);

                float ndotn         = dot(n, n_center); 
                float vdotn         = dot(v_increment, n_center); 
                float v2dotn        = dot(normalize(p - p_center), n_center); 
            
                ndotn *= max(0, 1.0 + fi * 0.5 * (1.0 - abs(v2dotn)));

                if(abs(vdotn)  > max_v_s || abs(v2dotn) > max_c_p) break;       

                if(ndotn > max_n_n)
                {
                    float d = distance(p, p_center) / worldradius;
                    float w = saturate(4.0 - 2.0 * d) * smoothstep(max_n_n, lerp(max_n_n, 1.0, 2), ndotn); //special recipe
                    w = stp < 1.5 && d < 2.0 ? 1 : w;  //special recipe       
                    n_candidate = lerp(n_candidate, n, w);
                    n_candidate = normalize(n_candidate);
                }

                p_prev = p;
                n_sum += n_candidate;
            }
        }

    }

    n_sum = normalize(n_sum);

    if(abs(NORMAL_BUMP) > 0.0001)
        n_sum = normalize(blend_normals(n_sum, get_normal_from_color(i.uv, NORMAL_BUMP)));
    gbuffer = float4(n_sum, gbuf_center.w);
}

void PSWriteZ(in VSOUT i, out float o : SV_Target0)
{
    float4 texels; //TODO: replace with gather()
    texels.x = Depth::get_linear_depth(i.uv + float2( 0.5, 0.5) * BUFFER_PIXEL_SIZE);
    texels.y = Depth::get_linear_depth(i.uv + float2(-0.5, 0.5) * BUFFER_PIXEL_SIZE);
    texels.z = Depth::get_linear_depth(i.uv + float2( 0.5,-0.5) * BUFFER_PIXEL_SIZE);
    texels.w = Depth::get_linear_depth(i.uv + float2(-0.5,-0.5) * BUFFER_PIXEL_SIZE);

    o = max(max(texels.x, texels.y), max(texels.z, texels.w));
    o = Projection::depth_to_z(o);
}

float3 get_pos(float2 uv, float t)
{
    float mip = max(0, log2(t) - 3);
    float z = tex2Dlod(sZTexLow, uv, mip).x;
    return Projection::uv_to_proj(uv, z);
}

struct RayDesc 
{
    float3 origin;
    float3 pos;
    float3 dir;
    float2 uv;
    float length;
    float width; //faux cone tracing
};

float3 get_jitter(uint2 texelpos, uint framecount)
{
    uint2 texel_in_tile = texelpos % 128u;
    uint frame = framecount % 64u;
    uint2 tile;  
    tile.x = frame % 8u;
    tile.y = frame / 8u;

    uint2 texturepos = tile * 128u + texel_in_tile;
    float3 jitter = tex2Dfetch(sJitterTexHi, texturepos).xyz;
    return jitter;
}

float4 ProcessLight(in VSOUT i, Light light, float quality_mult, bool trace_shadows)
{
    if(light.intensity < 0.001) return 0;

    float3 n = tex2Dlod(sGBTex, i.uv, 0).xyz;

    float3 p = Projection::uv_to_proj(i.uv);
    float  d = Projection::z_to_depth(p.z); 
    float p_dist = length(p);
    float3 e = p / p_dist;
    p *= 0.999;

    uint jitter_idx = dot(floor(i.vpos.xy) % 4, float2(1, 4)); 
    float jitter = (jitter_idx * 11u) % 16u; //prime shuffle, avoids duplicates
    jitter = (jitter + 0.5) * rcp(4 * 4);  

    if(USE_TEMPORAL_SS) jitter = get_jitter(i.vpos.xy, FRAMECOUNT).y;

    float4 lightsum = 0;

    float2 light_uv = float2(light.pos.x, 1 - light.pos.y);
    float3 light_pos = Projection::uv_to_proj(light_uv, Projection::depth_to_z(light.pos.z * light.pos.z * light.pos.z));//z cubed for easier adjustment
    float3 delta_vec = light_pos - p;
    float ldotl = dot(delta_vec, delta_vec);
    delta_vec *= rsqrt(ldotl);
    float ndotl = saturate(dot(n, delta_vec)); 

    float r2 = POINT_LIGHT_RADIUS * POINT_LIGHT_RADIUS;
    float fa = 2 * rcp(r2) * saturate(1 - sqrt(ldotl) * rsqrt(ldotl + r2));

    lightsum = float4(unpack_hdr(light.tint), 1) * fa * ndotl * light.intensity * light.intensity; //store unbiased intensity in A for ReSTIR!

    if(fa * ndotl < 0.00001) return 0;
    if(!trace_shadows) return lightsum * MAX_LIGHT_LUMINANCE;

    RayDesc ray;
    ray.dir = normalize(delta_vec);
    ray.length = 0;
    ray.origin = p;

    float2 delta_2D = light_uv - i.uv;
    float length_2D = length(delta_2D * BUFFER_ASPECT_RATIO.yx);
    float length_pixels = length(delta_2D * BUFFER_SCREEN_SIZE);
    float min_step = 2 / length_pixels;
    int steps = ceil(quality_mult * sqrt(length_2D)); 

    [loop]
    for(int j = 0; j < steps; j++)
    {
        float fi = saturate(float(j + jitter) / steps);
        fi *= sqrt(fi);
        fi += min_step;

        ray.uv = lerp(i.uv, light_uv, fi);
        float3 pos = get_pos(ray.uv, length(fi * BUFFER_ASPECT_RATIO * (ray.uv - i.uv))); 
        
        //trust me, I'm a polymath
        float t = rsqrt(dot(pos, pos));
        float cosA = dot(e, pos) * t;
        float cosB = dot(ray.dir, pos) * t;   

        ray.length = p_dist * sqrt((1 - cosA * cosA) / (1 - cosB * cosB));
        ray.pos = p + ray.dir * ray.length;

        float delta = dot(e, pos - ray.pos) * sign(cosA);
        
        float z_tolerance = SHADOWS_OBJ_THICKNESS * SHADOWS_OBJ_THICKNESS;
        z_tolerance *= ray.length + 1;

        [branch]
        if(abs(delta * 2.0 + z_tolerance + 0.05) < z_tolerance)
        {
            float ang = dot(normalize(pos - ray.origin), ray.dir);
            lightsum *= linearstep(SHADOWS_SHARPNESS, 1, 1 - acos(ang)) + 1e-2;   //lift a tiny bit so even in full shadow, weight for enabled lights is never truly 0, fixes blockyness in ReSTIR        
        }                     
    } 

    lightsum *= MAX_LIGHT_LUMINANCE;
    return lightsum;
}

void PS_ReSTIR_Initial(in VSOUT i, out float4 o : SV_Target0)
{
    if(!SHADOW_MODE) discard;

    Light lights[AMOUNT_OF_LIGHTS];
    setup_lights(lights);    

    const int quality_presets[4] = {16,32,48,80};
    float quality_mult = quality_presets[SHADOWS_QUALITY];

    o = 0;

    [unroll]
    for(int id = 0; id < AMOUNT_OF_LIGHTS; id++)
    {
        float4 t = ProcessLight(i, lights[id], quality_mult, true);
        o[id] = dot(t.xyz, float3(0.299, 0.587, 0.114));
    }
            
}

void PS_ReSTIR_Resampling(in VSOUT i, out float4 o : SV_Target0)
{   
    Light lights[AMOUNT_OF_LIGHTS];
    setup_lights(lights);

    int quality_presets[4] = {16,32,48,80};
    float quality_mult = quality_presets[SHADOWS_QUALITY];

    o = 0;

    [branch]
    if(SHADOW_MODE)
    {
        //No need for distributing samples spatially if we know the amount and can trace all of them
        //in lower res. So just sample all of them in low res, store weights in RGBA, get all potential reservoirs
        //in one single texture fetch, identify light by channel. Saves me pretty much 99% of ReSTIR's overhead
        float4 restir_weights = tex2D(sReSTIRReservoir, i.uv);

        float4 cumulative_weights = restir_weights.x;
        cumulative_weights.yzw += restir_weights.y;
        cumulative_weights.zw += restir_weights.z;
        cumulative_weights.w += restir_weights.w;
        cumulative_weights /= cumulative_weights.w + (cumulative_weights.w == 0 ? 1 : 0);
        cumulative_weights += float4(0,1,2,3) * 0.00001;

        float randv = get_jitter(i.vpos.xy, USE_TEMPORAL_SS ? FRAMECOUNT : 0).x;     

        //need to do it that way, otherwise the bracket select doesn't work right for some pixels   
        uint light_id_to_resample = randv < cumulative_weights.x ? 0 : 
                                    randv < cumulative_weights.y ? 1 :
                                    randv < cumulative_weights.z ? 2 :
                                    randv < cumulative_weights.w ? 3 :
                                    1337;     

        if(light_id_to_resample >= AMOUNT_OF_LIGHTS) { o = float4(0,0,0,1); return;} 
        light_id_to_resample %= AMOUNT_OF_LIGHTS; //just to be sure   

        o = ProcessLight(i, lights[light_id_to_resample], quality_mult, true);
        o *= dot(restir_weights, 1) / dot(restir_weights + 1e-3, light_id_to_resample.xxxx == uint4(0,1,2,3)); //scale weighting by relative weight
    }
    else 
    {
        [unroll]
        for(int id = 0; id < AMOUNT_OF_LIGHTS; id++)
            o += ProcessLight(i, lights[id], quality_mult, true);
    }   
}

void PS_TemporalBlend(in VSOUT i, out float4 o : SV_Target0)
{
    o = 0;

    float2 mv = tex2D(sMotionVectorTex, i.uv).xy;

    float4 lighting_prev = tex2D(sReLightPrev, i.uv + mv);

    float3 m1 = 0, m2 = 0;
    [unroll]for(int x = -2; x <= 2; x++)
    [unroll]for(int y = -2; y <= 2; y++)
    {
        float3 tv = tex2Doffset(sReLightCurr, i.uv, int2(x, y)).rgb;
        m1 += tv;
        m2 += tv * tv;
    }

    m1/=25;
    m2/=25;
    float3 std = sqrt(abs(m2 - m1 * m1));
    float4 lighting_curr = tex2D(sReLightCurr, i.uv);

    float3 difference_in_sigmas = abs(m1 - lighting_prev.rgb) / (std + 1);
    lighting_prev.rgb = clamp(lighting_prev.rgb, m1 - std, m1 + std); 

    float4 gbuf_curr = tex2D(sGBTex, i.uv);
    float4 gbuf_prev = tex2D(sGBTexPrev, i.uv + mv);

    float normal_weight = saturate(dot(normalize(gbuf_curr.xyz), normalize(gbuf_prev.xyz)) * 20 - 20 + 1);
    float depth_weight = 1 - 50 * abs(gbuf_curr.w - gbuf_prev.w)/(min(gbuf_curr.w, gbuf_prev.w) + 1e-6);
    o = lerp(lighting_prev, lighting_curr, lerp(0.9, 0.03, saturate(normal_weight * depth_weight)));
}

void PS_CopyPrev(in VSOUT i, out MRT2 o)
{
    o.t0 = tex2D(sReLightTmp, i.uv);
    o.t1 = tex2D(sGBTex, i.uv);
}

void PS_ApplyWithShadows(in VSOUT i, out float3 o : SV_Target0)
{
    float3 lighting = tex2D(sReLightTmp, i.uv).rgb;

    if(USE_SHADOW_FILTERING)
    {    
        lighting = 0;
        float4 gbc = tex2D(sGBTex, i.uv);
        float wsum = 0.0001;

        float3 g = float3(1, 0.64, 0.45);
        float4 maxv = 0;

        const float sigma_n = 20;
        const float sigma_z = 4;

        for(int x = -2; x <= 2; x++)
        for(int y = -2; y <= 2; y++)
        {
            float3 l = tex2Dlod(sReLightTmp, i.uv + BUFFER_PIXEL_SIZE * float2(x, y) * 1, 0).rgb;
            float4 gbt = tex2Dlod(sGBTex, i.uv + BUFFER_PIXEL_SIZE * float2(x, y) * 1, 0);

            float wn = saturate(dot(normalize(gbc.xyz), normalize(gbt.xyz)) * sigma_n - sigma_n + 1);
            float wz = saturate(1 - sigma_z * abs(gbc.w - gbt.w)/(min(gbc.w, gbt.w) + 1e-6));
            float w = wn * g[abs(x)] * g[abs(y)] * wz;
            lighting += l * w;
            wsum += w;
            maxv = max(maxv, float4(l, 1) * w);
        }

        lighting /= wsum;   
        float delta = length(maxv.xyz - lighting);
        lighting = lerp(lighting, (lighting * wsum - maxv.xyz) / max(0.0001, wsum - maxv.w), saturate(delta));
    }

    float3 color = tex2D(ColorInput, i.uv).rgb;
    color = unpack_hdr(color);

    color *= 1 + lighting;
    //color = lighting * 0.05;

    o = color;
    o.rgb = pack_hdr(o.rgb);
}

void PS_ApplyNoShadows(in VSOUT i, out float3 o : SV_Target0)
{
    Light lights[AMOUNT_OF_LIGHTS];
    setup_lights(lights); 

    float3 lighting = 0;

    [unroll]
    for(int id = 0; id < AMOUNT_OF_LIGHTS; id++)
        lighting += ProcessLight(i, lights[id], 0.0, false).rgb;

    float3 color = tex2D(ColorInput, i.uv).rgb;
    color = unpack_hdr(color);

    color *= 1 + lighting;

    o = color;
    o.rgb = pack_hdr(o.rgb);
}

void PSAddLightOverlay(in VSOUT i, out float3 o : SV_Target0)
{
    if(!(OVERLAY_OPEN && RT_SHOW_DEBUG) || SCREENSHOT) discard;

    float3 color = tex2D(ColorInput, i.uv).rgb; 
   
    //light debug view
    Light lights[AMOUNT_OF_LIGHTS];
    setup_lights(lights);

    float4 lightvis = float4(0, 0, 0, 10000000);
    float3 p = Projection::uv_to_proj(i.uv);
    float p_dist = length(p);
    float3 e = p / p_dist;

    [loop]
    for(int id = 0; id < AMOUNT_OF_LIGHTS; id++)    
    {
        if(lights[id].intensity < 0.001) 
            continue;

        float2 light_uv = float2(lights[id].pos.x, 1 - lights[id].pos.y);//flip Y because user expectation == !HLSL logic
        float3 light_pos = Projection::uv_to_proj(light_uv, Projection::depth_to_z(lights[id].pos.z * lights[id].pos.z * lights[id].pos.z));//z cubed for easier adjustment

        float3 light_color = unpack_hdr(lights[id].tint) * saturate(lights[id].intensity) * saturate(lights[id].intensity);           

        float hit = ray_sphere_intersection(0, e, light_pos, POINT_LIGHT_RADIUS).x;        
        lightvis = hit < lightvis.w ? float4(light_color, hit) : lightvis;
    }

    float3 closest_hit = e * lightvis.w;
    float3 hit_n = normalize(cross(ddx(closest_hit), ddy(closest_hit)));    
    color = lerp(color, pack_hdr(lightvis.rgb * saturate(hit_n.z * 0.5 + 0.5)), lightvis.w < p_dist);
    o = color;
}

/*=============================================================================
	Techniques
=============================================================================*/

technique qUINT_ReLight
{    
    pass { VertexShader = VSMain; PixelShader = PS_MakeInput_Gbuf;  RenderTarget = GBTexRaw; }
    pass { VertexShader = VSMain; PixelShader = PS_Smoothnormals;   RenderTarget = GBTex; }
    pass { VertexShader = VSMain; PixelShader = PSWriteZ;  RenderTarget = ZTexLow;} 
    pass { VertexShader = VSShadows; PixelShader = PS_ReSTIR_Initial;  RenderTarget = ReSTIRReservoir;}   
    pass { VertexShader = VSShadows; PixelShader = PS_ReSTIR_Resampling; RenderTarget = ReLightCurr;}
    pass { VertexShader = VSShadows; PixelShader = PS_TemporalBlend; RenderTarget = ReLightTmp; }
    pass { VertexShader = VSShadows; PixelShader = PS_CopyPrev; RenderTarget = ReLightPrev; RenderTarget1 = GBTexPrev;}    
    pass { VertexShader = VSShadows; PixelShader = PS_ApplyWithShadows; }
    pass { VertexShader = VSNoShadows; PixelShader = PS_ApplyNoShadows; }    
    pass { VertexShader = VSMain; PixelShader = PSAddLightOverlay;   }          
}

