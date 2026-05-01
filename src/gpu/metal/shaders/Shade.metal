// Shade.metal — per-pixel path tracing kernel (interactive / preview quality)
//
// One thread per pixel.  Performs up to maxDepth bounces using hardware
// intersection against the TLAS.  Direct lighting is sampled once per bounce
// from a randomly-selected light.  Indirect illumination uses cosine-weighted
// hemisphere sampling for Lambertian surfaces.
//
// Material support:
//   kMatLambertian — diffuse only
//   kMatEmissive   — emits Le, no scattering
//   kMatGGX        — GGX microfacet (roughness/metalness), no transmission
//   kMatGlass      — smooth dielectric: exact Fresnel + Snell refraction
//
// This is a "megakernel" path tracer suitable for interactive preview.

#include <metal_stdlib>
#include <metal_raytracing>
#include "SharedTypes.h"

using namespace metal;
using namespace raytracing;

// ---------------------------------------------------------------------------
// Packed vertex attribute types (match what the CPU uploaded)
// ---------------------------------------------------------------------------
struct PackedFloat3 { float x, y, z; };
struct PackedFloat2 { float x, y; };

// ---------------------------------------------------------------------------
// PCG random
// ---------------------------------------------------------------------------
static uint pcg(uint s) {
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
static float rand01(thread uint& s) {
    s = pcg(s * 747796405u + 2891336453u);
    return float(s) * (1.0f / 4294967296.0f);
}
static float2 rand2(thread uint& s) { return float2(rand01(s), rand01(s)); }

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static float3 interpolateNormal(
    uint   primID,
    float2 bary,
    const device PackedFloat3* normals,
    const device uint32_t*     indices,
    uint   indexOffset)           // element offset (not byte)
{
    uint i0 = indices[indexOffset + primID * 3 + 0];
    uint i1 = indices[indexOffset + primID * 3 + 1];
    uint i2 = indices[indexOffset + primID * 3 + 2];
    float3 n0 = float3(normals[i0].x, normals[i0].y, normals[i0].z);
    float3 n1 = float3(normals[i1].x, normals[i1].y, normals[i1].z);
    float3 n2 = float3(normals[i2].x, normals[i2].y, normals[i2].z);
    float w = 1.0f - bary.x - bary.y;
    return normalize(n0 * w + n1 * bary.x + n2 * bary.y);
}

// ---------------------------------------------------------------------------
// BSDF helpers
// ---------------------------------------------------------------------------
static float3 cosineSampleHemisphere(float2 u, float3 n) {
    float phi      = 2.0f * M_PI_F * u.x;
    float cosTheta = sqrt(u.y);
    float sinTheta = sqrt(1.0f - u.y);
    float3 t, bt;
    if (abs(n.x) > 0.9f)
        t = normalize(cross(float3(0,1,0), n));
    else
        t = normalize(cross(float3(1,0,0), n));
    bt = cross(n, t);
    return t * (sinTheta * cos(phi)) + bt * (sinTheta * sin(phi)) + n * cosTheta;
}

// GGX helpers for kMatGGX
static float ggxD(float cosH, float alpha2) {
    float d = cosH * cosH * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (M_PI_F * d * d + 1e-7f);
}
static float ggxG1(float cosV, float alpha2) {
    float denom = cosV + sqrt(alpha2 + (1.0f - alpha2) * cosV * cosV);
    return 2.0f * cosV / (denom + 1e-7f);
}
static float3 sampleGGX(float2 u, float alpha2) {
    float phi      = 2.0f * M_PI_F * u.x;
    float cosH2    = (1.0f - u.y) / (1.0f + (alpha2 - 1.0f) * u.y + 1e-7f);
    float cosH     = sqrt(max(0.0f, cosH2));
    float sinH     = sqrt(max(0.0f, 1.0f - cosH2));
    return float3(sinH * cos(phi), sinH * sin(phi), cosH);
}
static float3 schlick(float cosI, float3 F0) {
    float p = pow(1.0f - cosI, 5.0f);
    return F0 + (1.0f - F0) * p;
}
// Exact dielectric Fresnel reflectance (scalar, unpolarised).
// cosI: cosine of angle with surface normal (must be >= 0).
// eta:  relative IOR = n_inside / n_outside (eta > 1 = entering denser medium).
static float fresnelDielectric(float cosI, float eta) {
    // sinT = sinI / eta  (Snell's law), so sinT2 = sin2I / eta2.
    // The wrong formula eta2*sin2I triggers false TIR when entering glass from air,
    // making the dome nearly fully reflective and the interior dark.
    float sinT2 = (1.0f - cosI * cosI) / (eta * eta);
    if (sinT2 >= 1.0f) return 1.0f;  // total internal reflection
    float cosT  = sqrt(1.0f - sinT2);
    float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5f * (rs * rs + rp * rp);
}

// Build local ONB around n, transform v from tangent to world
static float3 toWorld(float3 v, float3 n) {
    float3 t, bt;
    if (abs(n.x) > 0.9f) t = normalize(cross(float3(0,1,0), n));
    else                  t = normalize(cross(float3(1,0,0), n));
    bt = cross(n, t);
    return v.x * t + v.y * bt + v.z * n;
}
static float3 toLocal(float3 v, float3 n) {
    float3 t, bt;
    if (abs(n.x) > 0.9f) t = normalize(cross(float3(0,1,0), n));
    else                  t = normalize(cross(float3(1,0,0), n));
    bt = cross(n, t);
    return float3(dot(v, t), dot(v, bt), dot(v, n));
}

// ---------------------------------------------------------------------------
// HDRI environment map helpers
// ---------------------------------------------------------------------------

// Apply world-to-envmap rotation stored as three row vectors in cam params
static float3 rotateToEnv(float3 wo, constant GpuCameraParams& cam) {
    float3 r0 = float3(cam.envRot0.x, cam.envRot0.y, cam.envRot0.z);
    float3 r1 = float3(cam.envRot1.x, cam.envRot1.y, cam.envRot1.z);
    float3 r2 = float3(cam.envRot2.x, cam.envRot2.y, cam.envRot2.z);
    return float3(dot(r0, wo), dot(r1, wo), dot(r2, wo));
}

// Sample the HDRI texture at world direction wo.
// Convention matches CPU DomeLight: theta=0 at +Y (row 0), u=phi/(2pi), v=theta/pi.
static float3 evalEnvmap(float3 wo,
                          constant GpuCameraParams& cam,
                          texture2d<float, access::sample> envTex) {
    float3 local = rotateToEnv(wo, cam);
    float theta  = acos(clamp(local.y, -1.0f, 1.0f));
    float phi    = atan2(local.x, local.z);
    if (phi < 0.0f) phi += 2.0f * M_PI_F;
    float u = phi  / (2.0f * M_PI_F);
    float v = theta / M_PI_F;
    constexpr sampler envSampler(s_address::repeat,
                                  t_address::clamp_to_edge,
                                  filter::linear,
                                  coord::normalized);
    float4 c = envTex.sample(envSampler, float2(u, v));
    return max(float3(0.0f), c.rgb) * cam.envIntensity;
}

// ---------------------------------------------------------------------------
// Direct-light sampling (one light chosen uniformly at random)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Shadow transmittance — steps through glass surfaces between hitPos and the
// light, accumulating tint. Returns (0,0,0) if an opaque surface blocks the
// path, or a tint <= (1,1,1) attenuated by any glass surfaces in between.
// ---------------------------------------------------------------------------
static float3 shadowTransmittance(
    float3                          origin,
    float3                          dir,
    float                           tMax,
    const device GpuMaterial*       materials,
    uint                            numMaterials,
    const device PackedFloat3*      normals,
    const device uint32_t*          indices,
    const device uint32_t*          meshIndexOffsets,
    instance_acceleration_structure accelStruct)
{
    float3 T = float3(1.0f);
    ray    stepRay;
    stepRay.direction   = dir;
    stepRay.min_distance = 1e-4f;
    stepRay.max_distance = tMax;
    stepRay.origin      = origin;

    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(false);  // need closest hit to step correctly

    for (int step = 0; step < 8; ++step) {
        intersection_result<triangle_data, instancing> res =
            isect.intersect(stepRay, accelStruct, 0xFF);

        if (res.type == intersection_type::none) break;  // clear path

        uint meshID = res.instance_id;
        uint matIdx = (meshID < numMaterials) ? meshID : 0;
        GpuMaterial mat = materials[matIdx];

        if (mat.type == kMatGlass) {
            // Attenuate by transmission weight, not baseColor.
            // baseColor is the diffuse reflectance — black for pure glass — which
            // would zero out all shadow rays and make objects inside appear black.
            T *= mat.transmission;
            if (max(T.x, max(T.y, T.z)) < 1e-4f) return float3(0);

            // Advance past this surface
            float remaining = stepRay.max_distance - res.distance;
            if (remaining <= 1e-4f) break;
            stepRay.origin      = stepRay.origin + dir * (res.distance + 1e-4f);
            stepRay.max_distance = remaining - 1e-4f;
        } else {
            // Opaque surface blocks the light
            return float3(0);
        }
    }
    return T;
}

static float3 sampleDirect(
    float3                          hitPos,
    float3                          n,
    float3                          wo,
    uint                            matType,
    float3                          baseColor,
    float                           roughness,
    float                           metalness,
    const device GpuLight*          lights,
    uint                            numLights,
    const device GpuMaterial*       materials,
    uint                            numMaterials,
    const device PackedFloat3*      normals,
    const device uint32_t*          indices,
    const device uint32_t*          meshIndexOffsets,
    thread uint&                    rng,
    instance_acceleration_structure accelStruct,
    constant GpuCameraParams&       cam,
    texture2d<float, access::sample> envTex)
{
    if (numLights == 0) return float3(0);

    uint lightIdx = uint(rand01(rng) * float(numLights)) % numLights;
    const device GpuLight& light = lights[lightIdx];
    float lightPick = 1.0f / float(numLights);

    float3 Li    = float3(0);
    float3 wi    = float3(0);
    float  tMax  = 0;
    float  pdfL  = 0;

    if (light.type == kLightRect) {
        float2 u = rand2(rng);
        float3 samplePos = float3(light.position.x, light.position.y, light.position.z)
                         + float3(light.uHalf.x, light.uHalf.y, light.uHalf.z) * (2.0f * u.x - 1.0f)
                         + float3(light.vHalf.x, light.vHalf.y, light.vHalf.z) * (2.0f * u.y - 1.0f);
        float3 toL  = samplePos - hitPos;
        float  dist = length(toL);
        wi    = toL / dist;
        tMax  = dist * 0.9999f;

        float3 lightN = float3(light.normal.x, light.normal.y, light.normal.z);
        float  cosL   = dot(-wi, lightN);
        if (cosL <= 0.0f) return float3(0);

        pdfL = (dist * dist) / (cosL * light.area) * lightPick;
        Li   = float3(light.Le.x, light.Le.y, light.Le.z);

    } else if (light.type == kLightDirectional) {
        float3 baseDir = float3(light.normal.x, light.normal.y, light.normal.z);
        float  cc = light.cosCone;
        if (cc < 0.9999f) {
            float2 uCone    = rand2(rng);
            float  cosTheta = 1.0f - uCone.x * (1.0f - cc);
            float  sinTheta = sqrt(max(0.0f, 1.0f - cosTheta * cosTheta));
            float  phi      = 2.0f * M_PI_F * uCone.y;
            float3 tangent, bitangent;
            if (abs(baseDir.x) > 0.9f)
                tangent = normalize(cross(float3(0,1,0), baseDir));
            else
                tangent = normalize(cross(float3(1,0,0), baseDir));
            bitangent = cross(baseDir, tangent);
            wi = normalize(tangent  * (sinTheta * cos(phi))
                         + bitangent * (sinTheta * sin(phi))
                         + baseDir   * cosTheta);
        } else {
            wi = baseDir;
        }
        tMax = 1e9f;
        pdfL = lightPick;
        Li   = float3(light.Le.x, light.Le.y, light.Le.z);

    } else if (light.type == kLightDome) {
        wi   = cosineSampleHemisphere(rand2(rng), n);
        tMax = 1e9f;
        float cosW = max(1e-7f, dot(n, wi));
        pdfL = (cosW / M_PI_F) * lightPick;
        Li   = evalEnvmap(wi, cam, envTex);

    } else {
        return float3(0);
    }

    float cosI = dot(n, wi);
    if (cosI <= 0.0f || pdfL <= 0.0f) return float3(0);

    // Transmittance along shadow ray — steps through glass surfaces
    float3 shadowOrigin = hitPos + n * 1e-4f;
    float3 Tr = shadowTransmittance(shadowOrigin, wi, tMax,
                                    materials, numMaterials,
                                    normals, indices, meshIndexOffsets,
                                    accelStruct);
    if (max(Tr.x, max(Tr.y, Tr.z)) <= 0.0f) return float3(0);

    // BSDF eval
    float3 f = float3(0);
    if (matType == kMatLambertian) {
        f = baseColor * (1.0f / M_PI_F);
    } else if (matType == kMatGGX) {
        float3 wh  = normalize(wo + wi);
        float  cosH = max(0.0f, dot(n, wh));
        float  cosO = max(0.0f, dot(n, wo));
        float  cosII = max(0.0f, dot(n, wi));
        float  alpha  = roughness * roughness;
        float  alpha2 = alpha * alpha;
        float  D  = ggxD(cosH, alpha2);
        float  G  = ggxG1(cosO, alpha2) * ggxG1(cosII, alpha2);
        float3 F0 = mix(float3(0.04f), baseColor, metalness);
        float3 F  = schlick(dot(wi, wh), F0);
        float3 specular = D * G * F / max(1e-7f, 4.0f * cosO * cosII);
        float3 diffuse  = (1.0f - metalness) * baseColor * (1.0f / M_PI_F);
        f = diffuse + specular;
    }
    // Glass is delta — no area PDF can be evaluated, skip direct lighting

    return f * Li * Tr * cosI / pdfL;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
kernel void shade(
    constant  GpuCameraParams&              cam           [[ buffer(0) ]],
    device    GpuAccumPixel*                accum         [[ buffer(1) ]],
    const device GpuLight*                  lights        [[ buffer(2) ]],
    constant  uint&                         numLights     [[ buffer(3) ]],
    const device GpuMaterial*               materials     [[ buffer(4) ]],
    constant  uint&                         numMaterials  [[ buffer(5) ]],
    const device PackedFloat3*              normals       [[ buffer(6) ]],
    const device uint32_t*                  indices       [[ buffer(7) ]],
    const device uint32_t*                  triMeshIDs    [[ buffer(8) ]],
    const device uint32_t*                  meshVertexOffsets [[ buffer(9) ]],
    const device uint32_t*                  meshIndexOffsets  [[ buffer(10) ]],
    constant  uint&                         sampleIndex   [[ buffer(11) ]],
    instance_acceleration_structure         accelStruct   [[ buffer(12) ]],
    texture2d<float, access::sample>        envTexture    [[ texture(0) ]],
    uint2                                   gid           [[ thread_position_in_grid ]])
{
    uint px = cam.tileX0 + gid.x;
    uint py = cam.tileY0 + gid.y;
    if (px >= cam.imageWidth || py >= cam.imageHeight) return;

    // accum is tile-sized; use local coordinates for the write index
    uint pixelIdx = gid.y * cam.tileWidth + gid.x;

    // Per-sample RNG seed — use global pixel position so tiles at different
    // screen positions produce different noise patterns.
    uint globalPixelIdx = py * cam.imageWidth + px;
    uint rng = pcg(pcg(globalPixelIdx) ^ (sampleIndex * 2654435761u));

    // Generate camera ray
    float jx = rand01(rng);
    float jy = rand01(rng);

    float u = (float(px) + jx) / float(cam.imageWidth);
    float v = (float(cam.imageHeight - 1 - py) + jy) / float(cam.imageHeight);

    float3 origin = float3(cam.origin.x, cam.origin.y, cam.origin.z);
    float3 horiz  = float3(cam.horizontal.x, cam.horizontal.y, cam.horizontal.z);
    float3 vert   = float3(cam.vertical.x,   cam.vertical.y,   cam.vertical.z);
    float3 ll     = float3(cam.lowerLeft.x,  cam.lowerLeft.y,  cam.lowerLeft.z);

    ray r;
    r.origin       = origin;
    r.direction    = normalize(ll + u * horiz + v * vert - origin);
    r.min_distance = 1e-4f;
    r.max_distance = 1e10f;

    // Path tracing loop
    float3 throughput = float3(1.0f);
    float3 L          = float3(0.0f);
    uint   glassDepth = 0;  // separate counter so glass doesn't exhaust bounce budget

    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(false);

    for (uint bounce = 0; bounce <= cam.maxDepth; ++bounce) {

        intersection_result<triangle_data, instancing> res =
            isect.intersect(r, accelStruct, 0xFF);

        if (res.type == intersection_type::none) {
            float3 envColor;
            if (cam.hasEnvLight) {
                envColor = evalEnvmap(r.direction, cam, envTexture);
            } else {
                float skyT = 0.5f * (r.direction.y + 1.0f);
                envColor = mix(float3(1.0f), float3(0.5f, 0.7f, 1.0f), skyT) * 0.5f;
            }
            L += throughput * envColor;
            break;
        }

        // Recover hit data
        uint instID   = res.instance_id;
        uint primID   = res.primitive_id;
        float2 bary   = res.triangle_barycentric_coord;
        float  t      = res.distance;

        float3 hitPos = r.origin + r.direction * t;

        // Global triangle index: primID is local within the BLAS instance (mesh)
        // We need the global triangle index for triMeshIDs lookup.
        // Since each mesh is one BLAS, instID == meshID.
        uint meshID = instID;

        // Index offset for this mesh (in elements, not bytes)
        uint idxOff = meshIndexOffsets[meshID] / 1;  // already element offset

        float3 geomN = interpolateNormal(primID, bary, normals, indices, idxOff);

        // geomN is the unflipped mesh normal — used by glass to detect entry vs exit.
        // n is flipped to always face the incoming ray for diffuse/specular shading.
        float3 n = geomN;
        if (dot(-r.direction, n) < 0.0f) n = -n;

        // Material lookup
        uint matIdx = (meshID < numMaterials) ? meshID : 0;
        GpuMaterial mat = materials[matIdx];
        float3 baseColor = float3(mat.baseColor.x, mat.baseColor.y, mat.baseColor.z);
        float3 emissive  = float3(mat.emissive.x,  mat.emissive.y,  mat.emissive.z);

        // Emission
        if (mat.type == kMatEmissive) {
            L += throughput * emissive;
            break;
        }

        // Direct lighting (skip for delta glass — no area-light PDF)
        float3 wo = -r.direction;
        if (mat.type != kMatGlass) {
            L += throughput * sampleDirect(hitPos, n, wo,
                                           mat.type, baseColor,
                                           mat.roughness, mat.metalness,
                                           lights, numLights,
                                           materials, numMaterials,
                                           normals, indices, meshIndexOffsets,
                                           rng, accelStruct, cam, envTexture);
        }

        // Russian roulette after bounce 3
        if (bounce >= 3) {
            float q = max(0.05f, 1.0f - max(throughput.x, max(throughput.y, throughput.z)));
            if (rand01(rng) < q) break;
            throughput /= (1.0f - q);
        }

        // Sample next direction
        float3 wi;
        float  bsdfPdf;
        float3 bsdfF;

        if (mat.type == kMatGlass) {
            // Use geomN (unflipped) to detect entry vs exit — n was already flipped
            // to face the ray so it cannot distinguish entry from exit.
            bool entering = dot(r.direction, geomN) < 0.0f;  // ray opposes outward normal → entering
            float3 faceN  = entering ? geomN : -geomN;       // points toward ray origin
            float  eta    = entering ? (1.0f / mat.specularIOR) : mat.specularIOR;

            float cosI = dot(-r.direction, faceN);
            float Fr   = fresnelDielectric(cosI, 1.0f / eta);  // eta = n2/n1, invert for function convention

            if (rand01(rng) < Fr) {
                // Reflect
                wi = reflect(r.direction, faceN);
                r.origin = hitPos + faceN * 1e-4f;
            } else {
                // Refract (Snell's law) — Metal's built-in refract(I, N, eta) where eta = n1/n2
                wi = refract(r.direction, faceN, eta);
                if (length_squared(wi) < 0.5f) {
                    // Total internal reflection fallback
                    wi = reflect(r.direction, faceN);
                    r.origin = hitPos + faceN * 1e-4f;
                } else {
                    r.origin = hitPos - faceN * 1e-4f;  // offset to inside surface
                }
            }
            // Delta BSDF: f/pdf = 1, throughput unchanged.
            // baseColor is the diffuse reflectance (often 0.5 grey for OslMaterial),
            // not a glass tint — using it here would darken everything behind glass.
            bsdfF   = float3(1.0f);
            bsdfPdf = 1.0f;
            r.direction    = normalize(wi);
            r.min_distance = 1e-4f;
            r.max_distance = 1e10f;
            throughput *= bsdfF;
            // Glass hits don't count against bounce budget — refracting through a
            // dome's two surfaces plus any interior bounces would exhaust maxDepth
            // before the background is ever reached.  Use a separate glass limiter.
            if (++glassDepth >= 16) break;
            bounce = (bounce > 0u) ? bounce - 1u : 0u;
            continue;

        } else if (mat.type == kMatGGX && mat.roughness < 0.95f) {
            float alpha2 = mat.roughness * mat.roughness;  // GGX alpha = roughness^2
            float3 wmLocal = sampleGGX(rand2(rng), alpha2);
            float3 wh = toWorld(wmLocal, n);
            if (dot(wh, n) < 0.0f) wh = -wh;
            wi = reflect(-wo, wh);
            if (dot(wi, n) <= 0.0f) break;

            float cosI = dot(n, wi);
            float cosO = dot(n, wo);
            float cosH = dot(n, wh);
            float D    = ggxD(cosH, alpha2);
            float G    = ggxG1(cosO, alpha2) * ggxG1(cosI, alpha2);
            float3 F0  = mix(float3(0.04f), baseColor, mat.metalness);
            float3 F   = schlick(dot(wi, wh), F0);
            bsdfPdf = D * cosH / max(1e-7f, 4.0f * dot(wo, wh));
            float3 spec = D * G * F / max(1e-7f, 4.0f * cosO * cosI);
            float3 diff = (1.0f - mat.metalness) * baseColor * (1.0f / M_PI_F);
            bsdfF = diff + spec;
        } else {
            // Cosine hemisphere (Lambertian)
            wi       = cosineSampleHemisphere(rand2(rng), n);
            bsdfPdf  = max(1e-7f, dot(n, wi)) / M_PI_F;
            bsdfF    = baseColor / M_PI_F;
        }

        float cosI = dot(n, wi);
        if (cosI <= 0.0f || bsdfPdf <= 0.0f) break;
        throughput *= bsdfF * cosI / bsdfPdf;

        // Spawn next ray
        r.origin       = hitPos + n * 1e-4f;
        r.direction    = wi;
        r.min_distance = 1e-4f;
        r.max_distance = 1e10f;
    }

    // Accumulate (per-sample; host averages after all samples)
    float lum = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
    device GpuAccumPixel& px_out = accum[pixelIdx];
    px_out.r        += L.x;
    px_out.g        += L.y;
    px_out.b        += L.z;
    px_out.weight   += 1.0f;
    px_out.sumLumSq += lum * lum;
}
