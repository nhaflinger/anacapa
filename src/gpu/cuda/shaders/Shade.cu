// Shade.cu — pure-CUDA path-tracing kernel.
//
// Replaces the OptiX raygen/closesthit/miss programs.  One thread per pixel,
// full path-tracing loop using iterative BVH traversal for each bounce.
// Material support: Lambertian, GGX, Emissive, Glass (same as before).

#include <cuda_runtime.h>
#include <math_constants.h>
#include <float.h>

#include "SharedTypes.h"
#include "LaunchParams.h"

// ---------------------------------------------------------------------------
// PCG random
// ---------------------------------------------------------------------------
static __forceinline__ __device__ uint32_t pcg(uint32_t s) {
    uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
static __forceinline__ __device__ float rand01(uint32_t& s) {
    s = pcg(s * 747796405u + 2891336453u);
    return float(s) * (1.0f / 4294967296.0f);
}
static __forceinline__ __device__ float2 rand2(uint32_t& s) {
    return make_float2(rand01(s), rand01(s));
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------
static __forceinline__ __device__ float3 make3(GpuFloat3 v) {
    return make_float3(v.x, v.y, v.z);
}
static __forceinline__ __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
static __forceinline__ __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
static __forceinline__ __device__ float3 operator*(float3 a, float3 b) {
    return make_float3(a.x*b.x, a.y*b.y, a.z*b.z);
}
static __forceinline__ __device__ float3 operator*(float3 a, float s) {
    return make_float3(a.x*s, a.y*s, a.z*s);
}
static __forceinline__ __device__ float3 operator*(float s, float3 a) { return a * s; }
static __forceinline__ __device__ float3 operator-(float s, float3 a) {
    return make_float3(s-a.x, s-a.y, s-a.z);
}
static __forceinline__ __device__ float3& operator+=(float3& a, float3 b) {
    a.x+=b.x; a.y+=b.y; a.z+=b.z; return a;
}
static __forceinline__ __device__ float3& operator*=(float3& a, float3 b) {
    a.x*=b.x; a.y*=b.y; a.z*=b.z; return a;
}
static __forceinline__ __device__ float3& operator*=(float3& a, float s) {
    a.x*=s; a.y*=s; a.z*=s; return a;
}
static __forceinline__ __device__ float  dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
static __forceinline__ __device__ float3 cross(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
static __forceinline__ __device__ float3 normalize(float3 v) {
    float inv = rsqrtf(dot(v,v)); return v * inv;
}
static __forceinline__ __device__ float3 reflect(float3 d, float3 n) {
    return d - 2.0f * dot(d, n) * n;
}
static __forceinline__ __device__ float3 refract(float3 i, float3 n, float eta) {
    float cosI = -dot(i, n);
    float sinT2 = eta * eta * (1.0f - cosI * cosI);
    if (sinT2 >= 1.0f) return make_float3(0,0,0);
    return eta * i + (eta * cosI - sqrtf(1.0f - sinT2)) * n;
}
static __forceinline__ __device__ float3 fmaxf3(float3 a, float3 b) {
    return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
static __forceinline__ __device__ float3 fminf3(float3 a, float3 b) {
    return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
static __forceinline__ __device__ float  compmax(float3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
static __forceinline__ __device__ float3 lerp3(float3 a, float3 b, float t) {
    return a * (1.0f - t) + b * t;
}

// ---------------------------------------------------------------------------
// BVH traversal
// ---------------------------------------------------------------------------

// AABB slab test — returns entry distance in tEntry (FLT_MAX on miss).
static __forceinline__ __device__
float rayAABBDist(float3 orig, float3 invDir, float tMin, float tMax,
                  GpuFloat3 bMin, GpuFloat3 bMax)
{
    float t0x = (bMin.x - orig.x) * invDir.x;
    float t1x = (bMax.x - orig.x) * invDir.x;
    float lo = fminf(t0x, t1x), hi = fmaxf(t0x, t1x);

    float t0y = (bMin.y - orig.y) * invDir.y;
    float t1y = (bMax.y - orig.y) * invDir.y;
    lo = fmaxf(lo, fminf(t0y, t1y));
    hi = fminf(hi, fmaxf(t0y, t1y));

    float t0z = (bMin.z - orig.z) * invDir.z;
    float t1z = (bMax.z - orig.z) * invDir.z;
    lo = fmaxf(lo, fminf(t0z, t1z));
    hi = fminf(hi, fmaxf(t0z, t1z));

    float entry = fmaxf(lo, tMin);
    return (entry <= fminf(hi, tMax)) ? entry : FLT_MAX;
}

// Möller–Trumbore ray-triangle intersection.
// Returns true and writes tHit, bary (u,v) if hit in [tMin, tMax).
static __forceinline__ __device__
bool rayTriangle(float3 orig, float3 dir, float tMin, float tMax,
                 uint32_t triIdx,
                 const float* positions, const uint32_t* indices,
                 float& tHit, float2& bary)
{
    uint32_t i0 = indices[triIdx * 3 + 0];
    uint32_t i1 = indices[triIdx * 3 + 1];
    uint32_t i2 = indices[triIdx * 3 + 2];
    float3 v0 = make_float3(positions[i0*3], positions[i0*3+1], positions[i0*3+2]);
    float3 v1 = make_float3(positions[i1*3], positions[i1*3+1], positions[i1*3+2]);
    float3 v2 = make_float3(positions[i2*3], positions[i2*3+1], positions[i2*3+2]);

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 h  = cross(dir, e2);
    float  a  = dot(e1, h);
    if (fabsf(a) < 1e-8f) return false;

    float  f  = 1.0f / a;
    float3 s  = orig - v0;
    float  u  = f * dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    float3 q  = cross(s, e1);
    float  v  = f * dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float  t  = f * dot(e2, q);
    if (t < tMin || t >= tMax) return false;

    tHit = t;
    bary = make_float2(u, v);
    return true;
}

struct TraceResult {
    uint32_t valid;   // 0 = miss
    uint32_t meshID;
    uint32_t primID;  // local triangle index within mesh
    float2   bary;
    float    t;
};

// Iterative BVH traversal — returns the closest hit.
static __forceinline__ __device__
TraceResult bvhTrace(float3 orig, float3 dir, float tMin, float tMax,
                     const BvhNode* bvh, const uint32_t* triIndices,
                     const float* positions, const uint32_t* indices,
                     const uint32_t* triMeshIDs, const uint32_t* meshIndexOffsets)
{
    float3 invDir = make_float3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);

    uint32_t stack[32];
    int      top = 0;
    stack[top++] = 0u;  // root

    TraceResult result{};
    result.valid = 0;
    float tHit = tMax;

    while (top > 0) {
        uint32_t       nodeIdx = stack[--top];
        const BvhNode& node    = bvh[nodeIdx];

        if (rayAABBDist(orig, invDir, tMin, tHit, node.aabbMin, node.aabbMax) == FLT_MAX)
            continue;

        if (node.triCount > 0) {
            // Leaf — test each triangle
            for (uint32_t i = 0; i < node.triCount; ++i) {
                uint32_t globalTriIdx = triIndices[node.leftFirst + i];
                float    tTemp;
                float2   tempBary;
                if (rayTriangle(orig, dir, tMin, tHit, globalTriIdx,
                                positions, indices, tTemp, tempBary)) {
                    tHit           = tTemp;
                    result.valid   = 1;
                    result.meshID  = triMeshIDs[globalTriIdx];
                    result.primID  = globalTriIdx - meshIndexOffsets[result.meshID] / 3;
                    result.bary    = tempBary;
                    result.t       = tTemp;
                }
            }
        } else {
            // Internal — test both children, push farther first (front-to-back LIFO)
            uint32_t L = node.leftFirst, R = node.leftFirst + 1;
            float tL = rayAABBDist(orig, invDir, tMin, tHit, bvh[L].aabbMin, bvh[L].aabbMax);
            float tR = rayAABBDist(orig, invDir, tMin, tHit, bvh[R].aabbMin, bvh[R].aabbMax);
            bool hitL = tL < FLT_MAX, hitR = tR < FLT_MAX;
            if (hitL && hitR) {
                if (tL < tR) { stack[top++] = R; stack[top++] = L; }
                else          { stack[top++] = L; stack[top++] = R; }
            } else if (hitL) { stack[top++] = L; }
            else if (hitR)   { stack[top++] = R; }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 interpolateNormal(uint32_t primId, float2 bary,
                         const GpuFloat3* normals,
                         const uint32_t*  indices,
                         uint32_t         indexOffset)
{
    uint32_t i0 = indices[indexOffset + primId * 3 + 0];
    uint32_t i1 = indices[indexOffset + primId * 3 + 1];
    uint32_t i2 = indices[indexOffset + primId * 3 + 2];
    float3 n0 = make3(normals[i0]);
    float3 n1 = make3(normals[i1]);
    float3 n2 = make3(normals[i2]);
    float w = 1.0f - bary.x - bary.y;
    return normalize(n0 * w + n1 * bary.x + n2 * bary.y);
}

// ---------------------------------------------------------------------------
// BSDF helpers
// ---------------------------------------------------------------------------
static __forceinline__ __device__ void buildONB(float3 n, float3& t, float3& bt) {
    if (fabsf(n.x) > 0.9f)
        t = normalize(cross(make_float3(0,1,0), n));
    else
        t = normalize(cross(make_float3(1,0,0), n));
    bt = cross(n, t);
}
static __forceinline__ __device__ float3 toWorld(float3 v, float3 n) {
    float3 t, bt; buildONB(n, t, bt);
    return v.x * t + v.y * bt + v.z * n;
}
static __forceinline__ __device__ float3 cosineSampleHemisphere(float2 u, float3 n) {
    float phi      = 2.0f * CUDART_PI_F * u.x;
    float cosTheta = sqrtf(u.y);
    float sinTheta = sqrtf(1.0f - u.y);
    float3 local   = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
    return toWorld(local, n);
}
static __forceinline__ __device__ float ggxD(float cosH, float a2) {
    float d = cosH * cosH * (a2 - 1.0f) + 1.0f;
    return a2 / (CUDART_PI_F * d * d + 1e-7f);
}
static __forceinline__ __device__ float ggxG1(float cosV, float a2) {
    float denom = cosV + sqrtf(a2 + (1.0f - a2) * cosV * cosV);
    return 2.0f * cosV / (denom + 1e-7f);
}
static __forceinline__ __device__ float3 sampleGGX(float2 u, float a2) {
    float phi  = 2.0f * CUDART_PI_F * u.x;
    float cosH = sqrtf(fmaxf(0.0f, (1.0f - u.y) / (1.0f + (a2 - 1.0f) * u.y + 1e-7f)));
    float sinH = sqrtf(fmaxf(0.0f, 1.0f - cosH * cosH));
    return make_float3(sinH * cosf(phi), sinH * sinf(phi), cosH);
}
static __forceinline__ __device__ float3 schlick(float cosI, float3 F0) {
    float p = powf(1.0f - cosI, 5.0f);
    return F0 + (1.0f - F0) * p;
}
static __forceinline__ __device__ float fresnelDielectric(float cosI, float eta) {
    float sinT2 = (1.0f - cosI * cosI) / (eta * eta);
    if (sinT2 >= 1.0f) return 1.0f;
    float cosT = sqrtf(1.0f - sinT2);
    float rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    float rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5f * (rs * rs + rp * rp);
}

// ---------------------------------------------------------------------------
// Environment map
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 evalEnvmap(float3 wo, const LaunchParams& p) {
    float3 r0 = make3(p.cam.envRot0);
    float3 r1 = make3(p.cam.envRot1);
    float3 r2 = make3(p.cam.envRot2);
    float3 local = make_float3(dot(r0, wo), dot(r1, wo), dot(r2, wo));
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, local.y)));
    float phi   = atan2f(local.x, local.z);
    if (phi < 0.0f) phi += 2.0f * CUDART_PI_F;
    float u = phi  / (2.0f * CUDART_PI_F);
    float v = theta / CUDART_PI_F;
    float4 c = tex2D<float4>(p.envTexture, u, v);
    return fmaxf3(make_float3(0.0f, 0.0f, 0.0f),
                  make_float3(c.x, c.y, c.z)) * p.cam.envIntensity;
}

// ---------------------------------------------------------------------------
// Shadow transmittance — steps through glass, returns (0,0,0) if blocked
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 shadowTransmittance(float3 origin, float3 dir, float tMax,
                           const LaunchParams& p)
{
    float3 T    = make_float3(1.0f, 1.0f, 1.0f);
    float3 orig = origin;
    float  remaining = tMax;

    for (int step = 0; step < 8; ++step) {
        TraceResult hit = bvhTrace(orig, dir, 1e-4f, remaining,
                                   p.bvh, p.triIndices,
                                   p.positions, p.indices,
                                   p.triMeshIDs, p.meshIndexOffsets);
        if (!hit.valid) break;

        uint32_t matIdx = (hit.meshID < p.numMaterials) ? hit.meshID : 0u;
        GpuMaterial mat = p.materials[matIdx];

        if (mat.type == kMatGlass) {
            T *= mat.transmission;
            if (compmax(T) < 1e-4f) return make_float3(0.0f, 0.0f, 0.0f);
            remaining -= hit.t + 1e-4f;
            if (remaining <= 0.0f) break;
            orig = orig + dir * (hit.t + 1e-4f);
        } else {
            return make_float3(0.0f, 0.0f, 0.0f);
        }
    }
    return T;
}

// ---------------------------------------------------------------------------
// Direct light sampling
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 sampleDirect(float3 hitPos, float3 n, float3 wo,
                    uint32_t matType, float3 baseColor,
                    float roughness, float metalness,
                    uint32_t& rng, const LaunchParams& p)
{
    if (p.numLights == 0) return make_float3(0.0f, 0.0f, 0.0f);

    uint32_t lightIdx = uint32_t(rand01(rng) * float(p.numLights)) % p.numLights;
    const GpuLight& light = p.lights[lightIdx];
    float lightPick = 1.0f / float(p.numLights);

    float3 Li   = make_float3(0.0f, 0.0f, 0.0f);
    float3 wi   = make_float3(0.0f, 0.0f, 0.0f);
    float  tMax = 0.0f;
    float  pdfL = 0.0f;

    if (light.type == kLightRect) {
        float2 u = rand2(rng);
        float3 lpos  = make3(light.position);
        float3 luH   = make3(light.uHalf);
        float3 lvH   = make3(light.vHalf);
        float3 samplePos = lpos + luH * (2.0f * u.x - 1.0f) + lvH * (2.0f * u.y - 1.0f);
        float3 toL   = samplePos - hitPos;
        float  dist  = sqrtf(dot(toL, toL));
        wi    = toL * (1.0f / dist);
        tMax  = dist * 0.9999f;
        float3 lightN = make3(light.normal);
        float  cosL   = dot(-1.0f * wi, lightN);
        if (cosL <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
        pdfL = (dist * dist) / (cosL * light.area) * lightPick;
        Li   = make3(light.Le);

    } else if (light.type == kLightDirectional) {
        float3 baseDir = make3(light.normal);
        float  cc = light.cosCone;
        if (cc < 0.9999f) {
            float2 uCone    = rand2(rng);
            float  cosTheta = 1.0f - uCone.x * (1.0f - cc);
            float  sinTheta = sqrtf(fmaxf(0.0f, 1.0f - cosTheta * cosTheta));
            float  phi      = 2.0f * CUDART_PI_F * uCone.y;
            float3 t, bt;
            buildONB(baseDir, t, bt);
            wi = normalize(t  * (sinTheta * cosf(phi))
                         + bt * (sinTheta * sinf(phi))
                         + baseDir * cosTheta);
        } else {
            wi = baseDir;
        }
        tMax = 1e9f;
        pdfL = lightPick;
        Li   = make3(light.Le);

    } else if (light.type == kLightDome) {
        wi   = cosineSampleHemisphere(rand2(rng), n);
        tMax = 1e9f;
        float cosW = fmaxf(1e-7f, dot(n, wi));
        pdfL = (cosW / CUDART_PI_F) * lightPick;
        Li   = (p.envTexture != 0) ? evalEnvmap(wi, p) : make3(p.cam.envLe);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float cosI = dot(n, wi);
    if (cosI <= 0.0f || pdfL <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 shadowOrigin = hitPos + n * 1e-4f;
    float3 Tr = shadowTransmittance(shadowOrigin, wi, tMax, p);
    if (compmax(Tr) <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 f = make_float3(0.0f, 0.0f, 0.0f);
    if (matType == kMatLambertian) {
        f = baseColor * (1.0f / CUDART_PI_F);
    } else if (matType == kMatGGX) {
        float3 wh   = normalize(wo + wi);
        float  cosH = fmaxf(0.0f, dot(n, wh));
        float  cosO = fmaxf(0.0f, dot(n, wo));
        float  cosII= fmaxf(0.0f, dot(n, wi));
        float  a    = roughness * roughness;
        float  a2   = a * a;
        float  D    = ggxD(cosH, a2);
        float  G    = ggxG1(cosO, a2) * ggxG1(cosII, a2);
        float3 F0   = lerp3(make_float3(0.04f, 0.04f, 0.04f), baseColor, metalness);
        float3 F    = schlick(dot(wi, wh), F0);
        float3 spec = D * G * F * (1.0f / fmaxf(1e-7f, 4.0f * cosO * cosII));
        float3 diff = (1.0f - metalness) * baseColor * (1.0f / CUDART_PI_F);
        f = diff + spec;
    }

    return f * Li * Tr * cosI * (1.0f / pdfL);
}

// ---------------------------------------------------------------------------
// Main path-tracing kernel — one thread per pixel, loops over all samples.
// __launch_bounds__ tells nvcc to target >=4 blocks/SM with 256 threads/block,
// which encourages register allocation that keeps occupancy high.
// ---------------------------------------------------------------------------
extern "C" __global__
__launch_bounds__(256, 4)
void shade(LaunchParams params)
{
    uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t ty = blockIdx.y * blockDim.y + threadIdx.y;
    if (tx >= params.cam.tileWidth || ty >= params.cam.tileHeight) return;

    uint32_t px = params.cam.tileX0 + tx;
    uint32_t py = params.cam.tileY0 + ty;
    if (px >= params.cam.imageWidth || py >= params.cam.imageHeight) return;

    uint32_t pixelIdx       = ty * params.cam.tileWidth + tx;
    uint32_t globalPixelIdx = py * params.cam.imageWidth + px;
    uint32_t nSamples  = params.cam.samplesPerPixel;

    float3 origin = make3(params.cam.origin);
    float3 horiz  = make3(params.cam.horizontal);
    float3 vert   = make3(params.cam.vertical);
    float3 ll     = make3(params.cam.lowerLeft);

    float rAcc = 0.0f, gAcc = 0.0f, bAcc = 0.0f, lumSqAcc = 0.0f;

    for (uint32_t s = 0; s < nSamples; ++s) {

    uint32_t rng = pcg(pcg(globalPixelIdx) ^ ((params.sampleIndex + s) * 2654435761u));

    // Camera ray
    float jx = rand01(rng);
    float jy = rand01(rng);
    float u  = (float(px) + jx) / float(params.cam.imageWidth);
    float v  = (float(params.cam.imageHeight - 1 - py) + jy) / float(params.cam.imageHeight);

    float3 rayOrig = origin;
    float3 rayDir  = normalize(ll + u * horiz + v * vert - origin);

    float3 throughput = make_float3(1.0f, 1.0f, 1.0f);
    float3 L          = make_float3(0.0f, 0.0f, 0.0f);
    uint32_t glassDepth = 0;

    for (uint32_t bounce = 0; bounce <= params.cam.maxDepth; ++bounce) {
        TraceResult hit = bvhTrace(rayOrig, rayDir, 1e-4f, 1e10f,
                                   params.bvh, params.triIndices,
                                   params.positions, params.indices,
                                   params.triMeshIDs, params.meshIndexOffsets);

        if (!hit.valid) {
            // Miss — environment
            float3 envColor;
            if (params.cam.hasEnvLight && params.envTexture != 0) {
                envColor = evalEnvmap(rayDir, params);
            } else if (params.cam.hasEnvLight) {
                envColor = make3(params.cam.envLe);
            } else {
                float skyT = 0.5f * (rayDir.y + 1.0f);
                envColor = lerp3(make_float3(1.0f, 1.0f, 1.0f),
                                 make_float3(0.5f, 0.7f, 1.0f), skyT) * 0.5f;
            }
            L += throughput * envColor;
            break;
        }

        float3 hitPos = rayOrig + rayDir * hit.t;

        uint32_t idxOff = params.meshIndexOffsets[hit.meshID];
        float3 geomN = interpolateNormal(hit.primID, hit.bary,
                                          params.normals, params.indices, idxOff);
        float3 n = geomN;
        if (dot(-1.0f * rayDir, n) < 0.0f) n = -1.0f * n;

        uint32_t matIdx  = (hit.meshID < params.numMaterials) ? hit.meshID : 0u;
        GpuMaterial mat  = params.materials[matIdx];
        float3 baseColor = make3(mat.baseColor);
        float3 emissive  = make3(mat.emissive);

        if (mat.type == kMatEmissive) {
            L += throughput * emissive;
            break;
        }

        float3 wo = -1.0f * rayDir;
        if (mat.type != kMatGlass) {
            L += throughput * sampleDirect(hitPos, n, wo,
                                           mat.type, baseColor,
                                           mat.roughness, mat.metalness,
                                           rng, params);
        }

        // Russian roulette
        if (bounce >= 3) {
            float q = fmaxf(0.05f, 1.0f - compmax(throughput));
            if (rand01(rng) < q) break;
            throughput *= (1.0f / (1.0f - q));
        }

        // Sample next direction
        float3 wi;
        float  bsdfPdf;
        float3 bsdfF;

        if (mat.type == kMatGlass) {
            bool   entering = dot(rayDir, geomN) < 0.0f;
            float3 faceN    = entering ? geomN : -1.0f * geomN;
            float  eta      = entering ? (1.0f / mat.specularIOR) : mat.specularIOR;
            float  cosI     = dot(-1.0f * rayDir, faceN);
            float  Fr       = fresnelDielectric(cosI, 1.0f / eta);

            if (rand01(rng) < Fr) {
                wi = reflect(rayDir, faceN);
                rayOrig = hitPos + faceN * 1e-4f;
            } else {
                wi = refract(rayDir, faceN, eta);
                if (dot(wi, wi) < 0.5f) {
                    wi = reflect(rayDir, faceN);
                    rayOrig = hitPos + faceN * 1e-4f;
                } else {
                    rayOrig = hitPos - faceN * 1e-4f;
                }
            }
            // Delta BSDF: f/pdf = 1, throughput unchanged.
            // Glass hits don't count against bounce budget — use a separate limiter.
            rayDir = normalize(wi);
            if (++glassDepth >= 16) break;
            if (bounce > 0) --bounce;
            continue;

        } else if (mat.type == kMatGGX && mat.roughness < 0.95f) {
            float  a2     = mat.roughness * mat.roughness;
            a2 = a2 * a2;
            float3 wmLocal = sampleGGX(rand2(rng), a2);
            float3 wh      = toWorld(wmLocal, n);
            if (dot(wh, n) < 0.0f) wh = -1.0f * wh;
            wi = reflect(-1.0f * wo, wh);
            if (dot(wi, n) <= 0.0f) break;

            float cosII = dot(n, wi);
            float cosO  = dot(n, wo);
            float cosH  = dot(n, wh);
            float D     = ggxD(cosH, a2);
            float G     = ggxG1(cosO, a2) * ggxG1(cosII, a2);
            float3 F0   = lerp3(make_float3(0.04f, 0.04f, 0.04f), baseColor, mat.metalness);
            float3 F    = schlick(dot(wi, wh), F0);
            bsdfPdf = D * cosH / fmaxf(1e-7f, 4.0f * dot(wo, wh));
            float3 spec = D * G * F * (1.0f / fmaxf(1e-7f, 4.0f * cosO * cosII));
            float3 diff = (1.0f - mat.metalness) * baseColor * (1.0f / CUDART_PI_F);
            bsdfF = diff + spec;
        } else {
            wi      = cosineSampleHemisphere(rand2(rng), n);
            bsdfPdf = fmaxf(1e-7f, dot(n, wi)) / CUDART_PI_F;
            bsdfF   = baseColor * (1.0f / CUDART_PI_F);
        }

        float cosI = dot(n, wi);
        if (cosI <= 0.0f || bsdfPdf <= 0.0f) break;
        throughput *= bsdfF * cosI * (1.0f / bsdfPdf);

        rayOrig = hitPos + n * 1e-4f;
        rayDir  = normalize(wi);
    }

    rAcc += L.x;
    gAcc += L.y;
    bAcc += L.z;
    float lum = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
    lumSqAcc += lum * lum;

    } // end sample loop

    GpuAccumPixel& out = params.accum[pixelIdx];
    out.r        += rAcc;
    out.g        += gAcc;
    out.b        += bAcc;
    out.weight   += float(nSamples);
    out.sumLumSq += lumSqAcc;
}
