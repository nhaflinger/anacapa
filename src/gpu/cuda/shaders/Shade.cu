// Shade.cu — OptiX raygen / closesthit / miss programs.
//
// Compiled to PTX (see anacapa_compile_ptx in src/gpu/cuda/CMakeLists.txt) and
// loaded at runtime by CudaPathIntegrator's pipeline.  The OptixDeviceContext
// is created in CudaContext; the GAS in CudaAccelStructure provides the
// traversable handle (single triangle GAS, motion-aware when any mesh has
// motion keys).
//
// Programs:
//   __raygen__rg     : pixel + sample loop, primary ray gen, bounce loop,
//                      BSDF / NEE evaluation, accumulation.  Uses optixTrace
//                      for both surface and shadow rays.
//   __closesthit__ch : writes hit info (primIdx, barycentrics, t) via 5
//                      payload registers.
//   __miss__ms       : sets payload p0 = 0 to signal miss.
//
// Per-ray time is sampled in raygen as t = lerp(shutterOpen, shutterClose,
// rand01()) and passed to optixTrace; the motion-aware GAS interpolates
// triangle vertex positions in hardware.

#include <optix.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <float.h>

#include "SharedTypes.h"
#include "LaunchParams.h"

extern "C" __constant__ LaunchParams params;

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
// float3 math helpers
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
static __forceinline__ __device__ float  compmax(float3 v) {
    return fmaxf(v.x, fmaxf(v.y, v.z));
}
static __forceinline__ __device__ float3 lerp3(float3 a, float3 b, float t) {
    return a * (1.0f - t) + b * t;
}

// ---------------------------------------------------------------------------
// optixTrace wrapper — surface query.  Payload layout:
//   p0 = hit valid (0 or 1)
//   p1 = primitive index (global triangle index in our single GAS)
//   p2 = barycentric u (bits)
//   p3 = barycentric v (bits)
//   p4 = ray-t (bits)
// On hit, meshID is looked up from params.triMeshIDs[primID] in the caller.
// ---------------------------------------------------------------------------
struct TraceResult {
    uint32_t valid;
    uint32_t meshID;
    uint32_t primID;
    float2   bary;
    float    t;
};

static __forceinline__ __device__
TraceResult trace(float3 orig, float3 dir, float tMin, float tMax, float rayTime)
{
    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0;
    optixTrace(
        params.handle,
        orig, dir,
        tMin, tMax,
        rayTime,
        OptixVisibilityMask(0xFF),
        OPTIX_RAY_FLAG_NONE,
        /*SBToffset=*/0u,
        /*SBTstride=*/1u,
        /*missSBT=*/0u,
        p0, p1, p2, p3, p4);

    TraceResult r{};
    r.valid = p0;
    if (p0) {
        r.primID = p1;
        r.bary.x = __uint_as_float(p2);
        r.bary.y = __uint_as_float(p3);
        r.t      = __uint_as_float(p4);
        r.meshID = params.triMeshIDs[p1];
    }
    return r;
}

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 interpolateNormal(uint32_t globalPrimId, float2 bary,
                         const GpuFloat3* normals,
                         const uint32_t*  indices)
{
    uint32_t i0 = indices[globalPrimId * 3 + 0];
    uint32_t i1 = indices[globalPrimId * 3 + 1];
    uint32_t i2 = indices[globalPrimId * 3 + 2];
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
static __forceinline__ __device__ float3 evalEnvmap(float3 wo) {
    float3 r0 = make3(params.cam.envRot0);
    float3 r1 = make3(params.cam.envRot1);
    float3 r2 = make3(params.cam.envRot2);
    float3 local = make_float3(dot(r0, wo), dot(r1, wo), dot(r2, wo));
    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, local.y)));
    float phi   = atan2f(local.x, local.z);
    if (phi < 0.0f) phi += 2.0f * CUDART_PI_F;
    float u = phi  / (2.0f * CUDART_PI_F);
    float v = theta / CUDART_PI_F;
    float4 c = tex2D<float4>(params.envTexture, u, v);
    return fmaxf3(make_float3(0.0f, 0.0f, 0.0f),
                  make_float3(c.x, c.y, c.z)) * params.cam.envIntensity;
}

// ---------------------------------------------------------------------------
// HDRI importance sampling — mirrors the Metal version
// ---------------------------------------------------------------------------

// Binary search into a normalized 1D CDF of (n+1) floats.  Returns bin
// index in [0,n), remapped u in [0,1), and bin probability (= pdf).
static __forceinline__ __device__
uint32_t sampleCDF1D(const float* cdf, uint32_t n, float u,
                     float& uRemapped, float& prob)
{
    uint32_t lo = 0, hi = n;
    while (lo < hi) {
        uint32_t mid = (lo + hi) >> 1;
        if (cdf[mid + 1] <= u) lo = mid + 1;
        else                   hi = mid;
    }
    uint32_t idx = (lo < n - 1) ? lo : n - 1;
    float    binProb = cdf[idx + 1] - cdf[idx];
    uRemapped = (binProb > 1e-7f)
              ? fminf(fmaxf((u - cdf[idx]) / binProb, 0.0f), 1.0f - 1e-7f)
              : 0.0f;
    prob = binProb;
    return idx;
}

// Solid-angle PDF for sampling world direction `dir` from the HDRI
// importance distribution.  Returns 0 outside the texture's domain.
static __forceinline__ __device__
float evalEnvPdf(float3 dir,
                 const float* marginalCdf,
                 const float* conditionalCdf)
{
    float3 r0 = make3(params.cam.envRot0);
    float3 r1 = make3(params.cam.envRot1);
    float3 r2 = make3(params.cam.envRot2);
    float3 local = make_float3(dot(r0, dir), dot(r1, dir), dot(r2, dir));

    float theta = acosf(fmaxf(-1.0f, fminf(1.0f, local.y)));
    float phi   = atan2f(local.x, local.z);
    if (phi < 0.0f) phi += 2.0f * CUDART_PI_F;

    uint32_t W = params.cam.envMapWidth;
    uint32_t H = params.cam.envMapHeight;
    uint32_t col = (uint32_t)(phi / (2.0f * CUDART_PI_F) * float(W));
    uint32_t row = (uint32_t)(theta / CUDART_PI_F * float(H));
    if (col >= W) col = W - 1u;
    if (row >= H) row = H - 1u;

    float sinTheta = fmaxf(sinf(theta), 1e-6f);
    float pdfRow   = marginalCdf[row + 1] - marginalCdf[row];
    float pdfCol   = conditionalCdf[row * (W + 1) + col + 1]
                   - conditionalCdf[row * (W + 1) + col];
    return (pdfRow * pdfCol * float(W * H))
           / (2.0f * CUDART_PI_F * CUDART_PI_F * sinTheta);
}

// Sample a direction from the 2D HDRI importance distribution.  Returns
// world-space direction; writes solid-angle PDF to `pdfOut`.
static __forceinline__ __device__
float3 sampleEnvDirection(float2 u,
                          const float* marginalCdf,
                          const float* conditionalCdf,
                          float& pdfOut)
{
    uint32_t W = params.cam.envMapWidth;
    uint32_t H = params.cam.envMapHeight;

    float    uRow, probRow, uCol, probCol;
    uint32_t row = sampleCDF1D(marginalCdf,                   H, u.y, uRow, probRow);
    uint32_t col = sampleCDF1D(conditionalCdf + row * (W + 1), W, u.x, uCol, probCol);

    float v     = (float(row) + uRow) / float(H);
    float uu    = (float(col) + uCol) / float(W);
    float theta = v * CUDART_PI_F;
    float phi   = uu * 2.0f * CUDART_PI_F;
    float sinT  = sinf(theta);
    float cosT  = cosf(theta);

    // envmap-local direction → world.  cam.envRot rows map world→env, so
    // the transpose maps env→world.
    float3 envDir = make_float3(sinT * sinf(phi), cosT, sinT * cosf(phi));
    float3 r0 = make3(params.cam.envRot0);
    float3 r1 = make3(params.cam.envRot1);
    float3 r2 = make3(params.cam.envRot2);
    float3 worldDir = make_float3(
        r0.x * envDir.x + r1.x * envDir.y + r2.x * envDir.z,
        r0.y * envDir.x + r1.y * envDir.y + r2.y * envDir.z,
        r0.z * envDir.x + r1.z * envDir.y + r2.z * envDir.z);

    float sinTabs = fmaxf(sinT, 1e-6f);
    pdfOut = (probRow * probCol * float(W * H))
           / (2.0f * CUDART_PI_F * CUDART_PI_F * sinTabs);
    if (pdfOut <= 0.0f) pdfOut = 1e-7f;
    return worldDir;
}

// ---------------------------------------------------------------------------
// Shadow transmittance — steps through glass; returns (0,0,0) if blocked.
// Uses optixTrace; the rayTime is constant across the loop (one shutter
// sample per primary ray, propagated through bounces).
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 shadowTransmittance(float3 origin, float3 dir, float tMax, float rayTime)
{
    float3 T    = make_float3(1.0f, 1.0f, 1.0f);
    float3 orig = origin;
    float  remaining = tMax;

    for (int step = 0; step < 8; ++step) {
        TraceResult hit = trace(orig, dir, 1e-4f, remaining, rayTime);
        if (!hit.valid) break;

        uint32_t matIdx = (hit.meshID < params.numMaterials) ? hit.meshID : 0u;
        GpuMaterial mat = params.materials[matIdx];

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
// MIS helpers
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float powerHeuristic(float pdfF, float pdfG) {
    float f = pdfF, g = pdfG;
    return (f * f) / (f * f + g * g + 1e-9f);
}

// Solid-angle PDF that the NEE rect-light strategy would use to sample
// direction wi from hitPos toward this light.  Returns 0 if wi doesn't
// land within the light's quad (so the strategy could not have produced
// it) — that case contributes 0 to the MIS denominator.
static __forceinline__ __device__
float rectLightSolidAnglePdf(const GpuLight& light,
                             float3 hitPos, float3 wi, float dist)
{
    float3 lightN = make3(light.normal);
    float  cosL   = dot(-1.0f * wi, lightN);
    if (cosL <= 0.0f) return 0.0f;
    float3 toHit = hitPos - make3(light.position);
    float3 uH    = make3(light.uHalf);
    float3 vH    = make3(light.vHalf);
    float  uLen  = sqrtf(dot(uH, uH));
    float  vLen  = sqrtf(dot(vH, vH));
    if (uLen < 1e-7f || vLen < 1e-7f) return 0.0f;
    float uCoord = dot(toHit, uH * (1.0f / uLen));
    float vCoord = dot(toHit, vH * (1.0f / vLen));
    if (fabsf(uCoord) > uLen || fabsf(vCoord) > vLen) return 0.0f;
    return (dist * dist) / (cosL * light.area);
}

// ---------------------------------------------------------------------------
// Direct light sampling
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 sampleDirect(float3 hitPos, float3 n, float3 wo,
                    uint32_t matType, float3 baseColor,
                    float roughness, float metalness,
                    uint32_t& rng, float rayTime)
{
    if (params.numLights == 0) return make_float3(0.0f, 0.0f, 0.0f);

    uint32_t lightIdx = uint32_t(rand01(rng) * float(params.numLights)) % params.numLights;
    const GpuLight& light = params.lights[lightIdx];
    float lightPick = 1.0f / float(params.numLights);

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
        // HDRI importance sampling when CDF tables are present, otherwise
        // fall back to a cosine-hemisphere prior.
        if (params.cam.envMapWidth > 0 && params.envMarginalCdf != nullptr
                && params.envConditionalCdf != nullptr) {
            float envPdf = 0.0f;
            wi = sampleEnvDirection(rand2(rng),
                                    params.envMarginalCdf,
                                    params.envConditionalCdf,
                                    envPdf);
            pdfL = envPdf * lightPick;
        } else {
            wi   = cosineSampleHemisphere(rand2(rng), n);
            float cosW = fmaxf(1e-7f, dot(n, wi));
            pdfL = (cosW / CUDART_PI_F) * lightPick;
        }
        tMax = 1e9f;
        Li   = (params.envTexture != 0) ? evalEnvmap(wi) : make3(params.cam.envLe);
    } else {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    float cosI = dot(n, wi);
    if (cosI <= 0.0f || pdfL <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 shadowOrigin = hitPos + n * 1e-4f;
    float3 Tr = shadowTransmittance(shadowOrigin, wi, tMax, rayTime);
    if (compmax(Tr) <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    float3 f = make_float3(0.0f, 0.0f, 0.0f);
    if (matType == kMatLambertian) {
        f = baseColor * (1.0f / CUDART_PI_F);
    } else if (matType == kMatGGX) {
        float3 wh   = normalize(wo + wi);
        float  cosH = fmaxf(0.0f, dot(n, wh));
        float  cosO = fmaxf(0.0f, dot(n, wo));
        float  cosII= fmaxf(0.0f, dot(n, wi));
        float  neeR = fmaxf(roughness, 0.2f);
        float  a    = neeR * neeR;
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

// ===========================================================================
// OptiX programs
// ===========================================================================

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint32_t tx = idx.x;
    const uint32_t ty = idx.y;
    if (tx >= params.cam.tileWidth || ty >= params.cam.tileHeight) return;

    const uint32_t px = params.cam.tileX0 + tx;
    const uint32_t py = params.cam.tileY0 + ty;
    if (px >= params.cam.imageWidth || py >= params.cam.imageHeight) return;

    const uint32_t pixelIdx       = ty * params.cam.tileWidth + tx;
    const uint32_t globalPixelIdx = py * params.cam.imageWidth + px;
    const uint32_t nSamples       = params.sampleBatch.batchSize;

    const float3 origin = make3(params.cam.origin);
    const float3 horiz  = make3(params.cam.horizontal);
    const float3 vert   = make3(params.cam.vertical);
    const float3 ll     = make3(params.cam.lowerLeft);

    float rAcc = 0.0f, gAcc = 0.0f, bAcc = 0.0f, lumSqAcc = 0.0f;

    for (uint32_t s = 0; s < nSamples; ++s) {
        uint32_t rng = pcg(pcg(globalPixelIdx) ^
                           ((params.sampleBatch.sampleStart + s) * 2654435761u));

        // Per-primary-ray shutter time.  Same value used for every bounce of
        // this primary so a single time slice is rendered consistently.
        float rayTime = params.cam.shutterOpen;
        if (params.cam.shutterClose > params.cam.shutterOpen) {
            rayTime = params.cam.shutterOpen
                    + rand01(rng) * (params.cam.shutterClose - params.cam.shutterOpen);
        }

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
        // MIS state: prevWasDelta=true on the first hit and after any delta
        // (glass) bounce so emitter Le on that vertex gets weight=1 (no NEE
        // was attempted at the previous vertex, no double-count risk).
        // Otherwise weight = powerHeuristic(prevBsdfPdf, lightSolidAnglePdf)
        // balances emitter-on-bounce against NEE for the same vertex.
        float  prevBsdfPdf  = 0.0f;
        float3 prevN        = make_float3(0.0f, 0.0f, 0.0f);
        bool   prevWasDelta = true;

        for (uint32_t bounce = 0; bounce <= params.cam.maxDepth; ++bounce) {
            TraceResult hit = trace(rayOrig, rayDir, 1e-4f, 1e10f, rayTime);

            if (!hit.valid) {
                // Match the CPU PathIntegrator: scenes without an environment
                // light produce black for escaped rays.  This branch used to
                // synthesise a blue/white sky gradient, which leaked into
                // Cornell-box indirect bounces through the open front face —
                // making GPU renders ~2× brighter and blue-shifted vs CPU.
                float3 envColor = make_float3(0.0f, 0.0f, 0.0f);
                if (params.cam.hasEnvLight && params.envTexture != 0) {
                    envColor = evalEnvmap(rayDir);
                } else if (params.cam.hasEnvLight) {
                    envColor = make3(params.cam.envLe);
                }
                if (compmax(envColor) > 0.0f) {
                    float weight = 1.0f;
                    if (!prevWasDelta && bounce > 0) {
                        // NEE dome-sampling PDF.  HDRI importance sampling when CDFs
                        // are uploaded; otherwise the cosine-hemisphere prior.
                        float lpdf = 0.0f;
                        for (uint32_t li = 0; li < params.numLights; ++li) {
                            if (params.lights[li].type == kLightDome) {
                                float domePdf;
                                if (params.cam.envMapWidth > 0
                                        && params.envMarginalCdf != nullptr
                                        && params.envConditionalCdf != nullptr) {
                                    domePdf = evalEnvPdf(rayDir,
                                                         params.envMarginalCdf,
                                                         params.envConditionalCdf);
                                } else {
                                    float cosW = fmaxf(0.0f, dot(rayDir, prevN));
                                    domePdf = cosW / CUDART_PI_F;
                                }
                                lpdf += domePdf / float(params.numLights);
                            }
                        }
                        if (lpdf > 0.0f) weight = powerHeuristic(prevBsdfPdf, lpdf);
                    }
                    L += throughput * envColor * weight;
                }
                break;
            }

            float3 hitPos = rayOrig + rayDir * hit.t;

            float3 geomN = interpolateNormal(hit.primID, hit.bary,
                                             params.normals, params.indices);
            float3 n = geomN;
            if (dot(-1.0f * rayDir, n) < 0.0f) n = -1.0f * n;

            uint32_t matIdx  = (hit.meshID < params.numMaterials) ? hit.meshID : 0u;
            GpuMaterial mat  = params.materials[matIdx];
            float3 baseColor = make3(mat.baseColor);
            float3 emissive  = make3(mat.emissive);

            // Emitter Le with MIS weight against NEE (rect-light) sampling.
            // prevWasDelta=true on the first hit and after any delta bounce
            // gets weight=1.  Otherwise the weight is the power heuristic
            // between the BSDF PDF that produced this ray and the rect-light
            // PDF, summed over all rect lights with uniform selection.
            if (mat.type == kMatEmissive) {
                float weight = 1.0f;
                if (!prevWasDelta && bounce > 0) {
                    float lpdf = 0.0f;
                    for (uint32_t li = 0; li < params.numLights; ++li) {
                        if (params.lights[li].type == kLightRect) {
                            float spdf = rectLightSolidAnglePdf(
                                params.lights[li], hitPos, rayDir, hit.t);
                            if (spdf > 0.0f) lpdf += spdf / float(params.numLights);
                        }
                    }
                    if (lpdf > 0.0f) weight = powerHeuristic(prevBsdfPdf, lpdf);
                }
                L += throughput * emissive * weight;
                break;
            }

            float3 wo = -1.0f * rayDir;
            if (mat.type != kMatGlass) {
                L += throughput * sampleDirect(hitPos, n, wo,
                                               mat.type, baseColor,
                                               mat.roughness, mat.metalness,
                                               rng, rayTime);
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
                rayDir = normalize(wi);
                // Glass is delta: emitter Le on next hit gets weight=1.
                prevBsdfPdf  = 1.0f;
                prevN        = faceN;
                prevWasDelta = true;
                if (++glassDepth >= 16) break;
                if (bounce > 0) --bounce;
                continue;

            } else if (mat.type == kMatGGX && mat.roughness < 0.95f) {
                float  alpha  = mat.roughness * mat.roughness;
                float  alpha2 = alpha * alpha;
                float3 F0     = lerp3(make_float3(0.04f, 0.04f, 0.04f), baseColor, mat.metalness);

                float lumSpec = (F0.x + F0.y + F0.z) / 3.0f;
                float lumDiff = (1.0f - mat.metalness) * (baseColor.x + baseColor.y + baseColor.z) / 3.0f;
                float pSpec   = lumSpec / fmaxf(1e-4f, lumSpec + lumDiff);
                float pDiff   = 1.0f - pSpec;

                float3 wh;
                if (rand01(rng) < pSpec) {
                    float3 wmLocal = sampleGGX(rand2(rng), alpha2);
                    wh = toWorld(wmLocal, n);
                    if (dot(wh, n) < 0.0f) wh = -1.0f * wh;
                    wi = reflect(-1.0f * wo, wh);
                } else {
                    wi = cosineSampleHemisphere(rand2(rng), n);
                    wh = normalize(wo + wi);
                }
                if (dot(wi, n) <= 0.0f) break;

                float cosII = dot(n, wi);
                float cosO  = dot(n, wo);
                float cosH  = fmaxf(0.0f, dot(n, wh));
                float D     = ggxD(cosH, alpha2);
                float G     = ggxG1(cosO, alpha2) * ggxG1(cosII, alpha2);
                float3 F    = schlick(fmaxf(0.0f, dot(wi, wh)), F0);
                float3 spec = D * G * F * (1.0f / fmaxf(1e-7f, 4.0f * cosO * cosII));
                float3 diff = (1.0f - mat.metalness) * baseColor * (1.0f / CUDART_PI_F);
                bsdfF = diff + spec;

                float ggxPdf = D * cosH / fmaxf(1e-7f, 4.0f * dot(wo, wh));
                float cosPdf = cosII / CUDART_PI_F;
                bsdfPdf = pSpec * ggxPdf + pDiff * cosPdf;
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
            // Diffuse / glossy bounce: emitter Le at next vertex must be
            // MIS-weighted against the NEE strategy that just sampled the
            // same vertex.
            prevBsdfPdf  = bsdfPdf;
            prevN        = n;
            prevWasDelta = false;
        }

        // Firefly clamp: scale L down if its luminance exceeds the threshold.
        // Done before per-pixel accumulation so the variance tracker (lumSq)
        // sees the clamped value too.
        float lum = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
        if (params.cam.fireflyClamp > 0.0f && lum > params.cam.fireflyClamp) {
            float k = params.cam.fireflyClamp / lum;
            L = L * k;
            lum = params.cam.fireflyClamp;
        }
        rAcc += L.x;
        gAcc += L.y;
        bAcc += L.z;
        lumSqAcc += lum * lum;
    }

    GpuAccumPixel& out = params.accum[pixelIdx];
    out.r        += rAcc;
    out.g        += gAcc;
    out.b        += bAcc;
    out.weight   += float(nSamples);
    out.sumLumSq += lumSqAcc;
}

extern "C" __global__ void __closesthit__ch()
{
    const uint32_t prim  = optixGetPrimitiveIndex();
    const float2   bary  = optixGetTriangleBarycentrics();
    const float    tHit  = optixGetRayTmax();
    optixSetPayload_0(1u);
    optixSetPayload_1(prim);
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(__float_as_uint(tHit));
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0u);
}
