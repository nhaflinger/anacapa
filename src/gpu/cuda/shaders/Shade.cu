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
    uint32_t meshID;       // resolved scene material slot (from triMeshIDs[] for tri hits)
    uint32_t primID;       // primitive index local to the BLAS instance
    uint32_t instanceID;   // IAS instance index (0 = mesh GAS, 1 = hair GAS)
    float2   bary;
    float    t;
};

static __forceinline__ __device__
TraceResult trace(float3 orig, float3 dir, float tMin, float tMax, float rayTime)
{
    uint32_t p0 = 0, p1 = 0, p2 = 0, p3 = 0, p4 = 0, p5 = 0;
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
        p0, p1, p2, p3, p4, p5);

    TraceResult r{};
    r.valid = p0;
    if (p0) {
        r.primID     = p1;
        r.bary.x     = __uint_as_float(p2);
        r.bary.y     = __uint_as_float(p3);
        r.t          = __uint_as_float(p4);
        r.instanceID = p5;
        // Triangle hits (instance 0) look up the per-triangle meshID;
        // hair hits use the per-strand material index from GpuHairTri.
        const bool isHair = (params.cam.hairMeshBaseID != 0xFFFFFFFFu
                             && p5 >= params.cam.hairMeshBaseID);
        r.meshID = isHair ? 0u : params.triMeshIDs[p1];
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

        // Hair is fully opaque to shadow rays (no transmission lobe yet).
        const bool isHair = (params.cam.hairMeshBaseID != 0xFFFFFFFFu
                              && hit.instanceID >= params.cam.hairMeshBaseID);
        if (isHair) return make_float3(0.0f, 0.0f, 0.0f);

        uint32_t matIdx = (hit.meshID < params.numMaterials) ? hit.meshID : 0u;
        GpuMaterial mat = params.materials[matIdx];

        if (mat.type == kMatGlass) {
            // When the photon map is active, glass is opaque to shadow rays:
            // the map is the sole carrier of transmitted/refracted light, so
            // attenuating NEE through glass too would double-count.  Without
            // a map, attenuate by transmission so glass still casts soft
            // colored shadows.
            if (params.cam.photonMapEnabled) return make_float3(0.0f, 0.0f, 0.0f);
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
// Energy compensation — mirrors the StandardSurface CPU implementation.
// Same LUTs uploaded by the host (see CudaPathIntegrator::prepare).
// ---------------------------------------------------------------------------

// Bilinear lookup of E_spec(cos_theta_o, roughness) — directional-hemispherical
// reflectance of the GGX dielectric spec lobe.  Used both for the spec/diff
// energy-conservation balance and as part of the Kulla-Conty multi-scatter
// compensation denominator.
static __forceinline__ __device__
float specAlbedoLookup(float cosO, float roughness) {
    if (params.specAlbedoLUT == nullptr) return 0.0f;
    const int N_COS = (int)params.specLUTCosBins;
    const int N_R   = (int)params.specLUTRoughBins;
    cosO      = fminf(1.0f, fmaxf(0.0f, cosO));
    roughness = fminf(1.0f, fmaxf(0.0f, roughness));
    float fc  = cosO      * float(N_COS - 1);
    float fr  = roughness * float(N_R   - 1);
    int   ic0 = min(N_COS - 1, (int)fc);
    int   ir0 = min(N_R   - 1, (int)fr);
    int   ic1 = min(N_COS - 1, ic0 + 1);
    int   ir1 = min(N_R   - 1, ir0 + 1);
    float tc  = fc - float(ic0);
    float tr  = fr - float(ir0);
    float v00 = params.specAlbedoLUT[ic0 * N_R + ir0];
    float v01 = params.specAlbedoLUT[ic0 * N_R + ir1];
    float v10 = params.specAlbedoLUT[ic1 * N_R + ir0];
    float v11 = params.specAlbedoLUT[ic1 * N_R + ir1];
    return (v00 * (1.0f - tc) + v10 * tc) * (1.0f - tr)
         + (v01 * (1.0f - tc) + v11 * tc) * tr;
}

// 1D lookup of cosine-weighted-average E_spec — used as the (1 - E_avg)
// Kulla-Conty denominator.
static __forceinline__ __device__
float specAvgAlbedoLookup(float roughness) {
    if (params.specAvgAlbedoLUT == nullptr) return 0.0f;
    const int N_R = (int)params.specLUTRoughBins;
    roughness     = fminf(1.0f, fmaxf(0.0f, roughness));
    float fr      = roughness * float(N_R - 1);
    int   i0      = min(N_R - 1, (int)fr);
    int   i1      = min(N_R - 1, i0 + 1);
    float t       = fr - float(i0);
    return params.specAvgAlbedoLUT[i0] * (1.0f - t)
         + params.specAvgAlbedoLUT[i1] * t;
}

// Kulla-Conty multi-scatter compensation BRDF for a GGX layer with
// effective F0 = F_ms.  Adds back the energy single-scatter masking
// loses; for dielectrics (F_ms ≈ 0.04) the magnitude is small, for
// metals (F_ms ≈ baseColor) it's large at high roughness.
static __forceinline__ __device__
float3 evalGGXMs(float cosO, float cosI, float roughness, float3 F_ms) {
    if (cosO <= 0.0f || cosI <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);
    float Eo = specAlbedoLookup(cosO, roughness);
    float Ei = specAlbedoLookup(cosI, roughness);
    float Ea = specAvgAlbedoLookup(roughness);
    if (Ea >= 0.999f) return make_float3(0.0f, 0.0f, 0.0f);
    float k = (1.0f - Eo) * (1.0f - Ei) / (CUDART_PI_F * (1.0f - Ea));
    return F_ms * k;
}

// Disney/Burley diffuse with roughness-dependent retro-reflection.
// Reduces to plain Lambertian when both cos_i and cos_o = 1; brightens
// at grazing for rough surfaces (Fd90 = 2.5 at α=1) to compensate for
// the diffuse energy lost to single-scatter spec.
static __forceinline__ __device__
float3 disneyDiffuseLobe(float3 wo, float3 wi, float3 n,
                          float3 baseColor, float roughness)
{
    float cosI = fmaxf(0.0f, dot(n, wi));
    float cosO = fmaxf(0.0f, dot(n, wo));
    float3 wh  = normalize(wo + wi);
    float cosD = fmaxf(0.0f, dot(wi, wh));
    float Fd90 = 0.5f + 2.0f * cosD * cosD * roughness;
    float Fview  = 1.0f + (Fd90 - 1.0f) * powf(1.0f - cosO, 5.0f);
    float Flight = 1.0f + (Fd90 - 1.0f) * powf(1.0f - cosI, 5.0f);
    return baseColor * (1.0f / CUDART_PI_F) * Fview * Flight;
}

// Layered StandardSurface BSDF — single GGX spec lobe (with metallic
// blending into F0) + Disney diffuse, weighted by the CPU's energy-
// conservation scheme: diffuse = (1 - metal) * (1 - spec * E_spec) *
// disney, plus full single-scatter spec + Kulla-Conty multi-scatter
// compensation.  Matches the CPU StandardSurface output up to the same
// systematic GPU/CPU offset that exists on plain Lambertian.
static __forceinline__ __device__
float3 evalLayeredBSDF(float3 wo, float3 wi, float3 n,
                        float3 baseColor, float roughness,
                        float metalness, float specular)
{
    float3 wh    = normalize(wo + wi);
    float  cosH  = fmaxf(0.0f, dot(n, wh));
    float  cosO  = fmaxf(0.0f, dot(n, wo));
    float  cosII = fmaxf(0.0f, dot(n, wi));
    if (cosO <= 0.0f || cosII <= 0.0f) return make_float3(0.0f, 0.0f, 0.0f);

    float  alpha  = fmaxf(1e-4f, roughness * roughness);
    float  a2     = alpha * alpha;
    float  D      = ggxD(cosH, a2);
    float  G      = ggxG1(cosO, a2) * ggxG1(cosII, a2);
    float  vdotH  = fmaxf(0.0f, dot(wi, wh));
    float  invDen = 1.0f / fmaxf(1e-7f, 4.0f * cosO * cosII);

    float3 F0     = lerp3(make_float3(0.04f, 0.04f, 0.04f), baseColor, metalness);
    float3 spec   = D * G * schlick(vdotH, F0) * invDen
                  + evalGGXMs(cosO, cosII, roughness, F0);

    float  E_spec = specAlbedoLookup(cosO, roughness);
    float  diffW  = (1.0f - metalness) * (1.0f - specular * E_spec);
    float3 diff   = disneyDiffuseLobe(wo, wi, n, baseColor, roughness) * diffW;

    return diff + spec;
}

// ===========================================================================
// Marschner (2003) hair BSDF — three lobes (R / TT / TRT).
// Port of Shade.metal, which itself follows PBRT-v4's Hair.cpp.
// ===========================================================================
static __forceinline__ __device__
float mh_I0(float x) {
    float sum = 0.f, x2i = 1.f, denom = 1.f;
    #pragma unroll
    for (int i = 0; i < 12; ++i) {
        if (i > 0) { x2i *= x * x; denom *= float(i) * float(i) * 4.f; }
        sum += x2i / denom;
    }
    return sum;
}
static __forceinline__ __device__
float mh_logI0(float x) {
    if (x > 12.f)
        return x + 0.5f * (-logf(2.f * CUDART_PI_F) - logf(x) + 1.f / (8.f * x));
    return logf(mh_I0(x));
}
static __forceinline__ __device__
float mh_Mp(float cosI, float sinI, float cosR, float sinR, float v) {
    v = fmaxf(v, 1e-5f);
    float a = cosI * cosR / v;
    float b = sinI * sinR / v;
    if (v <= 0.1f)
        return expf(mh_logI0(a) - b - 1.f/v + 0.6931472f + logf(0.5f/v));
    return expf(-b) * mh_I0(a) / (2.f * v * sinhf(1.f/v));
}
static __forceinline__ __device__
float mh_logistic(float x, float s) {
    float ex = expf(-fabsf(x) / s);
    return ex / (s * (1.f + ex) * (1.f + ex));
}
static __forceinline__ __device__
float mh_logisticCDF(float x, float s) { return 1.f / (1.f + expf(-x / s)); }
static __forceinline__ __device__
float mh_trimmedLogistic(float x, float s) {
    return mh_logistic(x, s)
         / (mh_logisticCDF(CUDART_PI_F, s) - mh_logisticCDF(-CUDART_PI_F, s));
}
static __forceinline__ __device__
float mh_sampleTrimmedLogistic(float u, float s) {
    float a = mh_logisticCDF(-CUDART_PI_F, s);
    float b = mh_logisticCDF( CUDART_PI_F, s);
    return fminf(CUDART_PI_F,
                 fmaxf(-CUDART_PI_F,
                       -s * logf(1.f / (a + u * (b - a)) - 1.f)));
}
static __forceinline__ __device__
float mh_Phi(int p, float gO, float gT) {
    return 2.f * float(p) * gT - 2.f * gO + float(p) * CUDART_PI_F;
}
static __forceinline__ __device__
float mh_wrapPhi(float x) {
    x = fmodf(x, 2.f * CUDART_PI_F);
    if (x >  CUDART_PI_F) x -= 2.f * CUDART_PI_F;
    if (x < -CUDART_PI_F) x += 2.f * CUDART_PI_F;
    return x;
}
static __forceinline__ __device__
float mh_Np(float phi, int p, float s, float gO, float gT) {
    return mh_trimmedLogistic(mh_wrapPhi(phi - mh_Phi(p, gO, gT)), s);
}
static __forceinline__ __device__
float hairLum(float3 c) { return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f; }

// Lobe attenuation A_p — writes R/TT/TRT into ap[0..2].
static __forceinline__ __device__
void mh_Ap(float cosThetaO, float eta, float h, float3 sigma_a, float3 ap[3]) {
    float sin2O = fmaxf(0.f, 1.f - cosThetaO * cosThetaO);
    float etaP  = sqrtf(fmaxf(0.f, eta * eta - sin2O)) / fmaxf(cosThetaO, 1e-5f);
    float sinGT = fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h / fmaxf(etaP, 1e-5f)));
    float cosGT = sqrtf(fmaxf(0.f, 1.f - sinGT * sinGT));
    float3 T    = make_float3(expf(-sigma_a.x * 2.f * cosGT),
                              expf(-sigma_a.y * 2.f * cosGT),
                              expf(-sigma_a.z * 2.f * cosGT));
    float cosGO = sqrtf(fmaxf(0.f, 1.f - h * h));
    float fr    = fresnelDielectric(fmaxf(0.f, cosThetaO * cosGO), eta);
    ap[0] = make_float3(fr, fr, fr);
    ap[1] = (1.f - fr) * (1.f - fr) * T;
    ap[2] = ap[1] * T * fr;
}

struct HairPrecomp { float3 v; float s; float3 alphaR; };

static __forceinline__ __device__
HairPrecomp makeHairPrecomp(const GpuHairMaterial& hm) {
    HairPrecomp hp;
    float bm = fminf(1.f, fmaxf(1e-3f, hm.beta_m));
    float v0 = 0.726f * bm + 0.812f * bm * bm + 3.7f * powf(bm, 20.f);
    v0 *= v0;
    hp.v = make_float3(v0, v0 * 0.25f, v0 * 4.f);

    float bn = fminf(1.f, fmaxf(1e-3f, hm.beta_n));
    hp.s = 0.626657069f * (0.265f * bn + 1.194f * bn * bn + 5.372f * powf(bn, 22.f));

    float ar = hm.alpha * (CUDART_PI_F / 180.f);
    hp.alphaR = make_float3(-ar, ar * 0.5f, -ar * 1.5f);
    return hp;
}

// Full Marschner BSDF evaluation (R + TT + TRT, no cosine factor).
static __forceinline__ __device__
float3 evalMarschnerLobes(
    float sinThetaO, float cosThetaO,
    float sinThetaI, float cosThetaI,
    float phi, float h, float3 sigma_a, float eta,
    float3 v, float s, float3 alphaR)
{
    float cosThetaD = sqrtf(fmaxf(0.f,
        0.5f * (1.f + cosThetaO * cosThetaI + sinThetaO * sinThetaI)));
    float denom = fmaxf(1e-5f, cosThetaD * cosThetaD);

    float sin2O  = 1.f - cosThetaO * cosThetaO;
    float etaP   = sqrtf(fmaxf(0.f, eta * eta - sin2O)) / fmaxf(cosThetaO, 1e-5f);
    float gammaO = asinf(fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h)));
    float sinGT  = fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h / fmaxf(etaP, 1e-5f)));
    float gammaT = asinf(sinGT);

    float3 ap[3]; mh_Ap(cosThetaO, eta, h, sigma_a, ap);

    float3 fsum = make_float3(0.f, 0.f, 0.f);
    const float ar[3] = { alphaR.x, alphaR.y, alphaR.z };
    const float vv[3] = { v.x, v.y, v.z };
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        float sinOs = sinThetaO * cosf(2.f * ar[p])
                    + cosThetaO * sinf(2.f * ar[p]);
        float cosOs = sqrtf(fmaxf(0.f, 1.f - sinOs * sinOs));
        float m_p = mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, vv[p]);
        float n_p = mh_Np(phi, p, s, gammaO, gammaT);
        fsum = fsum + ap[p] * (m_p * n_p);
    }
    return fsum * (1.0f / denom);
}

// Marschner PDF for the BSDF-sampling MIS weight.
static __forceinline__ __device__
float evalMarschnerPdf(
    float sinThetaO, float cosThetaO,
    float sinThetaI, float cosThetaI,
    float phi, float h, float3 sigma_a, float eta,
    float3 v, float s, float3 alphaR)
{
    float sin2O  = 1.f - cosThetaO * cosThetaO;
    float etaP   = sqrtf(fmaxf(0.f, eta * eta - sin2O)) / fmaxf(cosThetaO, 1e-5f);
    float gammaO = asinf(fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h)));
    float sinGT  = fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h / fmaxf(etaP, 1e-5f)));
    float gammaT = asinf(sinGT);

    float3 ap[3]; mh_Ap(cosThetaO, eta, h, sigma_a, ap);
    float w0 = hairLum(ap[0]), w1 = hairLum(ap[1]), w2 = hairLum(ap[2]);
    float wT = w0 + w1 + w2;
    if (wT < 1e-8f) return 0.f;

    const float ar[3] = { alphaR.x, alphaR.y, alphaR.z };
    const float vv[3] = { v.x, v.y, v.z };
    const float wp[3] = { w0, w1, w2 };

    float pdf = 0.f;
    #pragma unroll
    for (int p = 0; p < 3; ++p) {
        float sinOs = sinThetaO * cosf(2.f * ar[p])
                    + cosThetaO * sinf(2.f * ar[p]);
        float cosOs = sqrtf(fmaxf(0.f, 1.f - sinOs * sinOs));
        pdf += (wp[p] / wT)
             * mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, vv[p]) * cosThetaI
             * mh_Np(phi, p, s, gammaO, gammaT);
    }
    return fmaxf(0.f, pdf);
}

// ---------------------------------------------------------------------------
// Hair NEE — Marschner × Li / pdfL from a single sampled light direction.
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 sampleDirectHair(float3 hitPos, float3 wo, float3 hairT, float h,
                         float3 sigma_a, float eta, float3 v, float s, float3 alphaR,
                         float3 ribbonN, float rayTime, uint32_t& rng)
{
    if (params.numLights == 0) return make_float3(0.f, 0.f, 0.f);

    uint32_t lightIdx = uint32_t(rand01(rng) * float(params.numLights)) % params.numLights;
    const GpuLight& light = params.lights[lightIdx];
    float lightPick = 1.f / float(params.numLights);

    float3 Li = make_float3(0.f, 0.f, 0.f), wi = make_float3(0.f, 0.f, 0.f);
    float tMax = 0.f, pdfL = 0.f;

    if (light.type == kLightRect) {
        float2 u = rand2(rng);
        float3 lpos = make3(light.position);
        float3 luH  = make3(light.uHalf);
        float3 lvH  = make3(light.vHalf);
        float3 sp   = lpos + luH * (2.f * u.x - 1.f) + lvH * (2.f * u.y - 1.f);
        float3 toL  = sp - hitPos;
        float  dist = sqrtf(dot(toL, toL));
        wi   = toL * (1.f / dist);
        tMax = dist * 0.9999f;
        float3 lN = make3(light.normal);
        float  cosL = dot(-1.f * wi, lN);
        if (cosL <= 0.f) return make_float3(0.f, 0.f, 0.f);
        pdfL = (dist * dist) / (cosL * light.area) * lightPick;
        Li   = make3(light.Le);
    } else if (light.type == kLightDirectional) {
        float3 baseDir = make3(light.normal);
        float cc = light.cosCone;
        if (cc < 0.9999f) {
            float2 uc = rand2(rng);
            float cosT = 1.f - uc.x * (1.f - cc);
            float sinT = sqrtf(fmaxf(0.f, 1.f - cosT * cosT));
            float phi  = 2.f * CUDART_PI_F * uc.y;
            float3 t, bt;
            buildONB(baseDir, t, bt);
            wi = normalize(t * (sinT * cosf(phi))
                         + bt * (sinT * sinf(phi))
                         + baseDir * cosT);
        } else {
            wi = baseDir;
        }
        tMax = 1e9f;
        pdfL = lightPick;
        Li   = make3(light.Le);
    } else if (light.type == kLightDome) {
        tMax = 1e9f;
        if (params.cam.envMapWidth > 0 && params.envMarginalCdf != nullptr
                && params.envConditionalCdf != nullptr) {
            float ep = 0.f;
            wi = sampleEnvDirection(rand2(rng),
                                    params.envMarginalCdf,
                                    params.envConditionalCdf, ep);
            pdfL = ep * lightPick;
        } else {
            wi   = cosineSampleHemisphere(rand2(rng), ribbonN);
            float cw = fmaxf(1e-7f, dot(ribbonN, wi));
            pdfL = (cw / CUDART_PI_F) * lightPick;
        }
        Li = (params.envTexture != 0) ? evalEnvmap(wi) : make3(params.cam.envLe);
    } else {
        return make_float3(0.f, 0.f, 0.f);
    }
    if (pdfL <= 0.f) return make_float3(0.f, 0.f, 0.f);

    float3 shadowO = hitPos + ribbonN * 1e-4f;
    float3 Tr = shadowTransmittance(shadowO, wi, tMax, rayTime);
    if (compmax(Tr) <= 0.f) return make_float3(0.f, 0.f, 0.f);

    // Hair BSDF evaluation
    float sinThetaO = dot(wo, hairT);
    float cosThetaO = sqrtf(fmaxf(0.f, 1.f - sinThetaO * sinThetaO));
    float sinThetaI = dot(wi, hairT);
    float cosThetaI = sqrtf(fmaxf(0.f, 1.f - sinThetaI * sinThetaI));

    float3 woPerp = wo - hairT * sinThetaO;
    float3 wiPerp = wi - hairT * sinThetaI;
    float  lenO = sqrtf(dot(woPerp, woPerp));
    float  lenI = sqrtf(dot(wiPerp, wiPerp));
    float  phi  = 0.f;
    if (lenO > 1e-5f && lenI > 1e-5f) {
        woPerp = woPerp * (1.f / lenO);
        wiPerp = wiPerp * (1.f / lenI);
        float c  = fminf(1.f, fmaxf(-1.f, dot(woPerp, wiPerp)));
        float sp = dot(cross(wiPerp, woPerp), hairT);
        phi = atan2f(sp, c);
    }

    float3 f = evalMarschnerLobes(sinThetaO, cosThetaO, sinThetaI, cosThetaI,
                                   phi, h, sigma_a, eta, v, s, alphaR);
    return f * cosThetaI * Li * Tr * (1.f / pdfL);
}

// ---------------------------------------------------------------------------
// Direct light sampling
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 sampleDirect(float3 hitPos, float3 n, float3 wo,
                    uint32_t matType, float3 baseColor,
                    float roughness, float metalness, float specular,
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
        f = evalLayeredBSDF(wo, wi, n, baseColor, roughness, metalness, specular);
    }

    return f * Li * Tr * cosI * (1.0f / pdfL);
}

// ---------------------------------------------------------------------------
// Pixel reconstruction filter — separable inverse-CDF sampling
//
// Mirrors PixelFilter::sample1D() on the CPU.  Returns sub-pixel offset
// in [-radius, radius] plus the sign at the chosen bin (±1 for filters
// with negative lobes; +1 for all-positive filters).  When the host
// hasn't bound a filter (pixelFilterBins == 0) we fall back to a uniform
// [0, 1) box-1.0 jitter.
// ---------------------------------------------------------------------------
struct PixelFilterSample {
    float dx;
    float dy;
    float weight;
};

static __forceinline__ __device__
void samplePixelFilterAxis(float u, float& outX, float& outSign)
{
    const int   N    = (int)params.pixelFilterBins;
    const float R    = params.pixelFilterRadius;
    // Binary search for the upper bound bin
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (params.pixelFilterCdf[mid + 1] <= u) lo = mid + 1; else hi = mid;
    }
    int   bin  = (lo < N) ? lo : (N - 1);
    float c0   = params.pixelFilterCdf[bin];
    float c1   = params.pixelFilterCdf[bin + 1];
    float t    = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5f;
    float binW = (2.0f * R) / float(N);
    outX       = -R + (float(bin) + t) * binW;
    outSign    = params.pixelFilterSigns[bin];
}

static __forceinline__ __device__
PixelFilterSample samplePixelFilter(float u1, float u2)
{
    PixelFilterSample s;
    if (params.pixelFilterBins == 0
        || params.pixelFilterCdf == nullptr
        || params.pixelFilterSigns == nullptr) {
        // Fallback: legacy box-1.0 jitter centred on the pixel.
        s.dx     = u1 - 0.5f;
        s.dy     = u2 - 0.5f;
        s.weight = 1.0f;
        return s;
    }
    float sx, sgX, sy, sgY;
    samplePixelFilterAxis(u1, sx, sgX);
    samplePixelFilterAxis(u2, sy, sgY);
    s.dx     = sx;
    s.dy     = sy;
    s.weight = sgX * sgY;
    return s;
}

// ---------------------------------------------------------------------------
// Caustic photon map — uniform-cell hash grid density estimate.
//
// Mirrors the Metal queryHashGrid: 3x3x3 neighbour scan around the hit-cell,
// flat-kernel inside the search radius, BSDF evaluation at each photon's
// arrival direction.  When the camera params disable the map or the buffer
// pointers are null this is a fast no-op.
// ---------------------------------------------------------------------------
static __forceinline__ __device__
float3 queryPhotonMap(float3 p, float3 n, float3 wo,
                       uint32_t matType, float3 baseColor,
                       float roughness, float metalness, float specular)
{
    if (!params.cam.photonMapEnabled
        || params.hashCellStart == nullptr
        || params.photons       == nullptr)
        return make_float3(0.f, 0.f, 0.f);

    const float r  = params.cam.photonSearchRadius;
    const float r2 = r * r;
    const float cs = params.cam.hashCellSize;

    float3 orig = make3(params.cam.hashGridOrigin);
    float3 rel  = p - orig;

    int cx = (int)floorf(rel.x / cs);
    int cy = (int)floorf(rel.y / cs);
    int cz = (int)floorf(rel.z / cs);

    const int dimX = (int)params.cam.hashGridDimX;
    const int dimY = (int)params.cam.hashGridDimY;
    const int dimZ = (int)params.cam.hashGridDimZ;

    float3 Laccum = make_float3(0.f, 0.f, 0.f);

    #pragma unroll
    for (int dz = -1; dz <= 1; ++dz)
    #pragma unroll
    for (int dy = -1; dy <= 1; ++dy)
    #pragma unroll
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0) continue;
        if (nx >= dimX || ny >= dimY || nz >= dimZ) continue;

        uint32_t cellIdx = (uint32_t)nz * (uint32_t)dimX * (uint32_t)dimY
                         + (uint32_t)ny * (uint32_t)dimX
                         + (uint32_t)nx;
        // __ldg routes through the read-only cache, which dramatically
        // improves throughput for the scattered cellStart / photons reads
        // that dominate this kernel's runtime.
        uint32_t start = __ldg(params.hashCellStart + cellIdx);
        uint32_t end   = __ldg(params.hashCellStart + cellIdx + 1);

        for (uint32_t i = start; i < end; ++i) {
            // photons[] is now stored compacted (= grid-sorted), so each
            // cell's photons are contiguous in memory and the indirection
            // through sortedPhotonIdx is gone.  Better GPU cache behaviour.
            const GpuPhoton ph = params.photons[i];
            float3 phPos = make3(ph.position);
            float3 diff  = phPos - p;
            if (dot(diff, diff) > r2) continue;

            // wi (toward surface) negated → light direction at the hit.
            float3 wi   = -1.0f * make3(ph.wi);
            float  cosI = dot(n, wi);
            if (cosI <= 0.f) continue;

            float3 f = make_float3(0.f, 0.f, 0.f);
            if (matType == kMatLambertian) {
                f = baseColor * (1.0f / CUDART_PI_F);
            } else if (matType == kMatGGX) {
                f = evalLayeredBSDF(wo, wi, n, baseColor,
                                     roughness, metalness, specular);
            }
            Laccum += f * make3(ph.power);
        }
    }

    return Laccum * (1.0f / (CUDART_PI_F * r2));
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
    float weightAcc = 0.0f;

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

        // Pixel reconstruction filter — importance-sampled sub-pixel
        // jitter with per-sample sign weight (±1 for negative-lobe filters).
        PixelFilterSample fs = samplePixelFilter(rand01(rng), rand01(rng));
        float jx = 0.5f + fs.dx;
        float jy = 0.5f + fs.dy;
        float fw = fs.weight;
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
        // Caustic photon map is queried at the first non-glass hit along the
        // camera path (= the first diffuse/glossy surface, possibly after a
        // chain of refractions through glass).  Deeper diffuse bounces have
        // throughput < 0.5 and add negligible caustic contribution while
        // costing ~70% of the per-hit hash-grid work — skip them.
        bool   pmQueried    = false;

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

            // ----------------------------------------------------------------
            // Hair hit — handled before the standard material lookup because
            // hair uses its own per-primitive metadata buffer and Marschner
            // BSDF.  params.normals / params.indices index the mesh GAS only.
            // ----------------------------------------------------------------
            const bool isHair = (params.cam.hairMeshBaseID != 0xFFFFFFFFu
                                  && hit.instanceID >= params.cam.hairMeshBaseID);
            if (isHair && params.hairTris != nullptr && params.hairMats != nullptr) {
                GpuHairTri ht = params.hairTris[hit.primID];
                float bw = 1.f - hit.bary.x - hit.bary.y;
                float h  = fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f,
                    ht.h0 * bw + ht.h1 * hit.bary.x + ht.h2 * hit.bary.y));

                // Per-strand sigma_a: Beer-Lambert from color, or material default.
                float3 sc = make3(ht.color);
                float3 sigma_a;
                if (sc.x > 0.98f && sc.y > 0.98f && sc.z > 0.98f) {
                    sigma_a = make3(params.hairMats[ht.matIdx].sigma_a);
                } else {
                    sigma_a = make_float3(-logf(fmaxf(0.001f, sc.x)),
                                          -logf(fmaxf(0.001f, sc.y)),
                                          -logf(fmaxf(0.001f, sc.z)));
                }

                HairPrecomp hp = makeHairPrecomp(params.hairMats[ht.matIdx]);
                float3 hairT   = make3(ht.tangent);

                // Ribbon normal — matches the tessellator's convention.
                float3 refUp = (fabsf(hairT.y) > 0.9f) ? make_float3(1.f, 0.f, 0.f)
                                                       : make_float3(0.f, 1.f, 0.f);
                float3 widthDir = normalize(cross(hairT, refUp));
                float3 ribbonN  = normalize(cross(widthDir, hairT));
                if (dot(ribbonN, -1.0f * rayDir) < 0.f) ribbonN = -1.0f * ribbonN;

                float3 wo = -1.0f * rayDir;

                if (params.numLights > 0) {
                    L += throughput * sampleDirectHair(
                        hitPos, wo, hairT, h, sigma_a,
                        params.hairMats[ht.matIdx].eta,
                        hp.v, hp.s, hp.alphaR, ribbonN, rayTime, rng);
                }

                // Russian roulette
                if (bounce >= 3) {
                    float q = fmaxf(0.05f, 1.0f - compmax(throughput));
                    if (rand01(rng) < q) break;
                    throughput *= (1.0f / (1.0f - q));
                }

                // Sample next direction — lobe pick via A_p luminance weights,
                // longitudinal angle by Box-Muller around theta_o + alphaR,
                // azimuthal angle by the trimmed-logistic CDF inverse.
                float sinThetaO = dot(wo, hairT);
                float cosThetaO = sqrtf(fmaxf(0.f, 1.f - sinThetaO * sinThetaO));

                float3 ap[3];
                mh_Ap(cosThetaO, params.hairMats[ht.matIdx].eta, h, sigma_a, ap);
                float w0 = hairLum(ap[0]), w1 = hairLum(ap[1]), w2 = hairLum(ap[2]);
                float wT = w0 + w1 + w2;
                if (wT < 1e-8f) break;

                float uComp = rand01(rng);
                int lobe = 2;
                float uLobe;
                float cdf0 = w0 / wT, cdf1 = (w0 + w1) / wT;
                if (uComp < cdf0)      { lobe = 0; uLobe = uComp / fmaxf(cdf0, 1e-7f); }
                else if (uComp < cdf1) { lobe = 1; uLobe = (uComp - cdf0) / fmaxf(cdf1 - cdf0, 1e-7f); }
                else                   {            uLobe = (uComp - cdf1) / fmaxf(1.f - cdf1, 1e-7f); }

                float thetaO = asinf(fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, sinThetaO)));
                float u1 = fmaxf(rand01(rng), 1e-6f);
                float u2 = rand01(rng);
                float z  = sqrtf(-2.f * logf(u1)) * cosf(2.f * CUDART_PI_F * u2);
                const float arArr[3] = { hp.alphaR.x, hp.alphaR.y, hp.alphaR.z };
                const float vArr[3]  = { hp.v.x, hp.v.y, hp.v.z };
                float thetaI = fminf(CUDART_PI_F * 0.5f,
                                      fmaxf(-CUDART_PI_F * 0.5f,
                                            thetaO + arArr[lobe]
                                                   + z * sqrtf(vArr[lobe])));
                float sinThetaI_s = sinf(thetaI), cosThetaI_s = cosf(thetaI);

                float etaH = params.hairMats[ht.matIdx].eta;
                float sin2O  = 1.f - cosThetaO * cosThetaO;
                float etaP   = sqrtf(fmaxf(0.f, etaH * etaH - sin2O))
                             / fmaxf(cosThetaO, 1e-5f);
                float gammaO = asinf(fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h)));
                float sinGT  = fminf(1.f-1e-5f, fmaxf(-1.f+1e-5f, h / fmaxf(etaP, 1e-5f)));
                float gammaT = asinf(sinGT);
                float phi_s  = mh_sampleTrimmedLogistic(uLobe, hp.s)
                             + mh_Phi(lobe, gammaO, gammaT);

                float3 woPerp = wo - hairT * sinThetaO;
                float  lenO   = sqrtf(dot(woPerp, woPerp));
                if (lenO < 1e-5f) {
                    float3 arb = (fabsf(hairT.x) > 0.9f)
                               ? make_float3(0.f, 1.f, 0.f)
                               : make_float3(1.f, 0.f, 0.f);
                    woPerp = normalize(cross(hairT, arb));
                } else {
                    woPerp = woPerp * (1.f / lenO);
                }
                float3 ctPerp = cross(hairT, woPerp);

                float3 wi = normalize(hairT * sinThetaI_s
                                    + woPerp * (cosThetaI_s * cosf(phi_s))
                                    + ctPerp * (cosThetaI_s * sinf(phi_s)));

                float sinThetaI_a = dot(wi, hairT);
                float cosThetaI_a = sqrtf(fmaxf(0.f, 1.f - sinThetaI_a * sinThetaI_a));
                float3 wiPerp = wi - hairT * sinThetaI_a;
                float  lenI   = sqrtf(dot(wiPerp, wiPerp));
                float  phiA   = 0.f;
                if (lenI > 1e-5f && lenO > 1e-5f) {
                    wiPerp = wiPerp * (1.f / lenI);
                    float c  = fminf(1.f, fmaxf(-1.f, dot(woPerp, wiPerp)));
                    float sp = dot(cross(wiPerp, woPerp), hairT);
                    phiA = atan2f(sp, c);
                }

                float3 bsdfF = evalMarschnerLobes(sinThetaO, cosThetaO,
                                                   sinThetaI_a, cosThetaI_a,
                                                   phiA, h, sigma_a, etaH,
                                                   hp.v, hp.s, hp.alphaR);
                float bsdfPdf = evalMarschnerPdf(sinThetaO, cosThetaO,
                                                  sinThetaI_a, cosThetaI_a,
                                                  phiA, h, sigma_a, etaH,
                                                  hp.v, hp.s, hp.alphaR);
                if (bsdfPdf < 1e-8f) break;

                throughput *= bsdfF * cosThetaI_a * (1.f / bsdfPdf);

                rayOrig      = hitPos + ribbonN * 1e-4f;
                rayDir       = wi;
                prevBsdfPdf  = bsdfPdf;
                prevN        = ribbonN;
                prevWasDelta = false;
                continue;  // skip mesh material code
            }

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
                                               mat.specular,
                                               rng, rayTime);

                // Caustic photon map density estimate at the first non-glass
                // hit (= primary visible surface, possibly via a glass chain).
                if (!pmQueried) {
                    float3 Lcaustic = queryPhotonMap(hitPos, n, wo,
                                                      mat.type, baseColor,
                                                      mat.roughness, mat.metalness,
                                                      mat.specular);
                    L += throughput * Lcaustic;
                    pmQueried = true;
                }
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

                bsdfF = evalLayeredBSDF(wo, wi, n, baseColor,
                                         mat.roughness, mat.metalness,
                                         mat.specular);

                float ggxPdf = D * cosH / fmaxf(1e-7f, 4.0f * dot(wo, wh));
                float cosPdf = cosII / CUDART_PI_F;
                bsdfPdf = pSpec * ggxPdf + pDiff * cosPdf;
            } else {
                // kMatLambertian or very-rough kMatGGX (rough >= 0.95).
                // Sample cosine hemisphere; for kMatGGX, evaluate the
                // layered StandardSurface BSDF — this preserves the
                // CPU's energy compensation (Disney retro-reflection,
                // spec/diff balance, Kulla-Conty MS) on diffuse-flagged
                // walls.  Plain Lambertian for non-StandardSurface.
                wi      = cosineSampleHemisphere(rand2(rng), n);
                bsdfPdf = fmaxf(1e-7f, dot(n, wi)) / CUDART_PI_F;
                if (mat.type == kMatGGX) {
                    bsdfF = evalLayeredBSDF(wo, wi, n, baseColor,
                                             mat.roughness, mat.metalness,
                                             mat.specular);
                } else {
                    bsdfF = baseColor * (1.0f / CUDART_PI_F);
                }
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
        rAcc      += L.x * fw;
        gAcc      += L.y * fw;
        bAcc      += L.z * fw;
        weightAcc += fw;
        lumSqAcc  += lum * lum;
    }

    GpuAccumPixel& out = params.accum[pixelIdx];
    out.r        += rAcc;
    out.g        += gAcc;
    out.b        += bAcc;
    out.weight   += weightAcc;
    out.sumLumSq += lumSqAcc;
}

extern "C" __global__ void __closesthit__ch()
{
    const uint32_t prim  = optixGetPrimitiveIndex();
    const uint32_t inst  = optixGetInstanceId();
    const float2   bary  = optixGetTriangleBarycentrics();
    const float    tHit  = optixGetRayTmax();
    optixSetPayload_0(1u);
    optixSetPayload_1(prim);
    optixSetPayload_2(__float_as_uint(bary.x));
    optixSetPayload_3(__float_as_uint(bary.y));
    optixSetPayload_4(__float_as_uint(tHit));
    optixSetPayload_5(inst);
}

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(0u);
}

// ---------------------------------------------------------------------------
// __raygen__photon — one thread per photon slot.
//
// Selects a light, samples a position + direction, then traces the photon
// through the scene.  Stored on the first diffuse hit AFTER ≥1 specular
// (glass) bounce — caustic photons only.  Empty slots have power = (0,0,0)
// so the host hash-grid builder can skip them.
//
// Shares the closesthit/miss/SBT with __raygen__rg; the host dispatches it
// by swapping the SBT raygenRecord pointer at launch time.
// ---------------------------------------------------------------------------
extern "C" __global__ void __raygen__photon()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint32_t gid = idx.x;
    if (gid >= params.numPhotons) return;

    // Default to "invalid slot" — the host hash-grid builder filters these out.
    params.photons[gid].power = {0.f, 0.f, 0.f};

    if (params.numLights == 0) return;

    uint32_t rng = pcg(pcg(gid) ^ (params.frameIndex * 2654435761u + 1234567u));

    // Uniform light selection.  Same convention as sampleDirect; rectangles
    // are the only light type that produces useful caustics here.
    uint32_t lightIdx = uint32_t(rand01(rng) * float(params.numLights)) % params.numLights;
    const GpuLight& light = params.lights[lightIdx];
    float lightSelectPdf = 1.0f / float(params.numLights);

    float3 pos    = make_float3(0.f, 0.f, 0.f);
    float3 dir    = make_float3(0.f, 0.f, 0.f);
    float3 Le     = make_float3(0.f, 0.f, 0.f);
    float3 lightN = make_float3(0.f, 0.f, 0.f);
    float  pdfPos = 0.f, pdfDir = 0.f;

    if (light.type == kLightRect) {
        float3 lpos = make3(light.position);
        float3 luH  = make3(light.uHalf);
        float3 lvH  = make3(light.vHalf);
        float2 u    = rand2(rng);
        pos    = lpos + luH * (2.0f * u.x - 1.0f) + lvH * (2.0f * u.y - 1.0f);
        pdfPos = 1.0f / fmaxf(1e-7f, light.area);
        lightN = make3(light.normal);
        dir    = cosineSampleHemisphere(rand2(rng), lightN);
        float cosTheta = fmaxf(0.f, dot(dir, lightN));
        pdfDir = fmaxf(1e-7f, cosTheta / CUDART_PI_F);
        Le     = make3(light.Le);
    } else {
        // Directional / dome lights focus to infinity (no useful caustics),
        // and sphere lights aren't supported by this backend yet.  Bail.
        return;
    }

    if (pdfPos <= 0.f || pdfDir <= 0.f) return;

    // Initial radiant flux — matches CPU PhotonMapIntegrator + Metal photonTrace.
    float cosTheta = fmaxf(0.f, dot(dir, lightN));
    float invDen = 1.0f
                 / fmaxf(1e-30f,
                          lightSelectPdf * pdfPos * pdfDir * float(params.numPhotons));
    float3 power = Le * (cosTheta * invDen);

    float3 rayOrig = pos + lightN * 1e-3f;
    float3 rayDir  = dir;
    const float rayTime = 0.5f;  // mid-frame; matches Metal photonTrace

    int numSpecular = 0;

    for (int bounce = 0; bounce < 8; ++bounce) {
        TraceResult hit = trace(rayOrig, rayDir, 1e-4f, 1e10f, rayTime);
        if (!hit.valid) break;

        // Skip hair and other non-mesh hits — caustic photons should land on
        // mesh triangles where they'll be queried by the hash grid.
        const bool isHair = (params.cam.hairMeshBaseID != 0xFFFFFFFFu
                              && hit.instanceID >= params.cam.hairMeshBaseID);
        if (isHair) break;

        uint32_t matIdx = (hit.meshID < params.numMaterials) ? hit.meshID : 0u;
        GpuMaterial mat = params.materials[matIdx];

        if (mat.type == kMatEmissive) break;  // emitter — stop

        float3 hitPos = rayOrig + rayDir * hit.t;
        float3 geomN  = interpolateNormal(hit.primID, hit.bary,
                                           params.normals, params.indices);

        if (mat.type == kMatGlass) {
            // Delta refract/reflect with Russian-roulette branch.  Throughput
            // unchanged for delta paths; no tinting/absorption in glass here.
            bool   entering = dot(rayDir, geomN) < 0.f;
            float3 faceN    = entering ? geomN : (-1.0f * geomN);
            float  eta      = entering ? (1.0f / mat.specularIOR) : mat.specularIOR;
            float  cosI     = fmaxf(0.f, dot(-1.0f * rayDir, faceN));
            float  Fr       = fresnelDielectric(cosI, 1.0f / eta);

            if (rand01(rng) < Fr) {
                rayDir = reflect(rayDir, faceN);
                rayOrig = hitPos + faceN * 1e-4f;
            } else {
                float3 refr = refract(rayDir, faceN, eta);
                if (dot(refr, refr) < 0.5f) {
                    rayDir  = reflect(rayDir, faceN);
                    rayOrig = hitPos + faceN * 1e-4f;
                } else {
                    rayDir  = normalize(refr);
                    rayOrig = hitPos - faceN * 1e-4f;
                }
            }
            ++numSpecular;
            continue;
        }

        // Diffuse / glossy hit.  Store the photon iff we've had at least one
        // specular bounce — otherwise direct lighting already covers it.
        if (numSpecular > 0) {
            params.photons[gid].position = {hitPos.x, hitPos.y, hitPos.z};
            params.photons[gid].wi       = {rayDir.x, rayDir.y, rayDir.z};
            params.photons[gid].power    = {power.x, power.y, power.z};
        }
        break;
    }
}
