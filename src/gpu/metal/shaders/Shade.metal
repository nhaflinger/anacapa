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

// ============================================================
// Marschner (2003) hair BSDF — MSL port
// Three lobes: R (specular), TT (forward scatter), TRT (back-scatter)
// Reference: Marschner et al. 2003; d'Eon et al. 2011; PBRT-v4 Hair.cpp
// ============================================================

static float mh_I0(float x) {
    float sum = 0.f, x2i = 1.f, denom = 1.f;
    for (int i = 0; i < 12; ++i) {
        if (i > 0) { x2i *= x * x; denom *= float(i) * float(i) * 4.f; }
        sum += x2i / denom;
    }
    return sum;
}

static float mh_logI0(float x) {
    if (x > 12.f)
        return x + 0.5f * (-log(2.f * M_PI_F) - log(x) + 1.f / (8.f * x));
    return log(mh_I0(x));
}

// Von Mises–Fisher longitudinal scattering M_p
static float mh_Mp(float cosI, float sinI, float cosR, float sinR, float v) {
    v = max(v, 1e-5f);
    float a = cosI * cosR / v;
    float b = sinI * sinR / v;
    if (v <= 0.1f)
        return exp(mh_logI0(a) - b - 1.f/v + 0.6931472f + log(0.5f/v));
    return exp(-b) * mh_I0(a) / (2.f * v * sinh(1.f/v));
}

static float mh_logistic(float x, float s) {
    float ex = exp(-abs(x) / s);
    return ex / (s * (1.f + ex) * (1.f + ex));
}
static float mh_logisticCDF(float x, float s) { return 1.f / (1.f + exp(-x / s)); }
static float mh_trimmedLogistic(float x, float s) {
    return mh_logistic(x, s) / (mh_logisticCDF(M_PI_F, s) - mh_logisticCDF(-M_PI_F, s));
}
static float mh_sampleTrimmedLogistic(float u, float s) {
    float a = mh_logisticCDF(-M_PI_F, s);
    float b = mh_logisticCDF( M_PI_F, s);
    return clamp(-s * log(1.f / (a + u * (b - a)) - 1.f), -M_PI_F, M_PI_F);
}

static float mh_Phi(int p, float gO, float gT) {
    return 2.f * float(p) * gT - 2.f * gO + float(p) * M_PI_F;
}
static float mh_wrapPhi(float x) {
    x = fmod(x, 2.f * M_PI_F);
    if (x >  M_PI_F) x -= 2.f * M_PI_F;
    if (x < -M_PI_F) x += 2.f * M_PI_F;
    return x;
}
static float mh_Np(float phi, int p, float s, float gO, float gT) {
    return mh_trimmedLogistic(mh_wrapPhi(phi - mh_Phi(p, gO, gT)), s);
}

// Luminance weight for lobe selection
static float hairLum(float3 c) { return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f; }

// Lobe attenuation A_p — writes R/TT/TRT into ap[0..2]
static void mh_Ap(float cosThetaO, float eta, float h, float3 sigma_a,
                  thread float3 ap[3])
{
    float sin2O = max(0.f, 1.f - cosThetaO * cosThetaO);
    float etaP  = sqrt(max(0.f, eta * eta - sin2O)) / max(cosThetaO, 1e-5f);
    float sinGT = clamp(h / max(etaP, 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
    float cosGT = sqrt(max(0.f, 1.f - sinGT * sinGT));
    float3 T    = exp(-sigma_a * 2.f * cosGT);
    float cosGO = sqrt(max(0.f, 1.f - h * h));
    float fr    = fresnelDielectric(max(0.f, cosThetaO * cosGO), eta);
    ap[0] = float3(fr, fr, fr);
    ap[1] = (1.f - fr) * (1.f - fr) * T;
    ap[2] = ap[1] * T * fr;
}

// Precomputed per-hit hair parameters derived from GpuHairMaterial
struct HairPrecomp {
    float3 v;       // longitudinal variance [R, TT, TRT]
    float  s;       // azimuthal logistic scale
    float3 alphaR;  // lobe shift angles in radians [R, TT, TRT]
};

static HairPrecomp makeHairPrecomp(const device GpuHairMaterial& hm) {
    HairPrecomp hp;
    float bm = clamp(hm.beta_m, 1e-3f, 1.f);
    float v0 = 0.726f * bm + 0.812f * bm * bm + 3.7f * pow(bm, 20.f);
    v0 *= v0;
    hp.v = float3(v0, v0 * 0.25f, v0 * 4.f);

    float bn = clamp(hm.beta_n, 1e-3f, 1.f);
    hp.s = 0.626657069f * (0.265f * bn + 1.194f * bn * bn + 5.372f * pow(bn, 22.f));

    float ar = hm.alpha * (M_PI_F / 180.f);
    hp.alphaR = float3(-ar, ar * 0.5f, -ar * 1.5f);
    return hp;
}

// Full Marschner BSDF evaluation (R + TT + TRT, no cosine factor)
static float3 evalMarschnerLobes(
    float sinThetaO, float cosThetaO,
    float sinThetaI, float cosThetaI,
    float phi, float h, float3 sigma_a, float eta,
    float3 v, float s, float3 alphaR)
{
    float cosThetaD = sqrt(max(0.f, 0.5f * (1.f + cosThetaO * cosThetaI
                                                 + sinThetaO * sinThetaI)));
    float denom = max(1e-5f, cosThetaD * cosThetaD);

    float sin2O  = 1.f - cosThetaO * cosThetaO;
    float etaP   = sqrt(max(0.f, eta * eta - sin2O)) / max(cosThetaO, 1e-5f);
    float gammaO = asin(clamp(h, -1.f + 1e-5f, 1.f - 1e-5f));
    float sinGT  = clamp(h / max(etaP, 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
    float gammaT = asin(sinGT);

    float3 ap[3]; mh_Ap(cosThetaO, eta, h, sigma_a, ap);

    float3 fsum = float3(0.f);
    for (int p = 0; p < 3; ++p) {
        float sinOs = sinThetaO * cos(2.f * alphaR[p])
                    + cosThetaO * sin(2.f * alphaR[p]);
        float cosOs = sqrt(max(0.f, 1.f - sinOs * sinOs));
        fsum += ap[p] * (mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, v[p])
                       * mh_Np(phi, p, s, gammaO, gammaT));
    }
    return fsum / denom;
}

// Marschner PDF (solid-angle, weighted mixture over lobes)
static float evalMarschnerPdf(
    float sinThetaO, float cosThetaO,
    float sinThetaI, float cosThetaI,
    float phi, float h, float3 sigma_a, float eta,
    float3 v, float s, float3 alphaR)
{
    float sin2O  = 1.f - cosThetaO * cosThetaO;
    float etaP   = sqrt(max(0.f, eta * eta - sin2O)) / max(cosThetaO, 1e-5f);
    float gammaO = asin(clamp(h, -1.f + 1e-5f, 1.f - 1e-5f));
    float sinGT  = clamp(h / max(etaP, 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
    float gammaT = asin(sinGT);

    float3 ap[3]; mh_Ap(cosThetaO, eta, h, sigma_a, ap);
    float w0 = hairLum(ap[0]), w1 = hairLum(ap[1]), w2 = hairLum(ap[2]);
    float wT = w0 + w1 + w2;
    if (wT < 1e-8f) return 0.f;

    float pdf = 0.f;
    for (int p = 0; p < 3; ++p) {
        float sinOs = sinThetaO * cos(2.f * alphaR[p])
                    + cosThetaO * sin(2.f * alphaR[p]);
        float cosOs = sqrt(max(0.f, 1.f - sinOs * sinOs));
        float3 ap3[3]; mh_Ap(cosThetaO, eta, h, sigma_a, ap3);
        float wP = (p == 0) ? w0 : (p == 1) ? w1 : w2;
        pdf += (wP / wT)
             * mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, v[p]) * cosThetaI
             * mh_Np(phi, p, s, gammaO, gammaT);
    }
    return max(0.f, pdf);
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
// MIS helpers
// ---------------------------------------------------------------------------
static float powerHeuristic(float pdfF, float pdfG) {
    float f = pdfF, g = pdfG;
    return (f * f) / (f * f + g * g + 1e-9f);
}

// Solid-angle PDF of the NEE strategy sampling direction wi from fromPos
// toward rect light `light`.  Returns 0 if hitPos is not within the rect.
static float rectLightSolidAnglePdf(const device GpuLight& light,
                                     float3 hitPos, float3 wi, float dist)
{
    float3 lightN = float3(light.normal.x, light.normal.y, light.normal.z);
    float cosL = dot(-wi, lightN);
    if (cosL <= 0.0f) return 0.0f;
    float3 toHit = hitPos - float3(light.position.x, light.position.y, light.position.z);
    float3 uH = float3(light.uHalf.x, light.uHalf.y, light.uHalf.z);
    float3 vH = float3(light.vHalf.x, light.vHalf.y, light.vHalf.z);
    float uLen = length(uH);
    float vLen = length(vH);
    if (uLen < 1e-7f || vLen < 1e-7f) return 0.0f;
    float uCoord = dot(toHit, uH * (1.0f / uLen));
    float vCoord = dot(toHit, vH * (1.0f / vLen));
    if (abs(uCoord) > uLen || abs(vCoord) > vLen) return 0.0f;
    return (dist * dist) / (cosL * light.area);
}

// ---------------------------------------------------------------------------
// Energy-compensation LUT helpers — mirror the CPU StandardSurface tables
// ---------------------------------------------------------------------------

// Bilinear lookup of E_spec(cosO, roughness) from the 2D spec-albedo LUT.
static float specAlbedoLookup(float cosO, float roughness,
                               constant GpuCameraParams& cam,
                               const device float* lut) {
    if (cam.specLUTCosBins == 0 || lut == nullptr) return 0.0f;
    uint N_COS = cam.specLUTCosBins;
    uint N_R   = cam.specLUTRoughBins;
    cosO      = clamp(cosO,      0.0f, 1.0f);
    roughness = clamp(roughness, 0.0f, 1.0f);
    float fc  = cosO      * float(N_COS - 1);
    float fr  = roughness * float(N_R   - 1);
    uint  ic0 = min(N_COS - 1, uint(fc));
    uint  ir0 = min(N_R   - 1, uint(fr));
    uint  ic1 = min(N_COS - 1, ic0 + 1);
    uint  ir1 = min(N_R   - 1, ir0 + 1);
    float tc  = fc - float(ic0);
    float tr  = fr - float(ir0);
    float v00 = lut[ic0 * N_R + ir0];
    float v01 = lut[ic0 * N_R + ir1];
    float v10 = lut[ic1 * N_R + ir0];
    float v11 = lut[ic1 * N_R + ir1];
    return (v00 * (1.0f - tc) + v10 * tc) * (1.0f - tr)
         + (v01 * (1.0f - tc) + v11 * tc) * tr;
}

// 1D lookup of cosine-weighted average E_spec — Kulla-Conty denominator.
static float specAvgAlbedoLookup(float roughness,
                                  constant GpuCameraParams& cam,
                                  const device float* avgLut) {
    if (cam.specLUTRoughBins == 0 || avgLut == nullptr) return 0.0f;
    uint  N_R = cam.specLUTRoughBins;
    roughness = clamp(roughness, 0.0f, 1.0f);
    float fr  = roughness * float(N_R - 1);
    uint  i0  = min(N_R - 1, uint(fr));
    uint  i1  = min(N_R - 1, i0 + 1);
    float t   = fr - float(i0);
    return avgLut[i0] * (1.0f - t) + avgLut[i1] * t;
}

// Kulla-Conty multi-scatter GGX compensation.
static float3 evalGGXMs(float cosO, float cosI, float roughness, float3 F_ms,
                         constant GpuCameraParams& cam,
                         const device float* lut, const device float* avgLut) {
    if (cosO <= 0.0f || cosI <= 0.0f) return float3(0.0f);
    float Eo = specAlbedoLookup(cosO, roughness, cam, lut);
    float Ei = specAlbedoLookup(cosI, roughness, cam, lut);
    float Ea = specAvgAlbedoLookup(roughness, cam, avgLut);
    if (Ea >= 0.999f) return float3(0.0f);
    float k = (1.0f - Eo) * (1.0f - Ei) / (M_PI_F * (1.0f - Ea));
    return F_ms * k;
}

// Disney/Burley diffuse with roughness-dependent retro-reflection.
static float3 disneyDiffuseLobe(float3 wo, float3 wi, float3 n,
                                 float3 baseColor, float roughness) {
    float cosI = max(0.0f, dot(n, wi));
    float cosO = max(0.0f, dot(n, wo));
    float3 wh  = normalize(wo + wi);
    float cosD = max(0.0f, dot(wi, wh));
    float Fd90   = 0.5f + 2.0f * cosD * cosD * roughness;
    float Fview  = 1.0f + (Fd90 - 1.0f) * pow(1.0f - cosO, 5.0f);
    float Flight = 1.0f + (Fd90 - 1.0f) * pow(1.0f - cosI, 5.0f);
    return baseColor * (1.0f / M_PI_F) * Fview * Flight;
}

// Layered StandardSurface BSDF — single GGX spec lobe + Disney diffuse with
// energy-conserving spec/diff balance and Kulla-Conty multi-scatter correction.
static float3 evalLayeredBSDF(float3 wo, float3 wi, float3 n,
                               float3 baseColor, float roughness,
                               float metalness, float specular,
                               constant GpuCameraParams& cam,
                               const device float* specLUT,
                               const device float* specAvgLUT) {
    float3 wh    = normalize(wo + wi);
    float  cosH  = max(0.0f, dot(n, wh));
    float  cosO  = max(0.0f, dot(n, wo));
    float  cosII = max(0.0f, dot(n, wi));
    if (cosO <= 0.0f || cosII <= 0.0f) return float3(0.0f);

    float  alpha  = max(1e-4f, roughness * roughness);
    float  a2     = alpha * alpha;
    float  D      = ggxD(cosH, a2);
    float  G      = ggxG1(cosO, a2) * ggxG1(cosII, a2);
    float  vdotH  = max(0.0f, dot(wi, wh));
    float  invDen = 1.0f / max(1e-7f, 4.0f * cosO * cosII);

    float3 F0   = mix(float3(0.04f), baseColor, metalness);
    float3 spec = D * G * schlick(vdotH, F0) * invDen
                + evalGGXMs(cosO, cosII, roughness, F0, cam, specLUT, specAvgLUT);

    float  E_spec = specAlbedoLookup(cosO, roughness, cam, specLUT);
    float  diffW  = (1.0f - metalness) * (1.0f - specular * E_spec);
    float3 diff   = disneyDiffuseLobe(wo, wi, n, baseColor, roughness) * diffW;

    return diff + spec;
}

// ---------------------------------------------------------------------------
// HDRI importance sampling helpers
// ---------------------------------------------------------------------------

// Binary search into a normalized 1D CDF of (n+1) floats.
// Returns bin index in [0,n), remapped u in [0,1), and bin probability (= pdf).
static uint sampleCDF1D(const device float* cdf, uint n, float u,
                         thread float& uRemapped, thread float& prob) {
    uint lo = 0, hi = n;
    while (lo < hi) {
        uint mid = (lo + hi) >> 1;
        if (cdf[mid + 1] <= u) lo = mid + 1;
        else hi = mid;
    }
    uint idx = min(lo, n - 1);
    float binProb = cdf[idx + 1] - cdf[idx];
    uRemapped = (binProb > 1e-7f) ? clamp((u - cdf[idx]) / binProb, 0.0f, 1.0f - 1e-7f) : 0.0f;
    prob = binProb;
    return idx;
}

// Evaluate solid-angle PDF for world direction `dir` using uploaded HDRI CDFs.
static float evalEnvPdf(float3 dir,
                         constant GpuCameraParams& cam,
                         const device float* marginalCdf,
                         const device float* conditionalCdf) {
    float3 envRot0 = float3(cam.envRot0.x, cam.envRot0.y, cam.envRot0.z);
    float3 envRot1 = float3(cam.envRot1.x, cam.envRot1.y, cam.envRot1.z);
    float3 envRot2 = float3(cam.envRot2.x, cam.envRot2.y, cam.envRot2.z);
    float3 local = float3(dot(envRot0, dir), dot(envRot1, dir), dot(envRot2, dir));

    float theta = acos(clamp(local.y, -1.0f, 1.0f));
    float phi   = atan2(local.x, local.z);
    if (phi < 0.0f) phi += 2.0f * M_PI_F;

    uint W = cam.envMapWidth;
    uint H = cam.envMapHeight;
    uint col = min(uint(phi / (2.0f * M_PI_F) * float(W)), W - 1u);
    uint row = min(uint(theta / M_PI_F * float(H)), H - 1u);

    float sinTheta = max(sin(theta), 1e-6f);
    float pdfRow   = marginalCdf[row + 1] - marginalCdf[row];
    float pdfCol   = conditionalCdf[row * (W + 1) + col + 1] - conditionalCdf[row * (W + 1) + col];
    return (pdfRow * pdfCol * float(W * H)) / (2.0f * M_PI_F * M_PI_F * sinTheta);
}

// Sample a direction from the 2D HDRI importance distribution.
// Returns the world-space direction and sets pdfOut (solid-angle PDF).
static float3 sampleEnvDirection(float2 u,
                                  constant GpuCameraParams& cam,
                                  const device float* marginalCdf,
                                  const device float* conditionalCdf,
                                  thread float& pdfOut) {
    uint W = cam.envMapWidth;
    uint H = cam.envMapHeight;

    float uRow, probRow, uCol, probCol;
    uint row = sampleCDF1D(marginalCdf,                   H, u.y, uRow, probRow);
    uint col = sampleCDF1D(conditionalCdf + row * (W + 1), W, u.x, uCol, probCol);

    float v     = (float(row) + uRow) / float(H);
    float uu    = (float(col) + uCol) / float(W);
    float theta = v * M_PI_F;
    float phi   = uu * 2.0f * M_PI_F;
    float sinT  = sin(theta);
    float cosT  = cos(theta);

    // Direction in envmap local space, rotated to world space.
    // cam.envRot rows map world→env; transpose maps env→world.
    float3 envDir  = float3(sinT * sin(phi), cosT, sinT * cos(phi));
    float3 envRot0 = float3(cam.envRot0.x, cam.envRot0.y, cam.envRot0.z);
    float3 envRot1 = float3(cam.envRot1.x, cam.envRot1.y, cam.envRot1.z);
    float3 envRot2 = float3(cam.envRot2.x, cam.envRot2.y, cam.envRot2.z);
    float3 worldDir = float3(
        envRot0.x * envDir.x + envRot1.x * envDir.y + envRot2.x * envDir.z,
        envRot0.y * envDir.x + envRot1.y * envDir.y + envRot2.y * envDir.z,
        envRot0.z * envDir.x + envRot1.z * envDir.y + envRot2.z * envDir.z
    );

    float sinTabs = max(sinT, 1e-6f);
    pdfOut = (probRow * probCol * float(W * H)) / (2.0f * M_PI_F * M_PI_F * sinTabs);
    if (pdfOut <= 0.0f) pdfOut = 1e-7f;
    return worldDir;
}

// ---------------------------------------------------------------------------
// Shadow transmittance — steps through glass surfaces between hitPos and the
// light, accumulating tint. Returns (0,0,0) if an opaque surface blocks the
// path, or a tint <= (1,1,1) attenuated by any glass surfaces in between.
// ---------------------------------------------------------------------------
// blockGlass: when true (photon map enabled), glass is opaque to shadow rays.
// The photon map is the sole source of transmitted light; letting NEE count
// it too causes double-counting and kills the caustic shadow contrast.
// When false (no photon map), glass attenuates rather than blocks so the
// basic path tracer still shows some light through glass.
static float3 shadowTransmittance(
    float3                          origin,
    float3                          dir,
    float                           tMax,
    float                           rayTime,
    const device GpuMaterial*       materials,
    uint                            numMaterials,
    const device PackedFloat3*      normals,
    const device uint32_t*          indices,
    const device uint32_t*          meshIndexOffsets,
    bool                            blockGlass,
    acceleration_structure<instancing, primitive_motion> accelStruct)
{
    float3 T = float3(1.0f);
    ray    stepRay;
    stepRay.direction    = dir;
    stepRay.min_distance = 1e-4f;
    stepRay.max_distance = tMax;
    stepRay.origin       = origin;

    intersector<triangle_data, instancing, primitive_motion> isect;
    isect.accept_any_intersection(false);  // need closest hit to step correctly

    for (int step = 0; step < 8; ++step) {
        intersection_result<triangle_data, instancing, primitive_motion> res =
            isect.intersect(stepRay, accelStruct, 0xFF, rayTime);

        if (res.type == intersection_type::none) break;  // clear path

        uint meshID = res.instance_id;
        uint matIdx = (meshID < numMaterials) ? meshID : 0;
        GpuMaterial mat = materials[matIdx];

        if (mat.type == kMatGlass) {
            // Glass is opaque to shadow rays only when a photon map is active
            // AND this specific material is flagged as a caustic generator.
            // Non-flagged glass keeps attenuating NEE by its transmission tint
            // so ordinary windows / drinking glasses pass shadow light correctly
            // even when --integrator photon is enabled.
            if (blockGlass && mat.causticGenerator) return float3(0);

            // Without a photon map, attenuate by transmission so glass still
            // passes some light through (approximate but better than hard black).
            T *= mat.transmission;
            if (max(T.x, max(T.y, T.z)) < 1e-4f) return float3(0);

            // Advance past this surface
            float remaining = stepRay.max_distance - res.distance;
            if (remaining <= 1e-4f) break;
            stepRay.origin       = stepRay.origin + dir * (res.distance + 1e-4f);
            stepRay.max_distance = remaining - 1e-4f;
        } else {
            // Opaque surface blocks the light
            return float3(0);
        }
    }
    return T;
}

// Direct lighting contribution for a hair hit.
// Evaluates the Marschner BSDF × Li / pdfL from one sampled light direction.
static float3 sampleDirectHair(
    float3 hitPos, float3 wo, float3 hairT, float h,
    float3 sigma_a, float eta, float3 v, float s, float3 alphaR,
    float3 ribbonN,   // ribbon normal (for shadow ray offset)
    float  rayTime,
    const device GpuLight*    lights,    uint numLights,
    const device GpuMaterial* materials, uint numMaterials,
    const device PackedFloat3* normals,
    const device uint32_t*     indices,
    const device uint32_t*     meshIndexOffsets,
    thread uint& rng,
    acceleration_structure<instancing, primitive_motion> accelStruct,
    constant GpuCameraParams& cam,
    texture2d<float, access::sample> envTex,
    const device float* envMarginalCdf,
    const device float* envConditionalCdf)
{
    if (numLights == 0) return float3(0);

    uint lightIdx = uint(rand01(rng) * float(numLights)) % numLights;
    const device GpuLight& light = lights[lightIdx];
    float lightPick = 1.f / float(numLights);

    float3 Li = float3(0), wi = float3(0);
    float  tMax = 0, pdfL = 0;

    if (light.type == kLightRect) {
        float2 u = rand2(rng);
        float3 sp = float3(light.position.x, light.position.y, light.position.z)
                  + float3(light.uHalf.x, light.uHalf.y, light.uHalf.z) * (2.f*u.x-1.f)
                  + float3(light.vHalf.x, light.vHalf.y, light.vHalf.z) * (2.f*u.y-1.f);
        float3 toL = sp - hitPos;
        float  dist = length(toL);
        wi    = toL / dist;
        tMax  = dist * 0.9999f;
        float3 lightN = float3(light.normal.x, light.normal.y, light.normal.z);
        float  cosL   = dot(-wi, lightN);
        if (cosL <= 0.f) return float3(0);
        pdfL = (dist * dist) / (cosL * light.area) * lightPick;
        Li   = float3(light.Le.x, light.Le.y, light.Le.z);
    } else if (light.type == kLightDirectional) {
        float3 baseDir = float3(light.normal.x, light.normal.y, light.normal.z);
        float  cc = light.cosCone;
        if (cc < 0.9999f) {
            float2 uc = rand2(rng);
            float cosT = 1.f - uc.x * (1.f - cc);
            float sinT = sqrt(max(0.f, 1.f - cosT*cosT));
            float phi  = 2.f * M_PI_F * uc.y;
            float3 tang, bt;
            if (abs(baseDir.x) > 0.9f) tang = normalize(cross(float3(0,1,0), baseDir));
            else                         tang = normalize(cross(float3(1,0,0), baseDir));
            bt = cross(baseDir, tang);
            wi = normalize(tang*(sinT*cos(phi)) + bt*(sinT*sin(phi)) + baseDir*cosT);
        } else { wi = baseDir; }
        tMax = 1e9f; pdfL = lightPick;
        Li   = float3(light.Le.x, light.Le.y, light.Le.z);
    } else if (light.type == kLightDome) {
        tMax = 1e9f;
        if (cam.envMapWidth > 0) {
            float ep = 0.f;
            wi  = sampleEnvDirection(rand2(rng), cam, envMarginalCdf, envConditionalCdf, ep);
            pdfL = ep * lightPick;
        } else {
            // Cosine-hemisphere fallback (over ribbon normal hemisphere)
            wi   = cosineSampleHemisphere(rand2(rng), ribbonN);
            pdfL = max(1e-7f, dot(ribbonN, wi)) / M_PI_F * lightPick;
        }
        Li = evalEnvmap(wi, cam, envTex);
    } else {
        return float3(0);
    }
    if (pdfL <= 0.f) return float3(0);

    // Shadow transmittance
    float3 shadowO = hitPos + ribbonN * 1e-4f;
    float3 Tr = shadowTransmittance(shadowO, wi, tMax, rayTime,
                                    materials, numMaterials,
                                    normals, indices, meshIndexOffsets,
                                    cam.photonMapEnabled != 0,
                                    accelStruct);
    if (max(Tr.x, max(Tr.y, Tr.z)) <= 0.f) return float3(0);

    // Hair BSDF evaluation
    float sinThetaO = dot(wo, hairT);
    float cosThetaO = sqrt(max(0.f, 1.f - sinThetaO*sinThetaO));
    float sinThetaI = dot(wi, hairT);
    float cosThetaI = sqrt(max(0.f, 1.f - sinThetaI*sinThetaI));

    // Azimuthal difference angle phi
    float3 woPerp = wo - hairT * sinThetaO;
    float3 wiPerp = wi - hairT * sinThetaI;
    float  lenO = length(woPerp), lenI = length(wiPerp);
    float  phi = 0.f;
    if (lenO > 1e-5f && lenI > 1e-5f) {
        woPerp /= lenO; wiPerp /= lenI;
        float c  = clamp(dot(woPerp, wiPerp), -1.f, 1.f);
        float sp = dot(cross(wiPerp, woPerp), hairT);
        phi = atan2(sp, c);
    }

    float3 f = evalMarschnerLobes(sinThetaO, cosThetaO, sinThetaI, cosThetaI,
                                   phi, h, sigma_a, eta, v, s, alphaR);
    float hairCos = cosThetaI;
    return f * hairCos * Li * Tr / pdfL;
}

static float3 sampleDirect(
    float3                          hitPos,
    float3                          n,
    float3                          wo,
    uint                            matType,
    float3                          baseColor,
    float                           roughness,
    float                           metalness,
    float                           specular,
    float                           rayTime,
    const device GpuLight*          lights,
    uint                            numLights,
    const device GpuMaterial*       materials,
    uint                            numMaterials,
    const device PackedFloat3*      normals,
    const device uint32_t*          indices,
    const device uint32_t*          meshIndexOffsets,
    thread uint&                    rng,
    acceleration_structure<instancing, primitive_motion> accelStruct,
    constant GpuCameraParams&       cam,
    texture2d<float, access::sample> envTex,
    const device float*             envMarginalCdf,
    const device float*             envConditionalCdf,
    const device float*             specAlbedoLUT,
    const device float*             specAvgAlbedoLUT)
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
        tMax = 1e9f;
        if (cam.envMapWidth > 0) {
            // HDRI importance sampling via uploaded CDF tables
            float envPdf = 0.0f;
            wi  = sampleEnvDirection(rand2(rng), cam, envMarginalCdf, envConditionalCdf, envPdf);
            if (dot(n, wi) <= 0.0f) return float3(0);
            pdfL = envPdf * lightPick;
        } else {
            // Fallback: cosine hemisphere sampling
            wi = cosineSampleHemisphere(rand2(rng), n);
            float cosW = max(1e-7f, dot(n, wi));
            pdfL = (cosW / M_PI_F) * lightPick;
        }
        Li = evalEnvmap(wi, cam, envTex);

    } else {
        return float3(0);
    }

    float cosI = dot(n, wi);
    if (cosI <= 0.0f || pdfL <= 0.0f) return float3(0);

    // Transmittance along shadow ray — steps through glass surfaces
    float3 shadowOrigin = hitPos + n * 1e-4f;
    float3 Tr = shadowTransmittance(shadowOrigin, wi, tMax, rayTime,
                                    materials, numMaterials,
                                    normals, indices, meshIndexOffsets,
                                    cam.photonMapEnabled != 0,
                                    accelStruct);
    if (max(Tr.x, max(Tr.y, Tr.z)) <= 0.0f) return float3(0);

    // BSDF eval
    float3 f = float3(0);
    if (matType == kMatLambertian) {
        f = baseColor * (1.0f / M_PI_F);
    } else if (matType == kMatGGX) {
        f = evalLayeredBSDF(wo, wi, n, baseColor, roughness, metalness, specular,
                            cam, specAlbedoLUT, specAvgAlbedoLUT);
    }
    // Glass is delta — no area PDF can be evaluated, skip direct lighting

    return f * Li * Tr * cosI / pdfL;
}

// ---------------------------------------------------------------------------
// Pixel filter helpers (MSL does not support lambdas)
// ---------------------------------------------------------------------------
static void sampleFilterAxis(float u,
                              const device float* cdf,
                              const device float* signs,
                              int N, float R,
                              thread float& outX,
                              thread float& outSign) {
    const float binW = (2.0f * R) / float(N);
    int lo = 0, hi = N;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid + 1] <= u) lo = mid + 1; else hi = mid;
    }
    int   bin = (lo < N) ? lo : (N - 1);
    float c0  = cdf[bin];
    float c1  = cdf[bin + 1];
    float t   = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5f;
    outX    = -R + (float(bin) + t) * binW;
    outSign = signs[bin];
}

static void samplePixelFilter(float u1, float u2,
                               constant GpuCameraParams& cam,
                               const device float* cdf,
                               const device float* signs,
                               thread float& dx,
                               thread float& dy,
                               thread float& weight) {
    if (cam.pixelFilterBins == 0) {
        dx = u1 - 0.5f;
        dy = u2 - 0.5f;
        weight = 1.0f;
        return;
    }
    float sx, sgX, sy, sgY;
    sampleFilterAxis(u1, cdf, signs, (int)cam.pixelFilterBins, cam.pixelFilterRadius, sx, sgX);
    sampleFilterAxis(u2, cdf, signs, (int)cam.pixelFilterBins, cam.pixelFilterRadius, sy, sgY);
    dx = sx; dy = sy; weight = sgX * sgY;
}

// ---------------------------------------------------------------------------
// Caustic photon map — flat-kernel density estimate using the CPU-built
// hash grid uploaded at buffers 21/22/23.
// ---------------------------------------------------------------------------
static float3 queryHashGrid(
    float3                         p,
    float3                         n,
    float3                         wo,
    uint                           matType,
    float3                         baseColor,
    float                          roughness,
    float                          metalness,
    float                          specular,
    constant GpuCameraParams&      cam,
    const device uint32_t*         cellStart,
    const device uint32_t*         sortedPhotonIdx,
    const device GpuPhoton*        photons,
    const device float*            specAlbedoLUT,
    const device float*            specAvgAlbedoLUT)
{
    float r  = cam.photonSearchRadius;
    float r2 = r * r;
    float cs = cam.hashCellSize;

    float3 orig = float3(cam.hashGridOrigin.x, cam.hashGridOrigin.y, cam.hashGridOrigin.z);
    float3 rel  = p - orig;

    int cx = int(floor(rel.x / cs));
    int cy = int(floor(rel.y / cs));
    int cz = int(floor(rel.z / cs));

    uint dimX = cam.hashGridDimX;
    uint dimY = cam.hashGridDimY;
    uint dimZ = cam.hashGridDimZ;

    float3 Laccum = float3(0);

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0) continue;
        if (uint(nx) >= dimX || uint(ny) >= dimY || uint(nz) >= dimZ) continue;

        uint cellIdx = uint(nz) * dimX * dimY + uint(ny) * dimX + uint(nx);
        uint start   = cellStart[cellIdx];
        uint end     = cellStart[cellIdx + 1];

        for (uint i = start; i < end; ++i) {
            GpuPhoton ph = photons[sortedPhotonIdx[i]];

            float3 phPos = float3(ph.position.x, ph.position.y, ph.position.z);
            float3 diff  = phPos - p;
            if (dot(diff, diff) > r2) continue;

            // wi toward surface — negate to get the light direction at this point
            float3 wi = -float3(ph.wi.x, ph.wi.y, ph.wi.z);
            float cosI = dot(n, wi);
            if (cosI <= 0.f) continue;

            float3 f = float3(0);
            if (matType == kMatLambertian) {
                f = baseColor * (1.f / M_PI_F);
            } else if (matType == kMatGGX) {
                f = evalLayeredBSDF(wo, wi, n, baseColor, roughness, metalness, specular,
                                    cam, specAlbedoLUT, specAvgAlbedoLUT);
            }

            float3 power = float3(ph.power.x, ph.power.y, ph.power.z);
            Laccum += f * power;
        }
    }

    return Laccum * (1.f / (M_PI_F * r2));
}

// ---------------------------------------------------------------------------
// SSS photon map density estimate using the dipole-like kernel from CPU
// estimateSSSRadiance():  exp(-r_lat/d) * norm, with depth attenuation on
// the back-face.  Smoothstep density fade prevents bias at sparse areas.
// ---------------------------------------------------------------------------
static float3 querySSSHashGrid(
    float3                         p,
    float3                         n,
    float3                         subsurfaceColor,
    float                          d,
    constant GpuCameraParams&      cam,
    const device uint32_t*         sssCellStart,
    const device uint32_t*         sssSortedIdx,
    const device GpuPhoton*        sssPhotons)
{
    if (!cam.sssMapEnabled || d <= 0.f) return float3(0);

    // Hash grid cell size = 3*d_max (set by host), so ±1 traversal covers
    // 3d kernel support (~95% of exp(-r/d) weight — same visual result as 6d
    // at typical photon densities, with 8x fewer photons per query).
    float r  = 3.f * d;
    float r2 = r * r;
    float cs = cam.sssHashCellSize;

    float3 orig = float3(cam.sssHashOrigin.x, cam.sssHashOrigin.y, cam.sssHashOrigin.z);
    float3 rel  = p - orig;

    int cx = int(floor(rel.x / cs));
    int cy = int(floor(rel.y / cs));
    int cz = int(floor(rel.z / cs));

    uint dimX = cam.sssHashDimX;
    uint dimY = cam.sssHashDimY;
    uint dimZ = cam.sssHashDimZ;

    float norm = 1.f / (2.f * M_PI_F * M_PI_F * d * d);
    float3 Laccum = float3(0);
    int cnt = 0;

    for (int dz = -1; dz <= 1; ++dz)
    for (int dy = -1; dy <= 1; ++dy)
    for (int dx = -1; dx <= 1; ++dx) {
        int nx = cx + dx, ny = cy + dy, nz = cz + dz;
        if (nx < 0 || ny < 0 || nz < 0) continue;
        if (uint(nx) >= dimX || uint(ny) >= dimY || uint(nz) >= dimZ) continue;

        uint cellIdx = uint(nz) * dimX * dimY + uint(ny) * dimX + uint(nx);
        uint start   = sssCellStart[cellIdx];
        uint end     = sssCellStart[cellIdx + 1];

        for (uint i = start; i < end; ++i) {
            GpuPhoton ph = sssPhotons[sssSortedIdx[i]];
            float3 phPos = float3(ph.position.x, ph.position.y, ph.position.z);
            float3 diff  = phPos - p;
            if (dot(diff, diff) > r2) continue;
            ++cnt;
            float proj  = dot(diff, n);                              // signed depth
            float rLat2 = max(0.f, dot(diff, diff) - proj * proj);
            float rLat  = sqrt(rLat2);
            float w     = exp(-rLat / d) * norm;
            if (proj < 0.f) w *= exp(proj / d);                     // back-face depth attenuation
            float3 phPow = float3(ph.power.x, ph.power.y, ph.power.z);
            Laccum += phPow * w;
        }
    }

    if (cnt < 4) return float3(0);
    float t  = min(1.f, float(cnt - 4) / 20.f);
    float ds = t * t * (3.f - 2.f * t);
    return subsurfaceColor * (Laccum * ds);
}

// ---------------------------------------------------------------------------
// Halo disc particle traversal — software BVH matching CPU HaloAccel
// ---------------------------------------------------------------------------

// Camera-facing disc intersection.
// N = normalize(rayOrig - center); disc plane dot(N, P-center)=0.
// Returns hit t in outT on success.
static bool intersectHaloDisc(GpuHaloDesc h, float3 rayOrig, float3 rayDir,
                               float tMin, float tMax, float rayTime,
                               thread float& outT)
{
    float3 center = float3(h.center.x, h.center.y, h.center.z)
                  + (float3(h.centerClose.x, h.centerClose.y, h.centerClose.z)
                   - float3(h.center.x, h.center.y, h.center.z)) * rayTime;

    float3 toOrig = rayOrig - center;
    float  lenSq  = dot(toOrig, toOrig);
    if (lenSq < 1e-12f) return false;
    float3 N = toOrig * rsqrt(lenSq);

    float denom = dot(N, rayDir);
    if (abs(denom) < 1e-6f) return false;

    float t = dot(N, center - rayOrig) / denom;
    if (t < tMin || t >= tMax) return false;

    float3 hitPt = rayOrig + rayDir * t;
    float3 delta = hitPt - center;
    if (dot(delta, delta) > h.radius * h.radius) return false;

    outT = t;
    return true;
}

// Software BVH traversal over halo particles.
// Returns the hit halo index (into halos[]), or -1 on no hit.
// outT is set to the closest hit distance (starts at tMax).
static int traverseHaloBVH(
    float3 rayOrig, float3 rayDir, float tMin, float tMax, float rayTime,
    const device GpuHaloDesc* halos,
    const device GpuHaloNode* nodes,
    const device uint32_t*    primIdx,
    thread float& outT)
{
    outT = tMax;
    int bestIdx = -1;

    float3 safe = float3(abs(rayDir.x) > 1e-9f ? rayDir.x : 1e-9f,
                         abs(rayDir.y) > 1e-9f ? rayDir.y : 1e-9f,
                         abs(rayDir.z) > 1e-9f ? rayDir.z : 1e-9f);
    float3 invDir = 1.0f / safe;

    uint stack[64];
    int top = 0;
    stack[top++] = 0u;

    while (top > 0) {
        uint ni = stack[--top];
        GpuHaloNode node = nodes[ni];

        float3 bmin = float3(node.bmin[0], node.bmin[1], node.bmin[2]);
        float3 bmax = float3(node.bmax[0], node.bmax[1], node.bmax[2]);
        float3 t0 = (bmin - rayOrig) * invDir;
        float3 t1 = (bmax - rayOrig) * invDir;
        float tn = max3(min(t0.x, t1.x), min(t0.y, t1.y), min(t0.z, t1.z));
        float tf = min3(max(t0.x, t1.x), max(t0.y, t1.y), max(t0.z, t1.z));
        tn = max(tn, tMin);
        if (tn > tf || tn >= outT) continue;

        if (node.right_or_count & 0x80000000u) {
            uint first = node.left_or_prim;
            uint count = node.right_or_count & 0x7FFFFFFFu;
            for (uint j = 0; j < count; ++j) {
                uint idx = primIdx[first + j];
                float t;
                if (intersectHaloDisc(halos[idx], rayOrig, rayDir,
                                      tMin, outT, rayTime, t)) {
                    outT    = t;
                    bestIdx = int(idx);
                }
            }
        } else {
            stack[top++] = node.right_or_count;
            stack[top++] = node.left_or_prim;
        }
    }
    return bestIdx;
}

// ---------------------------------------------------------------------------
// Main kernel
// ---------------------------------------------------------------------------
kernel void shade(
    constant  GpuCameraParams&              cam               [[ buffer(0)  ]],
    device    GpuAccumPixel*                accum             [[ buffer(1)  ]],
    const device GpuLight*                  lights            [[ buffer(2)  ]],
    constant  uint&                         numLights         [[ buffer(3)  ]],
    const device GpuMaterial*               materials         [[ buffer(4)  ]],
    constant  uint&                         numMaterials      [[ buffer(5)  ]],
    const device PackedFloat3*              normals           [[ buffer(6)  ]],
    const device uint32_t*                  indices           [[ buffer(7)  ]],
    const device uint32_t*                  triMeshIDs        [[ buffer(8)  ]],
    const device uint32_t*                  meshVertexOffsets [[ buffer(9)  ]],
    const device uint32_t*                  meshIndexOffsets  [[ buffer(10) ]],
    constant  GpuSampleBatch&               batch             [[ buffer(11) ]],
    acceleration_structure<instancing, primitive_motion> accelStruct [[ buffer(12) ]],
    const device float*                     envMarginalCdf    [[ buffer(13) ]],
    const device float*                     envConditionalCdf [[ buffer(14) ]],
    const device float*                     specAlbedoLUT     [[ buffer(15) ]],
    const device float*                     specAvgAlbedoLUT  [[ buffer(16) ]],
    const device float*                     pixelFilterCdf    [[ buffer(17) ]],
    const device float*                     pixelFilterSigns  [[ buffer(18) ]],
    const device GpuHairTri*               hairTris          [[ buffer(19) ]],
    const device GpuHairMaterial*          hairMats          [[ buffer(20) ]],
    const device uint32_t*                 hashCellStart     [[ buffer(21) ]],
    const device uint32_t*                 sortedPhotonIdx   [[ buffer(22) ]],
    const device GpuPhoton*                photons           [[ buffer(23) ]],
    const device uint32_t*                 sssCellStart      [[ buffer(24) ]],
    const device uint32_t*                 sssSortedIdx      [[ buffer(25) ]],
    const device GpuPhoton*                sssPhotons        [[ buffer(26) ]],
    const device GpuHaloDesc*              halos             [[ buffer(27) ]],
    const device GpuHaloNode*              haloNodes         [[ buffer(28) ]],
    const device uint32_t*                 haloPrimIdx       [[ buffer(29) ]],
    texture2d<float, access::sample>        envTexture        [[ texture(0) ]],
    uint2                                   gid               [[ thread_position_in_grid ]])
{
    uint px = cam.tileX0 + gid.x;
    uint py = cam.tileY0 + gid.y;
    if (px >= cam.imageWidth || py >= cam.imageHeight) return;

    // accum is tile-sized; use local coordinates for the write index
    uint pixelIdx = gid.y * cam.tileWidth + gid.x;

    uint globalPixelIdx = py * cam.imageWidth + px;

    // Open-state camera vectors
    float3 camOrig0 = float3(cam.origin.x,     cam.origin.y,     cam.origin.z);
    float3 camH0    = float3(cam.horizontal.x,  cam.horizontal.y,  cam.horizontal.z);
    float3 camV0    = float3(cam.vertical.x,    cam.vertical.y,    cam.vertical.z);
    float3 camLL0   = float3(cam.lowerLeft.x,   cam.lowerLeft.y,   cam.lowerLeft.z);
    // Close-state camera vectors (== open-state when hasMotion==0, so mix() is a no-op)
    float3 camOrig1 = float3(cam.originClose.x,     cam.originClose.y,     cam.originClose.z);
    float3 camH1    = float3(cam.horizontalClose.x,  cam.horizontalClose.y,  cam.horizontalClose.z);
    float3 camV1    = float3(cam.verticalClose.x,    cam.verticalClose.y,    cam.verticalClose.z);
    float3 camLL1   = float3(cam.lowerLeftClose.x,   cam.lowerLeftClose.y,   cam.lowerLeftClose.z);

    intersector<triangle_data, instancing, primitive_motion> isect;
    isect.accept_any_intersection(false);

    // Local accumulators — written once after the whole batch
    float3 batchL      = float3(0.0f);
    float  batchLumSq  = 0.0f;
    float  batchWeight = 0.0f;

    for (uint s = 0; s < batch.batchSize; ++s) {
        uint sampleIndex = batch.sampleStart + s;

        // Per-sample RNG seed decorrelated by pixel and sample index
        uint rng = pcg(pcg(globalPixelIdx) ^ (sampleIndex * 2654435761u));

        // Sample shutter time uniformly within [shutterOpen, shutterClose]
        float rayTime = cam.shutterOpen + rand01(rng) * (cam.shutterClose - cam.shutterOpen);

        // Camera motion blur: lerp image-plane at rayTime
        float camT = (cam.shutterClose > cam.shutterOpen + 1e-6f)
                   ? clamp((rayTime - cam.shutterOpen) / (cam.shutterClose - cam.shutterOpen), 0.f, 1.f)
                   : 0.f;
        float3 origin = mix(camOrig0, camOrig1, camT);
        float3 horiz  = mix(camH0,    camH1,    camT);
        float3 vert   = mix(camV0,    camV1,    camT);
        float3 ll     = mix(camLL0,   camLL1,   camT);

        // Pixel reconstruction filter — importance-sampled jitter with
        // ±1 sign weight for filters with negative lobes.
        float fdx, fdy, fw;
        samplePixelFilter(rand01(rng), rand01(rng), cam, pixelFilterCdf, pixelFilterSigns, fdx, fdy, fw);
        float jx = 0.5f + fdx;
        float jy = 0.5f + fdy;

        float u = (float(px) + jx) / float(cam.imageWidth);
        float v = (float(cam.imageHeight - 1 - py) + jy) / float(cam.imageHeight);

        ray r;
        r.origin       = origin;
        r.direction    = normalize(ll + u * horiz + v * vert - origin);
        r.min_distance = 1e-4f;
        r.max_distance = 1e10f;

        // Path tracing loop
        float3 throughput  = float3(1.0f);
        float3 L           = float3(0.0f);
        uint   glassDepth  = 0;
        // MIS state: track BSDF PDF of the ray that spawned the current vertex.
        // prevWasDelta=true on first hit so emitter Le gets full weight (no NEE conflict).
        float  prevBsdfPdf  = 0.0f;
        bool   prevWasDelta = true;
        float3 prevPos      = float3(0.0f);
        float3 prevN        = float3(0.0f);

    for (uint bounce = 0; bounce <= cam.maxDepth; ++bounce) {

        intersection_result<triangle_data, instancing, primitive_motion> res =
            isect.intersect(r, accelStruct, 0xFF, rayTime);

        // Halo disc particles — drain all overlapping halos in front of the nearest mesh
        // without consuming main bounce budget.  Up to 256 halos per camera ray.
        if (cam.numHalos > 0 && halos != nullptr && haloNodes != nullptr && haloPrimIdx != nullptr) {
            float meshT = (res.type != intersection_type::none) ? res.distance : r.max_distance;
            for (uint hPass = 0; hPass < 256u; ++hPass) {
                float haloT;
                int haloIdx = traverseHaloBVH(r.origin, r.direction, r.min_distance,
                                               meshT, rayTime,
                                               halos, haloNodes, haloPrimIdx, haloT);
                if (haloIdx < 0) break;

                GpuHaloDesc hd = halos[haloIdx];
                float3 particleColor = float3(hd.color.x, hd.color.y, hd.color.z);
                float3 Le      = particleColor;
                float  opacity = 1.0f;
                uint   hMatIdx = hd.matIdx;
                if (hMatIdx < numMaterials) {
                    float3 emissive = float3(materials[hMatIdx].emissive.x,
                                            materials[hMatIdx].emissive.y,
                                            materials[hMatIdx].emissive.z);
                    if (emissive.x > 0.f || emissive.y > 0.f || emissive.z > 0.f)
                        Le = emissive * particleColor;
                    opacity = materials[hMatIdx].roughness;
                }
                L += throughput * Le;
                throughput *= (1.0f - opacity);

                float3 hitPt = r.origin + r.direction * haloT;
                r.origin     = hitPt + r.direction * 1e-4f;
                r.min_distance = 1e-4f;
                prevPos      = hitPt;
                prevWasDelta = true;

                if (throughput.x < 1e-4f && throughput.y < 1e-4f && throughput.z < 1e-4f)
                    break;
            }
        }

        if (res.type == intersection_type::none) {
            // Background / env light — apply MIS weight against NEE dome-sampling PDF.
            float3 envColor = float3(0.0f);
            if (cam.hasEnvLight) {
                envColor = evalEnvmap(r.direction, cam, envTexture);
            }
            if (envColor.x > 0.0f || envColor.y > 0.0f || envColor.z > 0.0f) {
                float weight = 1.0f;
                if (!prevWasDelta && bounce > 0) {
                    float lpdf = 0.0f;
                    for (uint li = 0; li < numLights; ++li) {
                        if (lights[li].type == kLightDome) {
                            float domePdf = 0.0f;
                            if (cam.envMapWidth > 0) {
                                domePdf = evalEnvPdf(r.direction, cam,
                                                     envMarginalCdf, envConditionalCdf);
                            } else {
                                float cosW = max(0.0f, dot(r.direction, prevN));
                                domePdf = cosW / M_PI_F;
                            }
                            lpdf += domePdf / float(numLights);
                        }
                    }
                    if (lpdf > 0.0f)
                        weight = powerHeuristic(prevBsdfPdf, lpdf);
                }
                L += throughput * envColor * weight;
            }
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

        // ============================================================
        // Hair hit — checked before the standard material table lookup.
        // hairTris is indexed by primID (local to the hair BLAS).
        // ============================================================
        bool isHair = (cam.hairMeshBaseID != 0xFFFFFFFFu && meshID >= cam.hairMeshBaseID);
        if (isHair) {
            GpuHairTri ht = hairTris[primID];
            float bw = 1.f - bary.x - bary.y;
            float h  = clamp(ht.h0*bw + ht.h1*bary.x + ht.h2*bary.y, -1.f+1e-5f, 1.f-1e-5f);

            // Per-strand sigma_a: Beer-Lambert from color, or material default
            float3 sc = float3(ht.color.x, ht.color.y, ht.color.z);
            float3 sigma_a;
            if (sc.x > 0.98f && sc.y > 0.98f && sc.z > 0.98f)
                sigma_a = float3(hairMats[ht.matIdx].sigma_a.x,
                                 hairMats[ht.matIdx].sigma_a.y,
                                 hairMats[ht.matIdx].sigma_a.z);
            else
                sigma_a = -log(max(sc, float3(0.001f)));

            HairPrecomp hp = makeHairPrecomp(hairMats[ht.matIdx]);
            float3 hairT   = float3(ht.tangent.x, ht.tangent.y, ht.tangent.z);

            // Ribbon normal for shadow ray offset (same orientation as tessellation)
            float3 refUp   = (abs(hairT.y) > 0.9f) ? float3(1,0,0) : float3(0,1,0);
            float3 widthDir= normalize(cross(hairT, refUp));
            float3 ribbonN = normalize(cross(widthDir, hairT));
            if (dot(ribbonN, -r.direction) < 0.f) ribbonN = -ribbonN;

            float3 wo = -r.direction;

            // Direct lighting
            if (numLights > 0) {
                L += throughput * sampleDirectHair(
                    hitPos, wo, hairT, h, sigma_a,
                    hairMats[ht.matIdx].eta, hp.v, hp.s, hp.alphaR,
                    ribbonN, rayTime,
                    lights, numLights, materials, numMaterials,
                    normals, indices, meshIndexOffsets,
                    rng, accelStruct, cam, envTexture,
                    envMarginalCdf, envConditionalCdf);
            }

            // Russian roulette after bounce 3
            if (bounce >= 3) {
                float q = max(0.05f, 1.f - max(throughput.x, max(throughput.y, throughput.z)));
                if (rand01(rng) < q) break;
                throughput /= (1.f - q);
            }

            // Longitudinal outgoing angles
            float sinThetaO = dot(wo, hairT);
            float cosThetaO = sqrt(max(0.f, 1.f - sinThetaO*sinThetaO));

            // Lobe selection via A_p luminance weights
            float3 ap[3]; mh_Ap(cosThetaO, hairMats[ht.matIdx].eta, h, sigma_a, ap);
            float w0 = hairLum(ap[0]), w1 = hairLum(ap[1]), w2 = hairLum(ap[2]);
            float wT = w0 + w1 + w2;
            if (wT < 1e-8f) break;

            float uComp = rand01(rng);
            int   lobe  = 2;
            float uLobe;
            float cdf0 = w0 / wT, cdf1 = (w0 + w1) / wT;
            if (uComp < cdf0)      { lobe = 0; uLobe = uComp / max(cdf0, 1e-7f); }
            else if (uComp < cdf1) { lobe = 1; uLobe = (uComp-cdf0) / max(cdf1-cdf0, 1e-7f); }
            else                   {            uLobe = (uComp-cdf1) / max(1.f-cdf1, 1e-7f); }

            // Sample longitudinal angle via Box–Muller
            float thetaO = asin(clamp(sinThetaO, -1.f+1e-5f, 1.f-1e-5f));
            float u1 = max(rand01(rng), 1e-6f), u2 = rand01(rng);
            float z  = sqrt(-2.f * log(u1)) * cos(2.f * M_PI_F * u2);
            float thetaI = clamp(thetaO + hp.alphaR[lobe] + z*sqrt(hp.v[lobe]),
                                 -M_PI_F*0.5f, M_PI_F*0.5f);
            float sinThetaI_s = sin(thetaI), cosThetaI_s = cos(thetaI);

            // Sample azimuthal angle
            float sin2O  = 1.f - cosThetaO*cosThetaO;
            float etaP   = sqrt(max(0.f, hairMats[ht.matIdx].eta*hairMats[ht.matIdx].eta - sin2O))
                         / max(cosThetaO, 1e-5f);
            float gammaO = asin(clamp(h, -1.f+1e-5f, 1.f-1e-5f));
            float sinGT  = clamp(h / max(etaP, 1e-5f), -1.f+1e-5f, 1.f-1e-5f);
            float gammaT = asin(sinGT);
            float phi_s  = mh_sampleTrimmedLogistic(uLobe, hp.s) + mh_Phi(lobe, gammaO, gammaT);

            // Reconstruct wi from sampled (thetaI, phi)
            float3 woPerp = wo - hairT * sinThetaO;
            float  lenO   = length(woPerp);
            if (lenO < 1e-5f) {
                float3 arb = (abs(hairT.x) > 0.9f) ? float3(0,1,0) : float3(1,0,0);
                woPerp = normalize(cross(hairT, arb));
            } else { woPerp /= lenO; }
            float3 ctPerp = cross(hairT, woPerp);

            float3 wi = normalize(hairT * sinThetaI_s
                                + woPerp * (cosThetaI_s * cos(phi_s))
                                + ctPerp * (cosThetaI_s * sin(phi_s)));

            // Re-derive actual longitudinal angles and phi for BSDF/PDF eval
            float sinThetaI_a = dot(wi, hairT);
            float cosThetaI_a = sqrt(max(0.f, 1.f - sinThetaI_a*sinThetaI_a));
            float3 wiPerp = wi - hairT * sinThetaI_a;
            float  lenI   = length(wiPerp);
            float  phiA   = 0.f;
            if (lenI > 1e-5f && lenO > 1e-5f) {
                wiPerp /= lenI;
                float c  = clamp(dot(woPerp, wiPerp), -1.f, 1.f);
                float sp = dot(cross(wiPerp, woPerp), hairT);
                phiA = atan2(sp, c);
            }

            float3 bsdfF = evalMarschnerLobes(sinThetaO, cosThetaO, sinThetaI_a, cosThetaI_a,
                                              phiA, h, sigma_a, hairMats[ht.matIdx].eta,
                                              hp.v, hp.s, hp.alphaR);
            float bsdfPdf = evalMarschnerPdf(sinThetaO, cosThetaO, sinThetaI_a, cosThetaI_a,
                                             phiA, h, sigma_a, hairMats[ht.matIdx].eta,
                                             hp.v, hp.s, hp.alphaR);
            if (bsdfPdf < 1e-8f) break;

            throughput *= bsdfF * cosThetaI_a / bsdfPdf;

            prevPos      = hitPos;
            prevN        = ribbonN;
            prevBsdfPdf  = bsdfPdf;
            prevWasDelta = false;

            r.origin       = hitPos + ribbonN * 1e-4f;
            r.direction    = wi;
            r.min_distance = 1e-4f;
            r.max_distance = 1e10f;
            continue;  // next bounce — skip surface material code below
        }

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

        // Emitter Le — add with MIS weight against the NEE light-sampling PDF.
        // prevWasDelta=true on first hit and after any delta bounce → weight=1.
        if (mat.type == kMatEmissive) {
            float weight = 1.0f;
            if (!prevWasDelta && bounce > 0) {
                // Sum rect-light solid-angle PDFs weighted by uniform selection (1/N)
                float lpdf = 0.0f;
                for (uint li = 0; li < numLights; ++li) {
                    if (lights[li].type == kLightRect) {
                        float spdf = rectLightSolidAnglePdf(lights[li], hitPos, r.direction, t);
                        if (spdf > 0.0f)
                            lpdf += spdf / float(numLights);
                    }
                }
                if (lpdf > 0.0f)
                    weight = powerHeuristic(prevBsdfPdf, lpdf);
            }
            L += throughput * emissive * weight;
            break;
        }

        // Direct lighting (skip for delta glass — no area-light PDF)
        float3 wo = -r.direction;
        if (mat.type != kMatGlass) {
            float3 Ldirect = sampleDirect(hitPos, n, wo,
                                          mat.type, baseColor,
                                          mat.roughness, mat.metalness, mat.specular,
                                          rayTime,
                                          lights, numLights,
                                          materials, numMaterials,
                                          normals, indices, meshIndexOffsets,
                                          rng, accelStruct, cam, envTexture,
                                          envMarginalCdf, envConditionalCdf,
                                          specAlbedoLUT, specAvgAlbedoLUT);
            L += throughput * Ldirect;

            // Caustic photon map density estimate — only queried when the map
            // has been built (photonMapEnabled != 0) and buffers are non-null.
            if (cam.photonMapEnabled && hashCellStart != nullptr && photons != nullptr) {
                float3 Lcaustic = queryHashGrid(hitPos, n, wo,
                                               mat.type, baseColor,
                                               mat.roughness, mat.metalness, mat.specular,
                                               cam, hashCellStart, sortedPhotonIdx, photons,
                                               specAlbedoLUT, specAvgAlbedoLUT);
                L += throughput * Lcaustic;
            }

            // SSS photon map density estimate — fires for any SSS material hit.
            if (mat.isSubsurface && cam.sssMapEnabled
                && sssCellStart != nullptr && sssPhotons != nullptr) {
                float3 sssColor = float3(mat.subsurfaceColor.x,
                                        mat.subsurfaceColor.y,
                                        mat.subsurfaceColor.z);
                float3 Lsss = querySSSHashGrid(hitPos, n, sssColor, mat.subsurfaceRadius,
                                               cam, sssCellStart, sssSortedIdx, sssPhotons);
                // Thickness attenuation disabled — visually preferred without it.
                // float thicknessScale = 1.f;
                // float d = mat.subsurfaceRadius;
                // if (d > 0.f) {
                //     ray inwardRay;
                //     inwardRay.origin       = hitPos - n * 1e-4f;
                //     inwardRay.direction    = -n;
                //     inwardRay.min_distance = 1e-4f;
                //     inwardRay.max_distance = 1e10f;
                //     intersector<instancing, primitive_motion> isect;
                //     auto thickResult = isect.intersect(inwardRay, accelStruct, rayTime);
                //     if (thickResult.type != intersection_type::none)
                //         thicknessScale = exp(-thickResult.distance / (d * 30.f));
                // }
                L += throughput * Lsss * mat.subsurfaceWeight * mat.subsurfaceStrength;
            }
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

            // Glass is a delta bounce: next emitter hit gets weight=1.
            prevPos      = hitPos;
            prevN        = faceN;
            prevBsdfPdf  = 1.0f;
            prevWasDelta = true;

            // Glass hits don't count against bounce budget — refracting through a
            // dome's two surfaces plus any interior bounces would exhaust maxDepth
            // before the background is ever reached.  Use a separate glass limiter.
            if (++glassDepth >= 16) break;
            bounce = (bounce > 0u) ? bounce - 1u : 0u;
            continue;

        } else if (mat.type == kMatGGX && mat.roughness < 0.95f) {
            float alpha  = mat.roughness * mat.roughness;
            float alpha2 = alpha * alpha;
            float3 F0    = mix(float3(0.04f), baseColor, mat.metalness);

            // Mixed diffuse/specular sampler — choose GGX with probability pSpec,
            // cosine hemisphere with pDiff.  This preserves indirect fill in shadowed
            // areas (where pure GGX sampling would produce near-zero diffuse weight).
            float lumSpec = (F0.x + F0.y + F0.z) / 3.0f;
            float lumDiff = (1.0f - mat.metalness) * (baseColor.x + baseColor.y + baseColor.z) / 3.0f;
            float pSpec   = lumSpec / max(1e-4f, lumSpec + lumDiff);
            float pDiff   = 1.0f - pSpec;

            float3 wh;
            if (rand01(rng) < pSpec) {
                float3 wmLocal = sampleGGX(rand2(rng), alpha2);
                wh = toWorld(wmLocal, n);
                if (dot(wh, n) < 0.0f) wh = -wh;
                wi = reflect(-wo, wh);
            } else {
                wi = cosineSampleHemisphere(rand2(rng), n);
                wh = normalize(wo + wi);
            }
            if (dot(wi, n) <= 0.0f) break;

            float cosI  = dot(n, wi);
            float cosO  = dot(n, wo);
            float cosH  = max(0.0f, dot(n, wh));
            float D     = ggxD(cosH, alpha2);

            bsdfF = evalLayeredBSDF(wo, wi, n, baseColor, mat.roughness,
                                    mat.metalness, mat.specular,
                                    cam, specAlbedoLUT, specAvgAlbedoLUT);

            float ggxPdf = D * cosH / max(1e-7f, 4.0f * dot(wo, wh));
            float cosPdf = cosI / M_PI_F;
            bsdfPdf = pSpec * ggxPdf + pDiff * cosPdf;
        } else {
            // Cosine hemisphere (Lambertian or very rough kMatGGX)
            wi      = cosineSampleHemisphere(rand2(rng), n);
            bsdfPdf = max(1e-7f, dot(n, wi)) / M_PI_F;
            if (mat.type == kMatGGX) {
                bsdfF = evalLayeredBSDF(wo, wi, n, baseColor, mat.roughness,
                                        mat.metalness, mat.specular,
                                        cam, specAlbedoLUT, specAvgAlbedoLUT);
            } else {
                bsdfF = baseColor / M_PI_F;
            }
        }

        float cosI = dot(n, wi);
        if (cosI <= 0.0f || bsdfPdf <= 0.0f) break;
        throughput *= bsdfF * cosI / bsdfPdf;

        // Update MIS tracking before spawning next ray
        prevPos      = hitPos;
        prevN        = n;
        prevBsdfPdf  = bsdfPdf;
        prevWasDelta = false;

        // Spawn next ray
        r.origin       = hitPos + n * 1e-4f;
        r.direction    = wi;
        r.min_distance = 1e-4f;
        r.max_distance = 1e10f;
    }  // end bounce loop

        // Firefly clamp: scale L down if its luminance exceeds the threshold.
        if (cam.fireflyClamp > 0.0f) {
            float lum = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
            if (lum > cam.fireflyClamp)
                L *= cam.fireflyClamp / lum;
        }

        // Accumulate this sample into batch locals (signed by filter weight)
        batchL      += L * fw;
        batchWeight += fw;
        float lum = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
        batchLumSq += lum * lum;

    }  // end sample batch loop

    // Single write to accum buffer for the entire batch
    device GpuAccumPixel& px_out = accum[pixelIdx];
    px_out.r        += batchL.x;
    px_out.g        += batchL.y;
    px_out.b        += batchL.z;
    px_out.weight   += batchWeight;
    px_out.sumLumSq += batchLumSq;
}

// ---------------------------------------------------------------------------
// photonTrace kernel — one thread per photon slot.
//
// Each thread selects a light, samples a position and direction, then traces
// the photon through the scene.  The photon is stored when it first hits a
// diffuse surface after ≥1 specular (glass) bounce — i.e. only caustic
// photons are stored.  Slots with no valid hit have power == (0,0,0).
// ---------------------------------------------------------------------------
kernel void photonTrace(
    const device GpuLight*      lights           [[ buffer(0) ]],
    constant uint&              numLights        [[ buffer(1) ]],
    const device GpuMaterial*   materials        [[ buffer(2) ]],
    constant uint&              numMaterials     [[ buffer(3) ]],
    const device PackedFloat3*  normals          [[ buffer(4) ]],
    const device uint32_t*      indices          [[ buffer(5) ]],
    const device uint32_t*      meshIndexOffsets [[ buffer(6) ]],
    device GpuPhoton*           photons          [[ buffer(7) ]],
    constant GpuPhotonParams&   params           [[ buffer(8) ]],
    acceleration_structure<instancing, primitive_motion> accelStruct [[ buffer(9) ]],
    device GpuPhoton*           sssPhotons       [[ buffer(10) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= params.numPhotons) return;

    // Default: invalid slot
    photons[gid].power    = {0.f, 0.f, 0.f};
    sssPhotons[gid].power = {0.f, 0.f, 0.f};

    if (numLights == 0) return;

    uint rng = pcg(pcg(gid) ^ (params.frameIndex * 2654435761u + 1234567u));

    // Select light uniformly
    uint lightIdx = uint(rand01(rng) * float(numLights)) % numLights;
    const device GpuLight& light = lights[lightIdx];
    float lightSelectPdf = 1.f / float(numLights);

    float3 pos    = float3(0);
    float3 dir    = float3(0);
    float3 Le     = float3(0);
    float3 lightN = float3(0);
    float  pdfPos = 0.f;
    float  pdfDir = 0.f;

    if (light.type == kLightRect) {
        float3 uH = float3(light.uHalf.x, light.uHalf.y, light.uHalf.z);
        float3 vH = float3(light.vHalf.x, light.vHalf.y, light.vHalf.z);
        float2 u  = rand2(rng);
        pos = float3(light.position.x, light.position.y, light.position.z)
            + uH * (2.f*u.x - 1.f) + vH * (2.f*u.y - 1.f);
        pdfPos = 1.f / max(1e-7f, light.area);
        lightN = float3(light.normal.x, light.normal.y, light.normal.z);
        dir    = cosineSampleHemisphere(rand2(rng), lightN);
        float cosTheta = max(0.f, dot(dir, lightN));
        pdfDir = max(1e-7f, cosTheta / M_PI_F);
        Le     = float3(light.Le.x, light.Le.y, light.Le.z);
    } else if (light.type == kLightDirectional) {
        // Directional lights emit from an infinite plane — not useful for
        // caustics (focus to a point).  Skip.
        return;
    } else {
        return;
    }

    if (pdfPos <= 0.f || pdfDir <= 0.f) return;

    // Initial photon power — matches CPU PhotonMapIntegrator formula
    float cosTheta = max(0.f, dot(dir, lightN));
    float3 power = Le * cosTheta
                 / (lightSelectPdf * pdfPos * pdfDir * float(params.numPhotons));

    ray r;
    r.origin       = pos + lightN * 1e-3f;
    r.direction    = dir;
    r.min_distance = 1e-4f;
    r.max_distance = 1e10f;

    intersector<triangle_data, instancing, primitive_motion> isect;
    isect.accept_any_intersection(false);

    int numSpecular = 0;

    for (int bounce = 0; bounce < 8; ++bounce) {
        intersection_result<triangle_data, instancing, primitive_motion> res =
            isect.intersect(r, accelStruct, 0xFF, 0.5f);  // mid-frame time

        if (res.type == intersection_type::none) break;

        uint  meshID  = res.instance_id;
        uint  matIdx  = (meshID < numMaterials) ? meshID : 0;
        GpuMaterial mat = materials[matIdx];

        float2 bary   = res.triangle_barycentric_coord;
        uint   idxOff = meshIndexOffsets[meshID];
        float3 n      = interpolateNormal(res.primitive_id, bary, normals, indices, idxOff);
        float3 hitPos = r.origin + r.direction * res.distance;

        if (mat.type == kMatEmissive) break;  // hit emitter — stop

        if (mat.type == kMatGlass) {
            // Specular refraction/reflection bounce
            bool   entering = dot(r.direction, n) < 0.f;
            float3 faceN    = entering ? n : -n;
            float  eta      = entering ? (1.f / mat.specularIOR) : mat.specularIOR;
            float  cosI     = dot(-r.direction, faceN);
            float  Fr       = fresnelDielectric(cosI, 1.f / eta);

            if (rand01(rng) < Fr) {
                r.direction = reflect(r.direction, faceN);
                r.origin    = hitPos + faceN * 1e-4f;
            } else {
                float3 refr = refract(r.direction, faceN, eta);
                if (length_squared(refr) < 0.5f) {
                    r.direction = reflect(r.direction, faceN);
                    r.origin    = hitPos + faceN * 1e-4f;
                } else {
                    r.direction = normalize(refr);
                    r.origin    = hitPos - faceN * 1e-4f;
                }
            }
            r.min_distance = 1e-4f;
            r.max_distance = 1e10f;
            // Only specular bounces through caustic-flagged glass count toward
            // "this photon is a caustic photon" — see ShadowRay.h / IMaterial.
            if (mat.causticGenerator) ++numSpecular;
            // Delta surface — throughput unchanged

        } else if (mat.isSubsurface && mat.subsurfaceWeight > 0.f) {
            // SSS hit — deposit first SSS photon, scatter Lambertian, continue
            float entrycos  = max(0.f, -dot(r.direction, n));
            float absorbed  = entrycos * mat.subsurfaceWeight;
            if (absorbed > 0.f
                && sssPhotons[gid].power.x == 0.f
                && sssPhotons[gid].power.y == 0.f
                && sssPhotons[gid].power.z == 0.f) {
                float3 sp = power * absorbed;
                sssPhotons[gid].position = {hitPos.x, hitPos.y, hitPos.z};
                sssPhotons[gid].wi       = {r.direction.x, r.direction.y, r.direction.z};
                sssPhotons[gid].power    = {sp.x, sp.y, sp.z};
            }
            float transmitted = 1.f - absorbed;
            power *= transmitted;
            if (max(max(power.x, power.y), power.z) < 1e-6f) break;
            // Lambertian scatter and continue
            r.direction    = cosineSampleHemisphere(rand2(rng), n);
            r.origin       = hitPos + n * 1e-4f;
            r.min_distance = 1e-4f;
            r.max_distance = 1e10f;

        } else {
            // Diffuse / GGX hit
            if (numSpecular > 0) {
                // Valid caustic photon — store and stop
                photons[gid].position = {hitPos.x, hitPos.y, hitPos.z};
                photons[gid].wi       = {r.direction.x, r.direction.y, r.direction.z};
                photons[gid].power    = {power.x, power.y, power.z};
            }
            break;
        }
    }
}
