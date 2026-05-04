#pragma once

// ---------------------------------------------------------------------------
// Marschner (2003) hair BSDF — "Light Scattering from Human Hair Fibers"
//
// Three-lobe model:
//   R   (p=0): specular reflection off the outer cuticle
//   TT  (p=1): two transmissions through the cortex (forward scatter)
//   TRT (p=2): two transmissions + one internal reflection (back-scatter glow)
//
// Parameterization:
//   θ (theta): longitudinal angle from the fiber cross-section plane
//              sinθ = dot(direction, fiber_tangent)
//   φ (phi):   azimuthal angle in the cross-section plane
//
// Coordinate convention (in ShadingContext):
//   ctx.t  = fiber tangent (dpdu, set by CurveBrute intersection)
//   ctx.n  = ribbon facing normal (toward camera at hit time)
//   ctx.uv.y = strand parameter v ∈ [0,1] (root → tip)
//
// The impact parameter h is derived from the ribbon normal:
//   h = dot(wo_cross_section_normalized, ctx.n)  ∈ [-1, 1]
//
// References:
//   Marschner et al., 2003 — original paper
//   d'Eon et al., 2011     — energy-conserving extension
//   PBRT v4 Hair.cpp       — reference implementation
// ---------------------------------------------------------------------------

#include <anacapa/shading/IMaterial.h>
#include <cmath>
#include <array>
#include <algorithm>

namespace anacapa {

// ---------------------------------------------------------------------------
// Math constants
// ---------------------------------------------------------------------------
static constexpr float kMH_Pi     = 3.14159265358979323846f;
static constexpr float kMH_InvPi  = 0.31830988618379067154f;
static constexpr float kMH_2Pi    = 6.28318530717958647692f;
static constexpr float kMH_Sqrt2Pi= 2.50662827463100050242f;

// ---------------------------------------------------------------------------
// Fresnel (dielectric, exact) — duplicated from StandardSurface to avoid
// header coupling.
// ---------------------------------------------------------------------------
inline float mh_fresnelDielectric(float cosI, float eta) {
    float sin2I = std::max(0.f, 1.f - cosI * cosI);
    float sin2T = sin2I / (eta * eta);
    if (sin2T >= 1.f) return 1.f;  // TIR
    float cosT = std::sqrt(1.f - sin2T);
    float Rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    float Rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// ---------------------------------------------------------------------------
// Modified Bessel I₀ (series expansion, sufficient for |x| < ~15)
// ---------------------------------------------------------------------------
inline float mh_I0(float x) {
    float sum  = 0.f;
    float x2i  = 1.f;  // x^(2i)
    float denom = 1.f; // (i!)^2 * 4^i
    for (int i = 0; i < 12; ++i) {
        if (i > 0) { x2i *= x * x; denom *= float(i) * float(i) * 4.f; }
        sum += x2i / denom;
    }
    return sum;
}

inline float mh_logI0(float x) {
    if (x > 12.f)
        return x + 0.5f * (-std::log(kMH_2Pi) + std::log(1.f / x) + 1.f / (8.f * x));
    return std::log(mh_I0(x));
}

// ---------------------------------------------------------------------------
// M_p — longitudinal scattering (von Mises–Fisher / trimmed Gaussian)
//
// M_p(θ_i, θ_r) ≈ VMF(cosθ_i · cosθ_r / v) · exp(−sinθ_i · sinθ_r / v)
//                  normalised over θ_i ∈ [−π/2, π/2]
//
// v is the variance parameter (σ²) for lobe p.
// The lobe shift α_p is applied to θ_r before calling: sinR_shifted, cosR_shifted.
// ---------------------------------------------------------------------------
inline float mh_Mp(float cosI, float sinI,
                   float cosR, float sinR,
                   float v)
{
    if (v < 1e-5f) v = 1e-5f;
    float a = cosI * cosR / v;
    float b = sinI * sinR / v;
    if (v <= 0.1f)
        return std::exp(mh_logI0(a) - b - 1.f / v + 0.6931472f + std::log(0.5f / v));
    return std::exp(-b) * mh_I0(a) / (2.f * v * std::sinh(1.f / v));
}

// ---------------------------------------------------------------------------
// Logistic distribution for azimuthal N_p
// ---------------------------------------------------------------------------
inline float mh_logistic(float x, float s) {
    float ex = std::exp(-std::abs(x) / s);
    return ex / (s * (1.f + ex) * (1.f + ex));
}

inline float mh_logisticCDF(float x, float s) {
    return 1.f / (1.f + std::exp(-x / s));
}

// Trimmed logistic on [−π, π], normalised to integrate to 1
inline float mh_trimmedLogistic(float x, float s) {
    float norm = mh_logisticCDF(kMH_Pi, s) - mh_logisticCDF(-kMH_Pi, s);
    return mh_logistic(x, s) / norm;
}

// Inversion-method sample from the trimmed logistic
inline float mh_sampleTrimmedLogistic(float u, float s) {
    float a = mh_logisticCDF(-kMH_Pi, s);
    float b = mh_logisticCDF( kMH_Pi, s);
    float x = -s * std::log(1.f / (a + u * (b - a)) - 1.f);
    return std::clamp(x, -kMH_Pi, kMH_Pi);
}

// ---------------------------------------------------------------------------
// Phi(p, γ_o, γ_t) — specular azimuthal direction for lobe p
// ---------------------------------------------------------------------------
inline float mh_Phi(int p, float gammaO, float gammaT) {
    return 2.f * float(p) * gammaT - 2.f * gammaO + float(p) * kMH_Pi;
}

inline float mh_wrapPhi(float x) {
    while (x >  kMH_Pi) x -= kMH_2Pi;
    while (x < -kMH_Pi) x += kMH_2Pi;
    return x;
}

// Azimuthal scattering N_p: logistic centred at Phi(p, γ_o, γ_t)
inline float mh_Np(float phi, int p, float s, float gammaO, float gammaT) {
    return mh_trimmedLogistic(mh_wrapPhi(phi - mh_Phi(p, gammaO, gammaT)), s);
}

// ---------------------------------------------------------------------------
// A_p — lobe attenuations [R, TT, TRT]
// ---------------------------------------------------------------------------
inline std::array<Spectrum, 3> mh_Ap(float cosThetaO, float eta, float h,
                                      const Spectrum& sigma_a)
{
    // Modified IOR for the longitudinal angle
    float sin2ThetaO = std::max(0.f, 1.f - cosThetaO * cosThetaO);
    float etaP = std::sqrt(std::max(0.f, eta * eta - sin2ThetaO)) / cosThetaO;

    // Impact parameter refraction angle
    float sinGammaT = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
    float cosGammaT = std::sqrt(std::max(0.f, 1.f - sinGammaT * sinGammaT));

    // Absorption: Beer's law along one chord through unit cylinder
    // Path length inside = 2 * cosGammaT
    auto expAbs = [&](float c) { return std::exp(-sigma_a.x * 2.f * c); };
    Spectrum T = {
        std::exp(-sigma_a.x * 2.f * cosGammaT),
        std::exp(-sigma_a.y * 2.f * cosGammaT),
        std::exp(-sigma_a.z * 2.f * cosGammaT)
    };

    // Fresnel at the outer surface
    float cosGammaO = std::sqrt(std::max(0.f, 1.f - h * h));
    float cosAngle  = std::max(0.f, cosThetaO * cosGammaO);
    float fr        = mh_fresnelDielectric(cosAngle, eta);

    std::array<Spectrum, 3> ap;
    ap[0] = { fr, fr, fr };                              // R
    ap[1] = (1.f - fr) * (1.f - fr) * T;                // TT
    ap[2] = ap[1] * T * fr;                              // TRT
    return ap;
}

// ---------------------------------------------------------------------------
// Box–Muller: two uniform samples → one standard-normal sample
// ---------------------------------------------------------------------------
inline float mh_sampleGaussian(float u1, float u2) {
    u1 = std::max(u1, 1e-6f);
    return std::sqrt(-2.f * std::log(u1)) * std::cos(kMH_2Pi * u2);
}

// ===========================================================================
// MarschnerHairMaterial
// ===========================================================================
class MarschnerHairMaterial : public IMaterial {
public:
    struct Params {
        float    eta     = 1.55f;  // IOR of the cortex (1.55 typical for human hair)
        Spectrum sigma_a = { 0.84f, 1.39f, 2.74f }; // absorption (medium brown hair, PBRT v4)
        float    beta_m  = 0.30f;  // longitudinal roughness ∈ [0, 1]
        float    beta_n  = 0.30f;  // azimuthal roughness    ∈ [0, 1]
        float    alpha   = 2.f;    // cuticle scale tilt in degrees (typical 2–4°)
    };

    explicit MarschnerHairMaterial(const Params& p) : m_p(p) {
        // Longitudinal variance — PBRT v4 curve-fit mapping from β_m
        float bm  = std::clamp(m_p.beta_m, 1e-3f, 1.f);
        float v0  = 0.726f * bm + 0.812f * bm * bm + 3.7f * std::pow(bm, 20.f);
        m_v[0] = v0 * v0;
        m_v[1] = m_v[0] * 0.25f;
        m_v[2] = m_v[0] * 4.0f;

        // Azimuthal logistic scale — PBRT v4 curve-fit from β_n
        float bn = std::clamp(m_p.beta_n, 1e-3f, 1.f);
        m_s = 0.626657069f * (0.265f * bn + 1.194f * bn * bn + 5.372f * std::pow(bn, 22.f));

        // Lobe shift angles in radians (signed)
        m_alpha[0] = -m_p.alpha * kMH_Pi / 180.f;          // R:   −α
        m_alpha[1] =  m_p.alpha * kMH_Pi / 180.f * 0.5f;   // TT: +α/2
        m_alpha[2] = -m_p.alpha * kMH_Pi / 180.f * 1.5f;   // TRT: −3α/2
    }

    bool     isDelta()    const override { return false; }
    float    roughness()  const override { return m_p.beta_m; }
    uint32_t flags()      const override {
        return BSDFFlag_Glossy | BSDFFlag_Reflection | BSDFFlag_Transmission;
    }

    Spectrum reflectance(const ShadingContext& ctx) const override {
        // Rough estimate: energy that isn't absorbed
        Spectrum absorbed = effectiveSigmaA(ctx);
        return {
            std::exp(-absorbed.x),
            std::exp(-absorbed.y),
            std::exp(-absorbed.z)
        };
    }

    // -----------------------------------------------------------------------
    // evaluate — pure BSDF f(wo, wi), no cosine factor
    // -----------------------------------------------------------------------
    BSDFEval evaluate(const ShadingContext& ctx,
                      Vec3f wo, Vec3f wi) const override
    {
        float sinThetaO, cosThetaO, sinThetaI, cosThetaI;
        hairLongitudinal(wo, ctx.t, sinThetaO, cosThetaO);
        hairLongitudinal(wi, ctx.t, sinThetaI, cosThetaI);

        float phi = hairPhi(wo, wi, ctx.t, sinThetaO, sinThetaI);
        float h   = hairImpactParam(wo, ctx.t, ctx.n, sinThetaO);

        Spectrum sigA = effectiveSigmaA(ctx);
        Spectrum f    = evalLobes(sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi, h, sigA);
        float    pd   = evalPdf(sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi, h, sigA);

        BSDFEval e;
        e.f      = f;
        e.pdf    = pd;
        e.pdfRev = evalPdf(sinThetaI, cosThetaI, sinThetaO, cosThetaO, -phi, h, sigA);
        return e;
    }

    // -----------------------------------------------------------------------
    // sample — importance-sample a wi direction
    // -----------------------------------------------------------------------
    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float uComp) const override
    {
        float sinThetaO, cosThetaO;
        hairLongitudinal(wo, ctx.t, sinThetaO, cosThetaO);

        float h = hairImpactParam(wo, ctx.t, ctx.n, sinThetaO);

        // Lobe attenuations for lobe selection
        Spectrum sigA = effectiveSigmaA(ctx);
        auto ap = mh_Ap(cosThetaO, m_p.eta, h, sigA);
        float weights[3];
        for (int p = 0; p < 3; ++p)
            weights[p] = luminance(ap[p]);

        // Normalise + CDF for discrete lobe selection
        float total = weights[0] + weights[1] + weights[2];
        if (total < 1e-8f) return {};
        float invTotal = 1.f / total;
        float cdf[3];
        cdf[0] = weights[0] * invTotal;
        cdf[1] = cdf[0] + weights[1] * invTotal;
        cdf[2] = 1.f;

        // Select lobe and remap uComp to [0,1] within the chosen bin
        int lobe = 0;
        float uRemapped = uComp;
        if (uComp < cdf[0]) {
            uRemapped = uComp / cdf[0];
        } else if (uComp < cdf[1]) {
            lobe      = 1;
            uRemapped = (uComp - cdf[0]) / (cdf[1] - cdf[0]);
        } else {
            lobe      = 2;
            uRemapped = (uComp - cdf[1]) / (cdf[2] - cdf[1]);
        }

        // Sample longitudinal angle for the selected lobe via Box–Muller
        float thetaO = std::asin(std::clamp(sinThetaO, -1.f + 1e-5f, 1.f - 1e-5f));
        float z      = mh_sampleGaussian(u.x, u.y);
        float thetaI = thetaO + m_alpha[lobe] + z * std::sqrt(m_v[lobe]);
        thetaI       = std::clamp(thetaI, -kMH_Pi * 0.5f, kMH_Pi * 0.5f);
        float sinThetaI = std::sin(thetaI);
        float cosThetaI = std::cos(thetaI);

        // Geometry for azimuthal sampling
        float sin2ThetaO = 1.f - cosThetaO * cosThetaO;
        float etaP = std::sqrt(std::max(0.f, m_p.eta*m_p.eta - sin2ThetaO)) / cosThetaO;
        float gammaO  = std::asin(std::clamp(h, -1.f + 1e-5f, 1.f - 1e-5f));
        float sinGammaT = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
        float gammaT    = std::asin(sinGammaT);

        // Sample azimuthal angle from trimmed logistic centred at Phi(lobe)
        float phiSampled = mh_sampleTrimmedLogistic(uRemapped, m_s)
                         + mh_Phi(lobe, gammaO, gammaT);

        // Reconstruct wi from (sinThetaI, cosThetaI, phi)
        // phi is measured relative to wo's cross-section projection
        Vec3f woPerp = safeNormalize(wo - ctx.t * sinThetaO);
        if (woPerp.lengthSq() < 1e-8f) {
            // wo is nearly parallel to tangent — pick an arbitrary perp
            Vec3f tmp;
            Vec3f dummy;
            buildOrthonormalBasis(ctx.t, woPerp, dummy);
        }
        Vec3f ctPerp = cross(ctx.t, woPerp);  // 3rd axis of cross-section frame

        Vec3f wi = ctx.t   * sinThetaI
                 + woPerp  * (cosThetaI * std::cos(phiSampled))
                 + ctPerp  * (cosThetaI * std::sin(phiSampled));
        wi = safeNormalize(wi);

        float sinThetaI_actual, cosThetaI_actual;
        hairLongitudinal(wi, ctx.t, sinThetaI_actual, cosThetaI_actual);
        float phiActual = hairPhi(wo, wi, ctx.t, sinThetaO, sinThetaI_actual);

        Spectrum f  = evalLobes(sinThetaO, cosThetaO,
                                sinThetaI_actual, cosThetaI_actual,
                                phiActual, h, sigA);
        float    pd = evalPdf(sinThetaO, cosThetaO,
                              sinThetaI_actual, cosThetaI_actual,
                              phiActual, h, sigA);
        if (pd < 1e-8f) return {};

        float cosSurf = std::abs(dot(wi, ctx.n));  // integrator's cosine factor

        BSDFSample bs;
        bs.wi    = wi;
        bs.f     = f * cosSurf;    // convention: f includes |cos θᵢ|
        bs.pdf   = pd;
        bs.pdfRev = evalPdf(sinThetaI_actual, cosThetaI_actual,
                             sinThetaO, cosThetaO, -phiActual, h, sigA);
        bs.flags = BSDFFlag_Glossy | BSDFFlag_Reflection | BSDFFlag_Transmission;
        bs.eta   = 1.f;
        return bs;
    }

    float pdf(const ShadingContext& ctx,
              Vec3f wo, Vec3f wi) const override
    {
        float sinThetaO, cosThetaO, sinThetaI, cosThetaI;
        hairLongitudinal(wo, ctx.t, sinThetaO, cosThetaO);
        hairLongitudinal(wi, ctx.t, sinThetaI, cosThetaI);
        float phi = hairPhi(wo, wi, ctx.t, sinThetaO, sinThetaI);
        float h   = hairImpactParam(wo, ctx.t, ctx.n, sinThetaO);
        return evalPdf(sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi, h, effectiveSigmaA(ctx));
    }

private:
    // -----------------------------------------------------------------------
    // Per-strand color → absorption coefficient
    // If ctx.color is non-white (< 0.999 on any channel), derive sigma_a from it.
    // Formula: sigma_a = -log(max(c, 0.001))  (Beer-Lambert inversion, scale=1)
    // White color (default) falls back to m_p.sigma_a for the material default.
    // -----------------------------------------------------------------------
    Spectrum effectiveSigmaA(const ShadingContext& ctx) const {
        const Vec3f& c = ctx.color;
        // White (1,1,1) is the default — no per-strand override assigned.
        // Threshold at 0.98 to catch near-white values (e.g. 0.991) that are
        // binary selection masks, not true hair color.
        if (c.x > 0.98f && c.y > 0.98f && c.z > 0.98f)
            return m_p.sigma_a;
        // Blender BYTE_COLOR is sRGB — linearise before Beer-Lambert inversion.
        auto toLinear = [](float s) { return std::pow(std::max(s, 0.0f), 2.2f); };
        float lr = toLinear(c.x), lg = toLinear(c.y), lb = toLinear(c.z);
        // Near-black (all channels < 0.01 linear) means an unpainted attribute
        // (Blender default black).  Fall back to material default rather than
        // rendering near-black hair.
        if (lr < 0.01f && lg < 0.01f && lb < 0.01f)
            return m_p.sigma_a;
        return {
            -std::log(std::max(lr, 0.001f)),
            -std::log(std::max(lg, 0.001f)),
            -std::log(std::max(lb, 0.001f))
        };
    }

    // -----------------------------------------------------------------------
    // Hair geometry helpers
    // -----------------------------------------------------------------------

    // Longitudinal angle of direction d relative to the fiber tangent t
    static void hairLongitudinal(Vec3f d, Vec3f t,
                                  float& sinTheta, float& cosTheta)
    {
        sinTheta = dot(d, t);
        cosTheta = std::sqrt(std::max(0.f, 1.f - sinTheta * sinTheta));
    }

    // Azimuthal difference angle φ = φ_i − φ_o, signed ∈ [−π, π]
    static float hairPhi(Vec3f wo, Vec3f wi, Vec3f t,
                          float sinThetaO, float sinThetaI)
    {
        Vec3f woPerp = wo - t * sinThetaO;
        Vec3f wiPerp = wi - t * sinThetaI;
        float lenO = woPerp.length(), lenI = wiPerp.length();
        if (lenO < 1e-5f || lenI < 1e-5f) return 0.f;
        woPerp = woPerp * (1.f / lenO);
        wiPerp = wiPerp * (1.f / lenI);
        float c = std::clamp(dot(woPerp, wiPerp), -1.f, 1.f);
        float s = dot(cross(wiPerp, woPerp), t);  // signed with tangent as axis
        return std::atan2(s, c);
    }

    // Impact parameter h from the outgoing direction projected onto the ribbon normal
    static float hairImpactParam(Vec3f wo, Vec3f t, Vec3f n, float sinThetaO) {
        Vec3f woPerp = wo - t * sinThetaO;
        float lenP = woPerp.length();
        if (lenP < 1e-5f) return 0.f;
        return std::clamp(dot(woPerp * (1.f / lenP), n), -1.f + 1e-5f, 1.f - 1e-5f);
    }

    // -----------------------------------------------------------------------
    // Core evaluation: Σ_p M_p · A_p · N_p / cos²(θ_d)
    // -----------------------------------------------------------------------
    Spectrum evalLobes(float sinThetaO, float cosThetaO,
                       float sinThetaI, float cosThetaI,
                       float phi, float h, Spectrum sigma_a) const
    {
        // cos(θ_d) where θ_d = (θ_i − θ_o)/2
        // Using the identity: cos(θ_d) = sqrt((1 + cosO·cosI + sinO·sinI) / 2)
        float cosThetaD = std::sqrt(std::max(0.f,
            0.5f * (1.f + cosThetaO * cosThetaI + sinThetaO * sinThetaI)));
        float denom = std::max(1e-5f, cosThetaD * cosThetaD);

        // Modified IOR and geometry
        float sin2ThetaO = 1.f - cosThetaO * cosThetaO;
        float etaP = std::sqrt(std::max(0.f, m_p.eta*m_p.eta - sin2ThetaO)) / cosThetaO;
        float gammaO = std::asin(std::clamp(h, -1.f + 1e-5f, 1.f - 1e-5f));
        float sinGT  = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
        float gammaT = std::asin(sinGT);

        auto ap = mh_Ap(cosThetaO, m_p.eta, h, sigma_a);

        Spectrum fsum{};
        for (int p = 0; p < 3; ++p) {
            // Apply longitudinal shift: rotate θ_o by α_p
            float sinOs = sinThetaO * std::cos(2.f * m_alpha[p])
                        + cosThetaO * std::sin(2.f * m_alpha[p]);
            float cosOs = std::sqrt(std::max(0.f, 1.f - sinOs * sinOs));

            float mp = mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, m_v[p]);
            float np = mh_Np(phi, p, m_s, gammaO, gammaT);
            fsum = fsum + ap[p] * (mp * np);
        }

        return fsum * (1.f / denom);
    }

    // -----------------------------------------------------------------------
    // PDF: mixture of M_p (Gaussian in θ_i) · N_p (logistic in φ)
    //      weighted by lobe energy, times cosThetaI for solid-angle normalisation
    // -----------------------------------------------------------------------
    float evalPdf(float sinThetaO, float cosThetaO,
                  float sinThetaI, float cosThetaI,
                  float phi, float h, Spectrum sigma_a) const
    {
        float sin2ThetaO = 1.f - cosThetaO * cosThetaO;
        float etaP = std::sqrt(std::max(0.f, m_p.eta*m_p.eta - sin2ThetaO)) / cosThetaO;
        float gammaO = std::asin(std::clamp(h, -1.f + 1e-5f, 1.f - 1e-5f));
        float sinGT  = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f + 1e-5f, 1.f - 1e-5f);
        float gammaT = std::asin(sinGT);

        auto ap = mh_Ap(cosThetaO, m_p.eta, h, sigma_a);
        float weights[3];
        for (int p = 0; p < 3; ++p) weights[p] = luminance(ap[p]);
        float total = weights[0] + weights[1] + weights[2];
        if (total < 1e-8f) return 0.f;

        float pdf = 0.f;
        for (int p = 0; p < 3; ++p) {
            float sinOs = sinThetaO * std::cos(2.f * m_alpha[p])
                        + cosThetaO * std::sin(2.f * m_alpha[p]);
            float cosOs = std::sqrt(std::max(0.f, 1.f - sinOs * sinOs));

            // M_p pdf is normalised over dθ_i; multiply by cosThetaI to get solid-angle pdf
            float mp = mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, m_v[p]) * cosThetaI;
            float np = mh_Np(phi, p, m_s, gammaO, gammaT);
            pdf += (weights[p] / total) * mp * np;
        }
        return std::max(0.f, pdf);
    }

    // -----------------------------------------------------------------------
    // Data
    // -----------------------------------------------------------------------
    Params  m_p;
    float   m_v[3];      // longitudinal variance [R, TT, TRT]
    float   m_s;         // azimuthal logistic scale
    float   m_alpha[3];  // lobe shift angles in radians [R, TT, TRT]
};

} // namespace anacapa
