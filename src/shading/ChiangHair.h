#pragma once

// ---------------------------------------------------------------------------
// Chiang et al. 2016 hair BSDF
// "A Practical and Controllable Hair and Fur Model for Production Path Tracing"
//
// Extends the Marschner (2003) R/TT/TRT model with an analytic multiple
// scattering term that approximates inter-fiber light transport.  The scatter
// lobe uses a broad Gaussian longitudinal distribution and a uniform azimuthal
// distribution.  This dramatically reduces variance for light-coloured hair,
// where inter-fiber bouncing carries a large fraction of the visible energy.
//
// Same physical parameters as MarschnerHairMaterial (eta, sigma_a, beta_m,
// beta_n, alpha) — the two shaders are drop-in replacements for each other.
//
// Reference implementation: PBRT v4 HairBxDF, Disney/Pixar production notes.
// ---------------------------------------------------------------------------

#include "MarschnerHair.h"   // reuse all mh_* math helpers

namespace anacapa {

class ChiangHairMaterial : public IMaterial {
public:
    using Params = MarschnerHairMaterial::Params;

    explicit ChiangHairMaterial(const Params& p) : m_p(p) {
        // ---- Longitudinal variance (same curve-fit as Marschner) ----
        float bm = std::clamp(m_p.beta_m, 1e-3f, 1.f);
        float v0 = 0.726f * bm + 0.812f * bm * bm + 3.7f * std::pow(bm, 20.f);
        m_v[0] = v0 * v0;
        m_v[1] = m_v[0] * 0.25f;
        m_v[2] = m_v[0] * 4.0f;
        m_v_ms = m_v[2] * 4.0f;  // scatter: 4× TRT variance → broad forward lobe

        // ---- Azimuthal logistic scale ----
        float bn = std::clamp(m_p.beta_n, 1e-3f, 1.f);
        m_s = 0.626657069f * (0.265f * bn + 1.194f * bn * bn + 5.372f * std::pow(bn, 22.f));

        // ---- Lobe shift angles (radians) ----
        m_alpha[0] = -m_p.alpha * kMH_Pi / 180.f;
        m_alpha[1] =  m_p.alpha * kMH_Pi / 180.f * 0.5f;
        m_alpha[2] = -m_p.alpha * kMH_Pi / 180.f * 1.5f;

        // ---- Precompute multiple scatter weight ----
        // Average A_TT and A_TRT over h ∈ [-1,1] at cosThetaO = 1, then form
        // the geometric series for inter-fiber multiple scattering:
        //   m_Ams = avg_TT × avg_TRT / (1 - avg_TRT)
        // This represents energy that has transmitted through one fiber (TT),
        // reflected off a second fiber (TRT), and so on to infinity.  Using
        // avg_TT alone (as a naive "forward albedo") double-counts the TT lobe
        // that single-scatter already handles, making hair far too bright.
        Spectrum avg_TT{}, avg_TRT{};
        constexpr int kN = 32;
        for (int i = 0; i < kN; ++i) {
            float h  = -1.f + 2.f * (i + 0.5f) / kN;
            auto  ap = mh_Ap(1.f, m_p.eta, h, m_p.sigma_a);
            avg_TT  = avg_TT  + ap[1] * (1.f / kN);
            avg_TRT = avg_TRT + ap[2] * (1.f / kN);
        }
        // Use scalar luminance for the denominator so the per-channel color comes
        // only from avg_TT and is not nonlinearly amplified — a per-channel
        // denominator (1 - avg_TT.r vs 1 - avg_TT.b) creates a strong warm/cool
        // tinge for hair with spectrally uneven absorption.
        float lum_TT  = luminance(avg_TT);
        float lum_TRT = luminance(avg_TRT);
        float scale   = lum_TRT / std::max(1.f - lum_TT, 1e-4f);
        m_Ams = avg_TT * scale;
    }

    bool     isDelta()   const override { return false; }
    float    roughness() const override { return m_p.beta_m; }
    uint32_t flags()     const override {
        return BSDFFlag_Glossy | BSDFFlag_Reflection | BSDFFlag_Transmission;
    }

    Spectrum reflectance(const ShadingContext& ctx) const override {
        Spectrum a = effectiveSigmaA(ctx);
        return { std::exp(-a.x), std::exp(-a.y), std::exp(-a.z) };
    }

    // -----------------------------------------------------------------------
    // evaluate
    // -----------------------------------------------------------------------
    BSDFEval evaluate(const ShadingContext& ctx,
                      Vec3f wo, Vec3f wi) const override
    {
        float sinThetaO, cosThetaO, sinThetaI, cosThetaI;
        hairLongitudinal(wo, ctx.t, sinThetaO, cosThetaO);
        hairLongitudinal(wi, ctx.t, sinThetaI, cosThetaI);

        float phi  = hairPhi(wo, wi, ctx.t, sinThetaO, sinThetaI);
        float h    = ctx.h;
        Spectrum sigA = effectiveSigmaA(ctx);

        Spectrum f  = evalLobes(sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi, h, sigA);
        float    pd = evalPdf(sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi, h, sigA);

        BSDFEval e;
        e.f      = f;
        e.pdf    = pd;
        e.pdfRev = evalPdf(sinThetaI, cosThetaI, sinThetaO, cosThetaO, -phi, h, sigA);
        return e;
    }

    // -----------------------------------------------------------------------
    // sample — 4-lobe importance sampling (R, TT, TRT, scatter)
    // -----------------------------------------------------------------------
    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float uComp) const override
    {
        float sinThetaO, cosThetaO;
        hairLongitudinal(wo, ctx.t, sinThetaO, cosThetaO);

        float    h    = ctx.h;
        Spectrum sigA = effectiveSigmaA(ctx);
        auto ap = mh_Ap(cosThetaO, m_p.eta, h, sigA);

        // Four-lobe discrete weights
        float weights[4];
        for (int p = 0; p < 3; ++p) weights[p] = luminance(ap[p]);
        weights[3] = luminance(m_Ams);

        float total = weights[0] + weights[1] + weights[2] + weights[3];
        if (total < 1e-8f) return {};
        float inv = 1.f / total;

        float cdf[4];
        cdf[0] = weights[0] * inv;
        cdf[1] = cdf[0] + weights[1] * inv;
        cdf[2] = cdf[1] + weights[2] * inv;
        cdf[3] = 1.f;

        int   lobe = 3;
        float uRemap = uComp;
        if      (uComp < cdf[0]) { lobe = 0; uRemap = uComp / cdf[0]; }
        else if (uComp < cdf[1]) { lobe = 1; uRemap = (uComp - cdf[0]) / (cdf[1] - cdf[0]); }
        else if (uComp < cdf[2]) { lobe = 2; uRemap = (uComp - cdf[1]) / (cdf[2] - cdf[1]); }
        else                     {            uRemap = (uComp - cdf[2]) / (cdf[3] - cdf[2]); }

        // Longitudinal sample — Gaussian for all lobes
        float thetaO = std::asin(std::clamp(sinThetaO, -1.f+1e-5f, 1.f-1e-5f));
        float z      = mh_sampleGaussian(u.x, u.y);

        float sinThetaI, cosThetaI;
        if (lobe < 3) {
            float thetaI = std::clamp(thetaO + m_alpha[lobe] + z * std::sqrt(m_v[lobe]),
                                      -kMH_Pi * 0.5f, kMH_Pi * 0.5f);
            sinThetaI = std::sin(thetaI);
            cosThetaI = std::cos(thetaI);
        } else {
            // Scatter lobe: no cuticle tilt shift, broader width
            float thetaI = std::clamp(thetaO + z * std::sqrt(m_v_ms),
                                      -kMH_Pi * 0.5f, kMH_Pi * 0.5f);
            sinThetaI = std::sin(thetaI);
            cosThetaI = std::cos(thetaI);
        }

        // Azimuthal sample
        float sin2ThetaO = 1.f - cosThetaO * cosThetaO;
        float etaP = std::sqrt(std::max(0.f, m_p.eta*m_p.eta - sin2ThetaO)) / cosThetaO;
        float gammaO = std::asin(std::clamp(h, -1.f+1e-5f, 1.f-1e-5f));
        float sinGT  = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f+1e-5f, 1.f-1e-5f);
        float gammaT = std::asin(sinGT);

        float phiSampled;
        if (lobe < 3) {
            phiSampled = mh_sampleTrimmedLogistic(uRemap, m_s) + mh_Phi(lobe, gammaO, gammaT);
        } else {
            // Scatter lobe: uniform azimuth
            phiSampled = uRemap * kMH_2Pi - kMH_Pi;
        }

        // Reconstruct wi
        Vec3f woPerp = safeNormalize(wo - ctx.t * sinThetaO);
        if (woPerp.lengthSq() < 1e-8f) {
            Vec3f dummy;
            buildOrthonormalBasis(ctx.t, woPerp, dummy);
        }
        Vec3f ctPerp = cross(ctx.t, woPerp);

        Vec3f wi = ctx.t   * sinThetaI
                 + woPerp  * (cosThetaI * std::cos(phiSampled))
                 + ctPerp  * (cosThetaI * std::sin(phiSampled));
        wi = safeNormalize(wi);

        float sinThetaI_a, cosThetaI_a;
        hairLongitudinal(wi, ctx.t, sinThetaI_a, cosThetaI_a);
        float phiActual = hairPhi(wo, wi, ctx.t, sinThetaO, sinThetaI_a);

        Spectrum f  = evalLobes(sinThetaO, cosThetaO, sinThetaI_a, cosThetaI_a, phiActual, h, sigA);
        float    pd = evalPdf(sinThetaO, cosThetaO, sinThetaI_a, cosThetaI_a, phiActual, h, sigA);
        if (pd < 1e-8f) return {};

        BSDFSample bs;
        bs.wi     = wi;
        bs.f      = f * cosThetaI_a;
        bs.pdf    = pd;
        bs.pdfRev = evalPdf(sinThetaI_a, cosThetaI_a, sinThetaO, cosThetaO, -phiActual, h, sigA);
        bs.flags  = BSDFFlag_Glossy | BSDFFlag_Reflection | BSDFFlag_Transmission;
        bs.eta    = 1.f;
        return bs;
    }

    float pdf(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const override
    {
        float sinThetaO, cosThetaO, sinThetaI, cosThetaI;
        hairLongitudinal(wo, ctx.t, sinThetaO, cosThetaO);
        hairLongitudinal(wi, ctx.t, sinThetaI, cosThetaI);
        float phi = hairPhi(wo, wi, ctx.t, sinThetaO, sinThetaI);
        return evalPdf(sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi, ctx.h, effectiveSigmaA(ctx));
    }

private:
    // -----------------------------------------------------------------------
    // Geometry helpers (identical to MarschnerHairMaterial)
    // -----------------------------------------------------------------------
    static void hairLongitudinal(Vec3f d, Vec3f t, float& sinTheta, float& cosTheta) {
        sinTheta = dot(d, t);
        cosTheta = std::sqrt(std::max(0.f, 1.f - sinTheta * sinTheta));
    }

    static float hairPhi(Vec3f wo, Vec3f wi, Vec3f t, float sinThetaO, float sinThetaI) {
        Vec3f woP = wo - t * sinThetaO, wiP = wi - t * sinThetaI;
        float lO = woP.length(), lI = wiP.length();
        if (lO < 1e-5f || lI < 1e-5f) return 0.f;
        woP = woP * (1.f / lO); wiP = wiP * (1.f / lI);
        float c = std::clamp(dot(woP, wiP), -1.f, 1.f);
        float s = dot(cross(wiP, woP), t);
        return std::atan2(s, c);
    }

    Spectrum effectiveSigmaA(const ShadingContext& ctx) const {
        const Vec3f& c = ctx.color;
        if (c.x > 0.98f && c.y > 0.98f && c.z > 0.98f)
            return m_p.sigma_a;
        return { -std::log(std::max(c.x, 0.001f)),
                 -std::log(std::max(c.y, 0.001f)),
                 -std::log(std::max(c.z, 0.001f)) };
    }

    // -----------------------------------------------------------------------
    // evalLobes — R + TT + TRT + multiple scatter
    // -----------------------------------------------------------------------
    Spectrum evalLobes(float sinThetaO, float cosThetaO,
                       float sinThetaI, float cosThetaI,
                       float phi, float h, Spectrum sigma_a) const
    {
        float cosThetaD = std::sqrt(std::max(0.f,
            0.5f * (1.f + cosThetaO*cosThetaI + sinThetaO*sinThetaI)));
        float denom = std::max(1e-5f, cosThetaD * cosThetaD);

        float sin2ThetaO = 1.f - cosThetaO * cosThetaO;
        float etaP  = std::sqrt(std::max(0.f, m_p.eta*m_p.eta - sin2ThetaO)) / cosThetaO;
        float gammaO = std::asin(std::clamp(h, -1.f+1e-5f, 1.f-1e-5f));
        float sinGT  = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f+1e-5f, 1.f-1e-5f);
        float gammaT = std::asin(sinGT);

        auto ap = mh_Ap(cosThetaO, m_p.eta, h, sigma_a);

        Spectrum fsum{};
        for (int p = 0; p < 3; ++p) {
            float sinOs = sinThetaO * std::cos(2.f * m_alpha[p])
                        + cosThetaO * std::sin(2.f * m_alpha[p]);
            float cosOs = std::sqrt(std::max(0.f, 1.f - sinOs * sinOs));
            float mp = mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, m_v[p]);
            float np = mh_Np(phi, p, m_s, gammaO, gammaT);
            fsum = fsum + ap[p] * (mp * np);
        }

        // Multiple scatter — broad Gaussian × uniform azimuth × precomputed albedo
        float M_ms = mh_Mp(cosThetaI, sinThetaI, cosThetaO, sinThetaO, m_v_ms);
        fsum = fsum + m_Ams * (M_ms * 0.5f * kMH_InvPi);

        return fsum * (1.f / denom);
    }

    // -----------------------------------------------------------------------
    // evalPdf — mixture over 4 lobes
    // -----------------------------------------------------------------------
    float evalPdf(float sinThetaO, float cosThetaO,
                  float sinThetaI, float cosThetaI,
                  float phi, float h, Spectrum sigma_a) const
    {
        float sin2ThetaO = 1.f - cosThetaO * cosThetaO;
        float etaP  = std::sqrt(std::max(0.f, m_p.eta*m_p.eta - sin2ThetaO)) / cosThetaO;
        float gammaO = std::asin(std::clamp(h, -1.f+1e-5f, 1.f-1e-5f));
        float sinGT  = std::clamp(h / (etaP > 1e-5f ? etaP : 1e-5f), -1.f+1e-5f, 1.f-1e-5f);
        float gammaT = std::asin(sinGT);

        auto ap = mh_Ap(cosThetaO, m_p.eta, h, sigma_a);
        float weights[4];
        for (int p = 0; p < 3; ++p) weights[p] = luminance(ap[p]);
        weights[3] = luminance(m_Ams);
        float total = weights[0]+weights[1]+weights[2]+weights[3];
        if (total < 1e-8f) return 0.f;

        float pd = 0.f;
        for (int p = 0; p < 3; ++p) {
            float sinOs = sinThetaO * std::cos(2.f * m_alpha[p])
                        + cosThetaO * std::sin(2.f * m_alpha[p]);
            float cosOs = std::sqrt(std::max(0.f, 1.f - sinOs * sinOs));
            float mp = mh_Mp(cosThetaI, sinThetaI, cosOs, sinOs, m_v[p]) * cosThetaI;
            float np = mh_Np(phi, p, m_s, gammaO, gammaT);
            pd += (weights[p] / total) * mp * np;
        }

        // Scatter lobe: Gaussian longitudinal, uniform azimuthal
        float M_ms = mh_Mp(cosThetaI, sinThetaI, cosThetaO, sinThetaO, m_v_ms) * cosThetaI;
        pd += (weights[3] / total) * M_ms * (0.5f * kMH_InvPi);

        return std::max(0.f, pd);
    }

    // -----------------------------------------------------------------------
    // Data
    // -----------------------------------------------------------------------
    Params   m_p;
    float    m_v[3];      // longitudinal variance [R, TT, TRT]
    float    m_v_ms;      // scatter lobe variance (4× TRT)
    float    m_s;         // azimuthal logistic scale
    float    m_alpha[3];  // lobe tilt shifts [R, TT, TRT]
    Spectrum m_Ams;       // precomputed average TT attenuation (multiple scatter weight)
};

} // namespace anacapa
