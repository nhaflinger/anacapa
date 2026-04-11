#pragma once

#include <anacapa/shading/IMaterial.h>
#include "Texture.h"
#include <cmath>

namespace anacapa {

// ---------------------------------------------------------------------------
// GGX (Trowbridge-Reitz) microfacet utilities
//
// alpha = roughness^2  (Disney perceptual remapping)
// ---------------------------------------------------------------------------

static constexpr float kSS_Pi    = 3.14159265358979323846f;
static constexpr float kSS_InvPi = 0.31830988618379067154f;

// GGX normal distribution D(wh): probability density of half-vector wh
// cosH = dot(wh, n) in local frame = wh.z
inline float D_GGX(float cosH, float alpha2) {
    if (cosH <= 0.f) return 0.f;
    float denom = cosH * cosH * (alpha2 - 1.f) + 1.f;
    return alpha2 / (kSS_Pi * denom * denom);
}

// Smith G1 (lambda form, exact for GGX):
//   G1(w) = 2 * |cos(theta)| / (|cos(theta)| + sqrt(alpha^2 + (1-alpha^2)*cos^2(theta)))
inline float G1_Smith(float absCos, float alpha2) {
    if (absCos <= 0.f) return 0.f;
    float cos2 = absCos * absCos;
    return 2.f * absCos / (absCos + std::sqrt(alpha2 + (1.f - alpha2) * cos2));
}

// Separable Smith G2 = G1(wo) * G1(wi)
inline float G2_Smith_Separable(float cosO, float cosI, float alpha2) {
    return G1_Smith(std::abs(cosO), alpha2) * G1_Smith(std::abs(cosI), alpha2);
}

// Sample a half-vector from the GGX NDF (spherical parameterization).
// Returns wh in local shading frame (+Z = normal).
inline Vec3f sampleGGX_halfvector(Vec2f u, float alpha2) {
    float phi       = 2.f * kSS_Pi * u.x;
    // Clamp to avoid divide-by-zero at u.y → 1
    float cosH2     = (1.f - u.y) / std::max(1.f + (alpha2 - 1.f) * u.y, 1e-10f);
    float cosH      = std::sqrt(std::max(0.f, cosH2));
    float sinH      = std::sqrt(std::max(0.f, 1.f - cosH2));
    return { sinH * std::cos(phi), sinH * std::sin(phi), cosH };
}

// PDF of sampling wi via GGX half-vector reflection:
//   p(wi) = D(wh) * G1(wo) * |dot(wo, wh)| / (4 * |cos(theta_o)|)
// Here we use the simpler form without G1 (standard NDF-based sampling).
//   p(wh) = D(wh) * cosH
//   p(wi) = p(wh) / (4 * |dot(wo, wh)|)
inline float pdfGGX_reflection(float cosH, float alpha2, float dotVH) {
    if (cosH <= 0.f || dotVH <= 0.f) return 0.f;
    return D_GGX(cosH, alpha2) * cosH / (4.f * dotVH);
}

// Schlick Fresnel for dielectric:  F0 = ((1-ior)/(1+ior))^2
inline float F0_fromIOR(float ior) {
    float r = (1.f - ior) / (1.f + ior);
    return r * r;
}

inline float schlickDielectric(float cosTheta, float f0) {
    float x  = 1.f - std::abs(cosTheta);
    float x5 = x * x * x * x * x;
    return f0 + (1.f - f0) * x5;
}

inline Spectrum schlickConductor(float cosTheta, Spectrum f0) {
    float x  = 1.f - std::abs(cosTheta);
    float x5 = x * x * x * x * x;
    return f0 + (Spectrum{1.f, 1.f, 1.f} - f0) * x5;
}

// ---------------------------------------------------------------------------
// StandardSurfaceMaterial
//
// Approximates the MaterialX standard_surface BSDF:
//   - Metallic GGX specular (conductor)
//   - Dielectric GGX specular (Fresnel-weighted)
//   - Lambertian diffuse (below the specular layer)
//   - Clearcoat GGX (thin dielectric layer on top, ior=1.5)
//   - Emission
//
// Sampling strategy (layered):
//   1. Evaluate coat Fresnel → sample coat GGX with probability coat*Fc
//   2. Evaluate specular Fresnel (conductor or dielectric)
//   3. Randomly choose: metallic GGX / dielectric GGX / diffuse
//      weighted by metalness, Fresnel, and remaining energy
// ---------------------------------------------------------------------------
class StandardSurfaceMaterial : public IMaterial {
public:
    struct Params {
        // Base layer
        SpectrumTOV base_color     = SpectrumTOV({0.8f, 0.8f, 0.8f});
        float        base          = 1.0f;       // overall base weight [0,1]
        FloatTOV     metalness     = FloatTOV(0.0f);

        // Specular layer
        FloatTOV     roughness     = FloatTOV(0.5f);
        FloatTOV     specular      = FloatTOV(1.0f);
        Spectrum     specular_color = {1.f,1.f,1.f}; // F0 tint for dielectric
        float        specular_IOR  = 1.5f;

        // Coat layer (thin dielectric on top)
        float        coat           = 0.0f;
        float        coat_roughness = 0.1f;

        // Normal map (tangent-space RGB texture, bias/scale from UsdUVTexture)
        SpectrumTOV  normal_map    = SpectrumTOV({0.5f, 0.5f, 1.0f});
        bool         has_normal_map = false;
        // UsdUVTexture bias/scale for normal: typically scale=(2,2,2,2) bias=(-1,-1,-1,-1)
        float        normal_scale  = 2.f;
        float        normal_bias   = -1.f;

        // Opacity
        FloatTOV     opacity       = FloatTOV(1.0f);

        // Emission
        float        emission       = 0.0f;
        Spectrum     emission_color = {0.f, 0.f, 0.f};
    };

    explicit StandardSurfaceMaterial(const Params& p) : m_p(p) {
        float r  = std::max(1e-4f, m_p.roughness.value);
        m_alpha  = r * r;
        m_alpha2 = m_alpha * m_alpha;
        m_coatAlpha  = std::max(1e-4f, m_p.coat_roughness * m_p.coat_roughness);
        m_coatAlpha2 = m_coatAlpha * m_coatAlpha;
        m_f0Dielectric = F0_fromIOR(m_p.specular_IOR);
        m_coatF0       = F0_fromIOR(1.5f);
    }

    bool isDelta() const override {
        float r = m_p.roughness.value;
        float m = m_p.metalness.value;
        float s = m_p.specular.value;
        return r < 0.001f && (m > 0.999f || s > 0.f);
    }

    float roughness() const override { return m_p.roughness.value; }
    float metalness() const override { return m_p.metalness.value; }

    uint32_t flags() const override {
        uint32_t f = BSDFFlag_Reflection;
        if (m_p.roughness.value < 0.001f)  f |= BSDFFlag_Specular;
        else if (m_p.roughness.value < 0.3f) f |= BSDFFlag_Glossy;
        else                          f |= BSDFFlag_Diffuse;
        return f;
    }

    Spectrum Le(const ShadingContext&, Vec3f) const override {
        if (m_p.emission > 0.f)
            return m_p.emission_color * m_p.emission;
        return {};
    }

    Spectrum reflectance(const ShadingContext& ctx) const override {
        Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
        float    metal      = evalTOV(m_p.metalness,  ctx.uv);
        float    spec       = evalTOV(m_p.specular,   ctx.uv);
        float    diff       = (1.f - metal) * (1.f - spec * 0.5f);
        return base_color * m_p.base * (diff + metal);
    }

    // -----------------------------------------------------------------------
    // sample
    // -----------------------------------------------------------------------
    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float uComponent) const override {
        ShadingContext sctx = applyNormalMap(ctx);
        Vec3f woLocal = sctx.toLocal(wo);
        if (woLocal.z <= 0.f) return {};   // backface

        Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
        float metal = evalTOV(m_p.metalness, ctx.uv);
        float spec  = evalTOV(m_p.specular,  ctx.uv);
        float rough = evalTOV(m_p.roughness, ctx.uv);
        float alpha  = std::max(1e-4f, rough * rough);
        float alpha2 = alpha * alpha;

        // --- Compute Fresnel at wo for layer selection ---
        float coatF = schlickDielectric(woLocal.z, m_coatF0);
        float specF = schlickDielectric(woLocal.z, m_f0Dielectric);

        // Layer selection weights (all in [0,1], then renormalize)
        float wCoat  = m_p.coat * coatF;
        float wMetal = metal * (1.f - wCoat);
        float wSpec  = spec * (1.f - metal) * specF * (1.f - wCoat);
        float wDiff  = m_p.base * (1.f - metal)
                     * (1.f - spec * specF) * (1.f - wCoat);

        float wSum = wCoat + wMetal + wSpec + wDiff;
        if (wSum <= 0.f) return {};
        float invW = 1.f / wSum;
        wCoat  *= invW;
        wMetal *= invW;
        wSpec  *= invW;
        wDiff  *= invW;

        // Choose component
        BSDFSample result;
        if (uComponent < wCoat) {
            result = sampleGGX(sctx, woLocal, u, m_coatAlpha, m_coatAlpha2,
                               Spectrum{m_coatF0, m_coatF0, m_coatF0}, false);
        } else if (uComponent < wCoat + wMetal) {
            // Conductor: F0 = base_color
            result = sampleGGX(sctx, woLocal, u, alpha, alpha2,
                               base_color * m_p.base, false);
        } else if (uComponent < wCoat + wMetal + wSpec) {
            // Dielectric specular: F0 = specular * specular_color * f0_from_IOR
            Spectrum f0 = m_p.specular_color * (spec * m_f0Dielectric);
            result = sampleGGX(sctx, woLocal, u, alpha, alpha2, f0, false);
        } else {
            // Diffuse
            result = sampleDiffuse(sctx, woLocal, u, base_color);
        }

        if (!result.isValid()) return {};

        // Compute combined PDF and f across all layers (for firefly-free evaluation)
        float pdfFwd = 0.f, pdfRev = 0.f;
        Spectrum fCombined = evalCombined(ctx, woLocal, sctx.toLocal(result.wi),
                                          wCoat, wMetal, wSpec, wDiff,
                                          base_color, spec, alpha2,
                                          pdfFwd, pdfRev);

        result.f      = fCombined * std::abs(sctx.toLocal(result.wi).z);
        result.pdf    = pdfFwd;
        result.pdfRev = pdfRev;
        result.flags  = flags();
        return result;
    }

    // -----------------------------------------------------------------------
    // evaluate
    // -----------------------------------------------------------------------
    BSDFEval evaluate(const ShadingContext& ctx,
                       Vec3f wo, Vec3f wi) const override {
        Vec3f woLocal = ctx.toLocal(wo);
        Vec3f wiLocal = ctx.toLocal(wi);
        if (!sameHemisphere(woLocal, wiLocal)) return {};

        Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
        float metal = evalTOV(m_p.metalness, ctx.uv);
        float spec  = evalTOV(m_p.specular,  ctx.uv);
        float rough = evalTOV(m_p.roughness, ctx.uv);
        float alpha  = std::max(1e-4f, rough * rough);
        float alpha2 = alpha * alpha;

        float coatF = schlickDielectric(woLocal.z, m_coatF0);
        float specF = schlickDielectric(woLocal.z, m_f0Dielectric);

        float wCoat  = m_p.coat * coatF;
        float wMetal = metal * (1.f - wCoat);
        float wSpec  = spec * (1.f - metal) * specF * (1.f - wCoat);
        float wDiff  = m_p.base * (1.f - metal)
                     * (1.f - spec * specF) * (1.f - wCoat);

        float wSum = wCoat + wMetal + wSpec + wDiff;
        if (wSum <= 0.f) return {};
        float invW = 1.f / wSum;
        wCoat  *= invW; wMetal *= invW; wSpec *= invW; wDiff *= invW;

        float pdfFwd, pdfRev;
        Spectrum f = evalCombined(ctx, woLocal, wiLocal,
                                  wCoat, wMetal, wSpec, wDiff,
                                  base_color, spec, alpha2,
                                  pdfFwd, pdfRev);

        BSDFEval e;
        e.f      = f;
        e.pdf    = pdfFwd;
        e.pdfRev = pdfRev;
        return e;
    }

    float pdf(const ShadingContext& ctx,
              Vec3f wo, Vec3f wi) const override {
        BSDFEval e = evaluate(ctx, wo, wi);
        return e.pdf;
    }

private:
    // -----------------------------------------------------------------------
    // sampleGGX — sample a GGX reflection lobe
    // -----------------------------------------------------------------------
    BSDFSample sampleGGX(const ShadingContext& ctx,
                          Vec3f woLocal, Vec2f u,
                          float alpha, float alpha2,
                          Spectrum f0, bool /*transmission*/) const {
        Vec3f wh = sampleGGX_halfvector(u, alpha2);
        if (wh.z <= 0.f) return {};

        // Reflect wo about wh to get wi
        float dotVH = dot(woLocal, wh);
        if (dotVH <= 0.f) return {};
        Vec3f wiLocal = wh * (2.f * dotVH) - woLocal;
        if (wiLocal.z <= 0.f) return {};

        Spectrum F  = schlickConductor(dotVH, f0);
        float D     = D_GGX(wh.z, alpha2);
        float G2    = G2_Smith_Separable(woLocal.z, wiLocal.z, alpha2);

        // BSDF: f = D*G2*F / (4 * cosO * cosI),  result.f includes cosI
        float cosO = std::abs(woLocal.z);
        float cosI = std::abs(wiLocal.z);
        if (cosO < 1e-6f || cosI < 1e-6f) return {};

        Spectrum bsdf = F * (D * G2 / (4.f * cosO));

        float pdfWi = pdfGGX_reflection(wh.z, alpha2, dotVH);

        BSDFSample s;
        s.wi     = ctx.toWorld(wiLocal);
        s.f      = bsdf * cosI;    // includes cosI (as per interface contract)
        s.pdf    = pdfWi;
        s.pdfRev = pdfWi;          // reflection: pdfRev == pdfFwd for GGX
        s.flags  = BSDFFlag_Specular | BSDFFlag_Reflection;
        return s;
    }

    // -----------------------------------------------------------------------
    // sampleDiffuse — cosine-weighted Lambertian
    // -----------------------------------------------------------------------
    BSDFSample sampleDiffuse(const ShadingContext& ctx,
                              Vec3f woLocal, Vec2f u,
                              Spectrum base_color) const {
        float phi = 2.f * kSS_Pi * u.x;
        float r   = std::sqrt(u.y);
        float z   = std::sqrt(std::max(0.f, 1.f - u.y));
        Vec3f wiLocal = { r * std::cos(phi), r * std::sin(phi), z };
        if (woLocal.z < 0.f) wiLocal.z = -wiLocal.z;

        float cosI = std::abs(wiLocal.z);
        float p    = cosI * kSS_InvPi;

        BSDFSample s;
        s.wi     = ctx.toWorld(wiLocal);
        s.f      = base_color * m_p.base * (kSS_InvPi * cosI);
        s.pdf    = p;
        s.pdfRev = std::abs(woLocal.z) * kSS_InvPi;
        s.flags  = BSDFFlag_Diffuse | BSDFFlag_Reflection;
        return s;
    }

    // -----------------------------------------------------------------------
    // evalGGX — evaluate GGX lobe at (woLocal, wiLocal)
    // -----------------------------------------------------------------------
    Spectrum evalGGX(Vec3f woLocal, Vec3f wiLocal,
                     float alpha2, Spectrum f0,
                     float& pdfFwd, float& pdfRev) const {
        pdfFwd = pdfRev = 0.f;
        if (woLocal.z <= 0.f || wiLocal.z <= 0.f) return {};

        Vec3f wh = safeNormalize(woLocal + wiLocal);
        if (wh.z <= 0.f) return {};

        float dotVH = std::max(0.f, dot(woLocal, wh));
        float dotLH = std::max(0.f, dot(wiLocal, wh));
        Spectrum F  = schlickConductor(dotVH, f0);
        float D     = D_GGX(wh.z, alpha2);
        float G2    = G2_Smith_Separable(woLocal.z, wiLocal.z, alpha2);

        float cosO = std::abs(woLocal.z);
        float cosI = std::abs(wiLocal.z);
        if (cosO < 1e-6f || cosI < 1e-6f) return {};

        // BSDF value (no cosine factor)
        Spectrum bsdf = F * (D * G2 / (4.f * cosO * cosI));

        pdfFwd = pdfGGX_reflection(wh.z, alpha2, dotVH);
        pdfRev = pdfGGX_reflection(wh.z, alpha2, dotLH);
        return bsdf;
    }

    // -----------------------------------------------------------------------
    // evalCombined — weighted sum of all BSDF layers
    // -----------------------------------------------------------------------
    Spectrum evalCombined(const ShadingContext& /*ctx*/,
                           Vec3f woLocal, Vec3f wiLocal,
                           float wCoat, float wMetal, float wSpec, float wDiff,
                           Spectrum base_color, float spec, float alpha2,
                           float& pdfFwd, float& pdfRev) const {
        Spectrum f = {};
        pdfFwd = pdfRev = 0.f;

        if (!sameHemisphere(woLocal, wiLocal)) return {};

        // Coat layer
        if (wCoat > 0.f) {
            float pF, pR;
            Spectrum fC = evalGGX(woLocal, wiLocal, m_coatAlpha2,
                                   Spectrum{m_coatF0, m_coatF0, m_coatF0}, pF, pR);
            f += fC * wCoat;
            pdfFwd += pF * wCoat;
            pdfRev += pR * wCoat;
        }

        // Metallic layer
        if (wMetal > 0.f) {
            float pF, pR;
            Spectrum fM = evalGGX(woLocal, wiLocal, alpha2,
                                   base_color * m_p.base, pF, pR);
            f += fM * wMetal;
            pdfFwd += pF * wMetal;
            pdfRev += pR * wMetal;
        }

        // Dielectric specular layer
        if (wSpec > 0.f) {
            float pF, pR;
            Spectrum f0 = m_p.specular_color * (spec * m_f0Dielectric);
            Spectrum fS = evalGGX(woLocal, wiLocal, alpha2, f0, pF, pR);
            f += fS * wSpec;
            pdfFwd += pF * wSpec;
            pdfRev += pR * wSpec;
        }

        // Diffuse layer
        if (wDiff > 0.f) {
            float cosI  = std::abs(wiLocal.z);
            float cosO  = std::abs(woLocal.z);
            Spectrum fD = base_color * m_p.base * kSS_InvPi;
            float pF    = cosI * kSS_InvPi;
            float pR    = cosO * kSS_InvPi;
            f += fD * wDiff;
            pdfFwd += pF * wDiff;
            pdfRev += pR * wDiff;
        }

        return f;
    }

    // -----------------------------------------------------------------------
    // applyNormalMap — perturb shading normal using tangent-space normal map
    // Returns a modified ShadingContext with the new normal and rebuilt basis.
    // -----------------------------------------------------------------------
    ShadingContext applyNormalMap(const ShadingContext& ctx) const {
        if (!m_p.has_normal_map || !m_p.normal_map.hasTexture()) return ctx;

        Spectrum s = evalTOV(m_p.normal_map, ctx.uv);
        // Decode tangent-space normal: typically scale=2, bias=-1
        Vec3f nts = {
            s.x * m_p.normal_scale + m_p.normal_bias,
            s.y * m_p.normal_scale + m_p.normal_bias,
            s.z * m_p.normal_scale + m_p.normal_bias
        };
        // Transform from tangent space to world space using TBN
        Vec3f nWorld = safeNormalize(ctx.t * nts.x + ctx.bt * nts.y + ctx.n * nts.z);

        ShadingContext result = ctx;
        result.n = nWorld;
        buildOrthonormalBasis(result.n, result.t, result.bt);
        return result;
    }

    Params   m_p;
    float    m_alpha, m_alpha2;
    float    m_coatAlpha, m_coatAlpha2;
    float    m_f0Dielectric;
    float    m_coatF0;
};

} // namespace anacapa
