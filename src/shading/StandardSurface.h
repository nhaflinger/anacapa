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
// Exact Fresnel equations for a dielectric interface.
// cosI: cosine of angle of incidence from the incident-medium side (> 0).
// eta:  n_transmitted / n_incident.
// Returns reflectance in [0, 1].  1.0 = total internal reflection (TIR).
// ---------------------------------------------------------------------------
inline float fresnelDielectric(float cosI, float eta) {
    float sin2I = std::max(0.f, 1.f - cosI * cosI);
    float sin2T = sin2I / (eta * eta);
    if (sin2T >= 1.f) return 1.f;                       // TIR
    float cosT = std::sqrt(1.f - sin2T);
    float Rs = (cosI - eta * cosT) / (cosI + eta * cosT);
    float Rp = (eta * cosI - cosT) / (eta * cosI + cosT);
    return 0.5f * (Rs * Rs + Rp * Rp);
}

// ---------------------------------------------------------------------------
// StandardSurfaceMaterial
//
// Approximates the MaterialX standard_surface BSDF:
//   - Dielectric glass (transmission > 0, roughness < 0.001): exact Fresnel +
//     Snell's law refraction — perfect specular reflect or transmit
//   - Rough dielectric glass (transmission > 0, roughness >= 0.001): GGX
//     reflection + microfacet transmission lobe (Walter et al. 2007)
//   - Metallic GGX specular (conductor)
//   - Dielectric GGX specular (Fresnel-weighted)
//   - Lambertian diffuse (below the specular layer)
//   - Clearcoat GGX (thin dielectric layer on top, ior=1.5)
//   - Emission
//
// Sampling strategy (layered):
//   1. If transmission > 0 and metalness ≈ 0 → dielectric glass path
//   2. Evaluate coat Fresnel → sample coat GGX with probability coat*Fc
//   3. Evaluate specular Fresnel (conductor or dielectric)
//   4. Randomly choose: metallic GGX / dielectric GGX / diffuse
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

        // Opacity / transmission
        FloatTOV     opacity       = FloatTOV(1.0f);
        float        transmission  = 0.0f;   // 0 = opaque, 1 = fully transmissive (glass)
        bool         alphaMask     = false;  // true = opacity driven by texture alpha channel

        // Emission
        float        emission       = 0.0f;
        SpectrumTOV  emission_color = SpectrumTOV({0.f, 0.f, 0.f});
    };

    const Params& params() const { return m_p; }

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
        // Smooth dielectric glass
        if (r < 0.001f && m_p.transmission > 0.001f && m < 0.001f) return true;
        return r < 0.001f && (m > 0.999f || s > 0.f);
    }

    float roughness() const override { return m_p.roughness.value; }
    float metalness() const override { return m_p.metalness.value; }

    uint32_t flags() const override {
        uint32_t f = BSDFFlag_Reflection;
        if (m_p.transmission > 0.001f) f |= BSDFFlag_Transmission;
        if (m_p.roughness.value < 0.001f)  f |= BSDFFlag_Specular;
        else if (m_p.roughness.value < 0.3f) f |= BSDFFlag_Glossy;
        else                          f |= BSDFFlag_Diffuse;
        return f;
    }

    Spectrum Le(const ShadingContext& ctx, Vec3f) const override {
        if (m_p.emission > 0.f)
            return evalTOV(m_p.emission_color, ctx.uv) * m_p.emission;
        return {};
    }

    Spectrum reflectance(const ShadingContext& ctx) const override {
        Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
        float    metal      = evalTOV(m_p.metalness,  ctx.uv);
        float    spec       = evalTOV(m_p.specular,   ctx.uv);
        float    diff       = (1.f - metal) * (1.f - spec * 0.5f);
        return base_color * m_p.base * (diff + metal);
    }

    float evalOpacity(const ShadingContext& ctx) const override {
        if (m_p.alphaMask)
            return evalTOV(m_p.opacity, ctx.uv);
        return 1.f;
    }

    // Shadow-ray transmittance tint.  For glass, this is the base color
    // weighted by the transmission amount.  Opaque materials return black.
    // We use the average Fresnel reflectance at normal incidence as a rough
    // correction so polished glass doesn't pass 100% of shadow light.
    Spectrum transmittanceColor(const ShadingContext& ctx) const override {
        // Alpha-masked surfaces: opacity texture tells us how much light passes.
        // opacity≈1 (opaque region) → blocks light; opacity≈0 (cutout) → passes light.
        if (m_p.alphaMask) {
            float alpha = evalTOV(m_p.opacity, ctx.uv);
            if (alpha > 0.5f) return {};                    // opaque region blocks light
            return {1.f, 1.f, 1.f};                        // cutout region: pass light through
        }
        if (m_p.transmission < 0.001f) return {};
        float metal = evalTOV(m_p.metalness, ctx.uv);
        if (metal > 0.001f) return {};
        Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
        // Fresnel reflectance at normal incidence (cosTheta = 1)
        float Fr = fresnelDielectric(1.f, m_p.specular_IOR);
        return base_color * (m_p.transmission * (1.f - Fr));
    }

    // -----------------------------------------------------------------------
    // sample
    // -----------------------------------------------------------------------
    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float uComponent) const override {
        ShadingContext sctx = applyNormalMap(ctx);
        Vec3f woLocal = sctx.toLocal(wo);
        if (woLocal.z <= 0.f) return {};   // backface — n is already flipped by ShadingContext

        Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
        float metal = evalTOV(m_p.metalness, ctx.uv);
        float spec  = evalTOV(m_p.specular,  ctx.uv);
        float rough = evalTOV(m_p.roughness, ctx.uv);

        // -------------------------------------------------------------------
        // Dielectric glass — smooth (delta) or rough (microfacet transmission)
        //
        // ShadingContext always flips n to face the incoming side, so:
        //   ctx.frontFace == true  → entering the glass  (eta = IOR / 1)
        //   ctx.frontFace == false → exiting  the glass  (eta = 1 / IOR)
        // woLocal.z > 0 always holds regardless of front/back face.
        // -------------------------------------------------------------------
        if (m_p.transmission > 0.001f && metal < 0.001f) {
            float eta = ctx.frontFace ? m_p.specular_IOR
                                      : (1.f / m_p.specular_IOR);
            float cosI = woLocal.z;  // angle of incidence, always > 0

            if (rough < 0.001f) {
                // ----- Perfect smooth glass (delta BSDF) -------------------
                float Fr = fresnelDielectric(cosI, eta);

                BSDFSample s;
                if (uComponent < Fr) {
                    // Specular reflection — mirror about shading normal
                    Vec3f wiLocal = {-woLocal.x, -woLocal.y, woLocal.z};
                    s.wi     = sctx.toWorld(wiLocal);
                    // f = BSDF * |cosI|  but BSDF = Fr/|cosI| for a delta →  f = Fr * tint
                    s.f      = base_color * Fr;
                    s.pdf    = Fr;
                    s.pdfRev = Fr;   // Fresnel is the same from both sides
                    s.eta    = 1.f;
                    s.flags  = BSDFFlag_Specular | BSDFFlag_Reflection;
                } else {
                    // Specular refraction (Snell's law in the local shading frame)
                    float sin2T = std::max(0.f, 1.f - cosI * cosI) / (eta * eta);
                    float cosT  = std::sqrt(std::max(0.f, 1.f - sin2T));
                    // Refracted direction: tangential component scaled by 1/eta,
                    // normal component becomes -cosT (goes through surface).
                    Vec3f wiLocal = {-woLocal.x / eta, -woLocal.y / eta, -cosT};
                    s.wi     = sctx.toWorld(wiLocal);
                    s.f      = base_color * (1.f - Fr);
                    s.pdf    = 1.f - Fr;
                    s.pdfRev = 1.f - Fr;
                    s.eta    = eta;
                    s.flags  = BSDFFlag_Specular | BSDFFlag_Transmission;
                }
                return s;

            } else {
                // ----- Rough dielectric glass (microfacet BSDF) ------------
                // Sample a GGX half-vector, then Fresnel-choose reflect vs refract.
                float alpha  = std::max(1e-4f, rough * rough);
                float alpha2 = alpha * alpha;
                Vec3f wh = sampleGGX_halfvector(u, alpha2);
                if (wh.z <= 0.f) return {};
                float cosIH = std::max(0.f, dot(woLocal, wh));
                if (cosIH <= 0.f) return {};

                float Fr = fresnelDielectric(cosIH, eta);

                BSDFSample s;
                if (uComponent < Fr) {
                    // GGX reflection
                    Vec3f wiLocal = wh * (2.f * cosIH) - woLocal;
                    if (wiLocal.z <= 0.f) return {};
                    float D  = D_GGX(wh.z, alpha2);
                    float G2 = G2_Smith_Separable(woLocal.z, wiLocal.z, alpha2);
                    float cosO = woLocal.z, cosR = wiLocal.z;
                    Spectrum f0{Fr, Fr, Fr};
                    Spectrum bsdf = f0 * (D * G2 / (4.f * cosO));
                    s.wi     = sctx.toWorld(wiLocal);
                    s.f      = bsdf * cosR * base_color;
                    s.pdf    = Fr * pdfGGX_reflection(wh.z, alpha2, cosIH);
                    s.pdfRev = s.pdf;
                    s.eta    = 1.f;
                    s.flags  = BSDFFlag_Glossy | BSDFFlag_Reflection;
                } else {
                    // GGX refraction (microfacet transmission, Walter et al. 2007)
                    float sin2T = std::max(0.f, 1.f - cosIH * cosIH) / (eta * eta);
                    if (sin2T >= 1.f) return {};  // TIR
                    float cosT_H = std::sqrt(1.f - sin2T);
                    // Refracted direction about wh (half-vector frame)
                    Vec3f wiLocal = woLocal * (-1.f / eta)
                                 + wh * (cosIH / eta - cosT_H);
                    if (wiLocal.z >= 0.f) return {};  // must go through surface
                    float cosO   = woLocal.z;
                    float cosI_t = std::abs(wiLocal.z);
                    float D      = D_GGX(wh.z, alpha2);
                    float G2     = G2_Smith_Separable(cosO, cosI_t, alpha2);
                    // Microfacet transmission BSDF (importance transport, no eta^2)
                    float denom  = cosIH + eta * std::abs(dot(wiLocal, wh));
                    if (denom < 1e-6f) return {};
                    float bsdf_scalar = (1.f - Fr) * D * G2
                                      * cosIH / (cosO * denom);
                    // PDF via half-vector Jacobian for refraction
                    float absCosT_wh = std::abs(dot(wiLocal, wh));
                    float pdf_wh     = D * wh.z;
                    float jacobian   = eta * eta * absCosT_wh / (denom * denom);
                    s.wi     = sctx.toWorld(wiLocal);
                    s.f      = base_color * bsdf_scalar * cosI_t;
                    s.pdf    = (1.f - Fr) * pdf_wh * jacobian;
                    s.pdfRev = s.pdf;
                    s.eta    = eta;
                    s.flags  = BSDFFlag_Glossy | BSDFFlag_Transmission;
                }
                return s;
            }
        }
        // -------------------------------------------------------------------
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

        // Rough dielectric transmission: wi is in the opposite hemisphere.
        if (!sameHemisphere(woLocal, wiLocal)) {
            if (m_p.transmission < 0.001f) return {};
            float rough = evalTOV(m_p.roughness, ctx.uv);
            if (rough < 0.001f) return {};  // smooth glass is delta, no area PDF
            float metal = evalTOV(m_p.metalness, ctx.uv);
            if (metal > 0.001f) return {};

            float eta = ctx.frontFace ? m_p.specular_IOR : (1.f / m_p.specular_IOR);
            float alpha2 = rough * rough; alpha2 *= alpha2;
            // Half-vector for refraction (Walter et al. 2007, eq. 16)
            Vec3f wh = safeNormalize(woLocal + wiLocal * eta);
            if (wh.z < 0.f) wh = -wh;  // ensure same hemisphere as normal
            float cosIH = std::max(0.f, dot(woLocal, wh));
            float cosTH = std::abs(dot(wiLocal, wh));
            float Fr    = fresnelDielectric(cosIH, eta);
            float D     = D_GGX(wh.z, alpha2);
            float G2    = G2_Smith_Separable(woLocal.z, std::abs(wiLocal.z), alpha2);
            float denom = cosIH + eta * cosTH;
            if (denom < 1e-6f) return {};
            float bsdf_scalar = (1.f - Fr) * D * G2 * cosIH
                              / (woLocal.z * denom);
            float jacobian = eta * eta * cosTH / (denom * denom);
            Spectrum base_color = evalTOV(m_p.base_color, ctx.uv);
            BSDFEval e;
            e.f      = base_color * bsdf_scalar;
            e.pdf    = (1.f - Fr) * D * wh.z * jacobian;
            e.pdfRev = e.pdf;
            return e;
        }

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
