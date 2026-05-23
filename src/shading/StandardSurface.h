#pragma once

#include <anacapa/shading/IMaterial.h>
#include "MarschnerHair.h"
#include "Texture.h"
#include <algorithm>
#include <cmath>
#include <cstdio>

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
// Specular directional-hemispherical albedo for the dielectric layer
//
// E_spec(cos_theta_o, roughness) = ∫ f_GGX(wo, wi) * cos(theta_i) dwi
//
// Used in the diffuse weight `(1 - spec * E_spec)` so the energy that the
// rough specular lobe DOESN'T reflect (because GGX*F integrates to less than
// the Fresnel point evaluation, especially at high roughness) returns to
// the diffuse layer instead of vanishing.  Replaces the previous point
// Fresnel `specF`, which over-counted spec reflectance and dimmed indirect
// diffuse — the "spec/diff balance" issue in the indirect-brightness memo.
//
// LUT is 32 × 32 (cos_theta_o × roughness), MC-integrated for F0 = 0.04
// (IOR ≈ 1.5).  Materials with significantly different IORs incur a small
// approximation error, smaller than the energy loss this fix prevents.
// ---------------------------------------------------------------------------
namespace detail {
struct SpecAlbedoLUT {
    static constexpr int N_COS = 32;
    static constexpr int N_R   = 32;
    float v    [N_COS * N_R];   // E_spec(cos_theta_o, roughness)
    float eAvg [N_R];           // E_avg(roughness) = ∫ E_spec(μ, α) * 2μ dμ

    SpecAlbedoLUT() {
        for (int ic = 0; ic < N_COS; ++ic) {
            float cosO = (ic + 0.5f) / float(N_COS);
            float sinO = std::sqrt(std::max(0.f, 1.f - cosO * cosO));
            Vec3f wo{sinO, 0.f, cosO};
            for (int ir = 0; ir < N_R; ++ir) {
                float r       = (ir + 0.5f) / float(N_R);
                float alpha   = std::max(1e-4f, r * r);
                float alpha2  = alpha * alpha;
                v[ic * N_R + ir] = computeCell(wo, alpha2);
            }
        }
        // Cosine-weighted directional average → "avg albedo" used as the
        // Kulla-Conty multi-scatter denominator (1 - E_avg).
        for (int ir = 0; ir < N_R; ++ir) {
            float sum = 0.f;
            for (int ic = 0; ic < N_COS; ++ic) {
                float cosO = (ic + 0.5f) / float(N_COS);
                sum += v[ic * N_R + ir] * 2.f * cosO;
            }
            eAvg[ir] = sum / float(N_COS);
        }
    }

    static float halton(int i, int b) {
        float f = 1.f, r = 0.f;
        while (i > 0) { f /= float(b); r += f * float(i % b); i /= b; }
        return r;
    }

    static float computeCell(Vec3f wo, float alpha2) {
        constexpr int N  = 4096;
        constexpr float F0 = 0.04f;       // dielectric, IOR ~ 1.5
        float E = 0.f;
        for (int s = 0; s < N; ++s) {
            float u1 = halton(s + 1, 2);
            float u2 = halton(s + 1, 3);
            // GGX half-vector NDF sample (Trowbridge-Reitz)
            float phi   = 2.f * kSS_Pi * u1;
            float cosH2 = (1.f - u2) / ((alpha2 - 1.f) * u2 + 1.f);
            float cosH  = std::sqrt(std::max(0.f, cosH2));
            float sinH  = std::sqrt(std::max(0.f, 1.f - cosH2));
            Vec3f wh{sinH * std::cos(phi), sinH * std::sin(phi), cosH};
            // Reflect wo about wh
            float dotVH = wo.x * wh.x + wo.y * wh.y + wo.z * wh.z;
            if (dotVH <= 0.f || wh.z <= 0.f) continue;
            Vec3f wi{2.f * dotVH * wh.x - wo.x,
                     2.f * dotVH * wh.y - wo.y,
                     2.f * dotVH * wh.z - wo.z};
            if (wi.z <= 0.f) continue;
            // Smith G2 separable
            auto G1 = [&](float c) {
                return 2.f * c / (c + std::sqrt(alpha2 + (1.f - alpha2) * c * c));
            };
            float G2 = G1(wo.z) * G1(wi.z);
            // Schlick Fresnel
            float Fr = F0 + (1.f - F0) * std::pow(1.f - dotVH, 5.f);
            // Estimator: G2 * F * |wo·wh| / (cos_theta_o * cos_theta_h)
            E += G2 * Fr * dotVH / (wo.z * wh.z);
        }
        return E / float(N);
    }

    float lookup(float cosO, float roughness) const {
        cosO      = std::min(1.f, std::max(0.f, cosO));
        roughness = std::min(1.f, std::max(0.f, roughness));
        float fc = cosO      * float(N_COS - 1);
        float fr = roughness * float(N_R   - 1);
        int   ic0 = std::min(N_COS - 1, int(fc));
        int   ir0 = std::min(N_R   - 1, int(fr));
        int   ic1 = std::min(N_COS - 1, ic0 + 1);
        int   ir1 = std::min(N_R   - 1, ir0 + 1);
        float tc  = fc - float(ic0);
        float tr  = fr - float(ir0);
        float v00 = v[ic0 * N_R + ir0];
        float v01 = v[ic0 * N_R + ir1];
        float v10 = v[ic1 * N_R + ir0];
        float v11 = v[ic1 * N_R + ir1];
        return (v00 * (1.f - tc) + v10 * tc) * (1.f - tr)
             + (v01 * (1.f - tc) + v11 * tc) * tr;
    }
};
}  // namespace detail

// Directional-hemispherical reflectance of the dielectric GGX spec lobe at
// (cos_theta_o, roughness).  Thread-safe lazy init via C++11 magic statics.
inline float specAlbedoDielectric(float cosO, float roughness) {
    static const detail::SpecAlbedoLUT lut;
    return lut.lookup(cosO, roughness);
}

// Cosine-weighted hemispherical average of E_spec — only depends on
// roughness.  Used as the Kulla-Conty multi-scatter compensation
// denominator: missing energy = (1 - E_avg).
inline float specAvgAlbedoDielectric(float roughness) {
    static const detail::SpecAlbedoLUT lut;
    roughness = std::min(1.f, std::max(0.f, roughness));
    float fr  = roughness * float(detail::SpecAlbedoLUT::N_R - 1);
    int   i0  = std::min(detail::SpecAlbedoLUT::N_R - 1, int(fr));
    int   i1  = std::min(detail::SpecAlbedoLUT::N_R - 1, i0 + 1);
    float t   = fr - float(i0);
    return lut.eAvg[i0] * (1.f - t) + lut.eAvg[i1] * t;
}

// LUT data accessors used by GPU backends to upload the same E_spec /
// E_avg tables to device memory.  v[] is 32*32 = 1024 floats indexed as
// [cos_idx * N_R + roughness_idx]; eAvg[] is 32 floats indexed by
// roughness only.  Both are computed once at program startup.
inline const float* specAlbedoLUTData() {
    static const detail::SpecAlbedoLUT lut;
    return lut.v;
}
inline const float* specAvgAlbedoLUTData() {
    static const detail::SpecAlbedoLUT lut;
    return lut.eAvg;
}
inline int specAlbedoLUTCosBins()       { return detail::SpecAlbedoLUT::N_COS; }
inline int specAlbedoLUTRoughnessBins() { return detail::SpecAlbedoLUT::N_R;   }

// Kulla-Conty multi-scatter compensation BRDF.  Adds the energy that
// single-scatter GGX loses to inter-microfacet masking back into the spec
// lobe.  The closer the surface gets to fully rough, the larger this term
// — for a chrome ball at roughness=1, single-scatter delivers ~50% of
// incoming; the rest comes through this term.  For dielectric F0=0.04
// the absolute magnitude is small; for metals (F0 ≈ baseColor) it's
// substantial and is the main reason rough metals look dim without it.
//
// f_ms(wo, wi) = F_ms * (1 - E(wo)) * (1 - E(wi)) / (π * (1 - E_avg))
//
// where F_ms ≈ F0 (the layer's incident reflectance — `base_color * spec`
// for metals, `m_f0Dielectric * spec` for dielectric spec).
inline Spectrum evalGGX_ms(Vec3f woLocal, Vec3f wiLocal,
                           float roughness, Spectrum F_ms)
{
    float cosO = woLocal.z;
    float cosI = wiLocal.z;
    if (cosO <= 0.f || cosI <= 0.f) return {};
    float Eo = specAlbedoDielectric(cosO, roughness);
    float Ei = specAlbedoDielectric(cosI, roughness);
    float Ea = specAvgAlbedoDielectric(roughness);
    if (Ea >= 0.999f) return {};   // saturated — no missing energy
    float k = (1.f - Eo) * (1.f - Ei) / (kSS_Pi * (1.f - Ea));
    return F_ms * k;
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
        bool         caustic        = false;  // photon map opt-in (see IMaterial::isCausticGenerator)
        float        caustic_radius = 0.f;   // 0 = use global scene default

        // Subsurface scattering layer (photon map path only)
        float        subsurface            = 0.0f;   // weight [0,1]
        SpectrumTOV  subsurface_color      = SpectrumTOV({1.f, 1.f, 1.f});
        float        subsurface_radius     = 0.1f;   // mean free path in world units
        float        subsurface_scale      = 1.0f;   // world-space multiplier; effective d = radius * scale
        float        subsurface_anisotropy = 0.0f;   // Henyey-Greenstein g [-1,1]
        float        subsurface_strength   = 1.0f;   // independent SSS radiance amplifier (> 1 boosts effect)

        // Diffuse translucency (Blender Translucent BSDF / ND_translucent_bsdf)
        float        translucency       = 0.0f;
        SpectrumTOV  translucency_color = SpectrumTOV({0.8f, 0.8f, 0.8f});
        // scatter: lobe width around straight-through direction.
        // 1.0 = full Lambertian (max blur/attenuation), 0.0 = delta straight-through (sharp/clear).
        float        scatter            = 1.0f;

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

        // Build Marschner hair fallback for when ctx.isCurve == true.
        // Used when this material is assigned to Alembic hair strands (via the
        // USD materialPath sidecar) and the USD material resolved as OpenPBR
        // rather than OslMaterial (e.g. no .oso file for unbound hair materials).
        {
            MarschnerHairMaterial::Params hp;
            const Spectrum& bc = m_p.base_color.value;
            float bcMax = std::max({bc.x, bc.y, bc.z});
            float bcMin = std::min({bc.x, bc.y, bc.z});
            if (bcMax - bcMin > 0.05f) {
                hp.sigma_a = {
                    -std::log(std::max(bc.x, 0.001f)),
                    -std::log(std::max(bc.y, 0.001f)),
                    -std::log(std::max(bc.z, 0.001f))
                };
            }
            hp.beta_m = std::clamp(m_p.roughness.value, 0.20f, 0.80f);
            hp.beta_n = std::clamp(m_p.roughness.value * 1.5f, 0.30f, 0.90f);
            if (m_p.specular_IOR > 1.f) hp.eta = m_p.specular_IOR;
            m_hairFallback = MarschnerHairMaterial(hp);
            fprintf(stderr,
                "[StandardSurface hair] baseColor=(%.3f,%.3f,%.3f) "
                "chroma=%.3f sigma_a=(%.3f,%.3f,%.3f)\n",
                bc.x, bc.y, bc.z, bcMax - bcMin,
                hp.sigma_a.x, hp.sigma_a.y, hp.sigma_a.z);
        }
    }

    bool isDelta() const override {
        float r = m_p.roughness.value;
        float m = m_p.metalness.value;
        float s = m_p.specular.value;
        // Smooth dielectric glass
        if (r < 0.001f && m_p.transmission > 0.001f && m < 0.001f) return true;
        return r < 0.001f && (m > 0.999f || s > 0.f);
    }

    bool  isCausticGenerator() const override { return m_p.caustic; }
    float causticRadius()       const override { return m_p.caustic_radius; }

    SubsurfaceParams subsurfaceParams() const override {
        return { m_p.subsurface,
                 evalTOV(m_p.subsurface_color, {}),
                 m_p.subsurface_radius,
                 m_p.subsurface_scale,
                 m_p.subsurface_anisotropy,
                 m_p.subsurface_strength };
    }

    float roughness() const override { return m_p.roughness.value; }
    float metalness() const override { return m_p.metalness.value; }

    uint32_t flags() const override {
        uint32_t f = BSDFFlag_Reflection;
        if (m_p.transmission > 0.001f || m_p.translucency > 0.001f)
            f |= BSDFFlag_Transmission;
        if (m_p.roughness.value < 0.001f)  f |= BSDFFlag_Specular;
        else if (m_p.roughness.value < 0.3f) f |= BSDFFlag_Glossy;
        else                          f |= BSDFFlag_Diffuse;
        return f;
    }

    Spectrum Le(const ShadingContext& ctx, Vec3f wo) const override {
        if (ctx.isCurve) return m_hairFallback.Le(ctx, wo);
        if (m_p.emission > 0.f)
            return evalTOV(m_p.emission_color, ctx.uv) * m_p.emission;
        return {};
    }

    Spectrum reflectance(const ShadingContext& ctx) const override {
        if (ctx.isCurve) return m_hairFallback.reflectance(ctx);
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
        // Alpha-masked surfaces: use opacity as a deterministic tint so shadow
        // rays get the correct expected transmittance at semi-transparent edges.
        // opacity=1 → fully blocks (returns black); opacity=0 → fully passes (returns white).
        // This is the expected-value equivalent of the stochastic test used in
        // the camera/bounce path loops, and avoids needing a sampler here.
        if (m_p.alphaMask) {
            float alpha = evalTOV(m_p.opacity, ctx.uv);
            float t = 1.f - alpha;
            if (t <= 0.f) return {};           // fully opaque
            return {t, t, t};                  // partial transmittance at edges
        }
        // Diffuse translucency: let shadow rays pass through, tinted by translucency_color.
        if (m_p.translucency > 0.001f)
            return evalTOV(m_p.translucency_color, ctx.uv) * m_p.translucency;

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
        if (ctx.isCurve) return m_hairFallback.sample(ctx, wo, u, uComponent);
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
                    // BSDFSample.f = f_bsdf * |cosI|.
                    // f_bsdf = Fr * D * G2 / (4 * cosO * cosR), so f_bsdf * cosR = Fr * D * G2 / (4 * cosO).
                    Spectrum bsdf = f0 * (D * G2 / (4.f * cosO));
                    s.wi     = sctx.toWorld(wiLocal);
                    s.f      = bsdf * base_color;
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
                    // Walter et al. 2007 microfacet transmission BSDF (importance transport, no eta²):
                    //   f_t = (1-Fr) * D * G2 * |dot(wo,wh)| * |dot(wi,wh)| / (|cosO| * |cosI| * denom²)
                    // BSDFSample.f = f_bsdf * |cosI|, so cosI_t cancels:
                    //   s.f = (1-Fr) * D * G2 * cosIH * absCosT_wh / (cosO * denom²)
                    float absCosT_wh = std::abs(dot(wiLocal, wh));
                    float pdf_wh     = D * wh.z;
                    float jacobian   = eta * eta * absCosT_wh / (denom * denom);
                    s.wi     = sctx.toWorld(wiLocal);
                    s.f      = base_color * ((1.f - Fr) * D * G2 * cosIH * absCosT_wh
                                            / (cosO * denom * denom));
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

        // Layer selection weights (all in [0,1], then renormalize).
        // Diffuse weight uses the spec lobe's directional-hemispherical
        // albedo E_spec — the average fraction the rough specular actually
        // reflects, integrated over the upper hemisphere — instead of the
        // point Fresnel `specF` at this single direction.  The latter
        // double-counted reflectance (especially at grazing angles where
        // specF→1 but rough GGX still scatters most energy elsewhere) and
        // dimmed indirect diffuse.
        float specE  = specAlbedoDielectric(woLocal.z, rough);
        float wCoat  = m_p.coat * coatF;
        float wMetal = metal * (1.f - wCoat);
        float wSpec  = spec * (1.f - metal) * specF * (1.f - wCoat);
        float wDiff  = m_p.base * (1.f - metal)
                     * (1.f - spec * specE) * (1.f - wCoat);
        // In path-tracing mode SSS is approximated as a colored diffuse lobe.
        // Blend base_color toward subsurface_color proportional to subsurface weight
        // so materials with subsurface=1.0 show the SSS color rather than pure white.
        // Near-white SSS color is a fallback default, not an intentional tint — skip
        // blending to preserve the base color (matches OslMaterial::applySssLobeTint guard).
        Spectrum sss_color  = evalTOV(m_p.subsurface_color, ctx.uv);
        float sssLum = 0.299f * sss_color.x + 0.587f * sss_color.y + 0.114f * sss_color.z;
        Spectrum diff_color = (sssLum > 0.65f)
                            ? base_color
                            : base_color * (1.f - m_p.subsurface)
                            + sss_color * m_p.subsurface;

        // Diffuse translucency (ND_translucent_bsdf) — lower hemisphere
        float wTrans = m_p.translucency;

        float wSum = wCoat + wMetal + wSpec + wDiff + wTrans;
        if (wSum <= 0.f) return {};
        float invW = 1.f / wSum;
        wCoat  *= invW;
        wMetal *= invW;
        wSpec  *= invW;
        wDiff  *= invW;
        wTrans *= invW;

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
        } else if (uComponent < wCoat + wMetal + wSpec + wDiff) {
            // Diffuse (uses diff_color which blends base_color toward sss_color)
            result = sampleDiffuse(sctx, woLocal, u, diff_color);
        } else {
            // Power-cosine lobe around the straight-through direction (-wo).
            // scatter=1 → Lambertian (max blur), scatter→0 → delta straight-through (sharp).
            // The lobe width controls both blur and attenuation: a tighter lobe puts
            // less energy toward off-axis directions, naturally dimming distant objects.
            // Returns directly: evalCombined only handles the upper hemisphere.
            Spectrum tColor = evalTOV(m_p.translucency_color, ctx.uv);
            float scatter = m_p.scatter;

            // Through direction in local shading space.
            Vec3f tLocal = { -woLocal.x, -woLocal.y, -woLocal.z };

            // True delta at scatter=0: straight-through with no spread.
            if (scatter < 0.01f) {
                BSDFSample ts;
                ts.wi     = sctx.toWorld(tLocal);
                ts.f      = tColor;
                ts.pdf    = 1.f;
                ts.pdfRev = 1.f;
                ts.eta    = 1.f;
                ts.flags  = BSDFFlag_Diffuse | BSDFFlag_Transmission;
                return ts;
            }

            // Lobe exponent n: scatter=1 → n=1 (Lambertian), scatter→0 → n→large.
            // Quadratic mapping gives better perceptual linearity across the range.
            float n = 1.f / (scatter * scatter);

            // Build ONB around tLocal (Gram-Schmidt, pick least-aligned axis).
            Vec3f bx;
            if (std::abs(tLocal.x) <= std::abs(tLocal.y) && std::abs(tLocal.x) <= std::abs(tLocal.z))
                bx = { 0.f, -tLocal.z,  tLocal.y };
            else if (std::abs(tLocal.y) <= std::abs(tLocal.z))
                bx = { -tLocal.z, 0.f,  tLocal.x };
            else
                bx = {  tLocal.y, -tLocal.x, 0.f };
            bx = safeNormalize(bx);
            Vec3f by = { tLocal.y*bx.z - tLocal.z*bx.y,
                         tLocal.z*bx.x - tLocal.x*bx.z,
                         tLocal.x*bx.y - tLocal.y*bx.x };

            // Importance-sample pdf = (n+1)/(2π) · cosα^n  where cosα = dot(wi, tLocal).
            float phi      = 2.f * kSS_Pi * u.x;
            float cosAlpha = std::pow(u.y, 1.f / (n + 1.f));
            float sinAlpha = std::sqrt(std::max(0.f, 1.f - cosAlpha * cosAlpha));
            Vec3f wiLocal  = bx * (sinAlpha * std::cos(phi))
                           + by * (sinAlpha * std::sin(phi))
                           + tLocal * cosAlpha;

            // f = tColor · pdf  →  f/pdf = tColor (energy-conserving, no grazing singularity).
            float pdfVal = (n + 1.f) * 0.5f * kSS_InvPi * std::pow(cosAlpha, n);

            BSDFSample ts;
            ts.wi     = sctx.toWorld(wiLocal);
            ts.f      = tColor * pdfVal;
            ts.pdf    = pdfVal;
            ts.pdfRev = pdfVal;
            ts.eta    = 1.f;
            ts.flags  = BSDFFlag_Diffuse | BSDFFlag_Transmission;
            return ts;
        }

        if (!result.isValid()) return {};

        // Compute combined PDF and f across all layers (for firefly-free evaluation)
        float pdfFwd = 0.f, pdfRev = 0.f;
        Spectrum fCombined = evalCombined(ctx, woLocal, sctx.toLocal(result.wi),
                                          wCoat, wMetal, wSpec, wDiff,
                                          diff_color, spec, rough, alpha2,
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
        if (ctx.isCurve) return m_hairFallback.evaluate(ctx, wo, wi);
        Vec3f woLocal = ctx.toLocal(wo);
        Vec3f wiLocal = ctx.toLocal(wi);

        // Opposite hemisphere: translucency or rough dielectric transmission.
        if (!sameHemisphere(woLocal, wiLocal)) {
            if (m_p.translucency > 0.001f && woLocal.z > 0.f && wiLocal.z < 0.f) {
                Spectrum tColor = evalTOV(m_p.translucency_color, ctx.uv);
                float scatter = m_p.scatter;
                // Delta at scatter=0: no area contribution to NEE.
                if (scatter < 0.01f) return {};
                float n = 1.f / (scatter * scatter);
                Vec3f tLocal   = { -woLocal.x, -woLocal.y, -woLocal.z };
                float cosAlpha = std::max(0.f, dot(wiLocal, tLocal));
                if (cosAlpha <= 0.f) return {};
                float absZ   = std::max(std::abs(wiLocal.z), 1e-4f);
                float pdfVal = (n + 1.f) * 0.5f * kSS_InvPi * std::pow(cosAlpha, n);
                // f = tColor * pdfVal / absZ so that f * cosI = tColor * pdfVal (bounded).
                BSDFEval e;
                e.f      = tColor * (pdfVal / absZ);
                e.pdf    = pdfVal;
                e.pdfRev = pdfVal;
                return e;
            }
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
            // Walter et al. 2007 microfacet transmission BSDF (importance transport, no eta²):
            //   f_t = (1-Fr) * D * G2 * |dot(wo,wh)| * |dot(wi,wh)| / (|cosO| * |cosI| * denom²)
            float bsdf_scalar = (1.f - Fr) * D * G2 * cosIH * cosTH
                              / (woLocal.z * std::abs(wiLocal.z) * denom * denom);
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

        // Glass (transmission > 0, non-metallic) in same hemisphere = Fresnel reflection only.
        // Without this guard, the function falls through to the opaque diffuse+specular BSDF,
        // which gives a full diffuse + specular response for a transmissive material — making
        // glass objects appear incandescent under direct lighting (NEE).
        if (m_p.transmission > 0.001f && metal < 0.001f) {
            if (rough < 0.001f) return {};   // smooth delta glass: no area PDF for reflection
            float eta    = ctx.frontFace ? m_p.specular_IOR : (1.f / m_p.specular_IOR);
            Vec3f wh     = safeNormalize(woLocal + wiLocal);
            if (wh.z < 0.f) return {};       // degenerate half-vector
            float cosIH  = std::max(0.f, dot(woLocal, wh));
            float Fr     = fresnelDielectric(cosIH, eta);
            float D      = D_GGX(wh.z, alpha2);
            float G2     = G2_Smith_Separable(woLocal.z, wiLocal.z, alpha2);
            float cosR   = wiLocal.z;
            float pdf    = Fr > 0.f ? Fr * pdfGGX_reflection(wh.z, alpha2, cosIH) : 0.f;
            BSDFEval e;
            // evaluate() returns pure BSDF (no cosine factor — integrator applies it externally).
            // Standard microfacet reflection BSDF: Fr * D * G2 / (4 * |cosO| * |cosI|)
            e.f      = base_color * (Fr * D * G2 / (4.f * woLocal.z * cosR));
            e.pdf    = pdf;
            e.pdfRev = pdf;   // symmetric Fresnel
            return e;
        }

        float coatF = schlickDielectric(woLocal.z, m_coatF0);
        float specF = schlickDielectric(woLocal.z, m_f0Dielectric);

        // Energy-conserving spec/diff balance — see the same calculation in
        // sample() for the rationale.  E_spec replaces specF in the diffuse
        // weight to avoid double-counting the spec lobe's reflectance at
        // grazing angles (where the rough GGX*F integrates to << specF).
        float specE  = specAlbedoDielectric(woLocal.z, rough);
        float wCoat  = m_p.coat * coatF;
        float wMetal = metal * (1.f - wCoat);
        float wSpec  = spec * (1.f - metal) * specF * (1.f - wCoat);
        float wDiff  = m_p.base * (1.f - metal)
                     * (1.f - spec * specE) * (1.f - wCoat);
        Spectrum sss_color_e  = evalTOV(m_p.subsurface_color, ctx.uv);
        float sssLumE = 0.299f * sss_color_e.x + 0.587f * sss_color_e.y + 0.114f * sss_color_e.z;
        Spectrum diff_color_e = (sssLumE > 0.65f)
                              ? base_color
                              : base_color * (1.f - m_p.subsurface)
                              + sss_color_e * m_p.subsurface;

        float wTrans = m_p.translucency;  // translucency takes probability from reflection pool
        float wSum = wCoat + wMetal + wSpec + wDiff + wTrans;
        if (wSum <= 0.f) return {};
        float invW = 1.f / wSum;
        wCoat  *= invW; wMetal *= invW; wSpec *= invW; wDiff *= invW;

        float pdfFwd, pdfRev;
        Spectrum f = evalCombined(ctx, woLocal, wiLocal,
                                  wCoat, wMetal, wSpec, wDiff,
                                  diff_color_e, spec, rough, alpha2,
                                  pdfFwd, pdfRev);

        BSDFEval e;
        e.f      = f;
        e.pdf    = pdfFwd;
        e.pdfRev = pdfRev;
        return e;
    }

    float pdf(const ShadingContext& ctx,
              Vec3f wo, Vec3f wi) const override {
        if (ctx.isCurve) return m_hairFallback.pdf(ctx, wo, wi);
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
                           Spectrum base_color, float spec, float roughness,
                           float alpha2,
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

        // Metallic layer.  Single-scatter GGX + Kulla-Conty multi-scatter
        // compensation — without the MS term, rough metals are visibly dim
        // because single-scatter GGX masks ~half the energy at α≈1.
        if (wMetal > 0.f) {
            float pF, pR;
            Spectrum F0 = base_color * m_p.base;
            Spectrum fM = evalGGX(woLocal, wiLocal, alpha2, F0, pF, pR);
            fM = fM + evalGGX_ms(woLocal, wiLocal, roughness, F0);
            f += fM * wMetal;
            pdfFwd += pF * wMetal;
            pdfRev += pR * wMetal;
        }

        // Dielectric specular layer (same MS compensation; effect is small at
        // F0=0.04 but consistent with the metal lobe).
        if (wSpec > 0.f) {
            float pF, pR;
            Spectrum f0 = m_p.specular_color * (spec * m_f0Dielectric);
            Spectrum fS = evalGGX(woLocal, wiLocal, alpha2, f0, pF, pR);
            fS = fS + evalGGX_ms(woLocal, wiLocal, roughness, f0);
            f += fS * wSpec;
            pdfFwd += pF * wSpec;
            pdfRev += pR * wSpec;
        }

        // Diffuse layer — Disney/Burley diffuse with roughness-dependent
        // retro-reflection (Burley 2012, "Physically-Based Shading at
        // Disney").  At low roughness Fd90 ≈ 0.5 so the term DARKENS the
        // diffuse at grazing (consistent with smooth dielectrics shedding
        // light forward); at roughness=1 Fd90 = 2.5 so the term BRIGHTENS
        // the diffuse at grazing — the retro-reflection peak that gives
        // Cycles its softer wall-bounce look.  Reduces to plain Lambertian
        // when both cos_i and cos_o = 1.
        if (wDiff > 0.f) {
            float cosI = std::abs(wiLocal.z);
            float cosO = std::abs(woLocal.z);
            Vec3f wh   = safeNormalize(woLocal + wiLocal);
            float cosD = std::max(0.f, dot(wiLocal, wh));
            float Fd90 = 0.5f + 2.f * cosD * cosD * roughness;
            float Fview  = 1.f + (Fd90 - 1.f) * std::pow(1.f - cosO, 5.f);
            float Flight = 1.f + (Fd90 - 1.f) * std::pow(1.f - cosI, 5.f);
            Spectrum fD  = base_color * m_p.base
                         * (kSS_InvPi * Fview * Flight);
            float pF = cosI * kSS_InvPi;
            float pR = cosO * kSS_InvPi;
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

    Params                m_p;
    float                 m_alpha, m_alpha2;
    float                 m_coatAlpha, m_coatAlpha2;
    float                 m_f0Dielectric;
    float                 m_coatF0;
    MarschnerHairMaterial m_hairFallback{MarschnerHairMaterial::Params{}};
};

} // namespace anacapa
