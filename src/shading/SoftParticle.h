#pragma once

#include <anacapa/shading/IMaterial.h>
#include <anacapa/core/Types.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// SoftParticleMaterial — additive emitter for particle disc halos.
//
// Each ray that intersects a particle disc accumulates Le = color * strength
// and then continues straight through (delta forward transmission, weight=1).
// Multiple overlapping discs stack additively: total = sum(color_i * strength_i).
//
// Shadow rays are fully unblocked (transmittanceColor = white).
// Not registered as an NEE light source (probeEmission returns {}).
// ---------------------------------------------------------------------------
class SoftParticleMaterial : public IMaterial {
public:
    struct Params {
        Spectrum color            = {1.f, 1.f, 1.f};
        float    emissionStrength = 1.f;
        float    opacity          = 0.05f; // low default: particles stack additively for glow effects
    };

    SoftParticleMaterial() : m_color{1.f, 1.f, 1.f}, m_strength{1.f}, m_opacity{0.05f} {}
    explicit SoftParticleMaterial(const Params& p)
        : m_color(p.color), m_strength(p.emissionStrength), m_opacity(p.opacity) {}

    bool     isDelta()  const override { return true; }
    uint32_t flags()    const override { return BSDFFlag_Specular | BSDFFlag_Transmission; }

    float opacity() const { return m_opacity; }

    // Throughput-decay transmission: f = (1-opacity).
    // Dense clusters converge to Le/opacity >> 1, blowing out to white.
    // A single isolated particle shows at full Le regardless of opacity.
    BSDFSample sample(const ShadingContext& /*ctx*/,
                      Vec3f wo, Vec2f /*u*/, float /*uComp*/) const override {
        BSDFSample s;
        s.wi     = -wo;
        s.pdf    = 1.f;
        s.pdfRev = 1.f;
        float t  = 1.f - m_opacity;
        s.f      = {t, t, t};
        s.flags  = BSDFFlag_Specular | BSDFFlag_Transmission;
        s.eta    = 1.f;
        return s;
    }

    BSDFEval evaluate(const ShadingContext&, Vec3f, Vec3f) const override {
        return {};  // delta — no area PDF
    }

    float pdf(const ShadingContext&, Vec3f, Vec3f) const override {
        return 0.f;
    }

    // Unscaled emission — opacity does not dim individual particles.
    // Dense clusters accumulate to Le/opacity (>> 1) and blow out to white.
    Spectrum Le(const ShadingContext& ctx, Vec3f) const override {
        return ctx.color * m_color * m_strength;
    }

    // Don't register this as an NEE light source — particles live in HaloAccel,
    // not the BVH, and are not sampled by direct light estimators.
    Spectrum probeEmission() const override { return {}; }

    // Fully transparent to shadow rays so particles don't cast shadows.
    Spectrum transmittanceColor(const ShadingContext&) const override {
        return {1.f, 1.f, 1.f};
    }

    Spectrum reflectance(const ShadingContext&) const override { return {}; }

private:
    Spectrum m_color;
    float    m_strength;
    float    m_opacity;
};

} // namespace anacapa
