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
    };

    SoftParticleMaterial() : m_color{1.f, 1.f, 1.f}, m_strength{1.f} {}
    explicit SoftParticleMaterial(const Params& p)
        : m_color(p.color), m_strength(p.emissionStrength) {}

    bool     isDelta()  const override { return true; }
    uint32_t flags()    const override { return BSDFFlag_Specular | BSDFFlag_Transmission; }

    // Pure delta forward transmission — ray continues straight through.
    BSDFSample sample(const ShadingContext& /*ctx*/,
                      Vec3f wo, Vec2f /*u*/, float /*uComp*/) const override {
        BSDFSample s;
        s.wi     = -wo;
        s.pdf    = 1.f;
        s.pdfRev = 1.f;
        s.f      = {1.f, 1.f, 1.f};
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

    // Emission accumulated by each disc hit.
    Spectrum Le(const ShadingContext&, Vec3f) const override {
        return m_color * m_strength;
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
};

} // namespace anacapa
