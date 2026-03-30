#pragma once

#include <anacapa/shading/IMaterial.h>
#include <cmath>

namespace anacapa {

static constexpr float kPi    = 3.14159265358979323846f;
static constexpr float kInvPi = 0.31830988618379067154f;

// Cosine-weighted hemisphere sampling
inline Vec3f sampleCosineHemisphere(Vec2f u) {
    // Malley's method
    float r   = std::sqrt(u.x);
    float phi = 2.f * kPi * u.y;
    float x   = r * std::cos(phi);
    float y   = r * std::sin(phi);
    float z   = std::sqrt(std::max(0.f, 1.f - u.x));
    return {x, y, z};
}

inline float cosineHemispherePdf(float cosTheta) {
    return std::max(0.f, cosTheta) * kInvPi;
}

// ---------------------------------------------------------------------------
// LambertianMaterial — perfectly diffuse (albedo * 1/pi)
// ---------------------------------------------------------------------------
class LambertianMaterial : public IMaterial {
public:
    explicit LambertianMaterial(Spectrum albedo) : m_albedo(albedo) {}

    bool isDelta() const override { return false; }
    uint32_t flags() const override { return BSDFFlag_Diffuse | BSDFFlag_Reflection; }

    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float) const override {
        Vec3f wiLocal = sampleCosineHemisphere(u);
        if (cosTheta(ctx.toLocal(wo)) < 0.f) wiLocal.z = -wiLocal.z;

        Vec3f wi = ctx.toWorld(wiLocal);
        float p  = cosineHemispherePdf(std::abs(wiLocal.z));

        BSDFSample s;
        s.wi     = wi;
        s.pdf    = p;
        s.pdfRev = p;  // Symmetric for diffuse
        s.f      = m_albedo * (kInvPi * std::abs(wiLocal.z));
        s.flags  = BSDFFlag_Diffuse | BSDFFlag_Reflection;
        return s;
    }

    BSDFEval evaluate(const ShadingContext& ctx,
                       Vec3f wo, Vec3f wi) const override {
        Vec3f wiLocal = ctx.toLocal(wi);
        Vec3f woLocal = ctx.toLocal(wo);

        if (!sameHemisphere(wiLocal, woLocal)) return {};

        float c = absCosTheta(wiLocal);
        BSDFEval e;
        e.f      = m_albedo * kInvPi;
        e.pdf    = cosineHemispherePdf(c);
        e.pdfRev = cosineHemispherePdf(absCosTheta(woLocal));
        return e;
    }

    float pdf(const ShadingContext& ctx,
              Vec3f wo, Vec3f wi) const override {
        Vec3f wiLocal = ctx.toLocal(wi);
        Vec3f woLocal = ctx.toLocal(wo);
        if (!sameHemisphere(wiLocal, woLocal)) return 0.f;
        return cosineHemispherePdf(absCosTheta(wiLocal));
    }

    Spectrum reflectance(const ShadingContext&) const override {
        return m_albedo;
    }

private:
    Spectrum m_albedo;
};

// ---------------------------------------------------------------------------
// EmissiveMaterial — diffuse emitter (wraps a Lambertian + Le term)
// Used for area lights that are also visible geometry.
// ---------------------------------------------------------------------------
class EmissiveMaterial : public LambertianMaterial {
public:
    EmissiveMaterial(Spectrum albedo, Spectrum emission)
        : LambertianMaterial(albedo), m_emission(emission) {}

    Spectrum Le(const ShadingContext&, Vec3f) const override {
        return m_emission;
    }

private:
    Spectrum m_emission;
};

} // namespace anacapa
