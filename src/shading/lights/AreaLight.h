#pragma once

#include <anacapa/shading/ILight.h>
#include <cmath>

namespace anacapa {

static constexpr float kAreaLightPi    = 3.14159265358979323846f;
static constexpr float kAreaLightInvPi = 0.31830988618379067154f;

// ---------------------------------------------------------------------------
// AreaLight — a planar quadrilateral emitter defined by a world-space
// transform. Uniform area sampling, cosine-weighted directional emission.
//
// For Phase 3 this will be driven by UsdLuxRectLight.
// ---------------------------------------------------------------------------
class AreaLight : public ILight {
public:
    // center: world-space center of the quad
    // u/v:    half-extent vectors along the two axes of the quad
    // Le:     emitted radiance
    AreaLight(Vec3f center, Vec3f uAxis, Vec3f vAxis, Spectrum Le)
        : m_center(center)
        , m_uAxis(uAxis)
        , m_vAxis(vAxis)
        , m_Le(Le)
    {
        m_normal = safeNormalize(cross(uAxis, vAxis));
        m_area   = cross(uAxis, vAxis).length() * 4.f;  // full quad area
    }

    LightSample sample(Vec3f p, Vec3f /*n*/, Vec2f u) const override {
        // Uniform sample on the quad surface
        Vec3f pos = m_center
                  + m_uAxis * (2.f * u.x - 1.f)
                  + m_vAxis * (2.f * u.y - 1.f);

        Vec3f toLight = pos - p;
        float dist2   = toLight.lengthSq();
        float dist    = std::sqrt(dist2);
        Vec3f wi      = toLight * (1.f / dist);

        float cosL = dot(-wi, m_normal);  // Cosine at light surface
        if (cosL <= 0.f) return {};       // Back face

        // Convert area PDF to solid-angle PDF
        float pdfSolidAngle = dist2 / (cosL * m_area);

        LightSample s;
        s.Li      = m_Le;
        s.wi      = wi;
        s.pdf     = pdfSolidAngle;
        s.dist    = dist * (1.f - 1e-4f);
        s.isDelta = false;
        return s;
    }

    float pdf(Vec3f p, Vec3f wi) const override {
        // Intersect the quad plane to get the sample point
        float denom = dot(wi, -m_normal);
        if (denom < 1e-6f) return 0.f;

        float t = dot(m_center - p, -m_normal) / denom;
        if (t <= 0.f) return 0.f;

        Vec3f hitPoint = p + wi * t;
        Vec3f offset   = hitPoint - m_center;

        // Check within quad extents
        float uLen = m_uAxis.length();
        float vLen = m_vAxis.length();
        float uProj = dot(offset, m_uAxis) / (uLen * uLen);
        float vProj = dot(offset, m_vAxis) / (vLen * vLen);
        if (std::abs(uProj) > 1.f || std::abs(vProj) > 1.f) return 0.f;

        float dist2 = t * t;
        float cosL  = denom;
        return dist2 / (cosL * m_area);
    }

    LightLeSample sampleLe(Vec2f uPos, Vec2f uDir) const override {
        // Uniform position on quad
        Vec3f pos = m_center
                  + m_uAxis * (2.f * uPos.x - 1.f)
                  + m_vAxis * (2.f * uPos.y - 1.f);

        // Cosine-weighted direction from the light surface
        float phi      = 2.f * kAreaLightPi * uDir.x;
        float cosTheta = std::sqrt(uDir.y);
        float sinTheta = std::sqrt(1.f - uDir.y);
        Vec3f t, bt;
        buildOrthonormalBasis(m_normal, t, bt);
        Vec3f dir = t * (sinTheta * std::cos(phi))
                  + bt * (sinTheta * std::sin(phi))
                  + m_normal * cosTheta;

        LightLeSample s;
        s.Le     = m_Le;
        s.pos    = pos;
        s.normal = m_normal;
        s.dir    = dir;
        s.pdfPos = 1.f / m_area;
        s.pdfDir = cosTheta * kAreaLightInvPi;
        return s;
    }

    // wo = direction from scene point toward the emitter (outgoing at the surface,
    // consistent with DomeLight::Le convention). The emitter is visible when wo
    // and m_normal point in opposite hemispheres (m_normal faces the scene).
    Spectrum Le(Vec3f /*pos*/, Vec3f normal, Vec3f wo) const override {
        return dot(wo, normal) < 0.f ? m_Le : Spectrum{};
    }

    float power() const override {
        // Integral of Le * cos over hemisphere * area
        return luminance(m_Le) * m_area * kAreaLightPi;
    }

private:
    Vec3f    m_center;
    Vec3f    m_uAxis, m_vAxis;
    Vec3f    m_normal;
    Spectrum m_Le;
    float    m_area = 1.f;
};

} // namespace anacapa
