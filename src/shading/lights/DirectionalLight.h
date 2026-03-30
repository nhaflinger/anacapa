#pragma once

#include <anacapa/shading/ILight.h>
#include <anacapa/core/Types.h>
#include <cmath>

namespace anacapa {

static constexpr float kDirLightPi    = 3.14159265358979323846f;
static constexpr float kDirLightInvPi = 0.31830988618379067154f;

// ---------------------------------------------------------------------------
// DirectionalLight — a sun-like light at infinite distance.
//
// All shadow rays travel in a single direction (delta distribution), so
// isDelta() == true.  Direct-lighting samples land at `dirToLight` regardless
// of the shading point.
//
// For BDPT light-subpath construction (sampleLe) we place an imaginary disk
// of radius `sceneRadius` at distance 100×sceneRadius from `sceneCenter` in
// the light direction.  This gives a finite spawn point while keeping the
// angular error from the true parallel-light approximation below ~0.6°.
// ---------------------------------------------------------------------------
class DirectionalLight : public ILight {
public:
    // dirToLight : unit vector FROM a surface point TOWARD the light source
    // Le         : emitted radiance
    // sceneRadius: half the scene bounding-sphere diameter (for sampleLe disk)
    // sceneCenter: center of the scene bounding sphere
    DirectionalLight(Vec3f dirToLight, Spectrum Le,
                     float sceneRadius, Vec3f sceneCenter)
        : m_dir(safeNormalize(dirToLight))
        , m_Le(Le)
        , m_sceneRadius(sceneRadius)
        , m_sceneCenter(sceneCenter)
    {
        m_diskArea = kDirLightPi * sceneRadius * sceneRadius;
    }

    // -----------------------------------------------------------------------
    // Direct-lighting interface
    // -----------------------------------------------------------------------
    LightSample sample(Vec3f /*p*/, Vec3f /*n*/, Vec2f /*u*/) const override {
        LightSample s;
        s.wi      = m_dir;          // toward the sun
        s.Li      = m_Le;
        s.pdf     = 1.f;            // delta distribution; weight handled by isDelta
        s.dist    = 1e10f;          // no finite position — let shadow rays exit freely
        s.isDelta = true;
        return s;
    }

    float pdf(Vec3f /*p*/, Vec3f /*wi*/) const override {
        return 0.f;  // delta: can never be sampled by BSDF sampling
    }

    // -----------------------------------------------------------------------
    // BDPT: spawn a ray from the light.
    // Disk placed at  sceneCenter + dirToLight * 100R,  radius = sceneRadius.
    // Rays travel in direction  -dirToLight  toward the scene.
    // -----------------------------------------------------------------------
    LightLeSample sampleLe(Vec2f uPos, Vec2f /*uDir*/) const override {
        // Uniform disk sampling
        float r   = m_sceneRadius * std::sqrt(uPos.x);
        float phi = 2.f * kDirLightPi * uPos.y;
        Vec3f t, bt;
        buildOrthonormalBasis(m_dir, t, bt);
        Vec3f diskCenter = m_sceneCenter + m_dir * (m_sceneRadius * 100.f);
        Vec3f pos        = diskCenter
                         + t  * (r * std::cos(phi))
                         + bt * (r * std::sin(phi));

        // Disk normal faces toward scene (-m_dir), ray direction is -m_dir
        Vec3f normal = -m_dir;
        Vec3f dir    = -m_dir;

        LightLeSample s;
        s.Le     = m_Le;
        s.pos    = pos;
        s.normal = normal;
        s.dir    = dir;
        s.pdfPos = 1.f / m_diskArea;
        s.pdfDir = 1.f;   // delta in direction; treated as fixed by BDPT
        return s;
    }

    // Radiance emitted from a point on the imaginary disk in direction wo.
    // Returns Le when wo points back toward scene (dot with disk normal > 0).
    Spectrum Le(Vec3f /*pos*/, Vec3f normal, Vec3f wo) const override {
        return dot(wo, normal) > 0.f ? m_Le : Spectrum{};
    }

    float power() const override {
        // Approximate: uniform irradiance over disk area × π steradians
        return luminance(m_Le) * m_diskArea * kDirLightPi;
    }

    bool isInfinite() const override { return true;  }
    bool isDelta()    const override { return true;  }

private:
    Vec3f    m_dir;          // from surface toward light
    Spectrum m_Le;
    float    m_sceneRadius;
    Vec3f    m_sceneCenter;
    float    m_diskArea = 1.f;
};

} // namespace anacapa
