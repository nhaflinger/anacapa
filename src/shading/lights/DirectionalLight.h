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
    // dirToLight  : unit vector FROM a surface point TOWARD the light source
    // Le          : emitted radiance
    // sceneRadius : half the scene bounding-sphere diameter (for sampleLe disk)
    // sceneCenter : center of the scene bounding sphere
    // halfAngleDeg: angular radius of the light disk in degrees (0 = delta/hard shadows,
    //               0.27 = sun, larger = softer shadows)
    DirectionalLight(Vec3f dirToLight, Spectrum Le,
                     float sceneRadius, Vec3f sceneCenter,
                     float halfAngleDeg = 0.f)
        : m_dir(safeNormalize(dirToLight))
        , m_Le(Le)
        , m_sceneRadius(sceneRadius)
        , m_sceneCenter(sceneCenter)
        , m_cosCone(std::cos(halfAngleDeg * kDirLightPi / 180.f))
    {
        m_diskArea = kDirLightPi * sceneRadius * sceneRadius;
        // Solid angle of the cone: 2π(1 - cos θ)
        m_conePdfInv = (halfAngleDeg > 0.f)
            ? 1.f / (2.f * kDirLightPi * (1.f - m_cosCone))
            : 1.f;
    }

    // -----------------------------------------------------------------------
    // Direct-lighting interface
    // -----------------------------------------------------------------------
    LightSample sample(Vec3f /*p*/, Vec3f /*n*/, Vec2f u) const override {
        LightSample s;
        s.dist    = 1e10f;
        s.Li      = m_Le;
        s.pdf     = 1.f;
        s.isDelta = true;  // always treated as delta so energy stays constant

        if (m_cosCone >= 1.f - 1e-6f) {
            // Hard shadow — exact direction
            s.wi = m_dir;
        } else {
            // Jitter direction within cone.  isDelta stays true and pdf stays 1
            // so Li/pdf = m_Le regardless of cone size — no brightness change.
            // The jitter spreads shadow rays across the penumbra each sample,
            // producing soft edges that converge with more SPP.
            float cosTheta = 1.f - u.x * (1.f - m_cosCone);
            float sinTheta = std::sqrt(std::max(0.f, 1.f - cosTheta * cosTheta));
            float phi      = 2.f * kDirLightPi * u.y;
            Vec3f t, bt;
            buildOrthonormalBasis(m_dir, t, bt);
            s.wi = safeNormalize(
                t  * (sinTheta * std::cos(phi)) +
                bt * (sinTheta * std::sin(phi)) +
                m_dir * cosTheta);
        }
        return s;
    }

    float pdf(Vec3f /*p*/, Vec3f /*wi*/) const override {
        return 0.f;  // always delta for MIS purposes
    }

    // -----------------------------------------------------------------------
    // BDPT: spawn a ray from the light.
    // Disk placed at  sceneCenter + dirToLight * 100R,  radius = sceneRadius.
    // Rays travel in direction  -dirToLight  toward the scene.
    // -----------------------------------------------------------------------
    LightLeSample sampleLe(Vec2f uPos, Vec2f uDir) const override {
        // Uniform disk sampling
        float r   = m_sceneRadius * std::sqrt(uPos.x);
        float phi = 2.f * kDirLightPi * uPos.y;
        Vec3f t, bt;
        buildOrthonormalBasis(m_dir, t, bt);
        Vec3f diskCenter = m_sceneCenter + m_dir * (m_sceneRadius * 100.f);
        Vec3f pos        = diskCenter
                         + t  * (r * std::cos(phi))
                         + bt * (r * std::sin(phi));

        // Ray direction: jitter within the cone using uDir, matching sample().
        // This gives BDPT light subpath rays the same angular spread as NEE rays,
        // so soft shadows from --light-angle work correctly in both integrators.
        Vec3f dir;
        if (m_cosCone >= 1.f - 1e-6f) {
            dir = -m_dir;  // hard shadow, no jitter
        } else {
            float cosTheta = 1.f - uDir.x * (1.f - m_cosCone);
            float sinTheta = std::sqrt(std::max(0.f, 1.f - cosTheta * cosTheta));
            float p        = 2.f * kDirLightPi * uDir.y;
            // Jitter around -m_dir (the travel direction away from source)
            Vec3f negDir = -m_dir;
            Vec3f dt, dbt;
            buildOrthonormalBasis(negDir, dt, dbt);
            dir = safeNormalize(
                dt    * (sinTheta * std::cos(p)) +
                dbt   * (sinTheta * std::sin(p)) +
                negDir * cosTheta);
        }

        LightLeSample s;
        s.Le     = m_Le;
        s.pos    = pos;
        s.normal = dir;   // disk normal aligns with travel direction
        s.dir    = dir;
        s.pdfPos = 1.f / m_diskArea;
        s.pdfDir = 1.f;   // delta in direction for MIS purposes
        return s;
    }

    // wo = direction from scene point toward the light (outgoing at the surface,
    // consistent with DomeLight::Le convention). Returns Le when wo aligns with
    // m_dir (the direction from scene toward the light source).
    Spectrum Le(Vec3f /*pos*/, Vec3f /*normal*/, Vec3f wo) const override {
        return dot(wo, m_dir) > m_cosCone - 1e-4f ? m_Le : Spectrum{};
    }

    float power() const override {
        // Approximate: uniform irradiance over disk area × π steradians
        return luminance(m_Le) * m_diskArea * kDirLightPi;
    }

    bool isInfinite() const override { return true; }
    bool isDelta()    const override { return true; }

    void setSceneRadius(float r) { m_sceneRadius = r; m_diskArea = kDirLightPi * r * r; }
    void setSceneCenter(Vec3f c) { m_sceneCenter = c; }
    void setHalfAngleDeg(float deg) {
        m_cosCone    = std::cos(deg * kDirLightPi / 180.f);
        m_conePdfInv = (deg > 0.f)
            ? 1.f / (2.f * kDirLightPi * (1.f - m_cosCone)) : 1.f;
    }

private:
    Vec3f    m_dir;
    Spectrum m_Le;
    float    m_sceneRadius;
    Vec3f    m_sceneCenter;
    float    m_diskArea   = 1.f;
    float    m_cosCone    = 1.f;   // cos(halfAngle); 1.0 = delta (hard shadows)
    float    m_conePdfInv = 1.f;   // 1 / solidAngle of cone
};

} // namespace anacapa
