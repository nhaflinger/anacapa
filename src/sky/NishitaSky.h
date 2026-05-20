#pragma once

#include <anacapa/core/Types.h>
#include <cmath>
#include <algorithm>

namespace anacapa {

// ---------------------------------------------------------------------------
// NishitaParams — physical sky configuration
// ---------------------------------------------------------------------------
struct NishitaParams {
    Vec3f sunDir        = {0.f, 1.f, 0.f}; // toward sun, normalized, Y-up world space
    float sunIntensity  = 1.f;              // sun disc + scene illumination scale
    float sunDiscAngle  = 0.03491f;         // half-angle in radians (~2°, matches Blender default)
    float altitude      = 0.f;             // viewer altitude above sea level (meters)
    float airDensity    = 1.f;             // Rayleigh scattering scale
    float dustDensity   = 1.f;             // Mie scattering scale (aerosols / haze)
    float ozoneDensity  = 1.f;             // Ozone absorption scale
    bool  transparentBg = false;           // render sky as transparent (alpha=0) for compositing
};

// ---------------------------------------------------------------------------
// NishitaSky — single-scattering atmospheric model
//
// Implements the Nishita (1993/1996) sky model with:
//   - Rayleigh scattering  (air molecules — blue sky)
//   - Mie scattering       (aerosols — haze, whitens horizon)
//   - Ozone absorption     (Chappuis band — desaturates blues)
//
// Coordinate convention: Y-up world space.  Earth centre is in the -Y
// direction; the viewer is placed at (0, rEarth + altitude, 0).
//
// Call evaluate(d) for any sky direction; it returns linear-light radiance
// suitable for direct use as a path-tracer environment radiance.
// ---------------------------------------------------------------------------
class NishitaSky {
public:
    // ---- Physical constants ------------------------------------------------

    static constexpr float kPi   = 3.14159265358979323846f;
    static constexpr float k2Pi  = 6.28318530717958647692f;
    static constexpr float kInvPi = 0.31830988618379067f;

    // Planetary geometry (meters)
    static constexpr float kREarth = 6360000.f;
    static constexpr float kRAtmos = 6420000.f;  // atmosphere top (60 km)

    // Atmospheric scale heights (meters)
    static constexpr float kHR = 8000.f;   // Rayleigh
    static constexpr float kHM = 1200.f;   // Mie

    // Rayleigh scattering coefficient at sea level (m⁻¹)
    // β_R for λ ≈ (680 nm, 550 nm, 440 nm) mapped to sRGB R/G/B channels
    static constexpr Vec3f kBetaR = {5.8e-6f, 13.5e-6f, 33.1e-6f};

    // Mie extinction coefficient at sea level (m⁻¹, wavelength-independent)
    static constexpr float kBetaMExt = 21.0e-6f;

    // Mie single-scatter albedo — fraction of extinction that is scattering
    static constexpr float kMieAlbedo = 0.9f;

    // Mie asymmetry parameter g (Henyey-Greenstein, forward peak for haze)
    static constexpr float kMieG = 0.76f;

    // Ozone Chappuis absorption peak (m⁻¹)
    // Values from Bruneton & Neyret (2008), mapped to R/G/B channels
    static constexpr Vec3f kBetaOzone = {0.650e-6f, 1.881e-6f, 0.085e-6f};

    // Ozone layer: Gaussian, peak altitude and width (meters)
    static constexpr float kOzonePeakAlt = 25000.f;
    static constexpr float kOzoneWidth   =  8000.f;

    // Solar radiance scale factor, calibrated to match Blender Cycles' Nishita
    // sky at strength=1 with default exposure.  At this value, sky scatter
    // delivers ~6 W/m² irradiance, giving a white Lambertian surface ~2.0
    // linear reflected radiance (filmic display ~0.82) — matching the
    // appearance of a sunlit surface in Blender.  Sun disc (2° disc angle)
    // is negligible for scene illumination; scatter dominates.
    static constexpr float kSolarRadiance = 150.0f;

    // ---- Construction ------------------------------------------------------

    explicit NishitaSky(const NishitaParams& p) : m_p(p) {}

    // ---- Evaluation --------------------------------------------------------

    // Sky radiance at direction d, excluding the sun disc.
    Spectrum skyRadiance(Vec3f d) const;

    // True if direction d falls inside the sun disc.
    bool inSunDisc(Vec3f d) const {
        return dot(d, m_p.sunDir) >= std::cos(m_p.sunDiscAngle);
    }

    // Radiance of the visible sun disc at direction d (call only when inSunDisc).
    // Models the direct solar beam attenuated by the atmosphere along the
    // viewer-to-atmosphere path in the sun's direction.
    Spectrum sunDiscRadiance() const;

    // Combined sky radiance + optional sun disc.
    Spectrum evaluate(Vec3f d) const {
        Spectrum L = skyRadiance(d);
        if (inSunDisc(d)) L = L + sunDiscRadiance();
        return L;
    }

    const NishitaParams& params() const { return m_p; }

private:
    NishitaParams m_p;

    // ---- Geometry helpers --------------------------------------------------

    // Viewer position in geocentric Y-up coordinates
    Vec3f viewerPos() const { return {0.f, kREarth + m_p.altitude, 0.f}; }

    // Distance to far intersection of ray (orig, dir) with sphere radius r.
    // Returns -1 if no intersection or sphere is behind ray.
    static float raySphereExit(Vec3f orig, Vec3f dir, float r) {
        float b    = dot(orig, dir);
        float c    = dot(orig, orig) - r * r;
        float disc = b * b - c;
        if (disc < 0.f) return -1.f;
        float sq = std::sqrt(disc);
        // We want the far (exit) intersection; take the larger t.
        float t = -b + sq;
        return (t > 0.f) ? t : -1.f;
    }

    // True if ray (orig, dir) intersects the Earth sphere from outside.
    static bool hitsEarth(Vec3f orig, Vec3f dir) {
        float b    = dot(orig, dir);
        if (b >= 0.f) return false;  // pointing away from centre
        float c    = dot(orig, orig) - kREarth * kREarth;
        if (c <= 0.f) return true;   // inside Earth (shouldn't happen for valid alt)
        float disc = b * b - c;
        return disc >= 0.f;
    }

    // Altitude above Earth surface at world-space position p
    static float heightAt(Vec3f p) { return p.length() - kREarth; }

    // ---- Density profiles --------------------------------------------------

    float densityR(float h) const {
        return m_p.airDensity   * std::exp(-std::max(h, 0.f) / kHR);
    }
    float densityM(float h) const {
        return m_p.dustDensity  * std::exp(-std::max(h, 0.f) / kHM);
    }
    float densityO(float h) const {
        float dh = h - kOzonePeakAlt;
        return m_p.ozoneDensity * std::exp(-(dh * dh) / (2.f * kOzoneWidth * kOzoneWidth));
    }

    // ---- Optical depth & transmittance ------------------------------------

    // Scalar optical depths accumulated along a ray segment.
    struct OD { float R = 0.f, M = 0.f, O = 0.f; };

    // Integrate densities along (from, dir, tMax) with N samples.
    OD marchOD(Vec3f from, Vec3f dir, float tMax, int N) const {
        float dt = tMax / static_cast<float>(N);
        OD od;
        for (int i = 0; i < N; ++i) {
            Vec3f p = from + dir * (dt * (i + 0.5f));
            float h = heightAt(p);
            od.R += densityR(h) * dt;
            od.M += densityM(h) * dt;
            od.O += densityO(h) * dt;
        }
        return od;
    }

    // Transmittance (per-channel) for given accumulated optical depths.
    Spectrum transmittance(OD od) const {
        // τ(λ) = β_R(λ)·od.R  +  β_M_ext·od.M  +  β_O(λ)·od.O
        float tauR = kBetaR.x * od.R + kBetaMExt * m_p.dustDensity * od.M + kBetaOzone.x * od.O;
        float tauG = kBetaR.y * od.R + kBetaMExt * m_p.dustDensity * od.M + kBetaOzone.y * od.O;
        float tauB = kBetaR.z * od.R + kBetaMExt * m_p.dustDensity * od.M + kBetaOzone.z * od.O;
        return {std::exp(-tauR), std::exp(-tauG), std::exp(-tauB)};
    }

    // ---- Phase functions ---------------------------------------------------

    static float phaseRayleigh(float cosTheta) {
        return (3.f / (16.f * kPi)) * (1.f + cosTheta * cosTheta);
    }

    static float phaseMie(float cosTheta) {
        constexpr float g  = kMieG;
        constexpr float g2 = g * g;
        float denom = 1.f + g2 - 2.f * g * cosTheta;
        return (1.f / (4.f * kPi)) * (1.f - g2) / (denom * std::sqrt(std::max(denom, 1e-6f)));
    }

    // ---- Ray marching steps ------------------------------------------------
    static constexpr int kNView  = 16;  // primary ray march
    static constexpr int kNLight =  8;  // shadow ray march per step
};

} // namespace anacapa
