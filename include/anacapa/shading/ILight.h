#pragma once

#include <anacapa/core/Types.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// LightSample — result of sampling a light toward a shading point
// ---------------------------------------------------------------------------
struct LightSample {
    Spectrum Li;       // Incident radiance (not divided by pdf)
    Vec3f    wi;       // Direction from shading point toward light (world)
    float    pdf = 0.f;
    float    dist = 0.f;   // Distance to light (for shadow ray tMax)
    bool     isDelta = false;  // Point/directional lights
};

// ---------------------------------------------------------------------------
// LightLeSample — for BDPT: sample a ray emitted from the light source
// ---------------------------------------------------------------------------
struct LightLeSample {
    Spectrum Le;        // Emitted radiance along the ray
    Vec3f    pos;       // Origin on the light surface
    Vec3f    normal;    // Surface normal at origin
    Vec3f    dir;       // Emission direction (world space)
    float    pdfPos = 0.f;  // Area PDF of the position
    float    pdfDir = 0.f;  // Directional PDF (solid angle from pos)
};

// ---------------------------------------------------------------------------
// ILight
// ---------------------------------------------------------------------------
class ILight {
public:
    virtual ~ILight() = default;

    // Sample radiance arriving at world-space point p with normal n.
    virtual LightSample sample(Vec3f p, Vec3f n, Vec2f u) const = 0;

    // PDF of the above sampling strategy (for MIS).
    virtual float pdf(Vec3f p, Vec3f wi) const = 0;

    // BDPT: sample a ray leaving the light (for building light subpaths).
    virtual LightLeSample sampleLe(Vec2f uPos, Vec2f uDir) const = 0;

    // Evaluate emitted radiance in direction wo from point pos on the light.
    virtual Spectrum Le(Vec3f pos, Vec3f normal, Vec3f wo) const = 0;

    // Approximate total emitted power (used to importance-sample lights)
    virtual float power() const = 0;

    // True for directional lights / environment maps (no finite position)
    virtual bool isInfinite() const { return false; }

    // True for point/directional lights (delta position or direction)
    virtual bool isDelta() const { return false; }
};

} // namespace anacapa
