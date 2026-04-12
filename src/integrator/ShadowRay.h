#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ShadingContext.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// shadowTransmittance
//
// Traces a shadow ray from `origin` toward `target`, stepping through
// transmissive surfaces and accumulating their tint.
//
// Returns:
//   White (1,1,1) — nothing between origin and target, full light.
//   Black (0,0,0) — at least one opaque surface blocks the path.
//   Tinted value   — one or more glass surfaces; multiply direct-lighting
//                    contribution by this value.
//
// `shadowRay` should already be spawned (offset + tMax set to the distance
//  to the light).  This function re-uses the same tMax but advances the
//  origin past each transparent surface it encounters.
// ---------------------------------------------------------------------------
inline Spectrum shadowTransmittance(Ray ray,
                                    const SceneView& scene,
                                    int maxTransparent = 8)
{
    Spectrum T = {1.f, 1.f, 1.f};

    for (int i = 0; i < maxTransparent; ++i) {
        TraceResult hit = scene.accel->trace(ray);
        if (!hit.hit) break;   // clear path — return T as-is

        const SurfaceInteraction& si = hit.si;
        const IMaterial* mat = (si.meshID < scene.materials.size())
                               ? scene.materials[si.meshID] : nullptr;
        if (!mat) return {};   // unknown geometry, treat as opaque

        ShadingContext ctx(si, ray.direction);
        Spectrum tint = mat->transmittanceColor(ctx);
        if (isBlack(tint)) return {};   // opaque surface blocks the light

        T = T * tint;
        if (isBlack(T)) return {};

        // Advance the ray past this surface and continue toward the light.
        // Spawn from the hit point in the same direction, keeping the
        // remaining tMax so we stop at the original light position.
        float remaining = ray.tMax - si.t;
        if (remaining <= 1e-4f) break;

        float sceneTime = ray.time;   // preserve scene time across the surface
        ray = spawnRay(si.p, si.ng, ray.direction);
        ray.tMax = remaining - 1e-4f;
        ray.time = sceneTime;
    }

    return T;
}

} // namespace anacapa
