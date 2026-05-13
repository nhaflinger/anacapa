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
//
// `deltaOpaque` — when true, delta (specular) surfaces block the shadow ray.
// This is the right behavior only when a photon map is the sole carrier of
// caustic light through those surfaces; otherwise NEE would double-count.
// In pure path/BDPT mode (no photon map) leave it false: blocking NEE
// through glass with no compensating photon contribution just deletes energy
// and produces a fake hard shadow with no physical justification.
// ---------------------------------------------------------------------------
inline Spectrum shadowTransmittance(Ray ray,
                                    const SceneView& scene,
                                    bool deltaOpaque = false,
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

        // Delta (specular) surfaces — only opaque when (a) the caller has a
        // photon map running and (b) this specific material is opted in as a
        // caustic generator.  Non-flagged glass keeps using transmittanceColor()
        // so windows, drinking glasses, etc. still pass shadow rays correctly
        // in --integrator photon.
        if (deltaOpaque && mat->isDelta() && mat->isCausticGenerator()) return {};

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
