#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/accel/IAccelerationStructure.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// ShadingContext
//
// All per-hit data needed by material/BSDF evaluation. Computed once after
// intersection and passed by const reference to all shading functions.
// Stateless — no material-specific data lives here.
// ---------------------------------------------------------------------------
struct ShadingContext {
    Vec3f p;        // World-space hit point
    Vec3f n;        // Shading normal (points toward exterior)
    Vec3f ng;       // Geometric normal
    Vec3f t;        // Primary tangent (dpdu direction, normalized)
    Vec3f bt;       // Bitangent = cross(n, t)
    Vec2f uv;       // Surface UV
    Vec3f color = {1.f, 1.f, 1.f};  // per-strand color (curves); white for surfaces

    // Whether the ray hit the front face (dot(wo, ng) > 0)
    bool  frontFace = true;

    // Construct from a SurfaceInteraction + incoming ray direction
    ShadingContext(const SurfaceInteraction& si, Vec3f rayDir) {
        p     = si.p;
        ng    = si.ng;
        uv    = si.uv;
        color = si.color;

        frontFace = dot(-rayDir, ng) > 0.f;
        // Flip normals for consistent outward convention
        n  = frontFace ? si.n  : -si.n;
        ng = frontFace ? si.ng : -si.ng;

        if (si.isCurve) {
            // For hair/curves the fiber tangent is stored in dpdu.
            // Use it directly so Marschner shading sees the correct fiber direction.
            // bt is derived from n and t; n was set to the ribbon facing normal.
            t  = si.dpdu;
            bt = safeNormalize(cross(n, t));
        } else {
            buildOrthonormalBasis(n, t, bt);
        }
    }

    // Transform a vector from world space to local shading frame
    // Local frame: +Z = shading normal
    Vec3f toLocal(Vec3f v) const {
        return {dot(v, t), dot(v, bt), dot(v, n)};
    }

    // Transform from local shading frame to world space
    Vec3f toWorld(Vec3f v) const {
        return t * v.x + bt * v.y + n * v.z;
    }
};

// Cosine of the angle between a local-frame direction and the surface normal
inline float cosTheta(Vec3f w)    { return w.z; }
inline float absCosTheta(Vec3f w) { return std::abs(w.z); }
inline bool  sameHemisphere(Vec3f a, Vec3f b) { return a.z * b.z > 0.f; }

} // namespace anacapa
