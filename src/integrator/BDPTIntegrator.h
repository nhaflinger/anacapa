#pragma once

#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/integrator/PathVertex.h>
#include <anacapa/integrator/MISWeight.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/shading/ShadingContext.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/sampling/ISampler.h>
#include "LightSampler.h"

namespace anacapa {

// ---------------------------------------------------------------------------
// BDPTIntegrator — bidirectional path tracer with MIS
//
// For each pixel sample:
//   1. Build a camera subpath of up to maxDepth+2 vertices
//   2. Build a light subpath of up to maxDepth+2 vertices
//   3. Connect all valid (s, t) strategy pairs
//   4. Weight each connection by bdptMISWeight(s, t)
//   5. Accumulate into the film (t==1 → splat; t>1 → tile buffer)
//
// The (0,t) strategies are pure camera paths — equivalent to the reference
// PathIntegrator but unified here for a single code path.
// The (s,1) strategies project light-subpath endpoints onto the film.
// The (1,t) strategies are direct light sampling (next-event estimation).
// ---------------------------------------------------------------------------
class BDPTIntegrator : public IIntegrator {
public:
    // fireflyClamp: maximum luminance of any single (s,t) contribution before
    // MIS weighting.  Values above this are scaled down to clamp luminance.
    // 0 = disabled.  Typical production values: 5–20.
    explicit BDPTIntegrator(uint32_t maxDepth = 8, float fireflyClamp = 10.f)
        : m_maxDepth(maxDepth), m_fireflyClamp(fireflyClamp) {}

    void prepare(const SceneView& scene) override;

    void renderTile(const SceneView& scene,
                    const TileRequest& tile,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    ISampler& sampler,
                    TileBuffer& localTile) override;

private:
    // Build the camera subpath starting from a primary ray.
    // Returns number of vertices written into `path`.
    uint32_t traceCameraSubpath(const SceneView& scene,
                                 const Ray& primaryRay,
                                 float primaryPdf,    // area PDF of the camera vertex
                                 ISampler& sampler,
                                 PathVertexBuffer& path) const;

    // Build the light subpath starting from a sampled point on a light.
    // sceneTime is the same time sampled for the camera subpath — ensures both
    // subpaths evaluate animated geometry at the same moment in time.
    // Returns number of vertices written into `path`.
    uint32_t traceLightSubpath(const SceneView& scene,
                                uint32_t lightIdx,
                                float sceneTime,
                                ISampler& sampler,
                                PathVertexBuffer& path) const;

    // Connect light vertex lp[s-1] to camera vertex cp[t-1].
    // Returns the unweighted contribution C(s,t).
    // Returns black if the connection is invalid (delta, occluded, etc.).
    Spectrum connect(const SceneView& scene,
                     const PathVertexBuffer& lp, uint32_t s,
                     const PathVertexBuffer& cp, uint32_t t,
                     Film* film,             // non-null → allowed to splat (t==1)
                     uint32_t filmWidth,
                     uint32_t filmHeight,
                     const Camera& cam) const;

    // Evaluate BSDF * |cosTheta| at a surface vertex
    static Spectrum evalVertex(const PathVertexBuffer& path, uint32_t i,
                                Vec3f toNext, float& fwdPdf, float& revPdf);

    // Geometry term G(v_i, v_j) = |cos theta_i| * |cos theta_j| / dist²
    static float geometryTerm(Vec3f posA, Vec3f nA, Vec3f posB, Vec3f nB);

    uint32_t    m_maxDepth = 8;
    float       m_fireflyClamp = 10.f;
    LightSampler m_lightSampler;

    // Camera stored at prepare() time — needed for (s,1) pixel projection
    Camera      m_camera;
};

} // namespace anacapa
