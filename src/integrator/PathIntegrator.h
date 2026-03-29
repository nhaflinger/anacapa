#pragma once

#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/sampling/ISampler.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// PathIntegrator — unidirectional path tracer with next-event estimation
//
// This is the Phase 1 reference integrator. It provides:
//   1. A known-correct baseline to validate the BVH and shading stack
//   2. A comparison target to verify BDPT produces equal-mean estimates
//
// Algorithm: For each pixel sample, trace a camera path of up to maxDepth
// bounces. At each bounce, sample the BSDF and also perform direct light
// sampling (next-event estimation) with MIS using the power heuristic.
// ---------------------------------------------------------------------------
class PathIntegrator : public IIntegrator {
public:
    explicit PathIntegrator(uint32_t maxDepth = 8, uint32_t minDepth = 2)
        : m_maxDepth(maxDepth), m_minDepth(minDepth) {}

    void prepare(const SceneView& scene) override {
        // Build alias table for light selection (uniform for now)
        m_lightCount = static_cast<uint32_t>(scene.lights.size());
    }

    void renderTile(const SceneView& scene,
                    const TileRequest& tile,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    ISampler& sampler,
                    TileBuffer& localTile) override;

private:
    Spectrum Li(const Ray& ray, const SceneView& scene,
                ISampler& sampler, uint32_t depth) const;

    // Direct lighting: sample one light, return MIS-weighted contribution
    Spectrum estimateDirect(const SurfaceInteraction& si,
                             const IMaterial& mat,
                             Vec3f wo,
                             const ILight& light,
                             const SceneView& scene,
                             ISampler& sampler) const;

    static float powerHeuristic(float nF, float pdfF,
                                  float nG, float pdfG) {
        float f = nF * pdfF, g = nG * pdfG;
        return (f * f) / (f*f + g*g);
    }

    uint32_t m_maxDepth  = 8;
    uint32_t m_minDepth  = 2;
    uint32_t m_lightCount = 0;
};

} // namespace anacapa
