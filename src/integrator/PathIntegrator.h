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
    explicit PathIntegrator(uint32_t maxDepth = 8, uint32_t minDepth = 2,
                            float fireflyClamp = 10.f)
        : m_maxDepth(maxDepth), m_minDepth(minDepth)
        , m_fireflyClamp(fireflyClamp) {}

    void prepare(const SceneView& /*scene*/) override {}

    void renderTile(const SceneView& scene,
                    const TileRequest& tile,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    ISampler& sampler,
                    TileBuffer& localTile) override;

    void setDebugMeshID(int32_t id) override { m_debugMeshID = id; }
    void setPixelFilter(const PixelFilter* f) override { m_pixelFilter = f; }

private:
    Spectrum Li(const Ray& ray, const SceneView& scene,
                ISampler& sampler,
                Spectrum& outAlbedo, Vec3f& outNormal) const;

    // Light sampling only — emitter Le is handled in Li via path continuation.
    Spectrum estimateDirect(const SurfaceInteraction& si,
                             const IMaterial& mat,
                             Vec3f wo,
                             const ILight& light,
                             const SceneView& scene,
                             ISampler& sampler,
                             float sceneTime) const;

    // Combined PDF of reaching direction wi from point `from` via NEE with
    // uniform light selection (1/N × light's solid-angle PDF, summed over lights).
    float emitterPdf(Vec3f from, Vec3f wi, const SceneView& scene) const;

    static float powerHeuristic(float nF, float pdfF,
                                  float nG, float pdfG) {
        float f = nF * pdfF, g = nG * pdfG;
        return (f * f) / (f*f + g*g);
    }

    uint32_t            m_maxDepth     = 8;
    uint32_t            m_minDepth     = 2;
    float               m_fireflyClamp = 10.f;
    int32_t             m_debugMeshID  = -1;
    const PixelFilter*  m_pixelFilter  = nullptr;  // owned by RenderSession
};

} // namespace anacapa
