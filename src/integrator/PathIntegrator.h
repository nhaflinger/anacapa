#pragma once

#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/sampling/ISampler.h>
#include "LightSampler.h"

namespace anacapa {

// ---------------------------------------------------------------------------
// PathIntegrator — unidirectional path tracer with next-event estimation
//
// MIS strategy (balance/power heuristic):
//   Direct lighting  : one light selected via power-weighted alias table;
//                      contribution MIS-weighted against the BSDF PDF.
//   Emitter Le       : path-continuation hits tracked with prevBsdfPdf so
//                      every emitter hit (not only specular) adds Le with
//                      the correct weight against the NEE light-sampling PDF.
//   BSDF sampling    : path-continuation ray; emitter hits handled in the
//                      main loop — no separate BSDF branch in estimateDirect.
// ---------------------------------------------------------------------------
class PathIntegrator : public IIntegrator {
public:
    explicit PathIntegrator(uint32_t maxDepth = 8, uint32_t minDepth = 2)
        : m_maxDepth(maxDepth), m_minDepth(minDepth) {}

    void prepare(const SceneView& scene) override {
        m_lightSampler.build(scene.lights);
    }

    void renderTile(const SceneView& scene,
                    const TileRequest& tile,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    ISampler& sampler,
                    TileBuffer& localTile) override;

    void setDebugMeshID(int32_t id) override { m_debugMeshID = id; }

private:
    Spectrum Li(const Ray& ray, const SceneView& scene,
                ISampler& sampler,
                Spectrum& outAlbedo, Vec3f& outNormal) const;

    // Light sampling only — no BSDF branch (emitter Le is handled in Li).
    Spectrum estimateDirect(const SurfaceInteraction& si,
                             const IMaterial& mat,
                             Vec3f wo,
                             const ILight& light,
                             const SceneView& scene,
                             ISampler& sampler,
                             float sceneTime) const;

    // Combined PDF of reaching direction wi from point `from` via the NEE
    // light sampler (selection probability × light's solid-angle PDF).
    float emitterPdf(Vec3f from, Vec3f wi, const SceneView& scene) const;

    static float powerHeuristic(float nF, float pdfF,
                                  float nG, float pdfG) {
        float f = nF * pdfF, g = nG * pdfG;
        return (f * f) / (f*f + g*g);
    }

    uint32_t     m_maxDepth    = 8;
    uint32_t     m_minDepth    = 2;
    int32_t      m_debugMeshID = -1;
    LightSampler m_lightSampler;
};

} // namespace anacapa
