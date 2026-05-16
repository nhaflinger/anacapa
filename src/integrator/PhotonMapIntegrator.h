#pragma once

#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/sampling/ISampler.h>
#include "PhotonMap.h"
#include "LightSampler.h"

namespace anacapa {

// ---------------------------------------------------------------------------
// PhotonMapIntegrator — path tracer augmented with a caustic photon map.
//
// prepare() traces N photons from light sources through the scene.  Photons
// that complete at least one specular bounce before hitting a diffuse surface
// are stored in a balanced kd-tree (the "caustic map").
//
// renderTile() runs the standard path tracing algorithm (identical to
// PathIntegrator) but adds a photon-map density estimate at every diffuse
// surface hit.  This provides clean caustics and glass-focused light without
// the extreme variance that path tracing produces for SDS paths.
//
// The photon map contribution is additive on top of the path integrator's
// direct illumination.  Because the path integrator almost never successfully
// traces SDS caustic paths (probability near zero for small light sources),
// double-counting is negligible for typical caustic configurations.
// ---------------------------------------------------------------------------
class PhotonMapIntegrator : public IIntegrator {
public:
    explicit PhotonMapIntegrator(uint32_t maxDepth     = 8,
                                  uint32_t minDepth     = 2,
                                  float    fireflyClamp = 10.f,
                                  int      numPhotons   = 500'000,
                                  float    searchRadius = 0.1f);

    void prepare(const SceneView& scene) override;

    void renderTile(const SceneView& scene,
                    const TileRequest& tile,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    ISampler& sampler,
                    TileBuffer& localTile) override;

    void setDebugMeshID(int32_t id) override { m_debugMeshID = id; }
    void setPixelFilter(const PixelFilter* f) override { m_pixelFilter = f; }

private:
    // Photon pass: trace photons from lights, populate m_causticMap.
    void tracePhotons(const SceneView& scene);

    // Camera pass: path tracing with photon map query at diffuse hits.
    Spectrum Li(const Ray& ray, const SceneView& scene,
                ISampler& sampler,
                Spectrum& outAlbedo, Vec3f& outNormal) const;

    // Direct illumination estimate (light sampling only, same as PathIntegrator).
    Spectrum estimateDirect(const SurfaceInteraction& si,
                             const IMaterial& mat, Vec3f wo,
                             const ILight& light,
                             const SceneView& scene,
                             ISampler& sampler, float sceneTime) const;

    // Combined NEE PDF for MIS (uniform light selection).
    float emitterPdf(Vec3f from, Vec3f wi, const SceneView& scene) const;

    static float powerHeuristic(float nF, float pdfF, float nG, float pdfG) {
        float f = nF * pdfF, g = nG * pdfG;
        return (f * f) / (f*f + g*g);
    }

    uint32_t           m_maxDepth     = 8;
    uint32_t           m_minDepth     = 2;
    float              m_fireflyClamp = 10.f;
    int                m_numPhotons   = 500'000;
    float              m_searchRadius = 0.1f;
    int32_t            m_debugMeshID  = -1;
    const PixelFilter* m_pixelFilter  = nullptr;

    PhotonMap    m_causticMap;
    PhotonMap    m_sssMap;
    LightSampler m_lightSampler;
};

} // namespace anacapa
