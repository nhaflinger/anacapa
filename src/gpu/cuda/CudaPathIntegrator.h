#pragma once

#include <anacapa/integrator/IIntegrator.h>
#include <memory>

namespace anacapa {

// ---------------------------------------------------------------------------
// CudaPathIntegrator — GPU-accelerated path tracer (pure CUDA backend)
//
// Implements the same IIntegrator interface as MetalPathIntegrator.
// Selected at runtime when --interactive is passed and ANACAPA_ENABLE_CUDA
// is defined.
//
// Feature parity with CPU PathIntegrator (not BDPT):
//   - Lambertian + GGX BSDF
//   - AreaLight, DirectionalLight, DomeLight
//   - HDRI environment map (CUDA texture)
//   - No BDPT, no OSL
// ---------------------------------------------------------------------------
class CudaPathIntegrator : public IIntegrator {
public:
    CudaPathIntegrator();
    ~CudaPathIntegrator() override;

    void prepare(const SceneView& scene) override;

    bool renderFrame(const SceneView& scene,
                     uint32_t filmWidth,
                     uint32_t filmHeight,
                     uint32_t sampleStart,
                     uint32_t sampleCount,
                     Film& film) override;

    void renderTile(const SceneView& scene,
                    const TileRequest& tile,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    ISampler& sampler,
                    TileBuffer& out) override;

    bool isValid() const;

    // Per-sample luminance ceiling.  When > 0, samples whose luminance
    // exceeds this threshold are scaled down to this value before being
    // added to the accumulation buffer.  Suppresses the bright outliers
    // ("fireflies") common to BDPT/path tracing in scenes with small lights
    // or strong caustics.  0 disables clamping.  Mirrors --firefly-clamp.
    void setFireflyClamp(float v);

    // Zero the persistent accumulation buffer.
    // Call before starting a fresh render (scene/camera change) so stale
    // samples from the previous render do not bleed into the new one.
    void clearAccum();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
