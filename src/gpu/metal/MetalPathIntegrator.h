#pragma once

#include <anacapa/integrator/IIntegrator.h>
#include <memory>
#include <string>

namespace anacapa {

class MetalContext;

// ---------------------------------------------------------------------------
// MetalPathIntegrator — GPU-accelerated wavefront path tracer (Metal backend)
//
// Implements the same IIntegrator interface as PathIntegrator/BDPTIntegrator.
// The CPU tile loop in RenderSession is unchanged; this integrator dispatches
// Metal compute kernels for each tile instead of running on CPU threads.
//
// Feature parity with CPU PathIntegrator (not BDPT):
//   - Lambertian + GGX BSDF (inline MSL, no virtual dispatch)
//   - AreaLight, DirectionalLight direct lighting
//   - DomeLight environment (texture sampled on GPU)
//   - No BDPT, no OSL
//
// This is appropriate for interactive/preview renders where speed > full
// accuracy. Use --interactive to select this integrator.
// ---------------------------------------------------------------------------
class MetalPathIntegrator : public IIntegrator {
public:
    // metallibPath: path to compiled anacapa.metallib (provided by CMake)
    explicit MetalPathIntegrator(const std::string& metallibPath);
    ~MetalPathIntegrator() override;

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

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
