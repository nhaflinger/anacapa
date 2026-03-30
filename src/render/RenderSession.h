#pragma once

#include <anacapa/film/Film.h>      // Film, DenoiseOptions
#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/accel/GeometryPool.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/sampling/ISampler.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include "ThreadPool.h"
#include <memory>
#include <string>

namespace anacapa {

enum class IntegratorType { Path, BDPT };

struct RenderSettings {
    uint32_t       imageWidth      = 800;
    uint32_t       imageHeight     = 800;
    uint32_t       samplesPerPixel = 64;
    uint32_t       maxDepth        = 8;
    uint32_t       tileSize        = 64;
    uint32_t       numThreads      = 0;   // 0 = hardware_concurrency
    std::string    outputPath      = "out.exr";
    IntegratorType integrator      = IntegratorType::BDPT;
    DenoiseOptions denoise;
};

// ---------------------------------------------------------------------------
// RenderSession — top-level coordinator
//
// Owns the GeometryPool, BVH, Film, Integrator, and ThreadPool.
// Drives the tile-parallel render loop.
// ---------------------------------------------------------------------------
class RenderSession {
public:
    explicit RenderSession(RenderSettings settings);

    // Build a hardcoded Cornell-box scene (Phase 1)
    void buildCornellBox();

    // Run the render, write output EXR
    void render();

private:
    void buildAccelStructure();
    void scheduleTiles(std::vector<TileRequest>& tiles) const;

    RenderSettings                        m_settings;
    GeometryPool                          m_geomPool;
    SceneView                             m_scene;
    std::unique_ptr<IAccelerationStructure> m_accel;
    std::unique_ptr<Film>                 m_film;
    std::unique_ptr<IIntegrator>          m_integrator;
    std::unique_ptr<ISampler>             m_baseSampler;
    std::unique_ptr<ThreadPool>           m_threadPool;

    // Owned materials and lights (scene takes non-owning pointers)
    std::vector<std::unique_ptr<IMaterial>> m_materials;
    std::vector<std::unique_ptr<ILight>>    m_lights;
};

} // namespace anacapa
