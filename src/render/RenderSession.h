#pragma once

#include <anacapa/film/Film.h>      // Film, DenoiseOptions
#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/accel/GeometryPool.h>
#include <anacapa/accel/CurvePool.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/sampling/ISampler.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include "ThreadPool.h"
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::string    scenePath;              // empty → built-in Cornell box
    std::string    cameraPath;             // empty → auto-select from USD
    std::string    curvesPath;             // Alembic .abc file for hair/fur (empty = none)
    // Material assignment files loaded in order; later entries override earlier ones.
    // Auto-discovered <abc>.matassign.json is always loaded first (if present).
    std::vector<std::string> matassignPaths;
    std::string    envPath;                // HDRI dome light (empty = none)
    float          envIntensity    = 1.f;

    // Thin lens / depth of field overrides.
    // If both are > 0 they override whatever the USD camera specifies (or add
    // DoF to a scene that has none). If either is 0 the USD values are used;
    // if USD also has none the camera falls back to pinhole.
    float          fStop         = 0.f;
    float          focusDistance = 0.f;

    // Frame to render (USD time code).  When frameSet=true, the USD loader
    // evaluates geometry transforms at this frame rather than the stage start.
    int            frameNumber   = 0;
    bool           frameSet      = false;

    // Motion blur shutter interval, relative to frameNumber (in USD time units /
    // frames).  Both 0 = no motion blur.  When shutterClose > shutterOpen, the
    // USD loader collects motion samples over [frame+shutterOpen, frame+shutterClose]
    // and rays are sampled with time in [0, 1] mapping to that window.
    float          shutterOpen   = 0.f;
    float          shutterClose  = 0.f;
    IntegratorType integrator      = IntegratorType::BDPT;
    float          fireflyClamp   = 10.f;  // BDPT: max luminance per (s,t) contribution; 0=off
    float          lightAngle     = 0.f;  // Angular radius for directional lights (degrees, 0=hard)
    bool           adaptive       = true;  // Enable adaptive per-tile sample allocation
    uint32_t       adaptiveBaseSpp = 0;   // 0 = auto (spp/4, min 16)
    bool           interactive     = false; // Use GPU (Metal) backend when available
    bool           overrideLights    = false; // Replace scene lights with a simple white directional
    bool           overrideMaterials = false; // Replace all scene materials with white Lambertian
    std::string    pngPath;                   // If set, write ACES-tonemapped PNG alongside EXR
    float          exposure          = 0.f;   // EV adjustment for PNG output
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

    // Load scene: if settings.scenePath is non-empty, loads from USD file.
    // Otherwise falls back to the built-in Cornell box.
    void loadScene();

    // Run the render, write output EXR
    void render();

private:
    void buildCornellBox();
    void buildAccelStructure();
    void scheduleTiles(std::vector<TileRequest>& tiles) const;
    // Load Alembic curves into m_curvePool (no-op if curvesPath is empty or
    // ANACAPA_ENABLE_ALEMBIC is not set)
    void appendAlembicCurves_();

    RenderSettings                        m_settings;
    GeometryPool                          m_geomPool;
    CurvePool                             m_curvePool;
    SceneView                             m_scene;
    std::optional<Camera>                 m_camera;   // set by USD loader if present
    std::unique_ptr<IAccelerationStructure> m_accel;
    std::unique_ptr<Film>                 m_film;
    std::unique_ptr<IIntegrator>          m_integrator;
    std::unique_ptr<ISampler>             m_baseSampler;
    std::unique_ptr<ThreadPool>           m_threadPool;

    // Shutter interval from the scene file (overridden by RenderSettings if set)
    float m_sceneShutterOpen  = 0.f;
    float m_sceneShutterClose = 0.f;

    // Owned materials and lights (scene takes non-owning pointers)
    std::vector<std::unique_ptr<IMaterial>> m_materials;
    std::vector<std::unique_ptr<ILight>>    m_lights;

    // USD prim path → m_materials index (populated from LoadedScene after USD load).
    // Used to assign USD materials to Alembic hair strands by material path.
    std::unordered_map<std::string, uint32_t> m_materialPathIndex;
};

} // namespace anacapa
