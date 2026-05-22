#pragma once

#include <anacapa/film/Film.h>          // Film, DenoiseOptions
#include <anacapa/film/PixelFilter.h>   // PixelFilterType
#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/render/IDisplayDriver.h>
#include <anacapa/accel/GeometryPool.h>
#include <anacapa/accel/CurvePool.h>
#include <anacapa/accel/HaloPool.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include "../accel/HaloAccel.h"
#include <anacapa/sampling/ISampler.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include "ThreadPool.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace anacapa {

enum class IntegratorType { Path, BDPT, PhotonMap };

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
    std::string    particlesPath;          // Alembic .abc file for IPoints particles (empty = none)
    // Material assignment files loaded in order; later entries override earlier ones.
    // Auto-discovered <abc>.matassign.json is always loaded first (if present).
    std::vector<std::string> matassignPaths;
    std::string    envPath;                // HDRI dome light (empty = none)
    float          envIntensity    = 1.f;
    std::string    skyConfigPath;          // Nishita sky JSON config (empty = none)

    // Thin lens / depth of field overrides.
    // If both are > 0 they override whatever the USD camera specifies (or add
    // DoF to a scene that has none). If either is 0 the USD values are used;
    // if USD also has none the camera falls back to pinhole.
    float          fStop         = 0.f;
    float          focusDistance = 0.f;
    bool           noDof         = false;  // --no-dof: force pinhole regardless of USD camera

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
    bool           gpuAssist       = false; // Use GPU (Metal/CUDA) backend when available
    bool           overrideLights    = false; // Replace scene lights with a simple white directional
    bool           overrideMaterials = false; // Replace all scene materials with white Lambertian
    bool           skipOSL           = false; // Skip OSL material loading; fall back to StandardSurface
    std::string    pngPath;                   // If set, write sRGB-encoded PNG alongside EXR (no tone mapping)
    std::string    previewExrPath;            // If set, write progressive linear EXR preview to this path
    float          exposure          = 0.f;   // EV adjustment for PNG output
    DenoiseOptions denoise;

    // Pixel reconstruction filter — replaces the legacy box-1.0 jitter.
    // pixelFilterRadius == 0 means "use the type's default radius".
    PixelFilterType pixelFilter      = PixelFilterType::Mitchell;
    float           pixelFilterRadius = 0.f;

    // Hair ribbon tessellation quality — quads per cubic Bézier segment.
    // Higher = smoother ribbons, more triangles, slower commit but same render cost.
    // Applied to both the CPU (CurveBrute) and GPU (MetalAccelStructure) pipelines.
    int            hairTessSteps     = 4;

    // Photon mapping parameters (IntegratorType::PhotonMap only).
    int   numPhotons    = 500'000; // Total photons emitted per frame
    float photonRadius  = 0.1f;   // Search radius for density estimate (world units)

    // Debug: when >= 0, primary rays that hit a different mesh return black.
    // Useful for isolating which pixels belong to a given mesh, and for testing
    // that mesh's shading in isolation.  Indirect bounces are unaffected.
    int32_t        debugMeshID       = -1;

    // Highlight: when >= 0, renders the target mesh as flat red and dims all
    // other geometry to 20% brightness so you can visually locate a mesh by ID.
    int32_t        highlightMeshID   = -1;
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

    // Optional display driver — must be set before render() is called.
    // Non-owning; caller is responsible for lifetime.
    void setDisplayDriver(IDisplayDriver* driver) { m_displayDriver = driver; }

    // Render control — safe to call from any thread (e.g. display driver callbacks).
    void requestCancel() { m_cancelRequested.store(true); }
    void requestPause()  { m_pauseRequested.store(true);  }
    void requestResume() {
        m_pauseRequested.store(false);
        m_pauseCV.notify_all();
    }
    void requestCrop(uint32_t cx, uint32_t cy, uint32_t cw, uint32_t ch);
    void requestClearCrop();

private:
    void buildCornellBox();
    void buildCornellBoxSSS();
    void buildAccelStructure();
    void scheduleTiles(std::vector<TileRequest>& tiles) const;
    // Load Alembic curves into m_curvePool (no-op if curvesPath is empty or
    // ANACAPA_ENABLE_ALEMBIC is not set)
    void appendAlembicCurves_();
    // Load Alembic IPoints into m_haloPool (no-op if particlesPath is empty)
    void appendAlembicParticles_();

    IDisplayDriver*                        m_displayDriver = nullptr;
    std::atomic<bool>                      m_cancelRequested{false};
    std::atomic<bool>                      m_pauseRequested{false};
    std::mutex                             m_pauseMtx;
    std::condition_variable                m_pauseCV;
    std::atomic<bool>                      m_cropActive{false};
    std::atomic<bool>                      m_cropCleared{false};
    std::mutex                             m_cropMtx;
    uint32_t                               m_cropX{0}, m_cropY{0}, m_cropW{0}, m_cropH{0};
    RenderSettings                        m_settings;
    GeometryPool                          m_geomPool;
    CurvePool                             m_curvePool;
    HaloPool                              m_haloPool;
    SceneView                             m_scene;
    std::optional<Camera>                 m_camera;   // set by USD loader if present
    std::unique_ptr<IAccelerationStructure> m_accel;
    std::unique_ptr<HaloAccel>            m_haloAccel;
    std::unique_ptr<Film>                 m_film;
    std::unique_ptr<IIntegrator>          m_integrator;
    std::unique_ptr<ISampler>             m_baseSampler;
    std::unique_ptr<ThreadPool>           m_threadPool;
    std::unique_ptr<PixelFilter>          m_pixelFilter;

    // Shutter interval from the scene file (overridden by RenderSettings if set)
    float m_sceneShutterOpen  = 0.f;
    float m_sceneShutterClose = 0.f;

    // Owned materials and lights (scene takes non-owning pointers)
    std::vector<std::unique_ptr<IMaterial>> m_materials;
    std::vector<std::unique_ptr<ILight>>    m_lights;

    // USD prim path → m_materials index (populated from LoadedScene after USD load).
    // Used to assign USD materials to Alembic hair strands by material path.
    std::unordered_map<std::string, uint32_t> m_materialPathIndex;

    // Active tile queue — non-null only while runTiles is executing.
    // requestCrop() grabs this mutex and calls reprioritize() on the queue.
    struct TileQueue;
    std::mutex    m_queueMtx;
    TileQueue*    m_activeTileQueue = nullptr;
};

} // namespace anacapa
