#include "RenderSession.h"
#include "../accel/BVHBackend.h"
#include "../integrator/PathIntegrator.h"
#include "../integrator/BDPTIntegrator.h"
#include "../sampling/HaltonSampler.h"
#include "../shading/Lambertian.h"
#include "../shading/lights/AreaLight.h"
#include "../shading/lights/DirectionalLight.h"
#include "../shading/lights/DomeLight.h"

#ifdef ANACAPA_ENABLE_USD
#  include "../scene/usd/USDLoader.h"
#endif

#include <spdlog/spdlog.h>
#include <cassert>
#include <chrono>
#include <cmath>

namespace anacapa {

// Rotate a point around the Y axis by `degrees`, about the given pivot.
static Vec3f rotateY(Vec3f p, Vec3f pivot, float degrees) {
    float rad = degrees * 3.14159265f / 180.f;
    float c   = std::cos(rad), s = std::sin(rad);
    Vec3f q   = p - pivot;
    return pivot + Vec3f{ c*q.x + s*q.z, q.y, -s*q.x + c*q.z };
}

// Compute the axis-aligned bounding box of all mesh positions in the pool.
static BBox3f computeSceneBounds(const GeometryPool& pool) {
    BBox3f bounds;
    for (size_t i = 0; i < pool.numMeshes(); ++i)
        for (const Vec3f& p : pool.mesh(i).positions)
            bounds.expand(p);
    return bounds;
}

RenderSession::RenderSession(RenderSettings settings)
    : m_settings(std::move(settings))
{}

// ---------------------------------------------------------------------------
// loadScene — dispatch to USD loader or built-in Cornell box
// ---------------------------------------------------------------------------
void RenderSession::loadScene() {
#ifdef ANACAPA_ENABLE_USD
    if (!m_settings.scenePath.empty()) {
        LoadedScene loaded = loadUSD(m_settings.scenePath,
                                      m_settings.imageWidth,
                                      m_settings.imageHeight,
                                      m_settings.cameraPath);
        if (!loaded.valid) {
            spdlog::error("Aborting: could not open scene '{}'",
                          m_settings.scenePath);
            std::exit(1);
        }
        m_geomPool  = std::move(loaded.geomPool);
        m_scene     = std::move(loaded.sceneView);
        m_camera    = std::move(loaded.camera);
        m_materials = std::move(loaded.materials);
        m_lights    = std::move(loaded.lights);
        m_scene.camera = m_camera;
        m_scene.accel  = nullptr;

        // ---- Auto-camera: frame the scene if no camera was found ------------
        if (!m_scene.camera) {
            BBox3f bounds = computeSceneBounds(m_geomPool);
            if (bounds.valid()) {
                Vec3f center   = bounds.centroid();
                float halfDiag = bounds.diagonal().length() * 0.5f;
                float fovY     = 45.f;
                float fovRad   = fovY * 3.14159265f / 180.f;
                float dist     = halfDiag / std::tan(fovRad * 0.5f);
                m_scene.camera = Camera::makePinhole(
                    center + Vec3f{0.f, 0.f, -(dist + halfDiag)},
                    center,
                    {0.f, 1.f, 0.f},
                    fovY,
                    m_settings.imageWidth, m_settings.imageHeight);
                spdlog::info("No camera in scene — auto-camera: "
                             "center=({:.2f},{:.2f},{:.2f}) fovY=45 dist={:.2f}",
                             center.x, center.y, center.z, dist);
            }
        }

        // ---- Auto-sun: add a headlamp if the scene has no lights -----------
        // The light is fixed to the camera and points in the camera's forward
        // direction so every surface in the frustum is illuminated.
        if (m_scene.lights.empty()) {
            BBox3f bounds = computeSceneBounds(m_geomPool);
            Vec3f  center = bounds.valid() ? bounds.centroid() : Vec3f{};
            float  radius = bounds.valid()
                          ? bounds.diagonal().length() * 0.5f : 1.f;

            // Derive camera forward from the stored camera (auto or USD).
            Vec3f forward = {0.f, 0.f, 1.f};   // fallback
            if (m_scene.camera) {
                const Camera& cam = *m_scene.camera;
                forward = safeNormalize(
                    cam.lowerLeftCorner
                    + cam.horizontal * 0.5f
                    + cam.vertical   * 0.5f
                    - cam.origin);
            }

            auto sun = std::make_unique<DirectionalLight>(
                forward,                        // dirToLight = camera forward
                Spectrum{8.f, 8.f, 7.5f},
                radius, center);
            m_scene.lights.push_back(sun.get());
            m_lights.push_back(std::move(sun));
            spdlog::info("No lights in scene — auto headlamp: "
                         "dir=({:.2f},{:.2f},{:.2f})",
                         forward.x, forward.y, forward.z);
        }

        return;
    }
#endif
    if (!m_settings.scenePath.empty())
        spdlog::warn("--scene requires ANACAPA_ENABLE_USD; falling back to Cornell box");
    buildCornellBox();
}

// ---------------------------------------------------------------------------
// Cornell Box — built-in fallback scene
//
// Coordinate system:
//   x ∈ [-1, 1]  (left wall = -1 red, right wall = +1 green)
//   y ∈ [-1, 1]  (floor = -1, ceiling = +1)
//   z ∈ [ 0, 2]  (open face at z=0 toward camera, back wall at z=2)
//
// All quads wound CCW when viewed from inside — normals face inward.
// ---------------------------------------------------------------------------
void RenderSession::buildCornellBox() {
    // -- Materials --
    auto white = std::make_unique<LambertianMaterial>(Spectrum{0.73f, 0.73f, 0.73f});
    auto red   = std::make_unique<LambertianMaterial>(Spectrum{0.65f, 0.05f, 0.05f});
    auto green = std::make_unique<LambertianMaterial>(Spectrum{0.12f, 0.45f, 0.15f});
    auto light = std::make_unique<EmissiveMaterial>(
        Spectrum{0.f}, Spectrum{15.f, 15.f, 15.f});

    const IMaterial* pWhite = white.get();
    const IMaterial* pRed   = red.get();
    const IMaterial* pGreen = green.get();
    const IMaterial* pLight = light.get();

    m_materials.push_back(std::move(white));
    m_materials.push_back(std::move(red));
    m_materials.push_back(std::move(green));
    m_materials.push_back(std::move(light));

    // addQuad: vertices in CCW order viewed from the direction the normal faces.
    // Explicit inward normal passed so there's no winding ambiguity.
    auto addQuad = [&](Vec3f v0, Vec3f v1, Vec3f v2, Vec3f v3,
                       Vec3f inwardNormal,
                       const IMaterial* mat) -> uint32_t {
        MeshDesc mesh;
        mesh.positions = {v0, v1, v2, v3};
        Vec3f n = safeNormalize(inwardNormal);
        mesh.normals = {n, n, n, n};
        mesh.uvs     = {{0,0},{1,0},{1,1},{0,1}};
        mesh.indices = {0,1,2, 0,2,3};
        uint32_t id  = m_geomPool.addMesh(std::move(mesh));
        (void)mat;
        return id;
    };


    // -- Room walls --
    // Floor: y=-1, normal +y
    uint32_t floorID = addQuad({-1,-1,0},{1,-1,0},{1,-1,2},{-1,-1,2}, {0, 1,0}, pWhite);
    // Ceiling: y=+1, normal -y
    uint32_t ceilID  = addQuad({-1, 1,2},{1, 1,2},{1, 1,0},{-1, 1,0}, {0,-1,0}, pWhite);
    // Back wall: z=2, normal -z
    uint32_t backID  = addQuad({-1,-1,2},{-1,1,2},{1,1,2},{1,-1,2},   {0,0,-1}, pWhite);
    // Left wall (red): x=-1, normal +x
    uint32_t leftID  = addQuad({-1,-1,2},{-1,1,2},{-1,1,0},{-1,-1,0}, { 1,0,0}, pRed);
    // Right wall (green): x=+1, normal -x
    uint32_t rightID = addQuad({1,-1,0},{1,1,0},{1,1,2},{1,-1,2},     {-1,0,0}, pGreen);
    // Ceiling light patch: slightly below y=1, normal -y (faces down into room)
    uint32_t lightID = addQuad({-0.25f,0.999f,0.85f},{0.25f,0.999f,0.85f},
                                {0.25f,0.999f,1.15f},{-0.25f,0.999f,1.15f},
                                {0,-1,0}, pLight);

    // -- Interior blocks (rotated around Y axis) --
    // addRotatedBox: builds an axis-aligned box then rotates all vertices
    // around the box's center (at floor level) by `yDeg` degrees.
    auto addRotatedBox = [&](Vec3f lo, Vec3f hi, float yDeg, const IMaterial* mat) {
        // Pivot at the horizontal center of the box, on the floor
        Vec3f pivot = { (lo.x + hi.x) * 0.5f, lo.y, (lo.z + hi.z) * 0.5f };

        auto rv = [&](Vec3f p) { return rotateY(p, pivot, yDeg); };

        float x0=lo.x, x1=hi.x, y0=lo.y, y1=hi.y, z0=lo.z, z1=hi.z;

        // Each face: 4 rotated verts + pre-rotated inward normal
        auto face = [&](Vec3f a, Vec3f b, Vec3f c, Vec3f d, Vec3f n) {
            Vec3f rn = rotateY(pivot + n, pivot, yDeg) - pivot; // rotate normal too
            uint32_t id = addQuad(rv(a), rv(b), rv(c), rv(d), rn, mat);
            if (id >= m_scene.materials.size())
                m_scene.materials.resize(id + 1, nullptr);
            m_scene.materials[id] = mat;
        };

        face({x0,y0,z0},{x1,y0,z0},{x1,y0,z1},{x0,y0,z1}, { 0, 1, 0}); // floor
        face({x0,y1,z1},{x1,y1,z1},{x1,y1,z0},{x0,y1,z0}, { 0,-1, 0}); // top
        face({x0,y0,z0},{x0,y1,z0},{x1,y1,z0},{x1,y0,z0}, { 0, 0,-1}); // front
        face({x1,y0,z1},{x1,y1,z1},{x0,y1,z1},{x0,y0,z1}, { 0, 0, 1}); // back
        face({x0,y0,z1},{x0,y1,z1},{x0,y1,z0},{x0,y0,z0}, { 1, 0, 0}); // left
        face({x1,y0,z0},{x1,y1,z0},{x1,y1,z1},{x1,y0,z1}, {-1, 0, 0}); // right
    };

    // Tall block — right side, rotated 15° CCW (classic Cornell box angle)
    addRotatedBox({ 0.10f, -1.0f, 0.55f}, { 0.65f,  0.0f, 1.10f}, -15.f, pWhite);

    // Short block — left side, rotated 18° CW (classic Cornell box angle)
    addRotatedBox({-0.65f, -1.0f, 0.80f}, {-0.10f, -0.5f, 1.35f},  18.f, pWhite);

    // -- Register all materials now that all geometry has been added --
    // (Must happen after all addQuad/addRotatedBox calls so meshID space is final)
    m_scene.materials.resize(m_geomPool.numMeshes(), nullptr);
    m_scene.materials[floorID] = pWhite;
    m_scene.materials[ceilID]  = pWhite;
    m_scene.materials[backID]  = pWhite;
    m_scene.materials[leftID]  = pRed;
    m_scene.materials[rightID] = pGreen;
    m_scene.materials[lightID] = pLight;
    // addRotatedBox registered its own face materials inline via m_scene.materials[id]

    // -- Area light (ILight for direct sampling) --
    auto areaLight = std::make_unique<AreaLight>(
        Vec3f{0.f, 0.999f, 1.0f},    // center (matches the ceiling patch)
        Vec3f{0.25f, 0.f, 0.f},       // u half-extent
        Vec3f{0.f,  0.f, 0.15f},      // v half-extent
        Spectrum{15.f, 15.f, 15.f}
    );
    m_scene.lights.push_back(areaLight.get());
    m_lights.push_back(std::move(areaLight));

    m_scene.envRadiance = {};
}

// ---------------------------------------------------------------------------
// Build BVH from the geometry pool
// ---------------------------------------------------------------------------
void RenderSession::buildAccelStructure() {
    auto bvh = std::make_unique<BVHBackend>(m_geomPool);
    bvh->commit();
    m_accel = std::move(bvh);
    m_scene.accel = m_accel.get();
}

// ---------------------------------------------------------------------------
// Partition film into tiles
// ---------------------------------------------------------------------------
void RenderSession::scheduleTiles(std::vector<TileRequest>& tiles) const {
    uint32_t ts = m_settings.tileSize;
    uint32_t W  = m_settings.imageWidth;
    uint32_t H  = m_settings.imageHeight;

    for (uint32_t y = 0; y < H; y += ts) {
        for (uint32_t x = 0; x < W; x += ts) {
            TileRequest t;
            t.x0          = x;
            t.y0          = y;
            t.width       = std::min(ts, W - x);
            t.height      = std::min(ts, H - y);
            t.sampleStart = 0;
            t.sampleCount = m_settings.samplesPerPixel;
            tiles.push_back(t);
        }
    }
}

// ---------------------------------------------------------------------------
// Main render loop
// ---------------------------------------------------------------------------
void RenderSession::render() {
    buildAccelStructure();

    // DomeLight (HDRI environment) — set up before prepare() so the integrator
    // can see envLight during subpath tracing.
    if (!m_settings.envPath.empty()) {
        BBox3f bounds = computeSceneBounds(m_geomPool);
        float radius  = bounds.valid()
            ? bounds.diagonal().length() * 0.5f * 2.f : 10.f;
        Vec3f center  = bounds.valid() ? bounds.centroid() : Vec3f{};

        auto dome = std::make_unique<DomeLight>(
            m_settings.envPath,
            m_settings.envIntensity,
            radius, center);
        m_scene.envLight = dome.get();
        m_scene.lights.push_back(dome.get());
        m_lights.push_back(std::move(dome));
        spdlog::info("DomeLight: '{}' intensity={:.2f}",
                     m_settings.envPath, m_settings.envIntensity);
    }

    m_film        = std::make_unique<Film>(m_settings.imageWidth,
                                           m_settings.imageHeight);
    if (m_settings.integrator == IntegratorType::BDPT)
        m_integrator = std::make_unique<BDPTIntegrator>(m_settings.maxDepth);
    else
        m_integrator = std::make_unique<PathIntegrator>(m_settings.maxDepth);
    m_baseSampler = std::make_unique<HaltonSampler>(m_settings.samplesPerPixel);
    m_threadPool  = std::make_unique<ThreadPool>(m_settings.numThreads);

    m_integrator->prepare(m_scene);

    std::vector<TileRequest> tiles;
    scheduleTiles(tiles);

    uint32_t totalTiles = static_cast<uint32_t>(tiles.size());
    spdlog::info("Rendering {}x{} @ {} spp — {} tiles on {} threads",
        m_settings.imageWidth, m_settings.imageHeight,
        m_settings.samplesPerPixel, totalTiles,
        m_threadPool->numThreads());

    auto t0 = std::chrono::steady_clock::now();

    std::atomic<uint32_t> tilesCompleted{0};

    m_threadPool->parallelFor(totalTiles, [&](uint32_t tileIdx) {
        const TileRequest& tile = tiles[tileIdx];

        // Each worker gets its own sampler clone (independent state)
        auto sampler = m_baseSampler->clone();

        TileBuffer localTile(tile.x0, tile.y0, tile.width, tile.height);
        m_integrator->renderTile(m_scene, tile,
                                  m_settings.imageWidth, m_settings.imageHeight,
                                  *sampler, localTile);
        m_film->mergeTile(localTile);

        uint32_t done = tilesCompleted.fetch_add(1, std::memory_order_relaxed) + 1;
        if (done % 16 == 0 || done == totalTiles)
            spdlog::info("  {}/{} tiles", done, totalTiles);
    });

    auto t1  = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    spdlog::info("Render complete in {:.1f} ms", ms);

    if (m_settings.denoise.enabled)
        m_film->denoise();

    if (!m_film->writeEXR(m_settings.outputPath, m_settings.denoise))
        spdlog::error("Failed to write {}", m_settings.outputPath);
    else
        spdlog::info("Written: {}", m_settings.outputPath);
}

} // namespace anacapa
