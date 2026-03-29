#include "RenderSession.h"
#include "../accel/BVHBackend.h"
#include "../integrator/PathIntegrator.h"
#include "../sampling/HaltonSampler.h"
#include "../shading/Lambertian.h"
#include "../shading/lights/AreaLight.h"

#include <spdlog/spdlog.h>
#include <cassert>
#include <chrono>

namespace anacapa {

RenderSession::RenderSession(RenderSettings settings)
    : m_settings(std::move(settings))
{}

// ---------------------------------------------------------------------------
// Cornell Box — hardcoded for Phase 1
// Classic Cornell box: white walls, red left, green right, white light above.
// ---------------------------------------------------------------------------
void RenderSession::buildCornellBox() {
    // -- Materials --
    auto white  = std::make_unique<LambertianMaterial>(Spectrum{0.73f, 0.73f, 0.73f});
    auto red    = std::make_unique<LambertianMaterial>(Spectrum{0.65f, 0.05f, 0.05f});
    auto green  = std::make_unique<LambertianMaterial>(Spectrum{0.12f, 0.45f, 0.15f});
    auto light  = std::make_unique<EmissiveMaterial>(
        Spectrum{0.f}, Spectrum{15.f, 15.f, 15.f});

    const IMaterial* pWhite = white.get();
    const IMaterial* pRed   = red.get();
    const IMaterial* pGreen = green.get();
    const IMaterial* pLight = light.get();

    m_materials.push_back(std::move(white));
    m_materials.push_back(std::move(red));
    m_materials.push_back(std::move(green));
    m_materials.push_back(std::move(light));

    // Helper: add a quad as two triangles
    auto addQuad = [&](Vec3f v0, Vec3f v1, Vec3f v2, Vec3f v3,
                       const IMaterial* mat) {
        MeshDesc mesh;
        mesh.positions = {v0, v1, v2, v3};
        Vec3f e1 = v1 - v0, e2 = v2 - v0;
        Vec3f n  = safeNormalize(cross(e1, e2));
        mesh.normals   = {n, n, n, n};
        mesh.uvs       = {{0,0},{1,0},{1,1},{0,1}};
        mesh.indices   = {0,1,2, 0,2,3};

        uint32_t id = m_geomPool.addMesh(std::move(mesh));
        // Map material pointer — we use meshID as material index
        (void)mat;
        return id;
    };

    // Cornell box spans [-1,1] in x and y, [0,2] in z (camera looks down -z)
    // Floor
    uint32_t floorID  = addQuad({-1,-1, 0},{1,-1, 0},{1,-1,2},{-1,-1,2}, pWhite);
    // Ceiling
    uint32_t ceilID   = addQuad({-1, 1, 0},{-1,1,2},{1,1,2},{1,1,0},     pWhite);
    // Back wall
    uint32_t backID   = addQuad({-1,-1,2},{1,-1,2},{1,1,2},{-1,1,2},     pWhite);
    // Left wall (red)
    uint32_t leftID   = addQuad({-1,-1,0},{-1,-1,2},{-1,1,2},{-1,1,0},   pRed);
    // Right wall (green)
    uint32_t rightID  = addQuad({1,-1,0},{1,1,0},{1,1,2},{1,-1,2},       pGreen);
    // Light quad (emissive patch on ceiling)
    uint32_t lightID  = addQuad({-0.3f,0.99f,0.9f},{0.3f,0.99f,0.9f},
                                 {0.3f,0.99f,1.1f},{-0.3f,0.99f,1.1f},   pLight);

    // -- Scene materials array (indexed by meshID) --
    m_scene.materials.resize(m_geomPool.numMeshes(), nullptr);
    m_scene.materials[floorID] = pWhite;
    m_scene.materials[ceilID]  = pWhite;
    m_scene.materials[backID]  = pWhite;
    m_scene.materials[leftID]  = pRed;
    m_scene.materials[rightID] = pGreen;
    m_scene.materials[lightID] = pLight;

    // -- Area light --
    auto areaLight = std::make_unique<AreaLight>(
        Vec3f{0.f, 0.99f, 1.0f},   // center
        Vec3f{0.3f, 0.f, 0.f},      // u half-extent
        Vec3f{0.f, 0.f, 0.1f},      // v half-extent
        Spectrum{15.f, 15.f, 15.f}
    );
    m_scene.lights.push_back(areaLight.get());
    m_lights.push_back(std::move(areaLight));

    m_scene.envRadiance = {0.f, 0.f, 0.f};
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

    m_film        = std::make_unique<Film>(m_settings.imageWidth,
                                           m_settings.imageHeight);
    m_integrator  = std::make_unique<PathIntegrator>(m_settings.maxDepth);
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

    if (!m_film->writeEXR(m_settings.outputPath))
        spdlog::error("Failed to write {}", m_settings.outputPath);
    else
        spdlog::info("Written: {}", m_settings.outputPath);
}

} // namespace anacapa
