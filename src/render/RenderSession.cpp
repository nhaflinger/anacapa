#include "RenderSession.h"
#include <anacapa/render/FileDisplayDriver.h>
#include "../accel/BVHBackend.h"
#include "../accel/CurveBrute.h"
#include "../integrator/PathIntegrator.h"
#include "../integrator/BDPTIntegrator.h"
#include "../integrator/PhotonMapIntegrator.h"
#ifdef ANACAPA_ENABLE_METAL
#  include "../gpu/metal/MetalPathIntegrator.h"
#endif
#ifdef ANACAPA_ENABLE_CUDA
#  include "../gpu/cuda/CudaPathIntegrator.h"
#endif
#include "../sampling/HaltonSampler.h"
#include "../shading/Lambertian.h"
#include "../shading/MarschnerHair.h"
#include "../shading/lights/AreaLight.h"
#include "../shading/lights/DirectionalLight.h"
#include "../shading/lights/DomeLight.h"

#ifdef ANACAPA_ENABLE_USD
#  include "../scene/usd/USDLoader.h"
#endif

#ifdef ANACAPA_ENABLE_ALEMBIC
#  include "../scene/alembic/AlembicLoader.h"
#  include "../scene/MatAssignLoader.h"
#endif

#include <spdlog/spdlog.h>
#include <cassert>
#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <limits>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <thread>

namespace anacapa {

// ---------------------------------------------------------------------------
// TileQueue — mutex-protected deque with crop-window support.
// Worker threads pop() from pending; setCrop() moves non-intersecting tiles
// to the deferred deque so they are skipped until CLEAR_CROP is received.
// ---------------------------------------------------------------------------
struct RenderSession::TileQueue {
    std::deque<TileRequest> pending;
    std::deque<TileRequest> deferred;
    std::mutex              mtx;

    void push(const TileRequest& t) { pending.push_back(t); }

    std::optional<TileRequest> pop() {
        std::lock_guard<std::mutex> lk(mtx);
        if (pending.empty()) return std::nullopt;
        TileRequest t = pending.front();
        pending.pop_front();
        return t;
    }

    // Partition pending tiles: those that intersect [cx,cy,cw,ch] stay in
    // pending; all others move to deferred. Call before dispatching workers.
    void setCrop(uint32_t cx, uint32_t cy, uint32_t cw, uint32_t ch) {
        std::lock_guard<std::mutex> lk(mtx);
        std::deque<TileRequest> cropTiles;
        for (auto& t : pending) {
            bool hit = (t.x0 < cx + cw && t.x0 + t.width  > cx &&
                        t.y0 < cy + ch && t.y0 + t.height > cy);
            (hit ? cropTiles : deferred).push_back(t);
        }
        pending = std::move(cropTiles);
    }

    std::vector<TileRequest> takeDeferredTiles() {
        std::lock_guard<std::mutex> lk(mtx);
        std::vector<TileRequest> out(deferred.begin(), deferred.end());
        deferred.clear();
        return out;
    }
};

void RenderSession::requestCrop(uint32_t cx, uint32_t cy, uint32_t cw, uint32_t ch) {
    {
        std::lock_guard<std::mutex> lk(m_cropMtx);
        m_cropX = cx; m_cropY = cy; m_cropW = cw; m_cropH = ch;
    }
    m_cropCleared.store(false);  // reset so the deferred wait detects a fresh CLEAR_CROP
    m_cropActive.store(true);
    // If a queue is already running, defer its remaining non-crop tiles now.
    std::lock_guard<std::mutex> lk(m_queueMtx);
    if (m_activeTileQueue)
        m_activeTileQueue->setCrop(cx, cy, cw, ch);
}

void RenderSession::requestClearCrop() {
    m_cropActive.store(false);
    m_cropCleared.store(true);
    m_pauseCV.notify_all();
}

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
        double renderFrame = m_settings.frameSet
                             ? static_cast<double>(m_settings.frameNumber)
                             : std::numeric_limits<double>::quiet_NaN();
        if (m_settings.frameSet)
            spdlog::info("RenderSession: rendering frame {} (--frame was provided)",
                         m_settings.frameNumber);
        else
            spdlog::info("RenderSession: no --frame specified — will use stage startTime");
        LoadedScene loaded = loadUSD(m_settings.scenePath,
                                      m_settings.imageWidth,
                                      m_settings.imageHeight,
                                      m_settings.cameraPath,
                                      renderFrame,
                                      m_settings.shutterOpen,
                                      m_settings.shutterClose);
        if (!loaded.valid) {
            spdlog::error("Aborting: could not open scene '{}'",
                          m_settings.scenePath);
            std::exit(1);
        }
        m_geomPool           = std::move(loaded.geomPool);
        m_curvePool          = std::move(loaded.curvePool);
        m_scene              = std::move(loaded.sceneView);
        m_camera             = std::move(loaded.camera);
        m_materials          = std::move(loaded.materials);
        m_lights             = std::move(loaded.lights);
        m_materialPathIndex  = std::move(loaded.materialPathIndex);
        m_sceneShutterOpen   = loaded.shutterOpen;   // unused after simplification below
        m_sceneShutterClose  = loaded.shutterClose;  // unused after simplification below
        m_scene.camera = m_camera;
        m_scene.accel  = nullptr;

        // ---- Override materials: replace all with white Lambertian ----------
        if (m_settings.overrideMaterials) {
            spdlog::info("--override-materials: replacing {} material(s) with white Lambertian",
                         m_materials.size());
            auto white = std::make_unique<LambertianMaterial>(Spectrum{0.8f, 0.8f, 0.8f});
            const IMaterial* pWhite = white.get();
            m_materials.clear();
            m_materials.push_back(std::move(white));
            // Point every mesh at the single white material
            std::fill(m_scene.materials.begin(), m_scene.materials.end(), pWhite);
        }

        // ---- Command-line DoF override --------------------------------------
        // If --fstop and --focus-distance are both provided they take priority
        // over whatever the USD camera specified. Applied after camera is set
        // so it works for both USD cameras and the auto-camera below.
        if (m_settings.fStop > 0.f && m_settings.focusDistance > 0.f
                && m_scene.camera) {
            Camera& cam = *m_scene.camera;
            // Use the physical focal length stored on the camera (set by USDLoader
            // from the raw USD focalLength attribute in scene units).  Fall back to
            // a frustum-derived approximation only if the camera has no stored value
            // (e.g. a synthetic auto-camera with no USD backing).
            float focalLen = (cam.focalLength > 0.f)
                             ? cam.focalLength
                             : cam.vertical.length() * 0.5f;  // fallback: frustum half-height at unit depth
            cam.apertureRadius = focalLen / (2.f * m_settings.fStop);
            cam.focalDistance  = m_settings.focusDistance;
            spdlog::info("DoF override: fStop={:.1f} focusDist={:.3f} focalLen={:.4f} apertureR={:.4f}",
                         m_settings.fStop, m_settings.focusDistance,
                         focalLen, cam.apertureRadius);
        }

        // ---- Motion blur shutter -------------------------------------------
        // The USDLoader has already used the provided frame/shutter params to
        // collect motion keys normalized to [0, 1].  loaded.shutterClose=1 when
        // motion blur is active, 0 when disabled.  Apply directly to the camera.
        if (m_scene.camera && loaded.shutterClose > loaded.shutterOpen) {
            Camera& cam = *m_scene.camera;
            cam.shutterOpen  = loaded.shutterOpen;
            cam.shutterClose = loaded.shutterClose;
            spdlog::info("Motion blur: shutter [{:.3f}, {:.3f}]",
                         loaded.shutterOpen, loaded.shutterClose);
        }

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

        // ---- Light angle: give directional lights an angular extent for soft shadows --
        if (m_settings.lightAngle > 0.f) {
            for (auto& l : m_lights) {
                if (auto* dl = dynamic_cast<DirectionalLight*>(l.get()))
                    dl->setHalfAngleDeg(m_settings.lightAngle);
            }

        }

        // ---- Override lights: replace all USD lights with a diagnostic light --
        if (m_settings.overrideLights &&
                (!m_scene.lights.empty() || m_scene.envLight != nullptr)) {
            spdlog::info("--override-lights: discarding {} USD light(s) + envLight, "
                         "installing white directional light", m_scene.lights.size());
            m_scene.lights.clear();
            m_scene.envLight = nullptr;
            m_lights.clear();
        }

        // ---- Auto-sun: add a headlamp if the scene has no punctual lights --
        // Also fires when override-lights cleared everything (envLight only scenes).
        if (m_scene.lights.empty() && m_scene.envLight == nullptr) {
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
                -forward,                       // dirToLight: FROM surface TOWARD light (camera is behind forward)
                Spectrum{8.f, 8.f, 7.5f},
                radius, center);
            m_scene.lights.push_back(sun.get());
            m_lights.push_back(std::move(sun));
            spdlog::info("No lights in scene — auto headlamp: "
                         "dir=({:.2f},{:.2f},{:.2f})",
                         forward.x, forward.y, forward.z);
        }

        appendAlembicCurves_();
        return;
    }
#endif
    if (!m_settings.scenePath.empty())
        spdlog::warn("--scene requires ANACAPA_ENABLE_USD; falling back to Cornell box");
    buildCornellBox();
    appendAlembicCurves_();
}

// ---------------------------------------------------------------------------
// appendAlembicCurves_ — load hair/fur from an Alembic file (if requested)
// ---------------------------------------------------------------------------
void RenderSession::appendAlembicCurves_() {
#ifdef ANACAPA_ENABLE_ALEMBIC
    if (m_settings.curvesPath.empty()) return;

    AlembicCurveOptions opts;
    // baseMaterialIndex: index into scene.materials (raw-pointer array indexed
    // by si.meshID).  Used as fallback when a USD mat path lookup fails; the
    // default MarschnerHairMaterial created by loadAlembicCurves is pushed at
    // this position after the load.
    opts.baseMaterialIndex = static_cast<uint32_t>(m_scene.materials.size());
    opts.motionBlur        = (m_settings.shutterClose > m_settings.shutterOpen);

    const size_t matsBefore = m_materials.size();

    std::vector<AlembicObjectRange> outRanges;
    bool ok = loadAlembicCurves(m_settings.curvesPath, opts,
                                m_curvePool, m_materials, &outRanges);
    if (!ok) {
        spdlog::error("Failed to load Alembic curves from '{}'",
                      m_settings.curvesPath);
        return;
    }

    // Mirror the default MarschnerHairMaterial into the raw-pointer list.
    for (size_t i = matsBefore; i < m_materials.size(); ++i)
        m_scene.materials.push_back(m_materials[i].get());

    // --- Load material assignments ---
    // Start with the auto-discovered sidecar (<abc>.matassign.json), then
    // layer any explicitly requested --matassign files on top.
    std::vector<MatAssignEntry> assignments;

    // Auto-discover: replace .abc extension with .matassign.json
    std::string autoPath = m_settings.curvesPath;
    {
        auto dot = autoPath.rfind('.');
        if (dot != std::string::npos)
            autoPath.replace(dot, std::string::npos, ".matassign.json");
        else
            autoPath += ".matassign.json";
    }
    if (std::ifstream(autoPath).good()) {
        auto f = loadMatAssign(autoPath);
        mergeMatAssign(assignments, f.assignments);
    }

    // Layer explicit --matassign files
    for (const auto& path : m_settings.matassignPaths) {
        auto f = loadMatAssign(path);
        mergeMatAssign(assignments, f.assignments);
    }

    // Build a name → MatAssignEntry lookup for fast per-range matching
    std::unordered_map<std::string, const MatAssignEntry*> assignMap;
    for (const auto& e : assignments)
        assignMap[e.objectName] = &e;

    // Apply per-object material assignments to strand ranges
    for (const auto& range : outRanges) {
        auto it = assignMap.find(range.objectName);
        if (it != assignMap.end()) {
            const MatAssignEntry& entry = *it->second;
            auto mat = buildMaterial(entry);
            uint32_t matIdx = static_cast<uint32_t>(m_scene.materials.size());
            m_scene.materials.push_back(mat.get());
            m_materials.push_back(std::move(mat));
            m_curvePool.setMaterialIndex(range.strandStart,
                                         range.strandStart + range.strandCount,
                                         matIdx);
            spdlog::info("appendAlembicCurves: '{}' → matassign ({})",
                         range.objectName,
                         entry.type == MatAssignType::Marschner ? "marschner" :
                         entry.type == MatAssignType::Chiang    ? "chiang"    :
                         entry.type == MatAssignType::Usd       ? "usd"       : "osl");
        } else {
            spdlog::info("appendAlembicCurves: '{}' not in matassign → default hair",
                         range.objectName);
        }
    }
    if (outRanges.empty()) {
        spdlog::info("appendAlembicCurves: no named objects found — "
                     "all strands use default hair material");
    }
#else
    if (!m_settings.curvesPath.empty())
        spdlog::warn("--curves requires ANACAPA_ENABLE_ALEMBIC; ignoring '{}'",
                     m_settings.curvesPath);
#endif
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
// Build acceleration structure from the geometry and curve pools.
// CurveBrute wraps the triangle BVH and adds brute-force curve intersection.
// When m_curvePool is empty (mesh-only scenes) the curve loop costs nothing.
// ---------------------------------------------------------------------------
void RenderSession::buildAccelStructure() {
    auto accel = std::make_unique<CurveBrute>(m_geomPool, m_curvePool, m_settings.hairTessSteps);
    accel->commit();
    m_accel = std::move(accel);
    m_scene.accel         = m_accel.get();
    m_scene.curvePool     = &m_curvePool;
    m_scene.hairTessSteps = m_settings.hairTessSteps;
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
        // Evict any DomeLight that loadUSD already installed (e.g. a dark USD
        // DomeLight) so NEE doesn't sample a stale envLight alongside the new one.
        if (m_scene.envLight) {
            auto* old = m_scene.envLight;
            m_scene.lights.erase(
                std::remove(m_scene.lights.begin(), m_scene.lights.end(), old),
                m_scene.lights.end());
            m_lights.erase(
                std::remove_if(m_lights.begin(), m_lights.end(),
                    [old](const auto& p){ return p.get() == old; }),
                m_lights.end());
            m_scene.envLight = nullptr;
        }

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
        spdlog::info("DomeLight override: '{}' intensity={:.2f}",
                     m_settings.envPath, m_settings.envIntensity);
    }

    m_film = std::make_unique<Film>(m_settings.imageWidth,
                                    m_settings.imageHeight);

#ifdef ANACAPA_ENABLE_METAL
    if (m_settings.interactive) {
        auto metalIntegrator = std::make_unique<MetalPathIntegrator>(
            std::string(ANACAPA_METALLIB_PATH));
        if (metalIntegrator->isValid()) {
            spdlog::info("Interactive mode: using Metal GPU backend");
            metalIntegrator->setFireflyClamp(m_settings.fireflyClamp);
            if (m_settings.integrator == IntegratorType::PhotonMap)
                metalIntegrator->setPhotonMap(m_settings.numPhotons, m_settings.photonRadius);
            m_integrator = std::move(metalIntegrator);
        } else {
            spdlog::warn("--interactive: Metal backend unavailable, "
                         "falling back to CPU path tracer");
        }
    }
#endif

#ifdef ANACAPA_ENABLE_CUDA
    if (m_settings.interactive && !m_integrator) {
        auto cudaIntegrator = std::make_unique<CudaPathIntegrator>();
        if (cudaIntegrator->isValid()) {
            spdlog::info("Interactive mode: using CUDA GPU backend");
            cudaIntegrator->setFireflyClamp(m_settings.fireflyClamp);
            if (m_settings.integrator == IntegratorType::PhotonMap)
                cudaIntegrator->setPhotonMap(m_settings.numPhotons, m_settings.photonRadius);
            m_integrator = std::move(cudaIntegrator);
        } else {
            spdlog::warn("--interactive: CUDA backend unavailable, "
                         "falling back to CPU path tracer");
        }
    }
#endif

    if (m_settings.interactive && !m_integrator)
        spdlog::warn("--interactive: GPU backend unavailable, "
                     "using CPU path tracer");

    if (!m_integrator) {
        if (m_settings.integrator == IntegratorType::BDPT)
            m_integrator = std::make_unique<BDPTIntegrator>(m_settings.maxDepth,
                                                               m_settings.fireflyClamp);
        else if (m_settings.integrator == IntegratorType::PhotonMap)
            m_integrator = std::make_unique<PhotonMapIntegrator>(m_settings.maxDepth,
                                                                    2,
                                                                    m_settings.fireflyClamp,
                                                                    m_settings.numPhotons,
                                                                    m_settings.photonRadius);
        else
            m_integrator = std::make_unique<PathIntegrator>(m_settings.maxDepth,
                                                               2,
                                                               m_settings.fireflyClamp);
    }
    m_baseSampler = std::make_unique<HaltonSampler>(m_settings.samplesPerPixel);
    m_threadPool  = std::make_unique<ThreadPool>(m_settings.numThreads);

    // Pixel reconstruction filter — built once and pointed at the integrator.
    // Lives on the session so its CDF tables stay alive for the whole render.
    m_pixelFilter = std::make_unique<PixelFilter>(m_settings.pixelFilter,
                                                    m_settings.pixelFilterRadius);
    spdlog::info("Pixel filter: {} (radius {:.2f})",
                 pixelFilterName(m_pixelFilter->type()), m_pixelFilter->radius());

    m_integrator->setDebugMeshID(m_settings.debugMeshID);
    m_integrator->setPixelFilter(m_pixelFilter.get());
    m_integrator->prepare(m_scene);

    std::vector<TileRequest> tiles;
    scheduleTiles(tiles);

    uint32_t totalTiles = static_cast<uint32_t>(tiles.size());
    spdlog::info("Rendering {}x{} @ {} spp — {} tiles on {} threads",
        m_settings.imageWidth, m_settings.imageHeight,
        m_settings.samplesPerPixel, totalTiles,
        m_threadPool->numThreads());

    auto t0 = std::chrono::steady_clock::now();

    // Display driver — use the caller-supplied driver if set, otherwise create
    // the default FileDisplayDriver (same preview-thread behaviour as before).
    const bool hasPng = !m_settings.pngPath.empty();
    const bool hasExrOut = [&] {
        const auto& p = m_settings.outputPath;
        if (p.size() < 4) return false;
        std::string ext = p.substr(p.size() - 4);
        for (auto& c : ext) c = static_cast<char>(std::tolower((unsigned char)c));
        return ext == ".exr";
    }();
    std::unique_ptr<FileDisplayDriver> ownedDriver;
    IDisplayDriver* displayDriver = m_displayDriver;
    if (!displayDriver) {
        ownedDriver = std::make_unique<FileDisplayDriver>(
            *m_film,
            hasPng    ? m_settings.pngPath    : std::string{},
            hasExrOut ? m_settings.outputPath : std::string{},
            m_settings.exposure);
        displayDriver = ownedDriver.get();
    }
    displayDriver->imageOpen(m_settings.imageWidth, m_settings.imageHeight);

    // Dedicated command-poll thread — runs at 100 ms intervals throughout the
    // entire render so viewer pause/cancel commands are always responsive,
    // even when the GPU renderFrame() blocks for many seconds at a time.
    std::atomic<bool> pollStop{false};
    std::thread pollThread([&] {
        while (!pollStop.load(std::memory_order_relaxed)) {
            displayDriver->pollCommands();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // When connected to a live viewer, poll until SET_CROP arrives or the
    // deadline passes.  The viewer sends SET_CROP in response to IMAGE_OPEN;
    // the round-trip is <1 ms on a Unix socket, but the viewer may be briefly
    // busy cycling through probe connections (_is_viewer_running checks from
    // the Blender addon), so we poll in a tight loop for up to 200 ms rather
    // than a single fixed sleep.  Skip when the crop is already established
    // from a previous render (m_cropActive persists across renders).
    if (m_displayDriver && !m_cropActive.load(std::memory_order_relaxed)) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        while (!m_cropActive.load(std::memory_order_relaxed) &&
               std::chrono::steady_clock::now() < deadline) {
            displayDriver->pollCommands();
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }

    // Helper: render a list of tiles and merge into Film.
    // Block until resumed or cancelled.
    auto waitIfPaused = [&]() {
        while (m_pauseRequested.load() && !m_cancelRequested.load()) {
            std::unique_lock<std::mutex> lk(m_pauseMtx);
            m_pauseCV.wait_for(lk, std::chrono::milliseconds(100));
        }
    };

    // Returns any tiles that were deferred due to an active crop window.
    // When m_cropActive is set, non-intersecting tiles are moved to the queue's
    // deferred list and returned so the caller can run them after CLEAR_CROP.
    auto runTiles = [&](const std::vector<TileRequest>& tileList,
                        uint32_t totalExpected,
                        std::atomic<uint32_t>& completed) -> std::vector<TileRequest> {
        TileQueue queue;
        for (const auto& t : tileList) queue.push(t);

        if (m_cropActive.load(std::memory_order_relaxed)) {
            uint32_t cx, cy, cw, ch;
            { std::lock_guard<std::mutex> lk(m_cropMtx); cx=m_cropX; cy=m_cropY; cw=m_cropW; ch=m_cropH; }
            queue.setCrop(cx, cy, cw, ch);
        }

        { std::lock_guard<std::mutex> lk(m_queueMtx); m_activeTileQueue = &queue; }

        uint32_t n = static_cast<uint32_t>(queue.pending.size());
        m_threadPool->parallelFor(n, [&](uint32_t) {
            if (m_cancelRequested.load(std::memory_order_relaxed)) return;
            waitIfPaused();
            if (m_cancelRequested.load(std::memory_order_relaxed)) return;

            auto maybeT = queue.pop();
            if (!maybeT) return;
            const TileRequest& tile = *maybeT;

            auto sampler = m_baseSampler->clone();
            TileBuffer localTile(tile.x0, tile.y0, tile.width, tile.height);
            m_integrator->renderTile(m_scene, tile,
                                      m_settings.imageWidth, m_settings.imageHeight,
                                      *sampler, localTile);
            m_film->mergeTile(localTile);

            if (displayDriver) {
                std::vector<float> rgb(tile.width * tile.height * 3);
                m_film->readTile(tile.x0, tile.y0, tile.width, tile.height, rgb.data());
                displayDriver->writeTile(tile.x0, tile.y0, tile.width, tile.height, rgb.data());
            }

            uint32_t done = completed.fetch_add(1, std::memory_order_relaxed) + 1;
            if (done % 16 == 0 || done >= totalExpected)
                spdlog::info("  {}/{} tiles", done, totalExpected);
        });

        { std::lock_guard<std::mutex> lk(m_queueMtx); m_activeTileQueue = nullptr; }
        return queue.takeDeferredTiles();
    };

    std::vector<TileRequest> allDeferred;

    // Push the entire film to the display driver as a single full-image tile.
    // Used after GPU renderFrame() calls which bypass the per-tile loop.
    // Returns false if the render should be aborted (cancelled).
    auto pushFrame = [&]() -> bool {
        if (!displayDriver) return !m_cancelRequested.load();
        uint32_t W = m_settings.imageWidth, H = m_settings.imageHeight;
        std::vector<float> rgb(W * H * 3);
        m_film->readTile(0, 0, W, H, rgb.data());
        displayDriver->writeTile(0, 0, W, H, rgb.data());
        waitIfPaused();
        return !m_cancelRequested.load();
    };

    // Reset GPU accumulation buffer before a new frame.
    auto clearGpuAccum = [&]() {
#ifdef ANACAPA_ENABLE_METAL
        if (auto* mp = dynamic_cast<MetalPathIntegrator*>(m_integrator.get()))
            mp->clearAccum();
#endif
#ifdef ANACAPA_ENABLE_CUDA
        if (auto* cp = dynamic_cast<CudaPathIntegrator*>(m_integrator.get()))
            cp->clearAccum();
#endif
    };

    std::atomic<uint32_t> tilesCompleted{0};

    if (!m_settings.adaptive) {
        // --- Single-pass: try whole-frame GPU path first ---
        clearGpuAccum();
        bool gpuDone = m_integrator->renderFrame(
            m_scene,
            m_settings.imageWidth, m_settings.imageHeight,
            0, m_settings.samplesPerPixel,
            *m_film);
        if (!gpuDone) {
            auto d = runTiles(tiles, totalTiles, tilesCompleted);
            allDeferred.insert(allDeferred.end(), d.begin(), d.end());
        } else if (!pushFrame())
            goto renderDone;
    } else {
        // --- Adaptive two-pass ---
        uint32_t totalSPP = m_settings.samplesPerPixel;
        uint32_t baseSPP  = m_settings.adaptiveBaseSpp > 0
                            ? m_settings.adaptiveBaseSpp
                            : std::min(32u, std::max(16u, totalSPP / 16));
        baseSPP = std::min(baseSPP, totalSPP);
        uint32_t extraSPP = totalSPP - baseSPP;

        // Base pass — use whole-frame GPU dispatch if available (same path as
        // non-adaptive), falling back to per-tile dispatch for CPU integrators.
        spdlog::info("  Adaptive base pass: {} spp", baseSPP);
        clearGpuAccum();
        bool baseGpuDone = m_integrator->renderFrame(
            m_scene,
            m_settings.imageWidth, m_settings.imageHeight,
            0, baseSPP,
            *m_film);
        if (!baseGpuDone) {
            for (auto& t : tiles) { t.sampleStart = 0; t.sampleCount = baseSPP; }
            auto d = runTiles(tiles, totalTiles, tilesCompleted);
            allDeferred.insert(allDeferred.end(), d.begin(), d.end());
        } else {
            tilesCompleted.store(totalTiles);
            if (!pushFrame()) goto renderDone;
        }

        if (extraSPP > 0 && baseGpuDone) {
            // GPU handled the base pass — add remaining samples via another
            // full-frame dispatch (no tile seam artifacts, accumulates on top).
            spdlog::info("  Adaptive GPU refinement: {} extra spp", extraSPP);
            m_integrator->renderFrame(
                m_scene,
                m_settings.imageWidth, m_settings.imageHeight,
                baseSPP, extraSPP,
                *m_film);
            if (!pushFrame()) goto renderDone;
        } else if (extraSPP > 0) {
            // Compute per-tile variance score = average pixel variance in tile
            uint32_t W  = m_settings.imageWidth;
            std::vector<float> tileVar(totalTiles, 0.f);
            for (uint32_t i = 0; i < totalTiles; ++i) {
                const TileRequest& t = tiles[i];
                float sum = 0.f;
                for (uint32_t ty = 0; ty < t.height; ++ty)
                    for (uint32_t tx = 0; tx < t.width; ++tx)
                        sum += m_film->varianceAt(t.x0 + tx, t.y0 + ty);
                tileVar[i] = sum / static_cast<float>(t.width * t.height);
            }

            // Blend uniform + proportional allocation.
            //
            // Pure proportional allocation (even with log-scaling) can leave
            // low-variance tiles with 0 extra samples, making them look worse
            // than a flat render.  Blending with a uniform floor ensures every
            // tile receives at least half the average budget while still
            // concentrating the other half on high-variance regions.
            //
            // Each tile gets: uniformShare + varianceBonus, capped at extraSPP.
            //   uniformShare  = extraSPP / 2  (same for every tile)
            //   varianceBonus = proportional share of the remaining extraSPP/2
            //                   budget, weighted by log(1+variance)
            std::vector<float> logWeight(totalTiles);
            for (uint32_t i = 0; i < totalTiles; ++i)
                logWeight[i] = std::log(1.f + tileVar[i]);
            float totalW = std::accumulate(logWeight.begin(), logWeight.end(), 0.f);

            std::vector<TileRequest> adaptiveTiles;
            {
                uint32_t uniformHalf = extraSPP / 2;
                uint32_t bonusHalf   = extraSPP - uniformHalf;  // remaining half
                std::vector<uint32_t> extraPerTile(totalTiles, 0u);
                for (uint32_t i = 0; i < totalTiles; ++i) {
                    uint32_t bonus = 0;
                    if (totalW > 1e-6f) {
                        float frac = logWeight[i] / totalW;
                        bonus = static_cast<uint32_t>(
                            std::round(frac * static_cast<float>(bonusHalf * totalTiles)));
                    }
                    extraPerTile[i] = std::min(uniformHalf + bonus, extraSPP);
                }

                for (uint32_t i = 0; i < totalTiles; ++i) {
                    if (extraPerTile[i] == 0) continue;
                    TileRequest t = tiles[i];
                    t.sampleStart = baseSPP;
                    t.sampleCount = extraPerTile[i];
                    adaptiveTiles.push_back(t);
                }
            }

            uint32_t nAdaptive = static_cast<uint32_t>(adaptiveTiles.size());
            spdlog::info("  Adaptive pass: {} active tiles (max {} extra spp per tile)",
                         nAdaptive, extraSPP);

            std::atomic<uint32_t> adaptiveCompleted{0};
            uint32_t grandTotal = totalTiles + nAdaptive;
            // shift progress so adaptive logs continue from base-pass count
            adaptiveCompleted.store(totalTiles);
            auto d = runTiles(adaptiveTiles, grandTotal, adaptiveCompleted);
            allDeferred.insert(allDeferred.end(), d.begin(), d.end());
        }
    }

    // --- Deferred tile phase ---
    // If a crop window was active, non-crop tiles were skipped. Wait for the
    // viewer to send CLEAR_CROP (user clicks outside the crop), then render
    // the deferred tiles at their configured sample ranges.
    if (!allDeferred.empty() && !m_cancelRequested.load()) {
        spdlog::info("Crop window active — {} tile(s) deferred. "
                     "Click 'Clear Crop' in viewer to render full image.",
                     allDeferred.size());
        while (!m_cropCleared.load(std::memory_order_relaxed) &&
               !m_cancelRequested.load(std::memory_order_relaxed)) {
            std::unique_lock<std::mutex> lk(m_pauseMtx);
            m_pauseCV.wait_for(lk, std::chrono::milliseconds(200));
        }
        if (!m_cancelRequested.load()) {
            spdlog::info("CLEAR_CROP received — rendering {} deferred tile(s)",
                         allDeferred.size());
            std::atomic<uint32_t> deferredCompleted{0};
            uint32_t deferredTotal = static_cast<uint32_t>(allDeferred.size());
            runTiles(allDeferred, deferredTotal, deferredCompleted);
        }
    }

    renderDone:
    pollStop.store(true);
    pollThread.join();

    // Signal render complete — driver flushes its final preview write
    displayDriver->imageClose();

    auto t1  = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    spdlog::info("Render complete in {:.1f} ms", ms);

    if (m_settings.denoise.enabled)
        m_film->denoise();

    if (!m_film->writeEXR(m_settings.outputPath, m_settings.denoise))
        spdlog::error("Failed to write {}", m_settings.outputPath);
    else
        spdlog::info("Written: {}", m_settings.outputPath);

    if (!m_settings.pngPath.empty()) {
        if (!m_film->writePNG(m_settings.pngPath, m_settings.exposure))
            spdlog::error("Failed to write PNG {}", m_settings.pngPath);
        else
            spdlog::info("Written: {}", m_settings.pngPath);
    }
}

} // namespace anacapa
