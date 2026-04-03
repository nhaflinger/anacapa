#ifdef ANACAPA_ENABLE_USD

#include "USDLoader.h"
#include "../../shading/Lambertian.h"
#include "../../shading/StandardSurface.h"
#include "../../shading/lights/AreaLight.h"
#include "../../shading/lights/DirectionalLight.h"
#include "../../shading/lights/DomeLight.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <spdlog/spdlog.h>
#include <cmath>
#include <unordered_map>

PXR_NAMESPACE_USING_DIRECTIVE

namespace anacapa {

// ---------------------------------------------------------------------------
// Helpers — convert Gf types to anacapa types
// ---------------------------------------------------------------------------
static Vec3f toVec3f(const GfVec3f& v) { return {v[0], v[1], v[2]}; }
static Vec3f toVec3f(const GfVec3d& v) {
    return {static_cast<float>(v[0]),
            static_cast<float>(v[1]),
            static_cast<float>(v[2])};
}

// Apply a 4x4 world transform to a point using GfMatrix4d::Transform
// (USD uses row-vector convention: p_world = m.Transform(p_local))
static Vec3f transformPoint(const GfMatrix4d& m, const GfVec3d& p) {
    GfVec3d r = m.Transform(p);
    return { static_cast<float>(r[0]),
             static_cast<float>(r[1]),
             static_cast<float>(r[2]) };
}

// Apply a 4x4 world transform to a normal using inverse-transpose
static Vec3f transformNormal(const GfMatrix4d& m, const GfVec3f& n) {
    GfVec3d nd(n[0], n[1], n[2]);
    GfVec3d r = m.GetInverse().GetTranspose().TransformDir(nd);
    return safeNormalize(Vec3f{static_cast<float>(r[0]),
                                static_cast<float>(r[1]),
                                static_cast<float>(r[2])});
}

// ---------------------------------------------------------------------------
// resolveColorInput — read a Vec3f color from a UsdShadeInput.
//
// If the input has a direct value, return it.
// If it is connected (e.g. to a UsdUVTexture), follow the connection and try:
//   1. UsdUVTexture.fallback  (Vec4f — common in Pixar sample scenes)
//   2. A hue derived from the texture filename so different materials get
//      different flat colors (better than every material being the same grey).
// ---------------------------------------------------------------------------
static Spectrum resolveColorInput(const UsdShadeInput& input,
                                   const Spectrum& defaultVal) {
    if (!input) return defaultVal;

    GfVec3f col;
    if (input.Get(&col)) return {col[0], col[1], col[2]};

    // Input is texture-connected — follow the connection.
    UsdShadeSourceInfoVector sources = input.GetConnectedSources();
    if (sources.empty()) return defaultVal;

    UsdShadeShader texShader(sources[0].source.GetPrim());
    if (!texShader) return defaultVal;

    TfToken shaderId;
    texShader.GetShaderId(&shaderId);

    if (shaderId == TfToken("UsdUVTexture")) {
        // Try the fallback value first (authored on the texture node).
        UsdShadeInput fallbackIn = texShader.GetInput(TfToken("fallback"));
        GfVec4f fb;
        if (fallbackIn && fallbackIn.Get(&fb))
            return {fb[0], fb[1], fb[2]};

        // No fallback authored — derive a stable hue from the texture path so
        // different materials are visually distinguishable.
        UsdShadeInput fileIn = texShader.GetInput(TfToken("file"));
        if (fileIn) {
            SdfAssetPath ap;
            if (fileIn.Get(&ap)) {
                // Hash the path to a hue in [0, 1), map to a saturated pastel.
                std::size_t h = std::hash<std::string>{}(ap.GetAssetPath());
                float hue = static_cast<float>(h % 360) / 360.f;
                // HSV→RGB with S=0.5, V=0.8 (pastel, not too bright)
                float H = hue * 6.f;
                int   i = static_cast<int>(H);
                float f = H - static_cast<float>(i);
                float p = 0.8f * 0.5f, q = 0.8f * (1.f - 0.5f * f),
                      t = 0.8f * (1.f - 0.5f * (1.f - f)), v = 0.8f;
                switch (i % 6) {
                    case 0: return {v, t, p};
                    case 1: return {q, v, p};
                    case 2: return {p, v, t};
                    case 3: return {p, q, v};
                    case 4: return {t, p, v};
                    default: return {v, p, q};
                }
            }
        }
    }

    return defaultVal;
}

// resolveIntensity — apply USD exposure attribute: finalIntensity = intensity * 2^exposure
static float resolveIntensity(const UsdLuxLightAPI& lightAPI) {
    float intensity = 1.f, exposure = 0.f;
    lightAPI.GetIntensityAttr().Get(&intensity);
    lightAPI.GetExposureAttr().Get(&exposure);
    return intensity * std::pow(2.f, exposure);
}

// computePoolBounds — fast AABB over all mesh positions in the pool
static BBox3f computePoolBounds(const GeometryPool& pool) {
    BBox3f b;
    for (size_t i = 0; i < pool.numMeshes(); ++i)
        for (const Vec3f& p : pool.mesh(static_cast<uint32_t>(i)).positions)
            b.expand(p);
    return b;
}

// resolveFloatInput — read a scalar UsdShadeInput, returning defaultVal if
// not authored or connected to a texture (textures are not yet supported).
static float resolveFloatInput(const UsdShadeInput& input, float defaultVal) {
    if (!input) return defaultVal;
    // Ignore texture connections — fall back to default
    if (input.HasConnectedSource()) return defaultVal;
    float val = defaultVal;
    input.Get(&val);
    return val;
}

// resolveMaterial — walk a UsdShadeMaterial's surface output to find
// a UsdPreviewSurface shader and extract diffuseColor, emissiveColor,
// roughness, and metallic. Returns the most appropriate material type:
//   - EmissiveMaterial   if emissiveColor is non-black
//   - StandardSurfaceMaterial if metallic > 0 or roughness < 0.9
//   - LambertianMaterial otherwise (pure diffuse)
// ---------------------------------------------------------------------------
static std::unique_ptr<IMaterial> resolveMaterial(const UsdShadeMaterial& mat) {
    Spectrum diffuse{0.5f, 0.5f, 0.5f};
    Spectrum emission{};
    float    roughness = 1.0f;
    float    metallic  = 0.0f;

    UsdShadeShader surface = mat.ComputeSurfaceSource();
    if (!surface)
        return std::make_unique<LambertianMaterial>(diffuse);

    TfToken shaderId;
    surface.GetShaderId(&shaderId);

    if (shaderId == TfToken("UsdPreviewSurface")) {
        diffuse   = resolveColorInput(surface.GetInput(TfToken("diffuseColor")),
                                      diffuse);
        emission  = resolveColorInput(surface.GetInput(TfToken("emissiveColor")),
                                      emission);
        roughness = resolveFloatInput(surface.GetInput(TfToken("roughness")),
                                      roughness);
        metallic  = resolveFloatInput(surface.GetInput(TfToken("metallic")),
                                      metallic);
    }

    if (!isBlack(emission))
        return std::make_unique<EmissiveMaterial>(diffuse, emission);

    // Use StandardSurfaceMaterial when the surface has any specular character.
    // Pure Lambertian (roughness=1, metallic=0) stays as LambertianMaterial.
    if (metallic > 0.01f || roughness < 0.95f) {
        StandardSurfaceMaterial::Params p;
        p.base_color = diffuse;
        p.roughness  = roughness;
        p.metalness  = metallic;
        p.specular   = metallic > 0.01f ? 0.f : 0.5f;  // no dielectric spec on metals
        return std::make_unique<StandardSurfaceMaterial>(p);
    }

    return std::make_unique<LambertianMaterial>(diffuse);
}

// ---------------------------------------------------------------------------
// loadMesh — triangulate a UsdGeomMesh and add it to the GeometryPool.
// Returns the meshID assigned by the pool, or ~0u on failure.
// ---------------------------------------------------------------------------
static uint32_t loadMesh(const UsdGeomMesh& usdMesh,
                         const GfMatrix4d& xform,
                         GeometryPool& pool) {
    VtArray<GfVec3f> points;
    usdMesh.GetPointsAttr().Get(&points);
    if (points.empty()) return ~0u;

    VtArray<int> fvcCounts, fvcIndices;
    usdMesh.GetFaceVertexCountsAttr().Get(&fvcCounts);
    usdMesh.GetFaceVertexIndicesAttr().Get(&fvcIndices);
    if (fvcCounts.empty() || fvcIndices.empty()) return ~0u;

    // Normals — try face-varying first, then vertex, then compute flat
    VtArray<GfVec3f> normals;
    TfToken normalInterp;
    usdMesh.GetNormalsAttr().Get(&normals);
    normalInterp = usdMesh.GetNormalsInterpolation();

    // UVs — look for primvar st (texCoord2f[])
    VtArray<GfVec2f> uvs;
    TfToken uvInterp;
    UsdGeomPrimvarsAPI pvAPI(usdMesh.GetPrim());
    UsdGeomPrimvar stPrimvar = pvAPI.GetPrimvar(TfToken("st"));
    if (!stPrimvar) stPrimvar = pvAPI.GetPrimvar(TfToken("UVMap"));
    if (stPrimvar) {
        stPrimvar.Get(&uvs);
        uvInterp = stPrimvar.GetInterpolation();
    }

    // --- Fan-triangulate all faces ---
    MeshDesc desc;
    desc.name = usdMesh.GetPrim().GetName().GetString();

    int faceStart = 0;
    for (int fi = 0; fi < (int)fvcCounts.size(); ++fi) {
        int nv = fvcCounts[fi];
        // Fan from vertex 0
        for (int tri = 0; tri < nv - 2; ++tri) {
            int i0 = fvcIndices[faceStart];
            int i1 = fvcIndices[faceStart + tri + 1];
            int i2 = fvcIndices[faceStart + tri + 2];

            // Face-varying indices for normals/uvs (one entry per vertex of face)
            int fvi0 = faceStart;
            int fvi1 = faceStart + tri + 1;
            int fvi2 = faceStart + tri + 2;

            uint32_t base = static_cast<uint32_t>(desc.positions.size());

            // Positions (baked to world space)
            desc.positions.push_back(transformPoint(xform, GfVec3d(points[i0])));
            desc.positions.push_back(transformPoint(xform, GfVec3d(points[i1])));
            desc.positions.push_back(transformPoint(xform, GfVec3d(points[i2])));

            // Normals
            auto getNormal = [&](int vi, int fvi) -> Vec3f {
                if (!normals.empty()) {
                    int ni = (normalInterp == UsdGeomTokens->faceVarying) ? fvi : vi;
                    if (ni < (int)normals.size())
                        return transformNormal(xform, normals[ni]);
                }
                // Compute geometric normal
                Vec3f a = transformPoint(xform, GfVec3d(points[i0]));
                Vec3f b = transformPoint(xform, GfVec3d(points[i1]));
                Vec3f c = transformPoint(xform, GfVec3d(points[i2]));
                return safeNormalize(cross(b - a, c - a));
            };
            desc.normals.push_back(getNormal(i0, fvi0));
            desc.normals.push_back(getNormal(i1, fvi1));
            desc.normals.push_back(getNormal(i2, fvi2));

            // UVs
            auto getUV = [&](int vi, int fvi) -> Vec2f {
                if (!uvs.empty()) {
                    int ui = (uvInterp == UsdGeomTokens->faceVarying) ? fvi : vi;
                    if (ui < (int)uvs.size())
                        return {uvs[ui][0], uvs[ui][1]};
                }
                return {};
            };
            desc.uvs.push_back(getUV(i0, fvi0));
            desc.uvs.push_back(getUV(i1, fvi1));
            desc.uvs.push_back(getUV(i2, fvi2));

            desc.indices.push_back(base);
            desc.indices.push_back(base + 1);
            desc.indices.push_back(base + 2);
        }
        faceStart += nv;
    }

    if (desc.positions.empty()) return ~0u;
    return pool.addMesh(std::move(desc));
}

// ---------------------------------------------------------------------------
// loadUSD
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// buildCamera — convert a UsdGeomCamera prim to an anacapa Camera
// ---------------------------------------------------------------------------
static Camera buildCamera(const UsdPrim& prim,
                           UsdGeomXformCache& xformCache,
                           uint32_t filmWidth, uint32_t filmHeight) {
    UsdGeomCamera usdCam(prim);

    bool resetXformStack = false;
    GfMatrix4d localToWorld(1.0);
    UsdGeomXformable xformable(prim);
    xformable.GetLocalTransformation(&localToWorld, &resetXformStack,
                                      UsdTimeCode::Default());
    GfMatrix4d parentXform = xformCache.GetParentToWorldTransform(prim);
    GfMatrix4d fullXform   = localToWorld * parentXform;

    Vec3f origin = transformPoint(fullXform, GfVec3d(0, 0,  0));
    Vec3f target = transformPoint(fullXform, GfVec3d(0, 0, -1));
    Vec3f upWS   = transformPoint(fullXform, GfVec3d(0, 1,  0));
    Vec3f up     = safeNormalize(upWS - origin);

    GfCamera gc  = usdCam.GetCamera(UsdTimeCode::Default());
    float focalLen  = gc.GetFocalLength();
    float hAperture = gc.GetHorizontalAperture();

    float aspectRatio = static_cast<float>(filmWidth) /
                        static_cast<float>(filmHeight);
    float vAperture = hAperture / aspectRatio;
    float vfovRad   = 2.f * std::atan(vAperture * 0.5f / focalLen);
    float vfovDeg   = vfovRad * 180.f / 3.14159265f;

    // Thin lens parameters — read fStop and focusDistance if authored.
    // fStop is dimensionless; aperture radius (in world units) = focalLength / (2 * fStop).
    // USD focalLength is in tenths of a world unit (scene units / 10) by convention,
    // so we convert: focalLen_world = focalLen / 10.
    // focusDistance is already in scene units.
    // If either is absent or zero, fall back to pinhole (no DoF).
    float apertureRadius = 0.f;
    float focalDistance  = 0.f;

    float fStop = 0.f;
    usdCam.GetFStopAttr().Get(&fStop, UsdTimeCode::Default());
    usdCam.GetFocusDistanceAttr().Get(&focalDistance, UsdTimeCode::Default());

    if (fStop > 0.f && focalDistance > 0.f) {
        float focalLen_world = focalLen / 10.f;   // USD tenths → scene units
        apertureRadius = focalLen_world / (2.f * fStop);
        spdlog::info("USDLoader: camera '{}' origin=({:.2f},{:.2f},{:.2f}) fov={:.1f}° "
                     "fStop={:.1f} focusDist={:.3f} apertureR={:.4f}",
                     prim.GetPath().GetString(),
                     origin.x, origin.y, origin.z, vfovDeg,
                     fStop, focalDistance, apertureRadius);
        return Camera::makeThinLens(origin, target, up, vfovDeg,
                                    filmWidth, filmHeight,
                                    apertureRadius, focalDistance);
    }

    spdlog::info("USDLoader: camera '{}' origin=({:.2f},{:.2f},{:.2f}) fov={:.1f}° (pinhole)",
                 prim.GetPath().GetString(),
                 origin.x, origin.y, origin.z, vfovDeg);

    return Camera::makePinhole(origin, target, up, vfovDeg, filmWidth, filmHeight);
}

// ---------------------------------------------------------------------------
// loadUSD
// ---------------------------------------------------------------------------
LoadedScene loadUSD(const std::string& path,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    const std::string& cameraOverridePath) {
    LoadedScene result;

    auto stage = UsdStage::Open(path);
    if (!stage) {
        spdlog::error("USDLoader: failed to open '{}'", path);
        return result;  // result.valid stays false
    }
    result.valid = true;

    spdlog::info("USDLoader: opened '{}' (up-axis: {})",
                 path,
                 UsdGeomGetStageUpAxis(stage).GetString());

    UsdGeomXformCache xformCache;

    // Cache material → IMaterial* to avoid duplicating per-mesh
    std::unordered_map<std::string, uint32_t> matPathToIdx;

    // Cache displayColor RGB (quantized to 8-bit key) → material index
    std::unordered_map<uint32_t, uint32_t> displayColorToIdx;

    // Default material (used when no binding exists)
    result.materials.push_back(
        std::make_unique<LambertianMaterial>(Spectrum{0.5f, 0.5f, 0.5f}));
    const uint32_t kDefaultMatIdx = 0;

    // Cameras collected during traversal; resolved after loop
    std::vector<UsdPrim> cameraPrims;

    // --- Traverse all prims ---
    for (const UsdPrim& prim : stage->Traverse()) {

        // ---- Mesh ----
        if (prim.IsA<UsdGeomMesh>()) {
            UsdGeomMesh usdMesh(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            uint32_t meshID = loadMesh(usdMesh, xform, result.geomPool);
            if (meshID == ~0u) {
                spdlog::warn("USDLoader: skipped mesh '{}' (no geometry)",
                             prim.GetPath().GetString());
                continue;
            }

            // Resolve material binding
            uint32_t matIdx = kDefaultMatIdx;
            UsdShadeMaterialBindingAPI bindAPI(prim);
            UsdShadeMaterial boundMat = bindAPI.ComputeBoundMaterial();
            if (boundMat) {
                std::string matPath = boundMat.GetPath().GetString();
                auto it = matPathToIdx.find(matPath);
                if (it != matPathToIdx.end()) {
                    matIdx = it->second;
                } else {
                    matIdx = static_cast<uint32_t>(result.materials.size());
                    result.materials.push_back(resolveMaterial(boundMat));
                    matPathToIdx[matPath] = matIdx;
                }
            } else {
                // No material binding — fall back to primvars:displayColor.
                // This is the common Blender USD export pattern.
                UsdGeomPrimvarsAPI pvAPI2(prim);
                UsdGeomPrimvar dcPrimvar = pvAPI2.GetPrimvar(TfToken("displayColor"));
                if (dcPrimvar) {
                    VtArray<GfVec3f> colors;
                    dcPrimvar.Get(&colors);
                    if (!colors.empty()) {
                        GfVec3f c = colors[0];
                        // Quantize to 8-bit per channel for cache key
                        uint32_t key = (static_cast<uint32_t>(c[0] * 255.f + 0.5f) << 16)
                                     | (static_cast<uint32_t>(c[1] * 255.f + 0.5f) << 8)
                                     |  static_cast<uint32_t>(c[2] * 255.f + 0.5f);
                        auto it2 = displayColorToIdx.find(key);
                        if (it2 != displayColorToIdx.end()) {
                            matIdx = it2->second;
                        } else {
                            matIdx = static_cast<uint32_t>(result.materials.size());
                            Spectrum color{c[0], c[1], c[2]};
                            result.materials.push_back(
                                std::make_unique<LambertianMaterial>(color));
                            displayColorToIdx[key] = matIdx;
                        }
                    }
                }
            }

            // Grow scene.materials to cover this meshID
            if (meshID >= result.sceneView.materials.size())
                result.sceneView.materials.resize(meshID + 1, nullptr);
            result.sceneView.materials[meshID] = result.materials[matIdx].get();

            spdlog::debug("USDLoader: mesh '{}' → meshID={} matIdx={}",
                          prim.GetPath().GetString(), meshID, matIdx);
        }

        // ---- RectLight ----
        else if (prim.IsA<UsdLuxRectLight>()) {
            UsdLuxRectLight rect(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float width = 1.f, height = 1.f;
            rect.GetWidthAttr().Get(&width);
            rect.GetHeightAttr().Get(&height);

            float intensity = 1.f;
            rect.GetIntensityAttr().Get(&intensity);

            GfVec3f color{1.f, 1.f, 1.f};
            rect.GetColorAttr().Get(&color);

            // Center of the light in world space
            Vec3f center = transformPoint(xform, GfVec3d(0, 0, 0));

            // Half-extents: rect light in USD lies in XY plane, normal = -Z
            Vec3f uHalf = transformPoint(xform, GfVec3d(width * 0.5, 0, 0)) - center;
            Vec3f vHalf = transformPoint(xform, GfVec3d(0, height * 0.5, 0)) - center;

            Spectrum Le = {color[0] * intensity,
                           color[1] * intensity,
                           color[2] * intensity};

            auto light = std::make_unique<AreaLight>(center, uHalf, vHalf, Le);
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));

            spdlog::debug("USDLoader: rectLight '{}' Le=({:.1f},{:.1f},{:.1f})",
                          prim.GetPath().GetString(), Le.x, Le.y, Le.z);
        }

        // ---- SphereLight — approximate as a small area light ----
        else if (prim.IsA<UsdLuxSphereLight>()) {
            UsdLuxSphereLight sphere(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float radius = 0.5f;
            sphere.GetRadiusAttr().Get(&radius);

            float intensity = 1.f;
            sphere.GetIntensityAttr().Get(&intensity);

            GfVec3f color{1.f, 1.f, 1.f};
            sphere.GetColorAttr().Get(&color);

            Vec3f center = transformPoint(xform, GfVec3d(0, 0, 0));
            // Represent as an area light facing -Y
            Vec3f uHalf = {radius, 0.f, 0.f};
            Vec3f vHalf = {0.f,  0.f, radius};

            Spectrum Le = {color[0] * intensity,
                           color[1] * intensity,
                           color[2] * intensity};

            auto light = std::make_unique<AreaLight>(center, uHalf, vHalf, Le);
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));
        }

        // ---- DistantLight (sun/directional) ----
        else if (prim.IsA<UsdLuxDistantLight>()) {
            UsdLuxDistantLight dist(prim);
            UsdLuxLightAPI lightAPI(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float intensity = resolveIntensity(lightAPI);
            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);

            // DistantLight emits along -Z local; dirToLight = +Z local in world
            Vec3f lightPos  = transformPoint(xform, GfVec3d(0, 0, 0));
            Vec3f lightPosZ = transformPoint(xform, GfVec3d(0, 0, 1));
            Vec3f dirToLight = safeNormalize(lightPosZ - lightPos);

            Spectrum Le = { color[0] * intensity,
                            color[1] * intensity,
                            color[2] * intensity };

            // Bounds needed for disk placement — use placeholder; updated below
            auto light = std::make_unique<DirectionalLight>(
                dirToLight, Le, /*sceneRadius=*/1.f, Vec3f{});
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));

            spdlog::info("USDLoader: distantLight '{}' dir=({:.2f},{:.2f},{:.2f}) intensity={:.2f}",
                         prim.GetPath().GetString(),
                         dirToLight.x, dirToLight.y, dirToLight.z, intensity);
        }

        // ---- DomeLight (HDRI environment) ----
        else if (prim.IsA<UsdLuxDomeLight>()) {
            UsdLuxDomeLight dome(prim);
            UsdLuxLightAPI lightAPI(prim);

            float intensity = resolveIntensity(lightAPI);
            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);
            float effectiveIntensity = intensity * (color[0]+color[1]+color[2]) / 3.f;

            // Check for a texture file
            std::string texturePath;
            SdfAssetPath ap;
            if (dome.GetTextureFileAttr().Get(&ap))
                texturePath = ap.GetResolvedPath().empty()
                            ? ap.GetAssetPath()
                            : ap.GetResolvedPath();

            // Bounds placeholder — updated after all meshes are loaded
            auto domeLight = std::make_unique<DomeLight>(
                texturePath, effectiveIntensity, /*sceneRadius=*/1.f, Vec3f{});
            result.sceneView.envLight = domeLight.get();
            result.sceneView.lights.push_back(domeLight.get());
            result.lights.push_back(std::move(domeLight));

            spdlog::info("USDLoader: domeLight '{}' intensity={:.2f} texture='{}'",
                         prim.GetPath().GetString(), effectiveIntensity, texturePath);
        }

        // ---- Camera — collect all; resolve selection after traversal ----
        else if (prim.IsA<UsdGeomCamera>()) {
            cameraPrims.push_back(prim);
        }
    }

    // --- Camera selection (three-level priority) ---
    //
    // 1. Explicit --camera path override
    // 2. UsdRenderSettings.camera relationship
    // 3. First camera found during traversal
    //
    // Always log all available cameras so users can see what's in the file.

    if (!cameraPrims.empty()) {
        spdlog::info("USDLoader: {} camera(s) found in scene:", cameraPrims.size());
        for (const auto& cp : cameraPrims)
            spdlog::info("  {}", cp.GetPath().GetString());
    }

    UsdPrim selectedCamPrim;

    // Priority 1: explicit --camera override
    if (!cameraOverridePath.empty()) {
        UsdPrim p = stage->GetPrimAtPath(SdfPath(cameraOverridePath));
        if (p && p.IsA<UsdGeomCamera>()) {
            selectedCamPrim = p;
            spdlog::info("USDLoader: using camera from --camera flag: '{}'",
                         cameraOverridePath);
        } else {
            spdlog::warn("USDLoader: --camera '{}' not found or not a camera; "
                         "falling back", cameraOverridePath);
        }
    }

    // Priority 2: UsdRenderSettings.camera relationship
    if (!selectedCamPrim) {
        for (const UsdPrim& prim : stage->Traverse()) {
            if (!prim.IsA<UsdRenderSettings>()) continue;
            UsdRenderSettings rs(prim);
            UsdRelationship camRel = rs.GetCameraRel();
            SdfPathVector targets;
            if (camRel && camRel.GetForwardedTargets(&targets) && !targets.empty()) {
                UsdPrim p = stage->GetPrimAtPath(targets[0]);
                if (p && p.IsA<UsdGeomCamera>()) {
                    selectedCamPrim = p;
                    spdlog::info("USDLoader: using camera from RenderSettings '{}': '{}'",
                                 prim.GetPath().GetString(),
                                 targets[0].GetString());
                    break;
                }
            }
        }
    }

    // Priority 3: first camera found
    if (!selectedCamPrim && !cameraPrims.empty()) {
        selectedCamPrim = cameraPrims[0];
        if (cameraPrims.size() > 1)
            spdlog::warn("USDLoader: multiple cameras found; using first '{}'. "
                         "Use --camera <path> to select another.",
                         selectedCamPrim.GetPath().GetString());
        else
            spdlog::info("USDLoader: using only camera '{}'",
                         selectedCamPrim.GetPath().GetString());
    }

    if (selectedCamPrim)
        result.camera = buildCamera(selectedCamPrim, xformCache, filmWidth, filmHeight);
    else
        spdlog::info("USDLoader: no camera in scene; renderer will use default");

    // Pad materials vector to cover all mesh IDs
    result.sceneView.materials.resize(result.geomPool.numMeshes(), nullptr);

    // Default env radiance — black
    result.sceneView.envRadiance = {};

    // Update scene-bounds-dependent lights (DistantLight, DomeLight) now that
    // all geometry is loaded and we can compute a tight bounding sphere.
    {
        BBox3f bounds = computePoolBounds(result.geomPool);
        if (bounds.valid()) {
            Vec3f center = bounds.centroid();
            float radius = bounds.diagonal().length() * 0.5f * 1.5f;  // 1.5× safety margin

            for (auto& lightPtr : result.lights) {
                if (auto* dl = dynamic_cast<DirectionalLight*>(lightPtr.get())) {
                    dl->setSceneRadius(radius);
                    dl->setSceneCenter(center);
                } else if (auto* dome = dynamic_cast<DomeLight*>(lightPtr.get())) {
                    dome->setSceneRadius(radius);
                    dome->setSceneCenter(center);
                }
            }
            spdlog::info("USDLoader: scene bounds center=({:.1f},{:.1f},{:.1f}) radius={:.1f}",
                         center.x, center.y, center.z, radius);
        }
    }

    spdlog::info("USDLoader: {} meshes, {} lights, camera={}",
                 result.geomPool.numMeshes(),
                 result.sceneView.lights.size(),
                 result.camera.has_value() ? "yes" : "none (using default)");

    return result;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_USD
