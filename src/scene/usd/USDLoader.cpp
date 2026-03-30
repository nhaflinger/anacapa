#ifdef ANACAPA_ENABLE_USD

#include "USDLoader.h"
#include "../../shading/Lambertian.h"
#include "../../shading/lights/AreaLight.h"

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
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
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
// resolveMaterial — walk a UsdShadeMaterial's surface output to find
// a UsdPreviewSurface shader and extract diffuseColor + emissiveColor.
// Returns a LambertianMaterial or EmissiveMaterial.
// ---------------------------------------------------------------------------
static std::unique_ptr<IMaterial> resolveMaterial(const UsdShadeMaterial& mat) {
    // Default grey if binding can't be resolved
    Spectrum diffuse{0.5f, 0.5f, 0.5f};
    Spectrum emission{};

    UsdShadeShader surface = mat.ComputeSurfaceSource();
    if (!surface) {
        return std::make_unique<LambertianMaterial>(diffuse);
    }

    TfToken shaderId;
    surface.GetShaderId(&shaderId);

    if (shaderId == TfToken("UsdPreviewSurface")) {
        // diffuseColor
        UsdShadeInput diffIn = surface.GetInput(TfToken("diffuseColor"));
        if (diffIn) {
            GfVec3f col;
            if (diffIn.Get(&col)) diffuse = {col[0], col[1], col[2]};
        }

        // emissiveColor
        UsdShadeInput emissIn = surface.GetInput(TfToken("emissiveColor"));
        if (emissIn) {
            GfVec3f col;
            if (emissIn.Get(&col)) emission = {col[0], col[1], col[2]};
        }
    }

    if (!isBlack(emission))
        return std::make_unique<EmissiveMaterial>(diffuse, emission);
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
LoadedScene loadUSD(const std::string& path,
                    uint32_t filmWidth,
                    uint32_t filmHeight) {
    LoadedScene result;

    auto stage = UsdStage::Open(path);
    if (!stage) {
        spdlog::error("USDLoader: failed to open '{}'", path);
        return result;
    }

    spdlog::info("USDLoader: opened '{}' (up-axis: {})",
                 path,
                 UsdGeomGetStageUpAxis(stage).GetString());

    UsdGeomXformCache xformCache;

    // Cache material → IMaterial* to avoid duplicating per-mesh
    std::unordered_map<std::string, uint32_t> matPathToIdx;

    // Default material (used when no binding exists)
    result.materials.push_back(
        std::make_unique<LambertianMaterial>(Spectrum{0.5f, 0.5f, 0.5f}));
    const uint32_t kDefaultMatIdx = 0;

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

        // ---- Camera — first camera found wins ----
        else if (prim.IsA<UsdGeomCamera>() && !result.camera.has_value()) {
            UsdGeomCamera usdCam(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            // Camera-to-world: compute directly from xformable (bypasses cache)
            bool resetXformStack = false;
            GfMatrix4d localToWorld(1.0);
            UsdGeomXformable xformable(prim);
            xformable.GetLocalTransformation(&localToWorld, &resetXformStack,
                                              UsdTimeCode::Default());
            // Compose with parent xform from cache
            GfMatrix4d parentXform = xformCache.GetParentToWorldTransform(prim);
            GfMatrix4d fullXform   = localToWorld * parentXform;

            spdlog::debug("USDLoader: camera xform t=({:.3f},{:.3f},{:.3f})",
                          fullXform[3][0], fullXform[3][1], fullXform[3][2]);

            // USD cameras look in -Z in local space
            Vec3f origin = transformPoint(fullXform, GfVec3d(0, 0,  0));
            Vec3f target = transformPoint(fullXform, GfVec3d(0, 0, -1));
            Vec3f upWS   = transformPoint(fullXform, GfVec3d(0, 1,  0));
            Vec3f up     = safeNormalize(upWS - origin);

            // Vertical FOV from focal length + aperture
            float focalLen = 50.f, hAperture = 20.943f;
            GfCamera gc = usdCam.GetCamera(UsdTimeCode::Default());
            focalLen  = gc.GetFocalLength();
            hAperture = gc.GetHorizontalAperture();

            float aspectRatio = static_cast<float>(filmWidth) /
                                static_cast<float>(filmHeight);
            float vAperture = hAperture / aspectRatio;
            float vfovRad   = 2.f * std::atan(vAperture * 0.5f / focalLen);
            float vfovDeg   = vfovRad * 180.f / 3.14159265f;

            result.camera = Camera::makePinhole(origin, target, up,
                                                 vfovDeg, filmWidth, filmHeight);
            spdlog::info("USDLoader: camera '{}' origin=({:.2f},{:.2f},{:.2f}) fov={:.1f}°",
                         prim.GetPath().GetString(),
                         origin.x, origin.y, origin.z, vfovDeg);
        }
    }

    // Pad materials vector to cover all mesh IDs
    result.sceneView.materials.resize(result.geomPool.numMeshes(), nullptr);

    // Default env radiance — black
    result.sceneView.envRadiance = {};

    spdlog::info("USDLoader: {} meshes, {} lights, camera={}",
                 result.geomPool.numMeshes(),
                 result.sceneView.lights.size(),
                 result.camera.has_value() ? "yes" : "none (using default)");

    return result;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_USD
