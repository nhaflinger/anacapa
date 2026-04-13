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
#include <algorithm>
#include <cmath>
#include <string>
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

// Z-up → Y-up correction: swap Y and Z, negate new Z (= old Y) to preserve handedness.
// Applied to every world-space point/vector/normal when upAxis == "Z".
// The matrix is:  [1  0  0]   i.e. x→x, y→z, z→-y
//                 [0  0 -1]
//                 [0  1  0]
// This rotates -90° around X: Z-up becomes Y-up.
static Vec3f applyUpCorrection(Vec3f v, bool zUp) {
    if (!zUp) return v;
    return { v.x, v.z, -v.y };
}

// Apply a 4x4 world transform to a point using GfMatrix4d::Transform
// (USD uses row-vector convention: p_world = m.Transform(p_local))
static Vec3f transformPoint(const GfMatrix4d& m, const GfVec3d& p, bool zUp = false) {
    GfVec3d r = m.Transform(p);
    return applyUpCorrection({ static_cast<float>(r[0]),
                               static_cast<float>(r[1]),
                               static_cast<float>(r[2]) }, zUp);
}

static Mat4f toMat4f(const GfMatrix4d& m) {
    // USD GfMatrix4d uses row-vector convention: p_world = p_local * M,
    // so translation is in the last ROW (m[3][0..2]).
    // Anacapa Mat4f uses column-vector convention: p_world = M * p_local,
    // so translation must be in the last COLUMN (m[0..2][3]).
    // Transpose on import to reconcile.
    Mat4f r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r.m[i][j] = static_cast<float>(m[j][i]);  // transposed
    return r;
}

// Apply a 4x4 world transform to a normal using inverse-transpose
static Vec3f transformNormal(const GfMatrix4d& m, const GfVec3f& n, bool zUp = false) {
    GfVec3d nd(n[0], n[1], n[2]);
    GfVec3d r = m.GetInverse().GetTranspose().TransformDir(nd);
    return safeNormalize(applyUpCorrection(Vec3f{static_cast<float>(r[0]),
                                                  static_cast<float>(r[1]),
                                                  static_cast<float>(r[2])}, zUp));
}

// ---------------------------------------------------------------------------
// UVTextureInfo — extracted data from a UsdUVTexture node
// ---------------------------------------------------------------------------
struct UVTextureInfo {
    std::string path;
    Vec2f       uvScale       = {1.f, 1.f};
    Vec2f       uvTranslation = {0.f, 0.f};
    float       uvRotation    = 0.f;
    GfVec4f     fallback      = {0.5f, 0.5f, 0.5f, 1.f};
    GfVec4f     scale         = {1.f, 1.f, 1.f, 1.f};
    GfVec4f     bias          = {0.f, 0.f, 0.f, 0.f};
    bool        isSRGB        = false; // true when sourceColorSpace == "sRGB" or "auto" + color file
    // Output channel from the UsdUVTexture: "r","g","b","a","rgb","rgba"
    // Empty means full RGB. Used for packed ORM textures.
    std::string outputChannel;
};

// resolveUVTexture — extract texture info from a UsdUVTexture shader node.
// Also follows the st input to pick up UsdTransform2d UV transforms.
// stageDir: directory of the USD file, used to resolve relative texture paths.
// Returns true if a valid file path was found.
static bool resolveUVTexture(const UsdShadeShader& texShader,
                              const std::string& stageDir,
                              UVTextureInfo& out) {
    TfToken shaderId;
    texShader.GetShaderId(&shaderId);
    if (shaderId != TfToken("UsdUVTexture")) return false;

    UsdShadeInput fileIn = texShader.GetInput(TfToken("file"));
    if (fileIn) {
        SdfAssetPath ap;
        if (fileIn.Get(&ap)) {
            // Prefer the USD-resolved absolute path; fall back to manual join
            // with the stage directory for relative paths.
            if (!ap.GetResolvedPath().empty()) {
                out.path = ap.GetResolvedPath();
            } else if (!ap.GetAssetPath().empty()) {
                const std::string& asset = ap.GetAssetPath();
                if (!asset.empty() && asset[0] == '/') {
                    out.path = asset;  // already absolute
                } else if (!stageDir.empty()) {
                    out.path = stageDir + "/" + asset;
                } else {
                    out.path = asset;
                }
            }
        }
    }

    UsdShadeInput fbIn = texShader.GetInput(TfToken("fallback"));
    if (fbIn) fbIn.Get(&out.fallback);

    UsdShadeInput scaleIn = texShader.GetInput(TfToken("scale"));
    if (scaleIn) scaleIn.Get(&out.scale);

    UsdShadeInput biasIn = texShader.GetInput(TfToken("bias"));
    if (biasIn) biasIn.Get(&out.bias);

    // sourceColorSpace: "sRGB" or "auto" means the file is gamma-encoded.
    // "raw" means linear (used for roughness, metallic, normals).
    UsdShadeInput csIn = texShader.GetInput(TfToken("sourceColorSpace"));
    if (csIn) {
        TfToken cs;
        if (csIn.Get(&cs)) {
            out.isSRGB = (cs == TfToken("sRGB") || cs == TfToken("auto"));
        }
    } else {
        // No sourceColorSpace authored — default is "auto" which treats
        // 8-bit files (PNG/JPG) as sRGB. Err on the side of linearizing
        // color inputs to avoid overly dark renders.
        out.isSRGB = true;
    }

    // Follow st → optional UsdTransform2d for UV transforms
    UsdShadeInput stIn = texShader.GetInput(TfToken("st"));
    if (stIn && stIn.HasConnectedSource()) {
        UsdShadeSourceInfoVector stSources = stIn.GetConnectedSources();
        if (!stSources.empty()) {
            UsdShadeShader stShader(stSources[0].source.GetPrim());
            if (stShader) {
                TfToken stId;
                stShader.GetShaderId(&stId);
                if (stId == TfToken("UsdTransform2d")) {
                    GfVec2f sc{1.f, 1.f}, tr{0.f, 0.f};
                    float   rot = 0.f;
                    UsdShadeInput scIn  = stShader.GetInput(TfToken("scale"));
                    UsdShadeInput trIn  = stShader.GetInput(TfToken("translation"));
                    UsdShadeInput rotIn = stShader.GetInput(TfToken("rotation"));
                    if (scIn)  scIn.Get(&sc);
                    if (trIn)  trIn.Get(&tr);
                    if (rotIn) rotIn.Get(&rot);
                    out.uvScale       = {sc[0], sc[1]};
                    out.uvTranslation = {tr[0], tr[1]};
                    out.uvRotation    = rot;
                }
            }
        }
    }
    return !out.path.empty();
}

// resolveColorTOV — read a color input, returning a SpectrumTOV that carries
// either a constant value or a file texture path + UV transform.
//
// Connections are checked before the authored constant value — see the comment
// in resolveFloatTOV for the rationale.
static SpectrumTOV resolveColorTOV(const UsdShadeInput& input,
                                    const Spectrum& defaultVal,
                                    const std::string& stageDir) {
    if (!input) return SpectrumTOV(defaultVal);

    // Check for a connected texture source first.
    UsdShadeSourceInfoVector sources = input.GetConnectedSources();
    if (!sources.empty()) {
        UsdShadeShader texShader(sources[0].source.GetPrim());
        if (texShader) {
            // Use the authored constant as the fallback value.
            GfVec3f authCol{defaultVal.x, defaultVal.y, defaultVal.z};
            input.Get(&authCol);
            UVTextureInfo info;
            info.fallback      = {authCol[0], authCol[1], authCol[2], 1.f};
            info.outputChannel = sources[0].sourceName.GetString(); // e.g. "rgb", "r", "g"
            if (resolveUVTexture(texShader, stageDir, info)) {
                SpectrumTOV tov(Spectrum{info.fallback[0], info.fallback[1], info.fallback[2]});
                tov.path          = info.path;
                tov.uvScale       = info.uvScale;
                tov.uvTranslation = info.uvTranslation;
                tov.uvRotation    = info.uvRotation;
                tov.linearize     = info.isSRGB;
                return tov;
            }
        }
    }

    // No connection (or connection failed to resolve) — use the constant value.
    GfVec3f col{defaultVal.x, defaultVal.y, defaultVal.z};
    input.Get(&col);
    return SpectrumTOV(Spectrum{col[0], col[1], col[2]});
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

// resolveFloatTOV — read a scalar input, returning a FloatTOV that carries
// either a constant value or a file texture path + UV transform.
//
// IMPORTANT: connections are checked BEFORE the authored constant value.
// USD shader inputs can legally have both a connection and an authored fallback
// value (the fallback is used when the connection is unresolvable at runtime).
// If we checked Get() first we would return the constant and silently ignore
// the connection — e.g. opacity=1.0 with a texture alpha connection would
// never resolve the texture, leaving alpha-masked surfaces always opaque.
static FloatTOV resolveFloatTOV(const UsdShadeInput& input, float defaultVal,
                                  const std::string& stageDir) {
    if (!input) return FloatTOV(defaultVal);

    // Check for a connected texture source first.
    UsdShadeSourceInfoVector sources = input.GetConnectedSources();
    if (!sources.empty()) {
        UsdShadeShader texShader(sources[0].source.GetPrim());
        if (texShader) {
            // Use the authored constant as the fallback value for when the
            // texture file is unavailable; fall back to defaultVal otherwise.
            float authVal = defaultVal;
            input.Get(&authVal);
            UVTextureInfo info;
            info.fallback      = {authVal, authVal, authVal, 1.f};
            info.outputChannel = sources[0].sourceName.GetString();
            if (resolveUVTexture(texShader, stageDir, info)) {
                FloatTOV tov(info.fallback[0]);
                tov.path          = info.path;
                tov.uvScale       = info.uvScale;
                tov.uvTranslation = info.uvTranslation;
                tov.uvRotation    = info.uvRotation;
                // Store channel in path suffix so evalTOV can select correctly.
                // Encode as a null-terminated tag after a pipe: "path|g" means G channel.
                if (!info.outputChannel.empty() && info.outputChannel != "r" && info.outputChannel != "rgb")
                    tov.path += "|" + info.outputChannel;
                return tov;
            }
        }
    }

    // No connection (or connection failed to resolve) — use the constant value.
    float val = defaultVal;
    input.Get(&val);
    return FloatTOV(val);
}

// resolveMaterial — walk a UsdShadeMaterial's surface output to find
// a UsdPreviewSurface shader and extract all material parameters including
// file textures, UV transforms, normal maps, opacity, and clearcoat.
// ---------------------------------------------------------------------------
static std::unique_ptr<IMaterial> resolveMaterial(const UsdShadeMaterial& mat,
                                                    const std::string& stageDir) {
    UsdShadeShader surface = mat.ComputeSurfaceSource();
    if (!surface)
        return std::make_unique<LambertianMaterial>(Spectrum{0.5f, 0.5f, 0.5f});

    TfToken shaderId;
    surface.GetShaderId(&shaderId);

    if (shaderId != TfToken("UsdPreviewSurface"))
        return std::make_unique<LambertianMaterial>(Spectrum{0.5f, 0.5f, 0.5f});

    StandardSurfaceMaterial::Params p;
    p.base_color = resolveColorTOV(surface.GetInput(TfToken("diffuseColor")),
                                    Spectrum{0.5f, 0.5f, 0.5f}, stageDir);
    p.roughness  = resolveFloatTOV(surface.GetInput(TfToken("roughness")),  1.0f, stageDir);
    p.metalness  = resolveFloatTOV(surface.GetInput(TfToken("metallic")),   0.0f, stageDir);
    p.opacity       = resolveFloatTOV(surface.GetInput(TfToken("opacity")),      1.0f, stageDir);
    p.specular_IOR  = resolveFloatTOV(surface.GetInput(TfToken("ior")),          1.5f, stageDir).value;

    // Derive transmission from whichever convention the exporter used:
    //
    //   1. inputs:transmission  — the explicit standard_surface / MaterialX
    //      convention (e.g. Houdini, Arnold, some DCC tools).
    //
    //   2. inputs:opacity = 0   — the UsdPreviewSurface convention used by
    //      Blender: a glass Cycles material is exported as opacity=0 with no
    //      transmission attribute at all.  We infer transmission = 1 - opacity
    //      so both conventions work transparently.
    //
    // Taking the max means a file that sets both attributes doesn't get double-
    // counted, and a fully opaque material (opacity=1, transmission=0) stays 0.
    float explicitTransmission = resolveFloatTOV(surface.GetInput(TfToken("transmission")), 0.0f, stageDir).value;
    float opacityVal = p.opacity.value;

    // If opacity is driven by a texture (e.g. alpha channel of the diffuse map),
    // treat it as an alpha mask rather than glass transmission.  The opacity
    // constant fallback is 1.0 in this case so 1-opacity would (incorrectly)
    // yield 0 — instead mark it as alphaMask and leave transmission alone.
    if (!p.opacity.path.empty()) {
        p.alphaMask   = true;
        p.transmission = explicitTransmission;   // texture-driven opacity ≠ glass
        spdlog::info("USDLoader: material '{}' alphaMask=true opacity path='{}'",
                     mat.GetPath().GetString(), p.opacity.path);
    } else {
        p.transmission = std::max(explicitTransmission, 1.f - opacityVal);
    }

    // Specular: USD UsdPreviewSurface has no separate specular weight input;
    // use 0 for metals (they have no dielectric specular) and 0.5 for others.
    // For transmissive materials, set specular weight = 1 so Fresnel is fully evaluated.
    if (p.transmission > 0.001f && p.metalness.value < 0.001f)
        p.specular = FloatTOV(1.0f);
    else
        p.specular = FloatTOV(p.metalness.value > 0.01f ? 0.f : 0.5f);

    // Clearcoat
    p.coat           = resolveFloatTOV(surface.GetInput(TfToken("clearcoat")), 0.f, stageDir).value;
    p.coat_roughness = resolveFloatTOV(surface.GetInput(TfToken("clearcoatRoughness")), 0.1f, stageDir).value;

    // Emission — emissiveColor may be a constant or a texture.
    // When it is texture-driven the constant fallback is (0,0,0) even though
    // the surface does emit — check the texture path too.
    {
        SpectrumTOV emissiveTOV = resolveColorTOV(
            surface.GetInput(TfToken("emissiveColor")), Spectrum{}, stageDir);
        p.emission_color = emissiveTOV;   // SpectrumTOV carries path + value
        bool hasEmission = !isBlack(emissiveTOV.value)
                        || !emissiveTOV.path.empty();
        p.emission = hasEmission ? 1.f : 0.f;
    }

    // Normal map (tangent-space): bias/scale come from UsdUVTexture inputs
    UsdShadeInput normalIn = surface.GetInput(TfToken("normal"));
    if (normalIn && normalIn.HasConnectedSource()) {
        UsdShadeSourceInfoVector nSources = normalIn.GetConnectedSources();
        if (!nSources.empty()) {
            UsdShadeShader nShader(nSources[0].source.GetPrim());
            if (nShader) {
                UVTextureInfo nInfo;
                nInfo.fallback = {0.5f, 0.5f, 1.0f, 1.0f};
                if (resolveUVTexture(nShader, stageDir, nInfo)) {
                    p.normal_map.path          = nInfo.path;
                    p.normal_map.value         = {nInfo.fallback[0], nInfo.fallback[1], nInfo.fallback[2]};
                    p.normal_map.uvScale       = nInfo.uvScale;
                    p.normal_map.uvTranslation = nInfo.uvTranslation;
                    p.normal_map.uvRotation    = nInfo.uvRotation;
                    p.normal_scale             = nInfo.scale[0];
                    p.normal_bias              = nInfo.bias[0];
                    p.has_normal_map           = true;
                }
            }
        }
    }

    return std::make_unique<StandardSurfaceMaterial>(p);
}

// Collect all authored time samples from xformOp:* attributes on a prim.
// Direct attribute enumeration is more reliable than GetOrderedXformOps(),
// which can silently return empty results on prims that don't satisfy all
// of UsdGeomXformable's requirements.
static void collectXformTimeSamples(const UsdPrim& prim, std::vector<double>& times) {
    for (const UsdAttribute& attr : prim.GetAttributes()) {
        // Match any attribute in the "xformOp" namespace (e.g. xformOp:translate,
        // xformOp:rotateXYZ).  Using GetNamespace() is more reliable than string
        // prefix matching on GetName(), which can behave unexpectedly across USD
        // versions when attribute names are stored as TfTokens.
        if (attr.GetNamespace() == TfToken("xformOp")) {
            std::vector<double> attrTimes;
            attr.GetTimeSamples(&attrTimes);
            if (!attrTimes.empty())
                spdlog::debug("USDLoader: '{}' attr '{}' has {} time sample(s)",
                              prim.GetPath().GetString(),
                              attr.GetName().GetString(),
                              attrTimes.size());
            for (double t : attrTimes)
                times.push_back(t);
        }
    }
}

// ---------------------------------------------------------------------------
// collectMotionKeys — build a sorted list of MotionKey from a USD prim's
// animated world transform.
//
// Gathers the union of all authored time samples from every xformOp on the
// prim (translate, rotate, scale, etc.) plus the parent hierarchy, evaluates
// the full local-to-world transform at each sample, and normalizes the time
// codes to [0, 1] relative to [startTime, endTime].
//
// Returns an empty vector for static prims.
// ---------------------------------------------------------------------------
static std::vector<MotionKey> collectMotionKeys(
        const UsdPrim& prim,
        double startTime,
        double endTime)
{
    // Collect the union of all authored xformOp time samples from the prim
    // and every ancestor, since animation may live on a parent Xform prim.
    std::vector<double> times;
    collectXformTimeSamples(prim, times);
    UsdPrim parent = prim.GetParent();
    while (parent && parent.IsValid()) {
        collectXformTimeSamples(parent, times);
        parent = parent.GetParent();
    }

    // Deduplicate and sort
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());

    {
        std::string timesStr;
        for (double t : times) { timesStr += std::to_string(t) + " "; }
        spdlog::info("USDLoader: collectMotionKeys for '{}': {} raw sample(s): [{}]",
                     prim.GetPath().GetString(), times.size(), timesStr);
    }

    // Always include the stage endpoints so the full shutter is covered
    times.insert(times.begin(), startTime);
    times.push_back(endTime);
    std::sort(times.begin(), times.end());
    times.erase(std::unique(times.begin(), times.end()), times.end());

    if (times.size() < 2) return {};

    double range = endTime - startTime;

    std::vector<MotionKey> keys;
    keys.reserve(times.size());
    for (double tc : times) {
        float normalizedTime = (range > 0.0)
            ? static_cast<float>((tc - startTime) / range)
            : 0.f;

        UsdGeomXformCache cache{UsdTimeCode(tc)};
        GfMatrix4d xfm = cache.GetLocalToWorldTransform(prim);

        MotionKey key;
        key.time          = normalizedTime;
        key.objectToWorld = toMat4f(xfm);
        key.worldToObject = toMat4f(xfm.GetInverse());
        keys.push_back(key);
    }
    // Log each key's translation so we can verify the arc is being captured
    for (const MotionKey& k : keys) {
        // Translation is in the last column of the column-major anacapa Mat4f
        spdlog::info("USDLoader:   key t={:.3f} translate=({:.3f},{:.3f},{:.3f})",
                     k.time,
                     k.objectToWorld.m[0][3],
                     k.objectToWorld.m[1][3],
                     k.objectToWorld.m[2][3]);
    }

    return keys;
}

// ---------------------------------------------------------------------------
// loadMesh — triangulate a UsdGeomMesh and add it to the GeometryPool.
// Returns the meshID assigned by the pool, or ~0u on failure.
//
// xform0     = world-from-object at shutter open (used for static mesh baking
//              and as a fallback).
// motionKeys = piecewise-linear transform samples, normalized to [0,1].
//              Empty for static meshes.
// ---------------------------------------------------------------------------
static uint32_t loadMesh(const UsdGeomMesh& usdMesh,
                         const GfMatrix4d& xform0,
                         std::vector<MotionKey> motionKeys,
                         GeometryPool& pool,
                         bool zUp = false) {
    const bool hasMotion = !motionKeys.empty();
    VtArray<GfVec3f> points;
    usdMesh.GetPointsAttr().Get(&points);
    if (points.empty()) return ~0u;

    VtArray<int> fvcCounts, fvcIndices;
    usdMesh.GetFaceVertexCountsAttr().Get(&fvcCounts);
    usdMesh.GetFaceVertexIndicesAttr().Get(&fvcIndices);
    if (fvcCounts.empty() || fvcIndices.empty()) return ~0u;

    // For animated meshes we keep positions in object space and carry both transforms.
    const GfMatrix4d& xform = xform0;

    // Normals — try face-varying first, then vertex, then compute flat
    VtArray<GfVec3f> normals;
    TfToken normalInterp;
    usdMesh.GetNormalsAttr().Get(&normals);
    normalInterp = usdMesh.GetNormalsInterpolation();

    // UVs — look for primvar st (texCoord2f[])
    // Use ComputeFlattened() to expand indexed primvars (Blender USD exports always
    // use primvars:st:indices, so Get() returns the raw deduplicated values and gives
    // wrong UVs when used with face-varying fvi indices).
    VtArray<GfVec2f> uvs;
    TfToken uvInterp;
    UsdGeomPrimvarsAPI pvAPI(usdMesh.GetPrim());
    UsdGeomPrimvar stPrimvar = pvAPI.GetPrimvar(TfToken("st"));
    if (!stPrimvar) stPrimvar = pvAPI.GetPrimvar(TfToken("UVMap"));
    if (stPrimvar) {
        stPrimvar.ComputeFlattened(&uvs);
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

            if (hasMotion) {
                // Object-space positions — BVH will apply interpolated transform at ray.time
                desc.positions.push_back({(float)points[i0][0], (float)points[i0][1], (float)points[i0][2]});
                desc.positions.push_back({(float)points[i1][0], (float)points[i1][1], (float)points[i1][2]});
                desc.positions.push_back({(float)points[i2][0], (float)points[i2][1], (float)points[i2][2]});
            } else {
                // Bake to world space (static mesh fast path)
                desc.positions.push_back(transformPoint(xform, GfVec3d(points[i0]), zUp));
                desc.positions.push_back(transformPoint(xform, GfVec3d(points[i1]), zUp));
                desc.positions.push_back(transformPoint(xform, GfVec3d(points[i2]), zUp));
            }

            // Normals
            auto getNormal = [&](int vi, int fvi) -> Vec3f {
                if (!normals.empty()) {
                    int ni = (normalInterp == UsdGeomTokens->faceVarying) ? fvi : vi;
                    if (ni < (int)normals.size()) {
                        if (hasMotion)
                            return applyUpCorrection(safeNormalize({normals[ni][0], normals[ni][1], normals[ni][2]}), zUp);
                        return transformNormal(xform, normals[ni], zUp);
                    }
                }
                // Compute geometric normal
                Vec3f a = hasMotion
                    ? applyUpCorrection(Vec3f{(float)points[i0][0], (float)points[i0][1], (float)points[i0][2]}, zUp)
                    : transformPoint(xform, GfVec3d(points[i0]), zUp);
                Vec3f b = hasMotion
                    ? applyUpCorrection(Vec3f{(float)points[i1][0], (float)points[i1][1], (float)points[i1][2]}, zUp)
                    : transformPoint(xform, GfVec3d(points[i1]), zUp);
                Vec3f c = hasMotion
                    ? applyUpCorrection(Vec3f{(float)points[i2][0], (float)points[i2][1], (float)points[i2][2]}, zUp)
                    : transformPoint(xform, GfVec3d(points[i2]), zUp);
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

    if (hasMotion) {
        desc.motionKeys = std::move(motionKeys);
    }
    // Static meshes: positions already in world space; motionKeys stays empty.

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
                           uint32_t filmWidth, uint32_t filmHeight,
                           bool zUp = false) {
    UsdGeomCamera usdCam(prim);

    bool resetXformStack = false;
    GfMatrix4d localToWorld(1.0);
    UsdGeomXformable xformable(prim);
    xformable.GetLocalTransformation(&localToWorld, &resetXformStack,
                                      UsdTimeCode::Default());
    GfMatrix4d parentXform = xformCache.GetParentToWorldTransform(prim);
    GfMatrix4d fullXform   = localToWorld * parentXform;

    Vec3f origin = transformPoint(fullXform, GfVec3d(0, 0,  0), zUp);
    Vec3f target = transformPoint(fullXform, GfVec3d(0, 0, -1), zUp);
    Vec3f upWS   = transformPoint(fullXform, GfVec3d(0, 1,  0), zUp);
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

    // Compute stage directory for resolving relative texture paths
    std::string stageDir;
    {
        std::string stagePath = stage->GetRootLayer()->GetRealPath();
        if (stagePath.empty()) stagePath = path;
        auto slash = stagePath.rfind('/');
        stageDir = (slash != std::string::npos) ? stagePath.substr(0, slash) : ".";
    }

    const bool zUp = (UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z);
    spdlog::info("USDLoader: opened '{}' (up-axis: {}{})",
                 path,
                 UsdGeomGetStageUpAxis(stage).GetString(),
                 zUp ? " — applying Y↔Z correction" : "");

    // Use the stage's authored time range for motion detection.
    // Falls back to 0/1 if the stage has no time codes (static scenes).
    double startTime = stage->HasAuthoredTimeCodeRange()
                         ? stage->GetStartTimeCode() : 0.0;
    double endTime   = stage->HasAuthoredTimeCodeRange()
                         ? stage->GetEndTimeCode()   : 1.0;

    // Normalized shutter interval: only enable motion blur if the stage has an
    // authored time range (i.e. it's actually animated). Static scenes get 0/0.
    result.shutterOpen  = 0.f;
    result.shutterClose = 0.f;
    if (stage->HasAuthoredTimeCodeRange()) {
        double tcps = stage->GetTimeCodesPerSecond();
        if (tcps <= 0.0) tcps = 24.0;
        double durationSec = (endTime - startTime) / tcps;
        if (durationSec > 0.0)
            result.shutterClose = 1.f;
        spdlog::info("USDLoader: animated time range [{}, {}] @ {} tcps ({:.4f}s) — motion blur enabled",
                     startTime, endTime, tcps, durationSec);
    } else {
        spdlog::info("USDLoader: static scene (no authored time range) — motion blur disabled");
    }

    UsdTimeCode tcOpen{startTime};
    UsdTimeCode tcClose{endTime};
    UsdGeomXformCache xformCache{tcOpen};    // shutter-open
    UsdGeomXformCache xformCacheT1{tcClose}; // shutter-close (motion detection)

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

            // Detect animated transforms by comparing the full world transform
            // (including parent hierarchy) at t=0 vs t=1.  This catches motion
            // that lives on a parent Xform prim rather than on the Mesh itself.
            GfMatrix4d xform0 = xformCache.GetLocalToWorldTransform(prim);
            GfMatrix4d xform1 = xformCacheT1.GetLocalToWorldTransform(prim);
            bool hasMotion = (xform0 != xform1);

            std::vector<MotionKey> motionKeys;
            if (hasMotion) {
                motionKeys = collectMotionKeys(prim, startTime, endTime);
                spdlog::info("USDLoader: animated mesh '{}' ({} motion key(s), motion blur active)",
                             prim.GetPath().GetString(), motionKeys.size());
            }

            uint32_t meshID = loadMesh(usdMesh, xform0, std::move(motionKeys), result.geomPool, zUp);
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
                    result.materials.push_back(resolveMaterial(boundMat, stageDir));
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
            UsdLuxLightAPI lightAPI(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float width = 1.f, height = 1.f;
            rect.GetWidthAttr().Get(&width);
            rect.GetHeightAttr().Get(&height);

            float intensity = 1.f;
            rect.GetIntensityAttr().Get(&intensity);

            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);

            bool normalize = false;
            lightAPI.GetNormalizeAttr().Get(&normalize);

            // Center of the light in world space
            Vec3f center = transformPoint(xform, GfVec3d(0, 0, 0), zUp);

            // Half-extents: rect light in USD lies in XY plane, normal = -Z
            Vec3f uHalf = transformPoint(xform, GfVec3d(width * 0.5, 0, 0), zUp) - center;
            Vec3f vHalf = transformPoint(xform, GfVec3d(0, height * 0.5, 0), zUp) - center;

            // When normalize=true, USD intensity is irradiance (W/m²).
            // Our AreaLight Le is radiance (W/m²/sr); for a Lambertian emitter: Le = E/π.
            float leScale = normalize ? (1.f / 3.14159265f) : 1.f;
            Spectrum Le = {color[0] * intensity * leScale,
                           color[1] * intensity * leScale,
                           color[2] * intensity * leScale};

            auto light = std::make_unique<AreaLight>(center, uHalf, vHalf, Le);
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));

            spdlog::debug("USDLoader: rectLight '{}' Le=({:.3f},{:.3f},{:.3f}) normalize={}",
                          prim.GetPath().GetString(), Le.x, Le.y, Le.z, normalize);
        }

        // ---- SphereLight — approximate as a small area light ----
        else if (prim.IsA<UsdLuxSphereLight>()) {
            UsdLuxSphereLight sphere(prim);
            UsdLuxLightAPI lightAPI(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float radius = 0.5f;
            sphere.GetRadiusAttr().Get(&radius);

            float intensity = 1.f;
            sphere.GetIntensityAttr().Get(&intensity);

            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);

            bool normalize = false;
            lightAPI.GetNormalizeAttr().Get(&normalize);

            Vec3f center = transformPoint(xform, GfVec3d(0, 0, 0), zUp);
            // Represent as an area light facing -Y (after up correction)
            Vec3f uHalf = {radius, 0.f, 0.f};
            Vec3f vHalf = {0.f,  0.f, radius};

            float leScale = normalize ? (1.f / 3.14159265f) : 1.f;
            Spectrum Le = {color[0] * intensity * leScale,
                           color[1] * intensity * leScale,
                           color[2] * intensity * leScale};

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
            Vec3f lightPos  = transformPoint(xform, GfVec3d(0, 0, 0), zUp);
            Vec3f lightPosZ = transformPoint(xform, GfVec3d(0, 0, 1), zUp);
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

            // Apply the DomeLight's world transform rotation to correctly orient the HDRI.
            // The xform rotates the envmap local space into world space; we store its
            // transpose (= inverse for orthogonal matrices) so we can rotate world-space
            // directions into envmap space for lookup.
            GfMatrix4d domeXform = xformCache.GetLocalToWorldTransform(prim);
            // Extract upper-left 3x3 (rotation + scale); USD DomeLights are typically
            // only rotated (no scale), but we normalize each column for safety.
            auto col0 = GfVec3d(domeXform[0][0], domeXform[0][1], domeXform[0][2]);
            auto col1 = GfVec3d(domeXform[1][0], domeXform[1][1], domeXform[1][2]);
            auto col2 = GfVec3d(domeXform[2][0], domeXform[2][1], domeXform[2][2]);
            col0.Normalize(); col1.Normalize(); col2.Normalize();
            // Convert from USD Z-up columns to Y-up by applying up-axis correction
            Vec3f c0 = applyUpCorrection({(float)col0[0], (float)col0[1], (float)col0[2]}, zUp);
            Vec3f c1 = applyUpCorrection({(float)col1[0], (float)col1[1], (float)col1[2]}, zUp);
            Vec3f c2 = applyUpCorrection({(float)col2[0], (float)col2[1], (float)col2[2]}, zUp);
            // setRotation takes the columns of the world-to-envmap matrix.
            // The local-to-world columns (c0, c1, c2) are the rows of world-to-local,
            // so pass them directly as the rows of the rotation matrix.
            domeLight->setRotation(c0, c1, c2);

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

    if (selectedCamPrim) {
        result.camera = buildCamera(selectedCamPrim, xformCache, filmWidth, filmHeight, zUp);

        // Read shutter:open / shutter:close from the camera prim.
        // Must use UsdTimeCode::Default() for non-time-varying attributes and
        // check the return value — if Get() returns false the output variable
        // is left unchanged at its initialised value (0.0), which would make
        // the condition camShutterClose > camShutterOpen silently false.
        UsdGeomCamera usdCam(selectedCamPrim);
        double camShutterOpen  = 0.0;
        double camShutterClose = 0.0;
        bool gotOpen  = usdCam.GetShutterOpenAttr() .Get(&camShutterOpen,  UsdTimeCode::Default());
        bool gotClose = usdCam.GetShutterCloseAttr().Get(&camShutterClose, UsdTimeCode::Default());
        spdlog::info("USDLoader: camera shutter:open={:.4f} (authored={}) "
                     "shutter:close={:.4f} (authored={})",
                     camShutterOpen, gotOpen, camShutterClose, gotClose);
        if (gotClose && camShutterClose > camShutterOpen) {
            result.shutterOpen  = static_cast<float>(camShutterOpen);
            result.shutterClose = static_cast<float>(camShutterClose);
            spdlog::info("USDLoader: using camera shutter [{:.4f}, {:.4f}]",
                         result.shutterOpen, result.shutterClose);
        }
    } else {
        spdlog::info("USDLoader: no camera in scene; renderer will use default");
    }

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

    // Auto-register emissive meshes as AreaLights so they contribute to direct
    // lighting (NEE) and illuminate surrounding geometry.
    //
    // We approximate each emissive mesh as a single quad AreaLight aligned to
    // the mesh AABB.  The dominant axis (longest diagonal component) determines
    // the quad normal; u/v half-extents come from the remaining two axes.
    // Le is the emission_color constant value scaled by emission weight — texture-
    // driven emission uses the constant fallback, which may be (0,0,0); in that
    // case we skip registration (the mesh still self-illuminates via Le() but
    // won't drive NEE).
    {
        uint32_t autoLightCount = 0;
        for (uint32_t mid = 0; mid < result.geomPool.numMeshes(); ++mid) {
            const IMaterial* imat = result.sceneView.materials[mid];
            if (!imat) continue;

            const auto* smat = dynamic_cast<const StandardSurfaceMaterial*>(imat);
            if (!smat) continue;

            const StandardSurfaceMaterial::Params& sp = smat->params();
            if (sp.emission <= 0.f) continue;

            // Use the constant emission color.  For texture-driven emission the
            // constant fallback is (0,0,0); fall back to a warm white so the mesh
            // still contributes to direct illumination (NEE).  The intensity is
            // approximate — texture-driven emitters self-illuminate via Le() at
            // full accuracy, but NEE needs a finite Le to sample toward them.
            Spectrum Le = sp.emission_color.value * sp.emission;
            if (isBlack(Le)) {
                if (!sp.emission_color.path.empty())
                    Le = {sp.emission * 1.f, sp.emission * 0.7f, sp.emission * 0.3f}; // warm orange approximation
                else
                    continue;  // truly black emitter, skip
            }

            // Compute tight AABB of this mesh in world space
            const MeshDesc& mesh = result.geomPool.mesh(mid);
            BBox3f mb;
            for (const Vec3f& p : mesh.positions)
                mb.expand(p);
            if (!mb.valid()) continue;

            Vec3f diag = mb.diagonal();
            Vec3f center = mb.centroid();

            // Choose the dominant axis as the normal direction, u/v from the other two.
            Vec3f uHalf, vHalf;
            if (diag.x <= diag.y && diag.x <= diag.z) {
                // X is smallest — normal points along X
                uHalf = {0.f, diag.y * 0.5f, 0.f};
                vHalf = {0.f, 0.f, diag.z * 0.5f};
            } else if (diag.y <= diag.x && diag.y <= diag.z) {
                // Y is smallest — normal points along Y
                uHalf = {diag.x * 0.5f, 0.f, 0.f};
                vHalf = {0.f, 0.f, diag.z * 0.5f};
            } else {
                // Z is smallest — normal points along Z
                uHalf = {diag.x * 0.5f, 0.f, 0.f};
                vHalf = {0.f, diag.y * 0.5f, 0.f};
            }

            auto light = std::make_unique<AreaLight>(center, uHalf, vHalf, Le);
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));
            ++autoLightCount;

            spdlog::info("USDLoader: emissive mesh {} → AreaLight center=({:.2f},{:.2f},{:.2f}) "
                         "Le=({:.3f},{:.3f},{:.3f})",
                         mid, center.x, center.y, center.z, Le.x, Le.y, Le.z);
        }
        if (autoLightCount > 0)
            spdlog::info("USDLoader: auto-registered {} emissive mesh(es) as AreaLight(s)",
                         autoLightCount);
    }

    spdlog::info("USDLoader: {} meshes, {} lights, camera={}",
                 result.geomPool.numMeshes(),
                 result.sceneView.lights.size(),
                 result.camera.has_value() ? "yes" : "none (using default)");

    return result;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_USD
