#ifdef ANACAPA_ENABLE_USD

#include "USDLoader.h"
#include "../../shading/Lambertian.h"
#include "../../shading/SoftParticle.h"
#include "../../shading/StandardSurface.h"
#include "../../shading/OslMaterial.h"
#include "../../shading/lights/AreaLight.h"
#include "../../shading/lights/DirectionalLight.h"
#include "../../shading/lights/DomeLight.h"
#include "../../sky/SkyConfig.h"
#include "../../sky/SkyLight.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/points.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/nodeGraph.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdRender/settings.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <spdlog/spdlog.h>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

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

// resolveAssetPath — shared helper for reading an SdfAssetPath input and
// resolving it to an absolute file path using stageDir.
static std::string resolveAssetPath(const UsdShadeInput& fileIn,
                                     const std::string& stageDir) {
    if (!fileIn) return {};
    SdfAssetPath ap;
    if (!fileIn.Get(&ap)) return {};
    if (!ap.GetResolvedPath().empty()) return ap.GetResolvedPath();
    const std::string& asset = ap.GetAssetPath();
    if (asset.empty()) return {};
    if (asset[0] == '/') return asset;
    return stageDir.empty() ? asset : stageDir + "/" + asset;
}

// resolveUVTexture — extract texture info from a shader node.
// Handles both UsdUVTexture (UsdPreviewSurface) and MaterialX ND_image_*
// nodes (Blender MaterialX export with generate_materialx_network=True).
// Also follows UV-transform connections (UsdTransform2d / ND_place2d_*).
// Returns true if a valid file path was found.
static bool resolveUVTexture(const UsdShadeShader& texShader,
                              const std::string& stageDir,
                              UVTextureInfo& out) {
    TfToken shaderId;
    texShader.GetShaderId(&shaderId);
    const std::string& sId = shaderId.GetString();

    // -----------------------------------------------------------------------
    // UsdUVTexture — UsdPreviewSurface texture node
    // -----------------------------------------------------------------------
    if (shaderId == TfToken("UsdUVTexture")) {
        out.path = resolveAssetPath(texShader.GetInput(TfToken("file")), stageDir);

        UsdShadeInput fbIn = texShader.GetInput(TfToken("fallback"));
        if (fbIn) fbIn.Get(&out.fallback);

        UsdShadeInput scaleIn = texShader.GetInput(TfToken("scale"));
        if (scaleIn) scaleIn.Get(&out.scale);

        UsdShadeInput biasIn = texShader.GetInput(TfToken("bias"));
        if (biasIn) biasIn.Get(&out.bias);

        // sourceColorSpace: "sRGB" or "auto" → gamma-encoded; "raw" → linear.
        UsdShadeInput csIn = texShader.GetInput(TfToken("sourceColorSpace"));
        if (csIn) {
            TfToken cs;
            if (csIn.Get(&cs))
                out.isSRGB = (cs == TfToken("sRGB") || cs == TfToken("auto"));
        } else {
            // No sourceColorSpace → default "auto" treats 8-bit as sRGB.
            out.isSRGB = true;
        }

        // Follow st → UsdTransform2d for UV transforms
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

    // -----------------------------------------------------------------------
    // MaterialX ND_image_* texture nodes (Blender MaterialX export)
    // Node IDs: ND_image_color3, ND_image_color4, ND_image_float,
    //           ND_image_vector2, ND_image_vector3
    // -----------------------------------------------------------------------
    if (sId.size() > 9 && sId.substr(0, 9) == "ND_image_") {
        out.path = resolveAssetPath(texShader.GetInput(TfToken("file")), stageDir);

        // ND_image nodes don't have a separate "fallback" — the caller's
        // fallback stays as-is (set before calling resolveUVTexture).

        // MaterialX image nodes are raw linear by default for non-color data.
        // Color images need linearization. Infer from node type suffix.
        // "color3" / "color4" → sRGB input files; "float" / "vector*" → linear.
        out.isSRGB = (sId.find("color") != std::string::npos);

        // Follow texcoord → ND_place2d_* for UV transforms.
        // Blender uses ND_place2d_vector2 with inputs: scale (Vec2), offset (Vec2), rotate (float, degrees).
        UsdShadeInput tcIn = texShader.GetInput(TfToken("texcoord"));
        if (tcIn && tcIn.HasConnectedSource()) {
            UsdShadeSourceInfoVector tcSources = tcIn.GetConnectedSources();
            if (!tcSources.empty()) {
                UsdShadeShader tcShader(tcSources[0].source.GetPrim());
                if (tcShader) {
                    TfToken tcId;
                    tcShader.GetShaderId(&tcId);
                    if (tcId.GetString().size() > 11 &&
                        tcId.GetString().substr(0, 11) == "ND_place2d_") {
                        GfVec2f sc{1.f, 1.f}, off{0.f, 0.f};
                        float rot = 0.f;
                        UsdShadeInput scIn  = tcShader.GetInput(TfToken("scale"));
                        UsdShadeInput offIn = tcShader.GetInput(TfToken("offset"));
                        UsdShadeInput rotIn = tcShader.GetInput(TfToken("rotate"));
                        if (scIn)  scIn.Get(&sc);
                        if (offIn) offIn.Get(&off);
                        if (rotIn) rotIn.Get(&rot);
                        out.uvScale       = {sc[0], sc[1]};
                        out.uvTranslation = {off[0], off[1]};
                        out.uvRotation    = rot;
                    }
                }
            }
        }
        return !out.path.empty();
    }

    return false;
}

// resolveShaderThroughNodeGraph — when a connection target is a NodeGraph prim
// (Blender wraps MaterialX subgraphs in NodeGraph containers), follow the named
// output back to the actual Shader prim inside.  Returns the resolved Shader, or
// an invalid Shader if the prim is not a NodeGraph or resolution fails.
// Depth-limited to avoid infinite loops.
static UsdShadeShader resolveShaderThroughNodeGraph(const UsdPrim& prim,
                                                     const std::string& outputName,
                                                     int depth = 0) {
    if (depth > 8 || !prim.IsValid()) return UsdShadeShader();
    if (!prim.IsA<UsdShadeNodeGraph>()) return UsdShadeShader(prim);

    UsdShadeNodeGraph ng(prim);
    // Try the named output first, fall back to first available output
    UsdShadeOutput out = outputName.empty()
                         ? UsdShadeOutput()
                         : ng.GetOutput(TfToken(outputName));
    if (!out) {
        std::vector<UsdShadeOutput> outs = ng.GetOutputs();
        if (!outs.empty()) out = outs[0];
    }
    if (!out || !out.HasConnectedSource()) return UsdShadeShader();

    UsdShadeSourceInfoVector innerSrcs = out.GetConnectedSources();
    if (innerSrcs.empty()) return UsdShadeShader();

    UsdPrim innerPrim = innerSrcs[0].source.GetPrim();
    std::string innerOut = innerSrcs[0].sourceName.GetString();
    return resolveShaderThroughNodeGraph(innerPrim, innerOut, depth + 1);
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
        // Resolve through NodeGraph wrappers (Blender wraps MaterialX subgraphs
        // in NodeGraph containers; the actual ND_image_* shader is inside).
        UsdShadeShader texShader = resolveShaderThroughNodeGraph(
            sources[0].source.GetPrim(), sources[0].sourceName.GetString());
        if (texShader) {
            // Use the authored constant as the fallback value.
            GfVec3f authCol{defaultVal.x, defaultVal.y, defaultVal.z};
            input.Get(&authCol);
            UVTextureInfo info;
            info.fallback      = {authCol[0], authCol[1], authCol[2], 1.f};
            info.outputChannel = sources[0].sourceName.GetString();
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
        // Resolve through NodeGraph wrappers — same as resolveColorTOV.
        UsdShadeShader texShader = resolveShaderThroughNodeGraph(
            sources[0].source.GetPrim(), sources[0].sourceName.GetString());
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

// makeGlassMaterial — create a standard glass material with the given IOR.
// Used as a fallback when a USD material has no surface shader but its name
// indicates it is glass (e.g. Blender's Glass BSDF doesn't export to USD).
static std::unique_ptr<IMaterial> makeGlassMaterial(float ior = 1.5f) {
    StandardSurfaceMaterial::Params p;
    p.base_color    = SpectrumTOV(Spectrum{1.f, 1.f, 1.f});
    p.roughness     = FloatTOV(0.f);
    p.metalness     = FloatTOV(0.f);
    p.transmission  = 1.f;
    p.specular_IOR  = ior;
    p.specular      = FloatTOV(1.f);
    p.emission      = 0.f;
    return std::make_unique<StandardSurfaceMaterial>(p);
}

// isGlassName — heuristic: does this material/prim name suggest a glass material?
static bool isGlassName(const std::string& name) {
    std::string lower = name;
    for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower.find("glass") != std::string::npos
        || lower.find("window") != std::string::npos
        || lower.find("glazing") != std::string::npos;
}

// findOpenPBRShader — search a material's prim tree for an
// ND_open_pbr_surface_surfaceshader node, which Blender 4.x+ emits when
// generate_materialx_network=True. Returns an invalid shader if not found.
static UsdShadeShader findOpenPBRShader(const UsdShadeMaterial& mat) {
    // The OpenPBR shader is wired to outputs:mtlx:surface, not outputs:surface,
    // so ComputeSurfaceSource() won't find it. Walk children directly.
    for (const UsdPrim& child : mat.GetPrim().GetAllDescendants()) {
        if (!child.IsA<UsdShadeShader>()) continue;
        UsdShadeShader sh(child);
        TfToken id;
        sh.GetShaderId(&id);
        if (id == TfToken("ND_open_pbr_surface_surfaceshader"))
            return sh;
    }
    return UsdShadeShader();
}

// findDirectTranslucentBsdf — return the ND_translucent_bsdf shader only when
// it is the DIRECT bsdf input to the top-level ND_surface node (possibly through
// a single NodeGraph passthrough).  This avoids falsely matching materials where
// ND_translucent_bsdf appears as one input to a Mix Shader.
static UsdShadeShader findDirectTranslucentBsdf(const UsdShadeMaterial& mat) {
    UsdStageWeakPtr stage = mat.GetPrim().GetStage();

    // Find any ND_surface node inside the material prim tree.
    for (const UsdPrim& child : mat.GetPrim().GetAllDescendants()) {
        if (!child.IsA<UsdShadeShader>()) continue;
        UsdShadeShader sh(child);
        TfToken id;
        sh.GetShaderId(&id);
        if (id != TfToken("ND_surface")) continue;

        UsdShadeInput bsdfIn = sh.GetInput(TfToken("bsdf"));
        if (!bsdfIn) continue;
        UsdShadeSourceInfoVector sources = bsdfIn.GetConnectedSources();
        for (const auto& src : sources) {
            UsdPrim srcPrim = src.source.GetPrim();

            // Direct shader connection
            if (srcPrim.IsA<UsdShadeShader>()) {
                UsdShadeShader srcSh(srcPrim);
                TfToken srcId;
                srcSh.GetShaderId(&srcId);
                if (srcId == TfToken("ND_translucent_bsdf")) return srcSh;
                continue;
            }

            // NodeGraph passthrough: follow the named output to its source shader.
            UsdAttribute outAttr = srcPrim.GetAttribute(
                TfToken("outputs:" + src.sourceName.GetString()));
            if (!outAttr) continue;
            SdfPathVector targets;
            outAttr.GetConnections(&targets);
            if (targets.empty()) continue;
            UsdPrim targetPrim = stage->GetPrimAtPath(targets[0].GetPrimPath());
            if (!targetPrim.IsA<UsdShadeShader>()) continue;
            UsdShadeShader targetSh(targetPrim);
            TfToken targetId;
            targetSh.GetShaderId(&targetId);
            if (targetId == TfToken("ND_translucent_bsdf")) return targetSh;
        }
    }
    return UsdShadeShader();
}

// resolveOpenPBRParams — extract StandardSurfaceMaterial::Params from an
// ND_open_pbr_surface_surfaceshader node. OpenPBR is a superset of
// UsdPreviewSurface with better physical parameterisation.
static StandardSurfaceMaterial::Params resolveOpenPBRParams(
    const UsdShadeShader& surface, const std::string& stageDir)
{
    StandardSurfaceMaterial::Params p;

    // Base layer
    p.base_color = resolveColorTOV(
        surface.GetInput(TfToken("base_color")),
        Spectrum{0.5f, 0.5f, 0.5f}, stageDir);
    p.roughness = resolveFloatTOV(
        surface.GetInput(TfToken("specular_roughness")), 0.5f, stageDir);
    p.metalness = resolveFloatTOV(
        surface.GetInput(TfToken("base_metalness")), 0.0f, stageDir);

    // IOR / specular
    p.specular_IOR = resolveFloatTOV(
        surface.GetInput(TfToken("specular_ior")), 1.5f, stageDir).value;
    // OpenPBR specular_weight maps to our specular param
    p.specular = resolveFloatTOV(
        surface.GetInput(TfToken("specular_weight")), 1.0f, stageDir);

    // Transmission / glass
    float transmissionWeight = resolveFloatTOV(
        surface.GetInput(TfToken("transmission_weight")), 0.0f, stageDir).value;
    p.transmission = transmissionWeight;
    p.opacity      = FloatTOV(1.0f - transmissionWeight);

    // Emission
    {
        float emWeight = resolveFloatTOV(
            surface.GetInput(TfToken("emission_luminance")), 0.0f, stageDir).value;
        SpectrumTOV emColor = resolveColorTOV(
            surface.GetInput(TfToken("emission_color")),
            Spectrum{1.f, 1.f, 1.f}, stageDir);
        p.emission_color = emColor;
        p.emission       = emWeight > 0.f ? emWeight : 0.f;
    }

    // Coat
    p.coat           = resolveFloatTOV(
        surface.GetInput(TfToken("coat_weight")), 0.0f, stageDir).value;
    p.coat_roughness = resolveFloatTOV(
        surface.GetInput(TfToken("coat_roughness")), 0.1f, stageDir).value;

    // Normal map — OpenPBR uses geometry_normal input
    UsdShadeInput normalIn = surface.GetInput(TfToken("geometry_normal"));
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

    // Specular defaults: fully evaluating Fresnel for glass, 0 for metals
    if (p.transmission > 0.001f && p.metalness.value < 0.001f)
        p.specular = FloatTOV(1.0f);
    else if (p.metalness.value > 0.01f)
        p.specular = FloatTOV(0.0f);

    // Subsurface scattering (OpenPBR / MaterialX standard_surface tokens).
    // subsurface_radius in MaterialX is a color3 (per-channel radius); we take
    // the average of R, G, B as our scalar mean free path.  subsurface_scale is
    // an additional scalar multiplier present in some DCC exports.
    {
        float sssWeight = resolveFloatTOV(
            surface.GetInput(TfToken("subsurface_weight")), 0.0f, stageDir).value;
        if (sssWeight <= 0.f)   // also try bare "subsurface" used by older exporters
            sssWeight = resolveFloatTOV(
                surface.GetInput(TfToken("subsurface")), 0.0f, stageDir).value;
        p.subsurface = sssWeight;

        if (sssWeight > 0.f) {
            p.subsurface_color = resolveColorTOV(
                surface.GetInput(TfToken("subsurface_color")),
                Spectrum{1.f, 1.f, 1.f}, stageDir);

            SpectrumTOV radVec = resolveColorTOV(
                surface.GetInput(TfToken("subsurface_radius")),
                Spectrum{0.1f, 0.1f, 0.1f}, stageDir);
            p.subsurface_radius = (radVec.value.x + radVec.value.y + radVec.value.z) / 3.f;

            p.subsurface_scale = resolveFloatTOV(
                surface.GetInput(TfToken("subsurface_scale")), 1.f, stageDir).value;
            if (p.subsurface_scale <= 0.f) p.subsurface_scale = 1.f;

            p.subsurface_anisotropy = resolveFloatTOV(
                surface.GetInput(TfToken("subsurface_anisotropy")), 0.0f, stageDir).value;

            p.subsurface_strength = resolveFloatTOV(
                surface.GetInput(TfToken("anacapa_subsurface_strength")), 1.0f, stageDir).value;
            if (p.subsurface_strength <= 0.f) p.subsurface_strength = 1.f;
        }
    }

    // Caustic opt-in: a custom bool input `anacapa_caustic` flags this surface
    // as a caustic generator for the photon map integrator.  Off by default so
    // existing scenes with ordinary glass keep using NEE transmittance.
    if (UsdShadeInput causticIn = surface.GetInput(TfToken("anacapa_caustic"))) {
        bool causticVal = false;
        causticIn.Get(&causticVal);
        p.caustic = causticVal;
    }
    if (UsdShadeInput crIn = surface.GetInput(TfToken("anacapa_caustic_radius"))) {
        float crVal = 0.f;
        crIn.Get(&crVal);
        p.caustic_radius = crVal;
    }

    return p;
}

// resolveColorTOVWithFallback — return `primary` if it has a texture path,
// otherwise return `fallback` (which may have one instead).
static SpectrumTOV resolveColorTOVWithFallback(const SpectrumTOV& primary,
                                                const SpectrumTOV& fallback) {
    return primary.path.empty() ? fallback : primary;
}
static FloatTOV resolveFloatTOVWithFallback(const FloatTOV& primary,
                                             const FloatTOV& fallback) {
    return primary.path.empty() ? fallback : primary;
}

// resolveMaterial — walk a UsdShadeMaterial's surface output to find
// a surface shader and extract all material parameters including
// file textures, UV transforms, normal maps, opacity, and clearcoat.
//
// Strategy when both OpenPBR and UsdPreviewSurface are present (Blender
// exports both with generate_materialx_network=True):
//   - OpenPBR is used for physical params: IOR, coat, transmission, etc.
//   - For textured inputs (base_color, roughness, metalness, normal),
//     we prefer whichever network actually has a connected texture.
//     Blender may generate the OpenPBR terminal with literal constants while
//     keeping the texture connection only in the UsdPreviewSurface network.
// ---------------------------------------------------------------------------
static std::unique_ptr<IMaterial> resolveMaterial(const UsdShadeMaterial& mat,
                                                    const std::string& stageDir) {
    // ND_translucent_bsdf — checked before OSL so the StandardSurface path
    // (which supports the scatter width parameter) handles it even when an
    // OSL shader file exists for the same material.
    {
        UsdShadeShader transSh = findDirectTranslucentBsdf(mat);
        if (transSh) {
            Spectrum color = {0.8f, 0.8f, 0.8f};
            UsdShadeInput colorIn = transSh.GetInput(TfToken("color"));
            if (colorIn) {
                GfVec3f v;
                if (colorIn.GetAttr().Get(&v)) color = {v[0], v[1], v[2]};
            }
            float scatter = 0.5f;
            UsdShadeInput strengthIn = transSh.GetInput(TfToken("strength"));
            if (strengthIn) {
                float s;
                if (strengthIn.GetAttr().Get(&s)) scatter = std::max(0.f, std::min(1.f, s));
            }
            spdlog::info("USDLoader: material '{}' → translucent "
                         "(color=({:.2f},{:.2f},{:.2f}) scatter={:.2f})",
                         mat.GetPath().GetString(), color.x, color.y, color.z, scatter);
            StandardSurfaceMaterial::Params p;
            p.base               = 0.f;
            p.translucency       = 1.f;
            p.translucency_color = SpectrumTOV(color);
            p.scatter            = scatter;
            p.roughness          = FloatTOV(1.f);
            p.specular           = FloatTOV(0.f);
            return std::make_unique<StandardSurfaceMaterial>(p);
        }
    }

#ifdef ANACAPA_ENABLE_OSL
    // If a matching .osl/.oso file exists in <stageDir>/materials/, prefer
    // OslMaterial — it evaluates the full procedural MaterialX graph.
    //
    // OSL compilation (.osl → .oso) is done via oslCompileShader() in
    // OslMaterial.cpp which isolates all OSL/OIIO includes from this TU.
    {
        auto tryOsl = [&](const std::string& name) -> std::unique_ptr<IMaterial> {
            if (name.empty()) return nullptr;
            std::string matDir = stageDir + "/materials";
            std::string osoPath = matDir + "/" + name + ".oso";
            std::string oslPath = matDir + "/" + name + ".osl";

            // Compile .osl → .oso if .oso is missing or older than .osl
            {
                namespace fs = std::filesystem;
                bool needCompile = false;
                if (!fs::exists(osoPath)) {
                    if (!fs::exists(oslPath)) return nullptr;
                    needCompile = true;
                } else if (fs::exists(oslPath) &&
                           fs::last_write_time(oslPath) > fs::last_write_time(osoPath)) {
                    needCompile = true;
                }
                if (needCompile) {
                    spdlog::info("USDLoader: compiling OSL shader '{}'", oslPath);
                    if (!oslCompileShader(oslPath, osoPath, matDir)) return nullptr;
                }
            }

            // Register the materials/ dir with the OSL ShadingSystem (once per dir)
            static std::string s_registeredDir;
            if (s_registeredDir != matDir) {
                oslAddSearchPath(matDir);
                s_registeredDir = matDir;
            }

            spdlog::info("USDLoader: material '{}' → OslMaterial ('{}')",
                         mat.GetPath().GetString(), name);
            auto m = makeOslMaterial(name);
            if (!m)
                spdlog::warn("USDLoader: OslMaterial load failed for '{}'; "
                             "falling back to StandardSurface", name);
            return m;  // nullptr → caller falls through to StandardSurface
        };

        // Prefer blender:data_name (the Blender material name used as filename)
        std::string blenderName;
        UsdAttribute nameAttr = mat.GetPrim().GetAttribute(
            TfToken("userProperties:blender:data_name"));
        if (nameAttr) nameAttr.Get(&blenderName);

        // MaterialX createValidName() replaces any character that isn't
        // alphanumeric or underscore with '_', so "Material.001" becomes
        // "Material_001".  The blender_prep_for_usd_export.py script uses this
        // when naming .mtlx → .oso files, but the USD blender:data_name
        // retains the original Blender name with dots.  Try the sanitized form
        // as a fallback so both "Material.001" and "Material_001" resolve.
        auto sanitizeName = [](const std::string& s) {
            std::string r = s;
            for (char& c : r)
                if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
                    c = '_';
            return r;
        };
        std::string blenderNameSanitized = sanitizeName(blenderName);

        auto osl = tryOsl(blenderName);
        if (!osl && blenderNameSanitized != blenderName)
            osl = tryOsl(blenderNameSanitized);
        if (!osl) osl = tryOsl(mat.GetPrim().GetName().GetString());
        if (osl) {
            // Inject SSS params from the UsdPreviewSurface shader (written by
            // the Blender addon post-processor) into the OslMaterial so the
            // photon map integrator can query them via subsurfaceParams().
            UsdShadeShader preview = mat.ComputeSurfaceSource();
            TfToken previewId;
            if (preview) preview.GetShaderId(&previewId);
            if (preview && previewId == TfToken("UsdPreviewSurface")) {
                float sssWeight = resolveFloatTOV(
                    preview.GetInput(TfToken("subsurface_weight")), 0.0f, stageDir).value;
                if (sssWeight <= 0.f)
                    sssWeight = resolveFloatTOV(
                        preview.GetInput(TfToken("subsurface")), 0.0f, stageDir).value;
                if (sssWeight > 0.f) {
                    SpectrumTOV sssColor = resolveColorTOV(
                        preview.GetInput(TfToken("subsurface_color")),
                        Spectrum{1.f, 1.f, 1.f}, stageDir);
                    SpectrumTOV rv = resolveColorTOV(
                        preview.GetInput(TfToken("subsurface_radius")),
                        Spectrum{0.1f, 0.1f, 0.1f}, stageDir);
                    float radius = (rv.value.x + rv.value.y + rv.value.z) / 3.f;
                    float scale = resolveFloatTOV(
                        preview.GetInput(TfToken("subsurface_scale")), 1.f, stageDir).value;
                    if (scale > 0.f) radius *= scale;
                    float aniso = resolveFloatTOV(
                        preview.GetInput(TfToken("subsurface_anisotropy")), 0.0f, stageDir).value;
                    float strength = resolveFloatTOV(
                        preview.GetInput(TfToken("anacapa_subsurface_strength")), 1.0f, stageDir).value;
                    if (strength <= 0.f) strength = 1.f;
                    oslSetSubsurfaceParams(osl.get(), sssWeight, sssColor.value, radius, aniso, strength);
                    spdlog::info("USDLoader: OSL material '{}' SSS weight={:.3f} radius={:.4f}",
                                 mat.GetPrim().GetName().GetString(), sssWeight, radius);
                }
            }
            return osl;
        }
    }
#endif

    // Try OpenPBR (MaterialX) first — richer parameterisation
    UsdShadeShader openPBR = findOpenPBRShader(mat);
    if (openPBR) {
        spdlog::debug("USDLoader: material '{}' using ND_open_pbr_surface",
                      mat.GetPath().GetString());
        StandardSurfaceMaterial::Params p = resolveOpenPBRParams(openPBR, stageDir);

        // If UsdPreviewSurface is also present, use it as a texture fallback
        // for inputs that OpenPBR resolved to a literal (no texture path).
        // This covers the common Blender case where MaterialX terminal nodes
        // are generated without ND_image_* children — the textures live only
        // in the UsdPreviewSurface subgraph.
        UsdShadeShader preview = mat.ComputeSurfaceSource();
        TfToken previewId;
        if (preview) preview.GetShaderId(&previewId);
        if (preview && previewId == TfToken("UsdPreviewSurface")) {
            SpectrumTOV pvColor = resolveColorTOV(
                preview.GetInput(TfToken("diffuseColor")),
                Spectrum{0.5f,0.5f,0.5f}, stageDir);
            FloatTOV pvRoughness = resolveFloatTOV(
                preview.GetInput(TfToken("roughness")), 0.5f, stageDir);
            FloatTOV pvMetalness = resolveFloatTOV(
                preview.GetInput(TfToken("metallic")), 0.0f, stageDir);
            FloatTOV pvOpacity   = resolveFloatTOV(
                preview.GetInput(TfToken("opacity")), 1.0f, stageDir);

            p.base_color = resolveColorTOVWithFallback(p.base_color, pvColor);
            p.roughness  = resolveFloatTOVWithFallback(p.roughness,  pvRoughness);
            p.metalness  = resolveFloatTOVWithFallback(p.metalness,  pvMetalness);
            p.opacity    = resolveFloatTOVWithFallback(p.opacity,    pvOpacity);

            // SSS: injected onto the UsdPreviewSurface shader by the Blender
            // addon post-processor — read it here if OpenPBR didn't have it.
            if (p.subsurface <= 0.f) {
                UsdShadeInput sssIn = preview.GetInput(TfToken("subsurface_weight"));
                float sssWeight = resolveFloatTOV(sssIn, 0.0f, stageDir).value;
                if (sssWeight <= 0.f)
                    sssWeight = resolveFloatTOV(
                        preview.GetInput(TfToken("subsurface")), 0.0f, stageDir).value;
                p.subsurface = sssWeight;
                if (sssWeight > 0.f) {
                    p.subsurface_color = resolveColorTOV(
                        preview.GetInput(TfToken("subsurface_color")),
                        Spectrum{1.f, 1.f, 1.f}, stageDir);
                    SpectrumTOV radVec = resolveColorTOV(
                        preview.GetInput(TfToken("subsurface_radius")),
                        Spectrum{0.1f, 0.1f, 0.1f}, stageDir);
                    p.subsurface_radius = (radVec.value.x + radVec.value.y + radVec.value.z) / 3.f;
                    float scale = resolveFloatTOV(
                        preview.GetInput(TfToken("subsurface_scale")), 1.f, stageDir).value;
                    p.subsurface_scale = (scale > 0.f) ? scale : 1.f;
                    p.subsurface_anisotropy = resolveFloatTOV(
                        preview.GetInput(TfToken("subsurface_anisotropy")), 0.0f, stageDir).value;
                }
            }

            // Normal map: OpenPBR uses geometry_normal; if not resolved,
            // fall back to UsdPreviewSurface normal input.
            if (!p.has_normal_map) {
                UsdShadeInput pvNormalIn = preview.GetInput(TfToken("normal"));
                if (pvNormalIn && pvNormalIn.HasConnectedSource()) {
                    UsdShadeSourceInfoVector nSrcs = pvNormalIn.GetConnectedSources();
                    if (!nSrcs.empty()) {
                        UsdShadeShader nSh(nSrcs[0].source.GetPrim());
                        if (nSh) {
                            UVTextureInfo nInfo;
                            nInfo.fallback = {0.5f, 0.5f, 1.0f, 1.0f};
                            if (resolveUVTexture(nSh, stageDir, nInfo)) {
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
            }
        }
        return std::make_unique<StandardSurfaceMaterial>(p);
    }


    UsdShadeShader surface = mat.ComputeSurfaceSource();
    if (!surface) {
        // No surface shader exported — check if the material name implies glass.
        // Blender's Glass BSDF nodes are not translated by the USD exporter and
        // result in an empty material shell. Detect by name and substitute glass.
        std::string matName = mat.GetPrim().GetName().GetString();
        // Also check blender:data_name attribute which carries the original name
        std::string blenderName;
        UsdAttribute nameAttr = mat.GetPrim().GetAttribute(
            TfToken("userProperties:blender:data_name"));
        if (nameAttr) nameAttr.Get(&blenderName);

        if (isGlassName(matName) || isGlassName(blenderName)) {
            spdlog::info("USDLoader: material '{}' has no surface shader — "
                         "name suggests glass, substituting glass material",
                         mat.GetPath().GetString());
            return makeGlassMaterial(1.5f);
        }
        return std::make_unique<LambertianMaterial>(Spectrum{0.5f, 0.5f, 0.5f});
    }

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
    } else {
        p.transmission = std::max(explicitTransmission, 1.f - opacityVal);
    }

    // Specular: UsdPreviewSurface has an optional inputs:specular weight (0–1,
    // default 0.5 for dielectrics).  Read it if present; otherwise fall back to
    // physically-motivated defaults: 0 for metals (no dielectric layer), 1 for
    // glass (Fresnel must be fully evaluated), 0.5 for ordinary dielectrics.
    {
        UsdShadeInput specIn = surface.GetInput(TfToken("specular"));
        float specularDefault = (p.transmission > 0.001f && p.metalness.value < 0.001f) ? 1.0f
                              : (p.metalness.value > 0.01f ? 0.f : 0.5f);
        p.specular = resolveFloatTOV(specIn, specularDefault, stageDir);
    }

    // Clearcoat
    p.coat           = resolveFloatTOV(surface.GetInput(TfToken("clearcoat")), 0.f, stageDir).value;
    p.coat_roughness = resolveFloatTOV(surface.GetInput(TfToken("clearcoatRoughness")), 0.1f, stageDir).value;

    // Subsurface scattering — UsdPreviewSurface has no official SSS spec but
    // Blender and some DCCs write these as extra inputs using MaterialX names.
    {
        float sssWeight = resolveFloatTOV(
            surface.GetInput(TfToken("subsurface_weight")), 0.0f, stageDir).value;
        if (sssWeight <= 0.f)
            sssWeight = resolveFloatTOV(
                surface.GetInput(TfToken("subsurface")), 0.0f, stageDir).value;
        p.subsurface = sssWeight;

        if (sssWeight > 0.f) {
            p.subsurface_color = resolveColorTOV(
                surface.GetInput(TfToken("subsurface_color")),
                Spectrum{1.f, 1.f, 1.f}, stageDir);

            SpectrumTOV radVec = resolveColorTOV(
                surface.GetInput(TfToken("subsurface_radius")),
                Spectrum{0.1f, 0.1f, 0.1f}, stageDir);
            p.subsurface_radius = (radVec.value.x + radVec.value.y + radVec.value.z) / 3.f;

            p.subsurface_scale = resolveFloatTOV(
                surface.GetInput(TfToken("subsurface_scale")), 1.f, stageDir).value;
            if (p.subsurface_scale <= 0.f) p.subsurface_scale = 1.f;

            p.subsurface_anisotropy = resolveFloatTOV(
                surface.GetInput(TfToken("subsurface_anisotropy")), 0.0f, stageDir).value;

            p.subsurface_strength = resolveFloatTOV(
                surface.GetInput(TfToken("anacapa_subsurface_strength")), 1.0f, stageDir).value;
            if (p.subsurface_strength <= 0.f) p.subsurface_strength = 1.f;
        }
    }

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

    // Caustic opt-in (UsdPreviewSurface): same custom bool as the OpenPBR
    // parser.  Off by default so existing scenes with ordinary glass keep
    // using NEE transmittance even in --integrator photon.
    if (UsdShadeInput causticIn = surface.GetInput(TfToken("anacapa_caustic"))) {
        bool causticVal = false;
        causticIn.Get(&causticVal);
        p.caustic = causticVal;
    }
    if (UsdShadeInput crIn = surface.GetInput(TfToken("anacapa_caustic_radius"))) {
        float crVal = 0.f;
        crIn.Get(&crVal);
        p.caustic_radius = crVal;
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
        double endTime,
        bool zUp = false)
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
        spdlog::debug("USDLoader: collectMotionKeys for '{}': {} raw sample(s): [{}]",
                      prim.GetPath().GetString(), times.size(), timesStr);
    }

    // Clamp to shutter window: discard any samples outside [startTime, endTime],
    // then ensure the endpoints themselves are always present.
    times.erase(std::remove_if(times.begin(), times.end(),
        [startTime, endTime](double t) { return t < startTime || t > endTime; }),
        times.end());
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

        if (zUp) {
            // Incorporate the Z-up → Y-up world-space correction into each motion key.
            // The correction matrix converts Z-up world coords to Y-up: {x, z, -y}.
            // In column-vector (anacapa) convention:
            //   M_corr.row0 = [1, 0,  0, 0]
            //   M_corr.row1 = [0, 0,  1, 0]
            //   M_corr.row2 = [0,-1,  0, 0]
            //   M_corr.row3 = [0, 0,  0, 1]
            // objectToWorld_corrected = M_corr * objectToWorld_usd
            Mat4f o2w = key.objectToWorld;
            Mat4f c;  // Z-up correction matrix
            c.m[0][0]=1; c.m[0][1]=0;  c.m[0][2]=0; c.m[0][3]=0;
            c.m[1][0]=0; c.m[1][1]=0;  c.m[1][2]=1; c.m[1][3]=0;
            c.m[2][0]=0; c.m[2][1]=-1; c.m[2][2]=0; c.m[2][3]=0;
            c.m[3][0]=0; c.m[3][1]=0;  c.m[3][2]=0; c.m[3][3]=1;
            key.objectToWorld = c * o2w;
            key.worldToObject = key.objectToWorld.inverse();
        }

        keys.push_back(key);
    }
    // Log only the shutter-open (t=0) and shutter-close (t=1) keys
    for (const MotionKey& k : keys) {
        if (k.time < 0.001f || (k.time > 0.999f && k.time < 1.001f)) {
            spdlog::info("USDLoader:   key t={:.3f} translate=({:.3f},{:.3f},{:.3f}) "
                         "scale=({:.3f},{:.3f},{:.3f})",
                         k.time,
                         k.objectToWorld.m[0][3], k.objectToWorld.m[1][3], k.objectToWorld.m[2][3],
                         k.objectToWorld.m[0][0], k.objectToWorld.m[1][1], k.objectToWorld.m[2][2]);
        }
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
                         bool zUp = false,
                         std::vector<uint32_t>* outFaceTriStart = nullptr,
                         std::vector<uint32_t>* outFaceTriCount = nullptr) {
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

    // Track which triangles each original face maps to (for GeomSubset splitting)
    std::vector<uint32_t> faceTriStart;
    std::vector<uint32_t> faceTriCount;
    if (outFaceTriStart) faceTriStart.reserve(fvcCounts.size());
    if (outFaceTriCount) faceTriCount.reserve(fvcCounts.size());

    int faceStart = 0;
    for (int fi = 0; fi < (int)fvcCounts.size(); ++fi) {
        int nv = fvcCounts[fi];
        uint32_t trisBefore = static_cast<uint32_t>(desc.indices.size() / 3);
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
                            return safeNormalize({normals[ni][0], normals[ni][1], normals[ni][2]});
                        return transformNormal(xform, normals[ni], zUp);
                    }
                }
                // Compute geometric normal
                Vec3f a = hasMotion
                    ? Vec3f{(float)points[i0][0], (float)points[i0][1], (float)points[i0][2]}
                    : transformPoint(xform, GfVec3d(points[i0]), zUp);
                Vec3f b = hasMotion
                    ? Vec3f{(float)points[i1][0], (float)points[i1][1], (float)points[i1][2]}
                    : transformPoint(xform, GfVec3d(points[i1]), zUp);
                Vec3f c = hasMotion
                    ? Vec3f{(float)points[i2][0], (float)points[i2][1], (float)points[i2][2]}
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
        uint32_t trisAfter = static_cast<uint32_t>(desc.indices.size() / 3);
        if (outFaceTriStart) faceTriStart.push_back(trisBefore);
        if (outFaceTriCount) faceTriCount.push_back(trisAfter - trisBefore);
        faceStart += nv;
    }

    if (desc.positions.empty()) return ~0u;

    if (hasMotion) {
        desc.motionKeys = std::move(motionKeys);
    } else {
        // Static meshes: positions baked to world space. Store the O2W so that
        // OSL position(space="object") can invert it to recover object coords.
        Mat4f o2w = toMat4f(xform);
        if (zUp) {
            // Same Z-up correction applied in collectMotionKeys
            Mat4f c;
            c.m[0][0]=1; c.m[0][1]=0;  c.m[0][2]=0; c.m[0][3]=0;
            c.m[1][0]=0; c.m[1][1]=0;  c.m[1][2]=1; c.m[1][3]=0;
            c.m[2][0]=0; c.m[2][1]=-1; c.m[2][2]=0; c.m[2][3]=0;
            c.m[3][0]=0; c.m[3][1]=0;  c.m[3][2]=0; c.m[3][3]=1;
            o2w = c * o2w;
        }
        desc.staticObjectToWorld = o2w;
        desc.staticWorldToObject = o2w.inverse();
    }

    if (outFaceTriStart) *outFaceTriStart = std::move(faceTriStart);
    if (outFaceTriCount) *outFaceTriCount = std::move(faceTriCount);

    return pool.addMesh(std::move(desc));
}

// ---------------------------------------------------------------------------
// extractSubsetMesh — build a MeshDesc containing only the triangles whose
// original face index appears in faceIndices.
//
// The full triangulated MeshDesc (already expanded to flat vertex arrays) is
// passed in.  We need the original face→triangle mapping, so we also pass the
// per-face triangle counts built during loadMesh triangulation.
// ---------------------------------------------------------------------------
static uint32_t extractSubsetMesh(
    const MeshDesc&                 fullMesh,
    const std::vector<uint32_t>&    faceTriStart,  // first triangle index for each orig face
    const std::vector<uint32_t>&    faceTriCount,  // triangle count for each orig face
    const VtArray<int>&             faceIndices,   // face indices belonging to this subset
    GeometryPool&                   pool)
{
    MeshDesc sub;
    sub.name                 = fullMesh.name;
    sub.motionKeys           = fullMesh.motionKeys;
    sub.staticObjectToWorld  = fullMesh.staticObjectToWorld;
    sub.staticWorldToObject  = fullMesh.staticWorldToObject;

    for (int fi : faceIndices) {
        if (fi < 0 || fi >= (int)faceTriStart.size()) continue;
        uint32_t triStart = faceTriStart[fi];
        uint32_t triCount = faceTriCount[fi];
        for (uint32_t ti = 0; ti < triCount; ++ti) {
            uint32_t srcBase = (triStart + ti) * 3;  // index into full flat arrays
            if (srcBase + 2 >= fullMesh.positions.size()) continue;

            uint32_t dstBase = static_cast<uint32_t>(sub.positions.size());
            sub.positions.push_back(fullMesh.positions[srcBase]);
            sub.positions.push_back(fullMesh.positions[srcBase + 1]);
            sub.positions.push_back(fullMesh.positions[srcBase + 2]);
            if (!fullMesh.normals.empty()) {
                sub.normals.push_back(fullMesh.normals[srcBase]);
                sub.normals.push_back(fullMesh.normals[srcBase + 1]);
                sub.normals.push_back(fullMesh.normals[srcBase + 2]);
            }
            if (!fullMesh.uvs.empty()) {
                sub.uvs.push_back(fullMesh.uvs[srcBase]);
                sub.uvs.push_back(fullMesh.uvs[srcBase + 1]);
                sub.uvs.push_back(fullMesh.uvs[srcBase + 2]);
            }
            sub.indices.push_back(dstBase);
            sub.indices.push_back(dstBase + 1);
            sub.indices.push_back(dstBase + 2);
        }
    }

    if (sub.positions.empty()) return ~0u;
    return pool.addMesh(std::move(sub));
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
                                      xformCache.GetTime());
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
    //
    // USD focalLength is stored in "tenths of scene unit" (= scene_unit * 0.1) per the
    // UsdGeomCamera spec.  We read the raw attribute here rather than using GfCamera's
    // GetFocalLength(), because some pxr versions (≥23.x) convert the stored value to
    // millimetres internally; using that mm value with a /10 divisor would give a focal
    // length in mm/10 = centimetres, not scene units — producing a wildly wrong aperture
    // radius and hence extreme/incorrect depth-of-field blur.
    //
    // Raw path: raw_focalLength * 0.1  →  scene units  (e.g. 0.13 * 0.1 = 0.013 m for a
    // 13 mm lens in a metre-scale scene).  focusDistance is already in scene units.
    // If either fStop or focusDistance is absent or zero, fall back to pinhole (no DoF).
    float apertureRadius = 0.f;
    float focalDistance  = 0.f;

    float fStop = 0.f;
    usdCam.GetFStopAttr().Get(&fStop, UsdTimeCode::Default());
    usdCam.GetFocusDistanceAttr().Get(&focalDistance, UsdTimeCode::Default());

    // Always read the raw focal length attribute (tenths of scene unit → scene unit).
    // Stored on the Camera struct so the --fstop CLI override can compute a correct
    // apertureRadius even when the USD camera itself has no fStop authored.
    float rawFocalLen = 0.f;
    usdCam.GetFocalLengthAttr().Get(&rawFocalLen, UsdTimeCode::Default());
    float focalLen_world = rawFocalLen * 0.1f;  // tenths of scene unit → scene unit

    if (fStop > 0.f && focalDistance > 0.f) {
        apertureRadius = focalLen_world / (2.f * fStop);
        spdlog::info("USDLoader: camera '{}' origin=({:.2f},{:.2f},{:.2f}) fov={:.1f}° "
                     "fStop={:.1f} focusDist={:.3f} apertureR={:.4f}",
                     prim.GetPath().GetString(),
                     origin.x, origin.y, origin.z, vfovDeg,
                     fStop, focalDistance, apertureRadius);
        Camera cam = Camera::makeThinLens(origin, target, up, vfovDeg,
                                          filmWidth, filmHeight,
                                          apertureRadius, focalDistance);
        cam.focalLength = focalLen_world;
        return cam;
    }

    Vec3f fwd = safeNormalize(target - origin);
    spdlog::info("USDLoader: camera '{}' origin=({:.2f},{:.2f},{:.2f}) "
                 "fwd=({:.2f},{:.2f},{:.2f}) fov={:.1f}° (pinhole)",
                 prim.GetPath().GetString(),
                 origin.x, origin.y, origin.z,
                 fwd.x, fwd.y, fwd.z, vfovDeg);

    Camera cam = Camera::makePinhole(origin, target, up, vfovDeg, filmWidth, filmHeight);
    cam.focalLength = focalLen_world;
    return cam;
}

// ---------------------------------------------------------------------------
// loadUSD
// ---------------------------------------------------------------------------
LoadedScene loadUSD(const std::string& path,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    const std::string& cameraOverridePath,
                    double frame,
                    float  shutterOpen,
                    float  shutterClose) {
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

    // Load the MaterialX JSON sidecar produced by blender_prep_for_usd_export.py.
    // For particle prims (GeomPoints / PointInstancer halos) this sidecar acts as
    // a live override: geometry_opacity and emission_color × emission_luminance drive
    // the SoftParticleMaterial created for each halo.  Editing the JSON and
    // re-rendering picks up the changes without touching the .usdc file.
    struct SidecarMat {
        Spectrum emission = {1.f, 1.f, 1.f};  // emission_color × emission_luminance
        float    opacity  = 1.f;               // geometry_opacity
    };
    std::map<std::string, SidecarMat> sidecarMats;
    {
        std::string sidecarPath = path + ".materials.json";
        std::ifstream sidecarFile(sidecarPath);
        if (sidecarFile.is_open()) {
            try {
                nlohmann::json sidecar = nlohmann::json::parse(sidecarFile);
                spdlog::info("USDLoader: MaterialX sidecar loaded — {} material(s) in '{}'",
                             static_cast<int>(sidecar.size()), sidecarPath);
                for (auto& [matPath, matData] : sidecar.items()) {
                    SidecarMat sm;
                    if (matData.contains("root") && matData.contains("nodes")) {
                        std::string rootPath = matData["root"].get<std::string>();
                        auto& nodes = matData["nodes"];
                        if (nodes.contains(rootPath)) {
                            auto& inp = nodes[rootPath]["inputs"];

                            // base_color is the primary particle tint color.
                            if (inp.contains("base_color") && inp["base_color"].is_array()
                                    && inp["base_color"].size() >= 3) {
                                sm.emission = {
                                    inp["base_color"][0].get<float>(),
                                    inp["base_color"][1].get<float>(),
                                    inp["base_color"][2].get<float>()
                                };
                            }
                            // emission_color × emission_luminance overrides base_color
                            // when emission is non-zero (user wants a glowing/bright particle).
                            float lum = inp.contains("emission_luminance")
                                        ? inp["emission_luminance"].get<float>() : 1.f;
                            if (inp.contains("emission_color") && inp["emission_color"].is_array()
                                    && inp["emission_color"].size() >= 3) {
                                float er = inp["emission_color"][0].get<float>();
                                float eg = inp["emission_color"][1].get<float>();
                                float eb = inp["emission_color"][2].get<float>();
                                if (er > 0.f || eg > 0.f || eb > 0.f) {
                                    sm.emission = { er * lum, eg * lum, eb * lum };
                                }
                            }
                            // If color is still zero (both base and emission unset), default
                            // to white so per-particle displayColor tinting shows through.
                            if (sm.emission.x == 0.f && sm.emission.y == 0.f && sm.emission.z == 0.f)
                                sm.emission = {1.f, 1.f, 1.f};

                            if (inp.contains("geometry_opacity"))
                                sm.opacity = inp["geometry_opacity"].get<float>();
                        }
                    }
                    sidecarMats[matPath] = sm;
                    spdlog::debug("USDLoader: sidecar '{}' — color=({:.2f},{:.2f},{:.2f}) opacity={:.2f}",
                                  matPath, sm.emission.x, sm.emission.y, sm.emission.z, sm.opacity);
                }
            } catch (const std::exception& e) {
                spdlog::warn("USDLoader: failed to parse MaterialX sidecar '{}': {}",
                             sidecarPath, e.what());
            }
        }
    }

    const bool zUp = (UsdGeomGetStageUpAxis(stage) == UsdGeomTokens->z);
    spdlog::info("USDLoader: opened '{}' (up-axis: {}{})",
                 path,
                 UsdGeomGetStageUpAxis(stage).GetString(),
                 zUp ? " — applying Y↔Z correction" : "");

    // Use the stage's authored time range to determine startTime (fallback frame).
    double startTime = stage->HasAuthoredTimeCodeRange()
                         ? stage->GetStartTimeCode() : 0.0;

    // Determine which frame to render.  If the caller didn't provide one, fall
    // back to the stage's start time code so static / single-frame scenes work.
    double renderFrame = std::isnan(frame) ? startTime : frame;

    // Motion blur is enabled only when the caller provides an explicit shutter
    // window (shutterClose > shutterOpen) AND the stage has animation data.
    bool enableMotionBlur = (shutterClose > shutterOpen)
                            && stage->HasAuthoredTimeCodeRange();

    // Shutter window in USD time codes (frames).
    double tcOpenVal  = renderFrame + (enableMotionBlur ? static_cast<double>(shutterOpen)  : 0.0);
    double tcCloseVal = renderFrame + (enableMotionBlur ? static_cast<double>(shutterClose) : 0.0);

    // The motion keys stored in each mesh are normalized to [0, 1] relative to
    // [tcOpenVal, tcCloseVal].  Camera ray.time is sampled in [0, 1] so that
    // ray.time=0 → shutter open and ray.time=1 → shutter close.
    result.shutterOpen  = 0.f;
    result.shutterClose = enableMotionBlur ? 1.f : 0.f;

    // Camera pre-pass: if caller didn't provide an explicit shutter window, look
    // for a camera prim with authored shutter:open / shutter:close and use those.
    // This must happen BEFORE the mesh traversal so that tcOpenVal/tcCloseVal and
    // the xform caches are set up correctly for motion-key collection.
    if (!enableMotionBlur && stage->HasAuthoredTimeCodeRange()) {
        std::string prepassCamPath = cameraOverridePath;
        UsdPrim prepassCamPrim;
        if (!prepassCamPath.empty()) {
            prepassCamPrim = stage->GetPrimAtPath(SdfPath(prepassCamPath));
        } else {
            for (const UsdPrim& p : stage->Traverse()) {
                if (p.IsA<UsdGeomCamera>()) { prepassCamPrim = p; break; }
            }
        }
        if (prepassCamPrim) {
            UsdGeomCamera usdCamPre(prepassCamPrim);
            double camOpen = 0.0, camClose = 0.0;
            bool gotOpen  = usdCamPre.GetShutterOpenAttr() .Get(&camOpen,  UsdTimeCode::Default());
            bool gotClose = usdCamPre.GetShutterCloseAttr().Get(&camClose, UsdTimeCode::Default());
            if (gotClose && camClose > camOpen) {
                shutterOpen  = static_cast<float>(camOpen);
                shutterClose = static_cast<float>(camClose);
                enableMotionBlur = true;
                tcOpenVal  = renderFrame + static_cast<double>(shutterOpen);
                tcCloseVal = renderFrame + static_cast<double>(shutterClose);
                result.shutterOpen  = 0.f;
                result.shutterClose = 1.f;
                spdlog::info("USDLoader: camera '{}' has authored shutter [{:.4f}, {:.4f}] — "
                             "enabling motion blur (time codes [{:.3f}, {:.3f}])",
                             prepassCamPrim.GetPath().GetString(),
                             camOpen, camClose, tcOpenVal, tcCloseVal);
            }
        }
    }

    if (enableMotionBlur)
        spdlog::info("USDLoader: motion blur enabled — frame {:.1f}, shutter [{:.3f}, {:.3f}] "
                     "(time codes [{:.3f}, {:.3f}])",
                     renderFrame, shutterOpen, shutterClose, tcOpenVal, tcCloseVal);
    else
        spdlog::info("USDLoader: motion blur disabled (frame {:.1f}{})",
                     renderFrame,
                     stage->HasAuthoredTimeCodeRange() ? "" : " — static scene");

    UsdTimeCode tcOpen{tcOpenVal};
    UsdTimeCode tcClose{tcCloseVal};
    UsdGeomXformCache xformCache{tcOpen};    // shutter-open (or render frame when no motion blur)
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

    // Pre-collect all prim paths that serve as PointInstancer prototypes so we
    // can skip them in the mesh branch (they are loaded with per-instance
    // transforms instead).  Prototypes may live anywhere in the stage, not
    // necessarily under the instancer prim.
    std::unordered_set<std::string> instancerProtoPaths;
    for (const UsdPrim& p : stage->Traverse()) {
        if (!p.IsA<UsdGeomPointInstancer>()) continue;
        SdfPathVector targets;
        UsdGeomPointInstancer(p).GetPrototypesRel().GetTargets(&targets);
        for (const SdfPath& tp : targets) {
            UsdPrim protoPrim = stage->GetPrimAtPath(tp);
            if (!protoPrim.IsValid()) continue;
            for (const UsdPrim& child : UsdPrimRange(protoPrim))
                instancerProtoPaths.insert(child.GetPath().GetString());
            instancerProtoPaths.insert(tp.GetString());
        }
    }

    // --- Traverse all prims ---
    for (const UsdPrim& prim : stage->Traverse()) {

        // ---- Mesh ----
        if (prim.IsA<UsdGeomMesh>()) {
            // Skip meshes that are PointInstancer prototypes — handled below.
            if (instancerProtoPaths.count(prim.GetPath().GetString())) continue;

            UsdGeomMesh usdMesh(prim);

            // Detect animated transforms by comparing the full world transform
            // (including parent hierarchy) at t=0 vs t=1.  This catches motion
            // that lives on a parent Xform prim rather than on the Mesh itself.
            GfMatrix4d xform0 = xformCache.GetLocalToWorldTransform(prim);
            GfMatrix4d xform1 = xformCacheT1.GetLocalToWorldTransform(prim);
            bool hasMotion = (xform0 != xform1);

            std::vector<MotionKey> motionKeys;
            if (enableMotionBlur && hasMotion) {
                motionKeys = collectMotionKeys(prim, tcOpenVal, tcCloseVal, zUp);
                spdlog::info("USDLoader: animated mesh '{}' — {} motion key(s)",
                             prim.GetPath().GetString(), motionKeys.size());
            } else if (enableMotionBlur) {
                spdlog::debug("USDLoader: mesh '{}' has no motion in [{:.3f}, {:.3f}]",
                              prim.GetPath().GetString(), tcOpenVal, tcCloseVal);
            }

            std::vector<uint32_t> faceTriStart, faceTriCount;
            uint32_t meshID = loadMesh(usdMesh, xform0, std::move(motionKeys),
                                       result.geomPool, zUp,
                                       &faceTriStart, &faceTriCount);
            if (meshID == ~0u) {
                spdlog::warn("USDLoader: skipped mesh '{}' (no geometry)",
                             prim.GetPath().GetString());
                continue;
            }

            // Helper: resolve a UsdShadeMaterial to a material index
            auto resolveMaterialIdx = [&](const UsdShadeMaterial& mat) -> uint32_t {
                if (!mat) return kDefaultMatIdx;
                std::string matPath = mat.GetPath().GetString();
                auto it = matPathToIdx.find(matPath);
                if (it != matPathToIdx.end()) return it->second;
                uint32_t idx = static_cast<uint32_t>(result.materials.size());
                result.materials.push_back(resolveMaterial(mat, stageDir));
                matPathToIdx[matPath] = idx;
                return idx;
            };

            // --- GeomSubset per-face material assignment ---
            // When a mesh has GeomSubset children with face-set type, each subset
            // carries its own material binding. We split those faces into separate
            // meshes so each can have the correct material (e.g. glass panes within
            // a larger window frame mesh).
            std::vector<int> faceCoveredBySubset(faceTriStart.size(), 0);
            for (const UsdGeomSubset& subset :
                     UsdGeomSubset::GetGeomSubsets(usdMesh,
                         UsdGeomTokens->face, TfToken()))
            {
                VtArray<int> subFaceIndices;
                subset.GetIndicesAttr().Get(&subFaceIndices);
                if (subFaceIndices.empty()) continue;

                UsdShadeMaterialBindingAPI subBindAPI(subset.GetPrim());
                UsdShadeMaterial subMat = subBindAPI.ComputeBoundMaterial();
                if (!subMat) continue;

                uint32_t subMatIdx = resolveMaterialIdx(subMat);

                // Extract the subset triangles into a new mesh
                const MeshDesc& fullMesh = result.geomPool.mesh(meshID);
                uint32_t subMeshID = extractSubsetMesh(fullMesh, faceTriStart, faceTriCount,
                                                        subFaceIndices, result.geomPool);
                if (subMeshID == ~0u) continue;

                if (subMeshID >= result.sceneView.materials.size())
                    result.sceneView.materials.resize(subMeshID + 1, nullptr);
                result.sceneView.materials[subMeshID] = result.materials[subMatIdx].get();

                for (int fi : subFaceIndices)
                    if (fi >= 0 && fi < (int)faceCoveredBySubset.size())
                        faceCoveredBySubset[fi] = 1;

                spdlog::info("USDLoader: GeomSubset '{}' → meshID={} matIdx={} ({} faces)",
                             subset.GetPrim().GetPath().GetString(),
                             subMeshID, subMatIdx, subFaceIndices.size());
            }

            // Resolve material binding for the whole mesh (faces not in any subset)
            uint32_t matIdx = kDefaultMatIdx;
            UsdShadeMaterialBindingAPI bindAPI(prim);
            UsdShadeMaterial boundMat = bindAPI.ComputeBoundMaterial();
            if (boundMat) {
                matIdx = resolveMaterialIdx(boundMat);
            } else {
                // No material binding — fall back to primvars:displayColor.
                UsdGeomPrimvarsAPI pvAPI2(prim);
                UsdGeomPrimvar dcPrimvar = pvAPI2.GetPrimvar(TfToken("displayColor"));
                if (dcPrimvar) {
                    VtArray<GfVec3f> colors;
                    dcPrimvar.Get(&colors);
                    if (!colors.empty()) {
                        GfVec3f c = colors[0];
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

            // Assign the mesh-level material to the full mesh
            if (meshID >= result.sceneView.materials.size())
                result.sceneView.materials.resize(meshID + 1, nullptr);
            result.sceneView.materials[meshID] = result.materials[matIdx].get();

            // If any subsets were extracted, rebuild the original mesh with only
            // the faces NOT covered by any subset.  Without this the original mesh
            // still contains the subset triangles (e.g. glass panes) assigned to
            // the mesh-level material (e.g. opaque frame), which would block light.
            {
                bool anySubsets = false;
                for (int v : faceCoveredBySubset) if (v) { anySubsets = true; break; }
                if (anySubsets) {
                    const MeshDesc& fullMesh = result.geomPool.mesh(meshID);
                    MeshDesc residual;
                    residual.name                = fullMesh.name;
                    residual.motionKeys          = fullMesh.motionKeys;
                    residual.staticObjectToWorld = fullMesh.staticObjectToWorld;
                    residual.staticWorldToObject = fullMesh.staticWorldToObject;

                    for (int fi = 0; fi < (int)faceCoveredBySubset.size(); ++fi) {
                        if (faceCoveredBySubset[fi]) continue;  // skip — already in a subset mesh
                        uint32_t triStart = faceTriStart[fi];
                        uint32_t triCount = faceTriCount[fi];
                        for (uint32_t ti = 0; ti < triCount; ++ti) {
                            uint32_t srcBase = (triStart + ti) * 3;
                            if (srcBase + 2 >= fullMesh.positions.size()) continue;
                            uint32_t dstBase = static_cast<uint32_t>(residual.positions.size());
                            residual.positions.push_back(fullMesh.positions[srcBase]);
                            residual.positions.push_back(fullMesh.positions[srcBase + 1]);
                            residual.positions.push_back(fullMesh.positions[srcBase + 2]);
                            if (!fullMesh.normals.empty()) {
                                residual.normals.push_back(fullMesh.normals[srcBase]);
                                residual.normals.push_back(fullMesh.normals[srcBase + 1]);
                                residual.normals.push_back(fullMesh.normals[srcBase + 2]);
                            }
                            if (!fullMesh.uvs.empty()) {
                                residual.uvs.push_back(fullMesh.uvs[srcBase]);
                                residual.uvs.push_back(fullMesh.uvs[srcBase + 1]);
                                residual.uvs.push_back(fullMesh.uvs[srcBase + 2]);
                            }
                            residual.indices.push_back(dstBase);
                            residual.indices.push_back(dstBase + 1);
                            residual.indices.push_back(dstBase + 2);
                        }
                    }

                    spdlog::info("USDLoader: mesh '{}' subset residual: {} → {} tris",
                                 fullMesh.name, fullMesh.numTriangles(),
                                 residual.numTriangles());
                    result.geomPool.replaceMesh(meshID, std::move(residual));
                }
            }

            spdlog::debug("USDLoader: mesh '{}' → meshID={} matIdx={}",
                          prim.GetPath().GetString(), meshID, matIdx);
        }

        // ---- PointInstancer (particle / instance scatter) ----
        else if (prim.IsA<UsdGeomPointInstancer>()) {
            UsdGeomPointInstancer instancer(prim);
            GfMatrix4d instancerToWorld = xformCache.GetLocalToWorldTransform(prim);

            // Material resolve helper — shared by both mesh-instance and halo paths.
            auto resolveMaterialIdx = [&](const UsdShadeMaterial& mat) -> uint32_t {
                if (!mat) return kDefaultMatIdx;
                std::string matPath = mat.GetPath().GetString();
                auto it = matPathToIdx.find(matPath);
                if (it != matPathToIdx.end()) return it->second;
                uint32_t idx = static_cast<uint32_t>(result.materials.size());
                result.materials.push_back(resolveMaterial(mat, stageDir));
                matPathToIdx[matPath] = idx;
                return idx;
            };

            // Halo-disc fallback: used when prototypes are missing/cyclic or
            // load 0 meshes (common with Blender 4.x GN simulation exports).
            auto tryHaloFallback = [&]() {
                VtArray<GfVec3f> rawPositions;
                instancer.GetPositionsAttr().Get(&rawPositions, tcOpen);
                if (rawPositions.empty()) {
                    spdlog::warn("USDLoader: instancer '{}' — no positions, skipped",
                                 prim.GetPath().GetString());
                    return;
                }
                VtArray<GfVec3f> rawScales;
                instancer.GetScalesAttr().Get(&rawScales, tcOpen);

                VtArray<GfVec3f> colors;
                UsdGeomPrimvarsAPI pvAPI(prim);
                auto dcPv = pvAPI.GetPrimvar(TfToken("displayColor"));
                if (dcPv) dcPv.ComputeFlattened(&colors, tcOpen);

                UsdShadeMaterialBindingAPI instBindAPI(prim);
                uint32_t instMatIdx =
                    resolveMaterialIdx(instBindAPI.ComputeBoundMaterial());

                int haloCount = 0;
                for (size_t ii = 0; ii < rawPositions.size(); ++ii) {
                    HaloDesc h;
                    GfVec3d pt3 = instancerToWorld.Transform(GfVec3d(rawPositions[ii]));
                    GfVec3f wpt((float)pt3[0], (float)pt3[1], (float)pt3[2]);
                    h.center = zUp ? Vec3f{wpt[0], wpt[2], -wpt[1]}
                                   : Vec3f{wpt[0], wpt[1],  wpt[2]};
                    h.centerClose = h.center;
                    float r = 0.05f;
                    if (ii < rawScales.size()) {
                        float s = std::max({rawScales[ii][0],
                                            rawScales[ii][1],
                                            rawScales[ii][2]});
                        r = std::max(s * 0.5f, 0.001f);
                    }
                    h.radius = r;
                    if (!colors.empty()) {
                        const GfVec3f& c = colors.size() == 1 ? colors[0]
                                         : ii < colors.size()  ? colors[ii]
                                         : colors.back();
                        h.color = {c[0], c[1], c[2]};
                    }
                    h.matIdx = instMatIdx;
                    result.haloPool.addHalo(h);
                    ++haloCount;
                }
                spdlog::info(
                    "USDLoader: instancer '{}' → {} halo discs (no prototype geometry)",
                    prim.GetPath().GetString(), haloCount);
            };

            SdfPathVector protoPaths;
            instancer.GetPrototypesRel().GetTargets(&protoPaths);
            if (protoPaths.empty()) {
                // Cycle or missing reference — prototype resolution failed.
                tryHaloFallback();
                continue;
            }

            VtArray<int> protoIndices;
            instancer.GetProtoIndicesAttr().Get(&protoIndices, tcOpen);
            if (protoIndices.empty()) {
                spdlog::warn("USDLoader: instancer '{}' has no instances",
                             prim.GetPath().GetString());
                continue;
            }

            // Per-instance transforms in instancer-local space, including each
            // prototype root's own local transform (IncludeProtoXform).
            // Multiplying by instancerToWorld gives the proto-root world transform
            // for each instance.
            VtArray<GfMatrix4d> instanceXforms;
            if (!instancer.ComputeInstanceTransformsAtTime(
                    &instanceXforms, tcOpen, tcOpen,
                    UsdGeomPointInstancer::IncludeProtoXform)) {
                spdlog::warn("USDLoader: instancer '{}' — failed to compute transforms",
                             prim.GetPath().GetString());
                tryHaloFallback();
                continue;
            }
            if (instanceXforms.size() != protoIndices.size()) continue;

            // Gather mesh prims per prototype with each mesh's transform relative
            // to the prototype root.  A single prototype may contain multiple
            // child meshes (e.g. a small hierarchy of leaf parts).
            struct ProtoMeshEntry { UsdGeomMesh usdMesh; GfMatrix4d relToRoot; };
            std::vector<std::vector<ProtoMeshEntry>> protoMeshes(protoPaths.size());
            for (size_t pi = 0; pi < protoPaths.size(); ++pi) {
                UsdPrim protoPrim = stage->GetPrimAtPath(protoPaths[pi]);
                if (!protoPrim.IsValid()) continue;
                GfMatrix4d protoRootWorldInv =
                    xformCache.GetLocalToWorldTransform(protoPrim).GetInverse();
                for (const UsdPrim& child : UsdPrimRange(protoPrim)) {
                    if (!child.IsA<UsdGeomMesh>()) continue;
                    GfMatrix4d childWorld = xformCache.GetLocalToWorldTransform(child);
                    protoMeshes[pi].push_back(
                        {UsdGeomMesh(child), protoRootWorldInv * childWorld});
                }
            }

            int loadedCount = 0;
            for (size_t ii = 0; ii < instanceXforms.size(); ++ii) {
                int pi = protoIndices[ii];
                if (pi < 0 || pi >= (int)protoPaths.size()) continue;

                // World transform of the prototype root for this instance.
                GfMatrix4d instanceProtoWorld = instancerToWorld * instanceXforms[ii];

                for (auto& pm : protoMeshes[pi]) {
                    GfMatrix4d meshWorld = instanceProtoWorld * pm.relToRoot;
                    uint32_t meshID = loadMesh(pm.usdMesh, meshWorld, {}, result.geomPool, zUp);
                    if (meshID == ~0u) continue;

                    if (meshID >= result.sceneView.materials.size())
                        result.sceneView.materials.resize(meshID + 1, nullptr);

                    UsdShadeMaterialBindingAPI bindAPI(pm.usdMesh.GetPrim());
                    UsdShadeMaterial boundMat = bindAPI.ComputeBoundMaterial();
                    result.sceneView.materials[meshID] =
                        result.materials[resolveMaterialIdx(boundMat)].get();
                    ++loadedCount;
                }
            }

            if (loadedCount > 0) {
                spdlog::info(
                    "USDLoader: instancer '{}' → {} mesh copies "
                    "({} instances, {} prototype(s))",
                    prim.GetPath().GetString(), loadedCount,
                    instanceXforms.size(), protoPaths.size());
            } else {
                tryHaloFallback();
            }
        }

        // ---- Points (camera-facing halo disc particles) ----
        else if (prim.IsA<UsdGeomPoints>()) {
            UsdGeomPoints pts(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            VtArray<GfVec3f> points;
            pts.GetPointsAttr().Get(&points, tcOpen);
            if (points.empty()) continue;

            // widths is diameter per-point; radius = width/2
            VtArray<float> widths;
            pts.GetWidthsAttr().Get(&widths, tcOpen);

            // Per-point display color from primvars:displayColor
            VtArray<GfVec3f> colors;
            UsdGeomPrimvarsAPI pvAPI(prim);
            auto dcPv = pvAPI.GetPrimvar(TfToken("displayColor"));
            if (dcPv) dcPv.ComputeFlattened(&colors, tcOpen);

            // Halo particles always use SoftParticleMaterial.
            // sidecarMats drives emission and opacity; per-particle color tinting
            // comes from hd.color (displayColor) at render time.
            uint32_t matIdx = kDefaultMatIdx;
            {
                SoftParticleMaterial::Params spp;
                std::string cacheKey = "softparticle::default";

                // Find the sidecar material for this particle prim.
                // 1. Try ComputeBoundMaterial() — works when the USD has an explicit binding.
                // 2. If that fails, infer by path convention:
                //    prim "{root}/{ObjName}_gn_particles" → look for sidecar keys under
                //    "{root}/{ObjName}/_materials/" (Blender USD exporter convention).
                // 3. Fallback: use the first available sidecar entry (single-particle scenes).
                auto applySidecar = [&](const std::string& matPath) {
                    cacheKey = "softparticle::" + matPath;
                    auto sit = sidecarMats.find(matPath);
                    if (sit != sidecarMats.end()) {
                        spp.color            = sit->second.emission;
                        spp.emissionStrength = 1.f;
                        spp.opacity          = sit->second.opacity;
                        spdlog::info("USDLoader: particle sidecar → color=({:.2f},{:.2f},{:.2f}) opacity={:.2f}",
                                     spp.color.x, spp.color.y, spp.color.z, spp.opacity);
                        return true;
                    }
                    return false;
                };

                bool found = false;
                UsdShadeMaterialBindingAPI bindAPI(prim);
                UsdShadeMaterial boundMat = bindAPI.ComputeBoundMaterial();
                if (boundMat) {
                    found = applySidecar(boundMat.GetPath().GetString());
                }

                if (!found && !sidecarMats.empty()) {
                    // Path-convention fallback: strip "_gn_particles" suffix to recover
                    // the Blender object name, then search for "/{objName}/_materials/".
                    std::string primName = prim.GetName();
                    std::string parentPath = prim.GetParent().GetPath().GetString();
                    const std::string gnSuffix = "_gn_particles";
                    std::string prefix;
                    if (primName.size() > gnSuffix.size() &&
                        primName.compare(primName.size() - gnSuffix.size(),
                                         gnSuffix.size(), gnSuffix) == 0) {
                        std::string objName = primName.substr(0, primName.size() - gnSuffix.size());
                        prefix = parentPath + "/" + objName + "/_materials/";
                    }

                    for (auto& [k, v] : sidecarMats) {
                        if (!prefix.empty() && k.compare(0, prefix.size(), prefix) == 0) {
                            found = applySidecar(k);
                            break;
                        }
                    }
                    // Last resort: use first sidecar entry.
                    if (!found) {
                        found = applySidecar(sidecarMats.begin()->first);
                    }
                }

                if (!found)
                    spdlog::info("USDLoader: GeomPoints '{}' — no sidecar material, using defaults",
                                 prim.GetPath().GetString());

                auto it = matPathToIdx.find(cacheKey);
                if (it != matPathToIdx.end()) {
                    matIdx = it->second;
                } else {
                    matIdx = static_cast<uint32_t>(result.materials.size());
                    result.materials.push_back(std::make_unique<SoftParticleMaterial>(spp));
                    matPathToIdx[cacheKey] = matIdx;
                }
            }

            // Motion blur: read anacapa:closePositions if present.
            // Not gated on enableMotionBlur — the attribute is written by our
            // Python exporter whenever shutter_close > 0, regardless of whether
            // the stage has a full authored time range.
            VtArray<GfVec3f> closePoints;
            {
                auto closeAttr = prim.GetAttribute(TfToken("anacapa:closePositions"));
                if (closeAttr)
                    closeAttr.Get(&closePoints);
            }

            uint32_t addedCount = 0;
            for (size_t i = 0; i < points.size(); ++i) {
                HaloDesc h;
                GfVec3d pt3 = xform.Transform(GfVec3d(points[i]));
                GfVec3f wpt((float)pt3[0], (float)pt3[1], (float)pt3[2]);
                h.center = zUp ? Vec3f{wpt[0], wpt[2], -wpt[1]}
                               : Vec3f{wpt[0], wpt[1],  wpt[2]};

                if (!closePoints.empty() && i < closePoints.size()) {
                    GfVec3d cp3 = xform.Transform(GfVec3d(closePoints[i]));
                    GfVec3f cwpt((float)cp3[0], (float)cp3[1], (float)cp3[2]);
                    h.centerClose = zUp ? Vec3f{cwpt[0], cwpt[2], -cwpt[1]}
                                        : Vec3f{cwpt[0], cwpt[1],  cwpt[2]};
                } else {
                    h.centerClose = h.center;
                }

                float w = widths.empty()           ? 0.02f
                        : widths.size() == 1       ? widths[0]
                        : i < widths.size()        ? widths[i]
                        : widths.back();
                h.radius = w * 0.5f;

                if (!colors.empty()) {
                    const GfVec3f& c = colors.size() == 1 ? colors[0]
                                     : i < colors.size()  ? colors[i]
                                     : colors.back();
                    h.color = {c[0], c[1], c[2]};
                }
                h.matIdx = matIdx;
                result.haloPool.addHalo(h);
                ++addedCount;
            }
            spdlog::info("USDLoader: GeomPoints '{}' → {} halo discs",
                         prim.GetPath().GetString(), addedCount);
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

            // Blender always exports normalize=true with intensity = energy/π.
            // To recover radiance: Le = intensity/area (Power = Le*area*π = energy).
            // normalize=false: intensity is raw radiance.
            float lightArea = cross(uHalf, vHalf).length() * 4.f;
            float leScale = normalize ? (1.f / lightArea) : 1.f;
            Spectrum Le = {color[0] * intensity * leScale,
                           color[1] * intensity * leScale,
                           color[2] * intensity * leScale};

            // Swap u/v so cross(uHalf,vHalf) = -Z local = USD emit direction
            auto light = std::make_unique<AreaLight>(center, vHalf, uHalf, Le);
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

        // ---- DiskLight — approximate as a square area light with matching area ----
        else if (prim.IsA<UsdLuxDiskLight>()) {
            UsdLuxDiskLight disk(prim);
            UsdLuxLightAPI lightAPI(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float radius = 0.5f;
            disk.GetRadiusAttr().Get(&radius);

            float intensity = resolveIntensity(lightAPI);
            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);

            bool normalize = false;
            lightAPI.GetNormalizeAttr().Get(&normalize);

            // Disk center and orientation in world space.
            // USD DiskLight lies in the XY plane, emitting along -Z local.
            Vec3f center = transformPoint(xform, GfVec3d(0, 0, 0), zUp);
            // Two orthogonal radii vectors on the disk plane (X and Y local axes).
            Vec3f xEdge = transformPoint(xform, GfVec3d(radius, 0, 0), zUp) - center;
            Vec3f yEdge = transformPoint(xform, GfVec3d(0, radius, 0), zUp) - center;

            // Same Blender convention: intensity=energy/π, normalize=true.
            float diskArea = 3.14159265f * radius * radius;
            float leScale = normalize ? (1.f / diskArea) : 1.f;
            Spectrum Le = {color[0] * intensity * leScale,
                           color[1] * intensity * leScale,
                           color[2] * intensity * leScale};

            // Approximate disk as an axis-aligned square with matching area (pi*r²).
            // Side length s so s²=pi*r² → s = r*sqrt(pi). Use half-extents = r*sqrt(pi)/2.
            float halfSide = radius * 0.8862f;  // sqrt(pi)/2 ≈ 0.8862
            Vec3f uDir = xEdge.lengthSq() > 1e-12f ? xEdge * (halfSide / xEdge.length()) : Vec3f{halfSide, 0, 0};
            Vec3f vDir = yEdge.lengthSq() > 1e-12f ? yEdge * (halfSide / yEdge.length()) : Vec3f{0, 0, halfSide};

            // Swap u/v so cross(uDir,vDir) = -Z local = USD emit direction
            auto light = std::make_unique<AreaLight>(center, vDir, uDir, Le);
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));

            spdlog::info("USDLoader: diskLight '{}' r={:.3f} Le=({:.3f},{:.3f},{:.3f})",
                          prim.GetPath().GetString(), radius, Le.x, Le.y, Le.z);
        }

        // ---- DistantLight (sun/directional) ----
        else if (prim.IsA<UsdLuxDistantLight>()) {
            UsdLuxDistantLight dist(prim);
            UsdLuxLightAPI lightAPI(prim);
            GfMatrix4d xform = xformCache.GetLocalToWorldTransform(prim);

            float intensity = resolveIntensity(lightAPI);
            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);

            // Blender exports normalize=true with intensity = energy/π (same convention
            // as area lights).  For a directional light there is no area factor to cancel
            // the π, so we must multiply by π to recover the irradiance (W/m²).
            bool normalize = false;
            lightAPI.GetNormalizeAttr().Get(&normalize);
            float leScale = normalize ? 3.14159265f : 1.f;

            // DistantLight emits along -Z local; dirToLight = +Z local in world
            Vec3f lightPos  = transformPoint(xform, GfVec3d(0, 0, 0), zUp);
            Vec3f lightPosZ = transformPoint(xform, GfVec3d(0, 0, 1), zUp);
            Vec3f dirToLight = safeNormalize(lightPosZ - lightPos);

            Spectrum Le = { color[0] * intensity * leScale,
                            color[1] * intensity * leScale,
                            color[2] * intensity * leScale };

            // Bounds needed for disk placement — use placeholder; updated below
            auto light = std::make_unique<DirectionalLight>(
                dirToLight, Le, /*sceneRadius=*/1.f, Vec3f{});
            result.sceneView.lights.push_back(light.get());
            result.lights.push_back(std::move(light));

            spdlog::info("USDLoader: distantLight '{}' dir=({:.2f},{:.2f},{:.2f}) intensity={:.2f} normalize={}",
                         prim.GetPath().GetString(),
                         dirToLight.x, dirToLight.y, dirToLight.z, intensity, normalize);
        }

        // ---- DomeLight or Nishita sky ----
        else if (prim.IsA<UsdLuxDomeLight>()) {
            UsdLuxDomeLight dome(prim);
            UsdLuxLightAPI lightAPI(prim);

            // Check for anacapa:sky:type = "nishita" custom attribute.
            // When present, build a SkyLight instead of loading an HDRI.
            {
                UsdAttribute skyTypeAttr = prim.GetAttribute(TfToken("anacapa:sky:type"));
                std::string skyType;
                if (skyTypeAttr && skyTypeAttr.Get(&skyType) && skyType == "nishita") {
                    NishitaParams sp;

                    // sun_elevation + sun_azimuth → sunDir
                    float elevDeg = 45.f, azimDeg = 180.f;
                    {
                        UsdAttribute a = prim.GetAttribute(TfToken("anacapa:sky:sun_elevation"));
                        if (a) a.Get(&elevDeg);
                    }
                    {
                        UsdAttribute a = prim.GetAttribute(TfToken("anacapa:sky:sun_azimuth"));
                        if (a) a.Get(&azimDeg);
                    }
                    constexpr float kDeg2Rad = 3.14159265f / 180.f;
                    float elev = elevDeg * kDeg2Rad;
                    float az   = azimDeg * kDeg2Rad;
                    sp.sunDir = normalize(Vec3f{
                        std::cos(elev) * std::sin(az),
                        std::sin(elev),
                        std::cos(elev) * std::cos(az)
                    });

                    auto readF = [&](const char* name, float& dst) {
                        UsdAttribute a = prim.GetAttribute(TfToken(name));
                        if (a) a.Get(&dst);
                    };
                    readF("anacapa:sky:sun_intensity",  sp.sunIntensity);
                    float discDeg = 2.0f;
                    readF("anacapa:sky:sun_disc_size",  discDeg);
                    sp.sunDiscAngle = discDeg * kDeg2Rad;
                    readF("anacapa:sky:altitude",       sp.altitude);
                    readF("anacapa:sky:air_density",    sp.airDensity);
                    readF("anacapa:sky:dust_density",   sp.dustDensity);
                    readF("anacapa:sky:ozone_density",  sp.ozoneDensity);
                    {
                        UsdAttribute a = prim.GetAttribute(TfToken("anacapa:sky:transparent_bg"));
                        int v = 0;
                        if (a) { a.Get(&v); sp.transparentBg = (v != 0); }
                    }

                    // Bounds placeholder — updated after all meshes are loaded
                    auto skyLight = std::make_unique<SkyLight>(sp, /*radius=*/1.f, Vec3f{});
                    result.sceneView.envLight = skyLight.get();
                    result.sceneView.lights.push_back(skyLight.get());
                    result.lights.push_back(std::move(skyLight));

                    spdlog::info("USDLoader: Nishita sky '{}' elev={:.1f}° az={:.1f}°",
                                 prim.GetPath().GetString(), elevDeg, azimDeg);
                    continue;  // skip DomeLight construction
                }
            }

            float intensity = resolveIntensity(lightAPI);
            GfVec3f color{1.f, 1.f, 1.f};
            lightAPI.GetColorAttr().Get(&color);
            float effectiveIntensity = intensity * (color[0]+color[1]+color[2]) / 3.f;

            // Check for a texture file
            std::string texturePath;
            SdfAssetPath ap;
            if (dome.GetTextureFileAttr().Get(&ap)) {
                texturePath = ap.GetResolvedPath().empty()
                            ? ap.GetAssetPath()
                            : ap.GetResolvedPath();

                // Blender may export environment textures with a <UDIM> token when
                // the filename ends in _NNNN (e.g. sky_1920.jpg → sky_<UDIM>.jpg).
                // USD's resolver cannot expand UDIM for DomeLights, so we recover the
                // actual file by scanning the directory for a match.
                if (texturePath.find("<UDIM>") != std::string::npos) {
                    // Make the path absolute relative to the USD stage directory.
                    namespace fs = std::filesystem;
                    fs::path raw(texturePath);
                    if (raw.is_relative())
                        raw = fs::path(stageDir) / raw;

                    // Build a glob prefix/suffix around the <UDIM> token.
                    std::string stem = raw.string();
                    auto udimPos = stem.find("<UDIM>");
                    std::string prefix = stem.substr(0, udimPos);
                    std::string suffix = stem.substr(udimPos + 6); // len("<UDIM>") = 6

                    std::string found;
                    std::error_code ec;
                    for (auto& entry : fs::directory_iterator(raw.parent_path(), ec)) {
                        std::string s = entry.path().string();
                        if (s.substr(0, prefix.size()) == prefix &&
                            s.size() >= prefix.size() + suffix.size() &&
                            s.substr(s.size() - suffix.size()) == suffix) {
                            found = s;
                            break;
                        }
                    }
                    if (!found.empty()) {
                        spdlog::info("USDLoader: domeLight UDIM path '{}' resolved to '{}'",
                                     texturePath, found);
                        texturePath = found;
                    } else {
                        spdlog::warn("USDLoader: domeLight UDIM path '{}' — no matching "
                                     "file found in '{}'", texturePath,
                                     raw.parent_path().string());
                    }
                } else if (!texturePath.empty()) {
                    // Resolve relative paths against the stage directory.
                    namespace fs = std::filesystem;
                    fs::path p(texturePath);
                    if (p.is_relative())
                        texturePath = (fs::path(stageDir) / p).string();
                }
            }

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

        // Camera motion blur: if the shutter window is active, read the close-state
        // transform and populate the Camera's motion fields.
        if (enableMotionBlur) {
            Camera camClose = buildCamera(selectedCamPrim, xformCacheT1, filmWidth, filmHeight, zUp);
            Camera& cam     = *result.camera;
            if (camClose.origin.x != cam.origin.x ||
                camClose.origin.y != cam.origin.y ||
                camClose.origin.z != cam.origin.z ||
                camClose.lowerLeftCorner.x != cam.lowerLeftCorner.x) {
                cam.originClose      = camClose.origin;
                cam.lowerLeftClose   = camClose.lowerLeftCorner;
                cam.horizontalClose  = camClose.horizontal;
                cam.verticalClose    = camClose.vertical;
                cam.hasMotion        = true;
                spdlog::info("USDLoader: camera motion blur — open=({:.3f},{:.3f},{:.3f}) "
                             "close=({:.3f},{:.3f},{:.3f})",
                             cam.origin.x, cam.origin.y, cam.origin.z,
                             cam.originClose.x, cam.originClose.y, cam.originClose.z);
            }
        }

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
        // Only apply the camera's authored shutter when the caller didn't request
        // an explicit motion blur window.  When motion blur is enabled via the
        // --shutter-open/close flags the motion keys are already normalized to
        // [0, 1] over that window and result.shutterClose=1; the camera's own
        // attribute must not override that range.
        if (!enableMotionBlur && gotClose && camShutterClose > camShutterOpen) {
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

    // Wire halo materials into sceneView.materials.
    // During traversal, hd.matIdx was assigned as an index into result.materials
    // (the unique_ptr vector).  sceneView.materials is indexed by meshID, so halo
    // materials must live AFTER all the mesh slots.  Remap now that numMeshes is final.
    {
        std::unordered_map<uint32_t, uint32_t> matIdxRemap; // result.materials idx → sceneView idx
        for (auto& h : result.haloPool.halos()) {
            uint32_t oldIdx = h.matIdx;
            if (matIdxRemap.count(oldIdx)) continue;
            if (oldIdx < result.materials.size() && result.materials[oldIdx]) {
                uint32_t svIdx = static_cast<uint32_t>(result.sceneView.materials.size());
                result.sceneView.materials.push_back(result.materials[oldIdx].get());
                matIdxRemap[oldIdx] = svIdx;
            }
        }
        for (auto& h : result.haloPool.halos())
            if (matIdxRemap.count(h.matIdx))
                h.matIdx = matIdxRemap[h.matIdx];
        if (!matIdxRemap.empty())
            spdlog::info("USDLoader: wired {} halo material(s) into sceneView at indices {}..{}",
                         matIdxRemap.size(),
                         result.geomPool.numMeshes(),
                         static_cast<uint32_t>(result.sceneView.materials.size()) - 1);
    }

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
                } else if (auto* sky = dynamic_cast<SkyLight*>(lightPtr.get())) {
                    sky->setSceneRadius(radius);
                    sky->setSceneCenter(center);
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

            // Only StandardSurface materials can be analytically detected as emissive
            // at load time.  OSL emitters contribute via Le() during path tracing and
            // do not need explicit NEE registration here.
            const auto* smat = dynamic_cast<const StandardSurfaceMaterial*>(imat);
            if (!smat) continue;
            const StandardSurfaceMaterial::Params& sp = smat->params();
            if (sp.emission <= 0.f) continue;
            Spectrum Le = sp.emission_color.value * sp.emission;
            if (isBlack(Le)) {
                if (!sp.emission_color.path.empty())
                    Le = {sp.emission * 1.f, sp.emission * 0.7f, sp.emission * 0.3f};
                else
                    continue;
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

    // Expose material path → index map for Alembic hair material assignment.
    // Start with the mesh-bound materials already indexed above, then scan for
    // any remaining UsdShadeMaterial prims that weren't bound to a mesh (e.g.
    // hair materials assigned only via the Alembic sidecar, not to any geometry).
    for (const UsdPrim& prim : stage->Traverse()) {
        if (!prim.IsA<UsdShadeMaterial>()) continue;
        std::string matPath = prim.GetPath().GetString();
        if (matPathToIdx.count(matPath)) continue;  // already indexed
        UsdShadeMaterial mat(prim);
        uint32_t idx = static_cast<uint32_t>(result.materials.size());
        auto m = resolveMaterial(mat, stageDir);
        if (!m) continue;
        result.materials.push_back(std::move(m));
        matPathToIdx[matPath] = idx;
        // Mirror into sceneView so the integrator can find it via meshID
        result.sceneView.materials.push_back(result.materials.back().get());
        spdlog::info("USDLoader: indexed unbound material '{}' (idx={})", matPath, idx);
    }
    result.materialPathIndex = matPathToIdx;

    return result;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_USD
