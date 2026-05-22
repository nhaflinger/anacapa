#include "SceneExporter.h"
#include "SceneFormat.h"

#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/tf/token.h>

#include <spdlog/spdlog.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>

PXR_NAMESPACE_USING_DIRECTIVE

namespace anacapa {

namespace {

// ---------------------------------------------------------------------------
// Binary reader helpers
// ---------------------------------------------------------------------------

struct Reader {
    const uint8_t* data;
    size_t         size;
    size_t         pos = 0;

    bool ok() const { return pos <= size; }

    bool read(void* dst, size_t n) {
        if (pos + n > size) return false;
        std::memcpy(dst, data + pos, n);
        pos += n;
        return true;
    }

    template<typename T>
    bool read(T& v) { return read(&v, sizeof(T)); }

    bool readStr(std::string& s) {
        uint32_t len = 0;
        if (!read(len)) return false;
        s.resize(len);
        return read(s.data(), len);
    }

    bool readFloats(std::vector<float>& v, size_t count) {
        v.resize(count);
        return read(v.data(), count * sizeof(float));
    }

    bool readUints(std::vector<uint32_t>& v, size_t count) {
        v.resize(count);
        return read(v.data(), count * sizeof(uint32_t));
    }
};

// ---------------------------------------------------------------------------
// Sanitise a name for use as a USD prim path component.
// Replaces any character that isn't alphanumeric or underscore with '_'.
// ---------------------------------------------------------------------------
static std::string sanitize(const std::string& s) {
    std::string r = s;
    for (char& c : r)
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_')
            c = '_';
    if (!r.empty() && std::isdigit(static_cast<unsigned char>(r[0])))
        r = "_" + r;
    return r;
}

// ---------------------------------------------------------------------------
// Build a GfMatrix4d from the 16 floats written by Python (row-major,
// column-vector convention).  USD GfMatrix4d is also row-major but uses
// row-vector convention, so we transpose.
// ---------------------------------------------------------------------------
static GfMatrix4d matFromBlob(const float m[16]) {
    // m[row*4+col] in column-vector convention
    // GfMatrix4d[row][col] in row-vector convention
    // Transpose: swap row and col indices.
    return GfMatrix4d(
        m[ 0], m[ 4], m[ 8], m[12],
        m[ 1], m[ 5], m[ 9], m[13],
        m[ 2], m[ 6], m[10], m[14],
        m[ 3], m[ 7], m[11], m[15]
    );
}

// ---------------------------------------------------------------------------
// Ensure a unique prim path within the given parent.
// ---------------------------------------------------------------------------
static SdfPath uniquePath(const SdfPath& parent, const std::string& base,
                          std::unordered_map<std::string, int>& counts) {
    std::string s = sanitize(base);
    if (s.empty()) s = "prim";
    auto it = counts.find(s);
    if (it == counts.end()) {
        counts[s] = 0;
        return parent.AppendChild(TfToken(s));
    }
    counts[s]++;
    return parent.AppendChild(TfToken(s + "_" + std::to_string(counts[s])));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// exportSceneToUsd
// ---------------------------------------------------------------------------
bool exportSceneToUsd(const std::string& blobPath, const std::string& usdPath) {

    // --- Load binary blob ---
    FILE* fp = std::fopen(blobPath.c_str(), "rb");
    if (!fp) {
        spdlog::error("SceneExporter: cannot open blob '{}'", blobPath);
        return false;
    }
    std::fseek(fp, 0, SEEK_END);
    size_t fsize = static_cast<size_t>(std::ftell(fp));
    std::rewind(fp);
    std::vector<uint8_t> buf(fsize);
    std::fread(buf.data(), 1, fsize, fp);
    std::fclose(fp);

    Reader r{buf.data(), buf.size()};

    // --- Validate header ---
    scene_fmt::SceneHeader hdr{};
    if (!r.read(hdr)) {
        spdlog::error("SceneExporter: truncated header");
        return false;
    }
    if (hdr.magic != scene_fmt::kMagic) {
        spdlog::error("SceneExporter: bad magic {:08x}", hdr.magic);
        return false;
    }
    if (hdr.version != scene_fmt::kVersion) {
        spdlog::error("SceneExporter: unsupported version {}", hdr.version);
        return false;
    }

    spdlog::info("SceneExporter: {} meshes, {} lights, camera={}",
                 hdr.num_meshes, hdr.num_lights, hdr.has_camera ? "yes" : "no");

    // --- Create USD stage ---
    auto stage = UsdStage::CreateNew(usdPath);
    if (!stage) {
        spdlog::error("SceneExporter: cannot create stage '{}'", usdPath);
        return false;
    }
    UsdGeomSetStageUpAxis(stage, UsdGeomTokens->y);

    SdfPath rootPath("/root");
    stage->DefinePrim(rootPath);

    SdfPath meshRoot  = rootPath.AppendChild(TfToken("Meshes"));
    SdfPath lightRoot = rootPath.AppendChild(TfToken("Lights"));
    SdfPath matRoot   = rootPath.AppendChild(TfToken("Materials"));
    SdfPath camRoot   = rootPath.AppendChild(TfToken("Cameras"));
    stage->DefinePrim(meshRoot);
    stage->DefinePrim(lightRoot);
    stage->DefinePrim(matRoot);
    stage->DefinePrim(camRoot);

    // Track authored time code range across all motion-blur meshes.
    // stage->HasAuthoredTimeCodeRange() must return true or USDLoader disables
    // motion blur regardless of time-sampled xformOps being present.
    double stageMinTime = std::numeric_limits<double>::max();
    double stageMaxTime = std::numeric_limits<double>::lowest();

    // Map material name → USD material path (created on demand)
    std::unordered_map<std::string, SdfPath> matPaths;
    std::unordered_map<std::string, int>     meshNames, lightNames;

    auto getOrCreateMaterial = [&](const std::string& matName,
                                    bool isTranslucent,
                                    const float* transColor,
                                    bool hasSss,
                                    const float* sssData) -> SdfPath {
        auto it = matPaths.find(matName);
        if (it != matPaths.end()) return it->second;

        std::string safe = sanitize(matName);
        if (safe.empty()) safe = "Material";
        SdfPath mp = matRoot.AppendChild(TfToken(safe));

        UsdShadeMaterial mat = UsdShadeMaterial::Define(stage, mp);

        // The USD loader resolves the .osl shader by looking up
        // userProperties:blender:data_name on the material prim.
        UsdAttribute nameAttr = mat.GetPrim().CreateAttribute(
            TfToken("userProperties:blender:data_name"),
            SdfValueTypeNames->String, false);
        nameAttr.Set(matName);

        // For pure translucent materials, write a custom attribute so
        // resolveMaterial() in USDLoader can create a correct StandardSurface
        // translucency material without needing embedded shader nodes.
        if (isTranslucent) {
            mat.GetPrim().CreateAttribute(
                TfToken("anacapa:translucency:enabled"),
                SdfValueTypeNames->Bool, false).Set(true);
            mat.GetPrim().CreateAttribute(
                TfToken("anacapa:translucency:color"),
                SdfValueTypeNames->Color3f, false)
                .Set(GfVec3f(transColor[0], transColor[1], transColor[2]));
        } else {
            // For all non-translucent materials create a UsdPreviewSurface so
            // USDLoader's StandardSurface fallback (used when OSL is off or absent)
            // can pick up the correct diffuseColor instead of defaulting to white.
            // transColor here carries the Principled BSDF base_color exported from
            // Blender; for SSS materials it is the base colour before SSS blending.
            SdfPath previewPath = mp.AppendChild(TfToken("Preview"));
            UsdShadeShader preview = UsdShadeShader::Define(stage, previewPath);
            preview.CreateIdAttr(VtValue(TfToken("UsdPreviewSurface")));
            preview.CreateInput(TfToken("diffuseColor"), SdfValueTypeNames->Color3f)
                .Set(GfVec3f(transColor[0], transColor[1], transColor[2]));

            // SSS params — sssData layout:
            // [weight, cr, cg, cb, rr, rg, rb, scale, anisotropy]
            if (hasSss && sssData) {
                preview.CreateInput(TfToken("subsurface_weight"), SdfValueTypeNames->Float)
                    .Set(sssData[0]);
                preview.CreateInput(TfToken("subsurface_color"), SdfValueTypeNames->Color3f)
                    .Set(GfVec3f(sssData[1], sssData[2], sssData[3]));
                // subsurface_radius is per-channel (color3f); USDLoader averages them
                preview.CreateInput(TfToken("subsurface_radius"), SdfValueTypeNames->Color3f)
                    .Set(GfVec3f(sssData[4], sssData[5], sssData[6]));
                preview.CreateInput(TfToken("subsurface_scale"), SdfValueTypeNames->Float)
                    .Set(sssData[7]);
                preview.CreateInput(TfToken("subsurface_anisotropy"), SdfValueTypeNames->Float)
                    .Set(sssData[8]);
            }

            // Connect surface output so ComputeSurfaceSource() can find this node
            mat.CreateSurfaceOutput().ConnectToSource(
                preview.CreateOutput(TfToken("surface"), SdfValueTypeNames->Token));
        }

        matPaths[matName] = mp;
        return mp;
    };

    // --- Write meshes ---
    for (uint32_t mi = 0; mi < hdr.num_meshes; ++mi) {
        scene_fmt::MeshHeader mhdr{};
        if (!r.read(mhdr)) {
            spdlog::error("SceneExporter: truncated mesh header {}", mi);
            return false;
        }

        // Motion keys: matrices then time codes
        std::vector<float> matrices, timeCodes;
        if (!r.readFloats(matrices,   mhdr.num_motion_keys * 16)) return false;
        if (!r.readFloats(timeCodes,  mhdr.num_motion_keys))      return false;

        std::string name;
        if (!r.readStr(name)) return false;

        std::vector<std::string> matNames(mhdr.num_mat_slots);
        for (auto& mn : matNames)
            if (!r.readStr(mn)) return false;

        std::vector<uint32_t> matFlags;
        std::vector<float>    matColors;
        std::vector<float>    matSssData;
        if (!r.readUints(matFlags,   mhdr.num_mat_slots))       return false;
        if (!r.readFloats(matColors, mhdr.num_mat_slots * 3))   return false;
        if (!r.readFloats(matSssData, mhdr.num_mat_slots * 9))  return false;

        std::vector<float> positions, normals;
        if (!r.readFloats(positions, mhdr.num_verts * 3)) return false;
        if (!r.readFloats(normals,   mhdr.num_loops * 3)) return false;

        struct UVLayer { std::string name; std::vector<float> uvs; };
        std::vector<UVLayer> uvLayers(mhdr.num_uvlayers);
        for (auto& uv : uvLayers) {
            if (!r.readStr(uv.name)) return false;
            if (!r.readFloats(uv.uvs, mhdr.num_loops * 2)) return false;
        }

        std::vector<uint32_t> loopStarts, loopTotals, matIndices, vertIndices;
        if (!r.readUints(loopStarts,  mhdr.num_polys)) return false;
        if (!r.readUints(loopTotals,  mhdr.num_polys)) return false;
        if (!r.readUints(matIndices,  mhdr.num_polys)) return false;
        if (!r.readUints(vertIndices, mhdr.num_loops)) return false;

        SdfPath meshPath = uniquePath(meshRoot, name, meshNames);
        UsdGeomMesh mesh = UsdGeomMesh::Define(stage, meshPath);

        // Transform — static or time-sampled for motion blur
        {
            auto xformOp = mesh.AddTransformOp();
            if (mhdr.num_motion_keys == 1) {
                xformOp.Set(matFromBlob(matrices.data()));
            } else {
                for (uint32_t k = 0; k < mhdr.num_motion_keys; ++k) {
                    double tc = static_cast<double>(timeCodes[k]);
                    xformOp.Set(matFromBlob(matrices.data() + k * 16),
                                UsdTimeCode(tc));
                    stageMinTime = std::min(stageMinTime, tc);
                    stageMaxTime = std::max(stageMaxTime, tc);
                }
            }
        }

        // Positions
        VtVec3fArray pts(mhdr.num_verts);
        for (uint32_t i = 0; i < mhdr.num_verts; ++i)
            pts[i] = GfVec3f(positions[i*3], positions[i*3+1], positions[i*3+2]);
        mesh.CreatePointsAttr().Set(pts);

        // Face topology
        VtIntArray faceVertCounts(mhdr.num_polys);
        VtIntArray faceVertIndices;
        faceVertIndices.reserve(mhdr.num_loops);
        for (uint32_t p = 0; p < mhdr.num_polys; ++p) {
            faceVertCounts[p] = static_cast<int>(loopTotals[p]);
            for (uint32_t l = loopStarts[p]; l < loopStarts[p] + loopTotals[p]; ++l)
                faceVertIndices.push_back(static_cast<int>(vertIndices[l]));
        }
        mesh.CreateFaceVertexCountsAttr().Set(faceVertCounts);
        mesh.CreateFaceVertexIndicesAttr().Set(faceVertIndices);

        // Normals (face-varying)
        VtVec3fArray norms(mhdr.num_loops);
        for (uint32_t i = 0; i < mhdr.num_loops; ++i)
            norms[i] = GfVec3f(normals[i*3], normals[i*3+1], normals[i*3+2]);
        mesh.CreateNormalsAttr().Set(norms);
        mesh.SetNormalsInterpolation(UsdGeomTokens->faceVarying);

        // UV primvars
        UsdGeomPrimvarsAPI primvarsAPI(mesh);
        bool firstUV = true;
        for (const auto& uv : uvLayers) {
            TfToken uvName = firstUV ? TfToken("st") : TfToken(sanitize(uv.name));
            firstUV = false;
            VtVec2fArray uvArr(mhdr.num_loops);
            for (uint32_t i = 0; i < mhdr.num_loops; ++i)
                uvArr[i] = GfVec2f(uv.uvs[i*2], uv.uvs[i*2+1]);
            auto pv = primvarsAPI.CreatePrimvar(uvName,
                          SdfValueTypeNames->TexCoord2fArray,
                          UsdGeomTokens->faceVarying);
            pv.Set(uvArr);
        }

        // Material binding — one subset per material slot used
        if (!matNames.empty()) {
            // Collect face indices per material slot
            std::unordered_map<uint32_t, VtIntArray> slotFaces;
            for (uint32_t p = 0; p < mhdr.num_polys; ++p)
                slotFaces[matIndices[p]].push_back(static_cast<int>(p));

            auto matTransColor = [&](uint32_t slot) -> const float* {
                static const float kWhite[3] = {0.8f, 0.8f, 0.8f};
                if (slot < matColors.size() / 3)
                    return matColors.data() + slot * 3;
                return kWhite;
            };
            auto matIsTranslucent = [&](uint32_t slot) -> bool {
                return slot < matFlags.size() && (matFlags[slot] & 1u);
            };
            auto matHasSss = [&](uint32_t slot) -> bool {
                return slot < matFlags.size() && (matFlags[slot] & 2u);
            };
            auto matSssPtr = [&](uint32_t slot) -> const float* {
                static const float kZero[9] = {};
                if (slot * 9 + 9 <= matSssData.size())
                    return matSssData.data() + slot * 9;
                return kZero;
            };

            if (slotFaces.size() == 1) {
                // Single material — bind directly on the mesh
                uint32_t slot = slotFaces.begin()->first;
                if (slot < matNames.size() && !matNames[slot].empty()) {
                    SdfPath mp = getOrCreateMaterial(matNames[slot],
                                     matIsTranslucent(slot), matTransColor(slot),
                                     matHasSss(slot), matSssPtr(slot));
                    UsdShadeMaterialBindingAPI::Apply(mesh.GetPrim())
                        .Bind(UsdShadeMaterial(stage->GetPrimAtPath(mp)));
                }
            } else {
                // Multiple materials — one UsdGeomSubset per slot
                for (auto& [slot, faces] : slotFaces) {
                    if (slot >= matNames.size() || matNames[slot].empty()) continue;
                    SdfPath mp = getOrCreateMaterial(matNames[slot],
                                     matIsTranslucent(slot), matTransColor(slot),
                                     matHasSss(slot), matSssPtr(slot));
                    std::string subName = "mat_" + std::to_string(slot);
                    auto subset = UsdGeomSubset::CreateGeomSubset(
                        mesh, TfToken(subName),
                        UsdGeomTokens->face, faces,
                        TfToken("materialBind"));
                    UsdShadeMaterialBindingAPI::Apply(subset.GetPrim())
                        .Bind(UsdShadeMaterial(stage->GetPrimAtPath(mp)));
                }
            }
        }
    }

    // --- Write lights ---
    for (uint32_t li = 0; li < hdr.num_lights; ++li) {
        scene_fmt::LightRecord lrec{};
        // Read fixed fields except name (name comes after fixed fields)
        if (!r.read(&lrec, offsetof(scene_fmt::LightRecord, params) + sizeof(lrec.params)))
            return false;
        std::string lname;
        if (!r.readStr(lname)) return false;

        SdfPath lpath = uniquePath(lightRoot, lname, lightNames);
        GfMatrix4d lmat = matFromBlob(lrec.matrix);

        auto ltype = static_cast<scene_fmt::LightType>(lrec.type);

        if (ltype == scene_fmt::LightType::Distant) {
            auto dl = UsdLuxDistantLight::Define(stage, lpath);
            dl.AddTransformOp().Set(lmat);
            UsdLuxLightAPI api(dl.GetPrim());
            api.CreateColorAttr().Set(GfVec3f(lrec.color[0], lrec.color[1], lrec.color[2]));
            api.CreateIntensityAttr().Set(lrec.intensity);
            api.CreateNormalizeAttr().Set(lrec.normalize != 0);
            if (lrec.params[0] > 0.f)
                dl.CreateAngleAttr().Set(lrec.params[0]);

        } else if (ltype == scene_fmt::LightType::Rect) {
            auto rl = UsdLuxRectLight::Define(stage, lpath);
            rl.AddTransformOp().Set(lmat);
            UsdLuxLightAPI api(rl.GetPrim());
            api.CreateColorAttr().Set(GfVec3f(lrec.color[0], lrec.color[1], lrec.color[2]));
            api.CreateIntensityAttr().Set(lrec.intensity);
            api.CreateNormalizeAttr().Set(lrec.normalize != 0);
            rl.CreateWidthAttr().Set(lrec.params[1]);
            rl.CreateHeightAttr().Set(lrec.params[2]);

        } else if (ltype == scene_fmt::LightType::Sphere) {
            auto sl = UsdLuxSphereLight::Define(stage, lpath);
            sl.AddTransformOp().Set(lmat);
            UsdLuxLightAPI api(sl.GetPrim());
            api.CreateColorAttr().Set(GfVec3f(lrec.color[0], lrec.color[1], lrec.color[2]));
            api.CreateIntensityAttr().Set(lrec.intensity);
            api.CreateNormalizeAttr().Set(lrec.normalize != 0);
            sl.CreateRadiusAttr().Set(lrec.params[3]);

        } else if (ltype == scene_fmt::LightType::Disk) {
            auto dkl = UsdLuxDiskLight::Define(stage, lpath);
            dkl.AddTransformOp().Set(lmat);
            UsdLuxLightAPI api(dkl.GetPrim());
            api.CreateColorAttr().Set(GfVec3f(lrec.color[0], lrec.color[1], lrec.color[2]));
            api.CreateIntensityAttr().Set(lrec.intensity);
            api.CreateNormalizeAttr().Set(lrec.normalize != 0);
            dkl.CreateRadiusAttr().Set(lrec.params[3]);
        }
    }

    // --- Write camera ---
    if (hdr.has_camera) {
        scene_fmt::CameraRecord crec{};
        if (!r.read(crec)) {
            spdlog::error("SceneExporter: truncated camera record");
            return false;
        }
        SdfPath camPath = camRoot.AppendChild(TfToken("Camera"));
        auto cam = UsdGeomCamera::Define(stage, camPath);
        cam.AddTransformOp().Set(matFromBlob(crec.matrix));
        cam.CreateFocalLengthAttr().Set(crec.lens);
        cam.CreateHorizontalApertureAttr().Set(crec.sensor_width);
        cam.CreateVerticalApertureAttr().Set(crec.sensor_height);
        cam.CreateClippingRangeAttr().Set(GfVec2f(crec.clip_start, crec.clip_end));
        if (crec.dof_distance > 0.f)
            cam.CreateFocusDistanceAttr().Set(crec.dof_distance);
        // Set render settings camera reference
        stage->SetDefaultPrim(stage->GetPrimAtPath(rootPath));
    }

    if (stageMinTime < stageMaxTime) {
        stage->SetStartTimeCode(stageMinTime);
        stage->SetEndTimeCode(stageMaxTime);
    }

    stage->Save();

    spdlog::info("SceneExporter: wrote '{}'", usdPath);
    return true;
}

} // namespace anacapa
