#include "MatAssignLoader.h"
#include "../shading/MarschnerHair.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <fstream>

namespace anacapa {

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static MatAssignType parseType(const std::string& s) {
    if (s == "usd")      return MatAssignType::Usd;
    if (s == "osl")      return MatAssignType::Osl;
    return MatAssignType::Marschner;  // "marschner" or unknown → safe default
}

static MatAssignEntry parseEntry(const json& j) {
    MatAssignEntry e;
    e.objectName = j.value("object",  "");
    e.comment    = j.value("comment", "");

    // "material" sub-object carries type and all material-specific fields.
    const json* mat = j.contains("material") ? &j["material"] : nullptr;
    e.type = parseType(mat ? mat->value("type", "marschner") : "marschner");

    if (e.type == MatAssignType::Marschner) {
        const json& m = mat ? *mat : j;
        if (m.contains("sigma_a") && m["sigma_a"].is_array()
                && m["sigma_a"].size() >= 3) {
            e.marschner.sigma_a[0] = m["sigma_a"][0].get<float>();
            e.marschner.sigma_a[1] = m["sigma_a"][1].get<float>();
            e.marschner.sigma_a[2] = m["sigma_a"][2].get<float>();
        }
        e.marschner.beta_m = m.value("beta_m", e.marschner.beta_m);
        e.marschner.beta_n = m.value("beta_n", e.marschner.beta_n);
        e.marschner.alpha  = m.value("alpha",  e.marschner.alpha);
    } else if (e.type == MatAssignType::Usd) {
        e.usdPath   = mat ? mat->value("path", "") : "";
    } else if (e.type == MatAssignType::Osl) {
        e.oslShader = mat ? mat->value("shader", "") : "";
    }

    return e;
}

// ---------------------------------------------------------------------------
// loadMatAssign
// ---------------------------------------------------------------------------
MatAssignFile loadMatAssign(const std::string& path) {
    MatAssignFile result;

    std::ifstream f(path);
    if (!f.is_open()) {
        spdlog::warn("MatAssignLoader: cannot open '{}'", path);
        return result;
    }

    json j;
    try {
        f >> j;
    } catch (const json::exception& ex) {
        spdlog::warn("MatAssignLoader: JSON parse error in '{}': {}", path, ex.what());
        return result;
    }

    result.version      = j.value("version",       1);
    result.exportedAt   = j.value("exported_at",   "");
    result.sourceApp    = j.value("source_app",    "");
    result.sourceFile   = j.value("source_file",   "");
    result.sourceFrame  = j.value("source_frame",  0);
    result.geometryCache= j.value("geometry_cache","");
    result.notes        = j.value("notes",         "");

    if (j.contains("assignments") && j["assignments"].is_array()) {
        for (const auto& entry : j["assignments"]) {
            MatAssignEntry e = parseEntry(entry);
            if (e.objectName.empty()) {
                spdlog::warn("MatAssignLoader: skipping entry with empty object name in '{}'",
                             path);
                continue;
            }
            result.assignments.push_back(std::move(e));
        }
    }

    spdlog::info("MatAssignLoader: loaded '{}' — {} assignment(s), exported {} by {}",
                 path, result.assignments.size(),
                 result.exportedAt.empty() ? "(unknown date)" : result.exportedAt,
                 result.sourceApp.empty()  ? "(unknown app)"  : result.sourceApp);

    return result;
}

// ---------------------------------------------------------------------------
// mergeMatAssign
// ---------------------------------------------------------------------------
void mergeMatAssign(std::vector<MatAssignEntry>&       dst,
                    const std::vector<MatAssignEntry>& src) {
    for (const auto& srcEntry : src) {
        bool found = false;
        for (auto& dstEntry : dst) {
            if (dstEntry.objectName == srcEntry.objectName) {
                dstEntry = srcEntry;  // later file wins
                found = true;
                break;
            }
        }
        if (!found)
            dst.push_back(srcEntry);
    }
}

// ---------------------------------------------------------------------------
// buildMaterial
// ---------------------------------------------------------------------------
std::unique_ptr<IMaterial> buildMaterial(const MatAssignEntry& entry) {
    if (entry.type == MatAssignType::Marschner) {
        MarschnerHairMaterial::Params p;
        p.sigma_a = { entry.marschner.sigma_a[0],
                      entry.marschner.sigma_a[1],
                      entry.marschner.sigma_a[2] };
        p.beta_m  = entry.marschner.beta_m;
        p.beta_n  = entry.marschner.beta_n;
        p.alpha   = entry.marschner.alpha;
        return std::make_unique<MarschnerHairMaterial>(p);
    }

    // Usd and Osl types are resolved by the caller (RenderSession) which has
    // access to the loaded USD scene.  Returning a default here is a safe
    // fallback if buildMaterial is called in a context without USD access.
    spdlog::warn("MatAssignLoader: buildMaterial called for type '{}' object '{}' "
                 "— USD/OSL resolution requires RenderSession; using default Marschner",
                 entry.type == MatAssignType::Usd ? "usd" : "osl",
                 entry.objectName);
    return std::make_unique<MarschnerHairMaterial>(MarschnerHairMaterial::Params{});
}

} // namespace anacapa
