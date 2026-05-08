#pragma once

#include <anacapa/shading/IMaterial.h>
#include <memory>
#include <string>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// MatAssignEntry
//
// One per-object material assignment from a .matassign.json file.
// The material type determines which fields are meaningful:
//
//   "marschner" — inline Marschner hair parameters; no external dependencies.
//   "usd"       — reference a material prim in the already-loaded USD scene.
//   "osl"       — reference an .oso shader file on disk with optional param
//                 overrides (future; not yet implemented).
//
// Multiple .matassign.json files may be loaded in order; a later file's entry
// for a given object name overrides an earlier one (layer semantics).
// ---------------------------------------------------------------------------

enum class MatAssignType { Marschner, Chiang, Usd, Osl };

struct MarschnerAssignParams {
    float sigma_a[3] = {0.06f, 0.10f, 0.20f};  // absorption coefficients (Beer-Lambert)
    float beta_m     = 0.30f;                    // longitudinal roughness ∈ [0,1]
    float beta_n     = 0.45f;                    // azimuthal roughness ∈ [0,1]
    float alpha      = -2.0f;                    // cuticle tilt angle in degrees
};

struct MatAssignEntry {
    std::string           objectName;  // DCC object name — primary key, human-readable
    std::string           comment;     // free-text annotation for artists/TDs

    MatAssignType         type        = MatAssignType::Marschner;

    MarschnerAssignParams marschner;   // valid when type == Marschner

    std::string           usdPath;    // USD material prim path, type == Usd
    std::string           oslShader;  // path to .oso file, type == Osl (future)
};

// ---------------------------------------------------------------------------
// MatAssignFile
//
// Top-level contents of one .matassign.json.
//
// Metadata fields are intended to carry enough context to reconstruct the
// work at a later date, on a different machine, or after a file is moved:
//
//   version       — schema version; bump when the format changes.
//   exported_at   — ISO 8601 UTC timestamp written by the DCC at export time.
//   source_app    — DCC application name and version (e.g. "Blender 5.1").
//   source_file   — Absolute path to the DCC scene file at export time.
//   source_frame  — Frame number exported (for animated caches).
//   geometry_cache— Filename (not full path) of the Alembic .abc this file
//                   accompanies; ties the matassign to a specific cache.
//   notes         — Free-text field for shot/sequence/department context,
//                   render farm wrangler notes, etc.  Multi-line OK.
// ---------------------------------------------------------------------------
struct MatAssignFile {
    int         version       = 1;
    std::string exportedAt;      // "2026-05-04T14:32:00Z"
    std::string sourceApp;       // "Blender 5.1"
    std::string sourceFile;      // "/jobs/proj/char/char_v14.blend"
    int         sourceFrame   = 0;
    std::string geometryCache;   // "char_v14.abc"
    std::string notes;           // free-form; survives round-trips unchanged

    std::vector<MatAssignEntry> assignments;
};

// ---------------------------------------------------------------------------
// loadMatAssign
//
// Parse a .matassign.json file.  Returns an empty MatAssignFile (version=0)
// on failure (file not found, malformed JSON) and logs a warning.
// ---------------------------------------------------------------------------
MatAssignFile loadMatAssign(const std::string& path);

// ---------------------------------------------------------------------------
// mergeMatAssign
//
// Merge src into dst.  For any object name present in both, src wins
// (layer semantics: later files override earlier ones).
// ---------------------------------------------------------------------------
void mergeMatAssign(std::vector<MatAssignEntry>&       dst,
                    const std::vector<MatAssignEntry>& src);

// ---------------------------------------------------------------------------
// buildMaterial
//
// Construct the appropriate IMaterial from a MatAssignEntry.
// Usd type requires the caller to resolve the prim path separately;
// if called here it returns a default MarschnerHairMaterial.
// ---------------------------------------------------------------------------
std::unique_ptr<IMaterial> buildMaterial(const MatAssignEntry& entry);

} // namespace anacapa
