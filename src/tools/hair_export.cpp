/**
 * hair_export — minimal Alembic writer for hair strand data.
 *
 * Reads a simple binary strand file written by the Blender addon's
 * export_hair_abc() function (which reads obj.data.attributes directly,
 * bypassing the crashy bpy.ops.wm.alembic_export operator in Blender 5.1).
 *
 * Binary format (little-endian):
 *
 *   AHAIR001 — original format:
 *     magic[8]         "AHAIR001"
 *     num_strands u32
 *     per strand:
 *       num_points u32
 *       per point: x f32, y f32, z f32, radius f32
 *
 *   AHAIR002 — adds per-strand color:
 *     magic[8]         "AHAIR002"
 *     num_strands u32
 *     per strand:
 *       num_points u32
 *       r f32, g f32, b f32   (linear sRGB strand color)
 *       per point: x f32, y f32, z f32, radius f32
 *
 * Usage: hair_export <input.hairbin> <output.abc> [--materials <materials.json>]
 *
 * If --materials is provided, the JSON file maps hair object names to their
 * USD material paths (e.g. { "Hair": "/Materials/PrincipledHair" }).
 * The material path is stored as a string attribute "materialPath" on the
 * ICurves schema so the AlembicLoader can look it up in the USD scene.
 */

#ifdef ANACAPA_ENABLE_ALEMBIC

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Alembic::AbcGeom;
using namespace Alembic::Abc;

static bool readLE32(FILE* f, uint32_t& out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return false;
    out = (uint32_t)b[0]
        | ((uint32_t)b[1] << 8)
        | ((uint32_t)b[2] << 16)
        | ((uint32_t)b[3] << 24);
    return true;
}

static bool readF32(FILE* f, float& out) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return false;
    uint32_t u = (uint32_t)b[0]
               | ((uint32_t)b[1] << 8)
               | ((uint32_t)b[2] << 16)
               | ((uint32_t)b[3] << 24);
    memcpy(&out, &u, 4);
    return true;
}

// Minimal JSON string-map parser: reads { "key": "value", ... }
// Only handles flat string-to-string maps (sufficient for material paths).
static std::unordered_map<std::string, std::string> parseJsonStringMap(const std::string& path) {
    std::unordered_map<std::string, std::string> result;
    std::ifstream f(path);
    if (!f.is_open()) return result;
    std::ostringstream ss;
    ss << f.rdbuf();
    std::string text = ss.str();

    // Extract "key": "value" pairs
    size_t pos = 0;
    while (pos < text.size()) {
        // Find next quoted key
        size_t ks = text.find('"', pos);
        if (ks == std::string::npos) break;
        size_t ke = text.find('"', ks + 1);
        if (ke == std::string::npos) break;
        std::string key = text.substr(ks + 1, ke - ks - 1);

        // Find colon then quoted value
        size_t colon = text.find(':', ke + 1);
        if (colon == std::string::npos) break;
        size_t vs = text.find('"', colon + 1);
        if (vs == std::string::npos) break;
        size_t ve = text.find('"', vs + 1);
        if (ve == std::string::npos) break;
        std::string val = text.substr(vs + 1, ve - vs - 1);

        result[key] = val;
        pos = ve + 1;
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: hair_export <input.hairbin> <output.abc> [--materials <materials.json>]\n");
        return 1;
    }

    // Parse optional --materials argument
    std::string materialsJsonPath;
    for (int i = 3; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--materials") {
            materialsJsonPath = argv[i + 1];
            ++i;
        }
    }

    FILE* f = fopen(argv[1], "rb");
    if (!f) {
        perror("hair_export: open input");
        return 1;
    }

    // Validate magic — accept both AHAIR001 (no color) and AHAIR002 (per-strand color)
    char magic[8];
    if (fread(magic, 1, 8, f) != 8) {
        fprintf(stderr, "hair_export: truncated magic in '%s'\n", argv[1]);
        fclose(f);
        return 1;
    }
    bool hasColor = false;
    if (memcmp(magic, "AHAIR002", 8) == 0) {
        hasColor = true;
    } else if (memcmp(magic, "AHAIR001", 8) != 0) {
        fprintf(stderr, "hair_export: invalid magic in '%s'\n", argv[1]);
        fclose(f);
        return 1;
    }

    uint32_t numStrands;
    if (!readLE32(f, numStrands)) {
        fprintf(stderr, "hair_export: truncated header\n");
        fclose(f);
        return 1;
    }

    // Flatten all strands into contiguous Alembic arrays
    std::vector<Imath::V3f>   allPoints;
    std::vector<int32_t>      curveCounts;
    std::vector<float>        allWidths;
    std::vector<Imath::C3f>   strandColors;  // one per strand (AHAIR002 only)

    allPoints.reserve(numStrands * 32);
    curveCounts.reserve(numStrands);
    allWidths.reserve(numStrands * 32);
    if (hasColor) strandColors.reserve(numStrands);

    for (uint32_t si = 0; si < numStrands; ++si) {
        uint32_t n;
        if (!readLE32(f, n)) {
            fprintf(stderr, "hair_export: truncated strand %u\n", si);
            fclose(f);
            return 1;
        }
        curveCounts.push_back(static_cast<int32_t>(n));

        if (hasColor) {
            float cr, cg, cb;
            if (!readF32(f, cr) || !readF32(f, cg) || !readF32(f, cb)) {
                fprintf(stderr, "hair_export: truncated color data (strand %u)\n", si);
                fclose(f);
                return 1;
            }
            strandColors.push_back({cr, cg, cb});
        }

        for (uint32_t pi = 0; pi < n; ++pi) {
            float x, y, z, r;
            if (!readF32(f, x) || !readF32(f, y) || !readF32(f, z) || !readF32(f, r)) {
                fprintf(stderr, "hair_export: truncated point data (strand %u, pt %u)\n", si, pi);
                fclose(f);
                return 1;
            }
            allPoints.push_back({x, y, z});
            allWidths.push_back(r * 2.0f);  // radius → diameter (Alembic width convention)
        }
    }
    fclose(f);

    if (curveCounts.empty()) {
        fprintf(stderr, "hair_export: no strands in input\n");
        return 1;
    }

    // Write Alembic (Ogawa format)
    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), argv[2]);
    OObject  top(archive, kTop);

    OCurves          curves(top, "hair_curves");
    OCurvesSchema&   schema = curves.getSchema();

    OFloatGeomParam::Sample widthSamp(
        FloatArraySample(allWidths.data(), allWidths.size()),
        kVertexScope
    );

    OCurvesSchema::Sample sample(
        P3fArraySample(allPoints.data(), allPoints.size()),
        Int32ArraySample(curveCounts.data(), curveCounts.size()),
        kLinear,        // CurveType  — AlembicLoader linearToBezier handles this
        kNonPeriodic,   // wrap
        widthSamp
    );

    schema.set(sample);

    // Write per-strand color as a uniform-scope C3f geom param (one value per curve)
    if (hasColor && !strandColors.empty()) {
        OC3fGeomParam colorParam(curves.getSchema().getArbGeomParams(),
                                 "color", false, kUniformScope, 1);
        OC3fGeomParam::Sample colorSamp(
            C3fArraySample(strandColors.data(), strandColors.size()),
            kUniformScope
        );
        colorParam.set(colorSamp);
    }

    // Write material paths from JSON sidecar as a string attribute "materialPath".
    // Each entry is "objectName:usdPath" separated by semicolons so the loader
    // can find the right material for each hair object.
    if (!materialsJsonPath.empty()) {
        auto matMap = parseJsonStringMap(materialsJsonPath);
        if (!matMap.empty()) {
            std::string encoded;
            for (const auto& kv : matMap) {
                if (!encoded.empty()) encoded += ";";
                encoded += kv.first + ":" + kv.second;
            }
            OCompoundProperty arbParams = curves.getSchema().getArbGeomParams();
            OStringProperty matProp(arbParams, "materialPath");
            matProp.set(encoded);
            printf("[hair_export] material paths: %s\n", encoded.c_str());
        }
    }

    printf("[hair_export] %u strands, %zu CVs%s → %s\n",
           numStrands, allPoints.size(),
           hasColor ? " (with color)" : "",
           argv[2]);
    return 0;
}

#else  // !ANACAPA_ENABLE_ALEMBIC

int main() {
    fprintf(stderr, "hair_export: built without Alembic support (ANACAPA_ENABLE_ALEMBIC=OFF)\n");
    return 1;
}

#endif
