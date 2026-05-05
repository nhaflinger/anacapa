/**
 * hair_export — Alembic writer for hair strand data.
 *
 * Reads a binary strand file written by the Blender addon's export_hair_abc()
 * function and writes an Alembic archive with one ICurves object per Blender
 * hair object, named after that object.  Material assignments live in the
 * companion .matassign.json file written by the Blender plugin; they are not
 * stored in the Alembic archive.
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
 * Usage:
 *   hair_export <input.hairbin> <output.abc> [--objects <counts.json>]
 *
 * If --objects is provided, counts.json maps hair object names (in export
 * order) to their strand counts:
 *   { "HairLong": 12450, "HairShort": 67890 }
 * Key order in the file must match the strand order in the binary.
 * hair_export writes one named OCurves per entry.
 *
 * Without --objects, all strands are written to a single "hair_curves" node
 * (backward-compatible with old exports).
 */

#ifdef ANACAPA_ENABLE_ALEMBIC

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreOgawa/All.h>

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace Alembic::AbcGeom;
using namespace Alembic::Abc;
using ordered_json = nlohmann::ordered_json;

// ---------------------------------------------------------------------------
// Binary reading helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Per-strand data (flat arrays, indexed by global strand index)
// ---------------------------------------------------------------------------
struct StrandData {
    std::vector<Imath::V3f> points;      // all CVs, concatenated
    std::vector<int32_t>    counts;      // CVs per strand
    std::vector<float>      widths;      // all widths, concatenated
    std::vector<Imath::C3f> colors;      // one per strand (empty if AHAIR001)
};

// ---------------------------------------------------------------------------
// writeOCurves — write one OCurves object for a slice of StrandData
// ---------------------------------------------------------------------------
static void writeOCurves(OObject& parent,
                          const std::string& name,
                          const StrandData& data,
                          uint32_t strandStart,
                          uint32_t strandCount) {
    if (strandCount == 0) return;

    // Build point and width slices from the global arrays.
    // First, find the CV offset for strandStart.
    size_t cvStart = 0;
    for (uint32_t si = 0; si < strandStart && si < data.counts.size(); ++si)
        cvStart += data.counts[si];

    size_t cvCount = 0;
    for (uint32_t si = strandStart;
         si < strandStart + strandCount && si < data.counts.size(); ++si)
        cvCount += data.counts[si];

    const Imath::V3f* pts    = data.points.data() + cvStart;
    const float*      wids   = data.widths.data() + cvStart;
    const int32_t*    cnts   = data.counts.data() + strandStart;

    OCurves        curves(parent, name);
    OCurvesSchema& schema = curves.getSchema();

    OFloatGeomParam::Sample widthSamp(
        FloatArraySample(wids, cvCount),
        kVertexScope
    );

    OCurvesSchema::Sample sample(
        P3fArraySample(pts,  cvCount),
        Int32ArraySample(cnts, strandCount),
        kLinear,
        kNonPeriodic,
        widthSamp
    );
    schema.set(sample);

    // Per-strand color (AHAIR002)
    if (!data.colors.empty()) {
        const Imath::C3f* cols = data.colors.data() + strandStart;
        OC3fGeomParam colorParam(schema.getArbGeomParams(),
                                 "color", false, kUniformScope, 1);
        OC3fGeomParam::Sample colorSamp(
            C3fArraySample(cols, strandCount),
            kUniformScope
        );
        colorParam.set(colorSamp);
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    if (argc < 3) {
        fprintf(stderr,
            "Usage: hair_export <input.hairbin> <output.abc> [--objects <counts.json>]\n");
        return 1;
    }

    std::string objectsJsonPath;
    for (int i = 3; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--objects") {
            objectsJsonPath = argv[i + 1];
            ++i;
        }
    }

    // ---- read binary ----
    FILE* f = fopen(argv[1], "rb");
    if (!f) { perror("hair_export: open input"); return 1; }

    char magic[8];
    if (fread(magic, 1, 8, f) != 8) {
        fprintf(stderr, "hair_export: truncated magic in '%s'\n", argv[1]);
        fclose(f); return 1;
    }
    bool hasColor = false;
    if (memcmp(magic, "AHAIR002", 8) == 0) {
        hasColor = true;
    } else if (memcmp(magic, "AHAIR001", 8) != 0) {
        fprintf(stderr, "hair_export: invalid magic in '%s'\n", argv[1]);
        fclose(f); return 1;
    }

    uint32_t numStrands;
    if (!readLE32(f, numStrands)) {
        fprintf(stderr, "hair_export: truncated header\n");
        fclose(f); return 1;
    }

    StrandData sd;
    sd.points.reserve(numStrands * 32);
    sd.counts.reserve(numStrands);
    sd.widths.reserve(numStrands * 32);
    if (hasColor) sd.colors.reserve(numStrands);

    for (uint32_t si = 0; si < numStrands; ++si) {
        uint32_t n;
        if (!readLE32(f, n)) {
            fprintf(stderr, "hair_export: truncated strand %u\n", si);
            fclose(f); return 1;
        }
        sd.counts.push_back(static_cast<int32_t>(n));

        if (hasColor) {
            float cr, cg, cb;
            if (!readF32(f, cr) || !readF32(f, cg) || !readF32(f, cb)) {
                fprintf(stderr, "hair_export: truncated color (strand %u)\n", si);
                fclose(f); return 1;
            }
            sd.colors.push_back({cr, cg, cb});
        }

        for (uint32_t pi = 0; pi < n; ++pi) {
            float x, y, z, r;
            if (!readF32(f, x) || !readF32(f, y) || !readF32(f, z) || !readF32(f, r)) {
                fprintf(stderr, "hair_export: truncated point (strand %u, pt %u)\n", si, pi);
                fclose(f); return 1;
            }
            sd.points.push_back({x, y, z});
            sd.widths.push_back(r * 2.0f);  // radius → diameter (Alembic convention)
        }
    }
    fclose(f);

    if (sd.counts.empty()) {
        fprintf(stderr, "hair_export: no strands in '%s'\n", argv[1]);
        return 1;
    }

    // ---- parse per-object counts (ordered: key order matches binary) ----
    // Each entry: { "ObjectName": strandCount }
    struct ObjEntry { std::string name; uint32_t count; };
    std::vector<ObjEntry> objects;

    if (!objectsJsonPath.empty()) {
        std::ifstream jf(objectsJsonPath);
        if (!jf.is_open()) {
            fprintf(stderr, "hair_export: cannot open objects file '%s'\n",
                    objectsJsonPath.c_str());
        } else {
            try {
                ordered_json j;
                jf >> j;
                for (auto it = j.begin(); it != j.end(); ++it) {
                    uint32_t count = 0;
                    if (it.value().is_number_integer())
                        count = it.value().get<uint32_t>();
                    else if (it.value().is_string()) {
                        // tolerate legacy "path|count" values written by older plugin
                        std::string s = it.value().get<std::string>();
                        auto pipe = s.rfind('|');
                        if (pipe != std::string::npos)
                            count = static_cast<uint32_t>(std::stoul(s.substr(pipe + 1)));
                    }
                    if (count > 0)
                        objects.push_back({it.key(), count});
                }
            } catch (const std::exception& ex) {
                fprintf(stderr, "hair_export: error parsing objects file: %s\n", ex.what());
            }
        }
    }

    // ---- write Alembic ----
    OArchive archive(Alembic::AbcCoreOgawa::WriteArchive(), argv[2]);
    OObject  top(archive, kTop);

    if (objects.empty()) {
        // No per-object info — write everything as one "hair_curves" node.
        writeOCurves(top, "hair_curves", sd, 0,
                     static_cast<uint32_t>(sd.counts.size()));
        printf("[hair_export] %u strands, %zu CVs%s → %s (single object)\n",
               numStrands, sd.points.size(),
               hasColor ? " (with color)" : "", argv[2]);
    } else {
        uint32_t runningStrand = 0;
        for (const auto& obj : objects) {
            writeOCurves(top, obj.name, sd, runningStrand, obj.count);
            printf("[hair_export] '%s': %u strands\n", obj.name.c_str(), obj.count);
            runningStrand += obj.count;
        }
        if (runningStrand != numStrands) {
            fprintf(stderr,
                "hair_export: warning: object counts sum to %u but binary has %u strands\n",
                runningStrand, numStrands);
        }
        printf("[hair_export] %u strands, %zu CVs%s → %s (%zu objects)\n",
               numStrands, sd.points.size(),
               hasColor ? " (with color)" : "", argv[2], objects.size());
    }

    return 0;
}

#else  // !ANACAPA_ENABLE_ALEMBIC

int main() {
    fprintf(stderr,
        "hair_export: built without Alembic support (ANACAPA_ENABLE_ALEMBIC=OFF)\n");
    return 1;
}

#endif
