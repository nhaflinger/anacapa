#ifdef ANACAPA_ENABLE_ALEMBIC

#include "AlembicLoader.h"
#include "../../shading/MarschnerHair.h"

#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreFactory/IFactory.h>

#include <spdlog/spdlog.h>
#include <cmath>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Imath M44d → anacapa Mat4f.
// Imath uses row-vector convention (v * M); anacapa uses column-vector (M * v).
// The transpose maps between the two.
static Mat4f toMat4f(const Imath::M44d& m) {
    Mat4f r;
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            r.m[i][j] = static_cast<float>(m[j][i]);   // transpose
    return r;
}

// Walk the parent chain collecting IXform transforms, then compose them
// root-first to produce the world transform of obj.
static Mat4f getWorldXform(const Alembic::AbcGeom::IObject& obj) {
    using namespace Alembic::AbcGeom;

    std::vector<Imath::M44d> stack;
    IObject cur = obj.getParent();
    while (cur.valid()) {
        if (IXform::matches(cur.getMetaData())) {
            IXform xf(cur, Alembic::Abc::kWrapExisting);
            XformSample xs;
            xf.getSchema().get(xs);
            stack.push_back(xs.getMatrix());
        }
        cur = cur.getParent();
    }

    Imath::M44d world;
    world.makeIdentity();
    for (auto it = stack.rbegin(); it != stack.rend(); ++it)
        world = world * (*it);

    return toMat4f(world);
}

// ---------------------------------------------------------------------------
// Curve-basis conversion utilities
// All functions produce endpoint-sharing cubic Bézier CVs (3n+1 points).
// ---------------------------------------------------------------------------

// B-spline → Bézier
// Input:  N uniform cubic B-spline CVs  → (N-3) segments
// Output: 3*(N-3)+1 endpoint-sharing Bézier CVs
static void bsplineToBezier(const std::vector<Vec3f>& sp,
                             const std::vector<float>& sw,
                             std::vector<Vec3f>&       bp,
                             std::vector<float>&       bw) {
    int n = static_cast<int>(sp.size()) - 3;
    if (n <= 0) return;

    bp.resize(3 * n + 1);
    bw.resize(3 * n + 1);

    for (int i = 0; i < n; ++i) {
        Vec3f p0 = sp[i],   p1 = sp[i+1], p2 = sp[i+2], p3 = sp[i+3];
        float w0 = sw[i],   w1 = sw[i+1], w2 = sw[i+2], w3 = sw[i+3];

        Vec3f b0 = (p0 + p1 * 4.f + p2) * (1.f / 6.f);
        Vec3f b1 = (p1 * 2.f + p2)      * (1.f / 3.f);
        Vec3f b2 = (p1 + p2 * 2.f)      * (1.f / 3.f);
        Vec3f b3 = (p1 + p2 * 4.f + p3) * (1.f / 6.f);

        float bw0 = (w0 + 4.f*w1 + w2) / 6.f;
        float bw1 = (2.f*w1 + w2)      / 3.f;
        float bw2 = (w1 + 2.f*w2)      / 3.f;
        float bw3 = (w1 + 4.f*w2 + w3) / 6.f;

        if (i == 0) { bp[0] = b0; bw[0] = bw0; }
        bp[3*i+1] = b1;  bw[3*i+1] = bw1;
        bp[3*i+2] = b2;  bw[3*i+2] = bw2;
        bp[3*i+3] = b3;  bw[3*i+3] = bw3;
    }
}

// Catmull-Rom → Bézier
// Input:  N CVs  → (N-3) interior segments (p1..p[N-2] are the curve endpoints)
// Output: 3*(N-3)+1 endpoint-sharing Bézier CVs
static void catmullromToBezier(const std::vector<Vec3f>& sp,
                                const std::vector<float>& sw,
                                std::vector<Vec3f>&       bp,
                                std::vector<float>&       bw) {
    int n = static_cast<int>(sp.size()) - 3;
    if (n <= 0) return;

    bp.resize(3 * n + 1);
    bw.resize(3 * n + 1);

    for (int i = 0; i < n; ++i) {
        Vec3f p0 = sp[i], p1 = sp[i+1], p2 = sp[i+2], p3 = sp[i+3];
        float f0 = sw[i], f1 = sw[i+1], f2 = sw[i+2], f3 = sw[i+3];

        Vec3f b0 = p1;
        Vec3f b1 = p1 + (p2 - p0) * (1.f / 6.f);
        Vec3f b2 = p2 - (p3 - p1) * (1.f / 6.f);
        Vec3f b3 = p2;

        float bw0 = f1;
        float bw1 = f1 + (f2 - f0) / 6.f;
        float bw2 = f2 - (f3 - f1) / 6.f;
        float bw3 = f2;

        if (i == 0) { bp[0] = b0; bw[0] = bw0; }
        bp[3*i+1] = b1;  bw[3*i+1] = bw1;
        bp[3*i+2] = b2;  bw[3*i+2] = bw2;
        bp[3*i+3] = b3;  bw[3*i+3] = bw3;
    }
}

// Linear → cubic Bézier (collinear interior CVs)
// Input:  N CVs  → (N-1) linear segments
// Output: 3*(N-1)+1 = 3N-2 endpoint-sharing Bézier CVs
static void linearToBezier(const std::vector<Vec3f>& sp,
                            const std::vector<float>& sw,
                            std::vector<Vec3f>&       bp,
                            std::vector<float>&       bw) {
    int n = static_cast<int>(sp.size()) - 1;
    if (n <= 0) return;

    bp.resize(3 * n + 1);
    bw.resize(3 * n + 1);

    for (int i = 0; i < n; ++i) {
        Vec3f p0 = sp[i], p1 = sp[i+1];
        float f0 = sw[i], f1 = sw[i+1];

        if (i == 0) { bp[0] = p0; bw[0] = f0; }
        bp[3*i+1] = p0 + (p1 - p0) * (1.f / 3.f);
        bp[3*i+2] = p0 + (p1 - p0) * (2.f / 3.f);
        bp[3*i+3] = p1;

        bw[3*i+1] = f0 + (f1 - f0) / 3.f;
        bw[3*i+2] = f0 + (f1 - f0) * 2.f / 3.f;
        bw[3*i+3] = f1;
    }
}

// ---------------------------------------------------------------------------
// Process a single ICurves object
// ---------------------------------------------------------------------------
static void processICurves(const Alembic::AbcGeom::ICurves& curvesObj,
                            const AlembicCurveOptions&       opts,
                            CurvePool&                       pool) {
    using namespace Alembic::AbcGeom;

    auto schema = curvesObj.getSchema();
    if (!schema.getNumSamples()) return;

    ICurvesSchema::Sample samp;
    schema.get(samp);

    auto positions = samp.getPositions();
    auto counts    = samp.getCurvesNumVertices();
    if (!positions || !counts || counts->size() == 0) return;

    const CurveType basisType = samp.getType();   // kLinear | kCubic
    const BasisType basis     = samp.getBasis();  // kBezierBasis | kBsplineBasis | …

    // ---- widths ----
    // Determine per-vertex widths by checking array size vs total CVs.
    size_t totalCVs    = positions->size();
    size_t numCurves   = counts->size();
    std::vector<float> allWidths;  // indexed by global CV index

    IFloatGeomParam wParam = schema.getWidthsParam();
    if (wParam.valid()) {
        IFloatGeomParam::Sample ws;
        wParam.getExpanded(ws);   // tries to expand to vertex scope
        if (ws.getVals() && ws.getVals()->size() > 0) {
            const float* wd = ws.getVals()->get();
            size_t       wn = ws.getVals()->size();

            if (wn == totalCVs) {
                // Per-vertex — apply widthScale here
                allWidths.resize(wn);
                for (size_t i = 0; i < wn; ++i)
                    allWidths[i] = wd[i] * opts.widthScale;
            } else if (wn == numCurves) {
                // Per-curve: replicate each value across the curve's CVs
                allWidths.reserve(totalCVs);
                for (size_t ci = 0; ci < numCurves; ++ci) {
                    int32_t nv = (*counts)[ci];
                    float   w  = (ci < wn) ? wd[ci] : opts.defaultWidth;
                    for (int32_t k = 0; k < nv; ++k)
                        allWidths.push_back(w * opts.widthScale);
                }
            } else if (wn == 1) {
                // Constant
                allWidths.assign(totalCVs, wd[0] * opts.widthScale);
            }
        }
    }
    if (allWidths.empty())
        allWidths.assign(totalCVs, opts.defaultWidth);

    // ---- world transform ----
    Mat4f xform = getWorldXform(curvesObj);

    // ---- per-strand color (AHAIR002 — optional) ----
    std::vector<Vec3f> strandColors;  // empty = no color data
    {
        using namespace Alembic::AbcGeom;
        ICompoundProperty arbParams = schema.getArbGeomParams();
        if (arbParams.valid() && arbParams.getPropertyHeader("color")) {
            IC3fGeomParam colorParam(arbParams, "color");
            if (colorParam.valid()) {
                IC3fGeomParam::Sample cs;
                colorParam.getExpanded(cs);
                if (cs.getVals() && cs.getVals()->size() == numCurves) {
                    const Imath::C3f* cols = cs.getVals()->get();
                    strandColors.resize(numCurves);
                    for (size_t ci2 = 0; ci2 < numCurves; ++ci2)
                        strandColors[ci2] = { cols[ci2].x, cols[ci2].y, cols[ci2].z };
                    spdlog::info("AlembicLoader: loaded {} strand colors (first={:.3f},{:.3f},{:.3f})",
                                 numCurves, cols[0].x, cols[0].y, cols[0].z);
                }
            }
        }
    }

    // ---- iterate curves ----
    const Imath::V3f* pts      = positions->get();
    const int32_t*    nVertsArr = counts->get();
    size_t cvOffset = 0;

    size_t strandsAdded = 0;
    size_t strandsSkipped = 0;

    for (size_t ci = 0; ci < numCurves; ++ci) {
        int32_t nv = nVertsArr[ci];
        if (nv < 2) { cvOffset += nv; continue; }

        // Collect CVs and transform to world space
        std::vector<Vec3f> rawCVs(nv);
        for (int32_t i = 0; i < nv; ++i) {
            const auto& p = pts[cvOffset + i];
            rawCVs[i] = xform.transformPoint({p.x, p.y, p.z});
        }

        // Collect per-CV widths for this curve
        std::vector<float> rawWidths(nv, opts.defaultWidth);
        if (cvOffset + (size_t)nv <= allWidths.size()) {
            rawWidths.assign(allWidths.begin() + cvOffset,
                             allWidths.begin() + cvOffset + nv);
        }

        // Convert to endpoint-sharing cubic Bézier
        std::vector<Vec3f> bezCVs;
        std::vector<float> bezWidths;

        if (basisType == kLinear) {
            linearToBezier(rawCVs, rawWidths, bezCVs, bezWidths);
        } else { // kCubic
            if (basis == kBezierBasis) {
                if (nv < 4 || (nv - 1) % 3 != 0) {
                    spdlog::warn("AlembicLoader: Bezier curve {} has {} CVs (expected 3n+1), skipping",
                                 ci, nv);
                    cvOffset += nv; ++strandsSkipped; continue;
                }
                bezCVs   = rawCVs;
                bezWidths = rawWidths;
            } else if (basis == kBsplineBasis) {
                if (nv < 4) { cvOffset += nv; ++strandsSkipped; continue; }
                bsplineToBezier(rawCVs, rawWidths, bezCVs, bezWidths);
            } else if (basis == kCatmullromBasis) {
                if (nv < 4) { cvOffset += nv; ++strandsSkipped; continue; }
                catmullromToBezier(rawCVs, rawWidths, bezCVs, bezWidths);
            } else {
                spdlog::warn("AlembicLoader: unsupported curve basis {} in '{}', skipping",
                             static_cast<int>(basis), curvesObj.getFullName());
                cvOffset += nv; ++strandsSkipped; continue;
            }
        }

        if (bezCVs.size() < 4) { cvOffset += nv; ++strandsSkipped; continue; }

        StrandDesc strand;
        strand.controlPoints  = std::move(bezCVs);
        strand.widths         = std::move(bezWidths);
        strand.materialIndex  = opts.baseMaterialIndex;
        if (!strandColors.empty())
            strand.color = strandColors[ci];
        pool.addStrand(std::move(strand));
        ++strandsAdded;

        cvOffset += nv;
    }

    if (strandsAdded || strandsSkipped) {
        spdlog::info("AlembicLoader: '{}' → {} strands added, {} skipped",
                     curvesObj.getFullName(), strandsAdded, strandsSkipped);
    }
}

// ---------------------------------------------------------------------------
// Recursive scene-graph traversal
// ---------------------------------------------------------------------------
static void traverse(const Alembic::AbcGeom::IObject& obj,
                     const AlembicCurveOptions&        opts,
                     CurvePool&                        pool) {
    using namespace Alembic::AbcGeom;

    if (ICurves::matches(obj.getMetaData())) {
        processICurves(ICurves(obj, Alembic::Abc::kWrapExisting), opts, pool);
    }

    for (size_t i = 0; i < obj.getNumChildren(); ++i)
        traverse(obj.getChild(i), opts, pool);
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
bool loadAlembicCurves(const std::string&                        path,
                       const AlembicCurveOptions&                opts,
                       CurvePool&                                outPool,
                       std::vector<std::unique_ptr<IMaterial>>&  outMaterials) {
    using namespace Alembic;

    AbcCoreFactory::IFactory factory;
    AbcCoreFactory::IFactory::CoreType coreType;
    Abc::IArchive archive = factory.getArchive(path, coreType);

    if (!archive.valid()) {
        spdlog::error("AlembicLoader: cannot open '{}'", path);
        return false;
    }

    spdlog::info("AlembicLoader: loading '{}'  ({})",
                 path,
                 coreType == AbcCoreFactory::IFactory::kOgawa ? "Ogawa" : "HDF5");

    size_t strandsBefore = outPool.numStrands();
    traverse(archive.getTop(), opts, outPool);
    size_t strandsLoaded = outPool.numStrands() - strandsBefore;

    if (strandsLoaded == 0) {
        spdlog::warn("AlembicLoader: no ICurves found in '{}'", path);
    } else {
        spdlog::info("AlembicLoader: {} total strands loaded from '{}'",
                     strandsLoaded, path);
    }

    // Create one MarschnerHairMaterial with default parameters
    outMaterials.push_back(std::make_unique<MarschnerHairMaterial>(
        MarschnerHairMaterial::Params{}));

    return true;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_ALEMBIC
