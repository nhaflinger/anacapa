#ifdef ANACAPA_ENABLE_ALEMBIC

#include "AlembicLoader.h"
#include "../../shading/MarschnerHair.h"
#include "../../shading/ChiangHair.h"
#include "../../shading/SoftParticle.h"

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
static void processICurves(const Alembic::AbcGeom::ICurves&  curvesObj,
                            const AlembicCurveOptions&        opts,
                            CurvePool&                        pool,
                            std::vector<AlembicObjectRange>*  outRanges) {
    using namespace Alembic::AbcGeom;

    auto schema = curvesObj.getSchema();
    if (!schema.getNumSamples()) return;

    ICurvesSchema::Sample samp;
    schema.get(samp, ISampleSelector(Alembic::Abc::index_t(0)));

    auto positions = samp.getPositions();
    auto counts    = samp.getCurvesNumVertices();
    if (!positions || !counts || counts->size() == 0) return;

    const CurveType basisType = samp.getType();   // kLinear | kCubic
    const BasisType basis     = samp.getBasis();  // kBezierBasis | kBsplineBasis | …

    // Load close positions for motion blur (sample index 1)
    bool hasClose = opts.motionBlur && schema.getNumSamples() >= 2;
    spdlog::info("AlembicLoader: '{}' — {} time sample(s), motionBlur={}, hasClose={}",
                 curvesObj.getFullName(), schema.getNumSamples(),
                 opts.motionBlur, hasClose);
    ICurvesSchema::Sample closeSamp;
    if (hasClose) {
        schema.get(closeSamp, ISampleSelector(Alembic::Abc::index_t(1)));
        auto closePts = closeSamp.getPositions();
        if (!closePts || closePts->size() != positions->size()) {
            spdlog::warn("AlembicLoader: close sample size mismatch in '{}' — disabling motion blur",
                         curvesObj.getFullName());
            hasClose = false;
        }
    }

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
    // ---- per-strand root UV (AHAIR003 — optional) ----
    std::vector<Vec2f> strandRootUVs; // empty = no UV data
    {
        using namespace Alembic::AbcGeom;
        ICompoundProperty arbParams = schema.getArbGeomParams();
        if (arbParams.valid()) {
            if (arbParams.getPropertyHeader("color")) {
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
            if (arbParams.getPropertyHeader("UVs")) {
                IV2fGeomParam uvParam(arbParams, "UVs");
                if (uvParam.valid()) {
                    IV2fGeomParam::Sample us;
                    uvParam.getExpanded(us);
                    if (us.getVals() && us.getVals()->size() == numCurves) {
                        const Imath::V2f* uvs = us.getVals()->get();
                        strandRootUVs.resize(numCurves);
                        for (size_t ci2 = 0; ci2 < numCurves; ++ci2)
                            strandRootUVs[ci2] = { uvs[ci2].x, uvs[ci2].y };
                        spdlog::info("AlembicLoader: loaded {} strand root UVs (first={:.3f},{:.3f})",
                                     numCurves, uvs[0].x, uvs[0].y);
                    }
                }
            }
        }
    }

    // Object name comes directly from the ICurves node name.
    // hair_export writes one OCurves per Blender object, named after it.
    // Falls back to the full Alembic path for archives not written by hair_export.
    const std::string objectName = curvesObj.getName();

    // Record pool size before adding any strands
    const uint32_t poolOffsetBefore = static_cast<uint32_t>(pool.numStrands());

    // ---- iterate curves ----
    const Imath::V3f* pts       = positions->get();
    const Imath::V3f* closePts  = hasClose ? closeSamp.getPositions()->get() : nullptr;
    const int32_t*    nVertsArr = counts->get();
    size_t cvOffset = 0;

    size_t strandsAdded = 0;
    size_t strandsSkipped = 0;

    // Dummy widths reused for close-sample basis conversion (widths don't change).
    static const std::vector<float> kDummyWidths1 = {1.f};

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

        // Convert open sample to endpoint-sharing cubic Bézier
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

        // Convert close sample to Bézier (positions only; widths are constant)
        std::vector<Vec3f> bezCVsClose;
        if (hasClose) {
            std::vector<Vec3f> rawClose(nv);
            for (int32_t i = 0; i < nv; ++i) {
                const auto& p = closePts[cvOffset + i];
                rawClose[i] = xform.transformPoint({p.x, p.y, p.z});
            }
            std::vector<float> dummyW(nv, 1.f);
            std::vector<float> dummyBW;
            if (basisType == kLinear) {
                linearToBezier(rawClose, dummyW, bezCVsClose, dummyBW);
            } else if (basis == kBezierBasis) {
                bezCVsClose = rawClose;
            } else if (basis == kBsplineBasis) {
                bsplineToBezier(rawClose, dummyW, bezCVsClose, dummyBW);
            } else if (basis == kCatmullromBasis) {
                catmullromToBezier(rawClose, dummyW, bezCVsClose, dummyBW);
            }
            if (bezCVsClose.size() != bezCVs.size())
                bezCVsClose.clear();  // mismatch — fall back to static
        }

        StrandDesc strand;
        strand.controlPoints      = std::move(bezCVs);
        strand.controlPointsClose = std::move(bezCVsClose);
        strand.widths             = std::move(bezWidths);
        strand.materialIndex      = opts.baseMaterialIndex;
        if (!strandColors.empty())
            strand.color = strandColors[ci];
        if (!strandRootUVs.empty())
            strand.rootUV = strandRootUVs[ci];
        pool.addStrand(std::move(strand));
        ++strandsAdded;

        cvOffset += nv;
    }

    if (strandsAdded || strandsSkipped) {
        spdlog::info("AlembicLoader: '{}' → {} strands added, {} skipped",
                     curvesObj.getFullName(), strandsAdded, strandsSkipped);
    }

    // Emit one AlembicObjectRange for this ICurves node.
    if (outRanges && strandsAdded > 0) {
        AlembicObjectRange r;
        r.objectName  = objectName;
        r.strandStart = poolOffsetBefore;
        r.strandCount = static_cast<uint32_t>(strandsAdded);
        spdlog::info("AlembicLoader: object '{}' → strands [{}, {})",
                     r.objectName, r.strandStart, r.strandStart + r.strandCount);
        outRanges->push_back(std::move(r));
    }
}

// ---------------------------------------------------------------------------
// Recursive scene-graph traversal
// ---------------------------------------------------------------------------
static void traverse(const Alembic::AbcGeom::IObject& obj,
                     const AlembicCurveOptions&        opts,
                     CurvePool&                        pool,
                     std::vector<AlembicObjectRange>*  outRanges) {
    using namespace Alembic::AbcGeom;

    if (ICurves::matches(obj.getMetaData())) {
        processICurves(ICurves(obj, Alembic::Abc::kWrapExisting), opts, pool,
                       outRanges);
    }

    for (size_t i = 0; i < obj.getNumChildren(); ++i)
        traverse(obj.getChild(i), opts, pool, outRanges);
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------
bool loadAlembicCurves(const std::string&                        path,
                       const AlembicCurveOptions&                opts,
                       CurvePool&                                outPool,
                       std::vector<std::unique_ptr<IMaterial>>&  outMaterials,
                       std::vector<AlembicObjectRange>*          outRanges) {
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
    traverse(archive.getTop(), opts, outPool, outRanges);
    size_t strandsLoaded = outPool.numStrands() - strandsBefore;

    if (strandsLoaded == 0) {
        spdlog::warn("AlembicLoader: no ICurves found in '{}'", path);
    } else {
        spdlog::info("AlembicLoader: {} total strands loaded from '{}'",
                     strandsLoaded, path);
    }

    outMaterials.push_back(std::make_unique<MarschnerHairMaterial>(
        MarschnerHairMaterial::Params{}));

    return true;
}

// ---------------------------------------------------------------------------
// IPoints — particle point cloud support
// ---------------------------------------------------------------------------

// Infer frames-per-second from the schema's first TimeSampling object.
// Returns 0 if timing info is unavailable.
static double inferFPS(const Alembic::AbcGeom::IPointsSchema& schema) {
    auto ts = schema.getTimeSampling();
    if (!ts) return 0.0;
    auto tsType = ts->getTimeSamplingType();
    double tpc = tsType.getTimePerCycle();
    if (tpc <= 0.0) return 0.0;
    // For uniform sampling, tpc == 1/fps; numSamplesPerCycle is usually 1.
    uint32_t spc = tsType.getNumSamplesPerCycle();
    return (spc > 0) ? (static_cast<double>(spc) / tpc) : (1.0 / tpc);
}

static void processIPoints(const Alembic::AbcGeom::IPoints& pointsObj,
                            const AlembicPointsOptions&      opts,
                            HaloPool&                        pool) {
    using namespace Alembic::AbcGeom;

    auto schema = pointsObj.getSchema();
    size_t numSamples = schema.getNumSamples();
    if (numSamples == 0) return;

    // --- resolve time selector for this frame ---
    ISampleSelector sel;  // defaults to index 0
    if (opts.frameNumber != 0 || numSamples > 1) {
        double fps = inferFPS(schema);
        if (fps <= 0.0)
            fps = (opts.framesPerSecond > 0.f) ? opts.framesPerSecond : 24.0;
        double sampleTime = opts.frameNumber / fps;
        sel = ISampleSelector(sampleTime, ISampleSelector::kNearIndex);
    }

    IPointsSchema::Sample samp;
    schema.get(samp, sel);

    auto positions = samp.getPositions();
    if (!positions || positions->size() == 0) return;

    size_t numPoints = positions->size();

    // --- widths / radius ---
    // Blender exports particle radius as a custom float attribute "radius".
    // Standard Alembic width (diameter) may also be present as "widths".
    // We prefer "radius" (Blender convention) over "widths" (RenderMan convention).
    std::vector<float> radii;
    {
        ICompoundProperty arb = schema.getArbGeomParams();
        if (arb.valid()) {
            // Try "radius" attribute first (Blender GN radius = actual radius)
            if (arb.getPropertyHeader("radius")) {
                IFloatGeomParam rParam(arb, "radius");
                if (rParam.valid()) {
                    IFloatGeomParam::Sample rs;
                    rParam.getExpanded(rs, sel);
                    if (rs.getVals() && rs.getVals()->size() > 0) {
                        const float* rd = rs.getVals()->get();
                        size_t rn = rs.getVals()->size();
                        if (rn == numPoints) {
                            radii.resize(numPoints);
                            for (size_t i = 0; i < numPoints; ++i)
                                radii[i] = rd[i] * opts.radiusScale;
                        } else if (rn == 1) {
                            radii.assign(numPoints, rd[0] * opts.radiusScale);
                        }
                    }
                }
            }
            // Fall back to "widths" (Alembic diameter convention, so halve it)
            if (radii.empty() && arb.getPropertyHeader("widths")) {
                IFloatGeomParam wParam(arb, "widths");
                if (wParam.valid()) {
                    IFloatGeomParam::Sample ws;
                    wParam.getExpanded(ws, sel);
                    if (ws.getVals() && ws.getVals()->size() > 0) {
                        const float* wd = ws.getVals()->get();
                        size_t wn = ws.getVals()->size();
                        if (wn == numPoints) {
                            radii.resize(numPoints);
                            for (size_t i = 0; i < numPoints; ++i)
                                radii[i] = wd[i] * 0.5f * opts.radiusScale;
                        } else if (wn == 1) {
                            radii.assign(numPoints, wd[0] * 0.5f * opts.radiusScale);
                        }
                    }
                }
            }
        }

        // Also check the built-in widths param on the schema itself
        if (radii.empty()) {
            IFloatGeomParam wParam = schema.getWidthsParam();
            if (wParam.valid()) {
                IFloatGeomParam::Sample ws;
                wParam.getExpanded(ws, sel);
                if (ws.getVals() && ws.getVals()->size() > 0) {
                    const float* wd = ws.getVals()->get();
                    size_t wn = ws.getVals()->size();
                    if (wn == numPoints) {
                        radii.resize(numPoints);
                        for (size_t i = 0; i < numPoints; ++i)
                            radii[i] = wd[i] * 0.5f * opts.radiusScale;
                    } else if (wn == 1) {
                        radii.assign(numPoints, wd[0] * 0.5f * opts.radiusScale);
                    }
                }
            }
        }
    }
    if (radii.empty())
        radii.assign(numPoints, opts.defaultRadius);

    // --- per-particle color (optional) ---
    std::vector<Vec3f> colors;
    {
        ICompoundProperty arb = schema.getArbGeomParams();
        if (arb.valid() && arb.getPropertyHeader("color")) {
            IC3fGeomParam cParam(arb, "color");
            if (cParam.valid()) {
                IC3fGeomParam::Sample cs;
                cParam.getExpanded(cs, sel);
                if (cs.getVals() && cs.getVals()->size() == numPoints) {
                    const Imath::C3f* cd = cs.getVals()->get();
                    colors.resize(numPoints);
                    for (size_t i = 0; i < numPoints; ++i)
                        colors[i] = {cd[i].x, cd[i].y, cd[i].z};
                }
            }
        }
    }

    // --- world transform ---
    Mat4f xform = getWorldXform(pointsObj);

    const Imath::V3f* pts = positions->get();
    for (size_t i = 0; i < numPoints; ++i) {
        const auto& p = pts[i];
        HaloDesc h;
        h.center = xform.transformPoint({p.x, p.y, p.z});
        h.radius = radii[i];
        h.color  = colors.empty() ? Vec3f{1.f, 1.f, 1.f} : colors[i];
        h.matIdx = opts.baseMaterialIndex;
        pool.addHalo(h);
    }

    spdlog::info("AlembicLoader: IPoints '{}' → {} particles @ frame {} ({} sample(s) total)",
                 pointsObj.getFullName(), numPoints, opts.frameNumber, numSamples);
}

static void traverseForPoints(const Alembic::AbcGeom::IObject& obj,
                               const AlembicPointsOptions&      opts,
                               HaloPool&                        pool) {
    using namespace Alembic::AbcGeom;

    if (IPoints::matches(obj.getMetaData())) {
        processIPoints(IPoints(obj, Alembic::Abc::kWrapExisting), opts, pool);
    }

    for (size_t i = 0; i < obj.getNumChildren(); ++i)
        traverseForPoints(obj.getChild(i), opts, pool);
}

bool loadAlembicPoints(const std::string&                        path,
                       const AlembicPointsOptions&               opts,
                       HaloPool&                                 outPool,
                       std::vector<std::unique_ptr<IMaterial>>&  outMaterials) {
    using namespace Alembic;

    AbcCoreFactory::IFactory factory;
    AbcCoreFactory::IFactory::CoreType coreType;
    Abc::IArchive archive = factory.getArchive(path, coreType);

    if (!archive.valid()) {
        spdlog::error("AlembicLoader: cannot open particles file '{}'", path);
        return false;
    }

    spdlog::info("AlembicLoader: loading particles from '{}' ({})",
                 path,
                 coreType == AbcCoreFactory::IFactory::kOgawa ? "Ogawa" : "HDF5");

    size_t halosBefore = outPool.numHalos();
    traverseForPoints(archive.getTop(), opts, outPool);
    size_t halosLoaded = outPool.numHalos() - halosBefore;

    if (halosLoaded == 0) {
        spdlog::warn("AlembicLoader: no IPoints objects found in '{}'", path);
    } else {
        spdlog::info("AlembicLoader: {} total particles loaded from '{}'",
                     halosLoaded, path);
    }

    // Default soft-particle material: white additive emitter.
    // Callers can override per-particle matIdx after loading if needed.
    outMaterials.push_back(std::make_unique<SoftParticleMaterial>());

    return true;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_ALEMBIC
