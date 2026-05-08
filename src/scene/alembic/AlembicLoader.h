#pragma once

#ifdef ANACAPA_ENABLE_ALEMBIC

#include <anacapa/accel/CurvePool.h>
#include <anacapa/shading/IMaterial.h>
#include <memory>
#include <string>
#include <vector>

namespace anacapa {

struct AlembicCurveOptions {
    float    defaultWidth  = 0.005f;  // fallback radius when no width channel present
    float    widthScale    = 1.0f;    // multiply all widths by this factor
    // baseMaterialIndex: index of the first material created by this call in
    // the owning scene's materials array.  All strands receive this index
    // (one MarschnerHairMaterial is created per loadAlembicCurves call).
    uint32_t baseMaterialIndex = 0;
    // When true, load sample 0 as shutter-open CVs and sample 1 (if present)
    // as shutter-close CVs, storing them in StrandDesc::controlPointsClose.
    bool     motionBlur    = false;
};

// ---------------------------------------------------------------------------
// AlembicObjectRange
//
// Describes one named hair object's strand range and its USD material path.
// Returned from loadAlembicCurves so the caller can assign per-object
// material indices via CurvePool::setMaterialIndex.
// ---------------------------------------------------------------------------
struct AlembicObjectRange {
    std::string objectName;   // Blender/DCC object name — matches matassign.json "object" keys
    uint32_t    strandStart;  // index of first strand in CurvePool for this object
    uint32_t    strandCount;  // number of strands belonging to this object
};

// ---------------------------------------------------------------------------
// loadAlembicCurves
//
// Reads all ICurves objects from an Alembic .abc file and appends cubic
// Bézier strands (endpoint-sharing, 3n+1 CVs) to outPool.
//
// Curve type support:
//   kLinear            — converted to cubic Bézier (collinear inner CVs)
//   kCubic/kBezierBasis — used as-is (assumes 3n+1 endpoint-sharing CVs)
//   kCubic/kBsplineBasis  — converted to Bézier (most common from Blender)
//   kCubic/kCatmullromBasis — converted to Bézier
//   other              — skipped with a warning
//
// One default MarschnerHairMaterial is appended to outMaterials.  All strands
// initially get materialIndex = opts.baseMaterialIndex.  The caller should
// use outRanges + a MatAssignLoader result to override per-object materials.
//
// Returns true if the archive was opened successfully (even if it has no
// ICurves objects).  Returns false if the file could not be opened.
// ---------------------------------------------------------------------------
// outRanges (optional): receives one AlembicObjectRange per named hair object
// found in the sidecar "materialPath" attribute.  Caller may pass nullptr.
bool loadAlembicCurves(const std::string&                        path,
                       const AlembicCurveOptions&                opts,
                       CurvePool&                                outPool,
                       std::vector<std::unique_ptr<IMaterial>>&  outMaterials,
                       std::vector<AlembicObjectRange>*          outRanges = nullptr);

} // namespace anacapa

#endif // ANACAPA_ENABLE_ALEMBIC
