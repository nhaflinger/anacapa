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
// One MarschnerHairMaterial with default parameters is created and appended
// to outMaterials.  All strands get materialIndex = opts.baseMaterialIndex.
//
// Returns true if the archive was opened successfully (even if it has no
// ICurves objects).  Returns false if the file could not be opened.
// ---------------------------------------------------------------------------
bool loadAlembicCurves(const std::string&                        path,
                       const AlembicCurveOptions&                opts,
                       CurvePool&                                outPool,
                       std::vector<std::unique_ptr<IMaterial>>&  outMaterials);

} // namespace anacapa

#endif // ANACAPA_ENABLE_ALEMBIC
