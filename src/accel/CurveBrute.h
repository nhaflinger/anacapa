#pragma once

#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/accel/CurvePool.h>
#include "BVHBackend.h"
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// SegRef — identifies one cubic Bézier segment within the CurvePool.
// ---------------------------------------------------------------------------
struct SegRef {
    uint32_t strandIdx;
    uint32_t segIdx;
};

// ---------------------------------------------------------------------------
// CurveNode — a node in the flat SAH-like BVH over curve segments.
//
// Interior node:  left_or_prim  = left child index
//                 right_or_count = right child index  (high bit clear)
// Leaf node:      left_or_prim  = first SegRef index in m_segRefs
//                 right_or_count = prim count | 0x80000000
// ---------------------------------------------------------------------------
struct CurveNode {
    float    bmin[3], bmax[3];
    uint32_t left_or_prim;    // interior: left child; leaf: first prim idx
    uint32_t right_or_count;  // interior: right child; leaf: count | 0x80000000

    bool     isLeaf()    const { return (right_or_count & 0x80000000u) != 0; }
    uint32_t primCount() const { return  right_or_count & 0x7FFFFFFFu; }
};

// ---------------------------------------------------------------------------
// CurveBrute — triangle BVH + SAH BVH over all curve segments.
//
// Triangle intersection: O(log N) via BVHBackend.
// Curve intersection:    O(log S) via CurveNode BVH over individual segments,
//                        then recursive de Casteljau subdivision (depth 6) at
//                        each leaf.  Replaces the former O(S) brute-force loop.
//
// Material lookup: si.meshID = strand.materialIndex, si.isCurve = true,
//                  si.strandID = index into CurvePool.
// ---------------------------------------------------------------------------
class CurveBrute : public IAccelerationStructure {
public:
    CurveBrute(const GeometryPool& triPool, const CurvePool& curvePool);

    void commit() override;

    TraceResult trace(const Ray& ray)    const override;
    bool        occluded(const Ray& ray) const override;

    const GeometryPool& pool()      const override { return m_triBvh.pool(); }
    const CurvePool&    curvePool() const          { return m_curvePool; }

private:
    static bool aabbHit(const float bmin[3], const float bmax[3],
                        const Ray& ray, float maxT);

    BVHBackend             m_triBvh;
    const CurvePool&       m_curvePool;
    std::vector<CurveNode> m_curveNodes;  // flat BVH over curve segments
    std::vector<SegRef>    m_segRefs;     // leaf prim array (ordered by build)
};

} // namespace anacapa
