#pragma once

#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/accel/CurvePool.h>
#include "BVHBackend.h"
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// CpuHairTri — baked per-triangle data for pre-tessellated hair ribbons.
// primID from the hair BVH indexes directly into the CpuHairTri array.
// Geometry stored in Möller-Trumbore form (v0, e1=v1-v0, e2=v2-v0).
// ---------------------------------------------------------------------------
struct CpuHairTri {
    Vec3f    v0, e1, e2;  // world-space triangle (shutter-open)
    Vec3f    tangent;     // fiber direction at quad midpoint (unit, open-time)
    Vec3f    ribbonN;     // outward ribbon normal (world-space, open-time)
    Vec3f    color;       // per-strand linear RGB
    float    h0, h1, h2; // impact parameter at barycentric verts 0/1/2
    uint32_t matIdx;      // strand.materialIndex → scene.materials[]
};

// ---------------------------------------------------------------------------
// CpuHairTriClose — close-time geometry, kept parallel to m_hairTris.
// Only populated when at least one strand carries motion keys, so static
// scenes pay zero extra memory.  Shading attributes (tangent, ribbonN) stay
// open-time only — the same approximation the GPU backends use.
// ---------------------------------------------------------------------------
struct CpuHairTriClose {
    Vec3f v0, e1, e2;
};

// ---------------------------------------------------------------------------
// HairNode — binary SAH BVH node over tessellated hair triangles.
//
// Interior: left_or_prim = left child, right_or_count = right child
// Leaf:     left_or_prim = first index in m_hairPrimIdx,
//           right_or_count = count | 0x80000000
// ---------------------------------------------------------------------------
struct HairNode {
    float    bmin[3], bmax[3];
    uint32_t left_or_prim;
    uint32_t right_or_count;

    bool     isLeaf()    const { return (right_or_count & 0x80000000u) != 0; }
    uint32_t primCount() const { return  right_or_count & 0x7FFFFFFFu; }
};

// ---------------------------------------------------------------------------
// CurveBrute — mesh triangle BVH4 + pre-tessellated hair ribbon triangle BVH.
//
// Triangle intersection: O(log N) via BVHBackend (BVH4 + SIMD).
// Hair intersection:     O(log T) via HairNode binary BVH, then O(1)
//                        Möller-Trumbore per leaf triangle. No per-ray
//                        recursive subdivision.
//
// Each cubic Bézier segment is tessellated into tessSteps quads (2*tessSteps
// triangles) at commit() time. tessSteps=4 matches the GPU default.
// ---------------------------------------------------------------------------
class CurveBrute : public IAccelerationStructure {
public:
    static constexpr int kDefaultTessSteps = 4;

    CurveBrute(const GeometryPool& triPool, const CurvePool& curvePool,
               int tessSteps = kDefaultTessSteps);

    void commit() override;

    TraceResult trace(const Ray& ray)    const override;
    bool        occluded(const Ray& ray) const override;

    const GeometryPool& pool()      const override { return m_triBvh.pool(); }
    const CurvePool&    curvePool() const          { return m_curvePool; }

private:
    BVHBackend                   m_triBvh;
    const CurvePool&             m_curvePool;
    int                          m_tessSteps;
    std::vector<HairNode>        m_hairNodes;    // binary BVH over hair triangles
    std::vector<uint32_t>        m_hairPrimIdx;  // leaf prim array → m_hairTris index
    std::vector<CpuHairTri>      m_hairTris;     // baked geometry + shade data (open-time)
    std::vector<CpuHairTriClose> m_hairTrisClose; // close-time positions; empty = static
};

} // namespace anacapa
