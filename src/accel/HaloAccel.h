#pragma once

#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/accel/HaloPool.h>
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// HaloNode — binary SAH BVH node over halo disc particles.
//
// Interior: left_or_prim = left child index,  right_or_count = right child index
// Leaf:     left_or_prim = first index in m_primIdx,
//           right_or_count = count | 0x80000000
//
// Each leaf AABB is the sphere bounding box [center-r, center+r] of the halos
// it contains.  Intersection tests the actual camera-facing disc at traversal.
// ---------------------------------------------------------------------------
struct HaloNode {
    float    bmin[3], bmax[3];
    uint32_t left_or_prim;
    uint32_t right_or_count;

    bool     isLeaf()    const { return (right_or_count & 0x80000000u) != 0; }
    uint32_t primCount() const { return  right_or_count & 0x7FFFFFFFu; }
};

// ---------------------------------------------------------------------------
// HaloAccel — spatial acceleration structure for camera-facing disc particles.
//
// Intersection: a disc whose normal is recomputed per-ray as
//   N = normalize(ray.origin - halo.center)
// so the disc always faces the incoming ray origin.  This gives the classic
// "halo" / billboard look in a path tracer.
//
// Phase 1 — static particles only.  Per-frame animation (Alembic IPoints)
// will be added in Phase 2 using the same motion-BVH pattern as CurveBrute.
// ---------------------------------------------------------------------------
class HaloAccel {
public:
    explicit HaloAccel(const HaloPool& pool);

    void commit();

    TraceResult trace(const Ray& ray) const;
    bool        occluded(const Ray& ray) const;

    // GPU upload accessors
    const HaloPool&               pool()    const { return m_pool; }
    const std::vector<HaloNode>&  nodes()   const { return m_nodes; }
    const std::vector<uint32_t>&  primIdx() const { return m_primIdx; }

private:
    const HaloPool&        m_pool;
    std::vector<HaloNode>  m_nodes;
    std::vector<uint32_t>  m_primIdx;  // leaf prim index array → HaloPool halo id
};

} // namespace anacapa
