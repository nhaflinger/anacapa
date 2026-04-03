#pragma once

#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/accel/GeometryPool.h>
#include <array>
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// BVH Node — 32 bytes, cache-line friendly
//
// Uses plain float[3] for bounds (not Vec3f) to avoid alignas(16) bloat.
//
// Encoding (32 bytes total):
//   boundsMin[3]  + dataA  (16 bytes)
//   boundsMax[3]  + dataB  (16 bytes)
//
//   Interior: dataA = rightChild index, dataB = splitAxis
//   Leaf:     dataA = primOffset,       dataB = primCount | kLeafFlag
//
// Left child of an interior node is always at nodeIndex+1 (depth-first).
// ---------------------------------------------------------------------------
struct BVHNode {
    float    boundsMin[3];
    uint32_t dataA;        // Interior: rightChild;  Leaf: primOffset
    float    boundsMax[3];
    uint32_t dataB;        // Interior: splitAxis;   Leaf: primCount | kLeafFlag

    static constexpr uint32_t kLeafFlag = 0x80000000u;

    bool     isLeaf()     const { return (dataB & kLeafFlag) != 0; }
    uint32_t rightChild() const { return dataA; }
    uint32_t primOffset() const { return dataA; }
    uint32_t primCount()  const { return dataB & ~kLeafFlag; }
    uint8_t  splitAxis()  const { return static_cast<uint8_t>(dataB); }
};
static_assert(sizeof(BVHNode) == 32, "BVHNode must be 32 bytes");

// ---------------------------------------------------------------------------
// Triangle — flattened, world-space, for fast intersection
// Precomputed edges avoid redundant subtractions in the Möller–Trumbore test.
// ---------------------------------------------------------------------------
struct BVHTriangle {
    Vec3f v0;
    Vec3f e1;        // v1 - v0
    Vec3f e2;        // v2 - v0
    Vec3f n;         // Geometric normal (normalized)
    Vec2f uv0, uv1, uv2;
    Vec3f sn0, sn1, sn2;  // Shading normals per vertex
    uint32_t meshID;
    uint32_t primID;
};

// ---------------------------------------------------------------------------
// BVHBackend — CPU SAH BVH over a GeometryPool
// ---------------------------------------------------------------------------
class BVHBackend : public IAccelerationStructure {
public:
    static constexpr int   kSAHBuckets      = 12;   // SAH evaluation buckets
    static constexpr int   kMaxLeafPrims    = 4;    // Max triangles per leaf
    static constexpr float kTraversalCost   = 1.f;  // Relative to intersection
    static constexpr float kIntersectCost   = 1.f;

    explicit BVHBackend(const GeometryPool& pool);

    void commit() override;

    TraceResult trace(const Ray& ray) const override;
    bool        occluded(const Ray& ray) const override;

    const GeometryPool& pool() const override { return m_pool; }

private:
    // -----------------------------------------------------------------------
    // Build
    // -----------------------------------------------------------------------
    struct PrimInfo {
        BBox3f bounds;
        Vec3f  centroid;
        uint32_t originalIndex;  // Index into m_tris
    };

    uint32_t buildRecursive(std::vector<PrimInfo>& primInfo,
                            uint32_t start, uint32_t end);

    // SAH split: returns best axis and split bucket index, or -1 for leaf
    int sahSplit(const std::vector<PrimInfo>& primInfo,
                 uint32_t start, uint32_t end,
                 const BBox3f& centroidBounds,
                 int& outAxis) const;

    // -----------------------------------------------------------------------
    // Traversal helpers
    // -----------------------------------------------------------------------
    struct Ray4 {
        Vec3f origin;
        Vec3f invDir;     // 1/direction (precomputed)
        int   dirNeg[3];  // sign(direction[i]) — used for slab test
        float tMin, tMax;
    };

    static Ray4 makeRay4(const Ray& ray);

    // Returns true if ray intersects the AABB, updates tMin/tMax
    static bool intersectAABB(const BVHNode& node, const Ray4& r,
                               float& tNear);

    // Möller–Trumbore ray-triangle intersection
    // Returns true on hit; fills t, u, v barycentric coordinates
    static bool intersectTriangle(const BVHTriangle& tri, const Ray4& r,
                                  float& t, float& u, float& v);

    void fillSurfaceInteraction(const BVHTriangle& tri,
                                float t, float u, float v,
                                SurfaceInteraction& si) const;

    // Actual traversal implementation (trace() delegates here)
    TraceResult traceImpl(const Ray& ray) const;

    // -----------------------------------------------------------------------
    // Data
    // -----------------------------------------------------------------------
    const GeometryPool&        m_pool;
    std::vector<BVHNode>       m_nodes;
    std::vector<BVHTriangle>   m_tris;      // Reordered by build
    std::vector<uint32_t>      m_primIndices; // Leaf prim index list
    bool                       m_built = false;
};

} // namespace anacapa
