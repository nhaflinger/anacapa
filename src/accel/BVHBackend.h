#pragma once

#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/accel/GeometryPool.h>
#include <array>
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// BVHNode — 64 bytes (one cache line), SOA dual-child layout.
//
// Each node stores the bounds of its TWO children in structure-of-arrays
// order so that both can be tested with a single SIMD operation.
//
//   minX[0..1], minY[0..1], minZ[0..1]  — children's AABB mins
//   maxX[0..1], maxY[0..1], maxZ[0..1]  — children's AABB maxs
//   childData[i] — interior: child node index; leaf: primOffset
//   childMeta[i] — interior: 0;            leaf: primCount | kLeafFlag
//
// Only interior nodes occupy a slot in m_nodes.  Leaves are encoded
// inline in their parent's childData/childMeta.
// ---------------------------------------------------------------------------
struct BVHNode {
    float    minX[2], minY[2], minZ[2];  // 24 bytes — children AABB mins
    float    maxX[2], maxY[2], maxZ[2];  // 24 bytes — children AABB maxs
    uint32_t childData[2];               //  8 bytes
    uint32_t childMeta[2];               //  8 bytes

    static constexpr uint32_t kLeafFlag = 0x80000000u;

    bool     isLeaf(int c)     const { return (childMeta[c] & kLeafFlag) != 0; }
    uint32_t primOffset(int c) const { return childData[c]; }
    uint32_t primCount(int c)  const { return childMeta[c] & ~kLeafFlag; }
    uint32_t childIdx(int c)   const { return childData[c]; }
};
static_assert(sizeof(BVHNode) == 64, "BVHNode must be 64 bytes");

// ---------------------------------------------------------------------------
// BuildBVHNode — temporary node format used only during SAH build.
//
// Classic own-bounds layout with depth-first left-child ordering.
// After buildRecursive() completes, repackBuildTree() converts this
// into the final SOA BVHNode format above.
// ---------------------------------------------------------------------------
struct BuildBVHNode {
    float    boundsMin[3];
    uint32_t dataA;   // Interior: rightChild index;  Leaf: primOffset
    float    boundsMax[3];
    uint32_t dataB;   // Interior: splitAxis;          Leaf: primCount | kLeafFlag

    static constexpr uint32_t kLeafFlag = 0x80000000u;
    bool     isLeaf()      const { return (dataB & kLeafFlag) != 0; }
    uint32_t rightChild()  const { return dataA; }
    uint32_t primOffset()  const { return dataA; }
    uint32_t primCount()   const { return dataB & ~kLeafFlag; }
};

// ---------------------------------------------------------------------------
// BVHTriTrav — traversal-only record, 64 bytes (one cache line).
//
// Contains only the data needed for Möller-Trumbore intersection and the
// animated-mesh transform lookup.  Kept separate from BVHTriAttrib so the
// hot traversal loop never fetches shading data it doesn't need.
//
// data encoding:  bits[30:0] = meshID,  bit[31] = isObjectSpace flag.
// Static meshes:  v0/e1/e2 in world space (isObjectSpace=false).
// Animated meshes: v0/e1/e2 in object space (isObjectSpace=true);
//   transform is interpolated at ray.time via MeshDesc::interpolateO2W.
// ---------------------------------------------------------------------------
struct BVHTriTrav {
    Vec3f    v0;    // 12 bytes
    Vec3f    e1;    // 12 bytes — v1 - v0
    Vec3f    e2;    // 12 bytes — v2 - v0
    uint32_t data;  //  4 bytes — meshID | (isObjectSpace << 31)

    static constexpr uint32_t kObjectSpaceFlag = 0x80000000u;
    bool     isObjectSpace() const { return (data & kObjectSpaceFlag) != 0; }
    uint32_t meshID()        const { return  data & ~kObjectSpaceFlag; }
};
static_assert(sizeof(BVHTriTrav) == 64, "BVHTriTrav must be 64 bytes (one cache line)");

// ---------------------------------------------------------------------------
// BVHTriAttrib — hit attributes, only loaded on confirmed intersection.
//
// Shading normals, UVs, and primID are irrelevant during traversal and live
// in a separate parallel array so they don't pollute traversal cache lines.
// Static meshes: n/sn* pre-baked to world space.
// Animated meshes: n/sn* in object space (transformed in fillSurfaceInteraction).
// ---------------------------------------------------------------------------
struct BVHTriAttrib {
    Vec3f    n;              // 12 bytes — geometric normal (normalized)
    Vec3f    sn0, sn1, sn2; // 36 bytes — shading normals per vertex
    Vec2f    uv0, uv1, uv2; // 24 bytes — texture coordinates per vertex
    uint32_t primID;         //  4 bytes
};
static_assert(sizeof(BVHTriAttrib) == 96, "BVHTriAttrib must be 96 bytes");

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
        uint32_t originalIndex;
    };

    // Recursive SAH build into a temporary BuildBVHNode array.
    uint32_t buildRecursive(std::vector<PrimInfo>& primInfo,
                            uint32_t start, uint32_t end,
                            std::vector<BuildBVHNode>& buildNodes);

    // SAH split: returns best split bucket, or -1 for leaf
    int sahSplit(const std::vector<PrimInfo>& primInfo,
                 uint32_t start, uint32_t end,
                 const BBox3f& centroidBounds,
                 int& outAxis) const;

    // Convert BuildBVHNode tree → final SOA BVHNode array in m_nodes.
    void repackBuildTree(const std::vector<BuildBVHNode>& build);

    // Recursive helper: store child c's bounds + data into m_nodes[parentIdx],
    // and if the child is interior allocate a new node and descend.
    void repackChild(const std::vector<BuildBVHNode>& build,
                     uint32_t oldIdx, uint32_t parentIdx, int slot);

    // -----------------------------------------------------------------------
    // Traversal helpers
    // -----------------------------------------------------------------------
    struct Ray4 {
        Vec3f origin;
        Vec3f invDir;  // 1/direction (precomputed, NaN-safe)
        float tMin, tMax;
    };

    static Ray4 makeRay4(const Ray& ray);
    static Ray4 makeObjectSpaceRay4(const Ray& ray, const Mat4f& worldToObject);

    // Test both children of node simultaneously using SIMD.
    // Returns hit mask: bit 0 = child 0 hit, bit 1 = child 1 hit.
    // tNear[c] is set for each hit child (entry distance).
    // closestT is used as tMax so already-found hits prune the test.
    //
    // Platform dispatch:
    //   __aarch64__ → NEON float32x2_t (2-wide, natural fit for 2 children)
    //   __SSE2__    → SSE  __m128 lower 2 lanes
    //   fallback    → scalar loop (compiler may auto-vectorize)
    static int intersectAABB2(const BVHNode& node, const Ray4& r,
                               float tNear[2], float closestT);

    // Möller–Trumbore ray-triangle intersection
    static bool intersectTriangle(const BVHTriTrav& trav, const Ray4& r,
                                  float& t, float& u, float& v);

    void fillSurfaceInteraction(const BVHTriTrav& trav, const BVHTriAttrib& attrib,
                                float t, float u, float v,
                                const Mat4f* worldXfm,
                                SurfaceInteraction& si) const;

    TraceResult traceImpl(const Ray& ray) const;

    // -----------------------------------------------------------------------
    // Data
    // -----------------------------------------------------------------------
    const GeometryPool&        m_pool;
    std::vector<BVHNode>       m_nodes;
    std::vector<BVHTriTrav>    m_trav;
    std::vector<BVHTriAttrib>  m_attribs;
    std::vector<uint32_t>      m_primIndices;
    bool                       m_built = false;
};

} // namespace anacapa
