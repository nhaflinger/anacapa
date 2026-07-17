#pragma once

#include <anacapa/accel/IAccelerationStructure.h>
#include <anacapa/accel/GeometryPool.h>
#include <array>
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// BVHNode — 128 bytes (2 cache lines), SOA 4-child layout.
//
// Each node stores the bounds of up to FOUR children in structure-of-arrays
// order so all four can be tested with a single full-width SIMD operation.
//
//   minX[4], minY[4], minZ[4]  — children AABB mins   (48 bytes, first cache line)
//   maxX[4], maxY[4], maxZ[4]  — children AABB maxs   (48 bytes, straddles lines)
//   childData[4]               — interior: child node index; leaf: primOffset
//   childMeta[4]               — interior: 0; leaf: primCount | kLeafFlag
//
// Unused slots (when a node has < 4 children) carry inverted AABBs that
// never intersect any ray, so SIMD tests remain branchless and correct.
//
// The BVH4 is built from a temporary binary SAH tree by collapsing one
// binary level per BVH4 level: each binary interior node's two children are
// expanded one step deeper, yielding 2-4 grandchildren per BVH4 node.
// ---------------------------------------------------------------------------
struct BVHNode {
    float    minX[4], minY[4], minZ[4];  // 48 bytes — children AABB mins
    float    maxX[4], maxY[4], maxZ[4];  // 48 bytes — children AABB maxs
    uint32_t childData[4];               // 16 bytes
    uint32_t childMeta[4];               // 16 bytes

    static constexpr uint32_t kLeafFlag = 0x80000000u;

    bool     isLeaf(int c)     const { return (childMeta[c] & kLeafFlag) != 0; }
    uint32_t primOffset(int c) const { return childData[c]; }
    uint32_t primCount(int c)  const { return childMeta[c] & ~kLeafFlag; }
    uint32_t childIdx(int c)   const { return childData[c]; }
};
static_assert(sizeof(BVHNode) == 128, "BVHNode must be 128 bytes");

// ---------------------------------------------------------------------------
// BuildBVHNode — temporary node format used only during SAH build.
// ---------------------------------------------------------------------------
struct BuildBVHNode {
    float    boundsMin[3];
    uint32_t dataA;   // Interior: rightChild index;  Leaf: primOffset
    float    boundsMax[3];
    uint32_t dataB;   // Interior: splitAxis;          Leaf: primCount | kLeafFlag

    static constexpr uint32_t kLeafFlag = 0x80000000u;
    bool     isLeaf()     const { return (dataB & kLeafFlag) != 0; }
    uint32_t rightChild() const { return dataA; }
    uint32_t primOffset() const { return dataA; }
    uint32_t primCount()  const { return dataB & ~kLeafFlag; }
};

// ---------------------------------------------------------------------------
// BVHTriTrav — traversal-only record, 64 bytes (one cache line).
// ---------------------------------------------------------------------------
struct BVHTriTrav {
    Vec3f    v0;
    Vec3f    e1;        // v1 - v0
    Vec3f    e2;        // v2 - v0
    uint32_t data;      // meshID | (isObjectSpace << 31)

    static constexpr uint32_t kObjectSpaceFlag = 0x80000000u;
    bool     isObjectSpace() const { return (data & kObjectSpaceFlag) != 0; }
    uint32_t meshID()        const { return  data & ~kObjectSpaceFlag; }
};
static_assert(sizeof(BVHTriTrav) == 64, "BVHTriTrav must be 64 bytes (one cache line)");

// ---------------------------------------------------------------------------
// BVHTriAttrib — hit attributes, only loaded on confirmed intersection.
// ---------------------------------------------------------------------------
struct BVHTriAttrib {
    Vec3f    n;
    Vec3f    sn0, sn1, sn2;
    Vec4f    t0, t1, t2;    // xyz = tangent, w = handedness sign (MeshDesc::tangents)
    Vec2f    uv0, uv1, uv2;
    uint32_t primID;
};
static_assert(sizeof(BVHTriAttrib) == 144, "BVHTriAttrib must be 144 bytes");

// ---------------------------------------------------------------------------
// BLAS — bottom-level acceleration structure for one prototype mesh.
// ---------------------------------------------------------------------------
struct BLAS {
    std::vector<BVHNode>      nodes;
    std::vector<BVHTriTrav>   trav;
    std::vector<BVHTriAttrib> attribs;
    std::vector<uint32_t>     primIndices;
};

// ---------------------------------------------------------------------------
// TLASInstance — one entry in the TLAS primitive list.
// ---------------------------------------------------------------------------
// Per-instance world-space AABB stored alongside groupID/instanceIdx so
// traversal can reject false-positive leaf hits without a full BLAS traversal.
struct TLASInstance {
    uint32_t groupID;
    uint32_t instanceIdx;
    float    bMin[3];   // world-space AABB min
    float    bMax[3];   // world-space AABB max
};

// ---------------------------------------------------------------------------
// BVHBackend — CPU SAH BVH4 over a GeometryPool
//
// Two-level structure when instance groups are present:
//   TLAS — BVH4 over per-instance world-space AABBs
//   BLAS — one BVH4 per prototype mesh (object-space triangles)
// Non-instanced meshes use the original single-level BVH.
// ---------------------------------------------------------------------------
class BVHBackend : public IAccelerationStructure {
public:
    static constexpr int   kSAHBuckets      = 12;
    static constexpr int   kSpatialBuckets  = 16;
    static constexpr int   kMaxLeafPrims    = 4;
    static constexpr int   kTLASMaxLeafPrims = 8;
    static constexpr float kTraversalCost   = 1.f;
    static constexpr float kIntersectCost   = 1.f;
    static constexpr float kSpatialAlpha    = 1e-5f;  // skip spatial splits on tiny nodes
    static constexpr float kMaxRefFactor    = 3.0f;   // max total refs = n * factor

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
        BBox3f   bounds;
        Vec3f    centroid;
        uint32_t originalIndex;
        Vec3f    v0, v1, v2;  // world-space vertices for spatial split clipping
        bool     animated;    // if true, skip spatial splits for this ref
    };

    // Build state threaded through recursive calls (totalRefs is mutable during build).
    struct BuildState {
        uint32_t totalRefs = 0;
    };

    uint32_t buildRecursive(std::vector<PrimInfo>& primInfo,
                            uint32_t start, uint32_t end,
                            std::vector<BuildBVHNode>& buildNodes,
                            std::vector<uint32_t>& primIndices,
                            BuildState& state,
                            int maxLeafPrims = kMaxLeafPrims);

    // Returns best SAH bucket (or -1 if no valid split). Caller compares outCost vs leafCost.
    int sahSplit(const std::vector<PrimInfo>& primInfo,
                 uint32_t start, uint32_t end,
                 const BBox3f& centroidBounds,
                 const BBox3f& nodeBounds,
                 int& outAxis,
                 float& outCost) const;

    // Evaluates spatial split candidates; returns best SAH cost (infinity if none useful).
    float spatialSplit(const std::vector<PrimInfo>& primInfo,
                       uint32_t start, uint32_t end,
                       const BBox3f& nodeBounds,
                       int& outAxis, float& outPos) const;

    // Clips triangle v[0..2] to the slab [sMin,sMax] on axis; returns clipped bbox.
    static BBox3f clipTriangleSlab(const Vec3f v[3], int axis, float sMin, float sMax);

    // Convert binary BuildBVHNode tree → BVH4 SOA BVHNode array.
    void repackBuildTree(const std::vector<BuildBVHNode>& build,
                         std::vector<BVHNode>& nodes);

    // Recursively build a BVH4 node for the subtree rooted at build[oldIdx].
    uint32_t buildBVH4Node(const std::vector<BuildBVHNode>& build, uint32_t oldIdx,
                           std::vector<BVHNode>& nodes);

    // Build a BLAS for one instance group's prototype mesh.
    void buildBLAS(uint32_t groupID);

    // Build the TLAS over all instance groups.
    void buildTLAS();

    // -----------------------------------------------------------------------
    // Traversal helpers
    // -----------------------------------------------------------------------
    struct Ray4 {
        Vec3f origin;
        Vec3f invDir;
        float tMin, tMax;
    };

    static Ray4 makeRay4(const Ray& ray);
    static Ray4 makeObjectSpaceRay4(const Ray& ray, const Mat4f& worldToObject);

    // Test all four children of node simultaneously using SIMD.
    static int intersectAABB4(const BVHNode& node, const Ray4& r,
                               float tNear[4], float closestT);

    static bool intersectTriangle(const BVHTriTrav& trav, const Ray4& r,
                                  float& t, float& u, float& v);

    void fillSurfaceInteraction(const BVHTriTrav& trav, const BVHTriAttrib& attrib,
                                float t, float u, float v,
                                const Mat4f* worldXfm,
                                SurfaceInteraction& si) const;

    // Traverse BLAS for one instance; update closestT/hit on success.
    void traceBLAS(const BLAS& blas, const Ray& ray, const Ray4& objRay,
                   uint32_t groupID, uint32_t instanceIdx, const Mat4f& o2w,
                   float& closestT, uint32_t& hitBlasTriIdx,
                   float& hitU, float& hitV,
                   uint32_t& hitGroupID, uint32_t& hitInstanceIdx) const;

    // Shadow variant — returns true immediately on any hit.
    bool occludedBLAS(const BLAS& blas, const Ray4& objRay) const;

    TraceResult traceImpl(const Ray& ray) const;

    // -----------------------------------------------------------------------
    // Data
    // -----------------------------------------------------------------------
    const GeometryPool&        m_pool;

    // Single-level BVH for non-instanced meshes
    std::vector<BVHNode>       m_nodes;
    std::vector<BVHTriTrav>    m_trav;
    std::vector<BVHTriAttrib>  m_attribs;
    std::vector<uint32_t>      m_primIndices;

    // Two-level BVH for instance groups
    std::vector<BLAS>          m_blasList;       // one per instance group
    std::vector<BVHNode>       m_tlasNodes;
    std::vector<TLASInstance>  m_tlasInstances;  // indexed by m_tlasPrimIndices
    std::vector<uint32_t>      m_tlasPrimIndices;

    bool m_built = false;

    // Spatial split budget — set before each build pass (single-BVH or BLAS).
    mutable float    m_buildRootSA    = 0.f;
    mutable uint32_t m_buildMaxRefs   = 0;
};

} // namespace anacapa
