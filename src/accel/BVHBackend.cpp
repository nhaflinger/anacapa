#include "BVHBackend.h"
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <spdlog/spdlog.h>

// Platform-specific SIMD headers — guarded so non-SIMD builds stay clean.
#if defined(__aarch64__)
#  include <arm_neon.h>
#elif defined(__SSE2__)
#  include <xmmintrin.h>
#  include <emmintrin.h>
#endif

namespace anacapa {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
BVHBackend::BVHBackend(const GeometryPool& pool)
    : m_pool(pool)
{}

void BVHBackend::commit() {
    m_trav.clear();
    m_attribs.clear();
    m_nodes.clear();
    m_primIndices.clear();

    for (uint32_t meshIdx = 0; meshIdx < m_pool.numMeshes(); ++meshIdx) {
        const MeshDesc& mesh = m_pool.mesh(meshIdx);
        const Mat4f xfm = Mat4f::identity();

        uint32_t numTris = mesh.numTriangles();
        for (uint32_t ti = 0; ti < numTris; ++ti) {
            uint32_t i0 = mesh.indices[ti * 3 + 0];
            uint32_t i1 = mesh.indices[ti * 3 + 1];
            uint32_t i2 = mesh.indices[ti * 3 + 2];

            BVHTriTrav   trav;
            BVHTriAttrib attrib;

            if (mesh.hasMotion()) {
                trav.v0 = mesh.positions[i0];
                Vec3f v1 = mesh.positions[i1];
                Vec3f v2 = mesh.positions[i2];
                trav.e1  = v1 - trav.v0;
                trav.e2  = v2 - trav.v0;
                trav.data = meshIdx | BVHTriTrav::kObjectSpaceFlag;

                attrib.n = safeNormalize(cross(trav.e1, trav.e2));
                auto getN = [&](uint32_t idx) -> Vec3f {
                    return idx < mesh.normals.size()
                        ? safeNormalize(mesh.normals[idx]) : attrib.n;
                };
                attrib.sn0 = getN(i0); attrib.sn1 = getN(i1); attrib.sn2 = getN(i2);
            } else {
                trav.v0 = xfm.transformPoint(mesh.positions[i0]);
                Vec3f v1 = xfm.transformPoint(mesh.positions[i1]);
                Vec3f v2 = xfm.transformPoint(mesh.positions[i2]);
                trav.e1  = v1 - trav.v0;
                trav.e2  = v2 - trav.v0;
                trav.data = meshIdx;

                attrib.n = safeNormalize(cross(trav.e1, trav.e2));
                auto getN = [&](uint32_t idx) -> Vec3f {
                    return idx < mesh.normals.size()
                        ? safeNormalize(xfm.transformNormal(mesh.normals[idx])) : attrib.n;
                };
                attrib.sn0 = getN(i0); attrib.sn1 = getN(i1); attrib.sn2 = getN(i2);
            }

            auto getUV = [&](uint32_t idx) -> Vec2f {
                return idx < mesh.uvs.size() ? mesh.uvs[idx] : Vec2f{0.f, 0.f};
            };
            attrib.uv0 = getUV(i0); attrib.uv1 = getUV(i1); attrib.uv2 = getUV(i2);
            attrib.primID = ti;

            m_trav.push_back(trav);
            m_attribs.push_back(attrib);
        }
    }

    if (!m_trav.empty()) {
        uint32_t n = static_cast<uint32_t>(m_trav.size());
        std::vector<PrimInfo> primInfo(n);
        for (uint32_t i = 0; i < n; ++i) {
            const BVHTriTrav& trav = m_trav[i];
            BBox3f b;
            if (trav.isObjectSpace()) {
                const MeshDesc& mesh = m_pool.mesh(trav.meshID());
                for (const MotionKey& key : mesh.motionKeys) {
                    Vec3f w0 = key.objectToWorld.transformPoint(trav.v0);
                    Vec3f w1 = key.objectToWorld.transformPoint(trav.v0 + trav.e1);
                    Vec3f w2 = key.objectToWorld.transformPoint(trav.v0 + trav.e2);
                    b.expand(w0); b.expand(w1); b.expand(w2);
                }
            } else {
                Vec3f v1 = trav.v0 + trav.e1, v2 = trav.v0 + trav.e2;
                b.expand(trav.v0); b.expand(v1); b.expand(v2);
            }
            for (int ax = 0; ax < 3; ++ax)
                if (b.diagonal()[ax] < 1e-7f)
                    { b.pMin[ax] -= 1e-7f; b.pMax[ax] += 1e-7f; }
            primInfo[i] = { b, b.centroid(), i,
                            trav.v0, trav.v0 + trav.e1, trav.v0 + trav.e2,
                            trav.isObjectSpace() };
        }

        BBox3f rootBounds;
        for (uint32_t i = 0; i < n; ++i) rootBounds.expand(primInfo[i].bounds);
        Vec3f rd = rootBounds.diagonal();
        m_buildRootSA  = 2.f * (rd.x*rd.y + rd.y*rd.z + rd.z*rd.x);
        m_buildMaxRefs = static_cast<uint32_t>(n * kMaxRefFactor);
        BuildState state;
        state.totalRefs = n;

        m_primIndices.clear();
        m_primIndices.reserve(n);

        std::vector<BuildBVHNode> buildNodes;
        buildNodes.reserve(2 * n);
        buildRecursive(primInfo, 0, n, buildNodes, m_primIndices, state);
        repackBuildTree(buildNodes, m_nodes);
    }

    // Build BLASes and TLAS for instance groups
    if (m_pool.numInstanceGroups() > 0) {
        m_blasList.resize(m_pool.numInstanceGroups());
        for (uint32_t gi = 0; gi < (uint32_t)m_pool.numInstanceGroups(); ++gi)
            buildBLAS(gi);
        buildTLAS();
    }

    m_built = true;
}

// ---------------------------------------------------------------------------
// Build — recursive SAH into BuildBVHNode array
// ---------------------------------------------------------------------------
static void storeBuildBounds(BuildBVHNode& node, const BBox3f& b) {
    node.boundsMin[0] = b.pMin.x; node.boundsMin[1] = b.pMin.y; node.boundsMin[2] = b.pMin.z;
    node.boundsMax[0] = b.pMax.x; node.boundsMax[1] = b.pMax.y; node.boundsMax[2] = b.pMax.z;
}

uint32_t BVHBackend::buildRecursive(std::vector<PrimInfo>& primInfo,
                                    uint32_t start, uint32_t end,
                                    std::vector<BuildBVHNode>& buildNodes,
                                    std::vector<uint32_t>& primIndices,
                                    BuildState& state,
                                    int maxLeafPrims) {
    uint32_t nodeIdx = static_cast<uint32_t>(buildNodes.size());
    buildNodes.emplace_back();

    BBox3f bounds, centroidBounds;
    for (uint32_t i = start; i < end; ++i) {
        bounds.expand(primInfo[i].bounds);
        centroidBounds.expand(primInfo[i].centroid);
    }
    storeBuildBounds(buildNodes[nodeIdx], bounds);

    uint32_t count = end - start;
    float leafCost = kIntersectCost * (float)count;

    // Object split
    float    objCost   = std::numeric_limits<float>::infinity();
    int      objAxis   = -1, objBucket = -1;
    if (count > (uint32_t)maxLeafPrims)
        objBucket = sahSplit(primInfo, start, end, centroidBounds, bounds, objAxis, objCost);

    // Spatial split — only when budget allows
    float spatCost = std::numeric_limits<float>::infinity();
    int   spatAxis = -1; float spatPos = 0.f;
    if (count > (uint32_t)maxLeafPrims && state.totalRefs < m_buildMaxRefs)
        spatCost = spatialSplit(primInfo, start, end, bounds, spatAxis, spatPos);

    // Spatial split path — try first if it beats both object split and leaf
    if (spatAxis >= 0 && spatCost < objCost && spatCost < leafCost) {
        std::vector<PrimInfo> leftRefs, rightRefs;
        leftRefs.reserve(count); rightRefs.reserve(count);

        for (uint32_t i = start; i < end; ++i) {
            const PrimInfo& pi = primInfo[i];
            bool pureLeft  = pi.bounds.pMax[spatAxis] <= spatPos;
            bool pureRight = pi.bounds.pMin[spatAxis] >= spatPos;

            if (pureLeft) {
                leftRefs.push_back(pi);
            } else if (pureRight) {
                rightRefs.push_back(pi);
            } else {
                // Straddle: clip to each half for tighter bounds
                Vec3f verts[3] = { pi.v0, pi.v1, pi.v2 };

                BBox3f lb = clipTriangleSlab(verts, spatAxis, bounds.pMin[spatAxis], spatPos);
                if (lb.pMin.x <= lb.pMax.x) {
                    lb.pMax[spatAxis] = std::min(lb.pMax[spatAxis], spatPos);
                    PrimInfo lpi = pi; lpi.bounds = lb; lpi.centroid = lb.centroid();
                    leftRefs.push_back(lpi);
                } else {
                    leftRefs.push_back(pi);  // degenerate clip — keep original
                }

                BBox3f rb = clipTriangleSlab(verts, spatAxis, spatPos, bounds.pMax[spatAxis]);
                if (rb.pMin.x <= rb.pMax.x) {
                    rb.pMin[spatAxis] = std::max(rb.pMin[spatAxis], spatPos);
                    PrimInfo rpi = pi; rpi.bounds = rb; rpi.centroid = rb.centroid();
                    rightRefs.push_back(rpi);
                } else {
                    rightRefs.push_back(pi);
                }
            }
        }

        if (!leftRefs.empty() && !rightRefs.empty()) {
            uint32_t extra = (uint32_t)(leftRefs.size() + rightRefs.size()) - count;
            state.totalRefs += extra;

            buildRecursive(leftRefs, 0, (uint32_t)leftRefs.size(), buildNodes, primIndices, state, maxLeafPrims);
            uint32_t rightIdx = buildRecursive(rightRefs, 0, (uint32_t)rightRefs.size(), buildNodes, primIndices, state, maxLeafPrims);

            buildNodes[nodeIdx].dataA = rightIdx;
            buildNodes[nodeIdx].dataB = (uint32_t)spatAxis;
            return nodeIdx;
        }
        // else fall through to object split
    }

    // Object split path
    if (objAxis >= 0 && objCost < leafCost) {
        float range = centroidBounds.diagonal()[objAxis];
        uint32_t mid = start;
        if (range > 0.f) {
            auto* p = std::partition(
                primInfo.data() + start, primInfo.data() + end,
                [&](const PrimInfo& pi) {
                    int b = (int)(kSAHBuckets *
                        ((pi.centroid[objAxis] - centroidBounds.pMin[objAxis]) / range));
                    return std::clamp(b, 0, kSAHBuckets - 1) <= objBucket;
                });
            mid = (uint32_t)(p - primInfo.data());
        }
        if (mid == start || mid == end) mid = start + count / 2;

        buildRecursive(primInfo, start, mid, buildNodes, primIndices, state, maxLeafPrims);
        uint32_t rightIdx = buildRecursive(primInfo, mid, end, buildNodes, primIndices, state, maxLeafPrims);

        buildNodes[nodeIdx].dataA = rightIdx;
        buildNodes[nodeIdx].dataB = (uint32_t)objAxis;
        return nodeIdx;
    }

    // Leaf
    uint32_t offset = (uint32_t)primIndices.size();
    for (uint32_t i = start; i < end; ++i)
        primIndices.push_back(primInfo[i].originalIndex);
    buildNodes[nodeIdx].dataA = offset;
    buildNodes[nodeIdx].dataB = count | BuildBVHNode::kLeafFlag;
    return nodeIdx;
}

int BVHBackend::sahSplit(const std::vector<PrimInfo>& primInfo,
                         uint32_t start, uint32_t end,
                         const BBox3f& centroidBounds,
                         const BBox3f& nodeBounds,
                         int& outAxis,
                         float& outCost) const {
    struct Bucket { BBox3f bounds; uint32_t count = 0; };

    float bestCost = std::numeric_limits<float>::infinity();
    int   bestSplit = -1;
    outAxis = -1; outCost = std::numeric_limits<float>::infinity();

    Vec3f d = nodeBounds.diagonal();
    float parentArea = 2.f * (d.x*d.y + d.y*d.z + d.z*d.x);
    if (parentArea < 1e-12f) return -1;

    for (int axis = 0; axis < 3; ++axis) {
        float range = centroidBounds.diagonal()[axis];
        if (range < 1e-7f) continue;

        std::array<Bucket, kSAHBuckets> buckets{};
        for (uint32_t i = start; i < end; ++i) {
            int b = static_cast<int>(kSAHBuckets *
                ((primInfo[i].centroid[axis] - centroidBounds.pMin[axis]) / range));
            b = std::clamp(b, 0, kSAHBuckets - 1);
            buckets[b].count++;
            buckets[b].bounds.expand(primInfo[i].bounds);
        }

        std::array<BBox3f,   kSAHBuckets - 1> leftBounds{};
        std::array<uint32_t, kSAHBuckets - 1> leftCount{};
        BBox3f lb; uint32_t lc = 0;
        for (int i = 0; i < kSAHBuckets - 1; ++i) {
            lb.expand(buckets[i].bounds); lc += buckets[i].count;
            leftBounds[i] = lb; leftCount[i] = lc;
        }

        auto area = [](const BBox3f& b) {
            Vec3f d2 = b.diagonal();
            return 2.f * (d2.x*d2.y + d2.y*d2.z + d2.z*d2.x);
        };
        BBox3f rb; uint32_t rc = 0;
        for (int i = kSAHBuckets - 2; i >= 0; --i) {
            rb.expand(buckets[i + 1].bounds); rc += buckets[i + 1].count;
            float cost = kTraversalCost +
                kIntersectCost * (area(leftBounds[i]) * leftCount[i]
                                + area(rb)            * rc) / parentArea;
            if (cost < bestCost) { bestCost = cost; bestSplit = i; outAxis = axis; }
        }
    }

    outCost = bestCost;
    return bestSplit;   // -1 if no valid split (degenerate bounds on all axes)
}

// ---------------------------------------------------------------------------
// SBVH — spatial split support
// ---------------------------------------------------------------------------

// Clips triangle v[0..2] to the slab [sMin, sMax] on axis.
// Returns the bounding box of the surviving polygon (empty BBox3f if nothing survives).
BBox3f BVHBackend::clipTriangleSlab(const Vec3f v[3], int axis, float sMin, float sMax) {
    Vec3f poly[9]; int n = 3;
    poly[0] = v[0]; poly[1] = v[1]; poly[2] = v[2];

    // Clip to axis >= sMin
    {
        Vec3f out[9]; int m = 0;
        for (int i = 0; i < n; ++i) {
            const Vec3f& a = poly[i];
            const Vec3f& b = poly[(i + 1) % n];
            bool aIn = (a[axis] >= sMin), bIn = (b[axis] >= sMin);
            if (aIn) out[m++] = a;
            if (aIn != bIn) {
                float t = (sMin - a[axis]) / (b[axis] - a[axis]);
                out[m++] = a + (b - a) * t;
            }
        }
        n = m; for (int i = 0; i < n; ++i) poly[i] = out[i];
    }

    // Clip to axis <= sMax
    {
        Vec3f out[9]; int m = 0;
        for (int i = 0; i < n; ++i) {
            const Vec3f& a = poly[i];
            const Vec3f& b = poly[(i + 1) % n];
            bool aIn = (a[axis] <= sMax), bIn = (b[axis] <= sMax);
            if (aIn) out[m++] = a;
            if (aIn != bIn) {
                float t = (sMax - a[axis]) / (b[axis] - a[axis]);
                out[m++] = a + (b - a) * t;
            }
        }
        n = m; for (int i = 0; i < n; ++i) poly[i] = out[i];
    }

    BBox3f result;
    for (int i = 0; i < n; ++i) result.expand(poly[i]);
    return result;
}

float BVHBackend::spatialSplit(const std::vector<PrimInfo>& primInfo,
                                uint32_t start, uint32_t end,
                                const BBox3f& nodeBounds,
                                int& outAxis, float& outPos) const {
    struct SpatBucket { BBox3f bounds; uint32_t in = 0, out = 0; };

    auto area = [](const BBox3f& b) -> float {
        Vec3f d = b.diagonal();
        return 2.f * (d.x*d.y + d.y*d.z + d.z*d.x);
    };
    auto isEmpty = [](const BBox3f& b) -> bool {
        return b.pMin.x > b.pMax.x;
    };

    float parentArea = area(nodeBounds);
    outAxis = -1; outPos = 0.f;

    if (parentArea < 1e-12f) return std::numeric_limits<float>::infinity();
    // Guard: skip nodes too small relative to scene root
    if (m_buildRootSA > 0.f && parentArea / m_buildRootSA < kSpatialAlpha)
        return std::numeric_limits<float>::infinity();

    // Skip if any ref is animated (motion-expanded bounds can't be usefully clipped)
    for (uint32_t i = start; i < end; ++i)
        if (primInfo[i].animated) return std::numeric_limits<float>::infinity();

    float bestCost = std::numeric_limits<float>::infinity();
    Vec3f diag = nodeBounds.diagonal();

    for (int axis = 0; axis < 3; ++axis) {
        if (diag[axis] < 1e-7f) continue;
        float binSize   = diag[axis] / kSpatialBuckets;
        float invBin    = 1.f / binSize;
        float nodeMin   = nodeBounds.pMin[axis];

        std::array<SpatBucket, kSpatialBuckets> buckets{};

        for (uint32_t i = start; i < end; ++i) {
            const PrimInfo& pi = primInfo[i];
            int bMin = std::clamp((int)((pi.bounds.pMin[axis] - nodeMin) * invBin), 0, kSpatialBuckets - 1);
            int bMax = std::clamp((int)((pi.bounds.pMax[axis] - nodeMin) * invBin), 0, kSpatialBuckets - 1);
            buckets[bMin].in++;
            buckets[bMax].out++;

            Vec3f verts[3] = { pi.v0, pi.v1, pi.v2 };
            for (int b = bMin; b <= bMax; ++b) {
                float sMin = nodeMin + b * binSize;
                float sMax = nodeMin + (b + 1) * binSize;
                BBox3f clipped = clipTriangleSlab(verts, axis, sMin, sMax);
                if (!isEmpty(clipped)) buckets[b].bounds.expand(clipped);
            }
        }

        // Prefix scan: left bounds and counts
        std::array<BBox3f,   kSpatialBuckets - 1> leftBounds{};
        std::array<uint32_t, kSpatialBuckets - 1> leftCount{};
        {
            BBox3f lb; uint32_t lc = 0;
            for (int i = 0; i < kSpatialBuckets - 1; ++i) {
                if (!isEmpty(buckets[i].bounds)) lb.expand(buckets[i].bounds);
                lc += buckets[i].in;
                leftBounds[i] = lb; leftCount[i] = lc;
            }
        }

        // Suffix scan: right bounds and counts
        std::array<BBox3f,   kSpatialBuckets - 1> rightBounds{};
        std::array<uint32_t, kSpatialBuckets - 1> rightCount{};
        {
            BBox3f rb; uint32_t rc = 0;
            for (int i = kSpatialBuckets - 2; i >= 0; --i) {
                if (!isEmpty(buckets[i + 1].bounds)) rb.expand(buckets[i + 1].bounds);
                rc += buckets[i + 1].out;
                rightBounds[i] = rb; rightCount[i] = rc;
            }
        }

        for (int i = 0; i < kSpatialBuckets - 1; ++i) {
            uint32_t lc = leftCount[i], rc = rightCount[i];
            if (lc == 0 || rc == 0) continue;
            if (isEmpty(leftBounds[i]) || isEmpty(rightBounds[i])) continue;

            float cost = kTraversalCost +
                kIntersectCost * (area(leftBounds[i]) * lc + area(rightBounds[i]) * rc) / parentArea;
            if (cost < bestCost) {
                bestCost = cost;
                outAxis  = axis;
                outPos   = nodeMin + (i + 1) * binSize;
            }
        }
    }

    return bestCost;
}

// ---------------------------------------------------------------------------
// BVH4 repack — collapse binary SAH tree into 4-wide SOA BVHNode array
// ---------------------------------------------------------------------------

// Fill an empty (never-hit) slot at index s in m_nodes[nodeIdx].
static void fillEmptySlot(BVHNode& node, int s) {
    constexpr float kInf = std::numeric_limits<float>::infinity();
    node.minX[s] = node.minY[s] = node.minZ[s] =  kInf;
    node.maxX[s] = node.maxY[s] = node.maxZ[s] = -kInf;
    node.childData[s] = 0;
    node.childMeta[s] = BVHNode::kLeafFlag;  // leaf with 0 prims — never matches
}

uint32_t BVHBackend::buildBVH4Node(const std::vector<BuildBVHNode>& build,
                                    uint32_t oldIdx,
                                    std::vector<BVHNode>& nodes) {
    // Gather up to 4 child slots by expanding the two binary children one level.
    // Each binary child is expanded into its own children if it is interior and
    // we still have room (total slots < 4).  This collapses two binary levels
    // into one BVH4 level without changing the SAH quality.
    struct Slot { uint32_t idx; };
    Slot slots[4];
    int  nSlots = 0;

    auto addSlot = [&](uint32_t idx) {
        const BuildBVHNode& n = build[idx];
        if (!n.isLeaf() && nSlots <= 2) {
            // Expand: use n's two children instead of n itself
            slots[nSlots++] = { (uint32_t)(idx + 1) };           // left child of n
            slots[nSlots++] = { n.rightChild() };                 // right child of n
        } else {
            slots[nSlots++] = { idx };
        }
    };

    const BuildBVHNode& root = build[oldIdx];
    addSlot(oldIdx + 1);          // binary left child
    addSlot(root.rightChild());   // binary right child
    // nSlots is now 2, 3, or 4

    // Allocate BVH4 node — do this AFTER gathering slots so no index is stale
    uint32_t newIdx = static_cast<uint32_t>(nodes.size());
    nodes.push_back(BVHNode{});

    // Fill each slot; recurse for interior children
    for (int s = 0; s < nSlots; ++s) {
        uint32_t si = slots[s].idx;
        const BuildBVHNode& sn = build[si];

        nodes[newIdx].minX[s] = sn.boundsMin[0];
        nodes[newIdx].minY[s] = sn.boundsMin[1];
        nodes[newIdx].minZ[s] = sn.boundsMin[2];
        nodes[newIdx].maxX[s] = sn.boundsMax[0];
        nodes[newIdx].maxY[s] = sn.boundsMax[1];
        nodes[newIdx].maxZ[s] = sn.boundsMax[2];

        if (sn.isLeaf()) {
            nodes[newIdx].childData[s] = sn.primOffset();
            nodes[newIdx].childMeta[s] = sn.primCount() | BVHNode::kLeafFlag;
        } else {
            uint32_t childNodeIdx = buildBVH4Node(build, si, nodes);
            // Re-index through newIdx — nodes may have reallocated during recursion
            nodes[newIdx].childData[s] = childNodeIdx;
            nodes[newIdx].childMeta[s] = 0;
        }
    }

    // Pad unused slots with empty AABBs
    for (int s = nSlots; s < 4; ++s)
        fillEmptySlot(nodes[newIdx], s);

    return newIdx;
}

void BVHBackend::repackBuildTree(const std::vector<BuildBVHNode>& build,
                                  std::vector<BVHNode>& nodes) {
    if (build.empty()) return;

    const BuildBVHNode& root = build[0];

    if (root.isLeaf()) {
        nodes.push_back(BVHNode{});
        nodes[0].minX[0] = root.boundsMin[0];
        nodes[0].minY[0] = root.boundsMin[1];
        nodes[0].minZ[0] = root.boundsMin[2];
        nodes[0].maxX[0] = root.boundsMax[0];
        nodes[0].maxY[0] = root.boundsMax[1];
        nodes[0].maxZ[0] = root.boundsMax[2];
        nodes[0].childData[0] = root.primOffset();
        nodes[0].childMeta[0] = root.primCount() | BVHNode::kLeafFlag;
        for (int s = 1; s < 4; ++s) fillEmptySlot(nodes[0], s);
    } else {
        buildBVH4Node(build, 0, nodes);
    }
}

// ---------------------------------------------------------------------------
// buildBLAS — build a BLAS for the prototype mesh of one instance group.
// ---------------------------------------------------------------------------
void BVHBackend::buildBLAS(uint32_t groupID) {
    BLAS& blas = m_blasList[groupID];
    const InstanceGroupDesc& grp   = m_pool.instanceGroup(groupID);
    const MeshDesc&          proto = m_pool.mesh(grp.protoMeshID);

    uint32_t numTris = proto.numTriangles();
    if (numTris == 0) return;

    blas.trav.reserve(numTris);
    blas.attribs.reserve(numTris);

    for (uint32_t ti = 0; ti < numTris; ++ti) {
        uint32_t i0 = proto.indices[ti * 3 + 0];
        uint32_t i1 = proto.indices[ti * 3 + 1];
        uint32_t i2 = proto.indices[ti * 3 + 2];

        BVHTriTrav   trav;
        BVHTriAttrib attrib;

        // Positions stored in object space (prototype mesh is never baked to world).
        trav.v0  = proto.positions[i0];
        Vec3f v1 = proto.positions[i1];
        Vec3f v2 = proto.positions[i2];
        trav.e1  = v1 - trav.v0;
        trav.e2  = v2 - trav.v0;
        trav.data = grp.protoMeshID;  // meshID used for material lookup

        attrib.n = safeNormalize(cross(trav.e1, trav.e2));
        auto getN = [&](uint32_t idx) -> Vec3f {
            return idx < proto.normals.size()
                ? safeNormalize(proto.normals[idx]) : attrib.n;
        };
        attrib.sn0 = getN(i0); attrib.sn1 = getN(i1); attrib.sn2 = getN(i2);

        auto getUV = [&](uint32_t idx) -> Vec2f {
            return idx < proto.uvs.size() ? proto.uvs[idx] : Vec2f{0.f, 0.f};
        };
        attrib.uv0 = getUV(i0); attrib.uv1 = getUV(i1); attrib.uv2 = getUV(i2);
        attrib.primID = ti;

        blas.trav.push_back(trav);
        blas.attribs.push_back(attrib);
    }

    // Build PrimInfo — object-space bounds (no transforms applied).
    std::vector<PrimInfo> primInfo(numTris);
    for (uint32_t i = 0; i < numTris; ++i) {
        const BVHTriTrav& t = blas.trav[i];
        Vec3f v1 = t.v0 + t.e1, v2 = t.v0 + t.e2;
        BBox3f b;
        b.expand(t.v0); b.expand(v1); b.expand(v2);
        for (int ax = 0; ax < 3; ++ax)
            if (b.diagonal()[ax] < 1e-7f)
                { b.pMin[ax] -= 1e-7f; b.pMax[ax] += 1e-7f; }
        primInfo[i] = { b, b.centroid(), i, t.v0, v1, v2, false };
    }

    BBox3f rootBounds;
    for (uint32_t i = 0; i < numTris; ++i) rootBounds.expand(primInfo[i].bounds);
    Vec3f rd = rootBounds.diagonal();
    m_buildRootSA  = 2.f * (rd.x*rd.y + rd.y*rd.z + rd.z*rd.x);
    m_buildMaxRefs = static_cast<uint32_t>(numTris * kMaxRefFactor);
    BuildState state;
    state.totalRefs = numTris;

    blas.primIndices.reserve(numTris);
    std::vector<BuildBVHNode> buildNodes;
    buildNodes.reserve(2 * numTris);
    buildRecursive(primInfo, 0, numTris, buildNodes, blas.primIndices, state);
    repackBuildTree(buildNodes, blas.nodes);
    spdlog::info("BVHBackend: BLAS[{}] built — {} tris, {} BVH4 nodes, {} instances",
                 groupID, numTris, blas.nodes.size(),
                 m_pool.instanceGroup(groupID).instances.size());
}

// ---------------------------------------------------------------------------
// buildTLAS — build a BVH4 over world-space instance AABBs.
// ---------------------------------------------------------------------------
void BVHBackend::buildTLAS() {
    uint32_t totalInstances = 0;
    for (uint32_t gi = 0; gi < (uint32_t)m_pool.numInstanceGroups(); ++gi)
        totalInstances += (uint32_t)m_pool.instanceGroup(gi).instances.size();
    if (totalInstances == 0) return;

    m_tlasInstances.reserve(totalInstances);
    m_tlasPrimIndices.reserve(totalInstances);

    std::vector<PrimInfo> primInfo;
    primInfo.reserve(totalInstances);

    for (uint32_t gi = 0; gi < (uint32_t)m_pool.numInstanceGroups(); ++gi) {
        const InstanceGroupDesc& grp   = m_pool.instanceGroup(gi);
        const MeshDesc&          proto = m_pool.mesh(grp.protoMeshID);

        // Object-space prototype bounds
        BBox3f protoBounds;
        for (const Vec3f& p : proto.positions) protoBounds.expand(p);

        for (uint32_t ii = 0; ii < (uint32_t)grp.instances.size(); ++ii) {
            const InstanceDesc& inst = grp.instances[ii];

            // World-space AABB: transform all 8 corners at each motion key
            BBox3f wb;
            for (const MotionKey& key : inst.keys) {
                for (int cx = 0; cx < 2; ++cx)
                for (int cy = 0; cy < 2; ++cy)
                for (int cz = 0; cz < 2; ++cz) {
                    Vec3f c = {
                        cx ? protoBounds.pMax.x : protoBounds.pMin.x,
                        cy ? protoBounds.pMax.y : protoBounds.pMin.y,
                        cz ? protoBounds.pMax.z : protoBounds.pMin.z
                    };
                    wb.expand(key.objectToWorld.transformPoint(c));
                }
            }
            for (int ax = 0; ax < 3; ++ax)
                if (wb.diagonal()[ax] < 1e-7f)
                    { wb.pMin[ax] -= 1e-7f; wb.pMax[ax] += 1e-7f; }

            uint32_t instIdx = (uint32_t)m_tlasInstances.size();
            TLASInstance ti;
            ti.groupID     = gi;
            ti.instanceIdx = ii;
            ti.bMin[0] = wb.pMin.x; ti.bMin[1] = wb.pMin.y; ti.bMin[2] = wb.pMin.z;
            ti.bMax[0] = wb.pMax.x; ti.bMax[1] = wb.pMax.y; ti.bMax[2] = wb.pMax.z;
            m_tlasInstances.push_back(ti);

            // animated=true disables SBVH spatial splits for TLAS prims
            primInfo.push_back({wb, wb.centroid(), instIdx, {}, {}, {}, true});
        }
    }

    BBox3f rootBounds;
    for (const PrimInfo& pi : primInfo) rootBounds.expand(pi.bounds);
    Vec3f rd = rootBounds.diagonal();
    m_buildRootSA  = 2.f * (rd.x*rd.y + rd.y*rd.z + rd.z*rd.x);
    m_buildMaxRefs = totalInstances;  // no spatial splits for TLAS
    BuildState state;
    state.totalRefs = totalInstances;

    std::vector<BuildBVHNode> buildNodes;
    buildNodes.reserve(2 * totalInstances);
    buildRecursive(primInfo, 0, totalInstances, buildNodes, m_tlasPrimIndices, state, kTLASMaxLeafPrims);
    repackBuildTree(buildNodes, m_tlasNodes);

    spdlog::info("BVHBackend: TLAS built — {} instance groups, {} instances total",
                 m_pool.numInstanceGroups(), totalInstances);
}

// ---------------------------------------------------------------------------
// traceBLAS — traverse one instance's BLAS; updates closestT on hit.
// ---------------------------------------------------------------------------
void BVHBackend::traceBLAS(const BLAS& blas, const Ray& ray, const Ray4& objRay,
                            uint32_t groupID, uint32_t instanceIdx, const Mat4f& o2w,
                            float& closestT, uint32_t& hitBlasTriIdx,
                            float& hitU, float& hitV,
                            uint32_t& hitGroupID, uint32_t& hitInstanceIdx) const {
    if (blas.nodes.empty()) return;

    uint32_t stack[64];
    int      top = 0;
    stack[top++] = 0;

    Ray4 r = objRay;
    r.tMax = closestT;

    while (top > 0) {
        const BVHNode& node = blas.nodes[stack[--top]];
        float tNear[4];
        int mask = intersectAABB4(node, r, tNear, r.tMax);
        if (!mask) continue;

        int hits[4], nHits = 0;
        for (int c = 0; c < 4; ++c)
            if (mask & (1 << c)) hits[nHits++] = c;
        for (int i = 1; i < nHits; ++i) {
            int key = hits[i], j = i - 1;
            while (j >= 0 && tNear[hits[j]] > tNear[key]) { hits[j+1] = hits[j]; --j; }
            hits[j+1] = key;
        }

        for (int i = nHits - 1; i >= 0; --i)
            if (!node.isLeaf(hits[i])) stack[top++] = node.childIdx(hits[i]);

        for (int i = 0; i < nHits; ++i) {
            int c = hits[i];
            if (!node.isLeaf(c)) continue;
            uint32_t offset = node.primOffset(c);
            uint32_t count  = node.primCount(c);
            for (uint32_t k = 0; k < count; ++k) {
                uint32_t idx = blas.primIndices[offset + k];
                float t, u, v;
                Ray4 rc = r; rc.tMax = r.tMax;
                if (intersectTriangle(blas.trav[idx], rc, t, u, v)) {
                    r.tMax = t;
                    hitBlasTriIdx  = idx;
                    hitU = u; hitV = v;
                    hitGroupID     = groupID;
                    hitInstanceIdx = instanceIdx;
                }
            }
        }
    }
    closestT = r.tMax;
}

// ---------------------------------------------------------------------------
// occludedBLAS — early-exit shadow test through one BLAS.
// ---------------------------------------------------------------------------
bool BVHBackend::occludedBLAS(const BLAS& blas, const Ray4& objRay) const {
    if (blas.nodes.empty()) return false;

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;
    float tNear[4];

    while (top > 0) {
        const BVHNode& node = blas.nodes[stack[--top]];
        int mask = intersectAABB4(node, objRay, tNear, objRay.tMax);
        if (!mask) continue;
        for (int c = 0; c < 4; ++c) {
            if (!(mask & (1 << c))) continue;
            if (node.isLeaf(c)) {
                uint32_t offset = node.primOffset(c);
                uint32_t count  = node.primCount(c);
                for (uint32_t k = 0; k < count; ++k) {
                    float t, u, v;
                    if (intersectTriangle(blas.trav[blas.primIndices[offset + k]], objRay, t, u, v))
                        return true;
                }
            } else {
                stack[top++] = node.childIdx(c);
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------
// Traversal helpers
// ---------------------------------------------------------------------------
BVHBackend::Ray4 BVHBackend::makeRay4(const Ray& ray) {
    Ray4 r;
    r.origin = ray.origin;
    r.tMin   = ray.tMin;
    r.tMax   = ray.tMax;
    auto safe = [](float v) { return std::abs(v) > 1e-9f ? v : 1e-9f; };
    r.invDir = { 1.f / safe(ray.direction.x),
                 1.f / safe(ray.direction.y),
                 1.f / safe(ray.direction.z) };
    return r;
}

BVHBackend::Ray4 BVHBackend::makeObjectSpaceRay4(const Ray& ray, const Mat4f& worldToObject) {
    Vec3f o = worldToObject.transformPoint(ray.origin);
    Vec3f d = worldToObject.transformVector(ray.direction);
    Ray objRay{o, d, ray.tMin, ray.tMax};
    return makeRay4(objRay);
}

// ---------------------------------------------------------------------------
// intersectAABB4 — simultaneous SIMD test of all four children.
//
// Returns hit mask: bit c = child c hit.
// tNear[c] receives the entry distance for each hit child.
// closestT is used as tMax to prune children behind a known hit.
// ---------------------------------------------------------------------------

#if defined(__aarch64__)

int BVHBackend::intersectAABB4(const BVHNode& node, const Ray4& r,
                                float tNear[4], float closestT) {
    // NEON 4-wide: one lane per child, all 4 tested simultaneously.
    float32x4_t tn = vdupq_n_f32(r.tMin);
    float32x4_t tf = vdupq_n_f32(closestT);

    float32x4_t ox = vdupq_n_f32(r.origin.x);
    float32x4_t oy = vdupq_n_f32(r.origin.y);
    float32x4_t oz = vdupq_n_f32(r.origin.z);
    float32x4_t dx = vdupq_n_f32(r.invDir.x);
    float32x4_t dy = vdupq_n_f32(r.invDir.y);
    float32x4_t dz = vdupq_n_f32(r.invDir.z);

    float32x4_t t0x = vmulq_f32(vsubq_f32(vld1q_f32(node.minX), ox), dx);
    float32x4_t t1x = vmulq_f32(vsubq_f32(vld1q_f32(node.maxX), ox), dx);
    tn = vmaxq_f32(tn, vminq_f32(t0x, t1x));
    tf = vminq_f32(tf, vmaxq_f32(t0x, t1x));

    float32x4_t t0y = vmulq_f32(vsubq_f32(vld1q_f32(node.minY), oy), dy);
    float32x4_t t1y = vmulq_f32(vsubq_f32(vld1q_f32(node.maxY), oy), dy);
    tn = vmaxq_f32(tn, vminq_f32(t0y, t1y));
    tf = vminq_f32(tf, vmaxq_f32(t0y, t1y));

    float32x4_t t0z = vmulq_f32(vsubq_f32(vld1q_f32(node.minZ), oz), dz);
    float32x4_t t1z = vmulq_f32(vsubq_f32(vld1q_f32(node.maxZ), oz), dz);
    tn = vmaxq_f32(tn, vminq_f32(t0z, t1z));
    tf = vminq_f32(tf, vmaxq_f32(t0z, t1z));

    uint32x4_t hit = vcleq_f32(tn, tf);
    vst1q_f32(tNear, tn);

    return (vgetq_lane_u32(hit, 0) ? 1 : 0)
         | (vgetq_lane_u32(hit, 1) ? 2 : 0)
         | (vgetq_lane_u32(hit, 2) ? 4 : 0)
         | (vgetq_lane_u32(hit, 3) ? 8 : 0);
}

#elif defined(__SSE2__)

int BVHBackend::intersectAABB4(const BVHNode& node, const Ray4& r,
                                float tNear[4], float closestT) {
    // SSE: __m128 is a natural 4-wide float — one lane per child.
    __m128 tn = _mm_set1_ps(r.tMin);
    __m128 tf = _mm_set1_ps(closestT);
    __m128 ox = _mm_set1_ps(r.origin.x);
    __m128 oy = _mm_set1_ps(r.origin.y);
    __m128 oz = _mm_set1_ps(r.origin.z);
    __m128 dx = _mm_set1_ps(r.invDir.x);
    __m128 dy = _mm_set1_ps(r.invDir.y);
    __m128 dz = _mm_set1_ps(r.invDir.z);

    __m128 t0x = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.minX), ox), dx);
    __m128 t1x = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.maxX), ox), dx);
    tn = _mm_max_ps(tn, _mm_min_ps(t0x, t1x));
    tf = _mm_min_ps(tf, _mm_max_ps(t0x, t1x));

    __m128 t0y = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.minY), oy), dy);
    __m128 t1y = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.maxY), oy), dy);
    tn = _mm_max_ps(tn, _mm_min_ps(t0y, t1y));
    tf = _mm_min_ps(tf, _mm_max_ps(t0y, t1y));

    __m128 t0z = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.minZ), oz), dz);
    __m128 t1z = _mm_mul_ps(_mm_sub_ps(_mm_loadu_ps(node.maxZ), oz), dz);
    tn = _mm_max_ps(tn, _mm_min_ps(t0z, t1z));
    tf = _mm_min_ps(tf, _mm_max_ps(t0z, t1z));

    _mm_storeu_ps(tNear, tn);
    return _mm_movemask_ps(_mm_cmple_ps(tn, tf));  // 4-bit mask, one per child
}

#else

int BVHBackend::intersectAABB4(const BVHNode& node, const Ray4& r,
                                float tNear[4], float closestT) {
    int mask = 0;
    for (int c = 0; c < 4; ++c) {
        float tn = r.tMin, tf = closestT;
        float t0, t1;
        t0 = (node.minX[c] - r.origin.x) * r.invDir.x;
        t1 = (node.maxX[c] - r.origin.x) * r.invDir.x;
        tn = std::max(tn, std::min(t0, t1)); tf = std::min(tf, std::max(t0, t1));
        t0 = (node.minY[c] - r.origin.y) * r.invDir.y;
        t1 = (node.maxY[c] - r.origin.y) * r.invDir.y;
        tn = std::max(tn, std::min(t0, t1)); tf = std::min(tf, std::max(t0, t1));
        t0 = (node.minZ[c] - r.origin.z) * r.invDir.z;
        t1 = (node.maxZ[c] - r.origin.z) * r.invDir.z;
        tn = std::max(tn, std::min(t0, t1)); tf = std::min(tf, std::max(t0, t1));
        if (tn <= tf) { tNear[c] = tn; mask |= (1 << c); }
    }
    return mask;
}

#endif

// ---------------------------------------------------------------------------
// intersectTriangle — Möller–Trumbore
// ---------------------------------------------------------------------------
bool BVHBackend::intersectTriangle(const BVHTriTrav& trav, const Ray4& r,
                                    float& t, float& u, float& v) {
    Vec3f dir = { 1.f / r.invDir.x, 1.f / r.invDir.y, 1.f / r.invDir.z };
    Vec3f h = cross(dir, trav.e2);
    float a = dot(trav.e1, h);
    if (std::abs(a) < 1e-8f) return false;
    float f = 1.f / a;
    Vec3f s = r.origin - trav.v0;
    u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;
    Vec3f q = cross(s, trav.e1);
    v = f * dot(dir, q);
    if (v < 0.f || u + v > 1.f) return false;
    t = f * dot(trav.e2, q);
    return t > r.tMin && t < r.tMax;
}

// ---------------------------------------------------------------------------
// fillSurfaceInteraction
// ---------------------------------------------------------------------------
void BVHBackend::fillSurfaceInteraction(const BVHTriTrav& trav,
                                         const BVHTriAttrib& attrib,
                                         float t, float u, float v,
                                         const Mat4f* worldXfm,
                                         SurfaceInteraction& si) const {
    float w = 1.f - u - v;
    si.t  = t;
    si.uv = attrib.uv0 * w + attrib.uv1 * u + attrib.uv2 * v;

    if (worldXfm) {
        si.p             = worldXfm->transformPoint(trav.v0 + trav.e1 * u + trav.e2 * v);
        si.objectToWorld = *worldXfm;
        si.worldToObject = worldXfm->inverse();
        // Normals transform by the inverse-transpose of objectToWorld.
        // worldToObject^T == (objectToWorld^{-1})^T — correct for any affine xfm.
        si.ng = safeNormalize(si.worldToObject.transformNormal(attrib.n));
        si.n  = safeNormalize(si.worldToObject.transformNormal(
                    attrib.sn0 * w + attrib.sn1 * u + attrib.sn2 * v));
    } else {
        si.p  = trav.v0 + trav.e1 * u + trav.e2 * v;
        si.ng = attrib.n;
        si.n  = safeNormalize(attrib.sn0 * w + attrib.sn1 * u + attrib.sn2 * v);
        const MeshDesc& mesh = m_pool.mesh(trav.meshID());
        si.objectToWorld = mesh.staticObjectToWorld;
        si.worldToObject = mesh.staticWorldToObject;
    }

    // Align geometric normal with the shading normal hemisphere.  USD scenes
    // can author shading normals that disagree with triangle-winding-derived
    // geometric normals (e.g. the Cornell box ceiling, where vertex order
    // makes ng point up while authored normals point down into the room).
    // The path tracer treats the shading normal as the intended outward
    // direction, so flip ng if they're opposed.  Without this, frontFace
    // tests in ShadingContext misclassify these surfaces as back-facing.
    if (dot(si.ng, si.n) < 0.f) si.ng = -si.ng;

    si.meshID = trav.meshID();
    si.primID = attrib.primID;
}

// ---------------------------------------------------------------------------
// trace() / traceImpl() — nearest-hit traversal
// ---------------------------------------------------------------------------
TraceResult BVHBackend::trace(const Ray& ray) const {
    assert(m_built);
    if (m_nodes.empty()) return {};
    return traceImpl(ray);
}

TraceResult BVHBackend::traceImpl(const Ray& ray) const {
    Ray4 r = makeRay4(ray);

    uint32_t stack[64];
    int      top = 0;
    stack[top++] = 0;

    float    closestT = r.tMax;
    uint32_t hitIdx   = ~0u;
    float    hitU = 0.f, hitV = 0.f;

    while (top > 0) {
        const BVHNode& node = m_nodes[stack[--top]];

        float tNear[4];
        int mask = intersectAABB4(node, r, tNear, closestT);
        if (!mask) continue;

        // Collect hit children and sort near-to-far (insertion sort, ≤4 elements).
        int hits[4], nHits = 0;
        for (int c = 0; c < 4; ++c)
            if (mask & (1 << c)) hits[nHits++] = c;
        for (int i = 1; i < nHits; ++i) {
            int key = hits[i]; int j = i - 1;
            while (j >= 0 && tNear[hits[j]] > tNear[key]) { hits[j+1] = hits[j]; --j; }
            hits[j+1] = key;
        }

        // Push interior children far-to-near so near is at stack top.
        // Process leaf children inline near-to-far to update closestT early.
        for (int i = nHits - 1; i >= 0; --i)
            if (!node.isLeaf(hits[i])) stack[top++] = node.childIdx(hits[i]);

        for (int i = 0; i < nHits; ++i) {
            int c = hits[i];
            if (!node.isLeaf(c)) continue;
            uint32_t offset = node.primOffset(c);
            uint32_t count  = node.primCount(c);
            for (uint32_t k = 0; k < count; ++k) {
                uint32_t idx = m_primIndices[offset + k];
                const BVHTriTrav& trav = m_trav[idx];
                float t, u, v;
                if (trav.isObjectSpace()) {
                    const MeshDesc& mesh = m_pool.mesh(trav.meshID());
                    Mat4f o2w = mesh.interpolateO2W(ray.time);
                    Ray4 ro = makeObjectSpaceRay4(ray, o2w.inverse());
                    ro.tMax = closestT;
                    if (intersectTriangle(trav, ro, t, u, v)) {
                        closestT = t; hitIdx = idx; hitU = u; hitV = v;
                    }
                } else {
                    Ray4 rc = r; rc.tMax = closestT;
                    if (intersectTriangle(trav, rc, t, u, v)) {
                        closestT = t; hitIdx = idx; hitU = u; hitV = v;
                    }
                }
            }
        }
    }

    // --- Query TLAS (instance groups), using single-BVH closestT as tMax ---
    // Single-BVH ran first so TLAS only finds hits strictly closer.
    // If TLAS wins, hitBlasTriIdx is set; hitIdx retains the single-BVH result.
    uint32_t hitBlasTriIdx  = ~0u;
    uint32_t hitGroupID     = ~0u;
    uint32_t hitInstanceIdx = ~0u;
    float    hitBU = 0.f, hitBV = 0.f;

    if (!m_tlasNodes.empty()) {
        uint32_t tlasStack[64];
        int      tlasTop = 0;
        tlasStack[tlasTop++] = 0;

        while (tlasTop > 0) {
            const BVHNode& node = m_tlasNodes[tlasStack[--tlasTop]];
            float tNear[4];
            int mask = intersectAABB4(node, r, tNear, closestT);
            if (!mask) continue;

            int hits[4], nHits = 0;
            for (int c = 0; c < 4; ++c)
                if (mask & (1 << c)) hits[nHits++] = c;
            for (int i = 1; i < nHits; ++i) {
                int key = hits[i], j = i - 1;
                while (j >= 0 && tNear[hits[j]] > tNear[key]) { hits[j+1] = hits[j]; --j; }
                hits[j+1] = key;
            }

            for (int i = nHits - 1; i >= 0; --i)
                if (!node.isLeaf(hits[i])) tlasStack[tlasTop++] = node.childIdx(hits[i]);

            for (int i = 0; i < nHits; ++i) {
                int c = hits[i];
                if (!node.isLeaf(c)) continue;
                uint32_t offset = node.primOffset(c);
                uint32_t count  = node.primCount(c);
                for (uint32_t k = 0; k < count; ++k) {
                    const TLASInstance& ti = m_tlasInstances[m_tlasPrimIndices[offset + k]];
                    // Tight per-instance AABB test — rejects false positives from the
                    // leaf's union AABB without paying full BLAS traversal cost.
                    {
                        float tn = r.tMin, tf = closestT;
                        for (int ax = 0; ax < 3; ++ax) {
                            float t0 = (ti.bMin[ax] - r.origin[ax]) * r.invDir[ax];
                            float t1 = (ti.bMax[ax] - r.origin[ax]) * r.invDir[ax];
                            tn = std::max(tn, std::min(t0, t1));
                            tf = std::min(tf, std::max(t0, t1));
                        }
                        if (tn > tf) continue;
                    }
                    const InstanceGroupDesc& grp  = m_pool.instanceGroup(ti.groupID);
                    const InstanceDesc&      inst = grp.instances[ti.instanceIdx];
                    Mat4f o2w = inst.interpolateO2W(ray.time);
                    Mat4f w2o = inst.interpolateW2O(ray.time);
                    Ray4  objRay = makeObjectSpaceRay4(ray, w2o);
                    traceBLAS(m_blasList[ti.groupID], ray, objRay,
                               ti.groupID, ti.instanceIdx, o2w,
                               closestT, hitBlasTriIdx, hitBU, hitBV,
                               hitGroupID, hitInstanceIdx);
                }
            }
        }
    }

    // TLAS ran second: if hitBlasTriIdx is set it is strictly closer than
    // any single-BVH hit (TLAS used single-BVH's closestT as its tMax).
    TraceResult result;
    if (hitBlasTriIdx != ~0u) {
        result.hit = true;
        const InstanceGroupDesc& grp   = m_pool.instanceGroup(hitGroupID);
        const InstanceDesc&      inst  = grp.instances[hitInstanceIdx];
        Mat4f o2w = inst.interpolateO2W(ray.time);
        fillSurfaceInteraction(m_blasList[hitGroupID].trav[hitBlasTriIdx],
                               m_blasList[hitGroupID].attribs[hitBlasTriIdx],
                               closestT, hitBU, hitBV, &o2w, result.si);
    } else if (hitIdx != ~0u) {
        result.hit = true;
        const BVHTriTrav&   trav   = m_trav[hitIdx];
        const BVHTriAttrib& attrib = m_attribs[hitIdx];
        if (trav.isObjectSpace()) {
            const MeshDesc& mesh = m_pool.mesh(trav.meshID());
            Mat4f o2w = mesh.interpolateO2W(ray.time);
            fillSurfaceInteraction(trav, attrib, closestT, hitU, hitV, &o2w, result.si);
        } else {
            fillSurfaceInteraction(trav, attrib, closestT, hitU, hitV, nullptr, result.si);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// occluded() — early-exit shadow ray
// ---------------------------------------------------------------------------
bool BVHBackend::occluded(const Ray& ray) const {
    assert(m_built);

    Ray4 r = makeRay4(ray);
    float tNear[4];

    // Single-level BVH
    if (!m_nodes.empty()) {
        uint32_t stack[64];
        int top = 0;
        stack[top++] = 0;

        while (top > 0) {
            const BVHNode& node = m_nodes[stack[--top]];
            int mask = intersectAABB4(node, r, tNear, r.tMax);
            if (!mask) continue;

            for (int c = 0; c < 4; ++c) {
                if (!(mask & (1 << c))) continue;
                if (node.isLeaf(c)) {
                    uint32_t offset = node.primOffset(c);
                    uint32_t count  = node.primCount(c);
                    for (uint32_t k = 0; k < count; ++k) {
                        uint32_t idx = m_primIndices[offset + k];
                        const BVHTriTrav& trav = m_trav[idx];
                        float t, u, v;
                        if (trav.isObjectSpace()) {
                            const MeshDesc& mesh = m_pool.mesh(trav.meshID());
                            Mat4f o2w = mesh.interpolateO2W(ray.time);
                            Ray4 ro = makeObjectSpaceRay4(ray, o2w.inverse());
                            if (intersectTriangle(trav, ro, t, u, v)) return true;
                        } else {
                            if (intersectTriangle(trav, r, t, u, v)) return true;
                        }
                    }
                } else {
                    stack[top++] = node.childIdx(c);
                }
            }
        }
    }

    // TLAS (instance groups)
    if (!m_tlasNodes.empty()) {
        uint32_t stack[64];
        int top = 0;
        stack[top++] = 0;

        while (top > 0) {
            const BVHNode& node = m_tlasNodes[stack[--top]];
            int mask = intersectAABB4(node, r, tNear, r.tMax);
            if (!mask) continue;

            for (int c = 0; c < 4; ++c) {
                if (!(mask & (1 << c))) continue;
                if (node.isLeaf(c)) {
                    uint32_t offset = node.primOffset(c);
                    uint32_t count  = node.primCount(c);
                    for (uint32_t k = 0; k < count; ++k) {
                        const TLASInstance& ti = m_tlasInstances[m_tlasPrimIndices[offset + k]];
                        {
                            float tn = r.tMin, tf = r.tMax;
                            for (int ax = 0; ax < 3; ++ax) {
                                float t0 = (ti.bMin[ax] - r.origin[ax]) * r.invDir[ax];
                                float t1 = (ti.bMax[ax] - r.origin[ax]) * r.invDir[ax];
                                tn = std::max(tn, std::min(t0, t1));
                                tf = std::min(tf, std::max(t0, t1));
                            }
                            if (tn > tf) continue;
                        }
                        const InstanceGroupDesc& grp  = m_pool.instanceGroup(ti.groupID);
                        const InstanceDesc&      inst = grp.instances[ti.instanceIdx];
                        Mat4f w2o = inst.interpolateW2O(ray.time);
                        Ray4  objRay = makeObjectSpaceRay4(ray, w2o);
                        if (occludedBLAS(m_blasList[ti.groupID], objRay)) return true;
                    }
                } else {
                    stack[top++] = node.childIdx(c);
                }
            }
        }
    }

    return false;
}

} // namespace anacapa
