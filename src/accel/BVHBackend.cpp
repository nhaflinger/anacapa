#include "BVHBackend.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

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

    if (m_trav.empty()) { m_built = true; return; }

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
        primInfo[i] = { b, b.centroid(), i };
    }

    m_primIndices.resize(n);
    std::iota(m_primIndices.begin(), m_primIndices.end(), 0u);

    // Phase 1: binary SAH build
    std::vector<BuildBVHNode> buildNodes;
    buildNodes.reserve(2 * n);
    buildRecursive(primInfo, 0, n, buildNodes);

    // Phase 2: collapse binary tree into BVH4 SOA nodes
    repackBuildTree(buildNodes);

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
                                    std::vector<BuildBVHNode>& buildNodes) {
    uint32_t nodeIdx = static_cast<uint32_t>(buildNodes.size());
    buildNodes.emplace_back();

    BBox3f bounds, centroidBounds;
    for (uint32_t i = start; i < end; ++i) {
        bounds.expand(primInfo[i].bounds);
        centroidBounds.expand(primInfo[i].centroid);
    }
    storeBuildBounds(buildNodes[nodeIdx], bounds);

    uint32_t count = end - start;
    int bestAxis = -1, splitBucket = -1;
    if (count > static_cast<uint32_t>(kMaxLeafPrims))
        splitBucket = sahSplit(primInfo, start, end, centroidBounds, bestAxis);

    if (bestAxis < 0 || splitBucket < 0) {
        uint32_t offset = static_cast<uint32_t>(m_primIndices.size());
        for (uint32_t i = start; i < end; ++i)
            m_primIndices.push_back(primInfo[i].originalIndex);
        buildNodes[nodeIdx].dataA = offset;
        buildNodes[nodeIdx].dataB = count | BuildBVHNode::kLeafFlag;
        return nodeIdx;
    }

    float range = centroidBounds.diagonal()[bestAxis];
    uint32_t mid = start;
    if (range > 0.f) {
        auto* p = std::partition(
            primInfo.data() + start, primInfo.data() + end,
            [&](const PrimInfo& pi) {
                int b = static_cast<int>(kSAHBuckets *
                    ((pi.centroid[bestAxis] - centroidBounds.pMin[bestAxis]) / range));
                return std::clamp(b, 0, kSAHBuckets - 1) <= splitBucket;
            });
        mid = static_cast<uint32_t>(p - primInfo.data());
    }
    if (mid == start || mid == end) mid = start + count / 2;

    buildRecursive(primInfo, start, mid, buildNodes);
    uint32_t rightIdx = buildRecursive(primInfo, mid, end, buildNodes);

    buildNodes[nodeIdx].dataA = rightIdx;
    buildNodes[nodeIdx].dataB = static_cast<uint32_t>(bestAxis);
    return nodeIdx;
}

int BVHBackend::sahSplit(const std::vector<PrimInfo>& primInfo,
                         uint32_t start, uint32_t end,
                         const BBox3f& centroidBounds,
                         int& outAxis) const {
    struct Bucket { BBox3f bounds; uint32_t count = 0; };

    float bestCost = std::numeric_limits<float>::infinity();
    int   bestSplit = -1;
    outAxis = -1;

    BBox3f parentBounds;
    for (uint32_t i = start; i < end; ++i) parentBounds.expand(primInfo[i].bounds);
    Vec3f d = parentBounds.diagonal();
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

    float leafCost = kIntersectCost * static_cast<float>(end - start);
    if (bestCost >= leafCost) { outAxis = -1; return -1; }
    return bestSplit;
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
                                    uint32_t oldIdx) {
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
    uint32_t newIdx = static_cast<uint32_t>(m_nodes.size());
    m_nodes.push_back(BVHNode{});

    // Fill each slot; recurse for interior children
    for (int s = 0; s < nSlots; ++s) {
        uint32_t si = slots[s].idx;
        const BuildBVHNode& sn = build[si];

        m_nodes[newIdx].minX[s] = sn.boundsMin[0];
        m_nodes[newIdx].minY[s] = sn.boundsMin[1];
        m_nodes[newIdx].minZ[s] = sn.boundsMin[2];
        m_nodes[newIdx].maxX[s] = sn.boundsMax[0];
        m_nodes[newIdx].maxY[s] = sn.boundsMax[1];
        m_nodes[newIdx].maxZ[s] = sn.boundsMax[2];

        if (sn.isLeaf()) {
            m_nodes[newIdx].childData[s] = sn.primOffset();
            m_nodes[newIdx].childMeta[s] = sn.primCount() | BVHNode::kLeafFlag;
        } else {
            uint32_t childNodeIdx = buildBVH4Node(build, si);
            // Re-index through newIdx — m_nodes may have reallocated during recursion
            m_nodes[newIdx].childData[s] = childNodeIdx;
            m_nodes[newIdx].childMeta[s] = 0;
        }
    }

    // Pad unused slots with empty AABBs
    for (int s = nSlots; s < 4; ++s)
        fillEmptySlot(m_nodes[newIdx], s);

    return newIdx;
}

void BVHBackend::repackBuildTree(const std::vector<BuildBVHNode>& build) {
    if (build.empty()) return;

    const BuildBVHNode& root = build[0];

    if (root.isLeaf()) {
        // Degenerate: single leaf — wrap it in slot 0, pad slots 1-3.
        m_nodes.push_back(BVHNode{});
        m_nodes[0].minX[0] = root.boundsMin[0];
        m_nodes[0].minY[0] = root.boundsMin[1];
        m_nodes[0].minZ[0] = root.boundsMin[2];
        m_nodes[0].maxX[0] = root.boundsMax[0];
        m_nodes[0].maxY[0] = root.boundsMax[1];
        m_nodes[0].maxZ[0] = root.boundsMax[2];
        m_nodes[0].childData[0] = root.primOffset();
        m_nodes[0].childMeta[0] = root.primCount() | BVHNode::kLeafFlag;
        for (int s = 1; s < 4; ++s) fillEmptySlot(m_nodes[0], s);
    } else {
        buildBVH4Node(build, 0);  // root BVH4 node will be at index 0
    }
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
        si.p  = worldXfm->transformPoint(trav.v0 + trav.e1 * u + trav.e2 * v);
        si.ng = safeNormalize(worldXfm->transformNormal(attrib.n));
        si.n  = safeNormalize(worldXfm->transformNormal(
                    attrib.sn0 * w + attrib.sn1 * u + attrib.sn2 * v));
    } else {
        si.p  = trav.v0 + trav.e1 * u + trav.e2 * v;
        si.ng = attrib.n;
        si.n  = safeNormalize(attrib.sn0 * w + attrib.sn1 * u + attrib.sn2 * v);
    }

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

    TraceResult result;
    if (hitIdx != ~0u) {
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
    if (m_nodes.empty()) return false;

    Ray4 r = makeRay4(ray);
    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;
    float tNear[4];

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
    return false;
}

} // namespace anacapa
