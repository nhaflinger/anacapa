#include "CurveBrute.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <spdlog/spdlog.h>

namespace anacapa {

// ===========================================================================
// Tessellation helpers
// ===========================================================================

namespace {

static Vec3f bezierPoint(const Vec3f* p, float t) {
    float mt = 1.f - t;
    float mt2 = mt * mt, t2 = t * t;
    return p[0]*(mt2*mt) + p[1]*(3.f*mt2*t) + p[2]*(3.f*mt*t2) + p[3]*(t2*t);
}

static Vec3f bezierTangent(const Vec3f* p, float t) {
    float mt = 1.f - t;
    return (p[1] - p[0]) * (3.f*mt*mt)
         + (p[2] - p[1]) * (6.f*mt*t)
         + (p[3] - p[2]) * (3.f*t*t);
}

// Tessellate one cubic Bézier segment into tessSteps ribbon quads.
// Appends 2*tessSteps CpuHairTri entries.
static void tessellateSegment(
    const Vec3f* cvOpen,
    float        wRoot,     // half-width at segment start
    float        wTip,      // half-width at segment end
    Vec3f        color,
    uint32_t     matIdx,
    int          tessSteps,
    std::vector<CpuHairTri>& out)
{
    const int N = tessSteps;

    // Sample N+1 positions and build left/right ribbon vertices
    std::vector<Vec3f> left(N + 1), right(N + 1);

    for (int k = 0; k <= N; ++k) {
        float t = float(k) / float(N);
        float w = (wRoot * (1.f - t) + wTip * t);
        if (w < 5e-5f) w = 5e-5f;  // minimum 0.1 mm radius

        Vec3f pos  = bezierPoint(cvOpen, t);
        Vec3f tang = bezierTangent(cvOpen, t);
        if (tang.lengthSq() < 1e-12f) tang = cvOpen[3] - cvOpen[0];
        tang = safeNormalize(tang);

        Vec3f refUp = (std::abs(tang.y) > 0.9f) ? Vec3f{1.f,0.f,0.f} : Vec3f{0.f,1.f,0.f};
        Vec3f perp  = safeNormalize(cross(tang, refUp));

        left[k]  = pos - perp * w;   // h = -1
        right[k] = pos + perp * w;   // h = +1
    }

    // Build N quads = 2N triangles
    for (int k = 0; k < N; ++k) {
        Vec3f l0 = left[k],    r0 = right[k];
        Vec3f l1 = left[k+1],  r1 = right[k+1];

        float tMid = (float(k) + 0.5f) / float(N);
        Vec3f tang = bezierTangent(cvOpen, tMid);
        if (tang.lengthSq() < 1e-12f) tang = cvOpen[3] - cvOpen[0];
        tang = safeNormalize(tang);

        Vec3f refUp  = (std::abs(tang.y) > 0.9f) ? Vec3f{1.f,0.f,0.f} : Vec3f{0.f,1.f,0.f};
        Vec3f perp   = safeNormalize(cross(tang, refUp));
        Vec3f ribN   = safeNormalize(cross(perp, tang));

        CpuHairTri ht;
        ht.tangent = tang;
        ht.ribbonN = ribN;
        ht.color   = color;
        ht.matIdx  = matIdx;

        // Tri 0: (l0, r0, r1) — h values (-1, +1, +1)
        ht.v0 = l0; ht.e1 = r0 - l0; ht.e2 = r1 - l0;
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = +1.f;
        out.push_back(ht);

        // Tri 1: (l0, r1, l1) — h values (-1, +1, -1)
        ht.v0 = l0; ht.e1 = r1 - l0; ht.e2 = l1 - l0;
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = -1.f;
        out.push_back(ht);
    }
}

// ===========================================================================
// Hair BVH traversal helpers
// ===========================================================================

struct HairRayTraversal {
    float ox, oy, oz;
    float invDx, invDy, invDz;
    float tMin;

    static HairRayTraversal make(const Ray& ray) {
        auto safe = [](float v) { return std::abs(v) > 1e-9f ? v : 1e-9f; };
        return { ray.origin.x, ray.origin.y, ray.origin.z,
                 1.f / safe(ray.direction.x),
                 1.f / safe(ray.direction.y),
                 1.f / safe(ray.direction.z),
                 ray.tMin };
    }
};

static bool aabbHit(const float bmin[3], const float bmax[3],
                    const HairRayTraversal& r, float tMax)
{
    float tn = r.tMin, tf = tMax;
    float t0, t1;
    t0 = (bmin[0]-r.ox)*r.invDx;  t1 = (bmax[0]-r.ox)*r.invDx;
    tn = std::max(tn, std::min(t0,t1));  tf = std::min(tf, std::max(t0,t1));
    t0 = (bmin[1]-r.oy)*r.invDy;  t1 = (bmax[1]-r.oy)*r.invDy;
    tn = std::max(tn, std::min(t0,t1));  tf = std::min(tf, std::max(t0,t1));
    t0 = (bmin[2]-r.oz)*r.invDz;  t1 = (bmax[2]-r.oz)*r.invDz;
    tn = std::max(tn, std::min(t0,t1));  tf = std::min(tf, std::max(t0,t1));
    return tn <= tf;
}

// Möller-Trumbore ray-triangle intersection.
// u, v are barycentric coords; hit point = v0*(1-u-v) + v1*u + v2*v.
static bool intersectHairTri(const CpuHairTri& ht, const Ray& ray,
                               float ray_tMin, float& inoutBestT,
                               float& outU, float& outV)
{
    Vec3f h = cross(ray.direction, ht.e2);
    float a = dot(ht.e1, h);
    if (std::abs(a) < 1e-9f) return false;
    float f = 1.f / a;
    Vec3f s = ray.origin - ht.v0;
    float u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;
    Vec3f q = cross(s, ht.e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.f || u + v > 1.f) return false;
    float t = f * dot(ht.e2, q);
    if (t < ray_tMin || t >= inoutBestT) return false;
    inoutBestT = t;
    outU = u; outV = v;
    return true;
}

static bool intersectHairTriAny(const CpuHairTri& ht, const Ray& ray,
                                 float ray_tMin, float ray_tMax)
{
    Vec3f h = cross(ray.direction, ht.e2);
    float a = dot(ht.e1, h);
    if (std::abs(a) < 1e-9f) return false;
    float f = 1.f / a;
    Vec3f s = ray.origin - ht.v0;
    float u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;
    Vec3f q = cross(s, ht.e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.f || u + v > 1.f) return false;
    float t = f * dot(ht.e2, q);
    return t >= ray_tMin && t < ray_tMax;
}

// ===========================================================================
// Hair BVH build (SAH, binary)
// ===========================================================================

constexpr int   kHairSAHBuckets    = 12;
constexpr int   kHairMaxLeaf       = 4;
constexpr float kHairTraversalCost = 1.f;
constexpr float kHairIntersectCost = 1.f;

static float boxArea(const float bmin[3], const float bmax[3]) {
    float d0 = bmax[0]-bmin[0], d1 = bmax[1]-bmin[1], d2 = bmax[2]-bmin[2];
    if (d0 < 0.f || d1 < 0.f || d2 < 0.f) return 0.f;
    return 2.f * (d0*d1 + d1*d2 + d2*d0);
}

static void boxUnion(float bmin[3], float bmax[3],
                     const float sbmin[3], const float sbmax[3]) {
    for (int k = 0; k < 3; ++k) {
        bmin[k] = std::min(bmin[k], sbmin[k]);
        bmax[k] = std::max(bmax[k], sbmax[k]);
    }
}

struct HairTriWork {
    uint32_t triIdx;
    float    bmin[3], bmax[3];
    float    centroid[3];
};

static uint32_t buildHairBVH(
    std::vector<HairTriWork>&  work,
    std::vector<uint32_t>&     outPrimIdx,
    std::vector<HairNode>&     nodes,
    uint32_t start, uint32_t end)
{
    uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back({});

    float bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i)
        boxUnion(bmin, bmax, work[i].bmin, work[i].bmax);

    uint32_t count    = end - start;
    float    leafCost = kHairIntersectCost * (float)count;

    auto makeLeaf = [&]() {
        uint32_t first = (uint32_t)outPrimIdx.size();
        for (uint32_t i = start; i < end; ++i)
            outPrimIdx.push_back(work[i].triIdx);
        for (int k = 0; k < 3; ++k) {
            nodes[nodeIdx].bmin[k] = bmin[k];
            nodes[nodeIdx].bmax[k] = bmax[k];
        }
        nodes[nodeIdx].left_or_prim   = first;
        nodes[nodeIdx].right_or_count = count | 0x80000000u;
    };

    if (count <= (uint32_t)kHairMaxLeaf) { makeLeaf(); return nodeIdx; }

    float cmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float cmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i)
        for (int k = 0; k < 3; ++k) {
            cmin[k] = std::min(cmin[k], work[i].centroid[k]);
            cmax[k] = std::max(cmax[k], work[i].centroid[k]);
        }

    float parentSA  = boxArea(bmin, bmax);
    float bestCost  = leafCost;
    int   bestAxis  = -1, bestBucket = -1;

    if (parentSA > 1e-12f) {
        struct Bucket {
            float    bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
            float    bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            uint32_t count   = 0;
        };

        for (int axis = 0; axis < 3; ++axis) {
            float range = cmax[axis] - cmin[axis];
            if (range < 1e-7f) continue;

            Bucket buckets[kHairSAHBuckets];
            for (uint32_t i = start; i < end; ++i) {
                int b = (int)(kHairSAHBuckets * ((work[i].centroid[axis] - cmin[axis]) / range));
                b = std::clamp(b, 0, kHairSAHBuckets - 1);
                buckets[b].count++;
                boxUnion(buckets[b].bmin, buckets[b].bmax, work[i].bmin, work[i].bmax);
            }

            float    lbMin[kHairSAHBuckets-1][3], lbMax[kHairSAHBuckets-1][3];
            uint32_t lCount[kHairSAHBuckets-1];
            {
                float lb[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
                float ub[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                uint32_t lc = 0;
                for (int i = 0; i < kHairSAHBuckets - 1; ++i) {
                    boxUnion(lb, ub, buckets[i].bmin, buckets[i].bmax);
                    lc += buckets[i].count;
                    for (int k = 0; k < 3; ++k) { lbMin[i][k] = lb[k]; lbMax[i][k] = ub[k]; }
                    lCount[i] = lc;
                }
            }

            float    rb[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
            float    ru[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            uint32_t rc = 0;
            for (int i = kHairSAHBuckets - 2; i >= 0; --i) {
                boxUnion(rb, ru, buckets[i+1].bmin, buckets[i+1].bmax);
                rc += buckets[i+1].count;
                if (lCount[i] == 0 || rc == 0) continue;
                float cost = kHairTraversalCost
                           + kHairIntersectCost
                             * (boxArea(lbMin[i], lbMax[i]) * lCount[i]
                             +  boxArea(rb, ru)              * rc) / parentSA;
                if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestBucket = i; }
            }
        }
    }

    if (bestAxis < 0) { makeLeaf(); return nodeIdx; }

    float range = cmax[bestAxis] - cmin[bestAxis];
    auto midIt = std::partition(
        work.begin() + start, work.begin() + end,
        [&](const HairTriWork& w) {
            int b = (int)(kHairSAHBuckets * ((w.centroid[bestAxis] - cmin[bestAxis]) / range));
            return std::clamp(b, 0, kHairSAHBuckets-1) <= bestBucket;
        });
    uint32_t mid = (uint32_t)(midIt - work.begin());
    if (mid == start || mid == end) mid = (start + end) / 2;

    uint32_t leftIdx  = buildHairBVH(work, outPrimIdx, nodes, start, mid);
    uint32_t rightIdx = buildHairBVH(work, outPrimIdx, nodes, mid,   end);

    for (int k = 0; k < 3; ++k) {
        nodes[nodeIdx].bmin[k] = bmin[k];
        nodes[nodeIdx].bmax[k] = bmax[k];
    }
    nodes[nodeIdx].left_or_prim   = leftIdx;
    nodes[nodeIdx].right_or_count = rightIdx;
    return nodeIdx;
}

}  // anonymous namespace

// ===========================================================================
// CurveBrute implementation
// ===========================================================================

CurveBrute::CurveBrute(const GeometryPool& triPool, const CurvePool& curvePool,
                       int tessSteps)
    : m_triBvh(triPool), m_curvePool(curvePool)
    , m_tessSteps(tessSteps > 0 ? tessSteps : kDefaultTessSteps)
{}

void CurveBrute::commit() {
    m_triBvh.commit();
    m_hairNodes.clear();
    m_hairPrimIdx.clear();
    m_hairTris.clear();

    const uint32_t S = (uint32_t)m_curvePool.numStrands();
    if (S == 0) return;

    // Tessellate all strands into ribbon triangles.
    for (uint32_t si = 0; si < S; ++si) {
        const StrandDesc& strand = m_curvePool.strand(si);
        const uint32_t    numSeg = strand.numSegments();

        float v0base = 0.f;
        const float vStep = 1.f / float(numSeg);

        for (uint32_t seg = 0; seg < numSeg; ++seg) {
            const Vec3f* cvOpen = &strand.controlPoints[seg * 3];
            float strandV0 = seg * vStep;
            float strandV1 = strandV0 + vStep;
            float wRoot = strand.widthAt(strandV0) * 0.5f;
            float wTip  = strand.widthAt(strandV1) * 0.5f;

            tessellateSegment(cvOpen, wRoot, wTip,
                              strand.color,
                              strand.materialIndex,
                              m_tessSteps,
                              m_hairTris);
        }
        (void)v0base;
    }

    if (m_hairTris.empty()) return;

    const uint32_t T = (uint32_t)m_hairTris.size();

    // Build per-triangle AABB work array
    std::vector<HairTriWork> work;
    work.reserve(T);
    for (uint32_t i = 0; i < T; ++i) {
        const CpuHairTri& ht = m_hairTris[i];
        Vec3f v1 = ht.v0 + ht.e1;
        Vec3f v2 = ht.v0 + ht.e2;

        HairTriWork w;
        w.triIdx = i;
        w.bmin[0] = std::min({ht.v0.x, v1.x, v2.x});
        w.bmin[1] = std::min({ht.v0.y, v1.y, v2.y});
        w.bmin[2] = std::min({ht.v0.z, v1.z, v2.z});
        w.bmax[0] = std::max({ht.v0.x, v1.x, v2.x});
        w.bmax[1] = std::max({ht.v0.y, v1.y, v2.y});
        w.bmax[2] = std::max({ht.v0.z, v1.z, v2.z});
        // Small epsilon to avoid degenerate AABBs on flat ribbon tris
        for (int k = 0; k < 3; ++k) {
            if (w.bmax[k] - w.bmin[k] < 1e-6f) {
                w.bmin[k] -= 1e-6f;
                w.bmax[k] += 1e-6f;
            }
        }
        w.centroid[0] = (w.bmin[0] + w.bmax[0]) * 0.5f;
        w.centroid[1] = (w.bmin[1] + w.bmax[1]) * 0.5f;
        w.centroid[2] = (w.bmin[2] + w.bmax[2]) * 0.5f;
        work.push_back(w);
    }

    uint32_t estLeaves = (T + kHairMaxLeaf - 1) / kHairMaxLeaf;
    m_hairNodes.reserve(2 * estLeaves + 4);
    m_hairPrimIdx.reserve(T);

    buildHairBVH(work, m_hairPrimIdx, m_hairNodes, 0, T);

    spdlog::info("CurveBrute (tessellated): {} strands → {} tris, {} BVH nodes (tessSteps={})",
                 S, T, m_hairNodes.size(), m_tessSteps);
}

TraceResult CurveBrute::trace(const Ray& ray) const {
    // Triangle BVH first.
    TraceResult result = m_triBvh.trace(ray);
    float bestT = result.hit ? result.si.t : ray.tMax;

    if (m_hairNodes.empty()) return result;

    HairRayTraversal rt = HairRayTraversal::make(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    float   bestU = 0.f, bestV = 0.f;
    uint32_t bestTriIdx = ~0u;

    while (top > 0) {
        const HairNode& node = m_hairNodes[stack[--top]];

        if (!aabbHit(node.bmin, node.bmax, rt, bestT)) continue;

        if (node.isLeaf()) {
            uint32_t end = node.left_or_prim + node.primCount();
            for (uint32_t i = node.left_or_prim; i < end; ++i) {
                uint32_t idx = m_hairPrimIdx[i];
                float u, v;
                if (intersectHairTri(m_hairTris[idx], ray, ray.tMin, bestT, u, v)) {
                    bestU      = u;
                    bestV      = v;
                    bestTriIdx = idx;
                }
            }
        } else {
            stack[top++] = node.right_or_count;
            stack[top++] = node.left_or_prim;
        }
    }

    if (bestTriIdx == ~0u) return result;

    // Build SurfaceInteraction from baked hair tri data.
    const CpuHairTri& ht = m_hairTris[bestTriIdx];
    SurfaceInteraction si;
    si.p       = ray.at(bestT);
    si.t       = bestT;
    si.meshID  = ht.matIdx;
    si.primID  = bestTriIdx;
    si.strandID = ~0u;  // normal-offset self-intersection avoidance; no strand skip needed
    si.isCurve = true;

    float bw = 1.f - bestU - bestV;
    si.h = std::clamp(ht.h0 * bw + ht.h1 * bestU + ht.h2 * bestV,
                      -1.f + 1e-5f, 1.f - 1e-5f);
    si.color   = ht.color;
    si.dpdu    = ht.tangent;

    // Flip ribbon normal to face incoming ray for spawnRay offset
    si.n  = (dot(-ray.direction, ht.ribbonN) >= 0.f) ? ht.ribbonN : -ht.ribbonN;
    si.ng = si.n;
    si.dpdv = safeNormalize(cross(ht.tangent, si.n));

    si.uv     = {0.f, 0.f};
    si.strandV = 0.f;

    result.hit = true;
    result.si  = si;
    return result;
}

bool CurveBrute::occluded(const Ray& ray) const {
    if (m_triBvh.occluded(ray)) return true;
    if (m_hairNodes.empty()) return false;

    HairRayTraversal rt = HairRayTraversal::make(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const HairNode& node = m_hairNodes[stack[--top]];

        if (!aabbHit(node.bmin, node.bmax, rt, ray.tMax)) continue;

        if (node.isLeaf()) {
            uint32_t end = node.left_or_prim + node.primCount();
            for (uint32_t i = node.left_or_prim; i < end; ++i) {
                uint32_t idx = m_hairPrimIdx[i];
                if (intersectHairTriAny(m_hairTris[idx], ray, ray.tMin, ray.tMax))
                    return true;
            }
        } else {
            stack[top++] = node.right_or_count;
            stack[top++] = node.left_or_prim;
        }
    }
    return false;
}

}  // namespace anacapa
