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
// Appends 2*tessSteps CpuHairTri entries (open-time geometry + shade data).
// When cvClose is non-null AND outClose is non-null, also appends matching
// close-time CpuHairTriClose entries — same algorithm against the close-time
// CV array.  Shading attributes (tangent, ribbonN) stay open-time only.
static void tessellateSegment(
    const Vec3f* cvOpen,
    const Vec3f* cvClose,    // nullptr for static segments
    float        wRoot,
    float        wTip,
    Vec3f        color,
    uint32_t     matIdx,
    int          tessSteps,
    std::vector<CpuHairTri>&      out,
    std::vector<CpuHairTriClose>* outClose)
{
    const int  N        = tessSteps;
    const bool hasMotion = (cvClose != nullptr) && (outClose != nullptr);

    // Sample N+1 ribbon cross-sections.  Width and perp orientation come
    // from the open-time tangent so the close-time ribbon stays parallel
    // to the open ribbon (no twist) — same convention as the GPU.
    std::vector<Vec3f> leftO (N + 1), rightO (N + 1);
    std::vector<Vec3f> leftC (N + 1), rightC (N + 1);

    for (int k = 0; k <= N; ++k) {
        float t = float(k) / float(N);
        float w = (wRoot * (1.f - t) + wTip * t);
        if (w < 5e-5f) w = 5e-5f;

        Vec3f posO = bezierPoint(cvOpen, t);
        Vec3f tang = bezierTangent(cvOpen, t);
        if (tang.lengthSq() < 1e-12f) tang = cvOpen[3] - cvOpen[0];
        tang = safeNormalize(tang);

        Vec3f refUp = (std::abs(tang.y) > 0.9f) ? Vec3f{1.f,0.f,0.f} : Vec3f{0.f,1.f,0.f};
        Vec3f perp  = safeNormalize(cross(tang, refUp));

        leftO [k] = posO - perp * w;
        rightO[k] = posO + perp * w;

        if (hasMotion) {
            Vec3f posC = bezierPoint(cvClose, t);
            leftC [k] = posC - perp * w;
            rightC[k] = posC + perp * w;
        }
    }

    // Build N quads = 2N triangles
    for (int k = 0; k < N; ++k) {
        Vec3f l0o = leftO[k],    r0o = rightO[k];
        Vec3f l1o = leftO[k+1],  r1o = rightO[k+1];

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
        ht.v0 = l0o; ht.e1 = r0o - l0o; ht.e2 = r1o - l0o;
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = +1.f;
        out.push_back(ht);
        if (hasMotion) {
            Vec3f l0c = leftC[k], r0c = rightC[k], r1c = rightC[k+1];
            outClose->push_back({ l0c, r0c - l0c, r1c - l0c });
        }

        // Tri 1: (l0, r1, l1) — h values (-1, +1, -1)
        ht.v0 = l0o; ht.e1 = r1o - l0o; ht.e2 = l1o - l0o;
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = -1.f;
        out.push_back(ht);
        if (hasMotion) {
            Vec3f l0c = leftC[k], r1c = rightC[k+1], l1c = leftC[k+1];
            outClose->push_back({ l0c, r1c - l0c, l1c - l0c });
        }
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

// Möller-Trumbore against the time-interpolated triangle.  When htc is
// nullptr (no motion) this collapses to the open-time tri at zero cost.
// u, v are barycentric coords; hit point = v0*(1-u-v) + v1*u + v2*v.
static bool intersectHairTri(const CpuHairTri& ht, const CpuHairTriClose* htc,
                               float time, const Ray& ray,
                               float ray_tMin, float& inoutBestT,
                               float& outU, float& outV)
{
    Vec3f v0 = ht.v0, e1 = ht.e1, e2 = ht.e2;
    if (htc) {
        v0 = v0 + (htc->v0 - v0) * time;
        e1 = e1 + (htc->e1 - e1) * time;
        e2 = e2 + (htc->e2 - e2) * time;
    }
    Vec3f h = cross(ray.direction, e2);
    float a = dot(e1, h);
    if (std::abs(a) < 1e-9f) return false;
    float f = 1.f / a;
    Vec3f s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;
    Vec3f q = cross(s, e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.f || u + v > 1.f) return false;
    float t = f * dot(e2, q);
    if (t < ray_tMin || t >= inoutBestT) return false;
    inoutBestT = t;
    outU = u; outV = v;
    return true;
}

static bool intersectHairTriAny(const CpuHairTri& ht, const CpuHairTriClose* htc,
                                 float time, const Ray& ray,
                                 float ray_tMin, float ray_tMax)
{
    Vec3f v0 = ht.v0, e1 = ht.e1, e2 = ht.e2;
    if (htc) {
        v0 = v0 + (htc->v0 - v0) * time;
        e1 = e1 + (htc->e1 - e1) * time;
        e2 = e2 + (htc->e2 - e2) * time;
    }
    Vec3f h = cross(ray.direction, e2);
    float a = dot(e1, h);
    if (std::abs(a) < 1e-9f) return false;
    float f = 1.f / a;
    Vec3f s = ray.origin - v0;
    float u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;
    Vec3f q = cross(s, e1);
    float v = f * dot(ray.direction, q);
    if (v < 0.f || u + v > 1.f) return false;
    float t = f * dot(e2, q);
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
    m_hairTrisClose.clear();

    const uint32_t S = (uint32_t)m_curvePool.numStrands();
    if (S == 0) return;

    // Decide once whether any strand carries motion keys; if so we pay
    // for parallel close-time triangles, otherwise the close vector stays
    // empty and static scenes pay nothing extra.
    bool anyMotion = false;
    for (uint32_t si = 0; si < S; ++si) {
        if (m_curvePool.strand(si).hasMotion()) { anyMotion = true; break; }
    }

    for (uint32_t si = 0; si < S; ++si) {
        const StrandDesc& strand = m_curvePool.strand(si);
        const uint32_t    numSeg = strand.numSegments();
        const bool        strandMotion = strand.hasMotion();
        const float       vStep = 1.f / float(numSeg);

        for (uint32_t seg = 0; seg < numSeg; ++seg) {
            const Vec3f* cvOpen  = &strand.controlPoints[seg * 3];
            // When the scene has motion but THIS strand is static, we still
            // need close-time entries so the parallel array stays aligned —
            // tessellateSegment treats cvClose == cvOpen as zero displacement.
            const Vec3f* cvClose = anyMotion
                                 ? (strandMotion
                                      ? &strand.controlPointsClose[seg * 3]
                                      : cvOpen)
                                 : nullptr;
            float strandV0 = seg * vStep;
            float strandV1 = strandV0 + vStep;
            float wRoot = strand.widthAt(strandV0) * 0.5f;
            float wTip  = strand.widthAt(strandV1) * 0.5f;

            tessellateSegment(cvOpen, cvClose, wRoot, wTip,
                              strand.color, strand.materialIndex,
                              m_tessSteps,
                              m_hairTris,
                              anyMotion ? &m_hairTrisClose : nullptr);
        }
    }

    if (m_hairTris.empty()) return;

    const uint32_t T = (uint32_t)m_hairTris.size();

    // Build per-triangle AABB work array — when motion is present, expand
    // each leaf bound to enclose both shutter-open and shutter-close tris
    // so motion-displaced hair isn't culled at traversal time.
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

        if (anyMotion) {
            const CpuHairTriClose& hc = m_hairTrisClose[i];
            Vec3f c1 = hc.v0 + hc.e1;
            Vec3f c2 = hc.v0 + hc.e2;
            w.bmin[0] = std::min({w.bmin[0], hc.v0.x, c1.x, c2.x});
            w.bmin[1] = std::min({w.bmin[1], hc.v0.y, c1.y, c2.y});
            w.bmin[2] = std::min({w.bmin[2], hc.v0.z, c1.z, c2.z});
            w.bmax[0] = std::max({w.bmax[0], hc.v0.x, c1.x, c2.x});
            w.bmax[1] = std::max({w.bmax[1], hc.v0.y, c1.y, c2.y});
            w.bmax[2] = std::max({w.bmax[2], hc.v0.z, c1.z, c2.z});
        }

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

    spdlog::info("CurveBrute (tessellated): {} strands → {} tris ({}), {} BVH nodes (tessSteps={})",
                 S, T, anyMotion ? "motion-aware" : "static",
                 m_hairNodes.size(), m_tessSteps);
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
                const CpuHairTriClose* hc = m_hairTrisClose.empty()
                                          ? nullptr : &m_hairTrisClose[idx];
                float u, v;
                if (intersectHairTri(m_hairTris[idx], hc, ray.time, ray,
                                       ray.tMin, bestT, u, v)) {
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
                const CpuHairTriClose* hc = m_hairTrisClose.empty()
                                          ? nullptr : &m_hairTrisClose[idx];
                if (intersectHairTriAny(m_hairTris[idx], hc, ray.time, ray,
                                          ray.tMin, ray.tMax))
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
