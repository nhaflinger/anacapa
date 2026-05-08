#include "CurveBrute.h"
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <spdlog/spdlog.h>

namespace anacapa {

// ===========================================================================
// Ribbon intersection helpers
//
// Algorithm (RenderMan-style ray-ribbon):
//   1. Build a ray-space coordinate frame where the ray travels along +Z.
//   2. Project each cubic Bézier segment's control points into ray space.
//      The 2D (x,y) projected curve is the "silhouette" seen from the ray.
//   3. Recursively subdivide with de Casteljau splitting until a leaf segment
//      is small enough to approximate as a straight 2D line.
//   4. At a leaf: find the closest 2D point on the line to the origin (0,0),
//      check the distance against the interpolated half-width, and recover
//      the ray parameter from the z component.
// ===========================================================================

namespace {

// ---------------------------------------------------------------------------
// SAH constants for curve BVH build
// ---------------------------------------------------------------------------
constexpr int   kCurveSAHBuckets   = 12;
constexpr int   kCurveMaxLeaf      = 4;
constexpr float kCurveTraversalCost = 1.f;
constexpr float kCurveIntersectCost = 1.f;

// ---------------------------------------------------------------------------
// Per-ray traversal state: precomputed invDir to avoid per-node divisions.
// ---------------------------------------------------------------------------
struct RayTraversal {
    float ox, oy, oz;
    float invDx, invDy, invDz;
    float tMin;

    static RayTraversal make(const Ray& ray) {
        auto safe = [](float v) { return std::abs(v) > 1e-9f ? v : 1e-9f; };
        return { ray.origin.x, ray.origin.y, ray.origin.z,
                 1.f / safe(ray.direction.x),
                 1.f / safe(ray.direction.y),
                 1.f / safe(ray.direction.z),
                 ray.tMin };
    }
};

// Branchless slab test — 3 multiplies, no divisions, no branches.
static bool aabbHit(const float bmin[3], const float bmax[3],
                    const RayTraversal& r, float tMax) {
    float tn = r.tMin, tf = tMax;
    float t0, t1;

    t0 = (bmin[0] - r.ox) * r.invDx;  t1 = (bmax[0] - r.ox) * r.invDx;
    tn = std::max(tn, std::min(t0, t1)); tf = std::min(tf, std::max(t0, t1));

    t0 = (bmin[1] - r.oy) * r.invDy;  t1 = (bmax[1] - r.oy) * r.invDy;
    tn = std::max(tn, std::min(t0, t1)); tf = std::min(tf, std::max(t0, t1));

    t0 = (bmin[2] - r.oz) * r.invDz;  t1 = (bmax[2] - r.oz) * r.invDz;
    tn = std::max(tn, std::min(t0, t1)); tf = std::min(tf, std::max(t0, t1));

    return tn <= tf;
}

// ---------------------------------------------------------------------------
// SAH build helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Ray-space frame: ray travels along +Z, origin at (0,0,0).
// ---------------------------------------------------------------------------
struct RayFrame {
    Vec3f origin;
    Vec3f xAxis;
    Vec3f yAxis;
    Vec3f zAxis;  // == ray.direction

    Vec3f project(Vec3f p) const {
        Vec3f q = p - origin;
        return { dot(q, xAxis), dot(q, yAxis), dot(q, zAxis) };
    }
};

static RayFrame makeRayFrame(const Ray& ray) {
    RayFrame f;
    f.origin = ray.origin;
    f.zAxis  = ray.direction;
    buildOrthonormalBasis(f.zAxis, f.xAxis, f.yAxis);
    return f;
}

// ---------------------------------------------------------------------------
// Cubic Bézier utilities
// ---------------------------------------------------------------------------

static Vec3f bezierEval(Vec3f P0, Vec3f P1, Vec3f P2, Vec3f P3, float t) {
    float mt = 1.f - t;
    float mt2 = mt * mt, t2 = t * t;
    return P0*(mt2*mt) + P1*(3.f*mt2*t) + P2*(3.f*mt*t2) + P3*(t2*t);
}

static Vec3f bezierTangent(Vec3f P0, Vec3f P1, Vec3f P2, Vec3f P3, float t) {
    float mt = 1.f - t;
    return (P1 - P0) * (3.f * mt * mt)
         + (P2 - P1) * (6.f * mt * t)
         + (P3 - P2) * (3.f * t  * t);
}

static void bezierSplitMid(
    Vec3f P0, Vec3f P1, Vec3f P2, Vec3f P3,
    Vec3f& L0, Vec3f& L1, Vec3f& L2, Vec3f& L3,
    Vec3f& R0, Vec3f& R1, Vec3f& R2, Vec3f& R3)
{
    Vec3f Q01 = (P0 + P1) * 0.5f;
    Vec3f Q12 = (P1 + P2) * 0.5f;
    Vec3f Q23 = (P2 + P3) * 0.5f;
    Vec3f R01 = (Q01 + Q12) * 0.5f;
    Vec3f R12 = (Q12 + Q23) * 0.5f;
    Vec3f M   = (R01 + R12) * 0.5f;

    L0 = P0;  L1 = Q01;  L2 = R01;  L3 = M;
    R0 = M;   R1 = R12;  R2 = Q23;  R3 = P3;
}

// ---------------------------------------------------------------------------
// Core recursive subdivision — unchanged from Phase 1.
// ---------------------------------------------------------------------------
static bool subdivide(
    const RayFrame& frame,
    Vec3f P0, Vec3f P1, Vec3f P2, Vec3f P3,
    float halfW0, float halfW3,
    float tOffset, float tScale,
    float ray_tMin,
    float& inoutBestT,
    int   depth,
    float& outTSeg,
    Vec3f& outTangent,
    float& outH)           // impact parameter: px/halfW ∈ [-1,1]
{
    Vec3f cp0 = frame.project(P0);
    Vec3f cp1 = frame.project(P1);
    Vec3f cp2 = frame.project(P2);
    Vec3f cp3 = frame.project(P3);

    float maxHalfW = std::max(halfW0, halfW3);

    float zMin = std::min({cp0.z, cp1.z, cp2.z, cp3.z});
    float zMax = std::max({cp0.z, cp1.z, cp2.z, cp3.z});
    if (zMin - maxHalfW > inoutBestT) return false;
    if (zMax + maxHalfW < ray_tMin)   return false;

    float xMin = std::min({cp0.x, cp1.x, cp2.x, cp3.x});
    float xMax = std::max({cp0.x, cp1.x, cp2.x, cp3.x});
    float yMin = std::min({cp0.y, cp1.y, cp2.y, cp3.y});
    float yMax = std::max({cp0.y, cp1.y, cp2.y, cp3.y});
    float cx = (0.f < xMin) ? xMin : (0.f > xMax ? xMax : 0.f);
    float cy = (0.f < yMin) ? yMin : (0.f > yMax ? yMax : 0.f);
    if (cx * cx + cy * cy > maxHalfW * maxHalfW) return false;

    if (depth == 0) {
        float ax = cp0.x, ay = cp0.y;
        float bx = cp3.x, by = cp3.y;
        float dx = bx - ax, dy = by - ay;
        float lenSq = dx * dx + dy * dy;

        float tLine;
        if (lenSq < 1e-12f) {
            tLine = 0.f;
        } else {
            tLine = std::max(0.f, std::min(1.f, (-ax * dx - ay * dy) / lenSq));
        }

        float px = ax + dx * tLine;
        float py = ay + dy * tLine;
        float dist2 = px * px + py * py;

        float halfW = halfW0 * (1.f - tLine) + halfW3 * tLine;
        if (dist2 > halfW * halfW) return false;

        float tRay = cp0.z + (cp3.z - cp0.z) * tLine;
        if (tRay < ray_tMin || tRay >= inoutBestT) return false;

        Vec3f tang = bezierTangent(P0, P1, P2, P3, tLine);
        if (tang.lengthSq() < 1e-12f) tang = P3 - P0;
        tang = safeNormalize(tang);

        inoutBestT = tRay;
        outTSeg    = tOffset + tLine * tScale;
        outTangent = tang;
        outH       = (halfW > 1e-8f) ? std::clamp(px / halfW, -1.f + 1e-5f, 1.f - 1e-5f) : 0.f;
        return true;
    }

    Vec3f L0, L1, L2, L3, R0, R1, R2, R3;
    bezierSplitMid(P0, P1, P2, P3, L0, L1, L2, L3, R0, R1, R2, R3);

    float halfWMid = (halfW0 + halfW3) * 0.5f;
    float tScaleH  = tScale * 0.5f;

    bool hit = false;
    float tmpTSeg = 0.f;
    Vec3f tmpTang;
    float tmpH = 0.f;

    if (subdivide(frame, L0, L1, L2, L3, halfW0, halfWMid,
                  tOffset, tScaleH,
                  ray_tMin, inoutBestT, depth - 1, tmpTSeg, tmpTang, tmpH)) {
        outTSeg    = tmpTSeg;
        outTangent = tmpTang;
        outH       = tmpH;
        hit = true;
    }

    if (subdivide(frame, R0, R1, R2, R3, halfWMid, halfW3,
                  tOffset + tScaleH, tScaleH,
                  ray_tMin, inoutBestT, depth - 1, tmpTSeg, tmpTang, tmpH)) {
        outTSeg    = tmpTSeg;
        outTangent = tmpTang;
        outH       = tmpH;
        hit = true;
    }

    return hit;
}

// ---------------------------------------------------------------------------
// subdivideAny — lean occlusion variant: returns true on first hit found.
// Does NOT compute tangent, normal, or SurfaceInteraction — skips all of
// that work since shadow rays only need a yes/no answer.
// ---------------------------------------------------------------------------
static bool subdivideAny(
    const RayFrame& frame,
    Vec3f P0, Vec3f P1, Vec3f P2, Vec3f P3,
    float halfW0, float halfW3,
    float ray_tMin, float ray_tMax,
    int   depth)
{
    Vec3f cp0 = frame.project(P0);
    Vec3f cp1 = frame.project(P1);
    Vec3f cp2 = frame.project(P2);
    Vec3f cp3 = frame.project(P3);

    float maxHalfW = std::max(halfW0, halfW3);

    float zMin = std::min({cp0.z, cp1.z, cp2.z, cp3.z});
    float zMax = std::max({cp0.z, cp1.z, cp2.z, cp3.z});
    if (zMin - maxHalfW > ray_tMax) return false;
    if (zMax + maxHalfW < ray_tMin) return false;

    float xMin = std::min({cp0.x, cp1.x, cp2.x, cp3.x});
    float xMax = std::max({cp0.x, cp1.x, cp2.x, cp3.x});
    float yMin = std::min({cp0.y, cp1.y, cp2.y, cp3.y});
    float yMax = std::max({cp0.y, cp1.y, cp2.y, cp3.y});
    float cx = (0.f < xMin) ? xMin : (0.f > xMax ? xMax : 0.f);
    float cy = (0.f < yMin) ? yMin : (0.f > yMax ? yMax : 0.f);
    if (cx*cx + cy*cy > maxHalfW*maxHalfW) return false;

    if (depth == 0) {
        float ax = cp0.x, ay = cp0.y;
        float bx = cp3.x, by = cp3.y;
        float dx = bx - ax, dy = by - ay;
        float lenSq = dx*dx + dy*dy;
        float tLine = (lenSq < 1e-12f) ? 0.f
                    : std::max(0.f, std::min(1.f, (-ax*dx - ay*dy) / lenSq));
        float px = ax + dx*tLine, py = ay + dy*tLine;
        float halfW = halfW0*(1.f-tLine) + halfW3*tLine;
        if (px*px + py*py > halfW*halfW) return false;
        float tRay = cp0.z + (cp3.z - cp0.z)*tLine;
        return tRay >= ray_tMin && tRay < ray_tMax;
    }

    Vec3f L0, L1, L2, L3, R0, R1, R2, R3;
    bezierSplitMid(P0, P1, P2, P3, L0, L1, L2, L3, R0, R1, R2, R3);
    float halfWMid = (halfW0 + halfW3) * 0.5f;

    return subdivideAny(frame, L0, L1, L2, L3, halfW0, halfWMid, ray_tMin, ray_tMax, depth-1)
        || subdivideAny(frame, R0, R1, R2, R3, halfWMid, halfW3, ray_tMin, ray_tMax, depth-1);
}

// Lean wrapper: one segment, occlusion only.
static bool intersectOneSegmentAny(
    const RayFrame&   frame,
    const Ray&        ray,
    const StrandDesc& strand,
    uint32_t          segIdx)
{
    const uint32_t N = strand.numSegments();
    if (segIdx >= N) return false;

    uint32_t base = segIdx * 3;
    Vec3f P0 = strand.controlPoints[base + 0];
    Vec3f P1 = strand.controlPoints[base + 1];
    Vec3f P2 = strand.controlPoints[base + 2];
    Vec3f P3 = strand.controlPoints[base + 3];

    if (strand.hasMotion()) {
        float t = ray.time, mt = 1.f - t;
        P0 = P0*mt + strand.controlPointsClose[base+0]*t;
        P1 = P1*mt + strand.controlPointsClose[base+1]*t;
        P2 = P2*mt + strand.controlPointsClose[base+2]*t;
        P3 = P3*mt + strand.controlPointsClose[base+3]*t;
    }

    float v0 = float(segIdx)     / float(N);
    float v3 = float(segIdx + 1) / float(N);
    return subdivideAny(frame, P0, P1, P2, P3,
                        strand.widthAt(v0) * 0.5f,
                        strand.widthAt(v3) * 0.5f,
                        ray.tMin, ray.tMax, 6);
}

// ---------------------------------------------------------------------------
// Intersect one segment (by index) of a strand against a ray.
// The RayFrame is pre-computed once per ray by the caller.
// ---------------------------------------------------------------------------
static bool intersectOneSegment(
    const RayFrame&      frame,
    const Ray&           ray,
    const StrandDesc&    strand,
    uint32_t             segIdx,
    uint32_t             strandID,
    float&               inoutBestT,
    SurfaceInteraction&  si)
{
    const uint32_t N = strand.numSegments();
    if (segIdx >= N) return false;

    const int MAX_DEPTH = 6;
    uint32_t base = segIdx * 3;
    Vec3f P0 = strand.controlPoints[base + 0];
    Vec3f P1 = strand.controlPoints[base + 1];
    Vec3f P2 = strand.controlPoints[base + 2];
    Vec3f P3 = strand.controlPoints[base + 3];

    // Interpolate control points at ray time for motion blur.
    if (strand.hasMotion()) {
        float t  = ray.time;
        float mt = 1.f - t;
        P0 = P0 * mt + strand.controlPointsClose[base + 0] * t;
        P1 = P1 * mt + strand.controlPointsClose[base + 1] * t;
        P2 = P2 * mt + strand.controlPointsClose[base + 2] * t;
        P3 = P3 * mt + strand.controlPointsClose[base + 3] * t;
    }

    float v0     = float(segIdx)     / float(N);
    float v3     = float(segIdx + 1) / float(N);
    float halfW0 = strand.widthAt(v0) * 0.5f;
    float halfW3 = strand.widthAt(v3) * 0.5f;

    float tSeg = 0.f;
    Vec3f tang;
    float h = 0.f;

    if (!subdivide(frame, P0, P1, P2, P3,
                   halfW0, halfW3,
                   v0, v3 - v0,
                   ray.tMin, inoutBestT, MAX_DEPTH,
                   tSeg, tang, h))
        return false;

    Vec3f hitPoint = ray.at(inoutBestT);
    Vec3f negDir   = -ray.direction;
    Vec3f normal   = negDir - tang * dot(negDir, tang);
    normal = safeNormalize(normal, frame.xAxis);

    si.p        = hitPoint;
    si.n        = normal;
    si.ng       = normal;
    si.dpdu     = tang;
    si.dpdv     = cross(tang, normal);
    si.uv       = strand.rootUV;   // root UV on emitter mesh (scalp position)
    si.strandV  = tSeg;            // parametric position along strand (root=0, tip=1)
    si.t        = inoutBestT;
    si.meshID   = strand.materialIndex;
    si.primID   = segIdx;
    si.strandID = strandID;
    si.isCurve  = true;
    si.h        = h;
    si.color    = strand.color;
    return true;
}

// ---------------------------------------------------------------------------
// BVH build — SAH over curve segment AABBs.
// ---------------------------------------------------------------------------

struct SegWork {
    SegRef ref;
    float  bmin[3], bmax[3];
    float  centroid[3];
};

static uint32_t buildCurveBVH(
    std::vector<SegWork>&   work,
    std::vector<SegRef>&    outRefs,
    std::vector<CurveNode>& nodes,
    uint32_t start, uint32_t end)
{
    uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back({});

    // Node AABB
    float bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i)
        boxUnion(bmin, bmax, work[i].bmin, work[i].bmax);

    uint32_t count  = end - start;
    float    leafCost = kCurveIntersectCost * (float)count;

    auto makeLeaf = [&]() {
        uint32_t firstPrim = (uint32_t)outRefs.size();
        for (uint32_t i = start; i < end; ++i) outRefs.push_back(work[i].ref);
        for (int k = 0; k < 3; ++k) { nodes[nodeIdx].bmin[k] = bmin[k]; nodes[nodeIdx].bmax[k] = bmax[k]; }
        nodes[nodeIdx].left_or_prim   = firstPrim;
        nodes[nodeIdx].right_or_count = count | 0x80000000u;
    };

    if (count <= (uint32_t)kCurveMaxLeaf) { makeLeaf(); return nodeIdx; }

    // Centroid AABB for bucket placement
    float cmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float cmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i)
        for (int k = 0; k < 3; ++k) {
            cmin[k] = std::min(cmin[k], work[i].centroid[k]);
            cmax[k] = std::max(cmax[k], work[i].centroid[k]);
        }

    float parentSA = boxArea(bmin, bmax);
    float bestCost = leafCost;
    int   bestAxis = -1, bestBucket = -1;

    if (parentSA > 1e-12f) {
        struct Bucket {
            float bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
            float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            uint32_t count = 0;
        };

        for (int axis = 0; axis < 3; ++axis) {
            float range = cmax[axis] - cmin[axis];
            if (range < 1e-7f) continue;

            Bucket buckets[kCurveSAHBuckets];
            for (uint32_t i = start; i < end; ++i) {
                int b = (int)(kCurveSAHBuckets * ((work[i].centroid[axis] - cmin[axis]) / range));
                b = std::clamp(b, 0, kCurveSAHBuckets - 1);
                buckets[b].count++;
                boxUnion(buckets[b].bmin, buckets[b].bmax, work[i].bmin, work[i].bmax);
            }

            // Prefix scan (left side)
            float    lbMin[kCurveSAHBuckets-1][3], lbMax[kCurveSAHBuckets-1][3];
            uint32_t lCount[kCurveSAHBuckets-1];
            {
                float lb[3] = { FLT_MAX, FLT_MAX, FLT_MAX };
                float ub[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                uint32_t lc = 0;
                for (int i = 0; i < kCurveSAHBuckets - 1; ++i) {
                    boxUnion(lb, ub, buckets[i].bmin, buckets[i].bmax);
                    lc += buckets[i].count;
                    for (int k = 0; k < 3; ++k) { lbMin[i][k] = lb[k]; lbMax[i][k] = ub[k]; }
                    lCount[i] = lc;
                }
            }

            // Suffix scan + evaluate cost
            float    rb[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
            float    ru[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            uint32_t rc = 0;
            for (int i = kCurveSAHBuckets - 2; i >= 0; --i) {
                boxUnion(rb, ru, buckets[i+1].bmin, buckets[i+1].bmax);
                rc += buckets[i+1].count;
                if (lCount[i] == 0 || rc == 0) continue;
                float cost = kCurveTraversalCost +
                    kCurveIntersectCost * (boxArea(lbMin[i], lbMax[i]) * lCount[i]
                                         + boxArea(rb, ru)              * rc) / parentSA;
                if (cost < bestCost) { bestCost = cost; bestAxis = axis; bestBucket = i; }
            }
        }
    }

    if (bestAxis < 0) { makeLeaf(); return nodeIdx; }

    float range = cmax[bestAxis] - cmin[bestAxis];
    auto midIt = std::partition(
        work.begin() + start, work.begin() + end,
        [&](const SegWork& sw) {
            int b = (int)(kCurveSAHBuckets * ((sw.centroid[bestAxis] - cmin[bestAxis]) / range));
            return std::clamp(b, 0, kCurveSAHBuckets-1) <= bestBucket;
        });
    uint32_t mid = (uint32_t)(midIt - work.begin());
    if (mid == start || mid == end) mid = (start + end) / 2;

    uint32_t leftIdx  = buildCurveBVH(work, outRefs, nodes, start, mid);
    uint32_t rightIdx = buildCurveBVH(work, outRefs, nodes, mid,   end);

    for (int k = 0; k < 3; ++k) { nodes[nodeIdx].bmin[k] = bmin[k]; nodes[nodeIdx].bmax[k] = bmax[k]; }
    nodes[nodeIdx].left_or_prim   = leftIdx;
    nodes[nodeIdx].right_or_count = rightIdx;
    return nodeIdx;
}

}  // anonymous namespace

// ===========================================================================
// CurveBrute implementation
// ===========================================================================

CurveBrute::CurveBrute(const GeometryPool& triPool, const CurvePool& curvePool)
    : m_triBvh(triPool), m_curvePool(curvePool)
{}

void CurveBrute::commit() {
    m_triBvh.commit();
    m_curveNodes.clear();
    m_segRefs.clear();

    const uint32_t S = (uint32_t)m_curvePool.numStrands();
    if (S == 0) return;

    // Enumerate all segments across all strands and compute per-segment AABBs.
    std::vector<SegWork> work;
    {
        size_t totalSegs = 0;
        for (uint32_t si = 0; si < S; ++si)
            totalSegs += m_curvePool.strand(si).numSegments();
        work.reserve(totalSegs);
    }

    for (uint32_t si = 0; si < S; ++si) {
        const StrandDesc& strand = m_curvePool.strand(si);
        const uint32_t    N     = strand.numSegments();

        const bool strandHasMotion = strand.hasMotion();

        for (uint32_t seg = 0; seg < N; ++seg) {
            uint32_t base = seg * 3;
            const Vec3f& P0 = strand.controlPoints[base + 0];
            const Vec3f& P1 = strand.controlPoints[base + 1];
            const Vec3f& P2 = strand.controlPoints[base + 2];
            const Vec3f& P3 = strand.controlPoints[base + 3];

            float v0       = float(seg)     / float(N);
            float v3       = float(seg + 1) / float(N);
            // Max tube radius for this segment (half the Alembic width/diameter).
            float maxHalfW = std::max(strand.widthAt(v0), strand.widthAt(v3)) * 0.5f;

            SegWork sw;
            sw.ref = { si, seg };

            // Conservative AABB: convex hull of CVs ± tube radius.
            // For motion blur, union open and close CV positions.
            float xMin = std::min({P0.x, P1.x, P2.x, P3.x});
            float xMax = std::max({P0.x, P1.x, P2.x, P3.x});
            float yMin = std::min({P0.y, P1.y, P2.y, P3.y});
            float yMax = std::max({P0.y, P1.y, P2.y, P3.y});
            float zMin = std::min({P0.z, P1.z, P2.z, P3.z});
            float zMax = std::max({P0.z, P1.z, P2.z, P3.z});

            if (strandHasMotion) {
                const Vec3f& C0 = strand.controlPointsClose[base + 0];
                const Vec3f& C1 = strand.controlPointsClose[base + 1];
                const Vec3f& C2 = strand.controlPointsClose[base + 2];
                const Vec3f& C3 = strand.controlPointsClose[base + 3];
                xMin = std::min({xMin, C0.x, C1.x, C2.x, C3.x});
                xMax = std::max({xMax, C0.x, C1.x, C2.x, C3.x});
                yMin = std::min({yMin, C0.y, C1.y, C2.y, C3.y});
                yMax = std::max({yMax, C0.y, C1.y, C2.y, C3.y});
                zMin = std::min({zMin, C0.z, C1.z, C2.z, C3.z});
                zMax = std::max({zMax, C0.z, C1.z, C2.z, C3.z});
            }

            sw.bmin[0] = xMin - maxHalfW;
            sw.bmin[1] = yMin - maxHalfW;
            sw.bmin[2] = zMin - maxHalfW;
            sw.bmax[0] = xMax + maxHalfW;
            sw.bmax[1] = yMax + maxHalfW;
            sw.bmax[2] = zMax + maxHalfW;

            sw.centroid[0] = (sw.bmin[0] + sw.bmax[0]) * 0.5f;
            sw.centroid[1] = (sw.bmin[1] + sw.bmax[1]) * 0.5f;
            sw.centroid[2] = (sw.bmin[2] + sw.bmax[2]) * 0.5f;

            work.push_back(sw);
        }
    }

    if (work.empty()) return;

    uint32_t estLeaves = ((uint32_t)work.size() + kCurveMaxLeaf - 1) / kCurveMaxLeaf;
    m_curveNodes.reserve(2 * estLeaves + 4);
    m_segRefs.reserve(work.size());

    buildCurveBVH(work, m_segRefs, m_curveNodes, 0, (uint32_t)work.size());

    spdlog::info("CurveBVH: {} strands, {} segments → {} nodes",
                 S, work.size(), m_curveNodes.size());
}

TraceResult CurveBrute::trace(const Ray& ray) const {
    // Triangle BVH first.
    TraceResult result = m_triBvh.trace(ray);
    float bestT = result.hit ? result.si.t : ray.tMax;

    if (m_curveNodes.empty()) return result;

    RayFrame      frame = makeRayFrame(ray);
    RayTraversal  rt    = RayTraversal::make(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const CurveNode& node = m_curveNodes[stack[--top]];

        if (!aabbHit(node.bmin, node.bmax, rt, bestT)) continue;

        if (node.isLeaf()) {
            uint32_t end = node.left_or_prim + node.primCount();
            for (uint32_t i = node.left_or_prim; i < end; ++i) {
                const SegRef& r = m_segRefs[i];
                if (r.strandIdx == ray.skipStrandID) continue;  // self-intersection avoidance
                SurfaceInteraction curveHit;
                if (intersectOneSegment(frame, ray,
                                        m_curvePool.strand(r.strandIdx),
                                        r.segIdx, r.strandIdx,
                                        bestT, curveHit)) {
                    result.hit = true;
                    result.si  = curveHit;
                }
            }
        } else {
            // Push both children; left is processed next (LIFO).
            stack[top++] = node.right_or_count;  // right child
            stack[top++] = node.left_or_prim;    // left child
        }
    }

    return result;
}

bool CurveBrute::occluded(const Ray& ray) const {
    if (m_triBvh.occluded(ray)) return true;
    if (m_curveNodes.empty()) return false;

    RayFrame     frame = makeRayFrame(ray);
    RayTraversal rt    = RayTraversal::make(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const CurveNode& node = m_curveNodes[stack[--top]];

        if (!aabbHit(node.bmin, node.bmax, rt, ray.tMax)) continue;

        if (node.isLeaf()) {
            uint32_t end = node.left_or_prim + node.primCount();
            for (uint32_t i = node.left_or_prim; i < end; ++i) {
                const SegRef& r = m_segRefs[i];
                if (r.strandIdx == ray.skipStrandID) continue;
                if (intersectOneSegmentAny(frame, ray,
                                           m_curvePool.strand(r.strandIdx),
                                           r.segIdx))
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
