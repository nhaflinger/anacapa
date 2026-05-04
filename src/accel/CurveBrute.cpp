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
    Vec3f& outTangent)
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
        return true;
    }

    Vec3f L0, L1, L2, L3, R0, R1, R2, R3;
    bezierSplitMid(P0, P1, P2, P3, L0, L1, L2, L3, R0, R1, R2, R3);

    float halfWMid = (halfW0 + halfW3) * 0.5f;
    float tScaleH  = tScale * 0.5f;

    bool hit = false;
    float tmpTSeg = 0.f;
    Vec3f tmpTang;

    if (subdivide(frame, L0, L1, L2, L3, halfW0, halfWMid,
                  tOffset, tScaleH,
                  ray_tMin, inoutBestT, depth - 1, tmpTSeg, tmpTang)) {
        outTSeg    = tmpTSeg;
        outTangent = tmpTang;
        hit = true;
    }

    if (subdivide(frame, R0, R1, R2, R3, halfWMid, halfW3,
                  tOffset + tScaleH, tScaleH,
                  ray_tMin, inoutBestT, depth - 1, tmpTSeg, tmpTang)) {
        outTSeg    = tmpTSeg;
        outTangent = tmpTang;
        hit = true;
    }

    return hit;
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

    float v0     = float(segIdx)     / float(N);
    float v3     = float(segIdx + 1) / float(N);
    float halfW0 = strand.widthAt(v0) * 0.5f;
    float halfW3 = strand.widthAt(v3) * 0.5f;

    float tSeg = 0.f;
    Vec3f tang;

    if (!subdivide(frame, P0, P1, P2, P3,
                   halfW0, halfW3,
                   v0, v3 - v0,
                   ray.tMin, inoutBestT, MAX_DEPTH,
                   tSeg, tang))
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
    si.uv       = { 0.f, tSeg };
    si.t        = inoutBestT;
    si.meshID   = strand.materialIndex;
    si.primID   = segIdx;
    si.strandID = strandID;
    si.isCurve  = true;
    si.color    = strand.color;
    return true;
}

// ---------------------------------------------------------------------------
// BVH build — top-down median-split over curve segment AABBs.
// ---------------------------------------------------------------------------

struct SegWork {
    SegRef ref;
    float  bmin[3], bmax[3];
    float  centroid[3];
};

// Returns index of the newly created node.
static uint32_t buildCurveBVH(
    std::vector<SegWork>&   work,
    std::vector<SegRef>&    outRefs,
    std::vector<CurveNode>& nodes,
    uint32_t start, uint32_t end,
    uint32_t maxLeaf)
{
    // Reserve a slot — written by index after recursive calls to stay valid
    // through any vector reallocations caused by child push_backs.
    uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back({});

    // Compute node AABB as union of all segment AABBs in [start, end).
    float bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (work[i].bmin[k] < bmin[k]) bmin[k] = work[i].bmin[k];
            if (work[i].bmax[k] > bmax[k]) bmax[k] = work[i].bmax[k];
        }
    }

    uint32_t count = end - start;

    if (count <= maxLeaf) {
        // Leaf: append prims to outRefs.
        uint32_t firstPrim = (uint32_t)outRefs.size();
        for (uint32_t i = start; i < end; ++i)
            outRefs.push_back(work[i].ref);

        for (int k = 0; k < 3; ++k) {
            nodes[nodeIdx].bmin[k] = bmin[k];
            nodes[nodeIdx].bmax[k] = bmax[k];
        }
        nodes[nodeIdx].left_or_prim   = firstPrim;
        nodes[nodeIdx].right_or_count = count | 0x80000000u;
        return nodeIdx;
    }

    // Interior: find longest axis of centroid AABB and split at median.
    float cmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float cmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i) {
        for (int k = 0; k < 3; ++k) {
            if (work[i].centroid[k] < cmin[k]) cmin[k] = work[i].centroid[k];
            if (work[i].centroid[k] > cmax[k]) cmax[k] = work[i].centroid[k];
        }
    }
    int axis = 0;
    float ext[3] = { cmax[0]-cmin[0], cmax[1]-cmin[1], cmax[2]-cmin[2] };
    if (ext[1] > ext[0])    axis = 1;
    if (ext[2] > ext[axis]) axis = 2;

    uint32_t mid = (start + end) / 2;
    std::nth_element(
        work.begin() + start,
        work.begin() + mid,
        work.begin() + end,
        [axis](const SegWork& a, const SegWork& b) {
            return a.centroid[axis] < b.centroid[axis];
        });

    uint32_t leftIdx  = buildCurveBVH(work, outRefs, nodes, start, mid, maxLeaf);
    uint32_t rightIdx = buildCurveBVH(work, outRefs, nodes, mid,   end, maxLeaf);

    // Write interior node — safe to index now that all push_backs are done.
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
            sw.bmin[0] = std::min({P0.x, P1.x, P2.x, P3.x}) - maxHalfW;
            sw.bmin[1] = std::min({P0.y, P1.y, P2.y, P3.y}) - maxHalfW;
            sw.bmin[2] = std::min({P0.z, P1.z, P2.z, P3.z}) - maxHalfW;
            sw.bmax[0] = std::max({P0.x, P1.x, P2.x, P3.x}) + maxHalfW;
            sw.bmax[1] = std::max({P0.y, P1.y, P2.y, P3.y}) + maxHalfW;
            sw.bmax[2] = std::max({P0.z, P1.z, P2.z, P3.z}) + maxHalfW;

            sw.centroid[0] = (sw.bmin[0] + sw.bmax[0]) * 0.5f;
            sw.centroid[1] = (sw.bmin[1] + sw.bmax[1]) * 0.5f;
            sw.centroid[2] = (sw.bmin[2] + sw.bmax[2]) * 0.5f;

            work.push_back(sw);
        }
    }

    if (work.empty()) return;

    constexpr uint32_t MAX_LEAF = 4;
    // Upper bound on node count: 2 * num_leaves, leaves ≈ work.size() / MAX_LEAF.
    uint32_t estLeaves = ((uint32_t)work.size() + MAX_LEAF - 1) / MAX_LEAF;
    m_curveNodes.reserve(2 * estLeaves + 4);
    m_segRefs.reserve(work.size());

    buildCurveBVH(work, m_segRefs, m_curveNodes, 0, (uint32_t)work.size(), MAX_LEAF);

    spdlog::info("CurveBVH: {} strands, {} segments → {} nodes",
                 S, work.size(), m_curveNodes.size());
}

TraceResult CurveBrute::trace(const Ray& ray) const {
    // Triangle BVH first.
    TraceResult result = m_triBvh.trace(ray);
    float bestT = result.hit ? result.si.t : ray.tMax;

    if (m_curveNodes.empty()) return result;

    // Build ray-space frame once for all segment tests.
    RayFrame frame = makeRayFrame(ray);

    // Iterative BVH traversal with explicit stack.
    // Stack depth: BVH height ≈ log2(totalSegs / MAX_LEAF) — 64 is ample.
    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;  // root

    while (top > 0) {
        const CurveNode& node = m_curveNodes[stack[--top]];

        if (!aabbHit(node.bmin, node.bmax, ray, bestT)) continue;

        if (node.isLeaf()) {
            uint32_t end = node.left_or_prim + node.primCount();
            for (uint32_t i = node.left_or_prim; i < end; ++i) {
                const SegRef& r = m_segRefs[i];
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

    RayFrame frame = makeRayFrame(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        const CurveNode& node = m_curveNodes[stack[--top]];

        // For occlusion the tMax bound doesn't shrink, so pass ray.tMax.
        if (!aabbHit(node.bmin, node.bmax, ray, ray.tMax)) continue;

        if (node.isLeaf()) {
            uint32_t end = node.left_or_prim + node.primCount();
            for (uint32_t i = node.left_or_prim; i < end; ++i) {
                const SegRef& r = m_segRefs[i];
                float t = ray.tMax;
                SurfaceInteraction dummy;
                if (intersectOneSegment(frame, ray,
                                        m_curvePool.strand(r.strandIdx),
                                        r.segIdx, r.strandIdx,
                                        t, dummy))
                    return true;
            }
        } else {
            stack[top++] = node.right_or_count;
            stack[top++] = node.left_or_prim;
        }
    }
    return false;
}

bool CurveBrute::aabbHit(const float bmin[3], const float bmax[3],
                          const Ray& ray, float maxT) {
    float tMin = ray.tMin;
    float tMax = maxT;

    for (int i = 0; i < 3; ++i) {
        float d = ray.direction[i];
        if (std::abs(d) < 1e-9f) {
            if (ray.origin[i] < bmin[i] || ray.origin[i] > bmax[i]) return false;
            continue;
        }
        float invD = 1.f / d;
        float t0 = (bmin[i] - ray.origin[i]) * invD;
        float t1 = (bmax[i] - ray.origin[i]) * invD;
        if (invD < 0.f) std::swap(t0, t1);
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMax < tMin) return false;
    }
    return true;
}

}  // namespace anacapa
