#include "HaloAccel.h"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace anacapa {

// ===========================================================================
// BVH traversal helpers  (identical pattern to CurveBrute hair traversal)
// ===========================================================================

struct HaloRay {
    float ox, oy, oz;
    float invDx, invDy, invDz;
    float tMin;

    static HaloRay make(const Ray& ray) {
        auto safe = [](float v) { return std::abs(v) > 1e-9f ? v : 1e-9f; };
        return { ray.origin.x, ray.origin.y, ray.origin.z,
                 1.f / safe(ray.direction.x),
                 1.f / safe(ray.direction.y),
                 1.f / safe(ray.direction.z),
                 ray.tMin };
    }
};

static bool aabbHit(const float bmin[3], const float bmax[3],
                    const HaloRay& r, float tMax)
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

// ---------------------------------------------------------------------------
// Camera-facing disc intersection.
//
// The disc normal is N = normalize(rayOrigin - center).  The disc lies in
// the plane:  dot(N, P - center) = 0.
//
// Substituting ray P = O + t*D:
//   t = dot(N, center - O) / dot(N, D)
//
// Accept if t in [tMin, bestT), hit point within radius.
// Returns true and updates bestT / outSI on success.
// ---------------------------------------------------------------------------
static bool intersectHaloDisc(const HaloDesc& h, const Ray& ray,
                               float tMin, float& bestT,
                               SurfaceInteraction& outSI)
{
    Vec3f toOrigin = ray.origin - h.center;
    float lenSq = dot(toOrigin, toOrigin);
    if (lenSq < 1e-12f) return false;  // ray origin at particle center
    float invLen = 1.f / std::sqrt(lenSq);
    Vec3f N = toOrigin * invLen;        // disc normal toward ray origin

    float denom = dot(N, ray.direction);
    if (std::abs(denom) < 1e-6f) return false;  // ray nearly parallel to disc

    float t = dot(N, h.center - ray.origin) / denom;
    if (t < tMin || t >= bestT) return false;

    Vec3f hitPt = ray.origin + ray.direction * t;
    Vec3f delta = hitPt - h.center;
    float distSq = dot(delta, delta);
    if (distSq > h.radius * h.radius) return false;

    // Build tangent frame for UV and dpdu/dpdv.
    // Pick a stable up vector orthogonal to N.
    Vec3f up = (std::abs(N.y) < 0.99f) ? Vec3f{0.f, 1.f, 0.f}
                                        : Vec3f{1.f, 0.f, 0.f};
    Vec3f T = normalize(cross(up, N));
    Vec3f B = cross(N, T);

    float du = dot(delta, T);
    float dv = dot(delta, B);
    float dist = std::sqrt(distSq);

    bestT = t;
    outSI.p   = hitPt;
    outSI.n   = N;
    outSI.ng  = N;
    // UV: u = normalized radius [0,1], v = polar angle [0,1]
    outSI.uv  = { dist / h.radius,
                  (std::atan2(dv, du) + 3.14159265358979323f) * (1.f / (2.f * 3.14159265358979323f)) };
    outSI.dpdu = T * h.radius;
    outSI.dpdv = B * h.radius;
    outSI.t      = t;
    outSI.meshID = h.matIdx;   // meshID drives scene.materials[] lookup
    outSI.primID = ~0u;
    outSI.color  = h.color;
    outSI.isHalo = true;
    return true;
}

// Occlusion-only variant — no SurfaceInteraction fill.
static bool intersectHaloDiscAny(const HaloDesc& h, const Ray& ray,
                                  float tMin, float tMax)
{
    Vec3f toOrigin = ray.origin - h.center;
    float lenSq = dot(toOrigin, toOrigin);
    if (lenSq < 1e-12f) return false;
    Vec3f N = toOrigin * (1.f / std::sqrt(lenSq));

    float denom = dot(N, ray.direction);
    if (std::abs(denom) < 1e-6f) return false;

    float t = dot(N, h.center - ray.origin) / denom;
    if (t < tMin || t >= tMax) return false;

    Vec3f hitPt = ray.origin + ray.direction * t;
    Vec3f delta = hitPt - h.center;
    return dot(delta, delta) <= h.radius * h.radius;
}

// ===========================================================================
// BVH build (SAH, binary — same algorithm as CurveBrute hair BVH)
// ===========================================================================

static constexpr int   kHaloSAHBuckets    = 12;
static constexpr int   kHaloMaxLeaf       = 4;
static constexpr float kHaloTraversalCost = 1.f;
static constexpr float kHaloIntersectCost = 1.f;

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

struct HaloWork {
    uint32_t haloIdx;
    float    bmin[3], bmax[3];
    float    centroid[3];
};

static uint32_t buildHaloBVH(std::vector<HaloWork>&  work,
                              std::vector<uint32_t>&  outPrimIdx,
                              std::vector<HaloNode>&  nodes,
                              uint32_t start, uint32_t end)
{
    uint32_t nodeIdx = (uint32_t)nodes.size();
    nodes.push_back({});

    float bmin[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
    float bmax[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
    for (uint32_t i = start; i < end; ++i)
        boxUnion(bmin, bmax, work[i].bmin, work[i].bmax);

    uint32_t count    = end - start;
    float    leafCost = kHaloIntersectCost * (float)count;

    auto makeLeaf = [&]() {
        uint32_t first = (uint32_t)outPrimIdx.size();
        for (uint32_t i = start; i < end; ++i)
            outPrimIdx.push_back(work[i].haloIdx);
        for (int k = 0; k < 3; ++k) {
            nodes[nodeIdx].bmin[k] = bmin[k];
            nodes[nodeIdx].bmax[k] = bmax[k];
        }
        nodes[nodeIdx].left_or_prim   = first;
        nodes[nodeIdx].right_or_count = count | 0x80000000u;
    };

    if (count <= (uint32_t)kHaloMaxLeaf) { makeLeaf(); return nodeIdx; }

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

            Bucket buckets[kHaloSAHBuckets];
            for (uint32_t i = start; i < end; ++i) {
                int b = (int)(kHaloSAHBuckets * ((work[i].centroid[axis] - cmin[axis]) / range));
                b = std::clamp(b, 0, kHaloSAHBuckets - 1);
                buckets[b].count++;
                boxUnion(buckets[b].bmin, buckets[b].bmax, work[i].bmin, work[i].bmax);
            }

            float    lbMin[kHaloSAHBuckets-1][3], lbMax[kHaloSAHBuckets-1][3];
            uint32_t lCount[kHaloSAHBuckets-1];
            {
                float lb[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
                float ub[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
                uint32_t lc = 0;
                for (int i = 0; i < kHaloSAHBuckets - 1; ++i) {
                    boxUnion(lb, ub, buckets[i].bmin, buckets[i].bmax);
                    lc += buckets[i].count;
                    for (int k = 0; k < 3; ++k) { lbMin[i][k] = lb[k]; lbMax[i][k] = ub[k]; }
                    lCount[i] = lc;
                }
            }

            float    rb[3] = {  FLT_MAX,  FLT_MAX,  FLT_MAX };
            float    ru[3] = { -FLT_MAX, -FLT_MAX, -FLT_MAX };
            uint32_t rc = 0;
            for (int i = kHaloSAHBuckets - 2; i >= 0; --i) {
                boxUnion(rb, ru, buckets[i+1].bmin, buckets[i+1].bmax);
                rc += buckets[i+1].count;
                if (lCount[i] == 0 || rc == 0) continue;
                float cost = kHaloTraversalCost
                           + kHaloIntersectCost
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
        [&](const HaloWork& w) {
            int b = (int)(kHaloSAHBuckets * ((w.centroid[bestAxis] - cmin[bestAxis]) / range));
            return std::clamp(b, 0, kHaloSAHBuckets - 1) <= bestBucket;
        });
    uint32_t mid = (uint32_t)(midIt - work.begin());
    if (mid == start || mid == end) mid = (start + end) / 2;

    uint32_t leftIdx  = buildHaloBVH(work, outPrimIdx, nodes, start, mid);
    uint32_t rightIdx = buildHaloBVH(work, outPrimIdx, nodes, mid,   end);

    for (int k = 0; k < 3; ++k) {
        nodes[nodeIdx].bmin[k] = bmin[k];
        nodes[nodeIdx].bmax[k] = bmax[k];
    }
    nodes[nodeIdx].left_or_prim   = leftIdx;
    nodes[nodeIdx].right_or_count = rightIdx;
    return nodeIdx;
}

// ===========================================================================
// HaloAccel public interface
// ===========================================================================

HaloAccel::HaloAccel(const HaloPool& pool) : m_pool(pool) {}

void HaloAccel::commit()
{
    const auto& halos = m_pool.halos();
    if (halos.empty()) return;

    std::vector<HaloWork> work;
    work.reserve(halos.size());
    for (uint32_t i = 0; i < (uint32_t)halos.size(); ++i) {
        const HaloDesc& h = halos[i];
        HaloWork w;
        w.haloIdx = i;
        w.bmin[0] = h.center.x - h.radius;  w.bmax[0] = h.center.x + h.radius;
        w.bmin[1] = h.center.y - h.radius;  w.bmax[1] = h.center.y + h.radius;
        w.bmin[2] = h.center.z - h.radius;  w.bmax[2] = h.center.z + h.radius;
        w.centroid[0] = h.center.x;
        w.centroid[1] = h.center.y;
        w.centroid[2] = h.center.z;
        work.push_back(w);
    }

    m_nodes.clear();
    m_primIdx.clear();
    m_nodes.reserve(2 * halos.size());
    m_primIdx.reserve(halos.size());
    buildHaloBVH(work, m_primIdx, m_nodes, 0, (uint32_t)work.size());
}

TraceResult HaloAccel::trace(const Ray& ray) const
{
    if (m_nodes.empty()) return {};

    TraceResult result;
    float bestT = ray.tMax;
    SurfaceInteraction best;

    HaloRay hr = HaloRay::make(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        uint32_t       ni   = stack[--top];
        const HaloNode& node = m_nodes[ni];

        if (!aabbHit(node.bmin, node.bmax, hr, bestT)) continue;

        if (node.isLeaf()) {
            uint32_t first = node.left_or_prim;
            uint32_t count = node.primCount();
            for (uint32_t j = 0; j < count; ++j) {
                uint32_t idx = m_primIdx[first + j];
                if (intersectHaloDisc(m_pool.halo(idx), ray, ray.tMin, bestT, best)) {
                    result.hit = true;
                    result.si  = best;
                }
            }
        } else {
            stack[top++] = node.right_or_count;
            stack[top++] = node.left_or_prim;
        }
    }

    return result;
}

bool HaloAccel::occluded(const Ray& ray) const
{
    if (m_nodes.empty()) return false;

    HaloRay hr = HaloRay::make(ray);

    uint32_t stack[64];
    int top = 0;
    stack[top++] = 0;

    while (top > 0) {
        uint32_t       ni   = stack[--top];
        const HaloNode& node = m_nodes[ni];

        if (!aabbHit(node.bmin, node.bmax, hr, ray.tMax)) continue;

        if (node.isLeaf()) {
            uint32_t first = node.left_or_prim;
            uint32_t count = node.primCount();
            for (uint32_t j = 0; j < count; ++j) {
                uint32_t idx = m_primIdx[first + j];
                if (intersectHaloDiscAny(m_pool.halo(idx), ray, ray.tMin, ray.tMax))
                    return true;
            }
        } else {
            stack[top++] = node.right_or_count;
            stack[top++] = node.left_or_prim;
        }
    }

    return false;
}

} // namespace anacapa
