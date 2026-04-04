#include "BVHBackend.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>

namespace anacapa {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------
BVHBackend::BVHBackend(const GeometryPool& pool)
    : m_pool(pool)
{}

void BVHBackend::commit() {
    m_tris.clear();
    m_nodes.clear();
    m_primIndices.clear();

    for (uint32_t meshID = 0; meshID < m_pool.numMeshes(); ++meshID) {
        const MeshDesc& mesh = m_pool.mesh(meshID);
        // Static meshes have positions pre-baked to world space by the loader;
        // the identity transform is a no-op and is used only for normal transform.
        const Mat4f xfm = Mat4f::identity();

        uint32_t numTris = mesh.numTriangles();
        for (uint32_t ti = 0; ti < numTris; ++ti) {
            uint32_t i0 = mesh.indices[ti * 3 + 0];
            uint32_t i1 = mesh.indices[ti * 3 + 1];
            uint32_t i2 = mesh.indices[ti * 3 + 2];

            BVHTriangle tri;
            if (mesh.hasMotion()) {
                // Keep vertices in object space; transform at ray intersection time.
                tri.v0 = mesh.positions[i0];
                Vec3f v1 = mesh.positions[i1];
                Vec3f v2 = mesh.positions[i2];
                tri.e1 = v1 - tri.v0;
                tri.e2 = v2 - tri.v0;
                tri.n  = safeNormalize(cross(tri.e1, tri.e2));

                auto getN = [&](uint32_t idx) -> Vec3f {
                    if (idx < mesh.normals.size())
                        return safeNormalize(mesh.normals[idx]);
                    return tri.n;
                };
                tri.sn0 = getN(i0); tri.sn1 = getN(i1); tri.sn2 = getN(i2);
                tri.isObjectSpace = true;
            } else {
                tri.v0 = xfm.transformPoint(mesh.positions[i0]);
                Vec3f v1 = xfm.transformPoint(mesh.positions[i1]);
                Vec3f v2 = xfm.transformPoint(mesh.positions[i2]);
                tri.e1 = v1 - tri.v0;
                tri.e2 = v2 - tri.v0;
                tri.n  = safeNormalize(cross(tri.e1, tri.e2));

                auto getN = [&](uint32_t idx) -> Vec3f {
                    if (idx < mesh.normals.size())
                        return safeNormalize(xfm.transformNormal(mesh.normals[idx]));
                    return tri.n;
                };
                tri.sn0 = getN(i0); tri.sn1 = getN(i1); tri.sn2 = getN(i2);
                tri.isObjectSpace = false;
            }

            auto getUV = [&](uint32_t idx) -> Vec2f {
                return idx < mesh.uvs.size() ? mesh.uvs[idx] : Vec2f{0.f, 0.f};
            };
            tri.uv0 = getUV(i0); tri.uv1 = getUV(i1); tri.uv2 = getUV(i2);

            tri.meshID = meshID;
            tri.primID = ti;
            m_tris.push_back(tri);
        }
    }

    if (m_tris.empty()) { m_built = true; return; }

    uint32_t n = static_cast<uint32_t>(m_tris.size());
    std::vector<PrimInfo> primInfo(n);
    for (uint32_t i = 0; i < n; ++i) {
        const BVHTriangle& tri = m_tris[i];
        BBox3f b;
        if (tri.isObjectSpace) {
            // Time-expanded bounds: union of world-space AABB across all motion keys
            const MeshDesc& mesh = m_pool.mesh(tri.meshID);
            for (const MotionKey& key : mesh.motionKeys) {
                Vec3f w0 = key.objectToWorld.transformPoint(tri.v0);
                Vec3f w1 = key.objectToWorld.transformPoint(tri.v0 + tri.e1);
                Vec3f w2 = key.objectToWorld.transformPoint(tri.v0 + tri.e2);
                b.expand(w0); b.expand(w1); b.expand(w2);
            }
        } else {
            Vec3f v1 = tri.v0 + tri.e1, v2 = tri.v0 + tri.e2;
            b.expand(tri.v0); b.expand(v1); b.expand(v2);
        }
        // Pad degenerate flat boxes so slabs never have zero width
        for (int ax = 0; ax < 3; ++ax)
            if (b.diagonal()[ax] < 1e-7f)
                { b.pMin[ax] -= 1e-7f; b.pMax[ax] += 1e-7f; }
        primInfo[i] = { b, b.centroid(), i };
    }

    m_primIndices.resize(n);
    std::iota(m_primIndices.begin(), m_primIndices.end(), 0u);
    m_nodes.reserve(2 * n);
    buildRecursive(primInfo, 0, n);
    m_built = true;
}

// ---------------------------------------------------------------------------
// Build — recursive SAH BVH
// ---------------------------------------------------------------------------
static void storeBounds(BVHNode& node, const BBox3f& b) {
    node.boundsMin[0] = b.pMin.x; node.boundsMin[1] = b.pMin.y; node.boundsMin[2] = b.pMin.z;
    node.boundsMax[0] = b.pMax.x; node.boundsMax[1] = b.pMax.y; node.boundsMax[2] = b.pMax.z;
}

uint32_t BVHBackend::buildRecursive(std::vector<PrimInfo>& primInfo,
                                    uint32_t start, uint32_t end) {
    uint32_t nodeIdx = static_cast<uint32_t>(m_nodes.size());
    m_nodes.emplace_back();

    BBox3f bounds, centroidBounds;
    for (uint32_t i = start; i < end; ++i) {
        bounds.expand(primInfo[i].bounds);
        centroidBounds.expand(primInfo[i].centroid);
    }
    storeBounds(m_nodes[nodeIdx], bounds);

    uint32_t count = end - start;
    int bestAxis = -1, splitBucket = -1;
    if (count > static_cast<uint32_t>(kMaxLeafPrims))
        splitBucket = sahSplit(primInfo, start, end, centroidBounds, bestAxis);

    if (bestAxis < 0 || splitBucket < 0) {
        // Leaf
        uint32_t offset = static_cast<uint32_t>(m_primIndices.size());
        for (uint32_t i = start; i < end; ++i)
            m_primIndices.push_back(primInfo[i].originalIndex);
        m_nodes[nodeIdx].dataA = offset;
        m_nodes[nodeIdx].dataB = count | BVHNode::kLeafFlag;
        return nodeIdx;
    }

    // Partition
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

    // Interior node — left child is always nodeIdx+1
    buildRecursive(primInfo, start, mid);  // left child
    uint32_t rightIdx = buildRecursive(primInfo, mid, end);

    m_nodes[nodeIdx].dataA = rightIdx;
    m_nodes[nodeIdx].dataB = static_cast<uint32_t>(bestAxis);  // no kLeafFlag
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

        // Prefix left bounds
        std::array<BBox3f,   kSAHBuckets - 1> leftBounds{};
        std::array<uint32_t, kSAHBuckets - 1> leftCount{};
        BBox3f lb; uint32_t lc = 0;
        for (int i = 0; i < kSAHBuckets - 1; ++i) {
            lb.expand(buckets[i].bounds); lc += buckets[i].count;
            leftBounds[i] = lb; leftCount[i] = lc;
        }

        // Sweep right, compute SAH cost per split
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
// Traversal helpers
// ---------------------------------------------------------------------------
BVHBackend::Ray4 BVHBackend::makeRay4(const Ray& ray) {
    Ray4 r;
    r.origin = ray.origin;
    r.tMin   = ray.tMin;
    r.tMax   = ray.tMax;
    // Guard against zero-direction components
    auto safe = [](float v) { return std::abs(v) > 1e-9f ? v : 1e-9f; };
    r.invDir = { 1.f / safe(ray.direction.x),
                 1.f / safe(ray.direction.y),
                 1.f / safe(ray.direction.z) };
    r.dirNeg[0] = ray.direction.x < 0.f ? 1 : 0;
    r.dirNeg[1] = ray.direction.y < 0.f ? 1 : 0;
    r.dirNeg[2] = ray.direction.z < 0.f ? 1 : 0;
    return r;
}

bool BVHBackend::intersectAABB(const BVHNode& node, const Ray4& r, float& tNear) {
    // bounds[0]=min, bounds[1]=max; dirNeg selects the far slab face
    const float* bounds[2] = { node.boundsMin, node.boundsMax };
    float tMin = r.tMin, tMax = r.tMax;
    for (int i = 0; i < 3; ++i) {
        float t0 = (bounds[    r.dirNeg[i]][i] - r.origin[i]) * r.invDir[i];
        float t1 = (bounds[1 - r.dirNeg[i]][i] - r.origin[i]) * r.invDir[i];
        tMin = std::max(tMin, t0);
        tMax = std::min(tMax, t1);
        if (tMax <= tMin) return false;
    }
    tNear = tMin;
    return true;
}

bool BVHBackend::intersectTriangle(const BVHTriangle& tri, const Ray4& r,
                                    float& t, float& u, float& v) {
    // Möller–Trumbore — reconstruct direction from invDir
    Vec3f dir = { 1.f / r.invDir.x, 1.f / r.invDir.y, 1.f / r.invDir.z };
    Vec3f h = cross(dir, tri.e2);
    float a = dot(tri.e1, h);
    if (std::abs(a) < 1e-9f) return false;

    float f  = 1.f / a;
    Vec3f s  = r.origin - tri.v0;
    u = f * dot(s, h);
    if (u < 0.f || u > 1.f) return false;

    Vec3f q = cross(s, tri.e1);
    v = f * dot(dir, q);
    if (v < 0.f || u + v > 1.f) return false;

    t = f * dot(tri.e2, q);
    return (t >= r.tMin && t <= r.tMax);
}

void BVHBackend::fillSurfaceInteraction(const BVHTriangle& tri,
                                         float t, float u, float v,
                                         const Mat4f* worldXfm,
                                         SurfaceInteraction& si) const {
    float w = 1.f - u - v;
    si.t  = t;

    if (worldXfm) {
        // Object-space triangle — transform everything to world space
        Vec3f p_obj  = tri.v0 + tri.e1 * u + tri.e2 * v;
        Vec3f n_obj  = safeNormalize(cross(tri.e1, tri.e2));
        Vec3f sn_obj = safeNormalize(tri.sn0 * w + tri.sn1 * u + tri.sn2 * v);

        si.p  = worldXfm->transformPoint(p_obj);
        si.ng = safeNormalize(worldXfm->transformNormal(n_obj));
        si.n  = safeNormalize(worldXfm->transformNormal(sn_obj));
        si.dpdu = worldXfm->transformVector(tri.e1);
        si.dpdv = worldXfm->transformVector(tri.e2);
    } else {
        si.p  = tri.v0 + tri.e1 * u + tri.e2 * v;
        si.ng = tri.n;
        si.n  = safeNormalize(tri.sn0 * w + tri.sn1 * u + tri.sn2 * v);
        si.dpdu = tri.e1;
        si.dpdv = tri.e2;
    }

    if (dot(si.n, si.ng) < 0.f) si.n = -si.n;
    si.uv = tri.uv0 * w + tri.uv1 * u + tri.uv2 * v;
    si.meshID     = tri.meshID;
    si.primID     = tri.primID;
    si.instanceID = ~0u;
    si.material   = nullptr;
}

// ---------------------------------------------------------------------------
// trace() — nearest-hit traversal
// ---------------------------------------------------------------------------
TraceResult BVHBackend::trace(const Ray& ray) const {
    assert(m_built);
    if (m_nodes.empty()) return {};
    return traceImpl(ray);
}

BVHBackend::Ray4 BVHBackend::makeObjectSpaceRay4(const Ray& ray, const Mat4f& worldToObject) {
    Vec3f o = worldToObject.transformPoint(ray.origin);
    Vec3f d = worldToObject.transformVector(ray.direction);
    Ray objRay{o, d, ray.tMin, ray.tMax};
    return makeRay4(objRay);
}

TraceResult BVHBackend::traceImpl(const Ray& ray) const {
    Ray4 r = makeRay4(ray);
    uint32_t stack[64];
    int      top = 0;
    stack[top++] = 0;

    float    closestT = r.tMax;
    uint32_t hitIdx   = ~0u;
    float    hitU = 0.f, hitV = 0.f;
    float    tNear;

    while (top > 0) {
        uint32_t nodeIdx = stack[--top];
        const BVHNode& node = m_nodes[nodeIdx];
        if (!intersectAABB(node, r, tNear) || tNear > closestT) continue;

        if (node.isLeaf()) {
            for (uint32_t i = 0; i < node.primCount(); ++i) {
                uint32_t idx = m_primIndices[node.primOffset() + i];
                const BVHTriangle& tri = m_tris[idx];
                float t, u, v;
                if (tri.isObjectSpace) {
                    // Piecewise-linear lerp of the FORWARD transform across all motion
                    // keys, then invert — lerping inverse matrices directly is wrong
                    // when rotation is present (lerp(A^-1,B^-1) ≠ lerp(A,B)^-1).
                    const MeshDesc& mesh = m_pool.mesh(tri.meshID);
                    Mat4f o2w = mesh.interpolateO2W(ray.time);
                    Mat4f w2o = o2w.inverse();
                    Ray4 ro = makeObjectSpaceRay4(ray, w2o);
                    ro.tMax = closestT;
                    if (intersectTriangle(tri, ro, t, u, v)) {
                        closestT = t; hitIdx = idx; hitU = u; hitV = v;
                    }
                } else {
                    Ray4 rc = r; rc.tMax = closestT;
                    if (intersectTriangle(tri, rc, t, u, v)) {
                        closestT = t; hitIdx = idx; hitU = u; hitV = v;
                    }
                }
            }
        } else {
            uint32_t left  = nodeIdx + 1;
            uint32_t right = node.rightChild();
            float tL = std::numeric_limits<float>::infinity();
            float tR = std::numeric_limits<float>::infinity();
            bool hL = intersectAABB(m_nodes[left],  r, tL);
            bool hR = intersectAABB(m_nodes[right], r, tR);
            // Push farther child first so nearer child is processed first
            if (hL && hR) {
                if (tL < tR) { stack[top++] = right; stack[top++] = left; }
                else         { stack[top++] = left;  stack[top++] = right; }
            } else if (hL) { stack[top++] = left; }
            else if (hR)   { stack[top++] = right; }
        }
    }

    TraceResult result;
    if (hitIdx != ~0u) {
        result.hit = true;
        const BVHTriangle& tri = m_tris[hitIdx];
        if (tri.isObjectSpace) {
            const MeshDesc& mesh = m_pool.mesh(tri.meshID);
            Mat4f o2w = mesh.interpolateO2W(ray.time);
            fillSurfaceInteraction(tri, closestT, hitU, hitV, &o2w, result.si);
        } else {
            fillSurfaceInteraction(tri, closestT, hitU, hitV, nullptr, result.si);
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
    float tNear, t, u, v;

    while (top > 0) {
        uint32_t nodeIdx = stack[--top];
        const BVHNode& node = m_nodes[nodeIdx];
        if (!intersectAABB(node, r, tNear)) continue;

        if (node.isLeaf()) {
            for (uint32_t i = 0; i < node.primCount(); ++i) {
                uint32_t idx = m_primIndices[node.primOffset() + i];
                const BVHTriangle& tri = m_tris[idx];
                if (tri.isObjectSpace) {
                    const MeshDesc& mesh = m_pool.mesh(tri.meshID);
                    Mat4f o2w = mesh.interpolateO2W(ray.time);
                    Mat4f w2o = o2w.inverse();
                    Ray4 ro = makeObjectSpaceRay4(ray, w2o);
                    if (intersectTriangle(tri, ro, t, u, v)) return true;
                } else {
                    if (intersectTriangle(tri, r, t, u, v)) return true;
                }
            }
        } else {
            stack[top++] = nodeIdx + 1;
            stack[top++] = node.rightChild();
        }
    }
    return false;
}

} // namespace anacapa
