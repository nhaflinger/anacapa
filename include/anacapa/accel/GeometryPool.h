#pragma once

#include <anacapa/core/Types.h>
#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// MotionKey — one transform sample within the shutter interval
//
// time is normalized to [0, 1] where 0 = shutter open, 1 = shutter close.
// Keys must be sorted by time ascending.
// ---------------------------------------------------------------------------
struct MotionKey {
    float time          = 0.f;
    Mat4f objectToWorld = Mat4f::identity();
    Mat4f worldToObject = Mat4f::identity();  // precomputed inverse of objectToWorld
};

// ---------------------------------------------------------------------------
// MeshDesc — a single triangle mesh in SoA layout
//
// SoA is mandatory for GPU migration: each array maps directly to an
// MTLBuffer or CUDA device pointer with a single cudaMemcpy / [MTLBuffer contents].
// ---------------------------------------------------------------------------
struct MeshDesc {
    std::string name;

    // Vertex attributes (parallel arrays, indexed by vertex index)
    std::vector<Vec3f>   positions;
    std::vector<Vec3f>   normals;
    std::vector<Vec2f>   uvs;
    // xyz = UV-derivative tangent (orthonormalized against the vertex normal),
    // w = bitangent handedness sign. Populated by computeTangents() below;
    // empty until then.
    std::vector<Vec4f>   tangents;

    // Indices: every 3 consecutive indices form a triangle
    std::vector<uint32_t> indices;

    // Motion keys sorted by normalized time in [0, 1].
    // Empty for static meshes; size >= 2 for animated meshes.
    // Positions are kept in object space when motionKeys is non-empty.
    std::vector<MotionKey> motionKeys;

    // Static O2W: the world transform baked into vertex positions for static
    // meshes (motionKeys empty). Needed to reconstruct object-space coords for
    // OSL position(space="object") calls. Identity if unknown.
    Mat4f staticObjectToWorld = Mat4f::identity();
    Mat4f staticWorldToObject = Mat4f::identity();

    bool hasMotion() const { return !motionKeys.empty(); }

    // Convenience: transform at shutter-open (first key)
    const Mat4f& objectToWorld() const { return motionKeys.front().objectToWorld; }

    // Interpolate the object-to-world transform at normalized time t in [0,1].
    // Uses piecewise-linear lerp between adjacent keys.
    Mat4f interpolateO2W(float t) const {
        if (motionKeys.size() == 1) return motionKeys[0].objectToWorld;
        if (t <= motionKeys.front().time) return motionKeys.front().objectToWorld;
        if (t >= motionKeys.back().time)  return motionKeys.back().objectToWorld;
        // Binary search for the bracketing interval
        size_t lo = 0, hi = motionKeys.size() - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
            if (motionKeys[mid].time <= t) lo = mid; else hi = mid;
        }
        float t0 = motionKeys[lo].time;
        float t1 = motionKeys[hi].time;
        float alpha = (t - t0) / (t1 - t0);
        return Mat4f::lerp(motionKeys[lo].objectToWorld, motionKeys[hi].objectToWorld, alpha);
    }

    // Material index into SceneGraph::materials
    uint32_t materialIndex = 0;

    uint32_t numTriangles() const {
        return static_cast<uint32_t>(indices.size() / 3);
    }

    uint32_t numVertices() const {
        return static_cast<uint32_t>(positions.size());
    }
};

// ---------------------------------------------------------------------------
// computeTangents — fills MeshDesc::tangents from position/normal/uv data
// (Lengyel per-triangle UV-derivative method). Call once after a mesh's
// positions/normals/uvs/indices are fully populated. Safe to call on meshes
// with no UVs (or degenerate/duplicate UVs) — falls back to an arbitrary
// per-vertex orthonormal basis in that case.
// ---------------------------------------------------------------------------
inline void computeTangents(MeshDesc& mesh) {
    const size_t numVerts = mesh.positions.size();
    mesh.tangents.assign(numVerts, Vec4f{1.f, 0.f, 0.f, 1.f});
    if (numVerts == 0 || mesh.indices.size() < 3) return;

    std::vector<Vec3f> tanAccum(numVerts, Vec3f{0.f, 0.f, 0.f});
    std::vector<Vec3f> bitanAccum(numVerts, Vec3f{0.f, 0.f, 0.f});

    const bool hasUVs = mesh.uvs.size() == numVerts;
    if (hasUVs) {
        const size_t numTris = mesh.indices.size() / 3;
        for (size_t t = 0; t < numTris; ++t) {
            uint32_t i0 = mesh.indices[t * 3 + 0];
            uint32_t i1 = mesh.indices[t * 3 + 1];
            uint32_t i2 = mesh.indices[t * 3 + 2];

            Vec3f e1 = mesh.positions[i1] - mesh.positions[i0];
            Vec3f e2 = mesh.positions[i2] - mesh.positions[i0];
            float duv1x = mesh.uvs[i1].x - mesh.uvs[i0].x;
            float duv1y = mesh.uvs[i1].y - mesh.uvs[i0].y;
            float duv2x = mesh.uvs[i2].x - mesh.uvs[i0].x;
            float duv2y = mesh.uvs[i2].y - mesh.uvs[i0].y;

            float r = duv1x * duv2y - duv2x * duv1y;
            if (std::abs(r) < 1e-12f) continue;  // degenerate UV triangle
            float invR = 1.f / r;

            Vec3f tangent   = (e1 * duv2y - e2 * duv1y) * invR;
            Vec3f bitangent = (e2 * duv1x - e1 * duv2x) * invR;

            tanAccum[i0] += tangent;   tanAccum[i1] += tangent;   tanAccum[i2] += tangent;
            bitanAccum[i0] += bitangent; bitanAccum[i1] += bitangent; bitanAccum[i2] += bitangent;
        }
    }

    const bool hasNormals = mesh.normals.size() == numVerts;
    for (size_t i = 0; i < numVerts; ++i) {
        const Vec3f n = hasNormals ? mesh.normals[i] : Vec3f{0.f, 0.f, 1.f};

        // Gram-Schmidt orthogonalize the accumulated tangent against the normal.
        Vec3f t = tanAccum[i] - n * dot(n, tanAccum[i]);
        if (t.lengthSq() < 1e-12f) {
            // No UVs, or degenerate accumulation (e.g. all-zero UVs): fall back
            // to an arbitrary but consistent per-vertex basis.
            Vec3f tArb, btArb;
            buildOrthonormalBasis(n, tArb, btArb);
            mesh.tangents[i] = Vec4f{tArb, 1.f};
            continue;
        }
        t = normalize(t);
        float handedness = dot(cross(n, t), bitanAccum[i]) < 0.f ? -1.f : 1.f;
        mesh.tangents[i] = Vec4f{t, handedness};
    }
}

// ---------------------------------------------------------------------------
// InstanceDesc — one instance of a prototype mesh.
// Static instances have exactly one key; motion-blur instances have >= 2.
// ---------------------------------------------------------------------------
struct InstanceDesc {
    std::vector<MotionKey> keys;

    bool hasMotion() const { return keys.size() >= 2; }

    Mat4f interpolateO2W(float t) const {
        if (keys.size() == 1) return keys[0].objectToWorld;
        if (t <= keys.front().time) return keys.front().objectToWorld;
        if (t >= keys.back().time)  return keys.back().objectToWorld;
        size_t lo = 0, hi = keys.size() - 1;
        while (hi - lo > 1) {
            size_t mid = (lo + hi) / 2;
            if (keys[mid].time <= t) lo = mid; else hi = mid;
        }
        float t0 = keys[lo].time, t1 = keys[hi].time;
        float alpha = (t - t0) / (t1 - t0);
        return Mat4f::lerp(keys[lo].objectToWorld, keys[hi].objectToWorld, alpha);
    }

    Mat4f interpolateW2O(float t) const {
        if (keys.size() == 1) return keys[0].worldToObject;
        return interpolateO2W(t).inverse();
    }
};

// ---------------------------------------------------------------------------
// InstanceGroupDesc — a prototype mesh shared by N instances.
// The prototype MeshDesc (at protoMeshID) must store positions in object space
// (motionKeys empty, positions NOT baked with any world transform).
// ---------------------------------------------------------------------------
struct InstanceGroupDesc {
    uint32_t                  protoMeshID;
    std::vector<InstanceDesc> instances;
};

// ---------------------------------------------------------------------------
// GeometryPool — owns all mesh data for the scene
//
// The BVH backend receives a const reference and builds acceleration
// structures over this data without copying it.
// ---------------------------------------------------------------------------
class GeometryPool {
public:
    uint32_t addMesh(MeshDesc mesh) {
        computeTangents(mesh);
        uint32_t id = static_cast<uint32_t>(m_meshes.size());
        m_meshes.push_back(std::move(mesh));
        return id;
    }

    const MeshDesc& mesh(uint32_t id) const { return m_meshes[id]; }
    size_t          numMeshes()        const { return m_meshes.size(); }

    // Replace the geometry of an existing mesh in-place.
    // Used by the USD loader to remove GeomSubset faces from the parent mesh
    // after extracting them into separate per-material submeshes.
    void replaceMesh(uint32_t id, MeshDesc replacement) {
        computeTangents(replacement);
        m_meshes[id] = std::move(replacement);
    }

    const std::vector<MeshDesc>& meshes() const { return m_meshes; }

    uint32_t addInstanceGroup(InstanceGroupDesc grp) {
        uint32_t id = static_cast<uint32_t>(m_instanceGroups.size());
        m_instanceGroups.push_back(std::move(grp));
        return id;
    }

    const InstanceGroupDesc& instanceGroup(uint32_t id) const { return m_instanceGroups[id]; }
    size_t                   numInstanceGroups()          const { return m_instanceGroups.size(); }

    const std::vector<InstanceGroupDesc>& instanceGroups() const { return m_instanceGroups; }

    // Mark a mesh as a prototype (object-space, referenced via instance groups).
    // Prototype meshes must NOT be placed directly in any BVH/TLAS as world-space geometry.
    void markPrototype(uint32_t meshID) { m_prototypeMeshIDs.insert(meshID); }
    bool isPrototype(uint32_t meshID)  const { return m_prototypeMeshIDs.count(meshID) > 0; }

private:
    std::vector<MeshDesc>          m_meshes;
    std::vector<InstanceGroupDesc> m_instanceGroups;
    std::unordered_set<uint32_t>   m_prototypeMeshIDs;
};

} // namespace anacapa
