#pragma once

#include <anacapa/core/Types.h>
#include <cstdint>
#include <string>
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

    // Indices: every 3 consecutive indices form a triangle
    std::vector<uint32_t> indices;

    // Motion keys sorted by normalized time in [0, 1].
    // Empty for static meshes; size >= 2 for animated meshes.
    // Positions are kept in object space when motionKeys is non-empty.
    std::vector<MotionKey> motionKeys;

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
// GeometryPool — owns all mesh data for the scene
//
// The BVH backend receives a const reference and builds acceleration
// structures over this data without copying it.
// ---------------------------------------------------------------------------
class GeometryPool {
public:
    uint32_t addMesh(MeshDesc mesh) {
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
        m_meshes[id] = std::move(replacement);
    }

    const std::vector<MeshDesc>& meshes() const { return m_meshes; }

private:
    std::vector<MeshDesc> m_meshes;
};

} // namespace anacapa
