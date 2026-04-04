#pragma once

#include <anacapa/core/Types.h>
#include <cstdint>
#include <string>
#include <vector>

namespace anacapa {

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

    // World-from-object transform at shutter-open (t=0)
    Mat4f objectToWorld  = Mat4f::identity();
    Mat4f worldToObject  = Mat4f::identity();  // Precomputed inverse

    // World-from-object transform at shutter-close (t=1).
    // Only used when hasMotion == true.
    Mat4f objectToWorld1 = Mat4f::identity();
    Mat4f worldToObject1 = Mat4f::identity();
    bool  hasMotion      = false;

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

    const std::vector<MeshDesc>& meshes() const { return m_meshes; }

private:
    std::vector<MeshDesc> m_meshes;
};

} // namespace anacapa
