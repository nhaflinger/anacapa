#pragma once

// Pure C++ header — no CUDA types exposed.
// Mirrors MetalAccelStructure.h.

#include <anacapa/accel/GeometryPool.h>
#include <cstdint>
#include <memory>

namespace anacapa {

class CudaContext;

// ---------------------------------------------------------------------------
// CudaAccelStructure — flat BVH2 over all scene triangles.
//
// Built on the CPU using recursive median-split then uploaded to the GPU.
// Geometry is assumed to be world-space (same assumption as Metal backend).
//
// Also uploads vertex attributes (positions, normals) and globalized indices
// so the shade kernel can test intersections and interpolate hit data.
// ---------------------------------------------------------------------------
class CudaAccelStructure {
public:
    CudaAccelStructure(CudaContext& ctx, const GeometryPool& pool);
    ~CudaAccelStructure();

    bool isValid() const;

    // Device pointers (raw uint64 = CUdeviceptr)
    uint64_t bvhBuffer()               const;  // BvhNode*   — flat node array
    uint64_t triIndexBuffer()          const;  // uint32_t*  — BVH-reordered triangle indices
    uint64_t positionBuffer()          const;  // float*     — packed float3, all meshes
    uint64_t normalBuffer()            const;  // GpuFloat3* — all meshes concatenated
    uint64_t indexBuffer()             const;  // uint32_t*  — globalized triangle indices
    uint64_t triMeshIDBuffer()         const;  // uint32_t*  — per-triangle meshID
    uint64_t meshVertexOffsetBuffer()  const;  // uint32_t*  — per-mesh vertex base
    uint64_t meshIndexOffsetBuffer()   const;  // uint32_t*  — per-mesh index base (elements)

    uint32_t totalVertices()  const;
    uint32_t totalTriangles() const;
    uint32_t numMeshes()      const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
