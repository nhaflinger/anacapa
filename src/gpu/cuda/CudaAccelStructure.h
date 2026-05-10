#pragma once

// Pure C++ header — no CUDA types exposed.
// Mirrors MetalAccelStructure.h.

#include <anacapa/accel/GeometryPool.h>
#include <cstdint>
#include <memory>

namespace anacapa {

class CudaContext;

// ---------------------------------------------------------------------------
// CudaAccelStructure — OptiX GAS over all scene triangles.
//
// Builds a single triangle GAS, motion-aware when any mesh has motion keys
// (vertex buffers at shutter-open and shutter-close).  Owns the world-space
// vertex / normal / index buffers used both for the GAS build and for
// shading-time attribute fetch in the OptiX programs.
// ---------------------------------------------------------------------------
class CudaAccelStructure {
public:
    CudaAccelStructure(CudaContext& ctx, const GeometryPool& pool);
    ~CudaAccelStructure();

    bool isValid() const;

    // Device pointers (raw uint64 = CUdeviceptr)
    uint64_t positionBuffer()          const;  // float*     — packed float3, world-space at shutter-open
    uint64_t positionBufferClose()     const;  // float*     — packed float3, world-space at shutter-close
    uint64_t normalBuffer()            const;  // GpuFloat3* — all meshes concatenated (baked at open)
    uint64_t indexBuffer()             const;  // uint32_t*  — globalized triangle indices
    uint64_t triMeshIDBuffer()         const;  // uint32_t*  — per-triangle meshID
    uint64_t meshVertexOffsetBuffer()  const;  // uint32_t*  — per-mesh vertex base
    uint64_t meshIndexOffsetBuffer()   const;  // uint32_t*  — per-mesh index base (elements)

    // OptiX traversable handle (returned uint64 = OptixTraversableHandle).
    // Zero unless ANACAPA_ENABLE_OPTIX was set at build time.
    uint64_t traversableHandle()       const;

    // True if any mesh has motion keys (=> OptiX GAS was built motion-aware
    // and positionBufferClose() differs from positionBuffer()).
    bool     hasMotion()               const;

    uint32_t totalVertices()  const;
    uint32_t totalTriangles() const;
    uint32_t numMeshes()      const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
