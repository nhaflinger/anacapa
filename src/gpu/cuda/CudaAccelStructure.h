#pragma once

// Pure C++ header — no CUDA types exposed.
// Mirrors MetalAccelStructure.h.

#include <anacapa/accel/GeometryPool.h>
#include <anacapa/accel/CurvePool.h>
#include <cstdint>
#include <memory>

namespace anacapa {

class CudaContext;

// ---------------------------------------------------------------------------
// CudaAccelStructure — OptiX acceleration structure over scene triangles
// and (optionally) tessellated hair ribbons.
//
// When the CurvePool argument is non-null and contains strands, hair is
// tessellated into ribbon quads (matching the Metal backend), built into a
// second triangle GAS, and exposed alongside the mesh GAS through an IAS.
// The IAS has two instances: index 0 = mesh GAS, index 1 = hair GAS.  The
// hairMeshBaseID() accessor returns that hair instance index so raygen can
// dispatch hits to the Marschner BSDF.
//
// When no curves are supplied, the build collapses to a single triangle GAS
// wrapped in a one-instance IAS — the traversable handle always points at
// an IAS, which keeps the OptiX pipeline configuration uniform.
// ---------------------------------------------------------------------------
class CudaAccelStructure {
public:
    // hairTessSteps: quads per cubic Bézier segment for hair ribbon
    // tessellation (default 4).  Same convention as MetalAccelStructure.
    CudaAccelStructure(CudaContext& ctx, const GeometryPool& pool,
                       const CurvePool* curvePool = nullptr,
                       int hairTessSteps = 4);
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

    // Hair accessors — valid only when hairMeshBaseID() != 0xFFFFFFFF.
    // hairTriBuffer: one GpuHairTri per tessellated triangle (primID-indexed
    // within the hair GAS).  The per-material hair-BSDF buffer is owned by
    // CudaPathIntegrator (it depends on scene.materials, not on geometry).
    uint64_t hairTriBuffer()     const;  // device GpuHairTri*
    uint32_t hairMeshBaseID()    const;  // IAS instance ID, or 0xFFFFFFFF

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
