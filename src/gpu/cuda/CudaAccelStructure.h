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
// Mirrors MetalAccelStructure layout: one GAS per pool mesh, plus an
// optional hair GAS, all referenced by IAS instances.  Prototype meshes
// (those referenced by InstanceGroupDesc) live in object space and appear
// only through per-instance IAS entries carrying actual objectToWorld
// transforms; other meshes get identity IAS transforms (geometry already
// baked to world space upstream).  hairMeshBaseID() returns a virtual
// meshID (= numMeshes) used by the shader to detect hair hits via
// instanceMeshIDs lookup.
//
// When no meshes are supplied (particles-only), the build collapses to a
// single dummy GAS wrapped in a one-instance IAS — the traversable handle
// always points at an IAS, which keeps the OptiX pipeline configuration
// uniform.
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
    uint64_t uvBuffer()                const;  // GpuFloat2* — all meshes concatenated
    uint64_t tangentBuffer()           const;  // GpuFloat4* — xyz=tangent, w=handedness; all meshes concatenated
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
    uint32_t hairMeshBaseID()    const;  // virtual hair meshID (= numMeshes), or 0xFFFFFFFF

    // Per-IAS-instance lookup tables — mirror MetalAccelStructure.
    // instanceMeshIDBuffer: one uint32_t per IAS instance → pool meshID.
    //   Regular meshes map 1:1; hair gets the virtual ID (= numMeshes);
    //   per-instance entries from InstanceGroupDesc point at protoMeshID.
    // instanceNormalMatrixBuffer: 12 floats per IAS instance = rows of
    //   worldToObject^T.  Identity for regular meshes (geometry world-space),
    //   actual w2o^T for prototype instances (geometry object-space).
    uint64_t instanceMeshIDBuffer()       const;  // uint32_t*
    uint64_t instanceNormalMatrixBuffer() const;  // float*
    // instanceTangentMatrixBuffer: 12 floats per IAS instance = plain
    //   objectToWorld rotation (tangents are ordinary directions, not
    //   normals — no inverse-transpose). Identity for regular meshes.
    uint64_t instanceTangentMatrixBuffer() const;  // float*
    // instancePositionMatrixBuffer: 12 floats per IAS instance = rows of
    //   plain worldToObject (translation included) — object-space hit
    //   position for MaterialX <position space="object"> nodes. NOT identity
    //   for regular meshes (unlike normal/tangent) — each mesh's real
    //   worldToObject is needed even though geometry is baked to world space.
    uint64_t instancePositionMatrixBuffer() const;  // float*

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
