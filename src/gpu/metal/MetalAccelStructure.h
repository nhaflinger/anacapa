#pragma once

// Pure C++ header — no Objective-C types.
// Wraps the Metal BLAS+TLAS build from a GeometryPool and optional CurvePool.

#include <anacapa/accel/GeometryPool.h>
#include <anacapa/accel/CurvePool.h>
#include <memory>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// MetalAccelStructure — builds and owns the Metal hardware ray-tracing
// acceleration structure (BLAS per mesh + TLAS over all instances).
//
// Geometry is assumed to be already in world space so all TLAS instance
// transforms are identity.
//
// Also uploads vertex attributes (positions, normals, UVs) and per-triangle
// meshID into MTLBuffers so the Shade kernel can interpolate hit data.
//
// Hair: if curvePool is non-null and has strands, each Bézier strand is
// tessellated into a flat ribbon mesh (kHairTessSteps quads per segment).
// The hair BLAS is added as the last instance in the TLAS.  Motion blur is
// supported via two-keyframe vertex buffers (shutter-open / shutter-close).
// ---------------------------------------------------------------------------
class MetalAccelStructure {
public:
    // device: id<MTLDevice> (void* to keep header ObjC-free)
    // cmdQueue: id<MTLCommandQueue>
    // curvePool: optional — pass nullptr when the scene has no hair
    // hairTessSteps: quads per Bézier segment (default 4)
    MetalAccelStructure(void* device, void* cmdQueue,
                        const GeometryPool& pool,
                        const CurvePool*    curvePool    = nullptr,
                        int                 hairTessSteps = 4);
    ~MetalAccelStructure();

    bool isValid() const;

    // TLAS handle — bind to Shade kernel as [[buffer(N)]] acceleration_structure
    void* tlas() const;  // id<MTLAccelerationStructure>

    // Vertex attribute buffers (interleaved per mesh, concatenated)
    // Layout: one entry per VERTEX across all meshes, in mesh order.
    void* positionBuffer() const;   // id<MTLBuffer> — packed_float3
    void* normalBuffer()   const;   // id<MTLBuffer> — packed_float3
    // uv + tangent share one buffer (see VertexExtra in Shade.metal /
    // MetalAccelStructure.mm) — Metal's shade kernel is at its 31-buffer-argument cap.
    void* vertexExtraBuffer() const;  // id<MTLBuffer> — VertexExtra { packed_float2 uv; packed_float4 tangent; }

    // Per-triangle meshID: triGlobalIdx → meshID
    void* triMeshIDBuffer() const;  // id<MTLBuffer> — uint32_t

    // Per-mesh vertex offset into the concatenated attribute buffers
    void* meshVertexOffsetBuffer() const;  // id<MTLBuffer> — uint32_t

    // Per-mesh index offset into the concatenated index buffer
    void* meshIndexOffsetBuffer()  const;  // id<MTLBuffer> — uint32_t

    // Concatenated index buffer (all meshes)
    void* indexBuffer() const;  // id<MTLBuffer> — uint32_t

    // Per-TLAS-instance lookup buffers (size = total TLAS instance count).
    // instanceMeshIDBuffer: uint32_t — TLAS instance → pool mesh ID for vertex/index/material lookup
    // instanceNormalMatrixBuffer: float4[9] per instance —
    //   rows [0-2] = worldToObject^T (normal transform)
    //   rows [3-5] = plain objectToWorld rotation (tangent transform)
    //   rows [6-8] = plain worldToObject, translation included (object-space
    //                position reconstruction for MaterialX <position space="object">)
    //   Rows [0-5] are identity for regular world-space meshes (geometry
    //   pre-baked to world space); rows [6-8] are NOT identity even for those
    //   meshes — each carries its own real worldToObject. Rows [0-8] all carry
    //   the actual transform for prototype mesh instances.
    void* instanceMeshIDBuffer()         const;  // id<MTLBuffer> — uint32_t
    void* instanceNormalMatrixBuffer()   const;  // id<MTLBuffer> — float4[9] per instance

    uint32_t totalVertices()  const;
    uint32_t totalTriangles() const;
    uint32_t numMeshes()      const;

    // Array of per-mesh BLAS handles — callers must useResource each entry
    // before dispatching a kernel that traverses the TLAS.
    std::vector<void*> blasHandles() const;  // id<MTLAccelerationStructure> per mesh

    // Hair ribbon buffers — valid only when hairMeshBaseID() != 0xFFFFFFFF.
    // hairTriBuffer: one GpuHairTri per tessellated triangle (primID-indexed).
    // hairMatBuffer: one GpuHairMaterial per scene material slot.
    void*    hairTriBuffer()     const;  // id<MTLBuffer>
    void*    hairMatBuffer()     const;  // id<MTLBuffer>
    uint32_t numHairMaterials()  const;
    uint32_t hairMeshBaseID()    const;  // TLAS instance ID of the hair mesh; 0xFFFFFFFF if none

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
