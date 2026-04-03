#pragma once

// Pure C++ header — no Objective-C types.
// Wraps the Metal BLAS+TLAS build from a GeometryPool.

#include <anacapa/accel/GeometryPool.h>
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
// ---------------------------------------------------------------------------
class MetalAccelStructure {
public:
    // device: id<MTLDevice> (void* to keep header ObjC-free)
    // cmdQueue: id<MTLCommandQueue>
    MetalAccelStructure(void* device, void* cmdQueue, const GeometryPool& pool);
    ~MetalAccelStructure();

    bool isValid() const;

    // TLAS handle — bind to Shade kernel as [[buffer(N)]] acceleration_structure
    void* tlas() const;  // id<MTLAccelerationStructure>

    // Vertex attribute buffers (interleaved per mesh, concatenated)
    // Layout: one entry per VERTEX across all meshes, in mesh order.
    void* positionBuffer() const;   // id<MTLBuffer> — packed_float3
    void* normalBuffer()   const;   // id<MTLBuffer> — packed_float3
    void* uvBuffer()       const;   // id<MTLBuffer> — float2

    // Per-triangle meshID: triGlobalIdx → meshID
    void* triMeshIDBuffer() const;  // id<MTLBuffer> — uint32_t

    // Per-mesh vertex offset into the concatenated attribute buffers
    void* meshVertexOffsetBuffer() const;  // id<MTLBuffer> — uint32_t

    // Per-mesh index offset into the concatenated index buffer
    void* meshIndexOffsetBuffer()  const;  // id<MTLBuffer> — uint32_t

    // Concatenated index buffer (all meshes)
    void* indexBuffer() const;  // id<MTLBuffer> — uint32_t

    uint32_t totalVertices()  const;
    uint32_t totalTriangles() const;
    uint32_t numMeshes()      const;

    // Array of per-mesh BLAS handles — callers must useResource each entry
    // before dispatching a kernel that traverses the TLAS.
    std::vector<void*> blasHandles() const;  // id<MTLAccelerationStructure> per mesh

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
