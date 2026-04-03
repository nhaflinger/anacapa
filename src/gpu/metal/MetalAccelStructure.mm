#ifdef ANACAPA_ENABLE_METAL

#include "MetalAccelStructure.h"
#include "MetalBuffer.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>

namespace anacapa {

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct MetalAccelStructure::Impl {
    id<MTLAccelerationStructure>              tlas = nil;
    NSMutableArray<id<MTLAccelerationStructure>>* blasArray = nil;

    // Concatenated vertex attribute buffers (all meshes in order)
    id<MTLBuffer> positionBuffer          = nil;  // packed_float3
    id<MTLBuffer> normalBuffer            = nil;  // packed_float3
    id<MTLBuffer> uvBuffer                = nil;  // float2
    id<MTLBuffer> indexBuffer             = nil;  // uint32_t
    id<MTLBuffer> triMeshIDBuffer         = nil;  // uint32_t per tri
    id<MTLBuffer> meshVertexOffsetBuffer  = nil;  // uint32_t per mesh
    id<MTLBuffer> meshIndexOffsetBuffer   = nil;  // uint32_t per mesh

    uint32_t totalVertices  = 0;
    uint32_t totalTriangles = 0;
    uint32_t numMeshes      = 0;
    bool     valid          = false;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static id<MTLBuffer> makeSharedBuffer(id<MTLDevice> device,
                                       const void* data,
                                       size_t bytes,
                                       const char* label) {
    id<MTLBuffer> buf = [device newBufferWithBytes:data
                                           length:bytes
                                          options:MTLResourceStorageModeShared];
    buf.label = [NSString stringWithUTF8String:label];
    return buf;
}

// Aligned float3 for Metal packed_float3 (12 bytes, no padding)
struct PackedFloat3 { float x, y, z; };
struct PackedFloat2 { float x, y; };

static id<MTLAccelerationStructure> buildAccelStructure(
    id<MTLDevice>              device,
    id<MTLCommandQueue>        cmdQueue,
    MTLAccelerationStructureDescriptor* desc)
{
    // Query scratch + AS sizes
    MTLAccelerationStructureSizes sizes =
        [device accelerationStructureSizesWithDescriptor:desc];

    id<MTLAccelerationStructure> as =
        [device newAccelerationStructureWithSize:sizes.accelerationStructureSize];
    id<MTLBuffer> scratch =
        [device newBufferWithLength:sizes.buildScratchBufferSize
                            options:MTLResourceStorageModePrivate];

    id<MTLCommandBuffer>                   cmdBuf  = [cmdQueue commandBuffer];
    id<MTLAccelerationStructureCommandEncoder> enc =
        [cmdBuf accelerationStructureCommandEncoder];

    [enc buildAccelerationStructure:as
                         descriptor:desc
                      scratchBuffer:scratch
                scratchBufferOffset:0];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    return as;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
MetalAccelStructure::MetalAccelStructure(void*             deviceVoid,
                                          void*             cmdQueueVoid,
                                          const GeometryPool& pool)
    : m_impl(std::make_unique<Impl>())
{
    id<MTLDevice>       device   = (__bridge id<MTLDevice>)      deviceVoid;
    id<MTLCommandQueue> cmdQueue = (__bridge id<MTLCommandQueue>) cmdQueueVoid;

    uint32_t numMeshes = static_cast<uint32_t>(pool.numMeshes());
    m_impl->numMeshes  = numMeshes;
    m_impl->blasArray  = [NSMutableArray arrayWithCapacity:numMeshes];

    if (numMeshes == 0) {
        spdlog::error("MetalAccelStructure: empty geometry pool");
        return;
    }

    // -----------------------------------------------------------------------
    // Pass 1: count totals and build concatenated CPU-side arrays
    // -----------------------------------------------------------------------
    uint32_t totalVerts = 0, totalTris = 0;
    for (uint32_t i = 0; i < numMeshes; ++i) {
        const MeshDesc& m = pool.mesh(i);
        totalVerts += m.numVertices();
        totalTris  += m.numTriangles();
    }

    std::vector<PackedFloat3> positions(totalVerts);
    std::vector<PackedFloat3> normals  (totalVerts);
    std::vector<PackedFloat2> uvs      (totalVerts);
    std::vector<uint32_t>     indices  (totalTris * 3);
    std::vector<uint32_t>     triMeshIDs(totalTris);
    std::vector<uint32_t>     vertexOffsets(numMeshes);
    std::vector<uint32_t>     indexOffsets (numMeshes);

    uint32_t vBase = 0, tBase = 0;
    for (uint32_t mi = 0; mi < numMeshes; ++mi) {
        const MeshDesc& m = pool.mesh(mi);
        vertexOffsets[mi] = vBase;
        indexOffsets [mi] = tBase * 3;  // byte/element offset in index array

        for (uint32_t v = 0; v < m.numVertices(); ++v) {
            positions[vBase + v] = {m.positions[v].x, m.positions[v].y, m.positions[v].z};
            normals  [vBase + v] = m.normals.empty()
                ? PackedFloat3{0,1,0}
                : PackedFloat3{m.normals[v].x, m.normals[v].y, m.normals[v].z};
            uvs[vBase + v] = m.uvs.empty()
                ? PackedFloat2{0,0}
                : PackedFloat2{m.uvs[v].x, m.uvs[v].y};
        }

        for (uint32_t t = 0; t < m.numTriangles(); ++t) {
            indices[(tBase + t) * 3 + 0] = vBase + m.indices[t * 3 + 0];
            indices[(tBase + t) * 3 + 1] = vBase + m.indices[t * 3 + 1];
            indices[(tBase + t) * 3 + 2] = vBase + m.indices[t * 3 + 2];
            triMeshIDs[tBase + t]         = mi;
        }

        vBase += m.numVertices();
        tBase += m.numTriangles();
    }

    m_impl->totalVertices  = totalVerts;
    m_impl->totalTriangles = totalTris;

    // -----------------------------------------------------------------------
    // Upload attribute buffers
    // -----------------------------------------------------------------------
    m_impl->positionBuffer = makeSharedBuffer(
        device, positions.data(),  totalVerts * sizeof(PackedFloat3), "positions");
    m_impl->normalBuffer   = makeSharedBuffer(
        device, normals.data(),    totalVerts * sizeof(PackedFloat3), "normals");
    m_impl->uvBuffer       = makeSharedBuffer(
        device, uvs.data(),        totalVerts * sizeof(PackedFloat2), "uvs");
    m_impl->indexBuffer    = makeSharedBuffer(
        device, indices.data(),    totalTris * 3 * sizeof(uint32_t),  "indices");
    m_impl->triMeshIDBuffer = makeSharedBuffer(
        device, triMeshIDs.data(), totalTris * sizeof(uint32_t),      "triMeshIDs");
    m_impl->meshVertexOffsetBuffer = makeSharedBuffer(
        device, vertexOffsets.data(), numMeshes * sizeof(uint32_t),   "vertexOffsets");
    m_impl->meshIndexOffsetBuffer  = makeSharedBuffer(
        device, indexOffsets.data(),  numMeshes * sizeof(uint32_t),   "indexOffsets");

    // -----------------------------------------------------------------------
    // Build one BLAS per mesh
    // -----------------------------------------------------------------------
    for (uint32_t mi = 0; mi < numMeshes; ++mi) {
        const MeshDesc& m = pool.mesh(mi);

        // Vertex and index sub-buffers are the full concatenated buffers with offsets.
        MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];

        // Use the full concatenated vertex buffer starting at offset 0.
        // Global indices (which include vBase) are correct because the vertex
        // buffer covers all meshes — no per-mesh vertex offset needed.
        geomDesc.vertexBuffer       = m_impl->positionBuffer;
        geomDesc.vertexBufferOffset = 0;
        geomDesc.vertexStride       = sizeof(PackedFloat3);
        geomDesc.vertexFormat       = MTLAttributeFormatFloat3;

        geomDesc.indexBuffer       = m_impl->indexBuffer;
        geomDesc.indexBufferOffset = indexOffsets[mi] * sizeof(uint32_t);
        geomDesc.indexType         = MTLIndexTypeUInt32;

        geomDesc.triangleCount = m.numTriangles();
        geomDesc.opaque        = YES;

        MTLPrimitiveAccelerationStructureDescriptor* blasDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        blasDesc.geometryDescriptors = @[geomDesc];

        id<MTLAccelerationStructure> blas =
            buildAccelStructure(device, cmdQueue, blasDesc);
        if (!blas) {
            spdlog::error("MetalAccelStructure: BLAS build failed for mesh {}", mi);
            return;
        }
        [m_impl->blasArray addObject:blas];
    }

    // -----------------------------------------------------------------------
    // Build TLAS with identity transforms (geometry is already world-space)
    // -----------------------------------------------------------------------
    // Populate instance descriptors in a CPU buffer.
    // MTLPackedFloat4x3 stores 4 rows of float3 (columns 0-2 = rotation/scale,
    // row 3 = translation).  Identity = rows {1,0,0}, {0,1,0}, {0,0,1}, {0,0,0}.
    size_t instDescSize = sizeof(MTLAccelerationStructureInstanceDescriptor);
    id<MTLBuffer> instBuf = [device newBufferWithLength:instDescSize * numMeshes
                                               options:MTLResourceStorageModeShared];
    instBuf.label = @"tlasInstances";
    MTLAccelerationStructureInstanceDescriptor* instDescs =
        (MTLAccelerationStructureInstanceDescriptor*)[instBuf contents];

    for (uint32_t i = 0; i < numMeshes; ++i) {
        MTLAccelerationStructureInstanceDescriptor d;
        memset(&d, 0, sizeof(d));
        // Identity MTLPackedFloat4x3: columns are rows of the matrix
        d.transformationMatrix.columns[0] = {1, 0, 0};
        d.transformationMatrix.columns[1] = {0, 1, 0};
        d.transformationMatrix.columns[2] = {0, 0, 1};
        d.transformationMatrix.columns[3] = {0, 0, 0};
        d.options            = MTLAccelerationStructureInstanceOptionOpaque;
        d.mask               = 0xFF;
        d.intersectionFunctionTableOffset = 0;
        d.accelerationStructureIndex      = i;
        instDescs[i] = d;
    }

    MTLInstanceAccelerationStructureDescriptor* tlasDesc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlasDesc.instancedAccelerationStructures = m_impl->blasArray;
    tlasDesc.instanceCount                   = numMeshes;
    tlasDesc.instanceDescriptorBuffer        = instBuf;

    m_impl->tlas = buildAccelStructure(device, cmdQueue, tlasDesc);
    if (!m_impl->tlas) {
        spdlog::error("MetalAccelStructure: TLAS build failed");
        return;
    }

    m_impl->valid = true;
    spdlog::info("MetalAccelStructure: built {} BLAS + TLAS ({} verts, {} tris)",
                 numMeshes, totalVerts, totalTris);
}

MetalAccelStructure::~MetalAccelStructure() = default;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
bool     MetalAccelStructure::isValid()     const { return m_impl->valid; }
uint32_t MetalAccelStructure::totalVertices()  const { return m_impl->totalVertices; }
uint32_t MetalAccelStructure::totalTriangles() const { return m_impl->totalTriangles; }
uint32_t MetalAccelStructure::numMeshes()      const { return m_impl->numMeshes; }

std::vector<void*> MetalAccelStructure::blasHandles() const {
    std::vector<void*> out;
    if (!m_impl->blasArray) return out;
    out.reserve([m_impl->blasArray count]);
    for (id<MTLAccelerationStructure> blas in m_impl->blasArray)
        out.push_back((__bridge void*)blas);
    return out;
}

void* MetalAccelStructure::tlas()                  const { return (__bridge void*)m_impl->tlas; }
void* MetalAccelStructure::positionBuffer()        const { return (__bridge void*)m_impl->positionBuffer; }
void* MetalAccelStructure::normalBuffer()          const { return (__bridge void*)m_impl->normalBuffer; }
void* MetalAccelStructure::uvBuffer()              const { return (__bridge void*)m_impl->uvBuffer; }
void* MetalAccelStructure::triMeshIDBuffer()       const { return (__bridge void*)m_impl->triMeshIDBuffer; }
void* MetalAccelStructure::meshVertexOffsetBuffer() const { return (__bridge void*)m_impl->meshVertexOffsetBuffer; }
void* MetalAccelStructure::meshIndexOffsetBuffer()  const { return (__bridge void*)m_impl->meshIndexOffsetBuffer; }
void* MetalAccelStructure::indexBuffer()            const { return (__bridge void*)m_impl->indexBuffer; }

} // namespace anacapa

#endif // ANACAPA_ENABLE_METAL
