#ifdef ANACAPA_ENABLE_METAL

#include "MetalAccelStructure.h"
#include "MetalBuffer.h"
#include "shaders/SharedTypes.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <anacapa/core/Types.h>

#include <spdlog/spdlog.h>
#include <vector>
#include <cmath>
#include <cstring>

namespace anacapa {

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct MetalAccelStructure::Impl {
    id<MTLAccelerationStructure>              tlas = nil;
    NSMutableArray<id<MTLAccelerationStructure>>* blasArray = nil;

    // Concatenated vertex attribute buffers (all meshes in order)
    id<MTLBuffer> positionBuffer          = nil;  // packed_float3 — world-space at shutter-open
    id<MTLBuffer> positionBufferClose     = nil;  // packed_float3 — world-space at shutter-close (= positionBuffer for static)
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

    // Hair ribbon geometry (separate from triangle mesh buffers)
    id<MTLBuffer> hairPosOpenBuf  = nil;  // GpuHairTri vertex positions at shutter-open
    id<MTLBuffer> hairPosCloseBuf = nil;  // ... at shutter-close (same as open for static hair)
    id<MTLBuffer> hairIndexBuf    = nil;  // uint32_t triangle indices
    id<MTLBuffer> hairTriBuf      = nil;  // GpuHairTri metadata, one entry per triangle
    id<MTLBuffer> hairMatBuf      = nil;  // GpuHairMaterial, one entry per scene material slot
    uint32_t      hairMeshBase    = 0xFFFFFFFFu;  // TLAS instance ID; sentinel = no hair
    uint32_t      numHairMats     = 0;
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

// ---------------------------------------------------------------------------
// Hair tessellation helpers
// ---------------------------------------------------------------------------

static Vec3f bezierPoint(const Vec3f* p, float t) {
    float u = 1.f - t;
    return p[0]*(u*u*u) + p[1]*(3.f*u*u*t) + p[2]*(3.f*u*t*t) + p[3]*(t*t*t);
}

static Vec3f bezierTangent(const Vec3f* p, float t) {
    float u = 1.f - t;
    Vec3f d = (p[1]-p[0])*(3.f*u*u) + (p[2]-p[1])*(6.f*u*t) + (p[3]-p[2])*(3.f*t*t);
    float len = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
    if (len < 1e-8f) return {0.f, 1.f, 0.f};
    return {d.x/len, d.y/len, d.z/len};
}

static Vec3f crossVec(Vec3f a, Vec3f b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

static Vec3f normalizeVec(Vec3f v) {
    float len = std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
    if (len < 1e-8f) return {0.f, 1.f, 0.f};
    return {v.x/len, v.y/len, v.z/len};
}

// Tessellate one cubic Bézier segment [cvs, cvs+3] into ribbon quads.
// Appends (tessSteps+1)*2 position entries to posOpen/posClose, the
// same number of UV/normal entries, and tessSteps*2 triangles + their
// GpuHairTri metadata.  vBase is the index of the first vertex this segment
// will produce in the combined position arrays.
static void tessellateSegment(
    const Vec3f* cvOpen,        // 4 CVs at shutter-open (world space)
    const Vec3f* cvClose,       // 4 CVs at shutter-close (nullptr = same as open)
    float        strandV0,      // strand parametric position at segment start
    float        strandV1,      // strand parametric position at segment end
    float        wRoot,         // width (diameter) at strandV0
    float        wTip,          // width (diameter) at strandV1
    Vec3f        strandColor,
    uint32_t     matIdx,
    uint32_t     vBase,
    int          tessSteps,
    std::vector<PackedFloat3>& posOpen,
    std::vector<PackedFloat3>& posClose,
    std::vector<PackedFloat2>& uvs,
    std::vector<PackedFloat3>& normals,
    std::vector<uint32_t>&     indices,
    std::vector<GpuHairTri>&   hairTris)
{
    // Sample tessSteps+1 positions along the segment
    for (int k = 0; k <= tessSteps; ++k) {
        float t  = float(k) / float(tessSteps);
        float v  = strandV0 + t * (strandV1 - strandV0);
        float w  = (wRoot * (1.f - t) + wTip * t) * 0.5f;  // half-width (radius)
        if (w < 5e-5f) w = 5e-5f;  // minimum 0.1 mm radius for robust intersection

        Vec3f posO = bezierPoint(cvOpen, t);
        Vec3f posC = cvClose ? bezierPoint(cvClose, t) : posO;
        Vec3f tang = bezierTangent(cvOpen, t);

        // Ribbon width direction: perpendicular to tangent, in the plane with world-Y
        // (fall back to world-X when tangent is nearly vertical)
        Vec3f refUp = (std::abs(tang.y) > 0.9f) ? Vec3f{1.f,0.f,0.f} : Vec3f{0.f,1.f,0.f};
        Vec3f perp  = normalizeVec(crossVec(tang, refUp));
        // Outward ribbon normal (perpendicular to both tangent and width direction)
        Vec3f ribbonN = normalizeVec(crossVec(perp, tang));

        // Left vertex (h = -1) and right vertex (h = +1)
        Vec3f lO = {posO.x - perp.x*w, posO.y - perp.y*w, posO.z - perp.z*w};
        Vec3f rO = {posO.x + perp.x*w, posO.y + perp.y*w, posO.z + perp.z*w};
        Vec3f lC = {posC.x - perp.x*w, posC.y - perp.y*w, posC.z - perp.z*w};
        Vec3f rC = {posC.x + perp.x*w, posC.y + perp.y*w, posC.z + perp.z*w};

        posOpen.push_back({lO.x, lO.y, lO.z});
        posOpen.push_back({rO.x, rO.y, rO.z});
        posClose.push_back({lC.x, lC.y, lC.z});
        posClose.push_back({rC.x, rC.y, rC.z});
        normals.push_back({ribbonN.x, ribbonN.y, ribbonN.z});
        normals.push_back({ribbonN.x, ribbonN.y, ribbonN.z});
        uvs.push_back({-1.f, v});  // left: h = -1
        uvs.push_back({ 1.f, v});  // right: h = +1
    }

    // Build tessSteps quads from adjacent row pairs
    for (int k = 0; k < tessSteps; ++k) {
        uint32_t l0 = vBase + uint32_t(2*k + 0);
        uint32_t r0 = vBase + uint32_t(2*k + 1);
        uint32_t l1 = vBase + uint32_t(2*k + 2);
        uint32_t r1 = vBase + uint32_t(2*k + 3);

        // Tangent at the midpoint of this sub-segment
        float tMid = (float(k) + 0.5f) / float(tessSteps);
        Vec3f tang = bezierTangent(cvOpen, tMid);

        GpuHairTri ht;
        ht.tangent = {tang.x, tang.y, tang.z};
        ht.matIdx  = matIdx;
        ht.color   = {strandColor.x, strandColor.y, strandColor.z};
        ht._pad    = 0.f;

        // Tri 0: (l0, r0, r1)  →  vertex h values: (-1, +1, +1)
        indices.push_back(l0); indices.push_back(r0); indices.push_back(r1);
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = +1.f;
        hairTris.push_back(ht);

        // Tri 1: (l0, r1, l1)  →  vertex h values: (-1, +1, -1)
        indices.push_back(l0); indices.push_back(r1); indices.push_back(l1);
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = -1.f;
        hairTris.push_back(ht);
    }
}

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
                                          const GeometryPool& pool,
                                          const CurvePool*  curvePool,
                                          int               hairTessSteps)
    : m_impl(std::make_unique<Impl>())
{
    id<MTLDevice>       device   = (__bridge id<MTLDevice>)      deviceVoid;
    id<MTLCommandQueue> cmdQueue = (__bridge id<MTLCommandQueue>) cmdQueueVoid;

    uint32_t numMeshes = static_cast<uint32_t>(pool.numMeshes());
    m_impl->numMeshes  = numMeshes;
    m_impl->blasArray  = [NSMutableArray arrayWithCapacity:numMeshes];

    if (numMeshes == 0) {
        // Particles-only scene — no triangle geometry at all.
        // Build a single dummy BLAS (one degenerate triangle far from the scene)
        // so the TLAS is valid.  The shader will never intersect this triangle.
        spdlog::info("MetalAccelStructure: no mesh geometry — building dummy BLAS for particles-only scene");

        static const float dv[3][3] = {{1e20f,0,0},{0,1e20f,0},{0,0,1e20f}};
        static const uint32_t di[3] = {0, 1, 2};
        id<MTLBuffer> dvBuf = [device newBufferWithBytes:dv length:sizeof(dv)
                                                 options:MTLResourceStorageModeShared];
        id<MTLBuffer> diBuf = [device newBufferWithBytes:di length:sizeof(di)
                                                 options:MTLResourceStorageModeShared];

        // All attribute buffers: 1 dummy element each so Metal doesn't see nil buffers.
        PackedFloat3 dummyN{0,1,0};
        PackedFloat2 dummyUV{0,0};
        uint32_t     dummyZero = 0;
        m_impl->positionBuffer        = dvBuf;
        m_impl->positionBufferClose   = dvBuf;
        m_impl->normalBuffer          = [device newBufferWithBytes:&dummyN length:sizeof(PackedFloat3)
                                                           options:MTLResourceStorageModeShared];
        m_impl->uvBuffer              = [device newBufferWithBytes:&dummyUV length:sizeof(PackedFloat2)
                                                           options:MTLResourceStorageModeShared];
        m_impl->indexBuffer           = diBuf;
        m_impl->triMeshIDBuffer       = [device newBufferWithBytes:&dummyZero length:sizeof(uint32_t)
                                                           options:MTLResourceStorageModeShared];
        m_impl->meshVertexOffsetBuffer = [device newBufferWithBytes:&dummyZero length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];
        m_impl->meshIndexOffsetBuffer  = [device newBufferWithBytes:&dummyZero length:sizeof(uint32_t)
                                                            options:MTLResourceStorageModeShared];

        MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
            [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
        geomDesc.vertexBuffer       = dvBuf;
        geomDesc.vertexBufferOffset = 0;
        geomDesc.vertexStride       = sizeof(float) * 3;
        geomDesc.vertexFormat       = MTLAttributeFormatFloat3;
        geomDesc.indexBuffer        = diBuf;
        geomDesc.indexBufferOffset  = 0;
        geomDesc.indexType          = MTLIndexTypeUInt32;
        geomDesc.triangleCount      = 1;
        geomDesc.opaque             = YES;

        MTLPrimitiveAccelerationStructureDescriptor* blasDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];
        blasDesc.geometryDescriptors = @[geomDesc];

        id<MTLAccelerationStructure> dummyBlas = buildAccelStructure(device, cmdQueue, blasDesc);
        if (!dummyBlas) {
            spdlog::error("MetalAccelStructure: dummy BLAS build failed");
            return;
        }
        [m_impl->blasArray addObject:dummyBlas];

        // Build TLAS with this single dummy instance.
        size_t instDescSize = sizeof(MTLAccelerationStructureInstanceDescriptor);
        id<MTLBuffer> instBuf = [device newBufferWithLength:instDescSize
                                                    options:MTLResourceStorageModeShared];
        MTLAccelerationStructureInstanceDescriptor* inst =
            (MTLAccelerationStructureInstanceDescriptor*)[instBuf contents];
        memset(inst, 0, instDescSize);
        inst->transformationMatrix.columns[0] = {1, 0, 0};
        inst->transformationMatrix.columns[1] = {0, 1, 0};
        inst->transformationMatrix.columns[2] = {0, 0, 1};
        inst->transformationMatrix.columns[3] = {0, 0, 0};
        inst->options                          = MTLAccelerationStructureInstanceOptionOpaque;
        inst->mask                             = 0xFF;
        inst->intersectionFunctionTableOffset  = 0;
        inst->accelerationStructureIndex       = 0;

        MTLInstanceAccelerationStructureDescriptor* tlasDesc =
            [MTLInstanceAccelerationStructureDescriptor descriptor];
        tlasDesc.instancedAccelerationStructures = m_impl->blasArray;
        tlasDesc.instanceCount                   = 1;
        tlasDesc.instanceDescriptorBuffer        = instBuf;

        m_impl->tlas = buildAccelStructure(device, cmdQueue, tlasDesc);
        if (!m_impl->tlas) {
            spdlog::error("MetalAccelStructure: dummy TLAS build failed");
            return;
        }
        m_impl->valid = true;
        spdlog::info("MetalAccelStructure: dummy BLAS+TLAS built for particles-only scene");
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

    std::vector<PackedFloat3> positions     (totalVerts);  // world-space at shutter-open
    std::vector<PackedFloat3> positionsClose(totalVerts);  // world-space at shutter-close
    std::vector<PackedFloat3> normals       (totalVerts);
    std::vector<PackedFloat2> uvs           (totalVerts);
    std::vector<uint32_t>     indices       (totalTris * 3);
    std::vector<uint32_t>     triMeshIDs   (totalTris);
    std::vector<uint32_t>     vertexOffsets(numMeshes);
    std::vector<uint32_t>     indexOffsets (numMeshes);

    uint32_t vBase = 0, tBase = 0;
    for (uint32_t mi = 0; mi < numMeshes; ++mi) {
        const MeshDesc& m = pool.mesh(mi);
        vertexOffsets[mi] = vBase;
        indexOffsets [mi] = tBase * 3;  // byte/element offset in index array

        for (uint32_t v = 0; v < m.numVertices(); ++v) {
            Vec3f p = m.positions[v];
            Vec3f n = m.normals.empty() ? Vec3f{0,1,0} : m.normals[v];
            PackedFloat3 pOpen, pClose;
            if (m.hasMotion()) {
                const Mat4f& xfOpen  = m.motionKeys.front().objectToWorld;
                const Mat4f& ixfOpen = m.motionKeys.front().worldToObject;
                const Mat4f& xfClose = m.motionKeys.back().objectToWorld;
                Vec3f woOpen  = xfOpen.transformPoint(p);
                Vec3f woClose = xfClose.transformPoint(p);
                pOpen  = {woOpen.x,  woOpen.y,  woOpen.z};
                pClose = {woClose.x, woClose.y, woClose.z};
                // Normal baked at shutter-open (acceptable approximation for rigid motion)
                n = ixfOpen.transformNormal(n);
                float len = n.length();
                if (len > 1e-6f) n = n * (1.f / len);
            } else {
                pOpen  = {p.x, p.y, p.z};
                pClose = {p.x, p.y, p.z};
            }
            positions     [vBase + v] = pOpen;
            positionsClose[vBase + v] = pClose;
            normals        [vBase + v] = {n.x, n.y, n.z};
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
        device, positions.data(),      totalVerts * sizeof(PackedFloat3), "positions");
    m_impl->positionBufferClose = makeSharedBuffer(
        device, positionsClose.data(), totalVerts * sizeof(PackedFloat3), "positionsClose");
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
    // Build one BLAS per mesh.
    // Animated meshes get a motion BLAS with 2 world-space vertex keyframes
    // (shutter-open and shutter-close). Static meshes get a regular BLAS.
    // The TLAS remains non-motion; primitive_motion interpolation happens within
    // each motion BLAS based on the ray time passed to intersect().
    // -----------------------------------------------------------------------
    for (uint32_t mi = 0; mi < numMeshes; ++mi) {
        const MeshDesc& m = pool.mesh(mi);

        MTLPrimitiveAccelerationStructureDescriptor* blasDesc =
            [MTLPrimitiveAccelerationStructureDescriptor descriptor];

        if (m.hasMotion()) {
            MTLAccelerationStructureMotionTriangleGeometryDescriptor* geomDesc =
                [MTLAccelerationStructureMotionTriangleGeometryDescriptor descriptor];

            MTLMotionKeyframeData* kf0 = [MTLMotionKeyframeData data];
            kf0.buffer = m_impl->positionBuffer;
            kf0.offset = 0;  // global indices have vertex base baked in

            MTLMotionKeyframeData* kf1 = [MTLMotionKeyframeData data];
            kf1.buffer = m_impl->positionBufferClose;
            kf1.offset = 0;

            geomDesc.vertexBuffers     = @[kf0, kf1];
            geomDesc.vertexStride      = sizeof(PackedFloat3);
            geomDesc.vertexFormat      = MTLAttributeFormatFloat3;
            geomDesc.indexBuffer       = m_impl->indexBuffer;
            geomDesc.indexBufferOffset = indexOffsets[mi] * sizeof(uint32_t);
            geomDesc.indexType         = MTLIndexTypeUInt32;
            geomDesc.triangleCount     = m.numTriangles();
            geomDesc.opaque            = YES;

            blasDesc.geometryDescriptors = @[geomDesc];
            blasDesc.motionKeyframeCount = 2;
            blasDesc.motionStartTime     = 0.0f;
            blasDesc.motionEndTime       = 1.0f;
        } else {
            MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
                [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];

            geomDesc.vertexBuffer       = m_impl->positionBuffer;
            geomDesc.vertexBufferOffset = 0;
            geomDesc.vertexStride       = sizeof(PackedFloat3);
            geomDesc.vertexFormat       = MTLAttributeFormatFloat3;
            geomDesc.indexBuffer        = m_impl->indexBuffer;
            geomDesc.indexBufferOffset  = indexOffsets[mi] * sizeof(uint32_t);
            geomDesc.indexType          = MTLIndexTypeUInt32;
            geomDesc.triangleCount      = m.numTriangles();
            geomDesc.opaque             = YES;

            blasDesc.geometryDescriptors = @[geomDesc];
        }

        id<MTLAccelerationStructure> blas = buildAccelStructure(device, cmdQueue, blasDesc);
        if (!blas) {
            spdlog::error("MetalAccelStructure: BLAS build failed for mesh {}", mi);
            return;
        }
        [m_impl->blasArray addObject:blas];
    }

    // -----------------------------------------------------------------------
    // Hair ribbon tessellation — build one BLAS for all hair strands.
    // Each cubic Bézier segment is tessellated into kHairTessSteps quads
    // (2 triangles each). Motion blur is handled via two position keyframes.
    // -----------------------------------------------------------------------
    uint32_t numHairTris = 0;
    if (curvePool && curvePool->numStrands() > 0) {
        std::vector<PackedFloat3> hairPosOpen, hairPosClose;
        std::vector<PackedFloat2> hairUVs;
        std::vector<PackedFloat3> hairNormals;
        std::vector<uint32_t>     hairIndices;
        std::vector<GpuHairTri>   hairTriData;

        bool anyHairMotion = false;
        uint32_t vBase = 0;

        for (size_t si = 0; si < curvePool->numStrands(); ++si) {
            const StrandDesc& strand = curvePool->strand(static_cast<uint32_t>(si));
            uint32_t numSeg = strand.numSegments();
            if (numSeg == 0) continue;

            bool hasMotion = strand.hasMotion();
            if (hasMotion) anyHairMotion = true;

            for (uint32_t seg = 0; seg < numSeg; ++seg) {
                const Vec3f* cvOpen  = &strand.controlPoints[seg * 3];
                const Vec3f* cvClose = hasMotion ? &strand.controlPointsClose[seg * 3] : nullptr;

                // Strand parameter range for this segment
                float v0 = float(seg)   / float(numSeg);
                float v1 = float(seg+1) / float(numSeg);

                float wRoot = strand.widthAt(v0) * 0.5f;
                float wTip  = strand.widthAt(v1) * 0.5f;

                tessellateSegment(cvOpen, cvClose,
                                  v0, v1, wRoot, wTip,
                                  strand.color,
                                  strand.materialIndex,
                                  vBase,
                                  hairTessSteps,
                                  hairPosOpen, hairPosClose,
                                  hairUVs, hairNormals,
                                  hairIndices, hairTriData);

                vBase += static_cast<uint32_t>(2 * (hairTessSteps + 1));
            }
        }

        numHairTris = static_cast<uint32_t>(hairTriData.size());

        if (numHairTris > 0) {
            size_t nV = hairPosOpen.size();
            size_t nI = hairIndices.size();

            m_impl->hairPosOpenBuf = makeSharedBuffer(device, hairPosOpen.data(),
                                                       nV * sizeof(PackedFloat3), "hairPosOpen");
            m_impl->hairPosCloseBuf = makeSharedBuffer(device, hairPosClose.data(),
                                                        nV * sizeof(PackedFloat3), "hairPosClose");
            m_impl->hairIndexBuf = makeSharedBuffer(device, hairIndices.data(),
                                                     nI * sizeof(uint32_t), "hairIndices");
            m_impl->hairTriBuf = makeSharedBuffer(device, hairTriData.data(),
                                                   numHairTris * sizeof(GpuHairTri), "hairTris");

            // Build hair BLAS (motion or static based on whether any strand had motion)
            MTLPrimitiveAccelerationStructureDescriptor* hairBlasDesc =
                [MTLPrimitiveAccelerationStructureDescriptor descriptor];

            if (anyHairMotion) {
                MTLAccelerationStructureMotionTriangleGeometryDescriptor* geomDesc =
                    [MTLAccelerationStructureMotionTriangleGeometryDescriptor descriptor];

                MTLMotionKeyframeData* kf0 = [MTLMotionKeyframeData data];
                kf0.buffer = m_impl->hairPosOpenBuf; kf0.offset = 0;
                MTLMotionKeyframeData* kf1 = [MTLMotionKeyframeData data];
                kf1.buffer = m_impl->hairPosCloseBuf; kf1.offset = 0;

                geomDesc.vertexBuffers     = @[kf0, kf1];
                geomDesc.vertexStride      = sizeof(PackedFloat3);
                geomDesc.vertexFormat      = MTLAttributeFormatFloat3;
                geomDesc.indexBuffer       = m_impl->hairIndexBuf;
                geomDesc.indexBufferOffset = 0;
                geomDesc.indexType         = MTLIndexTypeUInt32;
                geomDesc.triangleCount     = numHairTris;
                geomDesc.opaque            = YES;

                hairBlasDesc.geometryDescriptors = @[geomDesc];
                hairBlasDesc.motionKeyframeCount = 2;
                hairBlasDesc.motionStartTime     = 0.0f;
                hairBlasDesc.motionEndTime       = 1.0f;
            } else {
                MTLAccelerationStructureTriangleGeometryDescriptor* geomDesc =
                    [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];

                geomDesc.vertexBuffer       = m_impl->hairPosOpenBuf;
                geomDesc.vertexBufferOffset = 0;
                geomDesc.vertexStride       = sizeof(PackedFloat3);
                geomDesc.vertexFormat       = MTLAttributeFormatFloat3;
                geomDesc.indexBuffer        = m_impl->hairIndexBuf;
                geomDesc.indexBufferOffset  = 0;
                geomDesc.indexType          = MTLIndexTypeUInt32;
                geomDesc.triangleCount      = numHairTris;
                geomDesc.opaque             = YES;

                hairBlasDesc.geometryDescriptors = @[geomDesc];
            }

            id<MTLAccelerationStructure> hairBlas =
                buildAccelStructure(device, cmdQueue, hairBlasDesc);
            if (!hairBlas) {
                spdlog::error("MetalAccelStructure: hair BLAS build failed");
                return;
            }
            [m_impl->blasArray addObject:hairBlas];
            m_impl->hairMeshBase = numMeshes;  // hair is the next instance after all meshes

            spdlog::info("MetalAccelStructure: tessellated {} hair strands → {} tris (motion={})",
                         curvePool->numStrands(), numHairTris, anyHairMotion);
        }
    }

    // -----------------------------------------------------------------------
    // Build TLAS with identity transforms (geometry is already world-space).
    // Instance count = numMeshes + (1 if hair BLAS was built, else 0).
    // -----------------------------------------------------------------------
    uint32_t totalInstances = numMeshes + (m_impl->hairMeshBase != 0xFFFFFFFFu ? 1u : 0u);

    size_t instDescSize = sizeof(MTLAccelerationStructureInstanceDescriptor);
    id<MTLBuffer> instBuf = [device newBufferWithLength:instDescSize * totalInstances
                                               options:MTLResourceStorageModeShared];
    instBuf.label = @"tlasInstances";
    MTLAccelerationStructureInstanceDescriptor* instDescs =
        (MTLAccelerationStructureInstanceDescriptor*)[instBuf contents];

    auto makeIdentityInstance = [](uint32_t blasIdx) {
        MTLAccelerationStructureInstanceDescriptor d;
        memset(&d, 0, sizeof(d));
        d.transformationMatrix.columns[0] = {1, 0, 0};
        d.transformationMatrix.columns[1] = {0, 1, 0};
        d.transformationMatrix.columns[2] = {0, 0, 1};
        d.transformationMatrix.columns[3] = {0, 0, 0};
        d.options            = MTLAccelerationStructureInstanceOptionOpaque;
        d.mask               = 0xFF;
        d.intersectionFunctionTableOffset = 0;
        d.accelerationStructureIndex      = blasIdx;
        return d;
    };

    for (uint32_t i = 0; i < numMeshes; ++i)
        instDescs[i] = makeIdentityInstance(i);

    if (m_impl->hairMeshBase != 0xFFFFFFFFu)
        instDescs[numMeshes] = makeIdentityInstance(numMeshes);

    MTLInstanceAccelerationStructureDescriptor* tlasDesc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    tlasDesc.instancedAccelerationStructures = m_impl->blasArray;
    tlasDesc.instanceCount                   = totalInstances;
    tlasDesc.instanceDescriptorBuffer        = instBuf;

    m_impl->tlas = buildAccelStructure(device, cmdQueue, tlasDesc);
    if (!m_impl->tlas) {
        spdlog::error("MetalAccelStructure: TLAS build failed");
        return;
    }

    m_impl->valid = true;
    spdlog::info("MetalAccelStructure: built {} BLAS + TLAS ({} verts, {} tris, {} hair tris)",
                 totalInstances, totalVerts, totalTris, numHairTris);
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
void* MetalAccelStructure::hairTriBuffer()         const { return (__bridge void*)m_impl->hairTriBuf; }
void* MetalAccelStructure::hairMatBuffer()         const { return (__bridge void*)m_impl->hairMatBuf; }
uint32_t MetalAccelStructure::numHairMaterials()   const { return m_impl->numHairMats; }
uint32_t MetalAccelStructure::hairMeshBaseID()     const { return m_impl->hairMeshBase; }
void* MetalAccelStructure::uvBuffer()              const { return (__bridge void*)m_impl->uvBuffer; }
void* MetalAccelStructure::triMeshIDBuffer()       const { return (__bridge void*)m_impl->triMeshIDBuffer; }
void* MetalAccelStructure::meshVertexOffsetBuffer() const { return (__bridge void*)m_impl->meshVertexOffsetBuffer; }
void* MetalAccelStructure::meshIndexOffsetBuffer()  const { return (__bridge void*)m_impl->meshIndexOffsetBuffer; }
void* MetalAccelStructure::indexBuffer()            const { return (__bridge void*)m_impl->indexBuffer; }

} // namespace anacapa

#endif // ANACAPA_ENABLE_METAL
