#ifdef ANACAPA_ENABLE_CUDA

#include "CudaAccelStructure.h"
#include "CudaContext.h"
#include "CudaBuffer.h"
#include "shaders/SharedTypes.h"

#include <cuda_runtime.h>

#ifdef ANACAPA_ENABLE_OPTIX
#include <optix.h>
#include <optix_stubs.h>
#endif

#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <cstring>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[error] CUDA %s %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
} while(0)

#ifdef ANACAPA_ENABLE_OPTIX
#define OPTIX_CHECK(call) do { \
    OptixResult _r = (call); \
    if (_r != OPTIX_SUCCESS) \
        fprintf(stderr, "[error] OptiX %d (%s) at %s:%d\n", \
            int(_r), optixGetErrorName(_r), __FILE__, __LINE__); \
} while(0)
#endif

namespace anacapa {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct CudaAccelStructure::Impl {
    CudaByteBuffer        posBuffer;       // packed float3, world-space at shutter-open
    CudaByteBuffer        posBufferClose;  // packed float3, world-space at shutter-close
    CudaBuffer<GpuFloat3> normals;
    CudaBuffer<uint32_t>  indices;
    CudaBuffer<uint32_t>  triMeshIDs;
    CudaBuffer<uint32_t>  meshVertexOffsets;
    CudaBuffer<uint32_t>  meshIndexOffsets;

#ifdef ANACAPA_ENABLE_OPTIX
    // OptiX-built GAS storage.  Output buffer must outlive the handle.
    CudaByteBuffer         asBuffer;
    OptixTraversableHandle gasHandle = 0;
#endif

    uint32_t totalVertices  = 0;
    uint32_t totalTriangles = 0;
    uint32_t numMeshes_     = 0;
    bool     hasMotion      = false;
    bool     valid          = false;
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CudaAccelStructure::CudaAccelStructure(CudaContext& ctx, const GeometryPool& pool)
    : m_impl(std::make_unique<Impl>())
{
    uint32_t numMeshes = static_cast<uint32_t>(pool.numMeshes());
    m_impl->numMeshes_ = numMeshes;

    if (numMeshes == 0) {
        fprintf(stderr, "[error] CudaAccelStructure: empty geometry pool\n");
        return;
    }

    // -----------------------------------------------------------------------
    // Build concatenated CPU arrays.
    // Vertex positions are baked to world-space.  When a mesh has motionKeys,
    // the front and back keys' object-to-world transforms produce two
    // keyframes (positions + positionsClose); otherwise both keyframes are
    // identical and the GAS is built without motion options.
    // -----------------------------------------------------------------------
    uint32_t totalVerts = 0, totalTris = 0;
    for (uint32_t i = 0; i < numMeshes; ++i) {
        totalVerts += pool.mesh(i).numVertices();
        totalTris  += pool.mesh(i).numTriangles();
        if (pool.mesh(i).hasMotion()) m_impl->hasMotion = true;
    }

    std::vector<float>     positions     (totalVerts * 3);  // world-space @ shutterOpen
    std::vector<float>     positionsClose(totalVerts * 3);  // world-space @ shutterClose
    std::vector<GpuFloat3> normals  (totalVerts);
    std::vector<uint32_t>  indices  (totalTris * 3);
    std::vector<uint32_t>  triMeshIDs   (totalTris);
    std::vector<uint32_t>  vertexOffsets(numMeshes);
    std::vector<uint32_t>  indexOffsets (numMeshes);

    uint32_t vBase = 0, tBase = 0;
    for (uint32_t mi = 0; mi < numMeshes; ++mi) {
        const MeshDesc& m = pool.mesh(mi);
        vertexOffsets[mi] = vBase;
        indexOffsets [mi] = tBase * 3;

        const bool   meshMotion = m.hasMotion();
        const Mat4f  xfOpen     = meshMotion ? m.motionKeys.front().objectToWorld : Mat4f{};
        const Mat4f  xfClose    = meshMotion ? m.motionKeys.back ().objectToWorld : Mat4f{};
        const Mat4f& ixfOpen    = meshMotion ? m.motionKeys.front().worldToObject : Mat4f{};

        for (uint32_t v = 0; v < m.numVertices(); ++v) {
            const Vec3f p = m.positions[v];
            Vec3f pOpen, pClose;
            if (meshMotion) {
                pOpen  = xfOpen .transformPoint(p);
                pClose = xfClose.transformPoint(p);
            } else {
                pOpen  = p;
                pClose = p;
            }
            positions     [(vBase + v) * 3 + 0] = pOpen.x;
            positions     [(vBase + v) * 3 + 1] = pOpen.y;
            positions     [(vBase + v) * 3 + 2] = pOpen.z;
            positionsClose[(vBase + v) * 3 + 0] = pClose.x;
            positionsClose[(vBase + v) * 3 + 1] = pClose.y;
            positionsClose[(vBase + v) * 3 + 2] = pClose.z;

            // Normal baked at shutter-open — acceptable for rigid motion.
            Vec3f n = m.normals.empty() ? Vec3f{0, 1, 0} : m.normals[v];
            if (meshMotion) {
                n = ixfOpen.transformNormal(n);
                float len = n.length();
                if (len > 1e-6f) n = n * (1.f / len);
            }
            normals[vBase + v] = GpuFloat3{n.x, n.y, n.z};
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
    // Upload to GPU
    // -----------------------------------------------------------------------
    const size_t vbBytes = totalVerts * 3 * sizeof(float);

    m_impl->posBuffer = CudaByteBuffer(vbBytes);
    m_impl->posBuffer.upload(reinterpret_cast<const uint8_t*>(positions.data()), vbBytes);

    m_impl->posBufferClose = CudaByteBuffer(vbBytes);
    m_impl->posBufferClose.upload(reinterpret_cast<const uint8_t*>(positionsClose.data()), vbBytes);

    m_impl->normals = CudaBuffer<GpuFloat3>(totalVerts);
    m_impl->normals.upload(normals);

    m_impl->indices = CudaBuffer<uint32_t>(totalTris * 3);
    m_impl->indices.upload(indices);

    m_impl->triMeshIDs = CudaBuffer<uint32_t>(totalTris);
    m_impl->triMeshIDs.upload(triMeshIDs);

    m_impl->meshVertexOffsets = CudaBuffer<uint32_t>(numMeshes);
    m_impl->meshVertexOffsets.upload(vertexOffsets);

    m_impl->meshIndexOffsets = CudaBuffer<uint32_t>(numMeshes);
    m_impl->meshIndexOffsets.upload(indexOffsets);

#ifdef ANACAPA_ENABLE_OPTIX
    // -----------------------------------------------------------------------
    // OptiX GAS — single triangle GAS over all meshes' world-space vertices.
    // When any mesh has motion keys, build with two-keyframe motion options
    // (vertexBuffers[0] = open, vertexBuffers[1] = close); otherwise build
    // a static GAS with a single vertex buffer.  Per-mesh material dispatch
    // happens via the existing triMeshIDs buffer (Step 4 will replace this
    // with SBT records when the kernel is split).
    // -----------------------------------------------------------------------
    OptixDeviceContext optixCtx =
        static_cast<OptixDeviceContext>(ctx.optixContext());
    if (optixCtx) {
        CUstream stream = static_cast<CUstream>(ctx.cuStream());

        const CUdeviceptr vbOpen  = static_cast<CUdeviceptr>(m_impl->posBuffer.devPtr());
        const CUdeviceptr vbClose = static_cast<CUdeviceptr>(m_impl->posBufferClose.devPtr());
        const CUdeviceptr ibDev   = static_cast<CUdeviceptr>(m_impl->indices.devPtr());

        CUdeviceptr vbArr[2] = { vbOpen, vbClose };

        OptixBuildInput buildInput{};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto& tri = buildInput.triangleArray;
        tri.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        tri.vertexStrideInBytes = sizeof(float) * 3;
        tri.numVertices         = totalVerts;
        tri.vertexBuffers       = vbArr;  // first numKeys entries used
        tri.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        tri.indexStrideInBytes  = sizeof(uint32_t) * 3;
        tri.numIndexTriplets    = totalTris;
        tri.indexBuffer         = ibDev;

        const uint32_t geomFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
        tri.flags         = geomFlags;
        tri.numSbtRecords = 1;

        OptixAccelBuildOptions accelOpts{};
        accelOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                             | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;
        if (m_impl->hasMotion) {
            accelOpts.motionOptions.numKeys   = 2;
            accelOpts.motionOptions.timeBegin = 0.0f;
            accelOpts.motionOptions.timeEnd   = 1.0f;
            accelOpts.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
        } else {
            accelOpts.motionOptions.numKeys = 1;
        }

        OptixAccelBufferSizes sizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optixCtx, &accelOpts, &buildInput, /*numBuildInputs=*/1, &sizes));

        // Temp buffer (freed after build) and a property slot to query the
        // compacted size.
        CudaByteBuffer tempBuf(sizes.tempSizeInBytes);
        CudaByteBuffer outBufUncompacted(sizes.outputSizeInBytes);
        CudaBuffer<uint64_t> compactedSizeBuf(1);

        OptixAccelEmitDesc emit{};
        emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = static_cast<CUdeviceptr>(compactedSizeBuf.devPtr());

        OptixTraversableHandle handleUncompacted = 0;
        OPTIX_CHECK(optixAccelBuild(
            optixCtx, stream, &accelOpts, &buildInput, 1,
            static_cast<CUdeviceptr>(tempBuf.devPtr()), sizes.tempSizeInBytes,
            static_cast<CUdeviceptr>(outBufUncompacted.devPtr()), sizes.outputSizeInBytes,
            &handleUncompacted, &emit, 1));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        std::vector<uint64_t> compactedSizeHost(1);
        compactedSizeBuf.download(compactedSizeHost);
        const size_t compactedSize = static_cast<size_t>(compactedSizeHost[0]);

        if (compactedSize > 0 && compactedSize < sizes.outputSizeInBytes) {
            m_impl->asBuffer = CudaByteBuffer(compactedSize);
            OPTIX_CHECK(optixAccelCompact(
                optixCtx, stream, handleUncompacted,
                static_cast<CUdeviceptr>(m_impl->asBuffer.devPtr()), compactedSize,
                &m_impl->gasHandle));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            // Compaction wouldn't help — keep the uncompacted buffer.
            m_impl->asBuffer  = std::move(outBufUncompacted);
            m_impl->gasHandle = handleUncompacted;
        }
        printf("[info]  CudaAccelStructure: OptiX GAS built (%s, %u verts, %u tris, %.2f KiB)\n",
               m_impl->hasMotion ? "motion-aware" : "static",
               totalVerts, totalTris,
               m_impl->asBuffer.byteSize() / 1024.0);
    }
#endif

    m_impl->valid = true;
}

CudaAccelStructure::~CudaAccelStructure() = default;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
bool     CudaAccelStructure::isValid()        const { return m_impl->valid; }
uint32_t CudaAccelStructure::totalVertices()  const { return m_impl->totalVertices; }
uint32_t CudaAccelStructure::totalTriangles() const { return m_impl->totalTriangles; }
uint32_t CudaAccelStructure::numMeshes()      const { return m_impl->numMeshes_; }

uint64_t CudaAccelStructure::positionBuffer()         const { return m_impl->posBuffer.devPtr(); }
uint64_t CudaAccelStructure::positionBufferClose()    const { return m_impl->posBufferClose.devPtr(); }
uint64_t CudaAccelStructure::normalBuffer()           const { return m_impl->normals.devPtr(); }
uint64_t CudaAccelStructure::indexBuffer()            const { return m_impl->indices.devPtr(); }
uint64_t CudaAccelStructure::triMeshIDBuffer()        const { return m_impl->triMeshIDs.devPtr(); }
uint64_t CudaAccelStructure::meshVertexOffsetBuffer() const { return m_impl->meshVertexOffsets.devPtr(); }
uint64_t CudaAccelStructure::meshIndexOffsetBuffer()  const { return m_impl->meshIndexOffsets.devPtr(); }
bool     CudaAccelStructure::hasMotion()              const { return m_impl->hasMotion; }
uint64_t CudaAccelStructure::traversableHandle()      const {
#ifdef ANACAPA_ENABLE_OPTIX
    return static_cast<uint64_t>(m_impl->gasHandle);
#else
    return 0;
#endif
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
