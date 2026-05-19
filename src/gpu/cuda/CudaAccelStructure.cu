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

    // Hair ribbon geometry (separate from triangle buffers).
    CudaByteBuffer            hairPosOpenBuf;
    CudaByteBuffer            hairPosCloseBuf;
    CudaBuffer<uint32_t>      hairIndexBuf;
    CudaBuffer<GpuHairTri>    hairTriBuf;
    uint32_t                  numHairTris    = 0;
    uint32_t                  hairMeshBase   = 0xFFFFFFFFu;  // IAS instance ID; sentinel = no hair
    bool                      hairHasMotion  = false;

#ifdef ANACAPA_ENABLE_OPTIX
    // OptiX-built AS storage.  Output buffers must outlive the handles.
    CudaByteBuffer         meshAsBuffer;
    OptixTraversableHandle meshGasHandle = 0;
    CudaByteBuffer         hairAsBuffer;
    OptixTraversableHandle hairGasHandle = 0;
    CudaByteBuffer         iasBuffer;
    OptixTraversableHandle iasHandle     = 0;
    CudaByteBuffer         iasInstanceBuf;
#endif

    uint32_t totalVertices  = 0;
    uint32_t totalTriangles = 0;
    uint32_t numMeshes_     = 0;
    bool     hasMotion      = false;
    bool     valid          = false;
};

#ifdef ANACAPA_ENABLE_OPTIX
// ---------------------------------------------------------------------------
// Hair tessellation helpers — mirror MetalAccelStructure.mm:
//   per cubic Bézier segment, sample (hairTessSteps+1) ribbon cross-sections
//   and stitch them into hairTessSteps quads (= 2*hairTessSteps triangles).
//   hairTessSteps is plumbed in via the constructor (RenderSettings ->
//   SceneView -> CudaPathIntegrator -> here).
// ---------------------------------------------------------------------------

struct PackedFloat3 { float x, y, z; };

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

static void tessellateSegment(
    const Vec3f* cvOpen,
    const Vec3f* cvClose,        // nullptr = same as cvOpen
    float        strandV0,
    float        strandV1,
    float        wRoot,           // diameter at strandV0
    float        wTip,            // diameter at strandV1
    Vec3f        strandColor,
    uint32_t     matIdx,
    uint32_t     vBase,
    int          hairTessSteps,
    std::vector<PackedFloat3>& posOpen,
    std::vector<PackedFloat3>& posClose,
    std::vector<uint32_t>&     indices,
    std::vector<GpuHairTri>&   hairTris)
{
    for (int k = 0; k <= hairTessSteps; ++k) {
        float t = float(k) / float(hairTessSteps);
        float w = (wRoot * (1.f - t) + wTip * t) * 0.5f;  // half-width (radius)
        if (w < 5e-5f) w = 5e-5f;                          // robust intersection

        Vec3f posO = bezierPoint(cvOpen, t);
        Vec3f posC = cvClose ? bezierPoint(cvClose, t) : posO;
        Vec3f tang = bezierTangent(cvOpen, t);

        Vec3f refUp = (std::abs(tang.y) > 0.9f) ? Vec3f{1.f,0.f,0.f}
                                                : Vec3f{0.f,1.f,0.f};
        Vec3f perp  = normalizeVec(crossVec(tang, refUp));

        Vec3f lO = {posO.x - perp.x*w, posO.y - perp.y*w, posO.z - perp.z*w};
        Vec3f rO = {posO.x + perp.x*w, posO.y + perp.y*w, posO.z + perp.z*w};
        Vec3f lC = {posC.x - perp.x*w, posC.y - perp.y*w, posC.z - perp.z*w};
        Vec3f rC = {posC.x + perp.x*w, posC.y + perp.y*w, posC.z + perp.z*w};

        posOpen.push_back ({lO.x, lO.y, lO.z});
        posOpen.push_back ({rO.x, rO.y, rO.z});
        posClose.push_back({lC.x, lC.y, lC.z});
        posClose.push_back({rC.x, rC.y, rC.z});
    }

    for (int k = 0; k < hairTessSteps; ++k) {
        uint32_t l0 = vBase + uint32_t(2*k + 0);
        uint32_t r0 = vBase + uint32_t(2*k + 1);
        uint32_t l1 = vBase + uint32_t(2*k + 2);
        uint32_t r1 = vBase + uint32_t(2*k + 3);

        float tMid = (float(k) + 0.5f) / float(hairTessSteps);
        Vec3f tang = bezierTangent(cvOpen, tMid);

        GpuHairTri ht{};
        ht.tangent = {tang.x, tang.y, tang.z};
        ht.matIdx  = matIdx;
        ht.color   = {strandColor.x, strandColor.y, strandColor.z};

        // Tri 0: (l0, r0, r1) → h = (-1, +1, +1)
        indices.push_back(l0); indices.push_back(r0); indices.push_back(r1);
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = +1.f;
        hairTris.push_back(ht);

        // Tri 1: (l0, r1, l1) → h = (-1, +1, -1)
        indices.push_back(l0); indices.push_back(r1); indices.push_back(l1);
        ht.h0 = -1.f; ht.h1 = +1.f; ht.h2 = -1.f;
        hairTris.push_back(ht);
    }
}

// Build one triangle GAS (motion-aware when both vbClose and vbOpen differ).
// Returns the (handle, output buffer) pair via out-arguments; the buffer
// must outlive the handle.  Returns true on success.
static bool buildTriangleGAS(
    OptixDeviceContext optixCtx,
    CUstream           stream,
    CUdeviceptr        vbOpen,
    CUdeviceptr        vbClose,
    bool               motion,
    uint32_t           numVerts,
    CUdeviceptr        ibDev,
    uint32_t           numTris,
    CudaByteBuffer&    outBuf,
    OptixTraversableHandle& outHandle)
{
    CUdeviceptr vbArr[2] = { vbOpen, vbClose };

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    auto& tri = buildInput.triangleArray;
    tri.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    tri.vertexStrideInBytes = sizeof(float) * 3;
    tri.numVertices         = numVerts;
    tri.vertexBuffers       = vbArr;
    tri.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    tri.indexStrideInBytes  = sizeof(uint32_t) * 3;
    tri.numIndexTriplets    = numTris;
    tri.indexBuffer         = ibDev;

    const uint32_t geomFlags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    tri.flags         = geomFlags;
    tri.numSbtRecords = 1;

    OptixAccelBuildOptions accelOpts{};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                         | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;
    if (motion) {
        accelOpts.motionOptions.numKeys   = 2;
        accelOpts.motionOptions.timeBegin = 0.0f;
        accelOpts.motionOptions.timeEnd   = 1.0f;
        accelOpts.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
    } else {
        accelOpts.motionOptions.numKeys = 1;
    }

    OptixAccelBufferSizes sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optixCtx, &accelOpts, &buildInput, 1, &sizes));

    CudaByteBuffer tempBuf(sizes.tempSizeInBytes);
    CudaByteBuffer uncompactedBuf(sizes.outputSizeInBytes);
    CudaBuffer<uint64_t> compactedSizeBuf(1);
    // Fail cleanly if any of the build allocations failed (e.g. low-VRAM
    // GPU + a giant hair tessellation).  Callers can then skip hair and
    // continue with the mesh-only path instead of corrupting OptiX state.
    if (!tempBuf.isValid() || !uncompactedBuf.isValid() || !compactedSizeBuf.isValid()) {
        fprintf(stderr, "[error] CudaAccelStructure: GAS build OOM "
                        "(temp=%zu KiB, output=%zu KiB)\n",
                sizes.tempSizeInBytes / 1024,
                sizes.outputSizeInBytes / 1024);
        outHandle = 0;
        return false;
    }

    OptixAccelEmitDesc emit{};
    emit.type   = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit.result = static_cast<CUdeviceptr>(compactedSizeBuf.devPtr());

    OptixTraversableHandle hUncompacted = 0;
    OPTIX_CHECK(optixAccelBuild(
        optixCtx, stream, &accelOpts, &buildInput, 1,
        static_cast<CUdeviceptr>(tempBuf.devPtr()), sizes.tempSizeInBytes,
        static_cast<CUdeviceptr>(uncompactedBuf.devPtr()), sizes.outputSizeInBytes,
        &hUncompacted, &emit, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint64_t> hostCompactedSize(1);
    compactedSizeBuf.download(hostCompactedSize);
    size_t compactedSize = static_cast<size_t>(hostCompactedSize[0]);

    if (compactedSize > 0 && compactedSize < sizes.outputSizeInBytes) {
        outBuf = CudaByteBuffer(compactedSize);
        OPTIX_CHECK(optixAccelCompact(
            optixCtx, stream, hUncompacted,
            static_cast<CUdeviceptr>(outBuf.devPtr()), compactedSize,
            &outHandle));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        outBuf    = std::move(uncompactedBuf);
        outHandle = hUncompacted;
    }
    return true;
}
#endif  // ANACAPA_ENABLE_OPTIX

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CudaAccelStructure::CudaAccelStructure(CudaContext& ctx, const GeometryPool& pool,
                                        const CurvePool* curvePool,
                                        int hairTessSteps)
    : m_impl(std::make_unique<Impl>())
{
    if (hairTessSteps < 1)  hairTessSteps = 1;
    if (hairTessSteps > 32) hairTessSteps = 32;
    uint32_t numMeshes = static_cast<uint32_t>(pool.numMeshes());
    m_impl->numMeshes_ = numMeshes;

    if (numMeshes == 0) {
        // Particles-only scene — no triangle geometry at all.  Build a single
        // dummy GAS (one degenerate triangle far from the scene) so the IAS is
        // valid.  The shader will never intersect this triangle, but the
        // wavefront raygen needs a non-null traversable to call optixTrace on.
        printf("[info]  CudaAccelStructure: no mesh geometry — building dummy GAS for particles-only scene\n");

        const float    dv[9]   = {1e20f, 0.f, 0.f,  0.f, 1e20f, 0.f,  0.f, 0.f, 1e20f};
        const uint32_t di[3]   = {0u, 1u, 2u};

        m_impl->posBuffer = CudaByteBuffer(sizeof(dv));
        m_impl->posBuffer.upload(reinterpret_cast<const uint8_t*>(dv), sizeof(dv));
        m_impl->posBufferClose = CudaByteBuffer(sizeof(dv));
        m_impl->posBufferClose.upload(reinterpret_cast<const uint8_t*>(dv), sizeof(dv));

        m_impl->normals = CudaBuffer<GpuFloat3>(1);
        std::vector<GpuFloat3> dummyN{ GpuFloat3{0.f, 1.f, 0.f} };
        m_impl->normals.upload(dummyN);

        m_impl->indices = CudaBuffer<uint32_t>(3);
        m_impl->indices.upload(std::vector<uint32_t>(di, di + 3));

        m_impl->triMeshIDs = CudaBuffer<uint32_t>(1);
        m_impl->triMeshIDs.upload(std::vector<uint32_t>{0u});

        m_impl->meshVertexOffsets = CudaBuffer<uint32_t>(1);
        m_impl->meshVertexOffsets.upload(std::vector<uint32_t>{0u});

        m_impl->meshIndexOffsets = CudaBuffer<uint32_t>(1);
        m_impl->meshIndexOffsets.upload(std::vector<uint32_t>{0u});

        m_impl->totalVertices  = 3;
        m_impl->totalTriangles = 1;

#ifdef ANACAPA_ENABLE_OPTIX
        OptixDeviceContext optixCtx =
            static_cast<OptixDeviceContext>(ctx.optixContext());
        if (optixCtx) {
            CUstream stream = static_cast<CUstream>(ctx.cuStream());

            if (!buildTriangleGAS(optixCtx, stream,
                                  static_cast<CUdeviceptr>(m_impl->posBuffer.devPtr()),
                                  static_cast<CUdeviceptr>(m_impl->posBufferClose.devPtr()),
                                  false, 3,
                                  static_cast<CUdeviceptr>(m_impl->indices.devPtr()),
                                  1,
                                  m_impl->meshAsBuffer, m_impl->meshGasHandle)) {
                fprintf(stderr, "[error] CudaAccelStructure: dummy GAS build failed\n");
                return;
            }

            OptixInstance inst{};
            inst.transform[0]  = 1.f; inst.transform[5]  = 1.f; inst.transform[10] = 1.f;
            inst.instanceId        = 0;
            inst.sbtOffset         = 0;
            inst.visibilityMask    = 0xFF;
            inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = m_impl->meshGasHandle;

            m_impl->iasInstanceBuf = CudaByteBuffer(sizeof(OptixInstance));
            m_impl->iasInstanceBuf.upload(
                reinterpret_cast<const uint8_t*>(&inst), sizeof(OptixInstance));

            OptixBuildInput iasInput{};
            iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            iasInput.instanceArray.instances    =
                static_cast<CUdeviceptr>(m_impl->iasInstanceBuf.devPtr());
            iasInput.instanceArray.numInstances = 1;

            OptixAccelBuildOptions iasOpts{};
            iasOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
            iasOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;
            iasOpts.motionOptions.numKeys = 1;

            OptixAccelBufferSizes iasSizes{};
            OPTIX_CHECK(optixAccelComputeMemoryUsage(
                optixCtx, &iasOpts, &iasInput, 1, &iasSizes));

            CudaByteBuffer iasTempBuf(iasSizes.tempSizeInBytes);
            m_impl->iasBuffer = CudaByteBuffer(iasSizes.outputSizeInBytes);
            OPTIX_CHECK(optixAccelBuild(
                optixCtx, stream, &iasOpts, &iasInput, 1,
                static_cast<CUdeviceptr>(iasTempBuf.devPtr()), iasSizes.tempSizeInBytes,
                static_cast<CUdeviceptr>(m_impl->iasBuffer.devPtr()),
                iasSizes.outputSizeInBytes,
                &m_impl->iasHandle, nullptr, 0));
            CUDA_CHECK(cudaStreamSynchronize(stream));

            printf("[info]  CudaAccelStructure: dummy GAS+IAS built for particles-only scene\n");
        }
#endif

        m_impl->valid = true;
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
    OptixDeviceContext optixCtx =
        static_cast<OptixDeviceContext>(ctx.optixContext());
    if (optixCtx) {
        CUstream stream = static_cast<CUstream>(ctx.cuStream());

        // -------------------------------------------------------------------
        // 1) Triangle (mesh) GAS — one GAS over all scene triangles.
        // Per-mesh material dispatch happens via triMeshIDs in the shader.
        // -------------------------------------------------------------------
        buildTriangleGAS(optixCtx, stream,
                          static_cast<CUdeviceptr>(m_impl->posBuffer.devPtr()),
                          static_cast<CUdeviceptr>(m_impl->posBufferClose.devPtr()),
                          m_impl->hasMotion,
                          totalVerts,
                          static_cast<CUdeviceptr>(m_impl->indices.devPtr()),
                          totalTris,
                          m_impl->meshAsBuffer, m_impl->meshGasHandle);

        printf("[info]  CudaAccelStructure: mesh GAS (%s, %u verts, %u tris, %.2f KiB)\n",
               m_impl->hasMotion ? "motion-aware" : "static",
               totalVerts, totalTris,
               m_impl->meshAsBuffer.byteSize() / 1024.0);

        // -------------------------------------------------------------------
        // 2) Hair GAS — tessellate strands into ribbon quads and build a
        // second triangle GAS.  Matches the algorithm in MetalAccelStructure.
        // -------------------------------------------------------------------
        if (curvePool && curvePool->numStrands() > 0) {
            std::vector<PackedFloat3> hairPosOpen, hairPosClose;
            std::vector<uint32_t>     hairIndices;
            std::vector<GpuHairTri>   hairTriData;
            bool anyHairMotion = false;
            uint32_t vBase = 0;

            for (size_t si = 0; si < curvePool->numStrands(); ++si) {
                const StrandDesc& strand =
                    curvePool->strand(static_cast<uint32_t>(si));
                uint32_t numSeg = strand.numSegments();
                if (numSeg == 0) continue;

                bool hasMotion = strand.hasMotion();
                if (hasMotion) anyHairMotion = true;

                for (uint32_t seg = 0; seg < numSeg; ++seg) {
                    const Vec3f* cvOpen  = &strand.controlPoints[seg * 3];
                    const Vec3f* cvClose = hasMotion
                                         ? &strand.controlPointsClose[seg * 3]
                                         : nullptr;
                    float v0 = float(seg)     / float(numSeg);
                    float v1 = float(seg + 1) / float(numSeg);
                    float wRoot = strand.widthAt(v0);
                    float wTip  = strand.widthAt(v1);

                    tessellateSegment(cvOpen, cvClose, v0, v1, wRoot, wTip,
                                      strand.color, strand.materialIndex,
                                      vBase, hairTessSteps,
                                      hairPosOpen, hairPosClose,
                                      hairIndices, hairTriData);

                    vBase += static_cast<uint32_t>(2 * (hairTessSteps + 1));
                }
            }

            uint32_t numHairTris = static_cast<uint32_t>(hairTriData.size());
            if (numHairTris > 0) {
                size_t nV = hairPosOpen.size();
                size_t nI = hairIndices.size();

                // Upload all hair buffers.
                const size_t hairVBBytes = nV * sizeof(PackedFloat3);
                m_impl->hairPosOpenBuf  = CudaByteBuffer(hairVBBytes);
                m_impl->hairPosOpenBuf.upload(
                    reinterpret_cast<const uint8_t*>(hairPosOpen.data()),
                    hairVBBytes);
                m_impl->hairPosCloseBuf = CudaByteBuffer(hairVBBytes);
                m_impl->hairPosCloseBuf.upload(
                    reinterpret_cast<const uint8_t*>(hairPosClose.data()),
                    hairVBBytes);

                m_impl->hairIndexBuf = CudaBuffer<uint32_t>(nI);
                m_impl->hairIndexBuf.upload(hairIndices);

                m_impl->hairTriBuf = CudaBuffer<GpuHairTri>(numHairTris);
                m_impl->hairTriBuf.upload(hairTriData);

                bool hairOk = buildTriangleGAS(optixCtx, stream,
                                  static_cast<CUdeviceptr>(m_impl->hairPosOpenBuf.devPtr()),
                                  static_cast<CUdeviceptr>(m_impl->hairPosCloseBuf.devPtr()),
                                  anyHairMotion,
                                  static_cast<uint32_t>(nV),
                                  static_cast<CUdeviceptr>(m_impl->hairIndexBuf.devPtr()),
                                  numHairTris,
                                  m_impl->hairAsBuffer, m_impl->hairGasHandle);

                if (hairOk) {
                    m_impl->numHairTris   = numHairTris;
                    m_impl->hairHasMotion = anyHairMotion;
                    m_impl->hairMeshBase  = 1;  // IAS instance ID (mesh = 0, hair = 1)
                    printf("[info]  CudaAccelStructure: hair GAS (%s, %zu strands, %u tris, %.2f KiB)\n",
                           anyHairMotion ? "motion-aware" : "static",
                           curvePool->numStrands(), numHairTris,
                           m_impl->hairAsBuffer.byteSize() / 1024.0);
                } else {
                    // Hair GAS too big for this GPU; release tessellation
                    // buffers and fall through to mesh-only rendering.
                    m_impl->hairPosOpenBuf  = CudaByteBuffer{};
                    m_impl->hairPosCloseBuf = CudaByteBuffer{};
                    m_impl->hairIndexBuf    = CudaBuffer<uint32_t>{};
                    m_impl->hairTriBuf      = CudaBuffer<GpuHairTri>{};
                    m_impl->hairAsBuffer    = CudaByteBuffer{};
                    m_impl->hairGasHandle   = 0;
                    fprintf(stderr, "[warn]  CudaAccelStructure: hair GAS build failed "
                                    "(%zu strands, %u tris) — rendering without hair\n",
                            curvePool->numStrands(), numHairTris);
                }
            }
        }

        // -------------------------------------------------------------------
        // 3) Build an IAS over the (one or two) GASes.  Always wrap the mesh
        // GAS in an IAS so the rest of the pipeline gets a uniform traversable.
        // -------------------------------------------------------------------
        const uint32_t numInstances = (m_impl->hairMeshBase != 0xFFFFFFFFu) ? 2u : 1u;

        const bool iasMotion = m_impl->hasMotion || m_impl->hairHasMotion;
        std::vector<OptixInstance> insts(numInstances);
        memset(insts.data(), 0, numInstances * sizeof(OptixInstance));
        auto fillInstance = [&](OptixInstance& inst, uint32_t id,
                                 OptixTraversableHandle child) {
            inst.transform[0]  = 1.f; inst.transform[1] = 0.f; inst.transform[2]  = 0.f; inst.transform[3]  = 0.f;
            inst.transform[4]  = 0.f; inst.transform[5] = 1.f; inst.transform[6]  = 0.f; inst.transform[7]  = 0.f;
            inst.transform[8]  = 0.f; inst.transform[9] = 0.f; inst.transform[10] = 1.f; inst.transform[11] = 0.f;
            inst.instanceId    = id;
            inst.sbtOffset     = 0;
            inst.visibilityMask = 0xFF;
            inst.flags         = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = child;
        };
        fillInstance(insts[0], 0, m_impl->meshGasHandle);
        if (numInstances == 2)
            fillInstance(insts[1], 1, m_impl->hairGasHandle);

        const size_t instBytes = numInstances * sizeof(OptixInstance);
        m_impl->iasInstanceBuf = CudaByteBuffer(instBytes);
        m_impl->iasInstanceBuf.upload(
            reinterpret_cast<const uint8_t*>(insts.data()), instBytes);

        OptixBuildInput iasInput{};
        iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        iasInput.instanceArray.instances    =
            static_cast<CUdeviceptr>(m_impl->iasInstanceBuf.devPtr());
        iasInput.instanceArray.numInstances = numInstances;

        OptixAccelBuildOptions iasOpts{};
        iasOpts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
        iasOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;
        if (iasMotion) {
            iasOpts.motionOptions.numKeys   = 2;
            iasOpts.motionOptions.timeBegin = 0.f;
            iasOpts.motionOptions.timeEnd   = 1.f;
            iasOpts.motionOptions.flags     = OPTIX_MOTION_FLAG_NONE;
        } else {
            iasOpts.motionOptions.numKeys = 1;
        }

        OptixAccelBufferSizes iasSizes{};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optixCtx, &iasOpts, &iasInput, 1, &iasSizes));

        CudaByteBuffer iasTempBuf(iasSizes.tempSizeInBytes);
        m_impl->iasBuffer = CudaByteBuffer(iasSizes.outputSizeInBytes);
        OPTIX_CHECK(optixAccelBuild(
            optixCtx, stream, &iasOpts, &iasInput, 1,
            static_cast<CUdeviceptr>(iasTempBuf.devPtr()), iasSizes.tempSizeInBytes,
            static_cast<CUdeviceptr>(m_impl->iasBuffer.devPtr()),
            iasSizes.outputSizeInBytes,
            &m_impl->iasHandle, nullptr, 0));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        printf("[info]  CudaAccelStructure: IAS built (%u instance%s, %.2f KiB)\n",
               numInstances, numInstances == 1 ? "" : "s",
               m_impl->iasBuffer.byteSize() / 1024.0);
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
bool     CudaAccelStructure::hasMotion()              const { return m_impl->hasMotion || m_impl->hairHasMotion; }
uint64_t CudaAccelStructure::hairTriBuffer()          const { return m_impl->hairTriBuf.isValid() ? m_impl->hairTriBuf.devPtr() : 0u; }
uint32_t CudaAccelStructure::hairMeshBaseID()         const { return m_impl->hairMeshBase; }
uint64_t CudaAccelStructure::traversableHandle()      const {
#ifdef ANACAPA_ENABLE_OPTIX
    return static_cast<uint64_t>(m_impl->iasHandle);
#else
    return 0;
#endif
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
