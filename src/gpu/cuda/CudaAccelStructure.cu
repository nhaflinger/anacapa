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
    CudaBuffer<GpuFloat2> uvs;
    CudaBuffer<GpuFloat4> tangents;
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
    uint32_t                  hairMeshBase   = 0xFFFFFFFFu;  // virtual hair meshID; sentinel = no hair
    bool                      hairHasMotion  = false;

#ifdef ANACAPA_ENABLE_OPTIX
    // OptiX-built AS storage.  Output buffers must outlive the handles.
    // One GAS per pool mesh (mirrors Metal's per-mesh BLAS layout); referenced
    // by IAS instances via traversableHandle.  For particles-only scenes a
    // single dummy entry is created so the IAS is non-empty.
    std::vector<CudaByteBuffer>         perMeshGasBufs;
    std::vector<OptixTraversableHandle> perMeshGasHandles;
    CudaByteBuffer         hairAsBuffer;
    OptixTraversableHandle hairGasHandle = 0;
    CudaByteBuffer         iasBuffer;
    OptixTraversableHandle iasHandle     = 0;
    CudaByteBuffer         iasInstanceBuf;

    // Per-IAS-instance lookup buffers (see header for layout).
    CudaBuffer<uint32_t>    instanceMeshIDs;
    CudaBuffer<float>       instanceNormalMat;
    CudaBuffer<float>       instanceTangentMat;
    CudaBuffer<float>       instancePositionMat;
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

        m_impl->uvs = CudaBuffer<GpuFloat2>(1);
        m_impl->uvs.upload(std::vector<GpuFloat2>{ GpuFloat2{0.f, 0.f} });

        m_impl->tangents = CudaBuffer<GpuFloat4>(1);
        m_impl->tangents.upload(std::vector<GpuFloat4>{ GpuFloat4{1.f, 0.f, 0.f, 1.f} });

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

            m_impl->perMeshGasBufs.resize(1);
            m_impl->perMeshGasHandles.resize(1, 0);
            if (!buildTriangleGAS(optixCtx, stream,
                                  static_cast<CUdeviceptr>(m_impl->posBuffer.devPtr()),
                                  static_cast<CUdeviceptr>(m_impl->posBufferClose.devPtr()),
                                  false, 3,
                                  static_cast<CUdeviceptr>(m_impl->indices.devPtr()),
                                  1,
                                  m_impl->perMeshGasBufs[0],
                                  m_impl->perMeshGasHandles[0])) {
                fprintf(stderr, "[error] CudaAccelStructure: dummy GAS build failed\n");
                return;
            }

            OptixInstance inst{};
            inst.transform[0]  = 1.f; inst.transform[5]  = 1.f; inst.transform[10] = 1.f;
            inst.instanceId        = 0;
            inst.sbtOffset         = 0;
            inst.visibilityMask    = 0xFF;
            inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = m_impl->perMeshGasHandles[0];

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

            // Single IAS instance → mesh ID 0, identity normal matrix
            m_impl->instanceMeshIDs = CudaBuffer<uint32_t>(1);
            std::vector<uint32_t> miOne{0u};
            m_impl->instanceMeshIDs.upload(miOne);
            m_impl->instanceNormalMat = CudaBuffer<float>(12);
            std::vector<float> nmIdent{1,0,0,0,  0,1,0,0,  0,0,1,0};
            m_impl->instanceNormalMat.upload(nmIdent);
            m_impl->instanceTangentMat = CudaBuffer<float>(12);
            m_impl->instanceTangentMat.upload(nmIdent);  // identity, same layout
            m_impl->instancePositionMat = CudaBuffer<float>(12);
            m_impl->instancePositionMat.upload(nmIdent);  // identity, same layout

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
    std::vector<GpuFloat2> uvs      (totalVerts);
    std::vector<GpuFloat4> tangents (totalVerts);
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

            uvs[vBase + v] = v < m.uvs.size()
                ? GpuFloat2{m.uvs[v].x, m.uvs[v].y}
                : GpuFloat2{0.f, 0.f};

            Vec4f tan = v < m.tangents.size() ? m.tangents[v] : Vec4f{1.f, 0.f, 0.f, 1.f};
            if (meshMotion) {
                // Tangents are ordinary directions — transform by the plain
                // linear part of objectToWorld, same convention as positions
                // (baked at shutter-open).
                Vec3f tw = xfOpen.transformVector(tan.xyz());
                float len = tw.length();
                if (len > 1e-6f) tw = tw * (1.f / len);
                tangents[vBase + v] = GpuFloat4{tw.x, tw.y, tw.z, tan.w};
            } else {
                tangents[vBase + v] = GpuFloat4{tan.x, tan.y, tan.z, tan.w};
            }
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

    m_impl->uvs = CudaBuffer<GpuFloat2>(totalVerts);
    m_impl->uvs.upload(uvs);

    m_impl->tangents = CudaBuffer<GpuFloat4>(totalVerts);
    m_impl->tangents.upload(tangents);

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
        // 1) Build one triangle GAS per pool mesh — mirrors Metal's
        // per-mesh BLAS layout (MetalAccelStructure.mm "Build one BLAS per
        // mesh").  Each GAS slices into the concatenated index buffer at
        // meshIndexOffsets[mi]; indices are globalized (already include
        // vBase) so they reference the shared concat position buffer
        // correctly.  Prototype meshes for InstanceGroupDesc are built the
        // same way — they live in object space and are referenced by IAS
        // instances with per-instance objectToWorld transforms.
        // -------------------------------------------------------------------
        m_impl->perMeshGasBufs.resize(numMeshes);
        m_impl->perMeshGasHandles.resize(numMeshes, 0);
        size_t totalMeshGasBytes = 0;
        uint32_t numEmptyMeshes = 0;
        bool   gasOk = true;
        for (uint32_t mi = 0; mi < numMeshes; ++mi) {
            const MeshDesc& mesh = pool.mesh(mi);
            // Skip empty meshes — OptiX rejects build inputs with
            // numIndexTriplets == 0.  These pool entries still exist (e.g.
            // placeholder prims, USD GeomSubset stripped of all faces, or
            // GN intermediates) but have no triangle geometry to ray-trace.
            // perMeshGasHandles[mi] stays 0; the IAS-entry loop below skips
            // them so no IAS slot is wasted on a null traversable.
            if (mesh.numTriangles() == 0) { ++numEmptyMeshes; continue; }
            const bool motion    = mesh.hasMotion();
            CUdeviceptr vbOpen   = static_cast<CUdeviceptr>(m_impl->posBuffer.devPtr());
            CUdeviceptr vbClose  = static_cast<CUdeviceptr>(m_impl->posBufferClose.devPtr());
            CUdeviceptr ibSlice  = static_cast<CUdeviceptr>(m_impl->indices.devPtr())
                                 + static_cast<CUdeviceptr>(indexOffsets[mi]) * sizeof(uint32_t);
            if (!buildTriangleGAS(optixCtx, stream,
                                   vbOpen, vbClose, motion,
                                   totalVerts,
                                   ibSlice, mesh.numTriangles(),
                                   m_impl->perMeshGasBufs[mi],
                                   m_impl->perMeshGasHandles[mi])) {
                fprintf(stderr, "[error] CudaAccelStructure: per-mesh GAS build failed for mesh %u\n", mi);
                gasOk = false;
                break;
            }
            totalMeshGasBytes += m_impl->perMeshGasBufs[mi].byteSize();
        }
        if (!gasOk) return;

        printf("[info]  CudaAccelStructure: %u per-mesh GASes (%s, %u verts, %u tris, "
               "%u empty meshes skipped, %.2f MiB total)\n",
               numMeshes - numEmptyMeshes,
               m_impl->hasMotion ? "motion-aware (some)" : "static",
               totalVerts, totalTris,
               numEmptyMeshes,
               totalMeshGasBytes / (1024.0 * 1024.0));

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
                    // Virtual hair meshID = numMeshes; the shader's isHair test
                    // is now `meshID >= hairMeshBase` (mirrors Metal Shade.metal).
                    m_impl->hairMeshBase  = numMeshes;
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
        // 3) Build the IAS — mirrors MetalAccelStructure.mm TLAS construction.
        //
        // Prototype meshes (those referenced by an InstanceGroupDesc) do NOT
        // get a regular-mesh IAS entry; they appear only through per-instance
        // entries with actual objectToWorld transforms.  Other meshes get
        // identity transforms (geometry already in world space).
        //
        // IAS layout (order):
        //   [0 .. numRegularInst-1]    regular (non-prototype) meshes
        //   [numRegularInst]           hair (optional)
        //   [numRegularInst+1 .. N]    per-instance-group instance entries
        // -------------------------------------------------------------------
        // Only meshes that actually have a built GAS contribute IAS entries.
        // Empty meshes (numTriangles==0) were skipped above; prototypes appear
        // only via per-instance entries below.
        uint32_t numRegularInst = 0;
        for (uint32_t mi = 0; mi < numMeshes; ++mi)
            if (!pool.isPrototype(mi) && m_impl->perMeshGasHandles[mi] != 0)
                ++numRegularInst;

        uint32_t numInstGroupInst = 0;
        for (uint32_t g = 0; g < static_cast<uint32_t>(pool.numInstanceGroups()); ++g) {
            const InstanceGroupDesc& grp = pool.instanceGroup(g);
            // Skip groups whose prototype has no GAS (empty mesh).
            if (m_impl->perMeshGasHandles[grp.protoMeshID] == 0) continue;
            numInstGroupInst += static_cast<uint32_t>(grp.instances.size());
        }

        const bool     hasHair       = (m_impl->hairGasHandle != 0);
        const uint32_t totalInstances = numRegularInst + (hasHair ? 1u : 0u) + numInstGroupInst;
        const bool     iasMotion     = m_impl->hasMotion || m_impl->hairHasMotion;

        std::vector<OptixInstance> insts(totalInstances);
        memset(insts.data(), 0, totalInstances * sizeof(OptixInstance));
        std::vector<uint32_t> instMeshIDs(totalInstances);
        std::vector<float>    instNM(totalInstances * 12, 0.f);
        std::vector<float>    instTM(totalInstances * 12, 0.f);
        std::vector<float>    instPM(totalInstances * 12, 0.f);

        auto setIdentityXform = [](OptixInstance& inst) {
            inst.transform[0] = 1.f; inst.transform[1] = 0.f; inst.transform[2]  = 0.f; inst.transform[3]  = 0.f;
            inst.transform[4] = 0.f; inst.transform[5] = 1.f; inst.transform[6]  = 0.f; inst.transform[7]  = 0.f;
            inst.transform[8] = 0.f; inst.transform[9] = 0.f; inst.transform[10] = 1.f; inst.transform[11] = 0.f;
        };
        // OptixInstance::transform is row-major 3x4 = first 3 rows of o2w.
        auto setXformFromO2W = [](OptixInstance& inst, const Mat4f& o2w) {
            inst.transform[0]  = o2w.m[0][0]; inst.transform[1]  = o2w.m[0][1];
            inst.transform[2]  = o2w.m[0][2]; inst.transform[3]  = o2w.m[0][3];
            inst.transform[4]  = o2w.m[1][0]; inst.transform[5]  = o2w.m[1][1];
            inst.transform[6]  = o2w.m[1][2]; inst.transform[7]  = o2w.m[1][3];
            inst.transform[8]  = o2w.m[2][0]; inst.transform[9]  = o2w.m[2][1];
            inst.transform[10] = o2w.m[2][2]; inst.transform[11] = o2w.m[2][3];
        };
        auto setIdentityNM = [&](uint32_t i) {
            float* m = &instNM[i * 12];
            m[0]=1.f; m[1]=0.f; m[2] =0.f; m[3] =0.f;
            m[4]=0.f; m[5]=1.f; m[6] =0.f; m[7] =0.f;
            m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=0.f;
        };
        // Store rows of worldToObject^T = columns of worldToObject.
        // Shader computes geomN_world = normalize({dot(n,row0), dot(n,row1), dot(n,row2)}).
        auto setNMFromW2O = [&](uint32_t i, const Mat4f& w2o) {
            float* m = &instNM[i * 12];
            m[0]  = w2o.m[0][0]; m[1]  = w2o.m[1][0]; m[2]  = w2o.m[2][0]; m[3]  = 0.f;
            m[4]  = w2o.m[0][1]; m[5]  = w2o.m[1][1]; m[6]  = w2o.m[2][1]; m[7]  = 0.f;
            m[8]  = w2o.m[0][2]; m[9]  = w2o.m[1][2]; m[10] = w2o.m[2][2]; m[11] = 0.f;
        };
        auto setIdentityTM = [&](uint32_t i) {
            float* m = &instTM[i * 12];
            m[0]=1.f; m[1]=0.f; m[2] =0.f; m[3] =0.f;
            m[4]=0.f; m[5]=1.f; m[6] =0.f; m[7] =0.f;
            m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=0.f;
        };
        // Tangents are ordinary directions, not normals — transform by the
        // plain linear part of objectToWorld, not its inverse-transpose.
        auto setTMFromO2W = [&](uint32_t i, const Mat4f& o2w) {
            float* m = &instTM[i * 12];
            m[0]  = o2w.m[0][0]; m[1]  = o2w.m[0][1]; m[2]  = o2w.m[0][2]; m[3]  = 0.f;
            m[4]  = o2w.m[1][0]; m[5]  = o2w.m[1][1]; m[6]  = o2w.m[1][2]; m[7]  = 0.f;
            m[8]  = o2w.m[2][0]; m[9]  = o2w.m[2][1]; m[10] = o2w.m[2][2]; m[11] = 0.f;
        };
        auto setIdentityPM = [&](uint32_t i) {
            float* m = &instPM[i * 12];
            m[0]=1.f; m[1]=0.f; m[2] =0.f; m[3] =0.f;
            m[4]=0.f; m[5]=1.f; m[6] =0.f; m[7] =0.f;
            m[8]=0.f; m[9]=0.f; m[10]=1.f; m[11]=0.f;
        };
        // Full worldToObject (translation included) — row i = {w2o.m[i][0..3]},
        // so objPos[i] = dot(row[i].xyz, worldPos) + row[i].w. Used to recover
        // object-space hit position for MaterialX <position space="object">.
        auto setPMFromW2O = [&](uint32_t i, const Mat4f& w2o) {
            float* m = &instPM[i * 12];
            m[0]  = w2o.m[0][0]; m[1]  = w2o.m[0][1]; m[2]  = w2o.m[0][2]; m[3]  = w2o.m[0][3];
            m[4]  = w2o.m[1][0]; m[5]  = w2o.m[1][1]; m[6]  = w2o.m[1][2]; m[7]  = w2o.m[1][3];
            m[8]  = w2o.m[2][0]; m[9]  = w2o.m[2][1]; m[10] = w2o.m[2][2]; m[11] = w2o.m[2][3];
        };

        uint32_t tlasIdx = 0;

        // Regular (non-prototype) meshes — identity transform, geometry world-space.
        // Skip prototypes and empty meshes (perMeshGasHandles[mi] == 0).
        for (uint32_t mi = 0; mi < numMeshes; ++mi) {
            if (pool.isPrototype(mi)) continue;
            if (m_impl->perMeshGasHandles[mi] == 0) continue;
            OptixInstance& inst = insts[tlasIdx];
            setIdentityXform(inst);
            inst.instanceId        = tlasIdx;
            inst.sbtOffset         = 0;
            inst.visibilityMask    = 0xFF;
            inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = m_impl->perMeshGasHandles[mi];
            instMeshIDs[tlasIdx]   = mi;
            setIdentityNM(tlasIdx);
            setIdentityTM(tlasIdx);
            // Real per-mesh worldToObject — NOT identity, even though the
            // mesh's geometry is pre-baked to world space (see setPMFromW2O
            // comment above and MetalAccelStructure.mm's mirror of this).
            setPMFromW2O(tlasIdx, pool.mesh(mi).staticWorldToObject);
            ++tlasIdx;
        }

        // Hair (if present) — virtual meshID = numMeshes
        if (hasHair) {
            OptixInstance& inst = insts[tlasIdx];
            setIdentityXform(inst);
            inst.instanceId        = tlasIdx;
            inst.sbtOffset         = 0;
            inst.visibilityMask    = 0xFF;
            inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
            inst.traversableHandle = m_impl->hairGasHandle;
            instMeshIDs[tlasIdx]   = numMeshes;  // virtual hair ID
            setIdentityNM(tlasIdx);
            setIdentityTM(tlasIdx);
            setIdentityPM(tlasIdx);
            ++tlasIdx;
        }

        // Per-instance-group instances — actual o2w transforms, prototype GAS.
        // Skip groups whose prototype has no GAS (e.g. empty prototype mesh).
        for (uint32_t g = 0; g < static_cast<uint32_t>(pool.numInstanceGroups()); ++g) {
            const InstanceGroupDesc& grp = pool.instanceGroup(g);
            if (m_impl->perMeshGasHandles[grp.protoMeshID] == 0) continue;
            for (const InstanceDesc& instDesc : grp.instances) {
                const Mat4f& o2w = instDesc.keys[0].objectToWorld;
                const Mat4f& w2o = instDesc.keys[0].worldToObject;
                OptixInstance& inst = insts[tlasIdx];
                setXformFromO2W(inst, o2w);
                inst.instanceId        = tlasIdx;
                inst.sbtOffset         = 0;
                inst.visibilityMask    = 0xFF;
                inst.flags             = OPTIX_INSTANCE_FLAG_NONE;
                inst.traversableHandle = m_impl->perMeshGasHandles[grp.protoMeshID];
                instMeshIDs[tlasIdx]   = grp.protoMeshID;
                setNMFromW2O(tlasIdx, w2o);
                setTMFromO2W(tlasIdx, o2w);
                setPMFromW2O(tlasIdx, w2o);
                ++tlasIdx;
            }
        }

        // Upload per-instance lookup buffers
        m_impl->instanceMeshIDs = CudaBuffer<uint32_t>(totalInstances);
        m_impl->instanceMeshIDs.upload(instMeshIDs);
        m_impl->instanceNormalMat = CudaBuffer<float>(totalInstances * 12);
        m_impl->instanceNormalMat.upload(instNM);
        m_impl->instanceTangentMat = CudaBuffer<float>(totalInstances * 12);
        m_impl->instanceTangentMat.upload(instTM);
        m_impl->instancePositionMat = CudaBuffer<float>(totalInstances * 12);
        m_impl->instancePositionMat.upload(instPM);

        // Upload IAS instance descriptors and build
        const size_t instBytes = totalInstances * sizeof(OptixInstance);
        m_impl->iasInstanceBuf = CudaByteBuffer(instBytes);
        m_impl->iasInstanceBuf.upload(
            reinterpret_cast<const uint8_t*>(insts.data()), instBytes);

        OptixBuildInput iasInput{};
        iasInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
        iasInput.instanceArray.instances    =
            static_cast<CUdeviceptr>(m_impl->iasInstanceBuf.devPtr());
        iasInput.instanceArray.numInstances = totalInstances;

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

        printf("[info]  CudaAccelStructure: IAS built (%u instances = %u regular + %u hair + %u inst-group, "
               "%u prototype meshes, %.2f KiB)\n",
               totalInstances, numRegularInst, hasHair ? 1u : 0u, numInstGroupInst,
               numMeshes - numRegularInst, m_impl->iasBuffer.byteSize() / 1024.0);
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
uint64_t CudaAccelStructure::uvBuffer()               const { return m_impl->uvs.devPtr(); }
uint64_t CudaAccelStructure::tangentBuffer()          const { return m_impl->tangents.devPtr(); }
uint64_t CudaAccelStructure::indexBuffer()            const { return m_impl->indices.devPtr(); }
uint64_t CudaAccelStructure::triMeshIDBuffer()        const { return m_impl->triMeshIDs.devPtr(); }
uint64_t CudaAccelStructure::meshVertexOffsetBuffer() const { return m_impl->meshVertexOffsets.devPtr(); }
uint64_t CudaAccelStructure::meshIndexOffsetBuffer()  const { return m_impl->meshIndexOffsets.devPtr(); }
bool     CudaAccelStructure::hasMotion()              const { return m_impl->hasMotion || m_impl->hairHasMotion; }
uint64_t CudaAccelStructure::hairTriBuffer()          const { return m_impl->hairTriBuf.isValid() ? m_impl->hairTriBuf.devPtr() : 0u; }
uint32_t CudaAccelStructure::hairMeshBaseID()         const { return m_impl->hairMeshBase; }
uint64_t CudaAccelStructure::instanceMeshIDBuffer()   const {
#ifdef ANACAPA_ENABLE_OPTIX
    return m_impl->instanceMeshIDs.isValid() ? m_impl->instanceMeshIDs.devPtr() : 0u;
#else
    return 0u;
#endif
}
uint64_t CudaAccelStructure::instanceNormalMatrixBuffer() const {
#ifdef ANACAPA_ENABLE_OPTIX
    return m_impl->instanceNormalMat.isValid() ? m_impl->instanceNormalMat.devPtr() : 0u;
#else
    return 0u;
#endif
}
uint64_t CudaAccelStructure::instanceTangentMatrixBuffer() const {
#ifdef ANACAPA_ENABLE_OPTIX
    return m_impl->instanceTangentMat.isValid() ? m_impl->instanceTangentMat.devPtr() : 0u;
#else
    return 0u;
#endif
}
uint64_t CudaAccelStructure::instancePositionMatrixBuffer() const {
#ifdef ANACAPA_ENABLE_OPTIX
    return m_impl->instancePositionMat.isValid() ? m_impl->instancePositionMat.devPtr() : 0u;
#else
    return 0u;
#endif
}
uint64_t CudaAccelStructure::traversableHandle()      const {
#ifdef ANACAPA_ENABLE_OPTIX
    return static_cast<uint64_t>(m_impl->iasHandle);
#else
    return 0;
#endif
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
