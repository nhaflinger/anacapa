#ifdef ANACAPA_ENABLE_CUDA

#include "CudaAccelStructure.h"
#include "CudaContext.h"
#include "CudaBuffer.h"
#include "shaders/SharedTypes.h"

#include <cuda_runtime.h>

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

namespace anacapa {

// ---------------------------------------------------------------------------
// CPU-side BVH builder — binned SAH (matches quality of the CPU BVHBackend)
// ---------------------------------------------------------------------------
struct BvhBuilder {
    const float*    positions;
    const uint32_t* indices;
    uint32_t        numTris;

    std::vector<BvhNode>  nodes;
    std::vector<uint32_t> primOrder;

    struct TriData {
        float cX, cY, cZ;   // centroid
        float minX, minY, minZ;
        float maxX, maxY, maxZ;
    };
    std::vector<TriData> triData;

    void build() {
        primOrder.resize(numTris);
        for (uint32_t i = 0; i < numTris; ++i) primOrder[i] = i;

        triData.resize(numTris);
        for (uint32_t t = 0; t < numTris; ++t) {
            float mnX= FLT_MAX, mnY= FLT_MAX, mnZ= FLT_MAX;
            float mxX=-FLT_MAX, mxY=-FLT_MAX, mxZ=-FLT_MAX;
            for (int v = 0; v < 3; ++v) {
                uint32_t vi = indices[t * 3 + v];
                float px = positions[vi*3+0], py = positions[vi*3+1], pz = positions[vi*3+2];
                if (px<mnX) mnX=px; if (px>mxX) mxX=px;
                if (py<mnY) mnY=py; if (py>mxY) mxY=py;
                if (pz<mnZ) mnZ=pz; if (pz>mxZ) mxZ=pz;
            }
            triData[t] = { (mnX+mxX)*0.5f, (mnY+mxY)*0.5f, (mnZ+mxZ)*0.5f,
                            mnX, mnY, mnZ, mxX, mxY, mxZ };
        }

        nodes.reserve(2 * numTris);
        nodes.push_back(BvhNode{});
        buildNode(0, 0, numTris);
    }

private:
    static constexpr uint32_t MAX_LEAF_TRIS = 4;
    static constexpr int      NUM_BINS      = 8;

    // Half surface area of an AABB — proportional to expected ray hit probability
    static float halfArea(float mnX, float mnY, float mnZ,
                          float mxX, float mxY, float mxZ) {
        float dx = mxX-mnX, dy = mxY-mnY, dz = mxZ-mnZ;
        return dx*dy + dy*dz + dz*dx;
    }

    struct Bin {
        float mnX= FLT_MAX, mnY= FLT_MAX, mnZ= FLT_MAX;
        float mxX=-FLT_MAX, mxY=-FLT_MAX, mxZ=-FLT_MAX;
        uint32_t count = 0;
        void expand(float x, float y, float z,
                    float x2, float y2, float z2) {
            if (x <mnX) mnX=x;  if (x2>mxX) mxX=x2;
            if (y <mnY) mnY=y;  if (y2>mxY) mxY=y2;
            if (z <mnZ) mnZ=z;  if (z2>mxZ) mxZ=z2;
            ++count;
        }
    };

    void buildNode(uint32_t nodeIdx, uint32_t first, uint32_t count) {
        // Compute node AABB
        float bMinX= FLT_MAX, bMinY= FLT_MAX, bMinZ= FLT_MAX;
        float bMaxX=-FLT_MAX, bMaxY=-FLT_MAX, bMaxZ=-FLT_MAX;
        for (uint32_t i = first; i < first + count; ++i) {
            const TriData& td = triData[primOrder[i]];
            if (td.minX<bMinX) bMinX=td.minX; if (td.maxX>bMaxX) bMaxX=td.maxX;
            if (td.minY<bMinY) bMinY=td.minY; if (td.maxY>bMaxY) bMaxY=td.maxY;
            if (td.minZ<bMinZ) bMinZ=td.minZ; if (td.maxZ>bMaxZ) bMaxZ=td.maxZ;
        }
        nodes[nodeIdx].aabbMin = { bMinX, bMinY, bMinZ };
        nodes[nodeIdx].aabbMax = { bMaxX, bMaxY, bMaxZ };

        if (count <= MAX_LEAF_TRIS) {
            nodes[nodeIdx].leftFirst = first;
            nodes[nodeIdx].triCount  = count;
            return;
        }

        // SAH binned split — try all 3 axes, pick lowest cost
        float parentArea = halfArea(bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ);
        float leafCost   = float(count);  // cost of not splitting

        float bestCost = leafCost;
        int   bestAxis = -1;
        float bestSplit = 0.0f;

        // Centroid AABB for this set
        float cMinX= FLT_MAX, cMinY= FLT_MAX, cMinZ= FLT_MAX;
        float cMaxX=-FLT_MAX, cMaxY=-FLT_MAX, cMaxZ=-FLT_MAX;
        for (uint32_t i = first; i < first + count; ++i) {
            const TriData& td = triData[primOrder[i]];
            if (td.cX<cMinX) cMinX=td.cX; if (td.cX>cMaxX) cMaxX=td.cX;
            if (td.cY<cMinY) cMinY=td.cY; if (td.cY>cMaxY) cMaxY=td.cY;
            if (td.cZ<cMinZ) cMinZ=td.cZ; if (td.cZ>cMaxZ) cMaxZ=td.cZ;
        }

        float cRange[3] = { cMaxX-cMinX, cMaxY-cMinY, cMaxZ-cMinZ };
        float cMin [3]  = { cMinX, cMinY, cMinZ };

        for (int axis = 0; axis < 3; ++axis) {
            if (cRange[axis] < 1e-6f) continue;

            Bin bins[NUM_BINS]{};
            float scale = float(NUM_BINS) / cRange[axis];

            for (uint32_t i = first; i < first + count; ++i) {
                const TriData& td = triData[primOrder[i]];
                float c = (&td.cX)[axis];
                int b = std::min(int((c - cMin[axis]) * scale), NUM_BINS - 1);
                bins[b].expand(td.minX, td.minY, td.minZ,
                               td.maxX, td.maxY, td.maxZ);
            }

            // Prefix left bounds, suffix right bounds → evaluate NUM_BINS-1 splits
            float lMnX[NUM_BINS-1], lMnY[NUM_BINS-1], lMnZ[NUM_BINS-1];
            float lMxX[NUM_BINS-1], lMxY[NUM_BINS-1], lMxZ[NUM_BINS-1];
            uint32_t lCnt[NUM_BINS-1];
            {
                float mx= FLT_MAX, my= FLT_MAX, mz= FLT_MAX;
                float Mx=-FLT_MAX, My=-FLT_MAX, Mz=-FLT_MAX;
                uint32_t cnt = 0;
                for (int b = 0; b < NUM_BINS-1; ++b) {
                    if (bins[b].count > 0) {
                        if (bins[b].mnX<mx) mx=bins[b].mnX;
                        if (bins[b].mnY<my) my=bins[b].mnY;
                        if (bins[b].mnZ<mz) mz=bins[b].mnZ;
                        if (bins[b].mxX>Mx) Mx=bins[b].mxX;
                        if (bins[b].mxY>My) My=bins[b].mxY;
                        if (bins[b].mxZ>Mz) Mz=bins[b].mxZ;
                    }
                    cnt += bins[b].count;
                    lMnX[b]=mx; lMnY[b]=my; lMnZ[b]=mz;
                    lMxX[b]=Mx; lMxY[b]=My; lMxZ[b]=Mz;
                    lCnt[b]=cnt;
                }
            }

            {
                float mx= FLT_MAX, my= FLT_MAX, mz= FLT_MAX;
                float Mx=-FLT_MAX, My=-FLT_MAX, Mz=-FLT_MAX;
                uint32_t cnt = 0;
                for (int b = NUM_BINS-1; b >= 1; --b) {
                    if (bins[b].count > 0) {
                        if (bins[b].mnX<mx) mx=bins[b].mnX;
                        if (bins[b].mnY<my) my=bins[b].mnY;
                        if (bins[b].mnZ<mz) mz=bins[b].mnZ;
                        if (bins[b].mxX>Mx) Mx=bins[b].mxX;
                        if (bins[b].mxY>My) My=bins[b].mxY;
                        if (bins[b].mxZ>Mz) Mz=bins[b].mxZ;
                    }
                    cnt += bins[b].count;
                    int s = b - 1;  // split after bin s (left = [0..s], right = [s+1..])
                    if (lCnt[s] == 0 || cnt == 0) continue;
                    float cost = (halfArea(lMnX[s],lMnY[s],lMnZ[s],lMxX[s],lMxY[s],lMxZ[s]) / parentArea) * float(lCnt[s])
                               + (halfArea(mx,my,mz,Mx,My,Mz)                                / parentArea) * float(cnt);
                    if (cost < bestCost) {
                        bestCost  = cost;
                        bestAxis  = axis;
                        // World-space split plane position: boundary between bin s and s+1
                        bestSplit = cMin[axis] + float(s + 1) / scale;
                    }
                }
            }
        }

        if (bestAxis < 0) {
            // SAH found no profitable split — make a leaf (may exceed MAX_LEAF_TRIS)
            nodes[nodeIdx].leftFirst = first;
            nodes[nodeIdx].triCount  = count;
            return;
        }

        // Partition primOrder[first..first+count) around bestSplit on bestAxis
        auto* begin = primOrder.data() + first;
        auto* end   = begin + count;
        auto* mid   = std::partition(begin, end, [&](uint32_t t) {
            return (&triData[t].cX)[bestAxis] < bestSplit;
        });
        uint32_t leftCount = static_cast<uint32_t>(mid - begin);

        // Degenerate partition fallback
        if (leftCount == 0 || leftCount == count) {
            leftCount = count / 2;
        }

        uint32_t leftIdx = static_cast<uint32_t>(nodes.size());
        nodes.push_back(BvhNode{});
        nodes.push_back(BvhNode{});

        nodes[nodeIdx].leftFirst = leftIdx;
        nodes[nodeIdx].triCount  = 0;

        buildNode(leftIdx,     first,            leftCount);
        buildNode(leftIdx + 1, first + leftCount, count - leftCount);
    }
};

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------
struct CudaAccelStructure::Impl {
    CudaBuffer<BvhNode>  bvhNodes;
    CudaBuffer<uint32_t> triIndices;
    CudaByteBuffer       posBuffer;    // packed float3 positions
    CudaBuffer<GpuFloat3> normals;
    CudaBuffer<uint32_t>  indices;
    CudaBuffer<uint32_t>  triMeshIDs;
    CudaBuffer<uint32_t>  meshVertexOffsets;
    CudaBuffer<uint32_t>  meshIndexOffsets;

    uint32_t totalVertices  = 0;
    uint32_t totalTriangles = 0;
    uint32_t numMeshes_     = 0;
    bool     valid          = false;
};

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CudaAccelStructure::CudaAccelStructure(CudaContext& ctx, const GeometryPool& pool)
    : m_impl(std::make_unique<Impl>())
{
    (void)ctx;  // stream not needed for host-side build + cudaMemcpy

    uint32_t numMeshes = static_cast<uint32_t>(pool.numMeshes());
    m_impl->numMeshes_ = numMeshes;

    if (numMeshes == 0) {
        fprintf(stderr, "[error] CudaAccelStructure: empty geometry pool\n");
        return;
    }

    // -----------------------------------------------------------------------
    // Build concatenated CPU arrays
    // -----------------------------------------------------------------------
    uint32_t totalVerts = 0, totalTris = 0;
    for (uint32_t i = 0; i < numMeshes; ++i) {
        totalVerts += pool.mesh(i).numVertices();
        totalTris  += pool.mesh(i).numTriangles();
    }

    std::vector<float>     positions(totalVerts * 3);
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

        for (uint32_t v = 0; v < m.numVertices(); ++v) {
            positions[(vBase + v) * 3 + 0] = m.positions[v].x;
            positions[(vBase + v) * 3 + 1] = m.positions[v].y;
            positions[(vBase + v) * 3 + 2] = m.positions[v].z;
            normals[vBase + v] = m.normals.empty()
                ? GpuFloat3{0, 1, 0}
                : GpuFloat3{m.normals[v].x, m.normals[v].y, m.normals[v].z};
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
    // Build BVH on CPU
    // -----------------------------------------------------------------------
    BvhBuilder builder;
    builder.positions = positions.data();
    builder.indices   = indices.data();
    builder.numTris   = totalTris;
    builder.build();

    // -----------------------------------------------------------------------
    // Upload to GPU
    // -----------------------------------------------------------------------
    m_impl->posBuffer = CudaByteBuffer(totalVerts * 3 * sizeof(float));
    m_impl->posBuffer.upload(reinterpret_cast<const uint8_t*>(positions.data()),
                              totalVerts * 3 * sizeof(float));

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

    m_impl->bvhNodes = CudaBuffer<BvhNode>(builder.nodes.size());
    m_impl->bvhNodes.upload(builder.nodes);

    m_impl->triIndices = CudaBuffer<uint32_t>(builder.primOrder.size());
    m_impl->triIndices.upload(builder.primOrder);

    m_impl->valid = true;
    printf("[info]  CudaAccelStructure: BVH built — %u nodes, %u verts, %u tris\n",
           static_cast<uint32_t>(builder.nodes.size()), totalVerts, totalTris);
}

CudaAccelStructure::~CudaAccelStructure() = default;

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------
bool     CudaAccelStructure::isValid()        const { return m_impl->valid; }
uint32_t CudaAccelStructure::totalVertices()  const { return m_impl->totalVertices; }
uint32_t CudaAccelStructure::totalTriangles() const { return m_impl->totalTriangles; }
uint32_t CudaAccelStructure::numMeshes()      const { return m_impl->numMeshes_; }

uint64_t CudaAccelStructure::bvhBuffer()              const { return m_impl->bvhNodes.devPtr(); }
uint64_t CudaAccelStructure::triIndexBuffer()         const { return m_impl->triIndices.devPtr(); }
uint64_t CudaAccelStructure::positionBuffer()         const { return m_impl->posBuffer.devPtr(); }
uint64_t CudaAccelStructure::normalBuffer()           const { return m_impl->normals.devPtr(); }
uint64_t CudaAccelStructure::indexBuffer()            const { return m_impl->indices.devPtr(); }
uint64_t CudaAccelStructure::triMeshIDBuffer()        const { return m_impl->triMeshIDs.devPtr(); }
uint64_t CudaAccelStructure::meshVertexOffsetBuffer() const { return m_impl->meshVertexOffsets.devPtr(); }
uint64_t CudaAccelStructure::meshIndexOffsetBuffer()  const { return m_impl->meshIndexOffsets.devPtr(); }

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
