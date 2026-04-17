// LaunchParams.h — struct passed as a kernel argument to shade().
// Shared between host (CudaPathIntegrator.cu) and device (Shade.cu).
// Only included from .cu translation units compiled by nvcc.

#pragma once
#include "SharedTypes.h"
#include <cuda_runtime.h>

struct LaunchParams {
    GpuCameraParams          cam;
    GpuAccumPixel*           accum;           // device ptr — tile-sized output buffer
    const GpuLight*          lights;          // device ptr
    uint32_t                 numLights;
    const GpuMaterial*       materials;       // device ptr
    uint32_t                 numMaterials;
    const GpuFloat3*         normals;         // device ptr — all meshes concatenated
    const uint32_t*          indices;         // device ptr — globalized triangle indices
    const uint32_t*          triMeshIDs;      // device ptr — per-triangle meshID
    const uint32_t*          meshVertexOffsets; // device ptr — per-mesh vertex base
    const uint32_t*          meshIndexOffsets;  // device ptr — per-mesh index base (elements)
    const float*             positions;       // device ptr — packed float3 vertex positions
    const BvhNode*           bvh;            // device ptr — flat BVH node array
    const uint32_t*          triIndices;     // device ptr — BVH-reordered triangle indices
    uint32_t                 sampleIndex;
    cudaTextureObject_t      envTexture;      // 0 = no texture (use envLe fallback)
};
