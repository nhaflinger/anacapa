// LaunchParams.h — host/device shared launch parameter block.
//
// Used as the OptiX __constant__ variable referenced by raygen / closesthit /
// miss programs.  The host fills this struct, copies it into a device buffer,
// and passes that buffer's address to optixLaunch.

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
    GpuSampleBatch           sampleBatch;
    cudaTextureObject_t      envTexture;      // 0 = no texture (use envLe fallback)

    // OptiX traversable for the GAS (uint64 = OptixTraversableHandle).  Set
    // to CudaAccelStructure::traversableHandle().  Read in raygen / shadow
    // calls of optixTrace.
    unsigned long long       handle;
};
