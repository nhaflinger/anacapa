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
    const uint32_t*          triMeshIDs;      // device ptr — per-triangle meshID (legacy; unused after per-mesh GAS refactor)
    const uint32_t*          meshVertexOffsets; // device ptr — per-mesh vertex base
    const uint32_t*          meshIndexOffsets;  // device ptr — per-mesh index base (elements)

    // Per-IAS-instance lookup (mirrors MetalAccelStructure).
    // instanceMeshIDs[optixGetInstanceId()] → pool meshID (or virtual hairID).
    // instanceNormalMat: 12 floats per instance = rows of worldToObject^T.
    //   Identity for regular world-space meshes; actual w2o^T for prototype
    //   instances so geomN_world = normalize({dot(n_obj,row0),row1,row2}).
    const uint32_t*          instanceMeshIDs;
    const float*             instanceNormalMat;
    GpuSampleBatch           sampleBatch;
    cudaTextureObject_t      envTexture;      // 0 = no texture (use envLe fallback)

    // HDRI importance-sampling CDFs (when present — see cam.envMapWidth/Height).
    // marginal:    H+1 floats sampling the latitude (theta) axis
    // conditional: H * (W+1) floats — per-row CDFs across longitude (phi)
    const float*             envMarginalCdf;
    const float*             envConditionalCdf;

    // Energy-compensation LUTs for the GGX dielectric spec lobe, mirroring
    // the StandardSurface CPU tables.  Same layout: specAlbedo is 32x32
    // floats [cos_idx * 32 + roughness_idx]; specAvgAlbedo is 32 floats
    // indexed by roughness only.  Built once on the host and uploaded.
    const float*             specAlbedoLUT;
    const float*             specAvgAlbedoLUT;
    uint32_t                 specLUTCosBins;
    uint32_t                 specLUTRoughBins;

    // Pixel-reconstruction filter — separable inverse-CDF tables shared
    // with the CPU PixelFilter.  pixelFilterBins == 0 disables the filter
    // (raygen falls back to a unit box-1.0 jitter, the legacy behaviour).
    // pixelFilterCdf:    bins+1 floats, monotonic in [0,1]
    // pixelFilterSigns:  bins floats, ±1 per bin
    const float*             pixelFilterCdf;
    const float*             pixelFilterSigns;
    uint32_t                 pixelFilterBins;
    float                    pixelFilterRadius;

    // Hair — per-triangle metadata + per-material BSDF parameters.
    // Indexed by optixGetPrimitiveIndex() inside the hair GAS.  nullptr
    // when the scene has no hair (cam.hairMeshBaseID == 0xFFFFFFFF).
    const GpuHairTri*        hairTris;
    const GpuHairMaterial*   hairMats;

    // Halo disc particles — software BVH walked inline in wf_bounce.
    // halos      : numHalos GpuHaloDesc entries
    // haloNodes  : binary BVH over halos
    // haloPrimIdx: leaf-payload indices into halos[]
    // All nullptr when scene.haloAccel is absent or empty.
    const GpuHaloDesc*       halos;
    const GpuHaloNode*       haloNodes;
    const uint32_t*          haloPrimIdx;

    // Caustic photon map — populated when cam.photonMapEnabled != 0.
    //   photons          : numPhotons GpuPhoton entries written by the
    //                      photon-trace raygen; read by the shade raygen
    //                      via the hash-grid query.
    //   hashCellStart    : numCells+1 prefix-sum bucket starts
    //   sortedPhotonIdx  : validPhotons indices into photons[], grouped by cell
    //   numPhotons       : total photon slots (also the photon-raygen launch size)
    //   frameIndex       : bumped each photon-trace launch for seed variation
    GpuPhoton*               photons;
    const uint32_t*          hashCellStart;
    const uint32_t*          sortedPhotonIdx;
    uint32_t                 numPhotons;
    uint32_t                 frameIndex;

    // SSS photon map — populated when cam.sssMapEnabled != 0.
    //   sssPhotons       : cell-sorted SSS GpuPhoton entries
    //   sssHashCellStart : numSSS+1 prefix-sum bucket starts
    //   sssNumPhotons    : total SSS photon slots
    GpuPhoton*               sssPhotons;
    const uint32_t*          sssHashCellStart;
    uint32_t                 sssNumPhotons;

    // Wavefront ray state — used only by the wavefront raygens.
    // wfRays is a flat array of size wfNumRays = tileW * tileH * batchSize.
    // __raygen__wf_bounce skips entries with kWfTerminated set, so a single
    // launch handles all surviving paths at the current depth.
    WfRayState*              wfRays;
    uint32_t                 wfNumRays;

    // OptiX traversable for the GAS (uint64 = OptixTraversableHandle).  Set
    // to CudaAccelStructure::traversableHandle().  Read in raygen / shadow
    // calls of optixTrace.
    unsigned long long       handle;
};
