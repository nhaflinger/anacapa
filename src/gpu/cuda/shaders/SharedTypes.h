// SharedTypes.h — POD structs shared between C++ host and CUDA device code.
// CUDA adaptation of src/gpu/metal/shaders/SharedTypes.h.
//
// Rules:
//   - No C++ constructors, destructors, virtual functions, or std::
//   - All fields explicitly sized for cross-boundary ABI safety
//   - Include from both host (.cu) and device (Shade.cu) translation units

#ifndef ANACAPA_CUDA_SHARED_TYPES_H
#define ANACAPA_CUDA_SHARED_TYPES_H

#ifdef __CUDACC__
#  include <stdint.h>
#else
#  include <cstdint>
#endif

// Plain packed float types — identical layout on host and device.
// We avoid cuda's float3 (may have 16-byte alignment on device) to keep
// the struct layout exactly matching the CPU-side layout.
struct GpuFloat3 { float x, y, z; };
struct GpuFloat2 { float x, y; };

// ---------------------------------------------------------------------------
// GpuRay
// ---------------------------------------------------------------------------
struct GpuRay {
    GpuFloat3 origin;
    float     tMin;
    GpuFloat3 direction;
    float     tMax;
    uint32_t  pixelIdx;
    uint32_t  sampleIdx;
    uint32_t  bounce;
    uint32_t  _pad;
};

// ---------------------------------------------------------------------------
// GpuHit
// ---------------------------------------------------------------------------
struct GpuHit {
    GpuFloat3 position;
    float     t;
    GpuFloat3 normal;
    float     _pad0;
    GpuFloat2 uv;
    uint32_t  meshID;
    uint32_t  valid;
};

// ---------------------------------------------------------------------------
// GpuMaterial
// ---------------------------------------------------------------------------
enum GpuMaterialType : uint32_t {
    kMatLambertian  = 0,
    kMatGGX         = 1,
    kMatEmissive    = 2,
    kMatGlass       = 3,
};

struct GpuMaterial {
    GpuFloat3  baseColor;
    float      roughness;
    GpuFloat3  emissive;
    float      metalness;
    uint32_t   type;
    float      specularIOR;
    float      transmission;
    float      _pad0;
};

// ---------------------------------------------------------------------------
// GpuLight
// ---------------------------------------------------------------------------
enum GpuLightType : uint32_t {
    kLightRect        = 0,
    kLightSphere      = 1,
    kLightDirectional = 2,
    kLightDome        = 3,
};

struct GpuLight {
    GpuFloat3  Le;
    uint32_t   type;
    GpuFloat3  position;
    float      area;
    GpuFloat3  normal;
    float      _pad0;
    GpuFloat3  uHalf;
    float      _pad1;
    GpuFloat3  vHalf;
    float      _pad2;
};

// ---------------------------------------------------------------------------
// GpuCameraParams
// ---------------------------------------------------------------------------
struct GpuCameraParams {
    GpuFloat3 origin;
    float     _pad0;
    GpuFloat3 horizontal;
    float     _pad1;
    GpuFloat3 vertical;
    float     _pad2;
    GpuFloat3 lowerLeft;
    float     _pad3;
    uint32_t  imageWidth;
    uint32_t  imageHeight;
    uint32_t  samplesPerPixel;
    uint32_t  maxDepth;
    uint32_t  tileX0;
    uint32_t  tileY0;
    uint32_t  tileWidth;
    uint32_t  tileHeight;
    GpuFloat3 envLe;
    uint32_t  hasEnvLight;
    GpuFloat3 envRot0;
    GpuFloat3 envRot1;
    GpuFloat3 envRot2;
    float     envIntensity;
    float     _pad4, _pad5, _pad6;
};

// ---------------------------------------------------------------------------
// GpuAccumPixel
// ---------------------------------------------------------------------------
struct GpuAccumPixel {
    float r, g, b, weight;
    float sumLumSq;  // sum(luminance(sample)^2) — needed for variance-based adaptive sampling
};

// ---------------------------------------------------------------------------
// BvhNode — 32-byte flat BVH2 node
//
// Internal node: triCount == 0, left child at bvh[leftFirst], right at bvh[leftFirst+1]
// Leaf node:     triCount > 0, triangles in triIndices[leftFirst .. leftFirst+triCount)
// ---------------------------------------------------------------------------
struct BvhNode {
    GpuFloat3 aabbMin;
    uint32_t  leftFirst;
    GpuFloat3 aabbMax;
    uint32_t  triCount;
};

#endif // ANACAPA_CUDA_SHARED_TYPES_H
