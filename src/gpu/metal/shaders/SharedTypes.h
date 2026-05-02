// SharedTypes.h — POD structs shared between C++ host code and MSL kernels.
//
// Rules:
//   - No C++ constructors, destructors, virtual functions, or std::
//   - All fields explicitly sized for cross-boundary ABI safety
//   - Structs are aligned to 16 bytes where Metal requires it
//   - Include this file from both .mm (with ANACAPA_METAL_HOST defined)
//     and .metal files (no define needed — MSL sees packed_float3 etc.)

#ifndef ANACAPA_SHARED_TYPES_H
#define ANACAPA_SHARED_TYPES_H

#ifdef __METAL_VERSION__
// MSL side
#include <metal_stdlib>
using namespace metal;
// Use packed_float3 (12 bytes, 4-byte aligned) so the struct layout matches
// the C++ host struct GpuFloat3 { float x, y, z; } exactly.
// float3 in Metal constant-address-space is 16-byte aligned, which would
// create a different layout than the host.
typedef packed_float3 GpuFloat3;
typedef float2 GpuFloat2;
#else
// C++ side
#include <cstdint>
struct GpuFloat3 { float x, y, z; };
struct GpuFloat2 { float x, y; };
#endif

// ---------------------------------------------------------------------------
// GpuRay — one ray in the wavefront buffer
// ---------------------------------------------------------------------------
struct GpuRay {
    GpuFloat3 origin;     // world-space origin
    float     tMin;
    GpuFloat3 direction;  // unit vector
    float     tMax;
    uint32_t  pixelIdx;   // flat pixel index (y*width + x)
    uint32_t  sampleIdx;  // which sample within the spp loop
    uint32_t  bounce;     // current bounce depth
    uint32_t  _pad;
};

// ---------------------------------------------------------------------------
// GpuHit — result of a hardware intersection query
// ---------------------------------------------------------------------------
struct GpuHit {
    GpuFloat3 position;
    float     t;
    GpuFloat3 normal;     // shading normal (world space)
    float     _pad0;
    GpuFloat2 uv;
    uint32_t  meshID;     // indexes into material table
    uint32_t  valid;      // 1 = hit, 0 = miss
};

// ---------------------------------------------------------------------------
// GpuMaterial — flattened surface description (no virtual dispatch)
// ---------------------------------------------------------------------------
enum GpuMaterialType : uint32_t {
    kMatLambertian  = 0,
    kMatGGX         = 1,
    kMatEmissive    = 2,
    kMatGlass       = 3,   // smooth dielectric — delta Fresnel + Snell refraction
};

struct GpuMaterial {
    GpuFloat3       baseColor;
    float           roughness;
    GpuFloat3       emissive;
    float           metalness;
    uint32_t        type;         // GpuMaterialType
    float           specularIOR;  // IOR for glass; unused for other types
    float           transmission; // 0 = opaque, 1 = fully transmissive
    float           _pad0;
};

// ---------------------------------------------------------------------------
// GpuLight — all light types encoded as a tagged union
// ---------------------------------------------------------------------------
enum GpuLightType : uint32_t {
    kLightRect        = 0,
    kLightSphere      = 1,
    kLightDirectional = 2,
    kLightDome        = 3,
};

struct GpuLight {
    GpuFloat3   Le;           // emitted radiance / intensity
    uint32_t    type;         // GpuLightType
    GpuFloat3   position;     // rect/sphere center; unused for directional/dome
    float       area;         // rect/sphere area; unused for others
    GpuFloat3   normal;       // rect normal / directional direction-to-light
    float       cosCone;      // directional: cos(halfAngle), 1=hard shadow; unused for others
    GpuFloat3   uHalf;        // rect half-extent u (rect only)
    float       _pad1;
    GpuFloat3   vHalf;        // rect half-extent v (rect only)
    float       _pad2;
};

// ---------------------------------------------------------------------------
// GpuCameraParams — pinhole camera for RayGen kernel
// ---------------------------------------------------------------------------
struct GpuCameraParams {
    GpuFloat3 origin;
    float     _pad0;
    GpuFloat3 horizontal;   // full-width frustum vector
    float     _pad1;
    GpuFloat3 vertical;     // full-height frustum vector
    float     _pad2;
    GpuFloat3 lowerLeft;    // lower-left corner of the image plane
    float     _pad3;
    uint32_t  imageWidth;
    uint32_t  imageHeight;
    uint32_t  samplesPerPixel;
    uint32_t  maxDepth;
    // Tile sub-region: kernel dispatches tileWidth×tileHeight threads,
    // pixel coords = gid + (tileX0, tileY0).
    uint32_t  tileX0;
    uint32_t  tileY0;
    uint32_t  tileWidth;
    uint32_t  tileHeight;
    // Environment/dome light
    GpuFloat3 envLe;          // average Le (fallback when no texture bound)
    uint32_t  hasEnvLight;    // 1 if scene has a dome/environment light
    // World-to-envmap rotation: three rows of the 3x3 matrix
    GpuFloat3 envRot0;
    GpuFloat3 envRot1;
    GpuFloat3 envRot2;
    float     envIntensity;   // multiplier for HDRI pixels (DomeLight intensity)
    float     _pad4, _pad5, _pad6;
};

// ---------------------------------------------------------------------------
// GpuAccumPixel — per-pixel accumulation in the output buffer
// (non-atomic; one thread-group per pixel or written once per sample)
// ---------------------------------------------------------------------------
struct GpuAccumPixel {
    float r, g, b, weight;
    float sumLumSq;  // sum(luminance(sample)^2) — needed for variance-based adaptive sampling
};

// ---------------------------------------------------------------------------
// GpuSampleBatch — passed to the shade kernel instead of a bare sampleIndex.
// The kernel traces batchSize samples per thread and accumulates them locally,
// amortising command-buffer overhead across multiple samples per launch.
// ---------------------------------------------------------------------------
struct GpuSampleBatch {
    uint32_t sampleStart;
    uint32_t batchSize;
};

#endif // ANACAPA_SHARED_TYPES_H
