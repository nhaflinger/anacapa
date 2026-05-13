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
    kMatHair        = 4,   // Marschner — handled via hairTris/hairMats buffers
};

struct GpuMaterial {
    GpuFloat3  baseColor;
    float      roughness;
    GpuFloat3  emissive;
    float      metalness;
    uint32_t   type;
    float      specularIOR;
    float      transmission;
    // Specular layer strength in [0,1].  Matches MaterialX standard_surface's
    // `specular` knob — multiplies the dielectric Fresnel.  Defaults to 0.5
    // for UsdPreviewSurface, 1.0 for explicit metals, 0 for non-glossy.
    // Used by the energy-conserving spec/diff balance in the GPU shader.
    float      specular;
};

// ---------------------------------------------------------------------------
// GpuHairMaterial — Marschner BSDF parameters for one hair material slot.
// Indexed by GpuHairTri::matIdx.  scene.materials[] size determines the
// buffer length; non-hair slots are zero-initialised and never accessed.
// ---------------------------------------------------------------------------
struct GpuHairMaterial {
    GpuFloat3 sigma_a;  // default absorption when strand color is the (1,1,1) sentinel
    float     eta;      // IOR of cortex (typically 1.55)
    float     beta_m;   // longitudinal roughness ∈ [0,1]
    float     beta_n;   // azimuthal roughness    ∈ [0,1]
    float     alpha;    // cuticle tilt in degrees (typically 2–4)
    float     _pad;
};

// ---------------------------------------------------------------------------
// GpuHairTri — per-triangle metadata for tessellated hair ribbons.
// The GPU hair geometry is one GAS; primID indexes directly into this buffer.
// h values encode the impact parameter h ∈ [-1, 1] at each barycentric vertex
// so the shader can interpolate: h = h0*(1-u-v) + h1*u + h2*v.
// ---------------------------------------------------------------------------
struct GpuHairTri {
    GpuFloat3 tangent;  // fiber direction (world-space, unit)
    uint32_t  matIdx;   // index into hairMats[] buffer
    GpuFloat3 color;    // per-strand linear RGB; (1,1,1) = use GpuHairMaterial.sigma_a default
    float     h0;       // impact parameter at barycentric vertex 0 ∈ [-1, 1]
    float     h1;       // impact parameter at barycentric vertex 1
    float     h2;       // impact parameter at barycentric vertex 2
    float     _pad;
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
    float      cosCone;  // directional: cos(halfAngle), 1=hard shadow; unused for others
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
    // Normalized shutter open/close in [0, 1]; rays sample time uniformly.
    // For static frames both are 0 and the ray time is irrelevant (the GAS
    // is built without motion options).
    float     shutterOpen;
    float     shutterClose;
    uint32_t  envMapWidth;    // CDF table width  (0 = no HDRI importance sampling)
    uint32_t  envMapHeight;   // CDF table height
    float     fireflyClamp;   // max luminance per path sample; 0 = disabled
    // Hair: OptiX IAS instance ID where the tessellated hair geometry lives.
    // 0xFFFFFFFF when the scene contains no hair.  In raygen this is compared
    // against optixGetInstanceId() to route hits into the Marschner shader.
    uint32_t  hairMeshBaseID;

    // Caustic photon map hash grid (enabled when photonMapEnabled != 0).
    // hashCellStart[numCells+1] and sortedPhotonIdx[validPhotons] live in
    // LaunchParams; the photon array does too.  The grid is a uniform
    // spatial partition with cellSize = searchRadius, so the 3x3x3
    // neighbour scan inside queryHashGrid covers the full lookup radius.
    uint32_t  photonMapEnabled;    // 0 = off
    float     photonSearchRadius;
    GpuFloat3 hashGridOrigin;
    float     hashCellSize;
    uint32_t  hashGridDimX;
    uint32_t  hashGridDimY;
    uint32_t  hashGridDimZ;
};

// ---------------------------------------------------------------------------
// GpuAccumPixel
// ---------------------------------------------------------------------------
struct GpuAccumPixel {
    float r, g, b, weight;
    float sumLumSq;  // sum(luminance(sample)^2) — needed for variance-based adaptive sampling
};


// ---------------------------------------------------------------------------
// GpuSampleBatch — passed to the shade kernel instead of a bare sampleIndex.
// The kernel traces batchSize samples per thread and accumulates them locally,
// amortising launch overhead across multiple samples per dispatch.
// ---------------------------------------------------------------------------
struct GpuSampleBatch {
    uint32_t sampleStart;
    uint32_t batchSize;
};

// ---------------------------------------------------------------------------
// GpuPhoton — one caustic photon emitted by the photon-trace raygen.
// power == (0, 0, 0) means no valid caustic hit (no specular bounce before
// the first diffuse hit); those slots are skipped when building the grid.
// ---------------------------------------------------------------------------
struct GpuPhoton {
    GpuFloat3 position;
    GpuFloat3 wi;       // photon travel direction (toward surface)
    GpuFloat3 power;    // radiant flux RGB; (0,0,0) = invalid
    uint32_t  _pad;
};

#endif // ANACAPA_CUDA_SHARED_TYPES_H
