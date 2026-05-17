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
    float     time;       // normalized shutter time [0,1] for motion blur
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
    kMatHair        = 4,   // Marschner cylindrical fiber BSDF (handled via hairTris/hairMats buffers)
};

struct GpuMaterial {
    GpuFloat3       baseColor;
    float           roughness;
    GpuFloat3       emissive;
    float           metalness;
    uint32_t        type;         // GpuMaterialType
    float           specularIOR;  // IOR for glass; unused for other types
    float           transmission; // 0 = opaque, 1 = fully transmissive
    // Specular layer strength [0,1] — MaterialX standard_surface `specular` knob.
    // Multiplies the dielectric Fresnel in the energy-conserving spec/diff balance.
    float           specular;
    // Caustic-generator opt-in (see IMaterial::isCausticGenerator).  When 0
    // the surface uses ordinary NEE-through-glass transmittance even with a
    // photon map active; when 1 the photon-map gate fires here.
    uint32_t        causticGenerator;
    // Subsurface scattering layer (SSS photon map).
    uint32_t        isSubsurface;       // 1 = has SSS layer
    float           subsurfaceWeight;   // [0,1] mixing weight
    GpuFloat3       subsurfaceColor;    // scatter tint
    float           subsurfaceRadius;   // effective d = radius * scale (precomputed by host)
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
// The GPU hair mesh is one BLAS; primID indexes directly into this buffer.
// h values encode the impact parameter h ∈ [-1,1] at each barycentric vertex
// so the shader can interpolate: h = h0*(1-u-v) + h1*u + h2*v.
// ---------------------------------------------------------------------------
struct GpuHairTri {
    GpuFloat3 tangent;  // fiber direction (world-space, unit) — same for both tris of a quad
    uint32_t  matIdx;   // index into hairMats[] buffer
    GpuFloat3 color;    // per-strand linear RGB; (1,1,1) = use GpuHairMaterial.sigma_a default
    float     h0;       // impact parameter at barycentric vertex 0 ∈ [-1,1]
    float     h1;       // impact parameter at barycentric vertex 1
    float     h2;       // impact parameter at barycentric vertex 2
    float     _pad;
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
    float     shutterOpen;    // normalized shutter open time  (typically 0.0)
    float     shutterClose;   // normalized shutter close time (typically 1.0)
    uint32_t  envMapWidth;    // CDF table width  (0 = no HDRI importance sampling)
    uint32_t  envMapHeight;   // CDF table height
    float     fireflyClamp;   // max luminance per path sample; 0 = disabled
    uint32_t  specLUTCosBins;   // 0 = no LUT uploaded
    uint32_t  specLUTRoughBins;

    // Pixel reconstruction filter — separable inverse-CDF tables.
    // pixelFilterBins == 0 disables and the kernel falls back to a unit
    // box-1.0 jitter (legacy behaviour).  CDF/sign buffers are bound
    // separately in the kernel argument table.
    uint32_t  pixelFilterBins;
    float     pixelFilterRadius;

    // Hair: TLAS instance ID where the tessellated hair mesh starts.
    // 0xFFFFFFFF when the scene contains no hair.
    uint32_t  hairMeshBaseID;

    // Camera motion blur — close-state image plane (= open-state when hasMotion=0).
    // The shade kernel lerps between open and close at rayTime before shooting.
    // Aperture disk offset for DoF is applied after the lerp so both compose.
    GpuFloat3 originClose;
    float     _padOC;
    GpuFloat3 lowerLeftClose;
    float     _padLC;
    GpuFloat3 horizontalClose;
    float     _padHC;
    GpuFloat3 verticalClose;
    float     _padVC;
    uint32_t  hasMotion;      // 0 = static camera, 1 = lerp at rayTime

    // Caustic photon map hash grid (enabled when photonMapEnabled != 0).
    // cellStart[numCells+1] and sortedPhotonIdx[numPhotons] are bound at
    // buffers 21 and 22; the photon array is at buffer 23.
    uint32_t  photonMapEnabled;    // 0 = off
    float     photonSearchRadius;  // density estimate search radius
    GpuFloat3 hashGridOrigin;      // world-space grid min corner
    float     hashCellSize;        // cell edge length (= searchRadius)
    uint32_t  hashGridDimX;
    uint32_t  hashGridDimY;
    uint32_t  hashGridDimZ;

    // SSS photon map hash grid (Metal buffers 24/25/26 for sssPhotons/sssCellStart/sssSortedIdx).
    uint32_t  sssMapEnabled;
    float     sssSearchRadius;
    GpuFloat3 sssHashOrigin;
    float     sssHashCellSize;
    uint32_t  sssHashDimX;
    uint32_t  sssHashDimY;
    uint32_t  sssHashDimZ;
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

// ---------------------------------------------------------------------------
// GpuPhoton — one caustic photon written by the photonTrace kernel.
// power == (0,0,0) means no valid caustic hit (no specular bounce before
// first diffuse); these slots are skipped when building the hash grid.
// ---------------------------------------------------------------------------
struct GpuPhoton {
    GpuFloat3 position;   // world-space hit position
    GpuFloat3 wi;         // photon travel direction (toward surface)
    GpuFloat3 power;      // radiant flux RGB (0,0,0 = invalid)
    uint32_t  _pad;
};

// ---------------------------------------------------------------------------
// GpuPhotonParams — uniform data for the photonTrace kernel.
// ---------------------------------------------------------------------------
struct GpuPhotonParams {
    uint32_t numPhotons;   // total threads dispatched = total photon slots
    uint32_t frameIndex;   // seed variation per frame
    uint32_t _pad[2];
};

#endif // ANACAPA_SHARED_TYPES_H
