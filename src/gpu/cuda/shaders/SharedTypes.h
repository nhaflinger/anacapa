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
struct GpuFloat4 { float x, y, z, w; };

// Per-material texture atlas cap — kept in sync with the Metal backend's
// kMaxGpuTextures for parity, though CUDA has no per-kernel-argument limit
// forcing this; it's just a fixed-size device array of texture handles here.
#define ANACAPA_MAX_GPU_TEXTURES 48
constexpr uint32_t kMaxGpuTextures = ANACAPA_MAX_GPU_TEXTURES;

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
    kMatTranslucent = 5,   // diffuse transmission — power-cosine lobe around straight-through dir
    kMatMaterialX   = 6,   // MaterialX-codegen'd procedural surface params, feeds evalLayeredBSDF()
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
    // Caustic-generator opt-in (see IMaterial::isCausticGenerator).  When 0
    // the surface uses ordinary NEE-through-glass transmittance even with a
    // photon map active; when 1 the photon-map gate fires here.
    uint32_t   causticGenerator;
    // Subsurface scattering layer (SSS photon map).
    uint32_t   isSubsurface;       // 1 = has SSS layer
    float      subsurfaceWeight;   // [0,1] mixing weight
    GpuFloat3  subsurfaceColor;    // scatter tint
    float      subsurfaceRadius;   // effective d = radius * scale (precomputed by host)
    float      subsurfaceStrength; // independent amplifier (> 1 boosts SSS contribution)
    float      scatter;            // translucent lobe width: 0=delta straight-through, 1=Lambertian
    // Texture atlas indices; -1 = no texture, use the constant field above instead.
    int32_t    baseColorTexIdx;
    int32_t    normalMapTexIdx;
    float      normalScale;
    float      normalBias;
    // Index into LaunchParams::mxDispatch when type == kMatMaterialX; -1 otherwise.
    int32_t    mxDispatchIdx;
    // 1 = apply Kulla-Conty multi-scatter GGX compensation in evalLayeredBSDF
    // (matches CPU's StandardSurfaceMaterial, which uses the same LUTs — see
    // StandardSurface.h); 0 = pure single-scatter (matches CPU's OslMaterial
    // conductor/dielectric lobes, evalGGXReflLobe in OslMaterial.cpp, which
    // never applies this compensation). Set per-material in
    // extractGpuMaterial() — mirrors the identical Metal backend split.
    uint32_t   multiScatter;
};

// ---------------------------------------------------------------------------
// MaterialX GPU codegen dispatch — CUDA counterpart of the Metal backend's
// identically-named struct (src/gpu/metal/shaders/SharedTypes.h). Each
// kMatMaterialX GpuMaterial owns one MxDispatchEntry. A slot's funcIndex is
// kMxNoFunction when that surface input was a plain literal for this
// material. Unlike Metal (which needs a visible_function_table + argument
// buffer to dispatch dynamically), CUDA/OptiX dispatches generated functions
// via optixDirectCall<>(sbtIndex, ...) — funcIndex here IS that SBT index
// directly (offset by however many non-MaterialX callables precede them;
// see CudaPathIntegrator.cu's buildMaterialXCallables()).
// ---------------------------------------------------------------------------
#define ANACAPA_MX_NO_FUNCTION 0xFFFFFFFFu
constexpr uint32_t kMxNoFunction = ANACAPA_MX_NO_FUNCTION;

struct MxDispatchEntry {
    uint32_t baseColorFunc;    // optixDirectCall SBT index, or kMxNoFunction
    uint32_t baseColorOffset;  // float offset into LaunchParams::mxUniforms
    uint32_t roughnessFunc;
    uint32_t roughnessOffset;
    uint32_t metalnessFunc;
    uint32_t metalnessOffset;
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
// GpuHaloDesc — one camera-facing disc particle (matches CPU HaloDesc).
// center/centerClose are motion-blur endpoints; radius is disc radius.
// color is the per-particle tint (used as emission when matIdx has no Le).
// matIdx indexes into materials[] for Le / opacity lookup.
// ---------------------------------------------------------------------------
struct GpuHaloDesc {
    GpuFloat3 center;
    float     radius;
    GpuFloat3 centerClose;
    uint32_t  matIdx;
    GpuFloat3 color;
    float     _pad;
};

// ---------------------------------------------------------------------------
// GpuHaloNode — binary SAH BVH node over halo disc particles.
// Interior: left_or_prim = left child, right_or_count = right child.
// Leaf:     left_or_prim = first index in haloPrimIdx,
//           right_or_count = count | 0x80000000.
// Matches CPU HaloNode memory layout exactly.
// ---------------------------------------------------------------------------
struct GpuHaloNode {
    float    bmin[3];
    float    bmax[3];
    uint32_t left_or_prim;
    uint32_t right_or_count;
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

    // Halo disc particles (UsdGeomPoints).  Software BVH walked inline in
    // wf_bounce — halos are drained per-bounce in front of the mesh hit
    // without consuming the bounce budget.  0 = no halos.
    uint32_t  numHalos;

    // Camera motion blur — close-state image plane (= open-state when hasMotion=0).
    // The primary raygen lerps between open and close at rayTime before shooting.
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

    // Depth of field — thin lens (apertureRadius == 0 → pinhole)
    float     apertureRadius;  // half aperture diameter in world units
    float     focalDistance;   // distance to focal plane in world units
    GpuFloat3 basisU;          // camera right unit vector
    float     _padBU;
    GpuFloat3 basisV;          // camera up unit vector
    float     _padBV;

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

    // SSS photon map hash grid (enabled when sssMapEnabled != 0).
    uint32_t  sssMapEnabled;
    float     sssSearchRadius;
    GpuFloat3 sssHashOrigin;
    float     sssHashCellSize;
    uint32_t  sssHashDimX;
    uint32_t  sssHashDimY;
    uint32_t  sssHashDimZ;
    uint32_t  transparentBg;  // 1 = sky transparent at bounce 0 (compositing)
    uint32_t  _sky_pad[3];
};

// ---------------------------------------------------------------------------
// GpuAccumPixel
// ---------------------------------------------------------------------------
struct GpuAccumPixel {
    float r, g, b, weight;
    float sumLumSq;  // sum(luminance(sample)^2) — needed for variance-based adaptive sampling
    float alpha;     // sum of per-sample alpha weights (0=transparent sky, fw=opaque)
    float _pad;
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

// ---------------------------------------------------------------------------
// WfRayState — per-ray persistent state for the wavefront path tracer.
//
// The wavefront design moves the bounce loop out of the raygen kernel:
//   - __raygen__wf_primary  generates the camera ray and writes initial state
//   - __raygen__wf_bounce   runs ONE bounce iteration (trace + NEE + BSDF +
//                           sample next dir) and updates state.  Host loops
//                           this N=maxDepth times.
//   - __raygen__wf_finalize adds L into the film pixel.
//
// Splitting the loop reduces per-kernel register pressure, which dominates
// throughput on register-constrained GPUs (RTX A400 in particular).
// ---------------------------------------------------------------------------
struct WfRayState {
    GpuFloat3 origin;
    float     rayTime;       // shutter sample for this path
    GpuFloat3 dir;
    uint32_t  flags;         // bit 0: terminated, bit 1: prevWasDelta
    GpuFloat3 throughput;
    uint32_t  pixelIdx;      // flat index into the accum buffer
    GpuFloat3 L;
    uint32_t  rng;
    GpuFloat3 prevN;         // for emitter-PDF MIS weight on the next vertex
    float     prevBsdfPdf;
    float     pixelWeight;   // pixel-filter sign weight (±1)
    uint32_t  bounce;        // non-delta bounce depth (matches megakernel)
    uint32_t  glassDepth;    // delta bounces taken; capped at 16
    uint32_t  _pad;
};

enum : uint32_t {
    kWfTerminated    = 1u << 0,
    kWfPrevWasDelta  = 1u << 1,
    kWfCausticChain  = 1u << 2,  // delta chain crossed a caustic-flagged surface
    kWfTransparentSky = 1u << 3, // bounce-0 primary ray hit transparent sky (alpha=0)
};

#endif // ANACAPA_CUDA_SHARED_TYPES_H
