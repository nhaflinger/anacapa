#ifdef ANACAPA_ENABLE_CUDA

#include "CudaPathIntegrator.h"
#include "CudaContext.h"
#include "CudaBuffer.h"
#include "CudaAccelStructure.h"
#include "shaders/SharedTypes.h"
#include "shaders/LaunchParams.h"

#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/film/Film.h>
#include <anacapa/accel/GeometryPool.h>

#include "../../shading/Lambertian.h"
#include "../../shading/StandardSurface.h"
#include "../../shading/MarschnerHair.h"
#include "../../shading/ChiangHair.h"
#include "../../shading/lights/AreaLight.h"
#include "../../shading/lights/DirectionalLight.h"
#include "../../shading/lights/DomeLight.h"

#include <cuda_runtime.h>

#ifdef ANACAPA_ENABLE_OPTIX
#include <optix.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#endif

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
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
// Material / light extraction (identical logic to Metal backend)
// ---------------------------------------------------------------------------
static GpuMaterial extractGpuMaterial(const IMaterial* mat) {
    GpuMaterial gm{};
    gm.baseColor   = {0.5f, 0.5f, 0.5f};
    gm.emissive    = {0.f,  0.f,  0.f};
    gm.roughness   = 1.f;
    gm.metalness   = 0.f;
    gm.specularIOR = 1.5f;
    gm.specular    = 0.5f;       // matches MaterialX standard_surface default
    gm.type        = kMatLambertian;
    gm.causticGenerator = 0u;
    if (!mat) return gm;
    gm.causticGenerator = mat->isCausticGenerator() ? 1u : 0u;

    // Sample emission and reflectance up front for every material — UsdPreview-
    // Surface lights load as StandardSurfaceMaterial with emissive_color set, not
    // as EmissiveMaterial, so a dynamic_cast alone misses them.
    SurfaceInteraction si; si.n = si.ng = {0,0,1};
    ShadingContext ctx(si, {0,0,1});
    const Spectrum LeSamp  = mat->Le(ctx, {0,0,1});
    const Spectrum albSamp = mat->reflectance(ctx);
    gm.emissive  = {LeSamp.x,  LeSamp.y,  LeSamp.z};
    gm.baseColor = {albSamp.x, albSamp.y, albSamp.z};

    // Pure-emissive: an EmissiveMaterial subclass, or anything whose emission
    // dominates a near-black albedo.  Treated as a delta light surface — the
    // raygen adds Le and breaks without evaluating the BSDF.
    const EmissiveMaterial* em = dynamic_cast<const EmissiveMaterial*>(mat);
    const float emMax  = std::max({LeSamp.x,  LeSamp.y,  LeSamp.z});
    const float albMax = std::max({albSamp.x, albSamp.y, albSamp.z});
    const bool emissionDominated = emMax > 1e-3f && albMax < 0.05f;
    if (em || emissionDominated) {
        gm.type = kMatEmissive;
        return gm;
    }
    {
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum tint = mat->transmittanceColor(ctx);
        bool isTransmissive = (tint.x > 0.1f || tint.y > 0.1f || tint.z > 0.1f);

        const StandardSurfaceMaterial* ssm = dynamic_cast<const StandardSurfaceMaterial*>(mat);
        if (ssm && ssm->params().transmission > 0.001f && ssm->params().metalness.value < 0.001f) {
            gm.type         = kMatGlass;
            gm.specularIOR  = ssm->params().specular_IOR;
            gm.transmission = ssm->params().transmission;
            Spectrum alb = mat->reflectance(ctx);
            gm.baseColor = {alb.x, alb.y, alb.z};
            gm.roughness = ssm->params().roughness.value;
            return gm;
        }
        if (isTransmissive) {
            gm.type         = kMatGlass;
            gm.specularIOR  = 1.5f;
            gm.transmission = 1.0f;
            Spectrum alb = mat->reflectance(ctx);
            gm.baseColor = {alb.x, alb.y, alb.z};
            return gm;
        }
    }
    // Any StandardSurfaceMaterial (including diffuse-flagged ones — Cornell
    // walls land here) goes through the GGX layered model so the GPU shader
    // can apply the same spec/diff balance + Disney retro-reflection +
    // Kulla-Conty multi-scatter compensation as the CPU's evalCombined.
    if (const StandardSurfaceMaterial* ssm =
            dynamic_cast<const StandardSurfaceMaterial*>(mat)) {
        gm.type = kMatGGX;
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor   = {alb.x, alb.y, alb.z};
        gm.roughness   = ssm->params().roughness.value;
        gm.metalness   = ssm->params().metalness.value;
        gm.specular    = ssm->params().specular.value;
        gm.specularIOR = ssm->params().specular_IOR;
        return gm;
    }
    if (mat->flags() & BSDFFlag_Glossy) {
        gm.type = kMatGGX;
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
        gm.roughness = mat->roughness();
        gm.metalness = mat->metalness();
        return gm;
    }
    {
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
    }
    return gm;
}

static GpuHairMaterial extractGpuHairMaterial(const IMaterial* mat) {
    GpuHairMaterial hm{};
    hm.sigma_a = {0.06f, 0.10f, 0.20f};
    hm.eta     = 1.55f;
    hm.beta_m  = 0.40f;
    hm.beta_n  = 0.60f;
    hm.alpha   = 2.0f;
    hm._pad    = 0.f;

    if (!mat) return hm;

    if (const ChiangHairMaterial* ch = dynamic_cast<const ChiangHairMaterial*>(mat)) {
        const auto& p = ch->params();
        hm.sigma_a = {p.sigma_a.x, p.sigma_a.y, p.sigma_a.z};
        hm.eta     = p.eta;
        hm.beta_m  = p.beta_m;
        hm.beta_n  = p.beta_n;
        hm.alpha   = p.alpha;
        return hm;
    }
    if (const MarschnerHairMaterial* mh = dynamic_cast<const MarschnerHairMaterial*>(mat)) {
        const auto& p = mh->params();
        hm.sigma_a = {p.sigma_a.x, p.sigma_a.y, p.sigma_a.z};
        hm.eta     = p.eta;
        hm.beta_m  = p.beta_m;
        hm.beta_n  = p.beta_n;
        hm.alpha   = p.alpha;
        return hm;
    }
    return hm;
}

static GpuLight extractGpuLight(const ILight* light) {
    GpuLight gl{};
    if (!light) return gl;
    const AreaLight* al = dynamic_cast<const AreaLight*>(light);
    if (al) {
        gl.type = kLightRect;
        LightLeSample le  = al->sampleLe({0.5f,0.5f},{0.5f,0.5f});
        LightLeSample le0 = al->sampleLe({0.f, 0.5f},{0.5f,0.5f});
        LightLeSample le1 = al->sampleLe({1.f, 0.5f},{0.5f,0.5f});
        LightLeSample le2 = al->sampleLe({0.5f,0.f}, {0.5f,0.5f});
        LightLeSample le3 = al->sampleLe({0.5f,1.f}, {0.5f,0.5f});
        gl.Le       = {le.Le.x,     le.Le.y,     le.Le.z};
        gl.position = {le.pos.x,    le.pos.y,    le.pos.z};
        gl.normal   = {le.normal.x, le.normal.y, le.normal.z};
        gl.area     = 1.0f / le.pdfPos;
        Vec3f uF = {le1.pos.x-le0.pos.x, le1.pos.y-le0.pos.y, le1.pos.z-le0.pos.z};
        Vec3f vF = {le3.pos.x-le2.pos.x, le3.pos.y-le2.pos.y, le3.pos.z-le2.pos.z};
        gl.uHalf = {uF.x*0.5f, uF.y*0.5f, uF.z*0.5f};
        gl.vHalf = {vF.x*0.5f, vF.y*0.5f, vF.z*0.5f};
        return gl;
    }
    const DirectionalLight* dl = dynamic_cast<const DirectionalLight*>(light);
    if (dl) {
        gl.type = kLightDirectional;
        LightSample ls = dl->sample({0,0,0}, {0,1,0}, {0.5f, 0.5f});
        gl.Le     = {ls.Li.x, ls.Li.y, ls.Li.z};
        gl.normal = {ls.wi.x, ls.wi.y, ls.wi.z};
        LightLeSample le0 = dl->sampleLe({0.5f, 0.5f}, {0.f, 0.f});
        LightLeSample le1 = dl->sampleLe({0.5f, 0.5f}, {1.f, 0.f});
        Vec3f d0 = le0.dir, d1 = le1.dir;
        float cc = d0.x*d1.x + d0.y*d1.y + d0.z*d1.z;
        gl.cosCone = std::max(0.f, std::min(1.f, cc));
        return gl;
    }
    const DomeLight* dome = dynamic_cast<const DomeLight*>(light);
    if (dome) {
        gl.type = kLightDome;
        static const Vec3f kDirs[] = {
            {0,1,0},{0,-1,0},{1,0,0},{-1,0,0},{0,0,1},{0,0,-1},
            {0.577f,0.577f,0.577f},{-0.577f,0.577f,0.577f},
            {0.577f,0.577f,-0.577f},{-0.577f,0.577f,-0.577f},
            {0.577f,-0.577f,0.577f},{-0.577f,-0.577f,0.577f},
            {0.577f,-0.577f,-0.577f},{-0.577f,-0.577f,-0.577f},
        };
        Spectrum avg{};
        for (const Vec3f& d : kDirs) avg += dome->Le({},{},d);
        avg = avg * (1.f/14.f);
        gl.Le = {avg.x, avg.y, avg.z};
        return gl;
    }
    gl.type = kLightRect;
    gl.Le   = {0.f, 0.f, 0.f};
    return gl;
}

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct CudaPathIntegrator::Impl {
    std::unique_ptr<CudaContext>        ctx;
    std::unique_ptr<CudaAccelStructure> accel;

    CudaBuffer<GpuMaterial>  d_materials;
    CudaBuffer<GpuLight>     d_lights;

    // Persistent full-frame accumulation buffer.
    // Reused across renderFrame() calls so samples accumulate without
    // re-allocation. clearAccum() zeros it when the scene/camera changes.
    CudaBuffer<GpuAccumPixel> d_accum;
    uint32_t                  accumWidth  = 0;
    uint32_t                  accumHeight = 0;

    cudaArray_t         envArray   = nullptr;
    cudaTextureObject_t envTex     = 0;
    Vec3f               envRot[3]  = {{1,0,0},{0,1,0},{0,0,1}};
    float               envIntensity = 1.0f;

    // HDRI importance sampling — CDF tables (empty when no DomeLight or no
    // texture present).  envCdfWidth/Height = 0 signals the GPU to fall
    // back to the cosine-hemisphere prior.
    CudaBuffer<float>   d_envMarginalCdf;
    CudaBuffer<float>   d_envConditionalCdf;
    uint32_t            envCdfWidth  = 0;
    uint32_t            envCdfHeight = 0;

    // GGX dielectric energy-compensation LUTs, mirroring StandardSurface's CPU
    // tables.  specAlbedo: 32x32 floats indexed [cos * N_R + rough].
    // specAvgAlbedo: 32 floats indexed by roughness.  Uploaded once in
    // prepare() and reused across launches.
    CudaBuffer<float>   d_specAlbedoLUT;
    CudaBuffer<float>   d_specAvgAlbedoLUT;
    uint32_t            specLUTCosBins   = 0;
    uint32_t            specLUTRoughBins = 0;

    // Pixel reconstruction filter — host PixelFilter pointer, uploaded
    // CDF/sign tables.  pixelFilterBins == 0 means raygen falls back to
    // a unit box-1.0 jitter (legacy behaviour).
    const PixelFilter*  pixelFilter        = nullptr;
    CudaBuffer<float>   d_pixelFilterCdf;
    CudaBuffer<float>   d_pixelFilterSigns;
    uint32_t            pixelFilterBins    = 0;
    float               pixelFilterRadius  = 0.f;

    // Hair — per-material BSDF parameters (one slot per scene.materials entry).
    // The per-triangle GpuHairTri buffer lives in CudaAccelStructure; the
    // tessellated hair geometry is the second IAS instance (instanceID =
    // accel->hairMeshBaseID()).  hairMeshBaseID == 0xFFFFFFFF means no hair.
    CudaBuffer<GpuHairMaterial> d_hairMats;
    uint32_t                    hairMeshBaseID = 0xFFFFFFFFu;

    // Caustic photon map — GPU trace + CPU hash grid build + GPU query.
    CudaBuffer<GpuPhoton>       d_photons;
    CudaBuffer<uint32_t>        d_hashCellStart;
    CudaBuffer<uint32_t>        d_sortedPhotonIdx;
    bool                        photonMapEnabled   = false;
    uint32_t                    numPhotons         = 0;
    float                       photonSearchRadius = 0.1f;
    GpuFloat3                   hashGridOrigin     = {0.f, 0.f, 0.f};
    float                       hashCellSize       = 0.1f;
    uint32_t                    hashGridDimX       = 0;
    uint32_t                    hashGridDimY       = 0;
    uint32_t                    hashGridDimZ       = 0;
    uint32_t                    validPhotons       = 0;
    uint32_t                    photonFrameIndex   = 0;

    uint32_t numMaterials = 0;
    uint32_t numLights    = 0;
    uint32_t maxDepth     = 6;
    float    fireflyClamp = 10.f;
    bool     preparedOnce = false;

#ifdef ANACAPA_ENABLE_OPTIX
    // Pipeline state — created lazily on first render.  Lives until destructor.
    OptixModule           optixModule    = nullptr;
    OptixProgramGroup     pgRaygen       = nullptr;
    OptixProgramGroup     pgMiss         = nullptr;
    OptixProgramGroup     pgHit          = nullptr;
    OptixProgramGroup     pgPhotonRg     = nullptr;   // __raygen__photon
    OptixProgramGroup     pgWfPrimary    = nullptr;
    OptixProgramGroup     pgWfBounce     = nullptr;
    OptixProgramGroup     pgWfFinalize   = nullptr;
    OptixPipeline         optixPipeline  = nullptr;   // megakernel (shade + photon)
    OptixPipeline         wfPipeline     = nullptr;   // wavefront (3 raygens)
    CudaByteBuffer        sbtRaygenBuf;
    CudaByteBuffer        sbtPhotonRaygenBuf;        // SBT raygen record for photon pass
    CudaByteBuffer        sbtWfPrimaryBuf;
    CudaByteBuffer        sbtWfBounceBuf;
    CudaByteBuffer        sbtWfFinalizeBuf;
    CudaByteBuffer        sbtMissBuf;
    CudaByteBuffer        sbtHitBuf;
    OptixShaderBindingTable sbt           = {};   // megakernel: __raygen__rg
    OptixShaderBindingTable sbtPhoton     = {};   //              __raygen__photon
    OptixShaderBindingTable sbtWfPrimary  = {};
    OptixShaderBindingTable sbtWfBounce   = {};
    OptixShaderBindingTable sbtWfFinalize = {};
    CudaBuffer<LaunchParams> d_launchParams;  // single-element buffer in device mem
    bool                  optixReady     = false;

    // Wavefront state buffer (allocated lazily by renderFrameWavefront).
    CudaBuffer<WfRayState> d_wfRays;
    uint32_t               wfRayCount    = 0;
    bool                   wavefrontMode = false;

    bool buildOptixPipeline(OptixDeviceContext ctx);
    void destroyOptixPipeline();
    void buildPhotonMap(OptixDeviceContext ctx);
    bool renderFrameWavefront(const SceneView& scene,
                                uint32_t filmWidth, uint32_t filmHeight,
                                uint32_t sampleStart, uint32_t sampleCount,
                                Film& film);
    void renderTileWavefront(const SceneView& scene,
                              uint32_t filmWidth, uint32_t filmHeight,
                              uint32_t tileX0, uint32_t tileY0,
                              uint32_t tileW,  uint32_t tileH,
                              uint32_t sampleStart, uint32_t sampleCount,
                              TileBuffer& out);
#endif

    ~Impl() {
        if (envTex)   cudaDestroyTextureObject(envTex);
        if (envArray) cudaFreeArray(envArray);
#ifdef ANACAPA_ENABLE_OPTIX
        destroyOptixPipeline();
#endif
    }

    void ensureAccum(uint32_t w, uint32_t h) {
        if (d_accum.isValid() && accumWidth == w && accumHeight == h) return;
        d_accum     = CudaBuffer<GpuAccumPixel>(w * h);
        d_accum.zero();
        accumWidth  = w;
        accumHeight = h;
    }
    void clearAccum() {
        if (d_accum.isValid()) d_accum.zero();
    }

    void fillLaunchParams(LaunchParams& p, const SceneView& scene,
                          uint32_t filmWidth, uint32_t filmHeight,
                          uint32_t tileX0, uint32_t tileY0,
                          uint32_t tileW,   uint32_t tileH,
                          uint32_t sampleStart, uint32_t sampleCount,
                          GpuAccumPixel* d_accum) const;
};

#ifdef ANACAPA_ENABLE_OPTIX

namespace {
// SBT records carry only the OptiX header for now.  Per-record material data
// will be added when the SBT is used for material dispatch (Phase 6 step 6
// in the original plan; not yet wired).
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};
}  // namespace

bool CudaPathIntegrator::Impl::buildOptixPipeline(OptixDeviceContext ctx)
{
    if (optixReady) return true;

    // ---- Read PTX from disk -------------------------------------------------
    const std::string ptxPath = std::string(ANACAPA_PTX_DIR) + "/Shade.ptx";
    std::ifstream ptxFile(ptxPath, std::ios::binary);
    if (!ptxFile.is_open()) {
        fprintf(stderr, "[error] CudaPathIntegrator: could not open '%s' — "
                        "PTX target was not built\n", ptxPath.c_str());
        return false;
    }
    std::stringstream ss;
    ss << ptxFile.rdbuf();
    const std::string ptxSrc = ss.str();

    // ---- Module -------------------------------------------------------------
    OptixModuleCompileOptions modOpts{};
    modOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    modOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    modOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeOpts{};
    pipeOpts.usesMotionBlur                   = 1;  // motion always allowed; static GAS still works
    pipeOpts.traversableGraphFlags            =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeOpts.numPayloadValues                 = 6;
    pipeOpts.numAttributeValues               = 2;  // triangle barycentrics
    pipeOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "params";
    pipeOpts.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    char   log[4096];
    size_t logSize = sizeof(log);
    OPTIX_CHECK(optixModuleCreate(ctx, &modOpts, &pipeOpts,
                                  ptxSrc.c_str(), ptxSrc.size(),
                                  log, &logSize, &optixModule));

    // ---- Program groups -----------------------------------------------------
    OptixProgramGroupOptions pgOpts{};
    OptixProgramGroupDesc    pgDesc{};

    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = optixModule;
    pgDesc.raygen.entryFunctionName = "__raygen__rg";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts,
                                        log, &logSize, &pgRaygen));

    pgDesc = {};
    pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module            = optixModule;
    pgDesc.raygen.entryFunctionName = "__raygen__photon";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts,
                                        log, &logSize, &pgPhotonRg));

    pgDesc = {};
    pgDesc.kind                   = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.miss.module            = optixModule;
    pgDesc.miss.entryFunctionName = "__miss__ms";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts,
                                        log, &logSize, &pgMiss));

    pgDesc = {};
    pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH            = optixModule;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    logSize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(ctx, &pgDesc, 1, &pgOpts,
                                        log, &logSize, &pgHit));

    // ---- Pipeline -----------------------------------------------------------
    OptixPipelineLinkOptions linkOpts{};
    linkOpts.maxTraceDepth = 1;  // raygen is the only invoker; no recursion

    OptixProgramGroup pgs[4] = { pgRaygen, pgPhotonRg, pgMiss, pgHit };
    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(ctx, &pipeOpts, &linkOpts,
                                    pgs, 4, log, &logSize, &optixPipeline));

    // ---- Stack sizes --------------------------------------------------------
    OptixStackSizes stackSizes{};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgRaygen,   &stackSizes, optixPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgPhotonRg, &stackSizes, optixPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgMiss,     &stackSizes, optixPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgHit,      &stackSizes, optixPipeline));

    uint32_t dcStackTrav = 0, dcStackState = 0, contStack = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                           /*maxTraceDepth=*/1,
                                           /*maxCCDepth=*/0,
                                           /*maxDCDepth=*/0,
                                           &dcStackTrav, &dcStackState, &contStack));
    OPTIX_CHECK(optixPipelineSetStackSize(optixPipeline,
                                          dcStackTrav, dcStackState, contStack,
                                          /*maxTraversableDepth=*/2));

    // ---- Shader binding table -----------------------------------------------
    SbtRecord raygenRec{}, photonRaygenRec{}, missRec{}, hitRec{};
    OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen,   &raygenRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgPhotonRg, &photonRaygenRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss,     &missRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgHit,      &hitRec));

    sbtRaygenBuf = CudaByteBuffer(sizeof(SbtRecord));
    sbtRaygenBuf.upload(reinterpret_cast<const uint8_t*>(&raygenRec), sizeof(SbtRecord));

    sbtPhotonRaygenBuf = CudaByteBuffer(sizeof(SbtRecord));
    sbtPhotonRaygenBuf.upload(reinterpret_cast<const uint8_t*>(&photonRaygenRec),
                                sizeof(SbtRecord));

    sbtMissBuf = CudaByteBuffer(sizeof(SbtRecord));
    sbtMissBuf.upload(reinterpret_cast<const uint8_t*>(&missRec), sizeof(SbtRecord));

    sbtHitBuf = CudaByteBuffer(sizeof(SbtRecord));
    sbtHitBuf.upload(reinterpret_cast<const uint8_t*>(&hitRec), sizeof(SbtRecord));

    sbt                              = {};
    sbt.raygenRecord                 = sbtRaygenBuf.devPtr();
    sbt.missRecordBase               = sbtMissBuf.devPtr();
    sbt.missRecordStrideInBytes      = sizeof(SbtRecord);
    sbt.missRecordCount              = 1;
    sbt.hitgroupRecordBase           = sbtHitBuf.devPtr();
    sbt.hitgroupRecordStrideInBytes  = sizeof(SbtRecord);
    sbt.hitgroupRecordCount          = 1;

    // sbtPhoton shares miss/hit with sbt; only the raygen record differs.
    sbtPhoton                           = sbt;
    sbtPhoton.raygenRecord              = sbtPhotonRaygenBuf.devPtr();

    // -----------------------------------------------------------------------
    // Wavefront pipeline — built unconditionally (cheap; only consumes a few
    // KiB of SBT records) so the user can toggle setWavefrontMode at run
    // time without re-initialising OptiX.  The shade megakernel pipeline
    // above stays untouched, so non-wavefront renders pay zero overhead.
    // -----------------------------------------------------------------------
    auto makeRaygenPg = [&](const char* entry, OptixProgramGroup& outPg) {
        OptixProgramGroupDesc d{};
        d.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        d.raygen.module            = optixModule;
        d.raygen.entryFunctionName = entry;
        size_t lg = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(ctx, &d, 1, &pgOpts,
                                            log, &lg, &outPg));
    };
    makeRaygenPg("__raygen__wf_primary",  pgWfPrimary);
    makeRaygenPg("__raygen__wf_bounce",   pgWfBounce);
    makeRaygenPg("__raygen__wf_finalize", pgWfFinalize);

    OptixProgramGroup wfPgs[5] = {
        pgWfPrimary, pgWfBounce, pgWfFinalize, pgMiss, pgHit
    };
    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(ctx, &pipeOpts, &linkOpts,
                                    wfPgs, 5, log, &logSize, &wfPipeline));

    OptixStackSizes wfStack{};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgWfPrimary,  &wfStack, wfPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgWfBounce,   &wfStack, wfPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgWfFinalize, &wfStack, wfPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgMiss,       &wfStack, wfPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgHit,        &wfStack, wfPipeline));
    {
        uint32_t dcStackTrav2 = 0, dcStackState2 = 0, contStack2 = 0;
        OPTIX_CHECK(optixUtilComputeStackSizes(&wfStack,
                                               /*maxTraceDepth=*/1,
                                               /*maxCCDepth=*/0,
                                               /*maxDCDepth=*/0,
                                               &dcStackTrav2, &dcStackState2, &contStack2));
        OPTIX_CHECK(optixPipelineSetStackSize(wfPipeline,
                                              dcStackTrav2, dcStackState2, contStack2,
                                              /*maxTraversableDepth=*/2));
    }

    // Build three raygen SBT records; miss+hit are shared with the megakernel.
    auto packAndUpload = [&](OptixProgramGroup pg, CudaByteBuffer& buf,
                              OptixShaderBindingTable& out) {
        SbtRecord rec{};
        OPTIX_CHECK(optixSbtRecordPackHeader(pg, &rec));
        buf = CudaByteBuffer(sizeof(SbtRecord));
        buf.upload(reinterpret_cast<const uint8_t*>(&rec), sizeof(SbtRecord));
        out = sbt;
        out.raygenRecord = buf.devPtr();
    };
    packAndUpload(pgWfPrimary,  sbtWfPrimaryBuf,  sbtWfPrimary);
    packAndUpload(pgWfBounce,   sbtWfBounceBuf,   sbtWfBounce);
    packAndUpload(pgWfFinalize, sbtWfFinalizeBuf, sbtWfFinalize);

    // ---- Launch params buffer (re-used across launches) ---------------------
    d_launchParams = CudaBuffer<LaunchParams>(1);

    optixReady = true;
    printf("[info]  CudaPathIntegrator: OptiX pipeline ready\n");
    return true;
}

void CudaPathIntegrator::Impl::destroyOptixPipeline()
{
    if (!optixReady) return;
    if (wfPipeline)    optixPipelineDestroy(wfPipeline);
    if (optixPipeline) optixPipelineDestroy(optixPipeline);
    if (pgWfFinalize)  optixProgramGroupDestroy(pgWfFinalize);
    if (pgWfBounce)    optixProgramGroupDestroy(pgWfBounce);
    if (pgWfPrimary)   optixProgramGroupDestroy(pgWfPrimary);
    if (pgHit)         optixProgramGroupDestroy(pgHit);
    if (pgMiss)        optixProgramGroupDestroy(pgMiss);
    if (pgPhotonRg)    optixProgramGroupDestroy(pgPhotonRg);
    if (pgRaygen)      optixProgramGroupDestroy(pgRaygen);
    if (optixModule)   optixModuleDestroy(optixModule);
    wfPipeline = optixPipeline = nullptr;
    pgWfFinalize = pgWfBounce = pgWfPrimary = nullptr;
    pgHit = pgMiss = pgPhotonRg = pgRaygen = nullptr;
    optixModule = nullptr;
    optixReady = false;
}

#endif  // ANACAPA_ENABLE_OPTIX

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------
CudaPathIntegrator::CudaPathIntegrator()
    : m_impl(std::make_unique<Impl>())
{
    m_impl->ctx = CudaContext::create();
    if (!m_impl->ctx || !m_impl->ctx->isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator: context init failed\n");
        return;
    }
    printf("[info]  CudaPathIntegrator: ready on '%s'\n", m_impl->ctx->name().c_str());
}

CudaPathIntegrator::~CudaPathIntegrator() = default;

bool CudaPathIntegrator::isValid() const {
    return m_impl->ctx && m_impl->ctx->isValid();
}

void CudaPathIntegrator::clearAccum() {
    m_impl->clearAccum();
}

void CudaPathIntegrator::setFireflyClamp(float v) {
    m_impl->fireflyClamp = v;
}

void CudaPathIntegrator::setPixelFilter(const PixelFilter* f) {
    m_impl->pixelFilter = f;
    if (!f) {
        m_impl->d_pixelFilterCdf   = CudaBuffer<float>{};
        m_impl->d_pixelFilterSigns = CudaBuffer<float>{};
        m_impl->pixelFilterBins    = 0;
        m_impl->pixelFilterRadius  = 0.f;
        return;
    }
    const auto& cdf   = f->cdf();
    const auto& signs = f->signs();
    m_impl->d_pixelFilterCdf = CudaBuffer<float>(cdf.size());
    m_impl->d_pixelFilterCdf.upload(cdf);
    m_impl->d_pixelFilterSigns = CudaBuffer<float>(signs.size());
    m_impl->d_pixelFilterSigns.upload(signs);
    m_impl->pixelFilterBins   = static_cast<uint32_t>(signs.size());
    m_impl->pixelFilterRadius = f->radius();
}

void CudaPathIntegrator::setPhotonMap(int numPhotons, float searchRadius) {
    m_impl->photonMapEnabled   = (numPhotons > 0);
    m_impl->numPhotons         = static_cast<uint32_t>(std::max(0, numPhotons));
    m_impl->photonSearchRadius = searchRadius;
}

void CudaPathIntegrator::setWavefrontMode(bool on) {
    m_impl->wavefrontMode = on;
    if (on) printf("[info]  CudaPathIntegrator: wavefront mode enabled\n");
}

#ifdef ANACAPA_ENABLE_OPTIX
// ---------------------------------------------------------------------------
// buildPhotonMap — GPU photon trace → host hash grid build → device upload.
//
// Mirrors MetalPathIntegrator::buildPhotonMap step for step:
//   1. allocate the photon output buffer
//   2. fill LaunchParams in photon-trace mode and optixLaunch __raygen__photon
//   3. download photons, compute AABB of valid hits
//   4. assign each photon to a uniform-grid cell, sort by cell, build a
//      prefix-sum cellStart array + sortedPhotonIdx list
//   5. upload both to the device for the shade kernel's hash-grid query
// ---------------------------------------------------------------------------
void CudaPathIntegrator::Impl::buildPhotonMap(OptixDeviceContext optixCtx)
{
    if (!optixReady || numPhotons == 0) return;

    printf("[info]  CudaPathIntegrator: tracing %u photons on GPU...\n", numPhotons);

    // ---- 1. Photon-trace scratch buffer (full numPhotons slots).
    // Local to this function — replaced by a compacted, cell-sorted device
    // buffer at the end so the shade kernel doesn't drag 8 MB of mostly-
    // invalid photons through L2 on every hash lookup.
    CudaBuffer<GpuPhoton> traceBuf(numPhotons);
    if (!traceBuf.isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::buildPhotonMap: "
                        "photon alloc failed (%u entries)\n", numPhotons);
        photonMapEnabled = false;
        return;
    }
    traceBuf.zero();

    // ---- 2. Launch the photon raygen --------------------------------------
    CUstream stream = static_cast<CUstream>(ctx->cuStream());
    LaunchParams params{};
    params.cam.imageWidth      = 1;
    params.cam.imageHeight     = 1;
    params.cam.maxDepth        = 8;
    params.cam.hairMeshBaseID  = accel->hairMeshBaseID();
    params.cam.photonMapEnabled = 0;   // photon-trace doesn't consume the map
    params.cam.shutterOpen     = 0.f;
    params.cam.shutterClose    = 0.f;
    params.lights              = d_lights.ptr();
    params.numLights           = numLights;
    params.materials           = d_materials.ptr();
    params.numMaterials        = numMaterials;
    params.normals             = reinterpret_cast<const GpuFloat3*>(accel->normalBuffer());
    params.indices             = reinterpret_cast<const uint32_t*>(accel->indexBuffer());
    params.triMeshIDs          = reinterpret_cast<const uint32_t*>(accel->triMeshIDBuffer());
    params.meshVertexOffsets   = reinterpret_cast<const uint32_t*>(accel->meshVertexOffsetBuffer());
    params.meshIndexOffsets    = reinterpret_cast<const uint32_t*>(accel->meshIndexOffsetBuffer());
    params.photons             = traceBuf.ptr();   // raygen writes into this
    params.numPhotons          = numPhotons;
    params.frameIndex          = photonFrameIndex++;
    params.handle              = accel->traversableHandle();

    d_launchParams.upload(&params, 1);
    OPTIX_CHECK(optixLaunch(optixPipeline, stream,
                            d_launchParams.devPtr(),
                            sizeof(LaunchParams),
                            &sbtPhoton,
                            numPhotons, /*h=*/1, /*d=*/1));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // ---- 3. Download photons + compute AABB of valid hits ------------------
    std::vector<GpuPhoton> hostPhotons;
    traceBuf.download(hostPhotons);

    float minX =  1e30f, minY =  1e30f, minZ =  1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
    uint32_t validCount = 0;
    for (uint32_t i = 0; i < numPhotons; ++i) {
        const GpuPhoton& ph = hostPhotons[i];
        if (ph.power.x == 0.f && ph.power.y == 0.f && ph.power.z == 0.f) continue;
        ++validCount;
        minX = std::min(minX, ph.position.x);
        minY = std::min(minY, ph.position.y);
        minZ = std::min(minZ, ph.position.z);
        maxX = std::max(maxX, ph.position.x);
        maxY = std::max(maxY, ph.position.y);
        maxZ = std::max(maxZ, ph.position.z);
    }
    validPhotons = validCount;
    printf("[info]  CudaPathIntegrator: %u valid caustic photons (of %u traced)\n",
           validCount, numPhotons);

    if (validCount == 0) {
        d_photons         = CudaBuffer<GpuPhoton>{};
        d_hashCellStart   = CudaBuffer<uint32_t>{};
        d_sortedPhotonIdx = CudaBuffer<uint32_t>{};
        return;
    }

    // ---- 4. Build the uniform hash grid on the host ------------------------
    const float cs = photonSearchRadius;
    hashCellSize = cs;
    minX -= cs; minY -= cs; minZ -= cs;
    maxX += cs; maxY += cs; maxZ += cs;
    hashGridOrigin = {minX, minY, minZ};
    hashGridDimX = static_cast<uint32_t>(std::ceil((maxX - minX) / cs)) + 1u;
    hashGridDimY = static_cast<uint32_t>(std::ceil((maxY - minY) / cs)) + 1u;
    hashGridDimZ = static_cast<uint32_t>(std::ceil((maxZ - minZ) / cs)) + 1u;
    uint32_t numCells = hashGridDimX * hashGridDimY * hashGridDimZ;

    struct PhotonCell { uint32_t photonIdx; uint32_t cellIdx; };
    std::vector<PhotonCell> assignments;
    assignments.reserve(validCount);
    for (uint32_t i = 0; i < numPhotons; ++i) {
        const GpuPhoton& ph = hostPhotons[i];
        if (ph.power.x == 0.f && ph.power.y == 0.f && ph.power.z == 0.f) continue;
        uint32_t ix = static_cast<uint32_t>((ph.position.x - minX) / cs);
        uint32_t iy = static_cast<uint32_t>((ph.position.y - minY) / cs);
        uint32_t iz = static_cast<uint32_t>((ph.position.z - minZ) / cs);
        ix = std::min(ix, hashGridDimX - 1u);
        iy = std::min(iy, hashGridDimY - 1u);
        iz = std::min(iz, hashGridDimZ - 1u);
        uint32_t cellIdx = iz * hashGridDimX * hashGridDimY
                         + iy * hashGridDimX + ix;
        assignments.push_back({i, cellIdx});
    }
    std::sort(assignments.begin(), assignments.end(),
              [](const PhotonCell& a, const PhotonCell& b) {
                  return a.cellIdx < b.cellIdx;
              });

    std::vector<uint32_t> cellStart(numCells + 1, 0);
    for (const auto& pc : assignments) cellStart[pc.cellIdx + 1]++;
    for (uint32_t c = 0; c < numCells; ++c) cellStart[c + 1] += cellStart[c];

    // Compact photons in cell-sorted order so each cell's photons are
    // contiguous.  The shade-kernel scan now reads photons[i] directly
    // (no sortedPhotonIdx indirection) and stays inside one cache line
    // for typical small cells.
    std::vector<GpuPhoton> compacted(validCount);
    for (uint32_t i = 0; i < validCount; ++i)
        compacted[i] = hostPhotons[assignments[i].photonIdx];

    // ---- 5. Upload compacted photons + cellStart; drop the trace buffer ---
    d_photons = CudaBuffer<GpuPhoton>(validCount);
    d_photons.upload(compacted);
    d_hashCellStart = CudaBuffer<uint32_t>(cellStart.size());
    d_hashCellStart.upload(cellStart);
    d_sortedPhotonIdx = CudaBuffer<uint32_t>{};   // no longer needed

    printf("[info]  CudaPathIntegrator: photon hash grid %ux%ux%u (%u cells)\n",
           hashGridDimX, hashGridDimY, hashGridDimZ, numCells);
}

// ---------------------------------------------------------------------------
// renderFrameWavefront — experimental loop-in-host path tracer.
//
// Driver:
//   primary launch     — 3D grid (tileW, tileH, batchSize), one thread per
//                        (pixel, sample) initialises WfRayState.
//   bounce launch * N  — 1D grid (tileW * tileH * batchSize), one thread per
//                        ray-slot; runs one trace + shade + sample per call.
//                        Threads whose slot is already terminated return
//                        immediately, so the launch count stays constant
//                        without explicit compaction.  Adding compaction is
//                        a follow-up step if this POC pans out.
//   finalize launch    — same 1D grid, atomic-add each ray's L into the
//                        film accum (with firefly clamp).
// ---------------------------------------------------------------------------
bool CudaPathIntegrator::Impl::renderFrameWavefront(
    const SceneView& scene,
    uint32_t filmWidth, uint32_t filmHeight,
    uint32_t sampleStart, uint32_t sampleCount,
    Film& film)
{
    CUstream stream = static_cast<CUstream>(ctx->cuStream());

    ensureAccum(filmWidth, filmHeight);
    if (!d_accum.isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderFrameWavefront: "
                        "accum alloc failed (%u x %u)\n", filmWidth, filmHeight);
        return false;
    }

    constexpr uint32_t kBatchSize     = 4;
    constexpr uint32_t kMergeInterval = 4;

    // Allocate / resize the persistent ray-state buffer.
    uint32_t needRays = filmWidth * filmHeight * kBatchSize;
    if (!d_wfRays.isValid() || wfRayCount != needRays) {
        d_wfRays   = CudaBuffer<WfRayState>(needRays);
        wfRayCount = d_wfRays.isValid() ? needRays : 0;
        if (!d_wfRays.isValid()) {
            fprintf(stderr, "[error] CudaPathIntegrator: wavefront state "
                            "alloc failed (%u entries, %.2f MiB)\n",
                    needRays,
                    double(needRays * sizeof(WfRayState)) / (1024.0 * 1024.0));
            return false;
        }
    }

    auto flushToFilm = [&]() {
        std::vector<GpuAccumPixel> h_accum;
        d_accum.download(h_accum);
        TileBuffer tb(0, 0, filmWidth, filmHeight);
        for (uint32_t py = 0; py < filmHeight; ++py) {
            for (uint32_t px = 0; px < filmWidth; ++px) {
                const GpuAccumPixel& p = h_accum[py * filmWidth + px];
                float w = p.weight > 0.f ? p.weight : 1.f;
                tb.add(px, py, p.r / w, p.g / w, p.b / w, w);
                tb.addLumSq(px, py, p.sumLumSq);
            }
        }
        film.mergeTile(tb);
    };

    uint32_t dispatches = 0;
    for (uint32_t s = 0; s < sampleCount; s += kBatchSize) {
        uint32_t thisBatch = std::min(kBatchSize, sampleCount - s);

        LaunchParams params{};
        fillLaunchParams(params, scene,
            filmWidth, filmHeight,
            0, 0, filmWidth, filmHeight,
            sampleStart + s, thisBatch,
            d_accum.ptr());
        params.wfRays   = d_wfRays.ptr();
        params.wfNumRays = filmWidth * filmHeight * thisBatch;

        d_launchParams.upload(&params, 1);

        // Primary: 3D (w, h, batch).
        OPTIX_CHECK(optixLaunch(wfPipeline, stream,
                                d_launchParams.devPtr(),
                                sizeof(LaunchParams),
                                &sbtWfPrimary,
                                filmWidth, filmHeight, thisBatch));

        // Bounce: loop in host.  Threads that hit max depth or terminate
        // early early-out at the head of the kernel.
        for (uint32_t b = 0; b <= params.cam.maxDepth; ++b) {
            OPTIX_CHECK(optixLaunch(wfPipeline, stream,
                                    d_launchParams.devPtr(),
                                    sizeof(LaunchParams),
                                    &sbtWfBounce,
                                    params.wfNumRays, 1, 1));
        }

        // Finalize: gather L into the accum buffer.
        OPTIX_CHECK(optixLaunch(wfPipeline, stream,
                                d_launchParams.devPtr(),
                                sizeof(LaunchParams),
                                &sbtWfFinalize,
                                params.wfNumRays, 1, 1));

        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[error] CudaPathIntegrator::renderFrameWavefront: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        if ((++dispatches) % kMergeInterval == 0) flushToFilm();
    }
    flushToFilm();
    return true;
}

// ---------------------------------------------------------------------------
// renderTileWavefront — wavefront equivalent of renderTile.
//
// Called by the adaptive refinement pass for high-variance tiles.  Allocates
// a tile-local accum + ray-state buffer (no shared persistent state), runs
// primary -> bounce ×N -> finalize, then downloads accum into the host
// TileBuffer.  Same dispatch pattern as renderFrameWavefront, just sized
// for the tile instead of the full frame.
// ---------------------------------------------------------------------------
void CudaPathIntegrator::Impl::renderTileWavefront(
    const SceneView& scene,
    uint32_t filmWidth, uint32_t filmHeight,
    uint32_t tileX0, uint32_t tileY0,
    uint32_t tileW,  uint32_t tileH,
    uint32_t sampleStart, uint32_t sampleCount,
    TileBuffer& out)
{
    CUstream stream = static_cast<CUstream>(ctx->cuStream());

    CudaBuffer<GpuAccumPixel> d_tileAccum(tileW * tileH);
    if (!d_tileAccum.isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderTileWavefront: "
                        "tile accum alloc failed (%u x %u)\n", tileW, tileH);
        return;
    }
    d_tileAccum.zero();

    const uint32_t needRays = tileW * tileH * sampleCount;
    CudaBuffer<WfRayState> d_tileRays(needRays);
    if (!d_tileRays.isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderTileWavefront: "
                        "ray-state alloc failed (%u entries, %.2f MiB)\n",
                needRays,
                double(needRays * sizeof(WfRayState)) / (1024.0 * 1024.0));
        return;
    }

    LaunchParams params{};
    fillLaunchParams(params, scene,
        filmWidth, filmHeight,
        tileX0, tileY0, tileW, tileH,
        sampleStart, sampleCount,
        d_tileAccum.ptr());
    params.wfRays    = d_tileRays.ptr();
    params.wfNumRays = needRays;

    d_launchParams.upload(&params, 1);

    OPTIX_CHECK(optixLaunch(wfPipeline, stream,
                            d_launchParams.devPtr(),
                            sizeof(LaunchParams),
                            &sbtWfPrimary,
                            tileW, tileH, sampleCount));
    for (uint32_t b = 0; b <= params.cam.maxDepth; ++b) {
        OPTIX_CHECK(optixLaunch(wfPipeline, stream,
                                d_launchParams.devPtr(),
                                sizeof(LaunchParams),
                                &sbtWfBounce,
                                params.wfNumRays, 1, 1));
    }
    OPTIX_CHECK(optixLaunch(wfPipeline, stream,
                            d_launchParams.devPtr(),
                            sizeof(LaunchParams),
                            &sbtWfFinalize,
                            params.wfNumRays, 1, 1));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderTileWavefront: %s\n",
                cudaGetErrorString(err));
        return;
    }

    std::vector<GpuAccumPixel> h_accum;
    d_tileAccum.download(h_accum);
    for (uint32_t ty = 0; ty < tileH; ++ty) {
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            const GpuAccumPixel& p = h_accum[ty * tileW + tx];
            float w = p.weight > 0.f ? p.weight : 1.f;
            out.add(tx, ty, p.r / w, p.g / w, p.b / w, w);
            out.addLumSq(tx, ty, p.sumLumSq);
        }
    }
}

#endif  // ANACAPA_ENABLE_OPTIX

// ---------------------------------------------------------------------------
// prepare() — build accel, upload materials/lights/HDRI
// ---------------------------------------------------------------------------
void CudaPathIntegrator::prepare(const SceneView& scene) {
    if (!isValid() || !scene.accel) return;

    m_impl->accel = std::make_unique<CudaAccelStructure>(
        *m_impl->ctx, scene.accel->pool(), scene.curvePool,
        scene.hairTessSteps);
    if (!m_impl->accel->isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::prepare - accel build failed\n");
        return;
    }
    m_impl->hairMeshBaseID = m_impl->accel->hairMeshBaseID();

    // Materials
    uint32_t nMat = static_cast<uint32_t>(scene.materials.size());
    std::vector<GpuMaterial> gpuMats(std::max(nMat, 1u));
    for (uint32_t i = 0; i < nMat; ++i)
        gpuMats[i] = extractGpuMaterial(scene.materials[i]);
    m_impl->d_materials  = CudaBuffer<GpuMaterial>(gpuMats.size());
    m_impl->d_materials.upload(gpuMats);
    m_impl->numMaterials = nMat;

    // Hair materials — one slot per scene material, even on non-hair slots
    // (the raygen indexes by the strand's material index, not by GpuMaterial type).
    {
        size_t nSlots = std::max(scene.materials.size(), size_t(1));
        std::vector<GpuHairMaterial> hairMats(nSlots);
        for (size_t i = 0; i < scene.materials.size(); ++i)
            hairMats[i] = extractGpuHairMaterial(scene.materials[i]);
        m_impl->d_hairMats = CudaBuffer<GpuHairMaterial>(nSlots);
        m_impl->d_hairMats.upload(hairMats);
    }

    // Lights
    std::vector<GpuLight> gpuLights;
    for (const ILight* l : scene.lights)
        if (l) gpuLights.push_back(extractGpuLight(l));
    if (gpuLights.empty()) gpuLights.push_back({});
    m_impl->d_lights  = CudaBuffer<GpuLight>(gpuLights.size());
    m_impl->d_lights.upload(gpuLights);
    m_impl->numLights = static_cast<uint32_t>(scene.lights.size());

    // GGX energy-compensation LUTs — uploaded once.  The accessors return
    // pointers into a function-local static, so re-uploading is harmless.
    if (!m_impl->d_specAlbedoLUT.isValid()) {
        const int N_COS = specAlbedoLUTCosBins();
        const int N_R   = specAlbedoLUTRoughnessBins();
        m_impl->d_specAlbedoLUT = CudaBuffer<float>(size_t(N_COS) * N_R);
        m_impl->d_specAlbedoLUT.upload(specAlbedoLUTData(), size_t(N_COS) * N_R);
        m_impl->d_specAvgAlbedoLUT = CudaBuffer<float>(N_R);
        m_impl->d_specAvgAlbedoLUT.upload(specAvgAlbedoLUTData(), N_R);
        m_impl->specLUTCosBins   = static_cast<uint32_t>(N_COS);
        m_impl->specLUTRoughBins = static_cast<uint32_t>(N_R);
    }

    // HDRI texture
    if (m_impl->envTex)   { cudaDestroyTextureObject(m_impl->envTex); m_impl->envTex = 0; }
    if (m_impl->envArray) { cudaFreeArray(m_impl->envArray); m_impl->envArray = nullptr; }
    m_impl->d_envMarginalCdf    = CudaBuffer<float>{};
    m_impl->d_envConditionalCdf = CudaBuffer<float>{};
    m_impl->envCdfWidth  = 0;
    m_impl->envCdfHeight = 0;

    const DomeLight* dome = nullptr;
    for (const ILight* l : scene.lights)
        if ((dome = dynamic_cast<const DomeLight*>(l))) break;

    if (dome && dome->envWidth() > 0) {
        uint32_t ew = dome->envWidth(), eh = dome->envHeight();
        const float* rgb = dome->pixels();
        std::vector<float> rgba(size_t(ew) * eh * 4);
        for (uint32_t i = 0; i < ew * eh; ++i) {
            rgba[i*4+0] = rgb[i*3+0];
            rgba[i*4+1] = rgb[i*3+1];
            rgba[i*4+2] = rgb[i*3+2];
            rgba[i*4+3] = 1.f;
        }
        cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaMallocArray(&m_impl->envArray, &fmt, ew, eh));
        CUDA_CHECK(cudaMemcpy2DToArray(m_impl->envArray, 0, 0,
                                        rgba.data(), ew * 4 * sizeof(float),
                                        ew * 4 * sizeof(float), eh,
                                        cudaMemcpyHostToDevice));
        cudaResourceDesc resDesc{};
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = m_impl->envArray;
        cudaTextureDesc texDesc{};
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;
        CUDA_CHECK(cudaCreateTextureObject(&m_impl->envTex, &resDesc, &texDesc, nullptr));

        Vec3f r0, r1, r2;
        dome->getRotation(r0, r1, r2);
        m_impl->envRot[0]    = r0;
        m_impl->envRot[1]    = r1;
        m_impl->envRot[2]    = r2;
        m_impl->envIntensity = dome->intensity();

        // HDRI importance sampling — upload the marginal + conditional CDF
        // tables.  Sizes: marginal=(H+1), conditional=H*(W+1).
        const auto& margCdf = dome->marginalCdf();
        const auto  condCdf = dome->flatConditionalCdf();
        m_impl->d_envMarginalCdf = CudaBuffer<float>(margCdf.size());
        m_impl->d_envMarginalCdf.upload(margCdf);
        m_impl->d_envConditionalCdf = CudaBuffer<float>(condCdf.size());
        m_impl->d_envConditionalCdf.upload(condCdf);
        m_impl->envCdfWidth  = static_cast<uint32_t>(ew);
        m_impl->envCdfHeight = static_cast<uint32_t>(eh);

        printf("[info]  CudaPathIntegrator: uploaded %dx%d HDRI env texture + CDF tables\n", ew, eh);
    }

    printf("[info]  CudaPathIntegrator::prepare - %u materials, %u lights, %zu verts, %zu tris\n",
           m_impl->numMaterials, m_impl->numLights,
           m_impl->accel->totalVertices(),
           m_impl->accel->totalTriangles());

#ifdef ANACAPA_ENABLE_OPTIX
    // GPU caustic photon map.  Requires the OptiX pipeline (specifically the
    // __raygen__photon entry) to be built before launch.  buildPhotonMap is
    // a no-op when photonMapEnabled is false or numPhotons == 0.
    if (m_impl->photonMapEnabled) {
        if (m_impl->buildOptixPipeline(
                static_cast<OptixDeviceContext>(m_impl->ctx->optixContext()))) {
            m_impl->buildPhotonMap(
                static_cast<OptixDeviceContext>(m_impl->ctx->optixContext()));
        }
    }
#endif

    m_impl->preparedOnce = true;
}

// ---------------------------------------------------------------------------
// fillLaunchParams — shared setup for renderFrame and renderTile
// ---------------------------------------------------------------------------
void CudaPathIntegrator::Impl::fillLaunchParams(
    LaunchParams& p, const SceneView& scene,
    uint32_t filmWidth, uint32_t filmHeight,
    uint32_t tileX0, uint32_t tileY0, uint32_t tileW, uint32_t tileH,
    uint32_t sampleStart, uint32_t sampleCount,
    GpuAccumPixel* d_accum) const
{
    Camera cam = scene.camera.value_or(Camera::makePinhole(
        {0.f,0.f,-2.5f},{0.f,0.f,1.f},{0.f,1.f,0.f},
        50.f, filmWidth, filmHeight));

    p.cam.origin     = {cam.origin.x,          cam.origin.y,          cam.origin.z};
    p.cam.horizontal = {cam.horizontal.x,       cam.horizontal.y,      cam.horizontal.z};
    p.cam.vertical   = {cam.vertical.x,         cam.vertical.y,        cam.vertical.z};
    p.cam.lowerLeft  = {cam.lowerLeftCorner.x,  cam.lowerLeftCorner.y, cam.lowerLeftCorner.z};
    p.cam.imageWidth  = filmWidth;
    p.cam.imageHeight = filmHeight;
    p.cam.samplesPerPixel = sampleCount;
    p.cam.maxDepth        = maxDepth;
    p.cam.tileX0    = tileX0;
    p.cam.tileY0    = tileY0;
    p.cam.tileWidth  = tileW;
    p.cam.tileHeight = tileH;
    p.cam.hasEnvLight  = scene.envLight ? 1u : 0u;
    p.cam.envIntensity = envIntensity;
    p.cam.envRot0 = {envRot[0].x, envRot[0].y, envRot[0].z};
    p.cam.envRot1 = {envRot[1].x, envRot[1].y, envRot[1].z};
    p.cam.envRot2 = {envRot[2].x, envRot[2].y, envRot[2].z};
    // Shutter range — accel was already built motion-aware iff any mesh has
    // motion keys, so we always pass [0, 1] when motion is enabled.  When the
    // GAS is static both values are 0 and rays sample a fixed time slice.
    p.cam.shutterOpen  = accel->hasMotion() ? 0.0f : 0.0f;
    p.cam.shutterClose = accel->hasMotion() ? 1.0f : 0.0f;
    if (scene.envLight) {
        static const Vec3f kDirs[] = {{0,1,0},{0.577f,0.577f,0.577f},{-0.577f,0.577f,0.577f},
                                      {0.577f,0.577f,-0.577f},{-0.577f,0.577f,-0.577f}};
        Spectrum avg{};
        for (const Vec3f& d : kDirs) avg += scene.envLight->Le({},{},d);
        avg = avg * (1.f/5.f);
        p.cam.envLe = {avg.x, avg.y, avg.z};
    }
    p.accum             = d_accum;
    p.lights            = d_lights.ptr();
    p.numLights         = numLights;
    p.materials         = d_materials.ptr();
    p.numMaterials      = numMaterials;
    p.normals           = reinterpret_cast<const GpuFloat3*>(accel->normalBuffer());
    p.indices           = reinterpret_cast<const uint32_t*>(accel->indexBuffer());
    p.triMeshIDs        = reinterpret_cast<const uint32_t*>(accel->triMeshIDBuffer());
    p.meshVertexOffsets = reinterpret_cast<const uint32_t*>(accel->meshVertexOffsetBuffer());
    p.meshIndexOffsets  = reinterpret_cast<const uint32_t*>(accel->meshIndexOffsetBuffer());
    p.sampleBatch.sampleStart = sampleStart;
    p.sampleBatch.batchSize   = sampleCount;
    p.envTexture        = envTex;
    p.envMarginalCdf    = d_envMarginalCdf.isValid()    ? d_envMarginalCdf.ptr()    : nullptr;
    p.envConditionalCdf = d_envConditionalCdf.isValid() ? d_envConditionalCdf.ptr() : nullptr;
    p.cam.envMapWidth   = envCdfWidth;
    p.cam.envMapHeight  = envCdfHeight;
    p.cam.fireflyClamp  = fireflyClamp;
    p.cam.hairMeshBaseID = hairMeshBaseID;
    p.specAlbedoLUT     = d_specAlbedoLUT.isValid()    ? d_specAlbedoLUT.ptr()    : nullptr;
    p.specAvgAlbedoLUT  = d_specAvgAlbedoLUT.isValid() ? d_specAvgAlbedoLUT.ptr() : nullptr;
    p.specLUTCosBins    = specLUTCosBins;
    p.specLUTRoughBins  = specLUTRoughBins;
    p.pixelFilterCdf    = d_pixelFilterCdf.isValid()   ? d_pixelFilterCdf.ptr()   : nullptr;
    p.pixelFilterSigns  = d_pixelFilterSigns.isValid() ? d_pixelFilterSigns.ptr() : nullptr;
    p.pixelFilterBins   = pixelFilterBins;
    p.pixelFilterRadius = pixelFilterRadius;
    p.hairTris          = reinterpret_cast<const GpuHairTri*>(accel->hairTriBuffer());
    p.hairMats          = d_hairMats.isValid() ? d_hairMats.ptr() : nullptr;
    // Caustic photon map — gated on having actual valid photons so a
    // configured-but-empty map (e.g. no glass in the scene) doesn't query.
    // d_photons holds the cell-sorted compacted photons after buildPhotonMap;
    // d_sortedPhotonIdx is intentionally dropped post-compaction so we don't
    // require it in the gate.
    const bool pmActive = photonMapEnabled && validPhotons > 0
                           && d_hashCellStart.isValid()
                           && d_photons.isValid();
    p.cam.photonMapEnabled   = pmActive ? 1u : 0u;
    p.cam.photonSearchRadius = photonSearchRadius;
    p.cam.hashGridOrigin     = hashGridOrigin;
    p.cam.hashCellSize       = hashCellSize;
    p.cam.hashGridDimX       = hashGridDimX;
    p.cam.hashGridDimY       = hashGridDimY;
    p.cam.hashGridDimZ       = hashGridDimZ;
    p.photons                = d_photons.isValid()       ? d_photons.ptr()       : nullptr;
    p.hashCellStart          = d_hashCellStart.isValid() ? d_hashCellStart.ptr() : nullptr;
    p.sortedPhotonIdx        = nullptr;   // unused post-compaction
    p.numPhotons             = numPhotons;
    p.frameIndex             = photonFrameIndex;
    // Wavefront state — populated explicitly by renderFrameWavefront.  When
    // the megakernel path is in use these stay null/zero and are ignored.
    p.wfRays                 = nullptr;
    p.wfNumRays              = 0;
    p.handle            = accel->traversableHandle();
}

// ---------------------------------------------------------------------------
// renderFrame() — whole-image, single kernel launch
// ---------------------------------------------------------------------------
bool CudaPathIntegrator::renderFrame(const SceneView& scene,
                                      uint32_t filmWidth,
                                      uint32_t filmHeight,
                                      uint32_t sampleStart,
                                      uint32_t sampleCount,
                                      Film& film)
{
    if (!isValid() || !m_impl->preparedOnce) return false;

    cudaStream_t stream = static_cast<cudaStream_t>(m_impl->ctx->cuStream());

#ifdef ANACAPA_ENABLE_OPTIX
    if (!m_impl->buildOptixPipeline(
            static_cast<OptixDeviceContext>(m_impl->ctx->optixContext()))) {
        return false;
    }
    // Route to wavefront experimental path when enabled.  Photon-trace
    // still uses the megakernel pipeline (its raygen is small), but the
    // shade-time photon-map query is already implemented in
    // __raygen__wf_bounce, so photon mode benefits from wavefront too.
    // Hair scenes currently fall back — the Marschner BSDF path hasn't
    // been ported to wavefront yet.
    const bool wfPath = m_impl->wavefrontMode
                       && (m_impl->accel->hairMeshBaseID() == 0xFFFFFFFFu);
    if (wfPath) {
        return m_impl->renderFrameWavefront(scene,
                                              filmWidth, filmHeight,
                                              sampleStart, sampleCount, film);
    }
#else
    fprintf(stderr, "[error] CudaPathIntegrator: built without ANACAPA_ENABLE_OPTIX\n");
    return false;
#endif

    // Use the persistent accum buffer — accumulates across calls.
    // clearAccum() should be called when starting a fresh render (new scene/camera).
    m_impl->ensureAccum(filmWidth, filmHeight);
    CudaBuffer<GpuAccumPixel>& d_accum = m_impl->d_accum;
    // Bail cleanly on VRAM exhaustion so the caller falls back to per-tile
    // dispatch instead of submitting a launch with a null accum pointer.
    if (!d_accum.isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderFrame: "
                        "persistent accum alloc failed (%u x %u) — "
                        "falling back to tile dispatch\n",
                filmWidth, filmHeight);
        return false;
    }

    // Dispatch kBatchSize samples per launch — each thread traces the full
    // batch and accumulates locally before writing to the accum buffer.
    constexpr uint32_t kBatchSize    = 4;
    constexpr uint32_t kMergeInterval = 4;  // flush to film every N dispatches

    auto flushToFilm = [&]() {
        std::vector<GpuAccumPixel> h_accum;
        d_accum.download(h_accum);
        TileBuffer tb(0, 0, filmWidth, filmHeight);
        for (uint32_t py = 0; py < filmHeight; ++py) {
            for (uint32_t px = 0; px < filmWidth; ++px) {
                const GpuAccumPixel& p = h_accum[py * filmWidth + px];
                float w = p.weight > 0.f ? p.weight : 1.f;
                tb.add(px, py, p.r / w, p.g / w, p.b / w, w);
                tb.addLumSq(px, py, p.sumLumSq);
            }
        }
        film.mergeTile(tb);
    };

    uint32_t dispatches = 0;
    for (uint32_t s = 0; s < sampleCount; s += kBatchSize) {
        uint32_t thisBatch = std::min(kBatchSize, sampleCount - s);
        LaunchParams params{};
        m_impl->fillLaunchParams(params, scene,
            filmWidth, filmHeight,
            0, 0, filmWidth, filmHeight,
            sampleStart + s, thisBatch,
            d_accum.ptr());

#ifdef ANACAPA_ENABLE_OPTIX
        m_impl->d_launchParams.upload(&params, 1);
        OPTIX_CHECK(optixLaunch(m_impl->optixPipeline, stream,
                                m_impl->d_launchParams.devPtr(),
                                sizeof(LaunchParams),
                                &m_impl->sbt,
                                filmWidth, filmHeight, /*depth=*/1));
#endif
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[error] CudaPathIntegrator::renderFrame: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        if ((++dispatches) % kMergeInterval == 0)
            flushToFilm();
    }

    // Final flush
    flushToFilm();
    return true;
}

// ---------------------------------------------------------------------------
// renderTile() — used for adaptive passes; one kernel launch per tile
// ---------------------------------------------------------------------------
void CudaPathIntegrator::renderTile(const SceneView& scene,
                                     const TileRequest& tile,
                                     uint32_t filmWidth,
                                     uint32_t filmHeight,
                                     ISampler& /*sampler*/,
                                     TileBuffer& out)
{
    if (!isValid() || !m_impl->preparedOnce) return;

    cudaStream_t stream = static_cast<cudaStream_t>(m_impl->ctx->cuStream());

#ifdef ANACAPA_ENABLE_OPTIX
    if (!m_impl->buildOptixPipeline(
            static_cast<OptixDeviceContext>(m_impl->ctx->optixContext()))) {
        return;
    }
    // Wavefront route — same gating as renderFrame.  Hair scenes still
    // fall through to the megakernel tile dispatch below.
    if (m_impl->wavefrontMode
            && m_impl->accel->hairMeshBaseID() == 0xFFFFFFFFu) {
        uint32_t tw = std::min(tile.width,  filmWidth  - tile.x0);
        uint32_t th = std::min(tile.height, filmHeight - tile.y0);
        m_impl->renderTileWavefront(scene,
                                      filmWidth, filmHeight,
                                      tile.x0, tile.y0, tw, th,
                                      tile.sampleStart, tile.sampleCount,
                                      out);
        return;
    }
#else
    return;
#endif

    uint32_t tileW = std::min(tile.width,  filmWidth  - tile.x0);
    uint32_t tileH = std::min(tile.height, filmHeight - tile.y0);

    CudaBuffer<GpuAccumPixel> d_accum(tileW * tileH);
    // Bail cleanly on VRAM exhaustion.  Without this the launch below
    // would scribble into a null device pointer and the download would
    // produce arbitrary host memory contents — the source of the
    // "tiles in the wrong place" symptom on low-memory GPUs.
    if (!d_accum.isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderTile: "
                        "d_accum alloc failed (%u x %u)\n", tileW, tileH);
        return;
    }
    d_accum.zero();

    LaunchParams params{};
    m_impl->fillLaunchParams(params, scene,
        filmWidth, filmHeight,
        tile.x0, tile.y0, tileW, tileH,
        tile.sampleStart, tile.sampleCount,
        d_accum.ptr());

#ifdef ANACAPA_ENABLE_OPTIX
    m_impl->d_launchParams.upload(&params, 1);
    OPTIX_CHECK(optixLaunch(m_impl->optixPipeline, stream,
                            m_impl->d_launchParams.devPtr(),
                            sizeof(LaunchParams),
                            &m_impl->sbt,
                            tileW, tileH, /*depth=*/1));
#endif
    CUDA_CHECK(cudaStreamSynchronize(stream));
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "[error] CudaPathIntegrator::renderTile: "
                        "launch failed (%u x %u): %s\n",
                tileW, tileH, cudaGetErrorString(err));
        return;
    }

    std::vector<GpuAccumPixel> h_accum;
    d_accum.download(h_accum);

    for (uint32_t ty = 0; ty < tileH; ++ty) {
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            const GpuAccumPixel& p = h_accum[ty * tileW + tx];
            float w = p.weight > 0.f ? p.weight : 1.f;
            out.add(tx, ty, p.r / w, p.g / w, p.b / w, w);
            out.addLumSq(tx, ty, p.sumLumSq);
        }
    }
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_CUDA
