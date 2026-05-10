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
    gm.type        = kMatLambertian;
    if (!mat) return gm;

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
    OptixPipeline         optixPipeline  = nullptr;
    CudaByteBuffer        sbtRaygenBuf;
    CudaByteBuffer        sbtMissBuf;
    CudaByteBuffer        sbtHitBuf;
    OptixShaderBindingTable sbt          = {};
    CudaBuffer<LaunchParams> d_launchParams;  // single-element buffer in device mem
    bool                  optixReady     = false;

    bool buildOptixPipeline(OptixDeviceContext ctx);
    void destroyOptixPipeline();
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
    pipeOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues                 = 5;
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

    OptixProgramGroup pgs[3] = { pgRaygen, pgMiss, pgHit };
    logSize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(ctx, &pipeOpts, &linkOpts,
                                    pgs, 3, log, &logSize, &optixPipeline));

    // ---- Stack sizes --------------------------------------------------------
    OptixStackSizes stackSizes{};
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgRaygen, &stackSizes, optixPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgMiss,   &stackSizes, optixPipeline));
    OPTIX_CHECK(optixUtilAccumulateStackSizes(pgHit,    &stackSizes, optixPipeline));

    uint32_t dcStackTrav = 0, dcStackState = 0, contStack = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&stackSizes,
                                           /*maxTraceDepth=*/1,
                                           /*maxCCDepth=*/0,
                                           /*maxDCDepth=*/0,
                                           &dcStackTrav, &dcStackState, &contStack));
    OPTIX_CHECK(optixPipelineSetStackSize(optixPipeline,
                                          dcStackTrav, dcStackState, contStack,
                                          /*maxTraversableDepth=*/1));

    // ---- Shader binding table -----------------------------------------------
    SbtRecord raygenRec{}, missRec{}, hitRec{};
    OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen, &raygenRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss,   &missRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgHit,    &hitRec));

    sbtRaygenBuf = CudaByteBuffer(sizeof(SbtRecord));
    sbtRaygenBuf.upload(reinterpret_cast<const uint8_t*>(&raygenRec), sizeof(SbtRecord));

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

    // ---- Launch params buffer (re-used across launches) ---------------------
    d_launchParams = CudaBuffer<LaunchParams>(1);

    optixReady = true;
    printf("[info]  CudaPathIntegrator: OptiX pipeline ready\n");
    return true;
}

void CudaPathIntegrator::Impl::destroyOptixPipeline()
{
    if (!optixReady) return;
    if (optixPipeline) optixPipelineDestroy(optixPipeline);
    if (pgHit)         optixProgramGroupDestroy(pgHit);
    if (pgMiss)        optixProgramGroupDestroy(pgMiss);
    if (pgRaygen)      optixProgramGroupDestroy(pgRaygen);
    if (optixModule)   optixModuleDestroy(optixModule);
    optixPipeline = nullptr;
    pgHit = pgMiss = pgRaygen = nullptr;
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

// ---------------------------------------------------------------------------
// prepare() — build accel, upload materials/lights/HDRI
// ---------------------------------------------------------------------------
void CudaPathIntegrator::prepare(const SceneView& scene) {
    if (!isValid() || !scene.accel) return;

    m_impl->accel = std::make_unique<CudaAccelStructure>(
        *m_impl->ctx, scene.accel->pool());
    if (!m_impl->accel->isValid()) {
        fprintf(stderr, "[error] CudaPathIntegrator::prepare - accel build failed\n");
        return;
    }

    // Materials
    uint32_t nMat = static_cast<uint32_t>(scene.materials.size());
    std::vector<GpuMaterial> gpuMats(std::max(nMat, 1u));
    for (uint32_t i = 0; i < nMat; ++i)
        gpuMats[i] = extractGpuMaterial(scene.materials[i]);
    m_impl->d_materials  = CudaBuffer<GpuMaterial>(gpuMats.size());
    m_impl->d_materials.upload(gpuMats);
    m_impl->numMaterials = nMat;

    // Lights
    std::vector<GpuLight> gpuLights;
    for (const ILight* l : scene.lights)
        if (l) gpuLights.push_back(extractGpuLight(l));
    if (gpuLights.empty()) gpuLights.push_back({});
    m_impl->d_lights  = CudaBuffer<GpuLight>(gpuLights.size());
    m_impl->d_lights.upload(gpuLights);
    m_impl->numLights = static_cast<uint32_t>(scene.lights.size());

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
#else
    fprintf(stderr, "[error] CudaPathIntegrator: built without ANACAPA_ENABLE_OPTIX\n");
    return false;
#endif

    // Use the persistent accum buffer — accumulates across calls.
    // clearAccum() should be called when starting a fresh render (new scene/camera).
    m_impl->ensureAccum(filmWidth, filmHeight);
    CudaBuffer<GpuAccumPixel>& d_accum = m_impl->d_accum;

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
#else
    return;
#endif

    uint32_t tileW = std::min(tile.width,  filmWidth  - tile.x0);
    uint32_t tileH = std::min(tile.height, filmHeight - tile.y0);

    CudaBuffer<GpuAccumPixel> d_accum(tileW * tileH);
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
