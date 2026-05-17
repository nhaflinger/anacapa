#ifdef ANACAPA_ENABLE_METAL

#include "MetalPathIntegrator.h"
#include "MetalContext.h"
#include "MetalBuffer.h"
#include "MetalAccelStructure.h"
#include "shaders/SharedTypes.h"

#include <anacapa/integrator/IIntegrator.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/film/Film.h>
#include <anacapa/accel/GeometryPool.h>

// Material type-detection includes — used in prepare() only
#include "../../shading/Lambertian.h"
#include "../../shading/StandardSurface.h"
#include "../../shading/MarschnerHair.h"
#include "../../shading/ChiangHair.h"
#include "../../shading/lights/AreaLight.h"
#include "../../shading/lights/DirectionalLight.h"
#include "../../shading/lights/DomeLight.h"

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <spdlog/spdlog.h>
#include <vector>
#include <cstring>

namespace anacapa {

// ---------------------------------------------------------------------------
// PIMPL
// ---------------------------------------------------------------------------
struct MetalPathIntegrator::Impl {
    std::unique_ptr<MetalContext>         ctx;
    std::unique_ptr<MetalAccelStructure>  accel;

    id<MTLComputePipelineState> psoShade  = nil;

    // GPU-side scene data (uploaded once in prepare())
    std::unique_ptr<MetalBuffer<GpuMaterial>> matBuf;
    std::unique_ptr<MetalBuffer<GpuLight>>    lightBuf;

    // HDRI environment texture (RGBA32Float, or 1x1 white fallback)
    id<MTLTexture> envTexture     = nil;
    id<MTLTexture> fallbackEnvTex = nil;
    Vec3f          envRot[3]      = {{1,0,0},{0,1,0},{0,0,1}};
    float          envIntensity   = 1.0f;

    // HDRI importance sampling CDF buffers (uploaded once per scene load)
    id<MTLBuffer> envMarginalCdfBuf    = nil;  // (H+1) floats
    id<MTLBuffer> envConditionalCdfBuf = nil;  // H*(W+1) floats
    uint32_t      envCdfWidth          = 0;
    uint32_t      envCdfHeight         = 0;

    // GGX energy-compensation LUTs (uploaded once, reused across launches)
    id<MTLBuffer> specAlbedoLUTBuf    = nil;  // N_COS * N_R floats
    id<MTLBuffer> specAvgAlbedoLUTBuf = nil;  // N_R floats
    uint32_t      specLUTCosBins      = 0;
    uint32_t      specLUTRoughBins    = 0;

    // Pixel reconstruction filter — host pointer + uploaded CDF/sign tables.
    // pixelFilterBins == 0 disables and the kernel falls back to box-1.0.
    const PixelFilter* pixelFilter        = nullptr;
    id<MTLBuffer>      pixelFilterCdfBuf  = nil;
    id<MTLBuffer>      pixelFilterSignBuf = nil;
    uint32_t           pixelFilterBins    = 0;
    float              pixelFilterRadius  = 0.f;

    uint32_t numMaterials = 0;
    uint32_t numLights    = 0;
    uint32_t maxDepth     = 6;
    float    fireflyClamp = 10.f;

    // Persistent full-frame accumulation buffer.
    // Allocated once and reused across renderFrame() calls so samples accumulate
    // without re-allocation.  clearAccum() zeros it when the scene/camera changes.
    id<MTLBuffer> accumMTL    = nil;
    uint32_t      accumWidth  = 0;
    uint32_t      accumHeight = 0;

    // Hair material parameter buffer — one GpuHairMaterial per scene material slot.
    // Built in prepare() alongside hairTriBuf (which lives inside MetalAccelStructure).
    id<MTLBuffer> hairMatBuf      = nil;
    uint32_t      numHairMats     = 0;
    uint32_t      hairMeshBaseID  = 0xFFFFFFFFu;

    // Pointer to scene geometry (for building the accel structure)
    const GeometryPool* geomPool = nullptr;
    bool preparedOnce = false;

    // Caustic photon map — GPU trace + CPU hash grid build + GPU query
    id<MTLComputePipelineState> psoPhotonTrace = nil;
    id<MTLBuffer>  photonBuf          = nil;  // GpuPhoton[numPhotons] — trace output
    id<MTLBuffer>  hashCellStartBuf   = nil;  // uint32_t[numCells+1]
    id<MTLBuffer>  sortedPhotonIdxBuf = nil;  // uint32_t[validPhotons]
    uint32_t  numPhotons          = 0;
    float     photonSearchRadius  = 0.1f;
    bool      photonMapEnabled    = false;
    GpuFloat3 hashGridOrigin      = {0.f, 0.f, 0.f};
    float     hashCellSize        = 0.1f;
    uint32_t  hashGridDimX        = 0;
    uint32_t  hashGridDimY        = 0;
    uint32_t  hashGridDimZ        = 0;
    uint32_t  validPhotons        = 0;   // photons actually stored after trace
    uint32_t  frameIndex          = 0;

    // SSS photon map — built alongside the caustic map when SSS materials are present.
    id<MTLBuffer>  sssPhotonBuf          = nil;  // GpuPhoton[numPhotons] — SSS trace output
    id<MTLBuffer>  sssHashCellStartBuf   = nil;  // uint32_t[numCells+1]
    id<MTLBuffer>  sssSortedPhotonIdxBuf = nil;  // uint32_t[sssValidPhotons]
    bool      sssMapEnabled       = false;
    float     sssSearchRadius     = 0.1f;
    float     sssD_max            = 0.f;   // max(radius*scale) across all SSS materials
    GpuFloat3 sssHashGridOrigin   = {0.f, 0.f, 0.f};
    float     sssHashCellSize     = 0.1f;
    uint32_t  sssHashGridDimX     = 0;
    uint32_t  sssHashGridDimY     = 0;
    uint32_t  sssHashGridDimZ     = 0;
    uint32_t  sssValidPhotons     = 0;

    void buildPhotonMap(id<MTLDevice> dev, id<MTLCommandQueue> cq,
                        id<MTLAccelerationStructure> tlas,
                        id<MTLBuffer> lightBufMTL, uint32_t nLights,
                        id<MTLBuffer> matBufMTL,   uint32_t nMats,
                        id<MTLBuffer> normalBuf, id<MTLBuffer> indexBuf,
                        id<MTLBuffer> meshIdxOffBuf);

    // Ensure the persistent accum buffer is allocated and sized for (w x h).
    // Zeros the buffer if it had to be (re)allocated.
    void ensureAccum(id<MTLDevice> dev, uint32_t w, uint32_t h) {
        if (accumMTL && accumWidth == w && accumHeight == h) return;
        size_t bytes = (size_t)w * h * sizeof(GpuAccumPixel);
        accumMTL    = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        accumWidth  = w;
        accumHeight = h;
        memset([accumMTL contents], 0, bytes);
    }

    void clearAccum() {
        if (accumMTL)
            memset([accumMTL contents], 0, (size_t)accumWidth * accumHeight * sizeof(GpuAccumPixel));
    }
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static id<MTLComputePipelineState> makePSO(id<MTLDevice> device,
                                            id<MTLLibrary> library,
                                            const char* fnName) {
    NSString*       name = [NSString stringWithUTF8String:fnName];
    id<MTLFunction> fn   = [library newFunctionWithName:name];
    if (!fn) {
        spdlog::error("MetalPathIntegrator: kernel '{}' not in metallib", fnName);
        return nil;
    }
    NSError* err = nil;
    id<MTLComputePipelineState> pso =
        [device newComputePipelineStateWithFunction:fn error:&err];
    if (!pso)
        spdlog::error("MetalPathIntegrator: PSO '{}' failed: {}", fnName,
                      err ? [[err localizedDescription] UTF8String] : "?");
    return pso;
}

// Attempt to read base_color / emission from a CPU IMaterial*.
// Returns defaults for unknown types.
// Populate SSS fields in a GpuMaterial from the CPU IMaterial's subsurfaceParams().
// Called as a post-processing step after all type-detection branches.
static void fillSSSFields(GpuMaterial& gm, const IMaterial* mat) {
    if (!mat) return;
    auto sss = mat->subsurfaceParams();
    if (sss.weight <= 0.f) return;
    gm.isSubsurface      = 1u;
    gm.subsurfaceWeight  = sss.weight;
    gm.subsurfaceColor   = {sss.color.x, sss.color.y, sss.color.z};
    gm.subsurfaceRadius  = sss.radius * sss.scale;
}

static GpuMaterial extractGpuMaterial(const IMaterial* mat) {
    GpuMaterial gm{};
    gm.baseColor  = {0.5f, 0.5f, 0.5f};
    gm.emissive   = {0.f,  0.f,  0.f};
    gm.roughness  = 1.f;
    gm.metalness  = 0.f;
    gm.specularIOR = 1.5f;
    gm.specular    = 0.5f;   // matches MaterialX standard_surface default
    gm.type = kMatLambertian;
    gm.causticGenerator = 0u;

    if (!mat) return gm;
    gm.causticGenerator = mat->isCausticGenerator() ? 1u : 0u;

    // Emissive — covers EmissiveMaterial, StandardSurfaceMaterial with emission > 0,
    // OslMaterial with emission, etc.  Use a dummy front-face ShadingContext so Le
    // is evaluated from the correct (front) side of the surface.
    {
        SurfaceInteraction si;
        si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,-1});  // rayDir opposite ng → frontFace = true
        Spectrum Le = mat->Le(ctx, {0,0,1});
        if (Le.x > 0.f || Le.y > 0.f || Le.z > 0.f) {
            gm.type = kMatEmissive;
            gm.emissive = {Le.x, Le.y, Le.z};
            Spectrum alb = mat->reflectance(ctx);
            gm.baseColor = {alb.x, alb.y, alb.z};
            return gm;
        }
    }

    // Glass detection — works for StandardSurfaceMaterial and OslMaterial.
    // transmittanceColor() returns black for opaque materials; non-black means glass.
    {
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum tint = mat->transmittanceColor(ctx);
        // Use a higher threshold for OslMaterial to avoid misclassifying materials
        // with tiny incidental refraction lobes (e.g. eyes) as glass.
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

    // StandardSurfaceMaterial — promote to kMatGGX (including diffuse-flagged
    // ones like Cornell walls) so the layered eval picks up Disney retro-reflection,
    // spec/diff balance, and Kulla-Conty multi-scatter compensation on the GPU.
    {
        const StandardSurfaceMaterial* ssm = dynamic_cast<const StandardSurfaceMaterial*>(mat);
        if (ssm) {
            gm.type = kMatGGX;
            SurfaceInteraction si; si.n = si.ng = {0,0,1};
            ShadingContext ctx(si, {0,0,1});
            Spectrum alb = mat->reflectance(ctx);
            gm.baseColor   = {alb.x, alb.y, alb.z};
            gm.roughness   = ssm->params().roughness.value;
            gm.metalness   = ssm->params().metalness.value;
            gm.specular    = ssm->params().specular.value;
            gm.specularIOR = ssm->params().specular_IOR;
            fillSSSFields(gm, mat);
            return gm;
        }
    }

    // Non-StandardSurface glossy (OslMaterial, etc.) — GGX by flags
    if (mat->flags() & BSDFFlag_Glossy) {
        gm.type = kMatGGX;
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
        gm.roughness = mat->roughness();
        gm.metalness = mat->metalness();
        fillSSSFields(gm, mat);
        return gm;
    }

    // LambertianMaterial (and default)
    {
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
    }
    fillSSSFields(gm, mat);
    return gm;
}

static GpuLight extractGpuLight(const ILight* light) {
    GpuLight gl{};
    if (!light) return gl;

    const AreaLight* al = dynamic_cast<const AreaLight*>(light);
    if (al) {
        gl.type = kLightRect;
        // Use sampleLe for all geometry — it always returns Le regardless of orientation,
        // unlike sample() which returns {} when the query point is behind the light.
        LightLeSample le   = al->sampleLe({0.5f, 0.5f}, {0.5f, 0.5f});
        LightLeSample le0  = al->sampleLe({0.f,  0.5f}, {0.5f, 0.5f});
        LightLeSample le1  = al->sampleLe({1.f,  0.5f}, {0.5f, 0.5f});
        LightLeSample le2  = al->sampleLe({0.5f, 0.f},  {0.5f, 0.5f});
        LightLeSample le3  = al->sampleLe({0.5f, 1.f},  {0.5f, 0.5f});
        gl.Le       = {le.Le.x,     le.Le.y,     le.Le.z};
        gl.position = {le.pos.x,    le.pos.y,    le.pos.z};
        gl.normal   = {le.normal.x, le.normal.y, le.normal.z};
        gl.area     = 1.0f / le.pdfPos;
        Vec3f uFull = {le1.pos.x - le0.pos.x, le1.pos.y - le0.pos.y, le1.pos.z - le0.pos.z};
        Vec3f vFull = {le3.pos.x - le2.pos.x, le3.pos.y - le2.pos.y, le3.pos.z - le2.pos.z};
        gl.uHalf = {uFull.x * 0.5f, uFull.y * 0.5f, uFull.z * 0.5f};
        gl.vHalf = {vFull.x * 0.5f, vFull.y * 0.5f, vFull.z * 0.5f};
        return gl;
    }

    const DirectionalLight* dl = dynamic_cast<const DirectionalLight*>(light);
    if (dl) {
        gl.type = kLightDirectional;
        LightSample ls = dl->sample({0,0,0}, {0,1,0}, {0.5f, 0.5f});
        gl.Le     = {ls.Li.x, ls.Li.y, ls.Li.z};
        gl.normal = {ls.wi.x,  ls.wi.y,  ls.wi.z};  // dirToLight
        // Derive cosCone by sampling two directions at extremes of the u range.
        // If the light has an angular extent, the directions will differ; the dot
        // product of the two gives cos(halfAngle) of the cone.
        LightLeSample le0 = dl->sampleLe({0.5f, 0.5f}, {0.f, 0.f});
        LightLeSample le1 = dl->sampleLe({0.5f, 0.5f}, {1.f, 0.f});
        Vec3f d0 = le0.dir, d1 = le1.dir;
        float cc = d0.x*d1.x + d0.y*d1.y + d0.z*d1.z;
        gl.cosCone = std::max(0.f, std::min(1.f, cc));
        return gl;
    }

    // DomeLight (infinite environment light)
    const DomeLight* dome = dynamic_cast<const DomeLight*>(light);
    if (dome) {
        gl.type = kLightDome;
        // Approximate average Le by sampling cardinal and diagonal directions
        static const Vec3f kSampleDirs[] = {
            {0,1,0},  {0,-1,0}, {1,0,0}, {-1,0,0}, {0,0,1}, {0,0,-1},
            { 0.577f, 0.577f,  0.577f}, {-0.577f, 0.577f,  0.577f},
            { 0.577f, 0.577f, -0.577f}, {-0.577f, 0.577f, -0.577f},
            { 0.577f,-0.577f,  0.577f}, {-0.577f,-0.577f,  0.577f},
            { 0.577f,-0.577f, -0.577f}, {-0.577f,-0.577f, -0.577f},
        };
        Spectrum avg{};
        for (const Vec3f& d : kSampleDirs)
            avg += dome->Le({}, {}, d);
        avg = avg * (1.f / 14.f);
        gl.Le = {avg.x, avg.y, avg.z};
        return gl;
    }

    // Fallback: unknown light type — skip
    gl.type = kLightRect;
    gl.Le   = {0.f, 0.f, 0.f};
    return gl;
}

static GpuHairMaterial extractGpuHairMaterial(const IMaterial* mat) {
    GpuHairMaterial hm{};
    // Defaults matching MarschnerHairMaterial::Params
    hm.sigma_a = {0.06f, 0.10f, 0.20f};
    hm.eta     = 1.55f;
    hm.beta_m  = 0.40f;
    hm.beta_n  = 0.60f;
    hm.alpha   = 2.0f;
    hm._pad    = 0.f;

    if (!mat) return hm;

    // ChiangHairMaterial uses Params = MarschnerHairMaterial::Params
    const ChiangHairMaterial* ch = dynamic_cast<const ChiangHairMaterial*>(mat);
    if (ch) {
        const auto& p = ch->params();
        hm.sigma_a = {p.sigma_a.x, p.sigma_a.y, p.sigma_a.z};
        hm.eta     = p.eta;
        hm.beta_m  = p.beta_m;
        hm.beta_n  = p.beta_n;
        hm.alpha   = p.alpha;
        return hm;
    }
    const MarschnerHairMaterial* mh = dynamic_cast<const MarschnerHairMaterial*>(mat);
    if (mh) {
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

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
MetalPathIntegrator::MetalPathIntegrator(const std::string& metallibPath)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->ctx = MetalContext::create(metallibPath);
    if (!m_impl->ctx || !m_impl->ctx->isValid()) {
        spdlog::error("MetalPathIntegrator: context init failed");
        return;
    }

    id<MTLDevice>  dev = (__bridge id<MTLDevice>) m_impl->ctx->device();
    id<MTLLibrary> lib = (__bridge id<MTLLibrary>)m_impl->ctx->library();

    m_impl->psoShade      = makePSO(dev, lib, "shade");
    m_impl->psoPhotonTrace = makePSO(dev, lib, "photonTrace");

    if (m_impl->psoShade)
        spdlog::info("MetalPathIntegrator: ready on '{}'", m_impl->ctx->name());
}

MetalPathIntegrator::~MetalPathIntegrator() = default;

bool MetalPathIntegrator::isValid() const {
    return m_impl->ctx && m_impl->ctx->isValid() && m_impl->psoShade != nil;
}

void MetalPathIntegrator::clearAccum() {
    m_impl->clearAccum();
}

void MetalPathIntegrator::setFireflyClamp(float v) {
    m_impl->fireflyClamp = v;
}

void MetalPathIntegrator::setPixelFilter(const PixelFilter* f) {
    m_impl->pixelFilter = f;
    if (!f) {
        m_impl->pixelFilterCdfBuf  = nil;
        m_impl->pixelFilterSignBuf = nil;
        m_impl->pixelFilterBins    = 0;
        m_impl->pixelFilterRadius  = 0.f;
        return;
    }
    id<MTLDevice> dev = static_cast<id<MTLDevice>>(m_impl->ctx->device());
    const auto& cdf   = f->cdf();
    const auto& signs = f->signs();
    m_impl->pixelFilterCdfBuf = [dev newBufferWithBytes:cdf.data()
                                                 length:cdf.size() * sizeof(float)
                                                options:MTLResourceStorageModeShared];
    m_impl->pixelFilterSignBuf = [dev newBufferWithBytes:signs.data()
                                                  length:signs.size() * sizeof(float)
                                                 options:MTLResourceStorageModeShared];
    m_impl->pixelFilterBins   = static_cast<uint32_t>(signs.size());
    m_impl->pixelFilterRadius = f->radius();
}

void MetalPathIntegrator::setPhotonMap(int numPhotons, float searchRadius) {
    m_impl->photonMapEnabled   = (numPhotons > 0);
    m_impl->numPhotons         = static_cast<uint32_t>(std::max(0, numPhotons));
    m_impl->photonSearchRadius = searchRadius;
}

// ---------------------------------------------------------------------------
// buildPhotonMap — GPU trace → CPU hash grid build → GPU upload
// ---------------------------------------------------------------------------
void MetalPathIntegrator::Impl::buildPhotonMap(
    id<MTLDevice>               dev,
    id<MTLCommandQueue>         cq,
    id<MTLAccelerationStructure> tlas,
    id<MTLBuffer>               lightBufMTL, uint32_t nLights,
    id<MTLBuffer>               matBufMTL,   uint32_t nMats,
    id<MTLBuffer>               normalBuf,
    id<MTLBuffer>               indexBuf,
    id<MTLBuffer>               meshIdxOffBuf)
{
    if (!psoPhotonTrace || numPhotons == 0) return;

    spdlog::info("MetalPathIntegrator: tracing {} photons on GPU...", numPhotons);

    // --- 1. Allocate photon output buffers (caustic + SSS) ---
    size_t photonBytes = static_cast<size_t>(numPhotons) * sizeof(GpuPhoton);
    photonBuf = [dev newBufferWithLength:photonBytes options:MTLResourceStorageModeShared];
    memset([photonBuf contents], 0, photonBytes);
    sssPhotonBuf = [dev newBufferWithLength:photonBytes options:MTLResourceStorageModeShared];
    memset([sssPhotonBuf contents], 0, photonBytes);

    // --- 2. Upload params ---
    GpuPhotonParams params{};
    params.numPhotons  = numPhotons;
    params.frameIndex  = frameIndex++;
    id<MTLBuffer> paramsBuf = [dev newBufferWithBytes:&params
                                               length:sizeof(params)
                                              options:MTLResourceStorageModeShared];

    uint32_t nLightsVal = nLights;
    uint32_t nMatsVal   = nMats;
    id<MTLBuffer> nLightsBuf = [dev newBufferWithBytes:&nLightsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> nMatsBuf   = [dev newBufferWithBytes:&nMatsVal   length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    // --- 3. Dispatch photonTrace kernel ---
    id<MTLCommandBuffer>         cmdBuf = [cq commandBuffer];
    id<MTLComputeCommandEncoder> enc    = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:psoPhotonTrace];
    [enc setBuffer:lightBufMTL   offset:0 atIndex:0];
    [enc setBuffer:nLightsBuf    offset:0 atIndex:1];
    [enc setBuffer:matBufMTL     offset:0 atIndex:2];
    [enc setBuffer:nMatsBuf      offset:0 atIndex:3];
    [enc setBuffer:normalBuf     offset:0 atIndex:4];
    [enc setBuffer:indexBuf      offset:0 atIndex:5];
    [enc setBuffer:meshIdxOffBuf offset:0 atIndex:6];
    [enc setBuffer:photonBuf     offset:0 atIndex:7];
    [enc setBuffer:paramsBuf     offset:0 atIndex:8];
    [enc setAccelerationStructure:tlas atBufferIndex:9];
    [enc setBuffer:sssPhotonBuf  offset:0 atIndex:10];
    [enc useResource:tlas usage:MTLResourceUsageRead];
    for (void* blasVoid : accel->blasHandles())
        [enc useResource:(__bridge id<MTLAccelerationStructure>)blasVoid usage:MTLResourceUsageRead];

    uint32_t threadCount = numPhotons;
    MTLSize tpg = MTLSizeMake(64, 1, 1);
    MTLSize tpGrid = MTLSizeMake((threadCount + 63u) / 64u, 1, 1);
    [enc dispatchThreadgroups:tpGrid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    // --- 4. Download photons and build CPU hash grid ---
    const GpuPhoton* gpuPhotons = (const GpuPhoton*)[photonBuf contents];

    // Compute AABB of valid photons
    float minX =  1e30f, minY =  1e30f, minZ =  1e30f;
    float maxX = -1e30f, maxY = -1e30f, maxZ = -1e30f;
    uint32_t validCount = 0;
    for (uint32_t i = 0; i < numPhotons; ++i) {
        const GpuPhoton& ph = gpuPhotons[i];
        if (ph.power.x == 0.f && ph.power.y == 0.f && ph.power.z == 0.f) continue;
        ++validCount;
        minX = std::min(minX, ph.position.x); minY = std::min(minY, ph.position.y); minZ = std::min(minZ, ph.position.z);
        maxX = std::max(maxX, ph.position.x); maxY = std::max(maxY, ph.position.y); maxZ = std::max(maxZ, ph.position.z);
    }

    validPhotons = validCount;
    spdlog::info("MetalPathIntegrator: {} valid caustic photons (of {} traced)", validCount, numPhotons);

    if (validCount > 0) {
    // Expand AABB by one cell on each side
    float cs  = photonSearchRadius;
    hashCellSize   = cs;
    minX -= cs; minY -= cs; minZ -= cs;
    maxX += cs; maxY += cs; maxZ += cs;
    hashGridOrigin = {minX, minY, minZ};
    hashGridDimX   = static_cast<uint32_t>(std::ceil((maxX - minX) / cs)) + 1;
    hashGridDimY   = static_cast<uint32_t>(std::ceil((maxY - minY) / cs)) + 1;
    hashGridDimZ   = static_cast<uint32_t>(std::ceil((maxZ - minZ) / cs)) + 1;
    uint32_t numCells = hashGridDimX * hashGridDimY * hashGridDimZ;

    // Assign each valid photon to a cell
    struct PhotonCell { uint32_t photonIdx; uint32_t cellIdx; };
    std::vector<PhotonCell> assignments;
    assignments.reserve(validCount);
    for (uint32_t i = 0; i < numPhotons; ++i) {
        const GpuPhoton& ph = gpuPhotons[i];
        if (ph.power.x == 0.f && ph.power.y == 0.f && ph.power.z == 0.f) continue;
        uint32_t ix = static_cast<uint32_t>((ph.position.x - minX) / cs);
        uint32_t iy = static_cast<uint32_t>((ph.position.y - minY) / cs);
        uint32_t iz = static_cast<uint32_t>((ph.position.z - minZ) / cs);
        ix = std::min(ix, hashGridDimX - 1);
        iy = std::min(iy, hashGridDimY - 1);
        iz = std::min(iz, hashGridDimZ - 1);
        uint32_t cellIdx = iz * hashGridDimX * hashGridDimY + iy * hashGridDimX + ix;
        assignments.push_back({i, cellIdx});
    }

    // Sort by cell index
    std::sort(assignments.begin(), assignments.end(),
              [](const PhotonCell& a, const PhotonCell& b){ return a.cellIdx < b.cellIdx; });

    // Build prefix sum (cellStart[c] = first assignment index for cell c)
    std::vector<uint32_t> cellStart(numCells + 1, 0);
    for (const auto& pc : assignments)
        cellStart[pc.cellIdx + 1]++;
    for (uint32_t c = 0; c < numCells; ++c)
        cellStart[c + 1] += cellStart[c];

    // Extract sorted photon indices
    std::vector<uint32_t> sortedIdx(validCount);
    for (uint32_t i = 0; i < validCount; ++i)
        sortedIdx[i] = assignments[i].photonIdx;

    // Upload
    hashCellStartBuf = [dev newBufferWithBytes:cellStart.data()
                                        length:cellStart.size() * sizeof(uint32_t)
                                       options:MTLResourceStorageModeShared];
    sortedPhotonIdxBuf = [dev newBufferWithBytes:sortedIdx.data()
                                          length:sortedIdx.size() * sizeof(uint32_t)
                                         options:MTLResourceStorageModeShared];
    spdlog::info("MetalPathIntegrator: photon hash grid {}x{}x{} ({} cells)",
                 hashGridDimX, hashGridDimY, hashGridDimZ, numCells);
    } else {
        hashCellStartBuf   = nil;
        sortedPhotonIdxBuf = nil;
    }

    // --- 5. Build SSS hash grid from sssPhotonBuf ---
    const GpuPhoton* sssRaw = (const GpuPhoton*)[sssPhotonBuf contents];
    float sMinX =  1e30f, sMinY =  1e30f, sMinZ =  1e30f;
    float sMaxX = -1e30f, sMaxY = -1e30f, sMaxZ = -1e30f;
    uint32_t sssCount = 0;
    for (uint32_t i = 0; i < numPhotons; ++i) {
        const GpuPhoton& ph = sssRaw[i];
        if (ph.power.x == 0.f && ph.power.y == 0.f && ph.power.z == 0.f) continue;
        ++sssCount;
        sMinX = std::min(sMinX, ph.position.x); sMinY = std::min(sMinY, ph.position.y); sMinZ = std::min(sMinZ, ph.position.z);
        sMaxX = std::max(sMaxX, ph.position.x); sMaxY = std::max(sMaxY, ph.position.y); sMaxZ = std::max(sMaxZ, ph.position.z);
    }
    sssValidPhotons = sssCount;
    spdlog::info("MetalPathIntegrator: {} valid SSS photons (of {} traced)", sssCount, numPhotons);

    if (sssCount == 0) {
        sssMapEnabled         = false;
        sssHashCellStartBuf   = nil;
        sssSortedPhotonIdxBuf = nil;
    } else {
        // Cell size = search radius = 3*d_max so that a ±1 cell traversal
        // (27 cells) covers 3d support (~95% of weight, same visual result as 6d
        // at typical photon densities, with 8x fewer photons per query).
        // Floor at photonSearchRadius for scenes with no SSS or very small d.
        float scsIdeal = (sssD_max > 0.f) ? 3.f * sssD_max : photonSearchRadius;
        float scs = std::max(scsIdeal, photonSearchRadius);
        sssHashCellSize = scs;
        sMinX -= scs; sMinY -= scs; sMinZ -= scs;
        sMaxX += scs; sMaxY += scs; sMaxZ += scs;
        sssHashGridOrigin = {sMinX, sMinY, sMinZ};
        sssHashGridDimX   = static_cast<uint32_t>(std::ceil((sMaxX - sMinX) / scs)) + 1;
        sssHashGridDimY   = static_cast<uint32_t>(std::ceil((sMaxY - sMinY) / scs)) + 1;
        sssHashGridDimZ   = static_cast<uint32_t>(std::ceil((sMaxZ - sMinZ) / scs)) + 1;
        uint32_t sssNumCells = sssHashGridDimX * sssHashGridDimY * sssHashGridDimZ;

        struct SssCell { uint32_t photonIdx; uint32_t cellIdx; };
        std::vector<SssCell> sssAssignments;
        sssAssignments.reserve(sssCount);
        for (uint32_t i = 0; i < numPhotons; ++i) {
            const GpuPhoton& ph = sssRaw[i];
            if (ph.power.x == 0.f && ph.power.y == 0.f && ph.power.z == 0.f) continue;
            uint32_t ix = static_cast<uint32_t>((ph.position.x - sMinX) / scs);
            uint32_t iy = static_cast<uint32_t>((ph.position.y - sMinY) / scs);
            uint32_t iz = static_cast<uint32_t>((ph.position.z - sMinZ) / scs);
            ix = std::min(ix, sssHashGridDimX - 1);
            iy = std::min(iy, sssHashGridDimY - 1);
            iz = std::min(iz, sssHashGridDimZ - 1);
            uint32_t cellIdx = iz * sssHashGridDimX * sssHashGridDimY
                             + iy * sssHashGridDimX + ix;
            sssAssignments.push_back({i, cellIdx});
        }
        std::sort(sssAssignments.begin(), sssAssignments.end(),
                  [](const SssCell& a, const SssCell& b){ return a.cellIdx < b.cellIdx; });

        std::vector<uint32_t> sssCellStart(sssNumCells + 1, 0);
        for (const auto& sc : sssAssignments)
            sssCellStart[sc.cellIdx + 1]++;
        for (uint32_t c = 0; c < sssNumCells; ++c)
            sssCellStart[c + 1] += sssCellStart[c];

        std::vector<uint32_t> sssSortedIdx(sssCount);
        for (uint32_t i = 0; i < sssCount; ++i)
            sssSortedIdx[i] = sssAssignments[i].photonIdx;

        sssHashCellStartBuf = [dev newBufferWithBytes:sssCellStart.data()
                                               length:sssCellStart.size() * sizeof(uint32_t)
                                              options:MTLResourceStorageModeShared];
        sssSortedPhotonIdxBuf = [dev newBufferWithBytes:sssSortedIdx.data()
                                                 length:sssSortedIdx.size() * sizeof(uint32_t)
                                                options:MTLResourceStorageModeShared];
        sssMapEnabled = true;
        sssSearchRadius = scs;   // matches hash cell size → ±1 traversal in shader
        spdlog::info("MetalPathIntegrator: SSS hash grid {}x{}x{} ({} cells)",
                     sssHashGridDimX, sssHashGridDimY, sssHashGridDimZ, sssNumCells);
    }
}

// ---------------------------------------------------------------------------
// prepare() — build accel structure, upload materials + lights
// ---------------------------------------------------------------------------
void MetalPathIntegrator::prepare(const SceneView& scene) {
    if (!isValid() || !scene.accel) return;

    // Photon mode without any caustic-flagged material is wasted work —
    // mirror the CPU + CUDA warning.  See PhotonMapIntegrator.cpp.
    if (m_impl->photonMapEnabled) {
        bool anyCaustic = false;
        for (const IMaterial* mat : scene.materials) {
            if (mat && mat->isCausticGenerator()) { anyCaustic = true; break; }
        }
        if (!anyCaustic) {
            spdlog::warn("Photon map: no materials are flagged as caustic generators "
                         "(inputs:anacapa_caustic). The photon pass will produce no "
                         "caustic photons; consider --integrator path, or flag the "
                         "focusing surfaces.");
        }
    }

    id<MTLDevice>       dev = (__bridge id<MTLDevice>)      m_impl->ctx->device();
    id<MTLCommandQueue> cq  = (__bridge id<MTLCommandQueue>)m_impl->ctx->commandQueue();

    // Retrieve the GeometryPool through the CPU BVH accel interface.
    // BVHBackend stores a reference to the pool; we need it to build BLAS/TLAS.
    // Access it via the IAccelerationStructure's pool() method.
    m_impl->geomPool = &scene.accel->pool();

    // Build hardware acceleration structure (hair ribbons included when curvePool is present)
    m_impl->accel = std::make_unique<MetalAccelStructure>(
        (__bridge void*)dev, (__bridge void*)cq, *m_impl->geomPool,
        scene.curvePool, scene.hairTessSteps);

    if (!m_impl->accel->isValid()) {
        spdlog::error("MetalPathIntegrator::prepare — accel build failed");
        return;
    }

    // Hair material buffer — one slot per scene.materials entry (zero for non-hair slots).
    // hairMeshBaseID comes from the accel structure built above.
    m_impl->hairMeshBaseID = m_impl->accel->hairMeshBaseID();
    {
        uint32_t nSlots = static_cast<uint32_t>(scene.materials.size());
        if (nSlots == 0) nSlots = 1;  // Metal doesn't allow zero-length buffers
        std::vector<GpuHairMaterial> hairMats(nSlots);
        for (uint32_t i = 0; i < static_cast<uint32_t>(scene.materials.size()); ++i)
            hairMats[i] = extractGpuHairMaterial(scene.materials[i]);
        m_impl->hairMatBuf = [dev newBufferWithBytes:hairMats.data()
                                              length:nSlots * sizeof(GpuHairMaterial)
                                             options:MTLResourceStorageModeShared];
        m_impl->numHairMats = nSlots;
    }

    // Upload materials; compute sssD_max for SSS hash grid sizing
    uint32_t nMat = static_cast<uint32_t>(scene.materials.size());
    m_impl->matBuf  = std::make_unique<MetalBuffer<GpuMaterial>>(
        (__bridge void*)dev, std::max(nMat, 1u));
    m_impl->sssD_max = 0.f;
    for (uint32_t i = 0; i < nMat; ++i) {
        (*m_impl->matBuf)[i] = extractGpuMaterial(scene.materials[i]);
        if (scene.materials[i]) {
            auto sss = scene.materials[i]->subsurfaceParams();
            if (sss.weight > 0.f)
                m_impl->sssD_max = std::max(m_impl->sssD_max, sss.radius * sss.scale);
        }
    }
    m_impl->numMaterials = nMat;

    // Upload lights (including dome/infinite lights)
    std::vector<GpuLight> gpuLights;
    for (const ILight* l : scene.lights) {
        if (l)
            gpuLights.push_back(extractGpuLight(l));
    }
    m_impl->numLights = static_cast<uint32_t>(gpuLights.size());
    m_impl->lightBuf  = std::make_unique<MetalBuffer<GpuLight>>(
        (__bridge void*)dev, std::max((uint32_t)gpuLights.size(), 1u));
    for (uint32_t i = 0; i < gpuLights.size(); ++i)
        (*m_impl->lightBuf)[i] = gpuLights[i];

    // Upload HDRI environment map texture (RGBA32Float, padded from RGB)
    m_impl->envTexture = nil;
    const DomeLight* domeLight = nullptr;
    for (const ILight* l : scene.lights) {
        if ((domeLight = dynamic_cast<const DomeLight*>(l))) break;
    }
    if (domeLight && domeLight->envWidth() > 0) {
        uint32_t ew = domeLight->envWidth();
        uint32_t eh = domeLight->envHeight();
        const float* rgb = domeLight->pixels();

        MTLTextureDescriptor* td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:ew height:eh mipmapped:NO];
        td.usage       = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;

        id<MTLTexture> tex = [dev newTextureWithDescriptor:td];

        // Pad RGB → RGBA row by row
        std::vector<float> rgba(static_cast<size_t>(ew) * eh * 4);
        for (uint32_t i = 0; i < ew * eh; ++i) {
            rgba[i*4+0] = rgb[i*3+0];
            rgba[i*4+1] = rgb[i*3+1];
            rgba[i*4+2] = rgb[i*3+2];
            rgba[i*4+3] = 1.f;
        }
        [tex replaceRegion:MTLRegionMake2D(0, 0, ew, eh)
              mipmapLevel:0
                withBytes:rgba.data()
              bytesPerRow:ew * 4 * sizeof(float)];

        m_impl->envTexture = tex;

        // Store the rotation rows and intensity for use in renderTile()
        Vec3f r0, r1, r2;
        domeLight->getRotation(r0, r1, r2);
        m_impl->envRot[0]     = r0;
        m_impl->envRot[1]     = r1;
        m_impl->envRot[2]     = r2;
        m_impl->envIntensity  = domeLight->intensity();

        // Upload importance sampling CDF tables
        {
            const auto& margCdf = domeLight->marginalCdf();
            m_impl->envMarginalCdfBuf = [dev newBufferWithBytes:margCdf.data()
                                                         length:margCdf.size() * sizeof(float)
                                                        options:MTLResourceStorageModeShared];

            auto condCdf = domeLight->flatConditionalCdf();
            m_impl->envConditionalCdfBuf = [dev newBufferWithBytes:condCdf.data()
                                                            length:condCdf.size() * sizeof(float)
                                                           options:MTLResourceStorageModeShared];
            m_impl->envCdfWidth  = ew;
            m_impl->envCdfHeight = eh;
        }

        spdlog::info("MetalPathIntegrator: uploaded {}x{} HDRI env texture + CDF tables", ew, eh);
    }

    // GGX energy-compensation LUTs — uploaded once, reused across renders.
    if (!m_impl->specAlbedoLUTBuf) {
        int N_COS = specAlbedoLUTCosBins();
        int N_R   = specAlbedoLUTRoughnessBins();
        size_t lutBytes    = static_cast<size_t>(N_COS) * N_R * sizeof(float);
        size_t avgLutBytes = static_cast<size_t>(N_R) * sizeof(float);
        m_impl->specAlbedoLUTBuf = [dev newBufferWithBytes:specAlbedoLUTData()
                                                    length:lutBytes
                                                   options:MTLResourceStorageModeShared];
        m_impl->specAvgAlbedoLUTBuf = [dev newBufferWithBytes:specAvgAlbedoLUTData()
                                                       length:avgLutBytes
                                                      options:MTLResourceStorageModeShared];
        m_impl->specLUTCosBins   = static_cast<uint32_t>(N_COS);
        m_impl->specLUTRoughBins = static_cast<uint32_t>(N_R);
    }

    // 1×1 white fallback texture (used when no dome light is present)
    {
        MTLTextureDescriptor* td = [MTLTextureDescriptor
            texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
            width:1 height:1 mipmapped:NO];
        td.usage       = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModeShared;
        id<MTLTexture> fb = [dev newTextureWithDescriptor:td];
        float white[4] = {1.f, 1.f, 1.f, 1.f};
        [fb replaceRegion:MTLRegionMake2D(0, 0, 1, 1)
             mipmapLevel:0
               withBytes:white
             bytesPerRow:4 * sizeof(float)];
        m_impl->fallbackEnvTex = fb;
    }

    spdlog::info("MetalPathIntegrator::prepare — {} materials, {} lights, "
                 "{} verts, {} tris",
                 m_impl->numMaterials, m_impl->numLights,
                 m_impl->accel->totalVertices(),
                 m_impl->accel->totalTriangles());

    // Build GPU caustic photon map if enabled (requires accel structure to be ready)
    if (m_impl->photonMapEnabled && m_impl->psoPhotonTrace) {
        id<MTLAccelerationStructure> tlas = (__bridge id<MTLAccelerationStructure>)m_impl->accel->tlas();
        m_impl->buildPhotonMap(
            dev, cq, tlas,
            (__bridge id<MTLBuffer>)m_impl->lightBuf->handle(), m_impl->numLights,
            (__bridge id<MTLBuffer>)m_impl->matBuf->handle(),   m_impl->numMaterials,
            (__bridge id<MTLBuffer>)m_impl->accel->normalBuffer(),
            (__bridge id<MTLBuffer>)m_impl->accel->indexBuffer(),
            (__bridge id<MTLBuffer>)m_impl->accel->meshIndexOffsetBuffer());
    }

    m_impl->preparedOnce = true;
}

// ---------------------------------------------------------------------------
// renderFrame() — whole-image dispatch; used as the adaptive base pass.
// ---------------------------------------------------------------------------
bool MetalPathIntegrator::renderFrame(const SceneView& scene,
                                       uint32_t filmWidth,
                                       uint32_t filmHeight,
                                       uint32_t sampleStart,
                                       uint32_t sampleCount,
                                       Film& film)
{
    if (!isValid() || !m_impl->preparedOnce) return false;

    TileRequest tile;
    tile.x0          = 0;
    tile.y0          = 0;
    tile.width       = filmWidth;
    tile.height      = filmHeight;
    tile.sampleStart = sampleStart;
    tile.sampleCount = sampleCount;

    id<MTLDevice>       dev = (__bridge id<MTLDevice>)      m_impl->ctx->device();
    id<MTLCommandQueue> cq  = (__bridge id<MTLCommandQueue>)m_impl->ctx->commandQueue();

    Camera cam = scene.camera.value_or(Camera::makePinhole(
        {0.f, 0.f, -2.5f}, {0.f, 0.f, 1.f}, {0.f, 1.f, 0.f},
        50.f, filmWidth, filmHeight));

    GpuCameraParams camParams{};
    camParams.origin      = {cam.origin.x,         cam.origin.y,         cam.origin.z};
    camParams.horizontal  = {cam.horizontal.x,      cam.horizontal.y,     cam.horizontal.z};
    camParams.vertical    = {cam.vertical.x,        cam.vertical.y,       cam.vertical.z};
    camParams.lowerLeft   = {cam.lowerLeftCorner.x, cam.lowerLeftCorner.y,cam.lowerLeftCorner.z};
    camParams.imageWidth  = filmWidth;
    camParams.imageHeight = filmHeight;
    camParams.samplesPerPixel = sampleCount;
    camParams.maxDepth        = m_impl->maxDepth;
    camParams.tileX0     = 0;
    camParams.tileY0     = 0;
    camParams.tileWidth  = filmWidth;
    camParams.tileHeight = filmHeight;

    camParams.hasEnvLight  = 0;
    camParams.envLe        = {0.f, 0.f, 0.f};
    camParams.envIntensity = 1.0f;
    camParams.envRot0 = {1.f, 0.f, 0.f};
    camParams.envRot1 = {0.f, 1.f, 0.f};
    camParams.envRot2 = {0.f, 0.f, 1.f};
    if (scene.envLight) {
        camParams.hasEnvLight  = 1;
        camParams.envIntensity = m_impl->envIntensity;
        camParams.envRot0 = {m_impl->envRot[0].x, m_impl->envRot[0].y, m_impl->envRot[0].z};
        camParams.envRot1 = {m_impl->envRot[1].x, m_impl->envRot[1].y, m_impl->envRot[1].z};
        camParams.envRot2 = {m_impl->envRot[2].x, m_impl->envRot[2].y, m_impl->envRot[2].z};
        static const Vec3f kDirs[] = {
            {0,1,0},{0.577f,0.577f,0.577f},{-0.577f,0.577f,0.577f},
                    {0.577f,0.577f,-0.577f},{-0.577f,0.577f,-0.577f},
        };
        Spectrum avg{};
        for (const Vec3f& d : kDirs) avg += scene.envLight->Le({}, {}, d);
        avg = avg * (1.f / 5.f);
        camParams.envLe = {avg.x, avg.y, avg.z};
    }
    camParams.shutterOpen  = cam.shutterOpen;
    camParams.shutterClose = cam.shutterClose;
    camParams.envMapWidth    = m_impl->envCdfWidth;
    camParams.envMapHeight   = m_impl->envCdfHeight;
    camParams.fireflyClamp   = m_impl->fireflyClamp;
    camParams.specLUTCosBins   = m_impl->specLUTCosBins;
    camParams.specLUTRoughBins = m_impl->specLUTRoughBins;
    camParams.pixelFilterBins   = m_impl->pixelFilterBins;
    camParams.pixelFilterRadius = m_impl->pixelFilterRadius;
    camParams.hairMeshBaseID    = m_impl->hairMeshBaseID;
    camParams.photonMapEnabled  = m_impl->photonMapEnabled && (m_impl->validPhotons > 0) ? 1u : 0u;
    camParams.photonSearchRadius = m_impl->photonSearchRadius;
    camParams.hashGridOrigin    = m_impl->hashGridOrigin;
    camParams.hashCellSize      = m_impl->hashCellSize;
    camParams.hashGridDimX      = m_impl->hashGridDimX;
    camParams.hashGridDimY      = m_impl->hashGridDimY;
    camParams.hashGridDimZ      = m_impl->hashGridDimZ;
    // SSS photon map
    camParams.sssMapEnabled    = m_impl->sssMapEnabled && (m_impl->sssValidPhotons > 0) ? 1u : 0u;
    camParams.sssSearchRadius  = m_impl->sssSearchRadius;
    camParams.sssHashOrigin    = m_impl->sssHashGridOrigin;
    camParams.sssHashCellSize  = m_impl->sssHashCellSize;
    camParams.sssHashDimX      = m_impl->sssHashGridDimX;
    camParams.sssHashDimY      = m_impl->sssHashGridDimY;
    camParams.sssHashDimZ      = m_impl->sssHashGridDimZ;
    // Camera motion blur
    camParams.hasMotion = cam.hasMotion ? 1u : 0u;
    camParams.originClose     = {cam.originClose.x,     cam.originClose.y,     cam.originClose.z};
    camParams.lowerLeftClose  = {cam.lowerLeftClose.x,  cam.lowerLeftClose.y,  cam.lowerLeftClose.z};
    camParams.horizontalClose = {cam.horizontalClose.x, cam.horizontalClose.y, cam.horizontalClose.z};
    camParams.verticalClose   = {cam.verticalClose.x,   cam.verticalClose.y,   cam.verticalClose.z};

    // Use the persistent accum buffer — allocates only on first call or size change.
    // clearAccum() should be called when starting a fresh render (new scene/camera).
    m_impl->ensureAccum(dev, filmWidth, filmHeight);
    id<MTLBuffer> accumMTL = m_impl->accumMTL;

    id<MTLBuffer> camMTL   = [dev newBufferWithBytes:&camParams
                                              length:sizeof(GpuCameraParams)
                                             options:MTLResourceStorageModeShared];
    uint32_t numLightsVal = m_impl->numLights;
    uint32_t numMatsVal   = m_impl->numMaterials;
    id<MTLBuffer> numLightsMTL = [dev newBufferWithBytes:&numLightsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> numMatsMTL   = [dev newBufferWithBytes:&numMatsVal   length:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    id<MTLAccelerationStructure> tlas = (__bridge id<MTLAccelerationStructure>)m_impl->accel->tlas();
    id<MTLTexture> envTex = m_impl->envTexture ? m_impl->envTexture : m_impl->fallbackEnvTex;

    // Samples per kernel launch.  Each thread traces kBatchSize full paths
    // before writing to the accum buffer, amortising command-buffer overhead.
    // 4 is a good starting point: meaningfully reduces round-trips without
    // exhausting the register file on Apple Silicon.
    constexpr uint32_t kBatchSize    = 4;
    // Flush to film every kMergeInterval dispatches (= kMergeInterval*kBatchSize samples)
    constexpr uint32_t kMergeInterval = 4;

    auto flushToFilm = [&]() {
        const GpuAccumPixel* accumData = (const GpuAccumPixel*)[accumMTL contents];
        TileBuffer tb(0, 0, filmWidth, filmHeight);
        for (uint32_t py = 0; py < filmHeight; ++py) {
            for (uint32_t px = 0; px < filmWidth; ++px) {
                const GpuAccumPixel& p = accumData[py * filmWidth + px];
                float w = p.weight > 0.f ? p.weight : 1.f;
                tb.add(px, py, p.r / w, p.g / w, p.b / w, w);
                tb.addLumSq(px, py, p.sumLumSq);
            }
        }
        film.mergeTile(tb);
    };

    id<MTLBuffer> batchMTL = [dev newBufferWithLength:sizeof(GpuSampleBatch)
                                              options:MTLResourceStorageModeShared];

    MTLSize threadsPerGrid        = MTLSizeMake(filmWidth, filmHeight, 1);
    MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 1);

    uint32_t dispatches = 0;
    for (uint32_t s = 0; s < sampleCount; s += kBatchSize) {
        GpuSampleBatch sb;
        sb.sampleStart = sampleStart + s;
        sb.batchSize   = std::min(kBatchSize, sampleCount - s);
        memcpy([batchMTL contents], &sb, sizeof(sb));

        id<MTLCommandBuffer>         cmdBuf = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc    = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:m_impl->psoShade];
        [enc setBuffer:camMTL        offset:0 atIndex:0];
        [enc setBuffer:accumMTL      offset:0 atIndex:1];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->lightBuf->handle() offset:0 atIndex:2];
        [enc setBuffer:numLightsMTL  offset:0 atIndex:3];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->matBuf->handle()   offset:0 atIndex:4];
        [enc setBuffer:numMatsMTL    offset:0 atIndex:5];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->normalBuffer()           offset:0 atIndex:6];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->indexBuffer()            offset:0 atIndex:7];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->triMeshIDBuffer()        offset:0 atIndex:8];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->meshVertexOffsetBuffer() offset:0 atIndex:9];
        [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->meshIndexOffsetBuffer()  offset:0 atIndex:10];
        [enc setBuffer:batchMTL      offset:0 atIndex:11];
        [enc setAccelerationStructure:tlas atBufferIndex:12];
        // HDRI importance sampling CDF tables (nil when no dome light loaded)
        [enc setBuffer:m_impl->envMarginalCdfBuf    offset:0 atIndex:13];
        [enc setBuffer:m_impl->envConditionalCdfBuf offset:0 atIndex:14];
        // GGX energy-compensation LUTs
        [enc setBuffer:m_impl->specAlbedoLUTBuf    offset:0 atIndex:15];
        [enc setBuffer:m_impl->specAvgAlbedoLUTBuf offset:0 atIndex:16];
        // Pixel filter CDF / sign tables (nil → kernel falls back to box-1.0)
        [enc setBuffer:m_impl->pixelFilterCdfBuf   offset:0 atIndex:17];
        [enc setBuffer:m_impl->pixelFilterSignBuf  offset:0 atIndex:18];
        // Hair buffers (nil-safe — Metal silently ignores nil bindings)
        if (m_impl->accel->hairTriBuffer())
            [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->hairTriBuffer() offset:0 atIndex:19];
        if (m_impl->hairMatBuf)
            [enc setBuffer:m_impl->hairMatBuf offset:0 atIndex:20];
        // Photon map hash grid (nil-safe when photon map is off)
        if (m_impl->hashCellStartBuf)
            [enc setBuffer:m_impl->hashCellStartBuf   offset:0 atIndex:21];
        if (m_impl->sortedPhotonIdxBuf)
            [enc setBuffer:m_impl->sortedPhotonIdxBuf offset:0 atIndex:22];
        if (m_impl->photonBuf)
            [enc setBuffer:m_impl->photonBuf          offset:0 atIndex:23];
        // SSS photon map hash grid (use photonBuf as dummy when SSS not active)
        [enc setBuffer:(m_impl->sssHashCellStartBuf   ?: m_impl->photonBuf) offset:0 atIndex:24];
        [enc setBuffer:(m_impl->sssSortedPhotonIdxBuf ?: m_impl->photonBuf) offset:0 atIndex:25];
        [enc setBuffer:(m_impl->sssPhotonBuf          ?: m_impl->photonBuf) offset:0 atIndex:26];
        [enc setTexture:envTex atIndex:0];
        [enc useResource:tlas usage:MTLResourceUsageRead];
        for (void* blasVoid : m_impl->accel->blasHandles())
            [enc useResource:(__bridge id<MTLAccelerationStructure>)blasVoid usage:MTLResourceUsageRead];
        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            spdlog::error("MetalPathIntegrator::renderFrame GPU error: {}",
                          cmdBuf.error ? [[cmdBuf.error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        if ((++dispatches) % kMergeInterval == 0)
            flushToFilm();
    }

    // Final flush (catches remainder if sampleCount % kMergeInterval != 0)
    flushToFilm();
    return true;
}

// ---------------------------------------------------------------------------
// renderTile() — dispatch Shade kernel, read back into TileBuffer
// ---------------------------------------------------------------------------
void MetalPathIntegrator::renderTile(const SceneView& scene,
                                      const TileRequest& tile,
                                      uint32_t filmWidth,
                                      uint32_t filmHeight,
                                      ISampler& /*sampler*/,
                                      TileBuffer& out)
{
    if (!isValid() || !m_impl->preparedOnce) return;

    id<MTLDevice>       dev = (__bridge id<MTLDevice>)      m_impl->ctx->device();
    id<MTLCommandQueue> cq  = (__bridge id<MTLCommandQueue>)m_impl->ctx->commandQueue();

    Camera cam = scene.camera.value_or(Camera::makePinhole(
        {0.f, 0.f, -2.5f},
        {0.f, 0.f,  1.f},
        {0.f, 1.f,  0.f},
        50.f,
        filmWidth, filmHeight
    ));
    uint32_t tileW = std::min(tile.width,  filmWidth  - tile.x0);
    uint32_t tileH = std::min(tile.height, filmHeight - tile.y0);

    // Camera params — tile offset so the shader dispatches only this tile's pixels
    GpuCameraParams camParams{};
    camParams.origin     = {cam.origin.x,          cam.origin.y,          cam.origin.z};
    camParams.horizontal = {cam.horizontal.x,       cam.horizontal.y,      cam.horizontal.z};
    camParams.vertical   = {cam.vertical.x,         cam.vertical.y,        cam.vertical.z};
    camParams.lowerLeft  = {cam.lowerLeftCorner.x,  cam.lowerLeftCorner.y, cam.lowerLeftCorner.z};
    camParams.imageWidth  = filmWidth;
    camParams.imageHeight = filmHeight;
    camParams.samplesPerPixel = tile.sampleCount;
    camParams.maxDepth        = m_impl->maxDepth;
    camParams.tileX0    = tile.x0;
    camParams.tileY0    = tile.y0;
    camParams.tileWidth  = tileW;
    camParams.tileHeight = tileH;

    // Environment/dome light
    camParams.hasEnvLight  = 0;
    camParams.envLe        = {0.f, 0.f, 0.f};
    camParams.envIntensity = 1.0f;
    camParams.envRot0 = {1.f, 0.f, 0.f};
    camParams.envRot1 = {0.f, 1.f, 0.f};
    camParams.envRot2 = {0.f, 0.f, 1.f};
    if (scene.envLight) {
        camParams.hasEnvLight  = 1;
        camParams.envIntensity = m_impl->envIntensity;
        // Rotation rows for directional HDRI lookup in shader
        camParams.envRot0 = {m_impl->envRot[0].x, m_impl->envRot[0].y, m_impl->envRot[0].z};
        camParams.envRot1 = {m_impl->envRot[1].x, m_impl->envRot[1].y, m_impl->envRot[1].z};
        camParams.envRot2 = {m_impl->envRot[2].x, m_impl->envRot[2].y, m_impl->envRot[2].z};
        // Average Le as fallback for any path that doesn't hit the texture
        static const Vec3f kDirs[] = {
            {0,1,0}, {0.577f,0.577f,0.577f}, {-0.577f,0.577f,0.577f},
                     {0.577f,0.577f,-0.577f}, {-0.577f,0.577f,-0.577f},
        };
        Spectrum avg{};
        for (const Vec3f& d : kDirs) avg += scene.envLight->Le({}, {}, d);
        avg = avg * (1.f / 5.f);
        camParams.envLe = {avg.x, avg.y, avg.z};
    }
    camParams.shutterOpen  = cam.shutterOpen;
    camParams.shutterClose = cam.shutterClose;
    camParams.envMapWidth    = m_impl->envCdfWidth;
    camParams.envMapHeight   = m_impl->envCdfHeight;
    camParams.fireflyClamp   = m_impl->fireflyClamp;
    camParams.specLUTCosBins   = m_impl->specLUTCosBins;
    camParams.specLUTRoughBins = m_impl->specLUTRoughBins;
    camParams.pixelFilterBins   = m_impl->pixelFilterBins;
    camParams.pixelFilterRadius = m_impl->pixelFilterRadius;
    camParams.hairMeshBaseID    = m_impl->hairMeshBaseID;
    camParams.photonMapEnabled  = m_impl->photonMapEnabled && (m_impl->validPhotons > 0) ? 1u : 0u;
    camParams.photonSearchRadius = m_impl->photonSearchRadius;
    camParams.hashGridOrigin    = m_impl->hashGridOrigin;
    camParams.hashCellSize      = m_impl->hashCellSize;
    camParams.hashGridDimX      = m_impl->hashGridDimX;
    camParams.hashGridDimY      = m_impl->hashGridDimY;
    camParams.hashGridDimZ      = m_impl->hashGridDimZ;
    // SSS photon map
    camParams.sssMapEnabled    = m_impl->sssMapEnabled && (m_impl->sssValidPhotons > 0) ? 1u : 0u;
    camParams.sssSearchRadius  = m_impl->sssSearchRadius;
    camParams.sssHashOrigin    = m_impl->sssHashGridOrigin;
    camParams.sssHashCellSize  = m_impl->sssHashCellSize;
    camParams.sssHashDimX      = m_impl->sssHashGridDimX;
    camParams.sssHashDimY      = m_impl->sssHashGridDimY;
    camParams.sssHashDimZ      = m_impl->sssHashGridDimZ;
    // Camera motion blur
    camParams.hasMotion = cam.hasMotion ? 1u : 0u;
    camParams.originClose     = {cam.originClose.x,     cam.originClose.y,     cam.originClose.z};
    camParams.lowerLeftClose  = {cam.lowerLeftClose.x,  cam.lowerLeftClose.y,  cam.lowerLeftClose.z};
    camParams.horizontalClose = {cam.horizontalClose.x, cam.horizontalClose.y, cam.horizontalClose.z};
    camParams.verticalClose   = {cam.verticalClose.x,   cam.verticalClose.y,   cam.verticalClose.z};

    // Tile-sized accum buffer (gid is local; shader writes gid.y*tileW+gid.x)
    size_t accumBytes   = tileW * tileH * sizeof(GpuAccumPixel);
    id<MTLBuffer> accumMTL = [dev newBufferWithLength:accumBytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> camMTL   = [dev newBufferWithBytes:&camParams
                                              length:sizeof(GpuCameraParams)
                                             options:MTLResourceStorageModeShared];
    uint32_t numLightsVal = m_impl->numLights;
    uint32_t numMatsVal   = m_impl->numMaterials;
    id<MTLBuffer> numLightsMTL = [dev newBufferWithBytes:&numLightsVal
                                                  length:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> numMatsMTL   = [dev newBufferWithBytes:&numMatsVal
                                                  length:sizeof(uint32_t)
                                                 options:MTLResourceStorageModeShared];
    id<MTLBuffer> sampleIdxMTL = [dev newBufferWithLength:sizeof(uint32_t)
                                                   options:MTLResourceStorageModeShared];

    // Zero the accumulation buffer
    memset([accumMTL contents], 0, accumBytes);

    id<MTLAccelerationStructure> tlas =
        (__bridge id<MTLAccelerationStructure>)m_impl->accel->tlas();

    // Dispatch one sample at a time so the RNG seed changes per sample
    for (uint32_t s = 0; s < tile.sampleCount; ++s) {
        *(uint32_t*)[sampleIdxMTL contents] = tile.sampleStart + s;

        id<MTLCommandBuffer>         cmdBuf  = [cq commandBuffer];
        id<MTLComputeCommandEncoder> enc     = [cmdBuf computeCommandEncoder];
        [enc setComputePipelineState:m_impl->psoShade];

        auto setB = [&](uint32_t idx, id<MTLBuffer> b) {
            [enc setBuffer:b offset:0 atIndex:idx];
        };
        auto setBV = [&](uint32_t idx, void* vp) {
            setB(idx, (__bridge id<MTLBuffer>)vp);
        };

        setB (0,  camMTL);
        setB (1,  accumMTL);
        setB (2,  (__bridge id<MTLBuffer>)m_impl->lightBuf->handle());
        setB (3,  numLightsMTL);
        setB (4,  (__bridge id<MTLBuffer>)m_impl->matBuf->handle());
        setB (5,  numMatsMTL);
        setBV(6,  m_impl->accel->normalBuffer());
        setBV(7,  m_impl->accel->indexBuffer());
        setBV(8,  m_impl->accel->triMeshIDBuffer());
        setBV(9,  m_impl->accel->meshVertexOffsetBuffer());
        setBV(10, m_impl->accel->meshIndexOffsetBuffer());
        setB (11, sampleIdxMTL);

        [enc setAccelerationStructure:tlas atBufferIndex:12];

        // HDRI importance sampling CDF buffers (nil-safe: Metal ignores nil bindings)
        if (m_impl->envMarginalCdfBuf)
            [enc setBuffer:m_impl->envMarginalCdfBuf    offset:0 atIndex:13];
        if (m_impl->envConditionalCdfBuf)
            [enc setBuffer:m_impl->envConditionalCdfBuf offset:0 atIndex:14];
        // GGX energy-compensation LUTs
        if (m_impl->specAlbedoLUTBuf)
            [enc setBuffer:m_impl->specAlbedoLUTBuf    offset:0 atIndex:15];
        if (m_impl->specAvgAlbedoLUTBuf)
            [enc setBuffer:m_impl->specAvgAlbedoLUTBuf offset:0 atIndex:16];
        // Pixel filter CDF / sign tables (nil → kernel falls back to box-1.0)
        if (m_impl->pixelFilterCdfBuf)
            [enc setBuffer:m_impl->pixelFilterCdfBuf  offset:0 atIndex:17];
        if (m_impl->pixelFilterSignBuf)
            [enc setBuffer:m_impl->pixelFilterSignBuf offset:0 atIndex:18];
        // Hair buffers
        if (m_impl->accel->hairTriBuffer())
            [enc setBuffer:(__bridge id<MTLBuffer>)m_impl->accel->hairTriBuffer() offset:0 atIndex:19];
        if (m_impl->hairMatBuf)
            [enc setBuffer:m_impl->hairMatBuf offset:0 atIndex:20];
        // Photon map hash grid (nil-safe)
        if (m_impl->hashCellStartBuf)
            [enc setBuffer:m_impl->hashCellStartBuf   offset:0 atIndex:21];
        if (m_impl->sortedPhotonIdxBuf)
            [enc setBuffer:m_impl->sortedPhotonIdxBuf offset:0 atIndex:22];
        if (m_impl->photonBuf)
            [enc setBuffer:m_impl->photonBuf          offset:0 atIndex:23];
        // SSS photon map hash grid (use photonBuf as dummy when SSS not active)
        [enc setBuffer:(m_impl->sssHashCellStartBuf   ?: m_impl->photonBuf) offset:0 atIndex:24];
        [enc setBuffer:(m_impl->sssSortedPhotonIdxBuf ?: m_impl->photonBuf) offset:0 atIndex:25];
        [enc setBuffer:(m_impl->sssPhotonBuf          ?: m_impl->photonBuf) offset:0 atIndex:26];

        // Environment texture (index 0); fallback 1×1 white if no HDRI loaded
        id<MTLTexture> envTex = m_impl->envTexture
                              ? m_impl->envTexture
                              : m_impl->fallbackEnvTex;
        [enc setTexture:envTex atIndex:0];

        // Mark TLAS + all BLAS as used — required for GPU hazard tracking.
        // Without this, the GPU cannot access the BLAS data and intersections
        // silently return no-hit.
        [enc useResource:tlas usage:MTLResourceUsageRead];
        for (void* blasVoid : m_impl->accel->blasHandles()) {
            id<MTLAccelerationStructure> blas =
                (__bridge id<MTLAccelerationStructure>)blasVoid;
            [enc useResource:blas usage:MTLResourceUsageRead];
        }

        MTLSize threadsPerGrid        = MTLSizeMake(tileW, tileH, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
        [enc dispatchThreads:threadsPerGrid
           threadsPerThreadgroup:threadsPerThreadgroup];

        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.status == MTLCommandBufferStatusError) {
            spdlog::error("MetalPathIntegrator: GPU command buffer error: {}",
                          cmdBuf.error
                              ? [[cmdBuf.error localizedDescription] UTF8String]
                              : "unknown");
            return;
        }

    }

    // Read back tile-local accum buffer (indexed gid.y * tileW + gid.x)
    const GpuAccumPixel* accumData = (const GpuAccumPixel*)[accumMTL contents];
    for (uint32_t ty = 0; ty < tileH; ++ty) {
        for (uint32_t tx = 0; tx < tileW; ++tx) {
            const GpuAccumPixel& p = accumData[ty * tileW + tx];
            float w = p.weight > 0.f ? p.weight : 1.f;
            out.add(tx, ty, p.r / w, p.g / w, p.b / w, w);
            out.addLumSq(tx, ty, p.sumLumSq);
        }
    }
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_METAL
