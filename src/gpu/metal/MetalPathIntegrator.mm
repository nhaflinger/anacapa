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

    uint32_t numMaterials = 0;
    uint32_t numLights    = 0;
    uint32_t maxDepth     = 6;

    // Pointer to scene geometry (for building the accel structure)
    const GeometryPool* geomPool = nullptr;
    bool preparedOnce = false;
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
static GpuMaterial extractGpuMaterial(const IMaterial* mat) {
    GpuMaterial gm{};
    gm.baseColor  = {0.5f, 0.5f, 0.5f};
    gm.emissive   = {0.f,  0.f,  0.f};
    gm.roughness  = 1.f;
    gm.metalness  = 0.f;
    gm.specularIOR = 1.5f;
    gm.type = kMatLambertian;

    if (!mat) return gm;

    // EmissiveMaterial
    const EmissiveMaterial* em = dynamic_cast<const EmissiveMaterial*>(mat);
    if (em) {
        gm.type = kMatEmissive;
        // Le via a dummy ShadingContext — surface normal = +Z
        SurfaceInteraction si;
        si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum Le = mat->Le(ctx, {0,0,1});
        gm.emissive = {Le.x, Le.y, Le.z};
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
        return gm;
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

    // StandardSurfaceMaterial — GGX by flags
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

    // LambertianMaterial (and default)
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

    m_impl->psoShade = makePSO(dev, lib, "shade");

    if (m_impl->psoShade)
        spdlog::info("MetalPathIntegrator: ready on '{}'", m_impl->ctx->name());
}

MetalPathIntegrator::~MetalPathIntegrator() = default;

bool MetalPathIntegrator::isValid() const {
    return m_impl->ctx && m_impl->ctx->isValid() && m_impl->psoShade != nil;
}

// ---------------------------------------------------------------------------
// prepare() — build accel structure, upload materials + lights
// ---------------------------------------------------------------------------
void MetalPathIntegrator::prepare(const SceneView& scene) {
    if (!isValid() || !scene.accel) return;

    id<MTLDevice>       dev = (__bridge id<MTLDevice>)      m_impl->ctx->device();
    id<MTLCommandQueue> cq  = (__bridge id<MTLCommandQueue>)m_impl->ctx->commandQueue();

    // Retrieve the GeometryPool through the CPU BVH accel interface.
    // BVHBackend stores a reference to the pool; we need it to build BLAS/TLAS.
    // Access it via the IAccelerationStructure's pool() method.
    m_impl->geomPool = &scene.accel->pool();

    // Build hardware acceleration structure
    m_impl->accel = std::make_unique<MetalAccelStructure>(
        (__bridge void*)dev, (__bridge void*)cq, *m_impl->geomPool);

    if (!m_impl->accel->isValid()) {
        spdlog::error("MetalPathIntegrator::prepare — accel build failed");
        return;
    }

    // Upload materials
    uint32_t nMat = static_cast<uint32_t>(scene.materials.size());
    m_impl->matBuf  = std::make_unique<MetalBuffer<GpuMaterial>>(
        (__bridge void*)dev, std::max(nMat, 1u));
    for (uint32_t i = 0; i < nMat; ++i)
        (*m_impl->matBuf)[i] = extractGpuMaterial(scene.materials[i]);
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

        spdlog::info("MetalPathIntegrator: uploaded {}x{} HDRI env texture", ew, eh);
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

    size_t accumBytes = filmWidth * filmHeight * sizeof(GpuAccumPixel);
    id<MTLBuffer> accumMTL = [dev newBufferWithLength:accumBytes
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> camMTL   = [dev newBufferWithBytes:&camParams
                                              length:sizeof(GpuCameraParams)
                                             options:MTLResourceStorageModeShared];
    uint32_t numLightsVal = m_impl->numLights;
    uint32_t numMatsVal   = m_impl->numMaterials;
    id<MTLBuffer> numLightsMTL = [dev newBufferWithBytes:&numLightsVal length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> numMatsMTL   = [dev newBufferWithBytes:&numMatsVal   length:sizeof(uint32_t) options:MTLResourceStorageModeShared];
    id<MTLBuffer> sampleIdxMTL = [dev newBufferWithLength:sizeof(uint32_t) options:MTLResourceStorageModeShared];

    memset([accumMTL contents], 0, accumBytes);

    id<MTLAccelerationStructure> tlas = (__bridge id<MTLAccelerationStructure>)m_impl->accel->tlas();
    id<MTLTexture> envTex = m_impl->envTexture ? m_impl->envTexture : m_impl->fallbackEnvTex;

    // Merge interval: update the film every N samples so the progressive
    // preview watcher sees incremental updates during the base pass.
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

    for (uint32_t s = 0; s < sampleCount; ++s) {
        *(uint32_t*)[sampleIdxMTL contents] = sampleStart + s;

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
        [enc setBuffer:sampleIdxMTL  offset:0 atIndex:11];
        [enc setAccelerationStructure:tlas atBufferIndex:12];
        [enc setTexture:envTex atIndex:0];
        [enc useResource:tlas usage:MTLResourceUsageRead];
        for (void* blasVoid : m_impl->accel->blasHandles())
            [enc useResource:(__bridge id<MTLAccelerationStructure>)blasVoid usage:MTLResourceUsageRead];
        MTLSize threadsPerGrid        = MTLSizeMake(filmWidth, filmHeight, 1);
        MTLSize threadsPerThreadgroup = MTLSizeMake(8, 8, 1);
        [enc dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
        [enc endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        if (cmdBuf.status == MTLCommandBufferStatusError) {
            spdlog::error("MetalPathIntegrator::renderFrame GPU error: {}",
                          cmdBuf.error ? [[cmdBuf.error localizedDescription] UTF8String] : "unknown");
            return false;
        }

        // Progressive update — flush every kMergeInterval samples
        if ((s + 1) % kMergeInterval == 0)
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
