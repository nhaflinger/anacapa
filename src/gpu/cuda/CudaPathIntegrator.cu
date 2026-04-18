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

#include <algorithm>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) \
        fprintf(stderr, "[error] CUDA %s %s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); \
} while(0)

// Forward declaration of the device kernel defined in shaders/Shade.cu
extern "C" __global__ void shade(LaunchParams params);

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

    const EmissiveMaterial* em = dynamic_cast<const EmissiveMaterial*>(mat);
    if (em) {
        gm.type = kMatEmissive;
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum Le = mat->Le(ctx, {0,0,1});
        gm.emissive  = {Le.x, Le.y, Le.z};
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
        return gm;
    }
    const StandardSurfaceMaterial* ssm = dynamic_cast<const StandardSurfaceMaterial*>(mat);
    if (ssm && ssm->params().transmission > 0.001f && ssm->params().metalness.value < 0.001f) {
        gm.type        = kMatGlass;
        gm.specularIOR = ssm->params().specular_IOR;
        gm.transmission = ssm->params().transmission;
        SurfaceInteraction si; si.n = si.ng = {0,0,1};
        ShadingContext ctx(si, {0,0,1});
        Spectrum alb = mat->reflectance(ctx);
        gm.baseColor = {alb.x, alb.y, alb.z};
        gm.roughness = ssm->params().roughness.value;
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
        LightSample ls = dl->sample({0,0,0},{0,1,0},{0.5f,0.5f});
        gl.Le     = {ls.Li.x, ls.Li.y, ls.Li.z};
        gl.normal = {ls.wi.x, ls.wi.y, ls.wi.z};
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

    cudaArray_t         envArray   = nullptr;
    cudaTextureObject_t envTex     = 0;
    Vec3f               envRot[3]  = {{1,0,0},{0,1,0},{0,0,1}};
    float               envIntensity = 1.0f;

    uint32_t numMaterials = 0;
    uint32_t numLights    = 0;
    uint32_t maxDepth     = 6;
    bool     preparedOnce = false;

    ~Impl() {
        if (envTex)   cudaDestroyTextureObject(envTex);
        if (envArray) cudaFreeArray(envArray);
    }

    void fillLaunchParams(LaunchParams& p, const SceneView& scene,
                          uint32_t filmWidth, uint32_t filmHeight,
                          uint32_t tileX0, uint32_t tileY0,
                          uint32_t tileW,   uint32_t tileH,
                          uint32_t sampleStart, uint32_t sampleCount,
                          GpuAccumPixel* d_accum) const;
};

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
        printf("[info]  CudaPathIntegrator: uploaded %dx%d HDRI env texture\n", ew, eh);
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
    p.positions         = reinterpret_cast<const float*>(accel->positionBuffer());
    p.bvh               = reinterpret_cast<const BvhNode*>(accel->bvhBuffer());
    p.triIndices        = reinterpret_cast<const uint32_t*>(accel->triIndexBuffer());
    p.sampleIndex       = sampleStart;
    p.envTexture        = envTex;
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

    CudaBuffer<GpuAccumPixel> d_accum(filmWidth * filmHeight);
    d_accum.zero();

    dim3 block(16, 16, 1);
    dim3 grid((filmWidth + 15) / 16, (filmHeight + 15) / 16, 1);

    // One kernel per sample — keeps dispatches short to avoid GPU watchdog.
    // Flush to film every kMergeInterval samples for progressive preview.
    constexpr uint32_t kMergeInterval = 4;

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

    for (uint32_t s = 0; s < sampleCount; ++s) {
        LaunchParams params{};
        m_impl->fillLaunchParams(params, scene,
            filmWidth, filmHeight,
            0, 0, filmWidth, filmHeight,
            sampleStart + s, 1,
            d_accum.ptr());
        shade<<<grid, block, 0, stream>>>(params);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "[error] CudaPathIntegrator::renderFrame kernel: %s\n",
                    cudaGetErrorString(err));
            return false;
        }

        if ((s + 1) % kMergeInterval == 0)
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

    dim3 block(16, 16, 1);
    dim3 grid((tileW + 15) / 16, (tileH + 15) / 16, 1);
    shade<<<grid, block, 0, stream>>>(params);
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
