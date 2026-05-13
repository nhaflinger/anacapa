#include "PhotonMapIntegrator.h"
#include "ShadowRay.h"
#include <anacapa/shading/ShadingContext.h>
#include <spdlog/spdlog.h>
#include <random>
#include <cmath>

namespace anacapa {

PhotonMapIntegrator::PhotonMapIntegrator(uint32_t maxDepth,
                                          uint32_t minDepth,
                                          float    fireflyClamp,
                                          int      numPhotons,
                                          float    searchRadius)
    : m_maxDepth(maxDepth)
    , m_minDepth(minDepth)
    , m_fireflyClamp(fireflyClamp)
    , m_numPhotons(numPhotons)
    , m_searchRadius(searchRadius)
{}

// ---------------------------------------------------------------------------
// prepare — build the light sampler then trace the photon pass.
// Called once from the main thread before any renderTile() calls.
// ---------------------------------------------------------------------------
void PhotonMapIntegrator::prepare(const SceneView& scene) {
    m_lightSampler.build(scene.lights);
    if (!scene.lights.empty())
        tracePhotons(scene);
}

// ---------------------------------------------------------------------------
// tracePhotons — emit photons from lights, trace through specular surfaces,
// and store caustic photons (those that have made ≥1 specular bounce before
// their first diffuse hit) in the caustic kd-tree.
// ---------------------------------------------------------------------------
void PhotonMapIntegrator::tracePhotons(const SceneView& scene) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    auto rand1 = [&]() -> float { return u01(rng); };
    auto rand2 = [&]() -> Vec2f { return {u01(rng), u01(rng)}; };

    std::vector<Photon> photons;
    photons.reserve(static_cast<size_t>(m_numPhotons) / 4); // caustic photons are a subset

    for (int emitted = 0; emitted < m_numPhotons; ++emitted) {
        // Select a light proportional to power
        if (m_lightSampler.empty()) break;
        auto sel = m_lightSampler.sample(rand1());
        if (!sel.light || sel.pdf <= 0.f) continue;

        // Sample an emission ray from the light
        LightLeSample les = sel.light->sampleLe(rand2(), rand2());
        if (les.pdfPos <= 0.f || les.pdfDir <= 0.f || isBlack(les.Le)) continue;

        // Initial photon power: Le * |cos θ| / (lightSelectPdf * pdfPos * pdfDir * N)
        // This normalization ensures the density estimate produces radiance in W/(m²·sr).
        float cosTheta = std::abs(dot(les.dir, les.normal));
        if (cosTheta <= 0.f) continue;

        Spectrum power = les.Le * cosTheta
                       / (sel.pdf * les.pdfPos * les.pdfDir
                          * static_cast<float>(m_numPhotons));

        if (!power.isFinite() || isBlack(power)) continue;

        Ray photonRay{les.pos, les.dir};
        photonRay.tMin = 1e-4f;

        int numSpecular = 0;

        for (int bounce = 0; bounce < static_cast<int>(m_maxDepth); ++bounce) {
            TraceResult hit = scene.accel->trace(photonRay);
            if (!hit.hit) break;

            const SurfaceInteraction& si = hit.si;
            const IMaterial* mat = (si.meshID < scene.materials.size())
                                   ? scene.materials[si.meshID] : nullptr;
            if (!mat) break;

            ShadingContext ctx(si, photonRay.direction);

            // Opacity / alpha cutout: pass through without counting as a bounce
            float opacity = mat->evalOpacity(ctx);
            if (opacity <= 0.f) {
                photonRay = spawnRay(si.p, si.ng, photonRay.direction);
                --bounce;
                continue;
            }

            Vec3f wo = -photonRay.direction;

            if (mat->isDelta()) {
                // Specular surface (mirror / glass): bounce and keep tracing
                BSDFSample bs = mat->sample(ctx, wo, rand2(), rand1());
                if (!bs.isValid() || bs.pdf <= 0.f) break;

                power = power * bs.f / bs.pdf;
                if (!power.isFinite() || isBlack(power)) break;

                photonRay = spawnRay(si.p, si.n, bs.wi);
                photonRay.skipStrandID = si.isCurve ? si.strandID : ~0u;
                ++numSpecular;
            } else {
                // Diffuse surface: store as caustic photon if we bounced through specular
                if (numSpecular > 0) {
                    Photon ph;
                    ph.position = si.p;
                    ph.wi       = photonRay.direction; // toward surface
                    ph.power    = power;
                    ph.axis     = 0; // filled in by build()
                    photons.push_back(ph);
                }
                break; // caustic map only: stop at first diffuse hit
            }

            // Russian roulette after the first few bounces
            if (bounce >= 2) {
                float survival = std::min(luminance(power) * 10.f, 0.95f);
                if (rand1() > survival) break;
                power = power * (1.f / survival);
            }
        }
    }

    spdlog::info("PhotonMap: emitted {} photons, stored {} caustic photons",
                 m_numPhotons, photons.size());
    m_causticMap.build(std::move(photons));
}

// ---------------------------------------------------------------------------
// renderTile — identical to PathIntegrator::renderTile.
// ---------------------------------------------------------------------------
void PhotonMapIntegrator::renderTile(const SceneView& scene,
                                      const TileRequest& tile,
                                      uint32_t filmWidth,
                                      uint32_t filmHeight,
                                      ISampler& sampler,
                                      TileBuffer& localTile) {
    Camera cam = scene.camera.value_or(Camera::makePinhole(
        {0.f, 0.f, -2.5f},
        {0.f, 0.f,  1.f},
        {0.f, 1.f,  0.f},
        50.f,
        filmWidth, filmHeight
    ));

    for (uint32_t ty = 0; ty < tile.height; ++ty) {
        for (uint32_t tx = 0; tx < tile.width; ++tx) {
            uint32_t px = tile.x0 + tx;
            uint32_t py = tile.y0 + ty;

            Spectrum accum       = {};
            Spectrum accumAlbedo = {};
            Vec3f    accumNormal = {};
            uint32_t aovCount    = 0;
            float    weightSum   = 0.f;
            float    sumLumSq    = 0.f;

            for (uint32_t s = 0; s < tile.sampleCount; ++s) {
                sampler.startPixelSample(px, py, tile.sampleStart + s);
                Vec2f filt = sampler.get2D();
                Vec2f lens = sampler.get2D();
                float timeU = sampler.get1D();

                float jx = filt.x, jy = filt.y, fw = 1.f;
                if (m_pixelFilter) {
                    auto fs = m_pixelFilter->sample(filt.x, filt.y);
                    jx = 0.5f + fs.dx;
                    jy = 0.5f + fs.dy;
                    fw = fs.weight;
                }
                Ray ray = cam.generateRay(px, py, jx, jy, lens.x, lens.y, timeU);

                Spectrum albedo = {};
                Vec3f    normal = {};
                Spectrum sample = Li(ray, scene, sampler, albedo, normal);
                if (sample.isFinite()) {
                    accum     += sample * fw;
                    weightSum += fw;
                    float lum = luminance(sample);
                    sumLumSq += lum * lum;
                }
                accumAlbedo += albedo;
                accumNormal  = accumNormal + normal;
                ++aovCount;
            }

            if (weightSum != 0.f)
                localTile.add(tx, ty, accum * (1.f / weightSum), weightSum);
            localTile.addLumSq(tx, ty, sumLumSq);

            if (aovCount > 0) {
                float invN = 1.f / static_cast<float>(aovCount);
                localTile.addAlbedo(tx, ty, accumAlbedo * invN);
                Vec3f avgN = accumNormal * invN;
                float len  = avgN.length();
                if (len > 1e-6f) avgN = avgN * (1.f / len);
                localTile.addNormal(tx, ty, avgN);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Li — path tracing with photon map query at diffuse surface hits.
//
// Identical to PathIntegrator::Li except that after the NEE estimate at each
// diffuse hit, we query m_causticMap to add the caustic contribution.
// ---------------------------------------------------------------------------
Spectrum PhotonMapIntegrator::Li(const Ray& ray, const SceneView& scene,
                                  ISampler& sampler,
                                  Spectrum& outAlbedo, Vec3f& outNormal) const {
    Spectrum L    = {};
    Spectrum beta = {1.f, 1.f, 1.f};
    Ray      r    = ray;

    float prevBsdfPdf  = 0.f;
    Vec3f prevP        = {};
    bool  prevWasDelta = true;
    bool  firstHit     = true;

    for (uint32_t bounce = 0; bounce <= m_maxDepth; ++bounce) {
        TraceResult hit = scene.accel->trace(r);

        if (!hit.hit) {
            Spectrum bg = scene.envLight
                ? scene.envLight->Le({}, {}, r.direction)
                : scene.envRadiance;
            if (!isBlack(bg)) {
                float weight = 1.f;
                if (!prevWasDelta && bounce > 0) {
                    float lpdf = emitterPdf(prevP, r.direction, scene);
                    weight = powerHeuristic(1, prevBsdfPdf, 1, lpdf);
                }
                L += beta * bg * weight;
            }
            break;
        }

        SurfaceInteraction& si = hit.si;

        if (firstHit && m_debugMeshID >= 0
                && static_cast<int32_t>(si.meshID) != m_debugMeshID) {
            outAlbedo = {};
            outNormal = {};
            return {};
        }

        const IMaterial* mat = nullptr;
        if (si.meshID < scene.materials.size())
            mat = scene.materials[si.meshID];
        if (!mat) break;

        Vec3f wo = -r.direction;
        ShadingContext ctx(si, r.direction);

        // Alpha / opacity cutout
        {
            float opacity = mat->evalOpacity(ctx);
            bool passThrough = opacity <= 0.f
                || (opacity < 1.f && sampler.get1D() >= opacity);
            if (passThrough) {
                r = spawnRay(si.p, si.ng, r.direction);
                r.time = ray.time;
                continue;
            }
        }

        if (firstHit) {
            outAlbedo = mat->reflectance(ctx);
            outNormal = si.n;
            firstHit  = false;
        }

        // Emitter Le with MIS weight
        Spectrum Le = mat->Le(ctx, wo);
        if (!isBlack(Le)) {
            float weight = 1.f;
            if (!prevWasDelta && bounce > 0) {
                float lpdf = emitterPdf(prevP, r.direction, scene);
                weight = powerHeuristic(1, prevBsdfPdf, 1, lpdf);
            }
            L += beta * Le * weight;
        }

        // NEE + photon map query at diffuse surfaces
        if (!mat->isDelta()) {
            // Direct illumination via NEE
            if (!scene.lights.empty()) {
                uint32_t N = static_cast<uint32_t>(scene.lights.size());
                uint32_t lightIdx = std::min(
                    static_cast<uint32_t>(sampler.get1D() * static_cast<float>(N)), N - 1);
                Spectrum Ld = estimateDirect(si, *mat, wo,
                                              *scene.lights[lightIdx],
                                              scene, sampler, ray.time);
                L += beta * Ld * static_cast<float>(N);
            }

            // Caustic photon map contribution
            if (!m_causticMap.empty()) {
                Spectrum Lcaustic = m_causticMap.estimateRadiance(
                    si.p, si.n, wo, *mat, ctx, m_searchRadius);
                if (!isBlack(Lcaustic))
                    L += beta * Lcaustic;
            }
        }

        // BSDF sample for path continuation
        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) break;

        prevP        = si.p;
        prevBsdfPdf  = bs.pdf;
        prevWasDelta = bs.isDelta();

        beta *= bs.f / bs.pdf;
        r = spawnRay(si.p, si.n, bs.wi);
        r.time = ray.time;
        r.skipStrandID = si.isCurve ? si.strandID : ~0u;

        if (bounce >= m_minDepth) {
            float q = 1.f - std::min(beta.maxComponent(), 0.95f);
            if (sampler.get1D() < q) break;
            beta *= 1.f / (1.f - q);
        }
    }

    if (m_fireflyClamp > 0.f) {
        float lum = luminance(L);
        if (lum > m_fireflyClamp)
            L = L * (m_fireflyClamp / lum);
    }

    return L;
}

// ---------------------------------------------------------------------------
// estimateDirect — identical to PathIntegrator::estimateDirect.
// ---------------------------------------------------------------------------
Spectrum PhotonMapIntegrator::estimateDirect(const SurfaceInteraction& si,
                                              const IMaterial& mat,
                                              Vec3f wo,
                                              const ILight& light,
                                              const SceneView& scene,
                                              ISampler& sampler,
                                              float sceneTime) const {
    ShadingContext ctx(si, -wo);

    LightSample ls = light.sample(si.p, si.n, sampler.get2D());
    if (ls.pdf <= 0.f || isBlack(ls.Li)) return {};

    BSDFEval be = mat.evaluate(ctx, wo, ls.wi);
    if (isBlack(be.f)) return {};

    Ray shadowRay = spawnRayTo(si.p, si.n, si.p + ls.wi * ls.dist);
    shadowRay.time = sceneTime;
    shadowRay.skipStrandID = si.isCurve ? si.strandID : ~0u;
    Spectrum Tr = shadowTransmittance(shadowRay, scene);
    if (isBlack(Tr)) return {};

    float weight = ls.isDelta
        ? 1.f
        : powerHeuristic(1, ls.pdf, 1, be.pdf);

    float cosI = si.isCurve
        ? std::sqrt(std::max(0.f, 1.f - dot(ls.wi, ctx.t) * dot(ls.wi, ctx.t)))
        : absDot(ls.wi, si.n);

    return be.f * ls.Li * Tr * cosI * weight / ls.pdf;
}

// ---------------------------------------------------------------------------
// emitterPdf — uniform-selection combined solid-angle PDF for MIS.
// ---------------------------------------------------------------------------
float PhotonMapIntegrator::emitterPdf(Vec3f from, Vec3f wi,
                                       const SceneView& scene) const {
    if (scene.lights.empty()) return 0.f;
    float pdf    = 0.f;
    float weight = 1.f / static_cast<float>(scene.lights.size());
    for (const ILight* light : scene.lights) {
        float lpdf = light->pdf(from, wi);
        if (lpdf > 0.f)
            pdf += weight * lpdf;
    }
    return pdf;
}

} // namespace anacapa
