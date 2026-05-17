#include "PhotonMapIntegrator.h"
#include "ShadowRay.h"
#include <anacapa/shading/ShadingContext.h>
#include "../shading/Lambertian.h"
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

    // The caustic photon map is only useful when at least one material is
    // flagged as a caustic generator (inputs:anacapa_caustic = true).  If
    // nothing in the scene is opted in, the photon pass will emit photons,
    // skip every specular bounce as non-caustic, and store zero — wasting
    // time and giving the user no visible benefit over --integrator path.
    bool anyCaustic = false;
    bool anySSS     = false;
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        const IMaterial* mat = scene.materials[i];
        if (!mat) continue;
        if (mat->isCausticGenerator()) anyCaustic = true;
        auto sss = mat->subsurfaceParams();
        if (sss.weight > 0.f) {
            anySSS = true;
            spdlog::info("  mat[{}]: subsurface weight={:.3f} radius={:.4f}",
                         i, sss.weight, sss.radius);
        }
    }
    if (anyCaustic)
        spdlog::info("Photon map: caustic generators present — caustic photons will be traced.");
    if (anySSS)
        spdlog::info("Photon map: subsurface scattering enabled — SSS photons will be traced.");
    if (!anyCaustic && !anySSS)
        spdlog::info("Photon map: no caustic generators or SSS materials — "
                     "photon pass will build empty maps (consider --integrator path).");

    if (!scene.lights.empty())
        tracePhotons(scene);
}

// ---------------------------------------------------------------------------
// tracePhotons — emit photons from lights, trace through the scene and
// populate both the caustic map and the SSS map.
//
// Caustic photons: stored at the first diffuse hit after ≥1 specular bounce
// through a caustic-flagged surface (unchanged from the original behaviour).
//
// SSS photons: stored at every hit on a subsurface-enabled surface regardless
// of bounce history.  After depositing, the photon continues with a Lambertian
// bounce so that multi-bounce illumination still reaches SSS surfaces.
// ---------------------------------------------------------------------------
void PhotonMapIntegrator::tracePhotons(const SceneView& scene) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> u01(0.f, 1.f);

    auto rand1 = [&]() -> float { return u01(rng); };
    auto rand2 = [&]() -> Vec2f { return {u01(rng), u01(rng)}; };

    std::vector<Photon> causticPhotons;
    std::vector<Photon> sssPhotons;
    causticPhotons.reserve(static_cast<size_t>(m_numPhotons) / 4);
    sssPhotons.reserve(static_cast<size_t>(m_numPhotons) / 2);

    // Shutter interval for motion blur.  When both are 0 there is no motion
    // blur and all photons trace at t=0 (matching the static scene geometry).
    float shutterOpen  = 0.f;
    float shutterClose = 0.f;
    if (scene.camera) {
        shutterOpen  = scene.camera->shutterOpen;
        shutterClose = scene.camera->shutterClose;
    }
    bool motionBlur = (shutterClose > shutterOpen);

    for (int emitted = 0; emitted < m_numPhotons; ++emitted) {
        if (m_lightSampler.empty()) break;
        auto sel = m_lightSampler.sample(rand1());
        if (!sel.light || sel.pdf <= 0.f) continue;

        LightLeSample les = sel.light->sampleLe(rand2(), rand2());
        if (les.pdfPos <= 0.f || les.pdfDir <= 0.f || isBlack(les.Le)) continue;

        float cosTheta = std::abs(dot(les.dir, les.normal));
        if (cosTheta <= 0.f) continue;

        Spectrum power = les.Le * cosTheta
                       / (sel.pdf * les.pdfPos * les.pdfDir
                          * static_cast<float>(m_numPhotons));

        if (!power.isFinite() || isBlack(power)) continue;

        // Sample a shutter time for this photon path.  Distributing photons
        // uniformly across the shutter window means the photon map represents
        // time-averaged geometry, matching the statistical distribution of
        // camera rays that also sample random times in [shutterOpen, shutterClose].
        float photonTime = motionBlur
                         ? shutterOpen + rand1() * (shutterClose - shutterOpen)
                         : 0.f;

        Ray photonRay{les.pos, les.dir};
        photonRay.tMin = 1e-4f;
        photonRay.time = photonTime;

        int numSpecular = 0;

        for (int bounce = 0; bounce < static_cast<int>(m_maxDepth); ++bounce) {
            TraceResult hit = scene.accel->trace(photonRay);
            if (!hit.hit) break;

            const SurfaceInteraction& si = hit.si;
            const IMaterial* mat = (si.meshID < scene.materials.size())
                                   ? scene.materials[si.meshID] : nullptr;
            if (!mat) break;

            ShadingContext ctx(si, photonRay.direction);

            float opacity = mat->evalOpacity(ctx);
            if (opacity <= 0.f) {
                photonRay = spawnRay(si.p, si.ng, photonRay.direction);
                photonRay.time = photonTime;
                --bounce;
                continue;
            }

            Vec3f wo = -photonRay.direction;

            if (mat->isDelta()) {
                BSDFSample bs = mat->sample(ctx, wo, rand2(), rand1());
                if (!bs.isValid() || bs.pdf <= 0.f) break;

                power = power * bs.f / bs.pdf;
                if (!power.isFinite() || isBlack(power)) break;

                photonRay = spawnRay(si.p, si.n, bs.wi);
                photonRay.time = photonTime;
                photonRay.skipStrandID = si.isCurve ? si.strandID : ~0u;
                if (mat->isCausticGenerator())
                    ++numSpecular;
            } else if (mat->isSubsurface()) {
                // Deposit the absorbed fraction as an SSS photon, then
                // continue the photon path with the transmitted fraction.
                // This allows indirect photons to reach shadowed areas
                // (inner folds, lip edges) that receive no direct photons.
                //
                // Absorbed fraction = entrycos * sss.weight.
                // Transmitted fraction = 1 - absorbed.
                // Power is conserved: deposit + continue = original power.
                auto sss = mat->subsurfaceParams();
                float entrycos = std::max(0.f, -dot(photonRay.direction, si.n));
                float absorbed = entrycos * std::max(0.f, sss.weight);

                if (absorbed > 0.f) {
                    Photon ph;
                    ph.position = si.p;
                    ph.normal   = si.n;
                    ph.wi       = photonRay.direction;
                    ph.power    = power * absorbed;
                    ph.axis     = 0;
                    sssPhotons.push_back(ph);
                }

                // Continue with remaining energy via diffuse bounce so that
                // indirect photons reach geometrically shadowed SSS regions.
                float transmitted = 1.f - absorbed;
                power = power * transmitted;
                if (isBlack(power) || !power.isFinite()) break;

                ShadingContext ctx(si, photonRay.direction);
                Vec3f wiLocal = sampleCosineHemisphere(rand2());
                photonRay = spawnRay(si.p, si.n, ctx.toWorld(wiLocal));
                photonRay.time = photonTime;
                numSpecular = 0;
            } else {
                // Regular diffuse: store caustic photon if the path came through specular.
                if (numSpecular > 0) {
                    Photon ph;
                    ph.position = si.p;
                    ph.wi       = photonRay.direction;
                    ph.power    = power;
                    ph.axis     = 0;
                    causticPhotons.push_back(ph);
                }
                break;
            }

            if (bounce >= 2) {
                float survival = std::min(luminance(power) * 10.f, 0.95f);
                if (rand1() > survival) break;
                power = power * (1.f / survival);
            }
        }
    }

    spdlog::info("PhotonMap: emitted {} photons, stored {} caustic, {} SSS",
                 m_numPhotons, causticPhotons.size(), sssPhotons.size());
    m_causticMap.build(std::move(causticPhotons));
    m_sssMap.build(std::move(sssPhotons));
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
    // Tracks whether the current delta chain (since the last non-delta vertex)
    // has crossed at least one caustic-flagged surface.  When true, an emitter
    // hit at the end of the chain is already counted by the photon density
    // estimate that fired at the previous diffuse vertex — adding Le here too
    // would double-count, so we suppress it.
    bool  causticChain = false;

    for (uint32_t bounce = 0; bounce <= m_maxDepth; ++bounce) {
        TraceResult hit = scene.accel->trace(r);

        if (!hit.hit) {
            Spectrum bg = scene.envLight
                ? scene.envLight->Le({}, {}, r.direction)
                : scene.envRadiance;
            if (!isBlack(bg) && !causticChain) {
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

        // Emitter Le with MIS weight.  Suppressed if the path reached this
        // vertex through a caustic-flagged delta chain — the photon density
        // estimate at the previous diffuse vertex already counted it.
        Spectrum Le = mat->Le(ctx, wo);
        if (!isBlack(Le) && !causticChain) {
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
                float cr = mat->causticRadius() > 0.f ? mat->causticRadius() : m_searchRadius;
                Spectrum Lcaustic = m_causticMap.estimateRadiance(
                    si.p, si.n, wo, *mat, ctx, cr);
                if (!isBlack(Lcaustic))
                    L += beta * Lcaustic;
            }

            // SSS photon map contribution — replaces the diffuse component
            // in proportion to the subsurface weight (which already reduced
            // wDiff in the BSDF so the two don't double-count).
            if (!m_sssMap.empty()) {
                auto sss = mat->subsurfaceParams();
                if (sss.weight > 0.f) {
                    float d = sss.radius * sss.scale;
                    Spectrum Lsss = m_sssMap.estimateSSSRadiance(
                        si.p, si.n, sss.color, d);
                    if (!isBlack(Lsss)) {
                        // Thickness attenuation disabled — visually preferred without it.
                        // float thicknessScale = 1.f;
                        // if (d > 0.f) {
                        //     Ray inwardRay = spawnRay(si.p, si.n, -si.n);
                        //     inwardRay.time = ray.time;
                        //     TraceResult thickHit = scene.accel->trace(inwardRay);
                        //     if (thickHit.hit)
                        //         thicknessScale = std::exp(-thickHit.si.t / (d * 30.f));
                        // }
                        // L += beta * Lsss * sss.weight * sss.strength * thicknessScale;
                        L += beta * Lsss * sss.weight * sss.strength;
                    }
                }
            }
        }

        // BSDF sample for path continuation
        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) break;

        prevP        = si.p;
        prevBsdfPdf  = bs.pdf;
        prevWasDelta = bs.isDelta();
        // Maintain the "delta chain crossed a caustic-flagged surface" flag.
        // A non-delta bounce resets it (the next chain starts fresh from this
        // diffuse vertex, whose photon density estimate already fired above).
        if (!bs.isDelta())
            causticChain = false;
        else if (mat->isCausticGenerator())
            causticChain = true;

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
    Spectrum Tr = shadowTransmittance(shadowRay, scene, /*deltaOpaque=*/true);
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
