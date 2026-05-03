#include "PathIntegrator.h"
#include "ShadowRay.h"
#include <anacapa/shading/ShadingContext.h>
#include <cmath>

namespace anacapa {

void PathIntegrator::renderTile(const SceneView& scene,
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
            float    sumLumSq    = 0.f;

            for (uint32_t s = 0; s < tile.sampleCount; ++s) {
                sampler.startPixelSample(px, py, tile.sampleStart + s);
                Vec2f jitter = sampler.get2D();
                Vec2f lens   = sampler.get2D();
                float timeU  = sampler.get1D();
                Ray ray = cam.generateRay(px, py, jitter.x, jitter.y, lens.x, lens.y, timeU);

                Spectrum albedo = {};
                Vec3f    normal = {};
                Spectrum sample = Li(ray, scene, sampler, 0, albedo, normal);
                if (sample.isFinite()) {
                    accum += sample;
                    float lum = luminance(sample);
                    sumLumSq += lum * lum;
                }
                accumAlbedo += albedo;
                accumNormal = accumNormal + normal;
                ++aovCount;
            }

            // Weight by sampleCount so adaptive passes merge correctly:
            // Film computes weighted average across all passes per pixel.
            float invSPP = 1.f / static_cast<float>(tile.sampleCount);
            localTile.add(tx, ty, accum * invSPP, static_cast<float>(tile.sampleCount));
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

Spectrum PathIntegrator::Li(const Ray& ray, const SceneView& scene,
                              ISampler& sampler, uint32_t depth,
                              Spectrum& outAlbedo, Vec3f& outNormal) const {
    Spectrum L      = {};
    Spectrum beta   = {1.f, 1.f, 1.f};
    Ray      r      = ray;
    bool     specularBounce = false;
    bool     firstHit       = true;

    for (uint32_t bounce = 0; bounce <= m_maxDepth; ++bounce) {
        TraceResult hit = scene.accel->trace(r);

        if (!hit.hit) {
            if (bounce == 0 || specularBounce) {
                Spectrum bg = scene.envLight
                    ? scene.envLight->Le({}, {}, r.direction)
                    : scene.envRadiance;
                L += beta * bg;
            }
            break;
        }

        SurfaceInteraction& si = hit.si;

        const IMaterial* mat = nullptr;
        if (si.meshID < scene.materials.size())
            mat = scene.materials[si.meshID];
        if (!mat) break;

        Vec3f wo = -r.direction;
        ShadingContext ctx(si, r.direction);

        // Alpha test — if the surface is alpha-masked and this point is in the
        // cutout region, continue the ray past the surface without shading it.
        // Stochastic opacity: compare opacity against a random sample so
        // semi-transparent alpha-masked surfaces (fire, foliage) get a
        // properly anti-aliased soft edge rather than a hard cutout at 0.5.
        // Only consume a sampler dimension for partial alpha to preserve
        // low-discrepancy structure for subsequent bounce decisions.
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

        // Capture first-hit AOVs for denoising
        if (firstHit) {
            outAlbedo = mat->reflectance(ctx);
            outNormal = si.n;
            firstHit  = false;
        }

        Spectrum Le = mat->Le(ctx, wo);
        if (!isBlack(Le)) {
            if (bounce == 0 || specularBounce)
                L += beta * Le;
        }

        if (!mat->isDelta() && !scene.lights.empty()) {
            uint32_t lightIdx = static_cast<uint32_t>(
                sampler.get1D() * static_cast<float>(scene.lights.size()));
            lightIdx = std::min(lightIdx,
                static_cast<uint32_t>(scene.lights.size() - 1));

            Spectrum Ld = estimateDirect(si, *mat, wo,
                                          *scene.lights[lightIdx],
                                          scene, sampler, ray.time);
            L += beta * Ld * static_cast<float>(scene.lights.size());
        }

        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) {
            break;
        }

        specularBounce = bs.isDelta();
        beta *= bs.f / bs.pdf;
        r = spawnRay(si.p, si.n, bs.wi);
        r.time = ray.time;  // freeze scene at the same moment for all bounces

        if (bounce >= m_minDepth) {
            float q = 1.f - std::min(beta.maxComponent(), 0.95f);
            if (sampler.get1D() < q) break;
            beta *= 1.f / (1.f - q);
        }
    }

    return L;
}

Spectrum PathIntegrator::estimateDirect(const SurfaceInteraction& si,
                                         const IMaterial& mat,
                                         Vec3f wo,
                                         const ILight& light,
                                         const SceneView& scene,
                                         ISampler& sampler,
                                         float sceneTime) const {
    Spectrum Ld = {};
    ShadingContext ctx(si, -wo);

    // --- Light sampling ---
    {
        LightSample ls = light.sample(si.p, si.n, sampler.get2D());
        if (ls.pdf > 0.f && !isBlack(ls.Li)) {
            BSDFEval be = mat.evaluate(ctx, wo, ls.wi);
            if (!isBlack(be.f)) {
                Ray shadowRay = spawnRayTo(si.p, si.n, si.p + ls.wi * ls.dist);
                shadowRay.time = sceneTime;
                Spectrum Tr = shadowTransmittance(shadowRay, scene);
                if (!isBlack(Tr)) {
                    float weight = ls.isDelta
                        ? 1.f
                        : powerHeuristic(1, ls.pdf, 1, be.pdf);
                    float cosI = absDot(ls.wi, si.n);
                    Ld += be.f * ls.Li * Tr * cosI * weight / ls.pdf;
                }
            }
        }
    }

    // --- BSDF sampling (skip for delta lights) ---
    if (!light.isDelta()) {
        BSDFSample bs = mat.sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (bs.isValid()) {
            float lightPdf = light.pdf(si.p, bs.wi);
            if (lightPdf > 0.f) {
                float weight = bs.isDelta()
                    ? 1.f
                    : powerHeuristic(1, bs.pdf, 1, lightPdf);

                // Trace toward the light, stepping through any transparent surfaces.
                // We use shadowTransmittance which handles the chain of transparent
                // hits, then fall through to check if the final hit is an emitter.
                // Use geometric normal for offset to clear the actual surface.
                Ray shadowRay = spawnRay(si.p, si.ng, bs.wi);
                shadowRay.tMax = 1e10f;
                shadowRay.time = sceneTime;

                // Find the first non-transmissive surface (emitter or opaque blocker)
                Spectrum Tr = {1.f, 1.f, 1.f};
                TraceResult hit;
                {
                    Ray stepRay = shadowRay;
                    for (int step = 0; step < 8; ++step) {
                        hit = scene.accel->trace(stepRay);
                        if (!hit.hit) break;
                        const IMaterial* m = (hit.si.meshID < scene.materials.size())
                                             ? scene.materials[hit.si.meshID] : nullptr;
                        if (!m) { Tr = {}; break; }
                        ShadingContext hctx(hit.si, stepRay.direction);
                        Spectrum tint = m->transmittanceColor(hctx);
                        if (isBlack(tint)) break;   // opaque / emitter — stop here
                        Tr = Tr * tint;
                        if (isBlack(Tr)) break;
                        float remaining = stepRay.tMax - hit.si.t;
                        if (remaining <= 1e-4f) { hit.hit = false; break; }
                        float t = stepRay.time;
                        stepRay = spawnRay(hit.si.p, hit.si.ng, stepRay.direction);
                        stepRay.tMax = remaining - 1e-4f;
                        stepRay.time = t;
                    }
                }

                Spectrum Li = {};
                if (!hit.hit) {
                    Li = scene.envLight
                        ? scene.envLight->Le({}, {}, bs.wi)
                        : scene.envRadiance;
                } else if (hit.si.meshID < scene.materials.size()) {
                    const IMaterial* hitMat = scene.materials[hit.si.meshID];
                    if (hitMat) {
                        ShadingContext hitCtx(hit.si, -bs.wi);
                        Li = hitMat->Le(hitCtx, -bs.wi);
                    }
                }

                if (!isBlack(Li) && !isBlack(Tr)) {
                    float cosI = absDot(bs.wi, si.n);
                    Ld += bs.f * Li * Tr * cosI * weight / bs.pdf;
                }
            }
        }
    }

    return Ld;
}

} // namespace anacapa
