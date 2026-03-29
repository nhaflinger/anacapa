#include "PathIntegrator.h"
#include <anacapa/shading/ShadingContext.h>
#include <cmath>

namespace anacapa {

void PathIntegrator::renderTile(const SceneView& scene,
                                 const TileRequest& tile,
                                 uint32_t filmWidth,
                                 uint32_t filmHeight,
                                 ISampler& sampler,
                                 TileBuffer& localTile) {
    // Rebuild a simple pinhole camera from film dimensions.
    // In Phase 3 this will come from the USD scene.
    // Camera is outside the box opening (z < 0), looking into the interior.
    // Box spans x∈[-1,1], y∈[-1,1], z∈[0,2] with open face at z=0.
    Camera cam = Camera::makePinhole(
        {0.f, 0.f, -2.5f},  // from — outside in front of the opening
        {0.f, 0.f,  1.f},   // at   — center of the box interior
        {0.f, 1.f,  0.f},   // up
        50.f,
        filmWidth, filmHeight
    );

    for (uint32_t ty = 0; ty < tile.height; ++ty) {
        for (uint32_t tx = 0; tx < tile.width; ++tx) {
            uint32_t px = tile.x0 + tx;
            uint32_t py = tile.y0 + ty;

            Spectrum accum = {};
            for (uint32_t s = 0; s < tile.sampleCount; ++s) {
                sampler.startPixelSample(px, py, tile.sampleStart + s);
                Vec2f jitter = sampler.get2D();
                Ray ray = cam.generateRay(px, py, jitter.x, jitter.y);
                accum += Li(ray, scene, sampler, 0);
            }

            float invSPP = 1.f / static_cast<float>(tile.sampleCount);
            localTile.add(tx, ty, accum * invSPP);
        }
    }
}

Spectrum PathIntegrator::Li(const Ray& ray, const SceneView& scene,
                              ISampler& sampler, uint32_t depth) const {
    Spectrum L      = {};
    Spectrum beta   = {1.f, 1.f, 1.f};
    Ray      r      = ray;
    bool     specularBounce = false;

    for (uint32_t bounce = 0; bounce <= m_maxDepth; ++bounce) {
        TraceResult hit = scene.accel->trace(r);

        if (!hit.hit) {
            // Miss — add background radiance
            // For specular bounces or first bounce, include it directly;
            // otherwise it was already accounted for in estimateDirect.
            if (bounce == 0 || specularBounce)
                L += beta * scene.envRadiance;
            break;
        }

        SurfaceInteraction& si = hit.si;

        // Resolve material
        const IMaterial* mat = nullptr;
        if (si.meshID < scene.materials.size())
            mat = scene.materials[si.meshID];
        if (!mat) break;

        Vec3f wo = -r.direction;
        ShadingContext ctx(si, r.direction);

        // Add emitted radiance (emissive surfaces)
        Spectrum Le = mat->Le(ctx, wo);
        if (!isBlack(Le)) {
            // Include only on first bounce or after specular — otherwise
            // the direct lighting estimator already sampled it.
            if (bounce == 0 || specularBounce)
                L += beta * Le;
        }

        // Direct lighting — one light selected uniformly at random
        if (!mat->isDelta() && !scene.lights.empty()) {
            uint32_t lightIdx = static_cast<uint32_t>(
                sampler.get1D() * static_cast<float>(scene.lights.size()));
            lightIdx = std::min(lightIdx,
                static_cast<uint32_t>(scene.lights.size() - 1));

            Spectrum Ld = estimateDirect(si, *mat, wo,
                                          *scene.lights[lightIdx],
                                          scene, sampler);
            L += beta * Ld * static_cast<float>(scene.lights.size());
        }

        // Sample BSDF for next direction
        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) break;

        specularBounce = bs.isDelta();
        beta *= bs.f / bs.pdf;
        r = spawnRay(si.p, si.n, bs.wi);

        // Russian roulette after minDepth bounces
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
                                         ISampler& sampler) const {
    Spectrum Ld = {};
    ShadingContext ctx(si, -wo);

    // --- Light sampling ---
    {
        LightSample ls = light.sample(si.p, si.n, sampler.get2D());
        if (ls.pdf > 0.f && !isBlack(ls.Li)) {
            BSDFEval be = mat.evaluate(ctx, wo, ls.wi);
            if (!isBlack(be.f)) {
                // Visibility test
                Ray shadowRay = spawnRayTo(si.p, si.n, si.p + ls.wi * ls.dist);
                if (!scene.accel->occluded(shadowRay)) {
                    float weight = ls.isDelta
                        ? 1.f
                        : powerHeuristic(1, ls.pdf, 1, be.pdf);
                    float cosI   = absDot(ls.wi, si.n);
                    Ld += be.f * ls.Li * cosI * weight / ls.pdf;
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

                Ray shadowRay = spawnRay(si.p, si.n, bs.wi);
                shadowRay.tMax = 1e10f;
                TraceResult hit = scene.accel->trace(shadowRay);

                Spectrum Li = {};
                if (!hit.hit) {
                    Li = scene.envRadiance;
                } else if (hit.si.meshID < scene.materials.size()) {
                    const IMaterial* hitMat = scene.materials[hit.si.meshID];
                    if (hitMat) {
                        ShadingContext hitCtx(hit.si, -bs.wi);
                        Li = hitMat->Le(hitCtx, -bs.wi);
                    }
                }

                if (!isBlack(Li)) {
                    float cosI = absDot(bs.wi, si.n);
                    Ld += bs.f * Li * cosI * weight / bs.pdf;
                }
            }
        }
    }

    return Ld;
}

} // namespace anacapa
