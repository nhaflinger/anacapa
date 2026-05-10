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
                Spectrum sample = Li(ray, scene, sampler, albedo, normal);
                if (sample.isFinite()) {
                    accum += sample;
                    float lum = luminance(sample);
                    sumLumSq += lum * lum;
                }
                accumAlbedo += albedo;
                accumNormal = accumNormal + normal;
                ++aovCount;
            }

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
                              ISampler& sampler,
                              Spectrum& outAlbedo, Vec3f& outNormal) const {
    Spectrum L    = {};
    Spectrum beta = {1.f, 1.f, 1.f};
    Ray      r    = ray;

    // MIS state for emitter hits reached via path-continuation BSDF samples.
    // prevBsdfPdf  — solid-angle PDF of the BSDF sample that spawned the current ray.
    // prevWasDelta — true when the previous bounce was specular/delta; emitter Le
    //                then gets full weight (no NEE contribution to cancel against).
    // prevP        — world-space position of the previous surface (for light PDF query).
    float prevBsdfPdf  = 0.f;
    Vec3f prevP        = {};
    bool  prevWasDelta = true;   // first hit: treat as delta so Le weight = 1
    bool  firstHit     = true;

    for (uint32_t bounce = 0; bounce <= m_maxDepth; ++bounce) {
        TraceResult hit = scene.accel->trace(r);

        if (!hit.hit) {
            // Background / environment light
            Spectrum bg = scene.envLight
                ? scene.envLight->Le({}, {}, r.direction)
                : scene.envRadiance;
            if (!isBlack(bg)) {
                // emitterPdf returns 0 for directions with no area-light coverage,
                // so powerHeuristic(..., 0) = 1 — env always gets full weight unless
                // the DomeLight is in scene.lights and covers this direction.
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

        // Emitter Le — add with MIS weight against the NEE light-sampling PDF.
        // prevWasDelta is true on the first hit and after any delta bounce, so
        // those cases get weight 1 (no NEE was attempted, no double-count risk).
        Spectrum Le = mat->Le(ctx, wo);
        if (!isBlack(Le)) {
            float weight = 1.f;
            if (!prevWasDelta && bounce > 0) {
                float lpdf = emitterPdf(prevP, r.direction, scene);
                weight = powerHeuristic(1, prevBsdfPdf, 1, lpdf);
            }
            L += beta * Le * weight;
        }

        // NEE: sample one light via the power-weighted alias table.
        if (!mat->isDelta() && !m_lightSampler.empty()) {
            LightSampler::Selection sel = m_lightSampler.sample(sampler.get1D());
            if (sel.light && sel.pdf > 0.f) {
                Spectrum Ld = estimateDirect(si, *mat, wo, *sel.light,
                                              scene, sampler, ray.time);
                L += beta * Ld / sel.pdf;
            }
        }

        // BSDF sample for path continuation
        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) break;

        // Update MIS tracking before moving to the next vertex
        prevP         = si.p;
        prevBsdfPdf   = bs.pdf;
        prevWasDelta  = bs.isDelta();

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

    return L;
}

// ---------------------------------------------------------------------------
// estimateDirect — light-sampling branch only.
//
// The BSDF-sampling branch has been removed: emitter hits via path-continuation
// are now handled in Li with proper per-emitter MIS weights, which correctly
// covers all lights (not just the currently-selected one).
// ---------------------------------------------------------------------------
Spectrum PathIntegrator::estimateDirect(const SurfaceInteraction& si,
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

    // Power-heuristic MIS weight: balance light-sampling PDF against BSDF PDF.
    // Delta lights (point/directional) don't have a competing BSDF strategy.
    float weight = ls.isDelta
        ? 1.f
        : powerHeuristic(1, ls.pdf, 1, be.pdf);

    float cosI = si.isCurve
        ? std::sqrt(std::max(0.f, 1.f - dot(ls.wi, ctx.t) * dot(ls.wi, ctx.t)))
        : absDot(ls.wi, si.n);

    return be.f * ls.Li * Tr * cosI * weight / ls.pdf;
}

// ---------------------------------------------------------------------------
// emitterPdf — combined selection + solid-angle PDF for MIS.
//
// Returns the probability density (in solid angle measure at `from`) of
// reaching direction `wi` via the NEE light sampler.  Sums over all lights
// weighted by their alias-table selection probability so the result is the
// correct denominator for the MIS weight when a path-continuation ray hits
// an emitter.
// ---------------------------------------------------------------------------
float PathIntegrator::emitterPdf(Vec3f from, Vec3f wi,
                                   const SceneView& scene) const {
    float pdf = 0.f;
    for (uint32_t i = 0; i < static_cast<uint32_t>(scene.lights.size()); ++i) {
        float lpdf = scene.lights[i]->pdf(from, wi);
        if (lpdf > 0.f)
            pdf += m_lightSampler.pdf(i) * lpdf;
    }
    return pdf;
}

} // namespace anacapa
