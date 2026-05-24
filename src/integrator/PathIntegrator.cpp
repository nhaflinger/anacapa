#include "PathIntegrator.h"
#include "ShadowRay.h"
#include "../accel/HaloAccel.h"
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
            float    weightSum   = 0.f;
            float    sumLumSq    = 0.f;

            for (uint32_t s = 0; s < tile.sampleCount; ++s) {
                sampler.startPixelSample(px, py, tile.sampleStart + s);
                Vec2f filt = sampler.get2D();
                Vec2f lens = sampler.get2D();
                float timeU  = sampler.get1D();

                // Pixel-filter importance sampling.  Falls back to a unit
                // box (legacy behaviour) when no filter is bound.
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
                float    alpha  = 1.f;
                Spectrum sample = Li(ray, scene, sampler, albedo, normal, alpha);
                if (sample.isFinite()) {
                    accum     += sample * fw;
                    weightSum += fw;
                    float lum = luminance(sample);
                    sumLumSq += lum * lum;
                }
                accumAlbedo += albedo;
                accumNormal = accumNormal + normal;
                ++aovCount;
                localTile.addAlpha(tx, ty, alpha, fw);
            }

            // Deposit the signed-weighted estimator into the tile.  When
            // weightSum is 0 (extremely rare; only possible with negative-
            // lobe filters cancelling exactly) we drop the pixel for this
            // batch — variance is preserved through subsequent batches.
            if (weightSum != 0.f) {
                localTile.add(tx, ty, accum * (1.f / weightSum), weightSum);
            }
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
                              Spectrum& outAlbedo, Vec3f& outNormal,
                              float& outAlpha) const {
    Spectrum L    = {};
    Spectrum beta = {1.f, 1.f, 1.f};
    Ray      r    = ray;
    outAlpha      = 1.f;  // default: opaque (set to 0 for transparent-bg sky miss)

    // MIS state: track the BSDF PDF of the ray that spawned the current vertex.
    // prevWasDelta=true on the first hit and after any delta bounce so emitter Le
    // gets weight=1 (no NEE was attempted, no double-count risk).
    float prevBsdfPdf  = 0.f;
    Vec3f prevP        = {};
    bool  prevWasDelta = true;
    bool  firstHit     = true;

    uint32_t passthroughTotal = 0;
    for (uint32_t bounce = 0; bounce <= m_maxDepth; ++bounce) {
        TraceResult hit = scene.accel->trace(r);

        // Camera-facing halo disc particles — inner loop drains all halos in
        // front of the nearest mesh hit without consuming the bounce budget.
        if (scene.haloAccel) {
            for (uint32_t hPass = 0; hPass < 256u; ++hPass) {
                Ray haloRay = r;
                if (hit.hit) haloRay.tMax = hit.si.t;
                TraceResult haloHit = scene.haloAccel->trace(haloRay);
                if (!haloHit.hit) break;

                const SurfaceInteraction& hsi = haloHit.si;
                const IMaterial* hmat = nullptr;
                if (hsi.meshID < scene.materials.size())
                    hmat = scene.materials[hsi.meshID];

                if (hmat) {
                    ShadingContext hctx(hsi, r.direction);
                    Vec3f hwo = -r.direction;
                    Spectrum hLe = hmat->Le(hctx, hwo);
                    if (!isBlack(hLe)) L += beta * hLe;
                    BSDFSample bs = hmat->sample(hctx, hwo, {0.f, 0.f}, 0.f);
                    // Use pdf > 0 guard only (not isValid) so opacity=1 → f={0,0,0}
                    // correctly zeroes beta (full absorption is a valid physical state).
                    if (bs.pdf > 0.f) beta *= bs.f / bs.pdf;
                }

                // Advance ray past this halo disc
                Vec3f hitPt = r.origin + r.direction * haloHit.si.t;
                r.origin    = hitPt + r.direction * 1e-4f;
                r.tMin      = 1e-4f;
                prevP        = hitPt;
                prevWasDelta = true;

                if (beta.x < 1e-4f && beta.y < 1e-4f && beta.z < 1e-4f) { bounce = m_maxDepth + 1; break; }
            }
            if (bounce > m_maxDepth) break;
        }

        // Retrace mesh BVH from updated ray origin (halo inner loop may have moved r).
        hit = scene.accel->trace(r);

        if (!hit.hit) {
            // Background / environment light
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
            // Alpha=0 for background pixels so renders are always compositable.
            if (bounce == 0)
                outAlpha = 0.f;
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

        // Highlight mode: shade target mesh with flat red, others at 20% brightness
        if (m_highlightMeshID >= 0 && firstHit) {
            if (static_cast<int32_t>(si.meshID) == m_highlightMeshID) {
                outAlbedo = {1.f, 0.f, 0.f};
                outNormal = si.n;
                return {1.f, 0.f, 0.f};
            }
            // dim everything else so the highlight stands out
            beta *= 0.2f;
        }

        // Alpha / opacity cutout — cap passthroughs to avoid near-infinite loops
        // when instanced geometry has missing/zero-opacity materials.
        {
            float opacity = mat->evalOpacity(ctx);
            bool passThrough = (passthroughTotal < 64u)
                && (opacity <= 0.f || (opacity < 1.f && sampler.get1D() >= opacity));
            if (passThrough) {
                ++passthroughTotal;
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
        // those cases get weight=1 (no NEE was attempted, no double-count risk).
        Spectrum Le = mat->Le(ctx, wo);
        if (!isBlack(Le)) {
            float weight = 1.f;
            if (!prevWasDelta && bounce > 0) {
                float lpdf = emitterPdf(prevP, r.direction, scene);
                weight = powerHeuristic(1, prevBsdfPdf, 1, lpdf);
            }
            L += beta * Le * weight;
        }

        // NEE: uniform random light selection (includes env/dome light when present)
        {
            uint32_t N = static_cast<uint32_t>(scene.lights.size())
                       + (scene.envLight ? 1u : 0u);
            if (!mat->isDelta() && N > 0) {
                uint32_t lightIdx = std::min(
                    static_cast<uint32_t>(sampler.get1D() * static_cast<float>(N)), N - 1);
                const ILight* light = (scene.envLight && lightIdx >= scene.lights.size())
                    ? scene.envLight
                    : scene.lights[lightIdx];
                Spectrum Ld = estimateDirect(si, *mat, wo, *light,
                                              scene, sampler, ray.time);
                L += beta * Ld * static_cast<float>(N);
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

    if (m_fireflyClamp > 0.f) {
        float lum = luminance(L);
        if (lum > m_fireflyClamp)
            L = L * (m_fireflyClamp / lum);
    }

    return L;
}

// ---------------------------------------------------------------------------
// estimateDirect — light-sampling branch only.
//
// The BSDF-sampling branch has been removed: emitter hits via path-continuation
// are handled in Li with proper per-emitter MIS weights, which correctly covers
// all emitters across all bounce depths (not just the immediately adjacent hit).
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
// emitterPdf — uniform-selection combined solid-angle PDF for MIS.
//
// Returns (1/N) × sum of each light's solid-angle PDF for direction wi from
// point `from`.  This matches the NEE strategy used in Li: pick one light
// uniformly (prob 1/N), sample it (PDF ls.pdf).  The combined selection PDF
// is (1/N) × ls.pdf, which is what the MIS denominator needs.
// ---------------------------------------------------------------------------
float PathIntegrator::emitterPdf(Vec3f from, Vec3f wi,
                                   const SceneView& scene) const {
    uint32_t N = static_cast<uint32_t>(scene.lights.size())
               + (scene.envLight ? 1u : 0u);
    if (N == 0) return 0.f;
    float pdf    = 0.f;
    float weight = 1.f / static_cast<float>(N);
    for (const ILight* light : scene.lights) {
        float lpdf = light->pdf(from, wi);
        if (lpdf > 0.f)
            pdf += weight * lpdf;
    }
    if (scene.envLight) {
        float lpdf = scene.envLight->pdf(from, wi);
        if (lpdf > 0.f)
            pdf += weight * lpdf;
    }
    return pdf;
}

} // namespace anacapa
