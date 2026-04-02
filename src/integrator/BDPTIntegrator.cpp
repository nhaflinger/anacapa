#include "BDPTIntegrator.h"
#include <anacapa/film/Film.h>
#include <anacapa/shading/ShadingContext.h>
#include "../shading/Lambertian.h"   // kPi / kInvPi constants
#include <cmath>
#include <algorithm>

namespace anacapa {

// ---------------------------------------------------------------------------
// prepare
// ---------------------------------------------------------------------------
void BDPTIntegrator::prepare(const SceneView& scene) {
    m_lightSampler.build(scene.lights);

    // Camera stored for (s,1) pixel projection; overridden per tile if USD camera present
    m_camera = scene.camera.value_or(Camera::makePinhole(
        {0.f, 0.f, -2.5f}, {0.f, 0.f, 1.f}, {0.f, 1.f, 0.f},
        50.f, 800, 800));
}

// ---------------------------------------------------------------------------
// Geometry term  G(a,b) = |cos θ_a| |cos θ_b| / |ab|²
// ---------------------------------------------------------------------------
float BDPTIntegrator::geometryTerm(Vec3f posA, Vec3f nA,
                                    Vec3f posB, Vec3f nB) {
    Vec3f d = posB - posA;
    float dist2 = d.lengthSq();
    if (dist2 < 1e-10f) return 0.f;
    Vec3f dn = d * (1.f / std::sqrt(dist2));
    float cosA = std::abs(dot(dn,  nA));
    float cosB = std::abs(dot(-dn, nB));
    return cosA * cosB / dist2;
}

// ---------------------------------------------------------------------------
// evalVertex — evaluate f(wo, wi) * |cosTheta| at surface vertex i,
// given the direction `toNext` pointing toward the next vertex.
// Also fills forward and reverse solid-angle PDFs.
// ---------------------------------------------------------------------------
Spectrum BDPTIntegrator::evalVertex(const PathVertexBuffer& path, uint32_t i,
                                     Vec3f toNext,
                                     float& fwdPdf, float& revPdf) {
    fwdPdf = revPdf = 0.f;
    if (path.type(i) != PathVertexType::Surface) return {};
    const IMaterial* mat = path.material[i];
    if (!mat) return {};

    // Reconstruct ShadingContext from stored position/normal
    SurfaceInteraction si;
    si.p  = path.position[i];
    si.n  = path.normal[i];
    si.ng = path.normal[i];
    ShadingContext ctx(si, -path.wo[i]);

    Vec3f wo = path.wo[i];
    Vec3f wi = toNext;
    BSDFEval be = mat->evaluate(ctx, wo, wi);
    fwdPdf = be.pdf;
    revPdf = be.pdfRev;

    float cosI = std::abs(dot(wi, path.normal[i]));
    return be.f * cosI;
}

// ---------------------------------------------------------------------------
// traceCameraSubpath
// ---------------------------------------------------------------------------
uint32_t BDPTIntegrator::traceCameraSubpath(const SceneView& scene,
                                              const Ray& primaryRay,
                                              float primaryPdfA,
                                              ISampler& sampler,
                                              PathVertexBuffer& path) const {
    path.reset();
    if (path.full()) return 0;

    // Vertex 0: camera endpoint
    uint32_t vi = path.count++;
    path.position[vi] = primaryRay.origin;
    path.normal[vi]   = -primaryRay.direction;  // "normal" toward scene
    path.wo[vi]       = -primaryRay.direction;
    path.beta[vi]     = {1.f, 1.f, 1.f};
    path.Le[vi]       = {};
    path.pdfFwd[vi]   = primaryPdfA;
    path.pdfRev[vi]   = 0.f;
    path.flags[vi]    = static_cast<uint32_t>(PathVertexType::Camera);
    path.meshID[vi]   = ~0u;

    Spectrum beta     = {1.f, 1.f, 1.f};
    Ray      ray      = primaryRay;
    float    pdfFwd   = primaryPdfA;

    for (uint32_t depth = 0; depth < m_maxDepth && !path.full(); ++depth) {
        TraceResult hit = scene.accel->trace(ray);
        if (!hit.hit) {
            // Record an environment vertex so (s=0, t) strategies capture env light
            if (scene.envLight && !path.full()) {
                Vec3f rayDir = ray.direction;
                Spectrum Le  = scene.envLight->Le({}, {}, -rayDir);
                uint32_t ei  = path.count++;
                path.position[ei] = ray.origin + rayDir * 1e6f;
                path.normal[ei]   = -rayDir;
                path.wo[ei]       = -rayDir;
                path.beta[ei]     = beta;
                path.Le[ei]       = Le;
                path.pdfFwd[ei]   = 0.f;   // infinite light: area PDF undefined
                path.pdfRev[ei]   = 0.f;
                path.flags[ei]    = static_cast<uint32_t>(PathVertexType::Light)
                                  | kVertexInfiniteBit;
                path.meshID[ei]   = ~0u;
                path.light[ei]    = scene.envLight;
            }
            break;
        }

        SurfaceInteraction& si = hit.si;
        const IMaterial* mat = (si.meshID < scene.materials.size())
                               ? scene.materials[si.meshID] : nullptr;
        if (!mat) break;

        Vec3f wo = -ray.direction;
        ShadingContext ctx(si, ray.direction);

        // Convert solid-angle PDF of previous direction to area PDF at this vertex
        float pdfFwdArea = convertToArea(pdfFwd,
            path.position[path.count - 1], si.p, si.n);

        vi = path.count++;
        path.position[vi] = si.p;
        path.normal[vi]   = si.n;
        path.wo[vi]       = wo;
        path.beta[vi]     = beta;
        path.Le[vi]       = mat->Le(ctx, wo);
        path.pdfFwd[vi]   = pdfFwdArea;
        path.pdfRev[vi]   = 0.f;  // filled in by the previous vertex below
        path.flags[vi]    = static_cast<uint32_t>(PathVertexType::Surface)
                          | (mat->isDelta() ? kVertexDeltaBit : 0u);
        path.meshID[vi]   = si.meshID;
        path.material[vi] = mat;

        // Fill pdfRev on the previous vertex (reverse PDF from this vertex back)
        // We do this now because we have both ctx and wo
        if (vi > 0) {
            float dummy, revSA;
            evalVertex(path, vi, -wo, dummy, revSA);
            // pdfRev of previous vertex in area measure
            path.pdfRev[vi - 1] = convertToArea(revSA,
                si.p, path.position[vi - 1], path.normal[vi - 1]);
        }

        // Russian roulette
        if (depth >= 2) {
            float q = 1.f - std::min(beta.maxComponent(), 0.95f);
            if (sampler.get1D() < q) break;
            beta *= 1.f / (1.f - q);
        }

        // Sample next direction
        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) break;

        beta   *= bs.f / bs.pdf;
        pdfFwd  = bs.pdf;
        ray     = spawnRay(si.p, si.n, bs.wi);
    }

    return path.count;
}

// ---------------------------------------------------------------------------
// traceLightSubpath
// ---------------------------------------------------------------------------
uint32_t BDPTIntegrator::traceLightSubpath(const SceneView& scene,
                                             uint32_t lightIdx,
                                             ISampler& sampler,
                                             PathVertexBuffer& path) const {
    path.reset();
    if (path.full() || m_lightSampler.empty()) return 0;

    const ILight* light    = scene.lights[lightIdx];
    float         lightPdf = m_lightSampler.pdf(lightIdx);
    if (lightPdf <= 0.f) return 0;

    // Sample a ray from the light
    LightLeSample ls = light->sampleLe(sampler.get2D(), sampler.get2D());
    if (ls.pdfPos <= 0.f || ls.pdfDir <= 0.f) return 0;

    Spectrum beta = ls.Le * std::abs(dot(ls.dir, ls.normal))
                  / (lightPdf * ls.pdfPos * ls.pdfDir);

    // Vertex 0: light endpoint
    uint32_t vi  = path.count++;
    path.position[vi] = ls.pos;
    path.normal[vi]   = ls.normal;
    path.wo[vi]       = ls.dir;   // direction light ray travels
    path.beta[vi]     = ls.Le / (lightPdf * ls.pdfPos);
    path.Le[vi]       = ls.Le;
    path.pdfFwd[vi]   = lightPdf * ls.pdfPos;  // area PDF on the light surface
    path.pdfRev[vi]   = 0.f;
    path.flags[vi]    = static_cast<uint32_t>(PathVertexType::Light)
                      | (light->isDelta() ? kVertexDeltaBit : 0u)
                      | (light->isInfinite() ? kVertexInfiniteBit : 0u);
    path.meshID[vi]   = ~0u;
    path.light[vi]    = light;

    Ray   ray    = Ray{ls.pos, ls.dir};
    ray.tMin     = 1e-4f;
    float pdfFwd = ls.pdfDir;   // directional PDF (solid angle from light surface)

    for (uint32_t depth = 0; depth < m_maxDepth && !path.full(); ++depth) {
        TraceResult hit = scene.accel->trace(ray);
        if (!hit.hit) break;

        SurfaceInteraction& si = hit.si;
        const IMaterial* mat = (si.meshID < scene.materials.size())
                               ? scene.materials[si.meshID] : nullptr;
        if (!mat) break;

        Vec3f wo = -ray.direction;
        ShadingContext ctx(si, ray.direction);

        float pdfFwdArea = convertToArea(pdfFwd,
            path.position[path.count - 1], si.p, si.n);

        vi = path.count++;
        path.position[vi] = si.p;
        path.normal[vi]   = si.n;
        path.wo[vi]       = wo;
        path.beta[vi]     = beta;
        path.Le[vi]       = {};
        path.pdfFwd[vi]   = pdfFwdArea;
        path.pdfRev[vi]   = 0.f;
        path.flags[vi]    = static_cast<uint32_t>(PathVertexType::Surface)
                          | (mat->isDelta() ? kVertexDeltaBit : 0u);
        path.meshID[vi]   = si.meshID;
        path.material[vi] = mat;

        if (vi > 0) {
            float dummy, revSA;
            evalVertex(path, vi, -wo, dummy, revSA);
            path.pdfRev[vi - 1] = convertToArea(revSA,
                si.p, path.position[vi - 1], path.normal[vi - 1]);
        }

        if (depth >= 2) {
            float q = 1.f - std::min(beta.maxComponent(), 0.95f);
            if (sampler.get1D() < q) break;
            beta *= 1.f / (1.f - q);
        }

        BSDFSample bs = mat->sample(ctx, wo, sampler.get2D(), sampler.get1D());
        if (!bs.isValid()) break;

        beta   *= bs.f / bs.pdf;
        pdfFwd  = bs.pdf;
        ray     = spawnRay(si.p, si.n, bs.wi);
    }

    return path.count;
}

// ---------------------------------------------------------------------------
// connect — evaluate the contribution of strategy (s, t)
// ---------------------------------------------------------------------------
Spectrum BDPTIntegrator::connect(const SceneView& scene,
                                   const PathVertexBuffer& lp, uint32_t s,
                                   const PathVertexBuffer& cp, uint32_t t,
                                   Film* film,
                                   uint32_t filmWidth, uint32_t filmHeight,
                                   const Camera& cam) const {
    Spectrum L = {};

    if (t == 0) {
        // Strategy (s, 0): light subpath hits the camera lens — not implemented
        // for a pinhole camera (measure zero). Skip.
        return {};
    }

    if (s == 0) {
        // Strategy (0, t): pure camera path — emitted radiance at the last vertex.
        // Handles both emissive surfaces and environment (infinite) vertices.
        if (t < 2) return {};
        const uint32_t last = t - 1;
        PathVertexType ty = cp.type(last);
        if (ty == PathVertexType::Surface || ty == PathVertexType::Light)
            L = cp.beta[last] * cp.Le[last];
        return L;
    }

    if (s == 1) {
        // Strategy (1, t): sample a light toward the camera vertex cp[t-1].
        // This is next-event estimation from the camera side.
        if (t < 1) return {};
        const uint32_t ct = t - 1;
        if (!cp.isConnectible(ct)) return {};
        if (!cp.material[ct]) return {};

        // lp[0] is a point on the light — reconnect it toward cp[ct]
        const ILight* light = lp.light[0];
        if (!light) return {};

        Vec3f toLight = lp.position[0] - cp.position[ct];
        float dist    = toLight.length();
        if (dist < 1e-6f) return {};
        Vec3f wi = toLight * (1.f / dist);

        // Visibility
        Ray shadowRay = spawnRayTo(cp.position[ct], cp.normal[ct], lp.position[0]);
        if (scene.accel->occluded(shadowRay)) return {};

        // Evaluate BSDF at camera vertex
        SurfaceInteraction si;
        si.p = cp.position[ct]; si.n = cp.normal[ct]; si.ng = cp.normal[ct];
        ShadingContext ctx(si, -cp.wo[ct]);
        BSDFEval be = cp.material[ct]->evaluate(ctx, cp.wo[ct], wi);
        if (isBlack(be.f)) return {};

        // Light emitted toward cp[ct]
        Spectrum Le = light->Le(lp.position[0], lp.normal[0], -wi);
        if (isBlack(Le)) return {};

        float cosI = std::abs(dot(wi, cp.normal[ct]));
        float cosL = std::abs(dot(-wi, lp.normal[0]));
        float dist2 = dist * dist;

        L = cp.beta[ct] * be.f * cosI * Le * cosL
          / (dist2 * lp.pdfFwd[0]);
        return L;
    }

    if (t == 1) {
        // Strategy (s, 1): project light vertex lp[s-1] onto the film.
        // Not yet implemented for pinhole (requires camera ray PDF in area measure).
        // Will be added in a follow-up.
        return {};
    }

    // General strategy (s >= 2, t >= 2): connect lp[s-1] to cp[t-1]
    const uint32_t ls = s - 1;
    const uint32_t ct = t - 1;

    if (!lp.isConnectible(ls) || !cp.isConnectible(ct)) return {};
    if (!lp.material[ls] || !cp.material[ct]) return {};

    // Visibility
    Ray shadowRay = spawnRayTo(lp.position[ls], lp.normal[ls], cp.position[ct]);
    if (scene.accel->occluded(shadowRay)) return {};

    Vec3f d    = cp.position[ct] - lp.position[ls];
    float dist = d.length();
    if (dist < 1e-6f) return {};
    Vec3f wi = d * (1.f / dist);

    // Evaluate BSDF at light-subpath vertex lp[ls] toward cp[ct]
    float fwdL, revL;
    Spectrum fL = evalVertex(lp, ls,  wi, fwdL, revL);
    if (isBlack(fL)) return {};

    // Evaluate BSDF at camera-subpath vertex cp[ct] toward lp[ls]
    float fwdC, revC;
    Spectrum fC = evalVertex(cp, ct, -wi, fwdC, revC);
    if (isBlack(fC)) return {};

    float G = geometryTerm(lp.position[ls], lp.normal[ls],
                            cp.position[ct], cp.normal[ct]);

    L = lp.beta[ls] * fL * G * fC * cp.beta[ct];
    return L;
}

// ---------------------------------------------------------------------------
// renderTile
// ---------------------------------------------------------------------------
void BDPTIntegrator::renderTile(const SceneView& scene,
                                 const TileRequest& tile,
                                 uint32_t filmWidth,
                                 uint32_t filmHeight,
                                 ISampler& sampler,
                                 TileBuffer& localTile) {
    Camera cam = scene.camera.value_or(Camera::makePinhole(
        {0.f, 0.f, -2.5f}, {0.f, 0.f, 1.f}, {0.f, 1.f, 0.f},
        50.f, filmWidth, filmHeight));

    // Per-tile path vertex buffers — pre-allocated, reset each sample
    uint32_t maxVerts = m_maxDepth + 2;
    PathVertexBuffer camPath(maxVerts);
    PathVertexBuffer lightPath(maxVerts);

    for (uint32_t ty = 0; ty < tile.height; ++ty) {
        for (uint32_t tx = 0; tx < tile.width; ++tx) {
            uint32_t px = tile.x0 + tx;
            uint32_t py = tile.y0 + ty;

            Spectrum pixelL       = {};
            Spectrum accumAlbedo  = {};
            Vec3f    accumNormal  = {};
            uint32_t aovCount     = 0;

            for (uint32_t s = 0; s < tile.sampleCount; ++s) {
                sampler.startPixelSample(px, py, tile.sampleStart + s);

                // --- Camera subpath ---
                Vec2f jitter = sampler.get2D();
                Ray primaryRay = cam.generateRay(px, py, jitter.x, jitter.y);
                // Pinhole camera: area PDF = 1 (we treat it as a single point)
                traceCameraSubpath(scene, primaryRay, 1.f, sampler, camPath);

                // Denoising AOVs: first surface vertex (index 1, after the camera vertex)
                if (camPath.count >= 2 &&
                    camPath.type(1) == PathVertexType::Surface &&
                    camPath.material[1]) {
                    SurfaceInteraction si;
                    si.p = camPath.position[1]; si.n = camPath.normal[1]; si.ng = camPath.normal[1];
                    ShadingContext ctx(si, -camPath.wo[1]);
                    accumAlbedo = accumAlbedo + camPath.material[1]->reflectance(ctx);
                    accumNormal = accumNormal + camPath.normal[1];
                    ++aovCount;
                }

                // --- Light subpath ---
                uint32_t lightIdx = 0;
                if (!m_lightSampler.empty()) {
                    auto sel = m_lightSampler.sample(sampler.get1D());
                    lightIdx = sel.index;
                }
                traceLightSubpath(scene, lightIdx, sampler, lightPath);

                // --- Connect all (s, t) strategies ---
                Spectrum sampleL = {};

                uint32_t nLight  = lightPath.count;
                uint32_t nCamera = camPath.count;

                for (uint32_t t = 1; t <= nCamera; ++t) {
                    for (uint32_t s2 = 0; s2 <= nLight; ++s2) {
                        // Skip degenerate path lengths
                        if (s2 + t < 2) continue;

                        Spectrum C = connect(scene,
                                             lightPath, s2,
                                             camPath,   t,
                                             nullptr,
                                             filmWidth, filmHeight, cam);
                        if (isBlack(C)) continue;

                        float w = bdptMISWeight(lightPath, camPath, s2, t);
                        sampleL += C * w;
                    }
                }

                pixelL += sampleL;
            }

            float invSPP = 1.f / static_cast<float>(tile.sampleCount);
            localTile.add(tx, ty, pixelL * invSPP);

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

} // namespace anacapa
