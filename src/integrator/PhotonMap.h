#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ShadingContext.h>
#include <vector>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// Photon — a single photon stored in the caustic photon map.
//
// wi is the direction the photon was traveling when it hit the surface
// (i.e., toward the surface, like a ray direction).  During density
// estimation, -wi is used as the incoming light direction for BSDF queries.
//
// power carries the radiant flux (watts) accumulated along the path from the
// light through all intermediate specular bounces.  The cosine at the storage
// surface is NOT baked in; it is folded into the density estimate via the
// BSDF evaluation.
// ---------------------------------------------------------------------------
struct Photon {
    Vec3f    position;
    Vec3f    wi;       // photon travel direction (toward surface)
    Spectrum power;    // radiant flux along path (W)
    uint8_t  axis;     // kd-tree split axis: 0=x, 1=y, 2=z; 0xFF = empty slot
    uint8_t  _pad[3] = {};
};

// ---------------------------------------------------------------------------
// PhotonMap — balanced kd-tree for O(log N + k) radius searches.
//
// Build once from a flat vector of photons (build() rearranges them in-place
// into heap order).  estimateRadiance() is read-only and thread-safe after
// build completes.
// ---------------------------------------------------------------------------
class PhotonMap {
public:
    // Rearrange photons into kd-tree order and store internally.
    // The input vector is consumed.
    void build(std::vector<Photon> photons);

    // Estimate caustic radiance at a diffuse surface hit.
    //   p      — world-space hit point
    //   n      — surface normal
    //   wo     — outgoing direction (toward camera), world space
    //   mat    — surface material for BSDF evaluation
    //   ctx    — shading context at p
    //   radius — photon search radius in world units
    Spectrum estimateRadiance(Vec3f p, Vec3f n,
                               Vec3f wo,
                               const IMaterial& mat,
                               const ShadingContext& ctx,
                               float radius) const;

    // Estimate subsurface scattering radiance using a Gaussian diffusion kernel.
    //   p                 — world-space query point (surface hit)
    //   subsurface_color  — per-material scattering albedo
    //   sigma             — Gaussian σ = subsurface_radius (mean free path)
    // Searches within 3σ of p; weights each photon by exp(-r²/(2σ²)) / (2π σ²).
    Spectrum estimateSSSRadiance(Vec3f p,
                                  Spectrum subsurface_color,
                                  float sigma) const;

    bool   empty() const { return m_photons.size() <= 1; }
    size_t size()  const { return m_photons.empty() ? 0 : m_photons.size() - 1; }

private:
    void buildRecursive(std::vector<Photon>& src, int lo, int hi, int node);
    void search(int node, Vec3f p, float radius2,
                std::vector<const Photon*>& result) const;

    // KNN search: fills heap with the K nearest photons within the initial
    // radius2.  radius2 is shrunk progressively as the heap fills, guaranteeing
    // the K spatially closest photons with no kd-tree-order bias.
    struct PhotonHit {
        float        dist2;
        const Photon* ph;
        bool operator<(const PhotonHit& o) const { return dist2 < o.dist2; }
    };
    void searchKNN(int node, Vec3f p, float& radius2,
                   std::vector<PhotonHit>& heap, size_t maxK) const;

    std::vector<Photon> m_photons; // 1-indexed; slot 0 unused
};

} // namespace anacapa
