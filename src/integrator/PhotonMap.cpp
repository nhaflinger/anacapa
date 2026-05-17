#include "PhotonMap.h"
#include <algorithm>
#include <cmath>

namespace anacapa {

static constexpr float kPi = 3.14159265358979f;

// ---------------------------------------------------------------------------
// build — rearrange photons into a balanced heap-indexed kd-tree.
//
// Node layout: root at index 1, children of node i at 2i and 2i+1.
// For N photons built with median splitting, the maximum node index is
// bounded by 2N-1, so we allocate 2N+2 slots.
// Unfilled slots are marked with axis=0xFF and skipped during search.
// ---------------------------------------------------------------------------
void PhotonMap::build(std::vector<Photon> photons) {
    if (photons.empty()) {
        m_photons.clear();
        return;
    }
    m_photons.resize(2 * photons.size() + 2);
    for (auto& p : m_photons) p.axis = 0xFF;
    buildRecursive(photons, 0, static_cast<int>(photons.size()) - 1, 1);
}

void PhotonMap::buildRecursive(std::vector<Photon>& src, int lo, int hi, int node) {
    if (lo > hi || node >= static_cast<int>(m_photons.size())) return;

    if (lo == hi) {
        src[lo].axis = 0;
        m_photons[node] = src[lo];
        return;
    }

    // Choose split axis along the longest bounding-box dimension
    Vec3f bmin = src[lo].position, bmax = src[lo].position;
    for (int i = lo + 1; i <= hi; ++i) {
        bmin = min(bmin, src[i].position);
        bmax = max(bmax, src[i].position);
    }
    Vec3f ext = bmax - bmin;
    uint8_t axis = 0;
    if (ext.y > ext.x) axis = 1;
    if (ext.z > ext[axis]) axis = 2;

    // Partition so the median element is at index mid
    int mid = (lo + hi) / 2;
    std::nth_element(src.begin() + lo, src.begin() + mid, src.begin() + hi + 1,
        [axis](const Photon& a, const Photon& b) {
            return a.position[axis] < b.position[axis];
        });

    src[mid].axis   = axis;
    m_photons[node] = src[mid];

    buildRecursive(src, lo,      mid - 1, 2 * node);
    buildRecursive(src, mid + 1, hi,      2 * node + 1);
}

// ---------------------------------------------------------------------------
// search — collect all photons within radius² of p into result.
// ---------------------------------------------------------------------------
void PhotonMap::search(int node, Vec3f p, float radius2,
                        std::vector<const Photon*>& result) const {
    if (node >= static_cast<int>(m_photons.size())) return;
    const Photon& ph = m_photons[node];
    if (ph.axis == 0xFF) return;

    Vec3f d = p - ph.position;
    if (d.x*d.x + d.y*d.y + d.z*d.z < radius2)
        result.push_back(&ph);

    uint8_t axis = ph.axis;
    float   axisD = p[axis] - ph.position[axis];
    int nearChild = (axisD <= 0.f) ? 2*node   : 2*node+1;
    int farChild  = (axisD <= 0.f) ? 2*node+1 : 2*node;

    search(nearChild, p, radius2, result);
    if (axisD * axisD < radius2)
        search(farChild, p, radius2, result);
}

// ---------------------------------------------------------------------------
// searchKNN — K-nearest-neighbour search.
//
// radius2 is an in-out parameter: it starts as the initial search bound
// (e.g. (3σ)²) and is progressively shrunk to the distance of the Kth
// closest photon as the heap fills.  This pruning eliminates kd-subtrees
// that cannot contain a closer photon, giving O((K + log N)²)-ish cost.
//
// The heap is a max-heap keyed on dist² so the Kth-closest photon is always
// at the top and can be evicted in O(log K) when a closer one is found.
// ---------------------------------------------------------------------------
void PhotonMap::searchKNN(int node, Vec3f p, float& radius2,
                           std::vector<PhotonHit>& heap, size_t maxK) const {
    if (node >= static_cast<int>(m_photons.size())) return;
    const Photon& ph = m_photons[node];
    if (ph.axis == 0xFF) return;

    Vec3f d = p - ph.position;
    float r2 = d.x*d.x + d.y*d.y + d.z*d.z;

    if (r2 < radius2) {
        heap.push_back({r2, &ph});
        std::push_heap(heap.begin(), heap.end()); // max-heap: largest dist² at front
        if (heap.size() > maxK) {
            std::pop_heap(heap.begin(), heap.end());
            heap.pop_back();
        }
        // Once the heap is full, shrink the search radius to the farthest kept photon.
        if (heap.size() == maxK)
            radius2 = heap.front().dist2;
    }

    uint8_t axis  = ph.axis;
    float   axisD = p[axis] - ph.position[axis];
    int nearChild = (axisD <= 0.f) ? 2*node   : 2*node+1;
    int farChild  = (axisD <= 0.f) ? 2*node+1 : 2*node;

    searchKNN(nearChild, p, radius2, heap, maxK);
    if (axisD * axisD < radius2)
        searchKNN(farChild, p, radius2, heap, maxK);
}

// ---------------------------------------------------------------------------
// estimateRadiance — flat-kernel density estimate.
//
// Formula: L(p, wo) ≈ (1 / π r²) Σ_j f(p, -ph.wi, wo) * ph.power
//
// ph.wi is the photon travel direction (toward the surface), so -ph.wi is
// the incoming light direction for the BSDF.  The BSDF evaluation (be.f)
// returns the reflectance without the cosine factor; the photon power already
// incorporates all geometry terms from intermediate bounces, so no extra
// cosine is needed here.
// ---------------------------------------------------------------------------
Spectrum PhotonMap::estimateRadiance(Vec3f p, Vec3f n,
                                      Vec3f wo,
                                      const IMaterial& mat,
                                      const ShadingContext& ctx,
                                      float radius) const {
    if (m_photons.size() <= 1) return {};

    std::vector<const Photon*> nearby;
    nearby.reserve(64);
    search(1, p, radius * radius, nearby);
    if (nearby.empty()) return {};

    Spectrum L = {};
    for (const Photon* ph : nearby) {
        BSDFEval be = mat.evaluate(ctx, wo, -ph->wi);
        if (!isBlack(be.f))
            L += be.f * ph->power;
    }

    return L * (1.f / (kPi * radius * radius));
}

// ---------------------------------------------------------------------------
// estimateSSSRadiance — exponential lateral kernel with depth attenuation.
//
// Lateral kernel: R(r_lat) = exp(-r_lat/d) / (2π·d²), applied to the
// surface-projected distance. This is bounded (no singularity), integrates
// to 1, and decays with a heavier tail than Gaussian.
//
// Depth weighting: photons on the same face (dot(ph->pos - p, n) >= 0) get
// full lateral weight. Photons behind the surface (back face of thick
// geometry) get an extra exp(-|depth|/d) factor. For thin objects the depth
// is small → factor ≈ 1 (bleed-through preserved). For thick objects the
// depth is large → factor ≈ 0 (back-face contribution suppressed).
//
// Search radius reduced to 4d (captures 98.2% of exp(-r/d) weight vs 99.7%
// at 6d) for a ~2.25× reduction in photons per query.
// ---------------------------------------------------------------------------
Spectrum PhotonMap::estimateSSSRadiance(Vec3f p, Vec3f n,
                                         Spectrum subsurface_color,
                                         float d) const {
    if (m_photons.size() <= 1 || d <= 0.f) return {};

    // 6d captures 99.75% of exp(-r/d) weight; the kernel naturally decays to
    // exp(-6)≈0.0025 at the boundary so there's no hard edge cutoff.
    float searchR = 6.f * d;

    std::vector<const Photon*> nearby;
    nearby.reserve(256);
    search(1, p, searchR * searchR, nearby);
    if (nearby.empty()) return {};

    // Hard minimum: if fewer than 4 photons found there is not enough density
    // to produce a valid estimate — return nothing rather than a bright hotspot.
    // Above 4, smoothstep from 0 to 1 by 24 photons to fade at sparse boundaries.
    constexpr int kHardMin = 4;
    constexpr int kSoftMin = 24;
    if ((int)nearby.size() < kHardMin) return {};
    float t = std::min(1.f, (float)((int)nearby.size() - kHardMin)
                              / (float)(kSoftMin - kHardMin));
    float density_scale = t * t * (3.f - 2.f * t); // smoothstep

    float norm = 1.f / (2.f * kPi * kPi * d * d); // extra 1/π: irradiance → radiance
    Spectrum L = {};
    for (const Photon* ph : nearby) {
        // Vector from query point to photon deposit.
        Vec3f dv   = ph->position - p;
        // Signed depth: negative means photon is behind the surface (back face).
        float proj = dv.x*n.x + dv.y*n.y + dv.z*n.z;
        // Lateral (surface-plane) distance.
        float r_lat2 = std::max(0.f, dv.x*dv.x + dv.y*dv.y + dv.z*dv.z - proj*proj);
        float r_lat  = std::sqrt(r_lat2);
        // Lateral kernel weight.
        float w = std::exp(-r_lat / d) * norm;
        // Extra depth attenuation for back-face photons only.
        if (proj < 0.f)
            w *= std::exp(proj / d); // proj negative → exp(-|depth|/d)
        L += ph->power * w;
    }
    return subsurface_color * (L * density_scale);
}

} // namespace anacapa
