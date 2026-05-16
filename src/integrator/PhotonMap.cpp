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
// estimateSSSRadiance — Gaussian-kernel density estimate for subsurface
// scattering photons.
//
// The Gaussian profile exp(-r²/(2σ²)) / (2π σ²) approximates the
// single-order diffusion profile for the material's mean free path σ.
// We search within 3σ so we capture 99.7% of the kernel weight.
// ---------------------------------------------------------------------------
Spectrum PhotonMap::estimateSSSRadiance(Vec3f p,
                                         Spectrum subsurface_color,
                                         float sigma) const {
    if (m_photons.size() <= 1 || sigma <= 0.f) return {};

    // Fixed-σ Gaussian density estimate — the physically correct approach.
    //
    // Search within 2σ (captures 95% of the Gaussian kernel weight) and apply
    // a Gaussian weight exp(-r²/(2σ²)).  Normalise by 2πσ² (the 2-D Gaussian
    // integral).  No photon count cap: returning all photons in the radius is
    // essential — any hard cap in kd-tree traversal order creates visible
    // straight-line and circular artifacts at kd-node split boundaries.
    //
    // For performance: keep σ ≤ 0.1 world-units (10 cm) in typical scenes.
    // Large σ causes many photons per query and slow renders.
    float searchR = 2.f * sigma;

    std::vector<const Photon*> nearby;
    nearby.reserve(256);
    search(1, p, searchR * searchR, nearby);
    if (nearby.empty()) return {};

    float sigma2 = sigma * sigma;
    Spectrum L = {};
    for (const Photon* ph : nearby) {
        Vec3f d = p - ph->position;
        float r2 = d.x*d.x + d.y*d.y + d.z*d.z;
        float w  = std::exp(-r2 / (2.f * sigma2));
        L += ph->power * w;
    }
    return subsurface_color * L * (1.f / (2.f * kPi * sigma2));
}

} // namespace anacapa
