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

} // namespace anacapa
