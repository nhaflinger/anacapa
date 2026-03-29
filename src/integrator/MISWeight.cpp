#include <anacapa/integrator/MISWeight.h>
#include <cmath>

namespace anacapa {

// ---------------------------------------------------------------------------
// remapRoughness: a delta vertex has pdfFwd/pdfRev = 0 in area measure
// (the BSDFs have infinite density at a point). We treat these vertices as
// having weight 0 in the denominator sum — i.e. strategies that require
// explicitly sampling a delta vertex are invalid. This is implemented by
// returning 0 for any ratio that crosses through a delta vertex.
// ---------------------------------------------------------------------------

float bdptMISWeight(const PathVertexBuffer& lp,
                    const PathVertexBuffer& cp,
                    uint32_t s, uint32_t t) {
    // Single strategy — no MIS needed
    if (s + t == 2) return 1.f;

    // sumRi accumulates sum of (p_i / p_{s,t})^2 for all strategies i != (s,t)
    // We include the (s,t) strategy itself (ratio = 1) at the end.
    double sumRi = 0.0;
    double ri    = 1.0;  // current ratio relative to p(s,t)

    // --- Walk toward the light side: shift one vertex at a time from camera ---
    // Strategy (s-1, t+1), (s-2, t+2), ..., (0, s+t)
    // Each step: multiply ri by cp[t-1].pdfRev / lp[s-1].pdfFwd  (roughly)
    // Exact recurrence (Veach §10.3.4):
    //   p(s-i, t+i) = p(s, t) * prod_{k=0}^{i-1} [ pdfRev(lp, s-1-k) / pdfFwd(lp, s-1-k) ]
    //
    // Implemented as: after each step ri = p(s-i,t+i) / p(s,t)
    // The connecting vertex moves from the light subpath to the camera subpath.
    ri = 1.0;
    for (int i = static_cast<int>(s) - 1; i >= 0; --i) {
        // The vertex being "transferred" is lp[i] (0-indexed light subpath)
        // At each step the ratio changes by: pdfRev[i] / pdfFwd[i]
        // But we must skip strategies where a delta vertex would need to be
        // explicitly sampled.

        // Numerator: pdfRev of lp[i] — how likely is lp[i] sampled from lp[i+1]
        float num = (i < static_cast<int>(lp.count)) ? lp.pdfRev[i] : 0.f;
        // Denominator: pdfFwd of lp[i] — how likely was lp[i] sampled from lp[i-1]
        float den = (i < static_cast<int>(lp.count)) ? lp.pdfFwd[i] : 0.f;

        if (den == 0.f) break;  // Can't shift past a delta/zero-pdf vertex
        ri *= static_cast<double>(num) / static_cast<double>(den);

        // Strategy (i, s+t-i) is valid only if neither endpoint is delta
        bool prevDelta = (i > 0) && lp.isDelta(i - 1);
        bool currDelta = (i < static_cast<int>(lp.count)) && lp.isDelta(i);
        if (!prevDelta && !currDelta) sumRi += ri * ri;
    }

    // --- Walk toward the camera side: shift one vertex at a time from light ---
    // Strategy (s+1, t-1), (s+2, t-2), ..., (s+t, 0)
    ri = 1.0;
    for (int i = static_cast<int>(t) - 1; i >= 0; --i) {
        float num = (i < static_cast<int>(cp.count)) ? cp.pdfRev[i] : 0.f;
        float den = (i < static_cast<int>(cp.count)) ? cp.pdfFwd[i] : 0.f;

        if (den == 0.f) break;
        ri *= static_cast<double>(num) / static_cast<double>(den);

        bool prevDelta = (i > 0) && cp.isDelta(i - 1);
        bool currDelta = (i < static_cast<int>(cp.count)) && cp.isDelta(i);
        if (!prevDelta && !currDelta) sumRi += ri * ri;
    }

    // Include the (s,t) strategy itself: ratio = 1, contributes 1^2 = 1
    double totalSum = sumRi + 1.0;
    return static_cast<float>(1.0 / totalSum);
}

} // namespace anacapa
