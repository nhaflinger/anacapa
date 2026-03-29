#pragma once

#include <anacapa/integrator/PathVertex.h>

namespace anacapa {

// ---------------------------------------------------------------------------
// Power heuristic (Veach 1997, §9.2)
//
// For two sampling strategies with PDFs f and g and n_f, n_g samples:
//   w(f) = (n_f * f)^beta / sum_i (n_i * f_i)^beta
//
// With beta=2 and n_f=n_g=1 (one sample per strategy, standard in BDPT):
//   w(f) = f^2 / (f^2 + g^2)
// ---------------------------------------------------------------------------
inline float powerHeuristic(float pdfF, float pdfG, float beta = 2.f) {
    float f = std::pow(pdfF, beta);
    float g = std::pow(pdfG, beta);
    return f / (f + g);
}

// ---------------------------------------------------------------------------
// BDPT MIS weight — Veach thesis §10.3
//
// A full path of length n = s + t - 2 interior vertices can be generated
// by any of the (s+t+1) sampling strategies (s', t') where s'+t' = s+t.
// The MIS weight for strategy (s, t) is:
//
//   w(s,t) = p(s,t)^2 / sum_{s',t'} p(s',t')^2
//
// where p(s,t) is the unsigned area-measure PDF of the full path under
// strategy (s,t), and the sum is over all valid strategies.
//
// This is computed via the recursive PDF ratio trick:
//   r_i = p(s-i, t+i) / p(s, t)  for i = 1..s   (shifting from light side)
//   r_i = p(s+i, t-i) / p(s, t)  for i = 1..t-1 (shifting from camera side)
//
// Each ratio step multiplies or divides by consecutive pdfFwd / pdfRev values.
//
// Parameters:
//   lightPath  — the light subpath (vertices 0..s-1)
//   cameraPath — the camera subpath (vertices 0..t-1)
//   s          — number of light subpath vertices used in this connection
//   t          — number of camera subpath vertices used
//
// Returns the MIS weight in (0, 1].
// Returns 1 if only one strategy is valid (e.g., purely specular paths).
// ---------------------------------------------------------------------------
float bdptMISWeight(const PathVertexBuffer& lightPath,
                    const PathVertexBuffer& cameraPath,
                    uint32_t s, uint32_t t);

} // namespace anacapa
