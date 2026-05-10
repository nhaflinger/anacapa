// PixelFilter.h — pixel reconstruction filters with PBRT-v4 importance sampling.
//
// Replaces the previous hardcoded box-filter / uniform-jitter sampling.  The
// filter is built once per render and is queried from each per-sample loop:
//
//   PixelFilter::Sample fs = filter.sample(u);
//   Ray ray = cam.generateRay(px, py, 0.5f + fs.dx, 0.5f + fs.dy, ...);
//   Spectrum L = Li(ray, ...);
//   accum     += L * fs.weight;   // weight is +1 for positive lobes, -1 for negative
//   weightSum += fs.weight;
//
// Each filter is separable: the 1D PDF over [-radius, radius] is tabulated
// proportionally to |f(x)|, sample x and y independently from the same
// piecewise-constant inverse CDF, and recover the sign at the chosen bin.
// For all-positive filters (Box, Triangle, Gaussian, Blackman-Harris) the
// sign is always +1.  For Mitchell-Netravali, Catmull-Rom, and Lanczos the
// negative lobes contribute -1 weights so the resulting estimator converges
// to (∫f L) / (∫f) — exactly the PBRT-v4 formulation.

#pragma once

#include <anacapa/core/Types.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace anacapa {

enum class PixelFilterType : uint32_t {
    Box            = 0,
    Triangle       = 1,
    Gaussian       = 2,
    Mitchell       = 3,
    BlackmanHarris = 4,
    CatmullRom     = 5,
    Lanczos        = 6,
};

inline const char* pixelFilterName(PixelFilterType t) {
    switch (t) {
        case PixelFilterType::Box:            return "box";
        case PixelFilterType::Triangle:       return "triangle";
        case PixelFilterType::Gaussian:       return "gaussian";
        case PixelFilterType::Mitchell:       return "mitchell";
        case PixelFilterType::BlackmanHarris: return "blackman-harris";
        case PixelFilterType::CatmullRom:     return "catmull-rom";
        case PixelFilterType::Lanczos:        return "lanczos";
    }
    return "unknown";
}

inline bool parsePixelFilterName(const std::string& name, PixelFilterType& out) {
    if (name == "box")             { out = PixelFilterType::Box;            return true; }
    if (name == "triangle")        { out = PixelFilterType::Triangle;       return true; }
    if (name == "gaussian")        { out = PixelFilterType::Gaussian;       return true; }
    if (name == "mitchell")        { out = PixelFilterType::Mitchell;       return true; }
    if (name == "blackman-harris" || name == "blackman_harris" ||
        name == "blackmanharris" || name == "bh") {
        out = PixelFilterType::BlackmanHarris; return true;
    }
    if (name == "catmull-rom" || name == "catmullrom" || name == "catmull") {
        out = PixelFilterType::CatmullRom; return true;
    }
    if (name == "lanczos" || name == "sinc") { out = PixelFilterType::Lanczos; return true; }
    return false;
}

// Sensible default radius for each filter family, in pixel units.
inline float defaultPixelFilterRadius(PixelFilterType t) {
    switch (t) {
        case PixelFilterType::Box:            return 0.5f;
        case PixelFilterType::Triangle:       return 1.0f;
        case PixelFilterType::Gaussian:       return 1.5f;
        case PixelFilterType::Mitchell:       return 2.0f;
        case PixelFilterType::BlackmanHarris: return 1.5f;
        case PixelFilterType::CatmullRom:     return 2.0f;
        case PixelFilterType::Lanczos:        return 4.0f;
    }
    return 1.0f;
}

// ---------------------------------------------------------------------------
// PixelFilter — separable 2D filter with importance-sampled jitter.
// ---------------------------------------------------------------------------
class PixelFilter {
public:
    static constexpr int kBins = 64;

    struct Sample {
        float dx;     // offset from pixel centre, in [-radius, radius]
        float dy;     // same
        float weight; // ±1
    };

    // Build a filter of the given type and pixel radius.
    PixelFilter(PixelFilterType type = PixelFilterType::Mitchell,
                 float radius = 0.f) {
        m_type   = type;
        m_radius = (radius > 0.f) ? radius : defaultPixelFilterRadius(type);
        buildTables();
    }

    PixelFilterType type()   const { return m_type; }
    float           radius() const { return m_radius; }

    // Sample a 2D jitter in [-radius, radius]^2 from the filter's PDF.
    // u must be a uniform 2D random in [0, 1)^2.
    Sample sample(float u, float v) const {
        auto sx = sample1D(u);
        auto sy = sample1D(v);
        return { sx.x, sy.x, sx.sign * sy.sign };
    }
    Sample sample(Vec2f u) const { return sample(u.x, u.y); }

    // Read-only access to the 1D table (used by GPU backends to upload
    // the same CDF to device memory).
    const std::vector<float>& cdf()   const { return m_cdf;   }
    const std::vector<float>& signs() const { return m_signs; }

private:
    PixelFilterType    m_type;
    float              m_radius;
    std::vector<float> m_cdf;    // size kBins+1, monotone in [0, 1]
    std::vector<float> m_signs;  // size kBins, ±1 per bin

    struct Sample1D { float x; float sign; };

    Sample1D sample1D(float u) const {
        // Binary search for the upper bound bin.
        int lo = 0, hi = kBins;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (m_cdf[mid + 1] <= u) lo = mid + 1; else hi = mid;
        }
        int   bin = std::min(lo, kBins - 1);
        float c0  = m_cdf[bin];
        float c1  = m_cdf[bin + 1];
        float t   = (c1 > c0) ? (u - c0) / (c1 - c0) : 0.5f;
        float binW = (2.f * m_radius) / float(kBins);
        float x   = -m_radius + (float(bin) + t) * binW;
        return { x, m_signs[bin] };
    }

    void buildTables() {
        m_cdf.resize(kBins + 1);
        m_signs.resize(kBins);
        const float binW = (2.f * m_radius) / float(kBins);
        std::vector<float> mag(kBins);
        for (int i = 0; i < kBins; ++i) {
            float xc = -m_radius + (float(i) + 0.5f) * binW;
            float v  = eval1D(xc);
            mag[i]   = std::abs(v);
            m_signs[i] = (v >= 0.f) ? 1.f : -1.f;
        }
        // Normalise to a CDF; degenerate (all zero) → uniform.
        float sum = 0.f;
        for (float v : mag) sum += v;
        m_cdf[0] = 0.f;
        if (sum > 0.f) {
            float inv = 1.f / sum;
            for (int i = 0; i < kBins; ++i)
                m_cdf[i + 1] = m_cdf[i] + mag[i] * inv;
            m_cdf[kBins] = 1.f;  // guard against rounding drift
        } else {
            for (int i = 0; i < kBins; ++i)
                m_cdf[i + 1] = float(i + 1) / float(kBins);
            for (int i = 0; i < kBins; ++i) m_signs[i] = 1.f;
        }
    }

    // Single-axis filter evaluation — separable, so the 2D filter is the
    // product of two 1D evaluations.
    float eval1D(float x) const {
        const float ax = std::abs(x);
        if (ax > m_radius) return 0.f;
        const float Pi = 3.14159265358979323846f;
        switch (m_type) {
            case PixelFilterType::Box:
                return 1.f;

            case PixelFilterType::Triangle:
                return std::max(0.f, 1.f - ax / m_radius);

            case PixelFilterType::Gaussian: {
                // Truncated Gaussian: σ chosen so the radius is 2σ (Cycles-style).
                float sigma = m_radius * 0.5f;
                float k     = 1.f / (2.f * sigma * sigma);
                float edge  = std::exp(-m_radius * m_radius * k);
                return std::max(0.f, std::exp(-x * x * k) - edge);
            }

            case PixelFilterType::Mitchell:
                return mitchell(x, m_radius, 1.f / 3.f, 1.f / 3.f);

            case PixelFilterType::CatmullRom:
                return mitchell(x, m_radius, 0.f, 0.5f);

            case PixelFilterType::BlackmanHarris: {
                // Standard 4-term Blackman-Harris on [0, 1] mapped from [-R, R].
                float t  = (x + m_radius) * 0.5f / m_radius;
                float a0 = 0.35875f, a1 = 0.48829f;
                float a2 = 0.14128f, a3 = 0.01168f;
                return a0
                     - a1 * std::cos(2.f * Pi * t)
                     + a2 * std::cos(4.f * Pi * t)
                     - a3 * std::cos(6.f * Pi * t);
            }

            case PixelFilterType::Lanczos: {
                // Windowed sinc — τ = 3 (canonical Lanczos-3).
                const float tau = 3.f;
                float xR = ax / m_radius;
                if (xR >= 1.f) return 0.f;
                auto sinc = [Pi](float v) {
                    if (std::abs(v) < 1e-6f) return 1.f;
                    float pv = Pi * v;
                    return std::sin(pv) / pv;
                };
                return sinc(xR * tau) * sinc(xR);
            }
        }
        return 0.f;
    }

    // Mitchell-Netravali piecewise cubic, scaled so radius corresponds to
    // the canonical Mitchell radius of 2 pixels.
    static float mitchell(float x, float radius, float B, float C) {
        float xN = std::abs(x) * (2.f / radius);
        float xN2 = xN * xN;
        float xN3 = xN2 * xN;
        if (xN < 1.f) {
            return ((12.f - 9.f * B - 6.f * C) * xN3
                  + (-18.f + 12.f * B + 6.f * C) * xN2
                  + (6.f - 2.f * B)) * (1.f / 6.f);
        }
        if (xN < 2.f) {
            return ((-B - 6.f * C) * xN3
                  + (6.f * B + 30.f * C) * xN2
                  + (-12.f * B - 48.f * C) * xN
                  + (8.f * B + 24.f * C)) * (1.f / 6.f);
        }
        return 0.f;
    }
};

} // namespace anacapa
