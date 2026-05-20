#pragma once

#include <sky/NishitaSky.h>
#include <anacapa/shading/ILight.h>
#include <anacapa/core/Types.h>
#include <cmath>
#include <vector>
#include <algorithm>

namespace anacapa {

// ---------------------------------------------------------------------------
// SkyLight — ILight wrapper around NishitaSky
//
// Importance sampling strategy:
//   - Sky dome is sampled via a precomputed 2D luminance distribution
//     (lat-long grid of kResV × kResU cells).
//   - Sun disc is sampled analytically as a spherical cap.
//   - MIS combines the two strategies proportional to their power.
//
// PDF convention (solid angle):
//   p_sky(omega) = f(omega) / integral_f   where f is the luminance grid
//   p_sun(omega) = 1 / (2*pi*(1 - cos(sunDiscAngle)))  for the cap
// ---------------------------------------------------------------------------
class SkyLight : public ILight {
public:
    static constexpr float kPi    = NishitaSky::kPi;
    static constexpr float k2Pi   = NishitaSky::k2Pi;
    static constexpr float kInvPi = NishitaSky::kInvPi;

    // Resolution of the precomputed importance-sampling grid
    static constexpr int kResU = 512;
    static constexpr int kResV = 256;

    explicit SkyLight(const NishitaParams& p,
                      float sceneRadius = 10.f,
                      Vec3f sceneCenter = {})
        : m_sky(p)
        , m_sceneRadius(sceneRadius)
        , m_sceneCenter(sceneCenter)
    {
        buildDistribution();
        m_sunCapArea    = k2Pi * (1.f - std::cos(p.sunDiscAngle));
        m_sunPdfCap     = (m_sunCapArea > 0.f) ? 1.f / m_sunCapArea : 0.f;
        m_sunPower      = estimateSunPower();
        m_skyPower      = estimateSkyPower();
        float total     = m_sunPower + m_skyPower;
        m_pSunMIS       = (total > 0.f) ? m_sunPower / total : 0.5f;
    }

    // -----------------------------------------------------------------------
    // ILight interface
    // -----------------------------------------------------------------------

    LightSample sample(Vec3f /*p*/, Vec3f /*n*/, Vec2f u) const override
    {
        // Russian roulette: pick sun or sky stratum
        LightSample s;
        if (u.x < m_pSunMIS) {
            // Remap u.x into [0,1)
            Vec2f u2 = { u.x / m_pSunMIS, u.y };
            s.wi  = sampleSunCap(u2);
            s.pdf = pdfCombined(s.wi);
        } else {
            Vec2f u2 = { (u.x - m_pSunMIS) / (1.f - m_pSunMIS), u.y };
            s.wi  = sampleSkyGrid(u2);
            s.pdf = pdfCombined(s.wi);
        }
        if (s.pdf <= 0.f) return {};
        s.Li      = m_sky.evaluate(s.wi);
        s.dist    = 1e10f;
        s.isDelta = false;
        return s;
    }

    float pdf(Vec3f /*p*/, Vec3f wi) const override
    {
        return pdfCombined(wi);
    }

    LightLeSample sampleLe(Vec2f uPos, Vec2f uDir) const override
    {
        // Sample direction from the sky distribution
        float pdfDir;
        Vec3f envDir = sampleSkyGridRaw(uDir, pdfDir);
        if (pdfDir <= 0.f) return {};

        Vec3f t, bt;
        buildOrthonormalBasis(-envDir, t, bt);

        float r   = m_sceneRadius * std::sqrt(uPos.x);
        float phi = k2Pi * uPos.y;
        Vec3f pos = m_sceneCenter + envDir * m_sceneRadius
                  + t  * (r * std::cos(phi))
                  + bt * (r * std::sin(phi));

        float diskArea = kPi * m_sceneRadius * m_sceneRadius;

        LightLeSample s;
        s.Le     = m_sky.evaluate(envDir);
        s.pos    = pos;
        s.normal = -envDir;
        s.dir    = -envDir;
        s.pdfPos = 1.f / diskArea;
        s.pdfDir = pdfDir;
        return s;
    }

    Spectrum Le(Vec3f /*pos*/, Vec3f /*normal*/, Vec3f wo) const override
    {
        return m_sky.evaluate(wo);
    }

    float power() const override
    {
        return (m_sunPower + m_skyPower) * 4.f * kPi;
    }

    bool isInfinite()    const override { return true; }
    bool isDelta()       const override { return false; }
    bool transparentBg() const override { return m_sky.params().transparentBg; }

    // Accessors
    const NishitaSky&    sky()    const { return m_sky; }
    const NishitaParams& params() const { return m_sky.params(); }

    void setSceneRadius(float r) { m_sceneRadius = r; }
    void setSceneCenter(Vec3f c) { m_sceneCenter = c; }

    // -----------------------------------------------------------------------
    // GPU support — bake sky to an equirectangular RGBA32F texture and expose
    // CDF tables in the same format as DomeLight (for upload to Metal / CUDA).
    // -----------------------------------------------------------------------
    uint32_t texWidth()  const { return static_cast<uint32_t>(kResU); }
    uint32_t texHeight() const { return static_cast<uint32_t>(kResV); }

    // Returns kResU*kResV RGBA32F pixels, row-major, theta=0 at row 0 (+Y).
    std::vector<float> bakeRGBA() const {
        std::vector<float> out(static_cast<size_t>(kResU) * kResV * 4);
        for (int r = 0; r < kResV; ++r) {
            float v     = (r + 0.5f) / static_cast<float>(kResV);
            float theta = v * kPi;
            float sinT  = std::sin(theta), cosT = std::cos(theta);
            for (int c = 0; c < kResU; ++c) {
                float uu  = (c + 0.5f) / static_cast<float>(kResU);
                float phi = uu * k2Pi;
                Vec3f d   = { sinT * std::sin(phi), cosT, sinT * std::cos(phi) };
                Spectrum L = m_sky.evaluate(d);
                size_t idx = static_cast<size_t>(r * kResU + c) * 4;
                out[idx+0] = L.x;
                out[idx+1] = L.y;
                out[idx+2] = L.z;
                out[idx+3] = 1.f;
            }
        }
        return out;
    }

    // Marginal CDF: (kResV+1) floats, same format as DomeLight::marginalCdf().
    const std::vector<float>& marginalCdf() const { return m_marginal.cdf; }

    // Conditional CDFs packed row-major: kResV rows × (kResU+1) floats each.
    std::vector<float> flatConditionalCdf() const {
        std::vector<float> result;
        result.reserve(static_cast<size_t>(kResV) * (kResU + 1));
        for (const auto& d : m_conditional)
            result.insert(result.end(), d.cdf.begin(), d.cdf.end());
        return result;
    }

private:
    // -----------------------------------------------------------------------
    // Precomputed 2D luminance distribution (lat-long grid)
    // -----------------------------------------------------------------------
    struct Dist1D {
        std::vector<float> func;
        std::vector<float> cdf;
        float integral = 0.f;
        uint32_t n = 0;

        void build(const float* vals, uint32_t count) {
            n = count;
            func.assign(vals, vals + count);
            cdf.resize(count + 1);
            cdf[0] = 0.f;
            for (uint32_t i = 0; i < count; ++i)
                cdf[i + 1] = cdf[i] + func[i];
            integral = cdf[count];
            if (integral > 0.f) {
                float inv = 1.f / integral;
                for (auto& c : cdf) c *= inv;
            }
        }

        uint32_t sample(float u, float& pdf, float& uRemap) const {
            auto it  = std::lower_bound(cdf.begin(), cdf.end(), u);
            int  idx = static_cast<int>(it - cdf.begin()) - 1;
            idx = std::max(0, std::min(idx, static_cast<int>(n) - 1));
            float w  = cdf[idx + 1] - cdf[idx];
            uRemap   = (w > 1e-12f) ? std::max(0.f, std::min((u - cdf[idx]) / w, 1.f - 1e-7f)) : 0.f;
            pdf      = (integral > 0.f) ? func[idx] / integral : 0.f;
            return static_cast<uint32_t>(idx);
        }
    };

    void buildDistribution() {
        m_conditional.resize(kResV);
        std::vector<float> rowSums(kResV);
        std::vector<float> rowFunc(kResU);

        for (int r = 0; r < kResV; ++r) {
            float v         = (r + 0.5f) / static_cast<float>(kResV);
            float theta     = v * kPi;
            float sinTheta  = std::sin(theta);
            float cosTheta  = std::cos(theta);

            for (int c = 0; c < kResU; ++c) {
                float uu  = (c + 0.5f) / static_cast<float>(kResU);
                float phi = uu * k2Pi;
                Vec3f d = {
                    sinTheta * std::sin(phi),
                    cosTheta,
                    sinTheta * std::cos(phi)
                };
                Spectrum L = m_sky.evaluate(d);
                float lum  = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
                rowFunc[c] = lum * sinTheta;
            }
            m_conditional[r].build(rowFunc.data(), kResU);
            rowSums[r] = m_conditional[r].integral;
        }
        m_marginal.build(rowSums.data(), kResV);
    }

    // -----------------------------------------------------------------------
    // Sample sky grid: returns world direction and sets pdfSolidAngle
    // -----------------------------------------------------------------------
    Vec3f sampleSkyGridRaw(Vec2f u, float& pdfSolidAngle) const {
        float pdfRow, pdfCol, uRr, uRc;
        uint32_t row = m_marginal.sample(u.y, pdfRow, uRr);
        uint32_t col = m_conditional[row].sample(u.x, pdfCol, uRc);

        float v     = (row + uRr) / static_cast<float>(kResV);
        float uu    = (col + uRc) / static_cast<float>(kResU);
        float theta = v * kPi;
        float phi   = uu * k2Pi;

        float sinTheta = std::max(std::sin(theta), 1e-6f);
        float cosTheta = std::cos(theta);

        Vec3f d = { sinTheta * std::sin(phi), cosTheta, sinTheta * std::cos(phi) };

        pdfSolidAngle = (pdfRow * pdfCol * static_cast<float>(kResU * kResV))
                      / (k2Pi * kPi * sinTheta);
        if (pdfSolidAngle < 0.f) pdfSolidAngle = 0.f;
        return d;
    }

    Vec3f sampleSkyGrid(Vec2f u) const {
        float pdf;
        return sampleSkyGridRaw(u, pdf);
    }

    float evalSkyGridPdf(Vec3f d) const {
        float theta     = std::acos(std::max(-1.f, std::min(1.f, d.y)));
        float phi       = std::atan2(d.x, d.z);
        if (phi < 0.f) phi += k2Pi;

        float uu        = phi / k2Pi;
        float v         = theta / kPi;
        uint32_t col    = std::min(static_cast<uint32_t>(uu * kResU), static_cast<uint32_t>(kResU - 1));
        uint32_t row    = std::min(static_cast<uint32_t>(v  * kResV), static_cast<uint32_t>(kResV - 1));
        float sinTheta  = std::max(std::sin(theta), 1e-6f);

        float pdfRow = (m_marginal.integral > 0.f) ? m_marginal.func[row] / m_marginal.integral : 0.f;
        float pdfCol = (m_conditional[row].integral > 0.f)
                     ? m_conditional[row].func[col] / m_conditional[row].integral : 0.f;
        return (pdfRow * pdfCol * static_cast<float>(kResU * kResV)) / (k2Pi * kPi * sinTheta);
    }

    // -----------------------------------------------------------------------
    // Sample sun disc: uniform on spherical cap around sunDir
    // -----------------------------------------------------------------------
    Vec3f sampleSunCap(Vec2f u) const {
        const Vec3f& axis = m_sky.params().sunDir;
        float cosA = std::cos(m_sky.params().sunDiscAngle);

        // Uniform sampling on spherical cap
        float cosTheta = 1.f - u.x * (1.f - cosA);
        float sinTheta = std::sqrt(std::max(0.f, 1.f - cosTheta * cosTheta));
        float phi      = k2Pi * u.y;

        Vec3f t, bt;
        buildOrthonormalBasis(axis, t, bt);

        return normalize(t  * (sinTheta * std::cos(phi))
                       + bt * (sinTheta * std::sin(phi))
                       + axis * cosTheta);
    }

    bool inSunCap(Vec3f d) const {
        return dot(d, m_sky.params().sunDir) >= std::cos(m_sky.params().sunDiscAngle);
    }

    // -----------------------------------------------------------------------
    // MIS combined PDF
    // -----------------------------------------------------------------------
    float pdfCombined(Vec3f wi) const {
        float pSky = evalSkyGridPdf(wi);
        float pSun = inSunCap(wi) ? m_sunPdfCap : 0.f;
        return m_pSunMIS * pSun + (1.f - m_pSunMIS) * pSky;
    }

    // -----------------------------------------------------------------------
    // Power estimates for MIS weighting
    // -----------------------------------------------------------------------
    float estimateSunPower() const {
        Spectrum L = m_sky.sunDiscRadiance();
        float lum  = 0.2126f * L.x + 0.7152f * L.y + 0.0722f * L.z;
        return lum * m_sunCapArea;
    }

    float estimateSkyPower() const {
        // Already stored as the grid integral times 2*pi^2 / (kResU*kResV)
        return m_marginal.integral
             * (k2Pi * kPi / static_cast<float>(kResU * kResV));
    }

    // -----------------------------------------------------------------------
    // Members
    // -----------------------------------------------------------------------
    NishitaSky             m_sky;
    float                  m_sceneRadius;
    Vec3f                  m_sceneCenter;

    Dist1D                 m_marginal;
    std::vector<Dist1D>    m_conditional;

    float                  m_sunCapArea  = 0.f;
    float                  m_sunPdfCap   = 0.f;
    float                  m_sunPower    = 0.f;
    float                  m_skyPower    = 0.f;
    float                  m_pSunMIS     = 0.5f;  // fraction of samples toward sun disc
};

} // namespace anacapa
