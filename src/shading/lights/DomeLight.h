#pragma once

#include <anacapa/shading/ILight.h>
#include <anacapa/core/Types.h>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

namespace anacapa {

// ---------------------------------------------------------------------------
// Distribution1D — 1D piecewise-constant distribution for importance sampling
//
// Given N function values, builds a CDF and supports:
//   - sampleDiscrete: returns (bin index, pdf, remapped u within bin)
//   - integral:       sum of all function values (unnormalized)
// ---------------------------------------------------------------------------
struct Distribution1D {
    std::vector<float> func;   // raw function values [0..n)
    std::vector<float> cdf;    // cdf[0]=0, cdf[i+1] = cdf[i]+func[i], cdf[n]=integral (then normalized)
    float integral = 0.f;
    uint32_t n = 0;

    void build(const float* values, uint32_t count) {
        n = count;
        func.assign(values, values + count);
        cdf.resize(count + 1);
        cdf[0] = 0.f;
        for (uint32_t i = 0; i < count; ++i)
            cdf[i + 1] = cdf[i] + func[i];
        integral = cdf[count];
        if (integral > 0.f) {
            float invInt = 1.f / integral;
            for (auto& c : cdf) c *= invInt;
        }
    }

    // Returns bin index in [0, n), pdf for that bin, and xi remapped in [0,1).
    uint32_t sample(float u, float& pdf, float& uRemapped) const {
        // Binary search for u in the CDF
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        int idx = static_cast<int>(it - cdf.begin()) - 1;
        idx     = std::max(0, std::min(idx, static_cast<int>(n) - 1));

        float binWidth = cdf[idx + 1] - cdf[idx];
        uRemapped = (binWidth > 1e-12f)
            ? (u - cdf[idx]) / binWidth
            : 0.f;
        uRemapped = std::max(0.f, std::min(uRemapped, 1.f - 1e-7f));

        pdf = (integral > 0.f) ? func[idx] / integral : 0.f;
        return static_cast<uint32_t>(idx);
    }
};

// ---------------------------------------------------------------------------
// DomeLight — equirectangular HDRI environment map
//
// Loads a lat-long EXR or HDR image via OpenImageIO and builds a 2D
// importance sampling distribution weighted by luminance * sin(theta).
//
// UV ↔ world direction convention:
//   u = phi / (2*pi),  phi ∈ [0, 2pi)
//   v = theta / pi,    theta ∈ [0, pi], theta=0 at +Y, theta=pi at -Y
//
//   dir.x = sin(theta) * sin(phi)
//   dir.y = cos(theta)
//   dir.z = sin(theta) * cos(phi)
//
// PDF in solid angle measure:
//   p_omega = p_uv / (2*pi^2 * sin(theta))
//   where p_uv = (luminance * sin(theta)) / integral_2D
//   → p_omega = luminance / (integral_2D * 2 * pi^2)
//
// When the image cannot be loaded, the light emits a constant grey.
// ---------------------------------------------------------------------------
class DomeLight : public ILight {
public:
    static constexpr float kPi    = 3.14159265358979323846f;
    static constexpr float kInvPi = 0.31830988618379067154f;
    static constexpr float k2Pi   = 6.28318530717958647692f;

    // sceneRadius / sceneCenter: bounding sphere of the scene, used by sampleLe
    // to place ray origins on a disk facing the scene.
    explicit DomeLight(const std::string& hdriPath,
                       float intensity    = 1.f,
                       float sceneRadius  = 10.f,
                       Vec3f sceneCenter  = {})
        : m_intensity(intensity)
        , m_sceneRadius(sceneRadius)
        , m_sceneCenter(sceneCenter)
    {
        // Identity rotation by default
        m_rot[0] = {1,0,0}; m_rot[1] = {0,1,0}; m_rot[2] = {0,0,1};
        if (!hdriPath.empty())
            loadImage(hdriPath);
        buildDistribution();
    }

    // Set a world-to-envmap rotation (3x3, column-major like Mat4f).
    // When a world-space direction wo is looked up in the envmap, it is first
    // transformed by this matrix.  Set to the DomeLight's world xform (upper 3x3
    // of the local-to-world matrix, transposed to get world-to-local).
    void setRotation(Vec3f col0, Vec3f col1, Vec3f col2) {
        // Store as rows for efficient dot-product lookup
        // m_rot[i] = row i of the rotation matrix
        // We want to rotate wo: result = R * wo
        m_rot[0] = col0;   // first row  = first column of R
        m_rot[1] = col1;   // second row = second column of R
        m_rot[2] = col2;   // third row  = third column of R
    }

    // -----------------------------------------------------------------------
    // Direct-lighting interface: sample a direction toward the environment
    // -----------------------------------------------------------------------
    LightSample sample(Vec3f /*p*/, Vec3f /*n*/, Vec2f u) const override {
        float pdfDir;
        Vec3f wi = sampleDirection(u, pdfDir);
        if (pdfDir <= 0.f) return {};

        LightSample s;
        s.Li      = evalEnvmap(wi) * m_intensity;
        s.wi      = wi;
        s.pdf     = pdfDir;
        s.dist    = 1e10f;
        s.isDelta = false;
        return s;
    }

    float pdf(Vec3f /*p*/, Vec3f wi) const override {
        return evalPdf(wi);
    }

    // -----------------------------------------------------------------------
    // BDPT: emit a ray from the environment
    //
    // Strategy: sample an inward-facing direction from the distribution, then
    // place the ray origin on a disk perpendicular to that direction at the
    // far side of the scene bounding sphere.
    // -----------------------------------------------------------------------
    LightLeSample sampleLe(Vec2f uPos, Vec2f uDir) const override {
        float pdfDir;
        Vec3f envDir = sampleDirection(uDir, pdfDir);  // direction FROM environment TO scene
        if (pdfDir <= 0.f) return {};

        // Disk perpendicular to envDir, centered at sceneCenter + envDir * sceneRadius
        Vec3f t, bt;
        buildOrthonormalBasis(-envDir, t, bt);  // -envDir = disk normal (faces into scene)

        float r   = m_sceneRadius * std::sqrt(uPos.x);
        float phi = k2Pi * uPos.y;
        Vec3f diskCenter = m_sceneCenter + envDir * m_sceneRadius;
        Vec3f pos = diskCenter
                  + t  * (r * std::cos(phi))
                  + bt * (r * std::sin(phi));

        float diskArea = kPi * m_sceneRadius * m_sceneRadius;

        LightLeSample s;
        s.Le     = evalEnvmap(envDir) * m_intensity;
        s.pos    = pos;
        s.normal = -envDir;       // disk normal points inward (toward scene)
        s.dir    = -envDir;       // photon travels inward
        s.pdfPos = 1.f / diskArea;
        s.pdfDir = pdfDir;
        return s;
    }

    // Emitted radiance in direction wo from the environment
    // wo = direction from scene point toward the sky
    Spectrum Le(Vec3f /*pos*/, Vec3f /*normal*/, Vec3f wo) const override {
        return evalEnvmap(wo) * m_intensity;
    }

    float power() const override {
        // Approximate: integral * 4*pi (sphere solid angle) * intensity
        return m_envIntegral * 4.f * kPi * m_intensity;
    }

    bool isInfinite() const override { return true;  }
    bool isDelta()    const override { return false; }

    // -----------------------------------------------------------------------
    // Accessors (for RenderSession and GPU upload)
    // -----------------------------------------------------------------------
    void setSceneRadius(float r) { m_sceneRadius = r; }
    void setSceneCenter(Vec3f c) { m_sceneCenter = c; }
    float intensity() const { return m_intensity; }

    // Raw pixel data for GPU upload (RGB float, row-major, row 0 = zenith)
    const float* pixels()    const { return m_pixels.data(); }
    uint32_t     envWidth()  const { return m_width; }
    uint32_t     envHeight() const { return m_height; }

    // Rotation rows: world-to-envmap matrix stored as three row vectors
    void getRotation(Vec3f& r0, Vec3f& r1, Vec3f& r2) const {
        r0 = m_rot[0]; r1 = m_rot[1]; r2 = m_rot[2];
    }

private:
    // Apply the stored rotation: world direction → envmap local direction
    Vec3f rotateToEnv(Vec3f wo) const {
        return {
            dot(m_rot[0], wo),
            dot(m_rot[1], wo),
            dot(m_rot[2], wo)
        };
    }
    // Apply inverse rotation (transpose of orthogonal matrix): envmap dir → world dir
    Vec3f rotateFromEnv(Vec3f v) const {
        // m_rot rows are columns of the original matrix, so transpose = original columns
        return {
            m_rot[0].x * v.x + m_rot[1].x * v.y + m_rot[2].x * v.z,
            m_rot[0].y * v.x + m_rot[1].y * v.y + m_rot[2].y * v.z,
            m_rot[0].z * v.x + m_rot[1].z * v.y + m_rot[2].z * v.z
        };
    }

    // -----------------------------------------------------------------------
    // Image loading via OpenImageIO (defined in DomeLight.cpp)
    // -----------------------------------------------------------------------
    void loadImage(const std::string& path);

    // -----------------------------------------------------------------------
    // Build 2D piecewise-constant distribution
    // -----------------------------------------------------------------------
    void buildDistribution() {
        if (m_width == 0 || m_height == 0) {
            // Fallback: uniform distribution (1x1)
            m_width = m_height = 1;
            m_pixels = {1.f, 1.f, 1.f};
        }

        // f[r][c] = luminance(pixel) * sin(theta_r)
        // Used as the 2D sampling weight.
        std::vector<float> rowSums(m_height);
        m_conditional.resize(m_height);

        std::vector<float> rowFunc(m_width);

        for (uint32_t r = 0; r < m_height; ++r) {
            float theta = kPi * (r + 0.5f) / static_cast<float>(m_height);
            float sinTheta = std::sin(theta);
            for (uint32_t c = 0; c < m_width; ++c) {
                const float* px = pixelPtr(r, c);
                float lum = 0.2126f * px[0] + 0.7152f * px[1] + 0.0722f * px[2];
                rowFunc[c] = lum * sinTheta;
            }
            m_conditional[r].build(rowFunc.data(), m_width);
            rowSums[r] = m_conditional[r].integral;
        }

        m_marginal.build(rowSums.data(), m_height);

        // Total integral for power() estimate
        m_envIntegral = m_marginal.integral * k2Pi * kPi
                      / static_cast<float>(m_width * m_height);
    }

    // -----------------------------------------------------------------------
    // Sample a world-space direction from the distribution
    // -----------------------------------------------------------------------
    Vec3f sampleDirection(Vec2f u, float& pdfSolidAngle) const {
        float pdfRow, pdfCol, uRemap_r, uRemap_c;

        uint32_t row = m_marginal.sample(u.y, pdfRow, uRemap_r);
        uint32_t col = m_conditional[row].sample(u.x, pdfCol, uRemap_c);

        // Continuous UV: center of sampled bin + remapped offset
        float v = (static_cast<float>(row) + uRemap_r) / static_cast<float>(m_height);
        float uu = (static_cast<float>(col) + uRemap_c) / static_cast<float>(m_width);

        float theta    = v * kPi;
        float phi      = uu * k2Pi;
        float sinTheta = std::sin(theta);
        float cosTheta = std::cos(theta);

        // Direction in envmap local space, then rotate to world space
        Vec3f envDir = {
            sinTheta * std::sin(phi),
            cosTheta,
            sinTheta * std::cos(phi)
        };
        Vec3f dir = rotateFromEnv(envDir);

        // p_uv = pdfRow * pdfCol  (marginal * conditional, already normalized)
        // p_omega = p_uv / (2*pi^2 * sinTheta)
        float sinThetaAbs = std::max(sinTheta, 1e-6f);
        pdfSolidAngle = (pdfRow * pdfCol * static_cast<float>(m_width * m_height))
                      / (k2Pi * kPi * sinThetaAbs);

        if (pdfSolidAngle <= 0.f) pdfSolidAngle = 0.f;
        return dir;
    }

    // -----------------------------------------------------------------------
    // Evaluate solid-angle PDF for a given world direction
    // -----------------------------------------------------------------------
    float evalPdf(Vec3f wo) const {
        Vec3f local = rotateToEnv(wo);
        float theta = std::acos(std::max(-1.f, std::min(1.f, local.y)));
        float phi   = std::atan2(local.x, local.z);
        if (phi < 0.f) phi += k2Pi;

        float uu = phi / k2Pi;
        float v  = theta / kPi;

        uint32_t col = std::min(static_cast<uint32_t>(uu * m_width),  m_width  - 1u);
        uint32_t row = std::min(static_cast<uint32_t>(v  * m_height), m_height - 1u);

        float sinTheta = std::max(std::sin(theta), 1e-6f);

        float pdfRow = (m_marginal.integral > 0.f)
            ? m_marginal.func[row] / m_marginal.integral : 0.f;
        float pdfCol = (m_conditional[row].integral > 0.f)
            ? m_conditional[row].func[col] / m_conditional[row].integral : 0.f;

        return (pdfRow * pdfCol * static_cast<float>(m_width * m_height))
             / (k2Pi * kPi * sinTheta);
    }

    // -----------------------------------------------------------------------
    // Evaluate environment map color at world direction wo
    // -----------------------------------------------------------------------
    Spectrum evalEnvmap(Vec3f wo) const {
        Vec3f local = rotateToEnv(wo);
        float theta = std::acos(std::max(-1.f, std::min(1.f, local.y)));
        float phi   = std::atan2(local.x, local.z);
        if (phi < 0.f) phi += k2Pi;

        float uu = phi / k2Pi;
        float v  = theta / kPi;

        // Bilinear interpolation
        float fc = uu * static_cast<float>(m_width)  - 0.5f;
        float fr = v  * static_cast<float>(m_height) - 0.5f;

        int c0 = static_cast<int>(std::floor(fc));
        int r0 = static_cast<int>(std::floor(fr));
        float tc = fc - c0;
        float tr = fr - r0;

        auto clampC = [&](int c) { return static_cast<uint32_t>((c % static_cast<int>(m_width) + static_cast<int>(m_width)) % static_cast<int>(m_width)); };
        auto clampR = [&](int r) { return static_cast<uint32_t>(std::max(0, std::min(r, static_cast<int>(m_height) - 1))); };

        uint32_t c1 = clampC(c0 + 1);
        uint32_t c0u = clampC(c0);
        uint32_t r1u = clampR(r0 + 1);
        uint32_t r0u = clampR(r0);

        const float* p00 = pixelPtr(r0u, c0u);
        const float* p01 = pixelPtr(r0u, c1);
        const float* p10 = pixelPtr(r1u, c0u);
        const float* p11 = pixelPtr(r1u, c1);

        Spectrum s;
        for (int ch = 0; ch < 3; ++ch) {
            s[ch] = (1.f - tr) * ((1.f - tc) * p00[ch] + tc * p01[ch])
                  +        tr  * ((1.f - tc) * p10[ch] + tc * p11[ch]);
            s[ch] = std::max(0.f, s[ch]);
        }
        return s;
    }

    const float* pixelPtr(uint32_t row, uint32_t col) const {
        return m_pixels.data() + (row * m_width + col) * 3u;
    }

    // -----------------------------------------------------------------------
    // Member data
    // -----------------------------------------------------------------------
    std::vector<float>         m_pixels;         // RGB float, row-major
    uint32_t                   m_width  = 0;
    uint32_t                   m_height = 0;

    std::vector<Distribution1D> m_conditional;   // one 1D dist per row
    Distribution1D              m_marginal;       // distribution over rows

    float    m_intensity    = 1.f;
    float    m_sceneRadius  = 10.f;
    Vec3f    m_sceneCenter  = {};
    float    m_envIntegral  = 1.f;               // for power()

    // Rotation from world space to envmap local space (rows of rotation matrix).
    // Default = identity.  Set via setRotation() from the USD DomeLight xform.
    Vec3f    m_rot[3] = {{1,0,0},{0,1,0},{0,0,1}};
};

} // namespace anacapa
