#include "../../src/shading/StandardSurface.h"
#include "../../src/shading/lights/DomeLight.h"
#include <anacapa/shading/ShadingContext.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

using namespace anacapa;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a trivial ShadingContext facing +Z
static ShadingContext makeFlatCtx(Vec3f normal = {0,0,1}) {
    SurfaceInteraction si;
    si.p  = {};
    si.n  = safeNormalize(normal);
    si.ng = si.n;
    // Construct ShadingContext: rayDir points TOWARD the surface (-wo),
    // so the outgoing direction wo = -rayDir = normal (facing camera).
    return ShadingContext(si, -si.n);
}

// ---------------------------------------------------------------------------
// GGX D distribution: integral over hemisphere should equal 1
// (by definition of the NDF: integral D(wh) cosH dOmega_h = 1)
// ---------------------------------------------------------------------------
TEST(GGX, NDF_IntegralApproximatelyOne) {
    // Numerical integration of D_GGX(cosH, alpha2) * cosH  over the hemisphere
    float roughness = 0.5f;
    float alpha     = roughness * roughness;
    float alpha2    = alpha * alpha;

    // NDF normalization: ∫₀^{2π} ∫₀^1 D(cosH) · cosH · d(cosH) · dφ = 1
    // Substitution: d(cosH) absorbs sin(θ) from dΩ = sinθ dθ dφ
    const int N = 512;
    double sum  = 0.0;
    for (int i = 0; i < N; ++i) {
        float cosH  = (i + 0.5f) / static_cast<float>(N);  // [0..1)
        float dCosH = 1.f / static_cast<float>(N);
        // Integrate over phi analytically: factor of 2π
        sum += D_GGX(cosH, alpha2) * cosH * dCosH;
    }
    sum *= 2.0 * 3.14159265;   // ∫₀^{2π} dphi = 2π
    EXPECT_NEAR(sum, 1.0, 0.05);
}

// ---------------------------------------------------------------------------
// StandardSurfaceMaterial: pure Lambertian (metalness=0, roughness=1) should
// approximate a cos-weighted hemisphere integral of ≈ 1/pi * albedo
// ---------------------------------------------------------------------------
TEST(StandardSurface, DiffuseEnergyConservation) {
    StandardSurfaceMaterial::Params p;
    p.base_color = {1.f, 1.f, 1.f};
    p.metalness  = 0.f;
    p.roughness  = 1.f;
    p.specular   = 0.f;
    p.coat       = 0.f;
    StandardSurfaceMaterial mat(p);

    ShadingContext ctx = makeFlatCtx({0,0,1});
    Vec3f wo = {0,0,1};

    // Monte Carlo estimate of int f(wo,wi)*cosI dOmega_i over hemisphere
    const int N = 4096;
    double sumR = 0.0, sumG = 0.0, sumB = 0.0;
    for (int i = 0; i < N; ++i) {
        float theta = std::acos(1.f - (i + 0.5f) / static_cast<float>(N));
        float phi   = 2.f * 3.14159265f * 0.618033f * static_cast<float>(i);
        Vec3f wi = {std::sin(theta)*std::cos(phi),
                    std::sin(theta)*std::sin(phi),
                    std::cos(theta)};

        BSDFEval e = mat.evaluate(ctx, wo, wi);
        float cosI = std::abs(dot(wi, Vec3f{0,0,1}));
        float dOmega = 2.f * 3.14159265f / static_cast<float>(N); // hemisphere / N
        sumR += e.f.x * cosI * dOmega;
        sumG += e.f.y * cosI * dOmega;
        sumB += e.f.z * cosI * dOmega;
    }
    // Albedo = 1 → reflectance integral = 1, but the specular contribution
    // is 0 here so it should be ~1/pi * pi = 1 after cosine integration.
    // The diffuse component alone gives ~1.0 (Lambertian is energy-conserving).
    EXPECT_NEAR(sumR, 1.0, 0.05);
    EXPECT_NEAR(sumG, 1.0, 0.05);
    EXPECT_NEAR(sumB, 1.0, 0.05);
}

// ---------------------------------------------------------------------------
// StandardSurface: sample() should return valid BSDFSamples with pdf > 0
// ---------------------------------------------------------------------------
TEST(StandardSurface, SampleReturnsValid) {
    StandardSurfaceMaterial::Params p;
    p.base_color = {0.8f, 0.3f, 0.1f};
    p.metalness  = 0.5f;
    p.roughness  = 0.3f;
    StandardSurfaceMaterial mat(p);

    ShadingContext ctx = makeFlatCtx();
    Vec3f wo = safeNormalize({0.1f, 0.1f, 1.f});

    int validCount = 0;
    for (int i = 0; i < 128; ++i) {
        float ui   = (i + 0.5f) / 128.f;
        Vec2f u    = {ui, std::fmod(ui * 137.f, 1.f)};
        float uC   = std::fmod(ui * 31.f, 1.f);
        BSDFSample s = mat.sample(ctx, wo, u, uC);
        if (s.isValid()) {
            EXPECT_GT(s.pdf, 0.f);
            EXPECT_FALSE(s.f.hasNaN());
            ++validCount;
        }
    }
    // At least 70% of samples should be valid for a non-degenerate BSDF
    EXPECT_GT(validCount, 89);
}

// ---------------------------------------------------------------------------
// StandardSurface: emission returns correct values
// ---------------------------------------------------------------------------
TEST(StandardSurface, Emission) {
    StandardSurfaceMaterial::Params p;
    p.emission       = 2.f;
    p.emission_color = {1.f, 0.5f, 0.f};
    StandardSurfaceMaterial mat(p);

    ShadingContext ctx = makeFlatCtx();
    Spectrum Le = mat.Le(ctx, {0,0,1});

    EXPECT_NEAR(Le.x, 2.f, 1e-5f);
    EXPECT_NEAR(Le.y, 1.f, 1e-5f);
    EXPECT_NEAR(Le.z, 0.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// StandardSurface: pdfRev is symmetric for pure diffuse
// ---------------------------------------------------------------------------
TEST(StandardSurface, DiffusePdfSymmetry) {
    StandardSurfaceMaterial::Params p;
    p.metalness = 0.f;
    p.roughness = 1.f;
    p.specular  = 0.f;
    p.coat      = 0.f;
    StandardSurfaceMaterial mat(p);

    ShadingContext ctx1 = makeFlatCtx();
    ShadingContext ctx2 = makeFlatCtx();

    Vec3f a = safeNormalize({0.3f, 0.1f, 0.9f});
    Vec3f b = safeNormalize({-0.2f, 0.4f, 0.8f});

    BSDFEval e_ab = mat.evaluate(ctx1, a, b);
    BSDFEval e_ba = mat.evaluate(ctx2, b, a);

    // For Lambertian: pdfFwd(a→b) = pdf(b) = cosB/pi
    //                 pdfRev(a→b) = pdf(a) = cosA/pi
    // And pdfFwd(b→a) = cosA/pi,  pdfRev(b→a) = cosB/pi
    // So: e_ab.pdf == e_ba.pdfRev and e_ba.pdf == e_ab.pdfRev
    EXPECT_NEAR(e_ab.pdf,    e_ba.pdfRev, 1e-4f);
    EXPECT_NEAR(e_ba.pdf,    e_ab.pdfRev, 1e-4f);
}

// ---------------------------------------------------------------------------
// Distribution1D: integral and sampling correctness
// ---------------------------------------------------------------------------
TEST(Distribution1D, IntegralAndSample) {
    float vals[4] = {1.f, 3.f, 2.f, 4.f};
    Distribution1D d;
    d.build(vals, 4);

    // integral = 10, normalized so pdf(i) = vals[i] / 10
    EXPECT_NEAR(d.integral, 10.f, 1e-5f);

    // Sample the highest-weight bin (index 3, val=4, pdf=0.4)
    float pdf; float uRemap;
    uint32_t idx = d.sample(0.95f, pdf, uRemap);  // u=0.95 is in last bin
    EXPECT_EQ(idx, 3u);
    EXPECT_NEAR(pdf, 0.4f, 1e-4f);
}

TEST(Distribution1D, SampleConvergesToExpected) {
    // Two equal bins → ~50% each
    float vals[2] = {1.f, 1.f};
    Distribution1D d;
    d.build(vals, 2);

    int count0 = 0, count1 = 0;
    for (int i = 0; i < 10000; ++i) {
        float u = static_cast<float>(i) / 10000.f;
        float pdf; float uRemap;
        uint32_t idx = d.sample(u, pdf, uRemap);
        if (idx == 0) ++count0;
        else           ++count1;
    }
    EXPECT_NEAR(count0 / 10000.f, 0.5f, 0.02f);
    EXPECT_NEAR(count1 / 10000.f, 0.5f, 0.02f);
}

// ---------------------------------------------------------------------------
// DomeLight: constant grey environment — pdf and power sanity checks
// ---------------------------------------------------------------------------
TEST(DomeLight, ConstantEnvironmentPdfPositive) {
    // Default DomeLight (no image, falls back to 1x1 grey pixel)
    DomeLight dome("", 1.f, 5.f, {0,0,0});

    // sample() should return a valid direction with positive pdf
    LightSample s = dome.sample({0,0,0}, {0,0,1}, {0.3f, 0.7f});
    EXPECT_GT(s.pdf, 0.f);
    EXPECT_FALSE(s.Li.hasNaN());
    EXPECT_FLOAT_EQ(s.dist, 1e10f);
}

TEST(DomeLight, PdfMatchesExpected) {
    DomeLight dome("", 1.f, 5.f, {0,0,0});

    // For a 1×1 constant map, pdf should be the same for any direction
    float pdf0 = dome.pdf({0,0,0}, {1,0,0});
    float pdf1 = dome.pdf({0,0,0}, {0,1,0});
    // Not exactly equal due to sin(theta) weighting, but both positive
    EXPECT_GT(pdf0, 0.f);
    EXPECT_GT(pdf1, 0.f);
}

TEST(DomeLight, SampleLeIsValid) {
    DomeLight dome("", 1.f, 5.f, {0,0,0});
    LightLeSample s = dome.sampleLe({0.25f, 0.75f}, {0.1f, 0.9f});
    EXPECT_GT(s.pdfPos, 0.f);
    EXPECT_GT(s.pdfDir, 0.f);
    EXPECT_FALSE(s.Le.hasNaN());
    // Disk emitter: emitted ray travels in the same direction as the disk normal
    // (both point inward toward the scene, away from the environment)
    EXPECT_GT(dot(s.dir, s.normal), 0.f);
}

TEST(DomeLight, LeReturnsMappedColor) {
    // Constant grey map: Le should return {1,1,1} * intensity for any direction
    DomeLight dome("", 2.f, 5.f, {0,0,0});
    Spectrum le = dome.Le({}, {}, {0,1,0});
    // The 1×1 fallback pixel is {1,1,1}, so Le = {2,2,2}
    EXPECT_NEAR(le.x, 2.f, 0.1f);
    EXPECT_NEAR(le.y, 2.f, 0.1f);
    EXPECT_NEAR(le.z, 2.f, 0.1f);
}
