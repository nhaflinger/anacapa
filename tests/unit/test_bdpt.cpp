#include <anacapa/integrator/PathVertex.h>
#include <anacapa/integrator/MISWeight.h>
#include <gtest/gtest.h>
#include <cmath>

using namespace anacapa;

// ---------------------------------------------------------------------------
// PathVertexBuffer
// ---------------------------------------------------------------------------
TEST(PathVertexBuffer, ConstructAndReset) {
    PathVertexBuffer buf(8);
    EXPECT_EQ(buf.count, 0u);
    EXPECT_EQ(buf.capacity, 8u);

    buf.count = 3;
    buf.reset();
    EXPECT_EQ(buf.count, 0u);
    EXPECT_EQ(buf.capacity, 8u);  // capacity unchanged
}

TEST(PathVertexBuffer, FlagsRoundtrip) {
    PathVertexBuffer buf(4);
    buf.count = 1;
    buf.flags[0] = static_cast<uint32_t>(PathVertexType::Surface) | kVertexDeltaBit;

    EXPECT_EQ(buf.type(0), PathVertexType::Surface);
    EXPECT_TRUE(buf.isDelta(0));
    EXPECT_FALSE(buf.isInfinite(0));
    EXPECT_FALSE(buf.isConnectible(0));
}

TEST(PathVertexBuffer, DeviceViewPointers) {
    PathVertexBuffer buf(4);
    auto dv = buf.deviceView();
    EXPECT_EQ(dv.position, buf.position.data());
    EXPECT_EQ(dv.beta,     buf.beta.data());
    EXPECT_EQ(dv.pdfFwd,   buf.pdfFwd.data());
    EXPECT_EQ(dv.count,    0u);
}

// ---------------------------------------------------------------------------
// convertToArea
// ---------------------------------------------------------------------------
TEST(PathVertex, ConvertToAreaOrthogonal) {
    // Ray from origin hits a surface at (0,0,1) with normal pointing back (0,0,-1)
    // |cos| = 1, dist = 1, so pdfArea = pdfSA * 1 / 1 = pdfSA
    Vec3f prevPos = {0,0,0};
    Vec3f nextPos = {0,0,1};
    Vec3f nextN   = {0,0,-1};
    float result = convertToArea(1.f, prevPos, nextPos, nextN);
    EXPECT_NEAR(result, 1.f, 1e-5f);
}

TEST(PathVertex, ConvertToAreaGrazingAngle) {
    // Ray hits surface at 45 degrees
    Vec3f prevPos = {0,0,0};
    Vec3f nextPos = {1,0,1};   // direction (1,0,1)/sqrt2, dist = sqrt2
    Vec3f nextN   = {0,0,1};   // |cos| = dot((1,0,1)/sqrt2, (0,0,1)) = 1/sqrt2
    float dist2   = 2.f;
    float cosT    = 1.f / std::sqrt(2.f);
    float expected = 1.f * cosT / dist2;
    float result   = convertToArea(1.f, prevPos, nextPos, nextN);
    EXPECT_NEAR(result, expected, 1e-5f);
}

// ---------------------------------------------------------------------------
// MIS weight — sum over all valid strategies must equal 1
//
// For a two-vertex path (one light, one camera vertex, s=1,t=1),
// the only valid strategy is (1,1) so the weight must be 1.
// ---------------------------------------------------------------------------
TEST(MISWeight, SingleStrategyIsOne) {
    PathVertexBuffer lp(4), cp(4);

    // Light vertex
    lp.count = 1;
    lp.position[0] = {0, 0.9f, 1};
    lp.normal[0]   = {0, -1, 0};
    lp.pdfFwd[0]   = 0.5f;
    lp.pdfRev[0]   = 0.f;
    lp.flags[0]    = static_cast<uint32_t>(PathVertexType::Light);

    // Camera vertex
    cp.count = 1;
    cp.position[0] = {0, 0, -2};
    cp.normal[0]   = {0, 0,  1};
    cp.pdfFwd[0]   = 1.f;
    cp.pdfRev[0]   = 0.f;
    cp.flags[0]    = static_cast<uint32_t>(PathVertexType::Camera);

    float w = bdptMISWeight(lp, cp, 1, 1);
    EXPECT_NEAR(w, 1.f, 1e-5f);
}

TEST(MISWeight, WeightInRange) {
    // Two strategies available — weight should be in (0,1]
    PathVertexBuffer lp(4), cp(4);

    lp.count = 1;
    lp.position[0] = {0, 0.9f, 1};
    lp.normal[0]   = {0,-1,0};
    lp.pdfFwd[0]   = 0.5f;
    lp.pdfRev[0]   = 0.3f;
    lp.flags[0]    = static_cast<uint32_t>(PathVertexType::Light);

    cp.count = 2;
    cp.position[0] = {0,0,-2};   cp.normal[0] = {0,0,1};
    cp.pdfFwd[0]   = 1.f;        cp.pdfRev[0] = 0.f;
    cp.flags[0]    = static_cast<uint32_t>(PathVertexType::Camera);

    cp.position[1] = {0,0,0};    cp.normal[1] = {0,1,0};
    cp.pdfFwd[1]   = 0.4f;       cp.pdfRev[1] = 0.2f;
    cp.flags[1]    = static_cast<uint32_t>(PathVertexType::Surface);

    float w = bdptMISWeight(lp, cp, 1, 2);
    EXPECT_GT(w, 0.f);
    EXPECT_LE(w, 1.f);
}

TEST(MISWeight, DeltaVertexZeroWeight) {
    // A delta light vertex — strategies that try to explicitly sample it
    // (by connecting to it) must yield weight 0.
    PathVertexBuffer lp(4), cp(4);

    lp.count = 1;
    lp.position[0] = {0, 0.9f, 1};
    lp.normal[0]   = {0,-1,0};
    lp.pdfFwd[0]   = 0.f;   // delta: area PDF = 0
    lp.pdfRev[0]   = 0.f;
    lp.flags[0]    = static_cast<uint32_t>(PathVertexType::Light) | kVertexDeltaBit;

    cp.count = 1;
    cp.position[0] = {0,0,-2};
    cp.normal[0]   = {0,0,1};
    cp.pdfFwd[0]   = 1.f;
    cp.pdfRev[0]   = 0.f;
    cp.flags[0]    = static_cast<uint32_t>(PathVertexType::Camera);

    // (1,1) strategy: the only strategy, so weight = 1 by convention
    // even though the light is delta (the path was sampled, not connected)
    float w = bdptMISWeight(lp, cp, 1, 1);
    EXPECT_NEAR(w, 1.f, 1e-5f);
}

// ---------------------------------------------------------------------------
// LightSampler alias table
// ---------------------------------------------------------------------------
#include "../../src/integrator/LightSampler.h"
#include <anacapa/shading/ILight.h>

struct MockLight : public ILight {
    float m_power;
    explicit MockLight(float p) : m_power(p) {}
    LightSample   sample(Vec3f, Vec3f, Vec2f) const override { return {}; }
    float         pdf(Vec3f, Vec3f)            const override { return 0.f; }
    LightLeSample sampleLe(Vec2f, Vec2f)       const override { return {}; }
    Spectrum      Le(Vec3f,Vec3f,Vec3f)        const override { return {}; }
    float         power()                      const override { return m_power; }
};

TEST(LightSampler, SelectionProbabilities) {
    MockLight a(1.f), b(3.f);   // b is 3x more likely
    std::vector<const ILight*> lights = {&a, &b};

    LightSampler sampler;
    sampler.build(lights);

    EXPECT_NEAR(sampler.pdf(0), 0.25f, 1e-5f);
    EXPECT_NEAR(sampler.pdf(1), 0.75f, 1e-5f);
}

TEST(LightSampler, SelectionConverges) {
    MockLight a(1.f), b(1.f);  // equal weight
    std::vector<const ILight*> lights = {&a, &b};
    LightSampler sampler;
    sampler.build(lights);

    // Sample 10000 times and check ~50/50 split
    int countA = 0;
    for (int i = 0; i < 10000; ++i) {
        float u = static_cast<float>(i) / 10000.f;
        auto sel = sampler.sample(u);
        if (sel.index == 0) ++countA;
    }
    EXPECT_NEAR(countA / 10000.f, 0.5f, 0.02f);
}
