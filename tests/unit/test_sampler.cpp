#include <gtest/gtest.h>
#include "../../src/sampling/HaltonSampler.h"
#include "../../src/sampling/PCGRng.h"
#include <cmath>
#include <vector>

using namespace anacapa;

TEST(PCGRng, ValuesInRange) {
    PCGRng rng{42};
    for (int i = 0; i < 10000; ++i) {
        float v = rng.nextFloat();
        EXPECT_GE(v, 0.f);
        EXPECT_LT(v, 1.f);
    }
}

TEST(PCGRng, Reproducible) {
    PCGRng a{12345}, b{12345};
    for (int i = 0; i < 100; ++i)
        EXPECT_EQ(a.nextUint32(), b.nextUint32());
}

// Chi-squared uniformity test on 1D Halton samples
TEST(HaltonSampler, Uniformity1D) {
    HaltonSampler sampler(1024, /*seed=*/0);
    constexpr int kBins   = 10;
    constexpr int kSamples = 1000;
    std::vector<int> bins(kBins, 0);

    for (int i = 0; i < kSamples; ++i) {
        sampler.startPixelSample(0, 0, i);
        float v = sampler.get1D();
        ASSERT_GE(v, 0.f);
        ASSERT_LT(v, 1.f);
        int bin = static_cast<int>(v * kBins);
        bins[std::min(bin, kBins - 1)]++;
    }

    // Chi-squared test: expected = kSamples/kBins per bin
    float expected = static_cast<float>(kSamples) / kBins;
    float chi2 = 0.f;
    for (int b : bins) {
        float diff = b - expected;
        chi2 += diff * diff / expected;
    }

    // Chi-squared critical value for 9 DOF at p=0.001 is ~27.88
    // A good low-discrepancy sequence should have chi2 << this.
    EXPECT_LT(chi2, 27.88f)
        << "Halton samples failed uniformity test (chi2=" << chi2 << ")";
}

TEST(HaltonSampler, DifferentPixelsDiffer) {
    HaltonSampler sampler(64, 0);

    sampler.startPixelSample(0, 0, 0);
    float v00 = sampler.get1D();

    sampler.startPixelSample(1, 0, 0);
    float v10 = sampler.get1D();

    sampler.startPixelSample(0, 1, 0);
    float v01 = sampler.get1D();

    // Different pixels should (almost certainly) give different values
    EXPECT_NE(v00, v10);
    EXPECT_NE(v00, v01);
}
