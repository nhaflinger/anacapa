#pragma once

#include <anacapa/sampling/ISampler.h>
#include "PCGRng.h"
#include <array>
#include <cstdint>
#include <memory>

namespace anacapa {

// ---------------------------------------------------------------------------
// HaltonSampler
//
// Low-discrepancy sampler based on the Halton sequence.
// Dimension d uses base prime[d]; first two dimensions (2, 3) cover 2D samples.
// Scrambling (Owen-style via PCG) breaks inter-pixel correlation.
//
// Reference: Kollig & Keller 2002, "Efficient Multidimensional Sampling"
// ---------------------------------------------------------------------------
class HaltonSampler : public ISampler {
public:
    static constexpr uint32_t kMaxDimensions = 128;

    explicit HaltonSampler(uint32_t samplesPerPixel, uint64_t seed = 0)
        : m_samplesPerPixel(samplesPerPixel)
        , m_seed(seed)
    {}

    std::unique_ptr<ISampler> clone() const override {
        return std::make_unique<HaltonSampler>(*this);
    }

    void startPixelSample(uint32_t pixelX, uint32_t pixelY,
                           uint32_t sampleIndex) override {
        m_pixelX      = pixelX;
        m_pixelY      = pixelY;
        m_sampleIndex = sampleIndex;
        m_dimension   = 0;
        // Per-pixel scramble seed derived from pixel coords + global seed
        m_rng = PCGRng{m_seed ^ (uint64_t(pixelX) << 32 | pixelY)};
    }

    float get1D() override {
        uint32_t dim = m_dimension++;
        if (dim >= kMaxDimensions) {
            // Fallback to pure random beyond max dimensions
            return m_rng.nextFloat();
        }
        return scrambledRadicalInverse(dim, m_sampleIndex);
    }

    Vec2f get2D() override {
        return {get1D(), get1D()};
    }

    SamplerState getState() const override {
        SamplerState s;
        s.seed        = m_seed;
        s.pixelX      = m_pixelX;
        s.pixelY      = m_pixelY;
        s.sampleIndex = m_sampleIndex;
        s.dimension   = m_dimension;
        return s;
    }

private:
    // First 128 prime numbers (bases for Halton sequence)
    static constexpr std::array<uint32_t, 128> kPrimes = {
          2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,  41,
         43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97, 101,
        103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
        173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
        241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313,
        317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397,
        401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
        479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569,
        571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643,
        647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719
    };

    // Radical inverse in base b
    static float radicalInverse(uint32_t b, uint64_t n) {
        float invB = 1.f / static_cast<float>(b);
        float invBN = 1.f;
        float result = 0.f;
        while (n > 0) {
            uint64_t d = n % b;
            n /= b;
            invBN *= invB;
            result += static_cast<float>(d) * invBN;
        }
        return result;
    }

    // Scrambled radical inverse: XOR the digit sequence with per-pixel bits
    float scrambledRadicalInverse(uint32_t dim, uint64_t n) const {
        uint32_t base = kPrimes[dim];
        // Use the rng to derive a per-pixel, per-dimension scramble offset
        PCGRng rng{m_seed ^ (uint64_t(m_pixelX) * 2654435761ULL)
                           ^ (uint64_t(m_pixelY) * 805459861ULL)
                           ^ (uint64_t(dim)       * 999999893ULL)};
        uint64_t scramble = rng.nextUint32();
        return radicalInverse(base, n ^ scramble);
    }

    uint32_t m_samplesPerPixel = 1;
    uint64_t m_seed            = 0;
    uint32_t m_pixelX          = 0;
    uint32_t m_pixelY          = 0;
    uint32_t m_sampleIndex     = 0;
    uint32_t m_dimension       = 0;
    mutable PCGRng m_rng;
};

} // namespace anacapa
