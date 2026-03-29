#pragma once

#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// PCG32 — Permuted Congruential Generator
//
// Fast, high-quality 32-bit random number generator.
// Used as the stochastic fallback inside samplers and for Russian roulette.
// Reference: O'Neill, "PCG: A Family of Simple Fast Space-Efficient
//            Statistically Good Algorithms for Random Number Generation"
// ---------------------------------------------------------------------------
struct PCGRng {
    uint64_t state = 0x853c49e6748fea9bULL;
    uint64_t inc   = 0xda3e39cb94b95bdbULL;

    PCGRng() = default;

    explicit PCGRng(uint64_t seed, uint64_t seq = 1) {
        state = 0;
        inc = (seq << 1u) | 1u;
        nextUint32();
        state += seed;
        nextUint32();
    }

    uint32_t nextUint32() {
        uint64_t old = state;
        state = old * 6364136223846793005ULL + inc;
        uint32_t xorshifted = static_cast<uint32_t>(((old >> 18u) ^ old) >> 27u);
        uint32_t rot        = static_cast<uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    // Uniform float in [0, 1)
    float nextFloat() {
        // Multiply by 2^-32
        return static_cast<float>(nextUint32()) * 2.3283064365386963e-10f;
    }

    // Uniform integer in [0, bound)
    uint32_t nextUint(uint32_t bound) {
        // Rejection sampling to avoid modulo bias
        uint32_t threshold = (~bound + 1u) % bound;
        while (true) {
            uint32_t r = nextUint32();
            if (r >= threshold) return r % bound;
        }
    }
};

} // namespace anacapa
