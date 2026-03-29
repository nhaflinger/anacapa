#pragma once

#include <anacapa/core/Types.h>
#include <cstdint>
#include <memory>

namespace anacapa {

// ---------------------------------------------------------------------------
// SamplerState — POD view of sampler state
//
// Designed to be passed to GPU kernels by value (fits in registers).
// Each warp thread holds its own copy; no device-side heap needed.
// ---------------------------------------------------------------------------
struct alignas(64) SamplerState {
    uint64_t seed        = 0;
    uint32_t pixelX      = 0;
    uint32_t pixelY      = 0;
    uint32_t sampleIndex = 0;
    uint32_t dimension   = 0;
    uint8_t  opaque[40]  = {};  // Scramble bits, radical inverse state, etc.
};
static_assert(sizeof(SamplerState) == 64, "SamplerState must be one cache line");

// ---------------------------------------------------------------------------
// ISampler
//
// All samplers produce low-discrepancy sequences over [0,1).
// Clone() gives each tile worker an independent copy.
// startPixelSample() resets the dimension counter for each new (pixel, sample).
// ---------------------------------------------------------------------------
class ISampler {
public:
    virtual ~ISampler() = default;

    // Each tile worker calls clone() to get an independent instance
    virtual std::unique_ptr<ISampler> clone() const = 0;

    // Called once per (pixel, sample index) before any get1D/get2D calls
    virtual void startPixelSample(uint32_t pixelX, uint32_t pixelY,
                                   uint32_t sampleIndex) = 0;

    virtual float get1D() = 0;
    virtual Vec2f get2D() = 0;

    // POD snapshot for GPU kernel hand-off (Phase 5)
    virtual SamplerState getState() const = 0;
};

} // namespace anacapa
