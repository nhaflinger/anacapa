#pragma once

#include <anacapa/core/Types.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// PixelAccumulator — per-pixel accumulation bucket
//
// Separate r/g/b/w atomics allow lock-free concurrent writes from
// multiple tile workers and BDPT light-subpath splats.
// Uses relaxed atomics since splat ordering doesn't matter — only the
// final sum is read, and that read happens after all workers have joined.
//
// For GPU migration (Phase 5):
//   - CUDA:  replace std::atomic<float> with device-side atomicAdd()
//   - Metal: replace with atomic_fetch_add_explicit on device float*
// ---------------------------------------------------------------------------
struct PixelAccumulator {
    std::atomic<float> r{0.f};
    std::atomic<float> g{0.f};
    std::atomic<float> b{0.f};
    std::atomic<float> weight{0.f};

    // Welford online variance — updated per-sample for adaptive sampling
    std::atomic<float> m2{0.f};    // Sum of squared deviations (luminance)
    std::atomic<uint32_t> count{0};

    void add(float fr, float fg, float fb, float w = 1.f) {
        r.fetch_add(fr * w, std::memory_order_relaxed);
        g.fetch_add(fg * w, std::memory_order_relaxed);
        b.fetch_add(fb * w, std::memory_order_relaxed);
        weight.fetch_add(w,  std::memory_order_relaxed);
    }

    Spectrum resolve() const {
        float w = weight.load(std::memory_order_relaxed);
        if (w <= 0.f) return {};
        float invW = 1.f / w;
        return {
            r.load(std::memory_order_relaxed) * invW,
            g.load(std::memory_order_relaxed) * invW,
            b.load(std::memory_order_relaxed) * invW
        };
    }
};

// ---------------------------------------------------------------------------
// TileBuffer — non-atomic per-tile accumulation (no contention within a tile)
//
// Each tile worker accumulates into its own TileBuffer, then merges into
// the main Film atomically via Film::mergeTile(). This eliminates atomic
// ops for camera-path contributions (the common case) and reserves atomics
// only for BDPT light-subpath splats which are inherently non-local.
// ---------------------------------------------------------------------------
struct TileBuffer {
    struct Sample {
        float r = 0.f, g = 0.f, b = 0.f, weight = 0.f;
    };

    uint32_t           x0, y0;       // Tile origin in film space
    uint32_t           width, height;
    std::vector<Sample> pixels;       // [y * width + x]

    TileBuffer(uint32_t x0, uint32_t y0, uint32_t w, uint32_t h)
        : x0(x0), y0(y0), width(w), height(h), pixels(w * h)
    {}

    void add(uint32_t localX, uint32_t localY,
             float r, float g, float b, float w = 1.f) {
        auto& p = pixels[localY * width + localX];
        p.r      += r * w;
        p.g      += g * w;
        p.b      += b * w;
        p.weight += w;
    }

    void add(uint32_t localX, uint32_t localY, Spectrum s, float w = 1.f) {
        add(localX, localY, s.x, s.y, s.z, w);
    }

    void clear() {
        for (auto& p : pixels) p = {};
    }
};

// ---------------------------------------------------------------------------
// Film — the main render target
// ---------------------------------------------------------------------------
class Film {
public:
    Film(uint32_t width, uint32_t height);

    uint32_t width()  const { return m_width; }
    uint32_t height() const { return m_height; }

    // Thread-safe splat — used by BDPT for light-subpath connections that
    // land at an arbitrary pixel (not necessarily the current tile).
    // Coordinates are in continuous film space [0, width) x [0, height).
    void splatPixel(float x, float y, Spectrum value);

    // Merge a completed TileBuffer into the film (called by tile worker)
    void mergeTile(const TileBuffer& tile);

    // Write finalized image to an EXR file via OpenImageIO.
    // Divides each pixel by its accumulated weight before writing.
    bool writeEXR(const std::string& path) const;

    // Raw resolved pixel access (for preview / post-processing)
    Spectrum getPixel(uint32_t x, uint32_t y) const;

private:
    uint32_t m_width, m_height;
    std::vector<PixelAccumulator> m_pixels;  // [y * width + x]

    bool inBounds(int x, int y) const {
        return x >= 0 && x < static_cast<int>(m_width)
            && y >= 0 && y < static_cast<int>(m_height);
    }
};

} // namespace anacapa
