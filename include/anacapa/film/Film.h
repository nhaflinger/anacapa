#pragma once

#include <anacapa/core/Types.h>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>

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

    // Per-pixel variance tracking for adaptive sampling.
    // sumLumSq accumulates sum(luminance(sample)^2) across all merged tiles.
    // Combined with weight (= total sample count) and resolved luminance,
    // varianceAt() computes: E[L^2] - E[L]^2.
    std::atomic<float> sumLumSq{0.f};
    std::atomic<uint32_t> count{0};  // reserved / unused

#ifndef __CUDACC__
    void add(float fr, float fg, float fb, float w = 1.f) {
        r.fetch_add(fr * w, std::memory_order_relaxed);
        g.fetch_add(fg * w, std::memory_order_relaxed);
        b.fetch_add(fb * w, std::memory_order_relaxed);
        weight.fetch_add(w,  std::memory_order_relaxed);
    }
#endif

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
//
// albedo and normal are auxiliary buffers for denoising:
//   albedo — diffuse reflectance at the first camera-ray hit
//   normal — world-space normal at the first camera-ray hit (signed, unit)
// Both are averaged over spp and passed to OIDN as denoising hints.
// ---------------------------------------------------------------------------
struct TileBuffer {
    struct Sample {
        float r = 0.f, g = 0.f, b = 0.f, weight = 0.f;
    };

    // Denoising AOV sample — clamped average, not weighted sum
    struct AOVSample {
        float r = 0.f, g = 0.f, b = 0.f;
        uint32_t count = 0;
    };

    uint32_t           x0, y0;       // Tile origin in film space
    uint32_t           width, height;
    std::vector<Sample>    pixels;       // [y * width + x]
    std::vector<AOVSample> albedo;       // first-hit diffuse reflectance
    std::vector<AOVSample> normals;      // first-hit world-space normal
    std::vector<float>     sumLumSq;     // sum(luminance(sample)^2) per pixel

    TileBuffer(uint32_t x0, uint32_t y0, uint32_t w, uint32_t h)
        : x0(x0), y0(y0), width(w), height(h)
        , pixels(w * h), albedo(w * h), normals(w * h), sumLumSq(w * h, 0.f)
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

    // Record denoising AOVs (averaged, not splatted — only first hit per ray)
    void addAlbedo(uint32_t localX, uint32_t localY, Spectrum s) {
        auto& a = albedo[localY * width + localX];
        a.r += s.x; a.g += s.y; a.b += s.z; ++a.count;
    }

    void addNormal(uint32_t localX, uint32_t localY, Vec3f n) {
        auto& a = normals[localY * width + localX];
        a.r += n.x; a.g += n.y; a.b += n.z; ++a.count;
    }

    void addLumSq(uint32_t localX, uint32_t localY, float v) {
        sumLumSq[localY * width + localX] += v;
    }

    void clear() {
        for (auto& p : pixels)   p = {};
        for (auto& a : albedo)   a = {};
        for (auto& n : normals)  n = {};
        for (auto& v : sumLumSq) v = 0.f;
    }
};

// ---------------------------------------------------------------------------
// DenoiseOptions — controls Film::denoise() behaviour
// ---------------------------------------------------------------------------
struct DenoiseOptions {
    bool enabled   = false;  // Run OIDN on the beauty buffer
    bool writeAOVs = false;  // Include albedo + normals layers in output EXR
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

    // Merge a completed TileBuffer into the film (called by tile worker).
    // Sets the dirty flag so the progressive preview watcher knows to refresh.
    void mergeTile(const TileBuffer& tile);

    // Dirty flag — set by mergeTile(), cleared by the preview watcher after
    // each PNG write.  Render threads never block on this.
    bool isDirty() const { return m_dirty.load(std::memory_order_relaxed); }
    void clearDirty()    { m_dirty.store(false, std::memory_order_relaxed); }

    // Run OIDN denoiser on the beauty buffer, storing result in m_denoised.
    // No-op (returns false with a log warning) if OIDN is not compiled in.
    bool denoise();

    // Write finalized image to an EXR file via OpenImageIO.
    // If DenoiseOptions::enabled, includes a denoised beauty layer.
    // If DenoiseOptions::writeAOVs, includes albedo and normals layers.
    bool writeEXR(const std::string& path,
                  const DenoiseOptions& opts = {}) const;

    // Write a display-referred PNG/JPEG with ACES filmic tone mapping +
    // sRGB gamma. exposure: EV adjustment applied before tone mapping (0 = none).
    bool writePNG(const std::string& path, float exposure = 0.f) const;

    // Raw resolved pixel access (for preview / post-processing)
    Spectrum getPixel(uint32_t x, uint32_t y) const;

    // Per-pixel luminance variance: E[L^2] - E[L]^2.
    // Valid after at least one tile merge with adaptive-weight adds.
    // Returns 0 if the pixel has no samples yet.
    float varianceAt(uint32_t x, uint32_t y) const;

private:
    uint32_t m_width, m_height;
    std::vector<PixelAccumulator> m_pixels;   // [y * width + x]
    std::vector<PixelAccumulator> m_albedo;   // first-hit diffuse reflectance
    std::vector<PixelAccumulator> m_normals;  // first-hit world normal

    // Denoised beauty buffer (populated by denoise())
    std::vector<float> m_denoised;   // RGB interleaved, size = width*height*3

    // Set whenever mergeTile() completes; cleared by the preview watcher
    std::atomic<bool> m_dirty{false};

    bool inBounds(int x, int y) const {
        return x >= 0 && x < static_cast<int>(m_width)
            && y >= 0 && y < static_cast<int>(m_height);
    }
};

} // namespace anacapa
