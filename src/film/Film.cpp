#include <anacapa/film/Film.h>
#include <OpenImageIO/imageio.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdio>   // std::rename

#ifdef ANACAPA_ENABLE_OIDN
#  include <OpenImageDenoise/oidn.h>
#endif

namespace anacapa {

Film::Film(uint32_t width, uint32_t height)
    : m_width(width), m_height(height)
    , m_pixels(width * height)
    , m_albedo(width * height)
    , m_normals(width * height)
{}

void Film::splatPixel(float x, float y, Spectrum value) {
    // Clamp NaN/Inf before accumulating
    if (!value.isFinite()) return;

    int px = static_cast<int>(x);
    int py = static_cast<int>(y);
    if (!inBounds(px, py)) return;

    m_pixels[py * m_width + px].add(value.x, value.y, value.z, 1.f);
}

void Film::mergeTile(const TileBuffer& tile) {
    m_dirty.store(true, std::memory_order_relaxed);

    for (uint32_t ty = 0; ty < tile.height; ++ty) {
        for (uint32_t tx = 0; tx < tile.width; ++tx) {
            uint32_t fx = tile.x0 + tx;
            uint32_t fy = tile.y0 + ty;
            if (!inBounds(static_cast<int>(fx), static_cast<int>(fy))) continue;

            uint32_t fi = fy * m_width + fx;
            uint32_t ti = ty * tile.width + tx;

            // Beauty — tile stores pre-weighted values (color * weight), so add
            // directly to film atomics rather than re-multiplying via add().
            const auto& s = tile.pixels[ti];
            if (s.weight > 0.f) {
                m_pixels[fi].r.fetch_add(s.r, std::memory_order_relaxed);
                m_pixels[fi].g.fetch_add(s.g, std::memory_order_relaxed);
                m_pixels[fi].b.fetch_add(s.b, std::memory_order_relaxed);
                m_pixels[fi].weight.fetch_add(s.weight, std::memory_order_relaxed);
                m_pixels[fi].sumLumSq.fetch_add(tile.sumLumSq[ti],
                                                 std::memory_order_relaxed);
                m_pixels[fi].alpha.fetch_add(s.alpha, std::memory_order_relaxed);
            }

            // Albedo AOV
            const auto& a = tile.albedo[ti];
            if (a.count > 0) {
                float inv = 1.f / static_cast<float>(a.count);
                m_albedo[fi].add(a.r * inv, a.g * inv, a.b * inv, 1.f);
            }

            // Normal AOV
            const auto& n = tile.normals[ti];
            if (n.count > 0) {
                float inv = 1.f / static_cast<float>(n.count);
                m_normals[fi].add(n.r * inv, n.g * inv, n.b * inv, 1.f);
            }
        }
    }
}

Spectrum Film::getPixel(uint32_t x, uint32_t y) const {
    assert(inBounds(static_cast<int>(x), static_cast<int>(y)));
    return m_pixels[y * m_width + x].resolve();
}

void Film::readTile(uint32_t x0, uint32_t y0,
                    uint32_t w,  uint32_t h,
                    float* out) const {
    for (uint32_t row = 0; row < h; ++row) {
        uint32_t fy = y0 + row;
        for (uint32_t col = 0; col < w; ++col) {
            uint32_t fx = x0 + col;
            float* p = out + (row * w + col) * 4;
            if (inBounds(static_cast<int>(fx), static_cast<int>(fy))) {
                const auto& px = m_pixels[fy * m_width + fx];
                Spectrum s = px.resolve();
                p[0] = s.x; p[1] = s.y; p[2] = s.z;
                {
                    float wt = px.weight.load(std::memory_order_relaxed);
                    float a  = (wt > 0.f)
                        ? px.alpha.load(std::memory_order_relaxed) / wt
                        : 0.f;
                    p[3] = std::max(0.f, std::min(1.f, a));
                }
            } else {
                p[0] = p[1] = p[2] = p[3] = 0.f;
            }
        }
    }
}

float Film::varianceAt(uint32_t x, uint32_t y) const {
    const auto& p = m_pixels[y * m_width + x];
    float w = p.weight.load(std::memory_order_relaxed);
    if (w <= 0.f) return 0.f;
    float invW   = 1.f / w;
    float meanL  = (p.r.load(std::memory_order_relaxed) * 0.2126f
                  + p.g.load(std::memory_order_relaxed) * 0.7152f
                  + p.b.load(std::memory_order_relaxed) * 0.0722f) * invW;
    float eLsq   = p.sumLumSq.load(std::memory_order_relaxed) * invW;
    return std::max(0.f, eLsq - meanL * meanL);
}

// ---------------------------------------------------------------------------
// denoise — run Intel OIDN on the beauty buffer.
// Requires ANACAPA_ENABLE_OIDN; logs a warning and returns false otherwise.
// ---------------------------------------------------------------------------
bool Film::denoise() {
#ifndef ANACAPA_ENABLE_OIDN
    std::fprintf(stderr, "denoise() called but ANACAPA_ENABLE_OIDN is not enabled\n");
    return false;
#else
    const uint32_t N = m_width * m_height;

    // Resolve beauty, albedo, normals into flat RGB float arrays
    std::vector<float> color(N * 3);
    std::vector<float> albedo(N * 3);
    std::vector<float> normals(N * 3);

    for (uint32_t i = 0; i < N; ++i) {
        Spectrum c = m_pixels[i].resolve();
        color[i*3+0] = c.x; color[i*3+1] = c.y; color[i*3+2] = c.z;

        Spectrum a = m_albedo[i].resolve();
        albedo[i*3+0] = a.x; albedo[i*3+1] = a.y; albedo[i*3+2] = a.z;

        Spectrum n = m_normals[i].resolve();
        normals[i*3+0] = n.x; normals[i*3+1] = n.y; normals[i*3+2] = n.z;
    }

    m_denoised.resize(N * 3);

    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_CPU);
    oidnCommitDevice(device);

    OIDNFilter filter = oidnNewFilter(device, "RT");

    // OIDN 2.x API: oidnSetSharedFilterImage for host-side float buffers
    oidnSetSharedFilterImage(filter, "color",  color.data(),
                             OIDN_FORMAT_FLOAT3, m_width, m_height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "albedo", albedo.data(),
                             OIDN_FORMAT_FLOAT3, m_width, m_height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "normal", normals.data(),
                             OIDN_FORMAT_FLOAT3, m_width, m_height, 0, 0, 0);
    oidnSetSharedFilterImage(filter, "output", m_denoised.data(),
                             OIDN_FORMAT_FLOAT3, m_width, m_height, 0, 0, 0);

    oidnSetFilterBool(filter, "hdr", true);   // beauty is in linear HDR
    oidnCommitFilter(filter);
    oidnExecuteFilter(filter);

    const char* errMsg = nullptr;
    if (oidnGetDeviceError(device, &errMsg) != OIDN_ERROR_NONE) {
        std::fprintf(stderr, "OIDN error: %s\n", errMsg);
        oidnReleaseFilter(filter);
        oidnReleaseDevice(device);
        return false;
    }

    oidnReleaseFilter(filter);
    oidnReleaseDevice(device);
    std::fprintf(stderr, "OIDN denoising complete\n");
    return true;
#endif
}

// ---------------------------------------------------------------------------
// writeEXR — write beauty (and optionally denoised + AOV) layers to EXR
// ---------------------------------------------------------------------------
bool Film::writeEXR(const std::string& path,
                    const DenoiseOptions& opts) const {
    using namespace OIIO;

    const uint32_t N = m_width * m_height;

    const bool writeDenoised = opts.enabled && !m_denoised.empty();
    const bool writeAOVs    = opts.writeAOVs;

    // Build channel list
    std::vector<std::string> channelNames;
    channelNames.reserve(16);
    channelNames.push_back("Combined.R");
    channelNames.push_back("Combined.G");
    channelNames.push_back("Combined.B");
    channelNames.push_back("Combined.A");  // always written; 1.0 when not in alpha mode
    if (writeDenoised) { channelNames.push_back("denoised.R"); channelNames.push_back("denoised.G"); channelNames.push_back("denoised.B"); }
    if (writeAOVs)     { channelNames.push_back("albedo.R"); channelNames.push_back("albedo.G"); channelNames.push_back("albedo.B");
                         channelNames.push_back("normals.R"); channelNames.push_back("normals.G"); channelNames.push_back("normals.B"); }

    int nChannels = static_cast<int>(channelNames.size());

    // Build interleaved pixel buffer directly — one pixel at a time
    std::vector<float> interleaved(N * nChannels);
    for (uint32_t i = 0; i < N; ++i) {
        float* p = interleaved.data() + i * nChannels;
        int c = 0;

        // Beauty RGB
        Spectrum beauty = m_pixels[i].resolve();
        p[c++] = beauty.x;
        p[c++] = beauty.y;
        p[c++] = beauty.z;

        // Alpha — always accumulated; background pixels = 0, geometry = 1.
        {
            float w = m_pixels[i].weight.load(std::memory_order_relaxed);
            float a = (w > 0.f)
                ? m_pixels[i].alpha.load(std::memory_order_relaxed) / w
                : 0.f;
            p[c++] = std::max(0.f, std::min(1.f, a));
        }

        // Denoised RGB
        if (writeDenoised) {
            p[c++] = m_denoised[i*3+0];
            p[c++] = m_denoised[i*3+1];
            p[c++] = m_denoised[i*3+2];
        }

        // AOV layers
        if (writeAOVs) {
            Spectrum alb = m_albedo[i].resolve();
            p[c++] = alb.x; p[c++] = alb.y; p[c++] = alb.z;
            Spectrum nrm = m_normals[i].resolve();
            p[c++] = nrm.x; p[c++] = nrm.y; p[c++] = nrm.z;
        }
    }

    ImageSpec spec(static_cast<int>(m_width), static_cast<int>(m_height),
                   nChannels, TypeDesc::FLOAT);
    spec.attribute("compression", "zip");
    spec.channelnames = channelNames;

    auto out = ImageOutput::create(path);
    if (!out) return false;
    if (!out->open(path, spec)) return false;
    bool ok = out->write_image(TypeDesc::FLOAT, interleaved.data());
    out->close();
    return ok;
}

// ---------------------------------------------------------------------------
// writePNG — ACES filmic tone mapping + sRGB gamma, written as 8-bit PNG/JPEG
//
// Pipeline: linear → exposure → ACES RRT+ODT approximation → sRGB gamma
// This matches Cycles' default Filmic output mode closely enough that sky
// radiance values which would clip without tone mapping are compressed into
// a visually correct blue sky, matching the appearance Cycles produces at
// the same intensity values.
// ---------------------------------------------------------------------------
bool Film::writePNG(const std::string& path, float exposure) const {
    using namespace OIIO;

    const uint32_t N = m_width * m_height;
    const float evScale = std::pow(2.f, exposure);

    // Log-based filmic tone mapping, approximating Blender's "Filmic" default.
    // Operates in log2 space over a 16-stop range centred on 18% grey (0.18).
    // Key advantage over polynomial ACES: sky radiance values of 5–20 still
    // show distinct colour (bright saturated blue) rather than clipping to white.
    //   0.18 → 0.50   (mid-grey)
    //   0.47 → 0.63   (zenith blue sky at sunIntensity=1)
    //   4.7  → 0.89   (zenith blue sky at sunIntensity=10, still blue, not white)
    //   18   → 0.97   (sun disc at sunIntensity=1, bright but coloured)
    auto filmic = [](float x) -> float {
        if (x <= 0.f) return 0.f;
        float lx = std::log2(std::max(x, 1e-10f) / 0.18f) / 16.f + 0.5f;
        lx = std::max(0.f, std::min(1.f, lx));
        return lx * lx * (3.f - 2.f * lx);  // smoothstep S-curve
    };

    auto linearToSRGB = [](float x) -> float {
        if (x <= 0.0031308f) return 12.92f * x;
        return 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
    };

    std::vector<uint8_t> pixels(N * 3);
    for (uint32_t i = 0; i < N; ++i) {
        Spectrum c = m_pixels[i].resolve();
        c.x = linearToSRGB(filmic(c.x * evScale));
        c.y = linearToSRGB(filmic(c.y * evScale));
        c.z = linearToSRGB(filmic(c.z * evScale));
        pixels[i*3+0] = static_cast<uint8_t>(c.x * 255.f + 0.5f);
        pixels[i*3+1] = static_cast<uint8_t>(c.y * 255.f + 0.5f);
        pixels[i*3+2] = static_cast<uint8_t>(c.z * 255.f + 0.5f);
    }

    ImageSpec spec(static_cast<int>(m_width), static_cast<int>(m_height), 3, TypeDesc::UINT8);
    spec.attribute("oiio:ColorSpace", "sRGB");

    // Write to a temp file then atomically rename into place so viewers
    // never read a partially-written file.
    const std::string tmp = path + ".writing.png";

    auto out = ImageOutput::create(tmp);
    if (!out) return false;
    if (!out->open(tmp, spec)) return false;
    bool ok = out->write_image(TypeDesc::UINT8, pixels.data());
    out->close();
    if (!ok) return false;

    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

// ---------------------------------------------------------------------------
// writeEXRPreview — beauty-only linear EXR for progressive viewer updates.
// Atomic write (temp file → rename) so the viewer never sees a partial file.
// ---------------------------------------------------------------------------
bool Film::writeEXRPreview(const std::string& path) const {
    using namespace OIIO;
    const uint32_t N = m_width * m_height;
    std::vector<float> pixels(N * 4);
    for (uint32_t i = 0; i < N; ++i) {
        float* p = pixels.data() + i * 4;
        Spectrum c = m_pixels[i].resolve();
        p[0] = c.x;
        p[1] = c.y;
        p[2] = c.z;
        float w = m_pixels[i].weight.load(std::memory_order_relaxed);
        float a = (w > 0.f)
            ? m_pixels[i].alpha.load(std::memory_order_relaxed) / w
            : 0.f;
        p[3] = std::max(0.f, std::min(1.f, a));
    }

    ImageSpec spec(static_cast<int>(m_width), static_cast<int>(m_height),
                   4, TypeDesc::FLOAT);
    spec.attribute("compression", "none");  // fastest for progressive preview
    spec.channelnames = std::vector<std::string>{"R", "G", "B", "A"};

    const std::string tmp = path + ".writing.exr";
    auto out = ImageOutput::create(tmp);
    if (!out) return false;
    if (!out->open(tmp, spec)) return false;
    bool ok = out->write_image(TypeDesc::FLOAT, pixels.data());
    out->close();
    if (!ok) { std::remove(tmp.c_str()); return false; }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

} // namespace anacapa
