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

            // Beauty
            const auto& s = tile.pixels[ti];
            if (s.weight > 0.f)
                m_pixels[fi].add(s.r, s.g, s.b, s.weight);

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

    OIDNDevice device = oidnNewDevice(OIDN_DEVICE_TYPE_DEFAULT);
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

    // Build channel list: always beauty (R,G,B),
    // optionally denoised, albedo, normals
    std::vector<std::string> channelNames;
    std::vector<float>       buf;

    // Helper: append a resolved buffer of Spectrum values
    auto appendLayer = [&](const std::vector<PixelAccumulator>& accum,
                           const std::string& prefix) {
        channelNames.push_back(prefix + "R");
        channelNames.push_back(prefix + "G");
        channelNames.push_back(prefix + "B");
        size_t base = buf.size();
        buf.resize(base + N * 3);
        for (uint32_t i = 0; i < N; ++i) {
            Spectrum s = accum[i].resolve();
            buf[base + i*3+0] = s.x;
            buf[base + i*3+1] = s.y;
            buf[base + i*3+2] = s.z;
        }
    };

    // Beauty layer (always written)
    appendLayer(m_pixels, "");   // root-level R,G,B

    // Denoised layer
    if (opts.enabled && !m_denoised.empty()) {
        channelNames.push_back("denoised.R");
        channelNames.push_back("denoised.G");
        channelNames.push_back("denoised.B");
        buf.insert(buf.end(), m_denoised.begin(), m_denoised.end());
    }

    // AOV layers
    if (opts.writeAOVs) {
        appendLayer(m_albedo,  "albedo.");
        appendLayer(m_normals, "normals.");
    }

    int nChannels = static_cast<int>(channelNames.size());

    // OpenImageIO multi-channel EXR: all channels in one subimage
    ImageSpec spec(static_cast<int>(m_width), static_cast<int>(m_height),
                   nChannels, TypeDesc::FLOAT);
    spec.attribute("compression", "zip");
    spec.channelnames = channelNames;

    auto out = ImageOutput::create(path);
    if (!out) return false;
    if (!out->open(path, spec)) return false;

    // Interleaved layout — OIIO expects pixel-by-pixel channel order
    std::vector<float> interleaved(N * nChannels);
    int layers = nChannels / 3;
    for (uint32_t i = 0; i < N; ++i) {
        for (int l = 0; l < layers; ++l) {
            interleaved[i * nChannels + l*3 + 0] = buf[l * N * 3 + i*3 + 0];
            interleaved[i * nChannels + l*3 + 1] = buf[l * N * 3 + i*3 + 1];
            interleaved[i * nChannels + l*3 + 2] = buf[l * N * 3 + i*3 + 2];
        }
    }

    bool ok = out->write_image(TypeDesc::FLOAT, interleaved.data());
    out->close();
    return ok;
}

// ---------------------------------------------------------------------------
// writePNG — ACES filmic tone map + sRGB gamma, written as 8-bit PNG/JPEG
// ---------------------------------------------------------------------------
bool Film::writePNG(const std::string& path, float exposure) const {
    using namespace OIIO;

    const uint32_t N = m_width * m_height;
    const float evScale = std::pow(2.f, exposure);

    // ACES RRT+ODT approximation (Krzysztof Narkowicz, 2016)
    auto aces = [](float x) -> float {
        x *= 0.6f;  // pre-exposure to bring scene-linear into ACES range
        const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
        return std::max(0.f, std::min(1.f,
            (x * (a * x + b)) / (x * (c * x + d) + e)));
    };

    // linear -> sRGB
    auto linearToSRGB = [](float x) -> float {
        x = std::max(0.f, x);
        if (x <= 0.0031308f) return 12.92f * x;
        return 1.055f * std::pow(x, 1.f / 2.4f) - 0.055f;
    };

    std::vector<uint8_t> pixels(N * 3);
    for (uint32_t i = 0; i < N; ++i) {
        Spectrum c = m_pixels[i].resolve();
        c.x *= evScale;
        c.y *= evScale;
        c.z *= evScale;
        pixels[i*3+0] = static_cast<uint8_t>(std::min(255.f, linearToSRGB(aces(c.x)) * 255.f + 0.5f));
        pixels[i*3+1] = static_cast<uint8_t>(std::min(255.f, linearToSRGB(aces(c.y)) * 255.f + 0.5f));
        pixels[i*3+2] = static_cast<uint8_t>(std::min(255.f, linearToSRGB(aces(c.z)) * 255.f + 0.5f));
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

} // namespace anacapa
