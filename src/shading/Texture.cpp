#include "Texture.h"
#include <OpenImageIO/imageio.h>
#include <OpenImageIO/texture.h>
#include <unordered_set>
#include <cstdio>

namespace anacapa {

// ---------------------------------------------------------------------------
// TextureSampler::Impl — wraps OIIO TextureSystem
// ---------------------------------------------------------------------------
struct TextureSampler::Impl {
    std::shared_ptr<OIIO::TextureSystem> tsys;

    Impl() {
        tsys = OIIO::TextureSystem::create(/*shared=*/true);
    }
};

TextureSampler::TextureSampler() : m_impl(std::make_unique<Impl>()) {}
TextureSampler::~TextureSampler() = default;

TextureSampler& TextureSampler::global() {
    static TextureSampler instance;
    return instance;
}

Spectrum TextureSampler::sample(const std::string& path, Vec2f uv,
                                 const Spectrum& defaultVal) const {
    if (path.empty()) return defaultVal;

    // Strip any channel suffix (e.g. "|g") — only relevant for sampleFloat
    const std::string* cleanPath = &path;
    std::string stripped;
    auto pipe = path.rfind('|');
    if (pipe != std::string::npos) {
        stripped = path.substr(0, pipe);
        cleanPath = &stripped;
    }

    OIIO::TextureOpt opt;
    opt.swrap = OIIO::TextureOpt::WrapPeriodic;
    opt.twrap = OIIO::TextureOpt::WrapPeriodic;

    // Query channel count so we can handle grayscale textures correctly.
    // OIIO fills only the channels the texture has and leaves the rest at 0,
    // so a 1-channel greyscale file sampled for 3 channels gives (grey, 0, 0) —
    // which looks red. We detect this and broadcast the single channel to all three.
    const OIIO::ImageSpec* texSpec =
        m_impl->tsys->imagespec(OIIO::ustring(*cleanPath));
    int nchans = texSpec ? texSpec->nchannels : 3;

    // Initialize to defaultVal — OIIO sets result to 0 on failure, not defaultVal,
    // so we check ok and return defaultVal explicitly on failure.
    float result[3] = {defaultVal.x, defaultVal.y, defaultVal.z};
    bool ok = m_impl->tsys->texture(
        OIIO::ustring(*cleanPath),
        opt,
        uv.x, 1.f - uv.y,   // flip V: USD/OpenGL origin is bottom-left
        0.f, 0.f, 0.f, 0.f, // no derivatives (no mipmapping)
        3, result
    );
    if (!ok) {
        // Log each unique failed path once
        static std::unordered_set<std::string> s_failed;
        if (s_failed.insert(*cleanPath).second)
            fprintf(stderr, "TextureSampler: failed to load '%s'\n", cleanPath->c_str());
        return defaultVal;
    }
    // Broadcast single-channel (greyscale) to RGB
    if (nchans == 1) {
        result[1] = result[0];
        result[2] = result[0];
    }
    return {result[0], result[1], result[2]};
}

float TextureSampler::sampleFloat(const std::string& path, Vec2f uv,
                                   float defaultVal) const {
    // Path may have a channel suffix: "filepath|g" means use G channel.
    // Strip the suffix before loading, then select the channel.
    int channel = 0; // default: R
    std::string cleanPath = path;
    auto pipe = path.rfind('|');
    if (pipe != std::string::npos) {
        std::string ch = path.substr(pipe + 1);
        cleanPath = path.substr(0, pipe);
        if      (ch == "g") channel = 1;
        else if (ch == "b") channel = 2;
        else if (ch == "a") channel = 3;
    }

    // For the alpha channel we must request 4 channels from OIIO; otherwise
    // the texture system only fills 3 (RGB) and alpha is never fetched.
    if (channel == 3) {
        OIIO::TextureOpt opt;
        opt.swrap = OIIO::TextureOpt::WrapPeriodic;
        opt.twrap = OIIO::TextureOpt::WrapPeriodic;
        float result[4] = {defaultVal, defaultVal, defaultVal, defaultVal};
        bool ok = m_impl->tsys->texture(
            OIIO::ustring(cleanPath),
            opt,
            uv.x, 1.f - uv.y,
            0.f, 0.f, 0.f, 0.f,
            4, result
        );
        return ok ? result[3] : defaultVal;
    }

    Spectrum s = sample(cleanPath, uv, {defaultVal, defaultVal, defaultVal});
    if (channel == 0) return s.x;
    if (channel == 1) return s.y;
    return s.z;
}

// ---------------------------------------------------------------------------
// evalTOV implementations
// ---------------------------------------------------------------------------
Spectrum evalTOV(const SpectrumTOV& t, Vec2f uv) {
    if (!t.hasTexture()) return t.value;
    Spectrum s = TextureSampler::global().sample(t.path, t.transformUV(uv), t.value);
    if (t.linearize) s = srgbToLinear(s);
    return s;
}

float evalTOV(const FloatTOV& t, Vec2f uv) {
    if (!t.hasTexture()) return t.value;
    return TextureSampler::global().sampleFloat(t.path, t.transformUV(uv), t.value);
}

} // namespace anacapa
