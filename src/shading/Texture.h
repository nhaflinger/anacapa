#pragma once

#include <anacapa/core/Types.h>
#include <cmath>
#include <memory>
#include <string>

namespace anacapa {

// ---------------------------------------------------------------------------
// TextureSampler — thin wrapper around OIIO TextureSystem for bilinear/
// trilinear sampling of image textures.
//
// Implementation lives in Texture.cpp to avoid exposing OIIO headers here
// (OIIO and spdlog both bundle fmtlib, causing redefinition errors when
// included in the same translation unit).
// ---------------------------------------------------------------------------
class TextureSampler {
public:
    static TextureSampler& global();

    // Sample a texture at UV coordinates, returning RGB.
    Spectrum sample(const std::string& path, Vec2f uv,
                    const Spectrum& defaultVal = {0.5f, 0.5f, 0.5f}) const;

    // Sample a single channel (R) — used for roughness, metallic, etc.
    float sampleFloat(const std::string& path, Vec2f uv,
                      float defaultVal = 0.5f) const;

private:
    TextureSampler();
    ~TextureSampler();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};


// ---------------------------------------------------------------------------
// TextureOrValue<T> — a channel that is either a constant value or a
// file texture. Used for diffuseColor, roughness, metallic, etc.
// ---------------------------------------------------------------------------
template<typename T>
struct TextureOrValue {
    T           value;    // constant fallback / value when no texture
    std::string path;     // empty = constant, non-empty = texture file

    // UV transform from UsdTransform2d: uv_out = uv * scale + translation
    Vec2f       uvScale       = {1.f, 1.f};
    Vec2f       uvTranslation = {0.f, 0.f};
    float       uvRotation    = 0.f;  // degrees (rarely used)

    // When true, apply sRGB→linear conversion after sampling (color textures only)
    bool        linearize     = false;

    TextureOrValue() : value(T{}) {}
    TextureOrValue(T v) : value(v) {}  // implicit: allows p.roughness = 0.5f

    // Allow direct assignment from the underlying value type
    TextureOrValue& operator=(const T& v) { value = v; path.clear(); return *this; }

    bool hasTexture() const { return !path.empty(); }

    Vec2f transformUV(Vec2f uv) const {
        if (uvRotation != 0.f) {
            float rad = uvRotation * 3.14159265f / 180.f;
            float c = std::cos(rad), s = std::sin(rad);
            float u = uv.x * c - uv.y * s;
            float v = uv.x * s + uv.y * c;
            uv = {u, v};
        }
        return {uv.x * uvScale.x + uvTranslation.x,
                uv.y * uvScale.y + uvTranslation.y};
    }
};

// Specializations for the two types we use
using SpectrumTOV = TextureOrValue<Spectrum>;
using FloatTOV    = TextureOrValue<float>;

// sRGB → linear conversion (applied to color textures before shading)
inline float srgbToLinear(float x) {
    if (x <= 0.04045f) return x / 12.92f;
    return std::pow((x + 0.055f) / 1.055f, 2.4f);
}
inline Spectrum srgbToLinear(Spectrum s) {
    return { srgbToLinear(s.x), srgbToLinear(s.y), srgbToLinear(s.z) };
}

// Evaluate a Spectrum channel — applies sRGB→linear if t.linearize is set
Spectrum evalTOV(const SpectrumTOV& t, Vec2f uv);

// Evaluate a float channel (raw — no colorspace conversion)
float evalTOV(const FloatTOV& t, Vec2f uv);

} // namespace anacapa
