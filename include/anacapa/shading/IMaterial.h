#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/shading/ShadingContext.h>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// BSDF flags — packed into a uint32 for fast GPU-side branching
// ---------------------------------------------------------------------------
enum BSDFFlags : uint32_t {
    BSDFFlag_None         = 0,
    BSDFFlag_Diffuse      = 1 << 0,
    BSDFFlag_Glossy       = 1 << 1,
    BSDFFlag_Specular     = 1 << 2,   // Delta distribution (perfect mirror/glass)
    BSDFFlag_Transmission = 1 << 3,
    BSDFFlag_Reflection   = 1 << 4,
    BSDFFlag_All          = 0xFFFFFFFF,
};

// ---------------------------------------------------------------------------
// BSDFSample — returned by IMaterial::sample()
// ---------------------------------------------------------------------------
struct BSDFSample {
    Spectrum f;              // BSDF value f(wo, wi) * |cos(theta_i)|
    Vec3f    wi;             // Sampled incident direction (world space)
    float    pdf  = 0.f;    // PDF of sampling wi
    float    pdfRev = 0.f;  // Reverse PDF p(wo | wi) — needed for BDPT MIS
    float    eta  = 1.f;    // Relative IOR for refraction (1 = no refraction)
    uint32_t flags = BSDFFlag_None;

    bool isValid()   const { return pdf > 0.f && !isBlack(f); }
    bool isDelta()   const { return (flags & BSDFFlag_Specular) != 0; }
};

// ---------------------------------------------------------------------------
// BSDFEval — returned by IMaterial::evaluate()
// ---------------------------------------------------------------------------
struct BSDFEval {
    Spectrum f;           // BSDF value (not multiplied by cosine)
    float    pdf    = 0.f;
    float    pdfRev = 0.f;  // Reverse PDF for BDPT MIS weight computation
};

// ---------------------------------------------------------------------------
// IMaterial
//
// All methods are const and stateless. Per-evaluation state is in
// ShadingContext. This maps directly to a device function on GPU.
// ---------------------------------------------------------------------------
class IMaterial {
public:
    virtual ~IMaterial() = default;

    // Returns true if this material contains a delta (measure-zero) component.
    // Delta materials cannot be connected in BDPT — they must be sampled.
    virtual bool isDelta() const = 0;

    // Sample an incident direction given outgoing direction wo (world space).
    // u:         2D stratified sample for direction sampling
    // uComponent: 1D sample for selecting between material components
    virtual BSDFSample sample(const ShadingContext& ctx,
                              Vec3f wo,
                              Vec2f u,
                              float uComponent) const = 0;

    // Evaluate f(wo, wi) and forward/reverse PDFs.
    // Returns zero for delta materials (they have no area PDF).
    virtual BSDFEval evaluate(const ShadingContext& ctx,
                               Vec3f wo,
                               Vec3f wi) const = 0;

    // PDF of sampling wi given wo.
    virtual float pdf(const ShadingContext& ctx,
                      Vec3f wo,
                      Vec3f wi) const = 0;

    // Emitted radiance from this surface in direction wo (non-zero on emitters)
    virtual Spectrum Le(const ShadingContext& ctx, Vec3f wo) const {
        return {};
    }

    // Hemispherical albedo — the surface reflectance, used as a denoising hint.
    // Returns black for emitters and unimplemented materials.
    virtual Spectrum reflectance(const ShadingContext& ctx) const {
        return {};
    }

    virtual uint32_t flags() const = 0;

    // Perceptual roughness in [0, 1].  0 = perfectly smooth, 1 = fully diffuse.
    // Used by BDPTIntegrator to skip connection attempts through near-specular
    // vertices where the geometry-sampling PDF would be near zero.
    // Default (1.0) is safe for any material that doesn't override this.
    virtual float roughness() const { return 1.0f; }
    virtual float metalness() const { return 0.0f; }
};

} // namespace anacapa
