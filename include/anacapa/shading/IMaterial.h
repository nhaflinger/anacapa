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

    // Parameters for the subsurface scattering layer.  weight == 0 means no SSS.
    // radius is the mean free path in world units (controls Gaussian blur radius).
    struct SubsurfaceParams {
        float    weight     = 0.f;
        Spectrum color      = {1.f, 1.f, 1.f};
        float    radius     = 0.1f;
        float    scale      = 1.f;   // world-space multiplier; effective d = radius * scale
        float    anisotropy = 0.f;
    };

    // Returns non-zero weight when this surface has a subsurface scattering layer.
    // The photon map integrator queries this to deposit and look up SSS photons.
    virtual SubsurfaceParams subsurfaceParams() const { return {}; }
    bool isSubsurface() const { return subsurfaceParams().weight > 0.f; }

    // True if this material should be treated as a caustic-generating surface.
    // The photon map integrator uses this as an opt-in: photon storage and
    // NEE-through-delta gating only apply to surfaces flagged here.  Non-flagged
    // glass keeps using NEE with transmittanceColor() — so a windowpane or
    // drinking glass still passes light correctly when --integrator photon is
    // active but the user hasn't asked for photons through it.
    // Default: false (path/BDPT ignore this; photon mode skips caustic logic
    // on this material).
    virtual bool isCausticGenerator() const { return false; }

    // Per-material caustic search radius override (world units).
    // 0 means "use the global scene default from --photon-radius".
    virtual float causticRadius() const { return 0.f; }

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

    // Representative emission from a probe evaluation at uv=(0.5,0.5).
    // Non-zero if this material emits light; used to auto-register emissive
    // meshes as AreaLights for NEE.  Returns {} for non-emitting materials.
    virtual Spectrum probeEmission() const { return {}; }

    // Hemispherical albedo — the surface reflectance, used as a denoising hint.
    // Returns black for emitters and unimplemented materials.
    virtual Spectrum reflectance(const ShadingContext& ctx) const {
        return {};
    }

    // Transmittance color for shadow rays — the fraction of light that passes
    // through this surface unscattered.  Returns black for opaque materials
    // (fully blocks light) and a tint color for transmissive ones.
    // Used by the integrators to let shadow rays pass through glass.
    virtual Spectrum transmittanceColor(const ShadingContext& ctx) const {
        return {};   // opaque by default
    }

    // Opacity at the given surface point in [0, 1].
    // 1 = fully opaque (default), 0 = fully transparent (cutout).
    // Used by the path integrators to continue rays through alpha-masked
    // surfaces without shading them (alpha testing).
    virtual float evalOpacity(const ShadingContext& ctx) const {
        return 1.f;
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
