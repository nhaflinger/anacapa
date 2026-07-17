#pragma once

// ---------------------------------------------------------------------------
// MaterialXMaterial — holds MaterialX-generated shader source + uniforms for
// a single .mtlx material, keyed by surface parameter (base_color, etc.).
//
// This class exists so GPU material extraction (extractGpuMaterial() in
// MetalPathIntegrator.mm / the CUDA equivalent) has something to
// dynamic_cast onto and pull generated MSL/GLSL + uniforms from. CPU
// sample()/evaluate() fall back to a plain diffuse BSDF using the default
// (literal) uniform values baked into the generated shader — CPU rendering
// of these exact node graphs already works via OSL (see OslMaterial.h);
// this class is not meant to be the CPU-side renderer for them.
// ---------------------------------------------------------------------------

#include <anacapa/shading/IMaterial.h>
#include "MaterialXCodegen.h"
#include <string>
#include <unordered_map>

namespace anacapa {

#ifdef ANACAPA_ENABLE_MATERIALX

class MaterialXMaterial : public IMaterial {
public:
    // mtlxPath: path to the per-material .mtlx sidecar file exported by the
    // Blender addon. Runs codegen immediately for every input in
    // kGeneratedInputs, caching whichever ones turn out to be procedural.
    explicit MaterialXMaterial(const std::string& mtlxPath);

    bool isDelta() const override { return false; }
    uint32_t flags() const override { return BSDFFlag_Diffuse | BSDFFlag_Reflection; }
    float roughness() const override { return m_roughness; }
    float metalness() const override { return m_metalness; }

    BSDFSample sample(const ShadingContext& ctx, Vec3f wo, Vec2f u, float uComponent) const override;
    BSDFEval   evaluate(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const override;
    float      pdf(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const override;
    Spectrum   reflectance(const ShadingContext& ctx) const override { return m_baseColor; }

    // Generated shader for a given surface input ("base_color", "roughness",
    // "metalness", "normal", "emission_color"), if that input turned out to
    // be procedural. Empty/.valid==false if it's a plain literal — GPU
    // material extraction should fall back to the existing literal path.
    const MxGeneratedShader* generated(const std::string& inputName) const;

    const std::string& mtlxPath() const { return m_mtlxPath; }

private:
    std::string m_mtlxPath;
    std::unordered_map<std::string, MxGeneratedShader> m_generated;

    // Default (literal) values used for the CPU fallback BSDF, pulled from
    // the .mtlx surface node's literal inputs when procedural generation
    // isn't applicable.
    Spectrum m_baseColor = {0.8f, 0.8f, 0.8f};
    float    m_roughness = 0.5f;
    float    m_metalness = 0.f;
};

#endif // ANACAPA_ENABLE_MATERIALX

} // namespace anacapa
