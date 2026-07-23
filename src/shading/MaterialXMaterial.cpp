#include "MaterialXMaterial.h"

#ifdef ANACAPA_ENABLE_MATERIALX

#include <MaterialXCore/Document.h>
#include <MaterialXFormat/XmlIo.h>

#include <spdlog/spdlog.h>
#include <cmath>

namespace mx = MaterialX;

namespace anacapa {

namespace {

constexpr float kMxInvPi = 0.31830988618379067154f;

// Surface parameters we attempt codegen for. Mirrors the set
// evalLayeredBSDF() actually consumes on the GPU path (Phase 2+); CPU only
// needs base_color/roughness/metalness for its fallback diffuse BSDF.
constexpr const char* kGeneratedInputs[] = {
    "base_color", "specular_roughness", "base_metalness",
};

mx::NodePtr findSurfaceShaderNode(mx::DocumentPtr doc) {
    for (mx::ElementPtr elem : doc->traverseTree()) {
        mx::NodePtr node = elem->asA<mx::Node>();
        if (!node) continue;
        const std::string& cat = node->getCategory();
        if (cat == "open_pbr_surface" || cat == "standard_surface") {
            return node;
        }
    }
    return nullptr;
}

// Reads a literal float/color3 value from `inputName` if it isn't driven by
// a node connection. Leaves `out` untouched (and returns false) for
// procedural or missing inputs.
bool tryReadLiteralFloat(mx::NodePtr node, const std::string& inputName, float& out) {
    mx::InputPtr input = node->getInput(inputName);
    if (!input || input->getConnectedNode() || input->hasNodeGraphString()) return false;
    if (!input->hasValue()) return false;
    out = input->getValue()->asA<float>();
    return true;
}

bool tryReadLiteralColor3(mx::NodePtr node, const std::string& inputName, Spectrum& out) {
    mx::InputPtr input = node->getInput(inputName);
    if (!input || input->getConnectedNode() || input->hasNodeGraphString()) return false;
    if (!input->hasValue()) return false;
    mx::Color3 c = input->getValue()->asA<mx::Color3>();
    out = { c[0], c[1], c[2] };
    return true;
}

} // namespace

MaterialXMaterial::MaterialXMaterial(const std::string& mtlxPath) : m_mtlxPath(mtlxPath) {
    MaterialXCodegen codegen;
    for (const char* inputName : kGeneratedInputs) {
        // Cache whichever GPU shading language the active backend consumes.
        // The Metal backend wraps MSL into a compiled shader library; the
        // CUDA/OptiX backend's MxGlslToCuda rewrites GLSL as CUDA source.
        // Using the wrong one silently fails every single material's
        // adapter step (the stripper looks for GLSL patterns like
        // 'out vec4 X;' and 'void main()' that don't exist in MSL).
        //
        // If both backends are enabled in one build the Metal path wins here
        // and CUDA gets nothing — add a second cache + explicit accessor
        // if that combined build is ever needed.
#if defined(ANACAPA_ENABLE_METAL)
        MxGeneratedShader gen = codegen.generateMsl(mtlxPath, inputName);
#elif defined(ANACAPA_ENABLE_CUDA)
        MxGeneratedShader gen = codegen.generateGlsl(mtlxPath, inputName);
#else
        MxGeneratedShader gen = codegen.generateMsl(mtlxPath, inputName);
#endif
        if (gen.valid) {
            m_generated.emplace(inputName, std::move(gen));
        }
    }

    // Pull CPU fallback literals directly — the generated-shader path above
    // only tells us whether an input is procedural, not its resting value.
    mx::DocumentPtr doc = mx::createDocument();
    try {
        mx::readFromXmlFile(doc, mtlxPath);
    } catch (const std::exception& e) {
        spdlog::warn("MaterialXMaterial: failed to read '{}' for literal defaults: {}",
                     mtlxPath, e.what());
        return;
    }

    mx::NodePtr surfaceNode = findSurfaceShaderNode(doc);
    if (!surfaceNode) return;

    tryReadLiteralColor3(surfaceNode, "base_color", m_baseColor);
    tryReadLiteralFloat(surfaceNode, "specular_roughness", m_roughness);
    tryReadLiteralFloat(surfaceNode, "base_metalness", m_metalness);
}

const MxGeneratedShader* MaterialXMaterial::generated(const std::string& inputName) const {
    auto it = m_generated.find(inputName);
    return it != m_generated.end() ? &it->second : nullptr;
}

// CPU fallback: a plain Lambertian diffuse BSDF using the literal base color.
// Real CPU shading of these graphs happens via OSL (OslMaterial); this path
// only matters if a MaterialXMaterial is ever handed to the CPU integrator
// directly, which Phase 1 does not do.
BSDFSample MaterialXMaterial::sample(const ShadingContext& ctx, Vec3f wo, Vec2f u, float) const {
    Vec3f woLocal = ctx.toLocal(wo);
    if (woLocal.z <= 0.f) return {};   // backface

    float phi = 2.f * 3.14159265358979323846f * u.x;
    float r   = std::sqrt(u.y);
    float z   = std::sqrt(std::max(0.f, 1.f - u.y));
    Vec3f wiLocal = { r * std::cos(phi), r * std::sin(phi), z };

    float cosI = std::abs(wiLocal.z);

    BSDFSample s;
    s.wi     = ctx.toWorld(wiLocal);
    s.f      = m_baseColor * (kMxInvPi * cosI);
    s.pdf    = cosI * kMxInvPi;
    s.pdfRev = std::abs(woLocal.z) * kMxInvPi;
    s.flags  = BSDFFlag_Diffuse | BSDFFlag_Reflection;
    return s;
}

BSDFEval MaterialXMaterial::evaluate(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const {
    BSDFEval e;
    Vec3f woLocal = ctx.toLocal(wo);
    Vec3f wiLocal = ctx.toLocal(wi);
    if (woLocal.z * wiLocal.z <= 0.f) return e;
    e.f      = m_baseColor * kMxInvPi;
    e.pdf    = std::abs(wiLocal.z) * kMxInvPi;
    e.pdfRev = std::abs(woLocal.z) * kMxInvPi;
    return e;
}

float MaterialXMaterial::pdf(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const {
    Vec3f woLocal = ctx.toLocal(wo);
    Vec3f wiLocal = ctx.toLocal(wi);
    if (woLocal.z * wiLocal.z <= 0.f) return 0.f;
    return std::abs(wiLocal.z) * kMxInvPi;
}

} // namespace anacapa

#endif // ANACAPA_ENABLE_MATERIALX
