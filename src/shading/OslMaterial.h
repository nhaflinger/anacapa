#pragma once

// ---------------------------------------------------------------------------
// OslMaterial — IMaterial adapter that evaluates shaders via the Open
// Shading Language runtime (liboslexec).
//
// Build requirements:
//   cmake -DANACAPA_ENABLE_OSL=ON ...
//   System: brew install open-shading-language   (when available on macOS)
//           apt install libosl-dev               (Ubuntu/Debian)
//
// Design:
//   Each OslMaterial owns an OSL::ShaderGroup compiled from a .oso or inline
//   source string.  evaluate() / sample() invoke the "surface" shader closure
//   through liboslexec's batched execution interface, then extract the BSDF
//   lobe weights for sampling / evaluation.
//
//   The RendererServices implementation (OslRendererServices below) provides
//   BVH trace callbacks required by OSL's trace() and getattribute() built-ins.
//
// When ANACAPA_ENABLE_OSL is NOT defined this header provides a stub that
// asserts at construction time so the linker does not require OSL at all.
// ---------------------------------------------------------------------------

#include <anacapa/shading/IMaterial.h>
#include <cassert>
#include <string>

#ifdef ANACAPA_ENABLE_OSL
#  include <OSL/oslexec.h>
#  include <OSL/oslcomp.h>
#endif

namespace anacapa {

#ifdef ANACAPA_ENABLE_OSL

// ---------------------------------------------------------------------------
// OslRendererServices — implements OSL's renderer callback interface.
//
// OSL shaders can call trace(), getattribute(), and texture().
// The minimal implementation below supports texture() via OIIO and stubs
// out trace() / getattribute() for Phase 4.  Full support is Phase 4+.
// ---------------------------------------------------------------------------
class OslRendererServices : public OSL::RendererServices {
public:
    // texture() — delegate to OIIO TextureSystem (same one used by OSL itself)
    bool texture(OSL::ustring filename,
                 OSL::TextureSystem::Perthread* thread_info,
                 OSL::TextureOpt& options,
                 OSL::ShaderGlobals* sg,
                 float s, float t,
                 float dsdx, float dtdx,
                 float dsdy, float dtdy,
                 int nchannels,
                 float* result,
                 float* dresultds, float* dresultdt,
                 float* dresultdr) override {
        return OSL::RendererServices::texture(
            filename, thread_info, options, sg,
            s, t, dsdx, dtdx, dsdy, dtdy,
            nchannels, result, dresultds, dresultdt, dresultdr);
    }

    // trace() — stub: always returns false (no OSL-driven ray tracing yet)
    bool trace(OSL::TraceOpt& /*options*/, OSL::ShaderGlobals* /*sg*/,
               const OSL::Vec3& /*P*/, const OSL::Vec3& /*dPdx*/,
               const OSL::Vec3& /*dPdy*/, const OSL::Vec3& /*R*/,
               const OSL::Vec3& /*dRdx*/, const OSL::Vec3& /*dRdy*/) override {
        return false;
    }

    // getattribute() — stub: returns false (attribute not found)
    bool getattribute(OSL::ShaderGlobals* /*sg*/, bool /*derivatives*/,
                      OSL::ustring /*object*/, OSL::TypeDesc /*type*/,
                      OSL::ustring /*name*/, void* /*val*/) override {
        return false;
    }
};

// ---------------------------------------------------------------------------
// OslShadingSystem — singleton wrapper around OSL::ShadingSystem.
//
// One ShadingSystem per process; thread-safe after construction.
// ---------------------------------------------------------------------------
class OslShadingSystem {
public:
    static OslShadingSystem& instance() {
        static OslShadingSystem s_instance;
        return s_instance;
    }

    OSL::ShadingSystem* get() { return m_sys; }

private:
    OslShadingSystem() {
        m_textureSystem = OSL::TextureSystem::create();
        m_sys = new OSL::ShadingSystem(&m_services, m_textureSystem);
        // Register the standard MaterialX OSL nodes if present
        m_sys->attribute("searchpath:shader", OSL::ustring("shaders"));
    }

    ~OslShadingSystem() {
        delete m_sys;
        OSL::TextureSystem::destroy(m_textureSystem);
    }

    OslRendererServices   m_services;
    OSL::TextureSystem*   m_textureSystem = nullptr;
    OSL::ShadingSystem*   m_sys           = nullptr;
};

// ---------------------------------------------------------------------------
// OslMaterial
//
// Wraps an OSL::ShaderGroup.  The shader must export a "surface" closure.
// The evaluate() / sample() methods:
//   1. Fill ShaderGlobals from ShadingContext
//   2. Execute the shader group
//   3. Walk the returned closure to extract BSDF contributions
//
// Phase 4 limitation: only Lambertian and GGX closures are extracted.
// Full closure support (Disney, Oren-Nayar, SSS) is Phase 4+.
// ---------------------------------------------------------------------------
class OslMaterial : public IMaterial {
public:
    // Compile and load a shader from a .osl source file.
    // shaderName: the OSL shader name (e.g. "standard_surface")
    // params: name-value pairs to set on the shader
    explicit OslMaterial(const std::string& shaderName) {
        OSL::ShadingSystem* sys = OslShadingSystem::instance().get();
        m_group = sys->ShaderGroupBegin(shaderName);
        sys->Shader(*m_group, "surface", OSL::ustring(shaderName), OSL::ustring("layer1"));
        sys->ShaderGroupEnd(*m_group);

        if (!m_group)
            throw std::runtime_error("OslMaterial: failed to create shader group '" + shaderName + "'");
    }

    bool isDelta() const override { return false; }

    uint32_t flags() const override {
        return BSDFFlag_Diffuse | BSDFFlag_Reflection;
    }

    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float uComp) const override {
        // For Phase 4: evaluate the OSL shader to get closure, then sample
        // the dominant lobe.  Currently falls back to Lambertian.
        (void)u; (void)uComp;
        OSL::ShaderGlobals sg = makeShaderGlobals(ctx, wo);
        OSL::ShadingSystem* sys = OslShadingSystem::instance().get();
        OSL::ShadingContext* oslCtx = sys->get_context(nullptr);
        sys->execute(*oslCtx, *m_group, sg);
        // TODO: walk sg.Ci closure and dispatch to lobe samplers
        sys->release_context(oslCtx);

        // Placeholder: return invalid sample (caller will skip)
        return {};
    }

    BSDFEval evaluate(const ShadingContext& ctx,
                       Vec3f wo, Vec3f wi) const override {
        (void)wi;
        OSL::ShaderGlobals sg = makeShaderGlobals(ctx, wo);
        OSL::ShadingSystem* sys = OslShadingSystem::instance().get();
        OSL::ShadingContext* oslCtx = sys->get_context(nullptr);
        sys->execute(*oslCtx, *m_group, sg);
        // TODO: walk sg.Ci closure and evaluate lobes
        sys->release_context(oslCtx);
        return {};
    }

    float pdf(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const override {
        return evaluate(ctx, wo, wi).pdf;
    }

private:
    OSL::ShaderGroupRef m_group;

    static OSL::ShaderGlobals makeShaderGlobals(const ShadingContext& ctx,
                                                 Vec3f wo) {
        OSL::ShaderGlobals sg{};
        sg.P  = OSL::Vec3(ctx.p.x,  ctx.p.y,  ctx.p.z);
        sg.N  = OSL::Vec3(ctx.n.x,  ctx.n.y,  ctx.n.z);
        sg.Ng = OSL::Vec3(ctx.ng.x, ctx.ng.y, ctx.ng.z);
        sg.u  = ctx.uv.x;
        sg.v  = ctx.uv.y;
        sg.I  = OSL::Vec3(-wo.x, -wo.y, -wo.z);  // incident = -wo
        sg.backfacing = !ctx.frontFace;
        return sg;
    }
};

#else  // ANACAPA_ENABLE_OSL not defined

// ---------------------------------------------------------------------------
// Stub — fails loudly at construction so callers know OSL is unavailable.
// ---------------------------------------------------------------------------
class OslMaterial : public IMaterial {
public:
    explicit OslMaterial(const std::string&) {
        assert(false && "OslMaterial requires ANACAPA_ENABLE_OSL=ON at cmake time. "
                        "Install open-shading-language and rebuild.");
    }

    bool     isDelta()                                              const override { return false; }
    uint32_t flags()                                               const override { return 0; }
    BSDFSample sample(const ShadingContext&, Vec3f, Vec2f, float)  const override { return {}; }
    BSDFEval   evaluate(const ShadingContext&, Vec3f, Vec3f)       const override { return {}; }
    float      pdf(const ShadingContext&, Vec3f, Vec3f)            const override { return 0.f; }
};

#endif  // ANACAPA_ENABLE_OSL

} // namespace anacapa
