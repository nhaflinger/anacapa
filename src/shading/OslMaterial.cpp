// OslMaterial.cpp — implementation of the OSL material adapter.
//
// This file intentionally does NOT include spdlog.  OSL headers pull in OIIO
// which re-exports system fmt 12.x (via OpenImageIO/detail/fmt/), while
// spdlog bundles fmt 10.x.  These two fmt versions are incompatible in the
// same TU because fmt/ostream.h from the system includes fmt/chrono.h which
// expects fmt 12 internals.  By keeping spdlog out of this file we avoid the
// conflict entirely.  Logging here uses fprintf(stderr) instead.
#include <cstdio>
#include <cstring>

#include "OslMaterial.h"
#include "MarschnerHair.h"

#ifdef ANACAPA_ENABLE_OSL

#include <OSL/oslexec.h>
#include <OSL/oslcomp.h>
#include <OSL/genclosure.h>
#include <OSL/oslclosure.h>
#include <OSL/rendererservices.h>

#include "StandardSurface.h"
#include <anacapa/shading/ShadingContext.h>

#include <cassert>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace anacapa {

// reflect(I, N) = I - 2*dot(N,I)*N
inline Vec3f oslReflect(Vec3f I, Vec3f N) {
    return I - N * (2.f * dot(N, I));
}

// ===========================================================================
// Closure param structs
//
// These must match the OSL stdosl.h function signatures EXACTLY in field
// order and type.  OSL's JIT fills them in based on our register_closure()
// calls.  We omit optional keyword params (label, thinfilm_*) to keep the
// structs minimal; OSL simply skips any unregistered optional params.
// ===========================================================================

struct OslEmptyParams   {};

// diffuse(N)  [standard OSL]
struct OslDiffuseParams { OSL::Vec3 N; };

// oren_nayar(N, sigma)  [standard OSL]
struct OslOrenNayarParams { OSL::Vec3 N; float sigma; };

// OSL 1.13+ uses ustringhash for string closure params; 1.12 uses ustring.
#if OSL_LIBRARY_VERSION_MINOR >= 13
    using OslStringParam = OSL::ustringhash;
#else
    using OslStringParam = OSL::ustring;
#endif

// microfacet(dist, N, U, xalpha, yalpha, eta, refract)  [standard OSL]
struct OslMicrofacetParams {
    OslStringParam dist;
    OSL::Vec3 N, U;
    float xalpha, yalpha, eta;
    int refract; // 0=refl, 1=refr, 2=both
};

// reflection(N [, eta])  [standard OSL – two registrations]
struct OslReflectionParams { OSL::Vec3 N; float eta; };

// refraction(N, eta)  [standard OSL]
struct OslRefractionParams { OSL::Vec3 N; float eta; };

// oren_nayar_diffuse_bsdf(N, albedo, roughness)  [MaterialX]
struct OslMxONDParams { OSL::Vec3 N; OSL::Color3 albedo; float roughness; };

// burley_diffuse_bsdf(N, albedo, roughness)  [MaterialX]
struct OslMxBurleyParams { OSL::Vec3 N; OSL::Color3 albedo; float roughness; };

// translucent_bsdf(N, albedo)  [MaterialX]
struct OslMxTranslucentParams { OSL::Vec3 N; OSL::Color3 albedo; };

// sheen_bsdf(N, albedo, roughness)  [MaterialX]
struct OslMxSheenParams { OSL::Vec3 N; OSL::Color3 albedo; float roughness; };

// uniform_edf(emittance)  [MaterialX]
struct OslMxUniformEdfParams { OSL::Color3 emittance; };

// transparent_bsdf()  [MaterialX]  – uses OslEmptyParams

// dielectric_bsdf(N, U, refl_tint, refr_tint, rx, ry, ior, dist)  [MaterialX]
struct OslMxDielectricParams {
    OSL::Vec3    N, U;
    OSL::Color3  refl_tint;
    OSL::Color3  refr_tint;
    float        roughness_x, roughness_y;
    float        ior;
    OslStringParam distribution;
};

// conductor_bsdf(N, U, rx, ry, ior, extinction, dist)  [MaterialX]
struct OslMxConductorParams {
    OSL::Vec3    N, U;
    float        roughness_x, roughness_y;
    OSL::Color3  ior;
    OSL::Color3  extinction;
    OslStringParam distribution;
};

// generalized_schlick_bsdf(N, U, refl_tint, refr_tint, rx, ry, f0, f90, exp, dist)
struct OslMxSchlickParams {
    OSL::Vec3    N, U;
    OSL::Color3  refl_tint;
    OSL::Color3  refr_tint;
    float        roughness_x, roughness_y;
    OSL::Color3  f0, f90;
    float        exponent;
    OslStringParam distribution;
};

// layer(top, base)  [MaterialX]
struct OslMxLayerParams {
    OSL::ClosureColor* top;
    OSL::ClosureColor* base;
};

// subsurface_bssrdf(N, albedo, radius, anisotropy)  [MaterialX]
struct OslMxSubsurfaceParams {
    OSL::Vec3   N;
    OSL::Color3 albedo;
    OSL::Color3 radius;
    float       anisotropy;
};

// anisotropic_vdf(albedo, extinction, anisotropy)  [MaterialX]
struct OslMxAnisotropicVdfParams {
    OSL::Color3 albedo;
    OSL::Color3 extinction;
    float       anisotropy;
};

// marschner_hair(sigma_a, beta_m, beta_n, alpha, eta)  [anacapa custom]
// sigma_a : absorption coefficient (linear, not artist color)
// beta_m  : longitudinal roughness ∈ [0,1]
// beta_n  : azimuthal roughness ∈ [0,1]
// alpha   : cuticle tilt in degrees (typically -2 to -4)
// eta     : index of refraction (typically 1.55)
struct OslMarschnerParams {
    OSL::Color3 sigma_a;
    float       beta_m;
    float       beta_n;
    float       alpha;
    float       eta;
};

// ===========================================================================
// Closure IDs (renderer-assigned — must be unique positive integers)
// ===========================================================================
enum OslClosureId : int {
    OSL_CID_EMISSION       = 1,
    OSL_CID_DIFFUSE        = 2,
    OSL_CID_OREN_NAYAR     = 3,
    OSL_CID_TRANSLUCENT    = 4,
    OSL_CID_MICROFACET     = 5,
    OSL_CID_REFLECTION     = 6,
    OSL_CID_REFLECTION_F   = 7,  // with eta
    OSL_CID_REFRACTION     = 8,
    OSL_CID_TRANSPARENT    = 9,
    // MaterialX
    OSL_CID_MX_OND         = 10,
    OSL_CID_MX_BURLEY      = 11,
    OSL_CID_MX_TRANSLUCENT = 12,
    OSL_CID_MX_SHEEN       = 13,
    OSL_CID_MX_EDF         = 14,
    OSL_CID_MX_TRANSPARENT = 15,
    OSL_CID_MX_DIELECTRIC  = 16,
    OSL_CID_MX_CONDUCTOR   = 17,
    OSL_CID_MX_SCHLICK     = 18,
    OSL_CID_MX_LAYER            = 19,
    OSL_CID_MX_SUBSURFACE       = 20,
    OSL_CID_MX_ANISOTROPIC_VDF  = 21,
    OSL_CID_MARSCHNER_HAIR      = 22,  // anacapa custom hair closure
};

// ===========================================================================
// OslRendererServices
// ===========================================================================
class OslRendererServices : public OSL::RendererServices {
public:
    explicit OslRendererServices(OSL::TextureSystem* texsys = nullptr)
        : OSL::RendererServices(texsys) {}

#if OSL_LIBRARY_VERSION_MINOR >= 13
    bool texture(OSL::ustringhash filename,
                 TextureHandle* texture_handle,
                 TexturePerthread* texture_thread_info,
                 OSL::TextureOpt& options,
                 OSL::ShaderGlobals* sg,
                 float s, float t,
                 float dsdx, float dtdx,
                 float dsdy, float dtdy,
                 int nchannels, float* result,
                 float* dresultds, float* dresultdt,
                 OSL::ustringhash* errormessage) override {
        return OSL::RendererServices::texture(
            filename, texture_handle, texture_thread_info, options, sg,
            s, t, dsdx, dtdx, dsdy, dtdy,
            nchannels, result, dresultds, dresultdt, errormessage);
    }

    bool trace(OSL::TraceOpt& opt, OSL::ShaderGlobals* sg,
               const OSL::Vec3& P, const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
               const OSL::Vec3& R, const OSL::Vec3& dRdx, const OSL::Vec3& dRdy) override {
        return false;
    }

    bool get_attribute(OSL::ShaderGlobals* sg, bool /*deriv*/,
                       OSL::ustringhash /*object*/, OSL::TypeDesc type,
                       OSL::ustringhash name, void* val) override {
        if (!sg || !sg->renderstate || !val) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);

        static const OSL::ustringhash h_is_curve    (OSL::ustring("is_curve"));
        static const OSL::ustringhash h_hair_h      (OSL::ustring("hair_h"));
        static const OSL::ustringhash h_hair_tangent(OSL::ustring("hair_tangent"));
        static const OSL::ustringhash h_hair_v      (OSL::ustring("hair_v"));
        static const OSL::ustringhash h_hair_color  (OSL::ustring("hair_color"));

        if (name == h_is_curve && type == OSL::TypeInt) {
            *static_cast<int*>(val) = ctx->isCurve ? 1 : 0;
            return true;
        }
        if (name == h_hair_h && type == OSL::TypeFloat) {
            *static_cast<float*>(val) = ctx->h;
            return true;
        }
        if (name == h_hair_tangent && type == OSL::TypeVector) {
            *static_cast<OSL::Vec3*>(val) =
                OSL::Vec3(ctx->t.x, ctx->t.y, ctx->t.z);
            return true;
        }
        if (name == h_hair_v && type == OSL::TypeFloat) {
            *static_cast<float*>(val) = ctx->strandV;
            return true;
        }
        if (name == h_hair_color && type == OSL::TypeColor) {
            *static_cast<OSL::Color3*>(val) =
                OSL::Color3(ctx->color.x, ctx->color.y, ctx->color.z);
            return true;
        }
        return false;
    }

    // String-based get_matrix/get_inverse_matrix: called by OSL when evaluating
    // transform("object", P) or transform("world", P) in shader code.
    // "object" → object2common = object-to-world; inverse = world-to-object.
    // Convention: Mat4f is column-vector; Imath Matrix44 is row-vector.
    // Transpose on copy: result.x[i][j] = m.m[j][i].
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::ustringhash from, float /*time*/) override {
        if (!sg || !sg->renderstate) return false;
        static const OSL::ustringhash s_object(OSL::ustring("object"));
        if (from != s_object) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);
        const Mat4f& m = ctx->objectToWorld;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.x[i][j] = m.m[j][i];
        return true;
    }
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::ustringhash from) override {
        return get_matrix(sg, result, from, 0.f);
    }
    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::ustringhash to, float /*time*/) override {
        if (!sg || !sg->renderstate) return false;
        static const OSL::ustringhash s_object(OSL::ustring("object"));
        if (to != s_object) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);
        const Mat4f& m = ctx->worldToObject;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.x[i][j] = m.m[j][i];
        return true;
    }
    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::ustringhash to) override {
        return get_inverse_matrix(sg, result, to, 0.f);
    }
#else
    bool texture(OSL::ustring filename,
                 TextureHandle* texture_handle,
                 TexturePerthread* texture_thread_info,
                 OSL::TextureOpt& options,
                 OSL::ShaderGlobals* sg,
                 float s, float t,
                 float dsdx, float dtdx,
                 float dsdy, float dtdy,
                 int nchannels, float* result,
                 float* dresultds, float* dresultdt,
                 OSL::ustring* errormessage) override {
        return OSL::RendererServices::texture(
            filename, texture_handle, texture_thread_info, options, sg,
            s, t, dsdx, dtdx, dsdy, dtdy,
            nchannels, result, dresultds, dresultdt, errormessage);
    }

    bool trace(OSL::RendererServices::TraceOpt& opt, OSL::ShaderGlobals* sg,
               const OSL::Vec3& P, const OSL::Vec3& dPdx, const OSL::Vec3& dPdy,
               const OSL::Vec3& R, const OSL::Vec3& dRdx, const OSL::Vec3& dRdy) override {
        return false;
    }

    bool get_attribute(OSL::ShaderGlobals* sg, bool /*deriv*/,
                       OSL::ustring /*object*/, OSL::TypeDesc type,
                       OSL::ustring name, void* val) override {
        if (!sg || !sg->renderstate || !val) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);

        if (name == OSL::ustring("is_curve") && type == OSL::TypeInt) {
            *static_cast<int*>(val) = ctx->isCurve ? 1 : 0;
            return true;
        }
        if (name == OSL::ustring("hair_h") && type == OSL::TypeFloat) {
            *static_cast<float*>(val) = ctx->h;
            return true;
        }
        if (name == OSL::ustring("hair_tangent") && type == OSL::TypeVector) {
            *static_cast<OSL::Vec3*>(val) =
                OSL::Vec3(ctx->t.x, ctx->t.y, ctx->t.z);
            return true;
        }
        if (name == OSL::ustring("hair_v") && type == OSL::TypeFloat) {
            *static_cast<float*>(val) = ctx->strandV;
            return true;
        }
        if (name == OSL::ustring("hair_color") && type == OSL::TypeColor) {
            *static_cast<OSL::Color3*>(val) =
                OSL::Color3(ctx->color.x, ctx->color.y, ctx->color.z);
            return true;
        }
        return false;
    }

    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::ustring from, float /*time*/) override {
        if (!sg || !sg->renderstate) return false;
        if (from != OSL::ustring("object")) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);
        const Mat4f& m = ctx->objectToWorld;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.x[i][j] = m.m[j][i];
        return true;
    }
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::ustring from) override {
        return get_matrix(sg, result, from, 0.f);
    }
    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::ustring to, float /*time*/) override {
        if (!sg || !sg->renderstate) return false;
        if (to != OSL::ustring("object")) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);
        const Mat4f& m = ctx->worldToObject;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.x[i][j] = m.m[j][i];
        return true;
    }
    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::ustring to) override {
        return get_inverse_matrix(sg, result, to, 0.f);
    }
#endif

    // get_matrix / get_inverse_matrix — called by OSL when evaluating
    // position(space="object") or any transform() call referencing "object".
    // sg.object2common is set to a non-null sentinel before shader execution;
    // OSL passes it back here so we know which space is being requested.
    // The actual matrices are fetched from the ShadingContext in renderstate.
    //
    // CONVENTION NOTE: our Mat4f is column-vector (translation in column 3,
    // result = M * v).  OSL's Imath::Matrix44 is row-vector (translation in
    // row 3, result = v * M).  A raw memcpy would give OSL the transpose of
    // the intended matrix, so we copy with i/j swapped to transpose on the way in.
    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::TransformationPtr xform, float /*time*/) override {
        if (!xform || !sg || !sg->renderstate) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);
        const Mat4f& m = ctx->objectToWorld;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.x[i][j] = m.m[j][i];
        return true;
    }

    bool get_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                    OSL::TransformationPtr xform) override {
        return get_matrix(sg, result, xform, 0.f);
    }

    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform, float /*time*/) override {
        if (!xform || !sg || !sg->renderstate) return false;
        const ShadingContext* ctx =
            static_cast<const ShadingContext*>(sg->renderstate);
        const Mat4f& m = ctx->worldToObject;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                result.x[i][j] = m.m[j][i];
        return true;
    }

    bool get_inverse_matrix(OSL::ShaderGlobals* sg, OSL::Matrix44& result,
                            OSL::TransformationPtr xform) override {
        return get_inverse_matrix(sg, result, xform, 0.f);
    }
};

// ===========================================================================
// registerOslClosures — call once on the ShadingSystem before any execution.
// ===========================================================================
static void registerOslClosures(OSL::ShadingSystem* sys) {
    // The CLOSURE_*_PARAM macros reference TypeDesc/TypeVector/etc unqualified.
    // register_closure takes const ClosureParam*, so we use static arrays.
    using OSL::TypeDesc;
    using OSL::TypeFloat;
    using OSL::TypeColor;
    using OSL::TypeVector;
    using OSL::TypeString;
    using OSL::TypeInt;
    using OSL::TypePoint;
    using OSL::TypeNormal;
    using OSL::TypeMatrix;
    using OSL::ClosureParam;

    // Helper: register from a static initializer list
    auto reg = [&](const char* name, int id,
                   std::initializer_list<ClosureParam> params) {
        sys->register_closure(name, id, params.begin(), nullptr, nullptr);
    };

    // standard OSL
    reg("emission",    OSL_CID_EMISSION,
        { CLOSURE_FINISH_PARAM(OslEmptyParams) });
    reg("transparent", OSL_CID_TRANSPARENT,
        { CLOSURE_FINISH_PARAM(OslEmptyParams) });
    reg("diffuse", OSL_CID_DIFFUSE, {
        CLOSURE_VECTOR_PARAM(OslDiffuseParams, N),
        CLOSURE_FINISH_PARAM(OslDiffuseParams) });
    reg("oren_nayar", OSL_CID_OREN_NAYAR, {
        CLOSURE_VECTOR_PARAM(OslOrenNayarParams, N),
        CLOSURE_FLOAT_PARAM(OslOrenNayarParams, sigma),
        CLOSURE_FINISH_PARAM(OslOrenNayarParams) });
    reg("translucent", OSL_CID_TRANSLUCENT, {
        CLOSURE_VECTOR_PARAM(OslDiffuseParams, N),
        CLOSURE_FINISH_PARAM(OslDiffuseParams) });
    reg("microfacet", OSL_CID_MICROFACET, {
        CLOSURE_STRING_PARAM(OslMicrofacetParams, dist),
        CLOSURE_VECTOR_PARAM(OslMicrofacetParams, N),
        CLOSURE_VECTOR_PARAM(OslMicrofacetParams, U),
        CLOSURE_FLOAT_PARAM(OslMicrofacetParams, xalpha),
        CLOSURE_FLOAT_PARAM(OslMicrofacetParams, yalpha),
        CLOSURE_FLOAT_PARAM(OslMicrofacetParams, eta),
        CLOSURE_INT_PARAM(OslMicrofacetParams, refract),
        CLOSURE_FINISH_PARAM(OslMicrofacetParams) });
    reg("reflection", OSL_CID_REFLECTION, {
        CLOSURE_VECTOR_PARAM(OslReflectionParams, N),
        CLOSURE_FINISH_PARAM(OslReflectionParams) });
    reg("reflection", OSL_CID_REFLECTION_F, {
        CLOSURE_VECTOR_PARAM(OslReflectionParams, N),
        CLOSURE_FLOAT_PARAM(OslReflectionParams, eta),
        CLOSURE_FINISH_PARAM(OslReflectionParams) });
    reg("refraction", OSL_CID_REFRACTION, {
        CLOSURE_VECTOR_PARAM(OslRefractionParams, N),
        CLOSURE_FLOAT_PARAM(OslRefractionParams, eta),
        CLOSURE_FINISH_PARAM(OslRefractionParams) });

    // MaterialX closures
    reg("oren_nayar_diffuse_bsdf", OSL_CID_MX_OND, {
        CLOSURE_VECTOR_PARAM(OslMxONDParams, N),
        CLOSURE_COLOR_PARAM(OslMxONDParams, albedo),
        CLOSURE_FLOAT_PARAM(OslMxONDParams, roughness),
        CLOSURE_FINISH_PARAM(OslMxONDParams) });
    reg("burley_diffuse_bsdf", OSL_CID_MX_BURLEY, {
        CLOSURE_VECTOR_PARAM(OslMxBurleyParams, N),
        CLOSURE_COLOR_PARAM(OslMxBurleyParams, albedo),
        CLOSURE_FLOAT_PARAM(OslMxBurleyParams, roughness),
        CLOSURE_FINISH_PARAM(OslMxBurleyParams) });
    reg("translucent_bsdf", OSL_CID_MX_TRANSLUCENT, {
        CLOSURE_VECTOR_PARAM(OslMxTranslucentParams, N),
        CLOSURE_COLOR_PARAM(OslMxTranslucentParams, albedo),
        CLOSURE_FINISH_PARAM(OslMxTranslucentParams) });
    reg("sheen_bsdf", OSL_CID_MX_SHEEN, {
        CLOSURE_VECTOR_PARAM(OslMxSheenParams, N),
        CLOSURE_COLOR_PARAM(OslMxSheenParams, albedo),
        CLOSURE_FLOAT_PARAM(OslMxSheenParams, roughness),
        CLOSURE_FINISH_PARAM(OslMxSheenParams) });
    reg("uniform_edf", OSL_CID_MX_EDF, {
        CLOSURE_COLOR_PARAM(OslMxUniformEdfParams, emittance),
        CLOSURE_FINISH_PARAM(OslMxUniformEdfParams) });
    reg("transparent_bsdf", OSL_CID_MX_TRANSPARENT,
        { CLOSURE_FINISH_PARAM(OslEmptyParams) });
    reg("dielectric_bsdf", OSL_CID_MX_DIELECTRIC, {
        CLOSURE_VECTOR_PARAM(OslMxDielectricParams, N),
        CLOSURE_VECTOR_PARAM(OslMxDielectricParams, U),
        CLOSURE_COLOR_PARAM(OslMxDielectricParams, refl_tint),
        CLOSURE_COLOR_PARAM(OslMxDielectricParams, refr_tint),
        CLOSURE_FLOAT_PARAM(OslMxDielectricParams, roughness_x),
        CLOSURE_FLOAT_PARAM(OslMxDielectricParams, roughness_y),
        CLOSURE_FLOAT_PARAM(OslMxDielectricParams, ior),
        CLOSURE_STRING_PARAM(OslMxDielectricParams, distribution),
        CLOSURE_FINISH_PARAM(OslMxDielectricParams) });
    reg("conductor_bsdf", OSL_CID_MX_CONDUCTOR, {
        CLOSURE_VECTOR_PARAM(OslMxConductorParams, N),
        CLOSURE_VECTOR_PARAM(OslMxConductorParams, U),
        CLOSURE_FLOAT_PARAM(OslMxConductorParams, roughness_x),
        CLOSURE_FLOAT_PARAM(OslMxConductorParams, roughness_y),
        CLOSURE_COLOR_PARAM(OslMxConductorParams, ior),
        CLOSURE_COLOR_PARAM(OslMxConductorParams, extinction),
        CLOSURE_STRING_PARAM(OslMxConductorParams, distribution),
        CLOSURE_FINISH_PARAM(OslMxConductorParams) });
    reg("generalized_schlick_bsdf", OSL_CID_MX_SCHLICK, {
        CLOSURE_VECTOR_PARAM(OslMxSchlickParams, N),
        CLOSURE_VECTOR_PARAM(OslMxSchlickParams, U),
        CLOSURE_COLOR_PARAM(OslMxSchlickParams, refl_tint),
        CLOSURE_COLOR_PARAM(OslMxSchlickParams, refr_tint),
        CLOSURE_FLOAT_PARAM(OslMxSchlickParams, roughness_x),
        CLOSURE_FLOAT_PARAM(OslMxSchlickParams, roughness_y),
        CLOSURE_COLOR_PARAM(OslMxSchlickParams, f0),
        CLOSURE_COLOR_PARAM(OslMxSchlickParams, f90),
        CLOSURE_FLOAT_PARAM(OslMxSchlickParams, exponent),
        CLOSURE_STRING_PARAM(OslMxSchlickParams, distribution),
        CLOSURE_FINISH_PARAM(OslMxSchlickParams) });
    // layer(top, base) — must declare CLOSURE_CLOSURE_PARAM for each child
    // so OSL's JIT fills the pointer fields into the params struct.
    reg("layer", OSL_CID_MX_LAYER, {
        CLOSURE_CLOSURE_PARAM(OslMxLayerParams, top),
        CLOSURE_CLOSURE_PARAM(OslMxLayerParams, base),
        CLOSURE_FINISH_PARAM(OslMxLayerParams) });
    reg("subsurface_bssrdf", OSL_CID_MX_SUBSURFACE, {
        CLOSURE_VECTOR_PARAM(OslMxSubsurfaceParams, N),
        CLOSURE_COLOR_PARAM(OslMxSubsurfaceParams, albedo),
        CLOSURE_COLOR_PARAM(OslMxSubsurfaceParams, radius),
        CLOSURE_FLOAT_PARAM(OslMxSubsurfaceParams, anisotropy),
        CLOSURE_FINISH_PARAM(OslMxSubsurfaceParams) });
    reg("anisotropic_vdf", OSL_CID_MX_ANISOTROPIC_VDF, {
        CLOSURE_COLOR_PARAM(OslMxAnisotropicVdfParams, albedo),
        CLOSURE_COLOR_PARAM(OslMxAnisotropicVdfParams, extinction),
        CLOSURE_FLOAT_PARAM(OslMxAnisotropicVdfParams, anisotropy),
        CLOSURE_FINISH_PARAM(OslMxAnisotropicVdfParams) });

    // anacapa custom — Marschner hair BSDF with explicit absorption coefficient
    reg("marschner_hair", OSL_CID_MARSCHNER_HAIR, {
        CLOSURE_COLOR_PARAM(OslMarschnerParams, sigma_a),
        CLOSURE_FLOAT_PARAM(OslMarschnerParams, beta_m),
        CLOSURE_FLOAT_PARAM(OslMarschnerParams, beta_n),
        CLOSURE_FLOAT_PARAM(OslMarschnerParams, alpha),
        CLOSURE_FLOAT_PARAM(OslMarschnerParams, eta),
        CLOSURE_FINISH_PARAM(OslMarschnerParams) });
}

// ===========================================================================
// OslShadingSystem — singleton
// ===========================================================================
class OslShadingSystem {
public:
    static OslShadingSystem& instance() {
        static OslShadingSystem s;
        return s;
    }

    OSL::ShadingSystem* sys() { return m_sys; }

    // Return the PerThreadInfo for the calling thread, creating it on first use.
    OSL::PerThreadInfo* threadInfo() {
        thread_local OSL::PerThreadInfo* ti = nullptr;
        if (!ti) ti = m_sys->create_thread_info();
        return ti;
    }

    // Add a directory to the .oso search path (call before loading shaders).
    void addSearchPath(const std::string& dir) {
        m_searchPaths += dir + ":";
        m_sys->attribute("searchpath:shader",
                         OSL::ustring(m_searchPaths));
    }

private:
    // OSL::TextureSystem::create() returns TextureSystem* in OIIO 2.x and
    // shared_ptr<TextureSystem> in OIIO 3+. Overload-dispatch handles both.
    static OSL::TextureSystem* assignTexSys(OslShadingSystem* self,
                                            OSL::TextureSystem* raw) {
        self->m_texRaw = raw;
        return raw;
    }
    static OSL::TextureSystem* assignTexSys(OslShadingSystem* self,
                                            std::shared_ptr<OSL::TextureSystem> sp) {
        self->m_textureSystem = std::move(sp);
        return self->m_textureSystem.get();
    }

    OslShadingSystem() {
        OSL::TextureSystem* texRaw =
            assignTexSys(this, OSL::TextureSystem::create(true));
        m_services = OslRendererServices(texRaw);
        m_sys = new OSL::ShadingSystem(&m_services, texRaw);
        registerOslClosures(m_sys);
    }
    ~OslShadingSystem() {
        delete m_sys;
        // OIIO 2.x: destroy the raw pointer we still own.  OIIO 3+: the
        // shared_ptr (m_textureSystem) handles its own teardown — leaving
        // m_texRaw null in that case so nothing to do here.
#if OIIO_VERSION_MAJOR >= 3
        if (m_textureSystem) OSL::TextureSystem::destroy(m_textureSystem);
#else
        if (m_texRaw) OSL::TextureSystem::destroy(m_texRaw);
#endif
    }

    OslRendererServices  m_services;
    OSL::ShadingSystem*  m_sys = nullptr;
    std::string          m_searchPaths;
    std::shared_ptr<OSL::TextureSystem> m_textureSystem;  // owned in OIIO 3+
    OSL::TextureSystem*                 m_texRaw = nullptr;  // owned in OIIO 2.x
};

// ===========================================================================
// Internal lobe accumulation
// ===========================================================================
struct OslLobe {
    enum class Kind {
        Diffuse,       // Lambertian (upper hemisphere)
        DiffuseTrans,  // translucent_bsdf — Lambertian through the surface (lower hemisphere)
        GGXRefl,       // GGX reflection only
        GGXTrans,      // GGX transmission only
        GGXBoth,       // GGX reflection + transmission (dielectric)
        Transparent,   // straight-through (alpha cut-out)
        Emission,      // emitted radiance (contributes to Le only)
        MarschnerHair, // Marschner R/TT/TRT — dispatches to C++ implementation
    } kind;

    Spectrum weight   = {};       // accumulated path weight
    float    alpha2   = 0.25f;    // GGX roughness^2 (perceptual r^2)
    Spectrum f0       = {0.04f, 0.04f, 0.04f};   // Schlick F0
    Spectrum f90      = {1.f, 1.f, 1.f};          // Schlick F90
    float    schlickExp = 5.f;
    Spectrum refl_tint  = {1.f, 1.f, 1.f};
    Spectrum refr_tint  = {};
    float    ior        = 1.5f;
    Spectrum albedo     = {0.8f, 0.8f, 0.8f};   // diffuse albedo
    Spectrum emittance  = {};

    // MarschnerHair kind only: params for the C++ Marschner implementation
    struct MarschnerParams {
        Spectrum sigma_a = {0.06f, 0.10f, 0.20f};
        float    beta_m  = 0.30f;
        float    beta_n  = 0.60f;
        float    alpha   = -2.f;
        float    eta     = 1.55f;
    } marschner;
};

// Schlick generalized Fresnel: f0 + (f90-f0)*(1-cos)^exp
inline Spectrum schlickGeneral(float cosTheta, const Spectrum& f0,
                                const Spectrum& f90, float exp) {
    float c = std::max(0.f, 1.f - std::abs(cosTheta));
    float p = std::pow(c, exp);
    return f0 + (f90 - f0) * p;
}

// Walk the OSL closure tree into a flat list of lobes.
// Uses an iterative stack to avoid recursion.
// frontFace: true if the ray hit the front side of the surface.
// For backface hits (exiting glass), transmission lobes flip their eta.
static void collectLobes(const OSL::ClosureColor* c,
                         Spectrum weight,
                         std::vector<OslLobe>& out,
                         bool frontFace = true) {
    if (!c) return;

    struct Frame { const OSL::ClosureColor* c; Spectrum w; };
    Frame stack[32];
    int depth = 0;
    stack[depth++] = { c, weight };

    while (depth > 0) {
        auto [cur, w] = stack[--depth];
        if (!cur) continue;

        switch (cur->id) {
        case OSL::ClosureColor::MUL: {
            const auto* m = cur->as_mul();
            Spectrum mw = { m->weight[0], m->weight[1], m->weight[2] };
            stack[depth++] = { m->closure, w * mw };
            break;
        }
        case OSL::ClosureColor::ADD: {
            const auto* a = cur->as_add();
            stack[depth++] = { a->closureA, w };
            stack[depth++] = { a->closureB, w };
            break;
        }
        case OSL_CID_MX_LAYER: {
            // layer(top, base) — OSL's JIT writes both closure pointers into
            // the data area sized by CLOSURE_FINISH_PARAM(OslMxLayerParams).
            const auto* p = cur->as_comp()->as<OslMxLayerParams>();
            if (depth + 2 <= 32) {
                stack[depth++] = { p->top,  w };
                stack[depth++] = { p->base, w };
            }
            break;
        }
        // ---- Emission ----
        case OSL_CID_EMISSION: {
            OslLobe lobe;
            lobe.kind = OslLobe::Kind::Emission;
            lobe.weight = w;
            lobe.emittance = w;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_MX_EDF: {
            const auto* p = cur->as_comp()->as<OslMxUniformEdfParams>();
            OslLobe lobe;
            lobe.kind = OslLobe::Kind::Emission;
            lobe.emittance = w * Spectrum{p->emittance[0],
                                          p->emittance[1],
                                          p->emittance[2]};
            out.push_back(lobe);
            break;
        }
        // ---- Transparent ----
        case OSL_CID_TRANSPARENT:
        case OSL_CID_MX_TRANSPARENT: {
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Transparent;
            lobe.weight = w;
            out.push_back(lobe);
            break;
        }
        // ---- Lambertian diffuse ----
        case OSL_CID_DIFFUSE: {
            const auto* p = cur->as_comp()->as<OslDiffuseParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Diffuse;
            lobe.weight = w;
            lobe.albedo = w;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_TRANSLUCENT: {
            const auto* p = cur->as_comp()->as<OslDiffuseParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::DiffuseTrans;
            lobe.weight = w;
            lobe.albedo = w;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_OREN_NAYAR: {
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Diffuse;
            lobe.weight = w;
            lobe.albedo = w;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_MX_OND:
        case OSL_CID_MX_BURLEY: {
            const auto* p = cur->as_comp()->as<OslMxONDParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Diffuse;
            Spectrum alb = { p->albedo[0], p->albedo[1], p->albedo[2] };
            lobe.albedo  = w * alb;
            lobe.weight  = lobe.albedo;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_MX_TRANSLUCENT: {
            const auto* p = cur->as_comp()->as<OslMxONDParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::DiffuseTrans;
            Spectrum alb = { p->albedo[0], p->albedo[1], p->albedo[2] };
            lobe.albedo  = w * alb;
            lobe.weight  = lobe.albedo;
            out.push_back(lobe);
            break;
        }
        // ---- Standard GGX microfacet ----
        case OSL_CID_MICROFACET: {
            const auto* p = cur->as_comp()->as<OslMicrofacetParams>();
            float r  = std::max(p->xalpha, p->yalpha);
            float a2 = r * r;
            OslLobe lobe;
            lobe.weight  = w;
            lobe.alpha2  = a2;
            lobe.ior     = p->eta;
            lobe.f0      = { F0_fromIOR(p->eta), F0_fromIOR(p->eta),
                             F0_fromIOR(p->eta) };
            lobe.f90     = {1,1,1};
            lobe.schlickExp = 5.f;
            lobe.refl_tint  = {1,1,1};
            lobe.kind    = (p->refract == 1) ? OslLobe::Kind::GGXTrans
                         : (p->refract == 2) ? OslLobe::Kind::GGXBoth
                                             : OslLobe::Kind::GGXRefl;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_REFLECTION:
        case OSL_CID_REFLECTION_F: {
            const auto* p = cur->as_comp()->as<OslReflectionParams>();
            OslLobe lobe;
            lobe.kind    = OslLobe::Kind::GGXRefl;
            lobe.weight  = w;
            lobe.alpha2  = 1e-6f;  // near-perfect mirror
            float f0s    = F0_fromIOR(p->eta > 0.f ? p->eta : 1.5f);
            lobe.f0      = { f0s, f0s, f0s };
            lobe.f90     = {1,1,1};
            lobe.schlickExp = 5.f;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_REFRACTION: {
            const auto* p = cur->as_comp()->as<OslRefractionParams>();
            OslLobe lobe;
            lobe.kind    = OslLobe::Kind::GGXTrans;
            lobe.weight  = w;
            lobe.alpha2  = 1e-6f;
            lobe.ior     = p->eta;
            out.push_back(lobe);
            break;
        }
        // ---- MaterialX dielectric ----
        case OSL_CID_MX_DIELECTRIC: {
            const auto* p = cur->as_comp()->as<OslMxDielectricParams>();
            float r  = std::max(p->roughness_x, p->roughness_y);
            float a  = r;   // perceptual → alpha = r (MaterialX convention)
            float a2 = a * a;
            OslLobe lobe;
            lobe.weight     = w;
            lobe.alpha2     = a2;
            // On backface (exiting glass), eta flips: glass/air → air/glass = 1/ior
            lobe.ior        = frontFace ? p->ior : (1.f / p->ior);
            float f0s       = F0_fromIOR(p->ior);
            lobe.f0         = { f0s, f0s, f0s };
            lobe.f90        = {1,1,1};
            lobe.schlickExp = 5.f;
            lobe.refl_tint  = { p->refl_tint[0], p->refl_tint[1], p->refl_tint[2] };
            lobe.refr_tint  = { p->refr_tint[0], p->refr_tint[1], p->refr_tint[2] };
            bool hasRefr = luminance(lobe.refr_tint) > 0.01f;
            bool hasRefl = luminance(lobe.refl_tint) > 0.01f;
            lobe.kind = hasRefr ? (hasRefl ? OslLobe::Kind::GGXBoth
                                           : OslLobe::Kind::GGXTrans)
                                : OslLobe::Kind::GGXRefl;
            out.push_back(lobe);
            break;
        }
        // ---- MaterialX conductor ----
        case OSL_CID_MX_CONDUCTOR: {
            const auto* p = cur->as_comp()->as<OslMxConductorParams>();
            float r  = std::max(p->roughness_x, p->roughness_y);
            float a2 = r * r;
            // Approximate conductor F0 from complex IOR (eta, k):
            // F0 ≈ ((eta-1)^2 + k^2) / ((eta+1)^2 + k^2)
            auto condF0 = [](float eta, float k) {
                float n1 = eta - 1.f, n2 = eta + 1.f;
                return (n1*n1 + k*k) / (n2*n2 + k*k);
            };
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::GGXRefl;
            lobe.weight = w;
            lobe.alpha2 = a2;
            lobe.f0     = { condF0(p->ior[0], p->extinction[0]),
                             condF0(p->ior[1], p->extinction[1]),
                             condF0(p->ior[2], p->extinction[2]) };
            lobe.f90    = {1,1,1};
            lobe.schlickExp = 5.f;
            out.push_back(lobe);
            break;
        }
        // ---- MaterialX generalized Schlick ----
        case OSL_CID_MX_SCHLICK: {
            const auto* p = cur->as_comp()->as<OslMxSchlickParams>();
            float r  = std::max(p->roughness_x, p->roughness_y);
            float a2 = r * r;
            OslLobe lobe;
            lobe.weight     = w;
            lobe.alpha2     = a2;
            lobe.f0         = { p->f0[0], p->f0[1], p->f0[2] };
            lobe.f90        = { p->f90[0], p->f90[1], p->f90[2] };
            lobe.schlickExp = p->exponent;
            lobe.refl_tint  = { p->refl_tint[0], p->refl_tint[1], p->refl_tint[2] };
            lobe.refr_tint  = { p->refr_tint[0], p->refr_tint[1], p->refr_tint[2] };
            // Derive IOR from F0 (average): ior = (1+√F0)/(1-√F0)
            float avgF0  = (lobe.f0.x + lobe.f0.y + lobe.f0.z) / 3.f;
            float sqrtF0 = std::sqrt(std::max(0.f, avgF0));
            float baseIor = (1.f + sqrtF0) / std::max(0.001f, 1.f - sqrtF0);
            // On backface (exiting glass), eta flips
            lobe.ior     = frontFace ? baseIor : (1.f / baseIor);
            bool hasRefr = luminance(lobe.refr_tint) > 0.01f;
            bool hasRefl = luminance(lobe.refl_tint) > 0.01f;
            lobe.kind = hasRefr ? (hasRefl ? OslLobe::Kind::GGXBoth
                                           : OslLobe::Kind::GGXTrans)
                                : OslLobe::Kind::GGXRefl;
            out.push_back(lobe);
            break;
        }
        // ---- Sheen / subsurface — approximate as diffuse ----
        case OSL_CID_MX_SHEEN: {
            const auto* p = cur->as_comp()->as<OslMxSheenParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Diffuse;
            lobe.albedo = w * Spectrum{p->albedo[0], p->albedo[1], p->albedo[2]};
            lobe.weight = lobe.albedo;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_MX_SUBSURFACE: {
            const auto* p = cur->as_comp()->as<OslMxSubsurfaceParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Diffuse;
            lobe.albedo = w * Spectrum{p->albedo[0], p->albedo[1], p->albedo[2]};
            lobe.weight = lobe.albedo;
            out.push_back(lobe);
            break;
        }
        case OSL_CID_MX_ANISOTROPIC_VDF: {
            // VDF: approximate as diffuse using the scattering albedo.
            const auto* p = cur->as_comp()->as<OslMxAnisotropicVdfParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::Diffuse;
            lobe.albedo = w * Spectrum{p->albedo[0], p->albedo[1], p->albedo[2]};
            lobe.weight = lobe.albedo;
            out.push_back(lobe);
            break;
        }
        // ---- anacapa custom: Marschner hair ----
        case OSL_CID_MARSCHNER_HAIR: {
            const auto* p = cur->as_comp()->as<OslMarschnerParams>();
            OslLobe lobe;
            lobe.kind   = OslLobe::Kind::MarschnerHair;
            lobe.weight = w;
            lobe.marschner.sigma_a = { p->sigma_a[0], p->sigma_a[1], p->sigma_a[2] };
            lobe.marschner.beta_m  = p->beta_m;
            lobe.marschner.beta_n  = p->beta_n;
            lobe.marschner.alpha   = p->alpha;
            lobe.marschner.eta     = p->eta;
            out.push_back(lobe);
            break;
        }
        default:
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Merge co-incident GGXRefl + GGXTrans lobe pairs into a single GGXBoth lobe.
//
// MaterialX OpenPBR emits dielectric reflection (scatter_mode "R") and
// transmission (scatter_mode "T") as separate lobes from the same dielectric.
// If we select between them using macro-surface Fresnel, then
// sampleGGXTransLobe can still return {} when a sampled microfacet hits TIR
// (because macro-Fresnel and microfacet-Fresnel disagree at grazing angles /
// high roughness).  This makes thick glass edges go black.
//
// Merging into GGXBoth lets the sampler use per-microfacet Fresnel: when a
// sampled half-vector would TIR, it reflects instead of killing the path.
// ---------------------------------------------------------------------------

static void mergeGGXPairs(std::vector<OslLobe>& lobes) {
    // The OSL closure tree can emit GGXTrans before or after the matching GGXRefl,
    // so we search all pairs (i, j) regardless of order.
    for (size_t i = 0; i < lobes.size(); ++i) {
        if (lobes[i].kind != OslLobe::Kind::GGXRefl) continue;
        if (!isBlack(lobes[i].refr_tint)) continue;  // already has refr — skip
        for (size_t j = 0; j < lobes.size(); ++j) {
            if (j == i) continue;
            if (lobes[j].kind != OslLobe::Kind::GGXTrans) continue;
            // Same ior and roughly same roughness → same dielectric
            if (std::abs(lobes[i].ior - lobes[j].ior) > 0.01f) continue;
            if (std::abs(lobes[i].alpha2 - lobes[j].alpha2) > 0.01f) continue;
            // Merge: combine tints into a single GGXBoth lobe
            lobes[i].kind      = OslLobe::Kind::GGXBoth;
            lobes[i].refr_tint = lobes[j].refr_tint;
            lobes.erase(lobes.begin() + j);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-lobe BSDF evaluation helpers (local frame: +Z = normal)
// ---------------------------------------------------------------------------

static BSDFEval evalDiffuseLobe(const OslLobe& lobe, Vec3f wo, Vec3f wi) {
    if (wo.z <= 0.f || wi.z <= 0.f) return {};
    float pdf = wi.z * kSS_InvPi;
    Spectrum f = lobe.albedo * kSS_InvPi;
    return { f, pdf, pdf };
}

static BSDFSample sampleDiffuseLobe(const OslLobe& lobe, Vec3f /*wo*/, Vec2f u) {
    // Cosine hemisphere sampling
    float phi = 2.f * kSS_Pi * u.x;
    float r   = std::sqrt(u.y);
    Vec3f wi  = { r * std::cos(phi), r * std::sin(phi),
                  std::sqrt(std::max(0.f, 1.f - u.y)) };
    float pdf = wi.z * kSS_InvPi;
    Spectrum f = lobe.albedo * kSS_InvPi;
    BSDFSample s;
    s.wi    = wi;
    s.f     = f * wi.z;
    s.pdf   = pdf;
    s.pdfRev = pdf;
    s.flags = BSDFFlag_Diffuse | BSDFFlag_Reflection;
    return s;
}

// translucent_bsdf: diffuse transmission — samples the LOWER hemisphere.
// wo is in the upper hemisphere (wo.z > 0); wi goes through the surface (wi.z < 0).
static BSDFEval evalDiffuseTransLobe(const OslLobe& lobe, Vec3f wo, Vec3f wi) {
    if (wo.z <= 0.f || wi.z >= 0.f) return {};
    float absCosWi = -wi.z;
    float pdf = absCosWi * kSS_InvPi;
    Spectrum f = lobe.albedo * kSS_InvPi;
    return { f, pdf, pdf };
}

static BSDFSample sampleDiffuseTransLobe(const OslLobe& lobe, Vec3f /*wo*/, Vec2f u) {
    // Cosine-weighted sampling of the lower hemisphere (wi.z < 0)
    float phi = 2.f * kSS_Pi * u.x;
    float r   = std::sqrt(u.y);
    float cosWi = std::sqrt(std::max(0.f, 1.f - u.y));
    Vec3f wi  = { r * std::cos(phi), r * std::sin(phi), -cosWi };
    float pdf = cosWi * kSS_InvPi;
    Spectrum f = lobe.albedo * kSS_InvPi;
    BSDFSample s;
    s.wi     = wi;
    s.f      = f * cosWi;
    s.pdf    = pdf;
    s.pdfRev = pdf;
    s.eta    = 1.f;
    s.flags  = BSDFFlag_Diffuse | BSDFFlag_Transmission;
    return s;
}

// Schlick Fresnel with generalized exponent
static Spectrum evalSchlick(float cosTheta, const OslLobe& lobe) {
    return schlickGeneral(cosTheta, lobe.f0, lobe.f90, lobe.schlickExp);
}

static BSDFEval evalGGXReflLobe(const OslLobe& lobe, Vec3f wo, Vec3f wi) {
    if (lobe.alpha2 < 1e-6f) return {};  // delta BSDF — cannot evaluate
    if (wo.z <= 0.f || wi.z <= 0.f) return {};
    Vec3f wh  = normalize(wo + wi);
    float cosH = wh.z;
    float dotVH = dot(wo, wh);
    if (cosH <= 0.f || dotVH <= 0.f) return {};
    float D   = D_GGX(cosH, lobe.alpha2);
    float G2  = G2_Smith_Separable(wo.z, wi.z, lobe.alpha2);
    Spectrum F  = evalSchlick(dotVH, lobe) * lobe.refl_tint;
    Spectrum f  = F * D * G2 / (4.f * wo.z * wi.z) * wi.z;
    float pdf   = pdfGGX_reflection(cosH, lobe.alpha2, dotVH);
    return { f * lobe.weight, pdf, pdf };
}

static BSDFSample sampleGGXReflLobe(const OslLobe& lobe, Vec3f wo, Vec2f u) {
    if (wo.z <= 0.f) return {};
    if (lobe.alpha2 < 1e-6f) {
        // Perfect mirror — avoids D_GGX→NaN for zero roughness.
        Vec3f wi = {-wo.x, -wo.y, wo.z};
        if (wi.z <= 0.f) return {};
        Spectrum Fr = evalSchlick(wo.z, lobe) * lobe.refl_tint;
        BSDFSample s;
        s.wi = wi; s.f = Fr * lobe.weight; s.pdf = 1.f; s.pdfRev = 1.f;
        s.eta = 1.f; s.flags = BSDFFlag_Specular | BSDFFlag_Reflection;
        return s;
    }
    Vec3f wh  = sampleGGX_halfvector(u, lobe.alpha2);
    float dotVH = dot(wo, wh);
    if (dotVH <= 0.f) return {};
    Vec3f wi  = oslReflect(-wo, wh);
    if (wi.z <= 0.f) return {};
    float cosH = wh.z;
    float D    = D_GGX(cosH, lobe.alpha2);
    float G2   = G2_Smith_Separable(wo.z, wi.z, lobe.alpha2);
    Spectrum F = evalSchlick(dotVH, lobe) * lobe.refl_tint;
    Spectrum f = F * D * G2 / (4.f * wo.z * wi.z) * wi.z;
    float pdf  = pdfGGX_reflection(cosH, lobe.alpha2, dotVH);
    if (pdf <= 0.f) return {};
    BSDFSample s;
    s.wi    = wi;
    s.f     = f * lobe.weight;
    s.pdf   = pdf;
    s.pdfRev = pdf;
    s.eta   = 1.f;
    s.flags = BSDFFlag_Glossy | BSDFFlag_Reflection;
    return s;
}

// GGX transmission lobe (refraction via Snell's law + GGX microfacet)
// Simplified: treat as a perfect refraction weighted by the GGX NDF.
static BSDFSample sampleGGXTransLobe(const OslLobe& lobe, Vec3f wo, Vec2f u) {
    if (wo.z <= 0.f) return {};
    if (lobe.alpha2 < 1e-6f) {
        // Perfect refraction — avoids D_GGX→NaN for zero roughness.
        float eta   = lobe.ior;
        float sin2T = std::max(0.f, 1.f - wo.z * wo.z) / (eta * eta);
        if (sin2T >= 1.f) return {};  // TIR
        float cosT  = std::sqrt(1.f - sin2T);
        // wi = (-wo.x/eta, -wo.y/eta, -cosT) — Snell's law in local frame
        Vec3f wi = {-wo.x / eta, -wo.y / eta, -cosT};
        if (wi.z >= 0.f) return {};
        float Ft = 1.f - fresnelDielectric(wo.z, eta);
        BSDFSample s;
        s.wi = wi; s.f = lobe.refr_tint * lobe.weight * Ft;
        s.pdf = Ft; s.pdfRev = Ft;
        s.eta = eta; s.flags = BSDFFlag_Specular | BSDFFlag_Transmission;
        return s;
    }
    // Sample a half-vector from GGX, then refract
    Vec3f wh = sampleGGX_halfvector(u, lobe.alpha2);
    float eta = lobe.ior;  // n_i / n_t
    float cosI = dot(wo, wh);
    if (cosI <= 0.f) return {};
    float sin2I = 1.f - cosI * cosI;
    float sin2T = sin2I / (eta * eta);
    if (sin2T >= 1.f) return {};  // TIR
    float cosT = std::sqrt(1.f - sin2T);
    // Refracted direction in local frame
    Vec3f wi = (-wo / eta) + wh * (cosI / eta - cosT);
    if (wi.z >= 0.f) return {};  // must go downward (transmission)
    wi = normalize(wi);

    float D   = D_GGX(wh.z, lobe.alpha2);
    float G2  = G2_Smith_Separable(wo.z, std::abs(wi.z), lobe.alpha2);
    float Ft  = 1.f - fresnelDielectric(cosI, eta);
    // btdf = |cosT| * D * G2 * Ft / (eta^2 * cosI * |cosT|) × Jacobian
    float jacobian = std::abs(cosT) / ((eta * eta) * std::abs(cosI + eta * std::abs(cosT)));
    float pdf  = D * wh.z * jacobian;
    Spectrum f = lobe.refr_tint * Ft * D * G2 * std::abs(wi.z) * jacobian;
    if (pdf <= 0.f) return {};
    BSDFSample s;
    s.wi    = wi;
    s.f     = f * lobe.weight;
    s.pdf   = pdf;
    s.pdfRev = pdf;
    s.eta   = 1.f / eta;
    s.flags = BSDFFlag_Glossy | BSDFFlag_Transmission;
    return s;
}

// ===========================================================================
// OslMaterial — concrete class, private to this translation unit.
// ===========================================================================
class OslMaterial : public IMaterial {
public:
    // Load a pre-compiled .oso shader by name (OSL searches searchpath:shader).
    // The shader name is the filename without the .oso extension.
    explicit OslMaterial(const std::string& shaderName)
        : m_shaderName(shaderName) {
        OSL::ShadingSystem* s = OslShadingSystem::instance().sys();
        // Load as "surface" so that OSL writes the shader's Ci value back to
        // sg.Ci after execute().  The generated MaterialX shaders inject
        // "Ci = out;" at the end of their body (via the Python exporter), so
        // loading as surface is what causes sg.Ci to be non-null.  Loading as
        // "shader" silently drops the Ci assignment — sg.Ci stays null.
        m_group = s->ShaderGroupBegin(shaderName);
        s->Shader(*m_group, "surface",
                  OSL::ustring(shaderName), OSL::ustring("layer1"));
        s->ShaderGroupEnd(*m_group);
        if (!m_group)
            throw std::runtime_error(
                "OslMaterial: failed to load shader '" + shaderName + "'");

        // Probe at normal incidence (unit vertical hit, front face) to detect
        // transmission lobes and cache the tint for transmittanceColor().
        // This is done once at construction so shadow-ray evaluation is free.
        {
            SurfaceInteraction si;
            si.n = si.ng = {0.f, 0.f, 1.f};
            si.dpdu = {1.f, 0.f, 0.f};
            si.dpdv = {0.f, 1.f, 0.f};
            si.uv   = {0.5f, 0.5f};
            ShadingContext probe(si, {0.f, 0.f, -1.f}); // ray going -z, front face
            Vec3f wo = {0.f, 0.f, 1.f};                 // outward = +z
            auto lobes = evalClosure(probe, wo);
            for (auto& l : lobes) {
                if ((l.kind == OslLobe::Kind::GGXTrans ||
                     l.kind == OslLobe::Kind::GGXBoth) && !isBlack(l.refr_tint)) {
                    m_transmittanceTint = {
                        std::min(1.f, l.refr_tint.x),
                        std::min(1.f, l.refr_tint.y),
                        std::min(1.f, l.refr_tint.z) };
                    break;
                }
            }
            // Cache the diffuse lobe weight as the base color for GPU/Metal preview.
            // Only use the diffuse lobe — GGX specular is handled separately by the
            // GPU shader and summing both would exceed 1.0 and make the scene too bright.
            // Fall back to 0.5 grey for pure-specular / glass materials.
            Spectrum diffuseColor{};
            for (auto& l : lobes) {
                if (l.kind == OslLobe::Kind::Diffuse) {
                    diffuseColor = diffuseColor + l.weight;
                }
            }
            float lum = luminance(diffuseColor);
            m_baseColor = (lum > 1e-4f) ? diffuseColor : Spectrum{0.5f, 0.5f, 0.5f};

            // Cache emission from probe for auto-registration of emissive meshes.
            for (auto& l : lobes) {
                if (l.kind == OslLobe::Kind::Emission)
                    m_probeEmission = m_probeEmission + l.emittance;
            }

            // Cache roughness from the dominant GGX lobe for the GPU preview.
            // alpha2 = roughness^4 in Disney convention, so roughness = (alpha2)^0.25.
            float bestWeight = 0.f;
            for (auto& l : lobes) {
                if (l.kind == OslLobe::Kind::GGXRefl ||
                    l.kind == OslLobe::Kind::GGXBoth) {
                    float w = luminance(l.weight);
                    if (w > bestWeight) {
                        bestWeight  = w;
                        // l.alpha2 is GGX alpha^2 where alpha = roughness^2,
                        // so perceptual roughness = sqrt(sqrt(l.alpha2))
                        // Floor at 0.25 — the GPU preview can't handle near-delta
                        // materials without MIS; values below this cause fireflies.
                        m_roughness = std::max(0.25f, std::sqrt(std::sqrt(std::max(0.f, l.alpha2))));
                    }
                }
            }
        }

        // Build the hair fallback from the probed OSL material parameters.
        // Used when ctx.isCurve == true to delegate to physically correct
        // Marschner hair shading instead of the normal-based GGX path.
        {
            MarschnerHairMaterial::Params hp;
            // sigma_a from m_baseColor only when the probed color has genuine
            // chroma (max - min > 0.05).  A grey probe (e.g. old metallic/conductor
            // shaders where no diffuse lobe exists) means the color carries no
            // useful hair information — keep the default medium-brown sigma_a.
            // Once the scene is re-exported with _tx_bsdf_hair_principled, the
            // new OSL shaders produce a diffuse lobe and m_baseColor will carry
            // the actual Principled Hair BSDF Color input.
            {
                const Spectrum& bc = m_baseColor;
                float bcMax = std::max({bc.x, bc.y, bc.z});
                float bcMin = std::min({bc.x, bc.y, bc.z});
                if (bcMax - bcMin > 0.05f) {
                    // Colorful probe → derive sigma_a via Beer-Lambert inversion
                    hp.sigma_a = {
                        -std::log(std::max(bc.x, 0.001f)),
                        -std::log(std::max(bc.y, 0.001f)),
                        -std::log(std::max(bc.z, 0.001f))
                    };
                }
                // else: grey/neutral probe → keep Params{} default sigma_a
            }
            // Longitudinal roughness from dominant GGX lobe roughness.
            // beta_m ∈ [0.2, 0.8]; clamp to physically plausible range.
            hp.beta_m = std::clamp(m_roughness, 0.20f, 0.80f);
            // Azimuthal roughness slightly wider than longitudinal (typical hair).
            hp.beta_n = std::clamp(m_roughness * 1.5f, 0.30f, 0.90f);
            m_hairFallback = MarschnerHairMaterial(hp);
            // Diagnostic: print probed base color and resulting sigma_a so we can
            // verify the correct color is flowing through from the OSL shader.
            fprintf(stderr,
                "[OslMaterial hair] shader='%s' baseColor=(%.3f,%.3f,%.3f) "
                "sigma_a=(%.3f,%.3f,%.3f)\n",
                m_shaderName.c_str(),
                m_baseColor.x, m_baseColor.y, m_baseColor.z,
                hp.sigma_a.x, hp.sigma_a.y, hp.sigma_a.z);
        }
    }

    bool     isDelta()  const override { return false; }
    uint32_t flags()    const override { return BSDFFlag_Diffuse | BSDFFlag_Glossy |
                                                BSDFFlag_Reflection | BSDFFlag_Transmission; }

    // Allow shadow rays to pass through transmissive OSL materials.
    // The tint is probed once at construction so this is free at render time.
    Spectrum transmittanceColor(const ShadingContext&) const override {
        return m_transmittanceTint;
    }
    // Cached base color for GPU/Metal preview (probed at construction).
    Spectrum reflectance(const ShadingContext& ctx) const override {
        return m_baseColor;
    }
    float    roughness() const override { return m_roughness; }
    Spectrum probeEmission() const override { return m_probeEmission; }

    // ---------- Le: emitted radiance ----------
    Spectrum Le(const ShadingContext& ctx, Vec3f wo) const override {
        auto lobes = evalClosure(ctx, wo);
        Spectrum Le = {};
        for (auto& l : lobes)
            if (l.kind == OslLobe::Kind::Emission)
                Le = Le + l.emittance;
        return Le;
    }

    // Build a MarschnerHairMaterial from lobe params (used for dispatch below).
    static MarschnerHairMaterial marschnerFromLobe(const OslLobe& lobe) {
        MarschnerHairMaterial::Params p;
        p.sigma_a = lobe.marschner.sigma_a;
        p.beta_m  = lobe.marschner.beta_m;
        p.beta_n  = lobe.marschner.beta_n;
        p.alpha   = lobe.marschner.alpha;
        p.eta     = lobe.marschner.eta;
        return MarschnerHairMaterial(p);
    }

    // ---------- sample ----------
    BSDFSample sample(const ShadingContext& ctx,
                      Vec3f wo, Vec2f u, float uComp) const override {
        Vec3f woLocal = ctx.toLocal(wo);
        auto lobes = evalClosure(ctx, wo);
        mergeGGXPairs(lobes);

        // Select lobe by luminance weight; GGXBoth handles R/T split internally
        float total = 0.f;
        {
            static int s_034count = 0;
            if (m_shaderName.find("Material_034") != std::string::npos && s_034count++ < 8) {
                fprintf(stderr, "[034 DBG] lobes=%zu wo.z=%.3f\n", lobes.size(), woLocal.z);
                for (size_t li = 0; li < lobes.size(); ++li) {
                    const auto& lb = lobes[li];
                    fprintf(stderr, "  [%zu] kind=%d wt=(%.3f,%.3f,%.3f) alb=(%.3f,%.3f,%.3f)\n",
                        li, (int)lb.kind, lb.weight.x, lb.weight.y, lb.weight.z,
                        lb.albedo.x, lb.albedo.y, lb.albedo.z);
                }
            }
        }
        for (auto& l : lobes)
            if (l.kind != OslLobe::Kind::Emission)
                total += luminance(l.weight);

        if (total <= 0.f) return {};

        float accum     = 0.f;
        float accumPrev = 0.f;
        for (size_t i = 0; i < lobes.size(); ++i) {
            const auto& lobe = lobes[i];
            if (lobe.kind == OslLobe::Kind::Emission) continue;
            accumPrev = accum;
            accum += luminance(lobe.weight);
            if (uComp * total <= accum || i == lobes.size() - 1) {
                float selPdf = luminance(lobe.weight) / total;

                // Remap uComp into [0,1] within this lobe's selection interval
                // so GGXBoth can use it as an independent R/T Fresnel decision.
                float uFresnel = (total > 0.f && luminance(lobe.weight) > 0.f)
                    ? std::min(0.9999f, (uComp * total - accumPrev)
                               / luminance(lobe.weight))
                    : 0.5f;

                // Marschner: dispatch to the C++ hair implementation with full ctx
                if (lobe.kind == OslLobe::Kind::MarschnerHair) {
                    auto mat = marschnerFromLobe(lobe);
                    return mat.sample(ctx, wo, u, uComp);
                }

                BSDFSample s = sampleLobe(lobe, woLocal, u, uFresnel);
                if (!s.isValid()) return {};

                // Convert to world space
                s.wi  = ctx.toWorld(s.wi);
                // Scale pdf by lobe selection probability
                s.pdf    *= selPdf;
                s.pdfRev *= selPdf;

                // Add contributions from other lobes (MIS accumulation)
                Vec3f wiLocal = ctx.toLocal(s.wi);
                for (size_t j = 0; j < lobes.size(); ++j) {
                    if (j == i || lobes[j].kind == OslLobe::Kind::Emission)
                        continue;
                    auto ev = evalLobe(lobes[j], woLocal, wiLocal);
                    if (ev.pdf > 0.f) {
                        float wj = luminance(lobes[j].weight) / total;
                        s.f   = s.f + ev.f;
                        s.pdf += ev.pdf * wj;
                    }
                }
                return s;
            }
        }
        return {};
    }

    // ---------- evaluate ----------
    BSDFEval evaluate(const ShadingContext& ctx,
                       Vec3f wo, Vec3f wi) const override {
        Vec3f woLocal = ctx.toLocal(wo);
        Vec3f wiLocal = ctx.toLocal(wi);
        if (woLocal.z <= 0.f && wiLocal.z <= 0.f) return {};

        auto lobes = evalClosure(ctx, wo);
        mergeGGXPairs(lobes);

        float total = 0.f;
        for (auto& l : lobes)
            if (l.kind != OslLobe::Kind::Emission)
                total += luminance(l.weight);

        BSDFEval result = {};
        for (auto& lobe : lobes) {
            if (lobe.kind == OslLobe::Kind::Emission) continue;
            float selPdf = total > 0.f ? luminance(lobe.weight) / total : 0.f;
            if (lobe.kind == OslLobe::Kind::MarschnerHair) {
                auto mat = marschnerFromLobe(lobe);
                auto ev  = mat.evaluate(ctx, wo, wi);
                result.f      = result.f + ev.f;
                result.pdf    += ev.pdf * selPdf;
                result.pdfRev += ev.pdfRev * selPdf;
                continue;
            }
            auto ev = evalLobe(lobe, woLocal, wiLocal);
            result.f      = result.f + ev.f;
            result.pdf    += ev.pdf * selPdf;
            result.pdfRev += ev.pdfRev * selPdf;
        }
        return result;
    }

    float pdf(const ShadingContext& ctx, Vec3f wo, Vec3f wi) const override {
        return evaluate(ctx, wo, wi).pdf;
    }

    // SSS params injected from USD after construction (via setSubsurfaceParams).
    void setSubsurfaceParams(float weight, Spectrum color, float radius, float anisotropy,
                              float strength = 1.f) {
        m_sss.weight     = weight;
        m_sss.color      = color;
        m_sss.radius     = radius;
        m_sss.anisotropy = anisotropy;
        m_sss.strength   = strength;
    }

    SubsurfaceParams subsurfaceParams() const override { return m_sss; }

private:
    std::string                 m_shaderName;
    OSL::ShaderGroupRef         m_group;
    Spectrum                    m_transmittanceTint = {};
    Spectrum                    m_baseColor         = {0.5f, 0.5f, 0.5f};
    float                       m_roughness         = 1.0f;  // from dominant GGX lobe, cached at ctor
    Spectrum                    m_probeEmission     = {};    // emission at uv=(0.5,0.5), cached at ctor
    SubsurfaceParams            m_sss               = {};

    // Fallback Marschner material used when ctx.isCurve == true.
    // Constructed at OslMaterial init from the probed OSL closure parameters.
    // Allows any USD/OSL material assigned to hair strands (via materialPath
    // sidecar) to use physically correct Marschner hair shading rather than
    // the normal-based GGX path (which is meaningless on ribbon/curve geometry).
    MarschnerHairMaterial       m_hairFallback{MarschnerHairMaterial::Params{}};

    // Execute the shader and collect lobes.  Each call obtains its own
    // ShadingContext from the ShadingSystem (thread-safe).
    std::vector<OslLobe> evalClosure(const ShadingContext& ctx,
                                     Vec3f wo) const {
        OSL::ShadingSystem* s = OslShadingSystem::instance().sys();

        OSL::ShaderGlobals sg;
        memset(&sg, 0, sizeof(sg));
        sg.P  = OSL::Vec3(ctx.p.x,  ctx.p.y,  ctx.p.z);
        sg.N  = OSL::Vec3(ctx.n.x,  ctx.n.y,  ctx.n.z);
        sg.Ng = OSL::Vec3(ctx.ng.x, ctx.ng.y, ctx.ng.z);
        sg.u  = ctx.uv.x;
        sg.v  = ctx.uv.y;
        // sg.I is the incoming ray direction (toward surface, not negated)
        sg.I  = OSL::Vec3(-wo.x, -wo.y, -wo.z);
        // Always evaluate as front-face: MaterialX's backsurface defaults to
        // null_closure(), so setting sg.backfacing=true yields a black result
        // for back-face glass hits.  We handle the eta flip in collectLobes
        // via frontFace, so OSL doesn't need to know about backfacing.
        sg.backfacing = false;
        // Pass the ShadingContext so get_attribute can serve hair globals.
        sg.renderstate = const_cast<ShadingContext*>(&ctx);
        // Non-null sentinel; get_matrix/get_inverse_matrix read actual O2W
        // from renderstate, so any non-null pointer works as the "object" key.
        sg.object2common = static_cast<OSL::TransformationPtr>(&ctx);
        // dPdv carries the fiber tangent for curve hits; OSL hair shaders can
        // read it directly or via getattribute("hair_tangent", ...).
        if (ctx.isCurve)
            sg.dPdv = OSL::Vec3(ctx.t.x, ctx.t.y, ctx.t.z);

        OSL::PerThreadInfo* ti = OslShadingSystem::instance().threadInfo();
        OSL::ShadingContext* oslCtx = s->get_context(ti);
        s->execute(*oslCtx, *m_group, sg);

        // The generated shader writes "Ci = out;" at the end of its body
        // (injected by the Python exporter).  sg.Ci is an OSL built-in global
        // that is always preserved — no find_symbol / renderer_outputs needed.
        const OSL::ClosureColor* ci = sg.Ci;

        std::vector<OslLobe> lobes;
        collectLobes(ci, {1.f,1.f,1.f}, lobes, ctx.frontFace);

        s->release_context(oslCtx);
        return lobes;
    }

    // uFresnel: independent uniform [0,1] for GGXBoth R/T decision.
    // For all other lobe types it is unused.
    static BSDFSample sampleLobe(const OslLobe& lobe, Vec3f wo, Vec2f u,
                                  float uFresnel = 0.5f) {
        switch (lobe.kind) {
        case OslLobe::Kind::Diffuse:      return sampleDiffuseLobe(lobe, wo, u);
        case OslLobe::Kind::DiffuseTrans: return sampleDiffuseTransLobe(lobe, wo, u);
        case OslLobe::Kind::GGXRefl:      return sampleGGXReflLobe(lobe, wo, u);
        case OslLobe::Kind::GGXTrans:     return sampleGGXTransLobe(lobe, wo, u);
        case OslLobe::Kind::GGXBoth: {
            // Matches StandardSurface rough-glass path exactly:
            // ONE half-vector sampled from u (2D), R/T decided by uFresnel (1D).
            // This avoids the bug of using u.x for both wh-sampling and Fresnel
            // selection, and ensures the TIR check uses the same wh as the
            // Fresnel decision.
            if (wo.z <= 0.f) return {};

            // Smooth glass (alpha2≈0): D_GGX→NaN, so use perfect specular path.
            // This matches StandardSurface's roughness<0.001 specular branch.
            if (lobe.alpha2 < 1e-6f) {
                float Fr  = fresnelDielectric(wo.z, lobe.ior);
                if (uFresnel < Fr) {
                    // Perfect mirror reflection
                    Vec3f wi = {-wo.x, -wo.y, wo.z};
                    BSDFSample s;
                    s.wi = wi; s.f = lobe.refl_tint * lobe.weight * Fr;
                    s.pdf = Fr; s.pdfRev = Fr;
                    s.eta = 1.f; s.flags = BSDFFlag_Specular | BSDFFlag_Reflection;
                    return s;
                } else {
                    float eta   = lobe.ior;
                    float sin2T = std::max(0.f, 1.f - wo.z * wo.z) / (eta * eta);
                    if (sin2T >= 1.f) {
                        // TIR — reflect
                        Vec3f wi = {-wo.x, -wo.y, wo.z};
                        BSDFSample s;
                        s.wi = wi; s.f = lobe.refl_tint * lobe.weight;
                        s.pdf = 1.f; s.pdfRev = 1.f;
                        s.eta = 1.f; s.flags = BSDFFlag_Specular | BSDFFlag_Reflection;
                        return s;
                    }
                    float cosT  = std::sqrt(1.f - sin2T);
                    Vec3f wi    = {-wo.x / eta, -wo.y / eta, -cosT};
                    if (wi.z >= 0.f) return {};
                    float Ft    = 1.f - Fr;
                    BSDFSample s;
                    s.wi = wi; s.f = lobe.refr_tint * lobe.weight * Ft;
                    s.pdf = Ft; s.pdfRev = Ft;
                    s.eta = eta; s.flags = BSDFFlag_Specular | BSDFFlag_Transmission;
                    return s;
                }
            }

            Vec3f wh    = sampleGGX_halfvector(u, lobe.alpha2);
            float cosIH = std::max(0.f, dot(wo, wh));
            if (cosIH <= 0.f) return {};
            float Fr    = fresnelDielectric(cosIH, lobe.ior);

            if (uFresnel < Fr) {
                // ----- GGX reflection -----
                Vec3f wi = wh * (2.f * cosIH) - wo;
                if (wi.z <= 0.f) return {};
                float D   = D_GGX(wh.z, lobe.alpha2);
                float G2  = G2_Smith_Separable(wo.z, wi.z, lobe.alpha2);
                float pdf = Fr * pdfGGX_reflection(wh.z, lobe.alpha2, cosIH);
                if (pdf <= 0.f) return {};
                Spectrum f = lobe.refl_tint * lobe.weight
                             * (Fr * D * G2 / (4.f * wo.z));
                BSDFSample s;
                s.wi     = wi; s.f = f; s.pdf = pdf; s.pdfRev = pdf;
                s.eta    = 1.f;
                s.flags  = BSDFFlag_Glossy | BSDFFlag_Reflection;
                return s;
            } else {
                // ----- GGX refraction — same wh, TIR → reflect -----
                float eta   = lobe.ior;
                float sin2T = std::max(0.f, 1.f - cosIH * cosIH) / (eta * eta);
                if (sin2T >= 1.f) {
                    // TIR: reflect instead of killing the path
                    Vec3f wi = wh * (2.f * cosIH) - wo;
                    if (wi.z <= 0.f) return {};
                    float D   = D_GGX(wh.z, lobe.alpha2);
                    float G2  = G2_Smith_Separable(wo.z, wi.z, lobe.alpha2);
                    float pdf = pdfGGX_reflection(wh.z, lobe.alpha2, cosIH);
                    if (pdf <= 0.f) return {};
                    Spectrum f = lobe.refl_tint * lobe.weight
                                 * (D * G2 / (4.f * wo.z));
                    BSDFSample s;
                    s.wi = wi; s.f = f; s.pdf = pdf; s.pdfRev = pdf;
                    s.eta = 1.f;
                    s.flags = BSDFFlag_Glossy | BSDFFlag_Reflection;
                    return s;
                }
                float cosT_H     = std::sqrt(1.f - sin2T);
                Vec3f wi         = wo * (-1.f / eta) + wh * (cosIH / eta - cosT_H);
                if (wi.z >= 0.f) return {};
                float cosI_t     = std::abs(wi.z);
                float D          = D_GGX(wh.z, lobe.alpha2);
                float G2         = G2_Smith_Separable(wo.z, cosI_t, lobe.alpha2);
                float absCosT_wh = std::abs(dot(wi, wh));
                float denom      = cosIH + eta * absCosT_wh;
                if (denom < 1e-6f) return {};
                float jacobian   = eta * eta * absCosT_wh / (denom * denom);
                float pdf        = (1.f - Fr) * D * wh.z * jacobian;
                if (pdf <= 0.f) return {};
                Spectrum f = lobe.refr_tint * lobe.weight
                             * ((1.f - Fr) * D * G2 * cosIH * absCosT_wh
                                / (wo.z * denom * denom));
                BSDFSample s;
                s.wi    = wi; s.f = f; s.pdf = pdf; s.pdfRev = pdf;
                s.eta   = eta;
                s.flags = BSDFFlag_Glossy | BSDFFlag_Transmission;
                return s;
            }
        }
        case OslLobe::Kind::Transparent: {
            BSDFSample s;
            s.wi    = -wo;
            s.f     = lobe.weight;
            s.pdf   = 1e32f;
            s.pdfRev = 1e32f;
            s.eta   = 1.f;
            s.flags = BSDFFlag_Specular | BSDFFlag_Transmission;
            return s;
        }
        default: return {};
        }
    }

    static BSDFEval evalLobe(const OslLobe& lobe, Vec3f wo, Vec3f wi) {
        switch (lobe.kind) {
        case OslLobe::Kind::Diffuse:      return evalDiffuseLobe(lobe, wo, wi);
        case OslLobe::Kind::DiffuseTrans: return evalDiffuseTransLobe(lobe, wo, wi);
        case OslLobe::Kind::GGXRefl:  return evalGGXReflLobe(lobe, wo, wi);
        case OslLobe::Kind::GGXBoth:
        case OslLobe::Kind::GGXTrans:
            if (lobe.alpha2 < 1e-6f) return {};  // delta BSDF — eval = 0 a.e.
            if (wi.z < 0.f)  // transmission direction
                return {};  // GGX transmission eval is complex; skip for pdf
            return evalGGXReflLobe(lobe, wo, wi);
        case OslLobe::Kind::MarschnerHair:
            return {};  // handled in evaluate() via marschnerFromLobe — never reaches here
        default: return {};
        }
    }
};

// ===========================================================================
// Free function implementations (declared in OslMaterial.h)
// ===========================================================================

bool oslCompileShader(const std::string& oslPath,
                      const std::string& osoPath,
                      const std::string& matDir) {
    OSL::OSLCompiler compiler;
    // stdosl.h location — full path to the file; OSLCompiler prepends it as
    // #include "<stdoslpath>". Searches: $ANACAPA_OSL_STDOSL, then known
    // Blender install layouts on macOS / Linux, then nothing (compiler default).
    auto firstReadable = [](std::initializer_list<std::string> paths) -> std::string {
        for (const auto& p : paths)
            if (!p.empty() && std::ifstream(p).good()) return p;
        return {};
    };
    const char* envStdosl = std::getenv("ANACAPA_OSL_STDOSL");
    std::string stdosl = firstReadable({
        envStdosl ? envStdosl : "",
        // macOS Steam Blender
        "/Users/douglascreel/Library/Application Support/Steam/"
        "steamapps/common/Blender/Blender.app/Contents/Resources/"
        "5.1/scripts/addons_core/cycles/shader/stdosl.h",
        // macOS .app Blender
        "/Applications/Blender.app/Contents/Resources/"
        "5.1/scripts/addons_core/cycles/shader/stdosl.h",
        // Linux Blender (tarball install — common: ~/software, /opt, /usr/local)
        std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
            "/software/blender-5.1.1-linux-x64/5.1/scripts/addons_core/cycles/shader/stdosl.h",
        "/opt/blender/5.1/scripts/addons_core/cycles/shader/stdosl.h",
    });

    // mx_funcs.h is included by _mx_stdlib.h with a quoted include, which only
    // searches the including file's directory. Copy it to matDir if absent.
    const std::string mxFuncsDst = matDir + "/mx_funcs.h";
    if (!std::ifstream(mxFuncsDst).good()) {
        const char* envMxFuncs = std::getenv("ANACAPA_OSL_MX_FUNCS");
        std::string mxFuncsSrc = firstReadable({
            envMxFuncs ? envMxFuncs : "",
            // macOS Steam Blender
            "/Users/douglascreel/Library/Application Support/Steam/"
            "steamapps/common/Blender/Blender.app/Contents/Resources/"
            "lib/materialx/libraries/stdlib/genosl/include/mx_funcs.h",
            // macOS .app Blender
            "/Applications/Blender.app/Contents/Resources/"
            "lib/materialx/libraries/stdlib/genosl/include/mx_funcs.h",
            // Linux Blender
            std::string(std::getenv("HOME") ? std::getenv("HOME") : "") +
                "/software/blender-5.1.1-linux-x64/lib/materialx/libraries/stdlib/genosl/include/mx_funcs.h",
            "/opt/blender/lib/materialx/libraries/stdlib/genosl/include/mx_funcs.h",
        });
        if (!mxFuncsSrc.empty()) {
            std::ifstream  src(mxFuncsSrc, std::ios::binary);
            std::ofstream  dst(mxFuncsDst, std::ios::binary);
            dst << src.rdbuf();
        }
    }

    std::vector<std::string> opts;
    opts.push_back("-I"); opts.push_back(matDir);
    // Add the build-time _mx_stdlib.h directory so shaders can #include it
    // without it being present in the scene's materials directory.
#ifdef ANACAPA_MX_STDLIB_DIR
    opts.push_back("-I"); opts.push_back(ANACAPA_MX_STDLIB_DIR);
#endif
    opts.push_back("-o"); opts.push_back(osoPath);
    opts.push_back("-O2");
    bool ok = compiler.compile(oslPath, opts, stdosl);
    if (!ok)
        fprintf(stderr, "[OslMaterial] oslc compilation failed for '%s' (check stderr for details)\n",
                oslPath.c_str());
    return ok;
}

void oslAddSearchPath(const std::string& dir) {
    OslShadingSystem::instance().addSearchPath(dir);
}

std::unique_ptr<IMaterial> makeOslMaterial(const std::string& shaderName) {
    try {
        return std::make_unique<OslMaterial>(shaderName);
    } catch (const std::exception& e) {
        fprintf(stderr, "[OslMaterial] failed to load '%s': %s\n",
                shaderName.c_str(), e.what());
        return nullptr;
    }
}

void oslSetSubsurfaceParams(IMaterial* mat,
                             float weight, Spectrum color,
                             float radius, float anisotropy,
                             float strength) {
    if (auto* osl = dynamic_cast<OslMaterial*>(mat))
        osl->setSubsurfaceParams(weight, color, radius, anisotropy, strength);
}

} // namespace anacapa

#endif  // ANACAPA_ENABLE_OSL
