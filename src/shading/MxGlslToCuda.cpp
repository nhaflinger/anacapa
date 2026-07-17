#include "MxGlslToCuda.h"

#include <spdlog/spdlog.h>

#include <sstream>
#include <cstdlib>

namespace anacapa {

namespace {

// ---------------------------------------------------------------------------
// GLSL-compatible vector types + builtins for CUDA. MaterialX's raw GLSL
// output uses GLSL's native vec2/vec3/vec4 constructor-call syntax and a
// small set of builtin functions (mix/clamp/dot/cross/...) that CUDA's own
// float2/float3/float4 (plain C structs) don't support directly — this
// prelude gives CUDA C++ types with the SAME NAMES as GLSL's, so the raw
// MaterialX-generated function body compiles with no textual type rewriting.
//
// Scope: covers what the surface-parameter node graphs actually exercise
// (mix/clamp/arithmetic/basic math). No matrix types — a node graph that
// needs mat3/mat4 (e.g. a "place2d" UV transform) will fail to compile here;
// expand incrementally as real materials need it (see project memory
// project_materialx_codegen, Phase 3 risks).
// ---------------------------------------------------------------------------
constexpr const char* kCudaGlslPrelude = R"CUDA(
struct vec2 {
    float x, y;
    __device__ vec2() : x(0), y(0) {}
    __device__ vec2(float v) : x(v), y(v) {}
    __device__ vec2(float x_, float y_) : x(x_), y(y_) {}
};
struct vec3 {
    float x, y, z;
    __device__ vec3() : x(0), y(0), z(0) {}
    __device__ vec3(float v) : x(v), y(v), z(v) {}
    __device__ vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    __device__ vec3(vec2 xy, float z_) : x(xy.x), y(xy.y), z(z_) {}
};
struct vec4 {
    float x, y, z, w;
    __device__ vec4() : x(0), y(0), z(0), w(0) {}
    __device__ vec4(float v) : x(v), y(v), z(v), w(v) {}
    __device__ vec4(float x_, float y_, float z_, float w_) : x(x_), y(y_), z(z_), w(w_) {}
    __device__ vec4(vec3 xyz, float w_) : x(xyz.x), y(xyz.y), z(xyz.z), w(w_) {}
};

__device__ inline vec2 operator+(vec2 a, vec2 b) { return vec2(a.x+b.x, a.y+b.y); }
__device__ inline vec2 operator-(vec2 a, vec2 b) { return vec2(a.x-b.x, a.y-b.y); }
__device__ inline vec2 operator*(vec2 a, vec2 b) { return vec2(a.x*b.x, a.y*b.y); }
__device__ inline vec2 operator/(vec2 a, vec2 b) { return vec2(a.x/b.x, a.y/b.y); }
__device__ inline vec2 operator*(vec2 a, float s) { return vec2(a.x*s, a.y*s); }
__device__ inline vec2 operator*(float s, vec2 a) { return vec2(a.x*s, a.y*s); }
__device__ inline vec2 operator/(vec2 a, float s) { return vec2(a.x/s, a.y/s); }
__device__ inline vec2 operator-(vec2 a) { return vec2(-a.x, -a.y); }

__device__ inline vec3 operator+(vec3 a, vec3 b) { return vec3(a.x+b.x, a.y+b.y, a.z+b.z); }
__device__ inline vec3 operator-(vec3 a, vec3 b) { return vec3(a.x-b.x, a.y-b.y, a.z-b.z); }
__device__ inline vec3 operator*(vec3 a, vec3 b) { return vec3(a.x*b.x, a.y*b.y, a.z*b.z); }
__device__ inline vec3 operator/(vec3 a, vec3 b) { return vec3(a.x/b.x, a.y/b.y, a.z/b.z); }
__device__ inline vec3 operator*(vec3 a, float s) { return vec3(a.x*s, a.y*s, a.z*s); }
__device__ inline vec3 operator*(float s, vec3 a) { return vec3(a.x*s, a.y*s, a.z*s); }
__device__ inline vec3 operator/(vec3 a, float s) { return vec3(a.x/s, a.y/s, a.z/s); }
__device__ inline vec3 operator-(vec3 a) { return vec3(-a.x, -a.y, -a.z); }

__device__ inline vec4 operator+(vec4 a, vec4 b) { return vec4(a.x+b.x, a.y+b.y, a.z+b.z, a.w+b.w); }
__device__ inline vec4 operator-(vec4 a, vec4 b) { return vec4(a.x-b.x, a.y-b.y, a.z-b.z, a.w-b.w); }
__device__ inline vec4 operator*(vec4 a, vec4 b) { return vec4(a.x*b.x, a.y*b.y, a.z*b.z, a.w*b.w); }
__device__ inline vec4 operator/(vec4 a, vec4 b) { return vec4(a.x/b.x, a.y/b.y, a.z/b.z, a.w/b.w); }
__device__ inline vec4 operator*(vec4 a, float s) { return vec4(a.x*s, a.y*s, a.z*s, a.w*s); }
__device__ inline vec4 operator*(float s, vec4 a) { return vec4(a.x*s, a.y*s, a.z*s, a.w*s); }
__device__ inline vec4 operator/(vec4 a, float s) { return vec4(a.x/s, a.y/s, a.z/s, a.w/s); }
__device__ inline vec4 operator-(vec4 a) { return vec4(-a.x, -a.y, -a.z, -a.w); }

__device__ inline float mx_clampf(float x, float lo, float hi) { return x < lo ? lo : (x > hi ? hi : x); }
__device__ inline float mix(float a, float b, float t) { return a + (b - a) * t; }
__device__ inline vec2   mix(vec2 a, vec2 b, float t)   { return a + (b - a) * t; }
__device__ inline vec3   mix(vec3 a, vec3 b, float t)   { return a + (b - a) * t; }
__device__ inline vec4   mix(vec4 a, vec4 b, float t)   { return a + (b - a) * t; }
__device__ inline vec2   mix(vec2 a, vec2 b, vec2 t)    { return vec2(mix(a.x,b.x,t.x), mix(a.y,b.y,t.y)); }
__device__ inline vec3   mix(vec3 a, vec3 b, vec3 t)    { return vec3(mix(a.x,b.x,t.x), mix(a.y,b.y,t.y), mix(a.z,b.z,t.z)); }

__device__ inline float clamp(float x, float lo, float hi) { return mx_clampf(x, lo, hi); }
__device__ inline vec2  clamp(vec2 x, float lo, float hi)  { return vec2(mx_clampf(x.x,lo,hi), mx_clampf(x.y,lo,hi)); }
__device__ inline vec3  clamp(vec3 x, float lo, float hi)  { return vec3(mx_clampf(x.x,lo,hi), mx_clampf(x.y,lo,hi), mx_clampf(x.z,lo,hi)); }
__device__ inline vec3  clamp(vec3 x, vec3 lo, vec3 hi)    { return vec3(mx_clampf(x.x,lo.x,hi.x), mx_clampf(x.y,lo.y,hi.y), mx_clampf(x.z,lo.z,hi.z)); }

__device__ inline float dot(vec2 a, vec2 b) { return a.x*b.x + a.y*b.y; }
__device__ inline float dot(vec3 a, vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
__device__ inline vec3  cross(vec3 a, vec3 b) {
    return vec3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}
__device__ inline float length(vec2 a) { return sqrtf(dot(a,a)); }
__device__ inline float length(vec3 a) { return sqrtf(dot(a,a)); }
__device__ inline vec2  normalize(vec2 a) { float l = length(a); return l > 0.f ? a * (1.f/l) : a; }
__device__ inline vec3  normalize(vec3 a) { float l = length(a); return l > 0.f ? a * (1.f/l) : a; }

__device__ inline float mx_mod(float x, float y) { return x - y * floorf(x/y); }
__device__ inline vec2  mx_mod(vec2 x, float y)  { return vec2(mx_mod(x.x,y), mx_mod(x.y,y)); }
__device__ inline vec3  mx_mod(vec3 x, float y)  { return vec3(mx_mod(x.x,y), mx_mod(x.y,y), mx_mod(x.z,y)); }
#define mod mx_mod
#define mx_inversesqrt rsqrtf
#define mx_sin sinf
#define mx_cos cosf
#define mx_tan tanf
#define mx_asin asinf
#define mx_acos acosf
#define mx_atan atan2f
#define mx_radians(deg) ((deg) * 0.017453292519943295f)

__device__ inline vec3 pow(vec3 x, vec3 y) { return vec3(powf(x.x,y.x), powf(x.y,y.y), powf(x.z,y.z)); }
__device__ inline vec3 pow(vec3 x, float y) { return vec3(powf(x.x,y), powf(x.y,y), powf(x.z,y)); }
__device__ inline vec3 abs(vec3 x) { return vec3(fabsf(x.x), fabsf(x.y), fabsf(x.z)); }
__device__ inline vec3 floor(vec3 x) { return vec3(floorf(x.x), floorf(x.y), floorf(x.z)); }
__device__ inline vec3 max(vec3 a, vec3 b) { return vec3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z)); }
__device__ inline vec3 min(vec3 a, vec3 b) { return vec3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z)); }
__device__ inline vec3 max(vec3 a, float b) { return vec3(fmaxf(a.x,b), fmaxf(a.y,b), fmaxf(a.z,b)); }
__device__ inline vec3 min(vec3 a, float b) { return vec3(fminf(a.x,b), fminf(a.y,b), fminf(a.z,b)); }

typedef vec3 color3;
typedef vec4 color4;
#define M_PI_F 3.14159265358979323846f

// MaterialX's generator always emits a fixed boilerplate block (mx_square,
// mx_srgb_encode, ...) regardless of whether the target material actually
// calls it — it must still compile even when dead code. bvec3/greaterThan
// back mx_srgb_encode's branchless sRGB curve selection.
struct bvec2 { bool x, y; __device__ bvec2() : x(false), y(false) {} __device__ bvec2(bool x_, bool y_) : x(x_), y(y_) {} };
struct bvec3 { bool x, y, z; __device__ bvec3() : x(false), y(false), z(false) {} __device__ bvec3(bool x_, bool y_, bool z_) : x(x_), y(y_), z(z_) {} };
struct bvec4 { bool x, y, z, w; __device__ bvec4() : x(false), y(false), z(false), w(false) {} __device__ bvec4(bool x_, bool y_, bool z_, bool w_) : x(x_), y(y_), z(z_), w(w_) {} };

__device__ inline bvec2 greaterThan(vec2 a, vec2 b) { return bvec2(a.x > b.x, a.y > b.y); }
__device__ inline bvec3 greaterThan(vec3 a, vec3 b) { return bvec3(a.x > b.x, a.y > b.y, a.z > b.z); }
__device__ inline bvec2 lessThan(vec2 a, vec2 b) { return bvec2(a.x < b.x, a.y < b.y); }
__device__ inline bvec3 lessThan(vec3 a, vec3 b) { return bvec3(a.x < b.x, a.y < b.y, a.z < b.z); }

__device__ inline vec2 mix(vec2 a, vec2 b, bvec2 t) { return vec2(t.x ? b.x : a.x, t.y ? b.y : a.y); }
__device__ inline vec3 mix(vec3 a, vec3 b, bvec3 t) { return vec3(t.x ? b.x : a.x, t.y ? b.y : a.y, t.z ? b.z : a.z); }
)CUDA";

int componentCount(const std::string& type) {
    if (type == "float" || type == "integer" || type == "boolean") return 1;
    if (type == "vector2") return 2;
    if (type == "vector3" || type == "color3") return 3;
    if (type == "vector4" || type == "color4") return 4;
    return -1;
}

std::string glslType(const std::string& type) {
    if (type == "float")   return "float";
    if (type == "integer") return "int";
    if (type == "boolean") return "bool";
    if (type == "vector2") return "vec2";
    if (type == "vector3" || type == "color3") return "vec3";
    if (type == "vector4" || type == "color4") return "vec4";
    return "";
}

std::string unpackExpr(const std::string& type, int floatOffset) {
    auto elem = [&](int k) {
        return "uniformData[offset+" + std::to_string(floatOffset + k) + "]";
    };
    if (type == "float")   return elem(0);
    if (type == "integer") return "int(" + elem(0) + ")";
    if (type == "boolean") return "(" + elem(0) + " != 0.0f)";
    if (type == "vector2") return "vec2(" + elem(0) + ", " + elem(1) + ")";
    if (type == "vector3" || type == "color3")
        return "vec3(" + elem(0) + ", " + elem(1) + ", " + elem(2) + ")";
    if (type == "vector4" || type == "color4")
        return "vec4(" + elem(0) + ", " + elem(1) + ", " + elem(2) + ", " + elem(3) + ")";
    return "";
}

std::vector<float> parseValue(const std::string& value, int count) {
    std::vector<float> out(count, 0.f);
    if (value.empty()) return out;
    std::stringstream ss(value);
    std::string tok;
    int i = 0;
    while (i < count && std::getline(ss, tok, ',')) {
        out[i++] = static_cast<float>(std::atof(tok.c_str()));
    }
    return out;
}

// Removes every line whose trimmed content starts with `prefix`.
std::string stripLinesStartingWith(const std::string& src, const std::string& prefix) {
    std::istringstream in(src);
    std::ostringstream out;
    std::string line;
    while (std::getline(in, line)) {
        size_t start = line.find_first_not_of(" \t");
        if (start != std::string::npos && line.compare(start, prefix.size(), prefix) == 0) continue;
        out << line << "\n";
    }
    return out.str();
}

// MaterialX's GLSL output declares per-fragment varyings (UV, etc.) as a GLSL
// interface block: `in VertexData { vec2 texcoord_0; } vd;` — not valid CUDA
// C++ at all (interface blocks are GLSL-only syntax), so any material using
// <texcoord> would fail to compile as-is. Finds the block, returns its field
// names (same idea as MaterialXMslAdapter's findTexcoordFields()), and the
// [start,end) byte range of the whole `in VertexData { ... } vd;` statement
// so the caller can excise it and replace it with a real local declaration.
struct GlslVertexDataBlock {
    std::vector<std::string> texcoordFields;
    // True if the block declares `vec3 positionObject` — MaterialX emits this
    // for <position space="object"> (object-space noise, e.g. wood grain).
    // Only "object" space is handled, matching the CPU/OSL path's coverage.
    bool hasPositionObject = false;
    size_t start = std::string::npos;
    size_t end   = std::string::npos;
};

GlslVertexDataBlock findGlslVertexDataBlock(const std::string& source) {
    GlslVertexDataBlock result;
    size_t declPos = source.find("in VertexData");
    if (declPos == std::string::npos) return result;
    size_t braceOpen = source.find('{', declPos);
    size_t braceClose = source.find('}', braceOpen);
    if (braceOpen == std::string::npos || braceClose == std::string::npos) return result;
    // Statement ends at the `;` after `} vd` (or whatever instance name).
    size_t stmtEnd = source.find(';', braceClose);
    if (stmtEnd == std::string::npos) return result;
    result.start = declPos;
    result.end   = stmtEnd + 1;

    std::string body = source.substr(braceOpen, braceClose - braceOpen);
    const std::string needle = "vec2 ";
    size_t pos = 0;
    while ((pos = body.find(needle, pos)) != std::string::npos) {
        size_t nameStart = pos + needle.size();
        size_t nameEnd = body.find_first_of(" ;", nameStart);
        if (nameEnd == std::string::npos) break;
        result.texcoordFields.push_back(body.substr(nameStart, nameEnd - nameStart));
        pos = nameEnd;
    }
    result.hasPositionObject = body.find("vec3 positionObject") != std::string::npos;
    return result;
}

} // namespace

MxCudaAdapterResult buildCudaAdapter(const MxGeneratedShader& gen, const std::string& wrapperFnName) {
    MxCudaAdapterResult result;
    if (!gen.valid) return result;

    std::vector<std::string> unpackDecls;
    int floatOffset = 0;
    for (const auto& u : gen.uniforms) {
        int n = componentCount(u.type);
        std::string t = glslType(u.type);
        if (n < 0 || t.empty()) {
            spdlog::warn("MxGlslToCuda: unsupported uniform type '{}' for '{}' — "
                         "falling back to the literal GPU material path for this input",
                         u.type, u.name);
            return result;
        }
        unpackDecls.push_back("    " + t + " " + u.name + " = " + unpackExpr(u.type, floatOffset) + ";");
        std::vector<float> vals = parseValue(u.value, n);
        result.defaultUniformData.insert(result.defaultUniformData.end(), vals.begin(), vals.end());
        floatOffset += n;
    }

    // Strip the GLSL version pragma (meaningless in CUDA), every
    // `uniform ...;` global declaration (replaced by the unpacked locals
    // built above), and MaterialX's own `#define mx_X Y` aliases — these
    // collide with (and contradict) the CUDA-correct aliases the prelude
    // above already defines for the same names (e.g. the prelude's
    // `#define mx_inversesqrt rsqrtf` vs. MaterialX's own emitted
    // `#define mx_inversesqrt inversesqrt`, which isn't a real CUDA symbol).
    std::string body = stripLinesStartingWith(gen.source, "#version");
    body = stripLinesStartingWith(body, "uniform ");
    body = stripLinesStartingWith(body, "#define mx_");

    // MaterialX declares per-fragment varyings (UV, etc.) as a GLSL interface
    // block (`in VertexData { vec2 texcoord_0; } vd;`) — not valid CUDA C++
    // at all. Excise it; a local `vd` with the same field(s), populated from
    // the real per-fragment UV passed in below, takes its place (see
    // MaterialXMslAdapter.cpp's findTexcoordFields() comment for the full
    // story — the Metal backend had the identical gap: <texcoord>-driven
    // patterns silently got garbage/uninitialized UV).
    GlslVertexDataBlock vdBlock = findGlslVertexDataBlock(body);
    if (vdBlock.start != std::string::npos) {
        body.erase(vdBlock.start, vdBlock.end - vdBlock.start);
    }

    // The single `out vec4 <name>;` global becomes a plain local — turn it
    // into a declaration-only statement dropped in with the unpacked locals
    // below, and strip the original line so it isn't declared twice.
    std::string outDecl = "out vec4 " + gen.outputVarName + ";";
    size_t outPos = body.find(outDecl);
    if (outPos == std::string::npos) {
        spdlog::warn("MxGlslToCuda: expected 'out vec4 {}' declaration not found in generated GLSL", gen.outputVarName);
        return result;
    }
    body.erase(outPos, outDecl.size());

    size_t mainPos = body.find("void main()");
    if (mainPos == std::string::npos) {
        spdlog::warn("MxGlslToCuda: 'void main()' not found in generated GLSL for '{}'", wrapperFnName);
        return result;
    }
    size_t bracePos = body.find('{', mainPos);
    if (bracePos == std::string::npos) return result;

    size_t lastBrace = body.rfind('}');
    if (lastBrace == std::string::npos || lastBrace < bracePos) return result;

    // OptiX requires direct-callable entry points to be declared `extern "C"`
    // and named with the `__direct_callable__` prefix — this is how
    // optixProgramGroupCreate's entryFunctionNameDC finds them (mirrors the
    // `__raygen__`/`__closesthit__` convention already used elsewhere in
    // Shade.cu). wrapperFnName itself stays the plain base name — callers
    // (CudaPathIntegrator.cu) must prefix it with "__direct_callable__" when
    // building the OptixProgramGroupDesc. Every generated wrapper shares one
    // optixDirectCall<> signature, so `uv` and `objPos` are always parameters
    // — even for materials whose graph has no <texcoord>/<position> node,
    // where they're simply unused. The parameters are CUDA's *native*
    // float2/float3 (matching exactly what Shade.cu's
    // optixDirectCall<...>(..., texUV, objPos) passes — direct callables
    // are compiled in fully separate translation units with no C++ type
    // identity/ODR checking across the boundary, so the parameter type here
    // must be the same real type as the caller's argument, not just
    // layout-compatible; the prelude's own `vec2`/`vec3` are distinct C++
    // types by that measure, deliberately named to avoid colliding with
    // CUDA's built-in float2/float3/float4). Converted to the prelude's own
    // vec2/vec3 only for internal use inside the wrapper body.
    std::ostringstream sig;
    sig << "extern \"C\" __device__ float4 __direct_callable__" << wrapperFnName
        << "(const float* uniformData, unsigned int offset, float2 cudaUv, float3 cudaObjPos)";

    std::ostringstream prologue;
    prologue << "\n";
    for (const auto& decl : unpackDecls) prologue << decl << "\n";
    if (!vdBlock.texcoordFields.empty() || vdBlock.hasPositionObject) {
        if (!vdBlock.texcoordFields.empty()) prologue << "    vec2 uv = vec2(cudaUv.x, cudaUv.y);\n";
        if (vdBlock.hasPositionObject) prologue << "    vec3 objPos = vec3(cudaObjPos.x, cudaObjPos.y, cudaObjPos.z);\n";
        prologue << "    struct { ";
        for (const auto& f : vdBlock.texcoordFields) prologue << "vec2 " << f << "; ";
        if (vdBlock.hasPositionObject) prologue << "vec3 positionObject; ";
        prologue << "} vd;\n";
        for (const auto& f : vdBlock.texcoordFields) prologue << "    vd." << f << " = uv;\n";
        if (vdBlock.hasPositionObject) prologue << "    vd.positionObject = objPos;\n";
    }
    prologue << "    vec4 " << gen.outputVarName << ";\n";

    std::ostringstream epilogue;
    epilogue << "    return make_float4(" << gen.outputVarName << ".x, " << gen.outputVarName << ".y, "
              << gen.outputVarName << ".z, " << gen.outputVarName << ".w);\n";

    std::string adaptedBody =
        body.substr(0, mainPos) + sig.str() + "\n" +
        body.substr(bracePos, 1) + prologue.str() +
        body.substr(bracePos + 1, lastBrace - bracePos - 1) +
        epilogue.str() +
        body.substr(lastBrace);

    result.valid         = true;
    result.source         = std::string(kCudaGlslPrelude) + "\n" + adaptedBody;
    result.wrapperFnName  = wrapperFnName;
    return result;
}

} // namespace anacapa
