#include "MaterialXMslAdapter.h"

#include <spdlog/spdlog.h>

#include <sstream>
#include <cstdlib>

namespace anacapa {

namespace {

int componentCount(const std::string& type) {
    if (type == "float" || type == "integer" || type == "boolean") return 1;
    if (type == "vector2") return 2;
    if (type == "vector3" || type == "color3") return 3;
    if (type == "vector4" || type == "color4") return 4;
    return -1;  // unsupported (e.g. "filename"/string, "matrix33", ...)
}

// "float2(uniformData[offset+2], uniformData[offset+3])" style unpack
// expression, matching GlobalContext's constructor parameter type exactly.
std::string unpackExpr(const std::string& type, int floatOffset) {
    auto elem = [&](int k) {
        return "uniformData[offset+" + std::to_string(floatOffset + k) + "]";
    };
    if (type == "float")   return elem(0);
    if (type == "integer") return "int(" + elem(0) + ")";
    if (type == "boolean") return "(" + elem(0) + " != 0.0)";
    if (type == "vector2") return "float2(" + elem(0) + ", " + elem(1) + ")";
    if (type == "vector3" || type == "color3")
        return "float3(" + elem(0) + ", " + elem(1) + ", " + elem(2) + ")";
    if (type == "vector4" || type == "color4")
        return "float4(" + elem(0) + ", " + elem(1) + ", " + elem(2) + ", " + elem(3) + ")";
    return "";  // unreachable — caller checks componentCount() first
}

// Parses MaterialX's comma-separated value string ("0, 0, 1", "0.5", "")
// into up to 4 floats, zero-filled for missing/empty components.
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

// Finds every `vec2 <name> ;` field inside the generated `struct VertexData
// { ... };` block, excluding `pos` (the clip-space position MaterialX always
// emits but which our wrapper never uses). These are texcoord varyings
// (`texcoord_0`, `texcoord_1`, ...) that a node graph's <texcoord> nodes pull
// UV from — MaterialX's generator treats them as per-fragment inputs, not
// uniforms, so they don't appear in gen.uniforms at all and were previously
// left completely uninitialized (undefined values), which silently broke
// any material using UV-driven patterns (found via a real-scene regression:
// no procedural wood grain, random black regions from garbage UV feeding
// into noise/pattern nodes).
std::vector<std::string> findTexcoordFields(const std::string& source) {
    std::vector<std::string> fields;
    size_t structPos = source.find("struct VertexData");
    if (structPos == std::string::npos) return fields;
    size_t bodyStart = source.find('{', structPos);
    size_t bodyEnd   = source.find('}', bodyStart);
    if (bodyStart == std::string::npos || bodyEnd == std::string::npos) return fields;
    std::string body = source.substr(bodyStart, bodyEnd - bodyStart);

    const std::string needle = "vec2 ";
    size_t pos = 0;
    while ((pos = body.find(needle, pos)) != std::string::npos) {
        size_t nameStart = pos + needle.size();
        size_t nameEnd = body.find_first_of(" ;", nameStart);
        if (nameEnd == std::string::npos) break;
        fields.push_back(body.substr(nameStart, nameEnd - nameStart));
        pos = nameEnd;
    }
    return fields;
}

// True if the generated VertexData struct declares a `vec3 positionObject`
// field — MaterialX emits this for <position space="object">, the object-
// space hit position used by procedural patterns (e.g. object-space noise
// driving wood grain). Only "object" space is handled, matching the CPU/OSL
// path's coverage (OslMaterial.cpp only implements transform-to-"object").
bool hasPositionObjectField(const std::string& source) {
    size_t structPos = source.find("struct VertexData");
    if (structPos == std::string::npos) return false;
    size_t bodyStart = source.find('{', structPos);
    size_t bodyEnd   = source.find('}', bodyStart);
    if (bodyStart == std::string::npos || bodyEnd == std::string::npos) return false;
    std::string body = source.substr(bodyStart, bodyEnd - bodyStart);
    return body.find("vec3 positionObject") != std::string::npos;
}

} // namespace

MxAdapterResult buildMslAdapter(const MxGeneratedShader& gen, const std::string& wrapperFnName) {
    MxAdapterResult result;
    if (!gen.valid) return result;

    // Layout the packed-float block for this material's uniforms, in the
    // exact order GlobalContext's constructor expects (== gen.uniforms
    // order, which mirrors PublicUniforms' declaration order).
    std::vector<std::string> unpackExprs;
    unpackExprs.reserve(gen.uniforms.size());
    int floatOffset = 0;
    for (const auto& u : gen.uniforms) {
        int n = componentCount(u.type);
        if (n < 0) {
            spdlog::warn("MaterialXMslAdapter: unsupported uniform type '{}' for '{}' — "
                         "falling back to the literal GPU material path for this input",
                         u.type, u.name);
            return result;  // .valid stays false
        }
        unpackExprs.push_back(unpackExpr(u.type, floatOffset));
        std::vector<float> vals = parseValue(u.value, n);
        result.defaultUniformData.insert(result.defaultUniformData.end(), vals.begin(), vals.end());
        floatOffset += n;
    }

    // Every generated wrapper shares one visible_function_table signature
    // (Metal requires this), so `uv` and `objPos` are always parameters —
    // even for materials whose VertexData has no texcoord/positionObject
    // field, where they're simply unused. See findTexcoordFields()'s and
    // hasPositionObjectField()'s comments for why this exists.
    std::vector<std::string> texcoordFields = findTexcoordFields(gen.source);
    bool hasPositionObject = hasPositionObjectField(gen.source);

    std::ostringstream wrapper;
    wrapper << "\n[[visible]] float4 " << wrapperFnName
            << "(const device float* uniformData, uint offset, float2 uv, float3 objPos) {\n"
            << "    VertexData vd;\n"
            << "    vd.pos = float4(0.0, 0.0, 0.0, 1.0);\n";
    for (const auto& field : texcoordFields) wrapper << "    vd." << field << " = uv;\n";
    if (hasPositionObject) wrapper << "    vd.positionObject = objPos;\n";
    wrapper << "    GlobalContext ctx(vd";
    for (const auto& expr : unpackExprs) wrapper << ",\n        " << expr;
    wrapper << ");\n"
            << "    PixelOutputs po = ctx.FragmentMain();\n"
            << "    return po." << gen.outputVarName << ";\n"
            << "}\n";

    result.valid         = true;
    result.source        = gen.source + wrapper.str();
    result.wrapperFnName = wrapperFnName;
    return result;
}

} // namespace anacapa
