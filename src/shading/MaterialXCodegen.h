#pragma once

// MaterialXCodegen — generates MSL/GLSL source for a single procedural
// surface-parameter output (base_color, roughness, etc.) from a per-material
// .mtlx file, for GPU shading.
//
// Scope: does NOT generate a full lit/HW shader with lighting, world
// matrices, or vertex stages — our own path tracer already handles shading
// and lighting. We only want MaterialX to evaluate one scalar/color output
// of a procedural node graph (the same graphs already exported by the
// Blender addon's _export_materialx_graphs() step, currently only consumed
// by the CPU OSL path) — the result feeds directly into the existing
// evalLayeredBSDF() on GPU, unmodified.

#include <memory>
#include <string>
#include <vector>

namespace anacapa {

struct MxUniform {
    std::string name;     // emitted shader variable name (e.g. "u_...")
    std::string type;     // MaterialX type string: "float", "color3", "vector2", ...
    std::string value;     // default value, as a MaterialX value string
};

struct MxGeneratedShader {
    bool               valid = false;   // false if the input has no procedural graph to generate
    std::string        functionName;
    std::string        source;          // full generated MSL or GLSL source
    std::vector<MxUniform> uniforms;
    // Name of the single field on the generated PixelOutputs struct holding
    // the result (always emitted as vec4 by MaterialX's MSL generator,
    // regardless of the underlying MaterialX type — float/color3/etc. are
    // all broadcast into vec4.xyz with w=1.0). Callers building an adapter
    // that invokes GlobalContext::FragmentMain() directly read this field.
    std::string        outputVarName;
};

class MaterialXCodegen {
public:
    MaterialXCodegen();
    ~MaterialXCodegen();

    // Generate MSL for `inputName` (e.g. "base_color") on the OpenPBR/
    // standard_surface shader node found in the document at `mtlxPath`.
    // Returns .valid=false if the input isn't connected to anything (a
    // literal value — nothing to generate, caller should use the existing
    // literal/texture path) or on any load/generation failure (logged).
    MxGeneratedShader generateMsl(const std::string& mtlxPath, const std::string& inputName);

    // Same, targeting GLSL (for the later CUDA post-processing step).
    MxGeneratedShader generateGlsl(const std::string& mtlxPath, const std::string& inputName);

    // Public so the .cpp's free helper functions can operate on it directly
    // (this class is a thin PIMPL to keep MaterialX types out of the public
    // header, not a hard encapsulation boundary within its own translation unit).
    struct Impl;

private:
    std::unique_ptr<Impl> m_impl;
};

} // namespace anacapa
