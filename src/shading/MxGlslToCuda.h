#pragma once

// ---------------------------------------------------------------------------
// MxGlslToCuda — wraps MaterialX-generated GLSL (from
// MaterialXCodegen::generateGlsl()) into a callable CUDA __device__ function.
//
// Unlike the MSL backend (see MaterialXMslAdapter.h), MaterialX's GLSL output
// is NOT wrapped in a class with a constructor — it's a flat `void main()`
// that reads plain `uniform TYPE NAME = default;` globals and writes an
// `out vec4 <name>;` global. This means the adapter doesn't need to reverse-
// engineer a constructor call — it only needs to:
//   1. Turn each `uniform` global into a local variable initialized from a
//      packed float buffer (same technique as MaterialXMslAdapter).
//   2. Turn the `out vec4 <name>;` global into a local variable, returned at
//      the end of the function body.
//   3. Provide GLSL-compatible vec2/vec3/vec4/mix/clamp/etc. as real CUDA
//      types+functions (CUDA's own float2/float3/float4 don't support GLSL's
//      constructor-call syntax) — see the embedded prelude in the .cpp.
//
// Untested — no CUDA compiler is available in the environment this was
// written in. See project memory project_materialx_codegen, Phase 3 section,
// for what to check first if this doesn't compile/run on the target machine.
// ---------------------------------------------------------------------------

#include "MaterialXCodegen.h"
#include <string>
#include <vector>

namespace anacapa {

struct MxCudaAdapterResult {
    bool        valid = false;   // false if any uniform has an unsupported type
    std::string source;          // self-contained CUDA source: prelude + adapted body
    // Plain base name (e.g. "anacapa_mx_m2_base_color") — the actual emitted
    // function symbol is "__direct_callable__" + wrapperFnName, per OptiX's
    // direct-callable naming convention. Callers building the
    // OptixProgramGroupDesc must prefix it themselves.
    std::string wrapperFnName;
    std::vector<float> defaultUniformData;  // packed floats, offset 0 = this material's block start
};

// wrapperFnName must be unique across the whole compiled scene.
MxCudaAdapterResult buildCudaAdapter(const MxGeneratedShader& gen, const std::string& wrapperFnName);

} // namespace anacapa
