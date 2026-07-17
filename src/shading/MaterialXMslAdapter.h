#pragma once

// ---------------------------------------------------------------------------
// MaterialXMslAdapter — wraps raw MaterialX-generated MSL (a full fragment
// shader: `struct GlobalContext { ... FragmentMain() ... }; fragment
// PixelOutputs FragmentMain(VertexData vd [[stage_in]], constant
// PublicUniforms& u_pub [[buffer(0)]])`) with a plain `[[visible]]` device
// function callable via a Metal 3 visible_function_table from a compute
// kernel — which is what our path tracer needs, not a graphics fragment
// stage.
//
// The wrapper always has the signature `float4(const device float*, uint)`:
// reads its material's uniform values as a flat run of packed floats
// starting at the given offset (in the SAME order MaterialX declared them,
// which GlobalContext's constructor expects), constructs a GlobalContext,
// calls FragmentMain(), and returns the one output field — always a vec4
// per MaterialX's MSL generator, regardless of the underlying MaterialX
// type (float/color3/etc. are broadcast to vec4.xyz with w=1.0).
// ---------------------------------------------------------------------------

#include "MaterialXCodegen.h"
#include <string>
#include <vector>

namespace anacapa {

struct MxAdapterResult {
    bool        valid = false;   // false if any uniform has an unsupported type
    std::string source;          // gen.source + the appended [[visible]] wrapper
    std::string wrapperFnName;
    std::vector<float> defaultUniformData;  // packed floats, offset 0 = this material's block start
};

// wrapperFnName must be unique across the whole compiled scene (the visible
// function table is shared by every kMatMaterialX material).
MxAdapterResult buildMslAdapter(const MxGeneratedShader& gen, const std::string& wrapperFnName);

} // namespace anacapa
