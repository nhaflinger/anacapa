#pragma once

// ---------------------------------------------------------------------------
// OslMaterial — thin public header.
//
// This header is intentionally free of OSL, OIIO, and any fmt-pulling
// includes so that translation units (e.g. USDLoader.cpp) that include
// spdlog can safely include this header without triggering the
// fmt 10.x / fmt 12.x conflict.
//
// The concrete OslMaterial class and all OSL/OIIO headers live in
// OslMaterial.cpp only.
// ---------------------------------------------------------------------------

#include <anacapa/shading/IMaterial.h>
#include <memory>
#include <string>

namespace anacapa {

#ifdef ANACAPA_ENABLE_OSL

// Compile a .osl source file to a .oso object file.
// matDir is added to the compiler include path (for _mx_stdlib.h etc.).
// Returns true on success; logs a warning on failure.
bool oslCompileShader(const std::string& oslPath,
                      const std::string& osoPath,
                      const std::string& matDir);

// Register a directory with the OSL shader search path.
// Must be called at least once before makeOslMaterial() for any shader
// whose .oso lives in that directory.
void oslAddSearchPath(const std::string& dir);

// Create an OslMaterial from a compiled shader name (without .oso extension).
// The .oso file must already be present in a directory registered via
// oslAddSearchPath().  Returns nullptr on failure.
std::unique_ptr<IMaterial> makeOslMaterial(const std::string& shaderName);

#endif  // ANACAPA_ENABLE_OSL

} // namespace anacapa
