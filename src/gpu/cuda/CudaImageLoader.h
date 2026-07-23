#pragma once

// Small OIIO shim used by CudaPathIntegrator.cu.
//
// Kept as a plain C++ header (no OIIO includes) so the CUDA translation
// unit stays free of OIIO/spdlog/fmt — nvcc + OIIO 2.5's fmt-heavy detail
// headers don't compile cleanly together.  Implementation lives in
// CudaImageLoader.cpp which is the only site pulling OIIO into this backend.

#include <cstdint>
#include <string>
#include <vector>

namespace anacapa {

// Reads `path` into row-major float pixels.  On success: sets w/h/nchans and
// returns true.  On failure (missing file, decode error, invalid dims) prints
// a warning to stderr and returns false; pixels is left untouched.
bool readCudaImagePixels(const std::string& path,
                         std::vector<float>& pixels,
                         int& w, int& h, int& nchans);

} // namespace anacapa
