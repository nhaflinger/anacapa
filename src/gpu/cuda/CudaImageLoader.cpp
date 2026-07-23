#include "CudaImageLoader.h"

#include <OpenImageIO/imageio.h>

#include <cstdio>

namespace anacapa {

bool readCudaImagePixels(const std::string& path,
                         std::vector<float>& pixels,
                         int& w, int& h, int& nchans)
{
    auto in = OIIO::ImageInput::open(path);
    if (!in) {
        fprintf(stderr, "[warn] CudaPathIntegrator: failed to open texture '%s'\n",
                path.c_str());
        return false;
    }
    const OIIO::ImageSpec& spec = in->spec();
    w = spec.width; h = spec.height; nchans = spec.nchannels;
    pixels.assign(size_t(w) * size_t(h) * size_t(nchans), 0.f);
    const bool ok = in->read_image(0, 0, 0, nchans, OIIO::TypeDesc::FLOAT,
                                    pixels.data());
    in->close();
    if (!ok || w <= 0 || h <= 0) {
        fprintf(stderr, "[warn] CudaPathIntegrator: failed to read texture '%s'\n",
                path.c_str());
        return false;
    }
    return true;
}

} // namespace anacapa
