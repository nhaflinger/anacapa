#include "DomeLight.h"
#include <OpenImageIO/imageio.h>
#include <cstdio>

namespace anacapa {

// ---------------------------------------------------------------------------
// DomeLight::loadImage — load an equirectangular HDR image via OpenImageIO.
//
// Supported formats: EXR, HDR (.hdr/.rgbe), PNG, JPEG, etc.
// The image is converted to linear RGB float if needed.
// On failure, m_width / m_height remain 0 and buildDistribution() falls back
// to a uniform 1x1 grey environment.
// ---------------------------------------------------------------------------
void DomeLight::loadImage(const std::string& path) {
    using namespace OIIO;

    auto in = ImageInput::open(path);
    if (!in) {
        std::fprintf(stderr, "DomeLight: could not open '%s' — using constant grey\n", path.c_str());
        return;
    }

    const ImageSpec& spec = in->spec();
    int width    = spec.width;
    int height   = spec.height;
    int channels = spec.nchannels;

    if (width <= 0 || height <= 0) {
        std::fprintf(stderr, "DomeLight: empty image '%s' — using constant grey\n", path.c_str());
        in->close();
        return;
    }

    // Read into a temporary buffer (may have 1, 3, or 4 channels)
    std::vector<float> tmp(static_cast<size_t>(width * height * channels));
    bool ok = in->read_image(0, 0, 0, channels, TypeDesc::FLOAT, tmp.data());
    in->close();

    if (!ok) {
        std::fprintf(stderr, "DomeLight: failed to read pixels from '%s' — using constant grey\n", path.c_str());
        return;
    }

    // Determine if the image is stored in sRGB (i.e. LDR formats: JPEG/PNG/etc.)
    // EXR and HDR files are linear; everything else is assumed to be sRGB-encoded.
    // OIIO reads all formats to float in [0,1] without applying gamma,
    // so we must decode sRGB → linear for LDR images.
    bool isSrgb = false;
    {
        // Check OIIO's reported colorspace first
        std::string cs = spec.get_string_attribute("oiio:ColorSpace", "");
        if (cs == "sRGB" || cs == "srgb") {
            isSrgb = true;
        } else if (cs.empty() || cs == "Linear" || cs == "linear") {
            // Fall back to extension: EXR/HDR are linear, everything else sRGB
            const std::string& fn = path;
            auto dot = fn.rfind('.');
            if (dot != std::string::npos) {
                std::string ext = fn.substr(dot + 1);
                for (auto& c : ext) c = static_cast<char>(std::tolower(c));
                if (ext == "exr" || ext == "hdr" || ext == "rgbe")
                    isSrgb = false;
                else
                    isSrgb = true;  // jpg, jpeg, png, tiff, etc.
            }
        }
    }

    auto srgbToLinear = [](float x) -> float {
        x = std::max(0.f, x);
        return (x <= 0.04045f) ? x / 12.92f
                                : std::pow((x + 0.055f) / 1.055f, 2.4f);
    };

    // Convert to 3-channel RGB (drop alpha, replicate mono)
    m_width  = static_cast<uint32_t>(width);
    m_height = static_cast<uint32_t>(height);
    m_pixels.resize(static_cast<size_t>(m_width) * m_height * 3u);

    for (uint32_t i = 0; i < m_width * m_height; ++i) {
        const float* src = tmp.data() + static_cast<size_t>(i) * channels;
        float* dst = m_pixels.data() + i * 3u;
        if (channels == 1) {
            dst[0] = dst[1] = dst[2] = src[0];
        } else if (channels == 2) {
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = 0.f;
        } else {
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2];
        }
        // Clamp negatives (can appear in EXR due to firefly filters)
        dst[0] = std::max(0.f, dst[0]);
        dst[1] = std::max(0.f, dst[1]);
        dst[2] = std::max(0.f, dst[2]);
        // sRGB → linear decode for LDR formats
        if (isSrgb) {
            dst[0] = srgbToLinear(dst[0]);
            dst[1] = srgbToLinear(dst[1]);
            dst[2] = srgbToLinear(dst[2]);
        }
    }

    std::fprintf(stderr, "DomeLight: colorspace=%s\n", isSrgb ? "sRGB→linear" : "linear");

    std::fprintf(stderr, "DomeLight: loaded '%s' (%ux%u)\n",
                 path.c_str(), m_width, m_height);
}

} // namespace anacapa
