#include <anacapa/film/Film.h>
#include <OpenImageIO/imageio.h>
#include <cassert>
#include <cmath>
#include <vector>

namespace anacapa {

Film::Film(uint32_t width, uint32_t height)
    : m_width(width), m_height(height), m_pixels(width * height)
{}

void Film::splatPixel(float x, float y, Spectrum value) {
    // Clamp NaN/Inf before accumulating
    if (!value.isFinite()) return;

    int px = static_cast<int>(x);
    int py = static_cast<int>(y);
    if (!inBounds(px, py)) return;

    m_pixels[py * m_width + px].add(value.x, value.y, value.z, 1.f);
}

void Film::mergeTile(const TileBuffer& tile) {
    for (uint32_t ty = 0; ty < tile.height; ++ty) {
        for (uint32_t tx = 0; tx < tile.width; ++tx) {
            const auto& s = tile.pixels[ty * tile.width + tx];
            if (s.weight <= 0.f) continue;

            uint32_t fx = tile.x0 + tx;
            uint32_t fy = tile.y0 + ty;
            if (!inBounds(static_cast<int>(fx), static_cast<int>(fy))) continue;

            m_pixels[fy * m_width + fx].add(s.r, s.g, s.b, s.weight);
        }
    }
}

Spectrum Film::getPixel(uint32_t x, uint32_t y) const {
    assert(inBounds(static_cast<int>(x), static_cast<int>(y)));
    return m_pixels[y * m_width + x].resolve();
}

bool Film::writeEXR(const std::string& path) const {
    using namespace OIIO;

    // Resolve pixels into a linear float buffer (RGB)
    std::vector<float> buf(m_width * m_height * 3);
    for (uint32_t y = 0; y < m_height; ++y) {
        for (uint32_t x = 0; x < m_width; ++x) {
            Spectrum s = m_pixels[y * m_width + x].resolve();
            size_t off = (y * m_width + x) * 3;
            buf[off + 0] = s.x;
            buf[off + 1] = s.y;
            buf[off + 2] = s.z;
        }
    }

    auto out = ImageOutput::create(path);
    if (!out) return false;

    ImageSpec spec(static_cast<int>(m_width), static_cast<int>(m_height),
                   3, TypeDesc::FLOAT);
    spec.attribute("compression", "zip");
    spec.channelnames = {"R", "G", "B"};

    if (!out->open(path, spec)) return false;

    bool ok = out->write_image(TypeDesc::FLOAT, buf.data());
    out->close();
    return ok;
}

} // namespace anacapa
