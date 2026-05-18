#pragma once

#include <anacapa/core/Types.h>
#include <cstdint>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// HaloDesc — one camera-facing disc particle.
//
// center      — world-space position at shutter-open (or static position)
// centerClose — world-space position at shutter-close (= center when no motion)
// radius      — disc radius (= widths/2 from UsdGeomPoints, which stores diameters)
// color       — per-particle base color (from primvars:displayColor or white)
// matIdx      — index into LoadedScene::materials; drives shading + emission
// ---------------------------------------------------------------------------
struct HaloDesc {
    Vec3f    center      = {};
    Vec3f    centerClose = {};   // same as center when no motion blur
    float    radius      = 0.01f;
    Vec3f    color       = {1.f, 1.f, 1.f};
    uint32_t matIdx      = 0;

    bool hasMotion() const {
        return center.x != centerClose.x
            || center.y != centerClose.y
            || center.z != centerClose.z;
    }
};

// ---------------------------------------------------------------------------
// HaloPool — owns all halo disc particles for a scene.
// ---------------------------------------------------------------------------
class HaloPool {
public:
    uint32_t addHalo(const HaloDesc& h) {
        uint32_t id = static_cast<uint32_t>(m_halos.size());
        m_halos.push_back(h);
        return id;
    }

    const HaloDesc& halo(uint32_t id) const { return m_halos[id]; }

    size_t numHalos() const { return m_halos.size(); }

    const std::vector<HaloDesc>& halos() const { return m_halos; }

private:
    std::vector<HaloDesc> m_halos;
};

} // namespace anacapa
