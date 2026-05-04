#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/accel/GeometryPool.h>  // MotionKey
#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// StrandDesc — one hair/fur strand as a sequence of cubic Bézier segments.
//
// Control point layout (endpoint-sharing):
//   numSegments = (controlPoints.size() - 1) / 3
//   Segment i uses control points [3*i, 3*i+1, 3*i+2, 3*i+3].
//
// Width convention (diameter, not radius):
//   empty  → default 0.001 world units
//   size 2 → linearly interpolated from root (widths[0]) to tip (widths[1])
//   otherwise widths.size() must equal controlPoints.size()
//
// Control points are in world space for static strands; in object space when
// motionKeys is non-empty (same convention as MeshDesc).
// ---------------------------------------------------------------------------
struct StrandDesc {
    std::string            name;
    std::vector<Vec3f>     controlPoints;  // cubic Bézier CVs
    std::vector<float>     widths;         // per-CV diameter
    std::vector<MotionKey> motionKeys;     // empty = static
    uint32_t               materialIndex = 0;
    Vec3f                  color = {1.f, 1.f, 1.f};  // per-strand RGB (linear); white = use material default

    bool     hasMotion()   const { return !motionKeys.empty(); }

    // Number of cubic Bézier segments in this strand.
    uint32_t numSegments() const {
        if (controlPoints.size() < 4) return 0;
        return static_cast<uint32_t>((controlPoints.size() - 1) / 3);
    }

    // Width (diameter) at normalized strand parameter v ∈ [0, 1].
    // v=0 is the root, v=1 is the tip.
    float widthAt(float v) const {
        if (widths.empty()) return 0.001f;
        if (widths.size() == 1) return widths[0];
        if (widths.size() == 2) return widths[0] * (1.f - v) + widths[1] * v;
        float idx  = v * float(int(widths.size()) - 1);
        int   lo   = std::max(0, int(idx));
        int   hi   = std::min(int(widths.size()) - 1, lo + 1);
        float frac = idx - float(lo);
        return widths[lo] * (1.f - frac) + widths[hi] * frac;
    }
};

// ---------------------------------------------------------------------------
// CurvePool — owns all strand data for the scene.
//
// Parallel to GeometryPool; the BVH / curve acceleration structure receives
// a const reference and builds its data structures over this.
// ---------------------------------------------------------------------------
class CurvePool {
public:
    uint32_t addStrand(StrandDesc strand) {
        uint32_t id = static_cast<uint32_t>(m_strands.size());
        m_strands.push_back(std::move(strand));
        return id;
    }

    const StrandDesc& strand(uint32_t id)   const { return m_strands[id]; }
    size_t            numStrands()           const { return m_strands.size(); }
    const std::vector<StrandDesc>& strands() const { return m_strands; }

private:
    std::vector<StrandDesc> m_strands;
};

} // namespace anacapa
