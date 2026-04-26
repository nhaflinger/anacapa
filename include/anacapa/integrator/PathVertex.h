#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/accel/IAccelerationStructure.h>
#include <cstdint>
#include <vector>

namespace anacapa {

class IMaterial;
class ILight;

// ---------------------------------------------------------------------------
// PathVertexType — packed into flags field
// ---------------------------------------------------------------------------
enum class PathVertexType : uint8_t {
    Camera  = 0,
    Surface = 1,
    Light   = 2,
};

// Flags word layout:
//   bits 0–1  : PathVertexType
//   bit  2    : isDelta  (BSDF or light has a delta/measure-zero component)
//   bit  3    : isInfinite (environment map / directional light vertex)
//   bit  4    : isConnectible (not delta — can be explicitly connected)
static constexpr uint32_t kVertexTypeMask    = 0x03u;
static constexpr uint32_t kVertexDeltaBit    = 1u << 2;
static constexpr uint32_t kVertexInfiniteBit = 1u << 3;

// ---------------------------------------------------------------------------
// PathVertexBuffer — SoA storage for one subpath
//
// All arrays are parallel and indexed by vertex index [0, count).
// The SoA layout gives:
//   - Cache-friendly sequential access during MIS weight computation
//     (which walks the arrays linearly)
//   - Trivial migration to GPU: each std::vector maps to one MTLBuffer /
//     CUDA device pointer via a DeviceView
//
// Memory is pre-allocated at construction (maxVerts capacity).
// reset() sets count=0 with no deallocation.
// ---------------------------------------------------------------------------
struct PathVertexBuffer {
    // Geometry
    std::vector<Vec3f>    position;
    std::vector<Vec3f>    normal;     // Shading normal at the vertex
    std::vector<Vec3f>    wo;         // Direction leaving this vertex (toward previous)

    // Transport
    std::vector<Spectrum> beta;       // Path throughput up to and including this vertex
    std::vector<Spectrum> Le;         // Emitted radiance (non-zero at light vertices)

    // PDFs — stored in area measure for consistent MIS weight formulas
    std::vector<float>    pdfFwd;     // Area PDF of sampling this vertex from previous
    std::vector<float>    pdfRev;     // Area PDF of sampling previous vertex from this

    // Metadata
    std::vector<uint32_t> flags;      // PathVertexType | delta | infinite bits
    std::vector<uint32_t> meshID;     // ~0u for camera/light endpoint vertices

    // Non-owning pointers into scene data (not transferred to GPU — looked up by ID)
    std::vector<const IMaterial*> material;
    std::vector<const ILight*>    light;  // Non-null only at light endpoint vertices

    // Minimum perceptual roughness seen along the path from its origin up to
    // and including this vertex.  Starts at 1.0 (camera/light endpoint) and
    // is updated at each surface bounce.  Used by connect() to skip shadow
    // rays toward near-specular vertices where the geometry-sampling PDF is
    // near zero and the contribution would be discarded after MIS weighting.
    std::vector<float> pathMinRoughness;

    // lightPdf: probability of selecting the light used for this subpath from
    // the light sampler.  Stored for vertex 0 only (the light endpoint).
    // Used by connect() s=1 to correctly normalize infinite-light contributions
    // without conflating lightPdf with the position/direction sampling PDF.
    float lightPdf = 1.f;

    // Scene time sampled for this subpath — propagated to all spawned rays so
    // the entire path evaluates the scene at one consistent moment in time.
    float sceneTime = 0.f;

    uint32_t count    = 0;
    uint32_t capacity = 0;

    explicit PathVertexBuffer(uint32_t maxVerts) : capacity(maxVerts) {
        position.resize(maxVerts);
        normal.resize(maxVerts);
        wo.resize(maxVerts);
        beta.resize(maxVerts);
        Le.resize(maxVerts);
        pdfFwd.resize(maxVerts, 0.f);
        pdfRev.resize(maxVerts, 0.f);
        flags.resize(maxVerts, 0u);
        meshID.resize(maxVerts, ~0u);
        material.resize(maxVerts, nullptr);
        light.resize(maxVerts, nullptr);
        pathMinRoughness.resize(maxVerts, 1.0f);
    }

    void reset() { count = 0; }

    bool full() const { return count >= capacity; }

    // Typed accessors
    PathVertexType type(uint32_t i) const {
        return static_cast<PathVertexType>(flags[i] & kVertexTypeMask);
    }
    bool isDelta(uint32_t i)    const { return (flags[i] & kVertexDeltaBit)    != 0; }
    bool isInfinite(uint32_t i) const { return (flags[i] & kVertexInfiniteBit) != 0; }
    bool isConnectible(uint32_t i) const { return !isDelta(i); }

    // ---------------------------------------------------------------------------
    // DeviceView — POD snapshot for GPU kernel hand-off (Phase 5)
    // Returned by value; each field is a raw pointer into our std::vector data.
    // On GPU: replace std::vector backing with cudaMalloc / MTLBuffer.
    // ---------------------------------------------------------------------------
    struct DeviceView {
        Vec3f*    position;
        Vec3f*    normal;
        Vec3f*    wo;
        Spectrum* beta;
        Spectrum* Le;
        float*    pdfFwd;
        float*    pdfRev;
        uint32_t* flags;
        uint32_t* meshID;
        uint32_t  count;
    };

    DeviceView deviceView() {
        return { position.data(), normal.data(), wo.data(),
                 beta.data(), Le.data(),
                 pdfFwd.data(), pdfRev.data(),
                 flags.data(), meshID.data(),
                 count };
    }
};

// ---------------------------------------------------------------------------
// Area-measure PDF conversion helpers
//
// BDPT stores all PDFs in area measure so the MIS weight ratio formula
// (Veach §10.3) can be expressed as a simple product chain without
// repeatedly converting between solid-angle and area measure mid-computation.
//
// convertToArea: converts a solid-angle PDF at 'prev' for a direction
// toward 'next', into an area PDF at 'next'.
//   pdfArea = pdfSolidAngle * |cos(theta_next)| / dist²
// ---------------------------------------------------------------------------
inline float convertToArea(float pdfSolidAngle,
                            Vec3f prevPos, Vec3f nextPos, Vec3f nextNormal) {
    Vec3f d    = nextPos - prevPos;
    float dist2 = d.lengthSq();
    if (dist2 < 1e-10f) return 0.f;
    float cosTheta = std::abs(dot(safeNormalize(d), nextNormal));
    return pdfSolidAngle * cosTheta / dist2;
}

} // namespace anacapa
