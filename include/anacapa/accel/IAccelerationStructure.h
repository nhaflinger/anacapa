#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/accel/GeometryPool.h>
#include <cstdint>

namespace anacapa {

// ---------------------------------------------------------------------------
// SurfaceInteraction — result of a successful ray-surface intersection
// ---------------------------------------------------------------------------
struct SurfaceInteraction {
    Vec3f    p;           // World-space hit point
    Vec3f    n;           // Shading normal (normalized)
    Vec3f    ng;          // Geometric normal (normalized)
    Vec3f    dpdu, dpdv;  // Partial derivatives (tangent frame)
                          // For curves: dpdu = fiber/tangent direction along strand
    Vec2f    uv;          // Surface parameterization
                          // For curves: uv.y = strand parameter v ∈ [0,1] (root→tip)
    float    t = 0.f;     // Ray parameter at hit
    uint32_t meshID     = ~0u;
    uint32_t primID     = ~0u;  // Triangle index within mesh; segment index for curves
    uint32_t instanceID = ~0u;
    uint32_t strandID   = ~0u;  // Index into CurvePool (curves only; ~0u for triangles)
    bool     isCurve    = false; // True when the hit is a hair/curve primitive
    Vec3f    color      = {1.f, 1.f, 1.f};  // per-strand color propagated from StrandDesc

    // Filled in by the scene after intersection, not by the BVH
    const void* material = nullptr;  // IMaterial* — void* avoids circular include
};

// ---------------------------------------------------------------------------
// TraceResult
// ---------------------------------------------------------------------------
struct TraceResult {
    bool               hit = false;
    SurfaceInteraction si;
};

// ---------------------------------------------------------------------------
// IAccelerationStructure
//
// Abstracts the ray traversal backend. The CPU implementation is our custom
// SAH BVH (BVHBackend). Future backends:
//   - MetalBackend  (Phase 5, macOS) : MTLAccelerationStructure
//   - CUDABackend   (Phase 5, Linux) : optixLaunch
//
// Design rules:
//   - All methods are const after commit() — no mutable traversal state.
//   - traceBatch / occludedBatch are the kernel-boundary calls. On CPU they
//     loop over rays; on GPU they dispatch a kernel.
//   - No heap allocation inside trace() — must be safe to call from any
//     thread or GPU warp.
// ---------------------------------------------------------------------------
class IAccelerationStructure {
public:
    virtual ~IAccelerationStructure() = default;

    // Finalize geometry for rendering. Called once after all meshes are added.
    virtual void commit() = 0;

    // Access the underlying geometry pool (needed by GPU backends to build
    // hardware acceleration structures from the same data).
    virtual const GeometryPool& pool() const = 0;

    // Single-ray full intersection (used inside integrator loops on CPU)
    virtual TraceResult trace(const Ray& ray) const = 0;

    // Single-ray occlusion test — faster than trace(), no SurfaceInteraction
    virtual bool occluded(const Ray& ray) const = 0;

    // Batch interfaces — the GPU kernel boundary.
    // Default implementations loop over the single-ray variants.
    // GPU backends override with a single kernel dispatch.
    virtual void traceBatch(Span<const Ray>   rays,
                            Span<TraceResult> results) const {
        assert(rays.size() == results.size());
        for (size_t i = 0; i < rays.size(); ++i)
            results[i] = trace(rays[i]);
    }

    virtual void occludedBatch(Span<const Ray> rays,
                               Span<bool>      results) const {
        assert(rays.size() == results.size());
        for (size_t i = 0; i < rays.size(); ++i)
            results[i] = occluded(rays[i]);
    }
};

} // namespace anacapa
