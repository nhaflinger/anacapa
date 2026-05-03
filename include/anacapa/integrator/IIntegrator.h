#pragma once

#include <anacapa/core/Types.h>
#include <anacapa/film/Film.h>
#include <cstdint>
#include <optional>
#include <vector>

namespace anacapa {

class IAccelerationStructure;
class IMaterial;
class ILight;
class ISampler;
struct SurfaceInteraction;

// ---------------------------------------------------------------------------
// Camera — pinhole or thin-lens camera.
//
// Thin lens model:
//   apertureRadius > 0  enables depth of field.  A point on the aperture
//   disk is sampled (using lensU, lensV passed to generateRay) and the ray
//   is redirected to converge at the focal plane at distance focalDistance.
//
// Default (apertureRadius == 0): pure pinhole — lensU/lensV are ignored.
// ---------------------------------------------------------------------------
struct Camera {
    Vec3f  origin;
    Vec3f  lowerLeftCorner;
    Vec3f  horizontal;   // Full width vector
    Vec3f  vertical;     // Full height vector
    // Camera basis — needed to offset ray origin across the aperture disk
    Vec3f  basisU;       // Right vector (unit)
    Vec3f  basisV;       // Up vector (unit)
    uint32_t imageWidth;
    uint32_t imageHeight;

    // Thin lens parameters — both zero means pinhole (no DoF)
    float apertureRadius = 0.f;  // Half the aperture diameter in world units
    float focalDistance  = 0.f;  // Distance from origin to the focal plane
    float focalLength    = 0.f;  // Physical focal length in world units (metres for metre-scale scenes)

    // Motion blur shutter interval in normalized [0, 1] time.
    // shutterOpen == shutterClose (both 0) means no motion blur —
    // all rays get time=0 and transforms are sampled at t=0 only.
    float shutterOpen  = 0.f;
    float shutterClose = 0.f;

    static Camera makePinhole(Vec3f from, Vec3f at, Vec3f up,
                               float vfovDegrees,
                               uint32_t width, uint32_t height) {
        float theta       = vfovDegrees * 3.14159265f / 180.f;
        float h           = std::tan(theta * 0.5f);
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

        Vec3f w = normalize(from - at);
        Vec3f u = normalize(cross(up, w));
        Vec3f v = cross(w, u);

        float vpH = 2.f * h;
        float vpW = aspectRatio * vpH;

        Camera cam;
        cam.origin          = from;
        cam.horizontal      = u * vpW;
        cam.vertical        = v * vpH;
        cam.lowerLeftCorner = from - u*(vpW*0.5f) - v*(vpH*0.5f) - w;
        cam.basisU          = u;
        cam.basisV          = v;
        cam.imageWidth      = width;
        cam.imageHeight     = height;
        // apertureRadius and focalDistance stay 0 — pure pinhole
        return cam;
    }

    static Camera makeThinLens(Vec3f from, Vec3f at, Vec3f up,
                                float vfovDegrees,
                                uint32_t width, uint32_t height,
                                float apertureRadius, float focalDistance) {
        Camera cam = makePinhole(from, at, up, vfovDegrees, width, height);
        cam.apertureRadius = apertureRadius;
        cam.focalDistance  = focalDistance;
        return cam;
    }

    // Generate a ray for pixel (px, py).
    //   jitterU, jitterV — sub-pixel jitter in [0, 1)
    //   lensU,   lensV   — aperture sample in [0, 1); ignored when apertureRadius == 0
    //   timeU            — shutter time sample in [0, 1); mapped to [shutterOpen, shutterClose]
    Ray generateRay(uint32_t px, uint32_t py,
                    float jitterU, float jitterV,
                    float lensU = 0.5f, float lensV = 0.5f,
                    float timeU = 0.f) const {
        float s = (static_cast<float>(px) + jitterU) / static_cast<float>(imageWidth);
        // Flip py: row 0 is the top of the image, but lowerLeftCorner + vertical*0
        // is the bottom of the frustum. Invert so the image isn't upside-down.
        float t = (static_cast<float>(imageHeight - 1 - py) + jitterV)
                / static_cast<float>(imageHeight);

        Vec3f target = lowerLeftCorner + horizontal*s + vertical*t;

        // Map [0,1) time sample to the shutter interval
        float rayTime = shutterOpen + timeU * (shutterClose - shutterOpen);

        if (apertureRadius <= 0.f) {
            Ray r{origin, normalize(target - origin)};
            r.time = rayTime;
            return r;
        }

        // Thin lens: sample a point on the aperture disk using concentric
        // mapping (preserves uniform distribution, avoids clumping at center).
        // Map lensU/V from [0,1) to [-1,1) then apply concentric disk mapping.
        float lx = 2.f * lensU - 1.f;
        float ly = 2.f * lensV - 1.f;
        float diskX, diskY;
        if (lx == 0.f && ly == 0.f) {
            diskX = diskY = 0.f;
        } else if (std::abs(lx) >= std::abs(ly)) {
            float r   = lx;
            float phi = (3.14159265f / 4.f) * (ly / lx);
            diskX = r * std::cos(phi);
            diskY = r * std::sin(phi);
        } else {
            float r   = ly;
            float phi = (3.14159265f / 2.f) - (3.14159265f / 4.f) * (lx / ly);
            diskX = r * std::cos(phi);
            diskY = r * std::sin(phi);
        }
        Vec3f lensPoint = origin
                        + basisU * (diskX * apertureRadius)
                        + basisV * (diskY * apertureRadius);

        // The focal point is where the pinhole ray intersects the focal plane.
        // pinholeDir scaled so its z-component (along -w) equals focalDistance.
        Vec3f pinholeDir = normalize(target - origin);
        Vec3f focalPoint = origin + pinholeDir * focalDistance;

        // Rays from all lens points converge at the focal point.
        Ray r{lensPoint, normalize(focalPoint - lensPoint)};
        r.time = rayTime;
        return r;
    }
};

// ---------------------------------------------------------------------------
// SceneView — non-owning view of the scene passed to integrators.
// Separates the integrator from the scene ownership model.
// ---------------------------------------------------------------------------
struct SceneView {
    const IAccelerationStructure*        accel    = nullptr;
    std::vector<const IMaterial*>        materials;   // indexed by meshID
    std::vector<const ILight*>           lights;
    const ILight*                        envLight = nullptr;  // Infinite/dome light (nullptr = constant)
    Vec3f                                envRadiance = {};    // Constant background (used if envLight==nullptr)
    std::optional<Camera>                camera;              // set by scene loader
};

// ---------------------------------------------------------------------------
// TileRequest — describes a rectangular region of the film to render
// ---------------------------------------------------------------------------
struct TileRequest {
    uint32_t x0, y0;          // Pixel-space origin (inclusive)
    uint32_t width, height;
    uint32_t sampleStart;      // First sample index (for low-discrepancy offsets)
    uint32_t sampleCount;
};

// ---------------------------------------------------------------------------
// IIntegrator
// ---------------------------------------------------------------------------
class IIntegrator {
public:
    virtual ~IIntegrator() = default;

    // One-time setup after the scene is committed (build light samplers, etc.)
    virtual void prepare(const SceneView& scene) = 0;

    // Render a single tile. Must be thread-safe — called concurrently.
    // Writes to localTile; caller merges into Film.
    virtual void renderTile(const SceneView& scene,
                            const TileRequest& tile,
                            uint32_t filmWidth,
                            uint32_t filmHeight,
                            ISampler& sampler,
                            TileBuffer& localTile) = 0;

    // Optional: render the entire frame in one call and write directly into
    // film.  Returns true if the frame was rendered (caller skips tile
    // dispatch).  Default returns false — tile dispatch is used instead.
    // Not called for adaptive rendering passes.
    virtual bool renderFrame(const SceneView& scene,
                             uint32_t filmWidth,
                             uint32_t filmHeight,
                             uint32_t sampleStart,
                             uint32_t sampleCount,
                             Film& film) { return false; }
};

} // namespace anacapa
