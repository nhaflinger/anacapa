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
// Camera — pinhole camera (defined before SceneView so SceneView can embed it)
// ---------------------------------------------------------------------------
struct Camera {
    Vec3f  origin;
    Vec3f  lowerLeftCorner;
    Vec3f  horizontal;   // Full width vector
    Vec3f  vertical;     // Full height vector
    uint32_t imageWidth;
    uint32_t imageHeight;

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
        cam.imageWidth      = width;
        cam.imageHeight     = height;
        return cam;
    }

    // Generate a ray for pixel (px, py) with sub-pixel offset (u, v) in [0,1)
    Ray generateRay(uint32_t px, uint32_t py, float u, float v) const {
        float s = (static_cast<float>(px) + u) / static_cast<float>(imageWidth);
        // Flip py: row 0 is the top of the image, but lowerLeftCorner + vertical*0
        // is the bottom of the frustum. Invert so the image isn't upside-down.
        float t = (static_cast<float>(imageHeight - 1 - py) + v)
                / static_cast<float>(imageHeight);
        Vec3f dir = lowerLeftCorner + horizontal*s + vertical*t - origin;
        return Ray{origin, normalize(dir)};
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
    Vec3f                                envRadiance = {};  // Background color
    std::optional<Camera>                camera;            // set by scene loader
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
};

} // namespace anacapa
