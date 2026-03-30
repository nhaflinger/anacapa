#pragma once

#include <anacapa/accel/GeometryPool.h>
#include <anacapa/integrator/IIntegrator.h>  // SceneView, Camera
#include <anacapa/shading/IMaterial.h>
#include <anacapa/shading/ILight.h>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace anacapa {

// ---------------------------------------------------------------------------
// LoadedScene — owns all materials and lights produced by a scene loader.
// SceneView holds non-owning pointers into these vectors.
// Camera is optional — if not present the renderer uses its default.
// ---------------------------------------------------------------------------
struct LoadedScene {
    bool                                   valid = false; // false if stage failed to open
    GeometryPool                           geomPool;
    SceneView                              sceneView;
    std::vector<std::unique_ptr<IMaterial>> materials;
    std::vector<std::unique_ptr<ILight>>   lights;
    std::optional<Camera>                  camera;
};

} // namespace anacapa
