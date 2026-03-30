#pragma once

#include <anacapa/scene/SceneLoader.h>
#include <string>

namespace anacapa {

// ---------------------------------------------------------------------------
// USDLoader — loads a USD/USDA/USDC stage into a LoadedScene.
//
// Supported USD prims:
//   UsdGeomMesh        → GeometryPool mesh + LambertianMaterial
//   UsdLuxRectLight    → AreaLight
//   UsdLuxSphereLight  → AreaLight (approximated as a small rect)
//   UsdLuxDistantLight → directional (future)
//   UsdGeomCamera      → Camera (first camera found wins)
//   UsdShadeMaterial   → UsdPreviewSurface → LambertianMaterial or EmissiveMaterial
//
// Materials are resolved via UsdShadeMaterial bindings on each mesh.
// If no binding exists, a default grey Lambertian is assigned.
// Transforms: all positions are baked into world space via
// UsdGeomXformable::GetLocalToWorldTransform().
// ---------------------------------------------------------------------------
LoadedScene loadUSD(const std::string& path,
                    uint32_t filmWidth,
                    uint32_t filmHeight);

} // namespace anacapa
