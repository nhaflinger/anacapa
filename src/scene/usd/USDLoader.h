#pragma once

#include <anacapa/scene/SceneLoader.h>
#include <limits>
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
//   UsdGeomCamera      → Pinhole camera
//   UsdShadeMaterial   → UsdPreviewSurface → LambertianMaterial or EmissiveMaterial
//
// Camera selection priority:
//   1. cameraOverridePath — explicit prim path from --camera flag
//   2. UsdRenderSettings.camera relationship (first RenderSettings prim found)
//   3. First UsdGeomCamera encountered during traversal
//
// Pass --list-cameras to discover available camera paths without rendering.
// ---------------------------------------------------------------------------

LoadedScene loadUSD(const std::string& path,
                    uint32_t filmWidth,
                    uint32_t filmHeight,
                    const std::string& cameraOverridePath = "",
                    double frame        = std::numeric_limits<double>::quiet_NaN(),
                    float  shutterOpen  = 0.f,
                    float  shutterClose = 0.f,
                    bool   skipOSL      = false);

} // namespace anacapa
