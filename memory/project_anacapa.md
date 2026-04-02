---
name: Anacapa Renderer — Project State
description: Current state, architecture decisions, and phase plan for the Anacapa bidirectional path tracer
type: project
---

Anacapa is a bidirectional path tracer in C++20 at /Users/douglascreel/dev/anacapa.

**Why:** Learning/research project building toward a production-quality physically-based renderer with GPU acceleration.

**How to apply:** Use this as context for all implementation decisions.

## Architecture Decisions (locked in)

- **No compiled third-party dependencies in the core renderer** — custom SAH BVH, custom ThreadPool
- **Header-only deps only:** spdlog, CLI11, GoogleTest (via FetchContent)
- **One system dep:** OpenImageIO (EXR output + texture loading) — `brew install openimageio`
- **SoA memory layout throughout** — required for GPU migration
- **GPU backends:** Metal (Apple Silicon, Phase 5) and CUDA+OptiX (NVIDIA Linux/Windows, Phase 5)
- **No OpenCL** — user decided against it; CUDA only for non-Apple GPU
- **USD scene format** — Phase 3, optional via `ANACAPA_ENABLE_USD` CMake flag
- **IMPORTANT:** DomeLight.cpp must NOT include spdlog — OIIO 3.1 + spdlog bundled fmt clash in the same TU. Use std::fprintf instead.

## What's Built (Phase 1 — COMPLETE)

21/21 unit tests passing. Cornell box renders at 800x600 @ 64 SPP in ~11s on Apple Silicon (11 threads). EXR output confirmed.

All Phase 1 files: see original memory or git log.

## What's Built (Phase 2 — COMPLETE)

31/31 unit tests passing. BDPT renders Cornell box (400×400 @ 16 spp) in ~1.2s.

- `src/integrator/BDPTIntegrator.h/.cpp` — full BDPT with all (s,t) strategies
- `src/integrator/MISWeight.cpp` — Veach §10.3 power heuristic MIS weights
- `src/integrator/LightSampler.h` — Vose alias table light selection
- `include/anacapa/integrator/PathVertex.h` — SoA PathVertexBuffer + DeviceView
- `include/anacapa/integrator/MISWeight.h` — bdptMISWeight() interface
- `tests/unit/test_bdpt.cpp` — 10 BDPT unit tests

## What's Built (Phase 4 — COMPLETE)

42/42 unit tests passing. Smoke render still ~1.1s at 400×400 @ 16 spp.

### StandardSurfaceMaterial (`src/shading/StandardSurface.h`)
- GGX multi-layer BSDF: clearcoat + metallic conductor + dielectric specular + Lambertian diffuse
- Emission support
- Full pdfRev for BDPT MIS
- Parameters: base_color, metalness, roughness, specular, specular_color, specular_IOR, coat, coat_roughness, emission, emission_color

### DomeLight (`src/shading/lights/DomeLight.h` + `.cpp`)
- Loads equirectangular EXR/HDR via OIIO (`read_image(0, 0, 0, channels, TypeDesc::FLOAT, ...)`)
- 2D piecewise-constant importance sampling: Distribution1D (CDF-based, binary search)
- Weighted by `luminance * sin(theta)` for equirectangular warp correction
- `sample()`, `pdf()`, `sampleLe()`, `Le()` all implemented
- Falls back to 1×1 grey pixel if image path is empty or unreadable
- CLI: `--env path.exr --env-intensity 1.0`

### envLight integration
- `SceneView::envLight` (new field) — the infinite/dome ILight pointer
- PathIntegrator uses `envLight->Le({}, {}, direction)` when rays miss geometry
- BDPTIntegrator `traceCameraSubpath` adds an environment vertex (type=Light, isInfinite=true) when camera rays miss, so `(s=0, t)` strategies capture env light
- RenderSession sets `scene.envLight` when DomeLight is created

### OslMaterial (`src/shading/OslMaterial.h`)
- Full OSL adapter implementation when `ANACAPA_ENABLE_OSL=ON`
- OslRendererServices (texture/trace/getattribute callbacks)
- OslShadingSystem singleton
- Assertion stub when OSL not compiled in
- OSL not installed on this machine; install with: `brew install open-shading-language` (if/when available)

## Phase 3 (USD) — COMPLETE (was done before Phase 2)

Full UsdGeomMesh, UsdLuxLight, UsdGeomCamera, UsdShadeMaterial loader.
Built by default (USD found at ~/usd). Flag: `ANACAPA_ENABLE_USD`.

## What's Built (Phase 5 — COMPLETE)

42/42 unit tests passing. Kitchen_set.usdc renders in ~3.3s at 640×480 @ 32 spp with full color.

- `src/shading/StandardSurface.h` — GGX multi-layer BSDF (clearcoat, metallic, dielectric, diffuse)
- `src/shading/lights/DomeLight.h/.cpp` — equirectangular HDRI, 2D piecewise-constant importance sampling
- `src/shading/OslMaterial.h` — OSL adapter stub (enabled with `ANACAPA_ENABLE_OSL=ON`)
- `src/scene/usd/USDLoader.cpp` — added DistantLight, DomeLight, `primvars:displayColor` fallback
- `tests/unit/test_phase4.cpp` — 11 Phase 5 unit tests

### primvars:displayColor fallback
When no `UsdShadeMaterial` binding exists (Blender USD export pattern), the loader reads `primvars:displayColor` vertex colors as `base_color` for a `LambertianMaterial`. Colors are quantized to 8-bit and cached to avoid duplicate materials.

## Next Steps

- **Phase 6:** Metal backend (macOS/Apple Silicon), CUDA+OptiX backend (NVIDIA Linux/Windows)

## Key Implementation Notes

### OIIO 3.x API
`read_image(subimage, miplevel, chbegin, chend, TypeDesc, ptr)` — NOT the 2.x `read_image(TypeDesc, ptr)`.

### DomeLight sampleLe invariant
`dot(s.dir, s.normal) > 0` — disk emitter: emitted photon travels in the same direction as the disk normal (both point inward toward scene).

### StandardSurface sampling
The `sample()` method picks a layer via `uComponent`, then calls `evalCombined()` to compute the full weighted PDF and f across all layers. This avoids fireflies from single-lobe sampling.

### Phase 5 prep
PathVertexBuffer::DeviceView provides raw pointer snapshot for GPU kernel hand-off. All SoA arrays are std::vector<T> for trivial migration to cudaMalloc/MTLBuffer.
