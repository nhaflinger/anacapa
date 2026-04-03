# Anacapa

A physically-based bidirectional path tracer written in C++20.

Named after Anacapa Island, part of California's Channel Islands.

## Features

- **Bidirectional path tracing (BDPT)** with multiple importance sampling and Veach MIS weights
- **OpenUSD scene loading** — geometry, materials, lights, and camera from `.usda`/`.usdc` files
- **Intel OIDN denoising** — AI denoiser with albedo and normal auxiliary buffers, optional
- **Custom SAH BVH** — surface area heuristic build with 12-bucket binning, Möller–Trumbore traversal
- **Custom thread pool** — tile-parallel rendering with `std::thread`, no external threading library
- **Scrambled Halton sampler** — low-discrepancy sampling up to 128 dimensions
- **Multi-layer EXR output** — beauty, denoised, albedo, and normals layers via OpenImageIO
- **GGX multi-layer BSDF** — MaterialX `standard_surface` (metallic conductor, dielectric specular, Lambertian diffuse, clearcoat)
- **HDRI dome lights** — equirectangular EXR/HDR with 2D piecewise-constant importance sampling
- **OSL shading language** — optional (`-DANACAPA_ENABLE_OSL=ON`)
- **GPU-accelerated interactive rendering** — Metal backend (`--interactive`) for Apple Silicon (hardware ray tracing via `MTLAccelerationStructure`); CUDA+OptiX backend planned for NVIDIA
- **Zero compiled third-party dependencies** in the core renderer

## Dependencies

| Dependency | Required | Install |
|---|---|---|
| OpenImageIO | Yes | `brew install openimageio` |
| OpenUSD | No (`-DANACAPA_ENABLE_USD=ON`) | Build from source — see below |
| OpenImageDenoise | No (`-DANACAPA_ENABLE_OIDN=ON`) | `brew install open-image-denoise` |
| Open Shading Language | No (`-DANACAPA_ENABLE_OSL=ON`) | `brew install open-shading-language` (if available) |

Header-only dependencies (fetched automatically by CMake): spdlog, CLI11, GoogleTest.

## Building

```bash
# Prerequisites (macOS)
brew install openimageio

# Configure and build (no USD, no OIDN)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Build with Intel OIDN denoising support
brew install open-image-denoise
cmake -B build -DCMAKE_BUILD_TYPE=Release -DANACAPA_ENABLE_OIDN=ON \
  -DOpenImageDenoise_DIR=/opt/homebrew/lib/cmake/OpenImageDenoise-2.4.1
cmake --build build --parallel

# Build with OpenUSD support
# OpenUSD must be built from source using Pixar's build script:
#   python3 USD/build_scripts/build_usd.py ~/usd
# Then configure with:
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DANACAPA_ENABLE_USD=ON -DUSD_ROOT=~/usd \
  -DANACAPA_ENABLE_OIDN=ON \
  -DOpenImageDenoise_DIR=/opt/homebrew/lib/cmake/OpenImageDenoise-2.4.1
cmake --build build --parallel

# Run tests
cd build && ctest --output-on-failure
```

## Usage

```bash
# Render the built-in Cornell box scene (BDPT, 64 spp)
./build/anacapa -o images/render.exr

# Load a USD scene file
DYLD_LIBRARY_PATH=~/usd/lib \
./build/anacapa --scene scenes/cornell_box.usda -o images/render.exr

# Render with denoising
./build/anacapa --scene scenes/cornell_box.usda -o images/render.exr --denoise

# Render with denoising + write albedo and normals layers to the EXR
./build/anacapa --scene scenes/cornell_box.usda -o images/render.exr --denoise --write-aovs

# Render with depth of field (thin lens, f/4, focused at 5 units)
./build/anacapa --scene scenes/cornell_box.usda -o images/render.exr \
  --fstop 4 --focus-distance 5

# Fast GPU preview on Apple Silicon
./build/anacapa --scene scenes/kitchen_set.usdc --interactive \
  --width 800 --height 800 --spp 64 -o preview.exr

# Full options
./build/anacapa \
  --scene          scenes/cornell_box.usda \
  --camera         /World/RenderCam \
  --integrator     bdpt \
  --width          800  \
  --height         800  \
  --spp            256  \
  --depth          8    \
  --fstop          2.8  \
  --focus-distance 10   \
  --output         images/render.exr \
  --denoise             \
  --write-aovs          \
  --interactive

./build/anacapa --help
```

`--spp` controls quality vs. speed. Lower values (16–32) are useful for quick composition checks;
256+ is recommended for final renders. Defaults to 64.

`--fstop` and `--focus-distance` enable depth of field using a thin lens model. Both must be
provided together — either on the command line or via `fStop`/`focusDistance` attributes on the
`UsdGeomCamera` in the scene file. Command-line values take priority over USD values. If neither
is present the camera falls back to pinhole (infinite depth of field). Typical f-stops: 1.4
(very shallow DoF), 2.8 (moderate), 8 (near-infinite).

`--interactive` switches to the Metal GPU backend, which can be significantly faster — especially
on complex scenes — at the cost of some accuracy. See [Interactive (GPU) mode](#interactive-gpu-mode) below.
All three flags are optional and independent of each other.

### Interactive (GPU) mode

Pass `--interactive` to use the Metal GPU backend instead of the CPU path tracer.
This is intended for fast iteration — loading a scene, checking composition, or
previewing material changes — where render time matters more than accuracy.

```bash
# Fast preview render on Apple Silicon (Metal backend)
./build/anacapa --scene scenes/kitchen_set.usdc --interactive \
  --width 800 --height 800 --spp 64 -o preview.exr
```

The GPU backend requires a build with `ANACAPA_ENABLE_METAL` (enabled automatically
on macOS when Xcode is present). If the Metal device cannot be initialised the
renderer falls back to the CPU path tracer with a warning:

```
[warn] --interactive: Metal backend unavailable, falling back to CPU path tracer
```

The output is a linear HDR EXR file. Apply exposure compensation in your viewer — the scene uses physically-based light units.

When `--write-aovs` is used the EXR contains four layer groups:

| Channels | Contents |
|---|---|
| `R, G, B` | Raw beauty (full Monte Carlo integral) |
| `denoised.R/G/B` | OIDN-denoised beauty |
| `albedo.R/G/B` | First-hit diffuse reflectance (denoising hint) |
| `normals.R/G/B` | First-hit world-space normals (denoising hint) |

### Viewing EXR output

macOS Preview does not support multi-layer EXR or channel selection. To compare
the raw beauty against the denoised result, use `oiiotool` (installed with
OpenImageIO) to extract individual layers:

```bash
# Extract just the denoised beauty as a standard RGB EXR
oiiotool out.exr --ch "denoised.R,denoised.G,denoised.B" --chnames "R,G,B" -o denoised.exr

# Extract raw beauty
oiiotool out.exr --ch "R,G,B" -o beauty.exr
```

For interactive layer switching use one of these free viewers:

- **[mrViewer](https://mrviewer.sourceforge.io)** — macOS/Linux/Windows, designed for VFX
- **[DJV](https://darbyjohnston.github.io/DJV/)** — cross-platform, lightweight

## Scene Format

Scenes are authored in OpenUSD (`.usda` text, `.usdc` binary, or `.usd` auto-detect). The loader supports:

| USD Prim | Anacapa |
|---|---|
| `UsdGeomMesh` | Triangulated mesh, world-space baked |
| `UsdShadeMaterial` + `UsdPreviewSurface` | `LambertianMaterial` or `EmissiveMaterial` |
| `primvars:displayColor` | `LambertianMaterial` (Blender USD export fallback) |
| `UsdLuxRectLight` | `AreaLight` |
| `UsdLuxSphereLight` | `AreaLight` (approximated) |
| `UsdLuxDistantLight` | `DirectionalLight` |
| `UsdLuxDomeLight` | `DomeLight` (equirectangular HDRI) |
| `UsdGeomCamera` | Pinhole camera (focal length + aperture → FOV) |
| `UsdRenderSettings` | Declares the render camera via `.camera` relationship |

When no `UsdShadeMaterial` binding is present the loader falls back to `primvars:displayColor`, which is the default export path from Blender.
All mesh positions and normals are baked into world space at load time.

### Camera selection

When a scene contains multiple cameras the loader resolves which one to use in
priority order:

1. **`--camera /Prim/Path`** — explicit prim path passed on the command line
2. **`UsdRenderSettings.camera`** — the relationship declared in the scene's render settings
3. **First `UsdGeomCamera` found** — fallback when no other selection is made

Every render logs all cameras present in the file so you can see the available
prim paths without any extra tooling:

```
[info] USDLoader: 2 camera(s) found in scene:
[info]   /World/RenderCam
[info]   /World/CloseupCam
[info] USDLoader: multiple cameras found; using first '/World/RenderCam'.
       Use --camera <path> to select another.
```

The built-in scene is at [scenes/cornell_box.usda](scenes/cornell_box.usda).

## GPU-accelerated rendering

The Metal backend (`--interactive`) uses Apple's hardware ray-tracing API to accelerate preview renders. It is a simplified, single-bounce-per-pass megakernel designed for speed rather than accuracy.

### Performance

Measured on Apple M3 Pro, 400×400 @ 64 spp:

| Scene | CPU (BDPT, 11 threads) | GPU (Metal) | Speedup |
|---|---|---|---|
| Cornell box (36 tris) | ~1168 ms | ~301 ms | ~3.9× |
| Blender kitchen set | — | — | ~17× |

Speedup scales with scene complexity — the GPU's parallelism becomes more effective as triangle and material counts grow.

### Simplifications vs. the CPU renderer

The GPU backend trades correctness for speed in several areas:

| Feature | CPU renderer | GPU (`--interactive`) |
|---|---|---|
| Integrator | Bidirectional path tracing (BDPT) with MIS | Unidirectional path tracing (megakernel) |
| Direct lighting | Full MIS: BSDF + light sampling | Single random light sample per bounce, no MIS |
| Indirect lighting | Full multi-bounce with Russian roulette | Up to `--depth` bounces, Russian roulette after bounce 3 |
| Light types | Rect, sphere, directional, HDRI dome | Rect and directional only; dome lights ignored |
| Material model | Full GGX `standard_surface` (multi-lobe, clearcoat) | Lambertian, GGX (roughness/metalness), emissive |
| GGX parameters | Full `standard_surface` introspection | Roughness fixed at 0.5 for glossy materials |
| Caustics | Yes (via light subpaths) | No |
| AOVs | Albedo and normals written | Not written |
| Denoising | Supported (`--denoise`) | Not supported |

These simplifications are intentional — the goal is interactive feedback, not a reference render. Use the CPU renderer (default) for final-quality output.

### Architecture

The Metal backend is built only on macOS and is compiled as a separate set of Objective-C++ sources:

```
src/gpu/metal/
  MetalContext.{h,mm}           Device + command queue + .metallib loader
  MetalBuffer.{h,mm}            RAII MTLBuffer wrapper (MTLStorageModeShared)
  MetalAccelStructure.{h,mm}    BLAS-per-mesh + TLAS build from GeometryPool
  MetalPathIntegrator.{h,mm}    IIntegrator impl — prepare() and renderTile()
  shaders/
    SharedTypes.h               POD structs shared between C++ and MSL (packed_float3)
    Shade.metal                 Megakernel: ray generation, intersection, BSDF, direct lighting
    RayGen.metal                (reserved for future wavefront split)
```

The shaders are compiled to a `.metallib` at build time via `xcrun metal` / `metallib` and
loaded at runtime. The path is baked in as `ANACAPA_METALLIB_PATH`.

Key design points:
- **PIMPL everywhere** — all `id<MTL*>` types are hidden behind `struct Impl` so C++ headers stay ObjC-free.
- **`packed_float3`** — shared structs use `packed_float3` on the MSL side to match the 12-byte C++ layout; `float3` in constant address space is 16-byte aligned and would break the ABI.
- **`useResource` per BLAS** — Metal requires explicit resource declarations for all acceleration structures accessed indirectly through the TLAS; omitting them causes silent no-hit.
- **Tile-sized dispatch** — each `renderTile()` call dispatches only the tile's pixel region (not the full image), keeping GPU work proportional to tile area.

## Architecture

```
include/anacapa/
  core/         Types (Vec3f, Ray, Spectrum, BBox3f), ArenaAllocator
  accel/        IAccelerationStructure, GeometryPool
  shading/      IMaterial, ILight, ShadingContext
  sampling/     ISampler, SamplerState
  film/         Film, TileBuffer, DenoiseOptions
  integrator/   IIntegrator, Camera, SceneView
  scene/        SceneLoader (LoadedScene)

src/
  accel/        BVHBackend (custom SAH BVH)
  shading/      Lambertian, EmissiveMaterial, AreaLight
  sampling/     PCGRng, HaltonSampler
  integrator/   PathIntegrator (reference), BDPTIntegrator
                LightSampler (Vose alias table), MISWeight
  film/         Film (atomic accumulation, OIDN denoising, multi-layer EXR)
  render/       ThreadPool, RenderSession
  shading/lights/ AreaLight, DirectionalLight, DomeLight (HDRI)
  scene/usd/    USDLoader (geometry, lights, materials, camera)
  gpu/metal/    MetalContext, MetalAccelStructure, MetalPathIntegrator (macOS only)

scenes/
  cornell_box.usda   built-in Cornell box reference scene
```

All memory-owning data structures use SoA (Structure-of-Arrays) layout to enable zero-copy migration to GPU backends in Phase 6.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | Complete | CPU path tracer, custom BVH, Halton sampler, EXR output |
| 2 | Complete | Bidirectional path tracing with MIS, alias-table light sampler |
| 3 | Complete | Intel OIDN denoising, albedo/normal AOVs, multi-layer EXR |
| 4 | Complete | OpenUSD scene loading (geometry, materials, lights, camera) |
| 5 | Complete | GGX `standard_surface` BSDF, HDRI dome lights, OSL adapter, `primvars:displayColor` support |
| 6 | In Progress | Metal backend (Apple Silicon), CUDA+OptiX backend (NVIDIA) |

## License

MIT — see [LICENSE](LICENSE).
