# Anacapa

A physically-based bidirectional path tracer written in C++20.

Named after Anacapa Island, part of California's Channel Islands.

## Features

- **Bidirectional path tracing (BDPT)** with multiple importance sampling and Veach MIS weights
- **OpenUSD scene loading** — geometry, materials, lights, camera, and animated transforms from `.usda`/`.usdc` files
- **Transformation motion blur** — time-sampled USD xformOps interpolated per ray; shutter interval read automatically from the stage's `startTimeCode`/`endTimeCode`
- **Intel OIDN denoising** — AI denoiser with albedo and normal auxiliary buffers, optional
- **Custom SAH BVH** — surface area heuristic build with 12-bucket binning, Möller–Trumbore traversal; time-expanded bounds for animated meshes
- **Custom thread pool** — tile-parallel rendering with `std::thread`, no external threading library
- **Scrambled Halton sampler** — low-discrepancy sampling up to 128 dimensions
- **Multi-layer EXR output** — beauty, denoised, albedo, and normals layers via OpenImageIO
- **GGX multi-layer BSDF** — MaterialX `standard_surface` (metallic conductor, dielectric specular, Lambertian diffuse, clearcoat)
- **HDRI dome lights** — equirectangular EXR/HDR with 2D piecewise-constant importance sampling
- **Depth of field** — thin-lens model; f-stop and focus distance from the USD camera or CLI override
- **OSL shading language** — optional (`-DANACAPA_ENABLE_OSL=ON`)
- **GPU-accelerated interactive rendering** — Metal backend (`--interactive`) for Apple Silicon (hardware ray tracing via `MTLAccelerationStructure`); CUDA+OptiX backend planned for NVIDIA
- **Zero compiled third-party dependencies** in the core renderer

## Dependencies

| Dependency | Required | Install |
|---|---|---|
| OpenImageIO | Yes | `brew install openimageio` |
| OpenUSD | No (`-DANACAPA_ENABLE_USD=ON`) | Build from source — see below |
| OpenImageDenoise | No (`-DANACAPA_ENABLE_OIDN=ON`) | `brew install open-image-denoise` |
| Open Shading Language | No (`-DANACAPA_ENABLE_OSL=ON`) | `brew install open-shading-language` |

Header-only dependencies (fetched automatically by CMake): spdlog, CLI11, GoogleTest.

## Building

CMake presets place compiled output in architecture-specific subdirectories (`build/Darwin-arm64`, `build/Linux-x86_64`, etc.) so binaries are never mixed with source files.

```bash
# List available presets
cmake --list-presets

# Prerequisites (macOS)
brew install openimageio

# Configure and build — macOS arm64, no optional features
cmake --preset macos-arm64
cmake --build build/Darwin-arm64 --parallel

# Build with USD + Metal (typical development setup on Apple Silicon)
# OpenUSD must be built from source using Pixar's build script:
#   python3 USD/build_scripts/build_usd.py ~/usd
cmake --preset macos-arm64-usd \
  -DUSD_ROOT=~/usd \
  -DCMAKE_PREFIX_PATH=~/usd
cmake --build build/Darwin-arm64 --parallel

# Build with Intel OIDN denoising
brew install open-image-denoise
cmake --preset macos-arm64 \
  -DANACAPA_ENABLE_OIDN=ON \
  -DOpenImageDenoise_DIR=/opt/homebrew/lib/cmake/OpenImageDenoise-2.4.1
cmake --build build/Darwin-arm64 --parallel

# Run tests
cd build/Darwin-arm64 && ctest --output-on-failure
```

## Usage

```bash
# Render the built-in Cornell box scene (BDPT, 64 spp)
./build/Darwin-arm64/anacapa -o images/render.exr

# Load a USD scene file
DYLD_LIBRARY_PATH=~/usd/lib \
./build/Darwin-arm64/anacapa --scene scenes/cornell_box.usda -o images/render.exr

# Render with motion blur (shutter read automatically from USD startTimeCode/endTimeCode)
DYLD_LIBRARY_PATH=~/usd/lib \
./build/Darwin-arm64/anacapa --scene scenes/cornell_box_motion.usda -o images/render.exr

# Override shutter interval explicitly
./build/Darwin-arm64/anacapa --scene scene.usda \
  --shutter-open 0 --shutter-close 1 -o images/render.exr

# Render with denoising
./build/Darwin-arm64/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --denoise

# Render with denoising + write albedo and normals layers to the EXR
./build/Darwin-arm64/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --denoise --write-aovs

# Render with depth of field (thin lens, f/4, focused at 5 units)
./build/Darwin-arm64/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --fstop 4 --focus-distance 5

# Fast GPU preview on Apple Silicon
./build/Darwin-arm64/anacapa --scene scenes/cornell_box.usda \
  --interactive --width 800 --height 800 --spp 64 -o preview.exr

# Full options
./build/Darwin-arm64/anacapa \
  --scene          scenes/cornell_box.usda \
  --camera         /World/RenderCam       \
  --integrator     bdpt                   \
  --width          800                    \
  --height         800                    \
  --spp            256                    \
  --depth          8                      \
  --fstop          2.8                    \
  --focus-distance 10                     \
  --shutter-open   0                      \
  --shutter-close  1                      \
  --output         images/render.exr      \
  --denoise                               \
  --write-aovs                            \
  --interactive

./build/Darwin-arm64/anacapa --help
```

### Options reference

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `out.exr` | Output EXR path |
| `-W, --width` | `800` | Image width in pixels |
| `-H, --height` | `800` | Image height in pixels |
| `-s, --spp` | `64` | Samples per pixel |
| `-d, --depth` | `8` | Maximum path depth |
| `-t, --threads` | `0` (auto) | Thread count; 0 = hardware concurrency |
| `--tile-size` | `64` | Tile size in pixels |
| `--integrator` | `bdpt` | `bdpt` or `path` |
| `--scene` | — | USD/USDA/USDC scene file |
| `--camera` | — | USD prim path of camera (e.g. `/World/RenderCam`) |
| `--env` | — | Equirectangular HDRI environment map (EXR or HDR) |
| `--env-intensity` | `1.0` | Intensity multiplier for the environment map |
| `--fstop` | `0` | Lens f-stop; enables DoF when combined with `--focus-distance` |
| `--focus-distance` | `0` | Distance to focal plane in scene units |
| `--shutter-open` | `0` | Shutter open override (0 = `startTimeCode`) |
| `--shutter-close` | `0` | Shutter close override (1 = `endTimeCode`); leave at 0 to use the scene's time range |
| `--denoise` | off | Run Intel OIDN denoiser after rendering |
| `--write-aovs` | off | Include albedo and normals layers in the output EXR |
| `--interactive` | off | Use Metal GPU backend for fast preview (Apple Silicon) |

`--spp`: 16–32 for quick composition checks; 256+ for final renders.

`--fstop` and `--focus-distance` both must be provided to enable depth of field. They override the USD camera values when present; if neither is set the camera falls back to pinhole.

`--shutter-open`/`--shutter-close` override the motion blur shutter. When omitted, the shutter is derived from the stage's `startTimeCode`, `endTimeCode`, and `timeCodesPerSecond` automatically. Set both to 0 to disable motion blur on an animated scene.

## Motion Blur

Transformation motion blur is driven by time-sampled `xformOp` attributes on USD prims. A minimal animated scene:

```usda
#usda 1.0
(
    startTimeCode = 1
    endTimeCode   = 24
)

def Xform "MovingObject" {
    double3 xformOp:translate.timeSamples = {
        1:  (-0.5, 0, 0),
        24: ( 0.5, 0, 0)
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Box" { ... }
}
```

No CLI flags are needed — the shutter interval is inferred from `startTimeCode`/`endTimeCode` and `timeCodesPerSecond`. Motion blur activates automatically when animated meshes are detected.

See [scenes/cornell_box_motion.usda](scenes/cornell_box_motion.usda) for a complete example.

### Implementation notes

- Animated mesh vertices are stored in **object space**. The BVH stores time-expanded world-space bounds (union of bounds at start and end time) to correctly enclose all motion.
- At intersection time, `objectToWorld` is interpolated as `lerp(M_start, M_end, ray.time)` and inverted to transform the ray into object space. The lerped forward matrix is inverted — lerping the inverse matrices directly is incorrect when rotation is present.
- `ray.time` is sampled once per path on the primary camera ray and propagated to all secondary rays and shadow rays to keep the scene temporally consistent across a path.

## Scene Format

Scenes are authored in OpenUSD (`.usda` text, `.usdc` binary, or `.usd` auto-detect). The loader supports:

| USD Prim | Anacapa |
|---|---|
| `UsdGeomMesh` | Triangulated mesh — static (world-space baked) or animated (object-space + transform pair) |
| `UsdShadeMaterial` + `UsdPreviewSurface` | `LambertianMaterial` or `StandardSurfaceMaterial` |
| `primvars:displayColor` | `LambertianMaterial` (Blender USD export fallback) |
| `UsdLuxRectLight` | `AreaLight` |
| `UsdLuxSphereLight` | `AreaLight` (approximated) |
| `UsdLuxDistantLight` | `DirectionalLight` |
| `UsdLuxDomeLight` | `DomeLight` (equirectangular HDRI) |
| `UsdGeomCamera` | Pinhole or thin-lens camera |
| `UsdRenderSettings` | Declares the render camera via `.camera` relationship |

When no `UsdShadeMaterial` binding is present the loader falls back to `primvars:displayColor`. Static mesh positions and normals are baked into world space at load time; animated mesh positions are kept in object space and transformed per-ray.

### Camera selection

When a scene contains multiple cameras the loader resolves which one to use in priority order:

1. **`--camera /Prim/Path`** — explicit prim path on the command line
2. **`UsdRenderSettings.camera`** — relationship declared in the scene's render settings
3. **First `UsdGeomCamera` found** — fallback

All cameras found are logged so you can see available prim paths without any extra tooling:

```
[info] USDLoader: 2 camera(s) found in scene:
[info]   /World/RenderCam
[info]   /World/CloseupCam
[info] USDLoader: multiple cameras found; using first '/World/RenderCam'.
       Use --camera <path> to select another.
```

## EXR Output

Output is a linear HDR EXR file. Apply exposure or tone mapping in your viewer.

When `--write-aovs` is used the EXR contains additional channel layers:

| Channels | Contents |
|---|---|
| `R, G, B` | Raw beauty (full Monte Carlo integral) |
| `denoised.R/G/B` | OIDN-denoised beauty (requires `--denoise`) |
| `albedo.R/G/B` | First-hit diffuse reflectance |
| `normals.R/G/B` | First-hit world-space normals (signed, unit length) |

To extract individual layers with `oiiotool` (installed with OpenImageIO):

```bash
# Extract raw beauty
oiiotool out.exr --ch "R,G,B" -o beauty.exr

# Extract denoised beauty
oiiotool out.exr --ch "denoised.R,denoised.G,denoised.B" --chnames "R,G,B" -o denoised.exr

# Extract albedo
oiiotool out.exr --ch "albedo.R,albedo.G,albedo.B" --chnames "R,G,B" -o albedo.exr

# Extract normals (remap [-1,1] → [0,1] for viewing)
oiiotool out.exr --ch "normals.R,normals.G,normals.B" --chnames "R,G,B" \
  --addc 1,1,1 --mulc 0.5,0.5,0.5 -o normals.exr

# Apply gamma for display
oiiotool beauty.exr --powc 0.45,0.45,0.45,1.0 -o beauty.png
```

For interactive layer switching:

- **[mrViewer](https://mrviewer.sourceforge.io)** — macOS/Linux/Windows, designed for VFX
- **[DJV](https://darbyjohnston.github.io/DJV/)** — cross-platform, lightweight

## Interactive (GPU) Mode

Pass `--interactive` to use the Metal GPU backend instead of the CPU path tracer. Intended for fast iteration — checking composition, previewing lighting — where speed matters more than accuracy.

The GPU backend requires a build with `ANACAPA_ENABLE_METAL` (enabled via the `macos-arm64-usd` preset). If Metal cannot be initialised the renderer falls back to CPU with a warning.

### Performance

Measured on Apple M3 Pro, 400×400 @ 64 spp:

| Scene | CPU (BDPT, 11 threads) | GPU (Metal) | Speedup |
|---|---|---|---|
| Cornell box (36 tris) | ~1168 ms | ~301 ms | ~3.9× |

### Simplifications vs. the CPU renderer

| Feature | CPU renderer | GPU (`--interactive`) |
|---|---|---|
| Integrator | BDPT with MIS | Unidirectional path tracing |
| Direct lighting | Full MIS: BSDF + light sampling | Single random light sample, no MIS |
| Light types | Rect, sphere, directional, HDRI dome | Rect and directional only |
| Material model | Full GGX `standard_surface` | Lambertian + GGX (roughness fixed at 0.5) |
| Caustics | Yes | No |
| Motion blur | Yes | No |
| AOVs / denoising | Supported | Not supported |

## Architecture

```
include/anacapa/
  core/         Types (Vec3f, Ray, Spectrum, BBox3f), ArenaAllocator
  accel/        IAccelerationStructure, GeometryPool (MeshDesc with motion fields)
  shading/      IMaterial, ILight, ShadingContext
  sampling/     ISampler, SamplerState
  film/         Film, TileBuffer, DenoiseOptions
  integrator/   IIntegrator, Camera (shutter interval), SceneView
  scene/        SceneLoader (LoadedScene, shutter fields)

src/
  accel/        BVHBackend — SAH BVH; time-expanded bounds + object-space ray transform for animated meshes
  shading/      Lambertian, StandardSurface, EmissiveMaterial
  shading/lights/  AreaLight, DirectionalLight, DomeLight (HDRI)
  sampling/     PCGRng, HaltonSampler
  integrator/   PathIntegrator (reference), BDPTIntegrator
                LightSampler (Vose alias table), MISWeight
  film/         Film — atomic accumulation, OIDN denoising, multi-layer EXR
  render/       ThreadPool, RenderSession — shutter wiring from scene or CLI
  scene/usd/    USDLoader — geometry, lights, materials, camera, animated transforms
  gpu/metal/    MetalContext, MetalAccelStructure, MetalPathIntegrator (macOS only)

scenes/
  cornell_box.usda        Static Cornell box reference scene
  cornell_box_motion.usda Cornell box with animated ShortBlock (motion blur test)
```

All memory-owning data structures use SoA (Structure-of-Arrays) layout to enable zero-copy migration to GPU backends.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | Complete | CPU path tracer, custom BVH, Halton sampler, EXR output |
| 2 | Complete | Bidirectional path tracing with MIS, alias-table light sampler |
| 3 | Complete | Intel OIDN denoising, albedo/normal AOVs, multi-layer EXR |
| 4 | Complete | OpenUSD scene loading (geometry, materials, lights, camera) |
| 5 | Complete | GGX `standard_surface` BSDF, HDRI dome lights, depth of field |
| 6 | Complete | Transformation motion blur (time-sampled USD xforms, temporal BVH, per-ray time) |
| 7 | In Progress | Metal backend (Apple Silicon), CUDA+OptiX backend (NVIDIA) |

## License

MIT — see [LICENSE](LICENSE).
