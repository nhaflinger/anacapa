# Anacapa

A physically-based path tracer written in C++20.

Named after Anacapa Island, part of California's Channel Islands.

## Features

- **Unidirectional path tracing (Path)** with next event estimation
- **Bidirectional path tracing (BDPT)** with multiple importance sampling and Veach MIS weights
- **OpenUSD scene loading** — geometry, materials, lights, camera, and animated transforms from `.usda`/`.usdc` files
- **Transformation motion blur** — multi-sample USD xformOps with piecewise-linear interpolation per ray; arbitrary number of time samples supported for curved blur streaks; shutter interval read automatically from the camera prim's `shutter:open`/`shutter:close`, falling back to stage `startTimeCode`/`endTimeCode`
- **Hair and fur rendering** — RenderMan-style ray-ribbon intersection with recursive de Casteljau subdivision; Marschner 2003 BSDF (R/TT/TRT lobes, exact Fresnel, Beer's law absorption, PBRT v4 variance remapping)
- **Alembic curve loader** — loads hair/fur strands from `.abc` files directly into the ray tracer; supports cubic Bézier, B-spline, Catmull-Rom, and linear curves with automatic basis conversion to endpoint-sharing Bézier
- **Intel OIDN denoising** — AI denoiser with albedo and normal auxiliary buffers, enabled by default
- **Custom SAH BVH** — surface area heuristic build with 12-bucket binning, Möller–Trumbore traversal; time-expanded bounds for animated meshes
- **Custom thread pool** — tile-parallel rendering with `std::thread`, no external threading library
- **Scrambled Halton sampler** — low-discrepancy sampling up to 128 dimensions
- **Multi-layer EXR output** — beauty, denoised, albedo, and normals layers via OpenImageIO
- **GGX multi-layer BSDF** — MaterialX `standard_surface` (metallic conductor, dielectric specular, Lambertian diffuse, clearcoat)
- **HDRI dome lights** — equirectangular EXR/HDR with 2D piecewise-constant importance sampling
- **Depth of field** — thin-lens model; f-stop and focus distance from the USD camera or CLI override
- **OSL shading language** — optional (`-DANACAPA_ENABLE_OSL=ON`)
- **GPU-accelerated interactive rendering** — Metal backend (`--interactive`) for Apple Silicon (hardware ray tracing via `MTLAccelerationStructure`); pure-CUDA backend for NVIDIA GPUs (WSL2 and Linux)
- **Progressive render viewer** — SDL2 + Dear ImGui live preview with 8 comparison slots and real-time color controls (exposure, contrast, saturation, temperature)
- **Blender addon** — one-click render from Blender; exports USD + Alembic hair automatically, launches Anacapa, and loads the result back into Blender's image editor with a live progressive preview
- **Zero compiled third-party dependencies** in the core renderer

## Dependencies

| Dependency | Required | Install |
|---|---|---|
| OpenImageIO | Yes | `brew install openimageio` |
| OpenUSD | No (`-DANACAPA_ENABLE_USD=ON`) | Build from source — see below |
| Alembic | No (`-DANACAPA_ENABLE_ALEMBIC=ON`, default ON) | `brew install alembic` |
| OpenImageDenoise | Yes (all presets) | `brew install open-image-denoise` |
| Open Shading Language | No (`-DANACAPA_ENABLE_OSL=ON`) | `brew install open-shading-language` |

Header-only dependencies (fetched automatically by CMake): spdlog, CLI11, GoogleTest.

Viewer dependencies (fetched automatically when `ANACAPA_ENABLE_VIEWER=ON`): SDL2, Dear ImGui, glad, stb_image.

## Building

CMake presets place compiled output in OS-specific subdirectories (`build/Darwin`, `build/Linux`, etc.) so binaries are never mixed with source files.

```bash
# List available presets
cmake --list-presets

# Prerequisites (macOS) — OIDN is required by all presets
brew install openimageio open-image-denoise alembic

# Configure and build — macOS arm64
cmake --preset macos-arm64
cmake --build build/Darwin --parallel

# Build with USD + Metal (typical development setup on Apple Silicon)
# OpenUSD must be built from source using Pixar's build script:
#   python3 USD/build_scripts/build_usd.py ~/usd
cmake --preset macos-arm64-usd \
  -DUSD_ROOT=~/usd \
  -DCMAKE_PREFIX_PATH=~/usd
cmake --build build/Darwin --parallel

# Run tests
cd build/Darwin && ctest --output-on-failure
```

> **Note (macOS):** The Metal shader compiler requires a full Xcode installation, not just Command Line Tools. If `xcode-select -p` points at `/Library/Developer/CommandLineTools`, CMake will locate the compiler directly inside Xcode.app automatically. No manual intervention is needed as long as Xcode is installed.

## Usage

```bash
# Render the built-in Cornell box scene (BDPT, 64 spp)
./build/Darwin/anacapa -o images/render.exr

# Load a USD scene file
DYLD_LIBRARY_PATH=~/usd/lib \
./build/Darwin/anacapa --scene scenes/cornell_box.usda -o images/render.exr

# Render with motion blur (shutter read automatically from USD startTimeCode/endTimeCode)
DYLD_LIBRARY_PATH=~/usd/lib \
./build/Darwin/anacapa --scene scenes/cornell_box_motion.usda -o images/render.exr

# Override shutter interval explicitly
./build/Darwin/anacapa --scene scene.usda \
  --shutter-open 0 --shutter-close 1 -o images/render.exr

# Render with denoising
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --denoise

# Render with denoising + write albedo and normals layers to the EXR
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --denoise --write-aovs

# Render with depth of field (thin lens, f/4, focused at 5 units)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --fstop 4 --focus-distance 5

# Write a tone-mapped PNG alongside the EXR (no separate oiiotool step needed)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --png images/render.png

# PNG with exposure adjustment (+1 stop)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --png images/render.png --exposure 1.0

# Fast GPU preview on Apple Silicon
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  --interactive --width 800 --height 800 --spp 64 -o preview.exr

# Render a character with USD geometry and Alembic hair/fur
DYLD_LIBRARY_PATH=~/usd/lib \
./build/Darwin/anacapa --scene scenes/character.usda \
  --curves hair_sim.abc -o images/render.exr

# Isolate material issues (replace all lights with a single white directional)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --override-lights

# Isolate lighting issues (replace all materials with white Lambertian)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --override-materials

# Soft shadows from directional lights (2° angular radius, ~4× the sun)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --png images/render.png \
  --light-angle 2.0 --spp 256

# Adaptive sampling: 256 spp total, base pass at 64 spp, remaining 192 concentrated on high-variance tiles
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --png images/render.png \
  --spp 256 --adaptive --adaptive-base-spp 64

# Adaptive + firefly clamping (recommended for BDPT with complex lighting)
./build/Darwin/anacapa --scene scenes/cornell_box.usda \
  -o images/render.exr --png images/render.png \
  --integrator bdpt --spp 512 --adaptive --firefly-clamp 10

# Full options
./build/Darwin/anacapa \
  --scene            scenes/character.usda   \
  --curves           hair_sim.abc            \
  --camera           /World/RenderCam        \
  --integrator       bdpt                    \
  --width            800                     \
  --height           800                     \
  --spp              256                     \
  --depth            8                       \
  --fstop            2.8                     \
  --focus-distance   10                      \
  --shutter-open     0                       \
  --shutter-close    1                       \
  --output           images/render.exr       \
  --png              images/render.png       \
  --exposure         0.5                     \
  --denoise                                  \
  --write-aovs                               \
  --override-lights                          \
  --override-materials                       \
  --interactive                              \
  --firefly-clamp    10                      \
  --light-angle      2.0                     \
  --adaptive                                 \
  --adaptive-base-spp 0

./build/Darwin/anacapa --help
```

### Options reference

| Flag | Default | Description |
|---|---|---|
| `-o, --output` | `out.exr` | Output EXR path |
| `-W, --width` | `800` | Image width in pixels |
| `-H, --height` | `800` | Image height in pixels |
| `-s, --spp` | `64` | Samples per pixel |
| `-d, --depth` | `8` | Maximum number of light bounces (path depth). Lower values are faster but will miss indirect lighting and caustics through glass |
| `-t, --threads` | `0` (auto) | Thread count; 0 = hardware concurrency |
| `--tile-size` | `64` | Tile size in pixels |
| `--integrator` | `path` | `bdpt` or `path` |
| `--firefly-clamp` | `10` | BDPT: max luminance per strategy contribution; suppresses bright outliers. `0` = off |
| `--light-angle` | `0` | Angular radius for directional lights in degrees. Turns hard point sources into soft area lights with penumbras. `0` = hard shadows, `0.27` = sun, `2–5` = soft |
| `--adaptive` | off | Enable adaptive sampling: base pass + extra samples concentrated on high-variance tiles |
| `--adaptive-base-spp` | `0` | Adaptive base-pass SPP (`0` = auto: `spp/4`, minimum 16) |
| `--scene` | — | USD/USDA/USDC scene file |
| `--curves` | — | Alembic `.abc` file containing hair/fur curves |
| `--camera` | — | USD prim path of camera (e.g. `/World/RenderCam`) |
| `--env` | — | Equirectangular HDRI environment map (EXR or HDR) |
| `--env-intensity` | `1.0` | Intensity multiplier for the environment map |
| `--fstop` | `0` | Lens f-stop; enables DoF when combined with `--focus-distance` |
| `--focus-distance` | `0` | Distance to focal plane in scene units |
| `--shutter-open` | `0` | Shutter open override (0 = `startTimeCode`) |
| `--shutter-close` | `0` | Shutter close override (1 = `endTimeCode`); leave at 0 to use the scene's time range |
| `--denoise` | off | Run Intel OIDN denoiser after rendering |
| `--write-aovs` | off | Include albedo and normals layers in the output EXR |
| `--interactive` | off | Use GPU backend for fast preview — Metal on Apple Silicon, CUDA on NVIDIA |
| `--png` | — | Write ACES-tonemapped sRGB PNG alongside the EXR |
| `--exposure` | `0` | EV exposure adjustment for `--png` output (stops; positive = brighter) |
| `--override-lights` | off | Replace all scene lights with a single white directional light (isolate material issues) |
| `--override-materials` | off | Replace all scene materials with white Lambertian (isolate lighting issues) |

`--spp`: 16–32 for quick composition checks; 256+ for final renders.

`--fstop` and `--focus-distance` both must be provided to enable depth of field. They override the USD camera values when present; if neither is set the camera falls back to pinhole.

`--shutter-open`/`--shutter-close` override the motion blur shutter. When omitted, the shutter is derived from the stage's `startTimeCode`, `endTimeCode`, and `timeCodesPerSecond` automatically. Set both to 0 to disable motion blur on an animated scene.

## Choosing an Integrator

| Lighting scenario | Best integrator | Notes |
|---|---|---|
| Diffuse indirect lighting (rooms, interiors) | `bdpt` | Light and camera subpaths connect efficiently; much lower variance than path |
| Small / distant light sources | `bdpt` | Difficult for path tracer's random scatter to hit; BDPT connects directly |
| Glass and specular caustics | `bdpt` | BDPT can connect a light subpath through glass to the camera; path tracer cannot |
| Emissive surfaces (large area lights) | `path` | Large lights are easy for NEE to hit; BDPT connection overhead not worth it |
| Outdoor / HDRI dome lighting | `path` | Sky covers a wide solid angle; random scatter hits it reliably |
| Heavily occluded scenes (corners, crevices) | `bdpt` + `--adaptive` | BDPT handles indirect light better; adaptive concentrates samples where variance is highest |
| Fast preview / interactive | `path` + `--interactive` | GPU backend only supports path tracing |
| Unknown / general | `bdpt` | Default; handles more scenarios well at the cost of slightly higher per-sample overhead |

Both integrators support `--adaptive` sampling. As a rule of thumb: if your scene has difficult light transport (small lights, glass, deep indirection), prefer `bdpt`. If your scene is large and open with broad lighting, `path` is faster per sample and converges equally well.

## Hair and Fur

Hair and fur are rendered as cubic Bézier ribbon primitives using a RenderMan-style ray-ribbon intersection. Each strand is stored as a sequence of endpoint-sharing cubic Bézier segments; at render time the ribbon is subdivided recursively in ray space (de Casteljau, depth 6) until it can be approximated as a straight 2D line, then the closest point to the ray is tested against the interpolated half-width.

Shading uses the **Marschner 2003** hair BSDF with three scattering lobes:

| Lobe | Path | Description |
|---|---|---|
| R | Reflected | Single reflection off the cuticle surface |
| TT | Transmitted–Transmitted | Light passing straight through the fiber |
| TRT | Transmitted–Reflected–Transmitted | Internal reflection, produces the characteristic highlight |

Each lobe uses a von Mises-Fisher longitudinal distribution (M_p), exact Fresnel attenuation with Beer's law absorption (A_p), and a trimmed logistic azimuthal distribution (N_p). Variance remapping and logistic scale follow the PBRT v4 curve-fits.

Default parameters: η = 1.55 (glass-like cuticle), σ_a = (0.06, 0.10, 0.20) ACEScg (medium brown), β_m = β_n = 0.30 (moderately rough), α = 2° cuticle tilt.

### Loading hair from Alembic

Hair and fur strands are loaded from Alembic `.abc` files via `--curves`:

```bash
./build/Darwin/anacapa --scene character.usda --curves hair.abc -o render.exr
```

Supported curve bases and their automatic conversion to endpoint-sharing cubic Bézier:

| Alembic basis | Conversion |
|---|---|
| `kBezierBasis` (cubic) | Used directly (assumes 3n+1 endpoint-sharing CVs) |
| `kBsplineBasis` (cubic) | Converted via the standard B-spline → Bézier formula |
| `kCatmullromBasis` (cubic) | Converted to Bézier (centripetal form) |
| `kLinear` | Promoted to cubic Bézier with collinear inner CVs |

Width channels (per-vertex, per-curve, or constant scope) are read automatically. All transforms in the parent hierarchy are accumulated and applied at load time.

## Blender Addon

The Anacapa Blender addon provides a fully integrated render pipeline — from Blender's Properties panel straight to a finished EXR in Blender's image editor, with no manual export steps.

### Installation

Copy the `anacapa_renderer/` folder to your Blender addons directory and enable it in Preferences → Add-ons. Set the path to the `anacapa` binary in the addon preferences.

### Rendering from Blender

1. Set the render engine to **Anacapa** in Properties → Render
2. Configure sampling, integrator, lighting, and output in the Anacapa panels
3. Click **Render** — the addon handles everything automatically:
   - Hair/fur Curves objects and particle hair systems are exported to a temporary Alembic `.abc` file **first**, before any scene prep runs
   - The scene (meshes, lights, materials, camera) is prepped and exported to a temporary USD file
   - Anacapa is launched with `--scene … --curves …` assembled automatically
   - A live progressive preview updates in Blender's Image Editor every 500 ms as tiles complete
   - The final EXR is loaded when rendering finishes; press Escape to cancel mid-render

The hair export always runs before USD prep. This matters because the prep script converts Curves objects to meshes for USD export — exporting hair first ensures the original strand geometry is captured.

### Dirty tracking and caching

The addon caches both the USD and Alembic exports separately and only re-exports when the scene has actually changed:

| Change | Re-exports |
|---|---|
| Geometry or material change | USD + Alembic |
| Hair/fur strand change | Alembic only |
| Transform-only change | USD only |
| No change | Reuses both cached files |

### Export Scene operator

**Properties → Render → Output → Export Scene for Anacapa** exports the USD and Alembic files to a location you choose and prints the full `anacapa` command to the system console. Use this to hand off a render to a farm or inspect the export without launching a render.

### Known limitations

- Hair material parameters (color, roughness, η) use the Marschner defaults for all strands. Per-object hair material overrides are not yet exposed in the UI.
- Animated hair (simulated or cached) exports a single frame (the current frame). Animated hair sequences are not yet supported.
- The GPU (`--interactive`) backend does not support hair rendering. Hair strands are silently absent in interactive mode.

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

- Animated mesh vertices are stored in **object space**. The BVH stores time-expanded world-space bounds (union across all motion keys) to correctly enclose the full arc.
- All authored `xformOp:*` time samples are read from the prim and its parent hierarchy. The shutter interval is read from the camera prim's `shutter:open`/`shutter:close` attributes, falling back to the stage's `startTimeCode`/`endTimeCode`.
- At intersection time, a binary search finds the bracketing key pair and `objectToWorld` is piecewise-linearly interpolated, then inverted to transform the ray into object space. Lerping the forward matrix and inverting is correct; lerping inverse matrices directly is not when rotation is present.
- `ray.time` is sampled once per path on the primary camera ray and propagated to all secondary rays and shadow rays to keep the scene temporally consistent across a path.

## Scene Format

Scenes are authored in OpenUSD (`.usda` text, `.usdc` binary, or `.usd` auto-detect). The loader supports:

| USD Prim | Anacapa |
|---|---|
| `UsdGeomMesh` | Triangulated mesh — static (world-space baked) or animated (object-space + N motion keys) |
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

```

## Progressive Preview

When `--png` is set, Anacapa writes a tone-mapped preview PNG **during the render** — not just at the end. A background thread flushes whatever tiles have completed every 500 ms so you can watch the image build up in real time. The final write after all tiles are done guarantees the PNG on disk is always the complete render.

```bash
# Write both EXR and a progressively-updated tone-mapped PNG
./build/Darwin/anacapa --scene scene.usda -o render.exr --png render.png

# Adjust exposure before tone mapping (stops; positive = brighter)
./build/Darwin/anacapa --scene scene.usda -o render.exr --png render.png --exposure 1.0
```

`--exposure` applies an EV stop adjustment (multiply by 2^exposure) before the ACES filmic tone map. Pair `--png` with the `viewer` tool (see below) for a live render preview without a separate EXR viewer.

To produce a PNG manually from an existing EXR using `oiiotool`:

```bash
# Reinhard tone map → sRGB PNG
oiiotool beauty.exr --tonemap reinhard -o beauty_tm.png

# Exposure boost + Reinhard
oiiotool beauty.exr --mulc 2.0,2.0,2.0 --tonemap reinhard -o beauty_tm.png

# Tone map the denoised layer
oiiotool out.exr --ch "denoised.R,denoised.G,denoised.B" --chnames "R,G,B" \
  --tonemap reinhard -o denoised_tm.png
```

For interactive EXR viewing with layer switching:

- **[mrViewer](https://mrviewer.sourceforge.io)** — macOS/Linux/Windows, designed for VFX
- **[DJV](https://darbyjohnston.github.io/DJV/)** — cross-platform, lightweight

## Interactive (GPU) Mode

Pass `--interactive` to use the GPU backend instead of the CPU path tracer. Intended for fast iteration — checking composition, previewing lighting — where speed matters more than accuracy.

Two backends are supported:
- **Metal** (Apple Silicon) — hardware ray tracing via `MTLAccelerationStructure`; requires `ANACAPA_ENABLE_METAL` (enabled via the `macos-arm64-usd` preset)
- **CUDA** (NVIDIA, Linux/WSL2) — pure-CUDA software BVH path tracer; requires `ANACAPA_ENABLE_CUDA` (enabled via the `linux-x86_64-cuda` preset)

If the GPU backend cannot be initialised the renderer falls back to CPU with a warning.

### Performance

Measured on Apple M3 Pro, 400×400 @ 64 spp:

| Scene | CPU (BDPT, 11 threads) | GPU (Metal) | Speedup |
|---|---|---|---|
| Cornell box (36 tris) | ~1168 ms | ~301 ms | ~3.9× |

Measured on Linux/WSL2 (NVIDIA RTX A400, 10 CPU threads), 800×800:

| Scene | CPU (path, 256 spp) | GPU CUDA (256 spp) | Speedup |
|---|---|---|---|
| Blender 3.5 splash (4.4M tris) | ~83.7 s | ~23.1 s | ~3.6× |

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

## Tools

### `denoise` — standalone denoiser

Denoises any linear HDR EXR using Intel OIDN. The input does not need to have been rendered by Anacapa — any EXR with linear float RGB channels works. Built automatically when `ANACAPA_ENABLE_OIDN` is on.

```bash
# Denoise a plain beauty EXR (no AOV guidance)
./build/Darwin/denoise -i render.exr -o denoised.exr

# Denoise with albedo and normals AOVs from an anacapa --write-aovs EXR
./build/Darwin/denoise \
  -i render.exr \
  -o denoised.exr \
  --albedo-layer albedo \
  --normal-layer normals

# Denoise a non-standard beauty layer (e.g. multi-layer EXR from another renderer)
./build/Darwin/denoise -i render.exr -o denoised.exr --beauty-layer beauty
```

| Flag | Default | Description |
|---|---|---|
| `-i, --input` | required | Input EXR path |
| `-o, --output` | required | Output EXR path (denoised beauty, R/G/B channels) |
| `--beauty-layer` | — | EXR layer name for the noisy beauty (e.g. `beauty`). Omit to use root `R,G,B` channels |
| `--albedo-layer` | — | EXR layer name for the albedo AOV (e.g. `albedo`); omit to run without albedo guidance |
| `--normal-layer` | — | EXR layer name for the normals AOV (e.g. `normals`); omit to run without normal guidance |

AOV guidance significantly improves denoising quality, especially at low sample counts. Use `--write-aovs` when rendering to capture them.

---

### `blender_prep_for_usd_export.py` — Blender scene prep

Prepares a Blender scene for clean USD export to Anacapa. Many Blender features do not survive USD export without preprocessing — this script bakes them out before export.

**What it handles:**

| Step | Description |
|---|---|
| Realize instances | Converts collection instances (Alt+D linked duplicates) to real objects |
| Convert to mesh | Converts curves, text, metaballs, and NURBS surfaces to mesh |
| Apply modifiers | Applies the full modifier stack (boolean, subdivision, mirror, array, solidify, bevel, etc.) |
| Apply scale | Applies object scale so USD normals are correct |
| Bake shader nodes | Detects shader nodes that don't survive USD export (Invert Color, Hue/Saturation, Bright/Contrast) and bakes their effect into a new texture file, rewiring the material to use it directly |
| Remove hidden helpers | Removes render-hidden leaf objects (boolean cutters, etc.) |

**Usage:**

```bash
blender my_scene.blend --background \
  --python tools/blender_prep_for_usd_export.py \
  -- output.usda
```

The original `.blend` is never modified. The script exports directly to USD at the path you specify. Use `.usda` for human-readable ASCII or `.usdc` for binary.

**Known limitations** (printed as warnings, require manual attention):

- Particle hair and Curves objects are exported separately to Alembic by the addon and do not need to pass through this script
- Volume / VDB objects
- Grease Pencil objects
- Library-linked objects that cannot be made local
- Objects with shape keys (modifier application is skipped; apply or remove shape keys manually first)
- Complex shader graphs (RGB Curves, Color Ramp, procedural textures with no image source) — require a manual Cycles bake

---

### `viewer` — progressive render viewer

An SDL2 + Dear ImGui viewer that watches a PNG file on disk and updates live as Anacapa writes progressive previews. Pairs directly with `--png`. Built when `ANACAPA_ENABLE_VIEWER=ON`.

```bash
# Configure and build with the viewer (macOS Apple Silicon + USD)
cmake --preset macos-arm64-usd-viewer -DUSD_ROOT=~/usd -DCMAKE_PREFIX_PATH=~/usd
cmake --build build/Darwin --target viewer --parallel

# Open a PNG to watch
./build/Darwin/viewer render.png

# Poll more frequently (default is 500 ms)
./build/Darwin/viewer render.png --interval 250
```

Start a render in another terminal with `--png render.png` and the viewer refreshes automatically as each batch of tiles is written.

#### Slots

The viewer keeps 8 independent **slots**. Each slot holds one render and has its own color-adjustment state. Use slots to keep multiple renders in memory and flip between them without re-rendering.

1. **Select the target slot** in the viewer's sidebar (or press `1`–`8`) before starting a render. The active slot is highlighted in blue.
2. **Run Anacapa** with the same `--png` path as always — no extra flags needed.
3. When the file changes on disk the viewer loads it into whichever slot is currently active.
4. Switch between slots at any time to compare results.

Empty slots are shown in grey; selecting one turns it blue immediately, making it the target for the next render.

#### Real-time color controls

Each slot has independent adjustments applied via a GLSL shader — the source PNG on disk is never modified.

| Control | Range | Effect |
|---|---|---|
| Exposure | −4 to +4 EV | Overall brightness (linear scale) |
| Contrast | −1 to +1 | Pivot-at-0.5 contrast curve |
| Saturation | 0 to 2 | 0 = greyscale, 1 = original, 2 = vivid |
| Temperature | −1 to +1 | −1 = cool (blue shift), +1 = warm (red shift) |

#### Keyboard shortcuts

| Key | Action |
|---|---|
| `1` – `8` | Switch active slot |
| `R` | Reset active slot's color to defaults |
| `Q` | Quit |

## Architecture

```
include/anacapa/
  core/         Types (Vec3f, Ray, Spectrum, BBox3f), ArenaAllocator
  accel/        IAccelerationStructure, GeometryPool (MeshDesc with MotionKey),
                CurvePool (StrandDesc — cubic Bézier, endpoint-sharing)
  shading/      IMaterial, ILight, ShadingContext
  sampling/     ISampler, SamplerState
  film/         Film, TileBuffer, DenoiseOptions
  integrator/   IIntegrator, Camera (shutter interval), SceneView
  scene/        SceneLoader (LoadedScene, CurvePool, shutter fields)

src/
  accel/        BVHBackend — SAH BVH; time-expanded bounds + piecewise-linear interpolation
                CurveBrute — ribbon intersection (de Casteljau depth-6 subdivision in ray space)
  shading/      Lambertian, StandardSurface, EmissiveMaterial
                MarschnerHair — Marschner 2003 R/TT/TRT lobes, vMF longitudinal,
                                trimmed logistic azimuthal, exact Fresnel + Beer's law
  shading/lights/  AreaLight, DirectionalLight, DomeLight (HDRI)
  sampling/     PCGRng, HaltonSampler
  integrator/   PathIntegrator (reference), BDPTIntegrator
                LightSampler (Vose alias table), MISWeight
  film/         Film — atomic accumulation, OIDN denoising, multi-layer EXR
  render/       ThreadPool, RenderSession — shutter wiring from scene or CLI
  scene/usd/    USDLoader — geometry, lights, materials, camera, animated transforms
  scene/alembic/ AlembicLoader — ICurves traversal, basis conversion, world xform
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
| 7 | Complete | Metal backend (Apple Silicon), pure-CUDA backend (NVIDIA/Linux/WSL2) |
| 8 | In Progress | MaterialX/OSL shading — OpenPBR terminal resolution, UsdPreviewSurface texture fallback, JSON sidecar extraction |
| 9 | Complete | Hair and fur — Marschner BSDF, ray-ribbon intersection, Alembic loader, Blender addon integration |

## License

MIT — see [LICENSE](LICENSE).
