# Anacapa

A physically-based bidirectional path tracer written in C++20.

Named after Anacapa Island, part of California's Channel Islands.

## Features

- **Bidirectional path tracing (BDPT)** with multiple importance sampling *(Phase 2, in progress)*
- **Custom SAH BVH** — surface area heuristic build with 12-bucket binning, Möller–Trumbore traversal
- **Custom thread pool** — tile-parallel rendering with `std::thread`, no external threading library
- **Scrambled Halton sampler** — low-discrepancy sampling up to 128 dimensions
- **OpenEXR output** — HDR linear float images via OpenImageIO
- **USD scene format** — Phase 3, optional
- **OSL shading language** — Phase 4, optional
- **GPU backends** — Metal (Apple Silicon) and CUDA+OptiX (NVIDIA) planned for Phase 5
- **Zero compiled third-party dependencies** in the core renderer

## Dependencies

| Dependency | Required | Install |
|---|---|---|
| OpenImageIO | Yes | `brew install openimageio` |
| OpenUSD | No (`-DANACAPA_ENABLE_USD=ON`) | `brew install usd` |
| Open Shading Language | No (`-DANACAPA_ENABLE_OSL=ON`) | `brew install open-shading-language` |

Header-only dependencies (fetched automatically by CMake): spdlog, CLI11, GoogleTest.

## Building

```bash
# Prerequisites (macOS)
brew install openimageio

# Configure and build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Run tests
cd build && ctest --output-on-failure
```

## Usage

```bash
# Render the built-in Cornell box scene
./build/anacapa -o out.exr

# Full options
./build/anacapa \
  --width  800 \
  --height 600 \
  --spp    256 \
  --depth  8   \
  --output render.exr

./build/anacapa --help
```

The output is a linear HDR EXR file. Apply exposure compensation in your viewer — the scene uses physically-based light units.

## Architecture

```
include/anacapa/
  core/         Types (Vec3f, Ray, Spectrum, BBox3f), ArenaAllocator
  accel/        IAccelerationStructure, GeometryPool
  shading/      IMaterial, ILight, ShadingContext
  sampling/     ISampler, SamplerState
  film/         Film, TileBuffer
  integrator/   IIntegrator, Camera, SceneView

src/
  accel/        BVHBackend (custom SAH BVH)
  shading/      Lambertian, EmissiveMaterial, AreaLight
  sampling/     PCGRng, HaltonSampler
  integrator/   PathIntegrator (reference), BDPTIntegrator (Phase 2)
  film/         Film (atomic EXR output)
  render/       ThreadPool, RenderSession, TileScheduler
  scene/usd/    USD scene loader (Phase 3)
```

All memory-owning data structures use SoA (Structure-of-Arrays) layout to enable zero-copy migration to GPU backends in Phase 5.

## Roadmap

| Phase | Status | Description |
|---|---|---|
| 1 | Complete | CPU path tracer, custom BVH, Halton sampler, EXR output |
| 2 | Next | Bidirectional path tracing with MIS weight computation |
| 3 | Planned | OpenUSD scene loading (geometry, materials, lights, camera) |
| 4 | Planned | MaterialX `standard_surface`, OSL shading, HDRI dome lights |
| 5 | Planned | Metal backend (Apple Silicon), CUDA+OptiX backend (NVIDIA) |

## License

MIT — see [LICENSE](LICENSE).
