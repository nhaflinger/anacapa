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

## What's Built (Phase 2 — COMPLETE)

31/31 unit tests passing. BDPT renders Cornell box (400×400 @ 16 spp) in ~1.2s.

- `src/integrator/BDPTIntegrator.h/.cpp` — full BDPT with all (s,t) strategies
- `src/integrator/MISWeight.cpp` — Veach §10.3 power heuristic MIS weights
- `src/integrator/LightSampler.h` — Vose alias table light selection
- `include/anacapa/integrator/PathVertex.h` — SoA PathVertexBuffer + DeviceView
- `include/anacapa/integrator/MISWeight.h` — bdptMISWeight() interface

## Phase 3 (USD) — COMPLETE

Full UsdGeomMesh, UsdLuxLight, UsdGeomCamera, UsdShadeMaterial loader.

## What's Built (Phase 4/5 — COMPLETE)

42/42 unit tests passing. Kitchen_set.usdc renders in ~3.3s at 640×480 @ 32 spp with full color.

- `src/shading/StandardSurface.h` — GGX multi-layer BSDF (clearcoat, metallic, dielectric, diffuse)
- `src/shading/lights/DomeLight.h/.cpp` — equirectangular HDRI, 2D piecewise-constant importance sampling
- `src/shading/OslMaterial.h` — OSL adapter (enabled with `ANACAPA_ENABLE_OSL=ON`)
- Alembic hair pipeline complete: AHAIR002 per-strand color, O(log S) segment BVH, Marschner sigma_a from ctx.color
- GPU CUDA port complete (Apr 30 2026, all Metal fixes ported to Linux machine)
- Display driver complete: IDisplayDriver, FileDisplayDriver, SocketDisplayDriver, pause/cancel/resume

## Specular fixes — 2026-05-22 (commits ebf2c7f, 78210ee, 7d00898)

Three compounding bugs made specular nearly invisible in path renders:

1. **Env light excluded from NEE** — PathIntegrator and PhotonMapIntegrator only sampled `scene.lights`, not `scene.envLight`. Fixed: envLight added to NEE pool and emitterPdf in both integrators.

2. **StandardSurface double-Fresnel** — `evalCombined` scaled the BSDF value by `wSpec` (≈ spec × F0 / total ≈ 0.017) instead of the physical weight `spec` (0.5), making NEE ~29× too dim on dark dielectrics. Also `f0 = spec × m_f0Dielectric` put spec into the Fresnel, squaring it. Fix: `f += fS * spec` (physical BSDF), `pdfFwd += pF * wSpec` (sampling PDF), `f0 = m_f0Dielectric` (no spec in F0).

3. **OslMaterial extra cosine** — `evalGGXReflLobe` multiplied by `wi.z`, double-counting the cosI the integrator applies externally. Fixed.

4. **Experimental π on OSL GGX** — multiplying `evalGGXReflLobe` result by π brings path specular in line with BDPT. Hypothesis: `NG_open_pbr_surface_surfaceshader` bakes π into its closure weight normalization. Needs verification by logging actual `lobe.weight` values. See comment in OslMaterial.cpp.

## Path specular convergence roadmap

Full details in `docs/path_specular_improvements.md`. Priority order:
1. OIDN denoising — highest leverage, uses existing albedo/normal AOVs
2. Restore BSDF-sampling branch in `estimateDirect` (two-strategy MIS)
3. Adaptive lobe selection — boost `wSpec_sample`, compensate in f/pdf
4. BSDF-weighted area light sampling
5. ReSTIR

## Known issues / deferred

- **Light intensity off by π**: addon exports `intensity = ld.energy / π` with `normalize=true`; loader computes `Le = energy / (area × π²)` but should be `energy / (area × π)`. Lights are π× too dim vs Cycles. Deferred — scenes tuned around current values.
- **Weighted light sampling**: previous attempt broke renders. Root cause: `emitterPdf` and `estimateDirect` selection probability must change together. For weighted selection: `estimateDirect` divides by `p_k × ls.pdf` (no N multiply outside); `emitterPdf` returns `sum_k(p_k × light_k.pdf)`. Use a flag for A/B testing.
- **GPU remaining issues**: specular too strong, glass highlight too weak, highlights broader than CPU, no MIS on GPU.

## Key Implementation Notes

### OIIO 3.x API
`read_image(subimage, miplevel, chbegin, chend, TypeDesc, ptr)` — NOT the 2.x `read_image(TypeDesc, ptr)`.

### StandardSurface sampling
The `sample()` method picks a layer via `uComponent`, then calls `evalCombined()` to compute the full weighted PDF and f across all layers. BSDF value uses physical weights (spec, base), PDF uses selection weights (wSpec, wDiff). This split is intentional — do not re-merge them.

### Build directory
macOS → `build/Darwin`, Linux → `build/Linux`. Never bare `build/`.
