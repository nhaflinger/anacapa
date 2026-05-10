# Reading list

Annotated bibliography of the published research that lives inside Anacapa.
This is not a path-tracing introduction — it skips Whitted, Cook, Kajiya,
basic BVH construction, and the rendering equation itself. It indexes the
specific shader, sampling, and acceleration techniques that took
non-trivial domain knowledge to implement.

When the implementation is split across CPU and GPU paths, both are noted.

---

## Materials and BSDFs

### Microfacet theory

* **Walter, Marschner, Li, Torrance (2007). "Microfacet Models for Refraction through Rough Surfaces."** EGSR.
  GGX (Trowbridge-Reitz) NDF, the half-vector formulation for rough
  reflection and refraction, and the rough-glass transmission BSDF
  (Walter eq. 21). [`StandardSurface.h`](src/shading/StandardSurface.h) —
  `D_GGX`, `sampleGGX_halfvector`, the rough-glass branch of `sample()`.

* **Heitz (2014). "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs."** JCGT 3(2).
  Smith G2 derivation, separable vs height-correlated forms, and the
  argument for picking separable when sampling fidelity is more important
  than perfect energy bookkeeping. [`StandardSurface.h`](src/shading/StandardSurface.h)
  — `G1_Smith`, `G2_Smith_Separable`.

* **Schlick (1994). "An Inexpensive BRDF Model for Physically-Based Rendering."** Computer Graphics Forum.
  The (1-cos)^5 Fresnel approximation that we use everywhere instead of
  the full polarized reflectance. [`StandardSurface.h`](src/shading/StandardSurface.h)
  — `schlickDielectric`, `schlickConductor`.

### Energy conservation

* **Burley (2012). "Physically-Based Shading at Disney."** SIGGRAPH course "Practical Physically Based Shading in Film and Game Production".
  The "Principled" BRDF — the source of our diffuse retro-reflection term
  (`Fd90 = 0.5 + 2 cos²θ_d * roughness`), and conceptually the layered
  diff/spec/coat structure of `StandardSurface`.
  [`StandardSurface.h::evalCombined`](src/shading/StandardSurface.h) —
  the diffuse layer block.

* **Kulla, Conty (2017). "Revisiting Physically Based Shading at Imageworks."** SIGGRAPH course "Physically Based Shading in Theory and Practice".
  Multi-scatter GGX compensation. The `f_ms = F_ms (1-E(wo))(1-E(wi)) / (π(1-E_avg))`
  term that recovers the energy single-scatter GGX masks at high
  roughness. Particularly significant for rough metals where F_avg is
  large. [`StandardSurface.h`](src/shading/StandardSurface.h) — `evalGGX_ms`,
  `specAvgAlbedoDielectric`, the `SpecAlbedoLUT` precompute.

* **Karis (2013). "Real Shading in Unreal Engine 4."** SIGGRAPH talk.
  The "split-sum" approximation that motivated our use of E_spec(cosθ_o, α)
  as the directional-hemispherical reflectance for energy-conserving
  spec/diff balance — replacing the point-Fresnel `specF` in the diffuse
  weight with the integrated reflectance `E_spec`.
  [`StandardSurface.h`](src/shading/StandardSurface.h) —
  `specAlbedoDielectric`, the `wDiff` calculation in `sample()` and `evaluate()`.

### MaterialX / standard surface

* **Smith, Anderson, Carucci, Eberle, Hill, et al. (Autodesk, 2020). "MaterialX standard_surface" specification.**
  The layered shader our `StandardSurfaceMaterial` implements: base
  (Lambertian) + specular (GGX dielectric) + metalness (GGX conductor) +
  coat (GGX dielectric, IOR=1.5) + transmission (Walter rough glass) +
  emission. [`StandardSurface.h`](src/shading/StandardSurface.h) end-to-end.

* **Gritz, Stein, Kulla, Conty (2010). "OpenShadingLanguage."** SIGGRAPH course.
  The OSL specification and shading-system architecture. We link OSL's
  shading runtime so MaterialX networks compiled to `.osl/.oso` can
  evaluate inside our path tracer. [`OslMaterial.cpp`](src/shading/OslMaterial.cpp)
  — `OslShadingSystem`, `OslMaterial::sample` / `evaluate`.

### Hair

* **Marschner, Jensen, Cammarano, Worley, Hanrahan (2003). "Light Scattering from Human Hair Fibers."** SIGGRAPH.
  R / TT / TRT azimuthal × longitudinal lobes, the cone of forward-scattered
  light around the perfect specular angle, and the basis of every
  hair shader written since. [`MarschnerHair.h`](src/shading/MarschnerHair.h).

* **Chiang, Bitterli, Tappan, Burley (2016). "A Practical and Controllable Hair and Fur Model for Production Path Tracing."** EGSR.
  Inter-fiber multi-scatter approximation (broad Gaussian longitudinal ×
  uniform azimuthal lobe) that brightens the hair interior at low SPP.
  Used as an alternative model when `"type": "chiang"` is set in
  `matassign.json`. [`ChiangHair.h`](src/shading/ChiangHair.h).

---

## Sampling and Monte Carlo

* **Veach (1997). "Robust Monte Carlo Methods for Light Transport Simulation."** PhD thesis, Stanford.
  Multiple importance sampling, the power heuristic, balance heuristic,
  bidirectional path tracing as an MIS combination of all (s,t) strategies.
  All three appear in Anacapa: power-heuristic MIS in
  [`PathIntegrator::Li`](src/integrator/PathIntegrator.cpp), the BDPT
  implementation in [`BDPTIntegrator`](src/integrator/BDPTIntegrator.cpp),
  and the GPU port of MIS in [`Shade.cu`](src/gpu/cuda/shaders/Shade.cu)
  and [`Shade.metal`](src/gpu/metal/shaders/Shade.metal).

* **Lawrence, Rusinkiewicz, Ramamoorthi (2004). "Adaptive Numerical Cumulative Distribution Functions for Efficient Importance Sampling."** EGSR.
  2D marginal/conditional CDF construction for environment maps. Used
  across CPU `DomeLight`, the Metal HDRI sampler, and the CUDA HDRI sampler.
  [`DomeLight.h::marginalCdf`](src/shading/lights/DomeLight.h),
  [`Shade.metal::sampleEnvDirection`](src/gpu/metal/shaders/Shade.metal),
  [`Shade.cu::sampleEnvDirection`](src/gpu/cuda/shaders/Shade.cu).

* **Arvo, Kirk (1990). "Particle Transport and Image Synthesis."** SIGGRAPH.
  Russian roulette path termination with the
  unbiased-by-construction `1/(1-q)` survival weight. Both integrators.

* **Shirley, Chiu (1997). "A Low Distortion Map Between Disk and Square."** JGT.
  Concentric mapping for uniform disk sampling — used by the thin-lens
  camera for depth-of-field aperture sampling.
  [`IIntegrator.h::generateRay`](include/anacapa/integrator/IIntegrator.h).

* **Duff, Burgess, Christensen, Hery, Kensler, Liani, Villemin (2017). "Building an Orthonormal Basis, Revisited."** JCGT 6(1).
  The branchless sign-trick orthonormal basis from a single normal that
  avoids Frisvad's south-pole singularity. Used wherever we need a TBN
  frame for shading. [`Types.h::buildOrthonormalBasis`](include/anacapa/core/Types.h).

* **O'Neill (2014). "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation."** Tech report, Harvey Mudd.
  The PCG generator used inline in both GPU shaders. Two state words,
  one xorshift output, far better statistical quality than xorshift32 or
  the old "wang hash" used in early GPU path tracers.

* **Halton (1964). "Algorithm 247: Radical-Inverse Quasi-Random Point Sequence."** CACM.
  Halton sequence used for the offline LUT integration in `SpecAlbedoLUT`.

### Path tracer subtleties

* **Möller, Trumbore (1997). "Fast, Minimum Storage Ray-Triangle Intersection."** JGT.
  The standard ray-triangle intersection used in our software BVH path
  (now removed from the GPU in favor of OptiX's hardware intersector).
  Still in [`BVHBackend.cpp`](src/accel/BVHBackend.cpp) for the CPU path.

---

## Acceleration structures

* **MacDonald, Booth (1990). "Heuristics for Ray Tracing using Space Subdivision."** The Visual Computer.
  Surface Area Heuristic (SAH) — minimizes expected ray cost by weighting
  child cost by surface area. Used in our BVH builder and the curve BVH.

* **Stich, Friedrich, Dietrich (2009). "Spatial Splits in Bounding Volume Hierarchies."** HPG.
  SBVH — split a primitive across the partitioning plane when SAH cost
  improves. Used in [`BVHBackend.cpp`](src/accel/BVHBackend.cpp) ("BVH Step 4"
  in the commit history).

* **Wald, Boulos, Shirley (2007). "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies."** TOG.
  4-wide SIMD BVH (BVH4) collapse from binary, with SOA-laid child AABBs
  for vectorised slab tests. [`BVHBackend.cpp`](src/accel/BVHBackend.cpp)
  ("BVH Step 3").

* **Parker, Bigler, Dietrich, Friedrich, Hoberock, Luebke, McAllister, McGuire, Morley, Robison, Stich (2010). "OptiX: A General Purpose Ray Tracing Engine."** SIGGRAPH.
  The pipeline-of-programs model: raygen / closesthit / miss /
  intersection / any-hit, plus Shader Binding Tables. The CUDA backend
  is structured around this model.
  [`Shade.cu`](src/gpu/cuda/shaders/Shade.cu),
  [`CudaPathIntegrator::Impl::buildOptixPipeline`](src/gpu/cuda/CudaPathIntegrator.cu),
  [`CudaAccelStructure::CudaAccelStructure`](src/gpu/cuda/CudaAccelStructure.cu)
  for the GAS build.

* **Heitz, Hill, McGuire, et al. — NVIDIA OptiX 8 motion blur reference.**
  `OptixMotionOptions` and `primitive_motion` BLAS for hardware-interpolated
  motion blur. Mirrors Apple's Metal 3 `primitive_motion` AS — same shape
  on both GPUs. [`MetalAccelStructure.mm`](src/gpu/metal/MetalAccelStructure.mm)
  and [`CudaAccelStructure.cu`](src/gpu/cuda/CudaAccelStructure.cu) build the
  motion-aware AS; `__raygen__rg` samples ray-time.

---

## Light transport

* **Lafortune, Willems (1993). "Bi-Directional Path Tracing."** Compugraphics.
  Connecting eye- and light-subpaths.

* **Veach, Guibas (1995). "Bidirectional Estimators for Light Transport."** EGSR.
  MIS-weighted combination of all (s,t) connection strategies in BDPT.
  Both in [`BDPTIntegrator.cpp`](src/integrator/BDPTIntegrator.cpp).

* **Glassner (1988). "Spacetime Ray Tracing for Animation."** IEEE CGA.
  The original "ray time = animation time" formulation that all modern
  motion-blur AS APIs implement at the hardware level.

---

## GPU runtime

* **Pharr, Jakob, Humphreys (2023). "Physically Based Rendering: From Theory to Implementation, 4th edition."** ChisCRC. Online.
  PBRT v4 establishes the importance-sampled-filter pattern (rather than
  splatting) and the per-vertex MIS state tracking we use.

* **Heitz (2018). "Sampling the GGX Distribution of Visible Normals."** JCGT 7(4).
  VNDF sampling — superior to plain NDF sampling at high roughness
  (lower variance, no rejected samples below the horizon). **Not yet
  implemented**; tracked for a future improvement.

---

## Color management

* **The ACES Project. "Academy Color Encoding System Documentation."** Academy.
  The reference rendering and output transforms we expose through OCIO.
  Underlies the `ACES 1.0/1.1/2.0 - SDR/HDR Video` views in our viewer's
  loaded config.

* **Selan (Sony Imageworks). "OpenColorIO" specification.**
  Display Device → View Transform → Look pipeline. The viewer panels
  enumerate `Config::getDisplay/getView` and dispatch to
  `DisplayViewTransform`. [`viewer.cpp::ocioInit`](src/tools/viewer.cpp).

* **Sobotka. "Filmic Blender."** Open source. Contributed Filmic-Log-Encoding
  view transforms back into OCIO and Blender. We don't ship it directly
  yet (apt's OCIO 2.4.1 doesn't include Filmic in any built-in config),
  but the viewer can load a user-provided `$OCIO` that does.

* **Hable (2010). "Filmic Tonemapping for Real Rendering."** GDC course.
  The Hable / Uncharted 2 polynomial fit. Conceptually the basis of
  Filmic-style "soft shoulder" tone mapping.

* **Narkowicz (2016). "ACES Filmic Tone Mapping Curve."** Blog post.
  The 5-coefficient polynomial ACES approximation used as the viewer's
  fallback tone mapper when OCIO is unavailable.
  [`viewer.cpp::aces`](src/tools/viewer.cpp).

---

## Denoising

* **Áfra, Wald, et al. (2019). "Open Image Denoise."** Intel.
  AI denoiser used as a post-render pass.
  [`Film.cpp::denoise`](src/film/Film.cpp).

* **Bako, Vogels, McWilliams, Meyer, Novák, Harvill, Sen, Derose, Rousselle (2017). "Kernel-Predicting Convolutional Networks for Denoising Monte Carlo Renderings."** SIGGRAPH.
  The albedo + normal AOV-driven denoising approach OIDN's "RT" filter
  uses, so guidance images dramatically improve quality at low spp.
  Anacapa optionally writes both buffers via `--write-aovs`.

---

## What we have *not* yet implemented (queued)

For completeness — these appear in our project memory as future work:

* Heitz VNDF sampling (replace plain NDF sampling at high roughness).
* Pixel reconstruction filters with PBRT-v4 importance sampling
  (Mitchell-Netravali, Blackman-Harris, Gaussian, etc.) — currently a hardcoded box filter.
* Subsurface scattering (currently absent; would land between SSS-1 and Christensen-Burley).
* Bezier curve hair primitives with hardware ray tracing
  (we have a software curve BVH and Marschner/Chiang shading, but
  hardware curve primitives via OptiX 8's curve API would be faster).
