# Path Integrator — Specular Convergence Improvements

Specular on dark dielectrics (e.g. IOR=1.45, F0=3.4%) converges very slowly in
path tracing because the specular lobe gets only ~1.7% of the sampling budget
(`wSpec = spec * F0 ≈ 0.017`). Cycles mitigates this largely through OIDN
denoising and indefinite viewport accumulation. The ideas below address the
root convergence problem.

---

## 1. Denoising — Intel OIDN (highest leverage, near-term)

We already write albedo and normal AOVs to the tile buffer. OIDN can consume
those as auxiliary inputs to reconstruct clean specular from far fewer samples.
Most of the specular signal IS there — it is just noisy.

- API: `oidn::newDevice()` → `oidn::Filter("RT")` with color/albedo/normal buffers
- Can run as a post-process after each progressive batch
- Closest single change to matching Cycles' viewport quality per sample

---

## 2. Roughness-adaptive lobe selection in `sample()`

Currently `wSpec ≈ spec * F0 ≈ 0.017` — only 1.7% of continuation rays are
specular, causing very high variance. A better approach: use a higher surrogate
sampling probability `wSpec_sample = max(spec * specE, kMinSpecWeight)` (e.g.
floor at 0.1 for smooth materials), and compensate in `f/pdf` so the estimator
stays unbiased.

- `evaluate()` must return the same surrogate probability in `be.pdf` for MIS
  to stay correct
- Reduces fireflies versus naively boosting without compensation
- Low implementation cost, good payoff for dark dielectrics

---

## 3. Restore BSDF-sampling branch in `estimateDirect`

The BSDF-sampling branch was removed from `estimateDirect` because emitter hits
via path continuation handle it. But for specular surfaces, path continuation
only generates specular rays 1.7% of the time. Restoring the BSDF branch adds a
second strategy: sample a direction from the GGX lobe and test if it hits the
light. MIS combines both.

- Especially effective for small/bright lights whose solid angle aligns with the
  specular peak
- Standard two-strategy MIS (light + BSDF) is the textbook fix for this
- Moderate implementation cost

---

## 4. BSDF-weighted area light sampling

In `estimateDirect`, we currently sample the area light uniformly over its
surface. For a specular BSDF, most of those samples land far from the specular
peak and contribute ~0. Instead, clamp the sampling domain to the intersection
of the light's solid angle and the GGX lobe's significant support.

- Requires computing the specular reflection direction and finding the overlap
  with the light's projected solid angle
- Eliminates near-zero NEE contributions; variance drops dramatically
- Higher implementation cost; most useful when light and specular peak overlap

---

## 5. ReSTIR light sampling

Replace the single uniform light pick per bounce with a reservoir that maintains
M candidate light samples weighted by `f(wo, wi) * Li * cosI`. For specular
surfaces the reservoir naturally concentrates on lights overlapping the lobe.

- Most impactful for scenes with many lights or strong HDRI
- Enables temporal reuse across frames (progressive renders)
- Highest implementation cost of the options listed here

---

## Open question: OSL GGX π factor

As of 2026-05-22 we multiply `evalGGXReflLobe` by π (see `OslMaterial.cpp`).
This brings path specular in line with BDPT on OSL/OpenPBR materials. The
hypothesis is that `NG_open_pbr_surface_surfaceshader` bakes a π into its
closure weight normalization (a convention in some OSL layering models that
converts between reflectance and radiance units). Needs verification: log the
actual `lobe.weight` values the OpenPBR shader emits and check whether they
include a 1/π factor. If they do, the π multiply is correct; if not, it is
overcounting and should be removed.
