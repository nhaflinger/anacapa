# Adaptive Sampling — Future Improvements

## Current Approach and Its Limitation

The current adaptive pass renderer measures per-tile output variance and allocates additional
samples to high-variance tiles. This works well for most sources of noise but has a fundamental
blind spot for **translucent shadow regions**.

### The Translucent Shadow Problem

When a translucent sphere sits above a floor, the floor tiles in its shadow are noisy. The
variance measurement in those tiles is correct — they genuinely have high per-sample variance.
But adding more samples to the floor tiles doesn't reduce that variance proportionally, because
the variance comes from a low-probability event upstream: a path must scatter through the
translucent surface and then reach the floor. Every additional floor sample has the same low
probability of finding that path. The adaptive sampler can't know this because:

1. The floor tiles contain no translucent materials — they're white Lambertian.
2. The cause of the noise (the translucent sphere) is in a different tile entirely.
3. Variance is a backward-looking metric; it measures the symptom, not the cause.

The adaptive sampler's tile-local variance signal cannot distinguish "this tile is noisy because
the material here is hard to sample" from "this tile is noisy because a distant surface upstream
is hard to sample."

---

## Proposed Direction: Causal Pre-Pass

A cheap forward pre-pass that traces paths from the lights into the scene would produce a
**causal signal** — one that knows *why* a region is hard, not just that it is.

### How It Would Work

1. **Forward light trace (low SPP)**: Emit a small number of photon paths from each light
   source (e.g., 16–64 paths per light sample, coarse resolution).
2. **Record path history flags**: When a photon path passes through a translucent surface,
   flag the photon as carrying a "complex transport" marker.
3. **Build a density image**: Where complex-transport photons land (e.g., the floor under
   the translucent sphere), record high density.
4. **Seed the sample budget**: Use the density image to allocate extra samples in the main
   pass — both in the high-density landing regions (floor shadow) and in the upstream
   source regions (translucent sphere tiles).

The density image does not need to be accurate — it just needs to be directionally correct
about where hard transport concentrates. A 16-sample pre-pass is enough to identify the
regions that matter.

### Why This Fixes the Problem

The pre-pass signal is causal. A photon that scatters through the translucent sphere and lands
on the floor tells us:
- The floor tile will be noisy (landing region gets more samples).
- The translucent sphere is the cause (upstream tile also gets more samples).
- The path involved complex transport (we can specifically target path diversity there).

Neither the floor tiles nor the sphere tiles need to "discover" this through their own
variance — the pre-pass distributes that knowledge across the scene.

---

## Related Techniques

### Path Guiding (Müller et al., "Practical Path Guiding", 2017)
Iterative rather than a single pre-pass. Early render passes build a spatial data structure
(a 5D radiance distribution over position × direction, typically a SD-tree or similar) that
subsequent passes use to importance-sample toward productive directions. Later passes
naturally learn where translucent scattering concentrates light without explicitly tracking
material types. The guiding distribution is refined progressively.

**Tradeoff**: More complex to implement, but more general — it learns the right sampling
distribution for any kind of hard transport, not just translucency.

### Photon Mapping / SPPM
A full photon map is a bidirectional algorithm: trace photons forward from lights, store them
in a spatial hash, then gather during backward camera tracing. Stochastic Progressive Photon
Mapping (SPPM) combines this with progressive refinement to handle caustics and translucency
correctly without bias in the limit.

**Tradeoff**: Significant system to build. Correct in the limit but adds memory overhead
(photon store) and a second rendering mode.

### Path Complexity as a Per-Sample Flag (Lightweight Option)
A simpler intermediate approach: during the main render, when a path encounters a translucent
BSDF, set a flag on that sample's tile marking it as "complex-transport." After each adaptive
pass, tiles with a high fraction of complex-transport samples get a budget bonus above what
variance alone would allocate.

**Tradeoff**: Still reactive (looks backward at completed samples) rather than causal, but
is cheap to implement and requires no pre-pass infrastructure. Could be a useful first step.

---

## Implementation Priority

Roughly ordered from least to most infrastructure required:

1. **Per-sample path complexity flag** — minimal change to tile buffer and adaptive pass logic.
2. **Coarse forward light pre-pass** — requires a lightweight forward path tracer and a
   screen-space or world-space density accumulator.
3. **Path guiding (SD-tree or similar)** — requires a persistent spatial data structure rebuilt
   each pass, plus modified BSDF sampling to blend with the guiding distribution.
4. **Full photon mapping / SPPM** — requires a complete second rendering mode.

The coarse forward pre-pass (option 2) is likely the best near-term target: it directly solves
the translucent shadow noise problem with moderate implementation complexity and no change to
the main path integrator.
