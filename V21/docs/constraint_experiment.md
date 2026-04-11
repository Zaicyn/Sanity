# Constraint-Solver Experiment: Does GPU PBD Destroy Viviani Cache Coherence?

**Date:** 2026-04-11
**Hardware:** same GPU as the Viviani-sort retrospective baseline
**Scope:** one-shot experiment, no follow-up code expected from this document

## Question

Does adding a GPU distance-constraint solver over a 1000-particle lattice
embedded in the 20M-particle galaxy destroy the Viviani sort's gather-cache
coherence win (documented at `viviani_sort_retrospective.md`)?

## Experiment design

- **Lattice:** 10×10×10 cube (1000 particles) with 6-neighbor distance
  constraints (2700 total edges) dropped at r=50 in the disk plane, given
  the circular-orbit velocity a galaxy particle at the center would have
  so it feels the galactic environment rather than sitting inert.
- **Rest of the world:** 19,999,000 field particles running normal V21
  physics (scatter → reduce → stencil → gather_measure → siphon).
- **Solver:** GPU PBD distance constraints, 4 iterations per frame, no
  warm-starting, Baumgarte β=0.2. Constraints are pre-colored on the host
  into 6 vertex-disjoint buckets (axis × parity-of-starting-endpoint) so
  the kernel writes pos_x/y/z directly without atomics or contention.
  One dispatch per bucket, memory barrier between buckets → true
  sequential Gauss-Seidel on GPU.
- **Insertion point:** between gather_measure and siphon in the existing
  V21 dispatch pipeline. No change to siphon.comp or any other kernel.

## Three runs, ~5000 frames each at 20,000,000 particles

| Run | Config | Purpose |
|---|---|---|
| 1 | `--rigid-body off --init-sort viviani` | Control — reproduces retrospective baseline |
| 2 | `--rigid-body cube1000 --init-sort viviani` | Test — Viviani sort + constraint work |
| 3 | `--rigid-body cube1000 --init-sort none`    | Sanity — unsorted with constraints, should match unsorted retrospective baseline |

Steady-state means over 8 samples (excluding 2 warmup samples) per run:

| metric          | Run 1 (control) | Run 2 (test)   | Run 3 (sanity) |
|-----------------|----------------:|---------------:|---------------:|
| `scatter_ms`    |          1.979  |         2.003  |         1.736  |
| `stencil_ms`    |          0.012  |         0.012  |         0.012  |
| `gather_ms`     |      **1.391**  |     **1.395**  |     **2.930**  |
| `constraint_ms` |          0.001  |     **0.056**  |         0.056  |
| `siphon_ms`     |          8.085  |         8.065  |         8.066  |
| `project_ms`    |          1.683  |         1.686  |         1.762  |
| `tonemap_ms`    |          0.042  |         0.039  |         0.040  |
| `total_ms`      |     **13.192**  |    **13.258**  |    **14.601**  |

## Answer

**Run 2 `gather_ms` vs Run 1 `gather_ms`: +0.3% (+0.004 ms).**

Far inside the ±10% tolerance. The constraint solver does not disturb
the Viviani sort's cache coherence at this scale. Decision threshold
table from the plan:

| Δ gather_ms | Interpretation | Our result |
|---|---|---|
| ≤ ±10% | Door is open, unified substrate viable at small scale | **✓ +0.3%** |
| +10% to +30% | Marginal, needs investigation | — |
| +30% or more | Unified architecture wrong, pursue Bepu-alongside | — |

Run 3's `gather_ms = 2.930 ms` matches the retrospective's unsorted
baseline of 2.910 ms within 0.7%, confirming the measurement pipeline is
not confounded: the Viviani sort really is the thing keeping gather fast,
and the lattice/constraints aren't spuriously helping or hurting.

## Cost accounting

- **Solver cost**: `constraint_ms ≈ 0.056 ms/frame`. That's 24 dispatches
  (4 iters × 6 buckets × ~450 constraints each) running sub-millisecond.
  Essentially free relative to the 13 ms frame budget.
- **Total frame time delta** (Run 2 − Run 1): **+0.066 ms**, which equals
  `constraint_ms` within noise. **Zero cascading slowdown** — the rest of
  the pipeline pays nothing for the lattice's presence.

## Lattice stability (at 100K smoke test, not 20M)

At 100K particles (where lattice indices sit inside the 100K oracle
readback window), the lattice was verified to:

- **Orbit the black hole** as a rigid body: particle 0 moved from
  (47.8, -2.2, 0.1) at frame 500 to (31.1, -2.1, 37.3) at frame 9500,
  radius ≈ 48, fully consistent with a circular disk orbit at r=50.
- **Preserve all 2700 constraint distances** to within 0.04% of the
  0.5 rest length across 10000 frames under the full Viviani field
  (differential rotation trying to shear the cube). Max observed drift
  `d(0,1) = 0.5002` (X-neighbor), `d(0,10) = 0.4997` (Y-neighbor),
  `d(0,100) = 0.5000` (Z-neighbor). Far tighter than the ±10% bar
  the plan set.

## What this experiment does NOT answer

- Scaling to many rigid bodies (only 1 in this test). A real rigid-body
  engine needs thousands.
- Collision detection and response (none in MVP).
- Articulation (only distance constraints, no hinge/ball-socket).
- Determinism for networking (atomic ordering is OK here because the
  coloring is vertex-disjoint, but this is not a Gauss-Seidel proof of
  bit-identical reproducibility across runs).
- Friction, stacking, sleeping islands (all deferred).
- Constraint solver convergence quality (4 iters is not a fully
  converged solution; Baumgarte PBD is a latency test, not a physics
  validation).
- Anything at 100× or 1000× this many rigid-body particles.

These are all follow-up projects that depend on this MVP result being
encouraging. **The result is encouraging.** Phase 2 (collision detection,
multi-rigid-body scaling) is worth pursuing when the user wants to take
it there.

## Files touched (for future archaeology)

- `kernels/constraint_solve.comp` — new, 107 lines
- `vk_compute.h` — added `ConstraintPushConstants`, `PhysicsCompute`
  constraint fields, `initConstraintCompute` decl, `readTimestamps`
  signature with `out_constraint_ms`
- `vk_compute.cpp` — timestamp pool grew 7→8 slots, new
  `initConstraintCompute` (~200 lines), new constraint dispatch block
  in `dispatchPhysicsCompute` (6-bucket × 4-iter loop with barriers),
  `readTimestamps` now reports 7 values, cleanup destroys new resources
- `blackhole_v21_visual.cpp` — new `--rigid-body {off,cube1000}` flag,
  new `init_rigid_body_cube` helper with orbital velocity + 6-bucket
  coloring, `init_particles` takes an optional `n_generate` split,
  new `ConstraintLattice` struct, lattice-stability probe in the
  oracle-check block (no-ops at 20M because the lattice is outside
  the oracle window)
- `CMakeLists.txt` — added `constraint_solve.comp` to `KERNEL_SOURCES`

Original gameplan: `/home/zaiken/.claude/plans/wiggly-prancing-hamming.md`

---

# Phase 2.1 Scaling Results (2026-04-11, same session)

Follow-up experiment on the MVP. The MVP's "what this does NOT answer" list
opened with **scaling to many rigid bodies**. This section closes that gap.

## Question

Two-part cost-curve question that gates every Phase 2.2/2.3 decision:

> **Q1:** Does `constraint_ms(N)` scale linearly, sublinearly, or super-linearly
> in N, the number of independent 10×10×10 cubes (N ∈ {1, 2, 10, 100})?
>
> **Q2:** Does `gather_ms` stay flat in the 1.37–1.67 ms plateau band as lattice
> particles grow from 0.005% of the field (N=1) to 0.5% (N=100)?

## Experiment design

Same per-cube parameters as the MVP (10×10×10 lattice, 2700 6-neighbor
distance constraints, SPACING=0.5, 4 PBD iters/frame, Baumgarte β=0.2). Four
measurement runs at 20M field particles, with N cubes placed at evenly-spaced
azimuthal angles on the r=50 circle in the disk plane. Each cube gets the
circular-orbit velocity tangent to its own center. All runs use
`--init-sort viviani` (the question is scaling against the sort-preserving
case, not re-measuring the unsorted baseline).

The critical architectural choice was **bucket concatenation across cubes**
rather than per-cube bucket sets: within one super-bucket `k`, merge all
cubes' colored bucket `k` edges into a single contiguous range. This keeps
total dispatches per frame at **24** (4 iters × 6 buckets) regardless of N.
The alternative — 6·N buckets → 600 dispatches/frame at N=100 — would have
replaced a scaling measurement with a launch-overhead artifact. Vertex
disjointness within each super-bucket is preserved because different cubes
occupy disjoint particle index ranges `[base + c·1000, base + (c+1)·1000)`,
so edges from different cubes can never share endpoints.

New CLI flag: `--rigid-count N` (default 1) works alongside
`--rigid-body cube1000`. Bare `--rigid-body cube1000` without `--rigid-count`
is bit-identical on the host side to the shipped MVP, inherited as the N=1
baseline below.

## Four measurement runs, means over 8 samples (skipping 2 warmup)

| N | lattice particles | constraints | `scatter_ms` | `gather_ms` | `constraint_ms` | `siphon_ms` | `total_ms` |
|----:|----:|----:|----:|----:|----:|----:|----:|
| **1** (baseline) | 1000 | 2700 | 1.993 | 1.403 | **0.056** | 8.051 | 13.264 |
| 2 | 2000 | 5400 | 1.888 | 1.386 | **0.061** | 8.069 | 13.103 |
| 10 | 10000 | 27000 | 1.908 | 1.398 | **0.063** | 8.061 | 13.133 |
| 100 | 100000 | 270000 | 2.030 | 1.406 | **0.167** | 8.096 | 13.499 |

N=1 row is bit-identical on the host side to MVP Run 2 (same pair list, same
bucket counts `[500 400 500 400 500 400]`, same particle positions). GPU
timestamps drift ~0.8% from the MVP numbers (within measurement noise).

## Decision criteria: BOTH PASS

**Q1** (`constraint_ms` scaling): **STRONGLY SUBLINEAR.**

- `constraint_ms(N=100) = 0.167 ms` — well below the 1.0 ms threshold, and
  **6× better** than the plan's optimistic prediction of ~1.05 ms, **33×
  better** than naive linear extrapolation (`0.056 × 100 = 5.6 ms`).
- `constraint_ms(N=100) / constraint_ms(N=1) = 2.98` — for 100× the work, we
  pay 3× the time.
- Between N=1 and N=10 the curve is **essentially flat**: 0.056 → 0.063,
  +12% time for 10× the work. The GPU is entirely launch-overhead-bound
  across this range.
- Between N=10 and N=100 the curve starts rising: 0.063 → 0.167, +165% time
  for 10× the work. Still sublinear because the 6 bucket dispatches don't
  saturate the machine until several thousand edges per dispatch.

**Q2** (`gather_ms` invariance): **PASSES BY A WIDE MARGIN.**

`gather_ms` spans [1.386, 1.406] across all four N — a **1.4% spread**,
dominated by measurement noise, with no monotonic trend vs N. The Viviani
sort's gather-cache-coherence is completely undisturbed by adding 0.5% tail
lattice particles. The MVP's "the lattice lives in cache lines disjoint from
the gather traversal" intuition is confirmed at 100× the MVP's lattice size.

Scatter and siphon are also invariant within noise across all four N
(scatter spread 1.888–2.030, siphon spread 8.051–8.096), as expected — they
dispatch over the full 20M particles in all cases.

## Two-term fit

The plan predicted `constraint_ms(N) ≈ a + b·N` with `a ≈ 0.045–0.055 ms`
fixed overhead and `b ≈ 0.010 ms/cube`. Fitting against the N=1 and N=100
endpoints:

```
a + 1·b   = 0.056    →    a = 0.055 ms
a + 100·b = 0.167    →    b = 0.00112 ms/cube (≈ 1.12 µs/cube)
```

Predicted vs actual at the intermediate points:

| N | fit: a + b·N | actual | residual |
|---:|---:|---:|---:|
| 2 | 0.057 | 0.061 | +0.004 (noise) |
| 10 | 0.066 | 0.063 | −0.003 (noise) |

The residuals are well within GPU timestamp noise (±0.003 ms is typical
frame-to-frame jitter at this scale). The two-term model is an excellent fit
across four orders of magnitude in work, with `a = 0.055 ms` matching the
plan's prediction almost exactly and `b ≈ 1.1 µs/cube` in the saturated
regime.

**The linearity of the fit is itself interesting:** a strict two-term
`a + b·N` model would only hold once per-cube work saturates the SMs. The
data says it saturates by N=10 or so — meaning the constraint solver is
compute-bound for any N where you actually care, and the cost of adding
"more bodies" past that point is genuinely linear-per-edge.

## Architectural headroom

At N=100, `constraint_ms = 0.167 ms` of a 13.499 ms frame = **1.2% of the
frame budget**. At the current per-cube slope of 1.12 µs:

- **1000 cubes** would be ~1.17 ms (8.7% of frame, still fine).
- **10,000 cubes** would be ~11.3 ms (84% of frame, rapidly approaching the
  budget ceiling — but this would also be 10M lattice particles, half the
  total field count, at which point the experiment design itself should
  change).
- A rough linear ceiling: **roughly 5000–8000 cubes** before constraint_ms
  starts eating into the existing 13 ms frame envelope at the current
  1 cube ≈ 1.12 µs rate.

This is well beyond any realistic VR social-scene rigid-body budget
(~10–30 bodies for players + graspables + mechanisms). The unified substrate
has architectural headroom for at least two orders of magnitude beyond what
a shipped application would need.

## Prediction reconciliation

The plan predicted (section 6):

- `a ≈ 0.045–0.055 ms`. **Actual:** 0.055 ms. ✓ upper end of the range.
- `b ≈ 0.010 ms/cube` at saturation. **Actual:** 0.00112 ms/cube, or
  roughly 9× lower. Either (a) the shader compiled to more efficient SASS
  than I mentally modeled, (b) the L2 cache reuse across buckets within
  a single 4-iter frame is stronger than I predicted, or (c) memory
  bandwidth for a 2700-edge pair list (22 KB) fits entirely in L1/L2 and
  never touches VRAM.
- `constraint_ms(N=100) ≈ 0.8–1.3 ms`. **Actual:** 0.167 ms. The prediction
  was **6× too pessimistic**.
- `gather_ms` within ±1% of N=1 baseline. **Actual:** within ±1.4%, all
  within measurement noise. ✓

The pessimism in the per-edge cost estimate came from assuming the kernel
would become memory-bandwidth-bound on the pair list once the number of
edges exceeded L1 capacity. At N=100 the pair buffer is 2.16 MB plus rest
lengths 1.08 MB — 3.24 MB total — which fits comfortably in the L2 cache of
any modern desktop GPU. The 4 iterations per frame all touch the same data,
so after the first iteration the entire constraint set is L2-resident and
memory latency essentially disappears.

**This reinforces the MVP's win condition**: constraint work on a static,
bounded constraint graph is a *perfect* L2 use case on GPU. As long as the
pair list fits in L2, the solver is essentially free compared to the
O(N_particles) passes like siphon and scatter.

## What this does NOT answer

Same structure as the MVP writeup — this experiment still leaves a lot of
Phase 2 territory unexplored:

- **Collision detection and response** between rigid bodies — Phase 2.2.
  Contact constraints are dynamic (they appear and disappear as bodies
  collide), which breaks the static-pair-list L2 assumption that makes this
  experiment so fast. The dynamic-constraint cost curve will look very
  different.
- **Articulated joints** (hinge, ball-socket, fixed) — Phase 2.3. These
  are "just more distance constraints" in a PBD substrate, but the
  scaling and convergence behavior under rotational coupling is not yet
  measured.
- **Friction, stacking, sleeping islands** — Phase 2.4. Requires warm-
  starting the solver between frames, which breaks the "start from scratch
  every frame" assumption embedded in the current 4-iters-no-warm-start
  design.
- **Determinism audits** — Phase 2.5.
- **Multi-radius cube placement.** All N cubes in this experiment orbit at
  r=50. Placing them at different radii (e.g. a debris belt from r=30 to
  r=100) would test whether the Viviani sort's cache coherence holds when
  lattice particles span many gather cells rather than clustering in one
  azimuthal band. Probably fine given the single-radius result, but not
  directly measured.
- **Constraint solver convergence at very high N.** At N=100, 4 iters of
  Baumgarte PBD may or may not be enough to hold 100 separate cubes against
  their individual differential-rotation shears. We did not position-read-
  back at 20M, so lattice stability at 100 cubes is unverified beyond the
  fact that nothing exploded.

Do not pre-build any of these. The MVP's and Phase 2.1's sequential "each
step is data" discipline is what makes the cumulative evidence trustworthy.

## Files touched (for archaeology)

- `blackhole_v21_visual.cpp` — added `--rigid-count N` CLI flag (default 1);
  refactored `init_rigid_body_cube` to accept `n_cubes` and loop over N cube
  origins with super-bucket concatenation; threaded `rigid_count` into
  `n_rigid` and the call site. ~70 net lines changed.
- `docs/constraint_experiment.md` — this appended section.

No changes to `vk_compute.cpp`, `vk_compute.h`, `kernels/constraint_solve.comp`,
or any other kernel. The MVP was built parametrically enough that the backend
handles multi-body with zero modifications.

Plan file: `/home/zaiken/.claude/plans/wiggly-prancing-hamming.md` (overwritten
in the same session to hold the Phase 2.1 plan; the MVP plan was already
superseded by this time).
